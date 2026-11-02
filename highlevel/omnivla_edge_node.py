"""OmniVLA-edge node."""

from __future__ import annotations

import argparse
import collections
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image

from command_bus.bus import CommandPublisher
from command_bus.frames import FrameSubscriber

OMNIVLA_DIR = os.path.join(os.path.dirname(__file__), "..", "third_party", "OmniVLA")
sys.path[:0] = [OMNIVLA_DIR, os.path.join(OMNIVLA_DIR, "inference")]

import clip  # noqa: E402
from utils_policy import (  # noqa: E402
    load_model,
    transform_images_map,
    transform_images_PIL_mask,
)

MODEL_PARAMS = {
    "model_type": "omnivla-edge",
    "len_traj_pred": 8,
    "learn_angle": True,
    "context_size": 5,
    "obs_encoder": "efficientnet-b0",
    "encoding_size": 256,
    "obs_encoding_size": 1024,
    "goal_encoding_size": 1024,
    "late_fusion": False,
    "mha_num_attention_heads": 4,
    "mha_num_attention_layers": 4,
    "mha_ff_dim_factor": 4,
    "clip_type": "ViT-B/32",
}
LANGUAGE_GOAL = 7
WAYPOINT_SCALE = 0.1
TIMESTEP = 1 / 3
TARGET_WAYPOINT = 4


def waypoint_to_velocity(
    waypoint: np.ndarray, max_v: float, max_w: float, speed_cap: float = 0.5
) -> tuple[float, float]:
    """Steer towards a waypoint, same as the OmniVLA example code."""
    x, y, cos_yaw, sin_yaw = waypoint
    if abs(x) < 1e-8 and abs(y) < 1e-8:
        vx, wz = 0.0, np.arctan2(sin_yaw, cos_yaw) / TIMESTEP
    elif abs(x) < 1e-8:
        vx, wz = 0.0, np.sign(y) * np.pi / (2 * TIMESTEP)
    else:
        vx, wz = x / TIMESTEP, np.arctan(y / x) / TIMESTEP
    vx, wz = float(np.clip(vx, 0.0, speed_cap)), float(np.clip(wz, -1.0, 1.0))

    # Scale both so the turning radius stays the same
    scale = min(1.0, max_v / max(abs(vx), 1e-8), max_w / max(abs(wz), 1e-8))
    return vx * scale, wz * scale


def read_prompts() -> queue.Queue:
    """Collect prompts typed into the terminal."""
    prompts: queue.Queue = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            if line.strip():
                prompts.put(line.strip())

    threading.Thread(target=read, daemon=True).start()
    return prompts


def update_prompt(vla, prompts: queue.Queue) -> None:
    prompt = None
    while not prompts.empty():
        prompt = prompts.get()
    if prompt is not None:
        vla.set_prompt(prompt)
        vla.reset()
        print(f">>> prompt now {prompt!r}", flush=True)


class OmniVLAEdge:
    def __init__(self, prompt: str, device: torch.device):
        self.device = device
        checkpoint = os.path.join(OMNIVLA_DIR, "omnivla-edge", "omnivla-edge.pth")
        self.model, text_encoder, _ = load_model(checkpoint, MODEL_PARAMS, device)
        self.model = self.model.to(device).eval()
        self.text_encoder = text_encoder.to(device).eval()
        self.mask_small = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_large = np.ones((224, 224, 3), dtype=np.float32)
        self.history = collections.deque(maxlen=MODEL_PARAMS["context_size"] + 1)
        self.set_prompt(prompt)

        # Empty inputs for the goal types we don't use
        empty_map = transform_images_map(Image.new("RGB", (352, 352)))
        self.map_pair = torch.cat([empty_map] * 2, dim=1).to(device)
        self.goal_image = transform_images_PIL_mask(
            Image.new("RGB", (96, 96)), self.mask_small
        ).to(device)
        self.goal_pose = torch.zeros(1, 4, device=device)

    @torch.no_grad()
    def set_prompt(self, prompt: str) -> None:
        tokens = clip.tokenize(prompt, truncate=True).to(self.device)
        self.text_features = self.text_encoder.encode_text(tokens)

    def reset(self) -> None:
        self.history.clear()

    @torch.no_grad()
    def predict(self, frame: Image.Image) -> np.ndarray:
        """Predict 8 waypoints (x, y, cos, sin) in metres."""
        self.history.append(frame.resize((96, 96)))
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(self.history[0])
        history = transform_images_PIL_mask(list(self.history), self.mask_small)
        history = history.to(self.device)
        current = torch.split(history, 3, dim=1)[-1]
        current_large = transform_images_PIL_mask(
            frame.resize((224, 224)), self.mask_large
        ).to(self.device)
        actions, _, _ = self.model(
            history,
            self.goal_pose,
            torch.cat([self.map_pair, current], dim=1),
            self.goal_image,
            torch.tensor([LANGUAGE_GOAL], device=self.device),
            self.text_features,
            current_large,
        )
        waypoints = actions[0].float().cpu().numpy()
        waypoints[:, :2] *= WAYPOINT_SCALE
        return waypoints


def run_node(vla, args: argparse.Namespace) -> None:
    """Run the VLA on the newest camera frame and send velocity commands."""
    camera = None if args.image else FrameSubscriber()
    still_image = Image.open(args.image).convert("RGB") if args.image else None
    bus = CommandPublisher("highlevel")
    prompts = read_prompts()
    print("Type a new prompt and press Enter to change the goal.", flush=True)

    try:
        while True:
            update_prompt(vla, prompts)
            step_start = time.time()

            # Get frame
            if still_image is not None:
                frame = still_image
            else:
                latest = camera.latest()
                if latest is None:
                    time.sleep(0.05)
                    continue
                frame_time, frame = latest
                if step_start - frame_time > 1.0:
                    print(f"Warning: frame is {step_start - frame_time:.1f}s old")

            # Predict and send command
            waypoints = vla.predict(frame)
            vx, wz = waypoint_to_velocity(
                waypoints[TARGET_WAYPOINT], args.max_v, args.max_w, args.speed_cap
            )
            bus.send(vx, 0.0, wz)
            step_ms = (time.time() - step_start) * 1e3
            print(f"vx={vx:+.2f} wz={wz:+.2f}  ({step_ms:.0f} ms)", flush=True)

            if args.once:
                print("Waypoints (m):\n", np.round(waypoints, 2))
                return
            time.sleep(max(0.0, 1 / args.rate - (time.time() - step_start)))
    finally:
        bus.stop()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="move toward the door")
    parser.add_argument("--image", help="still image instead of the camera")
    parser.add_argument("--once", action="store_true", help="run once and exit")
    parser.add_argument("--rate", type=float, default=3.0, help="Hz")
    parser.add_argument("--max-v", type=float, default=0.3)
    parser.add_argument("--max-w", type=float, default=0.3)
    parser.add_argument("--speed-cap", type=float, default=0.5)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    if not torch.cuda.is_available():
        sys.exit("OmniVLA-edge needs a CUDA GPU.")
    device = torch.device("cuda:0")
    vla = OmniVLAEdge(args.prompt, device)
    print(f"OmniVLA-edge on {device}, prompt={args.prompt!r}", flush=True)
    run_node(vla, args)


if __name__ == "__main__":
    main()
