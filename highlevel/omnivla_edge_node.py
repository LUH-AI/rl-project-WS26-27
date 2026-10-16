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

# Config of the released omnivla-edge checkpoint
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
LANGUAGE_ONLY = 7
WAYPOINT_SPACING = 0.1
DT = 1 / 3
WAYPOINT_INDEX = 4


def waypoint_to_velocity(
    wp: np.ndarray, max_v: float, max_w: float, v_cap: float = 0.5
) -> tuple[float, float]:
    """Convert a waypoint to (v, w), keeping the turn radius when clipping."""
    dx, dy, hx, hy = wp
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        v, w = 0.0, np.arctan2(hy, hx) / DT
    elif abs(dx) < 1e-8:
        v, w = 0.0, np.sign(dy) * np.pi / (2 * DT)
    else:
        v, w = dx / DT, np.arctan(dy / dx) / DT
    v, w = float(np.clip(v, 0.0, v_cap)), float(np.clip(w, -1.0, 1.0))
    scale = min(1.0, max_v / max(abs(v), 1e-8), max_w / max(abs(w), 1e-8))
    return v * scale, w * scale


def prompt_reader() -> queue.Queue:
    """Read new prompts from stdin in the background."""
    prompts: queue.Queue = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            if line.strip():
                prompts.put(line.strip())

    threading.Thread(target=read, daemon=True).start()
    return prompts


def apply_new_prompt(vla, prompts: queue.Queue) -> None:
    new = None
    while not prompts.empty():
        new = prompts.get()
    if new is not None:
        vla.set_prompt(new)
        vla.reset()
        print(f">>> prompt now {new!r}", flush=True)


class OmniVLAEdge:
    def __init__(self, prompt: str, device: torch.device):
        self.device = device
        ckpt = os.path.join(OMNIVLA_DIR, "omnivla-edge", "omnivla-edge.pth")
        self.model, text_encoder, _ = load_model(ckpt, MODEL_PARAMS, device)
        self.model = self.model.to(device).eval()
        self.text_encoder = text_encoder.to(device).eval()
        self.ones96 = np.ones((96, 96, 3), dtype=np.float32)
        self.ones224 = np.ones((224, 224, 3), dtype=np.float32)
        self.context = collections.deque(maxlen=MODEL_PARAMS["context_size"] + 1)
        self.set_prompt(prompt)

        # Placeholders for the goal types we don't use
        black = Image.new("RGB", (352, 352))
        self.map_pair = torch.cat([transform_images_map(black)] * 2, dim=1).to(device)
        self.goal_image = transform_images_PIL_mask(
            Image.new("RGB", (96, 96)), self.ones96
        ).to(device)
        self.goal_pose = torch.zeros(1, 4, device=device)

    @torch.no_grad()
    def set_prompt(self, prompt: str) -> None:
        tokens = clip.tokenize(prompt, truncate=True).to(self.device)
        self.text_feat = self.text_encoder.encode_text(tokens)

    def reset(self) -> None:
        self.context.clear()

    @torch.no_grad()
    def predict(self, frame: Image.Image) -> np.ndarray:
        """Predict 8 waypoints (x, y, cos, sin) in metres."""
        self.context.append(frame.resize((96, 96)))
        while len(self.context) < self.context.maxlen:
            self.context.appendleft(self.context[0])
        obs = transform_images_PIL_mask(list(self.context), self.ones96).to(self.device)
        obs_cur = torch.split(obs, 3, dim=1)[-1]
        cur_large = transform_images_PIL_mask(
            frame.resize((224, 224)), self.ones224
        ).to(self.device)
        actions, _, _ = self.model(
            obs,
            self.goal_pose,
            torch.cat([self.map_pair, obs_cur], dim=1),
            self.goal_image,
            torch.tensor([LANGUAGE_ONLY], device=self.device),
            self.text_feat,
            cur_large,
        )
        wps = actions[0].float().cpu().numpy()
        wps[:, :2] *= WAYPOINT_SPACING
        return wps


def run_node(vla, args: argparse.Namespace) -> None:
    """Run the VLA on the newest frame and publish velocity commands."""
    frames = None if args.image else FrameSubscriber()
    still = Image.open(args.image).convert("RGB") if args.image else None
    bus = CommandPublisher("highlevel")
    prompts = prompt_reader()
    print("Type a new prompt and press Enter to change the goal.", flush=True)

    try:
        while True:
            apply_new_prompt(vla, prompts)
            t0 = time.time()

            # Get frame
            if still is not None:
                frame = still
            else:
                got = frames.latest()
                if got is None:
                    time.sleep(0.05)
                    continue
                stamp, frame = got
                if t0 - stamp > 1.0:
                    print(f"Warning: frame is {t0 - stamp:.1f}s old")

            # Predict and send command
            wps = vla.predict(frame)
            v, w = waypoint_to_velocity(
                wps[WAYPOINT_INDEX], args.max_v, args.max_w, args.speed_cap
            )
            bus.send(v, 0.0, w)
            print(
                f"vx={v:+.2f} wz={w:+.2f}  ({(time.time() - t0) * 1e3:.0f} ms)",
                flush=True,
            )
            if args.once:
                print("Waypoints (m):\n", np.round(wps, 2))
                return
            time.sleep(max(0.0, 1 / args.rate - (time.time() - t0)))
    finally:
        bus.stop()


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="move toward the door")
    p.add_argument("--image", help="still image instead of the camera")
    p.add_argument("--once", action="store_true", help="run once and exit")
    p.add_argument("--rate", type=float, default=3.0, help="inference rate in Hz")
    p.add_argument("--max-v", type=float, default=0.3)
    p.add_argument("--max-w", type=float, default=0.3)
    p.add_argument("--speed-cap", type=float, default=0.5)
    return p


def main() -> None:
    args = get_parser().parse_args()

    # The upstream model only runs on GPU
    if not torch.cuda.is_available():
        sys.exit("OmniVLA-edge needs a CUDA GPU.")
    device = torch.device("cuda:0")
    vla = OmniVLAEdge(args.prompt, device)
    print(f"OmniVLA-edge on {device}, prompt={args.prompt!r}", flush=True)
    run_node(vla, args)


if __name__ == "__main__":
    main()
