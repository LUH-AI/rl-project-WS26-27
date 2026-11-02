"""OmniVLA-edge, run with python -m vla.omnivla_edge."""

from __future__ import annotations

import collections
import os
import sys

import numpy as np
import torch
from PIL import Image

from vla.node import OMNIVLA_DIR, WAYPOINT_SCALE, get_parser, run_node

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
