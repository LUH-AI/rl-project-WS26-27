"""OmniVLA 7B node."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

from highlevel.omnivla_edge_node import (
    OMNIVLA_DIR,
    WAYPOINT_SCALE,
    get_parser,
    run_node,
)

sys.path[:0] = [OMNIVLA_DIR, os.path.join(OMNIVLA_DIR, "inference")]

import run_omnivla as upstream  # noqa: E402
from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.training.train_utils import (  # noqa: E402
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402

LANGUAGE_GOAL = 7


class OmniVLA7B:
    def __init__(self, prompt: str):
        cfg = upstream.InferenceConfig()
        cfg.vla_path = os.path.join(OMNIVLA_DIR, "omnivla-original")
        (
            self.vla,
            self.action_head,
            self.pose_projector,
            self.device,
            self.num_patches,
            tokenizer,
            processor,
        ) = upstream.define_model(cfg)
        self.vla.eval()
        self.action_head.eval()
        self.pose_projector.eval()

        # Only used to build the model inputs, so the goal fields stay empty
        self.builder = upstream.Inference(
            save_dir=None,
            lan_inst_prompt=prompt,
            goal_utm=None,
            goal_compass=0.0,
            goal_image_PIL=Image.new("RGB", (224, 224)),
            action_tokenizer=tokenizer,
            processor=processor,
        )
        self.set_prompt(prompt)

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def predict(self, frame: Image.Image) -> np.ndarray:
        """Predict 8 waypoints (x, y, cos, sin) in metres."""
        builder = self.builder
        batch = builder.data_transformer_omnivla(
            frame,
            self.prompt,
            builder.goal_image_PIL,
            np.zeros(4),
            prompt_builder=PurePromptBuilder,
            action_tokenizer=builder.action_tokenizer,
            processor=builder.processor,
        )
        modality = torch.tensor(
            [LANGUAGE_GOAL], dtype=torch.bfloat16, device=self.device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vla(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device),
                modality_id=modality,
                labels=batch["labels"].to(self.device),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(self.device),
                proprio_projector=self.pose_projector,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )

        # Get the hidden states of the action tokens
        labels = batch["labels"][:, 1:].to(self.device)
        mask = get_current_action_mask(labels) | get_next_actions_mask(labels)
        text_hidden = outputs.hidden_states[-1][:, self.num_patches : -1]
        action_hidden = (
            text_hidden[mask]
            .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )
        actions = self.action_head.predict_action(action_hidden, modality)
        waypoints = actions[0].float().cpu().numpy()
        waypoints[:, :2] *= WAYPOINT_SCALE
        return waypoints


def main() -> None:
    args = get_parser().parse_args()
    if not torch.cuda.is_available():
        sys.exit("OmniVLA 7B needs a CUDA GPU.")
    print(f"Loading OmniVLA 7B, prompt={args.prompt!r}")
    vla = OmniVLA7B(args.prompt)
    print(f"OmniVLA 7B on {vla.device}, prompt={args.prompt!r}", flush=True)
    run_node(vla, args)


if __name__ == "__main__":
    main()
