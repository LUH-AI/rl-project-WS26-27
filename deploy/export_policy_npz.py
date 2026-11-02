"""Export an Isaac Lab (rsl_rl) Go2 checkpoint + its deploy.yaml to a single .npz.

The .npz holds the actor MLP weights and everything needed to build
observations / apply actions on the real robot, so the deploy side only needs
numpy (no torch, no yaml). Run with the training env (.venv):

    .venv/bin/python deploy/export_policy_npz.py \
        --run_dir third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-09-02_14-23-20 \
        --checkpoint model_1999.pt \
        --out deploy/policies/go2_velocity.npz
"""

import argparse
import os

import numpy as np
import torch
import yaml

OBS_TERMS = [
    "base_ang_vel", "projected_gravity", "velocity_commands",
    "joint_pos_rel", "joint_vel_rel", "last_action",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="model_1999.pt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "params", "deploy.yaml")) as f:
        cfg = yaml.safe_load(f)

    obs_cfg = cfg["observations"]
    if list(obs_cfg.keys()) != OBS_TERMS:
        raise ValueError(f"unexpected observation terms: {list(obs_cfg.keys())}")
    for name, term in obs_cfg.items():
        if term["history_length"] != 1:
            raise ValueError(f"{name}: history_length != 1 not supported")

    ckpt = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    if any("normalizer" in k for k in sd):
        raise ValueError("checkpoint has an obs normalizer; export it too")

    out = {}
    layer_ids = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.")})
    for n, i in enumerate(layer_ids):
        out[f"W{n}"] = sd[f"actor.{i}.weight"].numpy()
        out[f"b{n}"] = sd[f"actor.{i}.bias"].numpy()
    out["num_layers"] = np.array(len(layer_ids))

    action = cfg["actions"]["JointPositionAction"]
    out["joint_ids_map"] = np.array(cfg["joint_ids_map"], dtype=np.int64)
    out["default_joint_pos"] = np.array(cfg["default_joint_pos"], dtype=np.float32)
    out["stiffness"] = np.array(cfg["stiffness"], dtype=np.float32)
    out["damping"] = np.array(cfg["damping"], dtype=np.float32)
    out["step_dt"] = np.array(cfg["step_dt"], dtype=np.float32)
    out["action_scale"] = np.array(action["scale"], dtype=np.float32)
    out["action_offset"] = np.array(action["offset"], dtype=np.float32)
    out["action_clip"] = np.array(action["clip"], dtype=np.float32)
    for name in OBS_TERMS:
        out[f"obs_scale_{name}"] = np.array(obs_cfg[name]["scale"], dtype=np.float32)
        out[f"obs_clip_{name}"] = np.array(obs_cfg[name]["clip"], dtype=np.float32)
    ranges = cfg["commands"]["base_velocity"]["ranges"]
    out["cmd_ranges"] = np.array(
        [ranges["lin_vel_x"], ranges["lin_vel_y"], ranges["ang_vel_z"]], dtype=np.float32
    )

    # Reference input/output pair to check the numpy forward pass against torch.
    actor = torch.nn.Sequential(*[
        m for i in layer_ids for m in (
            torch.nn.Linear(*sd[f"actor.{i}.weight"].shape[::-1]),
            torch.nn.ELU(),
        )
    ][:-1])
    actor.load_state_dict({
        f"{2 * n}.{p}": sd[f"actor.{i}.{p}"] for n, i in enumerate(layer_ids) for p in ("weight", "bias")
    })
    x = torch.randn(4, out["W0"].shape[1])
    with torch.no_grad():
        out["test_input"] = x.numpy()
        out["test_output"] = actor(x).numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}: obs_dim={out['W0'].shape[1]} act_dim={out[f'W{len(layer_ids) - 1}'].shape[0]}")


if __name__ == "__main__":
    main()
