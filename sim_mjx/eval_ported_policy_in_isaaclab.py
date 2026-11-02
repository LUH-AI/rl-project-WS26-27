"""Evaluate an MJX-trained (and ported-to-PyTorch) Go2 policy inside Isaac
Lab, to measure the sim2sim transfer gap. Reconstructs MJX's exact
observation (15-frame history of the 31-dim proprioceptive frame) from Isaac
Lab's simulation state, remaps between MJX's and Isaac Lab's joint orderings,
and overrides the robot's default pose to MJX's (0, 0.9, -1.8 per leg)
so the comparison isn't confounded by an arbitrary default-pose mismatch.

Run from third_party/unitree_rl_lab:
    python ../../sim_mjx/eval_ported_policy_in_isaaclab.py \
        --ported_policy ../../sim_mjx/ported_policy.pt --num_envs 256 --steps 500
"""

import argparse
import os
import sys

_UNITREE_RL_LAB_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "unitree_rl_lab", "scripts", "rsl_rl"
)
sys.path.insert(0, _UNITREE_RL_LAB_SCRIPTS)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ported_policy", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

MJX_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
MJX_DEFAULT_POSE = {  # from resources/unitree_go2/scene_mjx_vision.xml <keyframe> "home"
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
}


class PortedPolicy(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        self.register_buffer("obs_mean", torch.zeros(layer_sizes[0]))
        self.register_buffer("obs_std", torch.ones(layer_sizes[0]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / self.obs_std
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < len(self.linears) - 1:
                x = torch.nn.functional.silu(x)
        loc, _scale = torch.chunk(x, 2, dim=-1)
        return torch.tanh(loc)


def main():
    ckpt = torch.load(args_cli.ported_policy, map_location="cuda:0")
    policy = PortedPolicy(ckpt["layer_sizes"]).to("cuda:0")
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    env_cfg = parse_env_cfg("Unitree-Go2-Velocity-MJX-Matched", device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.scene.robot.init_state.joint_pos = MJX_DEFAULT_POSE
    env = gym.make("Unitree-Go2-Velocity-MJX-Matched", cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    robot = env.unwrapped.scene["robot"]
    isaac_names = list(robot.data.joint_names)
    isaac_to_mjx = torch.tensor([isaac_names.index(n) for n in MJX_ORDER], device="cuda:0")
    mjx_to_isaac = torch.tensor([MJX_ORDER.index(n) for n in isaac_names], device="cuda:0")

    num_envs = args_cli.num_envs
    history = torch.zeros(num_envs, 31 * 15, device="cuda:0")
    last_action_mjx_order = torch.zeros(num_envs, 12, device="cuda:0")

    sum_err_xy = torch.zeros(num_envs, device="cuda:0")
    sum_err_yaw = torch.zeros(num_envs, device="cuda:0")
    survived_steps = torch.zeros(num_envs, device="cuda:0")
    alive = torch.ones(num_envs, dtype=torch.bool, device="cuda:0")
    cmd_scale = torch.tensor([2.0, 2.0, 0.25], device="cuda:0")

    with torch.inference_mode():
        for step_idx in range(args_cli.steps):
            if step_idx % 100 == 0 or step_idx >= 390:
                print(f"PROGRESS step {step_idx}/{args_cli.steps}", flush=True)
            command = env.unwrapped.command_manager.get_command("base_velocity")  # (N,3) vx,vy,yaw_rate
            ang_vel_b = robot.data.root_ang_vel_b[:, 2:3] * 0.25
            grav_b = robot.data.projected_gravity_b
            joint_pos_rel_mjx = (robot.data.joint_pos - robot.data.default_joint_pos)[:, isaac_to_mjx]

            frame = torch.cat(
                [command * cmd_scale, ang_vel_b, grav_b, joint_pos_rel_mjx, last_action_mjx_order],
                dim=-1,
            )  # (N, 31)
            history = torch.roll(history, shifts=31, dims=1)
            history[:, :31] = frame

            action_mjx_order = policy(history)  # (N, 12), tanh-squashed, MJX joint order
            last_action_mjx_order = action_mjx_order
            action_isaac_order = action_mjx_order[:, mjx_to_isaac]

            _obs, _rew, terminated, truncated, _info = env.step(action_isaac_order)
            done = terminated | truncated

            lin_vel_xy = robot.data.root_lin_vel_b[:, :2]
            err_xy = torch.norm(lin_vel_xy - command[:, :2], dim=-1)
            err_yaw = torch.abs(robot.data.root_ang_vel_b[:, 2] - command[:, 2])
            sum_err_xy += torch.where(alive, err_xy, torch.zeros_like(err_xy))
            sum_err_yaw += torch.where(alive, err_yaw, torch.zeros_like(err_yaw))
            survived_steps += alive.float()
            alive = alive & (~done)
            # no early-exit .any() check here: it forces a GPU sync every step,
            # which was the actual cause of this loop being far slower than training

    mean_err_xy = (sum_err_xy / survived_steps.clamp(min=1)).mean().item()
    mean_err_yaw = (sum_err_yaw / survived_steps.clamp(min=1)).mean().item()
    results = {
        "mean_lin_vel_xy_tracking_error": mean_err_xy,
        "mean_yaw_rate_tracking_error": mean_err_yaw,
        "mean_survived_steps": survived_steps.mean().item(),
        "fraction_alive_at_end": alive.float().mean().item(),
        "steps": args_cli.steps,
        "num_envs": num_envs,
    }
    # Write to disk immediately, before any cleanup call that might hang the
    # process (observed: env.close()/Kit shutdown can hang and get killed by
    # an external timeout, silently dropping unflushed stdout).
    import json

    results_path = os.path.join(os.path.dirname(os.path.abspath(args_cli.ported_policy)), "transfer_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("=== MJX policy transferred into Isaac Lab ===", flush=True)
    print(f"Mean lin-vel-xy tracking error: {mean_err_xy:.4f} (native MJX eval: ~0.775-0.97 depending on phase)", flush=True)
    print(f"Mean yaw-rate tracking error:   {mean_err_yaw:.4f}", flush=True)
    print(f"Mean survived steps / {args_cli.steps}: {results['mean_survived_steps']:.1f}", flush=True)
    print(f"Fraction of envs alive at end:  {results['fraction_alive_at_end']:.3f}", flush=True)
    print(f"Results written to {results_path}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    simulation_app.close()
