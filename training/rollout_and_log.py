"""Roll out a trained Stage 1 checkpoint headless (no camera / no RTX renderer)
and log the joint + base trajectory to .npz, for offline rendering in MuJoCo.

Run from third_party/unitree_rl_lab (needs its cli_args.py on sys.path) with
the project venv active, e.g.:

    cd third_party/unitree_rl_lab
    python ../../training/rollout_and_log.py --task Unitree-Go2-Velocity \
        --num_envs 1 --steps 400 --out ../../training/outputs/rollout.npz
"""

import argparse
import os
import sys

# make third_party/unitree_rl_lab/scripts/rsl_rl's cli_args.py importable
_UNITREE_RL_LAB_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "unitree_rl_lab", "scripts", "rsl_rl"
)
sys.path.insert(0, _UNITREE_RL_LAB_SCRIPTS)

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Roll out a trained policy and log its trajectory.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="training/outputs/rollout.npz")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"),
                    help="fix the velocity command (e.g. 0 0 0 to compare with real shadow-mode logs)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.cmd is not None:
        vx, vy, wz = args_cli.cmd
        cmd_cfg = env_cfg.commands.base_velocity
        cmd_cfg.rel_standing_envs = 0.0
        for ranges in (cmd_cfg.ranges, getattr(cmd_cfg, "limit_ranges", None)):
            if ranges is not None:
                ranges.lin_vel_x = (vx, vx)
                ranges.lin_vel_y = (vy, vy)
                ranges.ang_vel_z = (wz, wz)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    env_origin = env.unwrapped.scene.env_origins[0].cpu().numpy()

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    joint_pos_log = []
    root_pos_log = []
    root_quat_log = []
    command_log = []
    obs_log = []
    action_log = []

    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = policy(obs)
            policy_obs = obs["policy"] if hasattr(obs, "keys") else obs
            obs_log.append(policy_obs[0].cpu().numpy().copy())
            action_log.append(actions[0].cpu().numpy().copy())
            obs, _, _, _ = env.step(actions)

            joint_pos_log.append(robot.data.joint_pos[0].cpu().numpy().copy())
            root_pos_log.append((robot.data.root_pos_w[0].cpu().numpy() - env_origin).copy())
            root_quat_log.append(robot.data.root_quat_w[0].cpu().numpy().copy())
            command_log.append(
                env.unwrapped.command_manager.get_command("base_velocity")[0].cpu().numpy().copy()
            )

    np.savez(
        args_cli.out,
        joint_names=np.array(joint_names),
        joint_pos=np.stack(joint_pos_log),
        root_pos=np.stack(root_pos_log),
        root_quat=np.stack(root_quat_log),
        command=np.stack(command_log),
        obs=np.stack(obs_log),
        action=np.stack(action_log),
        dt=env.unwrapped.step_dt,
    )
    print(f"[INFO] Saved {args_cli.steps}-step trajectory to {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
