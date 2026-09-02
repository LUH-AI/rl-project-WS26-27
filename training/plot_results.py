"""Plot PPO training curves from an rsl_rl / tensorboard run directory.

    python training/plot_results.py --run_dir third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/<run> \
        --out training/outputs/training_curves.png
"""

import argparse

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ACCENT = "#2a78d6"

PANELS = [
    ("Train/mean_reward", "Mean episode reward", "reward"),
    ("Train/mean_episode_length", "Mean episode length", "steps"),
    ("Metrics/base_velocity/error_vel_xy", "Linear velocity tracking error", "m/s"),
    ("Metrics/base_velocity/error_vel_yaw", "Yaw-rate tracking error", "rad/s"),
    ("Episode_Reward/track_lin_vel_xy", "Reward: track_lin_vel_xy", "reward"),
    ("Curriculum/terrain_levels", "Terrain curriculum level", "level"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

ea = EventAccumulator(args.run_dir)
ea.Reload()
available = set(ea.Tags()["scalars"])

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
fig.suptitle("Go2 Velocity Policy — PPO Training (Stage 1)", fontsize=14, fontweight="bold")

for ax, (tag, title, ylabel) in zip(axes.flat, PANELS):
    if tag not in available:
        ax.set_visible(False)
        continue
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    ax.plot(steps, values, color=ACCENT, linewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(args.out, dpi=150)
print(f"[INFO] Saved training curves to {args.out}")
