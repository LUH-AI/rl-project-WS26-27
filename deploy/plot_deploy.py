"""Plot a real-robot policy log (from run_policy.py).

Writes three PNGs next to --out_prefix:
  <prefix>_overview.png  base state over time: projected gravity, gyro, action size, command
  <prefix>_joints.png    per-joint measured q vs policy target (Isaac order, 3x4 grid)
  <prefix>_vs_sim.png    per-dimension obs / action mean ± std, real vs sim
                         (only with --sim, a training/rollout_and_log.py .npz with obs/action)

    .venv/bin/python deploy/plot_deploy.py --real deploy_log.npz \
        --sim training/outputs/rollout_cmd0.npz --out_prefix deploy/plots/run
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REAL = "#2a78d6"     # categorical slot 1
SECOND = "#eb6834"   # categorical slot 2 (policy target / sim)
THIRD = "#1baf7a"    # categorical slot 3
REF = "#8a8a85"      # reference lines (default pose, zero)
INK = "#3d3d3a"
XYZ = [REAL, SECOND, THIRD]

LEGS = ["FL", "FR", "RL", "RR"]
PARTS = ["hip", "thigh", "calf"]
OBS_SLICES = {  # term -> (start, end, scale); matches policy.OBS_TERMS
    "base_ang_vel": (0, 3, 0.2),
    "projected_gravity": (3, 6, 1.0),
    "velocity_commands": (6, 9, 1.0),
    "joint_pos_rel": (9, 21, 1.0),
    "joint_vel_rel": (21, 33, 0.05),
    "last_action": (33, 45, 1.0),
}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": REF, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3dc",
    "grid.linewidth": 0.6, "lines.linewidth": 1.5, "legend.frameon": False,
})


def overview(d, path):
    t, obs, a = d["t"], d["obs"], d["action"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    panels = [
        ("projected gravity (body)", obs[:, 3:6], ["x", "y", "z"]),
        ("base ang. vel. [rad/s]", obs[:, 0:3] / 0.2, ["x", "y", "z"]),
        ("velocity command", obs[:, 6:9], ["vx", "vy", "wz"]),
    ]
    for ax, (title, y, labels) in zip(axes, panels):
        for i, lab in enumerate(labels):
            ax.plot(t, y[:, i], color=XYZ[i], label=lab)
        ax.set_ylabel(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax = axes[3]
    ax.plot(t, np.abs(a).max(1), color=REAL, label="max |action|")
    ax.plot(t, np.abs(a).mean(1), color=SECOND, label="mean |action|")
    ax.set_ylabel("raw action size")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Real robot: base state and action size", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def joints(d, default_q, path):
    t, q, tgt = d["t"], d["joint_pos"], d["target"]
    fig, axes = plt.subplots(3, 4, figsize=(13, 7), sharex=True)
    for p, part in enumerate(PARTS):
        for l, leg in enumerate(LEGS):
            i = 4 * p + l  # Isaac order: all hips, then thighs, then calves
            ax = axes[p, l]
            ax.plot(t, q[:, i], color=REAL, label="measured q")
            ax.plot(t, tgt[:, i], color=SECOND, label="policy target")
            if default_q is not None:
                ax.axhline(default_q[i], color=REF, ls="--", lw=1, label="training default")
            ax.set_title(f"{leg}_{part}", loc="left")
            if l == 0:
                ax.set_ylabel("[rad]")
            if p == 2:
                ax.set_xlabel("time [s]")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Joint positions: measured vs policy target", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def vs_sim(real, sim, path):
    names = [f"{leg}_{part}" for part in PARTS for leg in LEGS]
    rows = list(OBS_SLICES.items()) + [("action", None)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.0 * len(rows)))
    for ax, (term, sl) in zip(axes, rows):
        if sl is None:
            r, s = real["action"], sim["action"]
        else:
            a, b, scale = sl
            r, s = real["obs"][:, a:b] / scale, sim["obs"][:, a:b] / scale
        x = np.arange(r.shape[1])
        for data, color, dx, label in ((s, SECOND, 0.12, "sim"), (r, REAL, -0.12, "real")):
            ax.errorbar(x + dx, data.mean(0), yerr=data.std(0), fmt="o", ms=4,
                        color=color, elinewidth=1.5, capsize=0, label=label)
        ax.axhline(0, color=REF, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names if r.shape[1] == 12 else ["x", "y", "z"], fontsize=7)
        ax.set_title(f"{term} (unscaled), mean ± std", loc="left")
    axes[0].legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="run_policy.py log .npz")
    parser.add_argument("--sim", default=None, help="rollout_and_log.py .npz with obs/action")
    parser.add_argument("--policy", default="deploy/example_policies/go2_velocity.npz",
                        help="for the training default pose line")
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    real = np.load(args.real)
    default_q = np.load(args.policy)["default_joint_pos"] if os.path.exists(args.policy) else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)

    overview(real, f"{args.out_prefix}_overview.png")
    joints(real, default_q, f"{args.out_prefix}_joints.png")
    written = ["overview", "joints"]
    if args.sim:
        sim = np.load(args.sim)
        if "obs" not in sim.files:
            raise ValueError(f"{args.sim} has no obs/action; re-run training/rollout_and_log.py")
        vs_sim(real, sim, f"{args.out_prefix}_vs_sim.png")
        written.append("vs_sim")
    print("wrote " + ", ".join(f"{args.out_prefix}_{w}.png" for w in written))


if __name__ == "__main__":
    main()
