#!/bin/bash
# Stage 1 Go2 velocity PPO training on the LUIS cluster's `ai` partition.
#
# Isaac Sim's official pip wheels need glibc>=2.35; this cluster runs Rocky
# Linux 9.x (glibc 2.34) on both login and `ai`-partition GPU nodes, so this
# uses the last glibc-2.34-compatible release line instead:
# isaaclab==2.1.0 / isaacsim==4.5.0.0 (Python 3.10), vs. 2.3.2.post1 / 5.1.0
# (Python 3.11) used on the local RTX 4090 workstation. unitree_rl_lab
# targets Isaac Lab 2.3.0 APIs, so third_party/unitree_rl_lab needed three
# small patches to run on 2.1.0 (ray_alignment -> attach_yaw_only, a
# joint_effort observation backport, and merge_fixed_joints=False on the
# Go2 URDF spawn config, since Isaac Sim 4.5's URDF importer collapses the
# *_foot links that 5.1's doesn't) — see git history / diffs on the cluster
# copy of third_party/unitree_rl_lab for the exact changes. A NGC-container
# (Apptainer) path that would let this track the same 2.3.x/5.1 versions as
# local is documented as a backup if this compatibility shim ever breaks.
#
# Submit from the login node:
#   ssh luis
#   cd /bigwork/nhwpchem/Repo/rl-project
#   sbatch scripts/train_go2_luis.slurm

#SBATCH -J go2-stage1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH -t 0-00:30:00
#SBATCH --mail-type fail,end
#SBATCH --mail-user m.che@ai.uni-hannover.de
#SBATCH --output logs/slurm/%x_%j.out
#SBATCH --error  logs/slurm/%x_%j.err

set -euo pipefail
cd /bigwork/nhwpchem/Repo/rl-project
source .venv/bin/activate
cd third_party/unitree_rl_lab

python scripts/rsl_rl/train.py \
    --task Unitree-Go2-Velocity --headless \
    --num_envs 1024 --max_iterations 200 --seed 42 --logger tensorboard
