#!/bin/bash
# Stage 1 Go2 velocity PPO training on LUIS's `ai` partition, JAX/MJX path.
#
# Unlike Isaac Sim, JAX/MJX training is pure CUDA compute with no graphics
# dependency, so it runs fine on this partition's MIG-partitioned A100s
# (confirmed: nvidia-smi/CUDA_VISIBLE_DEVICES show a MIG slice, and Isaac
# Sim's Kit renderer cannot initialize on one at all — see
# scripts/train_go2_luis.slurm's header for that story).
#
# Submit from the login node:
#   ssh luis
#   cd /bigwork/nhwpchem/Repo/rl-project
#   sbatch scripts/train_go2_mjx_luis.slurm

#SBATCH -J go2-mjx
#SBATCH -p ai
#SBATCH --reservation ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -t 0-00:40:00
#SBATCH --mail-type fail,end
#SBATCH --mail-user m.che@ai.uni-hannover.de
#SBATCH --output logs/slurm/%x_%j.out
#SBATCH --error  logs/slurm/%x_%j.err

set -euo pipefail
source "$HOME/.local/bin/env"
cd /bigwork/nhwpchem/Repo/rl-project
source .venv-mjx/bin/activate
cd third_party/unitree_go2_mjx_rl

python -u scripts/train_from_configs.py configs/joystick_basic.yaml
