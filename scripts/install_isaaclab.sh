#!/usr/bin/env bash
# One-time heavy install for Stage 1: Isaac Sim + Isaac Lab into the
# project's uv-managed .venv. Re-run to repair/update.
#
# Requires: an NVIDIA GPU + driver, ~20GB free disk, Python 3.11 (pinned in
# pyproject.toml since Isaac Sim 5.x requires it).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d .venv ]; then
    uv venv --python 3.11.16 --seed .venv
fi
source .venv/bin/activate

# Isaac Sim + Isaac Lab from NVIDIA's pip index. This alone pulls the
# correctly matched torch/torchvision/rsl-rl/gymnasium/tensorboard versions
# for Isaac Sim 5.x, so nothing else needs installing separately first.
# Version pinned to match unitree_rl_lab's tested Isaac Lab 2.3.x / Isaac
# Sim 5.x requirement.
uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

# First import triggers an interactive Omniverse Kit EULA prompt; accept it
# once here so later runs (including headless training) don't block on it.
echo "Yes" | python -c "import isaaclab"

echo "Done. Verify with: isaacsim"
