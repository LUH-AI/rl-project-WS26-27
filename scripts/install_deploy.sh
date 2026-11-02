#!/usr/bin/env bash
# Install the robot env with the Unitree SDK
# Usage: ./scripts/install_deploy.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SDK_DIR=third_party/unitree_sdk2_python
VENV=deploy/.venv

if [ ! -d "$SDK_DIR" ]; then
    git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python "$SDK_DIR"
fi
if [ ! -d "$VENV" ]; then
    uv venv --python 3.10 "$VENV"
fi
VIRTUAL_ENV="$VENV" uv pip install "cyclonedds==0.10.2" -e "$SDK_DIR" opencv-python-headless pillow

echo "Done!"
