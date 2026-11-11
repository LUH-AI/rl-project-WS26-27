#!/usr/bin/env bash
# Install the robot env (deploy/.venv) with the Unitree SDK, versions from deploy/uv.lock
# Usage: ./scripts/install_deploy.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SDK_DIR=third_party/unitree_sdk2_python
SDK_COMMIT=814556d15970dd2ecf1c9984e845ca02ab07e206

if [ ! -d "$SDK_DIR" ]; then
    git init -q "$SDK_DIR"
    git -C "$SDK_DIR" fetch -q --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python "$SDK_COMMIT"
    git -C "$SDK_DIR" checkout -q FETCH_HEAD
fi
uv sync --locked --project deploy

echo "Done!"
