#!/usr/bin/env bash
# Install the VLA env (vla/.venv) and OmniVLA checkpoints, versions from vla/uv.lock
# Usage: ./scripts/install_vla.sh [--full]
#   --full also downloads the 7B checkpoint (~16GB)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OMNIVLA_DIR=third_party/OmniVLA
OMNIVLA_COMMIT=5182600cb4a9ee07684e17cdd2a6cbafc56b8a68

if [ ! -d "$OMNIVLA_DIR" ]; then
    git init -q "$OMNIVLA_DIR"
    git -C "$OMNIVLA_DIR" fetch -q --depth 1 https://github.com/NHirose/OmniVLA "$OMNIVLA_COMMIT"
    git -C "$OMNIVLA_DIR" checkout -q FETCH_HEAD
fi
# Fix missing comma in upstream pyproject
sed -i 's/^\(\s*\)"av"\s*$/\1"av",/' "$OMNIVLA_DIR/pyproject.toml"

uv sync --locked --project vla

# Download checkpoints
hf_get() { vla/.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('$1', local_dir='$2')"; }
hf_get NHirose/omnivla-edge "$OMNIVLA_DIR/omnivla-edge"
if [ "${1:-}" = "--full" ]; then
    hf_get NHirose/omnivla-original "$OMNIVLA_DIR/omnivla-original"
fi

echo "Done!"
