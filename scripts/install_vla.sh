#!/usr/bin/env bash
# Install the VLA env and OmniVLA checkpoints
# Usage: ./scripts/install_vla.sh [--full]
#   --full also downloads the 7B checkpoint (~16GB)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OMNIVLA_DIR=third_party/OmniVLA
VENV=highlevel/.venv

if [ ! -d "$OMNIVLA_DIR" ]; then
    git clone https://github.com/NHirose/OmniVLA "$OMNIVLA_DIR"
fi
# Fix missing comma in upstream pyproject
sed -i 's/^\(\s*\)"av"\s*$/\1"av",/' "$OMNIVLA_DIR/pyproject.toml"

if [ ! -d "$VENV" ]; then
    uv venv --python 3.10 --seed "$VENV"
fi
source "$VENV/bin/activate"

uv pip install numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
uv pip install -e "$OMNIVLA_DIR"
# openai-clip still needs pkg_resources
uv pip install "setuptools<81"
# TF 2.15 needs protobuf 4
uv pip install "tensorflow-metadata<1.17"
uv pip install --no-deps "protobuf==4.25.9"

# Download checkpoints
hf_get() { python -c "from huggingface_hub import snapshot_download as d; d('$1', local_dir='$2')"; }
hf_get NHirose/omnivla-edge "$OMNIVLA_DIR/omnivla-edge"
if [ "${1:-}" = "--full" ]; then
    hf_get NHirose/omnivla-original "$OMNIVLA_DIR/omnivla-original"
fi

echo "Done!"
