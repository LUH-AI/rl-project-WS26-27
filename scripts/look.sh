#!/usr/bin/env bash
# Grab a camera frame from the Go2 and label the objects in it
# Usage: ./scripts/look.sh [--queries "red door,trash bin"]
set -euo pipefail
cd "$(dirname "$0")/.."

view=training/outputs/frames/view.jpg
deploy/.venv/bin/python deploy/grab_frame.py --out "$view" >/dev/null
vla/.venv/bin/python -m vla.detect_objects "$view" --out "$view" "$@" 2>/dev/null
