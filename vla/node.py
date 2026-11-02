"""Shared VLA loop: camera frame in, velocity command out."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
from PIL import Image

from command_bus.bus import CommandPublisher
from command_bus.frames import FrameSubscriber

OMNIVLA_DIR = os.path.join(os.path.dirname(__file__), "..", "third_party", "OmniVLA")
WAYPOINT_SCALE = 0.1
TIMESTEP = 1 / 3
TARGET_WAYPOINT = 4


def waypoint_to_velocity(
    waypoint: np.ndarray, max_v: float, max_w: float, speed_cap: float = 0.5
) -> tuple[float, float]:
    """Steer towards a waypoint, same as the OmniVLA example code."""
    x, y, cos_yaw, sin_yaw = waypoint
    if abs(x) < 1e-8 and abs(y) < 1e-8:
        vx, wz = 0.0, np.arctan2(sin_yaw, cos_yaw) / TIMESTEP
    elif abs(x) < 1e-8:
        vx, wz = 0.0, np.sign(y) * np.pi / (2 * TIMESTEP)
    else:
        vx, wz = x / TIMESTEP, np.arctan(y / x) / TIMESTEP
    vx, wz = float(np.clip(vx, 0.0, speed_cap)), float(np.clip(wz, -1.0, 1.0))

    # Scale both so the turning radius stays the same
    scale = min(1.0, max_v / max(abs(vx), 1e-8), max_w / max(abs(wz), 1e-8))
    return vx * scale, wz * scale


def read_prompts() -> queue.Queue:
    """Collect prompts typed into the terminal."""
    prompts: queue.Queue = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            if line.strip():
                prompts.put(line.strip())

    threading.Thread(target=read, daemon=True).start()
    return prompts


def update_prompt(vla, prompts: queue.Queue) -> None:
    prompt = None
    while not prompts.empty():
        prompt = prompts.get()
    if prompt is not None:
        vla.set_prompt(prompt)
        vla.reset()
        print(f">>> prompt now {prompt!r}", flush=True)


def run_node(vla, args: argparse.Namespace) -> None:
    """Run the VLA on the newest camera frame and send velocity commands."""
    camera = None if args.image else FrameSubscriber()
    still_image = Image.open(args.image).convert("RGB") if args.image else None
    bus = CommandPublisher("vla")
    prompts = read_prompts()
    print("Type a new prompt and press Enter to change the goal.", flush=True)

    try:
        while True:
            update_prompt(vla, prompts)
            step_start = time.time()

            # Get frame
            if still_image is not None:
                frame = still_image
            else:
                latest = camera.latest()
                if latest is None:
                    time.sleep(0.05)
                    continue
                frame_time, frame = latest
                if step_start - frame_time > 1.0:
                    print(f"Warning: frame is {step_start - frame_time:.1f}s old")

            # Predict and send command
            waypoints = vla.predict(frame)
            vx, wz = waypoint_to_velocity(
                waypoints[TARGET_WAYPOINT], args.max_v, args.max_w, args.speed_cap
            )
            bus.send(vx, 0.0, wz)
            step_ms = (time.time() - step_start) * 1e3
            print(f"vx={vx:+.2f} wz={wz:+.2f}  ({step_ms:.0f} ms)", flush=True)

            if args.once:
                print("Waypoints (m):\n", np.round(waypoints, 2))
                return
            time.sleep(max(0.0, 1 / args.rate - (time.time() - step_start)))
    finally:
        bus.stop()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="move toward the door")
    parser.add_argument("--image", help="still image instead of the camera")
    parser.add_argument("--once", action="store_true", help="run once and exit")
    parser.add_argument("--rate", type=float, default=3.0, help="Hz")
    parser.add_argument("--max-v", type=float, default=0.3)
    parser.add_argument("--max-w", type=float, default=0.3)
    parser.add_argument("--speed-cap", type=float, default=0.5)
    return parser
