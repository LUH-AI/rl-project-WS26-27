"""Save one frame from the Go2 front camera."""

from __future__ import annotations

import argparse
import os
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iface",
        default=os.environ.get("GO2_IFACE", "enxf8e43b9f046a"),
        help="network interface connected to the robot",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            "training/outputs/frames", time.strftime("%Y%m%d-%H%M%S") + ".jpg"
        ),
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.iface)
    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()
    code, data = video.GetImageSample()
    if code != 0 or not data:
        raise SystemExit(f"GetImageSample failed (code={code})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(bytes(data))
    print(args.out)


if __name__ == "__main__":
    main()
