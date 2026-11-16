"""Show or restore the Go2's sport mode, e.g. after run_policy.py released it.

    deploy/.venv/bin/python deploy/sport_mode.py        # show the current mode
    deploy/.venv/bin/python deploy/sport_mode.py --on   # switch sport mode back on, no reboot needed

Needs GO2_IFACE (or --iface). Switching sport mode on can make the robot stand up,
so keep ~1 m clear around it.
"""

import argparse
import os
import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def current_mode(msc: MotionSwitcherClient) -> str:
    code, result = msc.CheckMode()
    if code != 0:
        raise SystemExit(f"CheckMode failed with code {code}, is the robot connected?")
    return result["name"]


MODE_NAMES = ["mcf", "normal"]  # mcf on our Go2 firmware, normal on older ones


def switch_on(msc: MotionSwitcherClient, names: list[str] = MODE_NAMES) -> str:
    """Select the first accepted mode name, return the active mode or "" on failure."""
    for name in names:
        code, _ = msc.SelectMode(name)
        print(f"SelectMode('{name}'): code {code}")
        if code == 0:
            break
    else:
        return ""
    for _ in range(20):
        time.sleep(0.5)
        mode = current_mode(msc)
        if mode:
            return mode
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iface",
        default=os.environ.get("GO2_IFACE"),
        required="GO2_IFACE" not in os.environ,
        help="network interface connected to the robot (default: $GO2_IFACE)",
    )
    parser.add_argument("--on", action="store_true", help="switch sport mode back on")
    parser.add_argument(
        "--name",
        nargs="+",
        default=MODE_NAMES,
        help="mode names to try in order, the name depends on the firmware",
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.iface)
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    mode = current_mode(msc)
    print(f"current mode: {mode or 'none (sport mode off)'}")
    if not args.on or mode:
        return

    input("Press Enter to switch sport mode on, the robot may stand up (~1 m clear) ...")
    mode = switch_on(msc, args.name)
    if not mode:
        raise SystemExit("sport mode did not come back, reboot the robot")
    print(f"sport mode on: {mode}")


if __name__ == "__main__":
    main()
