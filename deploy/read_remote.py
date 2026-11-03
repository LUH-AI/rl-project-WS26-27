"""Read-only check of the remote parsing: prints buttons as they are pressed and
the stick values. Publishes nothing.

Run it while sport mode is OFF (e.g. after go2_stand_example.py), so the
button presses don't also trigger sport-mode actions on the robot.

    deploy/.venv/bin/python deploy/read_remote.py enxf8e43b9f046a
"""

import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

from remote import BUTTONS, Remote

latest = None


def handler(msg: LowState_):
    global latest
    latest = msg


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <network_interface>")
        sys.exit(1)
    ChannelFactoryInitialize(0, sys.argv[1])
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(handler, 10)

    remote = Remote()
    last_print = 0.0
    print("press buttons / move sticks (Ctrl+C to quit)")
    while True:
        time.sleep(0.02)
        if latest is None:
            continue
        remote.update(latest.wireless_remote)
        for b in BUTTONS:
            if remote.on_pressed(b):
                held = [h for h in BUTTONS if remote.pressed[h] and h != b]
                print(f"pressed {b}" + (f" (holding {'+'.join(held)})" if held else ""))
        now = time.time()
        if now - last_print > 0.5:
            last_print = now
            print(f"  sticks: lx={remote.lx:+.2f} ly={remote.ly:+.2f} rx={remote.rx:+.2f} ry={remote.ry:+.2f}")


if __name__ == "__main__":
    main()
