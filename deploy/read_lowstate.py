"""Read-only smoke test: subscribe to the Go2's rt/lowstate and print it.

Never publishes anything, so it is safe in any robot pose / mode.

    .venv-deploy/bin/python deploy/read_lowstate.py enxf8e43b9f046a
"""

import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

# Go2 motor order in LowState_.motor_state[0:12]
JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

latest = None
count = 0


def handler(msg: LowState_):
    global latest, count
    latest = msg
    count += 1


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <network_interface>")
        sys.exit(1)

    ChannelFactoryInitialize(0, sys.argv[1])
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(handler, 10)

    last_count, last_t = 0, time.time()
    while True:
        time.sleep(1.0)
        now = time.time()
        rate = (count - last_count) / (now - last_t)
        last_count, last_t = count, now

        if latest is None:
            print("no rt/lowstate received yet (check interface / cable)")
            continue

        imu = latest.imu_state
        print(f"\n--- lowstate @ {rate:.0f} Hz | tick {latest.tick} ---")
        print(f"rpy [rad]   : {[round(x, 3) for x in imu.rpy]}")
        print(f"gyro [rad/s]: {[round(x, 3) for x in imu.gyroscope]}")
        print(f"acc [m/s^2] : {[round(x, 3) for x in imu.accelerometer]}")
        print(f"foot force  : {list(latest.foot_force)}")
        print(f"battery     : {latest.power_v:.2f} V, {latest.power_a:.2f} A, "
              f"SOC {latest.bms_state.soc}%")
        for i, name in enumerate(JOINT_NAMES):
            m = latest.motor_state[i]
            print(f"  {name:9s} q={m.q:+.3f} dq={m.dq:+.3f} "
                  f"tau={m.tau_est:+.2f} T={m.temperature}C")


if __name__ == "__main__":
    main()
