"""Shadow mode: run the Isaac Lab policy on live rt/lowstate, publish nothing.

The robot stays under Unitree's sport mode (standing, or walking with the
remote). Every policy step (50 Hz) builds the observation from the real
robot exactly as in training, runs the actor, and logs obs / action / target.
Use it to check joint mapping, observation ranges and action sanity before
the policy is ever allowed to command motors.

    deploy/.venv/bin/python deploy/shadow_policy.py enxf8e43b9f046a \
        --policy deploy/policies/go2_velocity.npz --cmd 0 0 0 --out shadow_log.npz
"""

import argparse
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

from go2_policy import Go2Policy

ISAAC_JOINT_NAMES = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]

latest = None


def handler(msg: LowState_):
    global latest
    latest = msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interface")
    parser.add_argument("--policy", default="deploy/policies/go2_velocity.npz")
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("VX", "VY", "WZ"), help="fixed velocity command fed to the policy")
    parser.add_argument("--out", default="shadow_log.npz")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds, 0 = until Ctrl+C")
    args = parser.parse_args()

    policy = Go2Policy(args.policy)
    cmd = np.array(args.cmd, dtype=np.float32)
    print(f"policy loaded, step_dt={policy.step_dt}s, cmd={cmd.tolist()} (publishing nothing)")

    ChannelFactoryInitialize(0, args.interface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(handler, 10)

    while latest is None:
        print("waiting for rt/lowstate ...")
        time.sleep(0.5)

    log = {k: [] for k in ("t", "obs", "action", "target", "joint_pos", "tick")}
    t0 = next_t = time.time()
    last_print = t0
    try:
        while args.duration <= 0 or time.time() - t0 < args.duration:
            msg = latest
            state = policy.read_state(msg)
            obs, action, target = policy.step(state, cmd)

            now = time.time()
            log["t"].append(now - t0)
            log["obs"].append(obs)
            log["action"].append(action)
            log["target"].append(target)
            log["joint_pos"].append(state["joint_pos"])
            log["tick"].append(msg.tick)

            if now - last_print >= 1.0:
                last_print = now
                err = target - state["joint_pos"]
                print(f"\n--- t={now - t0:5.1f}s | gravity_b={np.round(state['projected_gravity_b'], 3).tolist()} "
                      f"| |action|max={np.abs(action).max():.2f}")
                for i, name in enumerate(ISAAC_JOINT_NAMES):
                    print(f"  {name:9s} q={state['joint_pos'][i]:+.3f} target={target[i]:+.3f} "
                          f"(target-q={err[i]:+.3f}) action={action[i]:+.2f}")

            next_t += policy.step_dt
            time.sleep(max(0.0, next_t - time.time()))
    except KeyboardInterrupt:
        pass
    finally:
        np.savez(args.out, cmd=cmd, joint_names=np.array(ISAAC_JOINT_NAMES),
                 **{k: np.array(v) for k, v in log.items()})
        print(f"\nsaved {len(log['t'])} steps to {args.out}")


if __name__ == "__main__":
    main()
