"""Closed-loop Go2 deployment of the Isaac Lab velocity policy over rt/lowcmd.

Python port of unitree_rl_lab's C++ Go2 FSM (deploy/robots/go2/config/config.yaml):

  PASSIVE  kp=0, kd=3 (damping)       --[L2+A] or key 's'-->  STAND
  STAND    current -> folded -> training default pose (kp 60/80, kd 5/4)
                                        --[start] or key 'p'-->  POLICY
  POLICY   policy at 50 Hz, kp=25, kd=0.5 (from deploy.yaml), sticks -> velocity cmd
  SIT      controlled lie-down to the folded pose (stand gains), rear legs lead so
           the nose stays up, then PASSIVE    <--[L2+X] or key 'l' from STAND/POLICY
  any      --[L2+B] or key 'd', lowstate timeout, or tilt > 1 rad (POLICY)--> PASSIVE

Quitting ('q' / Ctrl+C) from STAND or POLICY lies down through SIT first.

Motor commands are written at 500 Hz; the policy steps every 10th tick.

Start with the robot LYING DOWN on the floor: unless --dry_run, the script
releases sport mode (StandDown + ReleaseMode), after which the legs go limp.

    # dry run: full FSM + policy, publishes nothing, sport mode untouched
    deploy/.venv/bin/python deploy/run_policy.py enxf8e43b9f046a --dry_run

    # real
    deploy/.venv/bin/python deploy/run_policy.py enxf8e43b9f046a --cmd_scale 0.3

    # velocity commands from the command bus (e.g. the VLA), sticks still override
    deploy/.venv/bin/python deploy/run_policy.py enxf8e43b9f046a --cmd_source bus --bus_max_v 0.3
"""

import argparse
import os
import select
import sys
import threading
import time

import numpy as np
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from go2_policy import Go2Policy
from go2_remote import Remote

# command_bus/ lives in the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from command_bus.bus import CommandArbiter  # noqa: E402

CTRL_DT = 0.002
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

# SDK order: FR, FL, RR, RL x (hip, thigh, calf)
PASSIVE_KD = 3.0
STAND_KP = np.array([60, 80, 80] * 4, dtype=np.float32)
STAND_KD = np.array([5, 4, 4] * 4, dtype=np.float32)
FOLDED_Q = np.array([0.0, 1.36, -2.65] * 4, dtype=np.float32)
# Intersection of front/rear URDF limits, slightly inside the hard stops.
Q_MIN = np.array([-1.0, -0.5, -2.7] * 4, dtype=np.float32)
Q_MAX = np.array([1.0, 3.4, -0.85] * 4, dtype=np.float32)

LOWSTATE_TIMEOUT = 0.1  # s
TILT_LIMIT = 1.0  # rad, matches unitree_rl_lab's bad_orientation(1.0)
STICK_DEADZONE = 0.05
# Fraction of the sit-down by which the rear legs are fully folded; the front
# legs take the whole duration, so the rear touches down first (nose up).
SIT_REAR_LEAD = 0.7
SIT_HOLD = 0.5  # s at the folded pose before switching to damping


class LowStateBuffer:
    def __init__(self):
        self.msg = None
        self.t = 0.0

    def handler(self, msg: LowState_):
        self.msg = msg
        self.t = time.time()


class KeyboardInput:
    """Non-blocking single-key commands from stdin (s / p / d / q + Enter)."""

    def __init__(self):
        self.keys = []
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                line = sys.stdin.readline()
                if not line:
                    return
                with self._lock:
                    self.keys.extend(line.strip().lower())

    def pop(self):
        with self._lock:
            keys, self.keys = self.keys, []
        return keys


class Controller:
    def __init__(self, policy, lowstate, dry_run, cmd_scale, stand_time, sit_time,
                 arbiter=None, bus_max_v=0.3, bus_max_w=0.5):
        self.policy = policy
        self.lowstate = lowstate
        self.dry_run = dry_run
        self.cmd_scale = cmd_scale
        self.stand_time = stand_time
        self.sit_time = sit_time
        self.remote = Remote()
        self.arbiter = arbiter
        self.bus_max = np.array([bus_max_v, bus_max_v, bus_max_w], dtype=np.float32)
        self.cmd_from = "sticks"
        self.keyboard = KeyboardInput()
        self.crc = CRC()

        self.default_q_sdk = policy.to_sdk_order(policy.default_joint_pos)
        self.policy_kp_sdk = policy.to_sdk_order(policy.stiffness)
        self.policy_kd_sdk = policy.to_sdk_order(policy.damping)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            m = self.low_cmd.motor_cmd[i]
            m.mode = 0x01
            m.q, m.dq, m.kp, m.kd, m.tau = POS_STOP_F, VEL_STOP_F, 0.0, 0.0, 0.0

        self.publisher = None
        if not dry_run:
            self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.publisher.Init()

        self.state = None
        self.tick = 0
        self.cmd = np.zeros(3, dtype=np.float32)
        self.q_target_sdk = np.zeros(12, dtype=np.float32)
        self.log = {k: [] for k in ("t", "obs", "action", "target", "joint_pos", "tick")}
        self.t0 = time.time()
        self.running = True
        self.enter("PASSIVE", "start")

    # ---- state machine -------------------------------------------------
    def enter(self, state, reason):
        print(f"[{time.time() - self.t0:7.2f}s] -> {state} ({reason})", flush=True)
        self.state = state
        self.state_t0 = time.time()
        if state in ("STAND", "SIT"):
            self.start_q = self.current_q_sdk()
        elif state == "POLICY":
            self.policy.last_action[:] = 0.0
            self.policy_tick = 0

    def current_q_sdk(self):
        return np.array([m.q for m in self.lowstate.msg.motor_state[:12]], dtype=np.float32)

    def check_transitions(self):
        msg = self.lowstate.msg
        self.remote.update(msg.wireless_remote)
        keys = self.keyboard.pop()

        if "q" in keys:
            self.running = False
        if self.state != "PASSIVE":
            if time.time() - self.lowstate.t > LOWSTATE_TIMEOUT:
                return self.enter("PASSIVE", "lowstate timeout")
            if self.remote.combo("L2", "B") or "d" in keys:
                return self.enter("PASSIVE", "user damp")
        if self.state in ("STAND", "POLICY") and (self.remote.combo("L2", "X") or "l" in keys):
            return self.enter("SIT", "user")
        if self.state == "SIT" and time.time() - self.state_t0 > self.sit_time + SIT_HOLD:
            return self.enter("PASSIVE", "sit-down finished")
        if self.state == "POLICY":
            g = self.policy.read_state(msg)["projected_gravity_b"]
            tilt = np.arccos(np.clip(-g[2], -1.0, 1.0))
            if tilt > TILT_LIMIT:
                return self.enter("PASSIVE", f"tilt {tilt:.2f} rad > {TILT_LIMIT}")
        if self.state == "PASSIVE" and (self.remote.combo("L2", "A") or "s" in keys):
            return self.enter("STAND", "user")
        if self.state == "STAND" and (self.remote.on_pressed("start") or "p" in keys):
            if time.time() - self.state_t0 < self.stand_time:
                print("  stand-up not finished yet, ignoring", flush=True)
            else:
                return self.enter("POLICY", "user")

    # ---- per-state commands --------------------------------------------
    def run_passive(self):
        q = self.current_q_sdk()
        self.write(q, np.zeros(12), np.full(12, PASSIVE_KD))

    def run_stand(self):
        # current -> folded over the first half, folded -> default over the second
        s = (time.time() - self.state_t0) / self.stand_time
        if s < 0.5:
            a = s / 0.5
            q = (1 - a) * self.start_q + a * FOLDED_Q
        else:
            a = min((s - 0.5) / 0.5, 1.0)
            q = (1 - a) * FOLDED_Q + a * self.default_q_sdk
        self.write(q, STAND_KP, STAND_KD)

    def run_sit(self):
        s = min((time.time() - self.state_t0) / self.sit_time, 1.0)
        a = np.empty(12, dtype=np.float32)
        a[:6] = s  # front legs (FR, FL)
        a[6:] = min(s / SIT_REAR_LEAD, 1.0)  # rear legs (RR, RL)
        a = a * a * (3 - 2 * a)  # smoothstep: no velocity jump at start/end
        q = (1 - a) * self.start_q + a * FOLDED_Q
        self.write(q, STAND_KP, STAND_KD)

    def run_policy(self):
        if self.tick % round(self.policy.step_dt / CTRL_DT) == 0:
            msg = self.lowstate.msg
            self.cmd = self.command()
            state = self.policy.read_state(msg)
            obs, action, target = self.policy.step(state, self.cmd)
            self.q_target_sdk = self.policy.to_sdk_order(target)
            self.log["t"].append(time.time() - self.t0)
            self.log["obs"].append(obs)
            self.log["action"].append(action)
            self.log["target"].append(target)
            self.log["joint_pos"].append(state["joint_pos"])
            self.log["tick"].append(msg.tick)
            self.policy_tick += 1
        self.write(self.q_target_sdk, self.policy_kp_sdk, self.policy_kd_sdk)

    def stick_command(self):
        # same mapping as unitree_rl_lab: ly -> vx, -lx -> vy, -rx -> wz
        sticks = np.array([self.remote.ly, -self.remote.lx, -self.remote.rx], dtype=np.float32)
        sticks[np.abs(sticks) < STICK_DEADZONE] = 0.0
        lo, hi = self.policy.cmd_ranges[:, 0], self.policy.cmd_ranges[:, 1]
        return np.clip(sticks * self.cmd_scale * np.maximum(np.abs(lo), np.abs(hi)), lo, hi)

    def command(self):
        sticks = self.stick_command()
        if self.arbiter is None or np.any(sticks != 0.0):
            self.cmd_from = "sticks"
            return sticks
        # No fresh bus command means stand still
        bus = self.arbiter.get()
        self.cmd_from = bus.source
        cmd = np.array([bus.vx, bus.vy, bus.yaw_rate], dtype=np.float32)
        cmd = np.clip(cmd, -self.bus_max, self.bus_max)
        return np.clip(cmd, self.policy.cmd_ranges[:, 0], self.policy.cmd_ranges[:, 1])

    def write(self, q, kp, kd):
        q = np.clip(q, Q_MIN, Q_MAX)
        for i in range(12):
            m = self.low_cmd.motor_cmd[i]
            m.q, m.dq, m.kp, m.kd, m.tau = float(q[i]), 0.0, float(kp[i]), float(kd[i]), 0.0
        if self.publisher is not None:
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.publisher.Write(self.low_cmd)

    def step(self):
        self.check_transitions()
        {"PASSIVE": self.run_passive, "STAND": self.run_stand, "POLICY": self.run_policy,
         "SIT": self.run_sit}[self.state]()
        self.tick += 1

    def status_line(self):
        return (f"[{time.time() - self.t0:7.2f}s] {self.state:7s} cmd={np.round(self.cmd, 2).tolist()} "
                f"from={self.cmd_from} "
                f"battery={self.lowstate.msg.power_v:.1f}V")


def release_sport_mode():
    sc = SportClient()
    sc.SetTimeout(5.0)
    sc.Init()
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    _, result = msc.CheckMode()
    while result["name"]:
        print(f"releasing sport mode '{result['name']}' (robot lies down first) ...")
        sc.StandDown()
        time.sleep(2.0)
        msc.ReleaseMode()
        time.sleep(1.0)
        _, result = msc.CheckMode()
    print("sport mode released")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interface")
    parser.add_argument("--policy", default="deploy/policies/go2_velocity.npz")
    parser.add_argument("--dry_run", action="store_true",
                        help="run FSM + policy but publish nothing and leave sport mode alone")
    parser.add_argument("--cmd_scale", type=float, default=0.3,
                        help="fraction of the training command range the sticks can reach")
    parser.add_argument("--stand_time", type=float, default=2.0, help="seconds for the stand-up")
    parser.add_argument("--sit_time", type=float, default=2.0, help="seconds for the lie-down")
    parser.add_argument("--cmd_source", choices=["sticks", "bus"], default="sticks",
                        help="bus: velocity commands from command_bus (e.g. the VLA), sticks override")
    parser.add_argument("--bus_max_v", type=float, default=0.3, help="max |vx|, |vy| from the bus")
    parser.add_argument("--bus_max_w", type=float, default=0.5, help="max |wz| from the bus")
    parser.add_argument("--out", default="deploy_log.npz")
    args = parser.parse_args()

    policy = Go2Policy(args.policy)
    ChannelFactoryInitialize(0, args.interface)
    lowstate = LowStateBuffer()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lowstate.handler, 10)
    while lowstate.msg is None:
        print("waiting for rt/lowstate ...")
        time.sleep(0.5)

    if args.dry_run:
        print("DRY RUN: nothing is published, sport mode untouched")
    else:
        print("WARNING: the robot must be LYING DOWN with ~1 m clear around it.")
        input("Press Enter to release sport mode and start in PASSIVE (damping) ...")
        release_sport_mode()

    arbiter = CommandArbiter() if args.cmd_source == "bus" else None
    ctrl = Controller(policy, lowstate, args.dry_run, args.cmd_scale, args.stand_time, args.sit_time,
                      arbiter, args.bus_max_v, args.bus_max_w)
    print("controls: remote L2+A / key 's' = stand,  start / 'p' = policy,  "
          "L2+X / 'l' = lie down,  L2+B / 'd' = damp (emergency),  'q' or Ctrl+C = quit "
          "(keys need Enter)")

    next_t = time.time()
    last_status = 0.0
    try:
        while ctrl.running:
            ctrl.step()
            now = time.time()
            if now - last_status > 1.0:
                last_status = now
                print(ctrl.status_line(), flush=True)
            next_t += CTRL_DT
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.time()  # overran; don't try to catch up in a burst
    except KeyboardInterrupt:
        pass
    finally:
        if ctrl.state in ("STAND", "POLICY"):
            ctrl.enter("SIT", "exit")
        try:
            while ctrl.state == "SIT":  # still checks timeout / 'd' / L2+B
                ctrl.step()
                time.sleep(CTRL_DT)
        except KeyboardInterrupt:
            print("second Ctrl+C: skipping lie-down", flush=True)
        if ctrl.state != "PASSIVE":
            ctrl.enter("PASSIVE", "exit")
        # keep damping for a moment so the robot settles before commands stop
        t_end = time.time() + 1.0
        while time.time() < t_end:
            ctrl.run_passive()
            time.sleep(CTRL_DT)
        np.savez(args.out, cmd_scale=args.cmd_scale, **{k: np.array(v) for k, v in ctrl.log.items()})
        print(f"saved {len(ctrl.log['t'])} policy steps to {args.out}")


if __name__ == "__main__":
    main()
