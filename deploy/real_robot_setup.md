# Real Go2: setup, bring-up and policies

Details behind the root README: setup gotchas, what each bring-up step showed,
the deploy state machine and the two example policies.

## Setup notes

- `deploy/.venv` uses Python 3.10 because `cyclonedds==0.10.2` (pulled in by the SDK)
  only has prebuilt wheels up to 3.10. Newer Pythons need the CycloneDDS C library.
- Don't source `/opt/ros/*/setup.bash` in the robot shell, ROS 2 brings its own CycloneDDS.
- Shared machine: the SDK writes its DDS log to `/tmp/cdds.LOG`. If another user owns it,
  `ChannelFactoryInitialize` fails with `channel factory init error`. Fix in
  `third_party/unitree_sdk2_python/unitree_sdk2py/core/channel_config.py`
  (re-apply after re-cloning the SDK):
  ```diff
  -            <OutputFile>/tmp/cdds.LOG</OutputFile>
  +            <OutputFile>/tmp/cdds_${USER}.LOG</OutputFile>
  ```
- Network: workstation `192.168.123.51/24`, robot `192.168.123.161`.

## Bring-up steps

All passed on 2026-10-11, in this order.

| Step | Command | Robot | Result |
|------|---------|-------|--------|
| Read state | `deploy/read_lowstate.py` | anything | `rt/lowstate` at ~500 Hz. Motor order FR, FL, RR, RL × (hip, thigh, calf) |
| Sport API | `third_party/unitree_sdk2_python/example/go2/high_level/go2_sport_client.py` | on the floor | stand_down (2), stand_up (1), balanced stand (9), move (3), stop (6). Avoid damp (0) while standing and the acrobatic modes |
| Low-level | `third_party/unitree_sdk2_python/example/go2/low_level/go2_stand_example.py` | lying down, ~1 m clear | releases sport mode, stands (Kp 60, Kd 5) and lies down again |
| Dry run | `deploy/run_policy.py --dry_run` | lying down | policy runs on live state at 50 Hz, publishes nothing. Joint mapping verified |
| Remote | `deploy/read_remote.py` | sport mode off | buttons and sticks parsed correctly |
| Closed loop | `deploy/run_policy.py` | lying down, remote in hand | stable stand and walking with `--cmd_scale` 0.3 and 1.0, max tilt 6°. Plot logs with `deploy/plot_deploy.py` |

Sport mode stays released after any low-level run, until the robot reboots.

## `run_policy.py` state machine

A Python port of unitree_rl_lab's C++ Go2 FSM. Policy at 50 Hz, lowcmd at 500 Hz.
Keyboard keys need Enter.

| State | Gains | Enter via |
|-------|-------|-----------|
| PASSIVE | kp 0, kd 3 (damping) | start, L2+B / `d`, lowstate timeout >0.1 s, tilt >1 rad in POLICY |
| STAND | kp 60/80/80, kd 5/4/4, to the training default pose in 2 s | L2+A / `s` from PASSIVE |
| POLICY | kp 25, kd 0.5 (from `deploy.yaml`) | start / `p` from STAND |
| SIT | kp 60/80/80, kd 5/4/4, to the lying pose in 2 s, rear legs first, then PASSIVE | L2+X / `l` from STAND or POLICY, also on `q` / Ctrl+C |

- Sticks map to commands like in unitree_rl_lab (ly→vx, −lx→vy, −rx→wz), scaled by `--cmd_scale`
  (default 0.3 of the training range). With `--cmd_source bus` the commands come from the VLA,
  and any stick input overrides them.
- Use `d` only as an emergency stop. It drops the robot, and the head can hit the ground first.
- `--dry_run` publishes nothing and leaves sport mode on.

## Example policies

Both are blind (proprioceptive only) velocity-tracking policies trained with PPO (rsl_rl) in Isaac Lab,
4096 envs, 2000 iterations, flat terrain. They share one `deploy.yaml`, so `run_policy.py` runs either one.

| | `go2_velocity.npz` (standard) | `go2_crawl.npz` (crawl) |
|---|---|---|
| Task | `Unitree-Go2-Velocity` | `Unitree-Go2-Velocity-Crawl` |
| Run | `unitree_go2_velocity/2026-09-02_14-23-20` | `unitree_go2_velocity_crawl/2026-11-02_22-31-16` |
| Extra rewards | – | 4-beat gait (period 1 s), base height 0.22 m with weight −120 |
| Sim body height | 0.335 m | ~0.25 m |
| Tested on the robot | sticks, `--cmd_scale` 0.3 and 1.0 | VLA commands at 0.5 m/s |

Shared details:
- MLP 45 → 512 → 256 → 128 → 12 (ELU), no history and no obs normalizer.
- Observations: angular velocity (×0.2), projected gravity, command (vx, vy, wz), joint positions minus default, joint velocities (×0.05), last action. There is no base linear velocity, so no state estimator is needed.
- Actions: target = default pose + 0.25 × action, PD with Kp 25, Kd 0.5.
- Command range: vx ±1.0 m/s, vy ±0.4 m/s, wz ±1.0 rad/s.
- Domain randomization: friction 0.3–1.2, base mass −1…+3 kg, pushes, observation noise.

Re-export a run (`.venv`, writes to the gitignored `deploy/policies/`):
```bash
.venv/bin/python deploy/export_policy.py --out deploy/policies/go2_crawl.npz \
  --run_dir third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity_crawl/2026-11-02_22-31-16
```

### How the crawl height was tuned

The height reward alone barely lowers the body, so its weight was raised step by step:

| Run | Height weight | Sim body height |
|-----|---------------|-----------------|
| `unitree_go2_velocity_crawl/2026-10-11_16-37-47` | −10 | 0.306 m |
| `unitree_go2_velocity_crawl/2026-10-11_17-39-38` | −40 | 0.287 m |
| `unitree_go2_velocity_crawl/2026-11-02_22-31-16` (`go2_crawl.npz`) | −120 | ~0.25 m |

After `p` the robot drops from the stand pose to its crouch. That is the policy, not a fault.

### Crawl speed dead zone

The crawl policy barely moves for small forward commands (sim, `training/rollout_and_log.py --cmd VX 0 0`):

| Command vx [m/s] | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|
| Sim vx [m/s] | 0.01 | 0.08 | 0.27 | 0.62 | 0.81 |

The VLA always commands its `--max-v`, so `--max-v` and `--bus_max_v` set the crawl speed directly
(the lower one wins). At 0.3 the robot only turns on the spot, so use 0.5 or more.
For the same reason, driving with the sticks needs `--cmd_scale 1.0`.

Replaying the commands of a logged sport-mode VLA run in sim gives:

| Policy | vx cmd → actual | vx err | wz err |
|---|---|---|---|
| `go2_velocity` (sim) | 0.50 → 0.47 | 0.033 | 0.054 |
| `go2_crawl` (sim) | 0.50 → 0.35 | 0.155 | 0.067 |
| `go2_crawl` (sim, capped at 0.3) | 0.30 → 0.07 | 0.227 | 0.096 |
| sport mode (real, same log) | 0.50 → ~0.45 | 0.083 | 0.067 |

```bash
cd third_party/unitree_rl_lab
../../.venv/bin/python ../../training/rollout_and_log.py --task Unitree-Go2-Velocity-Crawl \
  --num_envs 1 --load_run 2026-11-02_22-31-16 --checkpoint model_1999.pt \
  --cmd_log ../../training/outputs/real_vla/<run>/commands.jsonl --out ../../training/outputs/replay.npz
```

## VLA with the crawl policy

Run the three terminals from the root README.
- In a dry run, after `s` and `p` terminal 2 should print `POLICY cmd=[0.5, 0.0, ...] from=vla`.
- With the VLA off, it shows `from=none` and the robot stands in place.
- Terminal 1 should keep showing `cam_err=0`. The camera keeps working after sport mode is released.
- Ctrl+C on the VLA stops the robot within ~0.5 s.
- If the robot walks at 0.3 m/s on the sticks, `--policy` or `--cmd_source bus` was forgotten.

Result (2026-11-02): crawled under VLA control for ~17 s at a 0.5 m/s command, mean |wz| 0.12 rad/s.
The real forward speed is unmeasured, since there is no odometry once sport mode is released.
