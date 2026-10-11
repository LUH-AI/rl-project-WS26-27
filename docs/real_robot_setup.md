# Real Go2 — SDK setup & bring-up log

Step-by-step record of getting `unitree_sdk2_python` talking to the real Go2 EDU
from the workstation. Test in increasing order of risk:

| Step | What | Robot pose | Status |
|------|------|-----------|--------|
| 0 | Read-only `rt/lowstate` subscriber | anything | ✅ done (2026-10-11) |
| 1 | High-level `SportClient` (StandUp/StandDown/Move) | on the floor, clear area | ✅ done (2026-10-11) |
| 2 | Low-level `rt/lowcmd` (sport mode released) | lying down on the floor | ✅ done (2026-10-11) |
| 3a | Policy in shadow mode, cmd=0, robot lying down | lying down | ✅ done (2026-10-11) |
| 3b | Shadow mode, cmd=0, robot standing in sport mode | standing, sport mode | ⬜ not started |
| 3c | Shadow mode, nonzero cmd, walking with remote | walking, sport mode | ⬜ not started |
| 4 | First live/closed-loop actuation (policy publishes to `lowcmd`) | standing, clear area, remote in hand | ✅ done (2026-10-11), `--cmd_scale` 0.3 and 1.0 |

> **Second, separate track:** the vendored C++ FSM deploy stack
> (`third_party/unitree_rl_lab/deploy`, built as `go2_ctrl`) is an independent
> path to the same goal, not yet reconciled with the steps above. See
> [§9](#9-c-fsm-deploy-track-go2_ctrl--separate-from-steps-above) — it has had
> no hardware testing at all so far, while the Python track above is already
> through Step 2 and mid-way through Step 3.

## 1. Install CycloneDDS 0.10.x (C library)

`cyclonedds==0.10.2` (pulled in by the SDK) fails to build with
`Could not locate cyclonedds` unless the C library is available.

```bash
cd ~/Repos
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build . --target install -j"$(nproc)"
```

`CYCLONEDDS_HOME=~/Repos/cyclonedds/install` is only needed when (re)installing,
not at runtime.

## 2. Install `unitree_sdk2_python` in a separate venv

Kept separate from the Isaac Lab env to avoid numpy/cyclonedds clashes.

```bash
cd ~/Repos/rl-project
git clone https://github.com/unitreerobotics/unitree_sdk2_python third_party/unitree_sdk2_python
uv venv .venv-deploy --python 3.11
CYCLONEDDS_HOME=~/Repos/cyclonedds/install \
  uv pip install --python .venv-deploy -e third_party/unitree_sdk2_python
```

### Local patch: per-user DDS log file

The SDK hardcodes the CycloneDDS log to `/tmp/cdds.LOG`. On this shared
machine that file is owned by another user, so `ChannelFactoryInitialize`
fails with `channel factory init error`. Patched
`third_party/unitree_sdk2_python/unitree_sdk2py/core/channel_config.py`:

```diff
-            <OutputFile>/tmp/cdds.LOG</OutputFile>
+            <OutputFile>/tmp/cdds_${USER}.LOG</OutputFile>
```

`third_party/` is gitignored — **re-apply this after re-cloning the SDK.**

## 3. Network

- Robot reachable over the USB-Ethernet adapter `enxf8e43b9f046a`
  (workstation IP `192.168.123.51/24`, robot `192.168.123.161`).
- Already configured, no `nmcli` setup was needed. Check with
  `ip -br addr | grep 192.168.123`.
- Don't source `/opt/ros/*/setup.bash` in the SDK shell (ROS 2 ships its own
  CycloneDDS / `CYCLONEDDS_URI`).

## 4. Step 0 — read-only `rt/lowstate` ✅

```bash
.venv-deploy/bin/python deploy/read_lowstate.py enxf8e43b9f046a
```

Prints IMU, foot force, battery and the 12 joints once per second; publishes
nothing. Result on first run (robot standing in sport mode):

- `rt/lowstate` at ~500 Hz
- rpy ≈ 0, joints ≈ thigh +0.7 / calf −1.4…−1.5 rad, feet loaded
- battery 31.5 V, SOC 99 %, motor temps 23–26 °C

Motor order in `motor_state[0:12]`: FR, FL, RR, RL × (hip, thigh, calf).
Check this against the sim joint order before running any policy.

## 5. Step 1 — high-level sport API ✅

```bash
.venv-deploy/bin/python third_party/unitree_sdk2_python/example/go2/high_level/go2_sport_client.py enxf8e43b9f046a
```

Robot on flat ground (not on a stand), clear space, remote in hand
(damping: L2+B). Sequence that worked:

1. `stand_down` (2)
2. `stand_up` (1)
3. `balanced stand` (9) — needed before move commands
4. `move forward` (3), then `stop_move` (6) immediately

Avoid: `damp` (0) while standing (legs go limp), and the acrobatic modes
(7 hand stand, 11/12 flips, 17 walk upright, 18 cross step, 19 free jump).

## 6. Step 2 — low-level `rt/lowcmd` stand-up ✅

No stand needed. Robot lying down on the floor first (`stand_down` via the
sport client), ~1 m clear around it.

```bash
.venv-deploy/bin/python third_party/unitree_sdk2_python/example/go2/low_level/go2_stand_example.py enxf8e43b9f046a
```

What the script does (Kp=60, Kd=5, commands at 500 Hz):

1. Calls `StandDown()` + `MotionSwitcherClient.ReleaseMode()` itself → sport mode off
2. ~1 s fold to crouch → ~1 s stand (thigh 0.67, calf −1.3) → ~2 s hold
3. ~1.8 s back down to a folded, hip-splayed pose, then holds it
4. Ctrl+C once it is lying down again

Sport mode stays released afterwards — reboot the robot (or re-enable via the
app/remote) before using the high-level API again.

Worked on first try.

## 7. Step 3 — Isaac Lab policy in shadow mode ✅

Policy: `third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-09-02_14-23-20/model_1999.pt`
(45-dim obs → 512/256/128 ELU → 12, no obs normalizer). Its `params/deploy.yaml`
gives joint map, Kp=25 / Kd=0.5, default pose, action scale 0.25, obs scales.

Export to a numpy-only `.npz` (weights + deploy.yaml; training venv, once):

```bash
.venv/bin/python deploy/export_policy_npz.py \
  --run_dir third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-09-02_14-23-20 \
  --checkpoint model_1999.pt --out deploy/policies/go2_velocity.npz
```

Run shadow mode (deploy venv; publishes nothing, Ctrl+C saves the log):

```bash
.venv-deploy/bin/python deploy/shadow_policy.py enxf8e43b9f046a \
  --policy deploy/policies/go2_velocity.npz --cmd 0 0 0 --out shadow_log.npz
```

- `deploy/go2_policy.py` mirrors unitree_rl_lab's C++ deploy: gyro → `base_ang_vel`,
  quaternion (w,x,y,z) → projected gravity, `q_isaac[i] = q_sdk[joint_ids_map[i]]`.
- First test run (robot lying in the folded pose after step 2): 50 Hz, fresh
  lowstate every step. Joint mapping verified: hips read FL +0.31 / FR −0.31 /
  RL +0.50 / RR −0.50, matching the stand example's final pose. Actions were large
  (|a| up to 3.5), as expected for an out-of-distribution lying pose.
- **Still to do (Step 3b/3c):**
  - [ ] Stand the robot up in sport mode, run shadow mode with `--cmd 0 0 0`.
        Actions should stay small (`|a|` well under the 3.5 seen lying down)
        and `target` should track close to the actual standing joint angles.
  - [ ] With the robot standing/walking under sport-mode remote control, run
        shadow mode with the remote's actual velocity (or sweep `--cmd` by
        hand) and confirm `target - q` stays small and smooth as the pose
        changes, no spikes or sign flips when turning.
  - [ ] Only after 3b/3c look sane: decide what Step 4 (first live actuation)
        looks like — most likely start from sport-mode standing, cmd=0, swap
        `shadow_policy.py`'s "log only" loop for one that publishes to
        `rt/lowcmd` via `MotionSwitcherClient.ReleaseMode()` (reuse the release
        mechanism already proven in Step 2, not necessarily the
        `RobotStateClient::ServiceSwitch` path from the C++ track — see §9),
        remote in hand on damp (`L2+B`) the whole time.

## 9. C++ FSM deploy track (`go2_ctrl`) — separate from steps above

Independent path through the vendored `unitree_rl_lab/deploy` C++ stack
(FSM + onnxruntime), built this session, **no hardware testing yet**. Not
merged into the Step 0–4 log above because it uses a different toolchain and
a different sport-mode release mechanism.

- `~/Repos/unitree_sdk2` cloned and built (own copy, not relying on another
  user's). Also already installed system-wide at `/usr/local` (pre-existing,
  not from this clone).
- `third_party/unitree_rl_lab/deploy/robots/go2/build/go2_ctrl` builds clean
  (fixed a missing `libonnxruntime.so` dev symlink in the vendored onnxruntime
  package).
- `~/Repos/unitree_sdk2/build/bin/sport_mode_switch <iface> list|on|off`
  built, to release/restore `sport_mode` via `RobotStateClient::ServiceSwitch`
  before/after running `go2_ctrl`.

**Open question before trusting this track:** Step 2 above released sport
mode via `MotionSwitcherClient.ReleaseMode()`, confirmed to stay released
until reboot or app/remote re-enable. `sport_mode_switch` instead uses
`RobotStateClient::ServiceSwitch("sport_mode", 0, ...)` — a different API.
Not yet confirmed these two are equivalent (same persistence, same effect on
`lowcmd` ownership). Check this empirically (`ServiceSwitch` off, reboot,
`ServiceList` to see if it's still off) before relying on it for a real
session, rather than assuming it matches `ReleaseMode()`'s behavior.

- [ ] Confirm `ServiceSwitch("sport_mode", 0, ...)` actually frees the
      `lowcmd` channel the same way `ReleaseMode()` did (run `go2_ctrl` after,
      check for the "other process is using the lowcmd channel" message).
- [ ] Confirm persistence across reboot, same as `ReleaseMode()`.
- [ ] `go2_ctrl`'s policy is loaded from
      `third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity` via
      `parser_policy_dir` (auto-picks latest run) — confirm this resolves to
      the same `2026-09-02_14-23-20/model_1999.pt` run already validated in
      Step 3, not a different/stale run.
- [ ] No shadow-mode equivalent exists for this track — first run of
      `go2_ctrl`'s `Velocity` FSM state will be live actuation immediately.
      Decide whether to add a dry-run/logging mode here too, or only ever use
      this track after the Python track's Step 4 has already validated the
      same policy live.

> **Unverified (revisit later):** tracking accuracy. Every controller has some
> admissible tracking error; the target-vs-measured offsets above have not been
> judged against an acceptance threshold or against sim
> (`plot_deploy.py --sim`). Treated as acceptable for now.

## 8. Step 4 — closed-loop policy (`deploy/run_policy.py`) ✅

Python port of unitree_rl_lab's C++ Go2 FSM. The C++ code is used only as a
reference for gains/thresholds; nothing is built in C++.

| State | Gains | Enter via |
|-------|-------|-----------|
| PASSIVE | kp 0, kd 3 (damping) | start, L2+B / `d`, lowstate timeout >0.1 s, tilt >1 rad in POLICY |
| STAND | kp 60/80/80, kd 5/4/4; current → folded → training default (2 s) | L2+A / `s` from PASSIVE |
| POLICY | kp 25, kd 0.5 (deploy.yaml); 50 Hz policy, 500 Hz lowcmd | start / `p` from STAND (after stand-up finished) |
| SIT | kp 60/80/80, kd 5/4/4; current → folded lying pose (2 s, smoothstep), rear legs fold by 70 % of the time so the rear touches down first; then PASSIVE after 0.5 s hold | L2+X / `l` from STAND or POLICY; also on `q` / Ctrl+C (second Ctrl+C skips it) |

- Sticks → command like unitree_rl_lab (ly→vx, −lx→vy, −rx→wz), scaled by
  `--cmd_scale` (default 0.3 of the training range), deadzone 0.05.
- Joint targets clamped inside URDF limits. Ctrl+C / `q` → 1 s damping, then
  exit; log saved in the same format as shadow mode (works with `plot_deploy.py`).
- Keyboard keys need Enter.

Dry run (publishes nothing, sport mode untouched) — tested 2026-10-11:
PASSIVE→STAND→POLICY→PASSIVE via keys, policy at 50.0 Hz, fresh lowstate every step.

```bash
.venv-deploy/bin/python deploy/run_policy.py enxf8e43b9f046a --dry_run
```

Before the real run:

1. **Verify the remote mapping** (parser in `deploy/go2_remote.py` is written
   from Unitree's struct layout, not yet checked). With sport mode off (e.g.
   right after `go2_stand_example.py`), run
   `.venv-deploy/bin/python deploy/read_remote.py enxf8e43b9f046a` and check
   L2, A, B, start and the sticks print correctly (ly>0 when pushing forward).
   Until verified, use the keyboard keys and keep Ctrl+C at hand.
2. Robot lying down, ~1 m clear, someone ready to catch/hold it.

Real run:

```bash
.venv-deploy/bin/python deploy/run_policy.py enxf8e43b9f046a --cmd_scale 0.3 --out deploy_stand.npz
```

`s` (stand) → wait for the stand-up → `p` (policy, sticks centered = stand in
place) → `l` (lie down) → `q`. Use `d` (damp) only as an emergency stop: it
drops the robot, and the head/lidar can hit the ground first (seen 2026-10-11),
which is why SIT was added. First goal: policy holds a stable stand at zero
command. Walking with small stick inputs comes after that.

Result (2026-10-11): real closed-loop runs work with `--cmd_scale 0.3` and
with `--cmd_scale 1.0` (full training command range via the sticks).

| Log | Policy time | Commands reached | Max tilt | Max \|gyro\| |
|-----|-------------|------------------|----------|---------------|
| `deploy_stand.npz` (scale 0.3) | 51 s | vx ±0.3, vy −0.12…0.09, wz ±0.3 | 2.7° | 1.6 rad/s |
| `deploy_stand_cmdscale_1.npz` (scale 1.0) | 25 s | vx ±1.0, vy −0.4…0.37, wz not used | 6.0° | 2.1 rad/s |

Both: 50 Hz, fresh lowstate every step (max step gap 20.3 ms). Plots:
`deploy/plots/deploy_stand*_{overview,joints}.png`. Velocity tracking accuracy
is still unverified (no measured base velocity to compare against).

## Deployed policy

Stock unitree_rl_lab **`Unitree-Go2-Velocity`** task: a blind (proprioceptive-only)
velocity-tracking walking policy.

| | |
|---|---|
| Checkpoint | `third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-09-02_14-23-20/model_1999.pt` (final of 2000 iterations) |
| Exported as | `deploy/policies/go2_velocity.npz` (`deploy/export_policy_npz.py`) |
| Algorithm | PPO (rsl_rl), Isaac Lab, 4096 parallel envs |
| Network | MLP 45 → 512 → 256 → 128 → 12, ELU; no memory/history, no obs normalizer |
| Observations (45) | base angular velocity (×0.2), projected gravity, velocity command (vx, vy, wz), joint pos − default (12), joint vel (×0.05, 12), last action (12). **No base linear velocity** → no state estimator needed |
| Actions (12) | joint position offsets: target = default pose + 0.25 × action; onboard PD with Kp=25, Kd=0.5 |
| Rates | policy 50 Hz, lowcmd 500 Hz |
| Default pose (Isaac order FL, FR, RL, RR) | hips ±0.1, thighs 0.8 front / 1.0 rear, calves −1.5 rad |
| Command range | vx ±1.0 m/s, vy ±0.4 m/s, wz ±1.0 rad/s (curriculum grows to this); ~10 % standing envs |
| Terrain | **flat only** |
| Domain randomization | friction 0.3–1.2, base mass −1…+3 kg, pushes ±0.5 m/s, observation noise |
| Main rewards | lin/ang velocity tracking (1.5 / 0.75); penalties on torque, energy, action rate, foot slip, undesired contacts; feet air time |

Limitations: never saw stairs/slopes/rough ground; trained with ideal PD
actuators (no actuator net yet). Deployed so far with `--cmd_scale` 0.3 and 1.0
(full training command range).

## 9b. Low-height "crawl" policy ⬜

Candidates trained 2026-10-11 (all share the deployed policy's `deploy.yaml`:
same obs, action scale, gains, default pose, so `run_policy.py` is unchanged):

| Run | Height reward (target, weight) | Extra | Sim body height |
|-----|-------------------------------|-------|-----------------|
| standard (deployed above) | – | – | 0.335 m |
| `unitree_go2_velocity_lowheight/2026-10-11_15-47-34` | 0.30 m, −10 | – | 0.329 m |
| `unitree_go2_velocity_crawl/2026-10-11_16-37-47` | 0.22 m, −10 | 4-beat gait reward (period 1 s) | 0.306 m |
| `unitree_go2_velocity_crawl/2026-10-11_17-39-38` | 0.22 m, **−40** | 4-beat gait reward (period 1 s) | **0.287 m** |

None reaches its height target. The weight-40 crawl run is the lowest (~5 cm
below standard) and is the one chosen for deployment:

```bash
.venv/bin/python deploy/export_policy_npz.py \
  --run_dir third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity_crawl/2026-10-11_17-39-38 \
  --checkpoint model_1999.pt --out deploy/policies/go2_crawl.npz

.venv-deploy/bin/python deploy/run_policy.py enxf8e43b9f046a \
  --policy deploy/policies/go2_crawl.npz --cmd_scale 0.3 --out deploy_crawl.npz
```

Expected: after `p` the robot drops from the stand pose (~0.33 m) to its
crouch (~0.29 m); that is the policy, not a fault.
