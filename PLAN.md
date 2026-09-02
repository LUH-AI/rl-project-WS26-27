# Go2 EDU Sim2Real RL Pipeline — Project Plan

## Goal

Build an extensible sim2real RL pipeline for the Unitree Go2 EDU, staged so each
part can be developed and validated independently before the next is bolted on:

1. **Stage 1 — Proprioceptive locomotion (RL core).** Train and deploy a
   blind, proprioception-only locomotion policy. This is the foundation
   everything else sits on.
2. **Stage 2 — High-level command layer (VLA / BFM).** Add a slow-rate
   high-level model that issues commands to the Stage 1 policy, without
   retraining the low-level controller from scratch.
3. **Stage 3 — Vision.** Add exteroceptive sensing (RealSense depth) so the
   low-level policy becomes terrain-aware, and/or so the high-level layer
   gets a camera feed for scene/language reasoning. Deferred until 1 and 2
   are solid.

The RL locomotion policy is the core of the system throughout — later stages
extend it via well-defined interfaces rather than replacing it.

```
Stage 3 (later): RealSense depth ────────────────┐
                                                  ▼
Stage 2 (next):  VLA / BFM  ──┐                                    Stage 1: RL policy ──► joints
                 (1-10 Hz)    ├──►  command bus (arbitrated)  ──►     (50 Hz, proprio)
Remote control   ────────────┘      (velocity cmd + mode/e-stop)
  (always overrides)
```

The remote control and the Stage 2 high-level model are two interchangeable
*sources* on the same command bus — the RC is not a separate, throwaway
teleop path. This lets the RC double as the Stage 1 hardware-validation
tool (driving the trained policy by hand) and as the permanent manual
override once Stage 2 is added. See "Command Bus & Remote Control" below.

---

## Stage 1 — Proprioceptive Sim2Real RL

### Objective
A single locomotion policy that takes proprioception + a velocity command
and produces joint targets, robust enough to walk/trot on the real Go2 EDU
after training entirely in simulation.

### Simulator & training stack
- **Simulator:** Isaac Lab (GPU-parallel, thousands of envs). Start from
  Unitree's own `unitree_rl_gym` config for Go2 if available — it already
  encodes the correct joint limits, mass properties, and URDF.
- **Algorithm:** PPO via `rsl_rl`.
- **Terrain:** procedurally generated flat + rough/uneven terrain curriculum
  (no stairs/obstacles yet — that's a Stage 3 concern once vision exists).

### Observation space (policy input)
- Joint positions, joint velocities (12 each)
- Base angular velocity, projected gravity vector (orientation, no absolute
  heading — keeps policy terrain/heading-invariant)
- Previous action
- **Command vector: (vx, vy, yaw-rate)** — see "Command Bus & Remote
  Control" below.

### Action space
- Joint position targets (12-dim), converted to torque via PD control /
  actuator network at a fixed control rate (50 Hz).

### Command Bus & Remote Control (decided now, reused in Stage 2)
The policy is conditioned on a **velocity command (vx, vy, yaw-rate)** from
the start, even though in Stage 1 this command is just sampled randomly
during training (no upstream model yet). This is the same interface both
the **remote control** and, later, the **Stage 2 VLA/BFM** will use — so
**no low-level retraining is needed when Stage 2 is added**, and the RC
isn't throwaway teleop code, it's the permanent manual-override path.

**Command bus, one format, multiple sources:**
- **Continuous command topic:** `(vx, vy, yaw-rate)`. In Stage 1 this has
  exactly one source (RC/joystick, or a scripted publisher for automated
  testing); in Stage 2, the VLA/BFM becomes a second, lower-priority source
  on the same topic.
- **Discrete mode/safety commands**, kept *out of* the learned command
  topic and handled by a lightweight supervisor process, not the RL
  policy itself:
  - E-stop / damping — bound to a physical RC button, lowest-latency path,
    independent of whatever else is running (autonomy stack, high-level
    model) so it degrades gracefully if those hang or crash.
  - Stand up / sit down / lie down — pose transitions outside the walking
    policy's scope (typically Unitree's built-in sport-mode routines).
  - Autonomy enable/disable — toggles whether the velocity-command topic
    is currently driven by RC or by the (future) high-level model.
- **Arbitration rule: RC always wins.** Any stick input (or a dedicated
  override button) immediately takes command authority away from
  whatever else is publishing to the command topic. This is a safety
  requirement now and remains one once Stage 2 adds an autonomous source.

To make the policy robust to an imperfect/discontinuous command source
later (RC glitches now, high-level model latency later), randomize
during Stage 1 training:
- Command resampling rate (not fixed-interval — vary how long a command
  holds, to mimic irregular human or high-level update rates)
- Command noise (small perturbations to vx/vy/yaw-rate)
- Occasional command dropout/staleness (hold last command briefly) to
  simulate a dropped RC packet or high-level model latency

This is cheap to do now and expensive to retrofit later — treat it as a
required part of Stage 1, not an optional extra.

### Sim2real gap closers
- **Actuator network:** identify real Go2 motor response (torque/velocity
  lag) from logged real joint data; replace ideal PD control in sim with
  this fitted model.
- **Domain randomization:** mass/COM, friction, motor strength, joint
  latency, external pushes.
- **Observation noise:** IMU and joint sensor noise matching real sensor
  specs.
- Teacher-student distillation is optional at this stage (mainly useful
  once terrain/vision privileged info exists in Stage 3) — a single-stage
  PPO policy is likely sufficient for flat/rough-terrain blind walking.

### Deployment
- Export trained policy to ONNX/TensorRT.
- Run on Go2 EDU's onboard Jetson Orin NX at 50 Hz, sending joint targets
  via `unitree_sdk2`'s low-level motor interface.
- Command source in Stage 1 deployment: the RC (joystick) publishing to
  the command bus, plus the supervisor process handling e-stop/mode
  commands — this is the same bus Stage 2 adds a second source to, not
  something Stage 2 replaces.

### Deliverables / milestones
- [ ] Isaac Lab env for Go2 with proprioceptive obs + velocity command
      conditioning, command randomization built in
- [ ] PPO training run reaching stable flat-ground walking in sim
- [ ] Rough-terrain curriculum added, policy robust to pushes/terrain
      randomization in sim
- [ ] Actuator network identified from real Go2 motor logs
- [ ] Command bus implemented: RC → velocity-command topic, plus
      supervisor-handled e-stop/stand/sit/autonomy-toggle
- [ ] Sim2real deployment: policy walks on real Go2 EDU tracking the RC's
      velocity command over the command bus
- [ ] Basic safety fallback (stop/damping mode on command loss or fall
      detection) implemented on-device, independent of the RL policy

### Open questions to resolve before starting
- Isaac Lab vs. Genesis for the simulator (Isaac Lab has more Go2-specific
  prior art via `unitree_rl_gym`; Genesis is newer/faster but less
  battle-tested for this exact robot) — recommend starting with Isaac Lab
  given the existing Unitree reference implementation, revisit if
  iteration speed becomes a bottleneck.
- Whether to reuse `unitree_rl_gym` as-is or reimplement in Isaac Lab
  directly for longer-term maintainability.

---

## Stage 2 — High-Level Command Layer (VLA / BFM)

### Objective
Add a slow-rate (1–10 Hz) model as a second source on the Stage 1 command
bus, without modifying or retraining the Stage 1 policy, and without
removing the RC's ability to override it at any time.

### Architecture
```
                RC (highest priority) ──────────┐
                                                 ▼
High-level model (VLA or BFM)  ──(vx,vy,yaw)──►  arbitrated command topic  ──►  Stage 1 policy (frozen)
      1-10 Hz                                                                       50 Hz
```
Only high-level commands are new here — for now the VLA/BFM is scoped to
issuing this one command type (no vision, no waypoints/subgoals yet); the
RC's arbitration priority and the supervisor's e-stop/mode handling from
Stage 1 are reused unchanged.

### Two paths (pick one to prototype first; not mutually exclusive long-term)
- **VLA-style / language-conditioned navigation:** a vision-language
  navigation model (e.g. ViNT/NoMaD-style, or a lightweight VLM used purely
  for scene/task reasoning) outputs the velocity command or a local
  waypoint. Natural fit once Stage 3 vision exists — can be prototyped
  earlier with a stubbed/simulated camera input or command scripts standing
  in for the model's output.
- **BFM-style / skill-conditioned:** a broader skill or zero-shot-RL-style
  policy layer where "high-level" is a skill selector or latent-conditioning
  mechanism. This more naturally overlaps with Stage 1 if pursued — flagging
  now as a possible alternative to the fixed single-behavior PPO policy, to
  decide once Stage 1 is working and we know whether more behavioral
  diversity is needed.

### Integration requirements
- Reuse the Stage 1 command bus and message format as-is — no new
  interface to design, only a new publisher on the existing velocity-
  command topic, arbitrated below RC per the Stage 1 rule.
- Fallback behavior: if the high-level layer stalls, produces
  out-of-distribution commands, or the RC operator intervenes, RC
  override + Stage 1's safety supervisor must take precedence immediately
  — this should already work from Stage 1's arbitration, but must be
  explicitly tested with the real VLA/BFM in the loop (it can fail in new
  ways, e.g. a model that publishes valid-looking but nonsensical commands
  fast enough to fight a slow human reaction time).
- Compute budget: confirm what's left on the Orin NX after the 50 Hz
  control loop before choosing model size; plan for on-device quantized
  model vs. offboard/wireless inference as a fallback if the chosen
  model doesn't fit.

### Deliverables / milestones
- [ ] Minimal high-level driver (even a scripted/rule-based one) added as
      a second publisher on the Stage 1 command bus, validated end-to-end
      on hardware with RC override still working
- [ ] First VLA or BFM candidate integrated, issuing commands in sim first,
      then on hardware
- [ ] RC-override tested against a live high-level model specifically
      (not just against a scripted stand-in), confirming a human always
      regains control instantly
- [ ] Safety fallback validated (kill high-level link mid-run, confirm
      Stage 1 policy degrades gracefully)

### Open questions to resolve before starting
- VLA vs. BFM as the first path to prototype — recommend deciding this
  after Stage 1 is deployed, since it doesn't block Stage 1 work.
- On-device vs. offboard inference for the high-level model.

---

## Stage 3 — Vision (deferred)

Placeholder for later planning. Known requirements from earlier discussion,
to expand once Stages 1–2 are stable:
- Intel RealSense (D435i/D455) depth input, processed off the control loop
- Depth → downsampled height-scan or small CNN encoder feeding either the
  Stage 1 policy (terrain-aware locomotion) or the Stage 2 high-level model
  (scene/language reasoning), or both
- Simulated depth/raycast sensor in Isaac Lab with matched noise/dropout
  model for sim2real transfer of the vision branch
- Latency budget accounting (RealSense pipeline adds ~30-60ms)

---

## Repo Structure (proposed, to scaffold at the start of Stage 1)

```
rl-project/
├── PLAN.md                  # this file
├── sim/                     # Isaac Lab env, task configs, domain randomization
├── training/                # rsl_rl PPO configs, training scripts
├── actuator_net/            # real motor data logging + system ID fitting
├── deploy/                  # ONNX export, unitree_sdk2 control loop, Orin NX runtime
├── command_bus/             # command topic def, arbitration, RC driver, e-stop supervisor
├── highlevel/                # (Stage 2) VLA/BFM driver, publishes to command_bus
├── vision/                   # (Stage 3) RealSense pipeline, depth encoders
└── docs/
```

---

## Immediate Next Step

Scaffold Stage 1 only: `sim/`, `training/`, `actuator_net/`, `deploy/`,
`command_bus/` directories plus the Isaac Lab environment definition with
the proprioceptive observation space and velocity-command conditioning
(including command randomization) described above.
