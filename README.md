# Go2 VLA Navigation

## Overview

Drive the real Unitree Go2 with a text prompt like "go to the white door".
We use [OmniVLA](https://github.com/NHirose/OmniVLA) to predict where to walk from the front camera,
and Unitree's sport mode controller to move the legs.

## Installation

The VLA needs an NVIDIA GPU with at least 20GB of memory for the 7B model (~2GB for the edge model)
and [uv](https://docs.astral.sh/uv/). To install the VLA and robot environments, simply run:
```bash
./scripts/install_vla.sh --full   # without --full, only the edge model is downloaded
./scripts/install_deploy.sh
```

Connect the robot via Ethernet and give the interface (see `ip -br link`) an IP in the robot's subnet:
```bash
export GO2_IFACE=enxf8e43b9f046a
sudo ip addr add 192.168.123.51/24 dev $GO2_IFACE
ping 192.168.123.161
```

## Usage

The VLA and the bridge to the robot run in two terminals.
Here are the commands for a full run:
```bash
# Check what the robot sees, saves a labelled image to training/outputs/frames/view.jpg
./scripts/look.sh
# Terminal 1: start the VLA, type a new prompt and press Enter to change the goal
highlevel/.venv/bin/python -m highlevel.omnivla_7b_node --prompt "go to the white door" --max-v 0.5 --max-w 0.5
# Terminal 2: dry run, the robot does not move
deploy/.venv/bin/python -m deploy.go2_vla_bridge --via sport --duration 5 --max-v 0.5 --min-v 0.35
# Terminal 2: live run, the robot walks for 5 seconds
deploy/.venv/bin/python -m deploy.go2_vla_bridge --via sport --duration 5 --max-v 0.5 --min-v 0.35 --live
```

> [!WARNING]
> The VLA does not stop at the goal or avoid obstacles. Keep the floor clear and the remote in hand.

### With our own RL policy

Instead of sport mode, the legs can be driven by our Isaac Lab crawl policy (`deploy/run_policy.py`).
Start with the robot lying down, `run_policy.py` turns sport mode off.
```bash
# Terminal 1: camera frames for the VLA only, the bridge does not move the robot
deploy/.venv/bin/python -m deploy.go2_vla_bridge --camera-only --duration 600
# Terminal 2: policy with velocity commands from the VLA, keys s = stand, p = policy, l = lie down, d = damp
deploy/.venv/bin/python deploy/run_policy.py $GO2_IFACE \
  --policy deploy/policies/go2_crawl_v3.npz --cmd_source bus --bus_max_v 0.5
# Terminal 3: once the robot is in the policy state, start the VLA, Ctrl+C stops the robot
highlevel/.venv/bin/python -m highlevel.omnivla_7b_node --prompt "go to the wooden pallet" --max-v 0.5 --max-w 0.5
```
Moving a stick overrides the VLA, and without VLA commands the robot stands still.
The crawl policy barely moves below ~0.5 m/s, so keep `--max-v` at 0.5 or more.
Details and test results: [docs/real_robot_setup.md](docs/real_robot_setup.md#9c-crawl-v3-driven-by-the-vla--2026-11-02).

## Tips

* Pick prompts with objects that `look.sh` finds. Distinct objects like doors work best,
  motion commands like "turn left" don't.
* Keep `--min-v` around 0.35, the Go2 stalls at lower speeds.
* `highlevel.omnivla_edge_node` is faster than the 7B model but mostly ignores the prompt.
* Each run is logged to `training/outputs/real_vla/`.
* If the robot is not found, check the cable and `GO2_IFACE`.

## Locomotion RL in Isaac Lab

This part is separate from the VLA and still work in progress: we train our own walking policy
for the Go2 in Isaac Lab. See above for running it with the VLA.

### Setup

Environment is managed with [uv](https://docs.astral.sh/uv/), pinned to
Python 3.11 (required by Isaac Sim 5.x). Isaac Sim + Isaac Lab are a
multi-GB install pulled from NVIDIA's pip index rather than PyPI, which
also pulls in the matched torch/rsl-rl/gymnasium versions — see the
comment in `pyproject.toml` for why this isn't a plain `uv add`:

```bash
./scripts/install_isaaclab.sh
```

This creates `.venv` (Python 3.11.16) and installs
`isaaclab[isaacsim,all]==2.3.2.post1`, giving torch 2.7.0+cu126,
rsl-rl-lib 3.0.1, gymnasium 1.2.0.

Unitree's own Isaac-Lab-based reference implementation for Go2
(`unitree_rl_lab`) is the recommended starting point for the env/task
config. Clone and install it into the same `.venv` once Isaac Lab is installed:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_lab third_party/unitree_rl_lab
cd third_party/unitree_rl_lab && ./unitree_rl_lab.sh -i
```

(Note: PLAN.md's "Open questions" section mentions `unitree_rl_gym` as the
Go2 reference — that repo targets the older Isaac Gym / MuJoCo stack.
Since the plan settled on Isaac Lab as the simulator, `unitree_rl_lab` is
the actual matching reference repo and is what's linked above.)

### Virtual environments

| Venv | Python | Used for | Contents | Created by |
|------|--------|----------|----------|------------|
| `.venv` (training) | 3.11 | Isaac Lab training/rollouts, policy export (`deploy/export_policy_npz.py`), plotting | Isaac Sim/Lab, torch, rsl-rl, matplotlib, numpy 1.26 | `./scripts/install_isaaclab.sh` |
| `highlevel/.venv` (VLA) | 3.10 | OmniVLA nodes, object detection | torch 2.2, OmniVLA, transformers | `./scripts/install_vla.sh` |
| `deploy/.venv` (robot) | 3.10 | Everything that talks to the real Go2 (`deploy/read_*.py`, `shadow_policy.py`, `run_policy.py`, `go2_vla_bridge.py`) | `unitree_sdk2py` (+ `cyclonedds==0.10.2`), numpy 2.x, pillow — **no torch, no Isaac** | `./scripts/install_deploy.sh` |

They are kept separate because their torch/numpy/cyclonedds pins clash, and the
robot side should stay small enough to also run on the Go2's onboard computer.
Isaac Sim 5.x needs Python 3.11; the robot env uses 3.10 because `cyclonedds==0.10.2`
only has prebuilt wheels up to 3.10, so no CycloneDDS C build is needed.
The envs only exchange data: policies as a numpy `.npz` (`deploy/export_policy_npz.py`),
velocity commands and camera frames over UDP (`command_bus/`).
Don't pickle anything that crosses envs, and keep shared code 3.10-compatible.

Details and the per-user DDS log-file patch: [docs/real_robot_setup.md](docs/real_robot_setup.md).

## Layout

- `sim/` — Isaac Lab env, task configs, domain randomization
- `training/` — rsl_rl PPO configs, training scripts
- `actuator_net/` — real motor data logging + system ID fitting
- `deploy/` — policy export to numpy, unitree_sdk2 shadow/closed-loop deploy, plotting (see `deploy/README.md`), VLA bridge
- `highlevel/` — VLA nodes and object detection
- `command_bus/` — command topic def, arbitration, RC driver, e-stop supervisor
- `docs/` — notes, design docs
