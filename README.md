# Go2 EDU Sim2Real RL Pipeline

## Overview

This repo has two parts: locomotion RL for the Go2 in Isaac Lab (Stage 1, see [PLAN.md](PLAN.md))
and a VLA that drives the real Go2 from its front camera and a text prompt (Stage 2).
For Stage 2 we use [OmniVLA](https://github.com/NHirose/OmniVLA) to predict where to walk,
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

## Tips

* Pick prompts with objects that `look.sh` finds. Distinct objects like doors work best,
  motion commands like "turn left" don't.
* Keep `--min-v` around 0.35, the Go2 stalls at lower speeds.
* `highlevel.omnivla_edge_node` is faster than the 7B model but mostly ignores the prompt.
* Each run is logged to `training/outputs/real_vla/`.
* If the robot is not found, check the cable and `GO2_IFACE`.

## Stage 1 setup (Isaac Lab)

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
config referenced in PLAN.md — clone and editable-install it against this
same `.venv` once Isaac Lab is installed:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_lab third_party/unitree_rl_lab
cd third_party/unitree_rl_lab && ./unitree_rl_lab.sh -i
```

(Note: PLAN.md's "Open questions" section mentions `unitree_rl_gym` as the
Go2 reference — that repo targets the older Isaac Gym / MuJoCo stack.
Since the plan settled on Isaac Lab as the simulator, `unitree_rl_lab` is
the actual matching reference repo and is what's linked above.)

## Layout

- `sim/` — Isaac Lab env, task configs, domain randomization
- `training/` — rsl_rl PPO configs, training scripts
- `actuator_net/` — real motor data logging + system ID fitting
- `deploy/` — ONNX export, unitree_sdk2 control loop, Orin NX runtime, VLA bridge
- `highlevel/` — VLA nodes and object detection
- `command_bus/` — command topic def, arbitration, RC driver, e-stop supervisor
- `docs/` — notes, design docs
