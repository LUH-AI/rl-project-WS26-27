# Go2 EDU Sim2Real RL Pipeline

See [PLAN.md](PLAN.md) for the full staged plan. This is Stage 1
(proprioceptive locomotion) scaffolding.

## Setup

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
- `deploy/` — ONNX export, unitree_sdk2 control loop, Orin NX runtime
- `command_bus/` — command topic def, arbitration, RC driver, e-stop supervisor
- `docs/` — notes, design docs
