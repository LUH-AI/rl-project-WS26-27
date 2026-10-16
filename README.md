# Go2 EDU Sim2Real RL Pipeline

See [PLAN.md](PLAN.md) for the full staged plan.
This repo contains two stages: locomotion RL for the Go2 in Isaac Lab (Stage 1) and a VLA that drives the real Go2 from its camera and a text goal (Stage 2).
If you want to get the robot walking to a goal, start with Stage 2 below.

## Stage 2: VLA on the real Go2
We use [OmniVLA](https://github.com/NHirose/OmniVLA) to turn the front camera image and a prompt like "go to the white door" into forward and turning velocities.
Unitree's own sport mode controller then moves the legs.
The VLA and the robot SDK run in separate environments and talk to each other over UDP on localhost.

You need an NVIDIA GPU with at least 20GB of memory for the 7B model (we use an RTX 4090), about 20GB of disk space and [uv](https://docs.astral.sh/uv/).
The smaller edge model runs with ~2GB.

### Installation
Install the VLA environment with the OmniVLA checkpoints and the robot environment with the Unitree SDK:

```bash
./scripts/install_vla.sh --full
./scripts/install_deploy.sh
```

Without `--full` you only get the edge model, which skips the 16GB download of the 7B checkpoint.

### Connect to the Robot
Connect the robot to your PC via Ethernet and give the interface an IP in the robot's subnet.
You can find the interface name with `ip -br link`:

```bash
export GO2_IFACE=enxf8e43b9f046a
sudo ip addr add 192.168.123.51/24 dev $GO2_IFACE
ping -c 3 192.168.123.161
```

All scripts read the interface from `GO2_IFACE`, alternatively you can pass `--iface`.

### See What the Robot Sees
Before picking a goal, it's useful to check what the robot can actually see:

```bash
./scripts/look.sh
```

This saves the camera image with labelled objects to `training/outputs/frames/view.jpg` and prints the detections.
You can also look for specific objects with `--queries "white door,trash bin"`.
The robot doesn't move during this.

### Run the VLA
Start the VLA in one terminal and wait until it prints `OmniVLA 7B on cuda:0`:

```bash
highlevel/.venv/bin/python -m highlevel.omnivla_7b_node --prompt "go to the white door" --max-v 0.5 --max-w 0.5
```

The edge model works the same way with `highlevel.omnivla_edge_node`, but it mostly ignores the prompt.
To change the goal, simply type a new prompt into this terminal and press Enter.
You can keep the VLA running between runs.

Then start the bridge to the robot in a second terminal.
We recommend a dry run first, this runs everything but doesn't send commands to the robot:

```bash
deploy/.venv/bin/python -m deploy.go2_vla_bridge --via sport --duration 5 --max-v 0.5 --min-v 0.35
```

If the commands look reasonable, add `--live` to let the robot walk:

```bash
deploy/.venv/bin/python -m deploy.go2_vla_bridge --via sport --duration 5 --max-v 0.5 --min-v 0.35 --live
```

Make sure the robot is standing in sport mode, the floor in front is clear and someone holds the remote.
The robot stops after `--duration` seconds, on Ctrl-C or if the VLA stops sending commands.
Each run is logged to `training/outputs/real_vla/`.
Keep `--min-v` around 0.35, the Go2 stalls at lower speeds.
For all other options, call the scripts with `--help`.

### Limitations
The VLA only walks forward and steers, it can't go backwards or sideways and always asks for full speed.
It works best with distinct objects like doors and does not understand motion commands like "turn left".
It also doesn't stop at the goal or avoid obstacles, so stay close to the robot!

### Troubleshooting
- `does not match an available interface`: the cable is unplugged, the robot is off or `GO2_IFACE` is wrong.
- `Sport mode is off`: stand the robot up with the remote or the app first.
- The robot shuffles and stops: increase `--min-v`.
- The 7B model fails on import with protobuf errors: rerun `./scripts/install_vla.sh`.

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
