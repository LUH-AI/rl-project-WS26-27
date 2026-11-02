#!/usr/bin/env bash
# Install Isaac Sim + Isaac Lab into .venv, plus unitree_rl_lab with our Go2 tasks.
# Re-run to repair/update.
#
# Requires: an NVIDIA GPU + driver, ~20GB free disk, Python 3.11 (pinned in
# pyproject.toml since Isaac Sim 5.x requires it).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RL_LAB=third_party/unitree_rl_lab
UNITREE_ROS=third_party/unitree_ros

if [ ! -d .venv ]; then
    uv venv --python 3.11.16 --seed .venv
fi
source .venv/bin/activate

# Isaac Sim + Isaac Lab from NVIDIA's pip index. This alone pulls the
# correctly matched torch/torchvision/rsl-rl/gymnasium/tensorboard versions
# for Isaac Sim 5.x, so nothing else needs installing separately first.
# Version pinned to match unitree_rl_lab's tested Isaac Lab 2.3.x / Isaac
# Sim 5.x requirement.
uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

# First import triggers an interactive Omniverse Kit EULA prompt; accept it
# once here so later runs (including headless training) don't block on it.
echo "Yes" | python -c "import isaaclab"

# unitree_rl_lab, our task configs, and the Go2 URDF from unitree_ros
[ -d "$RL_LAB" ] || git clone https://github.com/unitreerobotics/unitree_rl_lab "$RL_LAB"
[ -d "$UNITREE_ROS" ] || git clone --depth 1 https://github.com/unitreerobotics/unitree_ros "$UNITREE_ROS"
cp training/go2_tasks/*.py "$RL_LAB/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/"
# Spawn the Go2 from the unitree_ros URDF instead of the separately downloaded USD
python - "$RL_LAB/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py" "$PWD/$UNITREE_ROS" <<'EOF'
import sys
path, ros_dir = sys.argv[1:]
src = open(path).read()
src = src.replace('UNITREE_ROS_DIR = "path/to/unitree_ros"', f'UNITREE_ROS_DIR = "{ros_dir}"')
usd = '''    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd",
    ),'''
urdf = '''    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf",
    ),'''
if usd in src:
    src = src.replace(usd, urdf)
elif urdf not in src:
    sys.exit(f"Could not switch the Go2 spawn to URDF in {path}, edit it by hand")
open(path, "w").write(src)
EOF
(cd "$RL_LAB" && ./unitree_rl_lab.sh -i)

echo "Done. Verify with: isaacsim"
