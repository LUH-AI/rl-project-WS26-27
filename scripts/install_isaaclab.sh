#!/usr/bin/env bash
# Install Isaac Sim + Isaac Lab into .venv, plus unitree_rl_lab with our Go2 tasks.
# Versions come from pyproject.toml and uv.lock. Re-run to repair/update.
#
# Requires: an NVIDIA GPU + driver, ~20GB free disk.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RL_LAB=third_party/unitree_rl_lab
UNITREE_ROS=third_party/unitree_ros
# Upstream commits our tasks and example policies were trained with
RL_LAB_COMMIT=4960b84732b0c2ec593dccbfe963fda1bcd7b1e3
UNITREE_ROS_COMMIT=25746e8c73f097c3b32bed7b6c16d4e195dfacd8

# clone_at <url> <dir> <commit>: shallow clone of one commit, skipped if <dir> exists
clone_at() {
    [ -d "$2" ] && return
    git init -q "$2"
    git -C "$2" fetch -q --depth 1 "$1" "$3"
    git -C "$2" checkout -q FETCH_HEAD
}

# unitree_rl_lab, our task configs, and the Go2 URDF from unitree_ros
clone_at https://github.com/unitreerobotics/unitree_rl_lab "$RL_LAB" "$RL_LAB_COMMIT"
clone_at https://github.com/unitreerobotics/unitree_ros "$UNITREE_ROS" "$UNITREE_ROS_COMMIT"
cp training/go2_tasks/*.py "$RL_LAB/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/"
# Spawn the Go2 from the unitree_ros URDF instead of the separately downloaded USD
python3 - "$RL_LAB/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py" "$PWD/$UNITREE_ROS" <<'EOF'
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

uv sync --locked

# First import triggers an interactive Omniverse Kit EULA prompt; accept it
# once here so later runs (including headless training) don't block on it.
echo "Yes" | .venv/bin/python -c "import isaaclab"

echo "Done. Verify with: isaacsim"
