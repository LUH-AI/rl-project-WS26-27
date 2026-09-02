"""Render a logged Isaac Lab rollout (from rollout_and_log.py) as an mp4,
by replaying the joint/base trajectory kinematically in MuJoCo's offscreen
renderer. Sidesteps Isaac Sim's RTX camera renderer entirely.

Requires MUJOCO_GL=egl for headless GPU rendering:

    MUJOCO_GL=egl python training/render_mujoco.py \
        --trajectory training/outputs/rollout.npz \
        --model third_party/mujoco_go2/scene.xml \
        --out training/outputs/go2_rollout.mp4
"""

import argparse

import imageio
import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--trajectory", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=50)
args = parser.parse_args()

data_npz = np.load(args.trajectory)
joint_names = list(data_npz["joint_names"])
joint_pos = data_npz["joint_pos"]  # (T, 12)
root_pos = data_npz["root_pos"]  # (T, 3)
root_quat = data_npz["root_quat"]  # (T, 4) wxyz
command = data_npz["command"]  # (T, 3) vx, vy, yaw-rate
dt = float(data_npz["dt"])

model = mujoco.MjModel.from_xml_path(args.model)
data = mujoco.MjData(model)

# name -> qpos address, for the 12 hinge joints (freejoint occupies qpos[0:7])
qpos_adr = {name: model.joint(name).qposadr[0] for name in joint_names}

renderer = mujoco.Renderer(model, height=args.height, width=args.width)
cam = mujoco.MjvCamera()
cam.distance = 2.2
cam.elevation = -18
cam.azimuth = 120

frames = []
T = joint_pos.shape[0]
for t in range(T):
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = root_quat[t]
    for i, name in enumerate(joint_names):
        data.qpos[qpos_adr[name]] = joint_pos[t, i]
    mujoco.mj_forward(model, data)

    cam.lookat = root_pos[t]
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render())

imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
print(f"[INFO] Wrote {T} frames ({T / args.fps:.1f}s at {args.fps} fps) to {args.out}")
