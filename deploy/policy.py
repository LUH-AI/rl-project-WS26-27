"""Numpy-only Go2 policy: observation building + MLP forward pass.

Mirrors unitree_rl_lab's C++ deploy (deploy/include/unitree_articulation.h,
isaaclab/envs/mdp/observations/observations.h). All policy-side vectors are in
Isaac Lab joint order; `joint_ids_map[i]` is the SDK motor index of Isaac
joint i (SDK order: FR, FL, RR, RL x hip, thigh, calf).
"""

import numpy as np

OBS_TERMS = [
    "base_ang_vel", "projected_gravity", "velocity_commands",
    "joint_pos_rel", "joint_vel_rel", "last_action",
]
GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def quat_rotate_inverse(q_wxyz, v):
    """Rotate world vector v into the body frame given body quaternion (w, x, y, z)."""
    w, xyz = q_wxyz[0], q_wxyz[1:]
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


class Go2Policy:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        n = int(d["num_layers"])
        self.weights = [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)]
        self.joint_ids_map = d["joint_ids_map"]
        self.default_joint_pos = d["default_joint_pos"]
        self.stiffness = d["stiffness"]
        self.damping = d["damping"]
        self.step_dt = float(d["step_dt"])
        self.action_scale = d["action_scale"]
        self.action_offset = d["action_offset"]
        self.action_clip = d["action_clip"]
        self.cmd_ranges = d["cmd_ranges"]
        self.obs_scale = {k: d[f"obs_scale_{k}"] for k in OBS_TERMS}
        self.obs_clip = {k: d[f"obs_clip_{k}"] for k in OBS_TERMS}
        self.num_actions = len(self.joint_ids_map)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        out = self.forward(d["test_input"])
        err = np.abs(out - d["test_output"]).max()
        if err > 1e-4:
            raise RuntimeError(f"numpy forward pass mismatches torch export (max err {err})")

    def forward(self, x):
        for i, (W, b) in enumerate(self.weights):
            x = x @ W.T + b
            if i < len(self.weights) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
        return x

    def read_state(self, lowstate):
        """Pull base / joint state from a LowState_ into Isaac joint order."""
        imu = lowstate.imu_state
        q_sdk = np.array([m.q for m in lowstate.motor_state[:12]], dtype=np.float32)
        dq_sdk = np.array([m.dq for m in lowstate.motor_state[:12]], dtype=np.float32)
        return {
            "ang_vel_b": np.array(imu.gyroscope, dtype=np.float32),
            "projected_gravity_b": quat_rotate_inverse(
                np.array(imu.quaternion, dtype=np.float32), GRAVITY_W
            ),
            "joint_pos": q_sdk[self.joint_ids_map],
            "joint_vel": dq_sdk[self.joint_ids_map],
        }

    def build_obs(self, state, cmd):
        terms = {
            "base_ang_vel": state["ang_vel_b"],
            "projected_gravity": state["projected_gravity_b"],
            "velocity_commands": np.asarray(cmd, dtype=np.float32),
            "joint_pos_rel": state["joint_pos"] - self.default_joint_pos,
            "joint_vel_rel": state["joint_vel"],
            "last_action": self.last_action,
        }
        parts = []
        for k in OBS_TERMS:
            lo, hi = self.obs_clip[k]
            parts.append(np.clip(terms[k] * self.obs_scale[k], lo, hi))
        return np.concatenate(parts).astype(np.float32)

    def step(self, state, cmd):
        """Returns (obs, raw action, joint position targets in Isaac order)."""
        obs = self.build_obs(state, cmd)
        action = self.forward(obs[None])[0].astype(np.float32)
        self.last_action = action
        # Isaac Lab's JointPositionAction clips the processed target, not the raw action.
        target = self.action_offset + self.action_scale * action
        target = np.clip(target, self.action_clip[:, 0], self.action_clip[:, 1])
        return obs, action, target

    def to_sdk_order(self, x_isaac):
        x_sdk = np.zeros_like(x_isaac)
        x_sdk[self.joint_ids_map] = x_isaac
        return x_sdk
