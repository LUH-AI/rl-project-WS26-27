"""Roll out a trained MJX Go2 joystick policy and render it to mp4, using
plain MuJoCo (not MJX) for rendering — sidesteps a bug in this repo's own
render_policy_rollout utility for non-vision tasks (missing 'pipeline_states'
extra), and reuses the same replay-in-MuJoCo approach already used for the
Isaac Lab rollout animation.

    MUJOCO_GL=egl python sim_mjx/rollout_and_render_mjx.py \
        --experiment experiment_6 --steps 500 --out sim_mjx/go2_mjx_rollout.mp4
"""

import argparse
import pickle
import sys

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import linen

sys.path.insert(0, "third_party/unitree_go2_mjx_rl/scripts")
sys.path.insert(0, "third_party/unitree_go2_mjx_rl/src")
import paths  # noqa: E402
from quadruped_mjx_rl.configs import prepare_all_configs  # noqa: E402
from quadruped_mjx_rl.environments import get_env_factory  # noqa: E402
from quadruped_mjx_rl.terrain_gen import make_terrain  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--env_config", type=str, default="joystick_basic.yaml")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

experiment_dir = paths.EXPERIMENTS_DIRECTORY / args.experiment
robot_config, terrain_config, env_config, _vw, model_config, training_config = prepare_all_configs(
    paths.ROBOT_CONFIGS_DIRECTORY / "unitree_go2.yaml", paths.CONFIGS_DIRECTORY / args.env_config
)
env_model = make_terrain(resources_directory=paths.RESOURCES_DIRECTORY, terrain_config=terrain_config, robot_config=robot_config)
env = get_env_factory(robot_config=robot_config, environment_config=env_config, env_model=env_model, customize_model=True)()

with open(experiment_dir / "trained_policy", "rb") as f:
    params = pickle.load(f)
policy_params = params.network_params.policy["params"]
mean = jnp.asarray(params.preprocessor_params.mean["proprioceptive"])
std = jnp.asarray(params.preprocessor_params.std["proprioceptive"])
layer_names = ["Dense_0", "Dense_1", "Dense_2", "Dense_3", "Dense_4"]


def policy_fn(obs):
    x = (obs - mean) / std
    for i, name in enumerate(layer_names):
        x = x @ policy_params[name]["kernel"] + policy_params[name]["bias"]
        if i < len(layer_names) - 1:
            x = linen.swish(x)
    loc, _scale = jnp.split(x, 2, axis=-1)
    return jnp.tanh(loc)


reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)
policy_jit = jax.jit(policy_fn)

state = reset_fn(jax.random.PRNGKey(args.seed))
qpos_frames = []
commands = []
for _ in range(args.steps):
    action = policy_jit(state.obs["proprioceptive"])
    state = step_fn(state, action)
    qpos_frames.append(np.asarray(state.pipeline_state.data.qpos))
    commands.append(np.asarray(state.info["command"]))

print(f"Rolled out {len(qpos_frames)} steps. Sample command: {commands[0]}")

mj_model = mujoco.MjModel.from_xml_path(str(paths.RESOURCES_DIRECTORY / "unitree_go2" / "scene_mjx_vision.xml"))
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
cam = mujoco.MjvCamera()
cam.distance = 2.2
cam.elevation = -18
cam.azimuth = 120

frames = []
for qpos in qpos_frames:
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)
    cam.lookat = qpos[:3]
    renderer.update_scene(mj_data, camera=cam)
    frames.append(renderer.render())

imageio.mimsave(args.out, frames, fps=50, macro_block_size=1)
print(f"Wrote {len(frames)} frames to {args.out}")
