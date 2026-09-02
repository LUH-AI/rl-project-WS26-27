# training/

Training itself lives in `third_party/unitree_rl_lab` (its own `rsl_rl` PPO
configs and `scripts/rsl_rl/train.py`) — see the repo root README for setup.
This directory holds the glue around that:

- `plot_results.py` — reads an rsl_rl/tensorboard run directory and plots
  reward, episode length, velocity-tracking error, and curriculum progress.

  ```bash
  python training/plot_results.py \
    --run_dir third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/<run> \
    --out training/outputs/training_curves.png
  ```

- `rollout_and_log.py` — loads a trained checkpoint, runs it headless
  (no camera) for N steps, and logs the joint + base trajectory to `.npz`.
  Isaac Sim's RTX camera renderer crashes in this environment (confirmed:
  independent of sandboxing, num_envs, or --headless — reproduces with just
  `--video` on as few as 32 envs / 3 iterations), so video isn't recorded
  in-process; this script sidesteps that by logging state only. Run from
  `third_party/unitree_rl_lab` so its `cli_args.py` is importable:

  ```bash
  cd third_party/unitree_rl_lab
  python ../../training/rollout_and_log.py --task Unitree-Go2-Velocity \
    --num_envs 1 --steps 500 --out ../../training/outputs/rollout.npz
  ```

- `render_mujoco.py` — replays a logged `.npz` trajectory kinematically in
  MuJoCo (joint names/quaternion convention match Isaac Sim's, so no
  reordering needed) and renders it to mp4 via MuJoCo's own offscreen
  renderer — unaffected by the Isaac RTX bug above. Needs
  `third_party/mujoco_go2` (mujoco_menagerie's Go2 MJCF, fetched once) and
  `MUJOCO_GL=egl` for headless GPU rendering:

  ```bash
  MUJOCO_GL=egl python training/render_mujoco.py \
    --trajectory training/outputs/rollout.npz \
    --model third_party/mujoco_go2/scene.xml \
    --out training/outputs/go2_rollout.mp4
  ```

`training/outputs/` is gitignored (large binary artifacts).
