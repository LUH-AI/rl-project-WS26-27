# training/

`go2_tasks/` holds our Go2 tasks for [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab).
`scripts/install_isaaclab.sh` copies them into it. Edit them here, since `third_party/` is gitignored.

| Task | Change from the stock `Unitree-Go2-Velocity` |
|------|----------------------------------------------|
| `Unitree-Go2-Velocity-Crawl` | one-foot-at-a-time gait reward + base height reward (0.22 m, weight −120) |
| `Unitree-Go2-Velocity-LowHeight` | base height reward only (0.30 m) |
| `Unitree-Go2-Velocity-Pace` | pace gait reward |

Tools for checking trained policies (`.venv` active). `training/outputs/` is gitignored.

```bash
# training curves
python training/plot_results.py --out training/outputs/training_curves.png \
  --run_dir third_party/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/<run>
# headless rollout to .npz (Isaac's camera rendering crashes here, so state only), from third_party/unitree_rl_lab
python ../../training/rollout_and_log.py --task Unitree-Go2-Velocity --num_envs 1 --steps 500 \
  --out ../../training/outputs/rollout.npz
# render the rollout in MuJoCo (needs third_party/mujoco_go2)
MUJOCO_GL=egl python training/render_mujoco.py --trajectory training/outputs/rollout.npz \
  --model third_party/mujoco_go2/scene.xml --out training/outputs/go2_rollout.mp4
```
