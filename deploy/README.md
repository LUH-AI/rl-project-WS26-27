# deploy/

Runs Isaac Lab policies on the real Go2 over `unitree_sdk2_python` (DDS,
`rt/lowstate` / `rt/lowcmd`). The robot side needs only numpy + the SDK
(`deploy/.venv`); exporting and plotting use the training env (`.venv`).
Setup and bring-up log: [`docs/real_robot_setup.md`](../docs/real_robot_setup.md).

| Script | Venv | Purpose |
|--------|------|---------|
| `export_policy_npz.py` | `.venv` | Pack an rsl_rl checkpoint's actor weights + its `params/deploy.yaml` into one `.npz` in `policies/` (numpy-only deploy, self-checks against torch). |
| `go2_policy.py` | – | Library: builds the 45-dim observation from `LowState_` and runs the MLP; handles Isaac ↔ SDK joint order. |
| `go2_remote.py` | – | Library: parses the wireless remote (buttons, sticks) from `LowState_.wireless_remote`. |
| `read_lowstate.py` | `deploy/.venv` | Read-only connectivity check: prints IMU, battery and joint states. |
| `read_remote.py` | `deploy/.venv` | Read-only check of the remote button/stick mapping (run with sport mode off). |
| `shadow_policy.py` | `deploy/.venv` | Runs the policy on live robot state while publishing nothing; logs obs/actions. |
| `run_policy.py` | `deploy/.venv` | Closed-loop deployment: PASSIVE → STAND → POLICY → SIT state machine at 500 Hz (policy 50 Hz), remote/keyboard control, safety stops, `--dry_run`, `--cmd_source bus` for VLA commands. |
| `go2_vla_bridge.py` | `deploy/.venv` | Streams the camera to the VLA and drives sport mode with its commands; `--camera-only` just streams while `run_policy.py --cmd_source bus` walks. Run as `python -m deploy.go2_vla_bridge`. |
| `plot_deploy.py` | `.venv` | Plots a shadow/deploy log (base state, joints vs targets, optional real-vs-sim). |

Quick start (robot lying down, interface `enxf8e43b9f046a`):

```bash
.venv/bin/python deploy/export_policy_npz.py --run_dir <rsl_rl run dir> --out deploy/policies/go2_velocity.npz
deploy/.venv/bin/python deploy/run_policy.py enxf8e43b9f046a --policy deploy/policies/go2_velocity.npz
# keys (+Enter): s = stand, p = policy, l = lie down, d = emergency damp, q = quit
.venv/bin/python deploy/plot_deploy.py --real deploy_log.npz --out_prefix deploy/plots/run
```

`policies/` and `plots/` are gitignored.
