# command_bus/

Local UDP messages between the venvs (JSON and JPEG only, no pickle, Python 3.10 compatible).

| File | Port | Content |
|------|------|---------|
| `bus.py` | 15555 | Velocity commands `(vx, vy, yaw_rate, source)` from the VLA. Older than 0.5 s counts as stop |
| `frames.py` | 15556 | Front camera frames (224×224 JPEG) from `deploy/vla_bridge.py` to the VLA |
