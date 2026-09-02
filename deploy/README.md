# deploy/

ONNX export of the trained policy, and the `unitree_sdk2`-based low-level
control loop that runs it at 50 Hz on the Go2 EDU's onboard Jetson Orin NX.
Depends on the `deploy` extra (`unitree_sdk2py`, `onnxruntime-gpu`) — install
with `uv pip install -e ".[deploy]"`.
