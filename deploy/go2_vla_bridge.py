"""Bridge between the Go2 and the command bus."""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time

from PIL import Image
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import (
    ObstaclesAvoidClient,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from command_bus.bus import CommandArbiter
from command_bus.frames import FramePublisher

CONTROL_HZ = 20
CAMERA_HZ = 10


class RobotState:
    """Keeps the latest sport mode state of the robot."""

    def __init__(self):
        self.state = None
        self.subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.subscriber.Init(self.update, 10)

    def update(self, msg: SportModeState_) -> None:
        self.state = msg

    def to_dict(self) -> dict:
        if self.state is None:
            return {}
        return {
            "x": round(self.state.position[0], 3),
            "y": round(self.state.position[1], 3),
            "yaw": round(self.state.imu_state.rpy[2], 3),
            "vx_measured": round(self.state.velocity[0], 3),
            "wz_measured": round(self.state.yaw_speed, 3),
            "mode": self.state.mode,
        }


class CameraStream:
    """Sends robot camera frames to the VLA in the background."""

    def __init__(self, save_dir: str):
        self.video = VideoClient()
        self.video.SetTimeout(3.0)
        self.video.Init()
        self.frames = FramePublisher()
        self.save_dir = save_dir
        self.num_frames = 0
        self.num_errors = 0
        self.last_frame_time = time.time()
        self.last_error = ""
        self.running = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.running.set()
        self.thread.start()

    def stop(self) -> None:
        self.running.clear()

    def run(self) -> None:
        while self.running.is_set():
            step_start = time.time()
            try:
                code, data = self.video.GetImageSample()
                if code != 0 or not data:
                    raise RuntimeError(f"GetImageSample code={code}")
                image = Image.open(io.BytesIO(bytes(data))).convert("RGB")
                self.frames.send(image)
                self.num_frames += 1
                self.last_frame_time = time.time()

                # Save about one frame per second
                if self.num_frames % CAMERA_HZ == 0:
                    path = os.path.join(self.save_dir, f"{self.num_frames:05d}.jpg")
                    image.resize((480, 270)).save(path)
            except Exception as e:
                self.num_errors += 1
                self.last_error = repr(e)
            time.sleep(max(0.0, 1 / CAMERA_HZ - (time.time() - step_start)))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iface",
        default=os.environ.get("GO2_IFACE", "enxf8e43b9f046a"),
        help="network interface connected to the robot",
    )
    parser.add_argument(
        "--live", action="store_true", help="send commands to the robot"
    )
    parser.add_argument("--duration", type=float, default=30.0, help="seconds")
    parser.add_argument("--max-v", type=float, default=0.3, help="max forward speed")
    parser.add_argument(
        "--min-v",
        type=float,
        default=0.0,
        help="min forward speed when moving, the Go2 stalls below ~0.3",
    )
    parser.add_argument("--max-w", type=float, default=0.5, help="max turn rate")
    parser.add_argument(
        "--via",
        choices=["avoid", "sport"],
        default="avoid",
        help="move through obstacle avoidance or plain sport mode",
    )
    parser.add_argument(
        "--block-remote", action="store_true", help="ignore the remote during the run"
    )
    parser.add_argument("--log", default="training/outputs/real_vla")
    return parser


def main() -> None:
    args = get_parser().parse_args()

    # Make log dir
    run_name = time.strftime("%Y%m%d-%H%M%S") + ("-live" if args.live else "-dry")
    run_dir = os.path.join(args.log, run_name)
    os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)
    log_file = open(os.path.join(run_dir, "commands.jsonl"), "w")

    # Connect to robot
    ChannelFactoryInitialize(0, args.iface)
    robot = RobotState()
    camera = CameraStream(os.path.join(run_dir, "frames"))
    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()

    if args.live:
        motion_switcher = MotionSwitcherClient()
        motion_switcher.SetTimeout(5.0)
        motion_switcher.Init()
        _, mode = motion_switcher.CheckMode()
        if not mode.get("name"):
            raise SystemExit("Sport mode is off, stand the robot up first.")
        sport.BalanceStand()
        time.sleep(1.0)
        if args.block_remote:
            sport.SwitchJoystick(False)

    # With obstacle avoidance on, plain Move() gets overridden by the remote
    # after a few seconds, so we move through the avoidance service instead
    avoid = None
    if args.live and args.via == "avoid":
        avoid = ObstaclesAvoidClient()
        avoid.SetTimeout(3.0)
        avoid.Init()
        while not avoid.SwitchGet()[1]:
            avoid.SwitchSet(True)
            time.sleep(0.1)
        avoid.UseRemoteCommandFromApi(True)
    move = avoid.Move if avoid else sport.Move

    def stop_robot() -> None:
        move(0.0, 0.0, 0.0)
        if not avoid:
            sport.StopMove()

    arbiter = CommandArbiter()
    camera.start()
    mode_name = f"LIVE via {args.via}" if args.live else "DRY RUN"
    print(f"{mode_name} for {args.duration:.0f}s, logging to {run_dir}", flush=True)

    start = next_step = last_print = time.time()
    moving = False
    try:
        while time.time() - start < args.duration:
            # Clip command
            cmd = arbiter.get()
            vx = min(max(cmd.vx, 0.0), args.max_v)
            if 0.0 < vx < args.min_v:
                vx = min(args.min_v, args.max_v)
            wz = min(max(cmd.yaw_rate, -args.max_w), args.max_w)
            has_command = cmd.source != "none"

            # Send command, stop once if the VLA goes quiet
            if args.live:
                if has_command:
                    move(vx, 0.0, wz)
                    moving = True
                elif moving:
                    stop_robot()
                    moving = False

            # Log command and robot state
            now = time.time()
            measured = robot.to_dict()
            entry = {
                "t": round(now - start, 3),
                "source": cmd.source,
                "vx": round(vx, 3),
                "wz": round(wz, 3),
                "sent": args.live and has_command,
                **measured,
            }
            log_file.write(json.dumps(entry) + "\n")

            # Print status
            if now - last_print > 1.0:
                last_print = now
                status = (
                    f"t={now - start:5.1f}s frames={camera.num_frames} "
                    f"cam_err={camera.num_errors} src={cmd.source:>9} "
                    f"vx={vx:+.2f} wz={wz:+.2f}"
                )
                if not args.live:
                    status += "  (not sent)"
                if measured:
                    status += (
                        f" | robot x={measured['x']:+.2f} y={measured['y']:+.2f} "
                        f"v={measured['vx_measured']:+.2f}"
                    )
                print(status, flush=True)
                if now - camera.last_frame_time > 1.0:
                    print(
                        f"Warning: no camera frame for {now - camera.last_frame_time:.1f}s"
                        f" ({camera.last_error or 'no error'})",
                        flush=True,
                    )

            next_step += 1 / CONTROL_HZ
            time.sleep(max(0.0, next_step - time.time()))
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        camera.stop()
        if args.live:
            stop_robot()
            if avoid:
                time.sleep(0.2)
                avoid.UseRemoteCommandFromApi(False)
            if args.block_remote:
                sport.SwitchJoystick(True)
            print("Robot stopped.")
        log_file.close()
        print(f"Done! Sent {camera.num_frames} frames, log in {run_dir}")


if __name__ == "__main__":
    main()
