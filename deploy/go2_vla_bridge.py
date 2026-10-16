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


def camera_loop(
    video: VideoClient,
    frames: FramePublisher,
    stop: threading.Event,
    stats: dict,
    save_dir: str | None,
) -> None:
    while not stop.is_set():
        t0 = time.time()
        try:
            code, data = video.GetImageSample()
            if code != 0 or not data:
                raise RuntimeError(f"GetImageSample code={code}")
            img = Image.open(io.BytesIO(bytes(data))).convert("RGB")
            frames.send(img)
            stats["frames"] += 1
            stats["last_frame"] = time.time()

            # Save about one frame per second
            if save_dir and stats["frames"] % CAMERA_HZ == 0:
                path = os.path.join(save_dir, f"{stats['frames']:05d}.jpg")
                img.resize((480, 270)).save(path)
        except Exception as e:
            stats["cam_errors"] += 1
            stats["last_error"] = repr(e)
        time.sleep(max(0.0, 1 / CAMERA_HZ - (time.time() - t0)))


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
    cmd_log = open(os.path.join(run_dir, "commands.jsonl"), "w")

    # Connect to robot
    ChannelFactoryInitialize(0, args.iface)
    robot = {}
    state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    state_sub.Init(lambda m: robot.update(state=m), 10)
    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()
    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()

    if args.live:
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, mode = msc.CheckMode()
        if not mode.get("name"):
            raise SystemExit("Sport mode is off, select 'mcf' and stand the robot up.")
        print("BalanceStand ->", sport.BalanceStand())
        time.sleep(1.0)
        if args.block_remote:
            print("SwitchJoystick(False) ->", sport.SwitchJoystick(False))

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
        print("UseRemoteCommandFromApi(True) ->", avoid.UseRemoteCommandFromApi(True))
    move = avoid.Move if avoid else sport.Move

    def stop_move() -> None:
        move(0.0, 0.0, 0.0)
        if not avoid:
            sport.StopMove()

    # Start camera stream
    frames = FramePublisher()
    arbiter = CommandArbiter()
    stop = threading.Event()
    stats = {"frames": 0, "cam_errors": 0, "last_frame": time.time(), "last_error": ""}
    cam = threading.Thread(
        target=camera_loop,
        args=(video, frames, stop, stats, os.path.join(run_dir, "frames")),
        daemon=True,
    )
    cam.start()

    mode_str = f"LIVE via {args.via}" if args.live else "DRY RUN"
    print(
        f"{mode_str}: {args.duration:.0f}s, max_v={args.max_v} max_w={args.max_w}, "
        f"log -> {run_dir}",
        flush=True,
    )
    t0 = nxt = last_print = time.time()
    moving = False
    try:
        while time.time() - t0 < args.duration:
            # Clip command
            c = arbiter.get()
            vx = min(max(c.vx, 0.0), args.max_v)
            if 0.0 < vx < args.min_v:
                vx = min(args.min_v, args.max_v)
            wz = min(max(c.yaw_rate, -args.max_w), args.max_w)
            active = c.source != "none"

            # Send command, stop once if the VLA goes quiet
            if args.live:
                if active:
                    move(vx, 0.0, wz)
                    moving = True
                elif moving:
                    stop_move()
                    moving = False

            # Log command and robot state
            now = time.time()
            rs = robot.get("state")
            meas = {}
            if rs is not None:
                meas = {
                    "x": round(rs.position[0], 3),
                    "y": round(rs.position[1], 3),
                    "yaw": round(rs.imu_state.rpy[2], 3),
                    "vx_meas": round(rs.velocity[0], 3),
                    "wz_meas": round(rs.yaw_speed, 3),
                    "mode": rs.mode,
                }
            entry = {
                "t": round(now - t0, 3),
                "source": c.source,
                "vx": round(vx, 3),
                "wz": round(wz, 3),
                "sent": args.live and active,
                **meas,
            }
            cmd_log.write(json.dumps(entry) + "\n")

            # Print status
            if now - last_print > 1.0:
                last_print = now
                line = (
                    f"t={now - t0:5.1f}s frames={stats['frames']} "
                    f"cam_err={stats['cam_errors']} src={c.source:>9} "
                    f"vx={vx:+.2f} wz={wz:+.2f}"
                )
                if not args.live:
                    line += "  (not sent)"
                if meas:
                    line += (
                        f" | robot x={meas['x']:+.2f} y={meas['y']:+.2f} "
                        f"v={meas['vx_meas']:+.2f} mode={meas['mode']}"
                    )
                else:
                    line += " | no robot state"
                print(line, flush=True)
                if now - stats["last_frame"] > 1.0:
                    print(
                        f"  Warning: no camera frame for {now - stats['last_frame']:.1f}s "
                        f"(last error: {stats['last_error'] or 'none'})",
                        flush=True,
                    )

            nxt += 1 / CONTROL_HZ
            time.sleep(max(0.0, nxt - time.time()))
    except KeyboardInterrupt:
        print("Ctrl-C")
    finally:
        stop.set()
        if args.live:
            stop_move()
            if avoid:
                time.sleep(0.2)
                print(
                    "UseRemoteCommandFromApi(False) ->",
                    avoid.UseRemoteCommandFromApi(False),
                )
            if args.block_remote:
                print("SwitchJoystick(True) ->", sport.SwitchJoystick(True))
            print("Stopped.")
        cmd_log.close()
        print(f"Done: {stats['frames']} frames sent, log in {run_dir}")


if __name__ == "__main__":
    main()
