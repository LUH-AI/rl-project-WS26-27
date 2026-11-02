"""Velocity command bus: senders publish over UDP, the robot side reads the newest fresh command."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass

DEFAULT_ADDR = ("127.0.0.1", 15555)


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    source: str = "none"
    stamp: float = 0.0


class CommandPublisher:
    def __init__(self, source: str, addr: tuple[str, int] = DEFAULT_ADDR):
        self.source = source
        self.addr = addr
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, vx: float, vy: float, yaw_rate: float) -> None:
        cmd = VelocityCommand(vx, vy, yaw_rate, self.source, time.time())
        self.sock.sendto(json.dumps(asdict(cmd)).encode(), self.addr)

    def stop(self) -> None:
        self.send(0.0, 0.0, 0.0)


class CommandSubscriber:
    def __init__(self, addr: tuple[str, int] = DEFAULT_ADDR, timeout: float = 0.5):
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(addr)
        self.sock.setblocking(False)
        self.latest = VelocityCommand(source="none")

    def _drain(self) -> None:
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            try:
                cmd = VelocityCommand(**json.loads(data))
            except (ValueError, TypeError):
                continue
            # Use our own clock in case sender clocks differ
            cmd.stamp = time.time()
            self.latest = cmd

    def get(self) -> VelocityCommand:
        """Get the newest command, or a zero command if it is older than the timeout."""
        self._drain()
        now = time.time()
        if now - self.latest.stamp > self.timeout:
            return VelocityCommand(source="none", stamp=now)
        return self.latest


if __name__ == "__main__":
    subscriber = CommandSubscriber()
    while True:
        cmd = subscriber.get()
        print(
            f"\r{cmd.source:>10}  vx={cmd.vx:+.2f} vy={cmd.vy:+.2f} wz={cmd.yaw_rate:+.2f}",
            end="",
        )
        time.sleep(0.1)
