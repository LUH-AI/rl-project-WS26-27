"""Velocity command bus with source arbitration."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass

DEFAULT_ADDR = ("127.0.0.1", 15555)

# Lower number = higher priority. Unknown sources rank below everything.
PRIORITY = {"rc": 0, "highlevel": 10, "script": 20}


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    source: str = "script"
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


class CommandArbiter:
    def __init__(self, addr: tuple[str, int] = DEFAULT_ADDR, timeout: float = 0.5):
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(addr)
        self.sock.setblocking(False)
        self.latest: dict[str, VelocityCommand] = {}

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
            self.latest[cmd.source] = cmd

    def get(self) -> VelocityCommand:
        """Get the highest priority fresh command."""
        self._drain()
        now = time.time()
        fresh = [c for c in self.latest.values() if now - c.stamp <= self.timeout]
        if not fresh:
            return VelocityCommand(source="none", stamp=now)
        return min(fresh, key=lambda c: PRIORITY.get(c.source, 100))


if __name__ == "__main__":
    arb = CommandArbiter()
    while True:
        c = arb.get()
        print(
            f"\r{c.source:>10}  vx={c.vx:+.2f} vy={c.vy:+.2f} wz={c.yaw_rate:+.2f}",
            end="",
        )
        time.sleep(0.1)
