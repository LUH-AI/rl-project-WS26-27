"""Parse the Go2 wireless remote from LowState_.wireless_remote (40 bytes).

Layout (unitree xRockerBtnDataStruct): head[2], buttons uint16, lx, rx, ry,
L2 (analog), ly as little-endian float32, then padding. Verify the mapping on
the real remote with `deploy/read_remote.py` before relying on it.
"""

import struct

BUTTONS = [
    "R1", "L1", "start", "select", "R2", "L2", "F1", "F2",
    "A", "B", "X", "Y", "up", "right", "down", "left",
]


class Remote:
    def __init__(self):
        self.pressed = {b: False for b in BUTTONS}
        self._prev = dict(self.pressed)
        self.lx = self.ly = self.rx = self.ry = 0.0

    def update(self, wireless_remote):
        raw = bytes(wireless_remote)
        btn = struct.unpack_from("<H", raw, 2)[0]
        self.lx, self.rx, self.ry, _, self.ly = struct.unpack_from("<5f", raw, 4)
        self._prev = self.pressed
        self.pressed = {b: bool(btn >> i & 1) for i, b in enumerate(BUTTONS)}

    def on_pressed(self, button):
        """True only on the update where the button went down."""
        return self.pressed[button] and not self._prev[button]

    def combo(self, hold, press):
        """`hold` held and `press` just pressed, e.g. combo("L2", "B")."""
        return self.pressed[hold] and self.on_pressed(press)
