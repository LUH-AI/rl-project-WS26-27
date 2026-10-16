"""Camera frame stream over UDP."""

from __future__ import annotations

import io
import socket
import struct
import time

from PIL import Image

DEFAULT_FRAME_ADDR = ("127.0.0.1", 15556)
SEND_SIZE = (224, 224)


class FramePublisher:
    def __init__(self, addr: tuple[str, int] = DEFAULT_FRAME_ADDR):
        self.addr = addr
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, image) -> None:
        """Send a PIL image or uint8 array."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        buf = io.BytesIO()
        image.convert("RGB").resize(SEND_SIZE).save(buf, format="JPEG", quality=85)
        self.sock.sendto(struct.pack("!d", time.time()) + buf.getvalue(), self.addr)


class FrameSubscriber:
    def __init__(self, addr: tuple[str, int] = DEFAULT_FRAME_ADDR):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(addr)
        self.sock.setblocking(False)

    def latest(self) -> tuple[float, Image.Image] | None:
        """Get the newest frame and drop the rest."""
        data = None
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
        if data is None:
            return None
        (stamp,) = struct.unpack("!d", data[:8])
        return stamp, Image.open(io.BytesIO(data[8:])).convert("RGB")
