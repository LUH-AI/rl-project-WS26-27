"""Object detection on a robot camera frame with OWLv2."""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL = "google/owlv2-base-patch16-ensemble"
INDOOR = [
    "door",
    "red door",
    "door frame",
    "wall",
    "window",
    "whiteboard",
    "table",
    "desk",
    "chair",
    "armchair",
    "cardboard box",
    "trash bin",
    "wooden pallet",
    "shelf",
    "ladder",
    "wall bars",
    "3D printer",
    "printer",
    "computer monitor",
    "laptop",
    "keyboard",
    "cable",
    "power socket",
    "light switch",
    "soap dispenser",
    "radiator",
    "plant",
    "bottle",
    "cup",
    "bag",
    "backpack",
    "person",
    "robot",
    "box",
    "cabinet",
    "lamp",
    "pole",
    "carpet",
    "fire extinguisher",
    "poster",
]
PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def detect(img: Image.Image, queries: list[str], thresh: float) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(device).eval()
    inputs = processor(text=[queries], images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # OWLv2 pads the image to a square
    side = max(img.size)
    res = processor.post_process_object_detection(
        outputs,
        threshold=thresh,
        target_sizes=torch.tensor([[side, side]], device=device),
    )[0]
    dets = sorted(
        (
            (float(s), queries[int(label)], [float(v) for v in box])
            for s, label, box in zip(res["scores"], res["labels"], res["boxes"])
        ),
        reverse=True,
    )

    # Keep only the best label per region
    kept = []
    for d in dets:
        if all(iou(d[2], k[2]) < 0.5 for k in kept):
            kept.append(d)
    return kept


def draw_boxes(img: Image.Image, dets: list) -> None:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, img.width // 60))
    except OSError:
        font = ImageFont.load_default()
    for i, (score, label, (x0, y0, x1, y1)) in enumerate(dets):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.width - 1, x1), min(img.height - 1, y1)
        color = PALETTE[i % len(PALETTE)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(3, img.width // 400))
        text = f"{label} {score:.2f}"
        tb = draw.textbbox((x0, y0), text, font=font)
        h = tb[3] - tb[1]
        ty = y0 - h - 6 if y0 > h + 6 else y0 + 2
        draw.rectangle([x0, ty, x0 + (tb[2] - tb[0]) + 8, ty + h + 6], fill=color)
        draw.text((x0 + 4, ty + 1), text, fill="white", font=font)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--queries", help="comma separated object names")
    parser.add_argument("--thresh", type=float, default=0.2)
    parser.add_argument(
        "--out",
        default="training/outputs/detections",
        help="output directory or .jpg path",
    )
    args = parser.parse_args()
    if args.queries:
        queries = [q.strip() for q in args.queries.split(",")]
    else:
        queries = INDOOR

    img = Image.open(args.image).convert("RGB")
    dets = detect(img, queries, args.thresh)
    draw_boxes(img, dets)

    # Save image
    if args.out.lower().endswith((".jpg", ".jpeg")):
        out = args.out
    else:
        name = os.path.splitext(os.path.basename(args.image))[0]
        out = os.path.join(args.out, name + "_detections.jpg")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    img.save(out, quality=90)

    # Print detections
    for score, label, (x0, y0, x1, y1) in dets:
        cx = (x0 + x1) / 2 / img.width
        side = "left" if cx < 0.4 else "right" if cx > 0.6 else "centre"
        print(
            f"{score:.2f}  {label:<18} {side:<6} box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})"
        )
    print(f"{len(dets)} objects -> {out}")


if __name__ == "__main__":
    main()
