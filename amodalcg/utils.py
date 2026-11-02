import json
import math
import os
import random

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_by_id(path):
    """Load a jsonl file into a dict keyed by `id`. Returns {} if `path` is None or missing."""
    if path is None or not os.path.exists(path):
        return {}
    return {r["id"]: r for r in read_jsonl(path)}


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    """Load a binary mask as a {0, 1} uint8 array."""
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def save_mask(mask, path):
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Boxes. All boxes are [x1, y1, x2, y2] in pixel coordinates (inclusive).
# ---------------------------------------------------------------------------

def mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def add_margin(box, margin, h, w):
    """Enlarge `box` by `margin` (fraction of its size) in total, i.e. margin / 2 on each side."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    out = [
        box[0] - round(margin / 2 * bw),
        box[1] - round(margin / 2 * bh),
        box[2] + round(margin / 2 * bw),
        box[3] + round(margin / 2 * bh),
    ]
    return [max(0, out[0]), max(0, out[1]), min(w, out[2]), min(h, out[3])]


def enclose_box(box, inner, h, w):
    """Grow `box` so that it contains `inner`, clipped to the image."""
    return [
        max(0, min(inner[0], box[0])),
        max(0, min(inner[1], box[1])),
        min(w, max(inner[2], box[2])),
        min(h, max(inner[3], box[3])),
    ]


def box_to_mask(box, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[box[1]:box[3] + 1, box[0]:box[2] + 1] = 1
    return mask


def square_crop_region(mask, buffer=100):
    """Square region around `mask` with `buffer` pixels of context. Returns (y1, y2, x1, x2)."""
    h, w = mask.shape
    x1, y1, x2, y2 = mask_to_box(mask)
    y1, y2 = max(0, y1 - buffer), min(h, y2 + buffer)
    x1, x2 = max(0, x1 - buffer), min(w, x2 + buffer)

    size = max(y2 - y1, x2 - x1)
    if x2 - x1 < size:
        diff = (size - (x2 - x1)) / 2
        x1, x2 = max(0, x1 - math.floor(diff)), min(w, x2 + math.ceil(diff))
    if y2 - y1 < size:
        diff = (size - (y2 - y1)) / 2
        y1, y2 = max(0, y1 - math.floor(diff)), min(h, y2 + math.ceil(diff))
    return y1, y2, x1, x2


def mask_touches_box(mask, box, border=1, thresh=20):
    """
    Whether `mask` reaches any side of `box`, which signals that the completed object is
    likely truncated by the box. Sides lying on the image border are ignored.
    """
    h, w = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(w - 1, box[2]), min(h - 1, box[3])

    sides = []
    if x1 > 0:
        sides.append(((x1, y1), (x1, y2)))
    if x2 < w - 1:
        sides.append(((x2, y1), (x2, y2)))
    if y1 > 0:
        sides.append(((x1, y1), (x2, y1)))
    if y2 < h - 1:
        sides.append(((x1, y2), (x2, y2)))

    for p1, p2 in sides:
        line = np.zeros_like(m)
        cv2.line(line, p1, p2, 1, border)
        if (line & m).sum() >= min(thresh, line.sum() * 0.05):
            return True
    return False


# ---------------------------------------------------------------------------
# Visual prompts
# ---------------------------------------------------------------------------

def draw_box(image, box, color=(255, 0, 0), thickness=2):
    image = np.ascontiguousarray(image).copy()
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
    return image


def isolate_object(image, mask):
    """Keep only the pixels inside `mask`, paint everything else white."""
    return (image * mask[..., None] + 255 * (1 - mask[..., None])).astype(np.uint8)
