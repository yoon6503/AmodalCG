"""MLLM guidance: Guidance Decision (InternVL), Geometric and Semantic Guidance (GPT-4o)."""
import base64
import json
import re
from io import BytesIO

import numpy as np
from PIL import Image

from .prompts import (
    DECISION_PROMPT,
    GEOMETRIC_SYSTEM_PROMPT,
    GEOMETRIC_USER_PROMPT,
    SEMANTIC_SYSTEM_PROMPT,
    SEMANTIC_USER_PROMPT,
)
from .utils import draw_box, isolate_object, mask_to_box, square_crop_region


# ---------------------------------------------------------------------------
# Visual prompts
# ---------------------------------------------------------------------------

def decision_visual_prompt(image, modal_mask):
    """Visible target on a white background, boxed in red and cropped with a 100px margin."""
    h, w = modal_mask.shape
    x1, y1, x2, y2 = mask_to_box(modal_mask)
    box = [max(0, x1 - 5), max(0, y1 - 5), min(w, x2 + 6), min(h, y2 + 6)]
    canvas = draw_box(isolate_object(image, modal_mask), box)
    y1, y2, x1, x2 = square_crop_region(modal_mask, buffer=100)
    return Image.fromarray(canvas[y1:y2, x1:x2])


def geometric_visual_prompt(image, modal_mask):
    """Full-size image with only the visible target kept, its visible box marked in red."""
    return Image.fromarray(draw_box(isolate_object(image, modal_mask), mask_to_box(modal_mask)))


def semantic_visual_prompt(image, modal_mask, occluder_mask):
    """Occluders masked out, visible target marked in red, cropped with a 100px margin."""
    h, w = modal_mask.shape
    x1, y1, x2, y2 = mask_to_box(modal_mask)
    box = [max(0, x1 - 10), max(0, y1 - 10), min(w - 1, x2 + 10), min(h - 1, y2 + 10)]
    canvas = draw_box(image * (1 - occluder_mask[..., None]), box)
    y1, y2, x1, x2 = square_crop_region(modal_mask, buffer=100)
    return Image.fromarray(canvas[y1:y2, x1:x2].astype(np.uint8))


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_decision(text):
    """Parse the decision module's JSON output. Unparsable outputs default to requesting guidance."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    try:
        out = json.loads(match.group(0))
    except (AttributeError, json.JSONDecodeError):
        return {"requires_guidance": True, "category": None, "raw": text}
    answer = str(out.get("requires_extensive_completion", "yes")).strip().lower()
    return {"requires_guidance": not answer.startswith("no"), "category": out.get("category"), "raw": text}


def parse_boxes(text):
    """Parse 'Tight ...: [x1, y1, x2, y2]' style answers into a dict of three boxes."""
    boxes = [list(map(int, map(float, b))) for b in re.findall(
        r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", text)]
    if len(boxes) < 3:
        raise ValueError(f"Expected three boxes, got: {text!r}")
    return dict(zip(("tight", "moderate", "coarse"), boxes[:3]))


def parse_sd_prompt(text):
    return text.split("Prompt:")[-1].strip().strip('"')


# ---------------------------------------------------------------------------
# InternVL (Guidance Decision Module)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """InternVL dynamic tiling (from the official InternVL inference example)."""
    w, h = image.size
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    ratio = _closest_aspect_ratio(w / h, target_ratios, w, h, image_size)
    tw, th = image_size * ratio[0], image_size * ratio[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = [
        resized.crop(((i % cols) * image_size, (i // cols) * image_size,
                      (i % cols + 1) * image_size, (i // cols + 1) * image_size))
        for i in range(ratio[0] * ratio[1])
    ]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class InternVLDecision:
    def __init__(self, model_path="OpenGVLab/InternVL3_5-8B", device_map="auto"):
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _pixel_values(self, image):
        import torch

        tiles = _dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
        return torch.stack([self.transform(t) for t in tiles]).to(torch.bfloat16).to(self.model.device)

    def __call__(self, image, modal_mask, category=None):
        question = DECISION_PROMPT.format(category=category or "object")
        response = self.model.chat(
            self.tokenizer,
            self._pixel_values(decision_visual_prompt(image, modal_mask)),
            question,
            dict(max_new_tokens=100, do_sample=False),
        )
        return parse_decision(response)


# ---------------------------------------------------------------------------
# GPT-4o (Geometric and Semantic Guidance Modules)
# ---------------------------------------------------------------------------

def _encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def geometric_messages(image, modal_mask, category):
    h, w = modal_mask.shape
    x1, y1, x2, y2 = mask_to_box(modal_mask)
    text = GEOMETRIC_USER_PROMPT.format(category=category, x1=x1, y1=y1, x2=x2, y2=y2, height=h, width=w)
    return [
        {"role": "system", "content": GEOMETRIC_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": _encode_image(geometric_visual_prompt(image, modal_mask))}},
        ]},
    ]


def semantic_messages(image, modal_mask, occluder_mask, category):
    return [
        {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": SEMANTIC_USER_PROMPT.format(category=category)},
            {"type": "image_url", "image_url": {
                "url": _encode_image(semantic_visual_prompt(image, modal_mask, occluder_mask))}},
        ]},
    ]


def query_gpt(client, messages, model="gpt-4o", temperature=0.2, max_tokens=300):
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content
