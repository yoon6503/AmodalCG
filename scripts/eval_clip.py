"""Occluded object recognition: CLIP zero-shot Top-1 / Top-3 accuracy on completed objects (white background)."""
import argparse
import os

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from amodalcg.utils import read_jsonl

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pred_dir", required=True, help="<output_dir>/amodal_white from run_amodalcg.py")
    parser.add_argument("--clip_model", default="ViT-B/32")
    args = parser.parse_args()

    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(args.clip_model, device=device)
    res = model.visual.input_resolution
    preprocess = Compose([
        Resize((res, res), interpolation=Image.LANCZOS),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    with torch.no_grad():
        text = clip.tokenize([f"a photo of a {c}" for c in COCO_CLASSES]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    samples = read_jsonl(args.manifest or os.path.join(args.data_root, "manifest.jsonl"))
    top1 = top3 = n = 0
    for s in tqdm(samples):
        path = os.path.join(args.pred_dir, f"{s['id']}.png")
        if not os.path.exists(path):
            continue
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat /= feat.norm(dim=-1, keepdim=True)
            pred = (100.0 * feat @ text_features.T).softmax(dim=-1)[0].topk(3).indices.tolist()
        gt = COCO_CLASSES.index(s["category"])
        top1 += pred[0] == gt
        top3 += gt in pred
        n += 1

    print(f"Top-1: {100 * top1 / n:.2f}  Top-3: {100 * top3 / n:.2f}  (n={n})")


if __name__ == "__main__":
    main()
