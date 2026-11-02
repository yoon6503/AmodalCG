"""
Amodal segmentation mIoU, overall and per occlusion level.

Visible ratio r = |modal| / |amodal|:  easy r > 0.8,  moderate 0.5 < r <= 0.8,  hard r <= 0.5.
mIoU is computed as sum(intersection) / sum(union) within each group.
"""
import argparse
import os

import numpy as np
from tqdm import tqdm

from amodalcg.utils import load_mask, read_jsonl

LEVELS = ("all", "easy", "moderate", "hard")


def occlusion_level(modal, amodal):
    ratio = modal.sum() / (amodal.sum() + 1e-6)
    return "easy" if ratio > 0.8 else "moderate" if ratio > 0.5 else "hard"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pred_dir", required=True, help="<output_dir>/amodal_masks from run_amodalcg.py")
    parser.add_argument("--split", choices=["all", "occluded", "unoccluded"], default="all",
                        help="Restrict to targets whose modal and amodal masks differ (occluded) or not")
    args = parser.parse_args()

    samples = read_jsonl(args.manifest or os.path.join(args.data_root, "manifest.jsonl"))
    inter = {k: 0 for k in LEVELS}
    union = {k: 0 for k in LEVELS}
    count = {k: 0 for k in LEVELS}
    missing = 0

    for s in tqdm(samples):
        modal = load_mask(os.path.join(args.data_root, s["modal_mask"]))
        amodal = load_mask(os.path.join(args.data_root, s["amodal_mask"]))
        if not modal.any():
            continue
        occluded = not np.array_equal(modal, amodal)
        if (args.split == "occluded" and not occluded) or (args.split == "unoccluded" and occluded):
            continue

        pred_path = os.path.join(args.pred_dir, f"{s['id']}.png")
        if os.path.exists(pred_path):
            pred = load_mask(pred_path)
        else:
            pred, missing = modal, missing + 1

        i, u = int((pred & amodal).sum()), int((pred | amodal).sum())
        for level in ("all", occlusion_level(modal, amodal)):
            inter[level] += i
            union[level] += u
            count[level] += 1

    if missing:
        print(f"Warning: {missing} predictions missing, the modal mask was used instead.")
    for level in LEVELS:
        print(f"{level:>8}: mIoU {100 * inter[level] / max(union[level], 1):.2f}  (n={count[level]})")


if __name__ == "__main__":
    main()
