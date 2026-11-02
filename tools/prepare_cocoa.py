"""
Convert COCOA annotations (COCO_amodal_<split>.json) into per-target modal/amodal masks and a manifest.

    COCOA/
      test2014/                         COCO images
      annotations/COCO_amodal_test2014.json
      modal_mask/ amodal_mask/          <- written by this script
      manifest.jsonl                    <- written by this script
"""
import argparse
import json
import os
import re

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


def _load(field):
    return json.loads(field) if isinstance(field, str) else field


def region_masks(region, h, w):
    polygon = np.asarray(_load(region["segmentation"]), dtype=np.float64).flatten().tolist()
    amodal = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects([polygon], h, w))).astype(bool)
    if "visible_mask" in region:
        modal = mask_utils.decode(_load(region["visible_mask"])).astype(bool)
    elif "invisible_mask" in region:
        modal = amodal & ~mask_utils.decode(_load(region["invisible_mask"])).astype(bool)
    else:
        modal = amodal.copy()
    return modal, amodal


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="COCOA root directory")
    parser.add_argument("--split", default="test2014")
    args = parser.parse_args()

    with open(os.path.join(args.root, "annotations", f"COCO_amodal_{args.split}.json")) as f:
        ann = json.load(f)
    images = {im["id"]: im for im in ann["images"]}
    for sub in ("modal_mask", "amodal_mask"):
        os.makedirs(os.path.join(args.root, sub), exist_ok=True)

    idx = 0
    with open(os.path.join(args.root, "manifest.jsonl"), "w") as manifest:
        for a in tqdm(ann["annotations"]):
            im = images[a["image_id"]]
            for region in a["regions"]:
                idx += 1
                sid = f"{idx:05d}"
                label = re.sub(r"^[^0-9A-Za-z]+", "", region["name"].replace(" ", ""))
                name = f"{sid}_{os.path.splitext(im['file_name'])[0]}_{label}.png"
                modal, amodal = region_masks(region, im["height"], im["width"])
                Image.fromarray(modal.astype(np.uint8) * 255).save(os.path.join(args.root, "modal_mask", name))
                Image.fromarray(amodal.astype(np.uint8) * 255).save(os.path.join(args.root, "amodal_mask", name))
                manifest.write(json.dumps({
                    "id": sid,
                    "image": os.path.join(args.split, im["file_name"]),
                    "modal_mask": os.path.join("modal_mask", name),
                    "amodal_mask": os.path.join("amodal_mask", name),
                    "category": label,
                }) + "\n")


if __name__ == "__main__":
    main()
