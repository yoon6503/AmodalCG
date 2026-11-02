"""Run AmodalCG amodal completion on a dataset."""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from amodalcg import AmodalCG, AmodalDataset, Guidance
from amodalcg.occluders import OccluderEstimator, load_sam
from amodalcg.pipeline import SCALES
from amodalcg.utils import append_jsonl, load_jsonl_by_id, load_mask, save_mask, seed_everything


def build_guidance(sid, category, decisions, boxes, prompts, scales):
    """Use MLLM guidance only for targets the decision module flagged (all targets if no decisions given)."""
    guided = decisions[sid]["requires_guidance"] if sid in decisions else not decisions
    if not guided:
        return Guidance(prompt=category), False
    prompt = prompts[sid]["prompt"] if sid in prompts else category
    box_list = [boxes[sid][s] for s in scales] if sid in boxes else []
    return Guidance(prompt=prompt, boxes=box_list), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decisions", default=None, help="Output of guidance_decision.py")
    parser.add_argument("--boxes", default=None, help="Output of gpt4o_guidance.py --task geometric")
    parser.add_argument("--prompts", default=None, help="Output of gpt4o_guidance.py --task semantic")
    parser.add_argument("--occluder_dir", default=None,
                        help="Cached occluder masks; missing ones are estimated on the fly and saved here")
    parser.add_argument("--scales", default=",".join(SCALES), help="Box scales to try, from smallest to largest")
    parser.add_argument("--sd_model", default="sd2-community/stable-diffusion-2-inpainting")
    parser.add_argument("--sam_ckpt", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--instaorder_ckpt", default="checkpoints/InstaOrder_InstaOrderNet_od.pth.tar")
    parser.add_argument("--background_image", default="assets/gray_wallpaper.jpeg")
    parser.add_argument("--seed", type=int, default=900)
    args = parser.parse_args()

    seed_everything(args.seed)
    scales = args.scales.split(",")
    decisions = load_jsonl_by_id(args.decisions)
    boxes = load_jsonl_by_id(args.boxes)
    prompts = load_jsonl_by_id(args.prompts)

    for sub in ("amodal_completions", "amodal_masks", "amodal_white"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    if args.occluder_dir:
        os.makedirs(args.occluder_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "results.jsonl")
    done = load_jsonl_by_id(log_path)

    sam = load_sam(args.sam_ckpt)
    model = AmodalCG(
        sam,
        occluder_estimator=OccluderEstimator(sam, args.instaorder_ckpt),
        sd_model=args.sd_model,
        background_image=args.background_image,
    )
    dataset = AmodalDataset(args.data_root, args.manifest)

    for i in tqdm(range(len(dataset))):
        sid = dataset.samples[i]["id"]
        if sid in done:
            continue
        sample = dataset[i]
        image, modal_mask = sample["image"], sample["modal_mask"]
        if not modal_mask.any():
            continue

        occluder_mask = None
        occ_path = os.path.join(args.occluder_dir, f"{sid}.png") if args.occluder_dir else None
        if occ_path and os.path.exists(occ_path):
            occluder_mask = load_mask(occ_path)
        elif occ_path:
            occluder_mask = model.occluder_estimator(image, modal_mask)
            save_mask(occluder_mask, occ_path)

        category = sample.get("category") or decisions.get(sid, {}).get("category") or "object"
        guidance, guided = build_guidance(sid, category, decisions, boxes, prompts, scales)
        result = model.complete(image, modal_mask, guidance, occluder_mask)

        name = f"{sid}.png"
        result.image.save(os.path.join(args.output_dir, "amodal_completions", name))
        save_mask(result.amodal_mask, os.path.join(args.output_dir, "amodal_masks", name))
        mask = result.amodal_mask[..., None]
        white = np.array(result.image) * mask + 255 * (1 - mask)
        Image.fromarray(white.astype(np.uint8)).save(os.path.join(args.output_dir, "amodal_white", name))

        append_jsonl(log_path, {
            "id": sid, "category": category, "guided": guided, "prompt": guidance.prompt,
            "box": [int(v) for v in result.box], "num_scales": result.num_scales,
        })
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
