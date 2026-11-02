"""Estimate occluder masks (SAM + InstaOrder) for every target and cache them as PNGs."""
import argparse
import os

from tqdm import tqdm

from amodalcg import AmodalDataset
from amodalcg.occluders import OccluderEstimator, load_sam
from amodalcg.utils import save_mask, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sam_ckpt", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--instaorder_ckpt", default="checkpoints/InstaOrder_InstaOrderNet_od.pth.tar")
    parser.add_argument("--seed", type=int, default=900)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = AmodalDataset(args.data_root, args.manifest)
    estimator = OccluderEstimator(load_sam(args.sam_ckpt), args.instaorder_ckpt)

    for i in tqdm(range(len(dataset))):
        out_path = os.path.join(args.output_dir, f"{dataset.samples[i]['id']}.png")
        if os.path.exists(out_path):
            continue
        sample = dataset[i]
        if not sample["modal_mask"].any():
            continue
        save_mask(estimator(sample["image"], sample["modal_mask"]), out_path)


if __name__ == "__main__":
    main()
