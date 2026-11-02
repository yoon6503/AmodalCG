"""
Build a manifest for datasets stored as a folder of per-target modal masks (BSDS-A, MP3D-A, Occluded /
Separated COCO). Masks are sorted by filename; `--image_list` gives the image of each mask, line by line.
Masks named `<id>_<category>.png` also provide the category.

    python tools/make_manifest.py --root data/MP3D --image_dir mp3d_eval --modal_dir modal_mask \
        --amodal_dir gt_mask --image_list MP3D_img_filenames.txt
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Dataset root; all other paths are relative to it")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--modal_dir", required=True)
    parser.add_argument("--amodal_dir", default=None, help="Ground-truth amodal masks (same filenames)")
    parser.add_argument("--image_list", required=True, help="Text file with one image filename per mask")
    parser.add_argument("--output", default="manifest.jsonl")
    args = parser.parse_args()

    masks = sorted(f for f in os.listdir(os.path.join(args.root, args.modal_dir)) if f.endswith(".png"))
    with open(os.path.join(args.root, args.image_list)) as f:
        images = [line.strip() for line in f if line.strip()]
    if len(masks) != len(images):
        raise ValueError(f"{len(masks)} masks but {len(images)} images in {args.image_list}")

    with open(os.path.join(args.root, args.output), "w") as out:
        for mask, image in zip(masks, images):
            stem = os.path.splitext(mask)[0]
            _, _, category = stem.partition("_")
            record = {"id": stem, "image": os.path.join(args.image_dir, image),
                      "modal_mask": os.path.join(args.modal_dir, mask)}
            if args.amodal_dir:
                record["amodal_mask"] = os.path.join(args.amodal_dir, mask)
            if category:
                record["category"] = category
            out.write(json.dumps(record) + "\n")
    print(f"Wrote {len(masks)} targets to {os.path.join(args.root, args.output)}")


if __name__ == "__main__":
    main()
