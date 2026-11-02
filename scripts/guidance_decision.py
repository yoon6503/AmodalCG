"""Guidance Decision Module: ask InternVL3.5 whether each target needs MLLM guidance."""
import argparse
import os

from tqdm import tqdm

from amodalcg import AmodalDataset
from amodalcg.guidance import InternVLDecision
from amodalcg.utils import append_jsonl, load_jsonl_by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", required=True, help="Output jsonl (resumes if it exists)")
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-8B")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = load_jsonl_by_id(args.output)
    dataset = AmodalDataset(args.data_root, args.manifest)
    decider = InternVLDecision(args.model)

    for i in tqdm(range(len(dataset))):
        sid = dataset.samples[i]["id"]
        if sid in done:
            continue
        sample = dataset[i]
        if not sample["modal_mask"].any():
            continue
        result = decider(sample["image"], sample["modal_mask"], sample.get("category"))
        append_jsonl(args.output, {"id": sid, **result})


if __name__ == "__main__":
    main()
