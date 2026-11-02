"""
Geometric (`--task geometric`) and Semantic (`--task semantic`) Guidance Modules with GPT-4o.

  geometric -> {"id", "tight", "moderate", "coarse", "raw"}   (boxes of the full object)
  semantic  -> {"id", "prompt", "raw"}                         (Stable Diffusion prompt)

Only targets flagged by the Guidance Decision Module are queried unless `--decisions` is omitted.
Requires the OPENAI_API_KEY environment variable.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from amodalcg import AmodalDataset
from amodalcg.guidance import geometric_messages, parse_boxes, parse_sd_prompt, query_gpt, semantic_messages
from amodalcg.utils import append_jsonl, load_jsonl_by_id, load_mask


def build_messages(task, sample, occluder_dir, category):
    if task == "geometric":
        return geometric_messages(sample["image"], sample["modal_mask"], category)
    occluder_mask = load_mask(os.path.join(occluder_dir, f"{sample['id']}.png"))
    return semantic_messages(sample["image"], sample["modal_mask"], occluder_mask, category)


def parse(task, text):
    return parse_boxes(text) if task == "geometric" else {"prompt": parse_sd_prompt(text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["geometric", "semantic"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--decisions", default=None, help="Output of guidance_decision.py")
    parser.add_argument("--occluder_dir", default=None, help="Output of estimate_occluders.py (semantic task)")
    parser.add_argument("--output", required=True, help="Output jsonl (resumes if it exists)")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()
    if args.task == "semantic" and args.occluder_dir is None:
        parser.error("--occluder_dir is required for the semantic task")

    from openai import OpenAI

    client = OpenAI()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = load_jsonl_by_id(args.output)
    decisions = load_jsonl_by_id(args.decisions)
    dataset = AmodalDataset(args.data_root, args.manifest)

    todo = []
    for i, s in enumerate(dataset.samples):
        if s["id"] in done:
            continue
        if args.decisions is not None and not decisions.get(s["id"], {}).get("requires_guidance", False):
            continue
        todo.append(i)

    def run(i):
        sample = dataset[i]
        category = sample.get("category") or decisions.get(sample["id"], {}).get("category") or "object"
        messages = build_messages(args.task, sample, args.occluder_dir, category)
        for attempt in range(args.max_retries):
            try:
                text = query_gpt(client, messages, model=args.model, temperature=args.temperature)
                return {"id": sample["id"], **parse(args.task, text), "raw": text}
            except Exception as e:  # API errors and unparsable answers
                if attempt == args.max_retries - 1:
                    print(f"[{sample['id']}] failed: {e}")
                    return None
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(args.num_workers) as pool:
        futures = [pool.submit(run, i) for i in todo]
        for future in tqdm(as_completed(futures), total=len(futures)):
            record = future.result()
            if record is not None:
                append_jsonl(args.output, record)


if __name__ == "__main__":
    main()
