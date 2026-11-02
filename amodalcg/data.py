"""
Every dataset is described by a manifest (jsonl), one target object per line:

    {"id": "00002", "image": "test2014/COCO_test2014_000000000027.jpg",
     "modal_mask": "modal_mask/00002_....png", "amodal_mask": "gt_mask/00002_....png", "category": "Cycle"}

Paths are relative to the dataset root. `amodal_mask` is only needed for evaluation and
`category` may be omitted (the Guidance Decision Module then predicts it).
"""
import os

from .utils import load_image, load_mask, read_jsonl


class AmodalDataset:
    def __init__(self, root, manifest=None):
        self.root = root
        self.samples = read_jsonl(manifest or os.path.join(root, "manifest.jsonl"))

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.root, rel)

    def __getitem__(self, i):
        s = dict(self.samples[i])
        s["image"] = load_image(self.path(s["image"]))
        s["modal_mask"] = load_mask(self.path(s["modal_mask"]))
        if s.get("amodal_mask"):
            s["amodal_mask"] = load_mask(self.path(s["amodal_mask"]))
        return s
