"""Occluder estimation: SAM proposes instance masks, InstaOrder decides which of them occlude the target."""
import os
import sys

import cv2
import numpy as np

INSTAORDER_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "InstaOrder")

INSTAORDER_PARAMS = {
    "algo": "InstaOrderNet_od",
    "total_iter": 60000,
    "lr_steps": [32000, 48000],
    "lr_mults": [0.1, 0.1],
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "optim": "SGD",
    "warmup_lr": [],
    "warmup_steps": [],
    "use_rgb": True,
    "backbone_arch": "resnet50_cls",
    "backbone_param": {"in_channels": 5, "num_classes": [2, 3]},
    "overlap_weight": 0.1,
    "distinct_weight": 0.9,
}


def load_sam(checkpoint, model_type="vit_h", device="cuda"):
    from segment_anything import sam_model_registry

    return sam_model_registry[model_type](checkpoint=checkpoint).to(device)


def _import_instaorder():
    if INSTAORDER_ROOT not in sys.path:
        sys.path.insert(0, INSTAORDER_ROOT)
    import models
    from inference import net_forward_occ_depth
    from utils.data_utils import transform_resize

    return models, net_forward_occ_depth, transform_resize


class OccluderEstimator:
    def __init__(self, sam_model, instaorder_ckpt, input_size=384, dilate_iters=3):
        from segment_anything import SamAutomaticMaskGenerator

        models, self._net_forward, self._transform_resize = _import_instaorder()
        self.order_model = models.__dict__["InstaOrderNet_od"](INSTAORDER_PARAMS)
        self.order_model.load_state(instaorder_ckpt)
        self.order_model.switch_to("eval")

        self.mask_generator = SamAutomaticMaskGenerator(
            sam_model,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=25,
        )
        self.input_size = input_size
        self.dilate_iters = dilate_iters

    def find_occluders(self, image, modal_mask, candidates):
        """Return the candidate masks predicted to lie over the target."""
        import torch

        rgb = self._transform_resize(image, self.input_size, self.input_size)
        rgb = torch.from_numpy(rgb).cuda().unsqueeze(0)
        size = (rgb.shape[3], rgb.shape[2])
        query = cv2.resize(modal_mask.astype(np.float64), size, interpolation=cv2.INTER_NEAREST)

        occluders = []
        for cand in candidates:
            cand_resized = cv2.resize(cand.astype(np.float64), size, interpolation=cv2.INTER_NEAREST)
            _, _, cand_over_query = self._net_forward(self.order_model, rgb, query, cand_resized)
            if cand_over_query:
                occluders.append(cand)
        return occluders

    def __call__(self, image, modal_mask):
        """Binary mask of the region covered by the target's occluders (excluding the target itself)."""
        candidates = [m["segmentation"] for m in self.mask_generator.generate(image)]
        occluder_mask = np.zeros(modal_mask.shape, dtype=np.uint8)
        for occ in self.find_occluders(image, modal_mask, candidates):
            occluder_mask |= occ.astype(np.uint8)
        occluder_mask[modal_mask > 0] = 0

        kernel = np.ones((5, 5), np.uint8)
        occluder_mask = cv2.dilate(occluder_mask, kernel, iterations=self.dilate_iters)
        return occluder_mask * (1 - modal_mask)
