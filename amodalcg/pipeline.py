"""AmodalCG: MLLM-guided amodal completion with multi-scale box refinement."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from .utils import add_margin, box_to_mask, enclose_box, mask_to_box, mask_touches_box

SCALES = ("tight", "moderate", "coarse")


@dataclass
class Guidance:
    """MLLM guidance for one target. `boxes` is empty when guidance is skipped."""
    prompt: str
    boxes: List[List[int]] = field(default_factory=list)


@dataclass
class CompletionResult:
    image: Image.Image          # inpainted image (full resolution)
    amodal_mask: np.ndarray     # predicted amodal mask {0, 1}
    box: List[int]              # box used for the final completion
    num_scales: int             # number of scales tried


class AmodalCG:
    def __init__(
        self,
        sam_model,
        occluder_estimator=None,
        sd_model="sd2-community/stable-diffusion-2-inpainting",
        background_image="assets/gray_wallpaper.jpeg",
        min_background_pixels=300,
        device="cuda",
    ):
        from diffusers import StableDiffusionInpaintPipeline
        from segment_anything import SamPredictor

        self.inpainter = StableDiffusionInpaintPipeline.from_pretrained(sd_model, torch_dtype=torch.float16)
        self.inpainter.enable_attention_slicing()
        self.inpainter = self.inpainter.to(device)
        self.inpainter.set_progress_bar_config(disable=True)

        self.sam = SamPredictor(sam_model)
        self.occluder_estimator = occluder_estimator
        self.background = np.array(Image.open(background_image).convert("RGB"))
        self.min_background_pixels = min_background_pixels

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _background_like(self, h, w):
        bg = self.background
        if bg.shape[0] < h or bg.shape[1] < w:
            scale = max(h / bg.shape[0], w / bg.shape[1])
            bg = np.array(Image.fromarray(bg).resize((int(np.ceil(bg.shape[1] * scale)), int(np.ceil(bg.shape[0] * scale)))))
        return bg[:h, :w]

    def inpaint(self, image, modal_mask, inpaint_mask, prompt):
        """Place the visible target on a clean background and inpaint `inpaint_mask`."""
        h, w = modal_mask.shape
        canvas = np.where(modal_mask[..., None] > 0, image, self._background_like(h, w)).astype(np.uint8)
        out = self.inpainter(
            prompt=prompt,
            image=Image.fromarray(canvas),
            mask_image=Image.fromarray(inpaint_mask.astype(np.uint8) * 255),
        ).images[0]
        return out.resize((w, h))

    def segment_amodal(self, completion, modal_mask, inpaint_mask):
        """
        Segment the background of the inpainted image with SAM and invert it.
        Returns the amodal mask, or None if there is too little background to prompt SAM.
        """
        visible_bg = (inpaint_mask == 0) & (modal_mask == 0)
        fg_ys, fg_xs = np.where(modal_mask == 1)
        bg_ys, bg_xs = np.where(visible_bg)
        if len(bg_ys) <= self.min_background_pixels:
            return None

        fg = np.random.choice(len(fg_ys))
        bg = np.random.choice(len(bg_ys))
        # Negative click on the target, positive click on the background.
        points = np.array([[fg_xs[fg], fg_ys[fg]], [bg_xs[bg], bg_ys[bg]]])
        labels = np.array([0, 1])

        self.sam.set_image(np.array(completion))
        masks, scores, _ = self.sam.predict(point_coords=points, point_labels=labels, multimask_output=True)
        background = masks[int(np.argmax(scores))].astype(np.uint8)

        # Anything outside the inpainted region and the visible target is background.
        background = background | visible_bg.astype(np.uint8)
        return 1 - background

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def complete(self, image, modal_mask, guidance: Guidance, occluder_mask: Optional[np.ndarray] = None):
        h, w = modal_mask.shape
        modal_box = mask_to_box(modal_mask)

        if occluder_mask is None:
            if self.occluder_estimator is None:
                raise ValueError("Either pass `occluder_mask` or build AmodalCG with an `occluder_estimator`.")
            occluder_mask = self.occluder_estimator(image, modal_mask)
        occluder_mask = occluder_mask * (1 - modal_mask)

        # Without guidance, enlarge the visible box by a fixed 10% margin.
        boxes = guidance.boxes or [add_margin(modal_box, 0.1, h, w)]

        for num_scales, box in enumerate(boxes, start=1):
            box = enclose_box(box, modal_box, h, w)
            inpaint_mask = (occluder_mask * box_to_mask(box, h, w) > 0).astype(np.uint8)

            completion = self.inpaint(image, modal_mask, inpaint_mask, guidance.prompt)
            completed = self.segment_amodal(completion, modal_mask, inpaint_mask)
            if completed is None:
                completed = np.zeros_like(modal_mask)

            # A completion that reaches the box boundary is likely truncated: try the next scale.
            if not completed.any() or not mask_touches_box(completed, box):
                break

        amodal_mask = (completed | modal_mask).astype(np.uint8)
        return CompletionResult(image=completion, amodal_mask=amodal_mask, box=box, num_scales=num_scales)
