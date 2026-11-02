"""Prompts for the three MLLM guidance modules of AmodalCG."""

# Guidance Decision Module (InternVL3.5-8B).
DECISION_PROMPT = """<image>
You are an expert visual annotator.
You are given an image where a red bounding box highlights the {category}.
Follow these instructions carefully and output only a valid JSON object.

Instructions:
1. Determine amodal completion requirement:
   - Output "no" only if you are confident that the target object in the red box is already complete, minimally occluded, or cannot be further extended.
   - Otherwise, output "yes".
   - If you are uncertain, output "yes".

2. Identify the category name:
   - Output the category name of the target subject.

3. Do not include explanations, reasoning, or any text outside of the JSON.

Output format:
{{
  "requires_extensive_completion": "yes" | "no",
  "category": str
}}

Example output:
{{
  "requires_extensive_completion": "yes",
  "category": "Bear"
}}
"""

# Geometric Guidance Module (GPT-4o): extent of the full object.
GEOMETRIC_SYSTEM_PROMPT = (
    "You will be provided with an image of an object, the object's name, its visible bounding box in the "
    "format [x_min, y_min, x_max, y_max], and the size of the image as [height, width]. The bounding box is "
    "marked with a red box in the image. The object in the red box is partially obscured and your task is to "
    "estimate three bounding boxes for the entire object, including both the visible and invisible parts. "
    "Provide tight, moderate, and coarse bounding boxes for the full object in the format "
    "[x_min, y_min, x_max, y_max]. The tight bounding box should include minimal margin around the visible "
    "parts of the object. Just provide the three bounding boxes, without explanation."
)
GEOMETRIC_USER_PROMPT = (
    "The name of the object is {category}, and its visible bounding box is [{x1}, {y1}, {x2}, {y2}]. "
    "The height and width of the image are [{height}, {width}]."
)

# Semantic Guidance Module (GPT-4o): content of the missing region.
SEMANTIC_SYSTEM_PROMPT = (
    "Your job is to speculate the obscured part of the object inside the red region in the image and provide "
    "a Stable Diffusion prompt about the hidden part, using no more than 77 tokens. Do not include the names "
    "of the occluders and descriptions of the background in the prompt, focusing solely on the object. Your "
    "response should only contain the prompt. Your response should begin with a prefix that says 'Prompt:'."
)
SEMANTIC_USER_PROMPT = "A part of the {category} in the red box is masked."
