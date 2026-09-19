"""Local-shot prompt nodes for ComfyUI_Marquee_Director.

These nodes let you build a Ref2VA-compatible shot board from individual ComfyUI
IMAGE connections instead of embedding absolute image paths inside one giant JSON.
Each H3 Shot Prompt node receives a prompt and up to nine reference images;
H3 Shots Board merges them into the same board JSON that H3 Script Batch Render
already consumes.
"""

import hashlib
import json
import os

import folder_paths
import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

from .common import log


def _image_hash(tensor):
    """Content-addressed filename for a cached reference image."""
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha1()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    step = max(1, t.shape[0] // 8) if t.ndim > 0 and t.shape[0] > 8 else 1
    h.update(t[::step].numpy().tobytes())
    return h.hexdigest()[:16]


def _ensure_image_cache_dir():
    """Persistent cache under output/h3_continuous/_shot_refs/."""
    root = os.path.join(folder_paths.get_output_directory(), "h3_continuous", "_shot_refs")
    os.makedirs(root, exist_ok=True)
    return root


def _save_image_tensor(tensor, label="ref"):
    """Save a ComfyUI IMAGE tensor (B,H,W,C) in [0,1] to PNG and return its path.

    Batches larger than one are saved as separate files; the common case is one
    image per slot, so a single path is returned for B==1.
    """
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor, got %s" % (label, type(tensor).__name__))
    if tensor.ndim != 4:
        raise ValueError("%s must be a ComfyUI IMAGE tensor of shape (B,H,W,C), got %s"
                         % (label, tuple(tensor.shape)))
    cache = _ensure_image_cache_dir()
    paths = []
    for i in range(int(tensor.shape[0])):
        img = tensor[i].cpu().numpy()
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        # RGBA if the tensor has 4 channels, otherwise RGB.
        mode = "RGBA" if img.shape[-1] == 4 else "RGB"
        pil = Image.fromarray(img, mode=mode)
        name = "%s_%s.png" % (_image_hash(tensor[i:i + 1]), label.replace(" ", "_"))
        path = os.path.join(cache, name)
        if not os.path.exists(path):
            pil.save(path, "PNG")
        paths.append(path)
    return paths[0] if len(paths) == 1 else paths


def _ordered_autogrow(values):
    """Autogrow dict values -> ordered list by trailing ordinal."""
    out = []
    for name, value in (values or {}).items():
        if value is None:
            continue
        tail = name.rsplit("_", 1)[-1]
        try:
            ordinal = int(tail)
        except ValueError:
            ordinal = 0
        out.append((ordinal, value))
    out.sort(key=lambda kv: kv[0])
    return [v for _, v in out]


def _shot_json_from_inputs(prompt, seconds, shot_id, is_first_shot, images=None):
    """Build one shot entry JSON with saved reference-image paths."""
    image_paths = []
    for tensor in _ordered_autogrow(images):
        path = _save_image_tensor(tensor)
        if path:
            image_paths.append(path)
    shot = {
        "id": int(shot_id),
        "shot": str(prompt or "").strip(),
        "duration": float(seconds),
        "is_first_shots": bool(is_first_shot),
        "image_paths": image_paths,
    }
    return json.dumps(shot, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
CATEGORY = "ComfyUI_Marquee_Director"

SHOT_PROMPT_TOOLTIP = (
    "Build one shot entry for H3 Shots Board. Reference images wired here become "
    "<Picture 1..9> in the same order as the sockets, exactly like H3 Render Segment. "
    "The node persists the images under output/h3_continuous/_shot_refs/ so the "
    "downstream board JSON contains valid paths."
)

BOARD_TOOLTIP = (
    "Merge individual H3 Shot Prompt outputs into one board JSON. The result has the "
    "same schema that H3 Script Batch Render expects, but the reference images come "
    "from ComfyUI connections instead of hard-coded disk paths."
)


class H3ShotPromptNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ShotPrompt",
            display_name="H3 Shot Prompt",
            category=CATEGORY,
            description="One local shot: prompt + reference images + duration. Wire its "
                        "shot_json output into H3 Shots Board. Reference images are saved "
                        "to a persistent cache and their paths travel with the JSON.",
            inputs=[
                io.Int.Input("shot_id", default=1, min=1, max=9999,
                             tooltip="Shot number, used only for your own tracking."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False,
                                tooltip=SHOT_PROMPT_TOOLTIP),
                io.Float.Input("seconds", default=8.0, min=0.25, max=15.0, step=0.25,
                               round=False,
                               tooltip="Length of this shot, snapped up to H3's frame grid."),
                io.Boolean.Input("is_first_shot", default=False,
                                 tooltip="Marks the first shot of the take. The first shot "
                                         "has no replayed opening to continue from."),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    tooltip="Reference images -> <Picture 1..9> in socket order.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", optional=True),
                        prefix="ref_image_", min=0, max=9)),
            ],
            outputs=[io.String.Output(display_name="shot_json")],
        )

    @classmethod
    def execute(cls, shot_id, prompt, seconds, is_first_shot, ref_images=None) -> io.NodeOutput:
        if not str(prompt or "").strip():
            raise ValueError("H3 Shot Prompt %s: prompt is empty." % shot_id)
        if float(seconds) <= 0:
            raise ValueError("H3 Shot Prompt %s: seconds must be > 0." % shot_id)
        return io.NodeOutput(_shot_json_from_inputs(
            prompt, seconds, shot_id, is_first_shot, ref_images))


class H3ShotsBoardNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ShotsBoard",
            display_name="H3 Shots Board",
            category=CATEGORY,
            description="Assemble multiple H3 Shot Prompt outputs into one board JSON "
                        "for H3 Script Batch Render.",
            inputs=[
                io.String.Input("script_name", default="my_shots", multiline=False,
                                tooltip="Session base name. H3 Script Batch Render still "
                                        "uses its own session_name to decide the output folder."),
                io.String.Input("global_prompt", multiline=True, dynamic_prompts=False,
                                optional=True,
                                tooltip="Appended to every shot prompt unless the prompt "
                                        "already contains overall_soundscape:."),
                io.Int.Input("width", default=0, min=0, max=16384,
                             tooltip="Canvas width. 0 = let H3 Chain Settings decide."),
                io.Int.Input("height", default=0, min=0, max=16384,
                             tooltip="Canvas height. 0 = let H3 Chain Settings decide."),
                io.Autogrow.Input(
                    "shot_jsons", optional=False,
                    tooltip="Connect every H3 Shot Prompt output here, in shot order.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.String.Input("shot_json", optional=True, force_input=True),
                        prefix="shot_json_", min=1, max=99)),
            ],
            outputs=[io.String.Output(display_name="shots_json")],
        )

    @classmethod
    def execute(cls, script_name, global_prompt, width, height, shot_jsons) -> io.NodeOutput:
        raw = _ordered_autogrow(shot_jsons)
        shots = []
        for text in raw:
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                shot = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid shot_json: %s" % exc) from exc
            if not isinstance(shot, dict):
                raise ValueError("Each shot_json must be a JSON object.")
            shots.append(shot)
        if not shots:
            raise ValueError("No valid shot_json inputs connected.")

        global_prompt = str(global_prompt or "").strip()
        board = {
            "script_name": str(script_name or "my_shots").strip(),
            "global": {"global_prompt": global_prompt},
            "shots_info": shots,
        }
        if int(width or 0) >= 32:
            board["width"] = int(width)
        if int(height or 0) >= 32:
            board["height"] = int(height)
        log("assembled H3 Shots Board with %d shot(s)", len(shots))
        return io.NodeOutput(json.dumps(board, ensure_ascii=False, indent=2))


def register_with_extension(ext):
    """Return node classes for MarqueeDirectorExtension.get_node_list()."""
    return [H3ShotPromptNode, H3ShotsBoardNode]
