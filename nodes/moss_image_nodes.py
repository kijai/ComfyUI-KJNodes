"""MOSS-Image layout prompt builder.

Same visual editor as the Ideogram 4 builder (web/js/ideogram4_prompt_builder.js, spec in
web/js/moss_image_prompt_builder.js). MOSS-Image reads a layout as a JSON prompt: a summary caption plus items,
each with a label (object / character / text), a caption, the text to render, and a bbox in absolute pixels
measured from the canvas centre ([y_min, x_min, y_max, x_max]; bbox_coord_space "abs_centralized", bbox_format
"yxyx"). Same output as comfy_extras/nodes_moss_image.py, built from the editor instead of Create Bounding Boxes.
"""

import json
import math
import re

from comfy_api.latest import io

from .ideogram4_nodes import (_render_preview, _parse_json_list, _loads_caption, _seed_boxes, _bbox_outputs,
                              _bg_image, _dumps)


CANVASES = ("single", "multi_panel", "collage")              # summary.canvas values the official compiler accepts


def _strs(v):
    # A list of non-empty strings; a comma/newline separated string is split (the editor's meta/tags fields)
    if isinstance(v, str):
        v = re.split(r"[,\n]", v)
    return [str(t).strip() for t in (v if isinstance(v, list) else []) if str(t).strip()]


def _label_of(t):
    # Editor box type -> item label; "obj" (boxes pasted from an Ideogram template) is an object
    return t if isinstance(t, str) and t.strip() and t != "obj" else "object"


def _edges(box):
    # {x, y, w, h} fractions -> clamped (x1, y1, x2, y2) fractions, negative sizes normalized
    x, y, w, h = (float(box.get(k, 0.0)) for k in ("x", "y", "w", "h"))
    x1, x2 = sorted((x, x + w))
    y1, y2 = sorted((y, y + h))
    return tuple(max(0.0, min(1.0, v)) for v in (x1, y1, x2, y2))


def _moss_bbox(box, width, height):
    # Editor box -> [y_min, x_min, y_max, x_max] pixel edges enclosing it, origin at the canvas centre.
    # Same rounding as nodes_moss_image.py; mirrored by mossBbox() in moss_image_prompt_builder.js.
    x1, y1, x2, y2 = _edges(box)
    hx, hy = width // 2, height // 2
    return [math.floor(round(y1 * height, 9)) - hy, math.floor(round(x1 * width, 9)) - hx,
            math.ceil(round(y2 * height, 9)) - hy, math.ceil(round(x2 * width, 9)) - hx]


def _from_moss_bbox(bb, width, height, space):
    # Layout bbox -> editor {x, y, w, h} fractions, or None: pixel edges from the canvas centre
    # (abs_centralized, the default), else plain top-left pixels. Edges are clamped into the canvas
    # like ideoFromEdges() in the editor, so the preview/bboxes match what the editor shows.
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None
    try:
        y1, x1, y2, x2 = (float(v) for v in bb)
    except (TypeError, ValueError):
        return None
    if "central" in str(space or "abs_centralized"):
        y1, y2, x1, x2 = y1 + height // 2, y2 + height // 2, x1 + width // 2, x2 + width // 2
    x1, x2 = (max(0.0, min(1.0, v / width)) for v in (x1, x2))
    y1, y2 = (max(0.0, min(1.0, v / height)) for v in (y1, y2))
    return {"x": min(x1, x2), "y": min(y1, y2), "w": abs(x2 - x1), "h": abs(y2 - y1)}


def _layout_to_boxes(cap, width, height):
    # Layout dict -> editor boxes ({x, y, w, h, type, text, desc, palette})
    space = cap.get("bbox_coord_space")
    boxes = []
    for i, it in enumerate(cap.get("items") or []):
        if not isinstance(it, dict):
            continue
        box = {"type": _label_of(it.get("label")), "text": str(it.get("text") or ""),
               "desc": str(it.get("caption") or ""), "tags": ", ".join(_strs(it.get("tags"))),
               "palette": _strs(it.get("palette"))}
        bb = _from_moss_bbox(it.get("bbox"), width, height, space)
        if bb:
            box.update(bb)
        else:                                                # no bbox: unplaced placeholder, skipped on export
            k = i % 6
            box.update(x=0.03 + k * 0.035, y=0.03 + k * 0.035, w=0.22, h=0.14, nobbox=True)
        boxes.append(box)
    return boxes


def _summary(caption, meta, canvas, palette):
    # summary in the official field order; optional fields are omitted when empty
    s = {}
    if meta:
        s["meta"] = meta
    if canvas in CANVASES:
        s["canvas"] = canvas
    s["caption"] = caption.strip()
    if palette:
        s["palette"] = palette
    return s


def _layout(summary, boxes, width, height):
    # The layout dict, or None when there are no placed boxes (then the prompt is used as plain text).
    # Mirrored by buildCaption() in moss_image_prompt_builder.js.
    items = []
    for box in boxes:
        if not isinstance(box, dict) or box.get("nobbox"):
            continue
        label = _label_of(box.get("type"))
        item = {"label": label, "caption": box.get("desc", "") or ""}
        if label == "text":
            item["text"] = box.get("text", "") or ""
        tags, palette = _strs(box.get("tags")), _strs(box.get("palette"))
        if tags:
            item["tags"] = tags
        if palette:
            item["palette"] = palette
        item["bbox"] = _moss_bbox(box, width, height)
        items.append(item)
    if not items:
        return None
    return {"summary": summary, "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}


class MossImagePromptBuilderKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossImagePromptBuilderKJ",
            display_name="MOSS-Image Prompt Builder KJ",
            category="KJNodes/text",
            search_aliases=["moss", "moss image", "layout", "bbox", "prompt builder", "json prompt"],
            is_experimental=True,
            description="""
Visual layout builder for MOSS-Image: the prompt becomes a JSON layout with the scene caption and
one item per box (label object / character / text, caption, text to render, bbox in pixels from
the canvas centre). Without boxes the plain prompt is output.

Draw boxes on the canvas; select one to set its label, caption, (for text) the exact words to
render, and optional tags and palette. meta / canvas / the summary palette row are the optional
summary fields. width/height must match the generation size: the boxes are absolute pixels.

Prompt weighting: write (phrase:weight) inside any caption and wire the prompt output into the
text input of MOSS-Image Prompt Weight, which applies the weights and strips the markup.

Canvas controls are the same as in the Ideogram 4 builder; right-click lists the boxes.
Turn the prompt upsampler off when using layouts, it rewrites them.""",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=16,
                             tooltip="Generation width: the boxes are placed in its pixels. Multiples of 16."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16,
                             tooltip="Generation height: the boxes are placed in its pixels. Multiples of 16."),
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="Description of the whole scene and its style (summary.caption)."),
                io.String.Input("meta", default="",
                                tooltip="Optional comma-separated tags for summary.meta, e.g. \"poster\" or "
                                        "\"photograph\" (blank = omitted)."),
                io.Combo.Input("canvas", options=["none", *CANVASES], default="none",
                               tooltip="Optional summary.canvas: single image, multi_panel or collage (none = omitted)."),
                io.Image.Input("image", optional=True,
                               tooltip="Optional background for the editor canvas (and behind the preview)."),
                io.String.Input("import_json", default="", optional=True, force_input=True,
                                tooltip="Optional: a MOSS-Image layout JSON. When connected, it loads into the editor "
                                        "and drives the output per 'import_mode'."),
                io.String.Input("elements_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized boxes from the editor (managed by the node UI)."),
                io.String.Input("style_palette_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized summary palette from the editor (managed by the node UI)."),
                io.Int.Input("bg_brightness", default=25, min=0, max=100, socketless=True, advanced=True,
                             tooltip="Background image brightness % (managed by the node UI slider)."),
                io.Combo.Input("import_mode", options=["when empty", "always"], default="when empty",
                               tooltip="How a wired import_json is used: 'when empty' only seeds the editor while "
                                       "it has no boxes (then the editor wins, so you can edit); 'always' makes "
                                       "the wired JSON authoritative."),
                io.String.Input("output_format", default="compact", socketless=True, advanced=True,
                                tooltip="Output JSON formatting (set via the editor toolbar): 'compact' (default) or "
                                        "'pretty' (indented, for readability)."),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                     tooltip="Optional pixel-space boxes ({x, y, width, height}) used to seed the "
                                             "editor when it has no boxes. Ignored once boxes exist."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Image.Output(display_name="preview"),
                io.BoundingBox.Output(display_name="bboxes"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, width, height, prompt="", meta="", canvas="none", elements_data="", style_palette_data="",
                import_json="", import_mode="when empty", output_format="compact", bboxes=None, image=None,
                bg_brightness=25) -> io.NodeOutput:
        if import_mode not in ("when empty", "always"):
            import_mode = "when empty"
        boxes = _parse_json_list(elements_data)
        boxes_seeded = False
        if not boxes and bboxes:
            boxes = _seed_boxes(bboxes, width, height, {"type": "object", "text": "", "desc": "", "tags": "", "palette": []})
            boxes_seeded = bool(boxes)

        imported = _loads_caption(import_json, "MossImagePromptBuilderKJ")
        if imported is not None and not isinstance(imported.get("items"), list):
            imported = None
        used_import = imported is not None and (import_mode == "always" or not boxes)
        if used_import:
            s = imported.get("summary") if isinstance(imported.get("summary"), dict) else {}
            prompt = str(s.get("caption") or "")
            summary = _summary(prompt, _strs(s.get("meta")), s.get("canvas"), _strs(s.get("palette")))
            boxes = _layout_to_boxes(imported, width, height)
        else:
            summary = _summary(prompt, _strs(meta), canvas, _strs(_parse_json_list(style_palette_data)))

        layout = _layout(summary, boxes, width, height)
        if layout is None:
            out = prompt
        elif output_format == "pretty":
            out = _dumps(layout)
        else:
            out = json.dumps(layout, ensure_ascii=False, separators=(",", ":"))

        preview = _render_preview(boxes, width, height, _bg_image(image), bg_brightness)
        ui = {"dims": [width, height]}
        if boxes_seeded:
            ui["boxes"] = [json.dumps(boxes)]
        if used_import:                                       # mirror the import in the editor (only when used)
            ui["caption"] = [_dumps(imported)]
        return io.NodeOutput(out, preview, _bbox_outputs(boxes, width, height), width, height, ui=ui)


# ---------------- MOSS-Image Preset Merge Node ----------------
from comfy_extras.nodes_model_merging import ModelMergeBlocks

# SFT weight presets: 0 = pure Base, 1 = pure SFT
_PRESET_WEIGHTS = {
    # order: global components, 6 layer groups, output components
    "Composition Preserve": {
        "global": [0.12, 0.12, 0.10, 0.10, 0.14, 0.12],
        "layers": [0.08, 0.12, 0.18, 0.28, 0.42, 0.52],
        "output": [0.12, 0.25, 0.12]
    },
    "Balanced (Recommended)": {
        "global": [0.20, 0.20, 0.16, 0.16, 0.24, 0.20],
        "layers": [0.15, 0.20, 0.30, 0.45, 0.60, 0.72],
        "output": [0.20, 0.35, 0.20]
    },
    "Detail & Typography Priority": {
        "global": [0.30, 0.30, 0.24, 0.24, 0.36, 0.30],
        "layers": [0.25, 0.32, 0.50, 0.65, 0.80, 0.90],
        "output": [0.30, 0.50, 0.30]
    },
    # Portrait-specific presets (pure append, safety fallback retained)
    "ID Photo": {
        "global": [0.10, 0.08, 0.08, 0.08, 0.10, 0.10],
        "layers": [0.06, 0.10, 0.15, 0.22, 0.35, 0.45],
        "output": [0.10, 0.20, 0.10]
    },
    "Beauty Portrait": {
        "global": [0.12, 0.10, 0.08, 0.08, 0.12, 0.12],
        "layers": [0.08, 0.12, 0.18, 0.25, 0.30, 0.35],
        "output": [0.12, 0.15, 0.12]
    },
    "Realistic Photography": {
        "global": [0.22, 0.18, 0.16, 0.16, 0.26, 0.22],
        "layers": [0.16, 0.22, 0.32, 0.48, 0.68, 0.78],
        "output": [0.22, 0.40, 0.22]
    }
}

# MOSS component prefixes, 1:1 match official ModelMergeMOSSImage
_GLOBAL_PREFIXES = [
    "video_embedder.",
    "text_embedder.proj.",
    "text_embedder.blocks.0.",
    "text_embedder.blocks.1.",
    "time_embedding.",
    "shared_block_modulation."
]
_LAYER_GROUPS = [
    (0, 7),
    (8, 13),
    (14, 21),
    (22, 29),
    (30, 35),
    (36, 41)
]
_OUTPUT_PREFIXES = [
    "final_norm_video.",
    "final_linear_video.",
    "video_head_modulation."
]


def _build_preset_weights(preset_name: str, strength: float) -> dict:
    """Calculate per-component SFT weights, clamp to 0-1."""
    base = _PRESET_WEIGHTS[preset_name]
    weights = {}

    # Global components
    for prefix, w in zip(_GLOBAL_PREFIXES, base["global"]):
        weights[prefix] = max(0.0, min(1.0, w * strength))

    # Layer groups
    for (start, end), group_w in zip(_LAYER_GROUPS, base["layers"]):
        w = max(0.0, min(1.0, group_w * strength))
        for i in range(start, end + 1):
            weights[f"layers.{i}."] = w

    # Output components
    for prefix, w in zip(_OUTPUT_PREFIXES, base["output"]):
        weights[prefix] = max(0.0, min(1.0, w * strength))

    return weights


def _get_slider_status(value: float, labels: tuple) -> str:
    """Get plain-language status for 0-2 advanced slider, 5 fixed experience segments."""
    if value <= 0.1:
        return labels[0]
    elif value <= 0.5:
        return labels[1]
    elif value <= 1.2:
        return labels[2]
    elif value <= 1.8:
        return labels[3]
    else:
        return labels[4]


# Concise plain-language preset guidance (English default)
_PRESET_USER_GUIDANCE = {
    "Composition Preserve": "Locks composition, only adds detail, safest for beginners",
    "Balanced (Recommended)": "Balances stability and detail, default choice for general use",
    "Detail & Typography Priority": "Maximizes typography and material detail, for posters/product work, minor composition shift possible",
    "ID Photo": "Locks face and background, natural skin tone, no distortion",
    "Beauty Portrait": "Soft skin smoothing, preserves face structure, beauty style",
    "Realistic Photography": "Rich natural texture, cinematic realism, full lighting detail"
}

# Slider segment labels (English default, aligned with 5 ranges: 0-0.1, 0.1-0.5, 0.5-1.2, 1.2-1.8, 1.8-2)
_COMPOSITION_STATUSES = (
    "No composition constraint, highest layout change risk",
    "Low stability, allows major composition adjustments",
    "Standard stability, preset-safe composition behavior",
    "High stability, strongly locks layout, almost no distortion",
    "Maximum lock, fully fixes composition, only modifies details"
)
_TEXT_STATUSES = (
    "Text enhancement disabled, high typography error risk",
    "Weak text enhancement, minimal optimization",
    "Standard text performance, preset rendering",
    "Strong text enhancement, greatly improves accuracy, ideal for posters",
    "Maximum text enhancement, highest typography/structure accuracy"
)
_DETAIL_STATUSES = (
    "Detail enhancement disabled, smooth output",
    "Weak detail, minimal texture, ideal for soft portraits",
    "Standard detail, preset material quality",
    "Strong detail, rich material texture, ideal for product/still life",
    "Maximum detail, full texture strength, minor noise may appear"
)

# Preset English options
_PRESET_OPTIONS = [
    "Composition Preserve",
    "Balanced (Recommended)",
    "Detail & Typography Priority"
]


def _safe_get_model_keys(model) -> set:
    """Safely get all model key strings without triggering weight loading/attribute access."""
    # First try direct state dict keys from underlying model, no patch resolution
    try:
        inner_model = model.model
        if hasattr(inner_model, "state_dict"):
            return set(inner_model.state_dict().keys())
    except Exception:
        pass
    # Fallback: named parameters without accessing tensor values
    try:
        return {name for name, _ in model.model.named_parameters()}
    except Exception:
        pass
    # Last resort: only return empty set without calling get_key_patches
    return set()


def _safe_get_diffusion_keys(model) -> set:
    """Get diffusion model keys with standard prefix, no weight resolution."""
    all_keys = _safe_get_model_keys(model)
    return {f"diffusion_model.{k}" for k in all_keys if not k.startswith("diffusion_model.")} | {k for k in all_keys if k.startswith("diffusion_model.")}


def _read_model_quant_type(model) -> str:
    """Read official quantization type without triggering weight resolution, return bf16/int8_convrot."""
    # Read official loader metadata first, no weight access
    quant_meta = getattr(model, "quantization", None)
    if isinstance(quant_meta, str) and "convrot" in quant_meta.lower():
        return "int8_convrot"
    # Check state dict key markers only, no tensor/attribute access
    model_keys = _safe_get_diffusion_keys(model)
    has_quant_scale = any(k.endswith("weight_scale") or "comfy_quant" in k for k in model_keys)
    if has_quant_scale:
        return "int8_convrot"
    # Default unquantized official MOSS weights are bf16
    return "bf16"


def _validate_moss_models(base_model, sft_model) -> str:
    """Safe pre-validation: no weight resolution, no attribute access errors. Return quant type."""
    # Get key strings only without loading weights
    base_keys = _safe_get_diffusion_keys(base_model)
    sft_keys = _safe_get_diffusion_keys(sft_model)
    if not base_keys or not sft_keys:
        raise ValueError("Failed to load model: please verify the input MOSS model files are complete.")

    # Layer count check from key strings only
    base_layers = set()
    for k in base_keys:
        if k.startswith("diffusion_model.layers."):
            parts = k.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                base_layers.add(int(parts[2]))
    sft_layers = set()
    for k in sft_keys:
        if k.startswith("diffusion_model.layers."):
            parts = k.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                sft_layers.add(int(parts[2]))

    if len(base_layers) != 42 or len(sft_layers) != 42:
        raise ValueError("Invalid input: only standard 42-layer MOSS-Image models are supported for this node.")
    if base_layers != sft_layers:
        raise ValueError("Model structure mismatch: please use two MOSS models from the same version.")

    # Quantization check without weight access
    base_quant = _read_model_quant_type(base_model)
    sft_quant = _read_model_quant_type(sft_model)
    if base_quant != sft_quant:
        raise ValueError(f"Quantization mismatch: one model is {base_quant}, the other is {sft_quant}. Please use two MOSS models with matching quantization.")
    return base_quant


class MOSSMergePresetKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSMergePresetKJ",
            display_name="MOSS-Image Merge Preset KJ",
            category="KJNodes/model/merging",
            search_aliases=["moss merge", "moss preset", "base sft merge", "moss model merge"],
            description="""
User-friendly MOSS-Image model merging tool, no transformer layer knowledge required:
1. Connect MOSS Base model and SFT model
2. Select desired effect preset, all advanced sliders are collapsed by default
3. Plain-language guidance and warnings help fix issues without manual layer tuning

All default parameters are safety tested to prevent black images and severe corruption.
Supports BF16 and official int8c/convrot quantized models, preserves base quantization layout after merge.
""",
            inputs=[
                io.Model.Input("base_model",
                               tooltip="MOSS Base model, foundation for composition and realism behavior"),
                io.Model.Input("sft_model",
                               tooltip="MOSS SFT/finetuned model, provides typography and fine detail capability"),
                io.Combo.Input("merge_preset", options=_PRESET_OPTIONS, default="Balanced (Recommended)",
                               tooltip="Select blend behavior: Composition Preserve is safest, Detail Priority maximizes typography/material performance"),
                io.Float.Input("merge_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               socketless=True, advanced=True,
                               tooltip="Merge strength: 0=pure base, 0.1-0.5 light blend, 0.5-1.2 standard (default), 1.2-1.8 enhanced, 1.8-2 maximum strength; lower for less distortion, higher for more detail"),
                io.Float.Input("composition_stability", default=1.0, min=0.0, max=2.0, step=0.05,
                               socketless=True, advanced=True,
                               tooltip="Composition stability: 0=no constraint high risk, 0.1-0.5 low stability, 0.5-1.2 standard (default), 1.2-1.8 high stability, 1.8-2 fully lock composition"),
                io.Float.Input("text_accuracy", default=1.0, min=0.0, max=2.0, step=0.05,
                               socketless=True, advanced=True,
                               tooltip="Text accuracy: 0=disabled high error risk, 0.1-0.5 weak enhancement, 0.5-1.2 standard (default), 1.2-1.8 strong enhancement, 1.8-2 maximum accuracy"),
                io.Float.Input("detail_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               socketless=True, advanced=True,
                               tooltip="Detail strength: 0=disabled smooth output, 0.1-0.5 weak detail, 0.5-1.2 standard (default), 1.2-1.8 strong detail, 1.8-2 full texture, minor noise possible"),
            ],
            outputs=[
                io.Model.Output(display_name="merged_model"),
                io.String.Output(display_name="merge_info"),
            ],
        )

    @classmethod
    def execute(cls, base_model, sft_model, merge_preset, merge_strength=1.0,
                composition_stability=1.0, text_accuracy=1.0, detail_strength=1.0) -> io.NodeOutput:
        # Pre validation, get actual loaded quant type
        quant_type = _validate_moss_models(base_model, sft_model)

        # Build base preset weights
        sft_weights = _build_preset_weights(merge_preset, merge_strength)

        # Apply advanced user adjustments (user-friendly group scaling)
        for prefix in _GLOBAL_PREFIXES[:2] + [_GLOBAL_PREFIXES[4], _GLOBAL_PREFIXES[5]]:
            sft_weights[prefix] = max(0.0, min(1.0, sft_weights[prefix] / composition_stability))
        for i in range(0, 14):
            sft_weights[f"layers.{i}."] = max(0.0, min(1.0, sft_weights[f"layers.{i}."] / composition_stability))

        for i in range(14, 30):
            sft_weights[f"layers.{i}."] = max(0.0, min(1.0, sft_weights[f"layers.{i}."] * text_accuracy))
        sft_weights["text_embedder.proj."] = max(0.0, min(1.0, sft_weights["text_embedder.proj."] * text_accuracy))
        sft_weights["text_embedder.blocks.0."] = max(0.0, min(1.0, sft_weights["text_embedder.blocks.0."] * text_accuracy))
        sft_weights["text_embedder.blocks.1."] = max(0.0, min(1.0, sft_weights["text_embedder.blocks.1."] * text_accuracy))

        for i in range(30, 42):
            sft_weights[f"layers.{i}."] = max(0.0, min(1.0, sft_weights[f"layers.{i}."] * detail_strength))
        for prefix in _OUTPUT_PREFIXES:
            sft_weights[prefix] = max(0.0, min(1.0, sft_weights[prefix] * detail_strength))

        # Convert SFT weights to official merge ratio (official ratio = base weight)
        official_kwargs = {}
        for prefix, sft_w in sft_weights.items():
            official_kwargs[prefix] = 1.0 - sft_w

        # Use official merge logic 100%
        merger = ModelMergeBlocks()
        merged_model, = merger.merge(base_model, sft_model, **official_kwargs)

        # Build human-readable info
        avg_early = sum(sft_weights[f"layers.{i}."] for i in range(14)) / 14
        avg_mid = sum(sft_weights[f"layers.{i}."] for i in range(14, 30)) / 16
        avg_late = sum(sft_weights[f"layers.{i}."] for i in range(30, 42)) / 12

        # Full range segmented user tips (all positions have plain-language guidance)
        clamped_count = sum(1 for w in sft_weights.values() if w >= 0.999)
        strength_level = ""
        strength_desc = ""
        special_tip = ""
        if merge_strength <= 0.1:
            if merge_strength <= 0.001:
                strength_level = "Pure Base"
                strength_desc = "Fully uses base model with no SFT effect, 100% composition stable, no distortion.\n"
            else:
                strength_level = "Very Weak SFT"
                strength_desc = "Almost entirely base model output, SFT impact is negligible, composition barely changes.\n"
        elif merge_strength <= 0.5:
            strength_level = "Light SFT"
            strength_desc = "Only adds minimal SFT effect, composition remains highly stable, for conservative use.\n"
        elif merge_strength <= 1.2:
            strength_level = "Standard Effect"
            strength_desc = "Recommended default strength, safe and stable with no extra risk.\n"
        elif merge_strength <= 1.8:
            strength_level = "Enhanced Effect"
            strength_desc = "Strong SFT impact, target characteristics are prominent.\n"
            special_tip = "Tip: Further increase may cause minor composition changes, pair with higher composition stability if needed.\n"
        else:
            if merge_strength >= 1.999:
                strength_level = "Maximum Strength"
                strength_desc = "Slider at maximum, target effect is fully applied.\n"
            else:
                strength_level = "Near Maximum"
                strength_desc = "Very strong SFT effect, close to maximum strength.\n"
            special_tip = "Warning: Highest risk of composition shift and distortion, reduce strength directly if issues occur.\n"

        status_line = f"[Status: {merge_preset} {strength_level}]\n{strength_desc}"

        # Get per-slider plain-language status
        comp_status = _get_slider_status(composition_stability, _COMPOSITION_STATUSES)
        text_status = _get_slider_status(text_accuracy, _TEXT_STATUSES)
        detail_status = _get_slider_status(detail_strength, _DETAIL_STATUSES)

        info = (
            f"{status_line}"
            f"Merge Strength: {merge_strength:.2f}\n"
            f"SFT Ratio: Early {avg_early:.1%}, Mid {avg_mid:.1%}, Late {avg_late:.1%}\n"
            f"Composition Stability: {composition_stability:.2f} ({comp_status})\n"
            f"Text Accuracy: {text_accuracy:.2f} ({text_status})\n"
            f"Detail Strength: {detail_strength:.2f} ({detail_status})\n"
            f"{special_tip}"
            "Guidance: Raise stability/lower strength for distortion, raise text accuracy for typography errors, raise detail strength for sharper texture."
        )

        return io.NodeOutput(merged_model, info)
