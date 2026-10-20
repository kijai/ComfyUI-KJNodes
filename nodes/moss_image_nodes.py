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
