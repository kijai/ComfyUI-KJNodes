"""Ming-Image prompt builder.

Same visual editor as the Ideogram 4 builder (web/js/ideogram4_prompt_builder.js, spec in
web/js/ming_image_prompt_builder.js), emitting Ming-Image's Figma-style JSON caption:
canvas_settings + layers listed back (background) to front. Format reference:
https://github.com/inclusionAI/Ming-Image/blob/main/assets/t2i_rewriter_system_prompt.txt
"""

import json
import logging
import math
import re

from comfy_api.latest import io

from .ideogram4_nodes import (_render_preview, _parse_json_list, _loads_caption, _dumps, _palette,
                              _seed_boxes, _bbox_outputs, _bg_image)


# Ratios the auto aspect_ratio snaps to; the aspect/coordinate helpers mirror ming_image_prompt_builder.js.
_RATIOS = [(1, 1), (5, 4), (4, 3), (3, 2), (16, 10), (5, 3), (16, 9), (2, 1), (21, 9), (3, 1)]
_RATIO_TOL = 0.02                                            # |log ratio| ~2%
_ASPECT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)")          # "16:9"
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[×xX*]\s*(\d+(?:\.\d+)?)")         # "2048 × 2048 px"
_COORD_RE = re.compile(r"\b(cx|cy|w|h|width|height)\s*[:=]\s*(-?(?:\d+\.?\d*|\.\d+))", re.I)
_NUM_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)")


def _round_half_up(v, nd):
    return math.floor(v * 10 ** nd + 0.5) / 10 ** nd        # same as JS Math.floor(v * 10**nd + 0.5)


def _ratio(w, h):
    # "16:9": nearest common ratio within ~2%, else the reduced fraction, else decimal.
    r = w / h
    best, bd = None, float("inf")
    for a, b in _RATIOS:
        for p, q in (((a, b),) if a == b else ((a, b), (b, a))):
            d = abs(math.log(r * q / p))
            if d < bd:
                best, bd = (p, q), d
    if bd < _RATIO_TOL:
        return "%d:%d" % best
    g = math.gcd(w, h)
    if max(w, h) // g <= 32:
        return "%d:%d" % (w // g, h // g)
    return "%.2f:1" % _round_half_up(r, 2) if r >= 1 else "1:%.2f" % _round_half_up(1 / r, 2)


def _aspect(width, height, with_size=False):
    s = _ratio(width, height)
    return "%s, %d × %d px" % (s, width, height) if with_size else s


def _parse_aspect(s):
    # "16:9" / "1:1, 2048 × 2048 px" / "1920x1080" -> width/height ratio, or None.
    if not isinstance(s, str):
        return None
    for rx in (_ASPECT_RE, _SIZE_RE):
        m = rx.search(s)
        if m and float(m.group(1)) > 0 and float(m.group(2)) > 0:
            return float(m.group(1)) / float(m.group(2))
    return None


def _dims_for_aspect(r, width, height):
    # Reshape to ratio r at about the same pixel area (multiples of 16); unchanged if already within ~2%.
    if not r or abs(math.log(r * height / width)) < _RATIO_TOL:
        return width, height
    area = width * height
    def r16(v):
        return max(64, min(16384, int(math.floor(v / 16 + 0.5)) * 16))
    return r16(math.sqrt(area * r)), r16(math.sqrt(area / r))


def _coords(box):
    # {x, y, w, h} fractions -> "cx: 0.500, cy: 0.500, w: 1.000, h: 1.000", clipped to the canvas.
    def c(v):
        return max(0.0, min(1.0, v))
    x, y, w, h = box.get("x", 0.0), box.get("y", 0.0), box.get("w", 0.0), box.get("h", 0.0)
    x1, x2 = sorted((c(x), c(x + w)))
    y1, y2 = sorted((c(y), c(y + h)))
    vals = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return "cx: %.3f, cy: %.3f, w: %.3f, h: %.3f" % tuple(_round_half_up(v, 3) for v in vals)


def _parse_coords(c):
    # Layer coordinates ("cx: .., cy: .., w: .., h: ..", or a {cx,cy,w,h} dict / 4-list) -> {x, y, w, h}, or None.
    if isinstance(c, str):
        named = {}
        for k, v in _COORD_RE.findall(c):
            named[{"width": "w", "height": "h"}.get(k.lower(), k.lower())] = v
        vals = [named.get(k) for k in ("cx", "cy", "w", "h")]
        if None in vals:
            vals = _NUM_RE.findall(c)                        # bare "0.5, 0.5, 1, 1"
    elif isinstance(c, dict):
        vals = [c.get(k) for k in ("cx", "cy", "w", "h")]
    elif isinstance(c, (list, tuple)):
        vals = c
    else:
        return None
    try:
        cx, cy, w, h = (float(v) for v in vals)
    except (TypeError, ValueError):
        return None
    return {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h}


def _txt(v):
    return v if isinstance(v, str) else ("" if v is None else json.dumps(v, ensure_ascii=False))


def _layers_to_boxes(cap):
    # Caption dict -> editor boxes ({x, y, w, h, desc, rel, palette}) for the preview/bboxes outputs.
    boxes = []
    for el in cap.get("layers") or []:
        if not isinstance(el, dict):
            continue
        cs = el.get("color_specs")
        box = {"desc": _txt(el.get("description")), "rel": _txt(el.get("hierarchy_and_relation")),
               "palette": [c for c in cs if isinstance(c, str)] if isinstance(cs, list) else []}
        box.update(_parse_coords(el.get("coordinates")) or {"x": 0.03, "y": 0.03, "w": 0.22, "h": 0.14, "nobbox": True})
        boxes.append(box)
    return boxes


class MingImagePromptBuilderKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingImagePromptBuilderKJ",
            display_name="Ming-Image Prompt Builder KJ",
            category="KJNodes/text",
            search_aliases=["ming", "ming-image", "caption", "bbox", "layers", "prompt builder", "json prompt"],
            is_experimental=True,
            description="""
Visual prompt builder for Ming-Image's structured JSON caption (canvas_settings + layers).

Drag on the canvas to draw layers; select a layer to set its description,
hierarchy_and_relation and color_specs. Layers are listed back to front: 01 is the
bottom (usually the full-canvas background), the highest number the topmost overlay.
Outputs the assembled caption JSON string.

coordinates are normalized "cx, cy, w, h" (box center and size, 0-1); width/height set
the canvas aspect, and aspect_ratio follows them unless set to custom. Importing a caption
reshapes width/height to its aspect_ratio (at about the same pixel count).

Canvas controls:
- Drag: draw a new layer (also inside a canvas-filling layer such as the background)
- Ctrl/Cmd-drag: force-draw a new layer even on top of an existing one
- Click: select a layer · Alt-click: cycle overlapping layers
- Double-click: edit the description inline
- Right-click: layer list (top = front; select / delete / duplicate / lock / drag to reorder)
- Del / Backspace: remove the selected layer
- Ctrl/Cmd + C / V / D: copy / paste / duplicate the selected layer
- coordinate fields (px / out) in the panel are editable

Color swatches:
- Click: edit · Drag: reorder · Right-click: remove
- Hover + Ctrl/Cmd + C / V: copy / paste the hex
- "+": add a color (uses the clipboard color if it is one)

Toolbar:
- Live: use the live sampling preview as the background (and grab the final result)
- Grab BG / Clear BG: use the last generated image as the background
- brightness slider, token estimate, and Copy / Paste / Clear all""",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas width (also the pixel grid for the px coordinate field). Multiples of 16."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas height (also the pixel grid for the px coordinate field). Multiples of 16."),
                io.DynamicCombo.Input("aspect_ratio", options=[
                    io.DynamicCombo.Option("auto", []),
                    io.DynamicCombo.Option("auto_with_size", []),
                    io.DynamicCombo.Option("custom", [
                        io.String.Input("custom", default=""),
                    ]),
                ], tooltip="canvas_settings.aspect_ratio: 'auto' derives it from width/height (e.g. '16:9'), "
                           "'auto_with_size' adds the pixel size (e.g. '16:9, 1360 × 768 px'), 'custom' is verbatim."),
                io.String.Input("ambient_lighting", multiline=True, default="",
                                tooltip="canvas_settings.ambient_lighting: global lighting, mood and exposure."),
                io.String.Input("image_style", multiline=True, default="",
                                tooltip="canvas_settings.image_style: medium, rendering style, camera/lens, typography treatment."),
                io.Image.Input("image", optional=True,
                               tooltip="Optional reference image shown as the editor background (and behind the preview)."),
                io.String.Input("import_json", default="", optional=True, force_input=True,
                                tooltip="Optional: a full caption JSON. When connected, it loads into the editor "
                                        "and drives the output per 'import_mode'."),
                io.String.Input("elements_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized layers from the editor (managed by the node UI)."),
                io.Int.Input("bg_brightness", default=25, min=0, max=100, socketless=True, advanced=True,
                             tooltip="Background image brightness % (managed by the node UI slider)."),
                io.Combo.Input("import_mode", options=["when empty", "always"], default="when empty",
                               tooltip="How a wired import_json is used: 'when empty' only seeds the editor while "
                                       "it has no layers (then the editor wins, so you can edit); 'always' makes "
                                       "the wired JSON authoritative so its changes always propagate to the output."),
                io.String.Input("output_format", default="pretty", socketless=True, advanced=True,
                                tooltip="Output JSON formatting (set via the editor toolbar): 'pretty' (default, indented "
                                        "like Ming-Image's example prompts) or 'compact' (single line)."),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                     tooltip="Optional pixel-space boxes ({x, y, width, height}) used to seed the "
                                             "editor's layers when it has none. Ignored once layers exist."),
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
    def execute(cls, width, height, aspect_ratio, ambient_lighting="", image_style="",
                elements_data="", import_json="", import_mode="when empty", output_format="pretty",
                bboxes=None, image=None, bg_brightness=25) -> io.NodeOutput:
        if import_mode not in ("when empty", "always"):
            import_mode = "when empty"
        if output_format == "compact":
            dump = lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        else:
            dump = lambda v: _dumps(v, ind="  ")
        boxes = _parse_json_list(elements_data)
        boxes_seeded = False
        if not boxes and bboxes:
            boxes = _seed_boxes(bboxes, width, height, {"desc": "", "rel": "", "palette": []})
            boxes_seeded = bool(boxes)

        imported = _loads_caption(import_json, "MingImagePromptBuilderKJ")
        if imported is not None and not isinstance(imported.get("layers"), list):
            logging.warning("[MingImagePromptBuilderKJ] import_json has no 'layers' list; ignored")
            imported = None
        # Same import_mode semantics as the Ideogram 4 builder: "always" -> authoritative;
        # "when empty" -> only while the editor has no layers.
        used_import = imported is not None and (import_mode == "always" or not boxes)

        if used_import:
            caption = imported
            boxes = _layers_to_boxes(imported)
            if image is None:                                # follow its aspect (a connected image governs it otherwise)
                cs = imported.get("canvas_settings")
                r = _parse_aspect(cs.get("aspect_ratio") if isinstance(cs, dict) else None)
                width, height = _dims_for_aspect(r, width, height)
        else:
            mode = aspect_ratio.get("aspect_ratio", "auto") if isinstance(aspect_ratio, dict) else aspect_ratio
            if mode == "custom":
                ar = aspect_ratio.get("custom", "") if isinstance(aspect_ratio, dict) else ""
            else:
                ar = _aspect(width, height, mode == "auto_with_size")
            layers = []
            for box in boxes:                                # back to front, as Ming expects
                if not isinstance(box, dict):
                    continue
                layer = {"description": box.get("desc", "") or ""}   # key order matters
                if not box.get("nobbox"):                   # unplaced layers omit coordinates
                    layer["coordinates"] = _coords(box)
                layer["hierarchy_and_relation"] = box.get("rel", "") or ""
                layer["color_specs"] = _palette(box.get("palette", []))
                layers.append(layer)
            caption = {
                "canvas_settings": {"aspect_ratio": ar, "ambient_lighting": ambient_lighting, "image_style": image_style},
                "layers": layers,
            }

        preview = _render_preview(boxes, width, height, _bg_image(image), bg_brightness)
        ui = {"dims": [width, height]}
        if boxes_seeded:
            ui["boxes"] = [json.dumps(boxes)]
        if used_import:                                       # mirror the import in the editor (only when used)
            ui["caption"] = [_dumps(imported)]
        return io.NodeOutput(dump(caption), preview, _bbox_outputs(boxes, width, height), width, height, ui=ui)
