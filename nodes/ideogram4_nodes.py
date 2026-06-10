"""Ideogram 4 prompt builder V2 — freehand edition.

A single self-contained node with a visual editor: sketch shapes freehand,
bucket-fill closed areas like in Paint, and each drawn shape is automatically
outlined into its own bbox region. Set each region's type/desc/text/color
palette, and assemble the Ideogram 4 JSON caption prompt.

Identical inputs/outputs to "Ideogram 4 Prompt Builder KJ" (drop-in
compatible), plus the freehand ink layer and a background-image opacity
control managed by the node UI.
"""

import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

from comfy_api.latest import io


_FONT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", "FreeMono.ttf")


def _hex_rgb(h):
    h = (h or "").lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) if len(h) == 6 else (255, 255, 255)


def _readable(rgb):
    # Lighten toward white if too dark, so box-colored text stays legible on the dark canvas.
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < 130:
        t = (130 - lum) / (255 - lum)
        r, g, b = round(r + (255 - r) * t), round(g + (255 - g) * t), round(b + (255 - b) * t)
    return (r, g, b)


def _font(size):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        try:
            return ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    # Greedy word-wrap; words wider than the box are hard-broken so text never
    # escapes the box horizontally.
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            while len(word) > 1 and draw.textlength(word, font=font) > max_w:
                k = len(word)
                while k > 1 and draw.textlength(word[:k], font=font) > max_w:
                    k -= 1
                if line:
                    lines.append(line)
                    line = ""
                lines.append(word[:k])
                word = word[k:]
            test = word if not line else line + " " + word
            if line and draw.textlength(test, font=font) > max_w:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines


def _box_bg_lum(box, rgb, ink_arr, rw, rh):
    # Approximate luminance behind a box: its painted fill, or the ink under it.
    if box.get("fill"):
        return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) * 0.5 + 13
    if ink_arr is None:
        return 26.0
    total, n = 0.0, 0
    for iy in (1, 2, 3):
        for ix in (1, 2, 3):
            px = min(rw - 1, max(0, round((box.get("x", 0) + box.get("w", 0) * ix / 4) * rw)))
            py = min(rh - 1, max(0, round((box.get("y", 0) + box.get("h", 0) * iy / 4) * rh)))
            r2, g2, b2, a2 = (int(v) for v in ink_arr[py, px])
            a2 = a2 / 255.0
            total += (0.299 * r2 + 0.587 * g2 + 0.114 * b2) * a2 + 26 * (1 - a2)
            n += 1
    return total / n if n else 26.0


def _text_fill(rgb, bg_lum):
    # Dark shade of the color on a bright background, light shade on a dark one.
    if bg_lum > 110:
        l = max(1.0, 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        f = min(1.0, 50.0 / l)
        return (round(rgb[0] * f), round(rgb[1] * f), round(rgb[2] * f))
    return _readable(rgb)


def _flood_component(mask, sy, sx, conn8):
    """Boolean mask of the connected component of `mask` containing (sy, sx)."""
    if not mask[sy, sx]:
        return np.zeros_like(mask)
    try:
        from scipy import ndimage
        structure = np.ones((3, 3), bool) if conn8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        lbl, _ = ndimage.label(mask, structure=structure)
        return lbl == lbl[sy, sx]
    except Exception:
        from collections import deque                   # pure-python fallback (no scipy)
        h, w = mask.shape
        out = np.zeros_like(mask)
        out[sy, sx] = True
        dq = deque([(sy, sx)])
        neigh = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) if conn8 \
            else ((-1, 0), (1, 0), (0, -1), (0, 1))
        while dq:
            y, x = dq.popleft()
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                    out[ny, nx] = True
                    dq.append((ny, nx))
        return out


def _apply_fill(ink, sx, sy, rgb, rect=None):
    if rect is not None:
        # box-bounded fill: the box rect is the wall; flood the empty area inside
        # (strokes block, even unclosed), then the whole layer takes the color
        arr = np.array(ink, dtype=np.uint8)
        a = arr[..., 3]
        h, w = a.shape
        try:
            rx, ry, rw2, rh2 = (float(v) for v in rect)
        except Exception:
            return
        x1 = max(0, round(rx * w)); y1 = max(0, round(ry * h))
        x2 = min(w - 1, round((rx + rw2) * w)); y2 = min(h - 1, round((ry + rh2) * h))
        if int(a[sy, sx]) <= 24 and x1 <= sx <= x2 and y1 <= sy <= y2:
            cls = a <= 24
            bound = np.zeros_like(cls)
            bound[y1:y2 + 1, x1:x2 + 1] = True
            region = _flood_component(cls & bound, sy, sx, conn8=False)
            arr[..., :3][region] = rgb
            arr[..., 3][region] = 128
        m = arr[..., 3] > 24
        arr[..., :3][m] = rgb
        ink.paste(Image.fromarray(arr))
        return
    return _apply_fill_legacy(ink, sx, sy, rgb)


def _apply_fill_legacy(ink, sx, sy, rgb):
    """Paint-style bucket fill, mirroring the editor — one color per shape:
    seed on empty -> flood (alpha 128) and, if enclosed, recolor the surrounding
    drawing; seed on a previous fill -> refill it (and its outline); seed on a
    stroke -> recolor the whole connected drawing."""
    arr = np.array(ink, dtype=np.uint8)
    a = arr[..., 3].copy()
    a0 = int(a[sy, sx])
    INKA = 24
    if a0 >= 200:                                       # recolor the connected drawing
        m = _flood_component(a > INKA, sy, sx, conn8=True)
        arr[..., :3][m] = rgb
    else:
        cls = (a <= INKA) if a0 <= INKA else ((a > INKA) & (a < 200))
        region = _flood_component(cls, sy, sx, conn8=False)
        arr[..., :3][region] = rgb
        if a0 <= INKA:
            arr[..., 3][region] = 128
        touches = bool(region[0, :].any() or region[-1, :].any()
                       or region[:, 0].any() or region[:, -1].any())
        if not touches:                                 # enclosed: outline gets the same color
            ink_mask = (a > INKA) & ~region
            grown = np.zeros_like(region)
            grown[:-1, :] |= region[1:, :]
            grown[1:, :] |= region[:-1, :]
            grown[:, :-1] |= region[:, 1:]
            grown[:, 1:] |= region[:, :-1]
            grown[:-1, :-1] |= region[1:, 1:]
            grown[:-1, 1:] |= region[1:, :-1]
            grown[1:, :-1] |= region[:-1, 1:]
            grown[1:, 1:] |= region[:-1, :-1]
            seeds = np.argwhere(grown & ink_mask)
            if len(seeds):
                try:
                    from scipy import ndimage
                    lbl, _ = ndimage.label(ink_mask, structure=np.ones((3, 3), bool))
                    ids = np.unique(lbl[seeds[:, 0], seeds[:, 1]])
                    m = np.isin(lbl, ids[ids > 0])
                except Exception:
                    m = np.zeros_like(ink_mask)
                    for y, x in seeds:
                        if not m[y, x]:
                            m |= _flood_component(ink_mask, int(y), int(x), conn8=True)
                arr[..., :3][m] = rgb
    ink.paste(Image.fromarray(arr))


def _px_pts(pts, rw, rh):
    return [(min(rw - 1, max(0, x * rw)), min(rh - 1, max(0, y * rh))) for x, y in pts]


def _stroke_on(target, pts, width, color):
    if not pts:
        return
    r = max(1, width) / 2.0
    if len(pts) == 1:
        x, y = pts[0]
        target.ellipse([x - r, y - r, x + r, y + r], fill=color)
        return
    target.line(pts, fill=color, width=max(1, width), joint="curve")
    for x, y in (pts[0], pts[-1]):                      # round caps
        target.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _transform_layer(img, a, b, rw, rh):
    try:
        ax, ay, aw, ah = (float(v) for v in a)
        bx, by, bw, bh = (float(v) for v in b)
    except Exception:
        return img
    if min(aw, ah, bw, bh) <= 0:
        return img
    fx, fy = bw / aw, bh / ah
    return img.transform((rw, rh), Image.AFFINE,
                         (1 / fx, 0, ax * rw - (bx * rw) / fx,
                          0, 1 / fy, ay * rh - (by * rh) / fy),
                         resample=Image.BILINEAR)


def _render_layer(ops, L, rw, rh, limit=None):
    """Render ONE layer's raster: its own strokes/fills/erases (fills see only
    this layer's ink), global erases (L=0), its move/scale transforms, and its
    birth as a copy of another layer."""
    layer = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    long_edge = max(rw, rh)
    n = len(ops) if limit is None else limit
    for i in range(n):
        op = ops[i]
        if not isinstance(op, dict):
            continue
        t = op.get("t")
        oL = op.get("L") or 0
        scoped = oL == L
        try:
            if t == "s" and scoped:
                pts = _px_pts(op.get("pts") or [], rw, rh)
                w = max(1, round(float(op.get("s", 0.01)) * long_edge))
                _stroke_on(ImageDraw.Draw(layer), pts, w, _hex_rgb(op.get("c")) + (255,))
            elif t == "e" and (scoped or oL == 0):
                pts = _px_pts(op.get("pts") or [], rw, rh)
                w = max(1, round(float(op.get("s", 0.01)) * long_edge))
                mask = Image.new("L", (rw, rh), 0)
                _stroke_on(ImageDraw.Draw(mask), pts, w, 255)
                layer.paste((0, 0, 0, 0), (0, 0), mask)
            elif t == "E" and scoped:
                if op.get("w") is not None:                 # legacy rect erase
                    x1 = max(0, round(float(op.get("x", 0)) * rw) - 1)
                    y1 = max(0, round(float(op.get("y", 0)) * rh) - 1)
                    x2 = min(rw, round((float(op.get("x", 0)) + float(op.get("w", 0))) * rw) + 1)
                    y2 = min(rh, round((float(op.get("y", 0)) + float(op.get("h", 0))) * rh) + 1)
                    if x2 > x1 and y2 > y1:
                        layer.paste((0, 0, 0, 0), (x1, y1, x2, y2))
                else:                                       # erase the whole layer
                    layer = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
            elif t == "f" and scoped:
                sx = min(rw - 1, max(0, round(float(op.get("x", 0)) * rw)))
                sy = min(rh - 1, max(0, round(float(op.get("y", 0)) * rh)))
                _apply_fill(layer, sx, sy, _hex_rgb(op.get("c")), op.get("r"))
            elif t == "R" and scoped:                       # recolor the whole layer
                arr = np.array(layer, dtype=np.uint8)
                m = arr[..., 3] > 0
                arr[..., :3][m] = _hex_rgb(op.get("c"))
                layer.paste(Image.fromarray(arr))
            elif t == "M" and scoped:                       # move/scale the whole layer
                layer = _transform_layer(layer, op.get("a"), op.get("b"), rw, rh)
            elif t == "C" and op.get("N") == L:             # birth: copy of layer oL at this moment
                src_layer = _render_layer(ops, oL, rw, rh, i)
                layer = _transform_layer(src_layer, op.get("a"), op.get("b"), rw, rh)
        except Exception:
            continue                                        # a malformed op never breaks the preview
    return layer


def _render_ink(ops, rw, rh, z_layers=None, hidden=None):
    """Composite the region layers (back to front, hidden layers skipped)."""
    ops = [op for op in ops if isinstance(op, dict)]
    if any(op.get("L") is None for op in ops):              # migrate pre-layer drawings
        next_n = 2
        for op in ops:
            if op.get("L") is None:
                op["L"] = 0 if op.get("t") == "e" else 1
            if op.get("t") == "C" and op.get("N") is None:
                op["N"] = next_n
                next_n += 1
    ids = []
    for op in ops:
        lx = op.get("N") if op.get("t") == "C" else op.get("L")
        if lx and lx not in ids:
            ids.append(lx)
    hidden = hidden or set()
    order = [L for L in (z_layers or []) if L in ids]
    extras = [L for L in ids if L not in order]
    ink = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    for L in list(reversed(order)) + extras:                # back to front, new layers on top
        if L in hidden:
            continue
        ink.alpha_composite(_render_layer(ops, L, rw, rh))
    return ink


def _render_preview(boxes, width, height, bg=None, brightness=50, ops=None, bg_opacity=100):
    # Render the ink layer + regions + prompts over the reference image (or a blank canvas).
    if bg is not None:
        iw, ih = bg.size
        long_edge = max(iw, ih)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
        base = bg.convert("RGB").resize((rw, rh), Image.LANCZOS)
        if brightness < 100:                                # dim to match the editor's brightness slider
            base = ImageEnhance.Brightness(base).enhance(max(0.0, brightness / 100.0))
        if bg_opacity < 100:                                # blend toward the blank canvas (opacity layer)
            g = round(max(0, min(100, brightness)) / 100 * 128)
            grey = Image.new("RGB", (rw, rh), (g, g, g))
            base = Image.blend(grey, base, max(0.0, min(1.0, bg_opacity / 100.0)))
        img = base.convert("RGBA")
    else:
        long_edge = max(width, height)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw = max(1, round(width * scale))
        rh = max(1, round(height * scale))
        g = round(max(0, min(100, brightness)) / 100 * 128)  # blank canvas grey from the brightness slider
        img = Image.new("RGBA", (rw, rh), (g, g, g, 255))
    ink_arr = None
    if ops:                                                  # layered ink between background and boxes
        z = [b.get("L") for b in boxes if isinstance(b, dict) and b.get("L") is not None]
        hid = {b.get("L") for b in boxes if isinstance(b, dict) and b.get("hide") and b.get("L") is not None}
        ink = _render_ink(ops, rw, rh, z, hid)
        img = Image.alpha_composite(img, ink)
        ink_arr = np.asarray(ink)
    overlay = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fs = max(10, round(rh / 64))
    font = _font(fs)
    tag_font = _font(max(9, fs - 2))
    lh = fs + 2

    for i, box in enumerate(boxes):
        if not isinstance(box, dict) or box.get("nobbox") or box.get("hide"):
            continue                                        # skip unplaced and hidden elements
        palette = [c for c in (box.get("palette") or []) if c]
        r, g, b = _hex_rgb(palette[0]) if palette else (140, 140, 140)   # box = first palette color, else grey
        x1 = max(0, min(rw, round(box.get("x", 0) * rw)))
        y1 = max(0, min(rh, round(box.get("y", 0) * rh)))
        x2 = max(0, min(rw, round((box.get("x", 0) + box.get("w", 0)) * rw)))
        y2 = max(0, min(rh, round((box.get("y", 0) + box.get("h", 0)) * rh)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        if box.get("fill"):                                  # bucket-painted box: real 50% fill
            draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 128))
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)

        pal5 = palette[:5]                                   # palette shown as a strip along the top edge
        if pal5 and (x2 - x1) > 2:
            sh = max(5, fs // 2)
            seg = (x2 - x1) / len(pal5)
            for p, hexc in enumerate(pal5):
                sx = x1 + round(p * seg)
                draw.rectangle([sx, y1, x1 + round((p + 1) * seg), y1 + sh], fill=_hex_rgb(hexc))

        etype = "text" if box.get("type") == "text" else "obj"
        tag = str(i + 1).zfill(2)
        tw = draw.textlength(tag, font=tag_font)
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + fs + 2], fill=(r, g, b, 255))  # tag chip = box color
        tagfill = (0, 0, 0, 255) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255, 255)
        draw.text((x1 + 3, y1 + 1), tag, fill=tagfill, font=tag_font)

        body = box.get("desc", "") or ""
        if etype == "text" and box.get("text"):
            body = '"%s"%s' % (box["text"], " — " + body if body else "")
        if body and (x2 - x1) > 8:
            tfill = _text_fill((r, g, b), _box_bg_lum(box, (r, g, b), ink_arr, rw, rh))
            t_lum = 0.299 * tfill[0] + 0.587 * tfill[1] + 0.114 * tfill[2]
            halo = (0, 0, 0, 210) if t_lum >= 128 else (255, 255, 255, 220)
            ty = y1 + fs + 5
            for line in _wrap(draw, body, font, x2 - x1 - 8):
                if ty + lh > y2 + 2:                         # never pass the bottom edge
                    break
                draw.text((x1 + 4, ty), line, fill=tfill + (255,), font=font,
                          stroke_width=2, stroke_fill=halo)
                ty += lh

    img = Image.alpha_composite(img, overlay).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _norm_bbox(box):
    # Normalized {x, y, w, h} fractions (0-1) -> [ymin, xmin, ymax, xmax] on a 0-1000 grid.
    def c(v):
        return max(0, min(1000, round(v * 1000)))
    x, y, w, h = box.get("x", 0.0), box.get("y", 0.0), box.get("w", 0.0), box.get("h", 0.0)
    ymin, xmin, ymax, xmax = c(y), c(x), c(y + h), c(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def _palette(colors):
    # ["#rrggbb", ...] (or autogrow dict) -> ["#RRGGBB", ...] in order, dropping empties.
    if isinstance(colors, dict):
        colors = colors.values()
    return [c.upper() for c in colors if c]


def _dumps(v, lvl=0):
    # Like json.dumps(ensure_ascii=False, indent=4), but scalar arrays stay on one line.
    pad, end = "    " * (lvl + 1), "    " * lvl
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(_dumps(x, lvl) for x in v) + "]"
        return "[\n" + ",\n".join(pad + _dumps(x, lvl + 1) for x in v) + "\n" + end + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [pad + json.dumps(k, ensure_ascii=False) + ": " + _dumps(val, lvl + 1) for k, val in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    return json.dumps(v, ensure_ascii=False)


def _parse_json_list(s):
    if s:
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    return []


class Ideogram4PromptBuilderKJV2(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ideogram4PromptBuilderKJV2",
            display_name="Ideogram 4 Prompt Builder KJ V2",
            category="KJNodes/text",
            search_aliases=["ideogram", "caption", "bbox", "prompt builder", "json prompt",
                            "freehand", "draw", "paint", "fill"],
            is_experimental=True,
            description="""
Visual prompt builder for Ideogram 4's structured JSON caption format — freehand edition.

Sketch on the canvas with the brush, bucket-fill closed shapes like in Paint, and each
drawn shape is automatically outlined into its own region (bbox). Select a region to set
its type (obj/text), description, text, and color palette. A local image can be imported
as a background layer with adjustable opacity. Set the background and optional style
fields as widgets. Outputs the assembled caption JSON string.

bbox is normalized to a 0-1000 grid as [ymin, xmin, ymax, xmax]; width/height set
the canvas aspect ratio.""",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas aspect width (also the pixel grid the bbox is measured in). Ideogram 4 needs multiples of 16."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas aspect height (also the pixel grid the bbox is measured in). Ideogram 4 needs multiples of 16."),
                io.String.Input("high_level_description", multiline=True, default="",
                                tooltip="Optional one-line overview of the whole image (blank = omitted)."),
                io.String.Input("background", multiline=True, default="",
                                tooltip="Required scene background description."),
                io.DynamicCombo.Input("style", options=[
                    io.DynamicCombo.Option("none", []),
                    io.DynamicCombo.Option("photo", [
                        io.String.Input("photo", default=""),
                    ]),
                    io.DynamicCombo.Option("art_style", [
                        io.String.Input("art_style", default=""),
                    ]),
                ]),
                io.String.Input("aesthetics", default="", tooltip="Style descriptor (blank = omitted)."),
                io.String.Input("lighting", default="", tooltip="Style descriptor (blank = omitted)."),
                io.String.Input("medium", default="", tooltip="Style descriptor (blank = omitted)."),
                io.Image.Input("image", optional=True,
                               tooltip="Optional reference image shown as the editor background (and behind the preview)."),
                io.String.Input("import_json", default="", optional=True, force_input=True,
                                tooltip="Optional: a full caption JSON. When connected, it loads into the "
                                        "editor on run; the output always reflects the editor, never the raw input."),
                io.String.Input("style_palette_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized style color palette from the editor (managed by the node UI)."),
                io.String.Input("elements_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized regions from the editor (managed by the node UI)."),
                io.String.Input("draw_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized freehand drawing ops from the editor (managed by the node UI)."),
                io.Int.Input("bg_brightness", default=25, min=0, max=100, socketless=True, advanced=True,
                             tooltip="Background image brightness % (managed by the node UI slider)."),
                io.Int.Input("bg_opacity", default=100, min=0, max=100, socketless=True, advanced=True,
                             tooltip="Background image opacity % (managed by the node UI slider)."),
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
    def execute(cls, width, height, background, style,
                high_level_description="", aesthetics="", lighting="", medium="",
                style_palette_data="", elements_data="", draw_data="", import_json="",
                image=None, bg_brightness=25, bg_opacity=100) -> io.NodeOutput:
        boxes = _parse_json_list(elements_data)
        draw_ops = _parse_json_list(draw_data)

        caption = {}
        if high_level_description.strip():
            caption["high_level_description"] = high_level_description

        kind = style["style"]                               # "none" | "photo" | "art_style"
        if kind != "none":
            # The verifier requires every style key present (in order) once a style is
            # chosen; only color_palette is conditional. Emit blanks rather than omit.
            sd = {"aesthetics": aesthetics, "lighting": lighting}
            # photo: ...photo, medium...  |  art_style: ...medium, art_style...  (key order)
            if kind == "photo":
                sd["photo"] = style.get("photo", "")
                sd["medium"] = medium
            else:
                sd["medium"] = medium
                sd["art_style"] = style.get("art_style", "")
            palette = _palette(_parse_json_list(style_palette_data))
            if palette:
                sd["color_palette"] = palette
            caption["style_description"] = sd

        elements = []
        for box in boxes:
            if not isinstance(box, dict) or box.get("hide"):
                continue                                    # hidden regions are excluded, as if erased
            etype = "text" if box.get("type") == "text" else "obj"
            elem = {"type": etype}                          # key order matters
            if not box.get("nobbox"):                       # unplaced elements omit bbox
                elem["bbox"] = _norm_bbox(box)
            if etype == "text":
                elem["text"] = box.get("text", "")
            elem["desc"] = box.get("desc", "")
            palette = _palette(box.get("palette", []))
            if palette:
                elem["color_palette"] = palette[:5]
            elements.append(elem)

        caption["compositional_deconstruction"] = {
            "background": background,
            "elements": elements,
        }
        bg = None
        if image is not None:
            try:
                bg = Image.fromarray((image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            except Exception:
                bg = None
        preview = _render_preview(boxes, width, height, bg, bg_brightness, draw_ops, bg_opacity)

        # Pixel-space bboxes ({x, y, width, height}) for SAM3 / BoundingBox consumers.
        bbox_dicts = []
        for box in boxes:
            if not isinstance(box, dict) or box.get("nobbox") or box.get("hide"):
                continue
            x, y = box.get("x", 0.0), box.get("y", 0.0)
            bw, bh = box.get("w", 0.0), box.get("h", 0.0)
            if bw < 0:
                x += bw
                bw = -bw
            if bh < 0:
                y += bh
                bh = -bh
            bbox_dicts.append({"x": round(x * width), "y": round(y * height),
                               "width": round(bw * width), "height": round(bh * height)})
        # Per-frame nesting (list[list[dict]]) — the canonical BoundingBox shape that
        # SAM3 / crop nodes expect (bboxes[frame] -> list of boxes).
        bboxes_out = [bbox_dicts] if bbox_dicts else []

        # ui: send the resolved width/height so the editor canvas can follow connected
        # inputs; import_json (if wired) loads into the editor (output reflects editor only).
        ui = {"dims": [width, height]}
        if import_json and import_json.strip():
            try:
                cap = json.loads(import_json)
                if isinstance(cap, dict):
                    ui["caption"] = [_dumps(cap)]
            except json.JSONDecodeError:
                pass
        return io.NodeOutput(_dumps(caption), preview, bboxes_out, width, height, ui=ui)
