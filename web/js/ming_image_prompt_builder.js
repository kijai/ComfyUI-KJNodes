import { registerPromptBuilder, cleanPalette } from "./ideogram4_prompt_builder.js";

// Ming-Image caption format (nodes/ming_image_nodes.py): canvas_settings + layers listed back (background)
// to front, on the shared Ideogram 4 editor. The aspect/coordinate helpers mirror the Python side.
const RATIOS = [[1, 1], [5, 4], [4, 3], [3, 2], [16, 10], [5, 3], [16, 9], [2, 1], [21, 9], [3, 1]];
const RATIO_TOL = 0.02;                                    // |log ratio| ~2%
const gcd = (a, b) => (b ? gcd(b, a % b) : a);
const roundHalfUp = (v, nd) => Math.floor(v * 10 ** nd + 0.5) / 10 ** nd;   // same as Python _round_half_up
const f3 = (v) => roundHalfUp(v, 3).toFixed(3);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const txt = (v) => (typeof v === "string" ? v : v == null ? "" : JSON.stringify(v));

// "16:9": nearest common ratio within ~2%, else the reduced fraction, else decimal.
function ratioStr(w, h) {
  w = Math.round(w); h = Math.round(h);
  if (!(w > 0 && h > 0)) return "1:1";
  const r = w / h;
  let best = null, bd = Infinity;
  for (const [a, b] of RATIOS) {
    for (const [p, q] of (a === b ? [[a, b]] : [[a, b], [b, a]])) {
      const d = Math.abs(Math.log(r * q / p));
      if (d < bd) { best = [p, q]; bd = d; }
    }
  }
  if (bd < RATIO_TOL) return best[0] + ":" + best[1];
  const g = gcd(w, h);
  if (Math.max(w, h) / g <= 32) return w / g + ":" + h / g;
  return r >= 1 ? roundHalfUp(r, 2).toFixed(2) + ":1" : "1:" + roundHalfUp(1 / r, 2).toFixed(2);
}
function aspectStr(w, h, withSize) {
  const s = ratioStr(w, h);
  return withSize ? `${s}, ${w} × ${h} px` : s;
}
// "16:9" / "1:1, 2048 × 2048 px" / "1920x1080" -> width/height ratio, or null.
function parseAspect(s) {
  for (const rx of [/(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)/, /(\d+(?:\.\d+)?)\s*[×xX*]\s*(\d+(?:\.\d+)?)/]) {
    const m = s.match(rx);
    if (m && +m[1] > 0 && +m[2] > 0) return +m[1] / +m[2];
  }
  return null;
}
// Reshape to ratio r at about the same pixel area (multiples of 16); unchanged if already within ~2%.
// Repeated until stable, so reshaping the node's already-reshaped dims again is a no-op (mirrors Python).
function dimsForAspect(r, w, h) {
  const r16 = (v) => Math.max(64, Math.min(16384, Math.floor(v / 16 + 0.5) * 16));
  for (let i = 0; i < 8; i++) {
    if (!r || Math.abs(Math.log(r * h / w)) < RATIO_TOL) break;
    const area = w * h, nw = r16(Math.sqrt(area * r)), nh = r16(Math.sqrt(area / r));
    if (nw === w && nh === h) break;
    w = nw; h = nh;
  }
  return [w, h];
}

// Box -> [cx, cy, w, h] fractions, clipped to the canvas (the exported coordinates).
function centerBox(b) {
  let x1 = clamp01(b.x), x2 = clamp01(b.x + b.w), y1 = clamp01(b.y), y2 = clamp01(b.y + b.h);
  if (x1 > x2) [x1, x2] = [x2, x1];
  if (y1 > y2) [y1, y2] = [y2, y1];
  return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1];
}
function coordsStr(b) {
  const [cx, cy, w, h] = centerBox(b).map(f3);
  return `cx: ${cx}, cy: ${cy}, w: ${w}, h: ${h}`;
}
function fromCenter(cx, cy, w, h) {
  const x1 = clamp01(cx - Math.abs(w) / 2), x2 = clamp01(cx + Math.abs(w) / 2);
  const y1 = clamp01(cy - Math.abs(h) / 2), y2 = clamp01(cy + Math.abs(h) / 2);
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}
// Layer coordinates ("cx: .., cy: .., w: .., h: ..", or a {cx,cy,w,h} object / 4-array) -> {x, y, w, h}, or null.
function parseCoords(c) {
  let vals;
  if (typeof c === "string") {
    const named = {};
    for (const m of c.matchAll(/\b(cx|cy|w|h|width|height)\s*[:=]\s*(-?(?:\d+\.?\d*|\.\d+))/gi)) {
      const k = m[1].toLowerCase();
      named[k === "width" ? "w" : k === "height" ? "h" : k] = m[2];
    }
    vals = ["cx", "cy", "w", "h"].map((k) => named[k]);
    if (vals.includes(undefined)) vals = c.match(/-?(?:\d+\.?\d*|\.\d+)/g) || [];   // bare "0.5, 0.5, 1, 1"
  } else if (Array.isArray(c)) vals = c;
  else if (c && typeof c === "object") vals = ["cx", "cy", "w", "h"].map((k) => c[k]);
  else return null;
  const n = vals.map((v) => (v == null || v === "" ? NaN : Number(v)));
  if (n.length !== 4 || !n.every(Number.isFinite)) return null;
  const [cx, cy, w, h] = n;
  return { x: cx - w / 2, y: cy - h / 2, w, h };
}

registerPromptBuilder({
  nodeName: "MingImagePromptBuilderKJ",
  label: "Ming-Image",
  title: "Ming-Image editor",
  noun: "layer",
  tplDir: "kjnodes/ming_image/templates",
  captionKey: "layers",                             // a pasted/imported JSON must have it
  frontFirst: false,                                // layers[0] = background (back-most); new layers go on top
  maxElemColors: 16,                                // no model cap; keeps the swatch row manageable
  maxStyleColors: 0,                                // no global style palette
  format: "pretty",                                 // Ming-Image's example prompts are indented
  indent: "  ",
  compactTip: "Compact single-line JSON. Unchecked (default) = indented, like Ming-Image's example prompts.",
  tokenCap: 0,                                      // no hard prompt limit
  typeToggle: false,                                // rendered text goes in the description, quoted
  captionWidgets: ["aspect_ratio", "ambient_lighting", "image_style"],
  fields: [
    { key: "desc", grow: 2, placeholder: "description — content, typography, materials, lighting; quote rendered text exactly once" },
    { key: "rel", placeholder: "hierarchy_and_relation — ownership, alignment, containment, stacking, occlusion" },
  ],
  coords: {                                         // same order in both: center x, center y, width, height
    px: {
      hint: "cx, cy, w, h", width: 150,
      title: "Pixel box (of the node's width/height): center x, center y, width, height — editable",
      get: (b, W, H) => { const [cx, cy, w, h] = centerBox(b); return [Math.round(cx * W), Math.round(cy * H), Math.round(w * W), Math.round(h * H)]; },
      set: ([cx, cy, w, h], W, H) => fromCenter(cx / W, cy / H, w / W, h / H),
    },
    out: {
      hint: "cx, cy, w, h", width: 186,
      title: "Exported coordinates, normalized 0–1: center x, center y, width, height — editable",
      get: (b) => centerBox(b).map(f3),
      set: ([cx, cy, w, h]) => fromCenter(cx, cy, w, h),
    },
  },
  blankBox: () => ({ desc: "", rel: "", palette: [] }),
  elements: (cap) => (Array.isArray(cap.layers) ? cap.layers : []),
  elemToBox(el, idx) {
    if (!el || typeof el !== "object") return null;
    const cs = el.color_specs;
    const box = { desc: txt(el.description), rel: txt(el.hierarchy_and_relation),
      palette: Array.isArray(cs) ? cs.filter((c) => typeof c === "string") : [] };
    const xy = parseCoords(el.coordinates);
    if (xy) Object.assign(box, xy);
    else {                                          // no coordinates: "unplaced" placeholder, exported without them
      const k = (idx || 0) % 6;
      Object.assign(box, { x: 0.03 + k * 0.035, y: 0.03 + k * 0.035, w: 0.22, h: 0.14, nobbox: true });
    }
    return box;
  },
  buildCaption(ed) {                                // mirrors MingImagePromptBuilderKJ.execute key order
    const { getW, node } = ed;
    const [W, H] = ed.dims();
    const mode = getW("aspect_ratio") || "auto";
    const aspect = mode === "custom" ? getW("aspect_ratio.custom") : aspectStr(W, H, mode === "auto_with_size");
    const layers = node._boxes.map((b) => {
      const layer = { description: b.desc || "" };
      if (!b.nobbox) layer.coordinates = coordsStr(b);   // unplaced layers omit coordinates
      layer.hierarchy_and_relation = b.rel || "";
      layer.color_specs = cleanPalette(b.palette);
      return layer;
    });
    return {
      canvas_settings: { aspect_ratio: aspect, ambient_lighting: getW("ambient_lighting"), image_style: getW("image_style") },
      layers,
    };
  },
  applyCaption(cap, ed) {                           // canvas_settings -> widgets (the editor sets the boxes)
    const { setW, findW, node } = ed;
    const cs = cap.canvas_settings && typeof cap.canvas_settings === "object" ? cap.canvas_settings : {};
    const aspect = txt(cs.aspect_ratio).trim();
    let [w, h] = ed.dims();
    const r = parseAspect(aspect);
    if (r && !node._bgImg) {                        // follow the caption's aspect (a background image governs it otherwise)
      [w, h] = dimsForAspect(r, w, h);
      setW("width", w); setW("height", h);
    }
    // "auto" when the caption says what auto would write, else keep its string verbatim
    const mode = !aspect || aspect === aspectStr(w, h, false) ? "auto"
      : aspect === aspectStr(w, h, true) ? "auto_with_size" : "custom";
    const arW = findW("aspect_ratio");
    if (arW) arW.value = mode;                      // setter synchronously rebuilds the sub-widget
    if (mode === "custom") setW("aspect_ratio.custom", aspect);
    setW("ambient_lighting", txt(cs.ambient_lighting));
    setW("image_style", txt(cs.image_style));
  },
});
