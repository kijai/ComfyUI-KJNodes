import { registerPromptBuilder, ideoGrid, ideoFromEdges } from "./ideogram4_prompt_builder.js";

// FLUX.3 caption format (nodes/flux3_nodes.py): the prompt, one space, then a JSON list of box rows on the
// 0-1000 grid. Boxes live on canvases ("views"): 0 = output, k = reference input ref_k. Boxes sharing an id
// form one row: on a reference = src_bbox, on the output = tgt_bbox.

// Python's json.dumps(v, ensure_ascii=False) formatting, so the copied caption matches the node output.
function pyDumps(v) {
  if (Array.isArray(v)) return "[" + v.map(pyDumps).join(", ") + "]";
  if (v && typeof v === "object") return "{" + Object.entries(v).map(([k, x]) => JSON.stringify(k) + ": " + pyDumps(x)).join(", ") + "}";
  return JSON.stringify(v);
}

// Editor boxes -> rows; mirrors _rows() in flux3_nodes.py.
function rows(ed) {
  const slots = ed.refSlots(), rank = new Map(slots.map((k, i) => [k, i]));
  const order = [], src = new Map(), tgt = new Map();
  for (const b of ed.node._boxes) {
    if (!b || b.nobbox) continue;
    const view = b.view || 0;
    if (view && !rank.has(view)) continue;               // its reference is no longer connected
    const id = String(b.id || "");
    if (!src.has(id) && !tgt.has(id)) order.push(id);
    const m = view ? src : tgt;
    if (!m.has(id)) m.set(id, b);
  }
  return order.map((id) => {
    const s = src.get(id), t = tgt.get(id);
    const desc = (t && t.desc) || (s && s.desc) || "";
    if (slots.length) {
      return { id, from: s ? `ref_image_${rank.get(s.view)}` : "target",
        src_bbox: s ? ideoGrid(s) : null, tgt_bbox: t ? ideoGrid(t) : null, desc };
    }
    return t ? { id, bbox: ideoGrid(t), desc } : null;
  }).filter(Boolean);
}

// [y_min, x_min, y_max, x_max] on the 0-1000 grid -> {x, y, w, h}, or null.
function gridBox(bb) {
  if (!Array.isArray(bb) || bb.length !== 4) return null;
  const n = bb.map(Number);
  return n.every(Number.isFinite) ? ideoFromEdges(n, 1000, 1000) : null;
}

registerPromptBuilder({
  nodeName: "Flux3PromptBuilderKJ",
  label: "FLUX.3",
  title: "FLUX.3 editor",
  noun: "box",
  tplDir: "kjnodes/flux3/templates",
  frontFirst: true,
  maxElemColors: 0,                                 // no colour fields: colours go in the description
  maxStyleColors: 0,
  format: "compact",
  tokenCap: 0,
  typeToggle: false,
  textCaption: true,                                // plain text caption, not a JSON object
  views: "references",                              // one canvas per connected reference (ref_1..)
  linkKey: "id",                                    // boxes with the same id on different canvases are one row
  importInput: "import_caption",
  captionWidgets: ["prompt"],
  fields: [
    { key: "id", placeholder: "id — cite it as <id> in the prompt" },
    { key: "desc", grow: 3, placeholder: "description of this object" },
  ],
  coords: {                                         // same order in both: ymin, xmin, ymax, xmax
    px: {
      hint: "ymin, xmin, ymax, xmax", title: "Pixel box (of the current canvas): ymin, xmin, ymax, xmax — editable",
      get: (b, W, H) => [Math.round(b.y * H), Math.round(b.x * W), Math.round((b.y + b.h) * H), Math.round((b.x + b.w) * W)],
      set: (n, W, H) => ideoFromEdges(n, W, H),
    },
    out: {
      hint: "ymin, xmin, ymax, xmax", title: "Exported box on the 0–1000 grid: ymin, xmin, ymax, xmax — editable",
      get: (b) => ideoGrid(b),
      set: (n) => ideoFromEdges(n, 1000, 1000),
    },
  },
  tagText: (b) => b.id || "—",
  // Where the row lives: placed (reference -> output), removed (reference only) or new (output only).
  describe(b, ed) {
    if (!ed.refSlots().length) return "";
    const views = ed.node._boxes.filter((o) => o.id === b.id).map((o) => o.view || 0);
    const src = views.find((v) => v > 0);
    if (src && views.includes(0)) return `ref_${src} → output`;
    if (src) return `ref_${src} only (removed; draw it on output to place it)`;
    return "new in the output";
  },
  blankBox: () => ({ id: "", desc: "" }),
  // A box drawn right after switching canvases joins the row that was selected; otherwise a fresh obj_N.
  initBox(nb, ed, prev) {
    const boxes = ed.node._boxes, view = nb.view || 0;
    if (prev && prev.id && !boxes.some((o) => o !== nb && o.id === prev.id && (o.view || 0) === view)) {
      nb.id = prev.id; nb.desc = prev.desc || "";
      return;
    }
    let n = 0;
    for (const o of boxes) { const m = /^obj_(\d+)$/.exec(o.id || ""); if (m) n = Math.max(n, +m[1]); }
    nb.id = "obj_" + (n + 1);
  },
  // "prompt [rows]": the last "[" from which the rest parses as a JSON list; mirrors _split_caption().
  parseCaption(t) {
    if (!t) return null;
    for (let i = t.length - 1; i >= 0; i--) {
      if (t[i] !== "[") continue;
      try { const r = JSON.parse(t.slice(i)); if (Array.isArray(r)) return { prompt: t.slice(0, i).trimEnd(), rows: r }; }
      catch (e) {}
    }
    return { prompt: t.trim(), rows: [] };
  },
  elements: (cap) => (Array.isArray(cap.rows) ? cap.rows : []),
  // One row -> its boxes; ref_image_N goes to the canvas of ref_{N+1}. Mirrors _rows_to_boxes().
  elemToBox(el, idx) {
    if (!el || typeof el !== "object") return null;
    const base = { id: String(el.id || `obj_${(idx || 0) + 1}`), desc: String(el.desc || "") };
    const out = [];
    const m = /^ref_image_(\d+)$/.exec(String(el.from || ""));
    const src = gridBox(el.src_bbox);
    if (src && m) out.push({ ...src, ...base, view: +m[1] + 1 });
    const tgt = gridBox("tgt_bbox" in el ? el.tgt_bbox : el.bbox);
    if (tgt) out.push({ ...tgt, ...base, view: 0 });
    return out;
  },
  buildCaption(ed) {                                // mirrors Flux3PromptBuilderKJ.execute
    const prompt = ed.getW("prompt") || "", r = rows(ed);
    return r.length ? prompt.trimEnd() + " " + pyDumps(r) : prompt;
  },
  applyCaption(cap, ed) { ed.setW("prompt", cap.prompt || ""); },
});
