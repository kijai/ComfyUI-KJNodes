// Ideogram 4 Prompt Builder KJ V2 — freehand edition.
//
// Based on KJNodes' ideogram4_prompt_builder.js, extended with a real drawing
// editor:
//  - freehand brush, straight line, rectangle and ellipse outline tools
//  - Paint-style bucket fill of closed shapes
//  - automatic outlining: each drawn shape becomes its own bbox region
//  - eraser, color eyedropper, undo/redo for the ink layer
//  - local image import as a background layer with adjustable opacity
import { chainCallback, addMiddleClickPan, addWheelPassthrough, cursorForBboxMode, watchImageInputs, captureVideoFrame } from './utility.js';
const { app } = window.comfyAPI.app;

const HANDLE = 10;           // hit radius (canvas px) for corners/edges
const MAX_ELEM_COLORS = 5;   // Ideogram 4 per-element palette cap
const MAX_STYLE_COLORS = 16; // Ideogram 4 style palette cap
const INK_LONG = 768;        // ink raster long edge (fills/labeling resolution)
const INK_ALPHA = 24;        // alpha above this counts as ink (fill barrier / region pixels)
let copiedBox = null;        // internal clipboard for copy/paste of regions (shared across nodes)

// Track the most recent generated image so it can be grabbed as a background.
let lastResultImage = null;
try {
  app.api?.addEventListener?.("executed", (e) => {
    const imgs = e?.detail?.output?.images;
    if (Array.isArray(imgs) && imgs.length) {
      lastResultImage = imgs[imgs.length - 1];
      // Live nodes swap their preview for the full-res final result automatically.
      for (const n of livePreviewNodes) n._ideoGrabFinal?.();
    }
  });
} catch (e) {}
function resultViewUrl(img) {
  const p = new URLSearchParams({ filename: img.filename || "", subfolder: img.subfolder || "", type: img.type || "output" });
  return "/view?" + p.toString();
}

// Nodes opted into "live background": feed them the sampling preview frames as they arrive.
const livePreviewNodes = new Set();
try {
  app.api?.addEventListener?.("b_preview", (e) => {
    const blob = e?.detail;
    if (!blob || !livePreviewNodes.size) return;
    createImageBitmap(blob).then((bmp) => {
      let used = false;
      for (const n of livePreviewNodes) { n._ideoSetLiveBg?.(bmp); used = true; }
      if (!used && bmp.close) bmp.close();
    }).catch(() => {});
  });
} catch (e) {}

// Parse a #rrggbb hex into {r,g,b}, or null if malformed.
function hexRgb(hex) {
  const h = (hex || "").replace("#", "");
  if (h.length < 6) return null;
  return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
}
// Perceived luminance (0-255) of an {r,g,b}.
function luminance({ r, g, b }) { return 0.299 * r + 0.587 * g + 0.114 * b; }

// The hex color lightened toward white if too dark, so it stays readable on the dark canvas.
function readableText(hex) {
  const c = hexRgb(hex);
  if (!c) return "#d4d4d4";
  let { r, g, b } = c;
  const lum = luminance(c), MIN = 130;
  if (lum < MIN) {
    const t = (MIN - lum) / (255 - lum);
    r = Math.round(r + (255 - r) * t); g = Math.round(g + (255 - g) * t); b = Math.round(b + (255 - b) * t);
  }
  return `rgb(${r},${g},${b})`;
}

// Black or white, whichever contrasts better with the given hex background.
function textOn(hex) {
  const c = hexRgb(hex);
  if (!c) return "#000";
  return luminance(c) > 140 ? "#000" : "#fff";
}

function injectStyle() {
  if (document.getElementById("kjid2-style")) return;
  const s = document.createElement("style");
  s.id = "kjid2-style";
  s.textContent = `
    .kjid2-wrap { display:flex; flex-direction:column; overflow:hidden; position:relative; pointer-events:auto; gap:4px; }
    .kjid2-canvas { cursor:crosshair; display:block; width:100%; height:auto; flex:0 0 auto; background:#1a1a1a; border-radius:4px; outline:none; }
    .kjid2-bar { display:flex; align-items:center; gap:6px; font:11px sans-serif; color:#aaa; user-select:none; padding:0 2px; flex:0 0 auto; flex-wrap:wrap; }
    .kjid2-panel { display:flex; flex-direction:column; gap:5px; padding:6px; background:#262626; border-radius:4px; font:11px sans-serif; color:#bbb; flex:0 0 auto; }
    .kjid2-row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
    .kjid2-btn { background:#333; border:1px solid #555; border-radius:4px; color:#bbb; font:11px sans-serif; cursor:pointer; padding:2px 8px; line-height:16px; white-space:nowrap; flex-shrink:0; }
    .kjid2-btn:hover { border-color:#46b4e6; color:#fff; }
    .kjid2-btn.active { border-color:#46b4e6; color:#46b4e6; background:#2a3a42; }
    .kjid2-btn:disabled { opacity:.35; cursor:default; }
    .kjid2-btn:disabled:hover { border-color:#555; color:#bbb; }
    .kjid2-tool { padding:2px 6px; font-size:12px; }
    .kjid2-color { width:26px; height:20px; padding:0; border:1px solid #555; border-radius:3px; background:none; cursor:pointer; flex-shrink:0; }
    .kjid2-mid { display:flex; gap:6px; align-items:flex-start; flex:0 0 auto; }
    .kjid2-mid .kjid2-canvas { flex:1 1 0; min-width:0; width:auto; height:auto; }
    .kjid2-side { width:200px; flex:0 0 auto; box-sizing:border-box; background:#262626; border-radius:4px; padding:4px; font:11px sans-serif; color:#bbb; display:flex; flex-direction:column; gap:2px; overflow-y:auto; overflow-x:hidden; }
    .kjid2-side.collapsed { width:26px; }
    .kjid2-shdr { display:flex; align-items:center; gap:4px; color:#888; padding:0 2px 3px; flex:0 0 auto; }
    .kjid2-srow { display:flex; align-items:center; gap:5px; padding:2px 4px; border-radius:4px; cursor:pointer; user-select:none; flex:0 0 auto; transition:transform .18s ease, opacity .12s ease, background .12s; }
    .kjid2-srow:hover { background:#333; }
    .kjid2-srow.active { background:#2a3a42; box-shadow:inset 0 0 0 1px #46b4e6; }
    .kjid2-srow.dragging { opacity:.4; box-shadow:0 0 0 2px #46b4e6; background:#333; }
    .kjid2-srow.hidden-r { opacity:.45; }
    .kjid2-ssw { width:14px; height:14px; border-radius:3px; border:1px solid #666; flex:0 0 auto; position:relative; cursor:pointer; }
    .kjid2-ssw:hover { box-shadow:0 0 0 2px #46b4e6; }
    .kjid2-ssw input { position:absolute; opacity:0; width:0; height:0; pointer-events:none; }
    .kjid2-sdel:hover { color:#fff; background:#a33; border-radius:3px; }
    .kjid2-snum { font:bold 10px monospace; color:#888; width:16px; flex:0 0 auto; }
    .kjid2-stxt { flex:1 1 auto; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .kjid2-stxt.empty { color:#777; font-style:italic; }
    .kjid2-eye { background:none; border:none; cursor:pointer; font-size:12px; padding:0 2px; color:#bbb; flex:0 0 auto; line-height:1; }
    .kjid2-eye:hover { color:#fff; }
    .kjid2-area { width:100%; box-sizing:border-box; background:#1d1d1d; border:1px solid #444; border-radius:4px; color:#ddd; font:13px monospace; padding:4px 6px; resize:vertical; min-height:36px; }
    .kjid2-sw { width:20px; height:20px; border:1px solid #666; border-radius:3px; cursor:pointer; flex-shrink:0; position:relative; transition:transform .18s ease, box-shadow .12s ease, opacity .12s ease; }
    .kjid2-sw:hover { transform:scale(1.2); box-shadow:0 0 0 2px #46b4e6; z-index:3; }
    .kjid2-sw.dragging { opacity:.4; box-shadow:0 0 0 2px #46b4e6; }
    body.kjid2-dragging, body.kjid2-dragging * { cursor:move !important; }
    .kjid2-sw input { position:absolute; opacity:0; width:0; height:0; pointer-events:none; }
    .kjid2-inline { position:absolute; box-sizing:border-box; background:rgba(18,18,18,0.92); border:2px solid #46b4e6; border-radius:3px; color:#fff; font:13px monospace; padding:3px 4px; resize:none; outline:none; z-index:10; }
    .kjid2-menu { position:fixed; z-index:10000; background:#262626; border:1px solid #555; border-radius:6px; padding:4px; box-shadow:0 6px 20px rgba(0,0,0,0.55); font:12px sans-serif; color:#ddd; max-height:60vh; overflow-y:auto; min-width:210px; max-width:340px; }
    .kjid2-mhdr { font:11px sans-serif; color:#888; padding:2px 6px 4px; user-select:none; }
    .kjid2-lrow { display:flex; align-items:center; gap:6px; padding:3px 5px; border-radius:4px; cursor:move; user-select:none; transition:transform .18s ease, box-shadow .12s ease, opacity .12s ease, background .12s; }
    .kjid2-lrow:hover { background:#333; }
    .kjid2-lrow.active { background:#2a3a42; box-shadow:inset 0 0 0 1px #46b4e6; }
    .kjid2-lrow.dragging { opacity:.4; box-shadow:0 0 0 2px #46b4e6; background:#333; }
    .kjid2-lsw { width:16px; height:16px; border-radius:3px; border:1px solid #666; flex:0 0 auto; }
    .kjid2-lnum { font:bold 11px monospace; color:#888; flex:0 0 auto; width:18px; }
    .kjid2-ltext { flex:1 1 auto; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .kjid2-ltext.empty { color:#777; font-style:italic; }
    .kjid2-lbtn { background:none; border:none; color:#999; cursor:pointer; font:13px sans-serif; line-height:1; padding:2px 5px; border-radius:3px; flex:0 0 auto; }
    .kjid2-lbtn:hover { color:#fff; background:#444; }
    .kjid2-lbtn.del:hover { color:#fff; background:#a33; }
  `;
  document.head.appendChild(s);
}

app.registerExtension({
  name: "KJNodes.Ideogram4PromptBuilderV2",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "Ideogram4PromptBuilderKJV2") return;
    injectStyle();

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const findW = (n) => node.widgets?.find((w) => w.name === n);
      const elementsWidget = findW("elements_data");
      const stylePaletteWidget = findW("style_palette_data");
      const drawDataWidget = findW("draw_data");
      const bgBrightnessWidget = findW("bg_brightness");
      const bgOpacityWidget = findW("bg_opacity");
      if (bgBrightnessWidget && typeof bgBrightnessWidget.value !== "number") bgBrightnessWidget.value = 25;
      if (bgOpacityWidget && typeof bgOpacityWidget.value !== "number") bgOpacityWidget.value = 100;
      const wWidget = findW("width"), hWidget = findW("height");
      // Hide the data widgets while keeping them serializable.
      function hideDataWidgets() {
        for (const w of [elementsWidget, stylePaletteWidget, drawDataWidget, bgBrightnessWidget, bgOpacityWidget]) {
          if (!w) continue;
          w.hidden = true;
          w.computeSize = () => [0, -4];
        }
        for (const name of ["elements_data", "style_palette_data", "draw_data", "bg_brightness", "bg_opacity"]) {
          const i = node.inputs?.findIndex((inp) => inp.name === name);
          if (i != null && i !== -1) node.removeInput(i);
        }
      }
      hideDataWidgets();

      node.properties = node.properties || {};
      node._boxes = [];        // {x,y,w,h normalized 0-1, type, text, desc, palette[], auto?}
      node._stylePalette = []; // global style color palette (hex[])
      node._activeIdx = -1;
      node._drawing = false;   // box drag in progress
      node._placing = false;   // duplicate-placement mode: active box follows the cursor until clicked
      node._dragMode = null;
      node._dragStartN = null; // mouse-down point, normalized
      node._boxAtStart = null; // active box snapshot at drag start
      node._hoverTitle = null; // index of the title chip under the cursor
      node._hoverBox = null;   // index of the box under the cursor
      node._focused = false;   // editor (DOM) focused — gates the active-box highlight
      node._selected = false;  // node selected in the graph
      node._bgImg = null;      // optional reference image shown as the canvas background
      node._bgKind = null;     // null | "input" | "grab" | "import" | "live"
      node._lastImported = ""; // last import_json applied to the editor (avoid re-apply)
      node._areaH = node._areaH || {};      // remembered textarea heights (per field)
      node._areaObservers = [];             // live ResizeObservers to disconnect on rebuild

      // ── freehand ink state ──
      node._draws = [];        // ops: {t:"s"|"e", c?, s, pts:[[x,y]..]} | {t:"f", c, x, y} | {t:"E", x,y,w,h}
      node._redo = [];         // redo stack for ink ops
      node._stroke = null;     // in-progress stroke/shape op (not yet committed)
      node._poly = null;       // in-progress polyline (line tool, Paint-style clicks)
      node._polyCursor = null; // live cursor for the polyline preview segment
      node._editL = null;      // layer being edited (pencil button), null = not editing
      node._editStart = 0;     // ops count when the edit session began (for revert)
      node._editChips = null;  // canvas rects of the live edit chips (ok / cancel)
      node._strokeLastPx = null; // last recorded point (display px) for decimation
      node._inkCanvas = null;  // cached ink raster (offscreen canvas)
      node._inkData = null;    // cached ImageData of the ink raster (fills/labeling)
      node._cursorN = null;    // cursor position (normalized) for the brush ring
      node._tool = "brush";
      node._prevTool = "brush"; // tool to return to after the eyedropper
      node._brushColor = typeof node.properties.ideoBrushColor === "string" ? node.properties.ideoBrushColor : "#ff4040";
      node._brushSize = typeof node.properties.ideoBrushSize === "number" ? node.properties.ideoBrushSize : 14;
      let opsRev = 0;          // bumped on every ink change — drives the ink cache key
      let inkKey = "";
      let shapeStart = null;   // anchor point (normalized) of an in-progress line/rect/ellipse

      // ── DOM ──
      const wrap = document.createElement("div");
      wrap.className = "kjid2-wrap";
      const bar = document.createElement("div");
      bar.className = "kjid2-bar";
      const hint = document.createElement("span");
      hint.style.flex = "1";
      const copyBtn = document.createElement("button");
      copyBtn.className = "kjid2-btn";
      copyBtn.textContent = "Copy";
      copyBtn.title = "Copy the current caption JSON to the clipboard";
      const importBtn = document.createElement("button");
      importBtn.className = "kjid2-btn";
      importBtn.textContent = "Paste";
      importBtn.title = "Parse a caption JSON (clipboard, else paste prompt) and populate the node";
      const clearBtn = document.createElement("button");
      clearBtn.className = "kjid2-btn";
      clearBtn.textContent = "Clear all";
      clearBtn.title = "Remove all regions, ink and style colors";
      const tokenSpan = document.createElement("span");
      tokenSpan.style.cssText = "color:#888; white-space:nowrap;";
      tokenSpan.title = "Rough token estimate (~chars/4). Grey <256, green healthy, orange nearing, red ≥2048 (model cap — will error)";
      const grabBtn = document.createElement("button");
      grabBtn.className = "kjid2-btn";
      grabBtn.addEventListener("mousedown", (e) => e.stopPropagation());
      grabBtn.addEventListener("click", () => { (node._bgKind && node._bgKind !== "input" && node._bgImg) ? node._clearBg() : node._grabResultBg(); });
      function updateGrabBtn() {
        const clear = node._bgKind && node._bgKind !== "input" && node._bgImg;
        grabBtn.textContent = clear ? "Clear BG" : "Grab BG";
        grabBtn.title = clear ? "Remove the grabbed/imported background"
          : "Use the last generated image as the background";
      }
      const importBgBtn = document.createElement("button");
      importBgBtn.className = "kjid2-btn";
      importBgBtn.textContent = "Import BG";
      importBgBtn.title = "Import a local image as the background layer (use the ◐ slider for its opacity)";
      importBgBtn.addEventListener("mousedown", (e) => e.stopPropagation());
      const bgFileInput = document.createElement("input");
      bgFileInput.type = "file";
      bgFileInput.accept = "image/*";
      bgFileInput.style.display = "none";
      importBgBtn.addEventListener("click", () => bgFileInput.click());
      bgFileInput.addEventListener("change", () => {
        const f = bgFileInput.files && bgFileInput.files[0];
        bgFileInput.value = "";
        if (!f) return;
        const fr = new FileReader();
        fr.onload = () => {
          const img = new Image();
          img.onload = () => {
            // Downscale + recompress so the data URL stored in the workflow stays small.
            const long = Math.max(img.naturalWidth, img.naturalHeight) || 1;
            const sc = Math.min(1, 1536 / long);
            const c = document.createElement("canvas");
            c.width = Math.max(1, Math.round(img.naturalWidth * sc));
            c.height = Math.max(1, Math.round(img.naturalHeight * sc));
            c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
            let url;
            try { url = c.toDataURL("image/webp", 0.85); } catch (e) { url = c.toDataURL("image/png"); }
            node.properties.ideoBgImport = url;
            node._bgKind = "import";
            loadBg(url);
          };
          img.src = fr.result;
        };
        fr.readAsDataURL(f);
      });
      const liveLabel = document.createElement("label");
      liveLabel.style.cssText = "display:flex;align-items:center;gap:3px;flex:0 0 auto;cursor:pointer;";
      liveLabel.title = "Use the live sampling preview as the background while generating";
      const liveChk = document.createElement("input");
      liveChk.type = "checkbox";
      liveChk.checked = !!node.properties.liveBg;
      liveChk.addEventListener("mousedown", (e) => e.stopPropagation());
      liveChk.addEventListener("change", () => {
        node.properties.liveBg = liveChk.checked;
        if (liveChk.checked) livePreviewNodes.add(node); else livePreviewNodes.delete(node);
      });
      liveLabel.appendChild(liveChk); liveLabel.appendChild(document.createTextNode("Live"));
      if (liveChk.checked) livePreviewNodes.add(node);
      const briLbl = document.createElement("span");
      briLbl.textContent = "☀";
      briLbl.title = "Background brightness";
      const bgSlider = document.createElement("input");
      bgSlider.type = "range"; bgSlider.min = "0"; bgSlider.max = "100"; bgSlider.step = "1";
      bgSlider.value = bgBrightnessWidget ? bgBrightnessWidget.value : 25;
      bgSlider.title = "Background brightness (image or blank canvas)";
      bgSlider.style.cssText = "width:56px;flex:0 0 auto;";
      stopProp(bgSlider);
      bgSlider.addEventListener("input", () => { if (bgBrightnessWidget) bgBrightnessWidget.value = parseInt(bgSlider.value, 10); drawCanvas(); });
      const opaLbl = document.createElement("span");
      opaLbl.textContent = "◐";
      opaLbl.title = "Background image opacity (layer transparency)";
      const opaSlider = document.createElement("input");
      opaSlider.type = "range"; opaSlider.min = "0"; opaSlider.max = "100"; opaSlider.step = "1";
      opaSlider.value = bgOpacityWidget ? bgOpacityWidget.value : 100;
      opaSlider.title = "Background image opacity % (100 = opaque, 0 = invisible)";
      opaSlider.style.cssText = "width:56px;flex:0 0 auto;";
      stopProp(opaSlider);
      opaSlider.addEventListener("input", () => { if (bgOpacityWidget) bgOpacityWidget.value = parseInt(opaSlider.value, 10); drawCanvas(); });
      bar.appendChild(hint); bar.appendChild(liveLabel); bar.appendChild(grabBtn); bar.appendChild(importBgBtn);
      bar.appendChild(briLbl); bar.appendChild(bgSlider); bar.appendChild(opaLbl); bar.appendChild(opaSlider);
      bar.appendChild(tokenSpan); bar.appendChild(copyBtn); bar.appendChild(importBtn); bar.appendChild(clearBtn);
      updateGrabBtn();

      // ── drawing toolbar ──
      const drawBar = document.createElement("div");
      drawBar.className = "kjid2-bar";
      const TOOLS = [
        ["select",  "➤",  "Select / move / resize regions (V)"],
        ["brush",   "✏",  "Freehand brush — each separate shape becomes a region (B)"],
        ["line",    "╱",  "Polyline — click to chain segments, click near the start to close the shape, dbl-click/Enter to end (L)"],
        ["rect",    "▭",  "Rectangle outline — hold Shift for a square (R)"],
        ["ellipse", "◯",  "Ellipse outline — hold Shift for a circle (O)"],
        ["fill",    "🪣", "Bucket fill a closed area, like in Paint (G)"],
        ["erase",   "⌫",  "Eraser for the ink layer (E)"],
        ["pick",    "💧", "Eyedropper — click the canvas to pick a color (I)"],
        ["boxr",    "⬚",  "Drag a rectangle region, like the original node (X)"],
      ];
      const TOOL_HINTS = {
        select: "Click to select · drag to move/resize · alt-click overlap · dbl-click edit · right-click region list · Del remove",
        brush: "Draw freehand — each separate shape is auto-outlined into a region · Ctrl+Z undo",
        line: "Click to add points · click near the start to close · dbl-click/Enter ends · Esc cancels · Shift 45°",
        rect: "Drag a rectangle outline · Shift for a square · fill it with the bucket",
        ellipse: "Drag an ellipse outline · Shift for a circle · fill it with the bucket",
        fill: "Click a shape: the inside AND the outline take the color · click a line to recolor it",
        erase: "Erase ink — regions follow the remaining ink · Ctrl+Z undo",
        pick: "Click the canvas to pick that color as the brush color",
        boxr: "Drag to draw a rectangle region · click to select · dbl-click edit · Del remove",
      };
      const toolBtns = {};
      for (const [id, label, title] of TOOLS) {
        const btn = document.createElement("button");
        btn.className = "kjid2-btn kjid2-tool";
        btn.textContent = label;
        btn.title = title;
        stopProp(btn);
        btn.addEventListener("click", () => setTool(id));
        toolBtns[id] = btn;
        drawBar.appendChild(btn);
      }
      const colorInp = document.createElement("input");
      colorInp.type = "color";
      colorInp.className = "kjid2-color";
      colorInp.value = node._brushColor;
      colorInp.title = "Brush / shape / fill color";
      stopProp(colorInp);
      colorInp.addEventListener("input", () => {
        node._brushColor = colorInp.value;
        node.properties.ideoBrushColor = colorInp.value;
        drawCanvas();
      });
      drawBar.appendChild(colorInp);
      const sizeSlider = document.createElement("input");
      sizeSlider.type = "range"; sizeSlider.min = "2"; sizeSlider.max = "80"; sizeSlider.step = "1";
      sizeSlider.value = String(node._brushSize);
      sizeSlider.title = "Brush / line / eraser thickness";
      sizeSlider.style.cssText = "width:64px;flex:0 0 auto;";
      stopProp(sizeSlider);
      const sizeLbl = document.createElement("span");
      sizeLbl.style.cssText = "color:#888;width:26px;flex:0 0 auto;";
      sizeLbl.textContent = node._brushSize + "px";
      sizeSlider.addEventListener("input", () => {
        node._brushSize = parseInt(sizeSlider.value, 10);
        node.properties.ideoBrushSize = node._brushSize;
        sizeLbl.textContent = node._brushSize + "px";
        drawCanvas();
      });
      drawBar.appendChild(sizeSlider); drawBar.appendChild(sizeLbl);
      const undoBtn = document.createElement("button");
      undoBtn.className = "kjid2-btn"; undoBtn.textContent = "↶";
      undoBtn.title = "Undo last ink op (Ctrl+Z)";
      stopProp(undoBtn);
      undoBtn.addEventListener("click", () => undoOp());
      const redoBtn = document.createElement("button");
      redoBtn.className = "kjid2-btn"; redoBtn.textContent = "↷";
      redoBtn.title = "Redo ink op (Ctrl+Y / Ctrl+Shift+Z)";
      stopProp(redoBtn);
      redoBtn.addEventListener("click", () => redoOp());
      const clearInkBtn = document.createElement("button");
      clearInkBtn.className = "kjid2-btn"; clearInkBtn.textContent = "Clear ink";
      clearInkBtn.title = "Remove all freehand ink (auto regions disappear too)";
      stopProp(clearInkBtn);
      clearInkBtn.addEventListener("click", () => clearInk());
      drawBar.appendChild(undoBtn); drawBar.appendChild(redoBtn); drawBar.appendChild(clearInkBtn);
      function updateInkButtons() {
        undoBtn.disabled = !node._draws.length;
        redoBtn.disabled = !node._redo.length;
        clearInkBtn.disabled = !node._draws.length;
      }
      function setTool(id) {
        if (!toolBtns[id]) id = "brush";
        if (node._poly && id !== "line") finishPoly(); // leaving the polyline tool commits the shape
        if (id === "pick" && node._tool !== "pick") node._prevTool = node._tool;
        node._tool = id;
        node.properties.ideoTool = id;
        for (const k in toolBtns) toolBtns[k].classList.toggle("active", k === id);
        node._hoverTitle = null; node._hoverBox = null; node._cursorN = null;
        canvasEl.style.cursor = "crosshair";
        canvasEl.title = TOOL_HINTS[id] || "";
        if (node._activeIdx < 0) renderPanel();               // refresh the idle hint text
        drawCanvas();
      }

      // Persistent global style-palette row
      const styleBar = document.createElement("div");
      styleBar.className = "kjid2-bar";
      const styleLbl = document.createElement("span");
      styleLbl.textContent = "Style colors:";
      styleBar.appendChild(styleLbl);

      const canvasEl = document.createElement("canvas");
      canvasEl.className = "kjid2-canvas";
      canvasEl.tabIndex = 0;                                  // focusable, so it can receive key events
      const ctx = canvasEl.getContext("2d", { willReadFrequently: true });
      addWheelPassthrough(wrap);
      addMiddleClickPan(canvasEl);

      const panel = document.createElement("div");
      panel.className = "kjid2-panel";

      // ── side region list: color + prompt per region, eye to hide/show in the scene ──
      const sidePanel = document.createElement("div");
      sidePanel.className = "kjid2-side";
      stopProp(sidePanel);
      const sideHdr = document.createElement("div");
      sideHdr.className = "kjid2-shdr";
      const sideHdrLabel = document.createElement("span");
      sideHdrLabel.textContent = "Regions";
      sideHdrLabel.title = "Top = front (drawn above the others) — drag rows to reorder, like layers";
      const eyeAllBtn = document.createElement("button");
      eyeAllBtn.className = "kjid2-eye";
      eyeAllBtn.textContent = "👁";
      eyeAllBtn.title = "Show all / hide all regions in the scene";
      stopProp(eyeAllBtn);
      eyeAllBtn.addEventListener("click", () => {
        const anyHidden = node._boxes.some((b) => b.hide);
        for (const b of node._boxes) { if (anyHidden) delete b.hide; else b.hide = true; }
        commit();
      });
      const collBtn = document.createElement("button");
      collBtn.className = "kjid2-eye";
      collBtn.style.marginLeft = "auto";
      stopProp(collBtn);
      collBtn.addEventListener("click", () => {
        node.properties.ideoSideCollapsed = !node.properties.ideoSideCollapsed;
        rebuildSideList(); fitNode();
      });
      sideHdr.appendChild(sideHdrLabel); sideHdr.appendChild(eyeAllBtn); sideHdr.appendChild(collBtn);
      const sideRows = document.createElement("div");
      sideRows.style.cssText = "display:flex;flex-direction:column;gap:2px;min-height:0;";
      sidePanel.appendChild(sideHdr); sidePanel.appendChild(sideRows);

      const midRow = document.createElement("div");
      midRow.className = "kjid2-mid";
      midRow.appendChild(canvasEl);
      midRow.appendChild(sidePanel);

      // Canvas row above panel so the panel grows downward without shifting the canvas.
      wrap.appendChild(bar); wrap.appendChild(drawBar); wrap.appendChild(styleBar); wrap.appendChild(midRow); wrap.appendChild(panel);
      wrap.appendChild(bgFileInput);

      const TOOLBAR_H = 44;
      node._widgetHeight = 380;
      node.ideoEditor = node.addDOMWidget("ideo_editor", "Ideogram4EditorV2", wrap, {
        serialize: false, hideOnZoom: false,
        getMinHeight: () => node._widgetHeight,
      });
      node.resizable = true;

      // ── canvas sizing ──
      // The display size is CSS-driven (width:100% + aspect-ratio); the backing store
      // is sized to display × devicePixelRatio in _draw() so text/lines stay crisp.
      function setCanvasSize(w, h) {
        canvasEl.style.aspectRatio = `${w} / ${h}`;          // display shape only
        if (node.graph) node.graph.setDirtyCanvas(true, true);
      }
      function syncCanvasToDims() {
        const w = wWidget ? wWidget.value : 1024, h = hWidget ? hWidget.value : 1024;
        setCanvasSize(Math.max(1, w), Math.max(1, h));
        drawCanvas();
      }

      // Content height = panel's bottom edge in the wrapper (includes toolbars/canvas/gaps).
      function recalcWidgetHeight() {
        const contentH = panel.offsetTop + panel.offsetHeight;
        if (contentH > 0) {
          node._widgetHeight = contentH + 10;                  // margin pad
        } else {                                               // not laid out yet — estimate
          const ratio = (hWidget?.value || 1) / (wWidget?.value || 1);
          node._widgetHeight = Math.round(Math.max(100, node.size[0] - 30) * ratio) + TOOLBAR_H + 70;
        }
      }
      function fitNode() {
        recalcWidgetHeight();
        // computeSize (stable min-heights), not last_y which creeps with growable widgets above.
        const minH = node.computeSize()[1];
        if (node.size[1] < minH) node.setSize([node.size[0], minH]);
      }

      // ── geometry helpers ── (logical CSS px = the displayed canvas size)
      function logW() { return canvasEl.offsetWidth || 1; }
      function logH() { return canvasEl.offsetHeight || 1; }
      function toPx(b) {
        const W = logW(), H = logH();
        return { x1: b.x * W, y1: b.y * H, x2: (b.x + b.w) * W, y2: (b.y + b.h) * H };
      }
      function mouseN(e) {
        const r = canvasEl.getBoundingClientRect();
        return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
      }
      function clamp01(v) { return Math.max(0, Math.min(1, v)); }
      function r4(v) { return Math.round(v * 1e4) / 1e4; }
      function rect4(r) { return [r4(r.x), r4(r.y), r4(Math.abs(r.w)), r4(Math.abs(r.h))]; }
      // greedy word-wrap to maxW px; words wider than the box are hard-broken
      // so text never escapes the box (ctx.font must be set by caller)
      function wrapLines(text, maxW) {
        const lines = [];
        for (const para of text.split("\n")) {
          let line = "";
          for (let word of para.split(/\s+/)) {
            if (!word) continue;
            while (ctx.measureText(word).width > maxW && word.length > 1) {
              let k = word.length;
              while (k > 1 && ctx.measureText(word.slice(0, k)).width > maxW) k--;
              if (line) { lines.push(line); line = ""; }
              lines.push(word.slice(0, k));
              word = word.slice(k);
            }
            const test = line ? line + " " + word : word;
            if (line && ctx.measureText(test).width > maxW) { lines.push(line); line = word; }
            else line = test;
          }
          lines.push(line);
        }
        return lines;
      }
      // Average luminance behind a box: its painted fill, or the drawing under it.
      function boxBgLum(b, col) {
        if (b.fill) {
          const c = hexRgb(col);
          return c ? luminance(c) * 0.5 + 13 : 26;       // 50% color over the dark canvas
        }
        ensureDisplay();
        const id = node._inkData;
        if (!id || !node._draws.length) return 26;
        const RW = id.width, RH = id.height, d = id.data;
        let sum = 0, n = 0;
        for (let iy = 1; iy <= 3; iy++) {
          for (let ix = 1; ix <= 3; ix++) {
            const px = Math.min(RW - 1, Math.max(0, Math.round((b.x + b.w * ix / 4) * RW)));
            const py = Math.min(RH - 1, Math.max(0, Math.round((b.y + b.h * iy / 4) * RH)));
            const p = (py * RW + px) * 4;
            const a = d[p + 3] / 255;
            sum += luminance({ r: d[p], g: d[p + 1], b: d[p + 2] }) * a + 26 * (1 - a);
            n++;
          }
        }
        return n ? sum / n : 26;
      }
      // Text shade of the box color that stays readable on its background:
      // dark shade over a bright fill, light shade over the dark canvas.
      // `light` drives the contrasting halo drawn behind the text.
      function readableOn(b, col) {
        if (boxBgLum(b, col) > 110) {
          const c = hexRgb(col) || { r: 140, g: 140, b: 140 };
          const f = Math.min(1, 50 / Math.max(1, luminance(c)));
          const r = Math.round(c.r * f), g = Math.round(c.g * f), b2 = Math.round(c.b * f);
          return { css: `rgb(${r},${g},${b2})`, light: luminance({ r, g, b: b2 }) >= 128 };
        }
        return { css: readableText(col), light: true };
      }
      function normalizeBox(b) {
        // collapse negative size to positive top-left + w/h, clamp into canvas
        let x = b.x, y = b.y, w = b.w, h = b.h;
        if (w < 0) { x += w; w = -w; }
        if (h < 0) { y += h; h = -h; }
        x = clamp01(x); y = clamp01(y);
        w = Math.min(w, 1 - x); h = Math.min(h, 1 - y);
        return { ...b, x, y, w: Math.max(0, w), h: Math.max(0, h) };
      }

      // All boxes under the point, smallest first: clicking overlapping boxes picks
      // the most specific one, so a huge background region never swallows every
      // click. Regions hidden via the side panel are not hit-testable.
      function boxesAt(mN) {
        const rx = HANDLE / logW(), ry = HANDLE / logH();
        const res = [];
        for (let i = 0; i < node._boxes.length; i++) {
          const b = node._boxes[i];
          if (b.hide) continue;
          const mode = rectHitTestN(mN.x, mN.y, b.x, b.y, b.x + b.w, b.y + b.h, rx, ry);
          if (mode) res.push({ index: i, mode, area: Math.abs(b.w * b.h) });
        }
        res.sort((p, q) => p.area - q.area);
        return res;
      }
      // Hover / right-click: prefer a resize handle on the active box, else topmost.
      function hitTest(mN) {
        const cands = boxesAt(mN);
        if (!cands.length) return null;
        return cands.find((c) => c.index === node._activeIdx && c.mode !== "move") || cands[0];
      }
      // Tag-chip rects (canvas px), placed to avoid overlapping each other: each
      // box's tag tries top-left, top-right, bottom-right, bottom-left in turn.
      function tagRects() {
        ctx.font = "bold 11px monospace";
        const W = logW(), H = logH(), h = 14;
        const placed = [], rects = [];
        const hits = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
        for (let i = 0; i < node._boxes.length; i++) {
          const b = node._boxes[i];
          if (b.hide) { rects[i] = null; continue; }
          const x1 = b.x * W, y1 = b.y * H, x2 = (b.x + b.w) * W, y2 = (b.y + b.h) * H;
          const tag = String(i + 1).padStart(2, "0");
          const w = ctx.measureText(tag).width + 8;
          let pick = [x1, y1];
          for (const [cx, cy] of [[x1, y1], [x2 - w, y1], [x2 - w, y2 - h], [x1, y2 - h]]) {
            if (!placed.some((p) => hits({ x: cx, y: cy, w, h }, p))) { pick = [cx, cy]; break; }
          }
          const r = { x: pick[0], y: pick[1], w, h, tag };
          placed.push(r); rects[i] = r;
        }
        return rects;
      }
      function titleAt(mN) {
        const px = mN.x * logW(), py = mN.y * logH();
        const rects = tagRects();
        for (let i = node._boxes.length - 1; i >= 0; i--) {
          const r = rects[i];
          if (r && px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) return i;
        }
        return null;
      }
      // Click selection: active box's resize handle wins (corner resize); then a
      // title-chip click selects that box (drawn to front); Alt-click cycles the
      // overlap stack; else the topmost box.
      function pickForSelection(mN, cycle) {
        const cands = boxesAt(mN);
        if (!cands.length) return null;
        const ah = cands.find((c) => c.index === node._activeIdx && c.mode !== "move");
        if (ah && !cycle) return ah;
        const ti = titleAt(mN);
        if (ti !== null && !cycle) return { index: ti, mode: "move" };
        if (cycle && cands.length > 1) {
          const pos = cands.findIndex((c) => c.index === node._activeIdx);
          return cands[(pos + 1) % cands.length];
        }
        return cands.find((c) => c.index === node._activeIdx && c.mode !== "move") || cands[0];
      }
      // normalized variant of rectHitTest with separate x/y radii
      function rectHitTestN(mx, my, x1, y1, x2, y2, rx, ry) {
        const h = (cx, cy) => Math.abs(mx - cx) < rx && Math.abs(my - cy) < ry;
        if (h(x1, y1)) return "resize-tl";
        if (h(x2, y1)) return "resize-tr";
        if (h(x1, y2)) return "resize-bl";
        if (h(x2, y2)) return "resize-br";
        if (mx >= x1 && mx <= x2 && Math.abs(my - y1) < ry) return "resize-t";
        if (mx >= x1 && mx <= x2 && Math.abs(my - y2) < ry) return "resize-b";
        if (my >= y1 && my <= y2 && Math.abs(mx - x1) < rx) return "resize-l";
        if (my >= y1 && my <= y2 && Math.abs(mx - x2) < rx) return "resize-r";
        if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) return "move";
        return null;
      }

      function applyDrag(mode, start, dN) {
        let { x, y, w, h } = start;
        const dx = dN.x, dy = dN.y;
        switch (mode) {
          case "move": x += dx; y += dy; x = clamp01(Math.min(x, 1 - w)); y = clamp01(Math.min(y, 1 - h)); break;
          case "draw":
          case "resize-br": w += dx; h += dy; break;
          case "resize-tl": x += dx; y += dy; w -= dx; h -= dy; break;
          case "resize-tr": y += dy; w += dx; h -= dy; break;
          case "resize-bl": x += dx; w -= dx; h += dy; break;
          case "resize-t": y += dy; h -= dy; break;
          case "resize-b": h += dy; break;
          case "resize-l": x += dx; w -= dx; break;
          case "resize-r": w += dx; break;
        }
        return mode === "move" ? { ...start, x, y } : normalizeBox({ ...start, x, y, w, h });
      }

      // ── ink raster (strokes / shapes / eraser / bucket fills) ──
      function inkDims() {
        const w = Math.max(1, wWidget ? wWidget.value : 1024), h = Math.max(1, hWidget ? hWidget.value : 1024);
        const sc = INK_LONG / Math.max(w, h);
        return { RW: Math.max(8, Math.round(w * sc)), RH: Math.max(8, Math.round(h * sc)) };
      }
      // Draw one polyline stroke (round caps/joins); a single point becomes a dot.
      function strokePath(ictx, pts, lw, W, H, color, composite) {
        ictx.globalCompositeOperation = composite || "source-over";
        ictx.strokeStyle = color;
        ictx.fillStyle = color;
        ictx.lineWidth = lw;
        ictx.lineCap = "round";
        ictx.lineJoin = "round";
        if (!pts.length) return;
        if (pts.length === 1) {
          ictx.beginPath();
          ictx.arc(pts[0][0] * W, pts[0][1] * H, Math.max(0.5, lw / 2), 0, Math.PI * 2);
          ictx.fill();
          return;
        }
        ictx.beginPath();
        ictx.moveTo(pts[0][0] * W, pts[0][1] * H);
        for (let i = 1; i < pts.length; i++) ictx.lineTo(pts[i][0] * W, pts[i][1] * H);
        ictx.stroke();
      }
      // Paint-style bucket fill — one color per shape (Ideogram only honors a
      // single color per box):
      //  - seed on an empty area: flood it (50% alpha, 4-conn so no diagonal
      //    leaks) and, if the area is enclosed (doesn't touch the canvas border),
      //    recolor the surrounding drawing to the same color
      //  - seed on a previous fill: recolor that fill and its outline (refill)
      //  - seed on a stroke: recolor the whole connected drawing
      function floodFillImageData(id, op) {
        const RW = id.width, RH = id.height, data = id.data, N = RW * RH;
        const sx = Math.min(RW - 1, Math.max(0, Math.round(op.x * RW)));
        const sy = Math.min(RH - 1, Math.max(0, Math.round(op.y * RH)));
        const start = sy * RW + sx;
        const c = hexRgb(op.c) || { r: 255, g: 255, b: 255 };
        const a0 = data[start * 4 + 3];
        const stack = new Int32Array(N);
        const setRGB = (p) => { data[p * 4] = c.r; data[p * 4 + 1] = c.g; data[p * 4 + 2] = c.b; };

        if (Array.isArray(op.r)) {
          // box-bounded fill: the box rectangle is the wall; flood the empty area
          // inside it (strokes block, even unclosed), then the WHOLE layer takes
          // the color — one color per case, like Paint inside a frame
          const rx1 = Math.max(0, Math.round(op.r[0] * RW)), ry1 = Math.max(0, Math.round(op.r[1] * RH));
          const rx2 = Math.min(RW - 1, Math.round((op.r[0] + op.r[2]) * RW)), ry2 = Math.min(RH - 1, Math.round((op.r[1] + op.r[3]) * RH));
          if (a0 <= INK_ALPHA && sx >= rx1 && sx <= rx2 && sy >= ry1 && sy <= ry2) {
            const seen = new Uint8Array(N);
            let top = 0;
            stack[top++] = start; seen[start] = 1;
            while (top > 0) {
              const q = stack[--top];
              setRGB(q);
              data[q * 4 + 3] = 128;
              const qx = q % RW, qy = (q / RW) | 0;
              if (qx > rx1) { const p2 = q - 1; if (!seen[p2] && data[p2 * 4 + 3] <= INK_ALPHA) { seen[p2] = 1; stack[top++] = p2; } }
              if (qx < rx2) { const p2 = q + 1; if (!seen[p2] && data[p2 * 4 + 3] <= INK_ALPHA) { seen[p2] = 1; stack[top++] = p2; } }
              if (qy > ry1) { const p2 = q - RW; if (!seen[p2] && data[p2 * 4 + 3] <= INK_ALPHA) { seen[p2] = 1; stack[top++] = p2; } }
              if (qy < ry2) { const p2 = q + RW; if (!seen[p2] && data[p2 * 4 + 3] <= INK_ALPHA) { seen[p2] = 1; stack[top++] = p2; } }
            }
          }
          for (let p2 = 0; p2 < N; p2++) {                 // unify the case color
            if (data[p2 * 4 + 3] > INK_ALPHA) setRGB(p2);
          }
          return;
        }

        if (a0 >= 200) {
          // seed on a stroke: recolor the connected drawing (8-conn, alpha kept)
          const seen = new Uint8Array(N);
          let top = 0;
          stack[top++] = start; seen[start] = 1;
          while (top > 0) {
            const q = stack[--top];
            setRGB(q);
            const qx = q % RW, qy = (q / RW) | 0;
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (!dx && !dy) continue;
                const nx = qx + dx, ny = qy + dy;
                if (nx < 0 || ny < 0 || nx >= RW || ny >= RH) continue;
                const np = ny * RW + nx;
                if (!seen[np] && data[np * 4 + 3] > INK_ALPHA) { seen[np] = 1; stack[top++] = np; }
              }
            }
          }
          return;
        }

        // flood the seed's class — empty area, or a previous fill — 4-connected
        const empty = a0 <= INK_ALPHA;
        const match = (a) => empty ? a <= INK_ALPHA : (a > INK_ALPHA && a < 200);
        const region = new Uint8Array(N);
        let top = 0;
        stack[top++] = start; region[start] = 1;
        let touchesBorder = false;
        while (top > 0) {
          const q = stack[--top];
          setRGB(q);
          if (empty) data[q * 4 + 3] = 128;
          const qx = q % RW, qy = (q / RW) | 0;
          if (qx === 0 || qy === 0 || qx === RW - 1 || qy === RH - 1) touchesBorder = true;
          if (qx > 0) { const p = q - 1; if (!region[p] && match(data[p * 4 + 3])) { region[p] = 1; stack[top++] = p; } }
          if (qx < RW - 1) { const p = q + 1; if (!region[p] && match(data[p * 4 + 3])) { region[p] = 1; stack[top++] = p; } }
          if (qy > 0) { const p = q - RW; if (!region[p] && match(data[p * 4 + 3])) { region[p] = 1; stack[top++] = p; } }
          if (qy < RH - 1) { const p = q + RW; if (!region[p] && match(data[p * 4 + 3])) { region[p] = 1; stack[top++] = p; } }
        }
        if (touchesBorder) return;             // background fill — leave the outlines alone

        // recolor the connected drawing around the filled area (alpha kept)
        const seen = new Uint8Array(N);
        let top2 = 0;
        for (let p = 0; p < N; p++) {
          if (!region[p]) continue;
          const qx = p % RW, qy = (p / RW) | 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (!dx && !dy) continue;
              const nx = qx + dx, ny = qy + dy;
              if (nx < 0 || ny < 0 || nx >= RW || ny >= RH) continue;
              const np = ny * RW + nx;
              if (!region[np] && !seen[np] && data[np * 4 + 3] > INK_ALPHA) { seen[np] = 1; stack[top2++] = np; }
            }
          }
        }
        while (top2 > 0) {
          const q = stack[--top2];
          setRGB(q);
          const qx = q % RW, qy = (q / RW) | 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (!dx && !dy) continue;
              const nx = qx + dx, ny = qy + dy;
              if (nx < 0 || ny < 0 || nx >= RW || ny >= RH) continue;
              const np = ny * RW + nx;
              if (!seen[np] && !region[np] && data[np * 4 + 3] > INK_ALPHA) { seen[np] = 1; stack[top2++] = np; }
            }
          }
        }
      }
// ── layered ink engine ──
      // Each drawn region is its OWN layer: ops carry L (layer id). A layer
      // renders independently (its fills and erases see only its own ink), so
      // regions NEVER merge — wherever you move them, even drawn on top of each
      // other. Layers composite in box z-order with depth transparency.
      function opsMaxLayer() {
        let m = 0;
        for (const op of node._draws) {
          if (op.L > m) m = op.L;
          if (op.t === "C" && op.N > m) m = op.N;
        }
        return m;
      }
      function newLayerId() { return opsMaxLayer() + 1; }
      // The layer a new gesture targets: the edited layer, else a fresh one.
      function gestureLayer() { return node._editL != null ? node._editL : newLayerId(); }
      function eraseLayer() { return node._editL != null ? node._editL : 0; }   // 0 = erases every layer
      function drawTransformed(lctx, srcCanvas, RW, RH, a, b) {
        const [ax, ay, aw, ah] = a || [0, 0, 0, 0];
        const [bx, by, bw, bh] = b || [0, 0, 0, 0];
        if (aw <= 0 || ah <= 0 || bw <= 0 || bh <= 0) return;
        lctx.save();
        lctx.globalCompositeOperation = "source-over";
        lctx.translate(bx * RW, by * RH);
        lctx.scale(bw / aw, bh / ah);
        lctx.drawImage(srcCanvas, -ax * RW, -ay * RH);
        lctx.restore();
      }
      // Render one layer's raster (ops up to `limit` — used for copy snapshots).
      function renderLayer(L, RW, RH, limit) {
        const c = document.createElement("canvas");
        c.width = RW; c.height = RH;
        const lctx = c.getContext("2d", { willReadFrequently: true });
        const n = limit == null ? node._draws.length : limit;
        for (let i = 0; i < n; i++) {
          const op = node._draws[i];
          const oL = op.L || 0;
          const scoped = oL === L;
          if (op.t === "s" && scoped) {
            strokePath(lctx, op.pts || [], Math.max(1, op.s * Math.max(RW, RH)), RW, RH, op.c || "#ffffff", "source-over");
          } else if (op.t === "e" && (scoped || oL === 0)) {
            strokePath(lctx, op.pts || [], Math.max(1, op.s * Math.max(RW, RH)), RW, RH, "#000", "destination-out");
          } else if (op.t === "E" && scoped) {
            if (op.w != null) {                              // legacy rect erase
              lctx.globalCompositeOperation = "destination-out";
              lctx.fillStyle = "#000";
              lctx.fillRect(op.x * RW - 1, op.y * RH - 1, op.w * RW + 2, op.h * RH + 2);
              lctx.globalCompositeOperation = "source-over";
            } else {
              lctx.clearRect(0, 0, RW, RH);                  // erase the whole layer
            }
          } else if (op.t === "f" && scoped) {
            const id = lctx.getImageData(0, 0, RW, RH);
            floodFillImageData(id, op);                      // barriers = this layer only
            lctx.putImageData(id, 0, 0);
          } else if (op.t === "R" && scoped) {
            const id = lctx.getImageData(0, 0, RW, RH);      // recolor the whole layer (alpha kept)
            const d = id.data, col = hexRgb(op.c) || { r: 255, g: 255, b: 255 };
            for (let p = 0; p < RW * RH; p++) {
              if (d[p * 4 + 3] > 0) { d[p * 4] = col.r; d[p * 4 + 1] = col.g; d[p * 4 + 2] = col.b; }
            }
            lctx.putImageData(id, 0, 0);
          } else if (op.t === "M" && scoped) {
            const snap = document.createElement("canvas");   // move/scale the whole layer
            snap.width = RW; snap.height = RH;
            snap.getContext("2d").drawImage(c, 0, 0);
            lctx.clearRect(0, 0, RW, RH);
            drawTransformed(lctx, snap, RW, RH, op.a, op.b);
          } else if (op.t === "C" && op.N === L) {
            const src = renderLayer(oL, RW, RH, i);          // birth: copy of layer oL at this moment
            lctx.clearRect(0, 0, RW, RH);
            drawTransformed(lctx, src, RW, RH, op.a, op.b);
          }
        }
        return c;
      }
      function rebuildInk() {
        const { RW, RH } = inkDims();
        const ids = [];
        for (const op of node._draws) {
          const L = op.t === "C" ? op.N : op.L;
          if (L && !ids.includes(L)) ids.push(L);
        }
        const rasters = new Map();
        for (const L of ids) {
          const c = renderLayer(L, RW, RH, null);
          const d = c.getContext("2d").getImageData(0, 0, RW, RH).data;
          const alpha = new Uint8Array(RW * RH);
          let minx = RW, maxx = -1, miny = RH, maxy = -1, count = 0;
          const colors = new Map();
          for (let p = 0; p < RW * RH; p++) {
            const a = d[p * 4 + 3];
            alpha[p] = a;
            if (a <= INK_ALPHA) continue;
            count++;
            const x = p % RW, y = (p / RW) | 0;
            if (x < minx) minx = x;
            if (x > maxx) maxx = x;
            if (y < miny) miny = y;
            if (y > maxy) maxy = y;
            const key = (d[p * 4] << 16) | (d[p * 4 + 1] << 8) | d[p * 4 + 2];
            colors.set(key, (colors.get(key) || 0) + 1);
          }
          let best = 0x8c8c8c, bestN = 0;
          for (const [k, n2] of colors) { if (n2 > bestN) { bestN = n2; best = k; } }
          rasters.set(L, {
            canvas: c, alpha, count,
            bbox: count ? { x: minx / RW, y: miny / RH, w: (maxx - minx + 1) / RW, h: (maxy - miny + 1) / RH } : null,
            color: "#" + best.toString(16).padStart(6, "0"),
          });
        }
        node._layerRasters = rasters;
        node._inkSize = { RW, RH };
      }
      function ensureInk() {
        const { RW, RH } = inkDims();
        const key = RW + "x" + RH + ":" + opsRev;
        if (node._layerRasters && inkKey === key) return;
        inkKey = key;
        rebuildInk();
        displayKey = "";                                     // force a recomposite
      }
      // Composite the layers for display: box z-order (index 0 = front), depth
      // transparency per plane, hidden layers skipped, edit mode dims the others.
      let displayKey = "";
      function depthAlpha(rank) { return Math.max(0.3, Math.pow(0.85, rank)); }
      function buildComposite(excludeL) {
        const { RW, RH } = node._inkSize;
        const c = document.createElement("canvas");
        c.width = RW; c.height = RH;
        const cctx = c.getContext("2d", { willReadFrequently: true });
        const vis = node._boxes.filter((b) => !b.hide);
        // "extra" = layers with NO box at all (mid-gesture); a hidden box's layer
        // must NOT come back through here
        const known = new Set(node._boxes.map((b) => b.L).filter((v) => v != null));
        const extra = [...node._layerRasters.keys()].filter((L) => !known.has(L));
        for (let i = vis.length - 1; i >= 0; i--) {          // back to front, box-rank alpha
          const L = vis[i].L;
          if (L == null || L === excludeL) continue;
          const r = node._layerRasters.get(L);
          if (!r || !r.count) continue;
          let a = depthAlpha(i);
          if (node._editL != null) a = (L === node._editL) ? 1 : Math.min(a, 0.22);
          cctx.globalAlpha = a;
          cctx.drawImage(r.canvas, 0, 0);
        }
        for (const L of extra) {                             // brand-new layers (mid-gesture)
          if (L === excludeL) continue;
          const r = node._layerRasters.get(L);
          if (!r || !r.count) continue;
          cctx.globalAlpha = (node._editL != null && L !== node._editL) ? 0.22 : 1;
          cctx.drawImage(r.canvas, 0, 0);
        }
        cctx.globalAlpha = 1;
        return c;
      }
      function ensureDisplay() {
        ensureInk();
        const zsig = node._boxes.map((b) => (b.hide ? "h" : "") + (b.L != null ? b.L : "p")).join(",")
          + "|" + (node._editL == null ? "" : node._editL);
        const key = inkKey + "::" + zsig;
        if (node._inkCanvas && displayKey === key) return;
        displayKey = key;
        node._inkCanvas = buildComposite(null);
        node._inkData = node._inkCanvas.getContext("2d").getImageData(0, 0, node._inkSize.RW, node._inkSize.RH);
      }
      // ── regions ⇆ layers sync ──
      // One box per layer: the box bbox is the tight bbox of ITS OWN layer ink,
      // its color the layer's dominant color. Layers never merge — a region keeps
      // its identity wherever it sits, even drawn on top of another.
      function syncAutoBoxes() {
        ensureInk();
        const rasters = node._layerRasters;
        const activeObj = node._boxes[node._activeIdx] || null;
        node._boxes = node._boxes.filter((b) => {
          if (b.L == null) return true;                      // plain boxes are untouched
          const r = rasters.get(b.L);
          if (!r || !r.count || !r.bbox) return false;       // layer fully erased -> box gone
          b.x = r.bbox.x; b.y = r.bbox.y; b.w = r.bbox.w; b.h = r.bbox.h;
          b.src = { ...r.bbox };
          delete b.nobbox;
          if (!b.palUser) {
            b.palette = b.palette || [];
            b.palette[0] = r.color;
          }
          return true;
        });
        const have = new Set(node._boxes.map((b) => b.L).filter((v) => v != null));
        const created = [];
        for (const [L, r] of rasters) {
          if (have.has(L) || !r.count || !r.bbox) continue;
          const nb = { x: r.bbox.x, y: r.bbox.y, w: r.bbox.w, h: r.bbox.h, type: "obj", text: "", desc: "",
                       palette: [r.color], auto: true, L, src: { ...r.bbox } };
          node._boxes.unshift(nb);                           // a new drawing lands on the front plane
          created.push(nb);
        }
        node._activeIdx = activeObj ? node._boxes.indexOf(activeObj) : -1;
        return created;
      }

      // ── ink ops stack (undo/redo) ──
      function opsChanged(selectNewest) {
        opsRev++;
        ensureInk();
        const created = syncAutoBoxes();
        node._lastCreated = created;
        if (selectNewest && created.length) node._activeIdx = node._boxes.indexOf(created[created.length - 1]);
        if (node._activeIdx >= node._boxes.length) node._activeIdx = node._boxes.length ? node._boxes.length - 1 : -1;
        commit();
        updateInkButtons();
        fitNode();
      }
      function pushOp(op, selectNewest) {
        node._draws.push(op);
        node._redo.length = 0;
        opsChanged(selectNewest);
      }
      function undoOp() {
        if (!node._draws.length) return;
        if (node._editL != null && node._draws.length <= node._editStart) return;
        node._redo.push(node._draws.pop());
        opsChanged(false);
      }
      function redoOp() {
        if (!node._redo.length) return;
        node._draws.push(node._redo.pop());
        opsChanged(true);
      }
      function clearInk() {
        if (!node._draws.length) return;
        node._editL = null;
        node._draws = [];
        node._redo = [];
        opsChanged(false);
      }
      // Smallest visible box whose interior contains the point.
      function boxIndexAt(mN) {
        let best = -1, bestA = Infinity;
        node._boxes.forEach((b, i) => {
          if (b.hide) return;
          if (mN.x < b.x || mN.x > b.x + b.w || mN.y < b.y || mN.y > b.y + b.h) return;
          const a2 = Math.abs(b.w * b.h);
          if (a2 < bestA) { bestA = a2; best = i; }
        });
        return best;
      }
      // Bucket fill, Paint-style with the box as the frame:
      //  - on a drawing: that whole case takes the color
      //  - inside ANY box: fills the box area bounded by its edges + its strokes
      //    (works even when the drawing is not closed) — plain boxes paint fully
      //  - the background (outside every box) is NOT fillable
      function doFill(mN) {
        ensureInk();
        const { RW, RH } = node._inkSize;
        const px = Math.min(RW - 1, Math.max(0, Math.round(clamp01(mN.x) * RW)));
        const py = Math.min(RH - 1, Math.max(0, Math.round(clamp01(mN.y) * RH)));
        const fx = r4(clamp01(mN.x)), fy = r4(clamp01(mN.y));
        const rectOf = (b) => [r4(b.x), r4(b.y), r4(b.w), r4(b.h)];
        if (node._editL != null) {
          const eb = node._boxes.find((b) => b.L === node._editL);
          const r = node._layerRasters.get(node._editL);
          if (r && r.alpha[py * RW + px] > INK_ALPHA) {
            pushOp({ t: "R", L: node._editL, c: node._brushColor }, false);
          } else if (eb && mN.x >= eb.x && mN.x <= eb.x + eb.w && mN.y >= eb.y && mN.y <= eb.y + eb.h) {
            pushOp({ t: "f", L: node._editL, c: node._brushColor, x: fx, y: fy, r: rectOf(eb) }, false);
          } else {
            flashHint("Fill: click inside the edited box");
          }
          return;
        }
        const selectLayerBox = (L) => {
          const bi2 = node._boxes.findIndex((b) => b.L === L);
          if (bi2 !== -1) {
            node._activeIdx = bi2;
            const b = node._boxes[bi2];
            b.palette = b.palette || [];
            b.palette[0] = node._brushColor;
            b.palUser = true;
          }
        };
        // 1) directly on a visible drawing: that case takes the color
        for (const b of node._boxes) {
          if (b.L == null || b.hide) continue;
          const r = node._layerRasters.get(b.L);
          if (r && r.count && r.alpha[py * RW + px] > INK_ALPHA) {
            selectLayerBox(b.L);
            pushOp({ t: "R", L: b.L, c: node._brushColor }, false);
            return;
          }
        }
        // 2) inside a box: bounded fill (drawn) or painted rectangle (plain)
        const bi = boxIndexAt(mN);
        if (bi !== -1) {
          const b = node._boxes[bi];
          node._activeIdx = bi;
          b.palette = b.palette || [];
          b.palette[0] = node._brushColor;
          b.palUser = true;
          if (b.L != null) {
            pushOp({ t: "f", L: b.L, c: node._brushColor, x: fx, y: fy, r: rectOf(b) }, false);
          } else {
            b.fill = true;
            commit(); fitNode();
          }
          return;
        }
        // 3) the drawing background is not fillable
        flashHint("Fill: click inside a box — the background is not fillable");
      }
      // Eyedropper: sample the composited canvas at the click point.
      function doPick(mN) {
        try {
          const px = Math.min(canvasEl.width - 1, Math.max(0, Math.round(mN.x * canvasEl.width)));
          const py = Math.min(canvasEl.height - 1, Math.max(0, Math.round(mN.y * canvasEl.height)));
          const d = ctx.getImageData(px, py, 1, 1).data;
          const hex = "#" + (((d[0] << 16) | (d[1] << 8) | d[2]) >>> 0).toString(16).padStart(6, "0");
          node._brushColor = hex;
          node.properties.ideoBrushColor = hex;
          colorInp.value = hex;
          flashHint("Picked " + hex);
        } catch (err) {
          flashHint("Could not sample the canvas color");
        }
        setTool(node._prevTool || "brush");
      }
      let _hintTimer = null;
      function flashHint(msg) {
        hint.textContent = msg;
        if (_hintTimer) clearTimeout(_hintTimer);
        _hintTimer = setTimeout(() => { _hintTimer = null; renderPanel(); }, 1600);
      }

      // ── polyline tool (Paint-style polygon): click to chain segments ──
      const POLY_SNAP = 8;                  // display px: click this close to the start to close the shape
      function polySnapPoint(mN, shift) {
        // Shift snaps the new segment to 45° steps from the last vertex.
        if (!node._poly || !shift) return { x: clamp01(mN.x), y: clamp01(mN.y) };
        const last = node._poly.pts[node._poly.pts.length - 1];
        const W = logW(), H = logH();
        const dx = (mN.x - last[0]) * W, dy = (mN.y - last[1]) * H;
        const ang = Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) * (Math.PI / 4);
        const len = Math.hypot(dx, dy);
        return { x: clamp01(last[0] + Math.cos(ang) * len / W), y: clamp01(last[1] + Math.sin(ang) * len / H) };
      }
      function polyNearStart(pt) {
        if (!node._poly || node._poly.pts.length < 3) return false;
        const s = node._poly.pts[0];
        return Math.hypot((pt.x - s[0]) * logW(), (pt.y - s[1]) * logH()) <= POLY_SNAP;
      }
      function finishPoly() {
        const p = node._poly;
        node._poly = null;
        node._polyCursor = null;
        if (p && p.pts.length >= 2) pushOp(p, true);
        else drawCanvas();
      }
      function cancelPoly() {
        node._poly = null;
        node._polyCursor = null;
        drawCanvas();
      }

      // ── rect / ellipse shape points ──
      function shapePts(tool, a, b, constrain) {
        const W = logW(), H = logH();
        if (tool === "line") {
          let bx = b.x, by = b.y;
          if (constrain) {                                   // snap to 45° steps (in display px space)
            const dx = (b.x - a.x) * W, dy = (b.y - a.y) * H;
            const ang = Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) * (Math.PI / 4);
            const len = Math.hypot(dx, dy);
            bx = a.x + Math.cos(ang) * len / W;
            by = a.y + Math.sin(ang) * len / H;
          }
          return [[r4(clamp01(a.x)), r4(clamp01(a.y))], [r4(clamp01(bx)), r4(clamp01(by))]];
        }
        let x1 = a.x, y1 = a.y, x2 = b.x, y2 = b.y;
        if (constrain) {                                     // square / circle (in display px space)
          const dx = (x2 - x1) * W, dy = (y2 - y1) * H;
          const m = Math.max(Math.abs(dx), Math.abs(dy));
          x2 = x1 + (dx < 0 ? -m : m) / W;
          y2 = y1 + (dy < 0 ? -m : m) / H;
        }
        if (tool === "rect") {
          return [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
            .map(([x, y]) => [r4(clamp01(x)), r4(clamp01(y))]);
        }
        // ellipse — sampled polygon (closed)
        const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
        const rx = Math.abs(x2 - x1) / 2, ry = Math.abs(y2 - y1) / 2;
        const SEG = 48, pts = [];
        for (let i = 0; i <= SEG; i++) {
          const t = (i / SEG) * Math.PI * 2;
          pts.push([r4(clamp01(cx + Math.cos(t) * rx)), r4(clamp01(cy + Math.sin(t) * ry))]);
        }
        return pts;
      }

      // ── drawing ──
      let _rafPending = false;
      function drawCanvas() {
        if (_rafPending) return;
        _rafPending = true;
        requestAnimationFrame(() => {
          _rafPending = false;
          _draw();
        });
      }
      function _draw() {
        // Size the backing store to display × DPR and draw in logical px (crisp text/lines).
        const W = logW(), H = logH(), d = window.devicePixelRatio || 1;
        const bw = Math.round(W * d), bh = Math.round(H * d);
        if (canvasEl.width !== bw || canvasEl.height !== bh) { canvasEl.width = bw; canvasEl.height = bh; }
        ctx.setTransform(d, 0, 0, d, 0, 0);
        ctx.clearRect(0, 0, W, H);
        let bri = bgBrightnessWidget ? bgBrightnessWidget.value : 25;
        if (typeof bri !== "number" || isNaN(bri)) bri = 25;       // guard against unset widget value
        let opa = bgOpacityWidget ? bgOpacityWidget.value : 100;
        if (typeof opa !== "number" || isNaN(opa)) opa = 100;
        const g = Math.round(bri / 100 * 128);                     // blank canvas grey from brightness
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(0, 0, W, H);
        if (node._bgImg) {                                         // background layer, with opacity + brightness
          ctx.globalAlpha = Math.max(0, Math.min(1, opa / 100));
          ctx.drawImage(node._bgImg, 0, 0, W, H);
          ctx.globalAlpha = 1;
          const dim = (1 - bri / 100) * (opa / 100);
          if (dim > 0) { ctx.fillStyle = `rgba(0,0,0,${dim})`; ctx.fillRect(0, 0, W, H); }
        }
        // layered ink composite + live in-progress stroke/shape
        ensureDisplay();
        // keep the side region list exactly as tall as the canvas (it scrolls)
        if (sidePanel.style.height !== H + "px") sidePanel.style.height = H + "px";
        if (node._drawing && node._dragInk && node._boxAtStart) {
          // the dragged region's drawing follows the box live
          ctx.drawImage(node._dragInk.baseC, 0, 0, W, H);
          const b0 = node._boxAtStart, b1 = node._boxes[node._activeIdx] || b0;
          const fx = b0.w > 0 ? b1.w / b0.w : 1, fy = b0.h > 0 ? b1.h / b0.h : 1;
          ctx.save();
          ctx.translate(b1.x * W, b1.y * H);
          ctx.scale(fx, fy);
          ctx.drawImage(node._dragInk.regionC, -b0.x * W, -b0.y * H, W, H);
          ctx.restore();
        } else if (node._draws.length && node._inkCanvas) {
          ctx.drawImage(node._inkCanvas, 0, 0, W, H);
        }
        if (node._stroke && node._stroke.t !== "e") {      // erase shows live in the composite
          const s = node._stroke;
          const lw = Math.max(1, s.s * Math.max(W, H));
          strokePath(ctx, s.pts, lw, W, H, s.c, "source-over");
          ctx.globalCompositeOperation = "source-over";
        }
        if (node._poly) {
          // live polyline: committed segments + a preview segment to the cursor
          const p = node._poly;
          const lw = Math.max(1, p.s * Math.max(W, H));
          const cur = node._polyCursor;
          const near = cur && polyNearStart(cur);
          let pts = cur ? [...p.pts, near ? p.pts[0] : [cur.x, cur.y]] : p.pts;
          if (pts.length === 2 &&
              Math.hypot((pts[1][0] - pts[0][0]) * W, (pts[1][1] - pts[0][1]) * H) < 0.75) {
            pts = [pts[0]];                              // no movement yet → a brush-size dot, not a 0-length line
          }
          strokePath(ctx, pts, lw, W, H, p.c, "source-over");
          ctx.globalCompositeOperation = "source-over";
          ctx.fillStyle = "#fff";
          for (const [vx, vy] of p.pts) ctx.fillRect(vx * W - 2, vy * H - 2, 4, 4);   // vertex markers
          if (p.pts.length >= 3) {                       // snap target on the start point
            ctx.beginPath();
            ctx.arc(p.pts[0][0] * W, p.pts[0][1] * H, POLY_SNAP, 0, Math.PI * 2);
            ctx.strokeStyle = near ? "#46e67a" : "rgba(255,255,255,0.35)";
            ctx.lineWidth = near ? 2 : 1;
            ctx.stroke();
          }
        }
        // brush/eraser cursor ring
        if (node._cursorN && (node._tool === "brush" || node._tool === "erase") && !node._stroke) {
          const lw = Math.max(1, (node._brushSize / 1024) * Math.max(W, H));
          ctx.beginPath();
          ctx.arc(node._cursorN.x * W, node._cursorN.y * H, Math.max(1, lw / 2), 0, Math.PI * 2);
          ctx.strokeStyle = node._tool === "erase" ? "rgba(255,255,255,0.8)" : node._brushColor;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        // active box only when the editor is focused or the node is selected
        const aIdx = (node._focused || node._selected) ? node._activeIdx : -1;
        // index 0 = front (drawn last among non-active) so it matches the layers list (01 on top)
        const order = node._boxes.map((_, i) => i).filter((i) => i !== aIdx).reverse();
        if (aIdx >= 0 && aIdx < node._boxes.length) order.push(aIdx);  // active drawn last (on top)
        const tagR = tagRects();                              // collision-avoided tag positions
        const ranks = new Map();                              // rank among visible boxes (0 = front)
        {
          let k = 0;
          node._boxes.forEach((b2, i2) => { if (!b2.hide) ranks.set(i2, k++); });
        }
        for (const i of order) {
          const b = node._boxes[i], active = i === aIdx;
          if (b.hide) continue;                            // hidden in the scene (side panel eye)
          if (node._editL != null && b.L !== node._editL) continue;   // edit mode: only the edited case
          // depth transparency on the whole case — frame, fill AND text fade with the plane
          ctx.globalAlpha = node._editL != null ? 1 : depthAlpha(ranks.get(i) || 0);
          const pal = (b.palette || []).filter(Boolean);
          const col = pal.length ? pal[0] : "#8c8c8c";       // box color = first palette color, else neutral grey
          const { x1, y1, x2, y2 } = toPx(b);
          const w = x2 - x1, h = y2 - y1;
          const hovered = i === node._hoverBox || active;    // active box stays highlighted (on top)
          if (active && !b.auto) {                           // opaque backing so contents read clearly over boxes behind
            ctx.fillStyle = "rgba(26,26,26,0.88)";
            ctx.fillRect(x1, y1, w, h);
          }
          // auto regions stay light so the ink below them remains visible;
          // bucket-painted boxes get a real 50% fill like a drawn rectangle
          ctx.fillStyle = col + (b.auto ? (hovered ? "20" : "10")
            : b.fill ? (hovered ? "8c" : "80") : (hovered ? "3a" : "22"));
          ctx.fillRect(x1, y1, w, h);
          if (b.nobbox) ctx.setLineDash([6, 4]);             // unplaced (no bbox in source)
          else if (b.auto) ctx.setLineDash([4, 3]);          // auto-outlined (drawing region)
          const lw = active ? 2 : (hovered ? 1.5 : 1);
          ctx.strokeStyle = col; ctx.lineWidth = lw;
          ctx.strokeRect(x1 + lw / 2, y1 + lw / 2, w - lw, h - lw);  // inside the box so strip/badge align at y1
          ctx.setLineDash([]);
          // visible grab handles on the active box (corners + edge midpoints)
          if (active && (node._tool === "select" || node._tool === "boxr")) {
            const hs = 3.5;
            for (const [cx, cy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2],
                                    [(x1 + x2) / 2, y1], [(x1 + x2) / 2, y2],
                                    [x1, (y1 + y2) / 2], [x2, (y1 + y2) / 2]]) {
              ctx.fillStyle = "#fff";
              ctx.fillRect(cx - hs, cy - hs, hs * 2, hs * 2);
              ctx.strokeStyle = col; ctx.lineWidth = 1;
              ctx.strokeRect(cx - hs + 0.5, cy - hs + 0.5, hs * 2 - 1, hs * 2 - 1);
            }
          }
          if (pal.length) {                                  // palette shown as a strip along the top edge
            const sw = w / pal.length, n = pal.length, sh = 7;
            for (let p = 0; p < n; p++) {
              const sx = x1 + Math.round(p * sw);
              ctx.fillStyle = pal[p];
              ctx.fillRect(sx, y1, x1 + Math.round((p + 1) * sw) - sx, sh);
            }
          }
          // in-box content (clipped to the box): prompt text, tag chip on top
          ctx.save();
          ctx.beginPath(); ctx.rect(x1, y1, w, h); ctx.clip();

          let body = b.desc || "";
          if (b.type === "text" && b.text) body = `"${b.text}"` + (body ? " — " + body : "");
          if (body) {
            ctx.font = "12px monospace";
            // colored shade + contrasting halo: readable on the fill, the outline,
            // the drawing — on anything
            const tcol = readableOn(b, col);
            ctx.lineWidth = 3;
            ctx.lineJoin = "round";
            ctx.strokeStyle = tcol.light ? "rgba(0,0,0,0.8)" : "rgba(255,255,255,0.85)";
            ctx.fillStyle = tcol.css;
            const pad = 4, lh2 = 14;
            let ty = y1 + 15 + 12;                        // first line below the tag chip
            for (const line of wrapLines(body, w - pad * 2)) {
              if (ty > y1 + h - 3) break;                  // never pass the bottom edge
              ctx.strokeText(line, x1 + pad, ty);
              ctx.fillText(line, x1 + pad, ty);
              ty += lh2;
            }
          }
          // tag chip on top, at its collision-avoided position
          const tr = tagR[i];
          ctx.font = "bold 11px monospace";
          ctx.fillStyle = col;                                // tag chip = box color
          ctx.fillRect(tr.x, tr.y, tr.w, 14);
          if (i === node._hoverTitle) {                       // hover highlight
            ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.fillRect(tr.x, tr.y, tr.w, 14);
            ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.strokeRect(tr.x + 0.5, tr.y + 0.5, tr.w - 1, 13);
          }
          ctx.fillStyle = textOn(col);
          ctx.fillText(tr.tag, tr.x + 4, tr.y + 11);
          ctx.restore();
          ctx.globalAlpha = 1;
        }
        // \u2713 / \u2715 chips above the edited case
        node._editChips = null;
        if (node._editL != null) {
          const eb = node._boxes.find((b2) => b2.L === node._editL);
          if (eb) {
            const px = toPx(eb);
            const cw = 20, chh = 16, gap = 4;
            let cy = px.y1 - chh - 4;
            if (cy < 2) cy = px.y1 + 4;
            const cx = Math.min(W - 2 * cw - gap - 2, Math.max(2, px.x2 - 2 * cw - gap));
            const ok = { x: cx, y: cy, w: cw, h: chh };
            const no = { x: cx + cw + gap, y: cy, w: cw, h: chh };
            for (const [r2, bg2, glyph] of [[ok, "#2f8f4e", "\u2713"], [no, "#a33636", "\u2715"]]) {
              ctx.fillStyle = bg2;
              ctx.fillRect(r2.x, r2.y, r2.w, r2.h);
              ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
              ctx.strokeRect(r2.x + 0.5, r2.y + 0.5, r2.w - 1, r2.h - 1);
              ctx.fillStyle = "#fff";
              ctx.font = "bold 12px sans-serif";
              ctx.fillText(glyph, r2.x + 5, r2.y + 12);
            }
            node._editChips = { ok, no };
          }
        }
      }

      // ── serialization ──
      function serialize() {
        if (elementsWidget) elementsWidget.value = node._boxes.length ? JSON.stringify(node._boxes) : "";
        if (stylePaletteWidget) stylePaletteWidget.value = node._stylePalette.length ? JSON.stringify(node._stylePalette) : "";
        if (drawDataWidget) drawDataWidget.value = node._draws.length ? JSON.stringify(node._draws) : "";
      }

      function commit() { serialize(); renderPanel(); rebuildSideList(); drawCanvas(); updateTokens(); }
      // Live text edit: persist + repaint + token count, without rebuilding the panel.
      function touch() { serialize(); rebuildSideList(); drawCanvas(); updateTokens(); }

      // ── side region list rows ──
      function rebuildSideList() {
        const collapsed = !!node.properties.ideoSideCollapsed;
        sidePanel.classList.toggle("collapsed", collapsed);
        sideHdrLabel.style.display = collapsed ? "none" : "";
        eyeAllBtn.style.display = collapsed ? "none" : "";
        collBtn.textContent = collapsed ? "≡" : "⮞";
        collBtn.title = collapsed ? "Open the regions panel" : "Collapse the regions panel";
        sideRows.style.display = collapsed ? "none" : "";
        if (collapsed) return;
        sideRows.innerHTML = "";
        if (!node._boxes.length) {
          const empty = document.createElement("div");
          empty.style.cssText = "color:#777;font-style:italic;padding:2px 4px;";
          empty.textContent = "No regions yet.";
          sideRows.appendChild(empty);
          return;
        }
        node._boxes.forEach((b, i) => {
          const row = document.createElement("div");
          row.className = "kjid2-srow" + (i === node._activeIdx ? " active" : "") + (b.hide ? " hidden-r" : "");
          row._box = b;
          const eye = document.createElement("button");
          eye.className = "kjid2-eye";
          eye.textContent = b.hide ? "🚫" : "👁";
          eye.title = b.hide ? "Hidden — excluded from the prompt (click to restore)"
            : "Visible — click to hide it (excluded from the prompt while hidden)";
          const sw = document.createElement("div");
          sw.className = "kjid2-ssw";
          sw.style.background = (b.palette || []).find(Boolean) || "#8c8c8c";
          sw.title = "Click to pick this region's exact color";
          const cinp = document.createElement("input");
          cinp.type = "color";
          const curCol = (b.palette || []).find(Boolean);
          if (curCol && /^#[0-9a-fA-F]{6}$/.test(curCol)) cinp.value = curCol;
          sw.appendChild(cinp);
          sw.addEventListener("mousedown", (e) => { e.stopPropagation(); });
          sw.addEventListener("click", (e) => { e.stopPropagation(); cinp.click(); });
          cinp.addEventListener("input", () => {           // live color preview
            b.palette = b.palette || [];
            b.palette[0] = cinp.value;
            b.palUser = true;
            sw.style.background = cinp.value;
            serialize(); drawCanvas(); updateTokens();
          });
          cinp.addEventListener("change", () => {          // final: the drawing follows
            if (b.L != null) pushOp({ t: "R", L: b.L, c: cinp.value }, false);
            else commit();
          });
          const num = document.createElement("span");
          num.className = "kjid2-snum";
          num.textContent = String(i + 1).padStart(2, "0");
          const txt = document.createElement("span");
          const label = rowLabel(b);
          txt.className = "kjid2-stxt" + (label ? "" : " empty");
          txt.textContent = label ? ((b.auto ? "✏ " : "") + label)
            : (b.auto ? "✏ (drawn)" : (b.type === "text" ? "(text)" : "(empty)"));
          txt.title = (label ? label + " — " : "") + "drag to reorder (top = front)";
          const del = document.createElement("button");
          del.className = "kjid2-eye kjid2-sdel";
          del.textContent = "✕";
          del.title = "Delete this region (and its drawing)";
          del.addEventListener("mousedown", (e) => e.stopPropagation());
          del.addEventListener("click", (e) => {
            e.stopPropagation();
            const idx = node._boxes.indexOf(b);
            if (idx !== -1) removeBoxSmart(idx);
          });
          row.append(eye, sw, num, txt, del);
          sideRows.appendChild(row);
          eye.addEventListener("click", (e) => {
            e.stopPropagation();
            if (b.hide) delete b.hide; else b.hide = true;
            commit();
          });
          row.addEventListener("click", () => {
            if (row._dragged) { row._dragged = false; return; }
            if (node._editL != null && node._editL !== b.L) finishEdit(true);
            node._activeIdx = node._boxes.indexOf(b);
            commit();
          });
          // drag-reorder (vertical FLIP) — top row = front-most, like layer stacks
          row.addEventListener("mousedown", (e) => {
            if (e.button !== 0 || e.target === eye) return;
            e.preventDefault(); e.stopPropagation();
            const sx0 = e.clientX, sy0 = e.clientY;
            let dragging = false;
            const move = (me) => {
              if (!dragging) {
                if (Math.abs(me.clientX - sx0) + Math.abs(me.clientY - sy0) < 4) return;
                dragging = true; row.classList.add("dragging"); document.body.classList.add("kjid2-dragging");
              }
              for (const other of sideRows.querySelectorAll(".kjid2-srow")) {
                if (other === row) continue;
                const r = other.getBoundingClientRect();
                if (me.clientY >= r.top && me.clientY <= r.bottom) {
                  const ref = me.clientY > r.top + r.height / 2 ? other.nextSibling : other;
                  if (ref === row || ref === row.nextSibling) break;
                  const els = Array.from(sideRows.querySelectorAll(".kjid2-srow"));
                  const prev = els.map((el) => el.getBoundingClientRect().top);
                  sideRows.insertBefore(row, ref);
                  els.forEach((el, k) => {                        // FLIP: slide to new positions
                    const dy = prev[k] - el.getBoundingClientRect().top;
                    if (!dy) return;
                    el.style.transition = "none";
                    el.style.transform = `translateY(${dy}px)`;
                    el.getBoundingClientRect();                   // flush
                    el.style.transition = ""; el.style.transform = "";
                  });
                  // apply the new plane order LIVE: ink + frames + text re-fade now
                  const orderNow = Array.from(sideRows.querySelectorAll(".kjid2-srow")).map((el) => el._box);
                  if (orderNow.length === node._boxes.length) {
                    const act = node._boxes[node._activeIdx];
                    node._boxes = orderNow;
                    node._activeIdx = act ? node._boxes.indexOf(act) : -1;
                    drawCanvas();
                  }
                  break;
                }
              }
            };
            const up = () => {
              document.removeEventListener("mousemove", move);
              document.removeEventListener("mouseup", up);
              document.body.classList.remove("kjid2-dragging");
              if (dragging) {
                row.classList.remove("dragging");
                row._dragged = true;                             // suppress the trailing click
                const active = node._boxes[node._activeIdx];
                const order = Array.from(sideRows.querySelectorAll(".kjid2-srow")).map((el) => el._box);
                if (order.length === node._boxes.length) node._boxes = order;
                node._activeIdx = active ? node._boxes.indexOf(active) : -1;
                commit();
              }
            };
            document.addEventListener("mousemove", move);
            document.addEventListener("mouseup", up);
          });
        });
      }

      function removeBox(i) {
        node._boxes.splice(i, 1);
        if (node._boxes.length === 0) node._activeIdx = -1;
        else if (i <= node._activeIdx) node._activeIdx = Math.max(0, node._activeIdx - 1);
      }
      // Live drag layers: the box's own layer + the composite of all the others.
      function prepDragInk(hb) {
        node._dragInk = null;
        if (hb.auto && hb.L != null) {
          ensureDisplay();
          const r = node._layerRasters.get(hb.L);
          if (!r || !r.count) return;
          const { RW, RH } = node._inkSize;
          node._dragInk = { baseC: buildComposite(hb.L), regionC: renderLayer(hb.L, RW, RH, null) };
        }
      }
      // ── edit mode: add to / fix a region's drawing, then ✓ keep or ✕ revert ──
      function startEdit(b) {
        if (b.L == null) return;
        if (node._editL != null) finishEdit(true);
        node._editL = b.L;
        node._editStart = node._draws.length;
        node._activeIdx = node._boxes.indexOf(b);
        // entering edit mode arms a DRAWING tool — the eraser stays its own button
        if (!["brush", "line", "rect", "ellipse", "fill"].includes(node._tool)) setTool("brush");
        displayKey = "";
        commit(); fitNode();
      }
      function finishEdit(keep) {
        if (node._editL == null) return;
        finishPoly();                          // a pending polyline belongs to the session
        if (!keep && node._draws.length > node._editStart) {
          node._draws.length = node._editStart;
          node._redo = [];
        }
        node._editL = null;
        opsChanged(false);
      }
      // Delete a region. Drawn regions erase their whole layer (the sync then
      // drops the box); plain boxes are removed directly.
      function removeBoxSmart(i) {
        const b = node._boxes[i];
        if (!b) return;
        if (b.auto && b.L != null) {
          if (node._editL === b.L) finishEdit(true);
          pushOp({ t: "E", L: b.L }, false);
          return;
        }
        removeBox(i);
        commit(); fitNode();
      }

      // ── pointer interaction ──
      canvasEl.addEventListener("mousedown", (e) => {
        if (node._placing) {             // drop the duplicate being placed
          if (e.button === 0) { placeFollower(mouseN(e)); finishPlacing(); }
          else cancelPlacing();
          e.preventDefault(); e.stopPropagation();
          return;
        }
        if (e.button !== 0) return;
        canvasEl.focus();                // so Delete/Backspace targets this editor
        node._hoverTitle = null; node._hoverBox = null;  // clear hover highlight while interacting
        const mN = mouseN(e);

        // ✓ / ✕ chips of the edit mode (any tool)
        if (node._editL != null && node._editChips) {
          const cpx = mN.x * logW(), cpy = mN.y * logH();
          const hitChip = (r) => r && cpx >= r.x && cpx <= r.x + r.w && cpy >= r.y && cpy <= r.y + r.h;
          if (hitChip(node._editChips.ok)) { finishEdit(true); e.preventDefault(); e.stopPropagation(); return; }
          if (hitChip(node._editChips.no)) { finishEdit(false); e.preventDefault(); e.stopPropagation(); return; }
        }

        // ink tools: brush / eraser
        if (node._tool === "brush" || node._tool === "erase") {
          node._stroke = node._tool === "brush"
            ? { t: "s", L: gestureLayer(), c: node._brushColor, s: node._brushSize / 1024, pts: [[r4(clamp01(mN.x)), r4(clamp01(mN.y))]] }
            : { t: "e", L: eraseLayer(), s: node._brushSize / 1024, pts: [[r4(clamp01(mN.x)), r4(clamp01(mN.y))]] };
          node._strokeLastPx = { x: mN.x * logW(), y: mN.y * logH() };
          document.addEventListener("mousemove", onInkMove);
          document.addEventListener("mouseup", onInkUp);
          e.preventDefault(); e.stopPropagation();
          drawCanvas();
          return;
        }
        // polyline tool: click to chain segments, click near the start to close
        if (node._tool === "line") {
          const pt = polySnapPoint(mN, e.shiftKey);
          if (!node._poly) {
            node._poly = { t: "s", L: gestureLayer(), c: node._brushColor, s: node._brushSize / 1024,
                           pts: [[r4(pt.x), r4(pt.y)]] };
          } else if (polyNearStart(pt)) {
            node._poly.pts.push([node._poly.pts[0][0], node._poly.pts[0][1]]);   // snap-close the shape
            finishPoly();
            e.preventDefault(); e.stopPropagation();
            return;
          } else {
            const last = node._poly.pts[node._poly.pts.length - 1];
            if (Math.hypot((pt.x - last[0]) * logW(), (pt.y - last[1]) * logH()) >= 1.5) {
              node._poly.pts.push([r4(pt.x), r4(pt.y)]);
            }
          }
          node._polyCursor = pt;
          e.preventDefault(); e.stopPropagation();
          drawCanvas();
          return;
        }
        // shape tools: rect / ellipse (drag)
        if (node._tool === "rect" || node._tool === "ellipse") {
          shapeStart = { x: clamp01(mN.x), y: clamp01(mN.y) };
          node._stroke = { t: "s", L: gestureLayer(), c: node._brushColor, s: node._brushSize / 1024,
                           pts: shapePts(node._tool, shapeStart, shapeStart, false) };
          document.addEventListener("mousemove", onShapeMove);
          document.addEventListener("mouseup", onShapeUp);
          e.preventDefault(); e.stopPropagation();
          drawCanvas();
          return;
        }
        if (node._tool === "fill") {
          doFill(mN);
          e.preventDefault(); e.stopPropagation();
          return;
        }
        if (node._tool === "pick") {
          doPick(mN);
          e.preventDefault(); e.stopPropagation();
          return;
        }

        // select / region-box tools (parked while editing a drawing)
        if (node._editL != null) { e.preventDefault(); e.stopPropagation(); return; }
        // clicking a number chip brings that region to the front, instantly
        const ti0 = titleAt(mN);
        if (ti0 != null) {
          const [fb] = node._boxes.splice(ti0, 1);
          node._boxes.unshift(fb);
          node._activeIdx = 0;
          node._dragMode = "move";
          node._boxAtStart = { ...fb };
          prepDragInk(fb);
          node._drawing = true;
          node._dragStartN = mN;
          document.addEventListener("mousemove", onMove);
          document.addEventListener("mouseup", onUp);
          serialize(); renderPanel(); rebuildSideList(); updateTokens();
          e.preventDefault(); e.stopPropagation();
          drawCanvas();
          return;
        }
        const hit = pickForSelection(mN, e.altKey);
        if (hit) {
          node._activeIdx = hit.index;
          node._dragMode = hit.mode;
          const hb = node._boxes[hit.index];
          node._boxAtStart = { ...hb };
          prepDragInk(hb);                       // an auto region carries its drawing along
        } else if (node._tool === "boxr") {
          node._dragMode = "draw";
          const nb = { x: mN.x, y: mN.y, w: 0, h: 0, type: "obj", text: "", desc: "", palette: [] };
          node._boxes.push(nb);
          node._activeIdx = node._boxes.length - 1;
          node._boxAtStart = { ...nb };
        } else {                          // select tool on empty space: just deselect
          node._activeIdx = -1;
          renderPanel(); drawCanvas();
          e.preventDefault(); e.stopPropagation();
          return;
        }
        node._drawing = true;
        node._dragStartN = mN;
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        e.preventDefault(); e.stopPropagation();
        drawCanvas();   // panel rebuild/resize deferred to onUp so the canvas doesn't shift mid-drag
      });

      canvasEl.addEventListener("mousemove", (e) => {
        if (node._placing) { placeFollower(mouseN(e)); return; }
        const mN = mouseN(e);
        if (node._tool === "brush" || node._tool === "erase") {
          node._cursorN = mN;
          if (!node._stroke) drawCanvas();      // live stroke redraw handled by onInkMove
          canvasEl.style.cursor = "none";       // the ring replaces the cursor
          return;
        }
        node._cursorN = null;
        if (node._tool === "line") {
          node._polyCursor = node._poly ? polySnapPoint(mN, e.shiftKey) : null;
          canvasEl.style.cursor = "crosshair";
          if (node._poly) drawCanvas();
          return;
        }
        if (node._tool === "rect" || node._tool === "ellipse" ||
            node._tool === "fill" || node._tool === "pick") {
          canvasEl.style.cursor = "crosshair";
          return;
        }
        if (node._drawing) return;
        if (node._editL != null) { canvasEl.style.cursor = "crosshair"; return; }
        const ti = titleAt(mN);
        const hit = hitTest(mN);
        const hb = ti != null ? ti : (hit ? hit.index : null);
        if (ti !== node._hoverTitle || hb !== node._hoverBox) {
          node._hoverTitle = ti; node._hoverBox = hb; drawCanvas();
        }
        canvasEl.style.cursor = ti != null ? "pointer" : (hit ? (cursorForBboxMode(hit.mode) || "crosshair") : "crosshair");
      });
      canvasEl.addEventListener("mouseleave", () => {
        const hadCursor = node._cursorN !== null || node._polyCursor !== null;
        node._cursorN = null;
        node._polyCursor = null;
        if (node._hoverTitle !== null || node._hoverBox !== null || hadCursor) {
          node._hoverTitle = null; node._hoverBox = null; drawCanvas();
        }
      });

      // ── live sync: rasters + bboxes follow the stroke IN REAL TIME ──
      let _liveTick = 0;
      function liveInkSync(force) {
        const s = node._stroke;
        if (!s) return;
        const now = performance.now();
        if (!force && now - _liveTick < 120) return;
        _liveTick = now;
        node._draws.push(s);                               // temp op: current stroke so far
        opsRev++;
        rebuildInk();
        inkKey = node._inkSize.RW + "x" + node._inkSize.RH + ":" + opsRev;   // keep the temp rasters live
        displayKey = "";
        node._draws.pop();
        const before = node._boxes.length;
        syncAutoBoxes();
        if (node._boxes.length !== before) { renderPanel(); rebuildSideList(); fitNode(); }
        drawCanvas();
      }
      // ── freehand stroke recording ──
      function onInkMove(e) {
        if (!node._stroke) return;
        const mN = mouseN(e);
        node._cursorN = mN;
        const px = { x: mN.x * logW(), y: mN.y * logH() };
        const last = node._strokeLastPx;
        if (last && Math.hypot(px.x - last.x, px.y - last.y) < 1.5) return;   // decimate
        node._strokeLastPx = px;
        node._stroke.pts.push([r4(clamp01(mN.x)), r4(clamp01(mN.y))]);
        liveInkSync(false);
        drawCanvas();
      }
      function onInkUp() {
        document.removeEventListener("mousemove", onInkMove);
        document.removeEventListener("mouseup", onInkUp);
        const s = node._stroke;
        node._stroke = null;
        node._strokeLastPx = null;
        if (!s || !s.pts.length) { drawCanvas(); return; }
        pushOp(s, s.t === "s");           // a new brush shape selects its region
      }

      // ── shape recording (line / rect / ellipse) ──
      function onShapeMove(e) {
        if (!node._stroke || !shapeStart) return;
        const mN = mouseN(e);
        node._stroke.pts = shapePts(node._tool, shapeStart, { x: clamp01(mN.x), y: clamp01(mN.y) }, e.shiftKey);
        liveInkSync(false);
        drawCanvas();
      }
      function onShapeUp() {
        document.removeEventListener("mousemove", onShapeMove);
        document.removeEventListener("mouseup", onShapeUp);
        const s = node._stroke;
        node._stroke = null;
        shapeStart = null;
        if (!s) { drawCanvas(); return; }
        let minx = 1, maxx = 0, miny = 1, maxy = 0;
        for (const [x, y] of s.pts) {
          if (x < minx) minx = x;
          if (x > maxx) maxx = x;
          if (y < miny) miny = y;
          if (y > maxy) maxy = y;
        }
        // drop degenerate shapes (accidental click)
        if ((maxx - minx) * logW() < 3 && (maxy - miny) * logH() < 3) { drawCanvas(); return; }
        pushOp(s, true);
      }

      // ── inline description editing (double-click a region) ──
      let inlineTa = null;
      function closeInlineEditor() {
        if (inlineTa) { inlineTa.remove(); inlineTa = null; }
      }
      function openInlineEditor(idx) {
        closeInlineEditor();
        const b = node._boxes[idx];
        if (!b) return;
        node._activeIdx = idx;
        const dw = canvasEl.offsetWidth, dh = canvasEl.offsetHeight;       // CSS display size
        const ox = canvasEl.offsetLeft, oy = canvasEl.offsetTop;
        const w = Math.min(dw, Math.max(70, b.w * dw));
        const h = Math.min(dh, Math.max(42, b.h * dh));
        // clamp so the editor stays inside the canvas (wrapper is overflow:hidden)
        const left = Math.max(ox, Math.min(ox + b.x * dw, ox + dw - w));
        const top = Math.max(oy, Math.min(oy + b.y * dh, oy + dh - h));
        const ta = document.createElement("textarea");
        ta.className = "kjid2-inline";
        ta.value = b.desc || "";
        ta.style.left = left + "px";
        ta.style.top = top + "px";
        ta.style.width = w + "px";
        ta.style.height = h + "px";
        ta.style.borderColor = (b.palette || []).find(Boolean) || "#46b4e6";  // first palette color, else accent
        stopProp(ta);
        wrap.appendChild(ta);
        inlineTa = ta;
        ta.focus(); ta.select();
        const orig = b.desc || "";
        let cancelled = false;
        ta.addEventListener("input", () => { b.desc = ta.value; drawCanvas(); updateTokens(); });
        ta.addEventListener("keydown", (e) => {
          e.stopPropagation();
          if (e.key === "Escape") { cancelled = true; b.desc = orig; ta.blur(); }
          else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ta.blur();
        });
        ta.addEventListener("blur", () => {
          if (!cancelled) b.desc = ta.value;
          closeInlineEditor();
          commit();
        });
      }
      canvasEl.addEventListener("dblclick", (e) => {
        if (node._tool === "line") {                   // double-click ends the polyline (like Paint)
          e.preventDefault(); e.stopPropagation();
          finishPoly();
          return;
        }
        if (node._tool !== "select" && node._tool !== "boxr") return;
        if (node._editL != null) return;
        e.preventDefault(); e.stopPropagation();
        const cands = boxesAt(mouseN(e));     // edit the active box if it's under the cursor, else topmost
        const target = cands.find((c) => c.index === node._activeIdx) || cands[0];
        if (target) openInlineEditor(target.index);
      });

      // Transfer desc/type/text/palette from a prototype box onto the auto region
      // that a just-pushed ink-copy op created.
      function adoptCreated(proto) {
        const created = node._lastCreated || [];
        const nb = created[created.length - 1];
        if (!nb) return;
        nb.type = proto.type === "text" ? "text" : "obj";
        nb.text = proto.text || "";
        nb.desc = proto.desc || "";
        if (Array.isArray(proto.palette) && proto.palette.length) {
          nb.palette = proto.palette.slice();
          if (proto.palUser) nb.palUser = true;
        }
        node._activeIdx = node._boxes.indexOf(nb);
      }
      // Paste a clone of the clipboard box, offset slightly and clamped into the canvas.
      // Pasting a drawn region also duplicates its drawing (ink copy op).
      function pasteBox() {
        if (!copiedBox) return;
        const nb = JSON.parse(JSON.stringify(copiedBox));
        nb.x = Math.max(0, Math.min(clamp01(nb.x + 0.03), 1 - nb.w));
        nb.y = Math.max(0, Math.min(clamp01(nb.y + 0.03), 1 - nb.h));
        delete nb.nobbox;                              // a pasted box is placed
        delete nb.hide;
        if (nb.auto && nb.L != null && nb.src) {
          node._draws.push({ t: "C", L: nb.L, N: newLayerId(), a: rect4(nb.src), b: rect4(nb) });
          node._redo.length = 0;
          opsChanged(true);
          adoptCreated(nb);
          commit(); fitNode();
          return;
        }
        delete nb.auto; delete nb.src; delete nb.L;
        node._boxes.push(nb);
        node._activeIdx = node._boxes.length - 1;
        commit(); fitNode();
      }
      // Keyboard: Delete removes; Ctrl/Cmd C/V/D copy/paste/duplicate the active region;
      // Ctrl+Z/Y undo/redo ink; V/B/L/R/O/G/E/I/X switch tools.
      // Canvas must be focused; stop the event so LiteGraph doesn't act on the node.
      canvasEl.addEventListener("keydown", (e) => {
        if (node._placing) {
          if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancelPlacing(); }
          return;
        }
        if (node._poly) {                              // polyline in progress: Enter ends, Esc cancels
          if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancelPoly(); }
          else if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); finishPoly(); }
          return;
        }
        if (node._editL != null && (e.key === "Enter" || e.key === "Escape")) {
          e.preventDefault(); e.stopPropagation();
          finishEdit(e.key === "Enter");       // Enter keeps, Escape reverts the session
          return;
        }
        if (node._drawing || node._stroke) return;
        const ctrl = e.ctrlKey || e.metaKey;
        const k = (e.key || "").toLowerCase();
        if (ctrl && k === "z" && !e.shiftKey) {
          e.preventDefault(); e.stopPropagation();
          undoOp();
        } else if (ctrl && (k === "y" || (k === "z" && e.shiftKey))) {
          e.preventDefault(); e.stopPropagation();
          redoOp();
        } else if ((e.key === "Delete" || e.key === "Backspace") && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          removeBoxSmart(node._activeIdx);
        } else if (ctrl && k === "c" && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          copiedBox = JSON.parse(JSON.stringify(node._boxes[node._activeIdx]));
        } else if (ctrl && k === "v" && copiedBox) {
          e.preventDefault(); e.stopPropagation();
          pasteBox();
        } else if (ctrl && k === "d" && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          copiedBox = JSON.parse(JSON.stringify(node._boxes[node._activeIdx]));
          pasteBox();
        } else if (!ctrl && !e.altKey) {
          const tool = { v: "select", b: "brush", l: "line", r: "rect", o: "ellipse",
                         g: "fill", e: "erase", i: "pick", x: "boxr" }[k];
          if (tool) { e.preventDefault(); e.stopPropagation(); setTool(tool); }
        }
      });

      function onMove(e) {
        if (!node._drawing) return;
        const mN = mouseN(e);
        const dN = { x: mN.x - node._dragStartN.x, y: mN.y - node._dragStartN.y };
        const nb = applyDrag(node._dragMode, node._boxAtStart, dN);
        delete nb.nobbox;            // moving/resizing places the element (gives it a bbox)
        node._boxes[node._activeIdx] = nb;
        drawCanvas();
      }
      function onUp() {
        if (!node._drawing) return;
        node._drawing = false;
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        // drop zero-size boxes created by an accidental click
        const b = node._boxes[node._activeIdx];
        if (b && (b.w < 0.005 || b.h < 0.005) && node._dragMode === "draw") {
          removeBox(node._activeIdx);
        }
        // an auto region drags its drawing along: commit the ink move/scale
        if (node._dragInk) {
          node._dragInk = null;
          const b0 = node._boxAtStart;
          if (b && b.auto && b.L != null && b0 &&
              (Math.abs(b.x - b0.x) > 1e-4 || Math.abs(b.y - b0.y) > 1e-4 ||
               Math.abs(b.w - b0.w) > 1e-4 || Math.abs(b.h - b0.h) > 1e-4)) {
            pushOp({ t: "M", L: b.L, a: rect4(b0), b: rect4(b) }, false);
            return;                          // pushOp re-syncs and commits
          }
        }
        commit();
      }

      // ── duplicate placement: the new box follows the cursor until clicked ──
      function placeFollower(mN) {
        const b = node._boxes[node._activeIdx];
        if (!b) return;
        b.x = clamp01(Math.min(mN.x - b.w / 2, 1 - b.w));
        b.y = clamp01(Math.min(mN.y - b.h / 2, 1 - b.h));
        delete b.nobbox;
        drawCanvas();
      }
      function startPlacing(srcIdx) {
        const src = node._boxes[srcIdx];
        if (!src) return;
        const nb = JSON.parse(JSON.stringify(src));
        delete nb.nobbox;
        delete nb.hide;
        // duplicating a drawn region also duplicates its drawing once placed
        node._placeProto = (nb.auto && nb.L != null && nb.src) ? { srcRect: { ...nb.src }, proto: nb } : null;
        delete nb.auto; delete nb.src; delete nb.L;    // the follower is a plain box while placing
        node._boxes.push(nb);
        node._activeIdx = node._boxes.length - 1;
        node._placing = true;
        canvasEl.focus();
        canvasEl.style.cursor = "move";
        serialize(); renderPanel(); rebuildSideList(); drawCanvas(); updateTokens();
      }
      function finishPlacing() {
        if (!node._placing) return;
        node._placing = false;
        canvasEl.style.cursor = "crosshair";
        if (node._placeProto) {
          const pp = node._placeProto;
          node._placeProto = null;
          const temp = node._boxes[node._activeIdx];
          if (temp) {
            const dest = { x: temp.x, y: temp.y, w: temp.w, h: temp.h };
            removeBox(node._activeIdx);
            node._draws.push({ t: "C", L: pp.proto.L, N: newLayerId(), a: rect4(pp.srcRect), b: rect4(dest) });
            node._redo.length = 0;
            opsChanged(true);
            adoptCreated(pp.proto);
          }
        }
        commit(); fitNode();
      }
      function cancelPlacing() {
        if (!node._placing) return;
        node._placing = false;
        node._placeProto = null;
        canvasEl.style.cursor = "crosshair";
        removeBox(node._activeIdx);
        commit(); fitNode();
      }

      // ── right-click "layers" menu: list / select / delete / duplicate / reorder regions ──
      function closeLayersMenu() {
        if (node._layerMenu) { node._layerMenu.remove(); node._layerMenu = null; }
        if (node._layerMenuDismiss) {
          document.removeEventListener("mousedown", node._layerMenuDismiss, true);
          document.removeEventListener("pointerdown", node._layerMenuDismiss, true);
          node._layerMenuDismiss = null;
        }
      }
      function rowLabel(b) {
        if (b.type === "text") {
          const t = b.text ? `"${b.text}"` : "";
          return b.desc ? (t ? t + " — " + b.desc : b.desc) : t;
        }
        return b.desc || "";
      }
      function openLayersMenu(clientX, clientY) {
        closeLayersMenu();
        const menu = document.createElement("div");
        menu.className = "kjid2-menu";
        const hdr = document.createElement("div");
        hdr.className = "kjid2-mhdr";
        hdr.textContent = "Regions — top = front · click select · drag reorder";
        // 01 at the top of the list = front-most (drawn on top); see _draw / boxesAt.
        menu.appendChild(hdr);
        const list = document.createElement("div");
        menu.appendChild(list);
        node._layerMenu = menu;

        const renumber = () => Array.from(list.querySelectorAll(".kjid2-lrow")).forEach((row, k) => {
          row.querySelector(".kjid2-lnum").textContent = String(k + 1).padStart(2, "0");
        });
        function buildRows() {
          list.innerHTML = "";
          if (!node._boxes.length) {
            const empty = document.createElement("div");
            empty.className = "kjid2-mhdr"; empty.textContent = "No regions yet.";
            list.appendChild(empty);
            return;
          }
          node._boxes.forEach((b, i) => {
            const row = document.createElement("div");
            row.className = "kjid2-lrow" + (i === node._activeIdx ? " active" : "");
            row._box = b;
            const sw = document.createElement("div");
            sw.className = "kjid2-lsw";
            sw.style.background = (b.palette || []).find(Boolean) || "#8c8c8c";
            const num = document.createElement("span");
            num.className = "kjid2-lnum"; num.textContent = String(i + 1).padStart(2, "0");
            const txt = document.createElement("span");
            const label = rowLabel(b);
            txt.className = "kjid2-ltext" + (label ? "" : " empty");
            txt.textContent = label ? ((b.auto ? "✏ " : "") + label)
              : (b.auto ? "✏ (drawn)" : (b.type === "text" ? "(text)" : "(empty)"));
            txt.title = label;
            const dup = document.createElement("button");
            dup.className = "kjid2-lbtn"; dup.textContent = "⧉";
            dup.title = "Duplicate, then click on the canvas to place";
            const del = document.createElement("button");
            del.className = "kjid2-lbtn del"; del.textContent = "✕";
            del.title = b.auto ? "Delete region (erases its ink)" : "Delete region";
            row.append(sw, num, txt, dup, del);
            list.appendChild(row);

            row.addEventListener("click", () => {
              if (row._dragged) { row._dragged = false; return; }
              node._activeIdx = node._boxes.indexOf(b);
              commit();
              for (const r of list.querySelectorAll(".kjid2-lrow")) r.classList.toggle("active", r._box === b);
            });
            dup.addEventListener("click", (e) => {
              e.stopPropagation();
              const idx = node._boxes.indexOf(b);
              closeLayersMenu();
              startPlacing(idx);
            });
            del.addEventListener("click", (e) => {
              e.stopPropagation();
              const idx = node._boxes.indexOf(b);
              if (idx < 0) return;
              removeBoxSmart(idx);
              if (!node._boxes.length) { closeLayersMenu(); return; }
              buildRows();
            });
            // drag-reorder (vertical FLIP, mirrors the palette swatch reorder)
            row.addEventListener("mousedown", (e) => {
              if (e.button !== 0 || e.target === dup || e.target === del) return;
              e.preventDefault(); e.stopPropagation();
              const sx = e.clientX, sy = e.clientY;
              let dragging = false;
              const move = (me) => {
                if (!dragging) {
                  if (Math.abs(me.clientX - sx) + Math.abs(me.clientY - sy) < 4) return;
                  dragging = true; row.classList.add("dragging"); document.body.classList.add("kjid2-dragging");
                }
                for (const other of list.querySelectorAll(".kjid2-lrow")) {
                  if (other === row) continue;
                  const r = other.getBoundingClientRect();
                  if (me.clientY >= r.top && me.clientY <= r.bottom) {
                    const ref = me.clientY > r.top + r.height / 2 ? other.nextSibling : other;
                    if (ref === row || ref === row.nextSibling) break;
                    const els = Array.from(list.querySelectorAll(".kjid2-lrow"));
                    const prev = els.map((el) => el.getBoundingClientRect().top);
                    list.insertBefore(row, ref);
                    els.forEach((el, k) => {                        // FLIP: slide to new positions
                      const dy = prev[k] - el.getBoundingClientRect().top;
                      if (!dy) return;
                      el.style.transition = "none";
                      el.style.transform = `translateY(${dy}px)`;
                      el.getBoundingClientRect();                   // flush
                      el.style.transition = ""; el.style.transform = "";
                    });
                    break;
                  }
                }
              };
              const up = () => {
                document.removeEventListener("mousemove", move);
                document.removeEventListener("mouseup", up);
                document.body.classList.remove("kjid2-dragging");
                if (dragging) {
                  row.classList.remove("dragging");
                  row._dragged = true;                             // suppress the trailing click
                  const active = node._boxes[node._activeIdx];
                  const order = Array.from(list.querySelectorAll(".kjid2-lrow")).map((el) => el._box);
                  if (order.length === node._boxes.length) node._boxes = order;
                  node._activeIdx = active ? node._boxes.indexOf(active) : -1;
                  renumber();
                  commit();
                }
              };
              document.addEventListener("mousemove", move);
              document.addEventListener("mouseup", up);
            });
          });
        }
        buildRows();

        document.body.appendChild(menu);
        const r = menu.getBoundingClientRect();                    // clamp into the viewport
        let left = clientX, top = clientY;
        if (left + r.width > window.innerWidth) left = window.innerWidth - r.width - 4;
        if (top + r.height > window.innerHeight) top = window.innerHeight - r.height - 4;
        menu.style.left = Math.max(4, left) + "px";
        menu.style.top = Math.max(4, top) + "px";

        node._layerMenuDismiss = (e) => { if (!menu.contains(e.target)) closeLayersMenu(); };
        setTimeout(() => {
          document.addEventListener("mousedown", node._layerMenuDismiss, true);
          document.addEventListener("pointerdown", node._layerMenuDismiss, true);
        }, 0);
      }

      canvasEl.addEventListener("contextmenu", (e) => {
        e.preventDefault(); e.stopPropagation();
        if (node._placing) return;
        if (node._poly) { finishPoly(); return; }      // right-click ends the polyline (like Paint)
        closeInlineEditor();
        openLayersMenu(e.clientX, e.clientY);
      });
      stopProp(clearBtn);
      clearBtn.addEventListener("click", () => {
        closeInlineEditor();
        node._editL = null;
        node._boxes = []; node._activeIdx = -1; node._stylePalette = [];
        node._draws = []; node._redo = []; opsRev++;
        ensureInk();
        updateInkButtons();
        commit(); rebuildStylePalette(); fitNode();
      });

      // ── build caption JSON (mirrors Python key order) ──
      // pyJson: matches Python _dumps — indent=4, but scalar arrays stay on one line.
      function pyJson(v, lvl = 0) {
        if (v === null) return "null";
        if (typeof v === "number" || typeof v === "boolean") return String(v);
        if (typeof v === "string") return JSON.stringify(v);
        const pad = "    ".repeat(lvl + 1), end = "    ".repeat(lvl);
        if (Array.isArray(v)) {
          if (!v.length) return "[]";
          if (v.every((x) => x === null || typeof x !== "object"))   // scalar array → inline
            return "[" + v.map((x) => pyJson(x, lvl)).join(", ") + "]";
          return "[\n" + v.map((x) => pad + pyJson(x, lvl + 1)).join(",\n") + "\n" + end + "]";
        }
        const keys = Object.keys(v);
        if (!keys.length) return "{}";
        return "{\n" + keys.map((k) => pad + JSON.stringify(k) + ": " + pyJson(v[k], lvl + 1)).join(",\n") + "\n" + end + "}";
      }
      function getW(name) { const w = findW(name); return w ? w.value : ""; }
      function cleanPalette(arr) { return (arr || []).filter((c) => c).map((c) => c.toUpperCase()); }
      function normBboxJS(b) {
        const c = (v) => Math.max(0, Math.min(1000, Math.round(v * 1000)));
        let ymin = c(b.y), xmin = c(b.x), ymax = c(b.y + b.h), xmax = c(b.x + b.w);
        if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
        if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
        return [ymin, xmin, ymax, xmax];
      }
      function buildCaption() {
        const cap = {};
        if ((getW("high_level_description") || "").trim()) cap.high_level_description = getW("high_level_description");
        const styleW = findW("style");
        const kind = styleW ? styleW.value : "none";
        if (kind !== "none") {
          const sd = { aesthetics: getW("aesthetics"), lighting: getW("lighting") };
          if (kind === "photo") { sd.photo = getW("style.photo") || ""; sd.medium = getW("medium"); }
          else { sd.medium = getW("medium"); sd.art_style = getW("style.art_style") || ""; }
          const pal = cleanPalette(node._stylePalette);
          if (pal.length) sd.color_palette = pal;
          cap.style_description = sd;
        }
        // hidden regions are excluded from the prompt, as if erased
        const elements = node._boxes.filter((b) => !b.hide).map((b) => {
          const etype = b.type === "text" ? "text" : "obj";
          const el = { type: etype };
          if (!b.nobbox) el.bbox = normBboxJS(b);            // unplaced elements omit bbox
          if (etype === "text") el.text = b.text || "";
          el.desc = b.desc || "";
          const pal = cleanPalette(b.palette).slice(0, MAX_ELEM_COLORS);
          if (pal.length) el.color_palette = pal;
          return el;
        });
        cap.compositional_deconstruction = { background: getW("background"), elements };
        return pyJson(cap);
      }
      // Rough token estimate (~chars/4); exact count needs the Qwen tokenizer.
      function updateTokens() {
        const n = Math.ceil(buildCaption().length / 4);
        tokenSpan.textContent = "~" + n + " tok";
        // grey <256 (sparse) · green healthy · orange nearing · red ≥2048 (model hard cap)
        tokenSpan.style.color = n >= 2048 ? "#e05555" : n >= 1792 ? "#e6a23c" : n >= 256 ? "#6cc06c" : "#888";
      }
      async function doCopy() {
        const txt = buildCaption();
        try { await navigator.clipboard.writeText(txt); copyBtn.textContent = "Copied"; setTimeout(() => (copyBtn.textContent = "Copy"), 900); }
        catch (e) { window.prompt("Copy the caption JSON:", txt); }
      }
      stopProp(copyBtn);
      copyBtn.addEventListener("click", doCopy);

      // ── import a caption JSON and populate the node ──
      function setWidgetVal(name, val) {
        const w = findW(name);
        if (w) { w.value = val; w.callback?.(val); }
      }
      function bboxElemToBox(el, idx) {
        if (!el || typeof el !== "object") return null;
        const box = { type: el.type === "text" ? "text" : "obj",
          text: el.text || "", desc: el.desc || "",
          palette: Array.isArray(el.color_palette) ? el.color_palette.slice() : [] };
        const bb = el.bbox;
        if (Array.isArray(bb) && bb.length === 4) {
          const [ymin, xmin, ymax, xmax] = bb;
          box.x = xmin / 1000; box.y = ymin / 1000; box.w = (xmax - xmin) / 1000; box.h = (ymax - ymin) / 1000;
        } else {
          // No bbox: "unplaced" element — small placeholder, flagged so export omits bbox.
          const k = (idx || 0) % 6;
          box.x = 0.03 + k * 0.035; box.y = 0.03 + k * 0.035; box.w = 0.22; box.h = 0.14;
          box.nobbox = true;
        }
        return box;
      }
      function applyCaption(cap) {
        const cd = (cap && cap.compositional_deconstruction) || {};
        const els = Array.isArray(cd.elements) ? cd.elements : [];
        node._boxes = els.map((el, i) => bboxElemToBox(el, i)).filter(Boolean);
        node._activeIdx = node._boxes.length ? 0 : -1;
        setWidgetVal("high_level_description", cap.high_level_description || "");
        setWidgetVal("background", cd.background || "");
        const sd = cap.style_description || {};
        let kind = "none";
        if (typeof sd.photo === "string") kind = "photo";
        else if (typeof sd.art_style === "string") kind = "art_style";
        const styleW = findW("style");
        if (styleW) styleW.value = kind;            // setter synchronously rebuilds sub-widgets
        if (kind === "photo") setWidgetVal("style.photo", sd.photo || "");
        else if (kind === "art_style") setWidgetVal("style.art_style", sd.art_style || "");
        setWidgetVal("aesthetics", sd.aesthetics || "");
        setWidgetVal("lighting", sd.lighting || "");
        setWidgetVal("medium", sd.medium || "");
        node._stylePalette = Array.isArray(sd.color_palette) ? sd.color_palette.slice() : [];
      }
      function tryParseCaption(t) {
        if (!t) return null;
        try { const o = JSON.parse(t); return (o && typeof o === "object" && o.compositional_deconstruction) ? o : null; }
        catch (e) { return null; }
      }
      // Apply a parsed caption to the editor and refresh everything.
      // The imported boxes are authoritative — the ink layer is cleared so the
      // auto-outliner doesn't re-add regions on top of them.
      function loadCaption(cap) {
        closeInlineEditor();
        node._editL = null;
        applyCaption(cap);
        node._draws = []; node._redo = []; opsRev++;
        ensureInk();
        updateInkButtons();
        syncCanvasToDims(); commit(); rebuildStylePalette(); fitNode();
      }
      async function doImport() {
        let cap = null, txt = "";
        try { txt = (await navigator.clipboard.readText() || "").trim(); cap = tryParseCaption(txt); } catch (e) {}
        if (!cap) { txt = (window.prompt("Paste Ideogram 4 caption JSON:", "") || "").trim(); cap = tryParseCaption(txt); }
        if (!cap) { if (txt) alert("Not a valid Ideogram 4 caption JSON (needs 'compositional_deconstruction')."); return; }
        loadCaption(cap);
      }
      stopProp(importBtn);
      importBtn.addEventListener("click", doImport);

      // Populate the editor from a caption pushed back by execute() when import_json
      // is connected (a connected socket can't be read in the frontend directly).
      function applyImported(capStr) {
        if (!capStr || capStr === node._lastImported) return;
        const cap = tryParseCaption(capStr);
        if (!cap) return;
        node._lastImported = capStr;
        loadCaption(cap);
      }
      chainCallback(node, "onExecuted", function (message) {
        if (message?.caption) applyImported(message.caption[0]);
        // Reflect resolved width/height (e.g. from connected inputs) in the canvas aspect.
        // A connected background image governs the aspect itself, so skip then.
        if (message?.dims && !node._bgImg) {
          const [w, h] = message.dims;
          if (wWidget && w) wWidget.value = w;
          if (hWidget && h) hWidget.value = h;
          syncCanvasToDims(); fitNode();
        }
      });

      // ── property panel ──
      function stopProp(el) {
        for (const ev of ["mousedown", "pointerdown", "wheel"]) el.addEventListener(ev, (e) => e.stopPropagation());
      }
      // Color swatches: onEdit on change, onStruct on add/remove/reorder. Shared by both palettes.
      // Pointer-based drag (HTML5 DnD is unreliable inside LiteGraph DOM widgets) with live reorder.
      function buildSwatchRow(container, arr, max, onEdit, onStruct) {
        arr.forEach((hex, i) => {
          const sw = document.createElement("div");
          sw.className = "kjid2-sw";
          sw.style.background = hex;
          sw.dataset.hex = hex;
          sw.title = "Click edit · drag reorder · right-click remove";
          const inp = document.createElement("input");
          inp.type = "color"; inp.value = hex;
          sw.appendChild(inp);
          container.appendChild(sw);
          inp.addEventListener("input", () => { arr[i] = inp.value; sw.style.background = inp.value; sw.dataset.hex = inp.value; onEdit(); });
          sw.addEventListener("wheel", (e) => e.stopPropagation());
          sw.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); arr.splice(i, 1); onStruct(); });
          sw.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;
            e.preventDefault(); e.stopPropagation();
            const sx = e.clientX, sy = e.clientY;
            let dragging = false;
            const move = (me) => {
              if (!dragging) {
                if (Math.abs(me.clientX - sx) + Math.abs(me.clientY - sy) < 4) return;
                dragging = true; sw.classList.add("dragging"); document.body.classList.add("kjid2-dragging");
              }
              for (const other of container.querySelectorAll(".kjid2-sw")) {
                if (other === sw) continue;
                const r = other.getBoundingClientRect();
                if (me.clientX >= r.left && me.clientX <= r.right && me.clientY >= r.top - 6 && me.clientY <= r.bottom + 6) {
                  const ref = me.clientX > r.left + r.width / 2 ? other.nextSibling : other;
                  if (ref === sw || ref === sw.nextSibling) break;   // already there
                  const els = Array.from(container.querySelectorAll(".kjid2-sw"));
                  const prev = els.map((el) => el.getBoundingClientRect().left);
                  container.insertBefore(sw, ref);
                  els.forEach((el, k) => {                            // FLIP: slide to new positions
                    const dx = prev[k] - el.getBoundingClientRect().left;
                    if (!dx) return;
                    el.style.transition = "none";
                    el.style.transform = `translateX(${dx}px)`;
                    el.getBoundingClientRect();                       // flush
                    el.style.transition = ""; el.style.transform = "";
                  });
                  break;
                }
              }
            };
            const up = () => {
              document.removeEventListener("mousemove", move);
              document.removeEventListener("mouseup", up);
              document.body.classList.remove("kjid2-dragging");
              if (dragging) {
                sw.classList.remove("dragging");
                const order = Array.from(container.querySelectorAll(".kjid2-sw")).map((el) => el.dataset.hex);
                if (order.length === arr.length) { arr.length = 0; arr.push(...order); }
                onStruct();
              } else {
                inp.click();                                 // no drag → treat as click, open the picker
              }
            };
            document.addEventListener("mousemove", move);
            document.addEventListener("mouseup", up);
          });
        });
        if (arr.length < max) {
          const add = document.createElement("button");
          add.className = "kjid2-btn"; add.textContent = "+";
          stopProp(add);
          add.addEventListener("click", () => { arr.push("#ffffff"); onStruct(); });
          container.appendChild(add);
        }
      }

      // Swatch color changed (no add/remove): persist + repaint.
      function swatchEdit() { serialize(); drawCanvas(); }

      function rebuildStylePalette() {
        while (styleBar.children.length > 1) styleBar.removeChild(styleBar.lastChild);
        buildSwatchRow(styleBar, node._stylePalette, MAX_STYLE_COLORS,
          swatchEdit,
          () => { swatchEdit(); rebuildStylePalette(); fitNode(); });
      }

      // Textarea whose user-dragged height persists across panel rebuilds / box switches.
      function makeArea(field, value, placeholder, onInput, defaultH) {
        const ta = document.createElement("textarea");
        ta.className = "kjid2-area";
        ta.placeholder = placeholder;
        ta.value = value || "";
        const h = node._areaH[field] || defaultH;
        if (h) ta.style.height = h + "px";
        stopProp(ta);
        ta.addEventListener("input", onInput);
        const ro = new ResizeObserver(() => {
          if (ta.offsetHeight > 0) { node._areaH[field] = ta.offsetHeight; fitNode(); }
        });
        ro.observe(ta);
        node._areaObservers.push(ro);
        return ta;
      }
      function renderPanel() {
        for (const ro of node._areaObservers) ro.disconnect();
        node._areaObservers = [];
        panel.innerHTML = "";
        const b = node._boxes[node._activeIdx];
        if (!b) {
          hint.textContent = TOOL_HINTS[node._tool] ? TOOL_HINTS[node._tool].split(" · ")[0] : "Draw on the canvas";
          const p = document.createElement("div");
          p.style.color = "#888";
          p.textContent = node._boxes.length ? "Click a region to edit it (➤ select tool)." : "No regions yet — draw something!";
          panel.appendChild(p);
          requestAnimationFrame(fitNode);
          return;
        }
        const col = (b.palette || []).find(Boolean) || "#bbb";
        hint.innerHTML = `<b style="color:${col}">region ${node._activeIdx + 1}</b>` +
          (b.auto ? ` <span style="color:#888">· ✏ drawn — its drawing moves with it</span>`
            : " · dbl-click edit · alt-click overlap · del remove");

        // type toggle
        const typeRow = document.createElement("div");
        typeRow.className = "kjid2-row";
        const lbl = document.createElement("span"); lbl.textContent = "type:"; typeRow.appendChild(lbl);
        for (const t of ["obj", "text"]) {
          const btn = document.createElement("button");
          btn.className = "kjid2-btn" + (b.type === t ? " active" : "");
          btn.textContent = t;
          stopProp(btn);
          btn.addEventListener("click", () => { b.type = t; commit(); });
          typeRow.appendChild(btn);
        }
        if (b.auto && b.L != null) {
          const editBtn = document.createElement("button");
          editBtn.className = "kjid2-btn" + (node._editL === b.L ? " active" : "");
          editBtn.textContent = node._editL === b.L ? "\u2713 Done" : "\u270F Edit drawing";
          editBtn.title = "Add to or fix this region drawing, then \u2713 keep / \u2715 revert on the canvas (Enter / Esc)";
          stopProp(editBtn);
          editBtn.addEventListener("click", () => { if (node._editL === b.L) finishEdit(true); else startEdit(b); });
          typeRow.appendChild(editBtn);
          const note = document.createElement("span");
          note.style.cssText = "color:#777;font-style:italic;";
          note.textContent = node._editL === b.L ? "editing: draw freely, \u2713 keeps, \u2715 reverts"
            : "its drawing moves with the box";
          typeRow.appendChild(note);
        }
        panel.appendChild(typeRow);

        // text (only for text type)
        if (b.type === "text") {
          panel.appendChild(makeArea("text", b.text, "text to render (verbatim)",
            function () { b.text = this.value; touch(); }));
        }

        // desc — default ~3x the single-line min height
        panel.appendChild(makeArea("desc", b.desc, "description of this region",
          function () { b.desc = this.value; touch(); }, 110));

        // palette — editing it marks the box so auto-detection stops overwriting color 1
        const palRow = document.createElement("div");
        palRow.className = "kjid2-row";
        const pl = document.createElement("span"); pl.textContent = "colors:"; palRow.appendChild(pl);
        b.palette = b.palette || [];
        buildSwatchRow(palRow, b.palette, MAX_ELEM_COLORS,
          () => { b.palUser = true; swatchEdit(); },
          () => { b.palUser = true; commit(); });
        panel.appendChild(palRow);

        requestAnimationFrame(fitNode);
      }

      // ── width/height widget callbacks ──
      for (const w of [wWidget, hWidget]) {
        if (!w) continue;
        chainCallback(w, "callback", () => { syncCanvasToDims(); drawCanvas(); fitNode(); });
      }
      // Update the token estimate when the caption-level text widgets change.
      for (const name of ["background", "high_level_description", "aesthetics", "lighting", "medium", "style"]) {
        const w = findW(name);
        if (w) chainCallback(w, "callback", () => updateTokens());
      }

      // ── keep canvas + getMinHeight in sync while the node is resized ──
      let _resizing = false;
      chainCallback(node, "onResize", function () {
        if (_resizing) return;
        _resizing = true;
        recalcWidgetHeight();
        // Resize clamp reads computeSize() before getMinHeight refreshes; re-grow with fresh min.
        const minH = node.computeSize()[1];
        if (node.size[1] < minH) node.size[1] = minH;
        drawCanvas();
        _resizing = false;
      });

      // Optional reference image as the canvas background (matches ImageTransformKJ).
      function loadBg(src) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          node._bgImg = img;
          const r16 = (v) => Math.max(16, Math.round(v / 16) * 16);   // model needs multiples of 16
          if (wWidget) wWidget.value = r16(img.naturalWidth);          // match canvas aspect to the image
          if (hWidget) hWidget.value = r16(img.naturalHeight);
          syncCanvasToDims(); drawCanvas(); fitNode(); updateGrabBtn();
        };
        img.src = src;
      }
      // Restore the imported background (if any), else clear.
      function restoreImportOrClear() {
        if (node.properties.ideoBgImport) {
          node._bgKind = "import";
          loadBg(node.properties.ideoBgImport);
        } else {
          node._bgKind = null;
          node._bgImg = null;
          drawCanvas(); updateGrabBtn();
        }
      }
      watchImageInputs(node, "image", (sources) => {
        if (!sources.length) {
          if (node._bgKind === "input" || !node._bgImg) restoreImportOrClear();
          return;
        }
        node._bgKind = "input";                              // a connected image takes over
        const s = sources[0];
        if (s.isVideo && s.videoEl) captureVideoFrame(s.videoEl, (cv) => loadBg(cv.toDataURL("image/webp", 0.9)));
        else if (s.url) loadBg(s.url);
        updateGrabBtn();
      });
      // "Grab BG" button: use the last generated image as the background, or clear it.
      node._grabResultBg = () => {
        if (!lastResultImage) { alert("No sampling result yet — run a generation first."); return; }
        node._bgKind = "grab";
        loadBg(resultViewUrl(lastResultImage));
      };
      node._clearBg = () => {
        node._bgKind = null;
        node._bgImg = null;
        delete node.properties.ideoBgImport;
        if (node._liveBmp?.close) { try { node._liveBmp.close(); } catch (e) {} node._liveBmp = null; }
        drawCanvas(); updateGrabBtn();
      };
      // Feed a live sampling-preview frame as the background (no width/height change).
      node._ideoSetLiveBg = (bmp) => {
        if (node._liveBmp?.close && node._liveBmp !== bmp) { try { node._liveBmp.close(); } catch (e) {} }
        node._liveBmp = bmp; node._bgImg = bmp; node._bgKind = "live";
        drawCanvas(); updateGrabBtn();
      };
      // After generation, replace the live preview with the full-res final result.
      node._ideoGrabFinal = () => {
        if (!lastResultImage) return;
        if (node._liveBmp?.close) { try { node._liveBmp.close(); } catch (e) {} node._liveBmp = null; }
        node._bgKind = "grab";
        loadBg(resultViewUrl(lastResultImage));
      };

      // Active-box highlight only while the editor is focused or the node is selected.
      wrap.addEventListener("focusin", () => { if (!node._focused) { node._focused = true; drawCanvas(); } });
      wrap.addEventListener("focusout", (e) => {
        if (!wrap.contains(e.relatedTarget)) { node._focused = false; drawCanvas(); }
      });
      chainCallback(node, "onSelected", function () { node._selected = true; drawCanvas(); });
      // NOTE: deselecting the node must NOT touch the polyline — clicking the
      // editor canvas deselects the node in the graph, which would kill the
      // poly right after its first click.
      chainCallback(node, "onDeselected", function () { node._selected = false; finishPlacing(); closeLayersMenu(); drawCanvas(); });

      chainCallback(node, "onRemoved", function () {
        livePreviewNodes.delete(node);
        closeInlineEditor();
        closeLayersMenu();
        for (const ro of node._areaObservers) ro.disconnect();
        node._areaObservers = [];
      });

      // ── restore on load ──
      function _parseBoxes(s) {
        try {
          const p = JSON.parse(s);
          if (Array.isArray(p) && p.some((b) => b && typeof b.x === "number" && typeof b.w === "number")) return p;
        } catch (e) {}
        return null;
      }
      function _parseDraws(s) {
        try {
          const p = JSON.parse(s);
          if (Array.isArray(p) && p.length && p.every((op) => op && typeof op.t === "string")) return p;
        } catch (e) {}
        return null;
      }
      // Persist editor data by name (robust to widget-order changes across versions).
      chainCallback(node, "onSerialize", function (o) {
        if (o) o.ideo = { boxes: node._boxes, palette: node._stylePalette, draws: node._draws };
      });
      chainCallback(node, "onConfigure", function (o) {
        node._configured = true;                             // saved node — keep its saved size
        const raw = o && Array.isArray(o.widgets_values) ? o.widgets_values : [];
        // Recover regions: name-keyed blob → named widget → raw saved values (survives
        // any widget reorder/remap across versions) → live widgets.
        let boxes = (o && o.ideo && Array.isArray(o.ideo.boxes)) ? o.ideo.boxes : _parseBoxes(elementsWidget?.value || "");
        if (!boxes) { for (const v of raw) { const b = _parseBoxes(v); if (b) { boxes = b; break; } } }
        if (!boxes) { for (const w of node.widgets || []) { const b = _parseBoxes(w?.value); if (b) { boxes = b; break; } } }
        if (boxes) {
          node._boxes = boxes.filter((b) => b && typeof b.x === "number");
          node._activeIdx = node._boxes.length ? 0 : -1;
        }
        const isPal = (p) => Array.isArray(p) && p.length && p.every((c) => typeof c === "string" && c[0] === "#");
        let pal = (o && o.ideo && isPal(o.ideo.palette)) ? o.ideo.palette : null;
        if (!pal) { try { const p = JSON.parse(stylePaletteWidget?.value || ""); if (isPal(p)) pal = p; } catch (e) {} }
        if (!pal) { for (const v of raw) { try { const p = JSON.parse(v); if (isPal(p)) { pal = p; break; } } catch (e) {} } }
        if (pal) node._stylePalette = pal.slice();
        // Recover the ink layer: name-keyed blob → named widget.
        let draws = (o && o.ideo && Array.isArray(o.ideo.draws)) ? o.ideo.draws : _parseDraws(drawDataWidget?.value || "");
        if (draws) {
          node._draws = draws.filter((op) => op && typeof op.t === "string");
          node._redo = [];
        }
        // migrate pre-layer drawings: they all become one layer (1)
        if (node._draws.some((op) => op.L == null)) {
          let nextN = 2;
          for (const op of node._draws) {
            if (op.L == null) op.L = (op.t === "e") ? 0 : 1;
            if (op.t === "C" && op.N == null) op.N = nextN++;
          }
          for (const b of node._boxes) {
            if (b.auto && b.L == null) { delete b.auto; delete b.src; }
          }
        }
        opsRev++;                                            // saved boxes are authoritative — rebuild the raster only
        hideDataWidgets();
        serialize();                                         // realign widget values for Python + future saves
        if (bgBrightnessWidget) bgSlider.value = bgBrightnessWidget.value;
        if (bgOpacityWidget) opaSlider.value = bgOpacityWidget.value;
        node._brushColor = typeof node.properties.ideoBrushColor === "string" ? node.properties.ideoBrushColor : node._brushColor;
        if (typeof node.properties.ideoBrushSize === "number") node._brushSize = node.properties.ideoBrushSize;
        colorInp.value = node._brushColor;
        sizeSlider.value = String(node._brushSize);
        sizeLbl.textContent = node._brushSize + "px";
        setTool(node.properties.ideoTool || node._tool);
        if (node.properties.ideoBgImport && !node._bgImg) restoreImportOrClear();
        ensureInk();
        updateInkButtons();
        syncCanvasToDims();
        rebuildStylePalette();
        renderPanel();
        rebuildSideList();
        drawCanvas();
        updateTokens();
        requestAnimationFrame(fitNode);
      });

      // initial layout (deferred so size/last_y are settled)
      setTimeout(() => {
        hideDataWidgets();
        // fresh nodes start wide so the canvas + side panel are comfortably readable
        const minW = node._configured ? 420 : 660;
        if (node.size[0] < minW) node.setSize([minW, node.size[1]]);
        setTool(node.properties.ideoTool || node._tool);
        if (node.properties.ideoBgImport && !node._bgImg) restoreImportOrClear();
        syncCanvasToDims();
        rebuildStylePalette();
        renderPanel();
        rebuildSideList();
        drawCanvas();
        updateTokens();
        updateInkButtons();
        fitNode();
      }, 0);
    });
  },
});
