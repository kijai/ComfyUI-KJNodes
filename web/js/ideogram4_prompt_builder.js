import { chainCallback, addMiddleClickPan, addWheelPassthrough, cursorForBboxMode, watchImageInputs, captureVideoFrame } from './utility.js';
const { app } = window.comfyAPI.app;

const HANDLE = 8;            // hit radius (canvas px) for corners/edges
const MAX_ELEM_COLORS = 5;   // Ideogram 4 per-element palette cap
const MAX_STYLE_COLORS = 16; // Ideogram 4 style palette cap
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
  if (document.getElementById("kjideo-style")) return;
  const s = document.createElement("style");
  s.id = "kjideo-style";
  s.textContent = `
    .kjideo-wrap { display:flex; flex-direction:column; overflow:hidden; position:relative; pointer-events:auto; gap:4px; }
    .kjideo-canvas { cursor:crosshair; display:block; width:100%; height:auto; flex:0 0 auto; background:#1a1a1a; border-radius:4px; outline:none; }
    .kjideo-bar { display:flex; align-items:center; gap:6px; font:11px sans-serif; color:#aaa; user-select:none; padding:0 2px; flex:0 0 auto; }
    .kjideo-panel { display:flex; flex-direction:column; gap:5px; padding:6px; background:#262626; border-radius:4px; font:11px sans-serif; color:#bbb; flex:0 0 auto; }
    .kjideo-row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
    .kjideo-btn { background:#333; border:1px solid #555; border-radius:4px; color:#bbb; font:11px sans-serif; cursor:pointer; padding:2px 8px; line-height:16px; white-space:nowrap; flex-shrink:0; }
    .kjideo-btn:hover { border-color:#46b4e6; color:#fff; }
    .kjideo-btn.active { border-color:#46b4e6; color:#46b4e6; background:#2a3a42; }
    .kjideo-area { width:100%; box-sizing:border-box; background:#1d1d1d; border:1px solid #444; border-radius:4px; color:#ddd; font:13px monospace; padding:4px 6px; resize:vertical; min-height:36px; }
    .kjideo-sw { width:20px; height:20px; border:1px solid #666; border-radius:3px; cursor:pointer; flex-shrink:0; position:relative; transition:transform .18s ease, box-shadow .12s ease, opacity .12s ease; }
    .kjideo-sw:hover { transform:scale(1.2); box-shadow:0 0 0 2px #46b4e6; z-index:3; }
    .kjideo-sw.dragging { opacity:.4; box-shadow:0 0 0 2px #46b4e6; }
    body.kjideo-dragging, body.kjideo-dragging * { cursor:move !important; }
    .kjideo-sw input { position:absolute; opacity:0; width:0; height:0; pointer-events:none; }
    .kjideo-inline { position:absolute; box-sizing:border-box; background:rgba(18,18,18,0.92); border:2px solid #46b4e6; border-radius:3px; color:#fff; font:13px monospace; padding:3px 4px; resize:none; outline:none; z-index:10; }
  `;
  document.head.appendChild(s);
}

app.registerExtension({
  name: "KJNodes.Ideogram4PromptBuilder",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "Ideogram4PromptBuilderKJ") return;
    injectStyle();

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const findW = (n) => node.widgets?.find((w) => w.name === n);
      const elementsWidget = findW("elements_data");
      const stylePaletteWidget = findW("style_palette_data");
      const bgBrightnessWidget = findW("bg_brightness");
      if (bgBrightnessWidget && typeof bgBrightnessWidget.value !== "number") bgBrightnessWidget.value = 25;
      const wWidget = findW("width"), hWidget = findW("height");
      // Hide the data widgets while keeping them serializable.
      function hideDataWidgets() {
        for (const w of [elementsWidget, stylePaletteWidget, bgBrightnessWidget]) {
          if (!w) continue;
          w.hidden = true;
          w.computeSize = () => [0, -4];
        }
        for (const name of ["elements_data", "style_palette_data", "bg_brightness"]) {
          const i = node.inputs?.findIndex((inp) => inp.name === name);
          if (i != null && i !== -1) node.removeInput(i);
        }
      }
      hideDataWidgets();

      node._boxes = [];        // {x,y,w,h normalized 0-1, type, text, desc, palette[]}
      node._stylePalette = []; // global style color palette (hex[])
      node._activeIdx = -1;
      node._drawing = false;
      node._dragMode = null;
      node._dragStartN = null; // mouse-down point, normalized
      node._boxAtStart = null; // active box snapshot at drag start
      node._hoverTitle = null; // index of the title chip under the cursor
      node._hoverBox = null;   // index of the box under the cursor
      node._focused = false;   // editor (DOM) focused — gates the active-box highlight
      node._selected = false;  // node selected in the graph
      node._bgImg = null;      // optional reference image shown as the canvas background
      node._bgManual = false;  // bg set via "use last result" (not the image input)
      node._lastImported = ""; // last import_json applied to the editor (avoid re-apply)
      node._areaH = node._areaH || {};      // remembered textarea heights (per field)
      node._areaObservers = [];             // live ResizeObservers to disconnect on rebuild

      // ── DOM ──
      const wrap = document.createElement("div");
      wrap.className = "kjideo-wrap";
      const bar = document.createElement("div");
      bar.className = "kjideo-bar";
      const hint = document.createElement("span");
      hint.style.flex = "1";
      const copyBtn = document.createElement("button");
      copyBtn.className = "kjideo-btn";
      copyBtn.textContent = "Copy";
      copyBtn.title = "Copy the current caption JSON to the clipboard";
      const importBtn = document.createElement("button");
      importBtn.className = "kjideo-btn";
      importBtn.textContent = "Paste";
      importBtn.title = "Parse a caption JSON (clipboard, else paste prompt) and populate the node";
      const clearBtn = document.createElement("button");
      clearBtn.className = "kjideo-btn";
      clearBtn.textContent = "Clear all";
      const tokenSpan = document.createElement("span");
      tokenSpan.style.cssText = "color:#888; white-space:nowrap;";
      tokenSpan.title = "Rough token estimate (~chars/4). Grey <256, green healthy, orange nearing, red ≥2048 (model cap — will error)";
      const grabBtn = document.createElement("button");
      grabBtn.className = "kjideo-btn";
      grabBtn.addEventListener("mousedown", (e) => e.stopPropagation());
      grabBtn.addEventListener("click", () => { (node._bgManual && node._bgImg) ? node._clearBg() : node._grabResultBg(); });
      function updateGrabBtn() {
        const clear = node._bgManual && node._bgImg;
        grabBtn.textContent = clear ? "Clear BG" : "Grab BG";
        grabBtn.title = clear ? "Remove the grabbed background"
          : "Use the last generated image as the background";
      }
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
      const bgSlider = document.createElement("input");
      bgSlider.type = "range"; bgSlider.min = "0"; bgSlider.max = "100"; bgSlider.step = "1";
      bgSlider.value = bgBrightnessWidget ? bgBrightnessWidget.value : 25;
      bgSlider.title = "Background brightness (image or blank canvas)";
      bgSlider.style.cssText = "width:64px;flex:0 0 auto;";
      stopProp(bgSlider);
      bgSlider.addEventListener("input", () => { if (bgBrightnessWidget) bgBrightnessWidget.value = parseInt(bgSlider.value, 10); drawCanvas(); });
      bar.appendChild(hint); bar.appendChild(liveLabel); bar.appendChild(grabBtn); bar.appendChild(bgSlider); bar.appendChild(tokenSpan); bar.appendChild(copyBtn); bar.appendChild(importBtn); bar.appendChild(clearBtn);
      updateGrabBtn();

      // Persistent global style-palette row
      const styleBar = document.createElement("div");
      styleBar.className = "kjideo-bar";
      const styleLbl = document.createElement("span");
      styleLbl.textContent = "Style colors:";
      styleBar.appendChild(styleLbl);

      const canvasEl = document.createElement("canvas");
      canvasEl.className = "kjideo-canvas";
      canvasEl.tabIndex = 0;                                  // focusable, so it can receive key events
      canvasEl.title = "Drag to draw · click to select · alt-click overlap · dbl-click edit · " +
        "Del remove · Ctrl/Cmd+C/V/D copy/paste/duplicate";
      const ctx = canvasEl.getContext("2d");
      addWheelPassthrough(wrap);
      addMiddleClickPan(canvasEl);

      const panel = document.createElement("div");
      panel.className = "kjideo-panel";

      // Canvas above panel so the panel grows downward without shifting the canvas.
      wrap.appendChild(bar); wrap.appendChild(styleBar); wrap.appendChild(canvasEl); wrap.appendChild(panel);

      const TOOLBAR_H = 22;
      node._widgetHeight = 360;
      node.ideoEditor = node.addDOMWidget("ideo_editor", "Ideogram4Editor", wrap, {
        serialize: false, hideOnZoom: false,
        getMinHeight: () => node._widgetHeight,
      });
      node.resizable = true;

      // ── canvas sizing ──
      // The display size is CSS-driven (width:100% + aspect-ratio); the backing store
      // is sized to display × devicePixelRatio in prepCanvas() so text/lines stay crisp.
      function setCanvasSize(w, h) {
        canvasEl.style.aspectRatio = `${w} / ${h}`;          // display shape only
        if (node.graph) node.graph.setDirtyCanvas(true, true);
      }
      function syncCanvasToDims() {
        const w = wWidget ? wWidget.value : 1024, h = hWidget ? hWidget.value : 1024;
        setCanvasSize(Math.max(1, w), Math.max(1, h));
        drawCanvas();
      }

      // Content height = panel's bottom edge in the wrapper (includes toolbar/canvas/gaps).
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
      // greedy word-wrap to maxW px (ctx.font must be set by caller)
      function wrapLines(text, maxW) {
        const lines = [];
        for (const para of text.split("\n")) {
          let line = "";
          for (const word of para.split(/\s+/)) {
            if (!word) continue;
            const test = line ? line + " " + word : word;
            if (line && ctx.measureText(test).width > maxW) { lines.push(line); line = word; }
            else line = test;
          }
          lines.push(line);
        }
        return lines;
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

      // All boxes under the point, top-first to match draw order: the active box is
      // drawn last (on top), then the rest by index high→low.
      function boxesAt(mN) {
        const rx = HANDLE / logW(), ry = HANDLE / logH();
        const res = [];
        for (let i = node._boxes.length - 1; i >= 0; i--) {
          const b = node._boxes[i];
          const mode = rectHitTestN(mN.x, mN.y, b.x, b.y, b.x + b.w, b.y + b.h, rx, ry);
          if (mode) res.push({ index: i, mode });
        }
        const ai = res.findIndex((c) => c.index === node._activeIdx);
        if (ai > 0) res.unshift(res.splice(ai, 1)[0]);
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
        if (node._bgImg) {                                         // reference image, dimmed by brightness
          ctx.drawImage(node._bgImg, 0, 0, W, H);
          const dim = 1 - bri / 100;
          if (dim > 0) { ctx.fillStyle = `rgba(0,0,0,${dim})`; ctx.fillRect(0, 0, W, H); }
        } else {                                                   // blank canvas grey from brightness
          const g = Math.round(bri / 100 * 128);
          ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(0, 0, W, H);
        }
        // active box only when the editor is focused or the node is selected
        const aIdx = (node._focused || node._selected) ? node._activeIdx : -1;
        const order = node._boxes.map((_, i) => i).filter((i) => i !== aIdx);
        if (aIdx >= 0 && aIdx < node._boxes.length) order.push(aIdx);  // active drawn last (on top)
        const tagR = tagRects();                              // collision-avoided tag positions
        for (const i of order) {
          const b = node._boxes[i], active = i === aIdx;
          const pal = (b.palette || []).filter(Boolean);
          const col = pal.length ? pal[0] : "#8c8c8c";       // box color = first palette color, else neutral grey
          const { x1, y1, x2, y2 } = toPx(b);
          const w = x2 - x1, h = y2 - y1;
          const hovered = i === node._hoverBox || active;    // active box stays highlighted (on top)
          if (active) {                                      // opaque backing so contents read clearly over boxes behind
            ctx.fillStyle = "rgba(26,26,26,0.88)";
            ctx.fillRect(x1, y1, w, h);
          }
          ctx.fillStyle = col + (hovered ? "3a" : "22");     // tint of the box color
          ctx.fillRect(x1, y1, w, h);
          if (b.nobbox) ctx.setLineDash([6, 4]);             // unplaced (no bbox in source)
          const lw = active ? 2 : (hovered ? 1.5 : 1);
          ctx.strokeStyle = col; ctx.lineWidth = lw;
          ctx.strokeRect(x1 + lw / 2, y1 + lw / 2, w - lw, h - lw);  // inside the box so strip/badge align at y1
          ctx.setLineDash([]);
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
            ctx.fillStyle = readableText(col);              // box color, lightened if too dark
            const pad = 4, lh = 14;
            let ty = y1 + 15 + 12;                        // first line below the tag chip
            for (const line of wrapLines(body, w - pad * 2)) {
              if (ty > y1 + h) break;                      // clip overflow vertically
              ctx.fillText(line, x1 + pad, ty);
              ty += lh;
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
        }
      }

      // ── serialization ──
      function serialize() {
        if (elementsWidget) elementsWidget.value = node._boxes.length ? JSON.stringify(node._boxes) : "";
        if (stylePaletteWidget) stylePaletteWidget.value = node._stylePalette.length ? JSON.stringify(node._stylePalette) : "";
      }

      function commit() { serialize(); renderPanel(); drawCanvas(); updateTokens(); }
      // Live text edit: persist + repaint + token count, without rebuilding the panel.
      function touch() { serialize(); drawCanvas(); updateTokens(); }

      function removeBox(i) {
        node._boxes.splice(i, 1);
        if (node._boxes.length === 0) node._activeIdx = -1;
        else if (i <= node._activeIdx) node._activeIdx = Math.max(0, node._activeIdx - 1);
      }

      // ── pointer interaction ──
      canvasEl.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        canvasEl.focus();                // so Delete/Backspace targets this editor
        node._hoverTitle = null; node._hoverBox = null;  // clear hover highlight while interacting
        const mN = mouseN(e);
        const hit = pickForSelection(mN, e.altKey);
        if (hit) {
          node._activeIdx = hit.index;
          node._dragMode = hit.mode;
          node._boxAtStart = { ...node._boxes[hit.index] };
        } else {
          node._dragMode = "draw";
          const nb = { x: mN.x, y: mN.y, w: 0, h: 0, type: "obj", text: "", desc: "", palette: [] };
          node._boxes.push(nb);
          node._activeIdx = node._boxes.length - 1;
          node._boxAtStart = { ...nb };
        }
        node._drawing = true;
        node._dragStartN = mN;
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        e.preventDefault(); e.stopPropagation();
        drawCanvas();   // panel rebuild/resize deferred to onUp so the canvas doesn't shift mid-drag
      });

      canvasEl.addEventListener("mousemove", (e) => {
        if (node._drawing) return;
        const mN = mouseN(e);
        const ti = titleAt(mN);
        const hit = hitTest(mN);
        const hb = ti != null ? ti : (hit ? hit.index : null);
        if (ti !== node._hoverTitle || hb !== node._hoverBox) {
          node._hoverTitle = ti; node._hoverBox = hb; drawCanvas();
        }
        canvasEl.style.cursor = ti != null ? "pointer" : (hit ? (cursorForBboxMode(hit.mode) || "crosshair") : "crosshair");
      });
      canvasEl.addEventListener("mouseleave", () => {
        if (node._hoverTitle !== null || node._hoverBox !== null) {
          node._hoverTitle = null; node._hoverBox = null; drawCanvas();
        }
      });

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
        ta.className = "kjideo-inline";
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
        e.preventDefault(); e.stopPropagation();
        const cands = boxesAt(mouseN(e));     // edit the active box if it's under the cursor, else topmost
        const target = cands.find((c) => c.index === node._activeIdx) || cands[0];
        if (target) openInlineEditor(target.index);
      });

      // Paste a clone of the clipboard box, offset slightly and clamped into the canvas.
      function pasteBox() {
        if (!copiedBox) return;
        const nb = JSON.parse(JSON.stringify(copiedBox));
        nb.x = Math.max(0, Math.min(clamp01(nb.x + 0.03), 1 - nb.w));
        nb.y = Math.max(0, Math.min(clamp01(nb.y + 0.03), 1 - nb.h));
        delete nb.nobbox;                              // a pasted box is placed
        node._boxes.push(nb);
        node._activeIdx = node._boxes.length - 1;
        commit(); fitNode();
      }
      // Keyboard: Delete removes; Ctrl/Cmd C/V/D copy/paste/duplicate the active region.
      // Canvas must be focused; stop the event so LiteGraph doesn't act on the node.
      canvasEl.addEventListener("keydown", (e) => {
        if (node._drawing) return;
        const ctrl = e.ctrlKey || e.metaKey;
        if ((e.key === "Delete" || e.key === "Backspace") && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          removeBox(node._activeIdx); commit(); fitNode();
        } else if (ctrl && e.key === "c" && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          copiedBox = JSON.parse(JSON.stringify(node._boxes[node._activeIdx]));
        } else if (ctrl && e.key === "v" && copiedBox) {
          e.preventDefault(); e.stopPropagation();
          pasteBox();
        } else if (ctrl && e.key === "d" && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          copiedBox = JSON.parse(JSON.stringify(node._boxes[node._activeIdx]));
          pasteBox();
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
        commit();
      }

      canvasEl.addEventListener("contextmenu", (e) => e.preventDefault());
      stopProp(clearBtn);
      clearBtn.addEventListener("click", () => {
        closeInlineEditor();
        node._boxes = []; node._activeIdx = -1; node._stylePalette = [];
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
        const elements = node._boxes.map((b) => {
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
      function loadCaption(cap) {
        closeInlineEditor();
        applyCaption(cap);
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
          sw.className = "kjideo-sw";
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
                dragging = true; sw.classList.add("dragging"); document.body.classList.add("kjideo-dragging");
              }
              for (const other of container.querySelectorAll(".kjideo-sw")) {
                if (other === sw) continue;
                const r = other.getBoundingClientRect();
                if (me.clientX >= r.left && me.clientX <= r.right && me.clientY >= r.top - 6 && me.clientY <= r.bottom + 6) {
                  const ref = me.clientX > r.left + r.width / 2 ? other.nextSibling : other;
                  if (ref === sw || ref === sw.nextSibling) break;   // already there
                  const els = Array.from(container.querySelectorAll(".kjideo-sw"));
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
              document.body.classList.remove("kjideo-dragging");
              if (dragging) {
                sw.classList.remove("dragging");
                const order = Array.from(container.querySelectorAll(".kjideo-sw")).map((el) => el.dataset.hex);
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
          add.className = "kjideo-btn"; add.textContent = "+";
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
        ta.className = "kjideo-area";
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
          hint.textContent = "Drag on the canvas to add a region";
          const p = document.createElement("div");
          p.style.color = "#888";
          p.textContent = node._boxes.length ? "Click a region to edit it." : "No regions yet.";
          panel.appendChild(p);
          requestAnimationFrame(fitNode);
          return;
        }
        const col = (b.palette || []).find(Boolean) || "#bbb";
        hint.innerHTML = `<b style="color:${col}">region ${node._activeIdx + 1}</b> · dbl-click edit · alt-click overlap · del remove`;

        // type toggle
        const typeRow = document.createElement("div");
        typeRow.className = "kjideo-row";
        const lbl = document.createElement("span"); lbl.textContent = "type:"; typeRow.appendChild(lbl);
        for (const t of ["obj", "text"]) {
          const btn = document.createElement("button");
          btn.className = "kjideo-btn" + (b.type === t ? " active" : "");
          btn.textContent = t;
          stopProp(btn);
          btn.addEventListener("click", () => { b.type = t; commit(); });
          typeRow.appendChild(btn);
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

        // palette
        const palRow = document.createElement("div");
        palRow.className = "kjideo-row";
        const pl = document.createElement("span"); pl.textContent = "colors:"; palRow.appendChild(pl);
        b.palette = b.palette || [];
        buildSwatchRow(palRow, b.palette, MAX_ELEM_COLORS, swatchEdit, commit);
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
      watchImageInputs(node, "image", (sources) => {
        if (!sources.length) { if (!node._bgManual) { node._bgImg = null; drawCanvas(); updateGrabBtn(); } return; }
        node._bgManual = false;                              // a connected image takes over
        const s = sources[0];
        if (s.isVideo && s.videoEl) captureVideoFrame(s.videoEl, (cv) => loadBg(cv.toDataURL("image/webp", 0.9)));
        else if (s.url) loadBg(s.url);
      });
      // "Grab BG" button: use the last generated image as the background, or clear it.
      node._grabResultBg = () => {
        if (!lastResultImage) { alert("No sampling result yet — run a generation first."); return; }
        node._bgManual = true;
        loadBg(resultViewUrl(lastResultImage));
      };
      node._clearBg = () => {
        node._bgManual = false; node._bgImg = null;
        if (node._liveBmp?.close) { try { node._liveBmp.close(); } catch (e) {} node._liveBmp = null; }
        drawCanvas(); updateGrabBtn();
      };
      // Feed a live sampling-preview frame as the background (no width/height change).
      node._ideoSetLiveBg = (bmp) => {
        if (node._liveBmp?.close && node._liveBmp !== bmp) { try { node._liveBmp.close(); } catch (e) {} }
        node._liveBmp = bmp; node._bgImg = bmp; node._bgManual = true;
        drawCanvas(); updateGrabBtn();
      };
      // After generation, replace the live preview with the full-res final result.
      node._ideoGrabFinal = () => {
        if (!lastResultImage) return;
        if (node._liveBmp?.close) { try { node._liveBmp.close(); } catch (e) {} node._liveBmp = null; }
        node._bgManual = true;
        loadBg(resultViewUrl(lastResultImage));
      };

      // Active-box highlight only while the editor is focused or the node is selected.
      wrap.addEventListener("focusin", () => { if (!node._focused) { node._focused = true; drawCanvas(); } });
      wrap.addEventListener("focusout", (e) => {
        if (!wrap.contains(e.relatedTarget)) { node._focused = false; drawCanvas(); }
      });
      chainCallback(node, "onSelected", function () { node._selected = true; drawCanvas(); });
      chainCallback(node, "onDeselected", function () { node._selected = false; drawCanvas(); });

      chainCallback(node, "onRemoved", function () {
        livePreviewNodes.delete(node);
        closeInlineEditor();
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
      // Persist editor data by name (robust to widget-order changes across versions).
      chainCallback(node, "onSerialize", function (o) {
        if (o) o.ideo = { boxes: node._boxes, palette: node._stylePalette };
      });
      chainCallback(node, "onConfigure", function (o) {
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
        hideDataWidgets();
        serialize();                                         // realign widget values for Python + future saves
        if (bgBrightnessWidget) bgSlider.value = bgBrightnessWidget.value;
        syncCanvasToDims();
        rebuildStylePalette();
        renderPanel();
        drawCanvas();
        updateTokens();
        requestAnimationFrame(fitNode);
      });

      // initial layout (deferred so size/last_y are settled)
      setTimeout(() => {
        hideDataWidgets();
        if (node.size[0] < 380) node.setSize([380, node.size[1]]);
        syncCanvasToDims();
        rebuildStylePalette();
        renderPanel();
        drawCanvas();
        updateTokens();
        fitNode();
      }, 0);
    });
  },
});
