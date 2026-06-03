import { chainCallback, addMiddleClickPan, addWheelPassthrough, rectHitTest, cursorForBboxMode } from './utility.js';
const { app } = window.comfyAPI.app;

const BBOX_PALETTE = ["#46b4e6", "#e68246", "#82e646", "#e646b4", "#e6e646", "#46e6c8"];
const MAX_CANVAS_W = 1024, MAX_CANVAS_H = 768;
const HANDLE = 8;            // hit radius (canvas px) for corners/edges
const MAX_ELEM_COLORS = 5;   // Ideogram 4 per-element palette cap
const MAX_STYLE_COLORS = 16; // Ideogram 4 style palette cap

function clampToMaxCanvas(w, h) {
  if (w > MAX_CANVAS_W || h > MAX_CANVAS_H) {
    const s = Math.min(MAX_CANVAS_W / w, MAX_CANVAS_H / h);
    return [Math.round(w * s), Math.round(h * s)];
  }
  return [w, h];
}

function boxColor(index) { return BBOX_PALETTE[index % BBOX_PALETTE.length]; }

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
    .kjideo-area { width:100%; box-sizing:border-box; background:#1d1d1d; border:1px solid #444; border-radius:4px; color:#ddd; font:11px sans-serif; padding:4px 6px; resize:vertical; min-height:36px; }
    .kjideo-sw { width:20px; height:20px; border:1px solid #666; border-radius:3px; cursor:pointer; flex-shrink:0; position:relative; }
    .kjideo-sw input { position:absolute; opacity:0; width:0; height:0; pointer-events:none; }
    .kjideo-inline { position:absolute; box-sizing:border-box; background:rgba(18,18,18,0.92); border:2px solid #46b4e6; border-radius:3px; color:#fff; font:11px sans-serif; padding:3px 4px; resize:none; outline:none; z-index:10; }
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
      const wWidget = findW("width"), hWidget = findW("height");
      // Hide the data widgets while keeping them serializable.
      function hideDataWidgets() {
        for (const w of [elementsWidget, stylePaletteWidget]) {
          if (!w) continue;
          w.hidden = true;
          w.computeSize = () => [0, -4];
        }
        for (const name of ["elements_data", "style_palette_data"]) {
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
      tokenSpan.title = "Rough token estimate (~chars/4) of the caption prompt — not exact";
      bar.appendChild(hint); bar.appendChild(tokenSpan); bar.appendChild(copyBtn); bar.appendChild(importBtn); bar.appendChild(clearBtn);

      // Persistent global style-palette row
      const styleBar = document.createElement("div");
      styleBar.className = "kjideo-bar";
      const styleLbl = document.createElement("span");
      styleLbl.textContent = "Style colors:";
      styleBar.appendChild(styleLbl);

      const canvasEl = document.createElement("canvas");
      canvasEl.className = "kjideo-canvas";
      canvasEl.tabIndex = 0;                                  // focusable, so it can receive Delete
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

      // ── canvas sizing (aspect from width/height widgets) ──
      function setCanvasSize(cw, ch) {
        canvasEl.width = cw; canvasEl.height = ch;
        canvasEl.style.aspectRatio = `${cw} / ${ch}`;
        if (node.graph) node.graph.setDirtyCanvas(true, true);
      }
      function syncCanvasToDims() {
        const w = wWidget ? wWidget.value : 1024, h = hWidget ? hWidget.value : 1024;
        const [cw, ch] = clampToMaxCanvas(Math.max(1, w), Math.max(1, h));
        if (cw !== canvasEl.width || ch !== canvasEl.height) setCanvasSize(cw, ch);
      }

      // Content height = panel's bottom edge in the wrapper (includes toolbar/canvas/gaps).
      function recalcWidgetHeight() {
        if (canvasEl.width <= 0) return;
        const contentH = panel.offsetTop + panel.offsetHeight;
        if (contentH > 0) {
          node._widgetHeight = contentH + 10;                  // margin pad
        } else {                                               // not laid out yet
          const ch = Math.round(Math.max(100, node.size[0] - 30) * (canvasEl.height / canvasEl.width));
          node._widgetHeight = ch + TOOLBAR_H + 70;
        }
      }
      function fitNode() {
        recalcWidgetHeight();
        // computeSize (stable min-heights), not last_y which creeps with growable widgets above.
        const minH = node.computeSize()[1];
        if (node.size[1] < minH) node.setSize([node.size[0], minH]);
      }

      // ── geometry helpers ──
      function toPx(b) {
        return { x1: b.x * canvasEl.width, y1: b.y * canvasEl.height,
          x2: (b.x + b.w) * canvasEl.width, y2: (b.y + b.h) * canvasEl.height };
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
        const rx = HANDLE / canvasEl.width, ry = HANDLE / canvasEl.height;
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
      // Title-chip rect (canvas px) at a box's top-left, matching _draw.
      function titleChipPx(i) {
        const b = node._boxes[i];
        ctx.font = "bold 10px sans-serif";
        const tag = (b.type === "text" ? "T" : "O") + (i + 1);
        return { x: b.x * canvasEl.width, y: b.y * canvasEl.height, w: ctx.measureText(tag).width + 8, h: 14 };
      }
      function titleAt(mN) {
        const px = mN.x * canvasEl.width, py = mN.y * canvasEl.height;
        for (let i = node._boxes.length - 1; i >= 0; i--) {
          const r = titleChipPx(i);
          if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) return i;
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
        const W = canvasEl.width, H = canvasEl.height;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#1a1a1a"; ctx.fillRect(0, 0, W, H);
        // draw the active box last so it sits in front (kept by array index for color/tag)
        const order = node._boxes.map((_, i) => i).filter((i) => i !== node._activeIdx);
        if (node._activeIdx >= 0 && node._activeIdx < node._boxes.length) order.push(node._activeIdx);
        for (const i of order) {
          const b = node._boxes[i], col = boxColor(i), active = i === node._activeIdx;
          const { x1, y1, x2, y2 } = toPx(b);
          const w = x2 - x1, h = y2 - y1;
          ctx.fillStyle = col + "22";
          ctx.fillRect(x1, y1, w, h);
          if (b.nobbox) ctx.setLineDash([6, 4]);             // unplaced (no bbox in source)
          ctx.strokeStyle = col; ctx.lineWidth = active ? 2 : 1;
          ctx.strokeRect(x1, y1, w, h);
          ctx.setLineDash([]);
          if (active) {
            ctx.fillStyle = col;
            for (const [hx, hy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
              ctx.fillRect(hx - 4, hy - 4, 8, 8);
          }
          // in-box content: tag chip + wrapped prompt + palette dots, clipped to the box
          ctx.save();
          ctx.beginPath(); ctx.rect(x1, y1, w, h); ctx.clip();
          const tag = (b.type === "text" ? "T" : "O") + (i + 1);
          ctx.font = "bold 10px sans-serif";
          const chipW = ctx.measureText(tag).width + 8;
          ctx.fillStyle = col;
          ctx.fillRect(x1, y1, chipW, 14);
          if (i === node._hoverTitle) {                       // hover highlight on the title chip
            ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.fillRect(x1, y1, chipW, 14);
            ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.strokeRect(x1 + 0.5, y1 + 0.5, chipW - 1, 13);
          }
          ctx.fillStyle = "#000";
          ctx.fillText(tag, x1 + 4, y1 + 11);

          let body = b.desc || "";
          if (b.type === "text" && b.text) body = `"${b.text}"` + (body ? " — " + body : "");
          if (body) {
            ctx.font = "11px sans-serif";
            ctx.fillStyle = "#fff";
            ctx.shadowColor = "#000"; ctx.shadowBlur = 2;
            const pad = 4, lh = 13;
            let ty = y1 + 14 + 11;                        // first line below the tag chip
            for (const line of wrapLines(body, w - pad * 2)) {
              if (ty > y1 + h) break;                      // clip overflow vertically
              ctx.fillText(line, x1 + pad, ty);
              ty += lh;
            }
            ctx.shadowBlur = 0;
          }
          if (b.palette?.length) {
            for (let p = 0; p < b.palette.length; p++) {
              ctx.fillStyle = b.palette[p];
              ctx.fillRect(x1 + 2 + p * 9, y2 - 9, 7, 7);
              ctx.strokeStyle = "#000"; ctx.lineWidth = 1;
              ctx.strokeRect(x1 + 2 + p * 9, y2 - 9, 7, 7);
            }
          }
          ctx.restore();
        }
      }

      // ── serialization ──
      function serialize() {
        if (elementsWidget) elementsWidget.value = node._boxes.length ? JSON.stringify(node._boxes) : "";
        if (stylePaletteWidget) stylePaletteWidget.value = node._stylePalette.length ? JSON.stringify(node._stylePalette) : "";
      }

      function commit() { serialize(); renderPanel(); drawCanvas(); updateTokens(); }

      function removeBox(i) {
        node._boxes.splice(i, 1);
        if (node._boxes.length === 0) node._activeIdx = -1;
        else if (i <= node._activeIdx) node._activeIdx = Math.max(0, node._activeIdx - 1);
      }

      // ── pointer interaction ──
      canvasEl.addEventListener("mousedown", (e) => {
        if (e.button === 2) {            // right-click delete
          e.preventDefault();
          const hit = hitTest(mouseN(e));
          if (hit) removeBox(hit.index);
          else if (node._activeIdx >= 0) removeBox(node._activeIdx);
          commit(); fitNode();
          return;
        }
        if (e.button !== 0) return;
        canvasEl.focus();                // so Delete/Backspace targets this editor
        node._hoverTitle = null;         // clear hover highlight while interacting
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
        if (ti !== node._hoverTitle) { node._hoverTitle = ti; drawCanvas(); }
        if (ti !== null) { canvasEl.style.cursor = "pointer"; return; }
        const hit = hitTest(mN);
        canvasEl.style.cursor = hit ? (cursorForBboxMode(hit.mode) || "crosshair") : "crosshair";
      });
      canvasEl.addEventListener("mouseleave", () => {
        if (node._hoverTitle !== null) { node._hoverTitle = null; drawCanvas(); }
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
        ta.style.borderColor = boxColor(idx);
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

      // Delete/Backspace removes the active region (canvas must be focused; stop the
      // event so LiteGraph doesn't delete the whole node).
      canvasEl.addEventListener("keydown", (e) => {
        if ((e.key === "Delete" || e.key === "Backspace") && !node._drawing && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          removeBox(node._activeIdx);
          commit(); fitNode();
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
      clearBtn.addEventListener("mousedown", (e) => e.stopPropagation());
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
        tokenSpan.textContent = "~" + Math.ceil(buildCaption().length / 4) + " tok";
      }
      async function doCopy() {
        const txt = buildCaption();
        try { await navigator.clipboard.writeText(txt); copyBtn.textContent = "Copied"; setTimeout(() => (copyBtn.textContent = "Copy"), 900); }
        catch (e) { window.prompt("Copy the caption JSON:", txt); }
      }
      copyBtn.addEventListener("mousedown", (e) => e.stopPropagation());
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
      async function doImport() {
        let cap = null, txt = "";
        try { txt = (await navigator.clipboard.readText() || "").trim(); cap = tryParseCaption(txt); } catch (e) {}
        if (!cap) { txt = (window.prompt("Paste Ideogram 4 caption JSON:", "") || "").trim(); cap = tryParseCaption(txt); }
        if (!cap) { if (txt) alert("Not a valid Ideogram 4 caption JSON (needs 'compositional_deconstruction')."); return; }
        closeInlineEditor();
        applyCaption(cap);
        syncCanvasToDims(); commit(); rebuildStylePalette(); fitNode();
      }
      importBtn.addEventListener("mousedown", (e) => e.stopPropagation());
      importBtn.addEventListener("click", doImport);

      // ── property panel ──
      function stopProp(el) {
        for (const ev of ["mousedown", "pointerdown", "wheel"]) el.addEventListener(ev, (e) => e.stopPropagation());
      }
      // Color swatches: onEdit on change, onStruct on add/remove. Shared by both palettes.
      function buildSwatchRow(container, arr, max, onEdit, onStruct) {
        arr.forEach((hex, i) => {
          const sw = document.createElement("div");
          sw.className = "kjideo-sw";
          sw.style.background = hex;
          sw.title = "Click to edit · right-click to remove";
          const inp = document.createElement("input");
          inp.type = "color"; inp.value = hex;
          stopProp(sw);
          sw.addEventListener("click", () => inp.click());
          sw.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); arr.splice(i, 1); onStruct(); });
          inp.addEventListener("input", () => { arr[i] = inp.value; sw.style.background = inp.value; onEdit(); });
          sw.appendChild(inp);
          container.appendChild(sw);
        });
        if (arr.length < max) {
          const add = document.createElement("button");
          add.className = "kjideo-btn"; add.textContent = "+";
          stopProp(add);
          add.addEventListener("click", () => { arr.push("#ffffff"); onStruct(); });
          container.appendChild(add);
        }
      }

      function rebuildStylePalette() {
        while (styleBar.children.length > 1) styleBar.removeChild(styleBar.lastChild);
        buildSwatchRow(styleBar, node._stylePalette, MAX_STYLE_COLORS,
          () => { serialize(); drawCanvas(); },
          () => { serialize(); drawCanvas(); rebuildStylePalette(); fitNode(); });
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
        const col = boxColor(node._activeIdx);
        hint.innerHTML = `Editing <b style="color:${col}">region ${node._activeIdx + 1}</b> · dbl-click to edit · alt-click overlap · right-click/del to remove`;

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
            function () { b.text = this.value; serialize(); drawCanvas(); updateTokens(); }));
        }

        // desc — default ~3x the single-line min height
        panel.appendChild(makeArea("desc", b.desc, "description of this region",
          function () { b.desc = this.value; serialize(); drawCanvas(); updateTokens(); }, 110));

        // palette
        const palRow = document.createElement("div");
        palRow.className = "kjideo-row";
        const pl = document.createElement("span"); pl.textContent = "colors:"; palRow.appendChild(pl);
        b.palette = b.palette || [];
        buildSwatchRow(palRow, b.palette, MAX_ELEM_COLORS,
          () => { serialize(); drawCanvas(); }, commit);
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

      chainCallback(node, "onRemoved", function () {
        closeInlineEditor();
        for (const ro of node._areaObservers) ro.disconnect();
        node._areaObservers = [];
      });

      // ── restore on load ──
      chainCallback(node, "onConfigure", function () {
        if (elementsWidget?.value) {
          try {
            const parsed = JSON.parse(elementsWidget.value);
            if (Array.isArray(parsed)) {
              node._boxes = parsed.filter((b) => b && typeof b.x === "number");
              node._activeIdx = node._boxes.length ? 0 : -1;
            }
          } catch (e) {}
        }
        if (stylePaletteWidget?.value) {
          try {
            const sp = JSON.parse(stylePaletteWidget.value);
            if (Array.isArray(sp)) node._stylePalette = sp.filter((c) => typeof c === "string");
          } catch (e) {}
        }
        hideDataWidgets();
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
