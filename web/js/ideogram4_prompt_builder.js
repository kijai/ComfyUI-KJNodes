import { chainCallback, addMiddleClickPan, addWheelPassthrough, cursorForBboxMode, watchImageInputs, captureVideoFrame } from './utility.js';
const { app } = window.comfyAPI.app;

const HANDLE = 8;            // hit radius (canvas px) for corners/edges
const MAX_ELEM_COLORS = 5;   // Ideogram 4 per-element palette cap
const MAX_STYLE_COLORS = 16; // Ideogram 4 style palette cap
let copiedBoxes = null;      // internal clipboard for copy/paste of regions (array; shared across nodes)

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

// Parse a clipboard string into a normalized #rrggbb, or null if it isn't a color.
function parseColorString(s) {
  if (!s) return null;
  s = s.trim();
  let m = s.match(/^#?([0-9a-fA-F]{6})$/);
  if (m) return "#" + m[1].toLowerCase();
  m = s.match(/^#?([0-9a-fA-F]{3})$/);
  if (m) { const h = m[1]; return ("#" + h[0] + h[0] + h[1] + h[1] + h[2] + h[2]).toLowerCase(); }
  m = s.match(/^rgba?\(\s*(\d+)\s*[, ]\s*(\d+)\s*[, ]\s*(\d+)/i);
  if (m) {
    const h2 = (n) => Math.max(0, Math.min(255, parseInt(n, 10))).toString(16).padStart(2, "0");
    return "#" + h2(m[1]) + h2(m[2]) + h2(m[3]);
  }
  return null;
}
// The palette swatch under the cursor + a setter, so Ctrl+V pastes a color onto it
// and Ctrl+C copies its hex.
let hoveredSwatch = null;
document.addEventListener("keydown", (e) => {
  if (!hoveredSwatch) return;
  const ctrl = e.ctrlKey || e.metaKey;
  if (!ctrl) return;
  const key = e.key.toLowerCase();
  if (key !== "v" && key !== "c") return;
  const ae = document.activeElement;          // don't steal copy/paste from a focused text field
  if (ae && (ae.tagName === "TEXTAREA" || ae.tagName === "INPUT" || ae.isContentEditable)) return;
  const target = hoveredSwatch;
  if (!target.sw || !target.sw.isConnected) return;
  e.preventDefault(); e.stopPropagation();
  if (key === "c") {
    navigator.clipboard?.writeText?.(target.sw.dataset.hex || "").catch(() => {});
  } else {
    navigator.clipboard?.readText?.().then((txt) => {
      const c = parseColorString(txt);
      if (c && target.sw && target.sw.isConnected) target.setColor(c);
    }).catch(() => {});
  }
}, true);

// The node whose canvas is under the cursor — lets H toggle box visibility on hover (no click needed).
let hoveredCanvasNode = null;
document.addEventListener("keydown", (e) => {
  if (!hoveredCanvasNode) return;
  if (e.ctrlKey || e.metaKey || e.altKey || (e.key !== "h" && e.key !== "H")) return;
  const ae = document.activeElement;
  if (ae && (ae.tagName === "TEXTAREA" || ae.tagName === "INPUT" || ae.isContentEditable)) return;
  e.preventDefault(); e.stopPropagation();
  hoveredCanvasNode._toggleHideBoxes?.();
}, true);

function injectStyle() {
  if (document.getElementById("kjideo-style")) return;
  const s = document.createElement("style");
  s.id = "kjideo-style";
  s.textContent = `
    .kjideo-wrap { display:flex; flex-direction:column; overflow:hidden; position:relative; pointer-events:auto; gap:4px; }
    .kjideo-canvas { cursor:crosshair; display:block; width:100%; height:auto; flex:0 0 auto; background:#1a1a1a; border-radius:4px; outline:none; touch-action:none; }
    .kjideo-bar { display:flex; align-items:center; gap:6px; font:11px sans-serif; color:#aaa; user-select:none; padding:0 2px; flex:0 0 auto; }
    .kjideo-panel { display:flex; flex-direction:column; gap:5px; padding:6px; background:#262626; border-radius:4px; font:11px sans-serif; color:#bbb; flex:0 0 auto; }
    .kjideo-row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
    .kjideo-btn { background:#333; border:1px solid #555; border-radius:4px; color:#bbb; font:11px sans-serif; cursor:pointer; padding:2px 8px; line-height:16px; white-space:nowrap; flex-shrink:0; }
    .kjideo-btn:hover { border-color:#46b4e6; color:#fff; }
    .kjideo-btn.active { border-color:#46b4e6; color:#46b4e6; background:#2a3a42; }
    .kjideo-area { width:100%; box-sizing:border-box; background:#1d1d1d; border:1px solid #444; border-radius:4px; color:#ddd; font:13px monospace; padding:4px 6px; resize:vertical; min-height:36px; }
    .kjideo-sw { width:20px; height:20px; border:1px solid #666; border-radius:3px; cursor:pointer; flex-shrink:0; position:relative; touch-action:none; transition:transform .18s ease, box-shadow .12s ease, opacity .12s ease; }
    .kjideo-sw:hover { transform:scale(1.2); box-shadow:0 0 0 2px #46b4e6; z-index:3; }
    .kjideo-sw.dragging { opacity:.4; box-shadow:0 0 0 2px #46b4e6; }
    body.kjideo-dragging, body.kjideo-dragging * { cursor:move !important; }
    .kjideo-sw input { position:absolute; opacity:0; width:0; height:0; pointer-events:none; }
    .kjideo-inline { position:absolute; box-sizing:border-box; background:rgba(18,18,18,0.92); border:2px solid #46b4e6; border-radius:3px; color:#fff; font:13px monospace; padding:3px 4px; resize:none; outline:none; z-index:10; }
    .kjideo-bbox { width:128px; box-sizing:border-box; background:#1d1d1d; border:1px solid #444; border-radius:4px; color:#bbb; font:11px monospace; padding:2px 5px; }
    .kjideo-bbox:focus { border-color:#46b4e6; outline:none; color:#fff; }
    .kjideo-menu { position:fixed; z-index:10000; background:#262626; border:1px solid #555; border-radius:6px; padding:4px; box-shadow:0 6px 20px rgba(0,0,0,0.55); font:12px sans-serif; color:#ddd; max-height:60vh; overflow-y:auto; min-width:210px; max-width:340px; }
    .kjideo-mhdr { font:11px sans-serif; color:#888; padding:2px 6px 4px; user-select:none; }
    .kjideo-lrow { display:flex; align-items:center; gap:6px; padding:3px 5px; border-radius:4px; cursor:move; user-select:none; touch-action:none; transition:transform .18s ease, box-shadow .12s ease, opacity .12s ease, background .12s; }
    .kjideo-lrow:hover { background:#333; }
    .kjideo-lrow.active { background:#2a3a42; box-shadow:inset 0 0 0 1px #46b4e6; }
    .kjideo-lrow.dragging { opacity:.4; box-shadow:0 0 0 2px #46b4e6; background:#333; }
    .kjideo-lsw { width:16px; height:16px; border-radius:3px; border:1px solid #666; flex:0 0 auto; }
    .kjideo-lnum { font:bold 11px monospace; color:#888; flex:0 0 auto; width:18px; }
    .kjideo-ltext { flex:1 1 auto; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .kjideo-ltext.empty { color:#777; font-style:italic; }
    .kjideo-lbtn { background:none; border:none; color:#999; cursor:pointer; font:13px sans-serif; line-height:1; padding:2px 5px; border-radius:3px; flex:0 0 auto; }
    .kjideo-lbtn:hover { color:#fff; background:#444; }
    .kjideo-lbtn.del:hover { color:#fff; background:#a33; }
    .kjideo-fs { position:fixed; inset:0; z-index:9000; background:rgba(0,0,0,0.72); display:flex; align-items:center; justify-content:center; }
    .kjideo-fs-inner { position:relative; width:88vw; height:90vh; background:#1a1a1a; border:1px solid #444; border-radius:8px; box-shadow:0 12px 48px rgba(0,0,0,0.6); padding:12px; box-sizing:border-box; }
    .kjideo-fs-inner .kjideo-wrap { height:100%; }
    .kjideo-bgmenu { padding:7px; display:flex; flex-direction:column; gap:7px; min-width:170px; }
    .kjideo-bgrow { display:flex; align-items:center; gap:8px; }
    .kjideo-bglbl { color:#888; font:11px sans-serif; flex:0 0 auto; min-width:62px; }
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
      node._placing = false;   // duplicate-placement mode: active box follows the cursor until clicked
      node._dragMode = null;
      node._dragStartN = null; // mouse-down point, normalized
      node._boxAtStart = null; // active box snapshot at drag start
      node._selection = new Set();   // selected box indices (multi-select); always contains _activeIdx
      node._groupStart = null;       // {idx: {x,y,w,h}} snapshot of all selected boxes at drag start
      node._pendingCollapse = -1;    // box to collapse selection to if a click (not drag) on a multi-selection
      node._marquee = null;          // {x0,y0,x,y} rubber-band selection rect (shift-drag), normalized
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
      const guideSel = document.createElement("select");
      guideSel.className = "kjideo-btn"; guideSel.style.cssText = "flex:0 0 auto;";
      guideSel.title = "Composition guide overlay (editor view only)";
      for (const [val, label] of [["none", "no guide"], ["thirds", "thirds"], ["grid", "grid"], ["golden", "golden ratio"], ["spiral", "golden spiral"]]) {
        const o = document.createElement("option"); o.value = val; o.textContent = label; guideSel.appendChild(o);
      }
      guideSel.value = node.properties.guide || "none";
      stopProp(guideSel);
      guideSel.addEventListener("change", () => { node.properties.guide = guideSel.value; drawCanvas(); });
      const fsBtn = document.createElement("button");
      fsBtn.className = "kjideo-btn"; fsBtn.textContent = "⛶";
      fsBtn.title = "Open in a larger window (Esc to close)";
      stopProp(fsBtn);
      fsBtn.addEventListener("click", () => node._fullscreen ? exitFs() : enterFs());
      // Group the background/guide controls into one popup to keep the toolbar tidy.
      const bgBtn = document.createElement("button");
      bgBtn.className = "kjideo-btn"; bgBtn.textContent = "Background ▾";
      bgBtn.title = "Background & guides: live preview, grab/clear, brightness, composition guide";
      stopProp(bgBtn);
      const GRID_INV = 130;       // slider shows cell SIZE: position = GRID_INV - divisions (right = larger cells)
      const gridSlider = document.createElement("input");
      gridSlider.type = "range"; gridSlider.min = "2"; gridSlider.max = "128"; gridSlider.step = "1";
      gridSlider.value = GRID_INV - (node.properties.gridSize || 10);
      gridSlider.style.cssText = "width:90px;flex:0 0 auto;";
      gridSlider.title = "Grid cell size (drag right for larger cells); also the snap step";
      stopProp(gridSlider);
      gridSlider.addEventListener("input", () => { node.properties.gridSize = GRID_INV - parseInt(gridSlider.value, 10); drawCanvas(); });
      const snapLabel = document.createElement("label");
      snapLabel.style.cssText = "display:flex;align-items:center;gap:4px;cursor:pointer;";
      const snapChk = document.createElement("input");
      snapChk.type = "checkbox"; snapChk.checked = !!node.properties.snap;
      snapChk.addEventListener("change", () => { node.properties.snap = snapChk.checked; });
      snapLabel.appendChild(snapChk); snapLabel.appendChild(document.createTextNode("snap to grid"));
      const guideColor = document.createElement("input");
      guideColor.type = "color"; guideColor.value = node.properties.guideColor || "#ffffff";
      guideColor.style.cssText = "width:32px;height:20px;flex:0 0 auto;padding:0;border:1px solid #555;background:none;";
      stopProp(guideColor);
      guideColor.addEventListener("input", () => { node.properties.guideColor = guideColor.value; drawCanvas(); });
      const opacitySlider = document.createElement("input");
      opacitySlider.type = "range"; opacitySlider.min = "0"; opacitySlider.max = "100"; opacitySlider.step = "1";
      opacitySlider.value = node.properties.guideOpacity == null ? 100 : node.properties.guideOpacity;
      opacitySlider.style.cssText = "width:90px;flex:0 0 auto;";
      opacitySlider.title = "Guide/grid line opacity";
      stopProp(opacitySlider);
      opacitySlider.addEventListener("input", () => { node.properties.guideOpacity = parseInt(opacitySlider.value, 10); drawCanvas(); });
      const bgMenu = document.createElement("div");
      bgMenu.className = "kjideo-menu kjideo-bgmenu";
      bgMenu.style.display = "none";
      const bgRow = (labelText, el) => {
        const r = document.createElement("div"); r.className = "kjideo-bgrow";
        if (labelText) { const l = document.createElement("span"); l.className = "kjideo-bglbl"; l.textContent = labelText; r.appendChild(l); }
        r.appendChild(el); bgMenu.appendChild(r);
      };
      bgRow("", liveLabel); bgRow("", grabBtn); bgRow("Brightness", bgSlider);
      bgRow("Guide", guideSel); bgRow("Grid size", gridSlider); bgRow("", snapLabel);
      bgRow("Line color", guideColor); bgRow("Line opacity", opacitySlider);
      document.body.appendChild(bgMenu);
      node._bgMenu = bgMenu;
      function closeBgMenu() {
        bgMenu.style.display = "none";
        if (node._bgMenuDismiss) {
          document.removeEventListener("pointerdown", node._bgMenuDismiss, true);
          document.removeEventListener("mousedown", node._bgMenuDismiss, true);
          node._bgMenuDismiss = null;
        }
      }
      bgBtn.addEventListener("click", () => {
        if (bgMenu.style.display !== "none") { closeBgMenu(); return; }
        bgMenu.style.display = "";
        const r = bgBtn.getBoundingClientRect();
        bgMenu.style.left = Math.max(4, Math.min(r.left, window.innerWidth - bgMenu.offsetWidth - 4)) + "px";
        bgMenu.style.top = Math.min(r.bottom + 4, window.innerHeight - bgMenu.offsetHeight - 4) + "px";
        node._bgMenuDismiss = (e) => { if (!bgMenu.contains(e.target) && e.target !== bgBtn) closeBgMenu(); };
        setTimeout(() => {
          document.addEventListener("pointerdown", node._bgMenuDismiss, true);
          document.addEventListener("mousedown", node._bgMenuDismiss, true);
        }, 0);
      });
      bar.appendChild(hint); bar.appendChild(bgBtn); bar.appendChild(tokenSpan); bar.appendChild(copyBtn); bar.appendChild(importBtn); bar.appendChild(fsBtn); bar.appendChild(clearBtn);
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
      canvasEl.title = "Drag to draw · Ctrl-drag force-draw over a box · click to select · shift-drag marquee-select · " +
        "shift-click toggle · drag a group to move all · alt-click overlap · dbl-click edit · right-click region list · " +
        "Del remove (all selected) · Ctrl/Cmd+C/V/D copy/paste/duplicate · H hide boxes (view)";
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

      // DOM widgets are HTML layered over the canvas; ComfyUI only repositions/clips them
      // during a foreground draw. When the node returns from off-screen the element can
      // briefly float over the canvas until the next draw — force one when it re-enters view.
      try {
        node._visObserver = new IntersectionObserver((entries) => {
          if (entries.some((en) => en.isIntersecting)) {
            app.canvas?.setDirtyCanvas?.(true, true);
            drawCanvas();
          }
        });
        node._visObserver.observe(wrap);
      } catch (e) {}

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
        if (node._fullscreen) { fitFsCanvas(); return; }     // in the popup, size the canvas, not the node
        recalcWidgetHeight();
        // computeSize (stable min-heights), not last_y which creeps with growable widgets above.
        const minH = node.computeSize()[1];
        if (node.size[1] < minH) node.setSize([node.size[0], minH]);
      }

      // ── larger popup window (relocates the editor into a centered overlay) ──
      function fitFsCanvas() {                                // fit the fixed-aspect canvas into the popup
        if (!node._fullscreen || !node._fsInner) return;
        const availW = node._fsInner.clientWidth - 24;
        const availH = node._fsInner.clientHeight - 24 - 16 - bar.offsetHeight - styleBar.offsetHeight - panel.offsetHeight;
        const aspect = (wWidget?.value || 1) / (hWidget?.value || 1);
        let cw = availW, ch = cw / aspect;
        if (ch > availH) { ch = Math.max(60, availH); cw = ch * aspect; }
        if (cw > availW) { cw = availW; ch = cw / aspect; }
        canvasEl.style.width = Math.round(cw) + "px";
        canvasEl.style.height = Math.round(ch) + "px";
        canvasEl.style.alignSelf = "center";
        drawCanvas();
      }
      function onFsEsc(e) { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); exitFs(); } }
      function enterFs() {
        if (node._fullscreen) return;
        node._fullscreen = true;
        node._wrapHome = wrap.parentNode;                    // where to return the editor on close
        const ov = document.createElement("div"); ov.className = "kjideo-fs";
        const inner = document.createElement("div"); inner.className = "kjideo-fs-inner";
        inner.appendChild(wrap);                             // move the SAME editor element in (keeps all state)
        ov.appendChild(inner);
        ov.addEventListener("mousedown", (e) => { if (e.target === ov) exitFs(); });  // backdrop closes
        document.body.appendChild(ov);
        node._fsOverlay = ov; node._fsInner = inner;
        if (node.ideoEditor) node.ideoEditor.hidden = true;  // stop ComfyUI from reclaiming the element
        document.addEventListener("keydown", onFsEsc, true);
        window.addEventListener("resize", fitFsCanvas);
        requestAnimationFrame(fitFsCanvas);
      }
      function exitFs() {
        if (!node._fullscreen) return;
        node._fullscreen = false;
        document.removeEventListener("keydown", onFsEsc, true);
        window.removeEventListener("resize", fitFsCanvas);
        canvasEl.style.width = ""; canvasEl.style.height = ""; canvasEl.style.alignSelf = "";  // restore CSS sizing
        if (node._wrapHome) node._wrapHome.appendChild(wrap);
        node._fsOverlay?.remove(); node._fsOverlay = null; node._fsInner = null;
        if (node.ideoEditor) node.ideoEditor.hidden = false;
        if (node.graph) node.graph.setDirtyCanvas(true, true);
        requestAnimationFrame(() => { fitNode(); drawCanvas(); });
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
      // drawn last (on top), then the rest by index low→high (index 0 = front).
      function boxesAt(mN) {
        const rx = HANDLE / logW(), ry = HANDLE / logH();
        const res = [];
        for (let i = 0; i < node._boxes.length; i++) {
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

      // ── grid / guides ──
      function gridN() { return Math.max(2, Math.min(128, node.properties.gridSize || 10)); }
      function guideStroke(a) {
        const c = hexRgb(node.properties.guideColor || "#ffffff") || { r: 255, g: 255, b: 255 };
        const op = (node.properties.guideOpacity == null ? 100 : node.properties.guideOpacity) / 100;
        return `rgba(${c.r},${c.g},${c.b},${(a * op).toFixed(3)})`;
      }
      // Snap a box's free edges to the (square) grid when snap-to-grid is on.
      function snapBox(b, mode) {
        if (!node.properties.snap) return b;
        const W = logW(), H = logH(), cell = Math.min(W, H) / gridN();
        const sx = cell / W, sy = cell / H, sn = (v, s) => Math.round(v / s) * s;
        let { x, y, w, h } = b;
        if (mode === "move") { x = sn(x, sx); y = sn(y, sy); }
        else { const x2 = sn(x + w, sx), y2 = sn(y + h, sy); x = sn(x, sx); y = sn(y, sy); w = x2 - x; h = y2 - y; }
        return normalizeBox({ ...b, x, y, w, h });
      }

      // ── drawing ──
      // Golden spiral: largest φ-rectangle fitting the canvas, subdivided into squares with quarter arcs.
      function goldenSpiral(W, H) {
        const phi = 1.6180339887;
        let w, h;
        if (W >= H) { if (W / H >= phi) { h = H; w = H * phi; } else { w = W; h = W / phi; } }
        else { if (H / W >= phi) { w = W; h = W * phi; } else { h = H; w = H / phi; } }
        let x = (W - w) / 2, y = (H - h) / 2;
        ctx.save();
        ctx.strokeStyle = guideStroke(0.25); ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, w, h);
        ctx.strokeStyle = guideStroke(0.6); ctx.lineWidth = 1.5;
        let phase = w >= h ? 0 : 1;
        for (let i = 0; i < 12 && w > 1 && h > 1; i++, phase = (phase + 1) % 4) {
          const s = Math.min(w, h);
          let cx, cy, a0, a1;
          if (phase === 0)      { cx = x + s;     cy = y + s;     a0 = Math.PI;       a1 = Math.PI * 1.5; x += s; w -= s; }
          else if (phase === 1) { cx = x;         cy = y + s;     a0 = Math.PI * 1.5; a1 = Math.PI * 2;   y += s; h -= s; }
          else if (phase === 2) { cx = x + w - s; cy = y;         a0 = 0;             a1 = Math.PI * 0.5; w -= s; }
          else                  { cx = x + w;     cy = y + h - s; a0 = Math.PI * 0.5; a1 = Math.PI;       h -= s; }
          ctx.beginPath(); ctx.arc(cx, cy, s, a0, a1); ctx.stroke();
        }
        ctx.restore();
      }
      // Composition guide overlay (rule of thirds / grid / golden ratio / spiral), drawn on the bg.
      function drawGuide(W, H) {
        const kind = node.properties.guide;
        if (!kind || kind === "none") return;
        if (kind === "spiral") { goldenSpiral(W, H); return; }
        ctx.save();
        ctx.lineWidth = 1;
        if (kind === "grid") {                          // square cells (gridSize across the short edge), any aspect
          ctx.strokeStyle = guideStroke(0.24);
          const cell = Math.min(W, H) / gridN();
          for (let px = cell; px < W - 0.5; px += cell) { const X = Math.round(px) + 0.5; ctx.beginPath(); ctx.moveTo(X, 0); ctx.lineTo(X, H); ctx.stroke(); }
          for (let py = cell; py < H - 0.5; py += cell) { const Y = Math.round(py) + 0.5; ctx.beginPath(); ctx.moveTo(0, Y); ctx.lineTo(W, Y); ctx.stroke(); }
          ctx.restore(); return;
        }
        const v = [], h = [];                           // thirds / golden: per-axis fractions (correct at any aspect)
        if (kind === "thirds") { v.push(1 / 3, 2 / 3); h.push(1 / 3, 2 / 3); }
        else if (kind === "golden") { const g = 1 / 1.6180339887; v.push(1 - g, g); h.push(1 - g, g); }
        ctx.strokeStyle = guideStroke(kind === "golden" ? 0.45 : 0.28);
        for (const fx of v) { const px = Math.round(fx * W) + 0.5; ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke(); }
        for (const fy of h) { const py = Math.round(fy * H) + 0.5; ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke(); }
        ctx.restore();
      }
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
        drawGuide(W, H);                                          // composition guide overlay
        if (node._hideBoxes) return;                              // H: temporary background-only view
        // selection highlight only when the editor is focused or the node is selected
        const showSel = node._focused || node._selected;
        const aIdx = showSel ? node._activeIdx : -1;
        const selSet = new Set(showSel ? node._selection : []);
        if (aIdx >= 0) selSet.add(aIdx);
        // index 0 = front (drawn last); selected drawn above non-selected, active last of all
        const nonSel = node._boxes.map((_, i) => i).filter((i) => !selSet.has(i)).reverse();
        const selOthers = node._boxes.map((_, i) => i).filter((i) => selSet.has(i) && i !== aIdx).reverse();
        const order = [...nonSel, ...selOthers];
        if (aIdx >= 0 && aIdx < node._boxes.length) order.push(aIdx);
        const tagR = tagRects();                              // collision-avoided tag positions
        for (const i of order) {
          const b = node._boxes[i], active = i === aIdx, selected = selSet.has(i);
          const pal = (b.palette || []).filter(Boolean);
          const col = pal.length ? pal[0] : "#8c8c8c";       // box color = first palette color, else neutral grey
          const { x1, y1, x2, y2 } = toPx(b);
          const w = x2 - x1, h = y2 - y1;
          const hovered = i === node._hoverBox || selected;  // selected boxes stay highlighted
          if (selected) {                                    // opaque backing so contents read clearly over boxes behind
            ctx.fillStyle = "rgba(26,26,26,0.88)";
            ctx.fillRect(x1, y1, w, h);
          }
          ctx.fillStyle = col + (hovered ? "3a" : "22");     // tint of the box color
          ctx.fillRect(x1, y1, w, h);
          if (b.nobbox) ctx.setLineDash([6, 4]);             // unplaced (no bbox in source)
          const lw = selected ? 2 : (hovered ? 1.5 : 1);
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
          if (body && node.properties.showBoxText !== false) {  // toggle: show text in boxes
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
          if (selected) {                                    // orange selection ring on top (above strip/tag): solid = primary, dashed = others
            const olw = active ? 2 : 1;
            ctx.strokeStyle = "#ff8c00"; ctx.lineWidth = olw;
            if (!active) ctx.setLineDash([5, 3]);
            ctx.strokeRect(x1 + olw / 2, y1 + olw / 2, w - olw, h - olw);
            ctx.setLineDash([]);
          }
        }
        if (node._marquee && node._marqueeActive) {           // rubber-band selection rectangle
          const r = marqueeRect();
          const mx = r.x * W, my = r.y * H, mw = r.w * W, mh = r.h * H;
          ctx.fillStyle = "rgba(70,180,230,0.12)";
          ctx.fillRect(mx, my, mw, mh);
          ctx.strokeStyle = "#46b4e6"; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
          ctx.strokeRect(mx + 0.5, my + 0.5, mw - 1, mh - 1);
          ctx.setLineDash([]);
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
        node._selection = new Set();
        if (node._boxes.length === 0) node._activeIdx = -1;
        else if (i <= node._activeIdx) node._activeIdx = Math.max(0, node._activeIdx - 1);
        if (node._activeIdx >= 0) node._selection.add(node._activeIdx);
      }
      // Delete every selected box (or the active one) and reset the selection.
      function removeSelected() {
        const idxs = [...node._selection].sort((a, b) => b - a);
        if (!idxs.length && node._activeIdx >= 0) idxs.push(node._activeIdx);
        if (!idxs.length) return;
        for (const i of idxs) node._boxes.splice(i, 1);
        node._selection = new Set();
        node._activeIdx = node._boxes.length ? Math.min(idxs[idxs.length - 1], node._boxes.length - 1) : -1;
        if (node._activeIdx >= 0) node._selection.add(node._activeIdx);
      }
      // Replace the selection with a single box (used by clicks / programmatic selects).
      function selectOnly(idx) {
        node._activeIdx = idx;
        node._selection = idx >= 0 ? new Set([idx]) : new Set();
      }

      // ── pointer interaction ──
      canvasEl.addEventListener("pointerdown", (e) => {
        if (node._placing) {             // drop the duplicate being placed
          if (e.button === 0) { placeFollower(mouseN(e)); finishPlacing(); }
          else cancelPlacing();
          e.preventDefault(); e.stopPropagation();
          return;
        }
        if (e.button !== 0) return;
        if (node._hideBoxes) return;     // view-only while boxes are hidden (H)
        canvasEl.focus();                // so Delete/Backspace targets this editor
        node._hoverTitle = null; node._hoverBox = null;  // clear hover highlight while interacting
        const mN = mouseN(e);
        // Ctrl/Cmd forces drawing a new box even when starting over an existing one.
        const hit = (e.ctrlKey || e.metaKey) ? null : pickForSelection(mN, e.altKey);
        // Touch double-tap opens the inline editor (dblclick may not fire on touch).
        if (e.pointerType && e.pointerType !== "mouse") {
          const last = node._lastTap, now = e.timeStamp;
          if (hit && last && now - last.t < 350 && Math.abs(mN.x - last.x) < 0.03 && Math.abs(mN.y - last.y) < 0.03) {
            node._lastTap = null; openInlineEditor(hit.index);
            e.preventDefault(); e.stopPropagation(); return;
          }
          node._lastTap = { t: now, x: mN.x, y: mN.y };
        }
        node._pendingCollapse = -1;
        node._groupStart = null;
        if (e.shiftKey) {                              // shift-drag = marquee select; shift-click = toggle
          startMarquee(mN, hit ? hit.index : -1);
          e.preventDefault(); e.stopPropagation();
          return;
        }
        if (hit) {
          if (!node._selection.has(hit.index)) node._selection = new Set([hit.index]);  // outside selection → pick it
          else if (node._selection.size > 1) node._pendingCollapse = hit.index;          // click (no drag) collapses
          node._activeIdx = hit.index;
          node._dragMode = hit.mode;
          node._boxAtStart = { ...node._boxes[hit.index] };
          if (node._selection.size > 1) {                  // snapshot the whole group for group move/resize
            node._groupStart = {};
            for (const i of node._selection) node._groupStart[i] = { ...node._boxes[i] };
          }
        } else {
          node._dragMode = "draw";
          const nb = { x: mN.x, y: mN.y, w: 0, h: 0, type: "obj", text: "", desc: "", palette: [] };
          node._boxes.push(nb);
          node._activeIdx = node._boxes.length - 1;
          node._selection = new Set([node._activeIdx]);
          node._boxAtStart = { ...nb };
        }
        node._drawing = true;
        node._dragStartN = mN;
        document.addEventListener("pointermove", onMove);
        document.addEventListener("pointerup", onUp);
        e.preventDefault(); e.stopPropagation();
        drawCanvas();   // panel rebuild/resize deferred to onUp so the canvas doesn't shift mid-drag
      });

      canvasEl.addEventListener("pointermove", (e) => {
        node._lastMouseN = mouseN(e);                        // track cursor for paste-under-cursor
        if (node._placing) { placeFollower(node._lastMouseN); return; }
        if (node._drawing || node._marquee || node._hideBoxes) return;
        const mN = mouseN(e);
        const force = e.ctrlKey || e.metaKey;               // Ctrl/Cmd = force-draw
        const ti = force ? null : titleAt(mN);
        const hit = force ? null : hitTest(mN);
        const hb = ti != null ? ti : (hit ? hit.index : null);
        if (ti !== node._hoverTitle || hb !== node._hoverBox) {
          node._hoverTitle = ti; node._hoverBox = hb; drawCanvas();
        }
        canvasEl.style.cursor = ti != null ? "pointer" : (hit ? (cursorForBboxMode(hit.mode) || "crosshair") : "crosshair");
      });
      canvasEl.addEventListener("pointerleave", () => {
        if (hoveredCanvasNode === node) hoveredCanvasNode = null;
        if (node._hoverTitle !== null || node._hoverBox !== null) {
          node._hoverTitle = null; node._hoverBox = null; drawCanvas();
        }
      });
      canvasEl.addEventListener("pointerenter", () => { hoveredCanvasNode = node; });
      // H (while hovering the canvas): temporary background-only view (not serialized).
      node._toggleHideBoxes = () => { node._hideBoxes = !node._hideBoxes; drawCanvas(); };

      // ── inline description editing (double-click a region) ──
      let inlineTa = null;
      function closeInlineEditor() {
        if (inlineTa) { inlineTa.remove(); inlineTa = null; }
      }
      function openInlineEditor(idx) {
        closeInlineEditor();
        const b = node._boxes[idx];
        if (!b) return;
        selectOnly(idx);
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
        if (node._hideBoxes) return;
        e.preventDefault(); e.stopPropagation();
        const cands = boxesAt(mouseN(e));     // edit the active box if it's under the cursor, else topmost
        const target = cands.find((c) => c.index === node._activeIdx) || cands[0];
        if (target) openInlineEditor(target.index);
      });

      // Snapshot the current selection (or the active box) for the clipboard.
      function copySelection() {
        const idxs = node._selection.size ? [...node._selection].sort((a, b) => a - b)
          : (node._activeIdx >= 0 ? [node._activeIdx] : []);
        return idxs.map((i) => JSON.parse(JSON.stringify(node._boxes[i])));
      }
      // Paste the clipboard regions as a group, centered under the cursor, keeping their layout.
      function pasteBoxes() {
        if (!copiedBoxes || !copiedBoxes.length) return;
        const clones = copiedBoxes.map((b) => JSON.parse(JSON.stringify(b)));
        let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
        for (const b of clones) {
          minx = Math.min(minx, b.x); miny = Math.min(miny, b.y);
          maxx = Math.max(maxx, b.x + b.w); maxy = Math.max(maxy, b.y + b.h);
        }
        const gw = maxx - minx, gh = maxy - miny;
        const m = node._lastMouseN;
        let tx = m ? m.x - gw / 2 : minx + 0.03;       // target group top-left
        let ty = m ? m.y - gh / 2 : miny + 0.03;
        tx = Math.max(0, Math.min(tx, 1 - gw));         // clamp group into the canvas
        ty = Math.max(0, Math.min(ty, 1 - gh));
        const dx = tx - minx, dy = ty - miny;
        const start = node._boxes.length;
        for (const b of clones) {
          b.x = clamp01(b.x + dx); b.y = clamp01(b.y + dy);
          delete b.nobbox;                              // pasted boxes are placed
          node._boxes.push(b);
        }
        node._selection = new Set();
        for (let i = start; i < node._boxes.length; i++) node._selection.add(i);
        node._activeIdx = node._boxes.length - 1;
        commit(); fitNode();
      }
      // Keyboard: Delete removes; Ctrl/Cmd C/V/D copy/paste/duplicate the active region.
      // Canvas must be focused; stop the event so LiteGraph doesn't act on the node.
      canvasEl.addEventListener("keydown", (e) => {
        if (node._placing) {
          if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancelPlacing(); }
          return;
        }
        if (node._drawing || node._hideBoxes) return;   // view-only while boxes are hidden (H)
        const ctrl = e.ctrlKey || e.metaKey;
        if ((e.key === "Delete" || e.key === "Backspace") && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          removeSelected(); commit(); fitNode();      // removes all selected (or the active one)
        } else if (ctrl && e.key === "c" && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          copiedBoxes = copySelection();
        } else if (ctrl && e.key === "v" && copiedBoxes) {
          e.preventDefault(); e.stopPropagation();
          pasteBoxes();
        } else if (ctrl && e.key === "d" && node._activeIdx >= 0) {
          e.preventDefault(); e.stopPropagation();
          copiedBoxes = copySelection();
          pasteBoxes();
        }
      });

      function onMove(e) {
        if (!node._drawing) return;
        const mN = mouseN(e);
        const dN = { x: mN.x - node._dragStartN.x, y: mN.y - node._dragStartN.y };
        if (Math.abs(dN.x) + Math.abs(dN.y) > 0.001) node._pendingCollapse = -1;  // it's a drag, not a click
        if (node._dragMode === "move" && node._groupStart) {
          let dx = dN.x, dy = dN.y;                   // clamp delta so the whole group stays in bounds
          for (const i in node._groupStart) {
            const s = node._groupStart[i];
            dx = Math.min(Math.max(dx, -s.x), 1 - s.w - s.x);
            dy = Math.min(Math.max(dy, -s.y), 1 - s.h - s.y);
          }
          if (node.properties.snap) {                 // snap the group's movement to whole grid cells
            const cell = Math.min(logW(), logH()) / gridN();
            dx = Math.round(dx / (cell / logW())) * (cell / logW());
            dy = Math.round(dy / (cell / logH())) * (cell / logH());
          }
          for (const i in node._groupStart) {
            const s = node._groupStart[i];
            node._boxes[i] = { ...s, x: s.x + dx, y: s.y + dy };
            delete node._boxes[i].nobbox;
          }
          drawCanvas(); updateBboxLabel();
          return;
        }
        if (node._groupStart && node._dragMode.startsWith("resize")) {
          // Scale every selected box by the primary's resize, about the handle's fixed edge.
          const a = node._boxAtStart, na = applyDrag(node._dragMode, a, dN);
          const suf = node._dragMode.slice(7);       // "tl"|"tr"|"bl"|"br"|"t"|"b"|"l"|"r"
          const scaleX = (suf.includes("l") || suf.includes("r")) && a.w > 0 ? Math.max(0.02, na.w / a.w) : 1;
          const scaleY = (suf.includes("t") || suf.includes("b")) && a.h > 0 ? Math.max(0.02, na.h / a.h) : 1;
          const ax = suf.includes("l") ? a.x + a.w : a.x;   // fixed (anchor) edges
          const ay = suf.includes("t") ? a.y + a.h : a.y;
          for (const i in node._groupStart) {
            const s = node._groupStart[i];
            node._boxes[i] = normalizeBox({
              ...s, x: ax + (s.x - ax) * scaleX, y: ay + (s.y - ay) * scaleY,
              w: s.w * scaleX, h: s.h * scaleY,
            });
            delete node._boxes[i].nobbox;
          }
          drawCanvas(); updateBboxLabel();
          return;
        }
        const nb = snapBox(applyDrag(node._dragMode, node._boxAtStart, dN), node._dragMode);
        delete nb.nobbox;            // moving/resizing places the element (gives it a bbox)
        node._boxes[node._activeIdx] = nb;
        drawCanvas(); updateBboxLabel();
      }
      function onUp() {
        if (!node._drawing) return;
        node._drawing = false;
        document.removeEventListener("pointermove", onMove);
        document.removeEventListener("pointerup", onUp);
        // a click (no drag) on empty space drops the placeholder box and deselects everything
        const b = node._boxes[node._activeIdx];
        if (b && (b.w < 0.005 || b.h < 0.005) && node._dragMode === "draw") {
          node._boxes.splice(node._activeIdx, 1);
          selectOnly(-1);
        } else if (node._pendingCollapse >= 0) {     // click (no drag) on a group member → keep only it
          selectOnly(node._pendingCollapse);
        }
        node._pendingCollapse = -1; node._groupStart = null;
        commit();
      }

      // ── marquee (rubber-band) selection: shift-drag ──
      function marqueeRect() {
        const m = node._marquee;
        return { x: Math.min(m.x0, m.x), y: Math.min(m.y0, m.y),
                 w: Math.abs(m.x - m.x0), h: Math.abs(m.y - m.y0) };
      }
      function rectsOverlap(r, b) {
        return r.x < b.x + b.w && r.x + r.w > b.x && r.y < b.y + b.h && r.y + r.h > b.y;
      }
      function startMarquee(mN, startHit) {
        node._marquee = { x0: mN.x, y0: mN.y, x: mN.x, y: mN.y };
        node._marqueeBase = new Set(node._selection);   // additive: marquee unions with what's selected
        node._marqueeStartHit = startHit;               // for the shift-click (no drag) toggle fallback
        node._marqueeActive = false;
        canvasEl.focus();
        document.addEventListener("pointermove", onMarqueeMove);
        document.addEventListener("pointerup", onMarqueeUp);
        drawCanvas();
      }
      function onMarqueeMove(e) {
        if (!node._marquee) return;
        const mN = mouseN(e);
        node._marquee.x = mN.x; node._marquee.y = mN.y;
        if (Math.abs(mN.x - node._marquee.x0) + Math.abs(mN.y - node._marquee.y0) > 0.01) node._marqueeActive = true;
        if (node._marqueeActive) {
          const r = marqueeRect();
          const sel = new Set(node._marqueeBase);
          node._boxes.forEach((b, i) => { if (rectsOverlap(r, b)) sel.add(i); });
          node._selection = sel;
          if (node._activeIdx < 0 || !sel.has(node._activeIdx)) node._activeIdx = sel.size ? [...sel][0] : node._activeIdx;
        }
        drawCanvas();
      }
      function onMarqueeUp() {
        document.removeEventListener("pointermove", onMarqueeMove);
        document.removeEventListener("pointerup", onMarqueeUp);
        if (!node._marqueeActive && node._marqueeStartHit >= 0) {   // shift-click on a box → toggle it
          const idx = node._marqueeStartHit;
          if (node._selection.has(idx) && node._selection.size > 1) {
            node._selection.delete(idx);
            if (node._activeIdx === idx) node._activeIdx = node._selection.values().next().value;
          } else {
            node._selection.add(idx); node._activeIdx = idx;
          }
        }
        node._marquee = null; node._marqueeActive = false;
        if (node._activeIdx >= 0 && !node._selection.has(node._activeIdx)) {
          node._activeIdx = node._selection.size ? [...node._selection][0] : -1;
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
        drawCanvas(); updateBboxLabel();
      }
      function startPlacing(srcIdx) {
        const src = node._boxes[srcIdx];
        if (!src) return;
        const nb = JSON.parse(JSON.stringify(src));
        delete nb.nobbox;
        node._boxes.push(nb);
        selectOnly(node._boxes.length - 1);
        node._placing = true;
        canvasEl.focus();
        canvasEl.style.cursor = "move";
        serialize(); renderPanel(); drawCanvas(); updateTokens();
      }
      function finishPlacing() {
        if (!node._placing) return;
        node._placing = false;
        canvasEl.style.cursor = "crosshair";
        commit(); fitNode();
      }
      function cancelPlacing() {
        if (!node._placing) return;
        node._placing = false;
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
        menu.className = "kjideo-menu";
        const hdr = document.createElement("div");
        hdr.className = "kjideo-mhdr";
        hdr.textContent = "Regions — top = front · click select · drag reorder";
        // 01 at the top of the list = front-most (drawn on top); see _draw / boxesAt.
        menu.appendChild(hdr);
        const list = document.createElement("div");
        menu.appendChild(list);
        node._layerMenu = menu;

        const renumber = () => Array.from(list.querySelectorAll(".kjideo-lrow")).forEach((row, k) => {
          row.querySelector(".kjideo-lnum").textContent = String(k + 1).padStart(2, "0");
        });
        function buildRows() {
          list.innerHTML = "";
          if (!node._boxes.length) {
            const empty = document.createElement("div");
            empty.className = "kjideo-mhdr"; empty.textContent = "No regions yet.";
            list.appendChild(empty);
            return;
          }
          node._boxes.forEach((b, i) => {
            const row = document.createElement("div");
            row.className = "kjideo-lrow" + (i === node._activeIdx ? " active" : "");
            row._box = b;
            const sw = document.createElement("div");
            sw.className = "kjideo-lsw";
            sw.style.background = (b.palette || []).find(Boolean) || "#8c8c8c";
            const num = document.createElement("span");
            num.className = "kjideo-lnum"; num.textContent = String(i + 1).padStart(2, "0");
            const txt = document.createElement("span");
            const label = rowLabel(b);
            txt.className = "kjideo-ltext" + (label ? "" : " empty");
            txt.textContent = label || (b.type === "text" ? "(text)" : "(empty)");
            txt.title = label;
            const dup = document.createElement("button");
            dup.className = "kjideo-lbtn"; dup.textContent = "⧉";
            dup.title = "Duplicate, then click on the canvas to place";
            const del = document.createElement("button");
            del.className = "kjideo-lbtn del"; del.textContent = "✕";
            del.title = "Delete region";
            row.append(sw, num, txt, dup, del);
            list.appendChild(row);

            row.addEventListener("click", () => {
              if (row._dragged) { row._dragged = false; return; }
              selectOnly(node._boxes.indexOf(b));
              commit();
              for (const r of list.querySelectorAll(".kjideo-lrow")) r.classList.toggle("active", r._box === b);
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
              removeBox(idx); commit(); fitNode();
              if (!node._boxes.length) { closeLayersMenu(); return; }
              buildRows();
            });
            // drag-reorder (vertical FLIP, mirrors the palette swatch reorder)
            row.addEventListener("pointerdown", (e) => {
              if (e.button !== 0 || e.target === dup || e.target === del) return;
              e.preventDefault(); e.stopPropagation();
              const sx = e.clientX, sy = e.clientY;
              let dragging = false;
              const move = (me) => {
                if (!dragging) {
                  if (Math.abs(me.clientX - sx) + Math.abs(me.clientY - sy) < 4) return;
                  dragging = true; row.classList.add("dragging"); document.body.classList.add("kjideo-dragging");
                }
                for (const other of list.querySelectorAll(".kjideo-lrow")) {
                  if (other === row) continue;
                  const r = other.getBoundingClientRect();
                  if (me.clientY >= r.top && me.clientY <= r.bottom) {
                    const ref = me.clientY > r.top + r.height / 2 ? other.nextSibling : other;
                    if (ref === row || ref === row.nextSibling) break;
                    const els = Array.from(list.querySelectorAll(".kjideo-lrow"));
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
                document.removeEventListener("pointermove", move);
                document.removeEventListener("pointerup", up);
                document.body.classList.remove("kjideo-dragging");
                if (dragging) {
                  row.classList.remove("dragging");
                  row._dragged = true;                             // suppress the trailing click
                  const active = node._boxes[node._activeIdx];
                  const order = Array.from(list.querySelectorAll(".kjideo-lrow")).map((el) => el._box);
                  if (order.length === node._boxes.length) node._boxes = order;
                  selectOnly(active ? node._boxes.indexOf(active) : -1);  // reorder invalidates multi-select indices
                  renumber();
                  commit();
                }
              };
              document.addEventListener("pointermove", move);
              document.addEventListener("pointerup", up);
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
        if (node._placing || node._hideBoxes) return;
        closeInlineEditor();
        openLayersMenu(e.clientX, e.clientY);
      });
      stopProp(clearBtn);
      clearBtn.addEventListener("click", () => {
        closeInlineEditor();
        node._boxes = []; node._activeIdx = -1; node._selection = new Set(); node._stylePalette = [];
        node._lastImported = "";
        commit(); rebuildStylePalette(); fitNode();
        // Write a unique "empty" marker into elements_data so the next run isn't cache-skipped
        // (ComfyUI caches on the input signature; an empty value would match the prior run and the
        // node wouldn't re-execute). The server treats a non-list value as empty, then re-pulls the
        // wired import per import_mode and repopulates the editor via ui.
        if (elementsWidget) elementsWidget.value = JSON.stringify({ _cleared: (node._clearSeq = (node._clearSeq || 0) + 1) });
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
        selectOnly(node._boxes.length ? 0 : -1);
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
        if (!capStr) return;
        // "always" mode re-applies even an unchanged import so the editor snaps back to the
        // authoritative JSON after edits; "when empty" keeps the guard so edits stick.
        const always = findW("import_mode")?.value === "always";
        if (capStr === node._lastImported && !always) return;
        const cap = tryParseCaption(capStr);
        if (!cap) return;
        node._lastImported = capStr;
        loadCaption(cap);
      }
      chainCallback(node, "onExecuted", function (message) {
        if (message?.caption) applyImported(message.caption[0]);
        // Seed regions from the bboxes input only when the editor is empty, so user-drawn/edited
        // regions are never overwritten.
        if (message?.boxes && !node._boxes.length) {
          const seeded = JSON.parse(message.boxes[0]);
          if (Array.isArray(seeded) && seeded.length) {
            node._boxes = seeded.filter((b) => b && typeof b.x === "number" && typeof b.w === "number");
            selectOnly(node._boxes.length ? 0 : -1);
            commit(); fitNode();
          }
        }
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
          sw.title = "Click edit · drag reorder · Ctrl+C/V copy/paste hex · right-click remove";
          const inp = document.createElement("input");
          inp.type = "color"; inp.value = hex;
          sw.appendChild(inp);
          container.appendChild(sw);
          const setColor = (hex2) => { arr[i] = hex2; inp.value = hex2; sw.style.background = hex2; sw.dataset.hex = hex2; onEdit(); };
          inp.addEventListener("input", () => setColor(inp.value));
          sw.addEventListener("pointerenter", () => { hoveredSwatch = { sw, setColor }; });
          sw.addEventListener("pointerleave", () => { if (hoveredSwatch && hoveredSwatch.sw === sw) hoveredSwatch = null; });
          sw.addEventListener("wheel", (e) => e.stopPropagation());
          sw.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); arr.splice(i, 1); onStruct(); });
          sw.addEventListener("pointerdown", (e) => {
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
              document.removeEventListener("pointermove", move);
              document.removeEventListener("pointerup", up);
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
            document.addEventListener("pointermove", move);
            document.addEventListener("pointerup", up);
          });
        });
        if (arr.length < max) {
          const add = document.createElement("button");
          add.className = "kjideo-btn"; add.textContent = "+";
          add.title = "Add a color (uses the clipboard color if it is one)";
          stopProp(add);
          add.addEventListener("click", async () => {
            let col = "#ffffff";
            try { const c = parseColorString(await navigator.clipboard.readText()); if (c) col = c; } catch (e) {}
            arr.push(col); onStruct();
          });
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
      let bboxPx = null, bboxGrid = null;   // editable bbox fields: pixels and 0–1000 grid
      function dims() { return [wWidget ? wWidget.value : 1024, hWidget ? hWidget.value : 1024]; }
      function boxToPx(b) {                                   // same order as the grid: ymin, xmin, ymax, xmax
        const [W, H] = dims();
        return [Math.round(b.y * H), Math.round(b.x * W), Math.round((b.y + b.h) * H), Math.round((b.x + b.w) * W)];
      }
      function setField(inp, b, fn) {                          // skip while the field is being edited
        if (!inp || document.activeElement === inp) return;
        inp.value = (!b || b.nobbox) ? "" : fn(b).join(", ");
      }
      function updateBboxLabel() {
        const b = node._boxes[node._activeIdx];
        setField(bboxPx, b, boxToPx);
        setField(bboxGrid, b, normBboxJS);
      }
      function parse4(inp) {
        const nums = inp.value.split(/[,\s]+/).map(Number).filter((n) => !isNaN(n));
        return nums.length === 4 ? nums : null;
      }
      function commitPxEdit() {
        const b = node._boxes[node._activeIdx]; if (!b || !bboxPx) return;
        const nums = parse4(bboxPx); if (!nums) { updateBboxLabel(); return; }
        const [W, H] = dims();
        let [ymin, xmin, ymax, xmax] = nums;                  // ymin, xmin, ymax, xmax (matches grid)
        ymin = Math.max(0, Math.min(H, ymin)); ymax = Math.max(0, Math.min(H, ymax));
        xmin = Math.max(0, Math.min(W, xmin)); xmax = Math.max(0, Math.min(W, xmax));
        if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
        if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
        b.y = ymin / H; b.x = xmin / W; b.h = (ymax - ymin) / H; b.w = (xmax - xmin) / W;
        delete b.nobbox; commit(); fitNode();
      }
      function commitGridEdit() {
        const b = node._boxes[node._activeIdx]; if (!b || !bboxGrid) return;
        const nums = parse4(bboxGrid); if (!nums) { updateBboxLabel(); return; }
        let [ymin, xmin, ymax, xmax] = nums.map((n) => Math.max(0, Math.min(1000, n)));
        if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
        if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
        b.y = ymin / 1000; b.x = xmin / 1000; b.h = (ymax - ymin) / 1000; b.w = (xmax - xmin) / 1000;
        delete b.nobbox; commit(); fitNode();
      }
      function makeBboxField(placeholder, title, onCommit) {
        const inp = document.createElement("input");
        inp.type = "text"; inp.className = "kjideo-bbox";
        inp.placeholder = placeholder; inp.title = title;
        stopProp(inp);
        inp.addEventListener("keydown", (e) => {
          e.stopPropagation();
          if (e.key === "Enter") inp.blur();
          else if (e.key === "Escape") { updateBboxLabel(); inp.blur(); }
        });
        inp.addEventListener("change", onCommit);
        return inp;
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
          if (node._panelH) panel.style.minHeight = node._panelH + "px";  // reserve height so it doesn't pop
          requestAnimationFrame(fitNode);
          return;
        }
        panel.style.minHeight = "";
        const col = (b.palette || []).find(Boolean) || "#bbb";
        const selN = node._selection.size;
        hint.innerHTML = `<b style="color:${col}">region ${node._activeIdx + 1}</b>` +
          (selN > 1 ? ` <span style="color:#888">(${selN} selected)</span>` : "");

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
        const txtChk = document.createElement("label");
        txtChk.style.cssText = "display:flex; align-items:center; gap:3px; cursor:pointer;";
        txtChk.title = "Show each region's text inside its box on the canvas";
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.checked = node.properties.showBoxText !== false;
        stopProp(cb);
        cb.addEventListener("change", () => { node.properties.showBoxText = cb.checked; drawCanvas(); });
        txtChk.appendChild(cb); txtChk.appendChild(document.createTextNode("show text"));
        typeRow.appendChild(txtChk);
        const pxLbl = document.createElement("span");
        pxLbl.textContent = "px:"; pxLbl.style.cssText = "margin-left:auto; color:#888;";
        typeRow.appendChild(pxLbl);
        bboxPx = makeBboxField("ymin, xmin, ymax, xmax", "Pixel bbox (of the node's width/height): ymin, xmin, ymax, xmax — editable", commitPxEdit);
        typeRow.appendChild(bboxPx);
        const gl = document.createElement("span");
        gl.textContent = "out:"; gl.style.color = "#888";
        typeRow.appendChild(gl);
        bboxGrid = makeBboxField("ymin, xmin, ymax, xmax", "Exported bbox on the 0–1000 grid: ymin, xmin, ymax, xmax — editable", commitGridEdit);
        typeRow.appendChild(bboxGrid);
        updateBboxLabel();
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

        requestAnimationFrame(() => { node._panelH = panel.offsetHeight; fitNode(); });  // remember height for the deselected placeholder
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
        if (_resizing || node._fullscreen) return;
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
      chainCallback(node, "onDeselected", function () { node._selected = false; finishPlacing(); closeLayersMenu(); drawCanvas(); });

      chainCallback(node, "onRemoved", function () {
        livePreviewNodes.delete(node);
        if (hoveredCanvasNode === node) hoveredCanvasNode = null;
        if (node._fullscreen) {                       // tear down the popup if open
          document.removeEventListener("keydown", onFsEsc, true);
          window.removeEventListener("resize", fitFsCanvas);
          node._fsOverlay?.remove();
        }
        node._visObserver?.disconnect();
        closeBgMenu(); node._bgMenu?.remove();
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
      // Persist editor data by name (robust to widget-order changes across versions).
      chainCallback(node, "onSerialize", function (o) {
        if (o) o.ideo = { boxes: node._boxes, palette: node._stylePalette, importMode: findW("import_mode")?.value };
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
          selectOnly(node._boxes.length ? 0 : -1);
        }
        const isPal = (p) => Array.isArray(p) && p.length && p.every((c) => typeof c === "string" && c[0] === "#");
        let pal = (o && o.ideo && isPal(o.ideo.palette)) ? o.ideo.palette : null;
        if (!pal) { try { const p = JSON.parse(stylePaletteWidget?.value || ""); if (isPal(p)) pal = p; } catch (e) {} }
        if (!pal) { for (const v of raw) { try { const p = JSON.parse(v); if (isPal(p)) { pal = p; break; } } catch (e) {} } }
        if (pal) node._stylePalette = pal.slice();
        const im = o && o.ideo && o.ideo.importMode, imW = findW("import_mode");
        if (im && imW) imW.value = im;                        // restore import_mode (index-based restore is unreliable here)
        hideDataWidgets();
        serialize();                                         // realign widget values for Python + future saves
        if (bgBrightnessWidget) bgSlider.value = bgBrightnessWidget.value;
        // node.properties is restored after onNodeCreated, so resync the toolbar controls to it.
        liveChk.checked = !!node.properties.liveBg;
        if (liveChk.checked) livePreviewNodes.add(node); else livePreviewNodes.delete(node);
        guideSel.value = node.properties.guide || "none";
        gridSlider.value = GRID_INV - (node.properties.gridSize || 10);
        snapChk.checked = !!node.properties.snap;
        guideColor.value = node.properties.guideColor || "#ffffff";
        opacitySlider.value = node.properties.guideOpacity == null ? 100 : node.properties.guideOpacity;
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
