import { chainCallback, addMiddleClickPan, addWheelPassthrough, cursorForBboxMode, watchImageInputs, captureVideoFrame } from './utility.js';
const { app } = window.comfyAPI.app;

const HANDLE = 8;            // hit radius (canvas px) for corners/edges
const MAX_ELEM_COLORS = 5;   // Ideogram 4 per-element palette cap
const MAX_STYLE_COLORS = 16; // Ideogram 4 style palette cap
const DOCK_MINW = 300, DOCK_MINH = 240;   // dock min size — kept in sync with the .kjideo-dock CSS below
let copiedBoxes = null;      // internal clipboard for copy/paste of regions (array; shared across nodes)

// Track the most recent generated image so it can be grabbed as a background.
let lastResultImage = null;
try {
  // Keep the latest image of the prompt; by the time the prompt finishes this is the real output
  // (the output node runs after the sampler), so an early image-preview node never wins.
  app.api?.addEventListener?.("executed", (e) => {
    const imgs = e?.detail?.output?.images;
    if (Array.isArray(imgs) && imgs.length) lastResultImage = imgs[imgs.length - 1];
  });
  // Grab the full-res final once the prompt completes — independent of sampler previews, so it works
  // even when sampling preview is disabled.
  app.api?.addEventListener?.("execution_success", () => {
    for (const n of livePreviewNodes) n._ideoGrabFinal?.();
  });
} catch (e) {}
function resultViewUrl(img) {
  const p = new URLSearchParams({ filename: img.filename || "", subfolder: img.subfolder || "", type: img.type || "output" });
  return "/view?" + p.toString();
}

// Nodes opted into "live background": feed them the sampling preview frames as they arrive.
const livePreviewNodes = new Set();
let _lastPreviewBlob = null;
let _previewOffset = 0;            // byte offset of the image inside the preview blob (0 = clean image)
function pushPreview(img) { for (const n of livePreviewNodes) n._ideoSetLiveBg?.(img); }
// Decode a blob to an <img> via object URL (same path the final result uses); resolves null on failure.
function decodeToImg(blob) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { resolve(img); URL.revokeObjectURL(url); };   // img retains its data after revoke
    img.onerror = () => { resolve(null); URL.revokeObjectURL(url); };
    img.src = url;
  });
}
// Offset of an embedded JPEG (FF D8 FF) or PNG signature, or -1. Some servers frame the preview
// payload with header bytes the bundled frontend doesn't strip, so the image isn't at byte 0.
function imageStart(u) {
  for (let i = 0; i + 3 < u.length; i++) {
    if (u[i] === 0xff && u[i + 1] === 0xd8 && u[i + 2] === 0xff) return i;
    if (u[i] === 0x89 && u[i + 1] === 0x50 && u[i + 2] === 0x4e && u[i + 3] === 0x47) return i;
  }
  return -1;
}
async function feedPreviewBlob(blob) {
  // Both "b_preview" and "b_preview_with_metadata" can fire for the same frame (same Blob) — dedupe by identity.
  if (!blob || blob === _lastPreviewBlob || !livePreviewNodes.size) return;
  _lastPreviewBlob = blob;
  let img = await decodeToImg(_previewOffset ? blob.slice(_previewOffset) : blob);
  if (!img) {                       // wrong/stale offset — locate the embedded image and retry
    const off = imageStart(new Uint8Array(await blob.arrayBuffer()));
    _previewOffset = off > 0 ? off : 0;
    if (off >= 0) img = await decodeToImg(off > 0 ? blob.slice(off) : blob);
  }
  if (img) pushPreview(img);
}
try {
  app.api?.addEventListener?.("b_preview", (e) => feedPreviewBlob(e?.detail));
  app.api?.addEventListener?.("b_preview_with_metadata", (e) => feedPreviewBlob(e?.detail?.blob));
} catch (e) {}

// Named caption-JSON templates, each stored as its own file server-side via ComfyUI's userdata
// API (under the ComfyUI user dir) so they persist across browsers/machines and survive clears.
const TPL_DIR = "kjnodes/ideogram4/templates";
const tplSafe = (s) => (s || "").replace(/[\/\\:*?"<>|]+/g, "_").trim();   // filesystem-safe name
const tplFile = (name) => `${TPL_DIR}/${name}.json`;
async function listTemplateNames() {
  try {
    const items = await app.api.listUserDataFullInfo(TPL_DIR);
    return items.map((it) => it.path.split(/[\\/]/).pop() || "")
      .filter((f) => /\.json$/i.test(f))
      .map((f) => f.replace(/\.json$/i, ""))
      .filter(Boolean).sort((a, b) => a.localeCompare(b));
  } catch (e) { return []; }
}
async function loadTemplate(name) {
  try { const r = await app.api.getUserData(tplFile(name)); if (r.status === 200) return await r.text(); } catch (e) {}
  return null;
}
async function saveTemplate(name, caption) {
  try {
    await app.api.storeUserData(tplFile(name), caption, { overwrite: true, stringify: false, throwOnError: true });
    return true;
  } catch (e) { window.alert("Couldn't save the template to the server."); return false; }
}
async function deleteTemplate(name) {
  try { await app.api.deleteUserData(tplFile(name)); } catch (e) {}
}

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
// Alt held over a canvas: preview the alt-click cycle target even without moving the mouse.
document.addEventListener("keydown", (e) => { if (hoveredCanvasNode && e.key === "Alt") hoveredCanvasNode._altRefresh?.(true); }, true);
document.addEventListener("keyup", (e) => { if (hoveredCanvasNode && e.key === "Alt") hoveredCanvasNode._altRefresh?.(false); }, true);

// Pinned dock follows the node: Nodes 2.0 hosts it inside the node element (inherits its transform); legacy is body-fixed.
const pinnedDocks = new Set();
const liveDocks = new Set();      // every node with a dock — swept so an orphaned dock can't linger
window.addEventListener("resize", () => wakeDocks());   // canvas may shift without pan/zoom
// Safety net: a dock whose node has left the graph (deleted, workflow cleared) is torn down even if onRemoved didn't fire.
function sweepOrphanDocks() {
  for (const n of liveDocks) {
    if (n.graph?.getNodeById?.(n.id) === n) continue;   // still registered in its graph
    try { n._dockRO?.disconnect(); } catch (e) {}
    try { n._visObserver?.disconnect(); } catch (e) {}
    n._dockEl?.remove();
    pinnedDocks.delete(n); liveDocks.delete(n);
  }
}
// What the dock follows in the displayed graph, or null to hide: the node ("self", offset=dockGraph),
// or — opt-in via exposeToParent — the SubgraphNode wrapping its subgraph ("parent", offset=dockParent; v1 direct parent only).
function dockAnchor(n) {
  const c = app.canvas;
  if (!c || !c.graph) return null;
  if (n.graph === c.graph) return { host: n, ctx: "self" };
  if (n.properties.exposeToParent && n.graph) {
    const nodes = c.graph._nodes || c.graph.nodes || [];
    const sn = nodes.find((m) => m && m.subgraph === n.graph);   // SubgraphNode wrapping the node's subgraph
    if (sn) return { host: sn, ctx: "parent" };
  }
  return null;
}
// gr.x/gr.y = offset (graph units) from host.pos (host = the node, or its SubgraphNode at parent level);
// size (w/h) always lives in dockGraph. `rect` lets the tick loop share one canvas getBoundingClientRect.
function applyDockTransform(n, anchor, rect) {
  const c = app.canvas, fl = n._dockEl;
  if (!c || !fl) return;
  anchor = anchor || dockAnchor(n);
  if (!anchor) return;                                          // off-screen; visibility handled in tickDocks
  const host = anchor.host;
  n._dockHost = host; n._dockCtx = anchor.ctx;                  // drag/snap handlers read these
  const gr = anchor.ctx === "parent"
    ? (n.properties.dockParent || (n.properties.dockParent = { x: 0, y: (host.size ? host.size[1] : 0) + 2 }))
    : n.properties.dockGraph;
  if (!gr) return;
  let hostEl = null;
  if (window.LiteGraph?.vueNodesMode && host.id != null) {      // cache; re-query if the host changed or it's gone
    if (n._dockHostId !== host.id || !n._dockHostEl || !n._dockHostEl.isConnected) {
      n._dockHostEl = document.querySelector(`[data-node-id="${host.id}"]`);
      n._dockHostId = host.id; n._dockSig = "";
    }
    hostEl = n._dockHostEl;
  }
  if (hostEl) {                                                 // host inside the host element → inherits its transform
    const title = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;    // element top is the title bar; host.pos sits below it
    const sig = "v|" + host.id + "|" + gr.x + "|" + gr.y;
    if (fl.parentElement !== hostEl) {
      hostEl.appendChild(fl);
      fl.style.position = "absolute"; fl.style.transform = ""; fl.style.transformOrigin = ""; fl.style.zIndex = "";
      n._dockSig = "";
    }
    if (n._dockSig !== sig) { fl.style.left = gr.x + "px"; fl.style.top = (title + gr.y) + "px"; n._dockSig = sig; }
    return;
  }
  // Legacy / fallback (no Vue element, e.g. just after a mode switch): body-fixed from host.pos + transform.
  if (fl.parentElement !== document.body) { document.body.appendChild(fl); fl.style.left = ""; fl.style.top = ""; n._dockSig = ""; }  // re-attach if orphaned
  if (!host.pos) return;
  const ds = c.ds, scale = ds.scale;
  rect = rect || c.canvas.getBoundingClientRect();
  const baseLeft = rect.left + (host.pos[0] + gr.x + ds.offset[0]) * scale;
  const baseTop = rect.top + (host.pos[1] + gr.y + ds.offset[1]) * scale;
  const tf = `translate(${baseLeft}px,${baseTop}px) scale(${scale})`;
  if (n._dockSig !== tf) { fl.style.position = "fixed"; fl.style.transformOrigin = "top left"; fl.style.transform = tf; n._dockSig = tf; }
  const order = c.graph?.nodes?.indexOf(host);          // sit at the node DOM-widget layer (not above everything)
  if (order != null && order >= 0) fl.style.zIndex = String(order);
}
// Dock chrome follows the node: color theme (title → header, body → background; dark fallback when uncolored)
// and pinned state (node pinned → dock locked, no drag/resize).
function applyDockTheme(n) {
  const fl = n._dockEl; if (!fl) return;
  const pinned = !!(n.flags && n.flags.pinned);
  const sig = (n.color || "") + "|" + (n.bgcolor || "") + "|" + pinned;
  if (n._dockTheme === sig) return;
  n._dockTheme = sig;
  const set = (k, v) => (v ? fl.style.setProperty(k, v) : fl.style.removeProperty(k));
  set("--kj-dock-bg", n.bgcolor);
  set("--kj-dock-head", n.color);
  set("--kj-dock-border", n.color || n.bgcolor);
  fl.classList.toggle("pinned", pinned);
}
// Event-driven loop: woken by canvas redraws (dirty-gated) + resize, self-stops when settled.
const DOCK_IDLE_STOP = 6;        // stop after this many frames with no further wake
let _dockRAF = 0, _dockLastMode = null, _dockIdle = 0, _dockWakesInstalled = false;
function tickDocks() {
  const c = app.canvas;
  if (c && pinnedDocks.size) {
    const vue = !!window.LiteGraph?.vueNodesMode;
    if (vue !== _dockLastMode) {                  // legacy↔2.0 flip rebuilds the node DOM — force re-parent + re-place
      _dockLastMode = vue;
      for (const n of pinnedDocks) { n._dockSig = ""; n._dockHostEl = null; n._dockHostId = null; }
    }
    const rect = vue ? null : c.canvas.getBoundingClientRect();   // legacy: share one rect across all docks
    for (const n of pinnedDocks) {                // anchor = node (own graph on screen) or its SubgraphNode (exposed)
      applyDockTheme(n);
      if (n._exBtn) n._exBtn.style.display = (n.graph && !n.graph.isRootGraph) ? "" : "none";   // expose only matters inside a subgraph
      const anchor = dockAnchor(n);
      if (n._dockEl) n._dockEl.style.display = anchor ? "" : "none";
      if (anchor) applyDockTransform(n, anchor, rect);
    }
  }
  if (pinnedDocks.size && ++_dockIdle < DOCK_IDLE_STOP) _dockRAF = requestAnimationFrame(tickDocks);
  else _dockRAF = 0;                              // settled — stop until the next wake
}
function wakeDocks() {            // a canvas change happened — run/continue the loop until it settles
  _dockIdle = 0;
  if (!_dockRAF && pinnedDocks.size) _dockRAF = requestAnimationFrame(tickDocks);
}
function installDockWakes() {     // onDrawForeground only fires when dirty_canvas is set
  if (_dockWakesInstalled) return;
  const c = app.canvas; if (!c) return;
  _dockWakesInstalled = true;
  chainCallback(c, "onDrawForeground", () => { sweepOrphanDocks(); wakeDocks(); });
}
function startDockLoop() { installDockWakes(); wakeDocks(); }

// Pointer-capture drag: filters moves by pointerId, auto-removes its listeners on release, calls onEnd.
function dragPointer(e, target, onMove, onEnd) {
  try { target.setPointerCapture(e.pointerId); } catch (er) {}
  const move = (me) => { if (me.pointerId === e.pointerId) onMove(me); };
  const end = (ue) => {
    if (ue.pointerId !== e.pointerId) return;
    target.removeEventListener("pointermove", move);
    target.removeEventListener("pointerup", end);
    target.removeEventListener("pointercancel", end);
    if (onEnd) onEnd(ue);
  };
  target.addEventListener("pointermove", move);
  target.addEventListener("pointerup", end);
  target.addEventListener("pointercancel", end);
}

// Outside-click dismissal for a popup: arm() after showing, disarm() after hiding.
function outsideDismiss(menu, onDismiss, anchor) {
  let handler = null;
  const disarm = () => {
    if (!handler) return;
    document.removeEventListener("pointerdown", handler, true);
    document.removeEventListener("mousedown", handler, true);
    handler = null;
  };
  const arm = () => {
    disarm();
    handler = (e) => { if (!menu.contains(e.target) && e.target !== anchor) onDismiss(); };
    setTimeout(() => {                                  // defer so the opening click itself doesn't dismiss
      if (!handler) return;
      document.addEventListener("pointerdown", handler, true);
      document.addEventListener("mousedown", handler, true);
    }, 0);
  };
  return { arm, disarm };
}

function injectStyle() {
  if (document.getElementById("kjideo-style")) return;
  const s = document.createElement("style");
  s.id = "kjideo-style";
  s.textContent = `
    .kjideo-wrap { display:flex; flex-direction:column; overflow:hidden; position:relative; pointer-events:auto; gap:4px; }
    .kjideo-cv { flex:1 1 auto; min-height:60px; display:flex; align-items:center; justify-content:center; overflow:hidden; }
    .kjideo-canvas { cursor:crosshair; display:block; flex:0 0 auto; background:#1a1a1a; border-radius:4px; outline:none; touch-action:none; }
    .kjideo-bar { display:flex; align-items:center; gap:6px; font:11px sans-serif; color:#aaa; user-select:none; padding:0 2px; flex:0 0 auto; }
    .kjideo-panel { display:flex; flex-direction:column; gap:5px; padding:6px; background:#262626; border-radius:4px; font:11px sans-serif; color:#bbb; flex:0 0 auto; overflow-y:auto; min-height:0; }
    .kjideo-split { flex:0 0 auto; height:8px; cursor:ns-resize; position:relative; }
    .kjideo-split::before { content:""; position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:34px; height:3px; background:#555; border-radius:2px; }
    .kjideo-split:hover::before { background:#46b4e6; }
    .kjideo-row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
    .kjideo-btn { background:#333; border:1px solid #555; border-radius:4px; color:#bbb; font:11px sans-serif; cursor:pointer; padding:2px 8px; line-height:16px; white-space:nowrap; flex-shrink:0; }
    .kjideo-btn:hover { border-color:#46b4e6; color:#fff; }
    .kjideo-btn.active { border-color:#46b4e6; color:#46b4e6; background:#2a3a42; }
    .kjideo-area { width:100%; box-sizing:border-box; background:#1d1d1d; border:1px solid #444; border-radius:4px; color:#ddd; font:13px monospace; padding:4px 6px; resize:none; min-height:36px; flex:1 1 auto; }
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
    .kjideo-lbtn.on { background:#3a3320; }
    .kjideo-lock { filter:grayscale(1); opacity:0.4; }                 /* unlocked: faded grey */
    .kjideo-lock.on, .kjideo-lock:hover { filter:none; opacity:1; }    /* locked / hover: full colour */
    .kjideo-lbtn:disabled { opacity:0.25; cursor:default; background:none; }
    .kjideo-fs { position:fixed; inset:0; z-index:9000; background:rgba(0,0,0,0.72); display:flex; align-items:center; justify-content:center; }
    .kjideo-fs-inner { position:relative; width:88vw; height:90vh; background:#1a1a1a; border:1px solid #444; border-radius:8px; box-shadow:0 12px 48px rgba(0,0,0,0.6); padding:12px; box-sizing:border-box; }
    .kjideo-fs-inner .kjideo-wrap { height:100%; }
    .kjideo-fs-close { position:absolute; top:14px; right:18px; z-index:5; padding:4px 12px; font-size:14px; }
    .kjideo-dock { position:fixed; z-index:8500; pointer-events:auto; display:flex; flex-direction:column; background:var(--kj-dock-bg,#1a1a1a); border:1px solid var(--kj-dock-border,#555); border-radius:8px; box-shadow:0 8px 30px rgba(0,0,0,0.55); min-width:${DOCK_MINW}px; min-height:${DOCK_MINH}px; overflow:hidden; }
    .kjideo-rsz { position:absolute; z-index:20; touch-action:none; }
    .kjideo-rsz.n { top:0; left:11px; right:11px; height:6px; cursor:ns-resize; }
    .kjideo-rsz.s { bottom:0; left:11px; right:11px; height:6px; cursor:ns-resize; }
    .kjideo-rsz.e { right:0; top:11px; bottom:11px; width:6px; cursor:ew-resize; }
    .kjideo-rsz.w { left:0; top:11px; bottom:11px; width:6px; cursor:ew-resize; }
    .kjideo-rsz.ne { top:0; right:0; width:12px; height:12px; cursor:nesw-resize; }
    .kjideo-rsz.nw { top:0; left:0; width:12px; height:12px; cursor:nwse-resize; }
    .kjideo-rsz.se { bottom:0; right:0; width:12px; height:12px; cursor:nwse-resize; }
    .kjideo-rsz.sw { bottom:0; left:0; width:12px; height:12px; cursor:nesw-resize; }
    .kjideo-dock.minimized { min-height:0 !important; height:auto !important; }
    .kjideo-dock.minimized .kjideo-dock-body { display:none; }
    .kjideo-dock.minimized .kjideo-rsz { display:none; }
    .kjideo-dock.pinned .kjideo-rsz { display:none; }          /* node pinned → locked, like the node */
    .kjideo-dock.pinned .kjideo-dock-head { cursor:default; }
    .kjideo-dock.snap-ready { box-shadow:0 -3px 0 0 #46b4e6, 0 8px 30px rgba(0,0,0,0.55); }   /* upper border lit = release to snap */
    .kjideo-dock-head { display:flex; align-items:center; gap:6px; padding:4px 8px; background:var(--kj-dock-head,#262626); cursor:move; font:12px sans-serif; color:#ccc; user-select:none; border-bottom:1px solid rgba(0,0,0,0.25); flex:0 0 auto; }
    .kjideo-dock-head .kjideo-btn { padding:1px 7px; }
    .kjideo-dock-body { flex:1 1 auto; min-height:0; padding:8px; box-sizing:border-box; overflow:hidden; }
    .kjideo-dock-body .kjideo-wrap { height:100%; }
    .kjideo-bgmenu { padding:7px; display:flex; flex-direction:column; gap:7px; min-width:170px; }
    .kjideo-bgrow { display:flex; align-items:center; gap:8px; }
    .kjideo-bglbl { color:#888; font:11px sans-serif; flex:0 0 auto; min-width:62px; }
    .kjideo-trow { padding:2px 4px; border-radius:4px; }
    .kjideo-trow:hover { background:#333; }
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
      const formatWidget = findW("output_format");           // "pretty" | "compact" (set via the toolbar)
      const coordWidget = findW("coord_mode");               // "normalized" | "absolute" (set via the toolbar)
      const orderWidget = findW("bbox_order");               // "yx" (Ideogram) | "xy" (Qwen/standard) (set via the toolbar)
      if (bgBrightnessWidget && typeof bgBrightnessWidget.value !== "number") bgBrightnessWidget.value = 25;
      const wWidget = findW("width"), hWidget = findW("height");
      // Hide the data widgets while keeping them serializable.
      function hideDataWidgets() {
        for (const w of [elementsWidget, stylePaletteWidget, bgBrightnessWidget, formatWidget, coordWidget, orderWidget]) {
          if (!w) continue;
          w.hidden = true;
          w.computeSize = () => [0, -4];
        }
        for (const name of ["elements_data", "style_palette_data", "bg_brightness", "output_format", "coord_mode", "bbox_order"]) {
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
      node._areaObservers = [];             // (reserved) live ResizeObservers to disconnect on rebuild

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
      // Output-settings dropdown (formatting + bbox coordinate space/order) — writes the hidden widgets.
      const outBtn = document.createElement("button");
      outBtn.className = "kjideo-btn"; outBtn.textContent = "Output ▾";
      outBtn.title = "Output JSON settings: compact/pretty, bbox coordinate space and axis order";
      stopProp(outBtn);
      const outToggle = (get, set, label, title) => {                 // checkbox row bound to a hidden widget
        const l = document.createElement("label");
        l.style.cssText = "display:flex;align-items:center;gap:4px;cursor:pointer;"; l.title = title;
        const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = get();
        stopProp(cb);
        cb.addEventListener("change", () => { set(cb.checked); updateTokens(); flushChange(); });
        l.appendChild(cb); l.appendChild(document.createTextNode(label));
        l._cb = cb; return l;
      };
      const compactLbl = outToggle(
        () => (formatWidget?.value) !== "pretty",
        (on) => { if (formatWidget) formatWidget.value = on ? "compact" : "pretty"; },
        "compact JSON", "Compact output (the format Ideogram 4 expects). Uncheck for pretty/indented.");
      const absLbl = outToggle(
        () => (coordWidget?.value) === "absolute",
        (on) => { if (coordWidget) coordWidget.value = on ? "absolute" : "normalized"; },
        "absolute pixels", "bbox in absolute pixels (scaled by width/height) instead of the 0-1000 grid. NON-STANDARD for Ideogram 4.");
      const xyLbl = outToggle(
        () => (orderWidget?.value) === "xy",
        (on) => { if (orderWidget) orderWidget.value = on ? "xy" : "yx"; },
        "xy order (Qwen)", "bbox axis order [xmin,ymin,xmax,ymax] (x1,y1,x2,y2, as Qwen-VL uses) instead of Ideogram's [ymin,xmin,ymax,xmax].");
      const outMenu = document.createElement("div");
      outMenu.className = "kjideo-menu kjideo-bgmenu";
      outMenu.style.display = "none";
      for (const l of [compactLbl, absLbl, xyLbl]) { const r = document.createElement("div"); r.className = "kjideo-bgrow"; r.appendChild(l); outMenu.appendChild(r); }
      document.body.appendChild(outMenu);
      node._outMenu = outMenu;
      const outDismiss = outsideDismiss(outMenu, () => closeOutMenu(), outBtn);
      function closeOutMenu() { outMenu.style.display = "none"; outDismiss.disarm(); }
      outBtn.addEventListener("click", () => {
        if (outMenu.style.display !== "none") { closeOutMenu(); return; }
        outMenu.style.display = "";
        const r = outBtn.getBoundingClientRect();
        outMenu.style.left = Math.max(4, Math.min(r.left, window.innerWidth - outMenu.offsetWidth - 4)) + "px";
        outMenu.style.top = Math.min(r.bottom + 4, window.innerHeight - outMenu.offsetHeight - 4) + "px";
        outDismiss.arm();
      });
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
      const bgDismiss = outsideDismiss(bgMenu, () => closeBgMenu(), bgBtn);
      function closeBgMenu() { bgMenu.style.display = "none"; bgDismiss.disarm(); }
      bgBtn.addEventListener("click", () => {
        if (bgMenu.style.display !== "none") { closeBgMenu(); return; }
        bgMenu.style.display = "";
        const r = bgBtn.getBoundingClientRect();
        bgMenu.style.left = Math.max(4, Math.min(r.left, window.innerWidth - bgMenu.offsetWidth - 4)) + "px";
        bgMenu.style.top = Math.min(r.bottom + 4, window.innerHeight - bgMenu.offsetHeight - 4) + "px";
        bgDismiss.arm();
      });
      // ── Text style popup: show/hide, outline, font size, auto-placement ──
      const txtBtn = document.createElement("button");
      txtBtn.className = "kjideo-btn"; txtBtn.textContent = "Text ▾";
      txtBtn.title = "Region text: show/hide, outline, font size, overlap-avoiding placement";
      stopProp(txtBtn);
      const txtToggle = (prop, label, title) => {
        const l = document.createElement("label");
        l.style.cssText = "display:flex;align-items:center;gap:4px;cursor:pointer;"; l.title = title || "";
        const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = node.properties[prop] !== false;
        cb.addEventListener("change", () => { node.properties[prop] = cb.checked; drawCanvas(); });
        l.appendChild(cb); l.appendChild(document.createTextNode(label));
        l._cb = cb; return l;
      };
      const showLbl = txtToggle("showBoxText", "show text", "Draw each region's text inside its box");
      const strokeLbl = txtToggle("textStroke", "outline", "Dark halo behind the text for legibility");
      const autoLbl = txtToggle("textAutoPlace", "auto-place", "Stagger overlapping labels to reduce overlap");
      const sizeSlider = document.createElement("input");
      sizeSlider.type = "range"; sizeSlider.min = "8"; sizeSlider.max = "22"; sizeSlider.step = "1";
      sizeSlider.value = node.properties.textSize || 12;
      sizeSlider.style.cssText = "width:90px;flex:0 0 auto;";
      sizeSlider.title = "Region text font size";
      stopProp(sizeSlider);
      sizeSlider.addEventListener("input", () => { node.properties.textSize = parseInt(sizeSlider.value, 10); drawCanvas(); });
      const boxOpacSlider = document.createElement("input");
      boxOpacSlider.type = "range"; boxOpacSlider.min = "0"; boxOpacSlider.max = "100"; boxOpacSlider.step = "1";
      boxOpacSlider.value = node.properties.boxOpacity == null ? 14 : node.properties.boxOpacity;
      boxOpacSlider.style.cssText = "width:90px;flex:0 0 auto;";
      boxOpacSlider.title = "Box fill opacity";
      stopProp(boxOpacSlider);
      boxOpacSlider.addEventListener("input", () => { node.properties.boxOpacity = parseInt(boxOpacSlider.value, 10); drawCanvas(); });
      const txtMenu = document.createElement("div");
      txtMenu.className = "kjideo-menu kjideo-bgmenu";
      txtMenu.style.display = "none";
      const txtRow = (labelText, el) => {
        const r = document.createElement("div"); r.className = "kjideo-bgrow";
        if (labelText) { const l = document.createElement("span"); l.className = "kjideo-bglbl"; l.textContent = labelText; r.appendChild(l); }
        r.appendChild(el); txtMenu.appendChild(r);
      };
      txtRow("", showLbl); txtRow("Font size", sizeSlider); txtRow("Box opacity", boxOpacSlider); txtRow("", strokeLbl); txtRow("", autoLbl);
      document.body.appendChild(txtMenu);
      node._txtMenu = txtMenu;
      const txtDismiss = outsideDismiss(txtMenu, () => closeTxtMenu(), txtBtn);
      function closeTxtMenu() { txtMenu.style.display = "none"; txtDismiss.disarm(); }
      txtBtn.addEventListener("click", () => {
        if (txtMenu.style.display !== "none") { closeTxtMenu(); return; }
        txtMenu.style.display = "";
        const r = txtBtn.getBoundingClientRect();
        txtMenu.style.left = Math.max(4, Math.min(r.left, window.innerWidth - txtMenu.offsetWidth - 4)) + "px";
        txtMenu.style.top = Math.min(r.bottom + 4, window.innerHeight - txtMenu.offsetHeight - 4) + "px";
        txtDismiss.arm();
      });
      // ── Templates popup: save/load named caption JSONs (server-side userdata) ──
      const tplBtn = document.createElement("button");
      tplBtn.className = "kjideo-btn"; tplBtn.textContent = "Templates ▾";
      tplBtn.title = "Save / load the caption JSON as templates (stored on the server ComfyUI\\user\\default\\kjnodes\\ideogram4)";
      stopProp(tplBtn);
      const tplMenu = document.createElement("div");
      tplMenu.className = "kjideo-menu kjideo-bgmenu";
      tplMenu.style.display = "none";
      document.body.appendChild(tplMenu);
      node._tplMenu = tplMenu;
      const tplDismiss = outsideDismiss(tplMenu, () => closeTplMenu(), tplBtn);
      function closeTplMenu() { tplMenu.style.display = "none"; tplDismiss.disarm(); }
      async function buildTplMenu() {
        tplMenu.innerHTML = "";
        const saveRow = document.createElement("div"); saveRow.className = "kjideo-bgrow";
        const saveBtn = document.createElement("button"); saveBtn.className = "kjideo-btn"; saveBtn.textContent = "+ Save as…";
        saveBtn.addEventListener("click", async () => {
          const name = tplSafe(window.prompt("Save template as:", "") || "");
          if (!name) return;
          const existing = await listTemplateNames();
          if (existing.includes(name) && !window.confirm(`Overwrite template "${name}"?`)) return;
          if (await saveTemplate(name, buildCaption())) buildTplMenu();
        });
        saveRow.appendChild(saveBtn); tplMenu.appendChild(saveRow);
        const names = await listTemplateNames();
        if (!names.length) {
          const empty = document.createElement("div"); empty.className = "kjideo-mhdr"; empty.textContent = "No templates saved.";
          tplMenu.appendChild(empty);
        }
        for (const name of names) {
          const row = document.createElement("div"); row.className = "kjideo-bgrow kjideo-trow";
          const txt = document.createElement("span");
          txt.className = "kjideo-ltext"; txt.style.cssText = "flex:1 1 auto;cursor:pointer;"; txt.textContent = name; txt.title = "Load " + name + " (replaces everything)";
          const ins = document.createElement("button"); ins.className = "kjideo-lbtn"; ins.textContent = "⊞"; ins.title = "Insert this template's boxes only into the current canvas";
          const upd = document.createElement("button"); upd.className = "kjideo-lbtn"; upd.textContent = "⤓"; upd.title = "Save current (overwrite)";
          const del = document.createElement("button"); del.className = "kjideo-lbtn del"; del.textContent = "✕"; del.title = "Delete template";
          row.append(txt, ins, upd, del); tplMenu.appendChild(row);
          txt.addEventListener("click", async () => {
            const cap = tryParseCaption(await loadTemplate(name));
            if (!cap) { window.alert("That template isn't a valid caption JSON."); return; }
            loadCaption(cap); closeTplMenu();
          });
          ins.addEventListener("click", async (e) => {
            e.stopPropagation();
            const cap = tryParseCaption(await loadTemplate(name));
            if (!cap) { window.alert("That template isn't a valid caption JSON."); return; }
            insertCaptionBoxes(cap); closeTplMenu();
          });
          upd.addEventListener("click", async (e) => {
            e.stopPropagation();
            if (await saveTemplate(name, buildCaption())) buildTplMenu();
          });
          del.addEventListener("click", async (e) => {
            e.stopPropagation();
            if (!window.confirm(`Delete template "${name}"?`)) return;
            await deleteTemplate(name); buildTplMenu();
          });
        }
      }
      tplBtn.addEventListener("click", async () => {
        if (tplMenu.style.display !== "none") { closeTplMenu(); return; }
        tplMenu.style.display = "";
        await buildTplMenu();
        const r = tplBtn.getBoundingClientRect();
        tplMenu.style.left = Math.max(4, Math.min(r.left, window.innerWidth - tplMenu.offsetWidth - 4)) + "px";
        tplMenu.style.top = Math.min(r.bottom + 4, window.innerHeight - tplMenu.offsetHeight - 4) + "px";
        tplDismiss.arm();
      });
      bar.appendChild(hint); bar.appendChild(tokenSpan); bar.appendChild(outBtn); bar.appendChild(bgBtn); bar.appendChild(txtBtn); bar.appendChild(copyBtn); bar.appendChild(importBtn); bar.appendChild(tplBtn); bar.appendChild(clearBtn);
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
      addWheelPassthrough(wrap);   // middle-click pan is wired on the whole dock (fl) in undock()

      const panel = document.createElement("div");
      panel.className = "kjideo-panel";
      node._panelH = node._panelH || 150;                    // height of the prompt panel (set by the splitter)
      panel.style.height = node._panelH + "px";

      // Canvas letterbox-fits into a flex container (cvBox), so node resize never aspect-locks the height.
      const cvBox = document.createElement("div");
      cvBox.className = "kjideo-cv";
      cvBox.appendChild(canvasEl);

      // Draggable separator between the canvas and the prompt panel — drag up for more prompt, down for more canvas.
      const splitter = document.createElement("div");
      splitter.className = "kjideo-split";
      splitter.title = "Drag to resize the description area";
      splitter.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault(); e.stopPropagation();
        const scale = (node.properties.dockPinned && app.canvas) ? (app.canvas.ds.scale || 1) : 1;
        const sy = e.clientY, h0 = panel.offsetHeight;
        dragPointer(e, splitter, (me) => {
          const dy = (me.clientY - sy) / scale;
          const h = Math.max(60, Math.min(h0 - dy, wrap.clientHeight - 110));   // up → bigger panel; keep room for the canvas
          node._panelH = Math.round(h); panel.style.height = node._panelH + "px";   // persisted via the onSerialize blob
          fitCanvas();
        }, flushChange);
      });

      wrap.appendChild(bar); wrap.appendChild(styleBar); wrap.appendChild(cvBox); wrap.appendChild(splitter); wrap.appendChild(panel);

      // The editor is NOT a node widget — it lives in the floating dock (built in undock()), so the node
      // is an ordinary widgets-only node that ComfyUI sizes/persists with zero custom height handling.
      node.resizable = true;

      // The dock canvas measures 0 while hidden; re-letterbox it when the editor element re-enters view.
      try {
        node._visObserver = new IntersectionObserver((entries) => {
          if (entries.some((en) => en.isIntersecting)) fitCanvas();
        });
        node._visObserver.observe(wrap);
      } catch (e) {}

      // Letterbox-fit the canvas (keeping width/height aspect) into its flex container cvBox.
      function fitCanvas() {
        if (wrap.offsetParent === null) return;   // hidden (collapsed tab / minimized) — measurements are 0
        const availW = cvBox.clientWidth, availH = cvBox.clientHeight;
        if (availW < 4 || availH < 4) return;
        const aspect = (wWidget?.value || 1) / (hWidget?.value || 1);
        let cw = availW, ch = cw / aspect;
        if (ch > availH) { ch = availH; cw = ch * aspect; }
        canvasEl.style.width = Math.round(cw) + "px";
        canvasEl.style.height = Math.round(ch) + "px";
        drawCanvas();
      }
      const fitFsCanvas = fitCanvas;                           // dock/fullscreen use the same fitter
      function syncCanvasToDims() { fitCanvas(); }             // aspect changed (width/height widgets)
      function detachInto(container) {                        // move the editor element into a container (dock body / fullscreen)
        node._wrapHome = wrap.parentNode;
        container.appendChild(wrap);
        node._fsInner = container;
      }
      function reattach() {                                   // fullscreen exit → back into the floating dock
        canvasEl.style.width = ""; canvasEl.style.height = "";
        const home = node._dockBody || node._wrapHome;
        if (home) home.appendChild(wrap);
        node._fsInner = node._dockBody || null;               // stays detached, living in the dock
        if (node.graph) node.graph.setDirtyCanvas(true, true);
        requestAnimationFrame(fitCanvas);
      }

      function onFsEsc(e) { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); exitFs(); } }
      function enterFs() {
        if (node._fullscreen) return;
        node._fullscreen = true;
        const ov = document.createElement("div"); ov.className = "kjideo-fs";
        const inner = document.createElement("div"); inner.className = "kjideo-fs-inner";
        ov.appendChild(inner);
        ov.addEventListener("mousedown", (e) => { if (e.target === ov) exitFs(); });  // backdrop closes
        document.body.appendChild(ov);
        node._fsOverlay = ov;
        detachInto(inner);
        const closeBtn = document.createElement("button");                            // visible exit (⛶ in the dock header is hidden here)
        closeBtn.className = "kjideo-btn kjideo-fs-close"; closeBtn.textContent = "✕"; closeBtn.title = "Close (Esc)";
        stopProp(closeBtn); closeBtn.addEventListener("click", exitFs);
        ov.appendChild(closeBtn);                                                     // on the backdrop corner, clear of the editor
        document.addEventListener("keydown", onFsEsc, true);
        window.addEventListener("resize", fitFsCanvas);
        requestAnimationFrame(fitFsCanvas);
      }
      function exitFs() {
        if (!node._fullscreen) return;
        node._fullscreen = false;
        document.removeEventListener("keydown", onFsEsc, true);
        window.removeEventListener("resize", fitFsCanvas);
        node._fsOverlay?.remove(); node._fsOverlay = null;
        reattach();
      }

      // ── floating / dockable editor panel ──
      // Active-context position offset (set by applyDockTransform): dockParent (parent) or dockGraph (self).
      // dockParent is created by applyDockTransform with the right default; fall back to dockGraph if absent.
      function dockPos() {
        return node._dockCtx === "parent" ? (node.properties.dockParent || node.properties.dockGraph) : node.properties.dockGraph;
      }
      // The node the dock is currently anchored to: the node itself, or the SubgraphNode at parent level.
      function dockHostNode() { return node._dockHost || node; }
      // Persist the panel geometry — screen-space rect when unpinned, graph-space size when pinned.
      function saveDockGeom() {
        const fl = node._dockEl; if (!fl) return;
        if (node.properties.dockMin) return;                  // collapsed: don't persist the header-only size
        // sub-min = transient (hidden / mid-reparent) → bogus, don't persist. (offsetParent is null for the fixed dock.)
        if (fl.offsetWidth < DOCK_MINW || fl.offsetHeight < DOCK_MINH) return;
        if (node.properties.dockPinned) {
          node.properties.dockGraph = node.properties.dockGraph || { x: 0, y: 0 };
          node.properties.dockGraph.w = fl.offsetWidth; node.properties.dockGraph.h = fl.offsetHeight;
        } else {
          const r = fl.getBoundingClientRect();
          node.properties.dockRect = { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) };
        }
      }
      // Nudge ComfyUI's change tracker (it snapshots on mouseup, which our preventDefault'd drags suppress).
      function flushChange() { try { window.dispatchEvent(new MouseEvent("mouseup")); } catch (e) {} }
      function setPinned(on, pinBtn) {
        const fl = node._dockEl, c = app.canvas; if (!fl || !c) return;
        const rect = c.canvas.getBoundingClientRect(), ds = c.ds;
        if (on) {                                          // return to where it was pinned (the saved dockGraph)
          let g = node.properties.dockGraph;
          if (!g) {                                        // never pinned before — derive from the current float position
            const r = fl.getBoundingClientRect();
            const gx = (r.left - rect.left) / ds.scale - ds.offset[0], gy = (r.top - rect.top) / ds.scale - ds.offset[1];
            g = node.properties.dockGraph = {
              x: gx - (node.pos?.[0] ?? 0), y: gy - (node.pos?.[1] ?? 0),
              w: Math.round(fl.offsetWidth / ds.scale), h: Math.round(fl.offsetHeight / ds.scale),
            };
          }
          node.properties.dockPinned = true;
          fl.style.left = ""; fl.style.top = "";
          fl.style.width = g.w + "px"; fl.style.height = g.h + "px";
          node._dockSig = ""; node._dockHostEl = null; node._dockHostId = null;     // force a fresh place + re-parent
          pinnedDocks.add(node); startDockLoop();
          applyDockTransform(node);                        // place this frame (avoid a flash at the old float spot)
        } else {                                           // freeze current visual rect as a body-fixed screen panel
          const r = fl.getBoundingClientRect();
          node.properties.dockPinned = false;
          pinnedDocks.delete(node);
          node._dockSig = ""; node._dockHostEl = null; node._dockHostId = null;
          document.body.appendChild(fl);                   // back out of the node element
          fl.style.position = "fixed"; fl.style.transform = ""; fl.style.transformOrigin = ""; fl.style.zIndex = "";
          fl.style.left = Math.round(r.left) + "px"; fl.style.top = Math.round(r.top) + "px";
          fl.style.width = Math.round(r.width) + "px"; fl.style.height = Math.round(r.height) + "px";
          node.properties.dockRect = { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) };
        }
        if (pinBtn) { pinBtn.classList.toggle("active", on); pinBtn.title = on ? "Unpin from canvas (float in screen)" : "Pin to canvas (move/zoom with the graph)"; }
        c.setDirtyCanvas(true, true); requestAnimationFrame(fitFsCanvas);
        flushChange();   // click's mouseup already fired before this handler — snapshot the new state
      }
      // Resize from any edge/corner; N/W edges shift the anchor. Units: graph px pinned (÷scale), else screen px.
      function startDockResize(e, dir, fl) {
        if (e.button !== 0 || node.flags?.pinned) return;    // node pinned → not resizable, like the node
        e.preventDefault(); e.stopPropagation();
        const pinned = !!node.properties.dockPinned;
        const scale = (pinned && app.canvas) ? (app.canvas.ds.scale || 1) : 1;
        const sx = e.clientX, sy = e.clientY, w0 = fl.offsetWidth, h0 = fl.offsetHeight;
        const startPos = dockPos();                          // active context: dockGraph (self) or dockParent
        const gx0 = pinned ? startPos.x : (parseFloat(fl.style.left) || 0);
        const gy0 = pinned ? startPos.y : (parseFloat(fl.style.top) || 0);
        const MINW = DOCK_MINW, MINH = DOCK_MINH;
        dragPointer(e, e.currentTarget, (me) => {
          const dx = (me.clientX - sx) / scale, dy = (me.clientY - sy) / scale;
          let w = w0, h = h0, gx = gx0, gy = gy0;
          if (dir.indexOf("e") >= 0) w = w0 + dx;
          if (dir.indexOf("s") >= 0) h = h0 + dy;
          if (dir.indexOf("w") >= 0) { w = w0 - dx; gx = gx0 + dx; }
          if (dir.indexOf("n") >= 0) { h = h0 - dy; gy = gy0 + dy; }
          if (w < MINW) { if (dir.indexOf("w") >= 0) gx -= (MINW - w); w = MINW; }
          if (h < MINH) { if (dir.indexOf("n") >= 0) gy -= (MINH - h); h = MINH; }
          fl.style.width = Math.round(w) + "px"; fl.style.height = Math.round(h) + "px";
          if (pinned) { const p = dockPos(); p.x = gx; p.y = gy; node._dockSig = ""; applyDockTransform(node); }
          else { fl.style.left = Math.round(gx) + "px"; fl.style.top = Math.round(gy) + "px"; }
          fitFsCanvas();
        }, () => { saveDockGeom(); flushChange(); });
      }
      function addDockResizeHandles(fl) {
        for (const dir of ["n", "s", "e", "w", "ne", "nw", "se", "sw"]) {
          const h = document.createElement("div");
          h.className = "kjideo-rsz " + dir;
          h.addEventListener("pointerdown", (e) => startDockResize(e, dir, fl));
          fl.appendChild(h);
        }
      }
      function undock() {
        if (node._docked) return;
        if (node._fullscreen) exitFs();
        node._docked = true; node.properties.docked = true;
        const fl = document.createElement("div"); fl.className = "kjideo-dock";
        fl.dataset.captureWheel = "true";   // 2.0: let focused inputs in the dock scroll instead of zooming the graph
        stopProp(fl);   // hosted inside the node element — don't let dock interactions drag/zoom the node
        addMiddleClickPan(fl);   // middle-click pans the graph from anywhere in the dock (bubbles up from inner elements)
        fl.addEventListener("mousedown", (e) => { if (e.button === 1) e.stopPropagation(); });   // ...but don't let it reach the node behind
        const head = document.createElement("div"); head.className = "kjideo-dock-head";
        const title = document.createElement("span"); title.textContent = "Ideogram 4 editor"; title.style.flex = "1";
        const minBtn = document.createElement("button"); minBtn.className = "kjideo-btn";
        stopProp(minBtn);
        const applyMin = (on) => {                            // collapse the editor to just the header bar
          node.properties.dockMin = !!on;
          fl.classList.toggle("minimized", !!on);
          minBtn.textContent = on ? "▢" : "—";
          minBtn.title = on ? "Restore editor" : "Minimize editor";
          if (!on) requestAnimationFrame(fitFsCanvas);        // re-letterbox the canvas after it's visible again
        };
        minBtn.addEventListener("click", () => { applyMin(!node.properties.dockMin); flushChange(); });
        const fsBtn = document.createElement("button"); fsBtn.className = "kjideo-btn"; fsBtn.textContent = "⛶";
        fsBtn.title = "Open in a larger window (Esc to close)";
        stopProp(fsBtn); fsBtn.addEventListener("click", () => node._fullscreen ? exitFs() : enterFs());
        const pinBtn = document.createElement("button"); pinBtn.className = "kjideo-btn"; pinBtn.textContent = "📌";
        stopProp(pinBtn); pinBtn.addEventListener("click", () => setPinned(!node.properties.dockPinned, pinBtn));
        // Opt-in: also show this canvas at the parent (subgraph) level (tickDocks hides this button at root).
        const exBtn = document.createElement("button"); exBtn.className = "kjideo-btn"; exBtn.textContent = "⤴";
        exBtn.style.display = (node.graph && !node.graph.isRootGraph) ? "" : "none";
        const syncExBtn = () => {
          exBtn.classList.toggle("active", !!node.properties.exposeToParent);
          exBtn.title = (node.properties.exposeToParent ? "Hide" : "Show") + " this canvas at the parent (subgraph) level";
        };
        stopProp(exBtn);
        exBtn.addEventListener("click", () => { node.properties.exposeToParent = !node.properties.exposeToParent; syncExBtn(); node._dockSig = ""; wakeDocks(); flushChange(); });
        syncExBtn();
        node._syncExBtn = syncExBtn; node._exBtn = exBtn;     // onConfigure refreshes state; tickDocks toggles visibility
        head.append(title, fsBtn, minBtn, exBtn, pinBtn);
        const body = document.createElement("div"); body.className = "kjideo-dock-body";
        fl.append(head, body);
        addDockResizeHandles(fl);
        document.body.appendChild(fl);
        node._dockEl = fl; node._dockBody = body; node._dockSig = ""; node._dockHostEl = null; node._dockHostId = null; liveDocks.add(node);
        applyDockTheme(node);                                 // match the node's color theme right away
        detachInto(body);
        // initial geometry / mode
        const pinned = !!node.properties.dockPinned;
        if (pinned) {
          const g = node.properties.dockGraph || { x: 0, y: 0, w: 540, h: 470 };
          fl.style.position = "absolute"; fl.style.width = g.w + "px"; fl.style.height = g.h + "px";
          pinnedDocks.add(node); startDockLoop();
          applyDockTransform(node);                          // parent + place this frame (avoid a flash at 0,0)
          pinBtn.classList.add("active");
        } else {
          const rc = node.properties.dockRect || { x: 90, y: 90, w: 540, h: 470 };
          fl.style.left = Math.max(0, Math.min(rc.x, window.innerWidth - 80)) + "px";
          fl.style.top = Math.max(0, Math.min(rc.y, window.innerHeight - 40)) + "px";
          fl.style.width = rc.w + "px"; fl.style.height = rc.h + "px";
        }
        pinBtn.title = pinned ? "Unpin from canvas (float in screen)" : "Pin to canvas (move/zoom with the graph)";
        applyMin(!!node.properties.dockMin);                  // restore minimized state
        // drag the panel by its header (graph-space when pinned, screen-space otherwise)
        head.addEventListener("pointerdown", (e) => {
          if (e.target === pinBtn || e.target === minBtn || e.target === fsBtn || e.button !== 0) return;
          if (node.flags?.pinned) return;                    // node pinned → dock is locked too
          e.preventDefault();
          const sx0 = e.clientX, sy0 = e.clientY;
          const pinned = node.properties.dockPinned;
          const scale = pinned ? (app.canvas.ds.scale || 1) : 1;
          const dp0 = dockPos();                           // active context: dockGraph (self) or dockParent
          const gx0 = pinned ? dp0.x : 0, gy0 = pinned ? dp0.y : 0;
          const ox = fl.offsetLeft, oy = fl.offsetTop;
          let snapReady = false;
          dragPointer(e, head, (me) => {
            if (pinned) {                                  // move the panel by writing its transform directly (no full redraw)
              const gx = gx0 + (me.clientX - sx0) / scale, gy = gy0 + (me.clientY - sy0) / scale;
              const h = dockHostNode();                    // snap "home": flush under the anchor (node or SubgraphNode)
              const snap = 26 / scale, underY = h.size[1] + 2;
              snapReady = Math.abs(gx) < snap && Math.abs(gy - underY) < snap;   // just highlight; snap on release
              fl.classList.toggle("snap-ready", snapReady);
              const p = dockPos(); p.x = gx; p.y = gy;     // follow the cursor freely
              applyDockTransform(node);
            } else {
              fl.style.left = Math.max(0, Math.min(ox + me.clientX - sx0, window.innerWidth - 60)) + "px";
              fl.style.top = Math.max(0, Math.min(oy + me.clientY - sy0, window.innerHeight - 30)) + "px";
            }
          }, () => {
            if (snapReady) {                               // commit the snap: flush under the anchor, matching its width
              const h = dockHostNode(), w = Math.round(h.size[0]), p = dockPos();
              p.x = 0; p.y = h.size[1] + 2;
              if (node.properties.dockGraph.w !== w) { node.properties.dockGraph.w = w; fl.style.width = w + "px"; }
              node._dockSig = ""; applyDockTransform(node);
            }
            fl.classList.remove("snap-ready");
            saveDockGeom(); flushChange();
          });
        });
        // Re-letterbox the canvas and persist geometry on dock resize. The sub-min guard in
        // saveDockGeom drops the bogus transient reads from Vue re-parenting.
        try { node._dockRO = new ResizeObserver(() => { fitFsCanvas(); saveDockGeom(); }); node._dockRO.observe(fl); } catch (e) {}
        requestAnimationFrame(fitFsCanvas);   // letterbox the canvas once the dock has its size
      }
      // Pin the editor under the node. `fresh` (new node only) matches the node width; reloads honor the blob.
      function ensureDocked(fresh) {
        if (node._docked) return;
        if (node.properties.dockPinned == null) node.properties.dockPinned = true;
        const honor = !!node.properties.dockGraph && node._dockGeomRestored;   // only trust blob-restored geometry
        if (!honor) {
          // old workflow: reuse its saved (aspect-locked) node height so the canvas isn't tiny
          const sh = (!fresh && node._savedSize && node._savedSize[1] > 240) ? Math.round(node._savedSize[1]) : 470;
          node.properties.dockGraph = { x: 0, y: 0, w: Math.round(node.size[0]), h: sh };   // match the node width
        }
        undock();
        if (!honor) requestAnimationFrame(() => {             // no reliable geometry — place a default under the node, matching its width
          const g = node.properties.dockGraph;
          g.x = 0; g.y = node.size[1] + 2; g.w = Math.round(node.size[0]);
          if (node._dockEl) node._dockEl.style.width = g.w + "px";
          node._dockSig = ""; applyDockTransform(node); saveDockGeom();
        });
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
        const baseRx = HANDLE / logW(), baseRy = HANDLE / logH();
        const res = [];
        for (let i = 0; i < node._boxes.length; i++) {
          const b = node._boxes[i];
          if (b.locked) continue;                          // locked regions aren't grabbable on the canvas
          const rx = Math.min(baseRx, b.w / 3), ry = Math.min(baseRy, b.h / 3);  // shrink handles on small boxes so a central move zone remains
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
          // Box too small to hold the badge → place it just outside (above/below/beside) so it doesn't
          // cover the box and the box stays grabbable; otherwise tuck it into a free corner inside.
          const small = (x2 - x1) < w || (y2 - y1) < h;
          const cands = small
            ? [[x1, y1 - h - 1], [x1, y2 + 1], [x2 + 1, y1], [x1 - w - 1, y1]]
            : [[x1, y1], [x2 - w, y1], [x2 - w, y2 - h], [x1, y2 - h]];
          let pick = cands[0];
          for (const [cx, cy] of cands) {
            if (cx < 0 || cy < 0 || cx + w > W || cy + h > H) continue;   // keep on-canvas
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
          if (node._boxes[i].locked) continue;             // locked badges aren't interactive
          const r = rects[i];
          if (r && px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) return i;
        }
        return null;
      }
      // In-box text placement: when auto-place is on, put each box's text in whichever vertical slot overlaps
      // least with the text already placed for other boxes (greedy), so overlapping labels don't stack.
      function textBlocks() {
        if (node.properties.showBoxText === false) return [];
        const { fs, lh } = txtFont();
        ctx.font = fs + "px monospace";
        const W = logW(), H = logH(), pad = 4, tagH = 14, auto = node.properties.textAutoPlace !== false;
        const placed = [], blocks = [];
        const overlap = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) *
                                  Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
        for (let i = 0; i < node._boxes.length; i++) {
          const b = node._boxes[i];
          let body = b.desc || "";
          if (b.type === "text" && b.text) body = `"${b.text}"` + (body ? " — " + body : "");
          if (!body) { blocks[i] = null; continue; }
          const x1 = b.x * W, y1 = b.y * H, x2 = (b.x + b.w) * W, y2 = (b.y + b.h) * H;
          const lines = wrapLines(body, (x2 - x1) - pad * 2);   // full width
          let lw = 0; for (const ln of lines) lw = Math.max(lw, ctx.measureText(ln).width);
          const bw = Math.min(x2 - x1, lw + pad * 2), bh = Math.min(y2 - y1, lines.length * lh + 4);
          // candidates = several vertical slots (more when the box is taller); text spans the width, so only Y varies
          const top = y1 + ((y2 - y1) > bh + tagH ? tagH : 0), bottom = y2 - bh;
          const cands = [];
          if (!auto || bottom <= top) cands.push([x1, top]);    // off, or fills the box → fixed top
          else {
            const n = Math.min(6, 2 + Math.floor((bottom - top) / (lh * 2)));
            for (let k = 0; k < n; k++) cands.push([x1, top + ((bottom - top) * k) / (n - 1)]);
          }
          let best = cands[0], bestScore = Infinity;
          for (const [cx, cy] of cands) {
            const rect = { x: cx, y: cy, w: bw, h: bh };
            let s = 0; for (const p of placed) s += overlap(rect, p);
            if (s < bestScore) { bestScore = s; best = [cx, cy]; if (s === 0) break; }
          }
          const r = { x: best[0], y: best[1], w: bw, h: bh, lines, lh, fs, pad };
          placed.push(r); blocks[i] = r;
        }
        return blocks;
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
          // cycle over a stable order (boxesAt moves the active box to the front, which would otherwise ping-pong 2)
          const ordered = [...cands].sort((a, b) => a.index - b.index);
          const pos = ordered.findIndex((c) => c.index === node._activeIdx);
          return ordered[(pos + 1) % ordered.length];
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
        const { x, y, w, h } = start;
        const dx = dN.x, dy = dN.y;
        if (mode === "move") {
          return { ...start, x: clamp01(Math.min(x + dx, 1 - w)), y: clamp01(Math.min(y + dy, 1 - h)) };
        }
        if (mode === "draw") {
          const ax = clamp01(x), ay = clamp01(y);            // fixed corner (mousedown point)
          const cx = clamp01(x + dx), cy = clamp01(y + dy);  // moving corner (cursor), clamped to canvas
          return { ...start, x: Math.min(ax, cx), y: Math.min(ay, cy), w: Math.abs(cx - ax), h: Math.abs(cy - ay) };
        }
        // resize: move only the dragged edges (clamped to the canvas); the others stay anchored.
        const suf = mode.slice(7);                 // "tl"|"tr"|"bl"|"br"|"t"|"b"|"l"|"r"
        let l = x, t = y, r = x + w, b = y + h;
        if (suf.includes("l")) l = clamp01(l + dx);
        if (suf.includes("r")) r = clamp01(r + dx);
        if (suf.includes("t")) t = clamp01(t + dy);
        if (suf.includes("b")) b = clamp01(b + dy);
        if (r < l) [l, r] = [r, l];                // crossing the opposite edge flips, no anchor drift
        if (b < t) [t, b] = [b, t];
        return { ...start, x: l, y: t, w: r - l, h: b - t };
      }

      // ── grid / guides ──
      function gridN() { return Math.max(2, Math.min(128, node.properties.gridSize || 10)); }
      function txtFont() { const fs = Math.max(6, Math.min(40, node.properties.textSize || 12)); return { fs, lh: Math.round(fs * 1.2) }; }
      // Whole number of cells per axis so they fill the canvas exactly (no split cells at the edges);
      // near-square (the target size is rounded to fit). Shared by the grid guide and snap-to-grid.
      function gridStep() {
        const W = logW(), H = logH(), target = Math.min(W, H) / gridN();
        const nx = Math.max(1, Math.round(W / target)), ny = Math.max(1, Math.round(H / target));
        return { nx, ny, cw: W / nx, ch: H / ny, sx: 1 / nx, sy: 1 / ny };
      }
      function guideStroke(a) {
        const c = hexRgb(node.properties.guideColor || "#ffffff") || { r: 255, g: 255, b: 255 };
        const op = (node.properties.guideOpacity == null ? 100 : node.properties.guideOpacity) / 100;
        return `rgba(${c.r},${c.g},${c.b},${(a * op).toFixed(3)})`;
      }
      // Snap a box's free edges to the (square) grid when snap-to-grid is on.
      function snapBox(b, mode) {
        if (!node.properties.snap) return b;
        const { sx, sy } = gridStep();
        const sn = (v, s) => Math.round(v / s) * s;
        const { x, y, w, h } = b;
        if (mode === "move") {                          // snap position, preserve size
          return { ...b, x: clamp01(Math.min(sn(x, sx), 1 - w)), y: clamp01(Math.min(sn(y, sy), 1 - h)) };
        }
        const x2 = sn(x + w, sx), y2 = sn(y + h, sy);   // draw/resize: snap the edges
        return normalizeBox({ ...b, x: sn(x, sx), y: sn(y, sy), w: x2 - sn(x, sx), h: y2 - sn(y, sy) });
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
        if (kind === "grid") {                          // whole cells across both axes (no split cells at edges)
          ctx.strokeStyle = guideStroke(0.24);
          const { nx, ny, cw, ch } = gridStep();
          for (let i = 1; i < nx; i++) { const X = Math.round(i * cw) + 0.5; ctx.beginPath(); ctx.moveTo(X, 0); ctx.lineTo(X, H); ctx.stroke(); }
          for (let j = 1; j < ny; j++) { const Y = Math.round(j * ch) + 0.5; ctx.beginPath(); ctx.moveTo(0, Y); ctx.lineTo(W, Y); ctx.stroke(); }
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
        const textB = textBlocks();                           // collision-avoided in-box text positions
        for (const i of order) {
          const b = node._boxes[i], active = i === aIdx, selected = selSet.has(i) && !b.locked;  // locked never shows as selected
          const pal = (b.palette || []).filter(Boolean);
          const col = pal.length ? pal[0] : "#8c8c8c";       // box color = first palette color, else neutral grey
          const { x1, y1, x2, y2 } = toPx(b);
          const w = x2 - x1, h = y2 - y1;
          const hovered = (i === node._hoverBox && !b.locked) || selected;  // locked boxes don't hover; selected stay highlighted
          if (selected) {                                    // opaque backing so contents read clearly over boxes behind
            ctx.fillStyle = "rgba(26,26,26,0.88)";
            ctx.fillRect(x1, y1, w, h);
          }
          const baseA = (node.properties.boxOpacity == null ? 14 : node.properties.boxOpacity) / 100;
          const fillA = Math.min(1, hovered ? baseA + 0.1 : baseA);   // box-color tint at the chosen opacity
          ctx.fillStyle = col + Math.round(fillA * 255).toString(16).padStart(2, "0");
          ctx.fillRect(x1, y1, w, h);
          if (b.locked) ctx.setLineDash([3, 3]);             // locked: frozen on the canvas
          else if (b.nobbox) ctx.setLineDash([6, 4]);        // unplaced (no bbox in source)
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
          // in-box content (clipped to the box): prompt text (at its de-conflicted corner) + lock badge
          ctx.save();
          ctx.beginPath(); ctx.rect(x1, y1, w, h); ctx.clip();
          const tb = textB[i];
          if (tb) {
            const stroke = node.properties.textStroke !== false;
            ctx.font = tb.fs + "px monospace";
            if (stroke) { ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.strokeStyle = "rgba(0,0,0,0.85)"; }  // dark halo
            ctx.fillStyle = readableText(col);              // box color, lightened if too dark
            let ty = tb.y + tb.fs;
            for (const line of tb.lines) {
              if (stroke) ctx.strokeText(line, tb.x + tb.pad, ty);
              ctx.fillText(line, tb.x + tb.pad, ty);
              ty += tb.lh;
            }
          }
          ctx.restore();
          // tag chip on top, unclipped (sits outside boxes too small to hold it — its collision-avoided position)
          const tr = tagR[i];
          ctx.font = "bold 11px monospace";
          ctx.fillStyle = col;                                // tag chip = box color
          ctx.fillRect(tr.x, tr.y, tr.w, 14);
          if (i === node._hoverTitle && !b.locked) {          // hover highlight (never on locked)
            ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.fillRect(tr.x, tr.y, tr.w, 14);
            ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.strokeRect(tr.x + 0.5, tr.y + 0.5, tr.w - 1, 13);
          }
          ctx.fillStyle = textOn(col);
          ctx.fillText(tr.tag, tr.x + 4, tr.y + 11);
          if (b.locked) { ctx.font = "11px sans-serif"; ctx.fillStyle = "#ddd"; ctx.fillText("🔒", tr.x + tr.w + 2, tr.y + 11); }  // lock symbol next to the badge
          if (selected) {                                    // orange selection ring on top (above strip/tag): solid = primary, dashed = others
            const olw = active ? 2 : 1;
            ctx.strokeStyle = "#ff8c00"; ctx.lineWidth = olw;
            if (!active) ctx.setLineDash([5, 3]);
            ctx.strokeRect(x1 + olw / 2, y1 + olw / 2, w - olw, h - olw);
            ctx.setLineDash([]);
          }
        }
        const apIdx = node._altPreview ?? -1;                 // Alt-held: ring the box an alt-click would select next
        if (apIdx >= 0 && apIdx < node._boxes.length) {
          const { x1, y1, x2, y2 } = toPx(node._boxes[apIdx]);
          ctx.strokeStyle = "#46b4e6"; ctx.lineWidth = 2; ctx.setLineDash([5, 3]);
          ctx.strokeRect(x1 + 1, y1 + 1, (x2 - x1) - 2, (y2 - y1) - 2);
          ctx.setLineDash([]);
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
      // Wired AND the source node isn't muted(2)/bypassed(4) — those keep the link but emit nothing.
      const importConnected = () => {
        const link = node.graph?.links?.[node.inputs?.find((i) => i.name === "import_json")?.link];
        return !!link && ![2, 4].includes(node.graph.getNodeById(link.origin_id)?.mode);
      };
      function serialize() {                              // saved/restored value: clean boxes
        if (elementsWidget) elementsWidget.value = node._boxes.length ? JSON.stringify(node._boxes) : "";
        if (stylePaletteWidget) stylePaletteWidget.value = node._stylePalette.length ? JSON.stringify(node._stylePalette) : "";
      }
      // Queue-time value (not the saved value): when a wired import should drive the output — "always"
      // mode, or empty in "when empty" mode — return a unique marker so ComfyUI can't cache-skip the
      // node. It then re-executes and pushes the import back via ui, refreshing the editor. The server
      // treats a non-list elements_data as empty (and ignores it entirely in "always" mode), so the
      // nonce never affects the output.
      if (elementsWidget) {
        elementsWidget.serializeValue = () => {
          const always = findW("import_mode")?.value === "always";
          if (importConnected() && (always || !node._boxes.length)) {
            return JSON.stringify({ _refresh: (node._serialSeq = (node._serialSeq || 0) + 1) });
          }
          return node._boxes.length ? JSON.stringify(node._boxes) : "";
        };
      }

      function commit() { serialize(); renderPanel(); drawCanvas(); updateTokens(); flushChange(); }  // flush so canvas edits persist without a defocus
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
        const idxs = [...node._selection].sort((a, b) => b - a).filter((i) => !node._boxes[i]?.locked);  // keep locked
        if (!idxs.length && node._activeIdx >= 0 && !node._boxes[node._activeIdx]?.locked) idxs.push(node._activeIdx);
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
        // capture the pointer so move/up keep coming even when the cursor leaves the node
        try { canvasEl.setPointerCapture(e.pointerId); } catch (e2) {}
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
        canvasEl.addEventListener("pointermove", onMove);
        canvasEl.addEventListener("pointerup", onUp);
        canvasEl.addEventListener("pointercancel", onUp);   // touch can cancel instead of up
        e.preventDefault(); e.stopPropagation();
        drawCanvas();   // panel rebuild/resize deferred to onUp so the canvas doesn't shift mid-drag
      });

      // Hover targets: with Alt, preview what an alt-click would select next (the box under the current one).
      function hoverTargets(mN, alt, force) {
        if (alt && !force) { const pick = pickForSelection(mN, true); return { ti: null, hb: null, ap: pick ? pick.index : -1, hit: null }; }
        const ti = force ? null : titleAt(mN);
        const hit = force ? null : hitTest(mN);
        return { ti, hb: ti != null ? ti : (hit ? hit.index : null), ap: -1, hit };
      }
      function applyHover(t) {
        if (t.ti !== node._hoverTitle || t.hb !== node._hoverBox || t.ap !== (node._altPreview ?? -1)) {
          node._hoverTitle = t.ti; node._hoverBox = t.hb; node._altPreview = t.ap; drawCanvas();
        }
      }
      canvasEl.addEventListener("pointermove", (e) => {
        node._lastMouseN = mouseN(e);                        // track cursor for paste-under-cursor
        if (node._placing) { placeFollower(node._lastMouseN); return; }
        if (node._drawing || node._marquee || node._hideBoxes) return;
        const mN = node._lastMouseN;
        const force = e.ctrlKey || e.metaKey, alt = e.altKey;   // Ctrl/Cmd = force-draw, Alt = cycle preview
        const t = hoverTargets(mN, alt, force);
        applyHover(t);
        canvasEl.style.cursor = (alt && !force) ? (t.ap >= 0 ? "pointer" : "crosshair")
          : (t.ti != null ? "pointer" : (t.hit ? (cursorForBboxMode(t.hit.mode) || "crosshair") : "crosshair"));
      });
      // Alt pressed/released without moving the mouse — refresh the preview from the last cursor position.
      node._altRefresh = (altDown) => {
        if (node._drawing || node._marquee || node._placing || node._hideBoxes || !node._lastMouseN) return;
        applyHover(hoverTargets(node._lastMouseN, altDown, false));
      };
      canvasEl.addEventListener("pointerleave", () => {
        if (hoveredCanvasNode === node) hoveredCanvasNode = null;
        if (node._hoverTitle !== null || node._hoverBox !== null || (node._altPreview ?? -1) >= 0) {
          node._hoverTitle = null; node._hoverBox = null; node._altPreview = -1; drawCanvas();
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
        const clones = copiedBoxes.map((b) => { const c = JSON.parse(JSON.stringify(b)); delete c.locked; return c; });  // pasted = editable
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
        commit(); fitCanvas();
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
          removeSelected(); commit(); fitCanvas();      // removes all selected (or the active one)
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
            const { sx, sy } = gridStep();
            dx = Math.round(dx / sx) * sx;
            dy = Math.round(dy / sy) * sy;
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
        canvasEl.removeEventListener("pointermove", onMove);
        canvasEl.removeEventListener("pointerup", onUp);
        canvasEl.removeEventListener("pointercancel", onUp);
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
        canvasEl.addEventListener("pointermove", onMarqueeMove);
        canvasEl.addEventListener("pointerup", onMarqueeUp);
        canvasEl.addEventListener("pointercancel", onMarqueeUp);
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
          node._boxes.forEach((b, i) => { if (!b.locked && rectsOverlap(r, b)) sel.add(i); });   // marquee skips locked
          node._selection = sel;
          if (node._activeIdx < 0 || !sel.has(node._activeIdx)) node._activeIdx = sel.size ? [...sel][0] : node._activeIdx;
        }
        drawCanvas();
      }
      function onMarqueeUp() {
        canvasEl.removeEventListener("pointermove", onMarqueeMove);
        canvasEl.removeEventListener("pointerup", onMarqueeUp);
        canvasEl.removeEventListener("pointercancel", onMarqueeUp);
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
        delete nb.nobbox; delete nb.locked;               // a duplicate is editable, not born locked
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
        commit(); fitCanvas();
      }
      function cancelPlacing() {
        if (!node._placing) return;
        node._placing = false;
        canvasEl.style.cursor = "crosshair";
        removeBox(node._activeIdx);
        commit(); fitCanvas();
      }

      // ── right-click "layers" menu: list / select / delete / duplicate / reorder regions ──
      function closeLayersMenu() {
        if (node._layerMenu) { node._layerMenu.remove(); node._layerMenu = null; }
        node._layerDismiss?.disarm(); node._layerDismiss = null;
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
            const lock = document.createElement("button");
            lock.className = "kjideo-lbtn kjideo-lock" + (b.locked ? " on" : "");
            lock.textContent = b.locked ? "🔒" : "🔓";
            lock.title = b.locked ? "Unlock (allow moving/resizing)" : "Lock (freeze on the canvas)";
            const dup = document.createElement("button");
            dup.className = "kjideo-lbtn"; dup.textContent = "⧉";
            dup.title = "Duplicate, then click on the canvas to place";
            const del = document.createElement("button");
            del.className = "kjideo-lbtn del"; del.textContent = "✕";
            del.title = b.locked ? "Unlock to delete" : "Delete region";
            del.disabled = !!b.locked;
            row.append(sw, num, txt, lock, dup, del);
            list.appendChild(row);

            lock.addEventListener("click", (e) => {
              e.stopPropagation();
              b.locked = !b.locked;
              if (b.locked) {                              // a locked box drops out of the selection
                const idx = node._boxes.indexOf(b);
                node._selection.delete(idx);
                if (node._activeIdx === idx) node._activeIdx = node._selection.size ? [...node._selection][0] : -1;
              }
              commit(); buildRows();
            });

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
              removeBox(idx); commit(); fitCanvas();
              if (!node._boxes.length) { closeLayersMenu(); return; }
              buildRows();
            });
            // drag-reorder (vertical FLIP, mirrors the palette swatch reorder)
            row.addEventListener("pointerdown", (e) => {
              if (e.button !== 0 || e.target === lock || e.target === dup || e.target === del) return;
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
                document.removeEventListener("pointercancel", up);
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
              document.addEventListener("pointercancel", up);
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

        node._layerDismiss = outsideDismiss(menu, () => closeLayersMenu());
        node._layerDismiss.arm();
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
        commit(); rebuildStylePalette(); fitCanvas();   // a wired import re-seeds on the next run (serializeValue cache-busts)
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
      // Caption bbox: 0-1000 grid (default) or absolute pixels (scaled by width/height) — must match the
      // Python output. normBboxJS stays 0-1000 for the editor's grid input field, so keep this separate.
      function captionBboxJS(b) {
        const abs = (coordWidget?.value) === "absolute", xy = (orderWidget?.value) === "xy";
        const sx = abs ? (wWidget?.value || 1000) : 1000, sy = abs ? (hWidget?.value || 1000) : 1000;
        const cx = (v) => Math.max(0, Math.min(sx, Math.round(v * sx)));
        const cy = (v) => Math.max(0, Math.min(sy, Math.round(v * sy)));
        let ymin = cy(b.y), xmin = cx(b.x), ymax = cy(b.y + b.h), xmax = cx(b.x + b.w);
        if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
        if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
        return xy ? [xmin, ymin, xmax, ymax] : [ymin, xmin, ymax, xmax];
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
          if (!b.nobbox) el.bbox = captionBboxJS(b);         // unplaced elements omit bbox
          if (etype === "text") el.text = b.text || "";
          el.desc = b.desc || "";
          const pal = cleanPalette(b.palette).slice(0, MAX_ELEM_COLORS);
          if (pal.length) el.color_palette = pal;
          return el;
        });
        cap.compositional_deconstruction = { background: getW("background"), elements };
        return (formatWidget?.value) === "pretty" ? pyJson(cap) : JSON.stringify(cap);   // compact by default; matches the node output
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
      // Lenient fixer: slice the outermost {...} (drops ``` fences / prose), then strip trailing
      // commas before } or ] — the string alt matches first so quoted commas are left untouched.
      function repairJson(s) {
        const i = s.indexOf("{"), j = s.lastIndexOf("}");
        const t = (i !== -1 && j > i) ? s.slice(i, j + 1) : s;
        return t.replace(/("(?:[^"\\]|\\.)*")|,(\s*[}\]])/g, (m, str, close) => str || close);
      }
      function tryParseCaption(t) {
        if (!t) return null;
        for (const cand of [t, repairJson(t)]) {
          try { const o = JSON.parse(cand); if (o && typeof o === "object" && o.compositional_deconstruction) return o; }
          catch (e) {}
        }
        return null;
      }
      // Apply a parsed caption to the editor and refresh everything.
      function loadCaption(cap) {
        closeInlineEditor();
        // Setting the multiline widgets + rebuilding the style sub-widget makes ComfyUI grow the node;
        // an import shouldn't change its size, so restore it (now and after the deferred re-arrange).
        const sz = node.size ? [node.size[0], node.size[1]] : null;
        applyCaption(cap);
        if (sz) { node.setSize(sz); requestAnimationFrame(() => node.setSize(sz)); }
        syncCanvasToDims(); commit(); rebuildStylePalette(); fitCanvas();
      }
      // Append a caption's regions to the current canvas (keeps existing boxes + caption fields).
      function insertCaptionBoxes(cap) {
        closeInlineEditor();
        const cd = (cap && cap.compositional_deconstruction) || {};
        const els = Array.isArray(cd.elements) ? cd.elements : [];
        const added = els.map((el, i) => bboxElemToBox(el, i)).filter(Boolean);
        if (!added.length) return;
        const start = node._boxes.length;
        node._boxes.push(...added);
        node._selection = new Set();                      // select the inserted regions
        for (let i = start; i < node._boxes.length; i++) node._selection.add(i);
        node._activeIdx = node._boxes.length - 1;
        commit(); fitCanvas();
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
            commit(); fitCanvas();
          }
        }
        // Reflect resolved width/height (e.g. from connected inputs) in the canvas aspect.
        // A connected background image governs the aspect itself, so skip then.
        if (message?.dims && !node._bgImg) {
          const [w, h] = message.dims;
          if (wWidget && w) wWidget.value = w;
          if (hWidget && h) hWidget.value = h;
          syncCanvasToDims(); fitCanvas();
        }
      });

      // ── property panel ──
      function stopProp(el) {
        el.addEventListener("wheel", (e) => e.stopPropagation());
        // Let middle-click (button 1) bubble so it reaches the dock's pan handler; block left/right so
        // dock controls don't drag/select the node behind them.
        for (const ev of ["mousedown", "pointerdown"]) el.addEventListener(ev, (e) => { if (e.button !== 1) e.stopPropagation(); });
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
            try { sw.setPointerCapture(e.pointerId); } catch (e2) {}  // capture so Nodes 2.0's WidgetDOM .stop can't swallow the drag
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
              sw.removeEventListener("pointermove", move);
              sw.removeEventListener("pointerup", up);
              sw.removeEventListener("pointercancel", up);
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
            sw.addEventListener("pointermove", move);
            sw.addEventListener("pointerup", up);
            sw.addEventListener("pointercancel", up);
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
          () => { swatchEdit(); rebuildStylePalette(); fitCanvas(); });
      }

      // Textarea that flexes to fill the prompt panel (whose height is set by the splitter).
      function makeArea(field, value, placeholder, onInput) {
        const ta = document.createElement("textarea");
        ta.className = "kjideo-area";
        ta.placeholder = placeholder;
        ta.value = value || "";
        stopProp(ta);
        ta.addEventListener("input", onInput);
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
        delete b.nobbox; commit(); fitCanvas();
      }
      function commitGridEdit() {
        const b = node._boxes[node._activeIdx]; if (!b || !bboxGrid) return;
        const nums = parse4(bboxGrid); if (!nums) { updateBboxLabel(); return; }
        let [ymin, xmin, ymax, xmax] = nums.map((n) => Math.max(0, Math.min(1000, n)));
        if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
        if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
        b.y = ymin / 1000; b.x = xmin / 1000; b.h = (ymax - ymin) / 1000; b.w = (xmax - xmin) / 1000;
        delete b.nobbox; commit(); fitCanvas();
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
          hint.textContent = "";
          const p = document.createElement("div");
          p.style.color = "#888";
          p.textContent = node._boxes.length ? "Click a region to edit it." : "No regions yet.";
          panel.appendChild(p);
          return;
        }
        const col = (b.palette || []).find(Boolean) || "#bbb";
        const selN = node._selection.size;
        // Build with DOM + style.color (a CSS value) — never innerHTML — since col comes from
        // box data that may be loaded from an untrusted template/import (avoids HTML injection).
        hint.textContent = "";
        const tag = document.createElement("b");
        tag.style.color = col; tag.textContent = "region " + (node._activeIdx + 1);
        hint.appendChild(tag);
        if (selN > 1) {
          const s = document.createElement("span");
          s.style.color = "#888"; s.textContent = ` (${selN} selected)`;
          hint.appendChild(s);
        }

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

        panel.appendChild(makeArea("desc", b.desc, "description of this region",
          function () { b.desc = this.value; touch(); }));

        // palette
        const palRow = document.createElement("div");
        palRow.className = "kjideo-row";
        const pl = document.createElement("span"); pl.textContent = "colors:"; palRow.appendChild(pl);
        b.palette = b.palette || [];
        buildSwatchRow(palRow, b.palette, MAX_ELEM_COLORS, swatchEdit, commit);
        panel.appendChild(palRow);
      }

      // ── width/height widget callbacks ──
      for (const w of [wWidget, hWidget]) {
        if (!w) continue;
        chainCallback(w, "callback", () => { syncCanvasToDims(); drawCanvas(); fitCanvas(); updateTokens(); });  // absolute coords depend on w/h
      }
      // Update the token estimate when the caption-level text widgets change.
      for (const name of ["background", "high_level_description", "aesthetics", "lighting", "medium", "style"]) {
        const w = findW(name);
        if (w) chainCallback(w, "callback", () => updateTokens());
      }

      // Persist node resizes: the 2.0 resize drag preventDefaults the compat mouseup, so the change
      // tracker (snapshots on window 'mouseup') misses the new size — nudge it once the resize settles.
      let _sizeFlush = 0;
      chainCallback(node, "onResize", function () {
        clearTimeout(_sizeFlush);
        _sizeFlush = setTimeout(flushChange, 150);
      });

      // Optional reference image as the canvas background (matches ImageTransformKJ).
      function loadBg(src, keepAspect) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          node._bgImg = img;
          if (!keepAspect) {                                          // live mode keeps the user's canvas aspect
            const r16 = (v) => Math.max(16, Math.round(v / 16) * 16); // model needs multiples of 16
            if (wWidget) wWidget.value = r16(img.naturalWidth);        // match canvas aspect to the image
            if (hWidget) hWidget.value = r16(img.naturalHeight);
          }
          syncCanvasToDims(); drawCanvas(); fitCanvas(); updateGrabBtn();
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
        loadBg(resultViewUrl(lastResultImage), true);   // keep aspect — live mode must never resize the canvas
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
        pinnedDocks.delete(node); liveDocks.delete(node);
        node._dockRO?.disconnect(); node._dockEl?.remove();   // tear down the floating dock if open
        node._visObserver?.disconnect();
        if (node._liveBmp?.close) { try { node._liveBmp.close(); } catch (e) {} node._liveBmp = null; }  // release GPU bitmap
        closeBgMenu(); node._bgMenu?.remove();
        closeTxtMenu(); node._txtMenu?.remove();
        closeTplMenu(); node._tplMenu?.remove();
        closeOutMenu(); node._outMenu?.remove();
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
        if (!o) return;
        const p = node.properties;
        o.ideo = { boxes: node._boxes, palette: node._stylePalette, importMode: findW("import_mode")?.value,
          outputFormat: findW("output_format")?.value, coordMode: findW("coord_mode")?.value, bboxOrder: findW("bbox_order")?.value,
          dock: { pinned: p.dockPinned, graph: p.dockGraph, rect: p.dockRect, panelH: node._panelH, min: p.dockMin,
            exposeParent: p.exposeToParent, parent: p.dockParent } };
      });
      chainCallback(node, "onConfigure", function (o) {
        const raw = o && Array.isArray(o.widgets_values) ? o.widgets_values : [];
        // Restore dock geometry from the blob into node.properties BEFORE ensureDocked runs.
        // Use the SAVED size (o.size) — old workflows' tall embedded-editor height — so the dock gets a
        // sensible height. node.size here is already shrunk to content by configure's widget application.
        node._savedSize = (o && Array.isArray(o.size) && o.size.length === 2) ? [o.size[0], o.size[1]] : (node.size ? [node.size[0], node.size[1]] : null);
        const d = o && o.ideo && o.ideo.dock;
        if (d) {
          node._dockGeomRestored = true;                     // honor it; old workflows w/o the blob get a default
          if (d.graph) node.properties.dockGraph = d.graph;
          if (d.rect) node.properties.dockRect = d.rect;
          if (d.pinned != null) node.properties.dockPinned = d.pinned;
          if (d.min != null) node.properties.dockMin = d.min;
          if (d.exposeParent != null) node.properties.exposeToParent = d.exposeParent;
          if (d.parent) node.properties.dockParent = d.parent;
          if (typeof d.panelH === "number") { node._panelH = d.panelH; panel.style.height = d.panelH + "px"; }
        }
        node._syncExBtn?.();                                  // refresh the expose-to-parent toggle after restore
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
        const imW = findW("import_mode");                     // restore import_mode; coerce to a valid option so
        if (imW) {                                            // old workflows (saved before this widget) don't fail Combo validation
          const im = o && o.ideo && o.ideo.importMode, opts = ["when empty", "always"];
          imW.value = opts.includes(im) ? im : (opts.includes(imW.value) ? imW.value : "when empty");
        }
        if (formatWidget) {                                   // restore output format from the blob; default compact unless explicitly pretty
          formatWidget.value = (o && o.ideo && o.ideo.outputFormat) === "pretty" ? "pretty" : "compact";
          compactLbl._cb.checked = formatWidget.value === "compact";
        }
        if (coordWidget) {                                    // restore coord mode; default normalized unless explicitly absolute
          coordWidget.value = (o && o.ideo && o.ideo.coordMode) === "absolute" ? "absolute" : "normalized";
          absLbl._cb.checked = coordWidget.value === "absolute";
        }
        if (orderWidget) {                                    // restore bbox axis order; default yx unless explicitly xy
          orderWidget.value = (o && o.ideo && o.ideo.bboxOrder) === "xy" ? "xy" : "yx";
          xyLbl._cb.checked = orderWidget.value === "xy";
        }
        node._configured = true;                              // mark as loaded so initial layout keeps the restored size
        hideDataWidgets();
        serialize();                                         // realign widget values for Python + future saves
        if (bgBrightnessWidget) {
          if (typeof bgBrightnessWidget.value !== "number") bgBrightnessWidget.value = 25;  // old workflows may restore ""
          bgSlider.value = bgBrightnessWidget.value;
        }
        // node.properties is restored after onNodeCreated, so resync the toolbar controls to it.
        liveChk.checked = !!node.properties.liveBg;
        if (liveChk.checked) livePreviewNodes.add(node); else livePreviewNodes.delete(node);
        guideSel.value = node.properties.guide || "none";
        gridSlider.value = GRID_INV - (node.properties.gridSize || 10);
        snapChk.checked = !!node.properties.snap;
        guideColor.value = node.properties.guideColor || "#ffffff";
        opacitySlider.value = node.properties.guideOpacity == null ? 100 : node.properties.guideOpacity;
        showLbl._cb.checked = node.properties.showBoxText !== false;
        strokeLbl._cb.checked = node.properties.textStroke !== false;
        autoLbl._cb.checked = node.properties.textAutoPlace !== false;
        sizeSlider.value = node.properties.textSize || 12;
        boxOpacSlider.value = node.properties.boxOpacity == null ? 14 : node.properties.boxOpacity;
        syncCanvasToDims();
        rebuildStylePalette();
        renderPanel();
        drawCanvas();
        updateTokens();
        // configure() shrinks the node to content height (multiline widgets) before this runs — for a
        // current-version save, re-assert the saved widgets-only size. Skip for old (pre-dock) workflows:
        // their saved height included the embedded editor, so let those shrink to content instead.
        if (node._dockGeomRestored && o && Array.isArray(o.size) && o.size.length === 2) node.setSize([o.size[0], o.size[1]]);
        requestAnimationFrame(() => ensureDocked(false));    // reload: build the dock honoring saved geometry
      });

      // Initial layout (deferred so size/last_y settle). NEVER set height here — it must stay whatever
      // ComfyUI restored (touching it was the "height resets on reload/copy-paste" bug); only nudge a default width.
      setTimeout(() => {
        hideDataWidgets();
        if (!node._configured) node.setSize([Math.max(480, node.size[0]), node.size[1]]);                       // fresh: comfortable default width, keep natural height
        else if (!node._dockGeomRestored) node.setSize([Math.max(380, node.size[0]), node.computeSize()[1]]);  // old workflow: drop the embedded-editor height, shrink to content
        else if (node.size[0] < 380) node.setSize([380, node.size[1]]);                                          // current reload/paste: keep saved size, enforce min width
        syncCanvasToDims();
        rebuildStylePalette();
        renderPanel();
        drawCanvas();
        updateTokens();
        if (!node._configured) ensureDocked(true);           // fresh node: pop the dock under it, matching its width
      }, 0);
    });
  },
});
