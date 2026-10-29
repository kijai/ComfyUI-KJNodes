import { chainCallback, addMiddleClickPan, addWheelPassthrough, resolveSourcePreview, watchImageInputs, rectHitTest, cursorForBboxMode, captureVideoFrame } from './utility.js';
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

// Layers are real <img>/<video>/<canvas> elements composited by CSS, so the browser scales them at native
// quality and interaction only touches styles. Only diff mode draws, once per still or per video frame.

const MODES = ["slider", "box", "diff", "toggle", "mask"];
const MODE_HINTS = {
  slider: "Drag to split, Shift+drag rotates (Ctrl snaps 15°), double-click resets",
  box: "Drag to draw, move or resize the box, double-click clears",
  diff: "Drag sideways to boost, double-click resets",
  toggle: "Click to cycle, double-click returns to A",
  mask: "White in the mask shows the compared image",
};
const CURSORS = { box: "crosshair", diff: "ew-resize", toggle: "pointer", mask: "default" };
const PAIR_MODES = new Set(["box", "diff", "mask"]);
const MIN_HEIGHT = 120;
const DIFF_MAX_SIDE = 4096;
const BOOST_MAX_EXP = 5; // diff boost 1x..32x on a log scale
const DEFAULT_STATE = { mode: "slider", dividers: [{ x: 0.5, y: 0.5 }], angle: 0, box: null, toggleIdx: 0, cmpIdx: 1, boostExp: 0, showHandle: true };

const label = (i) => "ABCD"[i] ?? String(i + 1);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const clamp01 = (v) => clamp(v, 0, 1);
// Compared inputs are multi-type "IMAGE,VIDEO" slots, the mask input is not
const isMediaInput = (inp) => String(inp.type).split(",").some((t) => t === "IMAGE" || t === "VIDEO");
const wrap360 = (a) => ((a % 360) + 360) % 360;
const axisOf = (angle) => [Math.cos(angle * Math.PI / 180), Math.sin(angle * Math.PI / 180)];
const defaultDividers = (n) => Array.from({ length: Math.max(0, n - 1) }, (_, i) => ({ x: (i + 1) / n, y: 0.5 }));
const naturalSize = (el) => [el.videoWidth || el.naturalWidth || el.width || 1, el.videoHeight || el.naturalHeight || el.height || 1];
const viewUrl = (f) => api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder ?? "")}&type=${f.type ?? "temp"}`);

// Restore callbacks of live nodes, run once the frontend has put a reloaded tab's outputs back
const restorers = new Set();

const STYLE_ID = "kjnodes-image-compare-style";
const STYLE = `
.kjic-root { display:flex; flex-direction:column; width:100%; height:100%; min-height:${MIN_HEIGHT}px; overflow:hidden;
  font:11px sans-serif; color:var(--input-text, #ccc); user-select:none; }
.kjic-root [hidden] { display:none !important; }
.kjic-wrap { position:relative; flex:1 1 auto; min-height:0; container-type:size; display:flex; align-items:center; justify-content:center; }
.kjic-stage { position:relative; display:none; overflow:hidden; background:#282828;
  width:min(100cqw, calc(100cqh * var(--kjic-ar, 1))); height:min(100cqh, calc(100cqw / var(--kjic-ar, 1))); }
.kjic-root.has-media .kjic-stage { display:block; }
.kjic-root.has-media .kjic-empty, .kjic-root:not(.has-media) .kjic-bar { display:none; }
.kjic-empty { opacity:0.45; padding:8px; text-align:center; }
.kjic-layers, .kjic-overlay { position:absolute; inset:0; pointer-events:none; }
.kjic-layers { isolation:isolate; }
.kjic-layer { position:absolute; inset:0; width:100%; height:100%; display:block; object-fit:contain; background:#282828; mask-size:100% 100%; mask-repeat:no-repeat; }
.kjic-div { position:absolute; transform:rotate(var(--a, 0deg)); }
.kjic-div::before, .kjic-div::after { content:""; position:absolute; transform:translate(-50%, -50%); }
.kjic-div::before { height:calc(200cqw + 200cqh); border-left:1px solid rgba(255,255,255,0.75); filter:drop-shadow(0 0 1px rgba(0,0,0,0.8)); }
.kjic-div::after { width:24px; height:24px; box-sizing:border-box; border-radius:50%; border:1.5px solid rgba(255,255,255,0.85); box-shadow:0 0 0 1.5px rgba(0,0,0,0.4); }
.kjic-div.hover::after { border-color:#8af; background:rgba(130,170,255,0.25); }
.kjic-box { position:absolute; border:1px dashed rgba(255,255,255,0.7); box-shadow:0 0 0 1px rgba(0,0,0,0.35); }
.kjic-badge { position:absolute; left:6px; top:6px; padding:1px 6px; border-radius:3px; background:rgba(0,0,0,0.6); color:#fff; font-weight:bold; }
.kjic-bar { display:flex; flex:none; flex-wrap:wrap; align-items:center; gap:4px; padding:4px 2px 0; }
.kjic-group { display:flex; align-items:center; gap:3px; }
.kjic-btn { padding:2px 6px; font:inherit; color:inherit; border:1px solid var(--border-color, #555); border-radius:3px; cursor:pointer; background:var(--comfy-input-bg, #333); }
.kjic-btn:hover, .kjic-btn.active { border-color:#8af; }
.kjic-btn.active { background:#556; color:#fff; }
.kjic-btn.off { opacity:0.4; }
.kjic-boost { width:90px; }
.kjic-hint { flex:1 1 0; min-width:0; opacity:0.5; font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; text-align:right; }
.kjic-video { flex:1 0 100%; }
.kjic-video input { flex:1; min-width:60px; }
.kjic-frame { min-width:56px; text-align:right; font-variant-numeric:tabular-nums; }
`;

function el(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  parent?.appendChild(e);
  return e;
}

// Coalesces calls into one per animation frame
function rafOnce(fn) {
  let queued = false;
  return () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; fn(); });
  };
}

async function loadImageEl(url) {
  const img = new Image();
  img.src = url;
  try { await img.decode(); } catch { return null; }
  return img;
}

function releaseVideo(v) {
  v.pause();
  v.removeAttribute("src");
  v.load();
}

function loadVideoEl(url) {
  const v = Object.assign(document.createElement("video"), { muted: true, playsInline: true, preload: "auto" });
  return new Promise((resolve) => {
    const done = (ok) => {
      clearTimeout(timer);
      v.onloadeddata = v.onerror = null;
      if (!ok) releaseVideo(v);
      resolve(ok ? v : null);
    };
    const timer = setTimeout(done, 15000, false);
    v.onloadeddata = () => done(true);
    v.onerror = () => done(false);
    v.src = url;
  });
}

// Source rect is the item's content size, which leaves out the edge padding of odd-sized videos
function drawContain(ctx, { el, w, h }, cw, ch) {
  const s = Math.min(cw / w, ch / h);
  ctx.drawImage(el, 0, 0, w, h, (cw - w * s) / 2, (ch - h * s) / 2, w * s, h * s);
}

// Half-plane past a divider along its axis, as a clip-path in stage percent; geometry runs in reference
// pixels so angles match what is seen, percentages keep it valid at any display size
function halfPlaneClip(c, angle, W, H) {
  const [nx, ny] = axisOf(angle);
  const px = c.x * W, py = c.y * H, L = (W + H) * 2;
  const pts = [
    [px - ny * L, py + nx * L], [px + ny * L, py - nx * L],
    [px + ny * L + nx * L, py - nx * L + ny * L], [px - ny * L + nx * L, py + nx * L + ny * L],
  ];
  return `polygon(${pts.map(([x, y]) => `${(x / W * 100).toFixed(3)}% ${(y / H * 100).toFixed(3)}%`).join(", ")})`;
}

function snapAngle(a, fine) {
  const q = Math.round(a / 90) * 90;
  return wrap360(fine ? Math.round(a / 15) * 15 : Math.abs(a - q) < 3 ? q : a);
}

function sanitizeState(p) {
  const s = structuredClone(DEFAULT_STATE), fin = Number.isFinite;
  if (!p || typeof p !== "object") return s;
  if (MODES.includes(p.mode)) s.mode = p.mode;
  if (Array.isArray(p.dividers) && p.dividers.every((d) => fin(d?.x) && fin(d?.y))) s.dividers = p.dividers.map((d) => ({ x: clamp01(d.x), y: clamp01(d.y) }));
  if (fin(p.angle)) s.angle = wrap360(p.angle);
  const b = p.box;
  if (b && [b.x, b.y, b.w, b.h].every(fin) && b.w > 0 && b.h > 0) s.box = { x: b.x, y: b.y, w: b.w, h: b.h };
  for (const k of ["toggleIdx", "cmpIdx"]) if (Number.isInteger(p[k]) && p[k] >= 0) s[k] = p[k];
  if (fin(p.boostExp)) s.boostExp = clamp(p.boostExp, 0, BOOST_MAX_EXP);
  if (typeof p.showHandle === "boolean") s.showHandle = p.showHandle;
  return s;
}

app.registerExtension({
  name: "KJNodes.ImageCompare",

  // Like core previews, results live in the frontend's per-tab output store, not the workflow
  afterLoadGraph() {
    for (const restore of restorers) restore();
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "ImageCompareKJ") return;
    if (!document.getElementById(STYLE_ID)) document.head.appendChild(Object.assign(document.createElement("style"), { id: STYLE_ID, textContent: STYLE }));

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      let st = sanitizeState(null);
      let media = []; // { el, kind: "image"|"video"|"canvas", frames, url, w, h } with w/h the content size
      let refW = 1, refH = 1, maskUrl = null, dataFps = 0, viewSource = null; // viewSource: "exec" | "preview"
      let loadGen = 0, restoring = null, autoFit = true, fitQueued = false, configureSkip = false, sourceSig = null;
      let hoverIdx = -1, diffKey = "", dragMove = null, refreshTimer = null;
      let playing = false, master = null, frameReq = null;

      const root = el("div", "kjic-root");
      const wrap = el("div", "kjic-wrap", root);
      el("div", "kjic-empty", wrap, "Connect images or videos, or run the workflow to compare");
      const stage = el("div", "kjic-stage", wrap);
      const layers = el("div", "kjic-layers", stage);
      const diffCanvas = el("canvas", "kjic-layer", layers);
      const diffCtx = diffCanvas.getContext("2d");
      const overlay = el("div", "kjic-overlay", stage);
      const boxEl = el("div", "kjic-box", overlay);
      const badge = el("div", "kjic-badge", overlay);
      const divEls = [];

      const btn = (parent, text, onclick, title = "") => Object.assign(el("button", "kjic-btn", parent, text), { onclick, title });
      const bar = el("div", "kjic-bar", root);
      const modeGroup = el("div", "kjic-group", bar);
      const modeBtns = Object.fromEntries(MODES.map((m) => [m, btn(modeGroup, m[0].toUpperCase() + m.slice(1), () => {
        if (m === "toggle" && st.mode === "toggle") st.toggleIdx = (st.toggleIdx + 1) % Math.max(1, media.length);
        st.mode = m;
        commit();
      })]));
      const cmpBtn = btn(bar, "", () => { st.cmpIdx = cmpIdx() % (media.length - 1) + 1; commit(); }, "Image compared against A");
      const handleBtn = btn(bar, "◎", () => { st.showHandle = !st.showHandle; commit(); }, "Show or hide the dividers");
      const boostGroup = el("div", "kjic-group", bar);
      const boostSlider = Object.assign(el("input", "kjic-boost", boostGroup), {
        type: "range", min: 0, max: BOOST_MAX_EXP, step: 0.05, title: "Difference boost",
        oninput: () => { st.boostExp = parseFloat(boostSlider.value) || 0; scheduleApply(); },
        onchange: saveState,
      });
      const boostLabel = el("span", null, boostGroup);
      const hint = el("span", "kjic-hint", bar);
      const videoGroup = el("div", "kjic-group kjic-video", bar);
      const playBtn = btn(videoGroup, "▶", () => setPlaying(!playing));
      const scrubber = Object.assign(el("input", null, videoGroup), {
        type: "range", min: 0, max: 0, step: 1, value: 0,
        oninput: () => seekFrame(parseInt(scrubber.value, 10) || 0),
      });
      const frameLabel = el("span", "kjic-frame", videoGroup);

      const cleanupPan = addMiddleClickPan(root);
      addWheelPassthrough(root);
      // Left clicks on the bar's empty space shouldn't start a node drag
      root.addEventListener("mousedown", (e) => { if (e.button === 0) e.stopPropagation(); });

      node.addDOMWidget("compare_preview", "ImageCompareWidget", root, {
        serialize: false, hideOnZoom: false, getMinHeight: () => MIN_HEIGHT, onHide: () => setPlaying(false),
      });
      node.setSize([Math.max(node.size[0], 360), node.size[1]]);

      const fpsWidget = node.widgets?.find((w) => w.name === "fps");
      const fps = () => dataFps || fpsWidget?.value || 24;
      const mode = () => (st.mode === "mask" && !maskUrl ? "slider" : st.mode);
      const cmpIdx = () => clamp(st.cmpIdx, 1, Math.max(1, media.length - 1));
      const videoItems = () => media.filter((m) => m.kind === "video");
      const frameCount = () => Math.max(1, ...media.map((m) => m.frames));
      const boost = () => 2 ** st.boostExp;
      const scheduleApply = rafOnce(applyView), scheduleDiff = rafOnce(renderDiff);

      function saveState() { node.properties.imageCompare = JSON.parse(JSON.stringify(st)); }
      function commit() { saveState(); applyView(); }

      function applyView() {
        const m = mode(), n = media.length, cmp = cmpIdx(), b = st.box;
        media.forEach((item, i) => {
          let visible = true, clip = "", mask = "";
          if (m === "slider") clip = i > 0 && st.dividers[i - 1] ? halfPlaneClip(st.dividers[i - 1], st.angle, refW, refH) : "";
          else if (m === "toggle") visible = i === st.toggleIdx % n;
          else if (m === "diff") visible = n < 2 && i === 0;
          else {
            visible = i === 0 || i === cmp;
            if (i === cmp && m === "box") clip = b ? `inset(${b.y * 100}% ${(1 - b.x - b.w) * 100}% ${(1 - b.y - b.h) * 100}% ${b.x * 100}%)` : "inset(0 100% 0 0)";
            if (i === cmp && m === "mask") mask = `url("${maskUrl}")`;
          }
          Object.assign(item.el.style, { opacity: visible ? "" : "0", clipPath: clip, maskImage: mask });
        });
        const diff = m === "diff" && n > 1;
        diffCanvas.hidden = !diff;
        if (diff) {
          diffCanvas.style.filter = `grayscale(1) brightness(${boost()})`;
          // Boost, overlay and drags only touch CSS, video frames redraw from onFrame/onseeked
          const key = `${media[0].url}|${media[cmp].url}|${refW}x${refH}`;
          if (key !== diffKey) { diffKey = key; scheduleDiff(); }
        } else if (diffCanvas.width) {
          // Release the backing store, up to 64 MB at the size cap
          diffCanvas.width = diffCanvas.height = 0;
          diffKey = "";
        }

        overlay.style.setProperty("--a", `${st.angle}deg`);
        while (divEls.length < st.dividers.length) divEls.push(el("div", "kjic-div", overlay));
        divEls.forEach((d, i) => {
          const c = st.dividers[i];
          d.hidden = !(m === "slider" && st.showHandle && c && i < n - 1);
          if (!d.hidden) Object.assign(d.style, { left: `${c.x * 100}%`, top: `${c.y * 100}%` });
          d.classList.toggle("hover", i === hoverIdx);
        });
        boxEl.hidden = !(m === "box" && b);
        if (b) Object.assign(boxEl.style, { left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` });
        badge.textContent = m === "toggle" ? label(st.toggleIdx % Math.max(1, n)) : PAIR_MODES.has(m) && n > 2 ? `A vs ${label(cmp)}` : "";
        badge.hidden = !badge.textContent;

        for (const k of MODES) modeBtns[k].classList.toggle("active", k === m);
        modeBtns.mask.hidden = !maskUrl;
        modeBtns.toggle.textContent = m === "toggle" && n ? `Toggle ${label(st.toggleIdx % n)}` : "Toggle";
        cmpBtn.hidden = !(n > 2 && PAIR_MODES.has(m));
        cmpBtn.textContent = `vs ${label(cmp)}`;
        handleBtn.hidden = m !== "slider";
        handleBtn.classList.toggle("off", !st.showHandle);
        boostGroup.hidden = m !== "diff";
        boostSlider.value = st.boostExp;
        boostLabel.textContent = `${boost() < 10 ? boost().toFixed(1) : Math.round(boost())}x`;
        videoGroup.hidden = !videoItems().length;
        hint.textContent = hint.title = MODE_HINTS[m];
        stage.style.cursor = m === "slider" ? (Math.abs(Math.sin(st.angle * Math.PI / 180)) > 0.7 ? "row-resize" : "col-resize") : CURSORS[m];
      }

      // Diff at reference resolution, so fine differences survive the downscale to the node
      function renderDiff() {
        if (mode() !== "diff" || media.length < 2) return;
        const s = Math.min(1, DIFF_MAX_SIDE / Math.max(refW, refH));
        const cw = Math.max(1, Math.round(refW * s)), ch = Math.max(1, Math.round(refH * s));
        if (diffCanvas.width !== cw || diffCanvas.height !== ch) Object.assign(diffCanvas, { width: cw, height: ch });
        diffCtx.imageSmoothingQuality = "high";
        diffCtx.fillStyle = "#000";
        diffCtx.fillRect(0, 0, cw, ch);
        drawContain(diffCtx, media[0], cw, ch);
        diffCtx.globalCompositeOperation = "difference";
        drawContain(diffCtx, media[cmpIdx()], cw, ch);
        diffCtx.globalCompositeOperation = "source-over";
      }

      function clearMedia(keep) {
        setPlaying(false);
        for (const m of media) {
          if (keep?.has(m)) continue;
          if (m.kind === "video") releaseVideo(m.el);
          m.el.remove();
        }
        media = [];
        master = maskUrl = viewSource = null;
        root.classList.remove("has-media");
      }

      function setMedia(items, meta) {
        clearMedia(new Set(items));
        media = items;
        [refW, refH] = [items[0].w, items[0].h];
        maskUrl = meta.mask ?? null;
        dataFps = meta.fps ?? 0;
        viewSource = meta.source;
        for (const m of items) {
          layers.insertBefore(m.el, diffCanvas).classList.add("kjic-layer");
          const [nw, nh] = naturalSize(m.el);
          m.el.style.objectViewBox = nw > m.w || nh > m.h ? `inset(0 ${nw - m.w}px ${nh - m.h}px 0)` : "";
        }
        stage.style.setProperty("--kjic-ar", refW / refH);
        if (st.dividers.length !== items.length - 1) { st.dividers = defaultDividers(items.length); st.angle = 0; }
        st.toggleIdx %= Math.max(1, items.length);

        const vids = videoItems();
        master = vids.reduce((a, b) => (b.frames > a.frames ? b : a), vids[0])?.el ?? null;
        for (const { el: v } of vids) {
          if (v.currentTime) v.currentTime = 0;
          v.onseeked = () => { if (!playing) { updateFrameUI(); scheduleDiff(); } };
          v.onended = v === master ? () => { if (playing) restartAll(); } : null;
        }
        scrubber.max = frameCount() - 1;
        updateFrameUI();
        root.classList.add("has-media");
        applyView();
        fitQueued = true;
        fitNodeToMedia();
      }

      // Own element per source, an upstream video preview keeps playing undisturbed
      async function loadItem({ url, kind, frames, w, h }, rate) {
        let el = kind === "image" ? await loadImageEl(url) : await loadVideoEl(url);
        if (el && kind === "canvas") {
          const v = el;
          el = null;
          captureVideoFrame(v, (c) => { el = c; }); // synchronous once data is loaded
          releaseVideo(v);
        }
        if (!el) return null;
        // Streamed sources can report an Infinity duration
        frames ??= kind === "video" && Number.isFinite(el.duration) ? Math.max(1, Math.round(el.duration * rate)) : 1;
        const [nw, nh] = naturalSize(el);
        return { el, kind, frames, url, w: w ?? nw, h: h ?? nh };
      }

      // Items still showing the same file are kept as is, skipping the reload and decode.
      // Resolves true when shown, false when a file failed (the view stays as it was), undefined when superseded.
      async function loadMedia(specs, meta, maskSrc) {
        const gen = ++loadGen, pool = new Set(media);
        const reuse = ({ url, kind }) => {
          const m = [...pool].find((x) => x.url === url && x.kind === kind);
          pool.delete(m);
          return m;
        };
        const [items, mask] = await Promise.all([
          Promise.all(specs.map((s) => reuse(s) ?? loadItem(s, meta.fps))),
          maskSrc && (maskSrc === maskUrl || loadImageEl(maskSrc)),
        ]);
        if (gen === loadGen && items.every(Boolean)) {
          setMedia(items, { ...meta, mask: mask ? maskSrc : null });
          return true;
        }
        for (const m of items) if (m?.kind === "video" && !media.includes(m)) releaseVideo(m.el);
        return gen === loadGen ? false : undefined;
      }

      async function loadFromData(data, quiet = false) {
        const imgs = data?.images ?? [];
        if (!imgs.length) return;
        // By file type, a one-frame video still arrives as an mp4
        const specs = imgs.map((f) => ({ url: viewUrl(f), kind: /\.mp4$/i.test(f.filename) ? "video" : "image", frames: f.frames, w: f.width, h: f.height }));
        const ok = await loadMedia(specs, { fps: data.fps, source: "exec" }, data.mask && viewUrl(data.mask));
        // Temp is cleared on restart, so restored outputs fail quietly
        if (ok === false && !quiet) console.warn("[KJNodes] Image Compare: could not load the preview files", data);
      }

      // Tab switches and undo rebuild the node, the last result comes back from the frontend's output store
      function restoreOutput() {
        // Newer frontends clear graphs without onRemoved, drop nodes that left their graph
        if (node.graph?.getNodeById(node.id) !== node) return void restorers.delete(restoreOutput);
        const id = node.graph && node.graph !== app.rootGraph ? `${node.graph.id}:${node.id}` : String(node.id);
        const data = app.nodeOutputs?.[id]?.compare_data?.[0];
        if (data) restoring = loadFromData(data, true).finally(() => { restoring = null; });
      }
      restorers.add(restoreOutput);

      async function refreshSources() {
        if (restoring) await restoring;
        const slots = node.inputs?.flatMap((inp, i) => (isMediaInput(inp) && node.isInputConnected(i) ? [i] : [])) ?? [];
        const sources = slots.map((i) => resolveSourcePreview(node, i));
        const sig = JSON.stringify(slots.map((i, k) => [node.getInputLink(i)?.id, sources[k]?.videoEl?.src ?? sources[k]?.url ?? null]));
        const skip = configureSkip;
        configureSkip = false;
        // Mask or other non-image changes must not replace an execution result
        if (sig === sourceSig) return;
        sourceSig = sig;
        // Graph load also fires connection changes, keep the restored execution result then
        if (skip && viewSource === "exec") return;
        if (!slots.length) { ++loadGen; clearMedia(); return; }
        // Something unpreviewable (e.g. a VAE decode) is connected, keep what is shown
        if (sources.some((s) => !s)) return;
        // The source frame rate is unknown in the browser, frame steps use the fps widget as it was at load
        const allVideo = sources.every((s) => s.isVideo);
        await loadMedia(sources.map((s) => ({ url: s.videoEl?.src || s.url, kind: !s.isVideo ? "image" : allVideo ? "video" : "canvas" })), { source: "preview", fps: fpsWidget?.value || 24 });
      }

      const unwatchInputs = watchImageInputs(node, isMediaInput, () => {
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(refreshSources, 50);
      });
      chainCallback(node, "onConnectionsChange", function () {
        if (app.configuringGraph) configureSkip = true;
      });

      // Fresh nodes grow to the media aspect until the first execution result, loaded nodes are left alone
      function fitNodeToMedia() {
        if (!autoFit || !fitQueued || !media.length) return;
        const ww = wrap.clientWidth, wh = wrap.clientHeight;
        if (!ww) return; // not laid out yet, the ResizeObserver retries
        fitQueued = false;
        if (viewSource === "exec") autoFit = false;
        const want = Math.min(Math.round(ww * refH / refW), Math.max(wh, 768));
        if (want > wh) {
          node.setSize([node.size[0], node.size[1] + want - wh]);
          node.graph?.setDirtyCanvas(true, true);
        }
      }
      const resizeObserver = new ResizeObserver(fitNodeToMedia);
      resizeObserver.observe(wrap);

      const currentFrame = () => (master ? Math.min(frameCount() - 1, Math.floor(master.currentTime * fps() + 1e-3)) : 0);

      function updateFrameUI() {
        const f = currentFrame();
        scrubber.value = f;
        frameLabel.textContent = `${f + 1} / ${frameCount()}`;
      }

      function setPlaying(on) {
        const vids = videoItems().map((m) => m.el), was = playing;
        playing = on && vids.length > 0;
        playBtn.textContent = playing ? "⏸" : "▶";
        if (playing) {
          if (master.ended || master.currentTime >= master.duration - 0.5 / fps()) restartAll();
          // play() on an ended clip restarts it, shorter clips hold their last frame instead
          else for (const v of vids) if (!v.ended) v.play().catch(() => {});
          requestFrame();
          return;
        }
        for (const v of vids) v.pause();
        frameReq?.();
        frameReq = null;
        // Followers drift up to 1.5 frames while playing, a paused comparison shows one frame everywhere
        if (was) seekAll(currentFrame());
      }

      function restartAll() {
        for (const { el: v } of videoItems()) {
          v.currentTime = 0;
          if (playing) v.play().catch(() => {});
        }
      }

      // frameReq holds the cancel function of the pending frame callback
      function requestFrame() {
        if (frameReq || !master) return;
        const v = master;
        if (typeof v.requestVideoFrameCallback === "function") {
          const id = v.requestVideoFrameCallback(onFrame);
          frameReq = () => v.cancelVideoFrameCallback(id);
        } else {
          const id = requestAnimationFrame(onFrame);
          frameReq = () => cancelAnimationFrame(id);
        }
      }

      // Followers are only corrected when they drift, seeking every frame stutters
      function onFrame() {
        frameReq = null;
        if (!playing || !master) return;
        const t = master.currentTime, f = fps();
        for (const { el: v } of videoItems()) {
          if (v === master || t >= v.duration - 0.5 / f) continue; // shorter clips hold their last frame
          if (Math.abs(v.currentTime - t) > 1.5 / f) v.currentTime = t;
          if (v.paused && !v.ended) v.play().catch(() => {});
        }
        updateFrameUI();
        renderDiff();
        requestFrame();
      }

      function seekAll(f) {
        for (const m of videoItems()) m.el.currentTime = (Math.min(f, m.frames - 1) + 0.5) / fps();
        updateFrameUI();
      }

      function seekFrame(f) {
        playing = false; // skips the pause snap, the seek below replaces it
        setPlaying(false);
        seekAll(f);
      }

      function stagePoint(e) {
        const r = stage.getBoundingClientRect();
        return { r, nx: (e.clientX - r.left) / r.width, ny: (e.clientY - r.top) / r.height, sx: e.clientX - r.left, sy: e.clientY - r.top };
      }

      // Nearest divider by perpendicular screen distance, as [index, distance]. Stacked dividers tie,
      // the one on the pointer's side wins so a neighbour's clamp can't pin it
      function nearestDivider(p) {
        const [nx, ny] = axisOf(st.angle), k = p.r.width / refW;
        let best = [-1, Infinity];
        st.dividers.slice(0, media.length - 1).forEach((d, i) => {
          const sd = ((p.nx - d.x) * refW * nx + (p.ny - d.y) * refH * ny) * k, dist = Math.abs(sd);
          if (dist < best[1] - 0.5 || (dist <= best[1] + 0.5 && sd > 0)) best = [i, dist];
        });
        return best;
      }

      function hoveredDivider(p) {
        const [i, dist] = nearestDivider(p), c = st.dividers[i];
        return c && (dist < 8 || Math.hypot(p.sx - c.x * p.r.width, p.sy - c.y * p.r.height) < 16) ? i : -1;
      }

      // Moves divider i to a point, kept between its neighbours along the axis so every image stays reachable
      function placeDivider(i, p) {
        const [nx, ny] = axisOf(st.angle), proj = (d) => d.x * refW * nx + d.y * refH * ny;
        const x = p.nx * refW, y = p.ny * refH, s = x * nx + y * ny;
        const over = s - clamp(s, i > 0 ? proj(st.dividers[i - 1]) : -Infinity, i < st.dividers.length - 1 ? proj(st.dividers[i + 1]) : Infinity);
        st.dividers[i] = { x: clamp01((x - over * nx) / refW), y: clamp01((y - over * ny) / refH) };
      }

      // Rotates the whole split rigidly around handle i, other handles slide along their lines toward the center
      function startRotate(i, e) {
        const r = stage.getBoundingClientRect(), start = st.dividers.map((d) => ({ ...d })), startAngle = st.angle;
        const cx = r.left + start[i].x * r.width, cy = r.top + start[i].y * r.height;
        const px = start[i].x * refW, py = start[i].y * refH;
        let a0 = null;
        const update = (ev) => {
          const dx = ev.clientX - cx, dy = ev.clientY - cy;
          if (Math.hypot(dx, dy) < 10) return; // too close to the pivot for a stable angle
          const a = Math.atan2(dy, dx) * 180 / Math.PI;
          if (a0 === null) { a0 = a; return; }
          st.angle = snapAngle(startAngle + a - a0, ev.ctrlKey);
          const delta = (st.angle - startAngle) * Math.PI / 180, cos = Math.cos(delta), sin = Math.sin(delta);
          const [nx, ny] = axisOf(st.angle);
          st.dividers = start.map((d, j) => {
            if (j === i) return { ...d };
            const ox = d.x * refW - px, oy = d.y * refH - py, x = px + ox * cos - oy * sin, y = py + ox * sin + oy * cos;
            const t = (x - refW / 2) * ny - (y - refH / 2) * nx; // slide along the line (-ny, nx) to the center
            return { x: clamp01((x - t * ny) / refW), y: clamp01((y + t * nx) / refH) };
          });
        };
        update(e);
        return update;
      }

      const boxHit = (p, b = st.box) => b && rectHitTest(p.sx, p.sy, b.x * p.r.width, b.y * p.r.height, (b.x + b.w) * p.r.width, (b.y + b.h) * p.r.height, 8);

      function startBox(p) {
        const hit = boxHit(p), s = st.box && { ...st.box }, x0 = clamp01(p.nx), y0 = clamp01(p.ny);
        const fromCorners = (x1, y1, x2, y2) => ({ x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) });
        if (!hit) st.box = { x: x0, y: y0, w: 0, h: 0 };
        return (ev) => {
          const q = stagePoint(ev), dx = q.nx - p.nx, dy = q.ny - p.ny;
          if (!hit) st.box = fromCorners(x0, y0, clamp01(q.nx), clamp01(q.ny));
          else if (hit === "move") st.box = { ...s, x: clamp(s.x + dx, 0, 1 - s.w), y: clamp(s.y + dy, 0, 1 - s.h) };
          else {
            const e = hit.slice("resize-".length), x2 = s.x + s.w, y2 = s.y + s.h;
            st.box = fromCorners(e.includes("l") ? clamp01(s.x + dx) : s.x, e.includes("t") ? clamp01(s.y + dy) : s.y,
              e.includes("r") ? clamp01(x2 + dx) : x2, e.includes("b") ? clamp01(y2 + dy) : y2);
          }
        };
      }

      function onDocMove(e) {
        e.preventDefault();
        dragMove?.(e);
        scheduleApply();
      }

      function endDrag() {
        document.removeEventListener("mousemove", onDocMove);
        document.removeEventListener("mouseup", endDrag);
        if (!dragMove) return;
        dragMove = null;
        // A click or a sliver of a drag clears the box
        const b = st.box, r = stage.getBoundingClientRect();
        if (mode() === "box" && b && (b.w * r.width < 3 || b.h * r.height < 3)) st.box = null;
        commit();
      }

      stage.addEventListener("mousedown", (e) => {
        if (e.button !== 0 || !media.length) return;
        e.preventDefault();
        e.stopPropagation();
        const p = stagePoint(e), m = mode();
        if (m === "slider") {
          const [i] = nearestDivider(p);
          if (i < 0) return;
          if (e.shiftKey) dragMove = startRotate(i, e);
          else {
            placeDivider(i, p);
            dragMove = (ev) => placeDivider(i, stagePoint(ev));
          }
        } else if (m === "box") {
          dragMove = startBox(p);
        } else if (m === "diff") {
          const x0 = e.clientX, e0 = st.boostExp;
          dragMove = (ev) => { st.boostExp = clamp(e0 + (ev.clientX - x0) / 100, 0, BOOST_MAX_EXP); };
        } else if (m === "toggle") {
          st.toggleIdx = (st.toggleIdx + 1) % media.length;
          commit();
        }
        if (!dragMove) return;
        scheduleApply();
        document.addEventListener("mousemove", onDocMove);
        document.addEventListener("mouseup", endDrag);
      });

      stage.addEventListener("mousemove", (e) => {
        if (dragMove || !media.length) return;
        const p = stagePoint(e), m = mode();
        if (m === "box") stage.style.cursor = cursorForBboxMode(boxHit(p)) || "crosshair";
        const i = m === "slider" ? hoveredDivider(p) : -1;
        if (i !== hoverIdx) { hoverIdx = i; applyView(); }
      });

      stage.addEventListener("mouseleave", () => {
        if (hoverIdx !== -1) { hoverIdx = -1; applyView(); }
      });

      stage.addEventListener("dblclick", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const m = mode();
        if (m === "slider") { st.dividers = defaultDividers(media.length); st.angle = 0; }
        else if (m === "box") st.box = null;
        else if (m === "diff") st.boostExp = 0;
        else if (m === "toggle") st.toggleIdx = 0;
        commit();
      });

      chainCallback(node, "onExecuted", function (message) {
        const data = message?.compare_data?.[0];
        if (data) loadFromData(data);
      });

      chainCallback(node, "onConfigure", function () {
        autoFit = false;
        st = sanitizeState(node.properties?.imageCompare);
        applyView();
      });

      chainCallback(node, "onRemoved", function () {
        restorers.delete(restoreOutput);
        ++loadGen;
        clearTimeout(refreshTimer);
        endDrag();
        clearMedia();
        resizeObserver.disconnect();
        cleanupPan();
        unwatchInputs();
      });

      saveState();
      applyView();
    });
  },
});
