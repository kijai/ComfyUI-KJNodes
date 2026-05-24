import { chainCallback, addMiddleClickPan, addWheelPassthrough } from '../utility.js';
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const STYLE_ID = "kj-pov-stylesheet";
const _cssUrl = new URL("./preview_override.css", import.meta.url).href;
function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const link = document.createElement("link");
    link.id = STYLE_ID;
    link.rel = "stylesheet";
    link.href = _cssUrl;
    document.head.appendChild(link);
}

// Walks subgraph chain for IDs like "12:7:5". Mirrors getNodeByExecutionId (not exported).
function findNodeByQualifiedId(rootGraph, qid) {
    if (!rootGraph || !qid) return null;
    const parts = String(qid).split(":");
    let graph = rootGraph;
    for (let i = 0; i < parts.length - 1; i++) {
        const parentId = parseInt(parts[i], 10);
        if (!Number.isFinite(parentId)) return null;
        const parentNode = graph?.getNodeById?.(parentId);
        if (!parentNode?.subgraph) return null;
        graph = parentNode.subgraph;
    }
    const leafId = parseInt(parts[parts.length - 1], 10);
    if (!Number.isFinite(leafId)) return null;
    return graph?.getNodeById?.(leafId) || null;
}

api.addEventListener("kj_preview_override", (e) => {
    const data = e.detail;
    if (!data || data.node_id == null) return;
    const node = findNodeByQualifiedId(app.graph, data.node_id);
    if (node?._kjPreviewHandler) node._kjPreviewHandler(data);
});

const GRAPH_PAD_X = 4;

function fmt(n, d) {
    return Number.isFinite(n) ? n.toFixed(d) : "—";
}

function el(tag, className, parent) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    if (parent) parent.appendChild(e);
    return e;
}

function b64ToBlob(b64, mime) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
}

function syncCanvasDPR(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.width;
    const cssH = canvas.clientHeight || canvas.height;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, W: cssW, H: cssH };
}

function drawGridlines(ctx, W, H, padX, padY) {
    ctx.strokeStyle = "#1e1e1e";
    ctx.lineWidth = 1;
    for (let g = 1; g < 4; g++) {
        const y = Math.round(padY + (g / 4) * (H - 2 * padY)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(padX, y);
        ctx.lineTo(W - padX, y);
        ctx.stroke();
    }
}

// σ + Δ overlaid; each series normalised to its own max for comparable shapes.
function drawSigmaDeltaGraph(canvas, sigmas, deltas, step, totalSteps, hoverStep, dbCurve, lockedStep) {
    const { ctx, W, H } = syncCanvasDPR(canvas);
    const padX = GRAPH_PAD_X, padY = 3;
    const iW = W - 2 * padX, iH = H - 2 * padY;
    ctx.clearRect(0, 0, W, H);
    drawGridlines(ctx, W, H, padX, padY);

    const n = sigmas?.length || 0;
    const xSteps = Math.max(totalSteps || n, n, deltas?.length || 0);
    const xAt = i => padX + (i / Math.max(1, xSteps - 1)) * iW;

    let sYAt = null;
    if (n > 1) {
        let sMax = -Infinity, sMin = Infinity;
        for (const s of sigmas) { if (s > sMax) sMax = s; if (s < sMin) sMin = s; }
        if (sMin > 0) sMin = 0;
        const sRange = Math.max(sMax - sMin, 1e-6);
        sYAt = v => padY + (1 - (v - sMin) / sRange) * iH;

        ctx.strokeStyle = "rgba(208, 208, 208, 0.55)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const px = padX + (i / (n - 1)) * iW;
            const py = sYAt(sigmas[i]);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        const i = Math.max(0, Math.min(n - 1, step));
        const mx = padX + (i / Math.max(1, n - 1)) * iW;
        const my = sYAt(sigmas[i]);
        ctx.fillStyle = "#d0d0d0";
        ctx.beginPath();
        ctx.arc(mx, my, 2.5, 0, Math.PI * 2);
        ctx.fill();
    }

    let dYAt = null;
    if (deltas && deltas.length >= 1) {
        let dMax = -Infinity;
        for (const v of deltas) if (Number.isFinite(v) && v > dMax) dMax = v;
        const dRange = Math.max(dMax, 1e-6);
        dYAt = v => padY + (1 - v / dRange) * iH;

        // delta[i] is plotted at boundary (i+1); flat-extend delta[0] back to boundary 0.
        const lastB = deltas.length;
        ctx.beginPath();
        ctx.moveTo(xAt(0), H - padY);
        ctx.lineTo(xAt(0), dYAt(deltas[0]));
        for (let i = 0; i < deltas.length; i++) ctx.lineTo(xAt(i + 1), dYAt(deltas[i]));
        ctx.lineTo(xAt(lastB), H - padY);
        ctx.closePath();
        ctx.fillStyle = "rgba(230, 126, 34, 0.15)";
        ctx.fill();

        ctx.strokeStyle = "#e67e22";
        ctx.lineWidth = 1.3;
        if (deltas.length === 1) {
            ctx.fillStyle = "#e67e22";
            ctx.beginPath();
            ctx.arc(xAt(1), dYAt(deltas[0]), 2, 0, Math.PI * 2);
            ctx.fill();
        } else {
            ctx.beginPath();
            for (let i = 0; i < deltas.length; i++) {
                const px = xAt(i + 1), py = dYAt(deltas[i]);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }
    }

    // SamplerDetailBoost curve. `null` entries = outside the gate (drawn as faint baseline),
    // finite entries = active (drawn as the bright cyan line, auto-scaled to its own peak).
    if (Array.isArray(dbCurve) && dbCurve.length > 1) {
        let dbMaxAbs = 0;
        for (const v of dbCurve) if (Number.isFinite(v) && Math.abs(v) > dbMaxAbs) dbMaxAbs = Math.abs(v);
        const dbXAt = i => padX + (i / Math.max(1, dbCurve.length - 1)) * iW;
        const baselineY = H - padY - 0.5;
        const isActive = v => Number.isFinite(v);

        ctx.strokeStyle = "rgba(120, 200, 220, 0.25)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        let inactiveOpen = false;
        for (let i = 0; i < dbCurve.length; i++) {
            if (!isActive(dbCurve[i])) {
                if (!inactiveOpen) { ctx.moveTo(dbXAt(i), baselineY); inactiveOpen = true; }
                else ctx.lineTo(dbXAt(i), baselineY);
            } else {
                inactiveOpen = false;
            }
        }
        ctx.stroke();

        if (dbMaxAbs > 1e-9) {
            const dbYAt = v => padY + (1 - Math.abs(v) / dbMaxAbs) * iH;
            ctx.strokeStyle = "rgba(120, 200, 220, 0.85)";
            ctx.fillStyle = "rgba(120, 200, 220, 0.85)";
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            let segStart = -1;
            for (let i = 0; i <= dbCurve.length; i++) {
                const v = i < dbCurve.length ? dbCurve[i] : null;
                if (isActive(v)) {
                    if (segStart < 0) { segStart = i; ctx.moveTo(dbXAt(i), dbYAt(v)); }
                    else ctx.lineTo(dbXAt(i), dbYAt(v));
                } else if (segStart >= 0) {
                    if (i - segStart === 1) {
                        const v0 = dbCurve[segStart];
                        ctx.fillRect(dbXAt(segStart) - 1, dbYAt(v0) - 1, 2, 2);
                    }
                    segStart = -1;
                }
            }
            ctx.stroke();
        }
    }

    if (lockedStep != null && lockedStep >= 0 && lockedStep < xSteps) {
        const lx = xAt(lockedStep) + 0.5;
        ctx.strokeStyle = "rgba(245, 200, 60, 0.9)";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 2]);
        ctx.beginPath();
        ctx.moveTo(lx, padY);
        ctx.lineTo(lx, H - padY);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    if (hoverStep != null && hoverStep >= 0 && hoverStep < xSteps) {
        const hx = xAt(hoverStep) + 0.5;
        ctx.strokeStyle = "rgba(208, 208, 208, 0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(hx, padY);
        ctx.lineTo(hx, H - padY);
        ctx.stroke();
        if (sYAt && hoverStep < n) {
            ctx.fillStyle = "#d0d0d0";
            ctx.beginPath();
            ctx.arc(hx - 0.5, sYAt(sigmas[hoverStep]), 2.5, 0, Math.PI * 2);
            ctx.fill();
        }
        // delta[k-1] is plotted at boundary k.
        if (dYAt && hoverStep >= 1 && (hoverStep - 1) < deltas.length) {
            ctx.fillStyle = "#e67e22";
            ctx.beginPath();
            ctx.arc(hx - 0.5, dYAt(deltas[hoverStep - 1]), 2.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

// totalSteps fixes the x-axis so the line grows left-to-right, not stretching to fill.
function drawLineGraph(canvas, values, totalSteps) {
    const { ctx, W, H } = syncCanvasDPR(canvas);
    const padX = GRAPH_PAD_X, padY = 3;
    const iW = W - 2 * padX, iH = H - 2 * padY;
    ctx.clearRect(0, 0, W, H);
    drawGridlines(ctx, W, H, padX, padY);
    if (!values || values.length < 1) return;

    let vMax = -Infinity, vMin = Infinity;
    for (const v of values) { if (v > vMax) vMax = v; if (v < vMin) vMin = v; }
    if (vMin > 0) vMin = 0;
    const vRange = Math.max(vMax - vMin, 1e-6);

    const xSteps = Math.max(totalSteps || values.length, values.length);

    ctx.beginPath();
    ctx.moveTo(padX, H - padY);
    for (let i = 0; i < values.length; i++) {
        const px = padX + (i / Math.max(1, xSteps - 1)) * iW;
        const py = padY + (1 - (values[i] - vMin) / vRange) * iH;
        ctx.lineTo(px, py);
    }
    ctx.lineTo(padX + ((values.length - 1) / Math.max(1, xSteps - 1)) * iW, H - padY);
    ctx.closePath();
    ctx.fillStyle = "rgba(230, 126, 34, 0.15)";
    ctx.fill();

    ctx.strokeStyle = "#e67e22";
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
        const px = padX + (i / Math.max(1, xSteps - 1)) * iW;
        const py = padY + (1 - (values[i] - vMin) / vRange) * iH;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
}

app.registerExtension({
    name: "KJNodes.ModelPreviewOverride",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "ModelPreviewOverrideKJ") return;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            ensureStyles();
            const node = this;

            const root = el("div", "kj-pov-root");

            const imageArea = el("div", "kj-pov-image-area", root);
            // Double-buffered: decode() on the visible-to-be element so the bitmap is reused.
            const imgA = el("img", "kj-pov-img", imageArea);
            const imgB = el("img", "kj-pov-img", imageArea);
            imgA.draggable = false;
            imgB.draggable = false;
            imgB.style.opacity = "0";
            let visibleImg = imgA;
            let pendingImg = imgB;
            const img = imgA;  // alias used by class-toggle code; applied to both buffers
            // WebP path: ImageDecoder → VideoFrame[] → canvas, driven by a global timer.
            const videoCanvas = el("canvas", "kj-pov-img", imageArea);
            videoCanvas.style.opacity = "0";
            const videoCtx = videoCanvas.getContext("2d");
            // MP4 path: double-buffered <video> for same no-flash reason as imgA/B.
            function mkVideo() {
                const v = el("video", "kj-pov-img", imageArea);
                v.style.opacity = "0";
                v.muted = true;
                v.playsInline = true;
                v.loop = true;
                v.autoplay = true;
                v.disablePictureInPicture = true;
                return v;
            }
            const videoA = mkVideo();
            const videoB = mkVideo();
            let visibleVideo = videoA;
            let pendingVideo = videoB;
            // Hidden audio sources (LTXAV). Double-buffered like the video for gapless step
            // swaps; both ride the same playback clock as the visual preview.
            function mkAudio() {
                const a = el("audio", null, imageArea);
                a.style.display = "none";
                a.preload = "auto";
                a.loop = true;
                a.muted = true;  // unmuted only while hovering the preview or step plot
                return a;
            }
            const audioA = mkAudio();
            const audioB = mkAudio();
            let visibleAudio = audioA;
            let pendingAudio = audioB;
            // Keep audio quiet during normal graph work; it "opens up" only when the cursor is
            // over the preview frame or the σ/Δ plot. Starting muted also guarantees autoplay.
            let audioHoverImage = false;
            let audioHoverPlot = false;
            function refreshAudioMute() {
                const muted = !(audioHoverImage || audioHoverPlot);
                audioA.muted = muted;
                audioB.muted = muted;
            }
            const placeholder = el("div", "kj-pov-placeholder", imageArea);
            placeholder.textContent = "waiting for sample…";
            // Playback scrub bar — hidden until animated content is loaded.
            const scrubBar = el("div", "kj-pov-scrub", imageArea);
            const scrubFill = el("div", "kj-pov-scrub-fill", scrubBar);
            scrubBar.style.display = "none";

            const grip = el("div", "kj-pov-panel-grip", root);
            grip.title = "Drag to resize panel";

            const panel = el("div", "kj-pov-panel", root);
            const header = el("div", "kj-pov-panel-header", panel);
            const headerLeft = el("span", "kj-pov-header-left", header);
            const headerTitle = el("span", "kj-pov-panel-title", headerLeft);
            headerTitle.textContent = "Preview Override";
            const headerSummary = el("span", "kj-pov-panel-summary", header);
            headerSummary.textContent = "idle";

            node.properties = node.properties || {};
            if (typeof node.properties.kjPovPanelH === "number") {
                panel.style.height = node.properties.kjPovPanelH + "px";
            }

            grip.addEventListener("mousedown", (e) => {
                if (e.button !== 0) return;
                e.preventDefault();
                e.stopPropagation();
                const startY = e.clientY;
                const startH = panel.offsetHeight;
                const move = (ev) => {
                    const scale = app.canvas?.ds?.scale || 1;
                    const dy = (ev.clientY - startY) / scale;
                    const rootRect = root.getBoundingClientRect();
                    const maxH = Math.max(60, rootRect.height / scale - 80);
                    const newH = Math.max(60, Math.min(maxH, startH - dy));
                    panel.style.height = newH + "px";
                };
                const up = () => {
                    document.removeEventListener("mousemove", move);
                    document.removeEventListener("mouseup", up);
                    node.properties.kjPovPanelH = panel.offsetHeight;
                    node.graph?.change?.();
                };
                document.addEventListener("mousemove", move);
                document.addEventListener("mouseup", up);
            });

            chainCallback(node, "onConfigure", function () {
                if (typeof node.properties?.kjPovPanelH === "number") {
                    panel.style.height = node.properties.kjPovPanelH + "px";
                }
            });

            const grid = el("div", "kj-pov-graphs-grid", panel);

            function makeCell(labelText) {
                const cell = el("div", "kj-pov-graph-cell", grid);
                const head = el("div", "kj-pov-graph-head", cell);
                const lbl = el("span", "kj-pov-graph-label", head);
                lbl.textContent = labelText;
                const valEl = el("span", "kj-pov-graph-value", head);
                const canvas = el("canvas", "kj-pov-graph-canvas", cell);
                return { canvas, valEl, lbl };
            }
            const sdRow = makeCell("σ / Δ");
            const timeRow = makeCell("step time (ms)");
            sdRow.canvas.style.cursor = "crosshair";
            // sdRow.canvas.title = "Hover to scrub steps · click to lock · ← → to step · click again to unlock";
            timeRow.canvas.title = "Click to toggle ms ↔ s";

            // Click step-time cell to toggle ms ↔ s. Initial unit auto-picked from first value.
            let timeUnitSeconds = false;
            let lastStepMs = null;
            let lastAvgStepMs = null;
            let lastTotal = null;
            let lastStep = null;
            let lastW = null, lastH = null;
            function fmtTime(ms) {
                if (ms == null || !Number.isFinite(ms)) return "—";
                return timeUnitSeconds ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(0)}ms`;
            }
            // history.stepMs[k-1] = step k's duration.
            function stepTimeForDisplay() {
                let idx = lastCurrentStep;
                if (hoverStep != null) idx = hoverStep;
                else if (lockedStep != null) idx = lockedStep;
                const tIdx = idx - 1;
                if (tIdx >= 0 && tIdx < history.stepMs.length) return history.stepMs[tIdx];
                return null;
            }
            function renderTime() {
                timeRow.lbl.textContent = timeUnitSeconds ? "step time (s)" : "step time (ms)";
                timeRow.valEl.textContent = fmtTime(stepTimeForDisplay());
                if (lastAvgStepMs != null && lastTotal != null && lastStep != null) {
                    const eta = Math.max(0, lastTotal - lastStep) * lastAvgStepMs / 1000;
                    const avgTxt = timeUnitSeconds
                        ? `${(lastAvgStepMs / 1000).toFixed(2)}s/step`
                        : `${lastAvgStepMs.toFixed(0)}ms/step`;
                    headerSummary.textContent = `${lastW}×${lastH} · ${lastStep}/${lastTotal} · ${avgTxt} · ETA ${eta.toFixed(1)}s`;
                }
            }
            timeRow.canvas.style.cursor = "pointer";
            timeRow.lbl.style.cursor = "pointer";
            timeRow.valEl.style.cursor = "pointer";
            const toggleTimeUnit = (ev) => {
                ev.stopPropagation();
                timeUnitSeconds = !timeUnitSeconds;
                renderTime();
            };
            timeRow.canvas.addEventListener("click", toggleTimeUnit);
            timeRow.lbl.addEventListener("click", toggleTimeUnit);
            timeRow.valEl.addEventListener("click", toggleTimeUnit);

            // Per-run state. Declared together so helpers/handlers below can close over them.
            let hoverStep = null;
            let lockedStep = null;  // click-locked step; survives mouseleave for inspection
            let lastCurrentStep = -1;
            const history = { stepMs: [], delta: [] };
            let cachedSigmas = null;
            let cachedDbCurve = null;
            let totalSteps = 0;
            const stepBlobUrls = [];
            const stepVideoFrames = [];  // WebP path: per-step VideoFrame[]
            const stepMp4Urls = [];      // MP4 path: per-step blob URL
            const stepAudioUrls = [];    // LTXAV: per-step WAV blob URL
            let liveBlobUrl = null;
            let liveMp4Url = null;
            let liveAudioUrl = null;
            let currentAudioUrl = null;
            // Global timer never resets between steps — scrub continues at the equivalent elapsed.
            let playbackStartMs = null;
            let videoRafId = null;
            let currentMp4Url = null;

            const SD_LEG_BASE =
                '<span title="sigma — noise level at this boundary" style="color:#d0d0d0">σ</span>'
                + ' / '
                + '<span title="delta — magnitude of x0 change per step (orange fill = filled area, larger spike = bigger change)" style="color:#e67e22">Δ</span>';
            const SD_LEG_WITH_DB = SD_LEG_BASE
                + ' / '
                + '<span title="SamplerDetailBoost curve — per-σ boost applied by the wrapped sampler (peak = amount setting)" style="color:rgb(120,200,220)">DB</span>';
            function applySdLegend() {
                sdRow.lbl.innerHTML = cachedDbCurve ? SD_LEG_WITH_DB : SD_LEG_BASE;
            }
            applySdLegend();

            // Load into hidden buffer, await its decode(), then opacity-swap. See note on the
            // imgA/imgB definitions about why decode must run on the soon-to-be-visible element.
            function showLiveFrame(url) {
                const target = pendingImg;
                target.src = url;
                target.decode().then(() => {
                    if (hoverStep != null || lockedStep != null) return;
                    if (liveBlobUrl !== url) return;
                    if (target !== pendingImg) return;  // a newer frame already swapped
                    target.style.opacity = "1";
                    visibleImg.style.opacity = "0";
                    const prev = visibleImg;
                    visibleImg = target;
                    pendingImg = prev;
                }).catch(() => {});
            }
            // Baked WebP durations are dropped; rate comes live from preview_fps.
            async function decodeAnimatedBlob(blob) {
                if (typeof ImageDecoder === "undefined") return null;
                try {
                    const decoder = new ImageDecoder({ data: blob.stream(), type: blob.type });
                    await decoder.completed;
                    const track = decoder.tracks.selectedTrack;
                    if (!track || track.frameCount <= 1) { decoder.close?.(); return null; }
                    const frames = [];
                    for (let i = 0; i < track.frameCount; i++) {
                        const r = await decoder.decode({ frameIndex: i });
                        frames.push(r.image);
                    }
                    decoder.close?.();
                    return { frames };
                } catch {
                    return null;
                }
            }
            function closeStepVideo(stepIdx) {
                const v = stepVideoFrames[stepIdx];
                if (!v) return;
                for (const f of v.frames) { try { f.close(); } catch {} }
                stepVideoFrames[stepIdx] = null;
            }
            function activeStepIdx() {
                if (lockedStep != null) return lockedStep;
                if (hoverStep != null) return hoverStep;
                return lastCurrentStep;
            }
            // Read preview_fps live each tick so the widget retimes playback without re-encoding.
            const fpsWidget = node.widgets?.find(w => w.name === "preview_fps");
            function currentFps() {
                const v = +fpsWidget?.value;
                return Number.isFinite(v) && v > 0 ? v : 12;
            }

            // Pause state for the WebP path. MP4 uses videoEl.paused natively.
            let isPaused = false;
            let pauseAtMs = 0;
            function elapsedMs() {
                if (playbackStartMs == null) return 0;
                return (isPaused ? pauseAtMs : performance.now()) - playbackStartMs;
            }
            // MP4 encode-time fps; scales videoEl.playbackRate for live preview_fps retime.
            let bakedFps = null;

            function audioActive() {
                return currentAudioUrl != null && Number.isFinite(visibleAudio.duration) && visibleAudio.duration > 0;
            }
            // Mirror showMp4: load into the pending element, seek to the master clock, play,
            // then swap — keeps audio gapless across per-step buffer replacements.
            function showAudio(url) {
                if (url === currentAudioUrl) return;
                currentAudioUrl = url;
                if (playbackStartMs == null) playbackStartMs = performance.now();
                const target = pendingAudio;
                target.src = url;
                const promote = () => {
                    if (currentAudioUrl !== url || target !== pendingAudio) return;
                    try {
                        const dur = target.duration;
                        if (Number.isFinite(dur) && dur > 0) target.currentTime = (elapsedMs() / 1000) % dur;
                    } catch {}
                    if (!isPaused) target.play().catch(() => {});
                    const prev = visibleAudio;
                    prev.pause();
                    visibleAudio = target;
                    pendingAudio = prev;
                };
                const onLoaded = () => {
                    target.removeEventListener("loadeddata", onLoaded);
                    promote();
                };
                if (target.readyState >= 2) onLoaded();
                else target.addEventListener("loadeddata", onLoaded, { once: true });
            }
            function stopAudio() {
                currentAudioUrl = null;
                for (const a of [audioA, audioB]) {
                    try { a.pause(); a.removeAttribute("src"); a.load(); } catch {}
                }
            }

            function mp4Active() {
                return currentMp4Url != null && Number.isFinite(visibleVideo.duration) && visibleVideo.duration > 0;
            }
            function clipDurationMs() {
                if (mp4Active()) return visibleVideo.duration * 1000;
                const v = stepVideoFrames[activeStepIdx()];
                if (v && v.frames.length > 1) return v.frames.length * (1000 / currentFps());
                // Audio drives the scrub timeline when the video preview is a still frame.
                if (audioActive()) return visibleAudio.duration * 1000;
                return 0;
            }
            function getProgress() {
                if (mp4Active()) return visibleVideo.currentTime / visibleVideo.duration;
                const dur = clipDurationMs();
                if (dur <= 0 || playbackStartMs == null) return 0;
                return (elapsedMs() % dur) / dur;
            }
            function setProgress(pos) {
                pos = Math.max(0, Math.min(1, pos));
                const dur = clipDurationMs();
                if (dur <= 0) return;
                // Sync the global timer so cross-clip scrub picks up here.
                const ref = isPaused ? pauseAtMs : performance.now();
                if (playbackStartMs == null) playbackStartMs = ref;
                playbackStartMs = ref - pos * dur;
                if (mp4Active()) {
                    visibleVideo.currentTime = pos * visibleVideo.duration;
                }
                if (audioActive()) {
                    try { visibleAudio.currentTime = ((pos * dur) / 1000) % visibleAudio.duration; } catch {}
                }
            }
            function togglePause() {
                const willPause = !isPaused;
                if (mp4Active()) {
                    if (willPause) visibleVideo.pause();
                    else visibleVideo.play().catch(() => {});
                }
                if (audioActive()) {
                    if (willPause) visibleAudio.pause();
                    else visibleAudio.play().catch(() => {});
                }
                if (willPause) {
                    pauseAtMs = performance.now();
                    isPaused = true;
                } else {
                    if (playbackStartMs != null) playbackStartMs += performance.now() - pauseAtMs;
                    isPaused = false;
                }
                scrubBar.classList.toggle("kj-pov-paused", isPaused);
            }
            let scrubRafId = null;
            function tickScrub() {
                scrubRafId = requestAnimationFrame(tickScrub);
                const dur = clipDurationMs();
                if (dur > 0) {
                    if (scrubBar.style.display === "none") scrubBar.style.display = "block";
                    scrubFill.style.width = (getProgress() * 100) + "%";
                } else if (scrubBar.style.display !== "none") {
                    scrubBar.style.display = "none";
                }
                if (mp4Active() && bakedFps != null && bakedFps > 0) {
                    const rate = currentFps() / bakedFps;
                    if (Math.abs(visibleVideo.playbackRate - rate) > 0.001) {
                        visibleVideo.playbackRate = rate;
                    }
                }
                // Keep audio aligned to the master clock. Native playback advances on its own,
                // so we only re-seek on meaningful drift (after a step swap or a scrub).
                if (audioActive()) {
                    if (isPaused) {
                        if (!visibleAudio.paused) visibleAudio.pause();
                    } else {
                        if (visibleAudio.paused) visibleAudio.play().catch(() => {});
                        const want = (elapsedMs() / 1000) % visibleAudio.duration;
                        if (Math.abs(visibleAudio.currentTime - want) > 0.25) {
                            try { visibleAudio.currentTime = want; } catch {}
                        }
                    }
                }
            }
            scrubRafId = requestAnimationFrame(tickScrub);
            scrubBar.addEventListener("mousedown", (ev) => {
                if (ev.button !== 0) return;
                ev.stopPropagation();
                ev.preventDefault();
                const rect = scrubBar.getBoundingClientRect();
                const seek = (e) => setProgress((e.clientX - rect.left) / rect.width);
                seek(ev);
                const move = (e) => seek(e);
                const up = () => {
                    document.removeEventListener("mousemove", move);
                    document.removeEventListener("mouseup", up);
                };
                document.addEventListener("mousemove", move);
                document.addEventListener("mouseup", up);
            });
            // Click on the preview frame toggles pause (animated content only).
            imageArea.addEventListener("click", (ev) => {
                if (scrubBar.contains(ev.target)) return;
                if (clipDurationMs() <= 0) return;
                ev.stopPropagation();
                togglePause();
            });
            function drawCurrentVideoFrame() {
                const idx = activeStepIdx();
                const v = stepVideoFrames[idx];
                if (!v) return false;
                if (playbackStartMs == null) playbackStartMs = performance.now();
                const frameDurMs = 1000 / currentFps();
                const totalMs = v.frames.length * frameDurMs;
                const elapsed = elapsedMs() % totalMs;
                const fIdx = Math.min(v.frames.length - 1, Math.floor(elapsed / frameDurMs));
                const frame = v.frames[fIdx];
                if (videoCanvas.width !== frame.displayWidth || videoCanvas.height !== frame.displayHeight) {
                    videoCanvas.width = frame.displayWidth;
                    videoCanvas.height = frame.displayHeight;
                }
                videoCtx.drawImage(frame, 0, 0);
                return true;
            }
            function startVideoLoop() {
                if (videoRafId != null) return;
                const tick = () => {
                    videoRafId = requestAnimationFrame(tick);
                    if (!drawCurrentVideoFrame()) return;
                    if (videoCanvas.style.opacity !== "1") {
                        videoCanvas.style.opacity = "1";
                        imgA.style.opacity = "0";
                        imgB.style.opacity = "0";
                    }
                };
                videoRafId = requestAnimationFrame(tick);
            }
            function stopVideoLoop() {
                if (videoRafId != null) { cancelAnimationFrame(videoRafId); videoRafId = null; }
                videoCanvas.style.opacity = "0";
                visibleImg.style.opacity = "1";
            }
            // seeked → rVFC → double-rAF before hiding old video, to avoid the single-frame black gap.
            function showMp4(url) {
                if (url === currentMp4Url) return;
                currentMp4Url = url;
                if (playbackStartMs == null) playbackStartMs = performance.now();
                const target = pendingVideo;
                target.src = url;
                const promote = () => {
                    if (currentMp4Url !== url || target !== pendingVideo) return;
                    // Re-snap to NOW — initial seek drifted during load/paint. Paused-aware.
                    try {
                        const dur = target.duration;
                        if (Number.isFinite(dur) && dur > 0) {
                            target.currentTime = (elapsedMs() / 1000) % dur;
                        }
                    } catch {}
                    target.style.opacity = "1";
                    requestAnimationFrame(() => requestAnimationFrame(() => {
                        if (target !== visibleVideo && currentMp4Url === url) {
                            visibleVideo.style.opacity = "0";
                            visibleVideo.pause();
                            const prev = visibleVideo;
                            visibleVideo = target;
                            pendingVideo = prev;
                            imgA.style.opacity = "0";
                            imgB.style.opacity = "0";
                            videoCanvas.style.opacity = "0";
                            // Re-pause if user paused before this step swap.
                            if (isPaused) visibleVideo.pause();
                        }
                    }));
                };
                const afterSeek = () => {
                    if (typeof target.requestVideoFrameCallback === "function") {
                        target.requestVideoFrameCallback(() => promote());
                    } else {
                        promote();
                    }
                };
                const onLoaded = () => {
                    target.removeEventListener("loadeddata", onLoaded);
                    try {
                        const dur = target.duration;
                        if (Number.isFinite(dur) && dur > 0) {
                            // 'seeked' fires when the new currentTime is decoded and ready.
                            target.addEventListener("seeked", afterSeek, { once: true });
                            target.currentTime = (elapsedMs() / 1000) % dur;
                        } else {
                            afterSeek();
                        }
                        // Must play during load so rVFC fires; we re-pause after promote.
                        target.play().catch(() => {});
                    } catch { promote(); }
                };
                if (target.readyState >= 2) onLoaded();
                else target.addEventListener("loadeddata", onLoaded, { once: true });
            }
            function hideMp4() {
                visibleVideo.pause();
                visibleVideo.style.opacity = "0";
                pendingVideo.pause();
                pendingVideo.style.opacity = "0";
                visibleImg.style.opacity = "1";
                currentMp4Url = null;
            }
            function setStepBlob(stepIdx, blob) {
                const url = URL.createObjectURL(blob);
                if (placeholder.parentNode) placeholder.remove();
                if (blob.type === "video/mp4") {
                    if (stepMp4Urls[stepIdx]) {
                        try { URL.revokeObjectURL(stepMp4Urls[stepIdx]); } catch {}
                    }
                    stepMp4Urls[stepIdx] = url;
                    liveMp4Url = url;
                    if (hoverStep == null && lockedStep == null) showMp4(url);
                    return;
                }
                if (stepBlobUrls[stepIdx]) {
                    try { URL.revokeObjectURL(stepBlobUrls[stepIdx]); } catch {}
                }
                stepBlobUrls[stepIdx] = url;
                liveBlobUrl = url;
                if (blob.type === "image/webp") {
                    decodeAnimatedBlob(blob).then(v => {
                        if (!v) {
                            // Single-frame webp — show via the img path.
                            if (hoverStep == null && lockedStep == null && liveBlobUrl === url) {
                                showLiveFrame(url);
                            }
                            return;
                        }
                        closeStepVideo(stepIdx);
                        stepVideoFrames[stepIdx] = v;
                        startVideoLoop();
                    });
                } else {
                    if (hoverStep == null && lockedStep == null) {
                        showLiveFrame(url);
                    }
                }
            }
            function setStepAudio(stepIdx, blob) {
                const url = URL.createObjectURL(blob);
                if (stepAudioUrls[stepIdx]) {
                    try { URL.revokeObjectURL(stepAudioUrls[stepIdx]); } catch {}
                }
                stepAudioUrls[stepIdx] = url;
                liveAudioUrl = url;
                if (hoverStep == null && lockedStep == null) showAudio(url);
            }
            function resetFrames() {
                for (const u of stepBlobUrls) {
                    if (u) try { URL.revokeObjectURL(u); } catch {}
                }
                stepBlobUrls.length = 0;
                liveBlobUrl = null;
                for (let i = 0; i < stepVideoFrames.length; i++) closeStepVideo(i);
                stepVideoFrames.length = 0;
                stopVideoLoop();
                for (const u of stepMp4Urls) {
                    if (u) try { URL.revokeObjectURL(u); } catch {}
                }
                stepMp4Urls.length = 0;
                liveMp4Url = null;
                hideMp4();
                for (const u of stepAudioUrls) {
                    if (u) try { URL.revokeObjectURL(u); } catch {}
                }
                stepAudioUrls.length = 0;
                liveAudioUrl = null;
                stopAudio();
                playbackStartMs = null;
                bakedFps = null;
                isPaused = false;
                scrubBar.classList.remove("kj-pov-paused");
            }

            function updateSdHeader() {
                // Display priority: hover > locked > live. Hover gets [k] prefix, locked gets 🔒[k].
                let idx = null;
                let prefix = "";
                if (hoverStep != null && cachedSigmas && hoverStep < cachedSigmas.length) {
                    idx = hoverStep;
                    prefix = `[${idx}] `;
                } else if (lockedStep != null && cachedSigmas && lockedStep < cachedSigmas.length) {
                    idx = lockedStep;
                    prefix = `🔒[${idx}] `;
                } else if (lastCurrentStep >= 0 && cachedSigmas) {
                    idx = Math.min(lastCurrentStep, cachedSigmas.length - 1);
                }
                if (idx == null) return;
                const sig = cachedSigmas[idx];
                const dIdx = idx - 1;
                const d = (dIdx >= 0 && dIdx < history.delta.length) ? history.delta[dIdx] : null;
                let text = `${prefix}${fmt(sig, 3)} / ${fmt(d, 3)}`;
                if (cachedDbCurve && idx < cachedDbCurve.length) {
                    text += ` / ${fmt(cachedDbCurve[idx], 3)}`;
                }
                sdRow.valEl.textContent = text;
            }

            function redrawSd() {
                drawSigmaDeltaGraph(sdRow.canvas, cachedSigmas, history.delta, lastCurrentStep, totalSteps, hoverStep, cachedDbCurve, lockedStep);
                updateSdHeader();
                renderTime();
                // Scrub priority: locked > hover > live. Locked persists past mouseleave.
                const displayStep = lockedStep != null ? lockedStep : hoverStep;
                const targetIdx = displayStep != null ? displayStep : lastCurrentStep;
                if (stepMp4Urls[targetIdx]) {
                    stopVideoLoop();
                    showMp4(stepMp4Urls[targetIdx]);
                } else if (stepVideoFrames[targetIdx]) {
                    hideMp4();
                    startVideoLoop();
                } else if (displayStep != null && stepBlobUrls[displayStep]) {
                    stopVideoLoop();
                    hideMp4();
                    visibleImg.src = stepBlobUrls[displayStep];
                } else if (liveMp4Url) {
                    stopVideoLoop();
                    showMp4(liveMp4Url);
                } else if (liveBlobUrl) {
                    stopVideoLoop();
                    hideMp4();
                    showLiveFrame(liveBlobUrl);
                }
                // Audio follows the same locked > hover > live priority as the visual.
                if (displayStep != null && stepAudioUrls[displayStep]) {
                    showAudio(stepAudioUrls[displayStep]);
                } else if (liveAudioUrl) {
                    showAudio(liveAudioUrl);
                }
            }

            sdRow.canvas.addEventListener("mousemove", (ev) => {
                if (!cachedSigmas) return;
                const rect = sdRow.canvas.getBoundingClientRect();
                const iW = Math.max(1, rect.width - 2 * GRAPH_PAD_X);
                const xSteps = Math.max(totalSteps || cachedSigmas.length, cachedSigmas.length, history.delta.length);
                const fx = (ev.clientX - rect.left - GRAPH_PAD_X) / iW;
                const idx = Math.max(0, Math.min(xSteps - 1, Math.round(fx * (xSteps - 1))));
                if (idx !== hoverStep) {
                    hoverStep = idx;
                    redrawSd();
                }
            });
            sdRow.canvas.addEventListener("mouseleave", () => {
                if (hoverStep != null) {
                    hoverStep = null;
                    redrawSd();
                }
            });

            // Unmute audio only while hovering the preview frame or the σ/Δ step plot.
            imageArea.addEventListener("mouseenter", () => { audioHoverImage = true; refreshAudioMute(); });
            imageArea.addEventListener("mouseleave", () => { audioHoverImage = false; refreshAudioMute(); });
            sdRow.canvas.addEventListener("mouseenter", () => { audioHoverPlot = true; refreshAudioMute(); });
            sdRow.canvas.addEventListener("mouseleave", () => { audioHoverPlot = false; refreshAudioMute(); });

            // Click toggles a lock at the hover position so the preview survives mouseleave.
            sdRow.canvas.addEventListener("mousedown", (ev) => { ev.stopPropagation(); });
            sdRow.canvas.addEventListener("click", (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                if (lockedStep != null) {
                    lockedStep = null;
                } else if (cachedSigmas) {
                    const rect = sdRow.canvas.getBoundingClientRect();
                    const iW = Math.max(1, rect.width - 2 * GRAPH_PAD_X);
                    const xSteps = Math.max(totalSteps || cachedSigmas.length, cachedSigmas.length, history.delta.length);
                    const fx = (ev.clientX - rect.left - GRAPH_PAD_X) / iW;
                    lockedStep = Math.max(0, Math.min(xSteps - 1, Math.round(fx * (xSteps - 1))));
                }
                redrawSd();
            });

            // Arrow-key scrub: gated on mouseOverPanel so we don't shadow ComfyUI's global keys.
            let mouseOverPanel = false;
            root.addEventListener("mouseenter", () => { mouseOverPanel = true; });
            root.addEventListener("mouseleave", () => { mouseOverPanel = false; });
            const onKey = (ev) => {
                if (!mouseOverPanel) return;
                if (ev.key === " " && clipDurationMs() > 0) {
                    ev.preventDefault();
                    ev.stopPropagation();
                    togglePause();
                    return;
                }
                if (!cachedSigmas) return;
                if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
                const xSteps = Math.max(totalSteps || cachedSigmas.length, cachedSigmas.length, history.delta.length);
                const cur = lockedStep != null ? lockedStep : (hoverStep != null ? hoverStep : lastCurrentStep);
                const next = Math.max(0, Math.min(xSteps - 1, cur + (ev.key === "ArrowRight" ? 1 : -1)));
                ev.preventDefault();
                ev.stopPropagation();
                lockedStep = next;
                redrawSd();
            };
            document.addEventListener("keydown", onKey, true);
            chainCallback(node, "onRemoved", () => document.removeEventListener("keydown", onKey, true));

            addMiddleClickPan(root);
            addWheelPassthrough(root);

            node.addDOMWidget("preview", "kj_preview", root, { serialize: false });
            node.setSize([Math.max(node.size?.[0] ?? 360, 360), Math.max(node.size?.[1] ?? 480, 480)]);

            function resetHistory() {
                history.stepMs.length = 0;
                history.delta.length = 0;
                resetFrames();
                // Clear hover/lock — setStepBlob skips updates while either is set.
                hoverStep = null;
                lockedStep = null;
                // Allow the next run to re-pick its time unit from the first sample.
                lastStepMs = null;
            }

            const handler = (data) => {
                try {
                    if (Array.isArray(data.sigmas) && data.sigmas.length > 1) {
                        cachedSigmas = data.sigmas;
                        cachedDbCurve = Array.isArray(data.db_curve) ? data.db_curve : null;
                        resetHistory();
                        applySdLegend();
                    }
                    // Indexed by boundary; step 0 = initial noise (image optional).
                    if (typeof data.image === "string") {
                        const mime = typeof data.mime === "string" ? data.mime : "image/jpeg";
                        if (typeof data.fps === "number" && data.fps > 0) bakedFps = data.fps;
                        setStepBlob(data.step, b64ToBlob(data.image, mime));
                    }
                    // LTXAV audio for this step — rides the same playback clock as the visual.
                    if (typeof data.audio === "string") {
                        const amime = typeof data.audio_mime === "string" ? data.audio_mime : "audio/wav";
                        setStepAudio(data.step, b64ToBlob(data.audio, amime));
                    }

                    totalSteps = data.total || totalSteps;

                    if (data.step_ms != null) history.stepMs.push(data.step_ms);
                    if (data.delta != null) history.delta.push(data.delta);

                    // Auto-pick unit from the first sample so slow steps default to seconds.
                    if (lastStepMs == null && data.step_ms != null && data.step_ms > 1500) {
                        timeUnitSeconds = true;
                    }
                    lastStepMs = data.step_ms;
                    lastAvgStepMs = data.avg_step_ms;
                    lastStep = data.step;
                    lastTotal = data.total;
                    lastW = data.w;
                    lastH = data.h;
                    // data.step is 1-based; set BEFORE renderTime so stepTimeForDisplay sees it.
                    lastCurrentStep = data.step;
                    renderTime();

                    drawSigmaDeltaGraph(sdRow.canvas, cachedSigmas, history.delta, lastCurrentStep, totalSteps, hoverStep, cachedDbCurve);
                    updateSdHeader();
                    drawLineGraph(timeRow.canvas, history.stepMs, totalSteps);
                } catch (err) {
                    console.warn("[KJNodes.ModelPreviewOverride] preview decode failed:", err);
                }
            };
            node._kjPreviewHandler = handler;

            chainCallback(node, "onRemoved", function () {
                node._kjPreviewHandler = null;
                resetFrames();
                stopVideoLoop();
                if (scrubRafId != null) cancelAnimationFrame(scrubRafId);
            });
        });
    },
});
