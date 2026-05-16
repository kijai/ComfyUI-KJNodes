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
            // Double-buffered: decode() must run on the SAME element that becomes visible —
            // otherwise the bitmap isn't reused and the JPEG decodes progressively (tearing).
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
            // MP4 path: <video> element; currentTime preserves scrub continuity across steps.
            const videoEl = el("video", "kj-pov-img", imageArea);
            videoEl.style.opacity = "0";
            videoEl.muted = true;
            videoEl.playsInline = true;
            videoEl.loop = true;
            videoEl.autoplay = true;
            videoEl.disablePictureInPicture = true;
            const placeholder = el("div", "kj-pov-placeholder", imageArea);
            placeholder.textContent = "waiting for sample…";

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
            const timeRow = makeCell("ms");
            sdRow.canvas.style.cursor = "crosshair";

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
            let liveBlobUrl = null;
            let liveMp4Url = null;
            // Global timer never resets between steps — scrub continues at the equivalent elapsed.
            let playbackStartMs = null;
            let videoRafId = null;
            let currentMp4Url = null;

            const SD_LEG_BASE = '<span style="color:#d0d0d0">σ</span> / <span style="color:#e67e22">Δ</span>';
            const SD_LEG_WITH_DB = SD_LEG_BASE + ' / <span style="color:rgb(120,200,220)">DB</span>';
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
            // Returns null for single-frame webp / no ImageDecoder. Baked durations are dropped;
            // playback rate comes live from preview_fps so the widget can retime mid-run.
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
            function drawCurrentVideoFrame() {
                const idx = activeStepIdx();
                const v = stepVideoFrames[idx];
                if (!v) return false;
                if (playbackStartMs == null) playbackStartMs = performance.now();
                const frameDurMs = 1000 / currentFps();
                const totalMs = v.frames.length * frameDurMs;
                const elapsed = (performance.now() - playbackStartMs) % totalMs;
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
            // Seek to global elapsed so scrubbing between MP4s doesn't restart from 0.
            function showMp4(url) {
                if (url === currentMp4Url) return;
                currentMp4Url = url;
                if (playbackStartMs == null) playbackStartMs = performance.now();
                videoEl.src = url;
                const seekToElapsed = () => {
                    try {
                        const dur = videoEl.duration;
                        if (Number.isFinite(dur) && dur > 0) {
                            const elapsedSec = (performance.now() - playbackStartMs) / 1000;
                            videoEl.currentTime = elapsedSec % dur;
                        }
                        videoEl.play().catch(() => {});
                    } catch {}
                };
                if (videoEl.readyState >= 1) seekToElapsed();
                else videoEl.addEventListener("loadedmetadata", seekToElapsed, { once: true });
                if (videoEl.style.opacity !== "1") {
                    videoEl.style.opacity = "1";
                    imgA.style.opacity = "0";
                    imgB.style.opacity = "0";
                    videoCanvas.style.opacity = "0";
                }
            }
            function hideMp4() {
                videoEl.pause();
                videoEl.style.opacity = "0";
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
                playbackStartMs = null;
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

            // Arrow-key scrub: only when cursor is over the preview panel so we don't shadow
            // ComfyUI's global keys. First press initializes the lock at the current step.
            let mouseOverPanel = false;
            root.addEventListener("mouseenter", () => { mouseOverPanel = true; });
            root.addEventListener("mouseleave", () => { mouseOverPanel = false; });
            const onKey = (ev) => {
                if (!mouseOverPanel || !cachedSigmas) return;
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
            }

            const handler = (data) => {
                try {
                    if (Array.isArray(data.sigmas) && data.sigmas.length > 1) {
                        cachedSigmas = data.sigmas;
                        cachedDbCurve = Array.isArray(data.db_curve) ? data.db_curve : null;
                        resetHistory();
                        applySdLegend();
                    }
                    // Indexed by boundary (step 0 = initial noise). Image is optional —
                    // the boundary-0 message may arrive without one if encoding failed.
                    if (typeof data.image === "string") {
                        const mime = typeof data.mime === "string" ? data.mime : "image/jpeg";
                        setStepBlob(data.step, b64ToBlob(data.image, mime));
                    }

                    totalSteps = data.total || totalSteps;

                    if (data.step_ms != null) history.stepMs.push(data.step_ms);
                    if (data.delta != null) history.delta.push(data.delta);

                    const etaTxt = data.avg_step_ms != null
                        ? `${(Math.max(0, data.total - data.step) * data.avg_step_ms / 1000).toFixed(1)}s`
                        : "—";
                    const stepMsTxt = data.avg_step_ms != null ? `${data.avg_step_ms.toFixed(0)}ms/step` : "—";
                    headerSummary.textContent = `${data.w}×${data.h} · ${data.step}/${data.total} · ${stepMsTxt} · ETA ${etaTxt}`;

                    // data.step is 1-based; marker sits at boundary k after step k completes.
                    lastCurrentStep = data.step;
                    timeRow.valEl.textContent = data.step_ms != null ? `${data.step_ms.toFixed(0)}ms` : "—";

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
            });
        });
    },
});
