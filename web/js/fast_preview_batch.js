import { chainCallback, addMiddleClickPan, addWheelPassthrough } from './utility.js';
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const STYLE_ID = 'kj-batch-preview-style';
function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement('style');
    s.id = STYLE_ID;
    s.textContent = `
.kj-bp-root { width:100%; height:100%; position:relative; background:#1a1a1a; overflow:hidden; user-select:none; }
.kj-bp-grid { width:100%; height:100%; display:block; cursor:pointer; image-rendering:auto; }
.kj-bp-detail { position:absolute; inset:0; display:none; background:#0d0d0d; }
.kj-bp-detail.visible { display:flex; flex-direction:column; }
.kj-bp-detail-canvas { flex:1 1 auto; min-height:0; width:100%; display:block; }
.kj-bp-bar { display:flex; align-items:center; justify-content:space-between; padding:4px 8px; background:#222; color:#ddd; font:12px sans-serif; }
.kj-bp-btn { background:#333; color:#ddd; border:1px solid #444; padding:2px 10px; cursor:pointer; font:12px sans-serif; }
.kj-bp-btn:hover { background:#444; }
.kj-bp-btn:disabled { opacity:0.4; cursor:default; }
.kj-bp-scrubber { position:relative; flex:1 1 auto; height:22px; margin:0 8px; background:#2a2a2a; border:1px solid #444; cursor:ew-resize; user-select:none; touch-action:none; }
.kj-bp-scrubber-fill { position:absolute; top:0; bottom:0; left:0; width:0; background:#3a5a7a; pointer-events:none; }
.kj-bp-scrubber-text { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#ddd; font:11px sans-serif; pointer-events:none; text-shadow:0 1px 2px rgba(0,0,0,0.6); }
.kj-bp-status { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#888; font:12px sans-serif; pointer-events:none; }
`;
    document.head.appendChild(s);
}

function pickGridLayout(count, aspect, containerW, containerH) {
    let best = { cols: 1, rows: count, cellW: 0, cellH: 0, area: 0 };
    for (let cols = 1; cols <= count; cols++) {
        const rows = Math.ceil(count / cols);
        const cellW = containerW / cols;
        const cellH = containerH / rows;
        const fitW = Math.min(cellW, cellH * aspect);
        const fitH = fitW / aspect;
        const area = fitW * fitH;
        if (area > best.area) best = { cols, rows, cellW, cellH, area, fitW, fitH };
    }
    return best;
}

app.registerExtension({
    name: 'KJNodes.FastPreviewBatch',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== 'FastPreviewBatch') return;

        chainCallback(nodeType.prototype, 'onNodeCreated', function () {
            ensureStyles();
            this.setSize([520, 520]);

            const root = document.createElement('div');
            root.className = 'kj-bp-root';

            const status = document.createElement('div');
            status.className = 'kj-bp-status';
            status.textContent = 'No preview yet';
            root.appendChild(status);

            const gridCanvas = document.createElement('canvas');
            gridCanvas.className = 'kj-bp-grid';
            gridCanvas.style.display = 'none';
            root.appendChild(gridCanvas);

            const detail = document.createElement('div');
            detail.className = 'kj-bp-detail';
            const detailCanvas = document.createElement('canvas');
            detailCanvas.className = 'kj-bp-detail-canvas';
            const bar = document.createElement('div');
            bar.className = 'kj-bp-bar';
            const prevBtn = document.createElement('button');
            prevBtn.className = 'kj-bp-btn';
            prevBtn.textContent = '❮';
            const nextBtn = document.createElement('button');
            nextBtn.className = 'kj-bp-btn';
            nextBtn.textContent = '❯';
            const scrubber = document.createElement('div');
            scrubber.className = 'kj-bp-scrubber';
            const scrubberFill = document.createElement('div');
            scrubberFill.className = 'kj-bp-scrubber-fill';
            const scrubberText = document.createElement('span');
            scrubberText.className = 'kj-bp-scrubber-text';
            scrubber.appendChild(scrubberFill);
            scrubber.appendChild(scrubberText);
            const playBtn = document.createElement('button');
            playBtn.className = 'kj-bp-btn';
            playBtn.textContent = '▶';
            const closeBtn = document.createElement('button');
            closeBtn.className = 'kj-bp-btn';
            closeBtn.textContent = '✕';
            bar.appendChild(playBtn);
            bar.appendChild(prevBtn);
            bar.appendChild(scrubber);
            bar.appendChild(nextBtn);
            bar.appendChild(closeBtn);
            detail.appendChild(detailCanvas);
            detail.appendChild(bar);
            root.appendChild(detail);

            this.addDOMWidget('fast_preview_batch', 'div', root, { serialize: false });

            const removeMiddleClickPan = addMiddleClickPan(root);
            addWheelPassthrough(root);

            const state = {
                video: null,
                strip: null,
                stripCols: 1,
                stripCellW: 0,
                stripCellH: 0,
                frameCount: 0,
                fps: 30,
                thumbW: 0,
                thumbH: 0,
                layout: null,
                gridReady: false,
                currentIndex: 0,
                hoverIndex: -1,
            };

            const setStatus = (msg) => {
                if (msg) {
                    status.textContent = msg;
                    status.style.display = 'flex';
                } else {
                    status.style.display = 'none';
                }
            };

            const cleanupVideo = () => {
                if (state.video) {
                    try { state.video.pause(); } catch (_) { }
                    state.video.removeAttribute('src');
                    try { state.video.load(); } catch (_) { }
                    if (state.video.parentNode) state.video.parentNode.removeChild(state.video);
                    state.video = null;
                }
            };

            const seekToFrame = (i) => new Promise((resolve) => {
                if (!state.video) return resolve();
                const t = (i + 0.5) / state.fps;
                const onSeeked = () => {
                    state.video.removeEventListener('seeked', onSeeked);
                    // give the decoder a tick to ensure frame is drawable
                    requestAnimationFrame(() => resolve());
                };
                state.video.addEventListener('seeked', onSeeked);
                try {
                    state.video.currentTime = t;
                } catch (_) {
                    state.video.removeEventListener('seeked', onSeeked);
                    resolve();
                }
            });

            let drawGeneration = 0;
            const updateScrubberUI = () => {
                const t = state.frameCount > 1 ? state.currentIndex / (state.frameCount - 1) : 0;
                scrubberFill.style.width = (t * 100) + '%';
                scrubberText.textContent = `frame ${state.currentIndex + 1} / ${state.frameCount}`;
                prevBtn.disabled = state.currentIndex <= 0;
                nextBtn.disabled = state.currentIndex >= state.frameCount - 1;
            };
            const paintFrameToDetail = () => {
                const v = state.video;
                if (!v) return;
                const w = detailCanvas.clientWidth;
                const h = detailCanvas.clientHeight;
                if (w <= 0 || h <= 0) return;
                const dpr = window.devicePixelRatio || 1;
                const targetW = Math.max(1, Math.floor(w * dpr));
                const targetH = Math.max(1, Math.floor(h * dpr));
                if (detailCanvas.width !== targetW) detailCanvas.width = targetW;
                if (detailCanvas.height !== targetH) detailCanvas.height = targetH;
                const ctx = detailCanvas.getContext('2d');
                ctx.fillStyle = '#0d0d0d';
                ctx.fillRect(0, 0, detailCanvas.width, detailCanvas.height);
                const aspect = state.thumbW / state.thumbH;
                let dw = detailCanvas.width;
                let dh = dw / aspect;
                if (dh > detailCanvas.height) {
                    dh = detailCanvas.height;
                    dw = dh * aspect;
                }
                const dx = (detailCanvas.width - dw) / 2;
                const dy = (detailCanvas.height - dh) / 2;
                ctx.drawImage(v, dx, dy, dw, dh);
            };
            const drawDetail = async () => {
                if (!state.video || !state.frameCount) return;
                updateScrubberUI();
                const gen = ++drawGeneration;
                await seekToFrame(state.currentIndex);
                if (gen !== drawGeneration) return; // superseded by a newer scrub
                paintFrameToDetail();
            };

            let playing = false;
            let playHandle = 0;
            const updatePlayBtn = () => { playBtn.textContent = playing ? '⏸' : '▶'; };
            const stopPlayback = () => {
                if (!playing) return;
                playing = false;
                playHandle++;
                if (state.video) { try { state.video.pause(); } catch (_) { } }
                updatePlayBtn();
            };
            const startPlayback = async () => {
                if (!state.video || !state.frameCount || playing) return;
                if (typeof state.video.requestVideoFrameCallback !== 'function') return;
                if (state.currentIndex >= state.frameCount - 1) state.currentIndex = 0;
                drawGeneration++; // invalidate any pending drawDetail
                await seekToFrame(state.currentIndex);
                paintFrameToDetail();
                updateScrubberUI();
                playing = true;
                updatePlayBtn();
                state.video.playbackRate = 1;
                const myHandle = ++playHandle;
                try { await state.video.play(); } catch (_) { stopPlayback(); return; }
                if (myHandle !== playHandle) return;
                const onFrame = (_now, metadata) => {
                    if (!playing || myHandle !== playHandle) return;
                    const idx = Math.min(state.frameCount - 1,
                        Math.floor(metadata.mediaTime * state.fps + 1e-6));
                    if (idx !== state.currentIndex) {
                        state.currentIndex = idx;
                        updateScrubberUI();
                    }
                    paintFrameToDetail();
                    if (idx >= state.frameCount - 1) { stopPlayback(); return; }
                    state.video.requestVideoFrameCallback(onFrame);
                };
                state.video.requestVideoFrameCallback(onFrame);
            };

            const renderGrid = () => {
                if (!state.frameCount || !state.strip) return;
                const containerW = root.clientWidth;
                const containerH = root.clientHeight;
                if (containerW <= 0 || containerH <= 0) return;
                const dpr = window.devicePixelRatio || 1;
                const aspect = state.thumbW / state.thumbH;
                const layout = pickGridLayout(state.frameCount, aspect, containerW, containerH);
                state.layout = layout;
                gridCanvas.width = Math.floor(containerW * dpr);
                gridCanvas.height = Math.floor(containerH * dpr);
                gridCanvas.style.width = containerW + 'px';
                gridCanvas.style.height = containerH + 'px';
                const ctx = gridCanvas.getContext('2d');
                ctx.fillStyle = '#1a1a1a';
                ctx.fillRect(0, 0, gridCanvas.width, gridCanvas.height);

                const cellW = layout.cellW * dpr;
                const cellH = layout.cellH * dpr;
                const fitW = layout.fitW * dpr;
                const fitH = layout.fitH * dpr;
                const total = state.frameCount;
                const sw = state.stripCellW;
                const sh = state.stripCellH;
                const sCols = state.stripCols;
                for (let i = 0; i < total; i++) {
                    const sx = (i % sCols) * sw;
                    const sy = Math.floor(i / sCols) * sh;
                    const dCol = i % layout.cols;
                    const dRow = Math.floor(i / layout.cols);
                    const dx = dCol * cellW + (cellW - fitW) / 2;
                    const dy = dRow * cellH + (cellH - fitH) / 2;
                    ctx.drawImage(state.strip, sx, sy, sw, sh, dx, dy, fitW, fitH);
                }
                if (state.hoverIndex >= 0 && state.hoverIndex < total) {
                    const i = state.hoverIndex;
                    const sx = (i % sCols) * sw;
                    const sy = Math.floor(i / sCols) * sh;
                    const dCol = i % layout.cols;
                    const dRow = Math.floor(i / layout.cols);
                    const cx = dCol * cellW + cellW / 2;
                    const cy = dRow * cellH + cellH / 2;
                    const scale = 1.15;
                    const zw = fitW * scale;
                    const zh = fitH * scale;
                    let dx = cx - zw / 2;
                    let dy = cy - zh / 2;
                    dx = Math.max(0, Math.min(gridCanvas.width - zw, dx));
                    dy = Math.max(0, Math.min(gridCanvas.height - zh, dy));
                    ctx.save();
                    ctx.shadowColor = 'rgba(0,0,0,0.65)';
                    ctx.shadowBlur = 14 * dpr;
                    ctx.drawImage(state.strip, sx, sy, sw, sh, dx, dy, zw, zh);
                    ctx.restore();
                    ctx.strokeStyle = '#5a8ec4';
                    ctx.lineWidth = Math.max(2, 2 * dpr);
                    ctx.strokeRect(dx + ctx.lineWidth / 2, dy + ctx.lineWidth / 2, zw - ctx.lineWidth, zh - ctx.lineWidth);
                }
                state.gridReady = true;
                setStatus('');
                gridCanvas.style.display = 'block';
            };

            const enterDetail = (idx) => {
                state.currentIndex = idx;
                detail.classList.add('visible');
                requestAnimationFrame(drawDetail);
            };
            const exitDetail = () => {
                stopPlayback();
                detail.classList.remove('visible');
            };

            playBtn.addEventListener('click', () => {
                if (playing) stopPlayback();
                else startPlayback();
            });

            const cellIndexFromEvent = (e) => {
                if (!state.gridReady || !state.layout) return -1;
                const rect = gridCanvas.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) return -1;
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const cellW = rect.width / state.layout.cols;
                const cellH = rect.height / state.layout.rows;
                const col = Math.floor(x / cellW);
                const row = Math.floor(y / cellH);
                if (col < 0 || col >= state.layout.cols || row < 0 || row >= state.layout.rows) return -1;
                const idx = row * state.layout.cols + col;
                return (idx >= 0 && idx < state.frameCount) ? idx : -1;
            };

            gridCanvas.addEventListener('click', (e) => {
                const idx = cellIndexFromEvent(e);
                if (idx >= 0) enterDetail(idx);
            });

            gridCanvas.addEventListener('pointermove', (e) => {
                const idx = cellIndexFromEvent(e);
                if (idx !== state.hoverIndex) {
                    state.hoverIndex = idx;
                    renderGrid();
                }
            });
            gridCanvas.addEventListener('pointerleave', () => {
                if (state.hoverIndex !== -1) {
                    state.hoverIndex = -1;
                    renderGrid();
                }
            });

            prevBtn.addEventListener('click', () => {
                if (state.currentIndex > 0) {
                    stopPlayback();
                    state.currentIndex--;
                    drawDetail();
                }
            });
            nextBtn.addEventListener('click', () => {
                if (state.currentIndex < state.frameCount - 1) {
                    stopPlayback();
                    state.currentIndex++;
                    drawDetail();
                }
            });
            closeBtn.addEventListener('click', exitDetail);

            let scrubbing = false;
            const scrubFromEvent = (e) => {
                if (!state.frameCount) return;
                const rect = scrubber.getBoundingClientRect();
                const t = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                const idx = Math.round(t * (state.frameCount - 1));
                if (idx !== state.currentIndex) {
                    state.currentIndex = idx;
                    drawDetail();
                }
            };
            scrubber.addEventListener('pointerdown', (e) => {
                if (e.button !== 0) return; // let middle/right click bubble (canvas pan, etc.)
                stopPlayback();
                scrubbing = true;
                scrubber.setPointerCapture(e.pointerId);
                scrubFromEvent(e);
                e.preventDefault();
            });
            scrubber.addEventListener('pointermove', (e) => {
                if (scrubbing) scrubFromEvent(e);
            });
            const endScrub = (e) => {
                if (!scrubbing) return;
                scrubbing = false;
                try { scrubber.releasePointerCapture(e.pointerId); } catch (_) { }
            };
            scrubber.addEventListener('pointerup', endScrub);
            scrubber.addEventListener('pointercancel', endScrub);

            const keyHandler = (e) => {
                if (!detail.classList.contains('visible')) return;
                if (e.key === 'Escape') { exitDetail(); e.stopPropagation(); }
                else if (e.key === ' ') {
                    if (playing) stopPlayback(); else startPlayback();
                    e.preventDefault(); e.stopPropagation();
                }
                else if (e.key === 'ArrowLeft' && state.currentIndex > 0) {
                    stopPlayback(); state.currentIndex--; drawDetail(); e.stopPropagation();
                } else if (e.key === 'ArrowRight' && state.currentIndex < state.frameCount - 1) {
                    stopPlayback(); state.currentIndex++; drawDetail(); e.stopPropagation();
                }
            };
            window.addEventListener('keydown', keyHandler);

            const ro = new ResizeObserver(() => {
                if (detail.classList.contains('visible')) {
                    drawDetail();
                } else if (state.strip) {
                    renderGrid();
                }
            });
            ro.observe(root);

            const buildViewUrl = (filename, subfolder, type) => {
                const params = new URLSearchParams({
                    filename: filename,
                    subfolder: subfolder || '',
                    type: type || 'temp',
                    t: String(Date.now()),
                });
                return api.apiURL(`/view?${params.toString()}`);
            };

            const loadFromOutput = async (info) => {
                stopPlayback();
                cleanupVideo();
                state.strip = null;
                state.gridReady = false;
                gridCanvas.style.display = 'none';
                setStatus('Loading preview…');

                state.frameCount = info.frame_count;
                state.fps = info.fps || 30;
                state.thumbW = info.thumb_w;
                state.thumbH = info.thumb_h;
                state.stripCols = info.strip_cols || 1;
                state.stripCellW = info.strip_cell_w || info.thumb_w;
                state.stripCellH = info.strip_cell_h || info.thumb_h;
                if (state.currentIndex >= state.frameCount || state.currentIndex < 0) {
                    state.currentIndex = 0;
                }

                const v = document.createElement('video');
                v.muted = true;
                v.playsInline = true;
                v.preload = 'auto';
                v.style.position = 'absolute';
                v.style.left = '-99999px';
                v.style.width = '1px';
                v.style.height = '1px';
                document.body.appendChild(v);
                state.video = v;

                // Video loads in parallel for the detail view; failure is non-fatal
                // for the grid (strip handles that). Caught to avoid unhandled-rejection.
                new Promise((resolve, reject) => {
                    v.addEventListener('loadeddata', resolve, { once: true });
                    v.addEventListener('error', () => {
                        cleanupVideo();
                        reject(new Error('video load failed'));
                    }, { once: true });
                    v.src = buildViewUrl(info.filename, info.subfolder, info.type);
                }).catch(() => { });

                await new Promise((resolve, reject) => {
                    const img = new Image();
                    img.onload = () => { state.strip = img; resolve(); };
                    img.onerror = () => reject(new Error('strip load failed'));
                    img.src = buildViewUrl(info.strip_filename, info.subfolder, info.type);
                });
                renderGrid();
                if (detail.classList.contains('visible')) {
                    // Refresh detail canvas with the new video; drawDetail awaits seek,
                    // which itself queues until the new video element finishes loading.
                    drawDetail();
                }
            };

            const tryLoad = (info, onMissing) => {
                loadFromOutput(info).catch((err) => {
                    console.warn('FastPreviewBatch load failed', err);
                    if (onMissing) onMissing();
                    else setStatus('Preview failed: ' + (err && err.message || err));
                });
            };

            chainCallback(this, 'onExecuted', function (output) {
                if (!output || !output.kj_batch_preview) return;
                const info = output.kj_batch_preview[0];
                if (!info) return;
                this.properties = this.properties || {};
                this.properties.kj_batch_preview = info;
                tryLoad(info);
            });

            chainCallback(this, 'onConfigure', function () {
                const info = this.properties && this.properties.kj_batch_preview;
                if (!info) return;
                tryLoad(info, () => {
                    setStatus('Preview unavailable (re-run to regenerate)');
                    delete this.properties.kj_batch_preview;
                });
            });

            chainCallback(this, 'onRemoved', function () {
                window.removeEventListener('keydown', keyHandler);
                ro.disconnect();
                removeMiddleClickPan();
                cleanupVideo();
            });
        });
    },
});
