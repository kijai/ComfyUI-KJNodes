import { chainCallback, addMiddleClickPan, addWheelPassthrough } from './utility.js';
const { app } = window.comfyAPI.app;

// --- begin context window math (port of comfy/context_windows.py) ---
const MAX_WINDOWS = 512;
const TRUNC_MSG = `too many windows, truncated at ${MAX_WINDOWS}`;
function capReached(windows, errors) {
    if (windows.length < MAX_WINDOWS) return false;
    errors.push(TRUNC_MSG);
    return true;
}

function pythonRound(x) {
    // python round() is half-even
    const floor = Math.floor(x);
    const diff = x - floor;
    if (diff > 0.5) return floor + 1;
    if (diff < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
}

function orderedHalving(val) {
    const bin = BigInt(val).toString(2).padStart(64, '0').split('').reverse().join('');
    return Number(BigInt('0b' + bin)) / 18446744073709551616;
}

function range(start, stop) {
    const out = [];
    for (let i = start; i < stop; i++) out.push(i);
    return out;
}

// loop min/max; avoids Math.min(...arr) spread-limit crash on long windows
function arrMin(a) {
    let m = Infinity;
    for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i];
    return m;
}
function arrMax(a) {
    let m = -Infinity;
    for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
    return m;
}

function arraysEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

function doesWindowRollOver(window, numFrames) {
    let prev = -1;
    for (let i = 0; i < window.length; i++) {
        const val = ((window[i] % numFrames) + numFrames) % numFrames;
        if (val < prev) return [true, i];
        prev = val;
    }
    return [false, -1];
}

function shiftWindowToStart(window, numFrames) {
    const startVal = window[0];
    for (let i = 0; i < window.length; i++) {
        window[i] = ((window[i] - startVal) + numFrames) % numFrames;
    }
}

function shiftWindowToEnd(window, numFrames) {
    shiftWindowToStart(window, numFrames);
    const endDelta = numFrames - window[window.length - 1] - 1;
    for (let i = 0; i < window.length; i++) window[i] += endDelta;
}

// p: {length, overlap, stride, closedLoop, step}
function createWindows(numFrames, schedule, p, errors) {
    const L = p.length;
    if (schedule === 'standard_static') {
        if (numFrames <= L) return [range(0, numFrames)];
        const delta = L - p.overlap;
        if (delta <= 0) {
            errors.push(`context_overlap (${p.overlap}) must be smaller than context_length (${L}); the sampler would error`);
            return [];
        }
        const windows = [];
        for (let startIdx = 0; startIdx < numFrames; startIdx += delta) {
            const ending = startIdx + L;
            if (ending >= numFrames) {
                const finalStart = startIdx - (ending - numFrames);
                windows.push(range(finalStart, finalStart + L));
                break;
            }
            windows.push(range(startIdx, startIdx + L));
            if (capReached(windows, errors)) break;
        }
        return windows;
    }
    if (schedule === 'standard_static_balanced') {
        // standard_static count, placed evenly so seams are uniform (±1 frame)
        if (numFrames <= L) return [range(0, numFrames)];
        if (L - p.overlap <= 0) errors.push(`context_overlap (${p.overlap}) >= context_length (${L}); windows collapse to stride 1`);
        const strideTarget = Math.max(1, L - p.overlap);
        let N = Math.ceil((numFrames - L) / strideTarget) + 1; // stride <= L => no gaps
        if (N > MAX_WINDOWS) {
            errors.push(TRUNC_MSG);
            N = MAX_WINDOWS;
        }
        const windows = [];
        for (let i = 0; i < N; i++) {
            const start = Math.floor(i * (numFrames - L) / (N - 1) + 0.5);
            windows.push(range(start, start + L));
        }
        return windows;
    }
    if (schedule === 'batched') {
        if (numFrames <= L) return [range(0, numFrames)];
        const windows = [];
        for (let startIdx = 0; startIdx < numFrames; startIdx += L) {
            windows.push(range(startIdx, Math.min(startIdx + L, numFrames)));
            if (capReached(windows, errors)) break;
        }
        return windows;
    }
    if (schedule === 'batched_shifted') {
        // zero-overlap windows whose seams roll each step (~1.0x compute)
        if (numFrames <= L) return [range(0, numFrames)];
        const offset = pythonRound(numFrames * orderedHalving(p.step)) % L; // 0 at step 0 (plain batched)
        const windows = [];
        let start = 0;
        if (offset > 0) { windows.push(range(0, offset)); start = offset; }
        while (start < numFrames) {
            const end = Math.min(start + L, numFrames);
            windows.push(range(start, end));
            start = end;
            if (capReached(windows, errors)) break;
        }
        return windows;
    }
    if (schedule === 'looped_uniform' || schedule === 'standard_uniform') {
        const looped = schedule === 'looped_uniform';
        if (looped ? numFrames < L : numFrames <= L) return [range(0, numFrames)];
        const windows = [];
        const strideCount = Math.min(p.stride, Math.ceil(Math.log2(numFrames / L)) + 1);
        for (let sPow = 0; sPow < strideCount; sPow++) {
            const contextStep = 1 << sPow;
            const pad = pythonRound(numFrames * orderedHalving(p.step));
            const start = Math.floor(orderedHalving(p.step) * contextStep) + pad;
            const stop = numFrames + pad + (looped && p.closedLoop ? 0 : -p.overlap);
            const stepSize = L * contextStep - p.overlap;
            if (stepSize <= 0) {
                errors.push(`window advance is ${stepSize} (≤ 0) at stride ${contextStep}; the sampler would hang/error`);
                continue;
            }
            for (let j = start; j < stop; j += stepSize) {
                const w = [];
                for (let e = j; e < j + L * contextStep; e += contextStep) {
                    w.push(((e % numFrames) + numFrames) % numFrames);
                }
                windows.push(w);
                if (capReached(windows, errors)) return windows;
            }
        }
        if (!looped) {
            // shift roll-over windows to the end, fill gaps, drop duplicates
            const deleteIdxs = [];
            let winI = 0;
            while (winI < windows.length) {
                const [isRoll, rollIdx] = doesWindowRollOver(windows[winI], numFrames);
                if (isRoll) {
                    const rollVal = windows[winI][rollIdx];
                    shiftWindowToEnd(windows[winI], numFrames);
                    if (!windows[(winI + 1) % windows.length].includes(rollVal)) {
                        windows.splice(winI + 1, 0, range(rollVal, rollVal + L));
                    }
                }
                for (let preI = 0; preI < winI; preI++) {
                    if (arraysEqual(windows[winI], windows[preI])) {
                        deleteIdxs.push(winI);
                        break;
                    }
                }
                winI++;
            }
            deleteIdxs.reverse();
            for (const i of deleteIdxs) windows.splice(i, 1);
        }
        return windows;
    }
    return [range(0, numFrames)];
}

// extra overlap on the last static window from its shove-back; 0 = even tiling
function staticFinalDelta(F, L, overlap) {
    if (F <= L) return 0;
    const d = L - overlap;
    if (d <= 0) return 0; // invalid config, flagged elsewhere
    const mod = (F - L) % d;
    return mod === 0 ? 0 : d - mod;
}

function linspace(a, b, n) {
    if (n === 1) return [a];
    const out = [];
    for (let i = 0; i < n; i++) out.push(a + (b - a) * i / (n - 1));
    return out;
}

function weightsPyramid(length) {
    const seq = [];
    if (length % 2 === 0) {
        const m = length / 2;
        for (let i = 1; i <= m; i++) seq.push(i);
        for (let i = m; i >= 1; i--) seq.push(i);
    } else {
        const m = (length + 1) / 2;
        for (let i = 1; i < m; i++) seq.push(i);
        seq.push(m);
        for (let i = m - 1; i >= 1; i--) seq.push(i);
    }
    return seq;
}

function weightsOverlapLinear(length, fullLength, idxs, overlap, errors) {
    const w = new Array(length).fill(1);
    if (overlap > length && errors) {
        errors.push(`overlap-linear with context_overlap > context_length would error in the sampler`);
    }
    const ov = Math.min(overlap, length);
    if (ov > 0) {
        if (arrMin(idxs) > 0) {
            const ramp = linspace(1e-37, 1, ov);
            for (let i = 0; i < ov; i++) w[i] = ramp[i];
        }
        if (arrMax(idxs) < fullLength - 1) {
            const ramp = linspace(1, 1e-37, ov);
            for (let i = 0; i < ov; i++) w[length - ov + i] = ramp[i];
        }
    }
    return w;
}

function weightsRelative(idxList) {
    // per-index triangular bias (sequential weighted avg == bias/Σbias share)
    const first = idxList[0];
    const last = idxList[idxList.length - 1];
    return idxList.map((idx) => {
        const bias = 1 - Math.abs(idx - (first + last) / 2) / ((last - first + 1e-2) / 2);
        return Math.max(1e-2, bias);
    });
}

// smooth overlap-linear: flat interior, raised-cosine ramps that sum to 1
function weightsHann(length, fullLength, idxs, overlap, errors) {
    const w = new Array(length).fill(1);
    if (overlap > length && errors) {
        errors.push(`hann with context_overlap (${overlap}) > context_length (${length}); ramp clamped`);
    }
    const ov = Math.min(Math.max(overlap, 0), length);
    if (ov >= 1) {
        const denom = Math.max(ov - 1, 1);
        if (arrMin(idxs) > 0) {
            for (let j = 0; j < ov; j++) w[j] = Math.max(0.5 * (1 - Math.cos(Math.PI * j / denom)), 1e-37);
        }
        if (arrMax(idxs) < fullLength - 1) {
            for (let j = 0; j < ov; j++) w[length - ov + j] = Math.max(0.5 * (1 - Math.cos(Math.PI * (denom - j) / denom)), 1e-37);
        }
    }
    return w;
}

// smooth bell over the window (smooth twin of pyramid)
function weightsGaussian(length) {
    if (length <= 1) return new Array(length).fill(1);
    const c = (length - 1) / 2;
    const std = Math.max(length / 4, 1e-6);
    const out = [];
    for (let i = 0; i < length; i++) out.push(Math.exp(-0.5 * ((i - c) / std) ** 2));
    return out;
}

function computeViz(p) {
    const errors = [];
    const numFrames = Math.max(1, p.numFrames);
    const rawWindows = createWindows(numFrames, p.schedule, {
        length: Math.max(1, p.length),
        overlap: Math.max(0, p.overlap),
        stride: Math.max(1, p.stride),
        closedLoop: p.closedLoop,
        step: Math.max(0, p.step),
    }, errors);

    const windows = rawWindows.map((idxList) => {
        let anchor = -1;
        if (p.causalFix) {
            const a = idxList[0] - 1;
            if (a >= 0 && a < numFrames) anchor = a;
        }
        let weights;
        if (p.fuse === 'relative') weights = weightsRelative(idxList);
        else if (p.fuse === 'flat') weights = new Array(idxList.length).fill(1);
        else if (p.fuse === 'overlap-linear') weights = weightsOverlapLinear(idxList.length, numFrames, idxList, Math.max(0, p.overlap), errors);
        else if (p.fuse === 'hann') weights = weightsHann(idxList.length, numFrames, idxList, Math.max(0, p.overlap), errors);
        else if (p.fuse === 'gaussian') weights = weightsGaussian(idxList.length);
        else weights = weightsPyramid(idxList.length);
        const [rolls] = doesWindowRollOver(idxList, numFrames);
        return { idxList, anchor, weights, rolls };
    });

    // per-frame total weight (counts_final in the handler)
    const totals = new Float64Array(numFrames);
    for (const w of windows) {
        for (let pos = 0; pos < w.idxList.length; pos++) totals[w.idxList[pos]] += w.weights[pos];
    }
    const missing = [];
    for (let f = 0; f < numFrames; f++) if (totals[f] === 0) missing.push(f);
    let evals = 0;
    for (const w of windows) evals += w.idxList.length + (w.anchor >= 0 ? 1 : 0);

    return { numFrames, windows, totals, missing, evals, errors: [...new Set(errors)] };
}
// --- end context window math ---

const STYLE_ID = 'kj-cwviz-style';
function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement('style');
    s.id = STYLE_ID;
    s.textContent = `
.kj-cw-root { width:100%; height:100%; display:flex; flex-direction:column; background:#181818; overflow:hidden; user-select:none; border-radius:4px; }
.kj-cw-bar { display:flex; align-items:center; gap:6px; padding:4px 6px; background:#222; color:#ddd; font:12px sans-serif; flex:0 0 auto; }
.kj-cw-btn { background:#333; color:#ddd; border:1px solid #444; padding:1px 10px; cursor:pointer; font:12px sans-serif; border-radius:3px; }
.kj-cw-btn:hover { background:#444; }
.kj-cw-scrubber { position:relative; flex:1 1 auto; height:20px; background:#2a2a2a; border:1px solid #444; cursor:ew-resize; user-select:none; touch-action:none; border-radius:3px; overflow:hidden; }
.kj-cw-scrubber-fill { position:absolute; top:0; bottom:0; left:0; width:0; background:#3a5a7a; pointer-events:none; }
.kj-cw-scrubber-text { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#ddd; font:11px sans-serif; pointer-events:none; text-shadow:0 1px 2px rgba(0,0,0,0.6); }
.kj-cw-steps { width:42px; background:#2a2a2a; color:#ddd; border:1px solid #444; font:11px sans-serif; padding:2px 4px; border-radius:3px; }
.kj-cw-info { padding:2px 8px; color:#aaa; font:11px sans-serif; flex:0 0 auto; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kj-cw-info.error { color:#e06c5a; }
.kj-cw-canvas { flex:1 1 auto; min-height:0; width:100%; display:block; cursor:crosshair; }
`;
    document.head.appendChild(s);
}

const STEP_DEPENDENT = ['standard_uniform', 'looped_uniform', 'batched_shifted']; // schedules whose windows change per sampling step

const WINDOW_HUE_STEP = 137.508; // golden angle, distinct adjacent colors
function windowColor(i, sat = 65, light = 60, alpha = 1) {
    return `hsla(${(i * WINDOW_HUE_STEP) % 360}, ${sat}%, ${light}%, ${alpha})`;
}

function pickTickStep(numFrames, plotW) {
    const steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000];
    for (const s of steps) {
        if ((plotW / numFrames) * s >= 34) return s;
    }
    return steps[steps.length - 1];
}

app.registerExtension({
    name: 'KJNodes.ContextWindowsVisualizer',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== 'ContextWindowsVisualizerKJ') return;

        chainCallback(nodeType.prototype, 'onNodeCreated', function () {
            ensureStyles();
            const node = this;
            if (node.size[0] < 560 || node.size[1] < 480) node.setSize([Math.max(node.size[0], 560), Math.max(node.size[1], 480)]);

            const root = document.createElement('div');
            root.className = 'kj-cw-root';

            const bar = document.createElement('div');
            bar.className = 'kj-cw-bar';
            const playBtn = document.createElement('button');
            playBtn.className = 'kj-cw-btn';
            playBtn.textContent = '▶';
            playBtn.title = 'Animate sampling steps (uniform schedules shift their windows each step)';
            const scrubber = document.createElement('div');
            scrubber.className = 'kj-cw-scrubber';
            const scrubberFill = document.createElement('div');
            scrubberFill.className = 'kj-cw-scrubber-fill';
            const scrubberText = document.createElement('span');
            scrubberText.className = 'kj-cw-scrubber-text';
            scrubber.appendChild(scrubberFill);
            scrubber.appendChild(scrubberText);
            const stepsInput = document.createElement('input');
            stepsInput.className = 'kj-cw-steps';
            stepsInput.type = 'number';
            stepsInput.min = '1';
            stepsInput.max = '999';
            stepsInput.title = 'Total sampling steps to scrub through';
            bar.appendChild(playBtn);
            bar.appendChild(scrubber);
            bar.appendChild(stepsInput);

            const info = document.createElement('div');
            info.className = 'kj-cw-info';
            const canvas = document.createElement('canvas');
            canvas.className = 'kj-cw-canvas';

            root.appendChild(bar);
            root.appendChild(info);
            root.appendChild(canvas);
            node.addDOMWidget('context_windows_viz', 'div', root, { serialize: false });

            const removeMiddleClickPan = addMiddleClickPan(root);
            addWheelPassthrough(root);

            const state = {
                step: 0,
                steps: 20,
                viz: null,
                latent: null,     // converted params used for the viz
                hoverWin: -1,
                hoverFrame: -1,
                mouseX: 0,
                mouseY: 0,
                layout: null,     // hit-test geometry from last draw
            };

            const getWidgetValue = (name, fallback) => {
                const w = node.widgets?.find((w) => w.name === name);
                return w ? w.value : fallback;
            };

            const persist = () => {
                node.properties = node.properties || {};
                node.properties.cw_viz = { step: state.step, steps: state.steps };
            };

            const readParams = () => {
                const units = getWidgetValue('frame_units', 'pixel');
                const factor = Math.max(getWidgetValue('temporal_downscale', 4), 1);
                let numFrames = getWidgetValue('num_frames', 161);
                let length = getWidgetValue('context_length', 81);
                let overlap = getWidgetValue('context_overlap', 30);
                if (units === 'pixel') {
                    numFrames = Math.max(Math.floor((numFrames - 1) / factor) + 1, 1);
                    length = Math.max(Math.floor((length - 1) / factor) + 1, 1);
                    overlap = Math.max(Math.floor((overlap - 1) / factor) + 1, 0);
                }
                const schedule = getWidgetValue('context_schedule', 'standard_static');
                // flag the oversized final seam plain standard_static can leave
                const finalDelta = schedule === 'standard_static' ? staticFinalDelta(numFrames, length, overlap) : 0;
                return {
                    units,
                    factor,
                    numFrames,
                    length,
                    overlap,
                    finalDelta,
                    schedule,
                    stride: getWidgetValue('context_stride', 1),
                    closedLoop: !!getWidgetValue('closed_loop', false),
                    fuse: getWidgetValue('fuse_method', 'pyramid'),
                    causalFix: !!getWidgetValue('causal_window_fix', false),
                    step: state.step,
                };
            };

            const updateScrubberUI = () => {
                const t = state.steps > 1 ? state.step / (state.steps - 1) : 0;
                scrubberFill.style.width = (t * 100) + '%';
                const stepIndependent = state.latent && !STEP_DEPENDENT.includes(state.latent.schedule);
                scrubberText.textContent = `step ${state.step + 1} / ${state.steps}` + (stepIndependent ? ' (static schedule, no effect)' : '');
                stepsInput.value = String(state.steps);
            };

            const updateInfo = () => {
                if (!state.viz) return;
                const v = state.viz;
                if (v.errors.length) {
                    info.classList.add('error');
                    info.textContent = '⚠ ' + v.errors.join(' · ');
                    info.title = v.errors.join('\n');
                    return;
                }
                info.classList.remove('error');
                const lat = state.latent;
                const overhead = v.numFrames > 0 ? (v.evals / v.numFrames) : 0;
                const px = lat.units === 'pixel' ? ` (${(v.numFrames - 1) * lat.factor + 1}px frames)` : '';
                let txt = `${v.windows.length} windows over ${v.numFrames} latent frames${px} · ${v.evals} frame evals/step (${overhead.toFixed(2)}×)`;
                if (v.missing.length) txt += ` · ⚠ ${v.missing.length} uncovered frames`;
                if (lat.schedule === 'standard_static' && lat.finalDelta > 0) {
                    txt += ` · ⚠ last window overlaps +${lat.finalDelta} frames; standard_static_balanced spreads seams evenly`;
                } else if (lat.schedule === 'standard_static_balanced') {
                    txt += ` · balanced placement: even seams (±1 frame)`;
                } else if (lat.schedule === 'batched_shifted') {
                    const off = pythonRound(v.numFrames * orderedHalving(lat.step)) % lat.length;
                    txt += v.windows.length > 1 ? ` · zero-overlap rolling seams (step offset ${off})` : ` · zero-overlap`;
                }
                info.textContent = txt;
                info.title = txt;
            };

            let rafPending = false;
            const scheduleRender = () => {
                if (rafPending) return;
                rafPending = true;
                requestAnimationFrame(() => {
                    rafPending = false;
                    draw();
                });
            };

            const recompute = () => {
                state.latent = readParams();
                state.viz = computeViz(state.latent);
                state.hoverWin = -1;
                state.hoverFrame = -1;
                updateScrubberUI();
                updateInfo();
                scheduleRender();
            };

            const draw = () => {
                const viz = state.viz;
                if (!viz) return;
                const dpr = window.devicePixelRatio || 1;
                const cw = canvas.clientWidth;
                const ch = canvas.clientHeight;
                if (cw <= 0 || ch <= 0) return;
                const W = Math.floor(cw * dpr);
                const H = Math.floor(ch * dpr);
                if (canvas.width !== W) canvas.width = W;
                if (canvas.height !== H) canvas.height = H;
                const ctx = canvas.getContext('2d');
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                ctx.clearRect(0, 0, cw, ch);
                ctx.fillStyle = '#181818';
                ctx.fillRect(0, 0, cw, ch);

                const n = viz.numFrames;
                const pad = 8;
                const gutter = 30;
                const rulerH = 16;
                const chartH = Math.max(50, Math.min(110, ch * 0.28));
                const chartGap = 18;
                const plotX = pad + gutter;
                const plotW = cw - plotX - pad;
                const rowsY = pad + rulerH + 4;
                const chartY = ch - pad - chartH;
                const rowsH = chartY - chartGap - rowsY;
                if (plotW <= 0 || rowsH <= 10) return;
                const cellW = plotW / n;
                const barW = Math.max(cellW - (cellW > 3 ? 0.5 : 0), 0.5); // cell width minus a hairline gap
                const wins = viz.windows;
                const rowH = wins.length ? Math.min(20, Math.max(3, rowsH / wins.length)) : 0;
                state.layout = { plotX, plotW, rowsY, rowH, chartY, chartH, n };

                // ruler
                ctx.font = '10px sans-serif';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = '#777';
                ctx.strokeStyle = '#3a3a3a';
                const tick = pickTickStep(n, plotW);
                for (let f = 0; f < n; f += tick) {
                    const x = plotX + (f + 0.5) * cellW;
                    ctx.textAlign = 'center';
                    ctx.fillText(String(f), x, pad + rulerH / 2);
                    ctx.beginPath();
                    ctx.moveTo(x, rowsY);
                    ctx.lineTo(x, chartY + chartH);
                    ctx.stroke();
                }
                // uncovered frames
                if (viz.missing.length) {
                    ctx.fillStyle = 'rgba(220,70,60,0.25)';
                    for (const f of viz.missing) ctx.fillRect(plotX + f * cellW, rowsY, cellW, (chartY + chartH) - rowsY);
                }

                // window rows
                const hov = state.hoverWin;
                let drawnRows = wins.length;
                for (let i = 0; i < wins.length; i++) {
                    const w = wins[i];
                    const y = rowsY + i * rowH;
                    if (y + rowH > rowsY + rowsH + 1) { drawnRows = i; break; }
                    const dim = hov >= 0 && hov !== i;
                    const maxW = arrMax(w.weights);
                    // row label
                    if (rowH >= 9) {
                        ctx.fillStyle = dim ? '#555' : '#999';
                        ctx.textAlign = 'right';
                        ctx.fillText(String(i), plotX - 4, y + rowH / 2);
                    }
                    for (let pos = 0; pos < w.idxList.length; pos++) {
                        const f = w.idxList[pos];
                        const wt = maxW > 0 ? w.weights[pos] / maxW : 1;
                        const alpha = (0.25 + 0.7 * wt) * (dim ? 0.25 : 1);
                        // also encode weight as cell height when rows are tall
                        const hFrac = rowH >= 8 ? 0.35 + 0.65 * wt : 1;
                        const cellH = Math.max(1.5, (rowH - 1) * hFrac);
                        ctx.fillStyle = windowColor(i, 65, 60, alpha);
                        ctx.fillRect(plotX + f * cellW, y + (rowH - 1 - cellH), barW, cellH);
                    }
                    // causal anchor marker
                    if (w.anchor >= 0 && !dim) {
                        ctx.strokeStyle = windowColor(i, 65, 75, 0.9);
                        ctx.setLineDash([2, 2]);
                        ctx.strokeRect(plotX + w.anchor * cellW + 0.5, y + 0.5, Math.max(cellW - 1, 1), Math.max(rowH - 2, 1));
                        ctx.setLineDash([]);
                    }
                    if (hov === i) {
                        const lo = arrMin(w.idxList);
                        const hi = arrMax(w.idxList);
                        ctx.strokeStyle = '#fff';
                        ctx.lineWidth = 1;
                        ctx.strokeRect(plotX + lo * cellW, y, (hi - lo + 1) * cellW, rowH - 1);
                    }
                }

                state.layout.drawnRows = drawnRows;
                // blend chart: stacked normalized contribution per frame
                ctx.fillStyle = '#888';
                ctx.textAlign = 'left';
                ctx.fillText('blend (normalized contribution per frame)', plotX, chartY - 8);
                if (drawnRows < wins.length) {
                    ctx.textAlign = 'right';
                    ctx.fillText(`…+${wins.length - drawnRows} more windows (enlarge node)`, plotX + plotW, chartY - 8);
                }
                ctx.strokeStyle = '#3a3a3a';
                ctx.strokeRect(plotX, chartY, plotW, chartH);
                const stackOffsets = new Float64Array(n);
                for (let i = 0; i < wins.length; i++) {
                    const w = wins[i];
                    const dim = hov >= 0 && hov !== i;
                    ctx.fillStyle = windowColor(i, dim ? 25 : 65, dim ? 38 : 58, dim ? 0.6 : 0.92);
                    for (let pos = 0; pos < w.idxList.length; pos++) {
                        const f = w.idxList[pos];
                        const total = viz.totals[f];
                        if (total <= 0) continue;
                        const h = (w.weights[pos] / total) * chartH;
                        const yOff = (stackOffsets[f] / total) * chartH;
                        stackOffsets[f] += w.weights[pos];
                        ctx.fillRect(plotX + f * cellW, chartY + chartH - yOff - h, barW, h);
                    }
                }

                // hover guide + tooltip
                if (state.hoverFrame >= 0) {
                    const x = plotX + (state.hoverFrame + 0.5) * cellW;
                    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
                    ctx.beginPath();
                    ctx.moveTo(x, rowsY);
                    ctx.lineTo(x, chartY + chartH);
                    ctx.stroke();
                }
                const lines = [];
                if (hov >= 0 && hov < wins.length) {
                    const w = wins[hov];
                    const strided = w.idxList.length > 1 ? w.idxList[1] - w.idxList[0] : 1;
                    let l = `window ${hov + 1}/${wins.length}: ${w.idxList.length} frames [${w.idxList[0]}…${w.idxList[w.idxList.length - 1]}]`;
                    if (Math.abs(strided) > 1) l += ` stride ${strided}`;
                    if (w.rolls) l += ' (wraps)';
                    if (w.anchor >= 0) l += ` · anchor ${w.anchor}`;
                    lines.push(l);
                }
                if (state.hoverFrame >= 0) {
                    const f = state.hoverFrame;
                    const cover = [];
                    for (let i = 0; i < wins.length; i++) {
                        const pos = wins[i].idxList.indexOf(f);
                        if (pos >= 0) cover.push([i, wins[i].weights[pos]]);
                    }
                    const px = state.latent.units === 'pixel' ? ` (≈px ${f * state.latent.factor})` : '';
                    if (cover.length === 0) {
                        lines.push(`frame ${f}${px}: not covered by any window!`);
                    } else {
                        const total = viz.totals[f];
                        const shares = cover.map(([i, wt]) => `w${i + 1} ${(100 * wt / total).toFixed(0)}%`).join(', ');
                        lines.push(`frame ${f}${px}: ${shares}`);
                    }
                }
                if (lines.length) {
                    ctx.font = '11px sans-serif';
                    const tw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 12;
                    const th = lines.length * 15 + 8;
                    let tx = state.mouseX + 14;
                    let ty = state.mouseY + 14;
                    if (tx + tw > cw - 4) tx = state.mouseX - tw - 8;
                    if (ty + th > ch - 4) ty = state.mouseY - th - 8;
                    ctx.fillStyle = 'rgba(20,20,20,0.92)';
                    ctx.strokeStyle = '#555';
                    ctx.fillRect(tx, ty, tw, th);
                    ctx.strokeRect(tx, ty, tw, th);
                    ctx.fillStyle = '#ddd';
                    ctx.textAlign = 'left';
                    for (let i = 0; i < lines.length; i++) ctx.fillText(lines[i], tx + 6, ty + 12 + i * 15);
                }
            };

            // hover handling
            canvas.addEventListener('pointermove', (e) => {
                if (!state.viz || !state.layout) return;
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                state.mouseX = x;
                state.mouseY = y;
                const lay = state.layout;
                let frame = Math.floor((x - lay.plotX) / (lay.plotW / lay.n));
                if (x < lay.plotX || frame < 0 || frame >= lay.n) frame = -1;
                let win = -1;
                const visibleRows = Math.min(lay.drawnRows ?? state.viz.windows.length, state.viz.windows.length);
                if (y >= lay.rowsY && y < lay.rowsY + lay.rowH * visibleRows && lay.rowH > 0) {
                    win = Math.floor((y - lay.rowsY) / lay.rowH);
                    if (win >= visibleRows) win = -1;
                } else if (y >= lay.chartY && y <= lay.chartY + lay.chartH && frame >= 0) {
                    // pick the stacked band under the cursor
                    const total = state.viz.totals[frame];
                    if (total > 0) {
                        const fromBottom = (lay.chartY + lay.chartH - y) / lay.chartH * total;
                        let acc = 0;
                        for (let i = 0; i < state.viz.windows.length; i++) {
                            const w = state.viz.windows[i];
                            const pos = w.idxList.indexOf(frame);
                            if (pos < 0) continue;
                            acc += w.weights[pos];
                            if (fromBottom <= acc) { win = i; break; }
                        }
                    }
                }
                if (win !== state.hoverWin || frame !== state.hoverFrame) {
                    state.hoverWin = win;
                    state.hoverFrame = frame;
                }
                scheduleRender();
            });
            canvas.addEventListener('pointerleave', () => {
                state.hoverWin = -1;
                state.hoverFrame = -1;
                scheduleRender();
            });

            // step scrubber
            let scrubbing = false;
            const scrubFromEvent = (e) => {
                const rect = scrubber.getBoundingClientRect();
                const t = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                const step = Math.round(t * (state.steps - 1));
                if (step !== state.step) {
                    state.step = step;
                    persist();
                    recompute();
                }
            };
            scrubber.addEventListener('pointerdown', (e) => {
                if (e.button !== 0) return;
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

            stepsInput.addEventListener('change', () => {
                const v = parseInt(stepsInput.value, 10);
                if (Number.isFinite(v) && v >= 1 && v <= 999) {
                    state.steps = v;
                    if (state.step >= v) state.step = v - 1;
                    persist();
                    recompute();
                } else {
                    stepsInput.value = String(state.steps);
                }
            });
            stepsInput.addEventListener('pointerdown', (e) => e.stopPropagation());

            // step playback
            let playHandle = 0;
            const stopPlayback = () => {
                if (playHandle) {
                    clearInterval(playHandle);
                    playHandle = 0;
                    playBtn.textContent = '▶';
                }
            };
            playBtn.addEventListener('click', () => {
                if (playHandle) {
                    stopPlayback();
                    return;
                }
                playBtn.textContent = '⏸';
                playHandle = setInterval(() => {
                    state.step = (state.step + 1) % state.steps;
                    persist();
                    recompute();
                }, 300);
            });

            // live updates on widget changes
            const WATCHED = ['frame_units', 'temporal_downscale', 'num_frames', 'context_length', 'context_overlap',
                'context_schedule', 'context_stride', 'closed_loop', 'fuse_method', 'causal_window_fix'];
            for (const name of WATCHED) {
                const w = node.widgets?.find((w) => w.name === name);
                if (!w) continue;
                const orig = w.callback;
                w.callback = function (...args) {
                    const r = orig ? orig.apply(this, args) : undefined;
                    recompute();
                    return r;
                };
            }

            const ro = new ResizeObserver(() => scheduleRender());
            ro.observe(canvas);

            chainCallback(node, 'onConfigure', function () {
                const saved = this.properties?.cw_viz;
                if (saved) {
                    if (Number.isFinite(saved.steps)) state.steps = Math.min(999, Math.max(1, saved.steps));
                    if (Number.isFinite(saved.step)) state.step = Math.min(state.steps - 1, Math.max(0, saved.step));
                }
                recompute();
            });

            chainCallback(node, 'onRemoved', function () {
                stopPlayback();
                ro.disconnect();
                removeMiddleClickPan();
            });

            recompute();
        });
    },
});
