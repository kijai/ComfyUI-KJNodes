import { chainCallback, addMiddleClickPan, addWheelPassthrough } from './utility.js';
const { app } = window.comfyAPI.app;

// Shared across all HDR Preview nodes so synced nodes can drive each other.
const hdrSyncGroup = new Set();

const VERTEX_SHADER = `#version 300 es
out vec2 v_texCoord;
void main() {
    vec2 verts[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    v_texCoord = verts[gl_VertexID] * 0.5 + 0.5;
    v_texCoord.y = 1.0 - v_texCoord.y;
    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}`;

// LogC3 constants ported from ComfyUI-LTXVideo/hdr.py
const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 v_texCoord;
out vec4 fragColor;
uniform sampler2D u_image;
uniform float u_exposure;
uniform float u_saturation;
uniform float u_linearScale;
uniform int u_space;  // 0 = logc3, 1 = linear

const float LC_A = 5.555556;
const float LC_B = 0.052272;
const float LC_C = 0.247190;
const float LC_D = 0.385537;
const float LC_E = 5.367655;
const float LC_F = 0.092809;
const float LC_CUT = 0.010591;
const float LC_CUT_LOG = LC_E * LC_CUT + LC_F;  // ~0.14966

vec3 logc3_decompress(vec3 logc) {
    logc = clamp(logc, 0.0, 1.0);
    vec3 lin_from_log = (pow(vec3(10.0), (logc - LC_D) / LC_C) - LC_B) / LC_A;
    vec3 lin_from_lin = (logc - LC_F) / LC_E;
    vec3 is_log = step(vec3(LC_CUT_LOG), logc);
    return mix(lin_from_lin, lin_from_log, is_log);
}

vec3 linear_to_srgb(vec3 x) {
    vec3 cutoff = vec3(0.0031308);
    vec3 low = 12.92 * x;
    vec3 high = 1.055 * pow(max(x, cutoff), vec3(1.0 / 2.4)) - 0.055;
    return clamp(mix(low, high, step(cutoff, x)), 0.0, 1.0);
}

vec3 srgb_to_linear(vec3 x) {
    vec3 cutoff = vec3(0.04045);
    vec3 low = x / 12.92;
    vec3 high = pow(max(x, vec3(0.0)) + 0.055, vec3(2.4)) / pow(vec3(1.055), vec3(2.4));
    return mix(low, high, step(cutoff, x));
}

void main() {
    vec3 col = texture(u_image, v_texCoord).rgb;
    vec3 hdr;
    if (u_space == 0) {
        hdr = logc3_decompress(col);
    } else if (u_space == 1) {
        hdr = col * u_linearScale;
    } else {
        hdr = srgb_to_linear(col);
    }
    hdr = max(hdr, vec3(0.0));
    vec3 exposed = hdr * exp2(u_exposure);
    float luma = dot(exposed, vec3(0.2126, 0.7152, 0.0722));
    vec3 saturated = max(mix(vec3(luma), exposed, u_saturation), vec3(0.0));
    vec3 tm = (u_space == 2) ? clamp(saturated, 0.0, 1.0) : saturated / (1.0 + saturated);
    fragColor = vec4(linear_to_srgb(tm), 1.0);
}`;

function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh) || "shader compile failed";
        gl.deleteShader(sh);
        throw new Error(log);
    }
    return sh;
}

function createProgram(gl, vsSrc, fsSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(prog) || "program link failed";
        gl.deleteProgram(prog);
        throw new Error(log);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
}

function loadImageBitmapFromView(filename, type) {
    return fetch(`/view?filename=${encodeURIComponent(filename)}&type=${type}`)
        .then(r => r.ok ? r.blob() : null)
        .then(blob => blob ? createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" }) : null)
        .catch(() => null);
}

app.registerExtension({
    name: "KJNodes.HDRPreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "HDRPreviewKJ") return;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            const node = this;

            const state = {
                frames: [],
                frameCount: 0,
                width: 512, height: 512,
                fps: 24,
                inputSpace: "logc3",
                linearScale: 1.0,
                exposure: 0.0,
                saturation: 1.0,
                currentFrame: 0,
                playing: false,
                playTimer: null,
                gl: null, program: null, texture: null, vao: null,
                uniforms: {},
                uploadedFrame: -1,
                needsRender: false,
                execGen: 0,
            };
            node._hdrState = state;

            let controlsHeight = 28;  // measured from DOM after mount; this is the fallback
            node._widgetHeight = controlsHeight;

            const container = document.createElement("div");
            container.style.cssText =
                "position:relative;width:100%;background:#111;display:flex;flex-direction:column;align-items:center;overflow:hidden;";

            const viewport = document.createElement("div");
            // flex-shrink:0 — legacy container can be shorter than this height; without it the viewport collapses to 0.
            viewport.style.cssText =
                "position:relative;width:100%;height:0;background:#000;overflow:hidden;flex-shrink:0;";

            const canvas = document.createElement("canvas");
            canvas.style.cssText =
                "display:block;width:100%;height:100%;";
            viewport.appendChild(canvas);

            const frameRow = document.createElement("div");
            frameRow.style.cssText =
                "display:flex;align-items:center;gap:6px;padding:3px 6px;color:#ccc;font-size:11px;background:#1a1a1a;width:100%;box-sizing:border-box;flex:0 0 auto;";

            const playBtn = document.createElement("button");
            playBtn.type = "button";
            playBtn.textContent = "▶";
            playBtn.style.cssText =
                "background:#333;color:#ccc;border:1px solid #555;cursor:pointer;padding:1px 8px;font-size:11px;min-width:24px;";

            const syncBtn = document.createElement("button");
            syncBtn.type = "button";
            syncBtn.textContent = "⛓";
            syncBtn.title = "Sync playback with other HDR Preview nodes that have sync enabled.";
            syncBtn.style.cssText =
                "background:#333;color:#888;border:1px solid #555;cursor:pointer;padding:1px 6px;font-size:11px;min-width:22px;";

            const frameSlider = document.createElement("input");
            frameSlider.type = "range";
            frameSlider.min = "0";
            frameSlider.max = "0";
            frameSlider.value = "0";
            frameSlider.step = "1";
            frameSlider.style.cssText = "flex:1;min-width:40px;accent-color:#5af;";

            const frameLabel = document.createElement("span");
            frameLabel.textContent = "0/0";
            frameLabel.style.cssText = "min-width:50px;text-align:right;font-variant-numeric:tabular-nums;";

            frameRow.appendChild(playBtn);
            frameRow.appendChild(syncBtn);
            frameRow.appendChild(frameSlider);
            frameRow.appendChild(frameLabel);

            container.appendChild(viewport);
            container.appendChild(frameRow);

            const stopProp = (e) => e.stopPropagation();
            for (const el of [frameSlider, playBtn, syncBtn]) {
                el.addEventListener("pointerdown", stopProp);
                el.addEventListener("mousedown", stopProp);
            }

            addMiddleClickPan(container);
            addWheelPassthrough(canvas);

            function initGL() {
                try {
                    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false, alpha: false });
                    if (!gl) {
                        console.error("[HDRPreviewKJ] WebGL2 not available");
                        return false;
                    }
                    const program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
                    gl.useProgram(program);

                    const uniforms = {
                        u_image: gl.getUniformLocation(program, "u_image"),
                        u_exposure: gl.getUniformLocation(program, "u_exposure"),
                        u_saturation: gl.getUniformLocation(program, "u_saturation"),
                        u_space: gl.getUniformLocation(program, "u_space"),
                        u_linearScale: gl.getUniformLocation(program, "u_linearScale"),
                    };
                    gl.uniform1i(uniforms.u_image, 0);

                    const texture = gl.createTexture();
                    gl.activeTexture(gl.TEXTURE0);
                    gl.bindTexture(gl.TEXTURE_2D, texture);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

                    const vao = gl.createVertexArray();
                    gl.bindVertexArray(vao);

                    state.gl = gl;
                    state.program = program;
                    state.texture = texture;
                    state.vao = vao;
                    state.uniforms = uniforms;
                    return true;
                } catch (err) {
                    console.error("[HDRPreviewKJ] WebGL init failed:", err);
                    return false;
                }
            }

            function requestRender() {
                if (state.needsRender) return;
                state.needsRender = true;
                requestAnimationFrame(() => {
                    state.needsRender = false;
                    render();
                });
            }

            function render() {
                if (!state.gl || !state.frames.length) return;
                const frame = state.frames[state.currentFrame];
                if (!frame) return;

                const gl = state.gl;
                // Backing buffer sized to source resolution for quality; CSS handles display scaling
                if (canvas.width !== state.width || canvas.height !== state.height) {
                    canvas.width = state.width;
                    canvas.height = state.height;
                }
                gl.viewport(0, 0, canvas.width, canvas.height);
                gl.useProgram(state.program);
                gl.bindVertexArray(state.vao);
                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, state.texture);

                if (state.uploadedFrame !== state.currentFrame) {
                    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
                    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
                    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, frame);
                    state.uploadedFrame = state.currentFrame;
                }

                gl.uniform1f(state.uniforms.u_exposure, state.exposure);
                gl.uniform1f(state.uniforms.u_saturation, state.saturation);
                const spaceIdx = state.inputSpace === "logc3" ? 0 : state.inputSpace === "srgb" ? 2 : 1;
                gl.uniform1i(state.uniforms.u_space, spaceIdx);
                gl.uniform1f(state.uniforms.u_linearScale, state.linearScale);

                gl.clearColor(0, 0, 0, 1);
                gl.clear(gl.COLOR_BUFFER_BIT);
                gl.drawArrays(gl.TRIANGLES, 0, 3);
            }

            function stopPlayback() {
                if (state.playTimer !== null) {
                    clearInterval(state.playTimer);
                    state.playTimer = null;
                }
                state.playing = false;
                playBtn.textContent = "▶";
            }

            function updateFrameLabel() {
                const total = Math.max(state.frameCount, 1);
                frameLabel.textContent = `${state.currentFrame + 1}/${total}`;
            }

            const exposureWidget = node.widgets?.find(w => w.name === "exposure");
            const saturationWidget = node.widgets?.find(w => w.name === "saturation");

            function hookLiveWidget(widget, stateKey) {
                if (!widget) return;
                state[stateKey] = isFinite(widget.value) ? widget.value : state[stateKey];
                const original = widget.callback;
                widget.callback = function (value) {
                    const num = isFinite(value) ? value : state[stateKey];
                    state[stateKey] = num;
                    requestRender();
                    return original?.apply(this, arguments);
                };
            }
            hookLiveWidget(exposureWidget, "exposure");
            hookLiveWidget(saturationWidget, "saturation");

            // Sync handle: other nodes in the group call these to follow along.
            let syncEnabled = false;
            const syncHandle = {
                setFrameFraction(fraction) {
                    if (state.frameCount <= 0) return;
                    const maxIdx = Math.max(state.frameCount - 1, 0);
                    const frame = Math.min(maxIdx, Math.max(0, Math.round(fraction * maxIdx)));
                    if (frame === state.currentFrame) return;
                    state.currentFrame = frame;
                    frameSlider.value = String(frame);
                    updateFrameLabel();
                    requestRender();
                },
                showPlaying(play) {
                    state.playing = play;
                    playBtn.textContent = play ? "■" : "▶";
                    if (!play && state.playTimer !== null) {
                        clearInterval(state.playTimer);
                        state.playTimer = null;
                    }
                },
            };

            function broadcastFraction() {
                if (!syncEnabled || state.frameCount <= 1) return;
                const fraction = state.currentFrame / (state.frameCount - 1);
                for (const other of hdrSyncGroup) {
                    if (other !== syncHandle) other.setFrameFraction(fraction);
                }
            }

            function broadcastPlaying(play) {
                if (!syncEnabled) return;
                for (const other of hdrSyncGroup) {
                    if (other !== syncHandle) other.showPlaying(play);
                }
            }

            syncBtn.addEventListener("click", () => {
                syncEnabled = !syncEnabled;
                syncBtn.style.color = syncEnabled ? "#5fa" : "#888";
                syncBtn.style.borderColor = syncEnabled ? "#5fa" : "#555";
                if (syncEnabled) {
                    hdrSyncGroup.add(syncHandle);
                } else {
                    hdrSyncGroup.delete(syncHandle);
                }
            });

            frameSlider.addEventListener("input", () => {
                state.currentFrame = parseInt(frameSlider.value, 10) || 0;
                updateFrameLabel();
                requestRender();
                broadcastFraction();
            });
            frameSlider.addEventListener("pointerdown", () => {
                stopPlayback();
                broadcastPlaying(false);
            });
            function persistCurrentFrame() {
                if (!node.properties?.hdrLastPreview) return;
                node.properties.hdrLastPreview.current_frame = state.currentFrame;
                node.graph?.change?.();
                try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.(); } catch {}
            }
            frameSlider.addEventListener("change", persistCurrentFrame);

            playBtn.addEventListener("click", () => {
                if (state.playing) {
                    stopPlayback();
                    broadcastPlaying(false);
                    persistCurrentFrame();
                    return;
                }
                if (state.frameCount <= 1) return;
                // Ensure no other synced node has an active timer — we become the sole driver.
                if (syncEnabled) {
                    for (const other of hdrSyncGroup) {
                        if (other !== syncHandle) other.showPlaying(false);
                    }
                }
                const intervalMs = Math.max(16, 1000 / Math.max(state.fps, 1));
                state.playing = true;
                playBtn.textContent = "■";
                state.playTimer = setInterval(() => {
                    state.currentFrame = (state.currentFrame + 1) % state.frameCount;
                    frameSlider.value = String(state.currentFrame);
                    updateFrameLabel();
                    requestRender();
                    broadcastFraction();
                }, intervalMs);
                broadcastPlaying(true);
            });

            const domWidget = node.addDOMWidget("hdr_preview", "hdr_preview", container, {
                serialize: false,
                hideOnZoom: false,
                margin: 0,
                getMinHeight: () => node._widgetHeight,
                getMaxHeight: () => node._widgetHeight,
                getHeight: () => node._widgetHeight,
            });
            node.resizable = true;

            // LiteGraph's computeSize uses a fixed per-widget height that ignores DOM widgets — sum manually.
            const NATIVE_ROW_H = (LiteGraph?.NODE_WIDGET_HEIGHT ?? 20) + 4;
            const TITLE_H = LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
            function computeNodeHeight() {
                let nativeCount = 0;
                for (const w of node.widgets || []) {
                    if (w === domWidget) continue;
                    if (w.hidden || w.type === "converted-widget") continue;
                    nativeCount++;
                }
                return TITLE_H + nativeCount * NATIVE_ROW_H + node._widgetHeight + 8;
            }

            // Don't gate on state.frames — a fire during async bitmap load would wipe _widgetHeight and collapse the node.
            const controlsObserver = new ResizeObserver(() => {
                const measured = frameRow.offsetHeight;
                if (measured <= 0 || measured === controlsHeight) return;
                controlsHeight = measured;
                const viewportPx = parseInt(viewport.style.height, 10) || 0;
                node._widgetHeight = viewportPx + controlsHeight;
                node.setSize([node.size[0], computeNodeHeight()]);
                node.graph?.setDirtyCanvas(true, true);
            });
            controlsObserver.observe(frameRow);

            let resizing = false;
            function resizeToFit() {
                if (resizing) return;
                resizing = true;
                try {
                    const srcW = state.width || 512, srcH = state.height || 512;
                    const availW = Math.max(100, node.size[0] - 30);
                    const ratio = srcH / srcW;
                    const displayH = Math.round(availW * ratio);
                    const totalH = displayH + controlsHeight;
                    viewport.style.width = availW + "px";
                    viewport.style.height = displayH + "px";
                    // min-height keeps children visible when ComfyUI sets the container height from a stale computeSize.
                    container.style.minHeight = totalH + "px";
                    node._widgetHeight = totalH;
                    node.setSize([node.size[0], computeNodeHeight()]);
                    node.graph?.setDirtyCanvas(true, true);
                } finally {
                    resizing = false;
                }
            }

            chainCallback(node, "onResize", function () {
                if (state.frames.length) resizeToFit();
            });

            async function applyPreviewData(data) {
                const gen = ++state.execGen;

                stopPlayback();

                for (const f of state.frames) {
                    try { f.close?.(); } catch {}
                }
                state.frames = [];
                state.uploadedFrame = -1;

                state.frameCount = data.frame_count || 0;
                state.width = data.width || 512;
                state.height = data.height || 512;
                state.fps = data.fps || 24;
                state.inputSpace = data.input_space || "logc3";
                state.linearScale = data.linear_scale || 1.0;
                const restoredFrame = Number.isInteger(data.current_frame) ? data.current_frame : 0;
                state.currentFrame = Math.min(Math.max(0, restoredFrame), Math.max(0, state.frameCount - 1));

                frameSlider.max = String(Math.max(0, state.frameCount - 1));
                frameSlider.value = String(state.currentFrame);
                updateFrameLabel();

                if (!state.gl) initGL();

                resizeToFit();

                const bitmaps = await Promise.all(
                    (data.frames || []).map(f => loadImageBitmapFromView(f.filename, f.type))
                );

                if (gen !== state.execGen) {
                    for (const b of bitmaps) {
                        try { b?.close?.(); } catch {}
                    }
                    return;
                }

                state.frames = bitmaps.filter(Boolean);
                // Re-assert size in case the ResizeObserver fired mid-await with empty frames.
                resizeToFit();
                requestRender();
            }

            chainCallback(node, "onExecuted", async function (message) {
                const data = message?.hdr_preview_data?.[0];
                if (!data) return;

                node.properties = node.properties || {};
                node.properties.hdrLastPreview = {
                    frames: data.frames,
                    width: data.width,
                    height: data.height,
                    fps: data.fps,
                    input_space: data.input_space,
                    linear_scale: data.linear_scale,
                    frame_count: data.frame_count,
                };
                // Autosave snapshots before execution, so post-execute property updates need an explicit dirty nudge.
                node.graph?.change?.();
                try {
                    const ct = app.extensionManager?.workflow?.activeWorkflow?.changeTracker;
                    ct?.checkState?.();
                } catch {}
                try { app.workflowManager?.activeWorkflow?.changeTracker?.checkState?.(); } catch {}

                await applyPreviewData(data);
            });

            chainCallback(node, "onConfigure", function () {
                if (exposureWidget && isFinite(exposureWidget.value)) state.exposure = exposureWidget.value;
                if (saturationWidget && isFinite(saturationWidget.value)) state.saturation = saturationWidget.value;
                const saved = node.properties?.hdrLastPreview;
                if (saved?.frames?.length) applyPreviewData(saved);
            });

            chainCallback(node, "onRemoved", function () {
                stopPlayback();
                controlsObserver.disconnect();
                hdrSyncGroup.delete(syncHandle);
                for (const f of state.frames) {
                    try { f.close?.(); } catch {}
                }
                state.frames = [];
                if (state.gl) {
                    try {
                        state.gl.deleteProgram(state.program);
                        state.gl.deleteTexture(state.texture);
                        state.gl.deleteVertexArray(state.vao);
                        const lose = state.gl.getExtension("WEBGL_lose_context");
                        if (lose) lose.loseContext();
                    } catch {}
                    state.gl = null;
                }
            });
        });
    },
});
