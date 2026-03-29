import { addMiddleClickPan, addWheelPassthrough } from "./utility.js";
const { app } = window.comfyAPI.app;

app.registerExtension({
	name: "KJNodes.ScreencapStream",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "ScreencapStream") return;

		nodeType.prototype.onNodeCreated = function () {
			const node = this;
			const frameWidget = this.widgets.find(w => w.name === "frame_data");
			frameWidget.computeSize = () => [0, -4];

			// --- State ---
			let stream = null;
			let crop = null;               // { x, y, w, h } in video pixel coords, or null
			let previewVisible = true;
			let previewInterval = null;
			let bitmapInterval = null;
			let cachedBitmap = null;        // downscaled ImageBitmap for preview
			let bitmapPending = false;
			let drag = null;
			let drawPending = false;
			let cachedLayout = null;
			let layoutDirty = true;
			let updatingFromCrop = false;
			let captureStartTime = null;
			let timerInterval = null;

			const EDGE = 8;
			// separate canvases to avoid conflicts between preview and serialization
			const serializeCanvas = document.createElement("canvas");
			const serializeCtx = serializeCanvas.getContext("2d");

			// --- DOM elements ---
			const video = document.createElement("video");
			video.autoplay = true;
			video.muted = true;
			video.style.cssText = "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;overflow:hidden;";

			const container = document.createElement("div");
			container.style.cssText = "width:100%;height:100%;background:#000;border-radius:4px;overflow:hidden;position:relative;";
			container.appendChild(video);

			const overlay = document.createElement("canvas");
			overlay.style.cssText = "position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair;";
			container.appendChild(overlay);
			const overlayCtx = overlay.getContext("2d");

			addMiddleClickPan(container);
			addWheelPassthrough(container);

			// --- Widget references ---
			const cropWWidget = node.widgets.find(w => w.name === "crop_width");
			const cropHWidget = node.widgets.find(w => w.name === "crop_height");
			const PREVIEW_MAX_RES = 960; // preview-only cap for performance

			// --- Layout ---
			function invalidateLayout() { layoutDirty = true; }

			function getLayout() {
				if (!layoutDirty && cachedLayout) return cachedLayout;
				const oRect = overlay.getBoundingClientRect();
				const vw = video.videoWidth, vh = video.videoHeight;
				if (!vw || !vh || !oRect.width || !oRect.height) { cachedLayout = null; return null; }
				const scale = Math.min(oRect.width / vw, oRect.height / vh);
				cachedLayout = {
					scale, vw, vh,
					offX: (oRect.width - vw * scale) / 2,
					offY: (oRect.height - vh * scale) / 2,
				};
				layoutDirty = false;
				return cachedLayout;
			}

			function toVideoCoords(cx, cy) {
				const L = getLayout();
				if (!L) return { x: 0, y: 0 };
				return {
					x: Math.round(Math.max(0, Math.min(L.vw, (cx - L.offX) / L.scale))),
					y: Math.round(Math.max(0, Math.min(L.vh, (cy - L.offY) / L.scale))),
				};
			}

			function pointerToLocal(e) {
				const r = overlay.getBoundingClientRect();
				return { x: e.clientX - r.left, y: e.clientY - r.top };
			}

			// --- Output dimensions ---
			// Single source of truth for what the output size will be.
			function getOutputDimensions() {
				const vw = video.videoWidth, vh = video.videoHeight;
				if (!vw || !vh) return null;
				const sw = crop ? crop.w : vw;
				const sh = crop ? crop.h : vh;
				return { w: sw, h: sh };
			}

			function updateSizeWidgets() {
				updatingFromCrop = true;
				const dims = getOutputDimensions();
				if (dims) {
					cropWWidget.value = dims.w;
					cropHWidget.value = dims.h;
					// clamp max to full capture dimensions
					if (cropWWidget.options) cropWWidget.options.max = video.videoWidth || dims.w;
					if (cropHWidget.options) cropHWidget.options.max = video.videoHeight || dims.h;
				}
				app.graph.setDirtyCanvas(true);
				updatingFromCrop = false;
			}

			function updateCropFromWidgets() {
				if (updatingFromCrop || !video.videoWidth || !crop) return;
				const newW = Math.min(cropWWidget.value, video.videoWidth);
				const newH = Math.min(cropHWidget.value, video.videoHeight);
				if (newW <= 0 || newH <= 0) return;
				const cx = crop.x + crop.w / 2;
				const cy = crop.y + crop.h / 2;
				let x = Math.round(cx - newW / 2);
				let y = Math.round(cy - newH / 2);
				x = Math.max(0, Math.min(x, video.videoWidth - newW));
				y = Math.max(0, Math.min(y, video.videoHeight - newH));
				crop = { x, y, w: newW, h: newH };
				drawOverlay();
			}

			function clearCrop() {
				if (video.videoWidth && video.videoHeight) {
					crop = { x: 0, y: 0, w: video.videoWidth, h: video.videoHeight };
				} else {
					crop = null;
				}
				drawOverlay();
				updateSizeWidgets();
			}

			// --- Cached bitmap ---
			// Updated every 500ms for preview display.
			// Downscaled to PREVIEW_MAX_RES for performance.
			async function updateCachedBitmap() {
				if (bitmapPending || !stream || !video.videoWidth) return;
				bitmapPending = true;
				try {
					const vw = video.videoWidth, vh = video.videoHeight;
					if (vw <= 0 || vh <= 0) { bitmapPending = false; return; }
					// always full frame for preview — crop is drawn as overlay
					let outW = vw, outH = vh;
					if (Math.max(vw, vh) > PREVIEW_MAX_RES) {
						const s = PREVIEW_MAX_RES / Math.max(vw, vh);
						outW = Math.round(vw * s);
						outH = Math.round(vh * s);
					}
					const bmp = await createImageBitmap(video, {
						resizeWidth: outW,
						resizeHeight: outH,
						resizeQuality: "medium",
					});
					if (cachedBitmap) cachedBitmap.close();
					cachedBitmap = bmp;
				} catch { /* frame not ready */ }
				bitmapPending = false;
			}

			function startBitmapLoop() {
				stopBitmapLoop();
				bitmapInterval = setInterval(updateCachedBitmap, 500);
			}
			function stopBitmapLoop() {
				if (bitmapInterval) { clearInterval(bitmapInterval); bitmapInterval = null; }
			}

			// --- Preview drawing ---
			function drawOverlay() {
				if (drawPending) return;
				drawPending = true;
				requestAnimationFrame(() => {
					drawPending = false;
					renderOverlay();
				});
			}

			function renderOverlay() {
				const r = overlay.getBoundingClientRect();
				const w = Math.round(r.width), h = Math.round(r.height);
				if (overlay.width !== w || overlay.height !== h) {
					overlay.width = w; overlay.height = h;
					invalidateLayout();
				}
				overlayCtx.clearRect(0, 0, w, h);
				if (!previewVisible) return;
				const L = getLayout();
				if (!L) return;

				// draw from cached bitmap if available, else raw video
				const src = cachedBitmap || video;
				overlayCtx.drawImage(src, L.offX, L.offY, L.vw * L.scale, L.vh * L.scale);

				if (!crop) return;

				const rx = L.offX + crop.x * L.scale;
				const ry = L.offY + crop.y * L.scale;
				const rw = crop.w * L.scale;
				const rh = crop.h * L.scale;

				// darken outside crop
				overlayCtx.fillStyle = "rgba(0,0,0,0.5)";
				overlayCtx.fillRect(0, 0, w, ry);
				overlayCtx.fillRect(0, ry + rh, w, h - ry - rh);
				overlayCtx.fillRect(0, ry, rx, rh);
				overlayCtx.fillRect(rx + rw, ry, w - rx - rw, rh);

				overlayCtx.strokeStyle = "#fff";
				overlayCtx.lineWidth = 1.5;
				overlayCtx.setLineDash([4, 3]);
				overlayCtx.strokeRect(rx, ry, rw, rh);
				overlayCtx.setLineDash([]);
			}

			function startPreviewLoop() {
				stopPreviewLoop();
				previewInterval = setInterval(() => drawOverlay(), 100);
			}
			function stopPreviewLoop() {
				if (previewInterval) { clearInterval(previewInterval); previewInterval = null; }
			}

			// --- Hit testing ---
			function hitTest(lx, ly) {
				if (!crop) return "draw";
				const L = getLayout();
				if (!L) return "draw";
				const rx = L.offX + crop.x * L.scale;
				const ry = L.offY + crop.y * L.scale;
				const rw = crop.w * L.scale;
				const rh = crop.h * L.scale;

				const nearL = Math.abs(lx - rx) < EDGE;
				const nearR = Math.abs(lx - (rx + rw)) < EDGE;
				const nearT = Math.abs(ly - ry) < EDGE;
				const nearB = Math.abs(ly - (ry + rh)) < EDGE;
				const inX = lx > rx - EDGE && lx < rx + rw + EDGE;
				const inY = ly > ry - EDGE && ly < ry + rh + EDGE;

				if (nearT && nearL && inX && inY) return "nw";
				if (nearT && nearR && inX && inY) return "ne";
				if (nearB && nearL && inX && inY) return "sw";
				if (nearB && nearR && inX && inY) return "se";
				if (nearT && inX) return "n";
				if (nearB && inX) return "s";
				if (nearL && inY) return "w";
				if (nearR && inY) return "e";
				if (lx > rx && lx < rx + rw && ly > ry && ly < ry + rh) return "move";
				return "draw";
			}

			const cursorMap = {
				nw: "nwse-resize", se: "nwse-resize",
				ne: "nesw-resize", sw: "nesw-resize",
				n: "ns-resize", s: "ns-resize",
				w: "ew-resize", e: "ew-resize",
				move: "grab", draw: "crosshair",
			};

			// --- Widget callbacks ---
			const origWCb = cropWWidget.callback;
			const origHCb = cropHWidget.callback;
			cropWWidget.callback = function() { origWCb?.apply(this, arguments); updateCropFromWidgets(); };
			cropHWidget.callback = function() { origHCb?.apply(this, arguments); updateCropFromWidgets(); };

			// --- Pointer events ---
			overlay.addEventListener("pointermove", (e) => {
				if (drag) {
					e.stopPropagation();
					e.preventDefault();
					const p = pointerToLocal(e);
					const v = toVideoCoords(p.x, p.y);
					const dx = v.x - drag.startVX;
					const dy = v.y - drag.startVY;
					const o = drag.origCrop;
					const maxW = video.videoWidth, maxH = video.videoHeight;

					if (drag.type === "draw") {
						let w = Math.abs(v.x - drag.startVX);
						let h = Math.abs(v.y - drag.startVY);
						if (e.shiftKey && w > 0 && h > 0) {
							const aspect = drag.aspect || w / h;
							drag.aspect = aspect;
							if (w / h > aspect) w = Math.round(h * aspect);
							else h = Math.round(w / aspect);
						} else {
							drag.aspect = null;
						}
						crop = {
							x: v.x < drag.startVX ? drag.startVX - w : drag.startVX,
							y: v.y < drag.startVY ? drag.startVY - h : drag.startVY,
							w, h,
						};
					} else if (drag.type === "move") {
						let nx = o.x + dx, ny = o.y + dy;
						nx = Math.max(0, Math.min(maxW - o.w, nx));
						ny = Math.max(0, Math.min(maxH - o.h, ny));
						crop = { x: nx, y: ny, w: o.w, h: o.h };
					} else {
						let { x, y, w, h } = { ...o };
						if (drag.type.includes("w")) { x = o.x + dx; w = o.w - dx; }
						if (drag.type.includes("e")) { w = o.w + dx; }
						if (drag.type.includes("n")) { y = o.y + dy; h = o.h - dy; }
						if (drag.type.includes("s")) { h = o.h + dy; }
						if (w < 0) { x += w; w = -w; }
						if (h < 0) { y += h; h = -h; }
						if (e.shiftKey && o.w > 0 && o.h > 0 && w > 0 && h > 0) {
							const aspect = o.w / o.h;
							if (drag.type.length === 2) {
								if (w / h > aspect) w = Math.round(h * aspect);
								else h = Math.round(w / aspect);
								if (drag.type.includes("w")) x = o.x + o.w - w;
								if (drag.type.includes("n")) y = o.y + o.h - h;
							} else if (drag.type === "n" || drag.type === "s") {
								w = Math.round(h * aspect);
								x = o.x + Math.round((o.w - w) / 2);
							} else {
								h = Math.round(w / aspect);
								y = o.y + Math.round((o.h - h) / 2);
							}
						}
						x = Math.max(0, x); y = Math.max(0, y);
						w = Math.min(w, maxW - x); h = Math.min(h, maxH - y);
						crop = { x, y, w, h };
					}
					drawOverlay();
					updateSizeWidgets();
					return;
				}
				if (!previewVisible || !stream) return;
				const p = pointerToLocal(e);
				overlay.style.cursor = cursorMap[hitTest(p.x, p.y)];
			});

			overlay.addEventListener("pointerdown", (e) => {
				if (!previewVisible || !stream) return;
				e.stopPropagation();
				e.preventDefault();
				overlay.setPointerCapture(e.pointerId);
				const p = pointerToLocal(e);
				const type = hitTest(p.x, p.y);
				const v = toVideoCoords(p.x, p.y);
				drag = {
					type,
					startVX: v.x, startVY: v.y,
					origCrop: crop ? { ...crop } : null,
				};
				if (type === "move") overlay.style.cursor = "grabbing";
			});

			overlay.addEventListener("pointerup", (e) => {
				if (!drag) return;
				e.stopPropagation();
				if (drag.type === "move") overlay.style.cursor = "grab";
				drag = null;
				if (crop && (crop.w < 5 || crop.h < 5)) {
						crop = { x: 0, y: 0, w: video.videoWidth, h: video.videoHeight };
					}
				drawOverlay();
				updateSizeWidgets();
			});

			overlay.addEventListener("dblclick", (e) => {
				e.stopPropagation();
				e.preventDefault();
				clearCrop();
			});

			overlay.addEventListener("contextmenu", (e) => {
				e.stopPropagation();
				e.preventDefault();
				clearCrop();
			});

			// --- Resize observer ---
			const resizeObs = new ResizeObserver(() => { invalidateLayout(); drawOverlay(); });
			resizeObs.observe(container);

			// --- Timer ---
			function formatElapsed(ms) {
				const s = Math.floor(ms / 1000);
				const m = Math.floor(s / 60);
				const h = Math.floor(m / 60);
				const pad = (n) => String(n).padStart(2, "0");
				if (h > 0) return `${h}:${pad(m % 60)}:${pad(s % 60)}`;
				return `${m}:${pad(s % 60)}`;
			}

			function updateBtnTimer() {
				if (!captureStartTime || !btnWidget) return;
				const elapsed = formatElapsed(Date.now() - captureStartTime);
				btnWidget.name = `\uD83D\uDD34 ${elapsed} - Stop capture`;
				app.graph.setDirtyCanvas(true);
			}

			function startTimer() {
				captureStartTime = Date.now();
				timerInterval = setInterval(updateBtnTimer, 1000);
				updateBtnTimer();
			}

			function stopTimer() {
				if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
				captureStartTime = null;
			}

			// --- Stream management ---
			let btnWidget = null;

			function stopStream() {
				stopPreviewLoop();
				stopBitmapLoop();
				stopTimer();
				if (cachedBitmap) { cachedBitmap.close(); cachedBitmap = null; }
				if (stream) stream.getTracks().forEach(t => t.stop());
				stream = null;
				crop = null;
				video.srcObject = null;
				if (btnWidget) btnWidget.name = "Start capture";
				app.graph.setDirtyCanvas(true);
			}

			btnWidget = this.addWidget("button", "Start capture", null, async () => {
				if (stream) { stopStream(); return; }
				try {
					// capture at native resolution — browser constraints
					// cause green tint. Preview is downscaled via
					// createImageBitmap, output goes to Python at full res.
					stream = await navigator.mediaDevices.getDisplayMedia({
						video: { frameRate: { ideal: 5, max: 10 } },
					});
					video.srcObject = stream;
					startTimer();
					const track = stream.getVideoTracks()[0];
					track.addEventListener("ended", stopStream);
					video.addEventListener("loadedmetadata", () => {
						crop = { x: 0, y: 0, w: video.videoWidth, h: video.videoHeight };
						invalidateLayout();
						updateSizeWidgets();
						startBitmapLoop();
						if (previewVisible) startPreviewLoop();
					}, { once: true });
				} catch { /* user cancelled */ }
			});

			// --- DOM widget ---
			this.addDOMWidget("preview", "ScreencapPreview", container, {
				serialize: false,
				hideOnZoom: false,
				getMinHeight: () => 200,
			});

			const previewBtn = this.addWidget("button", "Hide preview", null, () => {
				previewVisible = !previewVisible;
				previewBtn.name = previewVisible ? "Hide preview" : "Show preview";
				if (previewVisible && stream) {
					invalidateLayout();
					startBitmapLoop();
					startPreviewLoop();
				} else {
					stopPreviewLoop();
					stopBitmapLoop();
					overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
				}
				app.graph.setDirtyCanvas(true);
			});

			// --- Frame serialization ---
			// Always send full crop resolution to Python.
			// Python handles downscaling with Lanczos for better quality.
			frameWidget.serializeValue = () => {
				if (!stream || !video.videoWidth) return "";
				const vw = video.videoWidth, vh = video.videoHeight;
				const sx = crop ? Math.min(crop.x, vw) : 0;
				const sy = crop ? Math.min(crop.y, vh) : 0;
				const sw = crop ? Math.min(crop.w, vw - sx) : vw;
				const sh = crop ? Math.min(crop.h, vh - sy) : vh;
				if (sw <= 0 || sh <= 0) return "";
				serializeCanvas.width = sw;
				serializeCanvas.height = sh;
				serializeCtx.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
				return serializeCanvas.toDataURL("image/png");
			};

			// --- Cleanup ---
			const origOnRemoved = this.onRemoved;
			this.onRemoved = function () {
				stopStream();
				resizeObs.disconnect();
				origOnRemoved?.apply(this, arguments);
			};
		};
	},
});
