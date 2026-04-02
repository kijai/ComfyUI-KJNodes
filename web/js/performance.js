const { app } = window.comfyAPI.app;

const _savedRoundRadius = typeof LiteGraph !== "undefined" ? LiteGraph.ROUND_RADIUS : 8;

function getSetting(id, fallback) {
	return app.ui.settings.getSettingValue(id) ?? fallback;
}

function toggleSetting(id) {
	const cur = app.ui.settings.getSettingValue(id);
	app.ui.settings.setSettingValue(id, !cur);
}

app.registerExtension({
	name: "KJNodes.Performance",

	commands: [
		{
			id: "KJNodes.perf.toggleSingleCanvasPan",
			label: "Single-canvas mode during pan",
			menubarLabel: "Single-canvas mode during pan",
			function: () => toggleSetting("KJNodes.perf.singleCanvasPan"),
			active: () => getSetting("KJNodes.perf.singleCanvasPan", false),
		},
		{
			id: "KJNodes.perf.toggleDisableShadows",
			label: "Disable node shadows",
			menubarLabel: "Disable node shadows",
			function: () => toggleSetting("KJNodes.perf.disableShadows"),
			active: () => getSetting("KJNodes.perf.disableShadows", false),
		},
		{
			id: "KJNodes.perf.toggleDisableConnectionBorders",
			label: "Disable connection borders",
			menubarLabel: "Disable connection borders",
			function: () => toggleSetting("KJNodes.perf.disableConnectionBorders"),
			active: () => getSetting("KJNodes.perf.disableConnectionBorders", false),
		},
		{
			id: "KJNodes.perf.toggleDisableRoundedCorners",
			label: "Disable rounded corners",
			menubarLabel: "Disable rounded corners",
			function: () => toggleSetting("KJNodes.perf.disableRoundedCorners"),
			active: () => getSetting("KJNodes.perf.disableRoundedCorners", false),
		},
		{
			id: "KJNodes.perf.toggleThrottleRenderInfo",
			label: "Throttle info overlay",
			menubarLabel: "Throttle info overlay",
			function: () => toggleSetting("KJNodes.perf.throttleRenderInfo"),
			active: () => getSetting("KJNodes.perf.throttleRenderInfo", false),
		},
	],

	menuCommands: [
		{
			path: ["KJNodes", "Performance"],
			commands: [
				"KJNodes.perf.toggleSingleCanvasPan",
				"KJNodes.perf.toggleDisableShadows",
				"KJNodes.perf.toggleDisableConnectionBorders",
				"KJNodes.perf.toggleDisableRoundedCorners",
				"KJNodes.perf.toggleThrottleRenderInfo",
			],
		},
	],

	settings: [
		{
			id: "KJNodes.perf.singleCanvasPan",
			name: "Single-canvas mode during pan",
			category: ["KJNodes", "Performance", "Single-canvas mode during pan"],
			tooltip: "Eliminates the expensive bgcanvas-to-canvas copy during panning. Significant improvement without hardware acceleration.",
			type: "boolean",
			defaultValue: false,
			onChange: (value) => {
				const canvas = app.canvas;
				if (!canvas) return;
				if (value) {
					installSingleCanvasPan(canvas);
				} else {
					uninstallSingleCanvasPan(canvas);
				}
			},
		},
		{
			id: "KJNodes.perf.disableShadows",
			name: "Disable node shadows",
			category: ["KJNodes", "Performance", "Disable node shadows"],
			tooltip: "Disable shadow rendering on nodes for better performance without hardware acceleration.",
			type: "boolean",
			defaultValue: false,
			onChange: (value) => {
				if (app.canvas) {
					app.canvas.render_shadows = !value;
					app.canvas.setDirty(true, true);
				}
			},
		},
		{
			id: "KJNodes.perf.disableConnectionBorders",
			name: "Disable connection borders",
			category: ["KJNodes", "Performance", "Disable connection borders"],
			tooltip: "Disable the outer border stroke on connection lines. Each link draws two strokes instead of one when enabled.",
			type: "boolean",
			defaultValue: false,
			onChange: (value) => {
				if (app.canvas) {
					app.canvas.render_connections_border = !value;
					app.canvas.setDirty(true, true);
				}
			},
		},
		{
			id: "KJNodes.perf.disableRoundedCorners",
			name: "Disable rounded corners",
			category: ["KJNodes", "Performance", "Disable rounded corners"],
			tooltip: "Use square corners on nodes instead of rounded. Avoids roundRect calls in software rendering.",
			type: "boolean",
			defaultValue: false,
			onChange: (value) => {
				if (typeof LiteGraph !== "undefined") {
					LiteGraph.ROUND_RADIUS = value ? 0 : _savedRoundRadius;
					app.canvas?.setDirty(true, true);
				}
			},
		},
		{
			id: "KJNodes.perf.throttleRenderInfo",
			name: "Throttle info overlay",
			category: ["KJNodes", "Performance", "Throttle info overlay"],
			tooltip: "Cache the FPS/info text overlay and only re-render it a few times per second. Saves ~20ms per frame in software rendering.",
			type: "boolean",
			defaultValue: false,
			onChange: (value) => {
				const canvas = app.canvas;
				if (!canvas) return;
				if (value) {
					installThrottleRenderInfo(canvas);
				} else {
					uninstallThrottleRenderInfo(canvas);
				}
			},
		},
	],

	setup() {
		const canvas = app.canvas;
		if (!canvas) return;

		if (getSetting("KJNodes.perf.singleCanvasPan", false)) {
			installSingleCanvasPan(canvas);
		}
		if (getSetting("KJNodes.perf.disableShadows", false)) {
			canvas.render_shadows = false;
		}
		if (getSetting("KJNodes.perf.disableConnectionBorders", false)) {
			canvas.render_connections_border = false;
		}
		if (getSetting("KJNodes.perf.disableRoundedCorners", false)) {
			LiteGraph.ROUND_RADIUS = 0;
		}
		if (getSetting("KJNodes.perf.throttleRenderInfo", false)) {
			installThrottleRenderInfo(canvas);
		}
	},
});

// ── Single-canvas mode during pan ───────────────────────────────────────────

let _origDraw = null;
let _panInstalledOn = null;

function installSingleCanvasPan(canvas) {
	if (_origDraw && _panInstalledOn === canvas) return;
	if (_origDraw && _panInstalledOn !== canvas) {
		// Canvas was replaced — discard stale reference
		_origDraw = null;
	}
	_panInstalledOn = canvas;
	_origDraw = canvas.draw;
	let panning = false;
	let savedBgCanvas = null;
	let savedBgCtx = null;

	canvas.draw = function (force_canvas, force_bgcanvas) {
		if (this.dragging_canvas) {
			if (!panning) {
				// Flush any pending bg redraw before aliasing to avoid a stale-frame flash
				if (this.dirty_bgcanvas) {
					_origDraw.call(this, false, true);
				}
				savedBgCanvas = this.bgcanvas;
				savedBgCtx = this.bgctx;
				panning = true;
				this.bgcanvas = this.canvas;
				this.bgctx = this.ctx;
			}
		} else if (panning) {
			panning = false;
			this.bgcanvas = savedBgCanvas;
			this.bgctx = savedBgCtx;
			this.dirty_bgcanvas = true;
			this.dirty_canvas = true;
		}
		_origDraw.call(this, force_canvas, force_bgcanvas);
	};
}

function uninstallSingleCanvasPan(canvas) {
	if (!_origDraw) return;
	if (_panInstalledOn === canvas) {
		canvas.draw = _origDraw;
	}
	_origDraw = null;
	_panInstalledOn = null;
	canvas.dirty_bgcanvas = true;
	canvas.dirty_canvas = true;
}

// ── Throttle renderInfo ─────────────────────────────────────────────────────

let _origRenderInfo = null;
let _infoCanvas = null;
let _infoInstalledOn = null;

function installThrottleRenderInfo(canvas) {
	if (_origRenderInfo && _infoInstalledOn === canvas) return;
	if (_origRenderInfo && _infoInstalledOn !== canvas) {
		_origRenderInfo = null;
		_infoCanvas = null;
	}
	_infoInstalledOn = canvas;
	_origRenderInfo = canvas.renderInfo;
	_infoCanvas = document.createElement("canvas");
	let infoCtx = _infoCanvas.getContext("2d");
	let lastInfoTime = 0;
	let cachedDpr = 0;
	let _infoDrawX = 10;
	let _infoDrawY = 0;

	canvas.renderInfo = function (ctx, x, y) {
		const dpr = window.devicePixelRatio || 1;

		// Resize offscreen canvas if DPR changed
		if (dpr !== cachedDpr) {
			cachedDpr = dpr;
			_infoCanvas.width = Math.ceil(200 * dpr);
			_infoCanvas.height = Math.ceil(100 * dpr);
			infoCtx = _infoCanvas.getContext("2d");
			infoCtx.scale(dpr, dpr);
			lastInfoTime = 0;
		}

		const now = performance.now();
		if (now - lastInfoTime > 250) {
			lastInfoTime = now;
			infoCtx.clearRect(0, 0, 200, 100);
			// Render with the same x/y the original would use, but offset
			// into offscreen canvas coordinates
			const origX = x || 10;
			const lineHeight = 13;
			const lineCount = (this.graph ? 5 : 1) + (this.info_text ? 1 : 0);
			const origY = y || (this.canvas.height / dpr - (lineCount + 1) * lineHeight);
			// Store the computed position for blitting
			_infoDrawX = origX;
			_infoDrawY = origY;
			// Draw into offscreen at (1, 1), text flows from there
			_origRenderInfo.call(this, infoCtx, 1, 1);
		}

		// Blit cached text at the original position
		ctx.drawImage(_infoCanvas, _infoDrawX - 1, _infoDrawY - 1, 200, 100);
	};
}

function uninstallThrottleRenderInfo(canvas) {
	if (!_origRenderInfo) return;
	if (_infoInstalledOn === canvas) {
		canvas.renderInfo = _origRenderInfo;
	}
	_origRenderInfo = null;
	_infoCanvas = null;
	_infoInstalledOn = null;
	canvas.setDirty(true, true);
}
