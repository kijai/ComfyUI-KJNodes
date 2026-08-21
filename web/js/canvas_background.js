const { app } = window.comfyAPI.app;

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Linearly mix two hex colors: 0 = pure a, 1 = pure b. */
function mixColors(a, b, t) {
	const parse = (hex) => {
		if (hex.length === 4) hex = `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`;
		return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
	};
	const ca = parse(a), cb = parse(b);
	const r = Math.round(ca[0] + (cb[0] - ca[0]) * t);
	const g = Math.round(ca[1] + (cb[1] - ca[1]) * t);
	const bl = Math.round(ca[2] + (cb[2] - ca[2]) * t);
	return `#${((1 << 24) | (r << 16) | (g << 8) | bl).toString(16).slice(1)}`;
}

/** Ensure a color string has a leading '#' (PrimeVue ColorPicker omits it). */
function asHex(color, fallback = "#000000") {
	if (!color) return fallback;
	return color.startsWith("#") ? color : `#${color}`;
}

function getSetting(id, fallback) {
	return app.ui.settings.getSettingValue(id) ?? fallback;
}

// ── Pattern tiles ─────────────────────────────────────────────────────────────
// Each pattern is described by:
//   tile(size) → [tileW, tileH]   the period of the pattern in graph units.
//   draw(ctx, size, t, fg, bg)    draws ONE seamless tile of the foreground
//                                  motif into a context already scaled so that
//                                  [0..tileW] × [0..tileH] graph units fill the
//                                  whole tile bitmap. The solid background is
//                                  painted separately, so tiles are transparent
//                                  except where the motif is drawn.
//
// At render time we rasterise one tile into a small OffscreenCanvas sized to the
// pattern's *on-screen* pixel size for the current zoom, then fill the canvas
// with createPattern("repeat"). This is O(1) per frame (the GPU repeats the
// tile) instead of stroking the whole graph every frame, and because the tile
// is rendered at native device resolution and blitted ~1:1

const patternDefs = {
	dots: {
		tile: (size) => [size, size],
		draw: (ctx, size, t, fg) => {
			ctx.fillStyle = fg;
			ctx.beginPath();
			ctx.arc(size / 2, size / 2, t * 2, 0, Math.PI * 2);
			ctx.fill();
		},
	},

	grid: {
		tile: (size) => [size, size],
		draw: (ctx, size, t, fg) => {
			ctx.fillStyle = fg;
			ctx.fillRect(size - t, 0, t, size - t);
			ctx.fillRect(0, size - t, size, t);
		},
	},

	"cross dots": {
		tile: (size) => [size, size],
		draw: (ctx, size, t, fg) => {
			const half = size / 2;
			const arm = Math.max(t + 1, size * 0.08);
			const ht = t / 2;
			ctx.fillStyle = fg;
			ctx.fillRect(half - arm, half - ht, arm * 2, t);
			ctx.fillRect(half - ht, half - arm, t, arm * 2);
		},
	},

	blueprint: {
		tile: (size) => [size, size],
		draw: (ctx, size, t, fg, bg) => {
			const sub = size / 4;
			ctx.fillStyle = mixColors(bg, fg, 0.3);
			for (let i = sub; i < size; i += sub) {
				ctx.fillRect(i, 0, t, size - t);
				ctx.fillRect(0, i, size, t);
			}
			ctx.fillStyle = fg;
			ctx.fillRect(size - t, 0, t, size - t);
			ctx.fillRect(0, size - t, size, t);
		},
	},

	isometric: {
		tile: (size) => [size, Math.round(size / 2)],
		draw: (ctx, size, t, fg) => {
			const w = size;
			const h = Math.round(size / 2);
			ctx.strokeStyle = fg;
			ctx.lineWidth = t;
			ctx.beginPath();
			ctx.moveTo(0, h / 2);
			ctx.lineTo(w / 2, 0);
			ctx.lineTo(w, h / 2);
			ctx.moveTo(0, h / 2);
			ctx.lineTo(w / 2, h);
			ctx.lineTo(w, h / 2);
			ctx.stroke();
		},
	},

	hexagons: {
		tile: (size) => [(size / 2) * 3, Math.round((size / 2) * Math.sqrt(3))],
		draw: (ctx, size, t, fg) => {
			const r = size / 2;
			const h = Math.round(r * Math.sqrt(3));
			ctx.strokeStyle = fg;
			ctx.lineWidth = t;
			const hex = (cx, cy) => {
				ctx.beginPath();
				for (let i = 0; i < 6; i++) {
					const a = Math.PI / 3 * i;
					const px = cx + r * Math.cos(a);
					const py = cy + r * Math.sin(a);
					if (i === 0) ctx.moveTo(px, py);
					else ctx.lineTo(px, py);
				}
				ctx.closePath();
				ctx.stroke();
			};
			hex(r, h / 2);
			hex(r * 2.5, 0);
			hex(r * 2.5, h);
		},
	},

	octagons: {
		tile: (size) => [size, size],
		draw: (ctx, size, t, fg) => {
			const s = size / (1 + Math.SQRT2);
			const d = (size - s) / 2;
			ctx.strokeStyle = fg;
			ctx.lineWidth = t;
			ctx.beginPath();
			ctx.moveTo(d, 0);
			ctx.lineTo(size - d, 0);
			ctx.lineTo(size, d);
			ctx.lineTo(size, size - d);
			ctx.lineTo(size - d, size);
			ctx.lineTo(d, size);
			ctx.lineTo(0, size - d);
			ctx.lineTo(0, d);
			ctx.closePath();
			ctx.stroke();
		},
	},

	waves: {
		tile: (size) => [size, size],
		draw: (ctx, size, t, fg, bg) => {
			const lines = Math.max(2, Math.round(size / 20));
			const spacing = size / lines;
			ctx.lineWidth = t;
			for (let i = 0; i < lines; i++) {
				const blend = 0.1 + (i % 3) * 0.07;
				const amp = spacing * (0.15 + (i % 4) * 0.05);
				const baseY = spacing * i + spacing / 2;
				ctx.strokeStyle = mixColors(bg, fg, blend);
				ctx.beginPath();
				for (let x = 0; x <= size; x++) {
					const y = baseY + Math.sin((x / size) * Math.PI * 2 + i * 0.8) * amp;
					if (x === 0) ctx.moveTo(x, y);
					else ctx.lineTo(x, y);
				}
				ctx.stroke();
			}
		},
	},

	"carbon fiber": {
		tile: (size) => {
			const cell = Math.max(2, Math.round(size / 4));
			return [cell * 4, cell * 4];
		},
		draw: (ctx, size, t, fg, bg) => {
			const cell = Math.max(2, Math.round(size / 4));
			const s = cell * 4;
			// Dark cells == background, left transparent so the bg shows through.
			ctx.fillStyle = mixColors(bg, fg, 0.2);
			for (let row = 0; row < 4; row++) {
				for (let col = 0; col < 4; col++) {
					if ((col + row) % 4 < 2) ctx.fillRect(col * cell, row * cell, cell, cell);
				}
			}
			ctx.fillStyle = mixColors(bg, fg, 0.07);
			const gt = Math.max(1, Math.round(t));
			for (let i = 1; i < 4; i++) {
				ctx.fillRect(0, i * cell, s, gt);
				ctx.fillRect(i * cell, 0, gt, s);
			}
		},
	},
};

// Ordered list of choices: 'default' (LiteGraph's own grid) + 'none' + patterns.
const PATTERN_KEYS = ["default", "none", ...Object.keys(patternDefs)];

// Below this on-screen tile size (device px) the motif is an indiscernible
// smear, so we skip it and show the plain background.
const MIN_TILE_PX = 3;
// Cap the tile bitmap. This bounds the only per-zoom-frame work (re-rendering
// one tile to match the new on-screen size); beyond this the tile is gently
// upscaled by the pattern fill (few, large tiles when zoomed in → fine).
const MAX_TILE_PX = 256;

// ── State ───────────────────────────────────────────────────────────────────

let currentType = "default";
let currentFg = "#444444";
let currentBg = "#212121";
let currentSize = 32;
let currentThickness = 2;
let currentTileW = 32;   // tile period (graph units) for the active pattern
let currentTileH = 32;
let dirty = true;

// Bumped whenever a setting changes; lets the per-frame cache check compare a
// number instead of building a key string each frame (avoids GC churn on drag).
let settingsVersion = 0;

// Cached tile bitmap + pattern — reused across frames; rebuilt only when the
// settings version or the on-screen tile size changes. Panning/dragging (scale
// unchanged) always hits the cache with zero allocation.
const patternCache = { ver: -1, iDw: -1, iDh: -1, pattern: null };

function invalidateCache() {
	dirty = true;
	settingsVersion++;
	app.canvas?.setDirty(true, true);
}

function readSettings() {
	currentType = getSetting("KJNodes.canvasBg.pattern", "default");
	currentFg = asHex(getSetting("KJNodes.canvasBg.patternColor", "#444444"));
	currentBg = asHex(getSetting("KJNodes.canvasBg.bgColor", "#212121"));
	currentSize = getSetting("KJNodes.canvasBg.scale", 32);
	currentThickness = getSetting("KJNodes.canvasBg.thickness", 2);
	// Resolve the tile period once per change so the render loop never allocates.
	const def = patternDefs[currentType];
	if (def) {
		const dims = def.tile(currentSize);
		currentTileW = dims[0];
		currentTileH = dims[1];
	}
}

/** Build (or reuse) a repeat pattern of one tile rendered at `iDw × iDh` px. */
function getPattern(ctx, def, tileW, tileH, iDw, iDh) {
	if (patternCache.ver === settingsVersion && patternCache.iDw === iDw && patternCache.iDh === iDh && patternCache.pattern) {
		return patternCache.pattern;
	}

	const oc = new OffscreenCanvas(iDw, iDh);
	const tctx = oc.getContext("2d");
	tctx.scale(iDw / tileW, iDh / tileH);
	// Bake the background into the tile so the whole canvas is painted with a
	// single opaque pattern fill (no separate solid-fill pass per frame).
	tctx.fillStyle = currentBg;
	tctx.fillRect(0, 0, tileW, tileH);
	def.draw(tctx, currentSize, currentThickness, currentFg, currentBg);

	patternCache.pattern = ctx.createPattern(oc, "repeat");
	patternCache.ver = settingsVersion;
	patternCache.iDw = iDw;
	patternCache.iDh = iDh;
	return patternCache.pattern;
}

// ── Commands ────────────────────────────────────────────────────────────────

const patternCommands = PATTERN_KEYS.map((key) => ({
	id: `KJNodes.canvasBg.setPattern.${key}`,
	label: key.charAt(0).toUpperCase() + key.slice(1),
	menubarLabel: key.charAt(0).toUpperCase() + key.slice(1),
	function: () => {
		app.ui.settings.setSettingValue("KJNodes.canvasBg.pattern", key);
	},
	active: () => getSetting("KJNodes.canvasBg.pattern", "default") === key,
}));

// Single source of truth for slider bounds — commands reference these same objects.
const SCALE_ATTRS = { min: 8, max: 512, step: 4 };
const THICKNESS_ATTRS = { min: 0.5, max: 20, step: 0.5 };

function stepSetting(id, fallback, attrs, direction) {
	const cur = getSetting(id, fallback);
	const next = cur + attrs.step * direction;
	if (next >= attrs.min && next <= attrs.max) {
		app.ui.settings.setSettingValue(id, next);
	}
}

const scaleCommands = [
	{
		id: "KJNodes.canvasBg.increaseScale",
		label: "Increase pattern scale",
		menubarLabel: "Increase scale",
		function: () => stepSetting("KJNodes.canvasBg.scale", 32, SCALE_ATTRS, 1),
		active: () => false,
	},
	{
		id: "KJNodes.canvasBg.decreaseScale",
		label: "Decrease pattern scale",
		menubarLabel: "Decrease scale",
		function: () => stepSetting("KJNodes.canvasBg.scale", 32, SCALE_ATTRS, -1),
		active: () => false,
	},
];

const thicknessCommands = [
	{
		id: "KJNodes.canvasBg.increaseThickness",
		label: "Increase pattern thickness",
		menubarLabel: "Increase thickness",
		function: () => stepSetting("KJNodes.canvasBg.thickness", 2, THICKNESS_ATTRS, 1),
		active: () => false,
	},
	{
		id: "KJNodes.canvasBg.decreaseThickness",
		label: "Decrease pattern thickness",
		menubarLabel: "Decrease thickness",
		function: () => stepSetting("KJNodes.canvasBg.thickness", 2, THICKNESS_ATTRS, -1),
		active: () => false,
	},
];

// ── Extension ───────────────────────────────────────────────────────────────

app.registerExtension({
	name: "KJNodes.CanvasBackground",

	commands: [...patternCommands, ...scaleCommands, ...thicknessCommands],

	menuCommands: [
		{
			path: ["KJNodes", "Canvas Background"],
			commands: patternCommands.map((c) => c.id),
		},
		{
			path: ["KJNodes", "Canvas Background"],
			commands: scaleCommands.map((c) => c.id),
		},
		{
			path: ["KJNodes", "Canvas Background"],
			commands: thicknessCommands.map((c) => c.id),
		},
	],

	settings: [
		{
			id: "KJNodes.canvasBg.pattern",
			name: "Background pattern",
			category: ["KJNodes", "Canvas Background", "Pattern"],
			tooltip: "Choose a background pattern for the node graph canvas. 'default' uses LiteGraph's built-in dot grid. 'none' disables all background drawing.",
			type: "combo",
			defaultValue: "default",
			options: PATTERN_KEYS,
			onChange: invalidateCache,
		},
		{
			id: "KJNodes.canvasBg.patternColor",
			name: "Pattern color",
			category: ["KJNodes", "Canvas Background", "Pattern color"],
			tooltip: "Foreground color of the pattern",
			type: "color",
			defaultValue: "#444444",
			onChange: invalidateCache,
		},
		{
			id: "KJNodes.canvasBg.bgColor",
			name: "Background color",
			category: ["KJNodes", "Canvas Background", "Background color"],
			tooltip: "Background fill color behind the pattern",
			type: "color",
			defaultValue: "#212121",
			onChange: invalidateCache,
		},
		{
			id: "KJNodes.canvasBg.scale",
			name: "Pattern scale",
			category: ["KJNodes", "Canvas Background", "Pattern scale"],
			tooltip: "Size of one pattern tile in pixels",
			type: "slider",
			defaultValue: 32,
			attrs: SCALE_ATTRS,
			onChange: invalidateCache,
		},
		{
			id: "KJNodes.canvasBg.thickness",
			name: "Pattern thickness",
			category: ["KJNodes", "Canvas Background", "Pattern thickness"],
			tooltip: "Thickness of lines and size of dots in pixels",
			type: "slider",
			defaultValue: 2,
			attrs: THICKNESS_ATTRS,
			onChange: invalidateCache,
		},
	],

	setup() {
		const canvas = app.canvas;
		if (!canvas) return;

		// onRenderBackground runs before LiteGraph applies its own graph
		// transform and before it paints the clear color / background pattern.
		// Returning true tells LiteGraph we handled the background, so it skips
		// both. We then paint the whole canvas with a single repeat-pattern fill
		const origCallback = canvas.onRenderBackground;
		canvas.onRenderBackground = function (cvs, ctx) {
			if (origCallback) {
				const result = origCallback.call(this, cvs, ctx);
				if (result) return true;
			}

			if (dirty) {
				readSettings();
				dirty = false;
			}

			// If ComfyUI's own background image setting is active, don't interfere.
			if (getSetting("Comfy.Canvas.BackgroundImage", "")) return false;

			// 'default' → let LiteGraph draw its built-in dot grid as usual.
			if (currentType === "default") return false;

			const scale = this.ds.scale;
			const offset = this.ds.offset;
			const dpr = window.devicePixelRatio || 1;

			ctx.save();

			const def = patternDefs[currentType];
			let painted = false;
			if (def) {
				const tileW = currentTileW;
				const tileH = currentTileH;
				const k = scale * dpr;
				const onW = tileW * k;   // on-screen tile size, device px
				const onH = tileH * k;

				if (onW >= MIN_TILE_PX && onH >= MIN_TILE_PX) {
					const iDw = Math.min(Math.round(onW), MAX_TILE_PX);
					const iDh = Math.min(Math.round(onH), MAX_TILE_PX);
					const pattern = getPattern(ctx, def, tileW, tileH, iDw, iDh);

					// Tile the (bg-baked, opaque) pattern over the whole canvas with a
					// single nearest-neighbor fill. The tile is rendered at device resolution
					const a = Math.round(onW) <= MAX_TILE_PX ? 1 : onW / iDw;
					const d = Math.round(onH) <= MAX_TILE_PX ? 1 : onH / iDh;
					const periodX = a * iDw;
					const periodY = d * iDh;
					let px = dpr * scale * offset[0];
					let py = dpr * scale * offset[1];
					px -= Math.floor(px / periodX) * periodX;
					py -= Math.floor(py / periodY) * periodY;

					ctx.imageSmoothingEnabled = false;
					ctx.setTransform(a, 0, 0, d, px, py);
					ctx.fillStyle = pattern;
					ctx.fillRect(-px / a, -py / d, cvs.width / a, cvs.height / d);
					painted = true;
				}
			}

			if (!painted) {
				// 'none', or tiles too small to discern → plain background only.
				ctx.setTransform(1, 0, 0, 1, 0, 0);
				ctx.fillStyle = currentBg;
				ctx.fillRect(0, 0, cvs.width, cvs.height);
			}

			ctx.restore();
			return true;
		};

		invalidateCache();
	},
});
