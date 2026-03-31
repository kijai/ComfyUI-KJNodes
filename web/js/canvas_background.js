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

// ── Pattern generators ──────────────────────────────────────────────────────
// Each returns an OffscreenCanvas to be used with createPattern("repeat").
// `size` = tile size, `fg`/`bg` = hex colors, `t` = feature thickness.

function makeDots(size, fg, bg, t) {
	const c = new OffscreenCanvas(size, size);
	const ctx = c.getContext("2d");
	ctx.fillStyle = bg;
	ctx.fillRect(0, 0, size, size);
	ctx.fillStyle = fg;
	ctx.beginPath();
	ctx.arc(size / 2, size / 2, t * 2, 0, Math.PI * 2);
	ctx.fill();
	return c;
}

function makeGrid(size, fg, bg, t) {
	const c = new OffscreenCanvas(size, size);
	const ctx = c.getContext("2d");
	ctx.fillStyle = bg;
	ctx.fillRect(0, 0, size, size);
	ctx.fillStyle = fg;
	ctx.fillRect(size - t, 0, t, size - t);
	ctx.fillRect(0, size - t, size, t);
	return c;
}

function makeCrossDots(size, fg, bg, t) {
	const c = new OffscreenCanvas(size, size);
	const ctx = c.getContext("2d");
	ctx.fillStyle = bg;
	ctx.fillRect(0, 0, size, size);
	const half = size / 2;
	const arm = Math.max(t + 1, size * 0.08);
	const ht = t / 2;
	ctx.fillStyle = fg;
	ctx.fillRect(half - arm, half - ht, arm * 2, t);
	ctx.fillRect(half - ht, half - arm, t, arm * 2);
	return c;
}

function makeBlueprint(size, fg, bg, t) {
	const c = new OffscreenCanvas(size, size);
	const ctx = c.getContext("2d");
	ctx.fillStyle = bg;
	ctx.fillRect(0, 0, size, size);
	const sub = size / 4;
	const subColor = mixColors(bg, fg, 0.3);
	ctx.fillStyle = subColor;
	for (let i = sub; i < size; i += sub) {
		ctx.fillRect(i, 0, t, size - t);
		ctx.fillRect(0, i, size, t);
	}
	ctx.fillStyle = fg;
	ctx.fillRect(size - t, 0, t, size - t);
	ctx.fillRect(0, size - t, size, t);
	return c;
}

function makeIsometric(size, fg, bg, t) {
	const w = size;
	const h = Math.round(size / 2);
	const c = new OffscreenCanvas(w, h);
	const ctx = c.getContext("2d");
	ctx.fillStyle = bg;
	ctx.fillRect(0, 0, w, h);
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
	return c;
}

function makeHexagons(size, fg, bg, t) {
	const r = size / 2;
	const w = r * 3;
	const h = Math.round(r * Math.sqrt(3));
	const c = new OffscreenCanvas(w, h);
	const ctx = c.getContext("2d");
	ctx.fillStyle = bg;
	ctx.fillRect(0, 0, w, h);
	ctx.strokeStyle = fg;
	ctx.lineWidth = t;

	function hex(cx, cy) {
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
	}

	hex(r, h / 2);
	hex(r * 2.5, 0);
	hex(r * 2.5, h);
	return c;
}

function makeOctagons(size, fg, bg, t) {
	const c = new OffscreenCanvas(size, size);
	const ctx = c.getContext("2d");
	ctx.fillStyle = bg;
	ctx.fillRect(0, 0, size, size);
	ctx.strokeStyle = fg;
	ctx.lineWidth = t;

	const s = size / (1 + Math.SQRT2);
	const d = (size - s) / 2;

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
	return c;
}

function makeWaves(size, fg, bg, t) {
	const c = new OffscreenCanvas(size, size);
	const ctx = c.getContext("2d");
	ctx.fillStyle = bg;
	ctx.fillRect(0, 0, size, size);
	ctx.lineWidth = t;

	const lines = Math.max(2, Math.round(size / 20));
	const spacing = size / lines;

	for (let i = 0; i < lines; i++) {
		const blend = 0.1 + (i % 3) * 0.07;
		ctx.strokeStyle = mixColors(bg, fg, blend);
		const baseY = spacing * i + spacing / 2;
		const amp = spacing * (0.15 + (i % 4) * 0.05);

		ctx.beginPath();
		for (let x = 0; x <= size; x++) {
			const y = baseY + Math.sin((x / size) * Math.PI * 2 + i * 0.8) * amp;
			if (x === 0) ctx.moveTo(x, y);
			else ctx.lineTo(x, y);
		}
		ctx.stroke();
	}
	return c;
}

function makeCarbonFiber(size, fg, bg, t) {
	const cell = Math.max(2, Math.round(size / 4));
	const s = cell * 4;
	const c = new OffscreenCanvas(s, s);
	const ctx = c.getContext("2d");

	const dark = bg;
	const light = mixColors(bg, fg, 0.2);
	const groove = mixColors(bg, fg, 0.07);

	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			const phase = (col + row) % 4;
			ctx.fillStyle = (phase < 2) ? light : dark;
			ctx.fillRect(col * cell, row * cell, cell, cell);
		}
	}

	ctx.fillStyle = groove;
	const gt = Math.max(1, Math.round(t));
	for (let i = 1; i < 4; i++) {
		ctx.fillRect(0, i * cell, s, gt);
		ctx.fillRect(i * cell, 0, gt, s);
	}
	return c;
}


const PATTERNS = {
	default: null,
	none: null,
	dots: makeDots,
	grid: makeGrid,
	"cross dots": makeCrossDots,
	blueprint: makeBlueprint,
	isometric: makeIsometric,
	hexagons: makeHexagons,
	octagons: makeOctagons,
	waves: makeWaves,
	"carbon fiber": makeCarbonFiber,
};

// ── State ───────────────────────────────────────────────────────────────────

let currentType = "default";
let currentFg = "#444444";
let currentBg = "#212121";
let currentSize = 32;
let currentThickness = 2;
let dirty = true;

function invalidateCache() {
	dirty = true;
	app.canvas?.setDirty(true, true);
}



function readSettings() {
	currentType = getSetting("KJNodes.canvasBg.pattern", "default");
	currentFg = asHex(getSetting("KJNodes.canvasBg.patternColor", "#444444"));
	currentBg = asHex(getSetting("KJNodes.canvasBg.bgColor", "#212121"));
	currentSize = getSetting("KJNodes.canvasBg.scale", 32);
	currentThickness = getSetting("KJNodes.canvasBg.thickness", 2);
}

// ── Commands ────────────────────────────────────────────────────────────────

const patternCommands = Object.keys(PATTERNS).map((key) => ({
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
			options: Object.keys(PATTERNS),
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

		let patternToken = null; // null = default, "" = none, string = custom
		// Snapshot palette values lazily on first override so we capture post-palette state
		let origBackgroundImage = null;
		let origClearBgColor = null;
		let origsCaptured = false;

		function captureOriginals() {
			if (origsCaptured) return;
			origsCaptured = true;
			origBackgroundImage = canvas.background_image;
			origClearBgColor = canvas.clear_background_color;
		}

		function applyToLiteGraph() {
			if (!dirty) return;
			dirty = false;
			readSettings();

			// If ComfyUI's own background image setting is active, don't interfere
			if (getSetting("Comfy.Canvas.BackgroundImage", "")) {
				patternToken = null;
				return;
			}

			if (currentType === "default") {
				if (origsCaptured) {
					canvas.background_image = origBackgroundImage;
					canvas.clear_background_color = origClearBgColor;
					canvas._pattern = undefined;
					canvas._bg_img = undefined;
				}
				patternToken = null;
				canvas.setDirty(true, true);
				return;
			}

			// Capture originals before we overwrite them for the first time
			captureOriginals();

			if (currentType === "none") {
				canvas.background_image = "";
				canvas.clear_background_color = currentBg;
				canvas._pattern = undefined;
				canvas._bg_img = undefined;
				patternToken = "";
				canvas.setDirty(true, true);
				return;
			}

			// Generate tile as OffscreenCanvas — passed directly to LiteGraph's
			// createPattern which accepts any CanvasImageSource.
			const gen = PATTERNS[currentType];
			if (!gen) return;

			const tile = gen(currentSize, currentFg, currentBg, currentThickness);
			patternToken = `kjbg_${currentType}_${currentSize}_${currentThickness}_${currentFg}_${currentBg}`;
			tile.name = patternToken;

			canvas.background_image = patternToken;
			canvas.clear_background_color = currentBg;
			canvas._bg_img = tile;
			canvas._pattern = undefined;
			canvas.setDirty(true, true);
		}

		applyToLiteGraph();

		// Enforce our values if palette service overwrites them.
		// Returns false — LiteGraph's native code does all drawing.
		const origCallback = canvas.onRenderBackground;
		canvas.onRenderBackground = function (cvs, ctx) {
			if (origCallback) {
				const result = origCallback.call(this, cvs, ctx);
				if (result) return true;
			}

			applyToLiteGraph();

			// If palette service overwrote our values, mark dirty for next frame.
			if (!dirty && patternToken !== null && this.background_image !== patternToken) {
				dirty = true;
			}

			return false;
		};
	},
});
