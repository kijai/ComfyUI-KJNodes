const { app } = window.comfyAPI.app;

// Nodes 2.0 forwards wheel to the canvas (TransformPane capture listener) unless the widget is focused; this lets overflowing widgets scroll on hover like legacy
const NODE = '[data-testid="transform-pane"] [data-node-id]';
let enabled = false;

const scrollsY = (el) => {
	if (el.scrollHeight <= el.clientHeight) return false;
	const overflow = getComputedStyle(el).overflowY;
	return overflow === "auto" || overflow === "scroll" || overflow === "overlay";
};

document.addEventListener("wheel", (e) => {
	if (!enabled || e.ctrlKey || e.metaKey || e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
	const node = e.target.closest?.(NODE);
	if (!node || e.target.closest('[data-capture-wheel="true"]')?.contains(document.activeElement)) return;
	for (let el = e.target; el && el !== node; el = el.parentElement) {
		if (!scrollsY(el)) continue;
		// no preventDefault: the browser scrolls it, the frontend's forward-to-canvas never runs
		e.stopPropagation();
		return;
	}
}, { capture: true, passive: true });

app.registerExtension({
	name: "KJNodes.VueWheel",
	settings: [
		{
			id: "KJNodes.vueWheel.scrollUnfocused",
			name: "Wheel scrolls unfocused widgets",
			category: ["KJNodes", "Nodes 2.0", "Wheel scrolls unfocused widgets"],
			tooltip: "Mouse wheel over a Nodes 2.0 text box or other widget with overflowing content scrolls it without clicking it first, like the legacy canvas. Elsewhere, and with Ctrl/Shift or horizontal swipes, the wheel still zooms/pans the canvas.",
			type: "boolean",
			defaultValue: false,
			onChange: (value) => (enabled = !!value),
		},
	],
});
