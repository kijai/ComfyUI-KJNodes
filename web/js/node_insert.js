const { app } = window.comfyAPI.app;
import { getSlotPos, clientToCanvas, getNodeAtPoint, typesCompatible, chainCallback } from "./utility.js";

// Max age of the last keydown for it to count as the command's trigger.
// Beyond this, treat the command as menu-fired (no release tracking).
const KEYDOWN_MAX_AGE_MS = 100;

const state = {
	pointerDown: false,
	hasMoved: false,
	insertKeyDown: false,
	activationKey: null,
	draggedNode: null,
	startPos: null,
	insertTargetLink: null,
	insertSlots: null,
	insertOriginNode: null,
	insertDestNode: null,
	lastScanTime: 0,
	animating: false,
};

let lastKeyDown = null;
let lastKeyDownTime = 0;

function bezierAt(p0, p1, p2, p3, t) {
	const u = 1 - t;
	const uu = u * u, uuu = uu * u;
	const tt = t * t, ttt = tt * t;
	return [
		uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0],
		uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1],
	];
}

// Horizontal control-point offset for the link bezier; must match what
// LiteGraph draws so the hit-test region matches the visible curve.
const bezierOffsetX = (from, to) => Math.max(Math.abs(to[0] - from[0]) * 0.5, 50);

function findLinkUnderNode(graph, draggedNode) {
	const bounds = draggedNode.getBounding?.();
	const nodeX = bounds ? bounds[0] : draggedNode.pos[0];
	const nodeY = bounds ? bounds[1] : draggedNode.pos[1];
	const nodeW = bounds ? bounds[2] : (draggedNode.size[0] || 100);
	const nodeH = bounds ? bounds[3] : (draggedNode.size[1] || 60);
	const nodeCx = nodeX + nodeW / 2;
	const nodeCy = nodeY + nodeH / 2;

	let bestLink = null;
	let bestDist = Infinity;

	const links = graph.links;
	if (!links) return null;

	for (const link of links.values()) {
		if (!link) continue;
		if (link.origin_id === draggedNode.id || link.target_id === draggedNode.id) continue;

		const originNode = graph.getNodeById(link.origin_id);
		const targetNode = graph.getNodeById(link.target_id);
		if (!originNode || !targetNode) continue;

		// Reject on node bounds before resolving slot positions — the latter
		// hits the DOM in Vue mode and is the expensive step.
		const oW = originNode.size?.[0] || 100, oH = originNode.size?.[1] || 60;
		const tW = targetNode.size?.[0] || 100, tH = targetNode.size?.[1] || 60;
		const cMinX = Math.min(originNode.pos[0], targetNode.pos[0]);
		const cMaxX = Math.max(originNode.pos[0] + oW, targetNode.pos[0] + tW);
		const cMinY = Math.min(originNode.pos[1], targetNode.pos[1]) - 50;
		const cMaxY = Math.max(originNode.pos[1] + oH, targetNode.pos[1] + tH) + 50;
		if (cMaxX < nodeX || cMinX > nodeX + nodeW || cMaxY < nodeY || cMinY > nodeY + nodeH) continue;

		const outPos = getSlotPos(originNode, false, link.origin_slot);
		const inPos = getSlotPos(targetNode, true, link.target_slot);

		const lMinX = Math.min(outPos[0], inPos[0]);
		const lMaxX = Math.max(outPos[0], inPos[0]);
		const lMinY = Math.min(outPos[1], inPos[1]) - 50; // bezier can sag
		const lMaxY = Math.max(outPos[1], inPos[1]) + 50;
		if (lMaxX < nodeX || lMinX > nodeX + nodeW || lMaxY < nodeY || lMinY > nodeY + nodeH) continue;

		const offsetX = bezierOffsetX(outPos, inPos);
		const p0 = outPos;
		const p1 = [outPos[0] + offsetX, outPos[1]];
		const p2 = [inPos[0] - offsetX, inPos[1]];
		const p3 = inPos;

		for (let i = 0; i <= 20; i++) {
			const pt = bezierAt(p0, p1, p2, p3, i / 20);
			if (pt[0] >= nodeX && pt[0] <= nodeX + nodeW && pt[1] >= nodeY && pt[1] <= nodeY + nodeH) {
				const d = Math.hypot(pt[0] - nodeCx, pt[1] - nodeCy);
				if (d < bestDist) {
					bestDist = d;
					bestLink = link;
				}
				break;
			}
		}
	}
	return bestLink;
}

function findInsertSlots(node, linkType) {
	const inputSlot = node.inputs?.findIndex(i => i.link == null && typesCompatible(linkType, i.type)) ?? -1;
	const outputSlot = node.outputs?.findIndex(o => typesCompatible(linkType, o.type)) ?? -1;
	if (inputSlot === -1 || outputSlot === -1) return null;
	return { inputSlot, outputSlot };
}

function executeNodeInsert(canvas, node, link) {
	const graph = canvas.graph || app.graph;

	const originNode = graph.getNodeById(link.origin_id);
	const targetNode = graph.getNodeById(link.target_id);
	if (!originNode || !targetNode) return;

	const slots = findInsertSlots(node, link.type);
	if (!slots) return;

	const originSlot = link.origin_slot;
	const targetSlot = link.target_slot;

	targetNode.disconnectInput(targetSlot);
	originNode.connect(originSlot, node, slots.inputSlot);
	node.connect(slots.outputSlot, targetNode, targetSlot);

	graph.change();
	canvas.setDirty(true, true);
}

function setInsertTarget(link, graph, slots = null) {
	if (link === state.insertTargetLink) return;
	// `_dragging` is LiteGraph's own runtime flag for hiding a link mid-rewire.
	// Not serialized, so save/copy/undo stay safe.
	if (state.insertTargetLink) delete state.insertTargetLink._dragging;
	if (link) {
		link._dragging = true;
		state.insertSlots = slots ?? findInsertSlots(state.draggedNode, link.type);
		state.insertOriginNode = graph.getNodeById(link.origin_id);
		state.insertDestNode = graph.getNodeById(link.target_id);
	} else {
		state.insertSlots = null;
		state.insertOriginNode = null;
		state.insertDestNode = null;
	}
	state.insertTargetLink = link;
}

function getTypeColor(lgCanvas, slotType) {
	if (slotType && LGraphCanvas.link_type_colors && LGraphCanvas.link_type_colors[slotType]) {
		return LGraphCanvas.link_type_colors[slotType];
	}
	return lgCanvas.default_link_color || "#AAA";
}

function drawGhostLink(ctx, from, to, color, alpha, drawTime) {
	const offsetX = bezierOffsetX(from, to);

	ctx.beginPath();
	ctx.moveTo(from[0], from[1]);
	ctx.bezierCurveTo(
		from[0] + offsetX, from[1],
		to[0] - offsetX, to[1],
		to[0], to[1],
	);

	ctx.strokeStyle = color;
	ctx.globalAlpha = alpha;
	ctx.lineWidth = 2.5;
	ctx.setLineDash([8, 4]);
	ctx.lineDashOffset = -(drawTime ?? performance.now()) / 50;

	ctx.stroke();
	ctx.setLineDash([]);
	ctx.globalAlpha = 1;
}

function drawSlotHighlight(ctx, pos, color, alpha) {
	ctx.beginPath();
	ctx.arc(pos[0], pos[1], 6, 0, Math.PI * 2);
	ctx.globalAlpha = alpha;
	ctx.fillStyle = color;
	ctx.fill();
	ctx.globalAlpha = 1;
}

function drawGhostSegment(ctx, from, to, color, now) {
	drawGhostLink(ctx, from, to, color, 0.8, now);
	drawSlotHighlight(ctx, from, color, 0.6);
	drawSlotHighlight(ctx, to, color, 0.6);
}

function drawInsertPreview(ctx, lgCanvas) {
	const { insertTargetLink: link, insertSlots: slots, insertOriginNode, insertDestNode, draggedNode } = state;
	if (!link || !slots || !insertOriginNode || !insertDestNode || !draggedNode) return;

	const now = performance.now();
	const color = getTypeColor(lgCanvas, link.type);

	drawGhostSegment(ctx,
		getSlotPos(insertOriginNode, false, link.origin_slot),
		getSlotPos(draggedNode, true, slots.inputSlot),
		color, now);
	drawGhostSegment(ctx,
		getSlotPos(draggedNode, false, slots.outputSlot),
		getSlotPos(insertDestNode, true, link.target_slot),
		color, now);
}

function startAnimLoop(lgCanvas) {
	if (state.animating) return;
	state.animating = true;

	function tick() {
		const link = state.insertTargetLink;
		if (!link) {
			state.animating = false;
			lgCanvas.setDirty(true, true);
			return;
		}
		// If the link was removed externally (e.g. undo) mid-preview, our
		// `_dragging` flag has nothing to land on — drop the preview.
		const graph = lgCanvas.graph || app.graph;
		const stored = graph?.links?.get(link.id);
		if (stored !== link) {
			setInsertTarget(null);
			state.animating = false;
			lgCanvas.setDirty(true, true);
			return;
		}
		// Steady-state: only the dash offset changes — front canvas only.
		lgCanvas.setDirty(true, false);
		requestAnimationFrame(tick);
	}
	requestAnimationFrame(tick);
}

function clearState() {
	setInsertTarget(null);
	state.pointerDown = false;
	state.hasMoved = false;
	state.draggedNode = null;
	state.startPos = null;
}

app.registerExtension({
	name: "KJNodes.NodeInsert",

	settings: [
		{
			id: "KJNodes.nodeInsertMode",
			name: "Node insert activation",
			category: ["KJNodes", "Node Insert", "Activation mode"],
			tooltip: "Always: dragging a compatible node onto a link previews insertion. Hotkey: only while the hotkey (default: D) is held. Disabled: feature off.",
			type: "combo",
			defaultValue: "hotkey",
			options: ["always", "hotkey", "disabled"],
		},
	],

	commands: [
		{
			id: "KJNodes.NodeInsertMode",
			label: "Node insert mode (hold to activate)",
			active: () => state.insertKeyDown,
			function: () => {
				state.insertKeyDown = true;
				// Snapshot the key that triggered this so any rebind works.
				state.activationKey = (performance.now() - lastKeyDownTime < KEYDOWN_MAX_AGE_MS)
					? lastKeyDown
					: null;
				if (state.draggedNode) {
					const graph = app.canvas?.graph || app.graph;
					if (graph) {
						const link = findLinkUnderNode(graph, state.draggedNode);
						if (link && findInsertSlots(state.draggedNode, link.type)) {
							setInsertTarget(link, graph);
							startAnimLoop(app.canvas);
						}
					}
				}
			},
		},
	],

	keybindings: [
		{
			commandId: "KJNodes.NodeInsertMode",
			// Default only — release detection captures the actually-triggering
			// key at command-fire time, so user rebinds work.
			combo: { key: "d" },
			targetElementId: "graph-canvas",
		},
	],

	async setup() {
		const lgCanvas = app.canvas;
		const canvasEl = lgCanvas.canvas;

		// Capture phase so `lastKeyDown` is set before ComfyUI fires the command.
		document.addEventListener("keydown", (e) => {
			if (e.repeat) return;
			lastKeyDown = e.key?.toLowerCase() ?? null;
			lastKeyDownTime = performance.now();
		}, true);

		document.addEventListener("keyup", (e) => {
			const key = e.key?.toLowerCase() ?? null;
			// Released key can't be a future trigger — invalidate so a stale
			// keydown can't be picked up by a later menu activation.
			if (key && key === lastKeyDown) lastKeyDownTime = 0;

			if (!state.insertKeyDown) return;
			// No activation key (menu-fired) → blur/pointercancel will clean up.
			if (!state.activationKey || key !== state.activationKey) return;
			state.insertKeyDown = false;
			state.activationKey = null;
			if (state.insertTargetLink) {
				setInsertTarget(null);
				lgCanvas.setDirty(false, true);
			}
		});

		// Releases outside the window never reach our keyup/pointerup listeners,
		// so without these the `_dragging` flag and hotkey state can stick.
		const dropTransient = () => {
			state.insertKeyDown = false;
			state.activationKey = null;
			clearState();
			lgCanvas.setDirty(false, true);
		};
		window.addEventListener("blur", dropTransient);
		document.addEventListener("visibilitychange", () => {
			if (document.hidden) dropTransient();
		});
		document.addEventListener("pointercancel", dropTransient, true);

		// Scope the Vue-mode node lookup to the graph canvas's container so
		// stray `[data-node-id]` elements elsewhere can't activate this flow.
		const graphRoot = canvasEl.parentElement;
		document.addEventListener("pointerdown", (e) => {
			if (e.button !== 0) return;
			const onCanvas = e.target === canvasEl;
			const vueNodeEl = graphRoot?.contains(e.target)
				? e.target?.closest?.("[data-node-id]")
				: null;
			if (!onCanvas && !vueNodeEl) return;

			state.pointerDown = true;
			state.hasMoved = false;

			const graph = lgCanvas.graph || app.graph;
			if (graph) {
				let node = null;
				if (vueNodeEl) {
					const nodeId = parseInt(vueNodeEl.getAttribute("data-node-id"));
					if (Number.isFinite(nodeId)) node = graph.getNodeById(nodeId);
				} else {
					const [cx, cy] = clientToCanvas(lgCanvas, e.clientX, e.clientY);
					node = getNodeAtPoint(graph, cx, cy);
				}
				if (node && (node.inputs || node.outputs)) {
					state.draggedNode = node;
					state.startPos = node.pos ? [node.pos[0], node.pos[1]] : null;
				}
			}
		}, true);

		document.addEventListener("pointermove", () => {
			if (!state.pointerDown) return;
			if (lgCanvas.connecting_links?.length) return;
			if (lgCanvas.resizing_node) return;
			if (lgCanvas.node_widget) return;

			const mode = app.ui.settings.getSettingValue("KJNodes.nodeInsertMode") ?? "always";
			if (mode === "disabled") return;

			state.hasMoved = true;

			const active = mode === "always" || state.insertKeyDown;
			if (!active) {
				setInsertTarget(null);
				return;
			}

			if (!state.draggedNode) return;

			// Multi-node drags move several nodes together — inserting one of
			// them into a link is almost never what the user wants.
			const selected = lgCanvas.selected_nodes;
			if (selected) {
				const count = selected instanceof Map ? selected.size : Object.keys(selected).length;
				if (count > 1) {
					setInsertTarget(null);
					return;
				}
			}

			// Vue mode: pointermove fires on the document during widget drags
			// (sliders etc.) too — require the node itself to have moved.
			const np = state.draggedNode.pos;
			if (!np || !state.startPos
				|| (np[0] === state.startPos[0] && np[1] === state.startPos[1])) {
				setInsertTarget(null);
				return;
			}

			const graph = lgCanvas.graph || app.graph;
			if (!graph) return;

			const now = performance.now();
			const nodeCount = graph._nodes?.length ?? 0;
			const throttle = nodeCount > 200 ? 50 : nodeCount > 100 ? 32 : 16;
			if (now - state.lastScanTime < throttle) return;
			state.lastScanTime = now;

			const link = findLinkUnderNode(graph, state.draggedNode);
			const slots = link ? findInsertSlots(state.draggedNode, link.type) : null;
			const wasIdle = !state.insertTargetLink;

			if (link && slots) {
				setInsertTarget(link, graph, slots);
				if (wasIdle) startAnimLoop(lgCanvas);
			} else {
				setInsertTarget(null);
			}
		}, true);

		document.addEventListener("pointerup", (e) => {
			if (e.button !== 0) return;
			if (!state.pointerDown) return;

			if (state.insertTargetLink && state.draggedNode && state.hasMoved
				&& !state.draggedNode.flags?.pinned) {
				delete state.insertTargetLink._dragging;
				executeNodeInsert(lgCanvas, state.draggedNode, state.insertTargetLink);
			}

			clearState();
		}, true);

		// onDrawForeground: ghost paints over the existing (hidden) link.
		chainCallback(lgCanvas, "onDrawForeground", function (ctx) {
			if (state.insertTargetLink) {
				drawInsertPreview(ctx, lgCanvas);
			}
		});
	},
});
