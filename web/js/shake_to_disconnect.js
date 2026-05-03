const { app } = window.comfyAPI.app;

const state = {
	pointerDown: false,
	lastDelta: [0, 0],
	lastPos: null,
	reversalTimes: [],
	lastScanTime: 0,
	triggered: false,
};

// Cached from each setting's onChange to avoid per-pointermove lookups.
const settings = {
	enabled: true,
	reversalsNeeded: 3,
};

// Top-left, not center: center shifts during resize, pos only moves on drag.
function getNodeAnchor(node) {
	return [node.pos[0], node.pos[1]];
}

function getDraggedNodes(lgCanvas) {
	const items = lgCanvas.selectedItems;
	if (!items || items.size === 0) return [];
	return [...items].filter(n => n.inputs || n.outputs);
}

function nodeHasLinks(node) {
	if (node.inputs?.some(i => i.link != null)) return true;
	if (node.outputs?.some(o => o.links?.length > 0)) return true;
	return false;
}

// Same end result as deleting all selected nodes except internal links between selected nodes are preserved.
function executeShakeBreak(draggedNodes, graph, lgCanvas) {
	if (draggedNodes.length === 0) return;

	const selectedIds = new Set(draggedNodes.map(n => n.id));

	lgCanvas.emitBeforeChange?.();
	graph.beforeChange?.();

	try {
		for (const node of draggedNodes) bypassExternal(node, selectedIds, graph);
		for (const node of draggedNodes) disconnectExternal(node, selectedIds, graph);
	} finally {
		graph.afterChange?.();
		lgCanvas.emitAfterChange?.();
	}

	lgCanvas.setDirty(true, true);

	state.triggered = true;
	state.reversalTimes = [];
}

function bypassExternal(node, selectedIds, graph) {
	const inputs = node.inputs || [];
	for (let i = 0; i < inputs.length; i++) {
		if (inputs[i].link == null) continue;
		const inLink = graph.getLink(inputs[i].link);
		if (!inLink || selectedIds.has(inLink.origin_id)) continue;
		const inNode = graph.getNodeById(inLink.origin_id);
		if (!inNode) continue;

		traceAndBypass(node, i, inNode, inLink, selectedIds, graph, new Set());
	}
}

// At each step, follows output[slot] (matching the input slot we arrived on).
// Internal targets recurse; external targets get a direct upstream → target link.
function traceAndBypass(currentNode, slot, inNode, inLink, selectedIds, graph, visited) {
	const key = `${currentNode.id}:${slot}`;
	if (visited.has(key)) return;
	visited.add(key);

	const output = currentNode.outputs?.[slot];
	if (!output?.links?.length) return;

	for (const outLinkId of [...output.links]) {
		const outLink = graph.getLink(outLinkId);
		if (!outLink) continue;

		if (selectedIds.has(outLink.target_id)) {
			const nextNode = graph.getNodeById(outLink.target_id);
			if (nextNode) {
				traceAndBypass(nextNode, outLink.target_slot, inNode, inLink, selectedIds, graph, visited);
			}
		} else {
			const outNode = graph.getNodeById(outLink.target_id);
			if (outNode) {
				inNode.connect(inLink.origin_slot, outNode, outLink.target_slot, inLink.parentId);
			}
		}
	}
}

function disconnectExternal(node, selectedIds, graph) {
	if (node.inputs) {
		for (let i = 0; i < node.inputs.length; i++) {
			const linkId = node.inputs[i].link;
			if (linkId == null) continue;
			const link = graph.getLink(linkId);
			if (!link || selectedIds.has(link.origin_id)) continue;
			node.disconnectInput(i, true);
		}
	}
	if (node.outputs) {
		for (let o = 0; o < node.outputs.length; o++) {
			const links = node.outputs[o].links;
			if (!links?.length) continue;
			for (const linkId of [...links]) {
				const link = graph.getLink(linkId);
				if (!link || selectedIds.has(link.target_id)) continue;
				const targetNode = graph.getNodeById(link.target_id);
				if (targetNode) targetNode.disconnectInput(link.target_slot, true);
			}
		}
	}
}

function clearState() {
	state.pointerDown = false;
	state.lastDelta = [0, 0];
	state.lastPos = null;
	state.reversalTimes = [];
	state.lastScanTime = 0;
	state.triggered = false;
}

app.registerExtension({
	name: "KJNodes.ShakeToDisconnect",

	settings: [
		{
			id: "KJNodes.shakeToDisconnect",
			name: "Enable shake to disconnect",
			category: ["KJNodes", "Shake to Disconnect", "Enabled"],
			tooltip: "Shake a selected node back and forth quickly to remove its links. Pass-through bypass is applied (same as deleting the node); when multiple nodes are selected, links between them are preserved and bypass propagates through the chain.",
			type: "boolean",
			defaultValue: false,
			onChange: (v) => { settings.enabled = !!v; },
		},
		{
			id: "KJNodes.shakeReversals",
			name: "Reversals required",
			category: ["KJNodes", "Shake to Disconnect", "Reversals required"],
			tooltip: "Number of back-and-forth direction reversals needed to trigger disconnect. Lower = more sensitive (triggers from a small wiggle), higher = needs a more deliberate shake.",
			type: "slider",
			defaultValue: 3,
			attrs: { min: 2, max: 6, step: 1 },
			onChange: (v) => { settings.reversalsNeeded = Number(v) || 3; },
		},
	],

	setup() {
		const lgCanvas = app.canvas;
		const canvasEl = lgCanvas.canvas;

		document.addEventListener("pointerdown", (e) => {
			if (e.button !== 0) return;
			// Legacy mode targets the canvas; Vue mode targets a node DOM element.
			const onCanvas = e.target === canvasEl;
			const onVueNode = e.target?.closest?.("[data-node-id]");
			if (!onCanvas && !onVueNode) return;

			state.pointerDown = true;
			state.lastDelta = [0, 0];
			state.lastPos = null;
			state.reversalTimes = [];
			state.triggered = false;
		}, true);

		document.addEventListener("pointermove", () => {
			if (!state.pointerDown) return;
			if (state.triggered) return;
			if (lgCanvas.connecting_links?.length) return;
			if (lgCanvas.resizing_node) return;
			if (lgCanvas.node_widget) return;

			if (!settings.enabled) return;

			const graph = lgCanvas.graph || app.graph;
			if (!graph) return;

			const now = performance.now();
			const nodeCount = graph._nodes?.length ?? 0;
			const throttle = nodeCount > 200 ? 50 : nodeCount > 100 ? 32 : 16;
			if (now - state.lastScanTime < throttle) return;
			state.lastScanTime = now;

			const draggedNodes = getDraggedNodes(lgCanvas);
			if (draggedNodes.length === 0) return;

			const curPos = getNodeAnchor(draggedNodes[0]);
			if (state.lastPos) {
				// Thresholds are in screen-pixels; deltas are in graph-coords, so scale.
				const scale = lgCanvas.ds?.scale ?? 1;
				const sdx = (curPos[0] - state.lastPos[0]) * scale;
				const sdy = (curPos[1] - state.lastPos[1]) * scale;
				const dot = sdx * state.lastDelta[0] + sdy * state.lastDelta[1];
				const magnitude = Math.sqrt(sdx * sdx + sdy * sdy);
				if (dot < 0 && magnitude > 5) state.reversalTimes.push(now);
				if (magnitude > 2) state.lastDelta = [sdx, sdy];
			}
			state.lastPos = curPos;
			const cutoff = now - 600;
			while (state.reversalTimes.length && state.reversalTimes[0] < cutoff) state.reversalTimes.shift();

			if (state.reversalTimes.length >= settings.reversalsNeeded && draggedNodes.some(nodeHasLinks)) {
				executeShakeBreak(draggedNodes, graph, lgCanvas);
			}
		}, true);

		document.addEventListener("pointerup", (e) => {
			if (e.button !== 0) return;
			clearState();
		}, true);
	},
});
