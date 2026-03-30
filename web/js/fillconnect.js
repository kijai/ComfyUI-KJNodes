const { app } = window.comfyAPI.app;
import { getSlotPos } from "./utility.js";

/**
 * Check type compatibility. Returns a match tier:
 *   2 = exact single-type match, 1 = multi-type or wildcard match, -1 = incompatible
 */
function typeMatchTier(outType, inType) {
	if (!LiteGraph.isValidConnection(outType, inType)) return -1;
	if (typeof outType === "string" && typeof inType === "string"
		&& !outType.includes(",") && !inType.includes(",")
		&& outType.toUpperCase() === inType.toUpperCase()) return 2;
	return 1;
}


/**
 * Order selected nodes for connection.
 * Respects existing links between them (topological sort),
 * then inserts unconnected nodes by spatial position (left-to-right, top-to-bottom).
 */
function orderNodes(nodes, graph) {
	const selectedIds = new Set(nodes.map(n => n.id));
	const nodeById = new Map(nodes.map(n => [n.id, n]));

	const outEdges = new Map();
	const inDegree = new Map();
	for (const n of nodes) {
		outEdges.set(n.id, new Set());
		inDegree.set(n.id, 0);
	}

	for (const node of nodes) {
		for (const out of (node.outputs || [])) {
			for (const linkId of (out.links || [])) {
				const link = graph.getLink(linkId);
				if (link && selectedIds.has(link.target_id) && link.target_id !== node.id) {
					if (!outEdges.get(node.id).has(link.target_id)) {
						outEdges.get(node.id).add(link.target_id);
						inDegree.set(link.target_id, inDegree.get(link.target_id) + 1);
					}
				}
			}
		}
	}

	const bySpatial = (a, b) => a.pos[0] - b.pos[0] || a.pos[1] - b.pos[1];
	const queue = nodes.filter(n => inDegree.get(n.id) === 0);
	queue.sort(bySpatial);

	const ordered = [];
	while (queue.length > 0) {
		const node = queue.shift();
		ordered.push(node);
		for (const targetId of outEdges.get(node.id)) {
			const deg = inDegree.get(targetId) - 1;
			inDegree.set(targetId, deg);
			if (deg === 0 && nodeById.has(targetId)) {
				queue.push(nodeById.get(targetId));
				queue.sort(bySpatial);
			}
		}
	}

	// Cycle fallback: append remaining by position
	if (ordered.length < nodes.length) {
		const inOrdered = new Set(ordered.map(n => n.id));
		const remaining = nodes.filter(n => !inOrdered.has(n.id));
		remaining.sort(bySpatial);
		ordered.push(...remaining);
	}

	return ordered;
}

/**
 * Collect all candidate connections and assign globally by closest 2D distance.
 * Each input and output is used at most once per invocation.
 */
function planConnections(ordered) {
	// Collect all candidates globally across all target nodes
	const candidates = [];

	for (let b = 1; b < ordered.length; b++) {
		const nodeB = ordered[b];
		if (!nodeB.inputs) continue;

		for (let inIdx = 0; inIdx < nodeB.inputs.length; inIdx++) {
			const inp = nodeB.inputs[inIdx];
			if (inp.link != null) continue;

			const inPos = getSlotPos(nodeB, true, inIdx);
			const inputCandidates = [];
			let hasExact = false;

			for (let a = b - 1; a >= 0; a--) {
				const nodeA = ordered[a];
				if (!nodeA.outputs) continue;

				for (let outIdx = 0; outIdx < nodeA.outputs.length; outIdx++) {
					const tier = typeMatchTier(nodeA.outputs[outIdx].type, inp.type);
					if (tier < 0) continue;
					if (tier === 2) hasExact = true;

					const outPos = getSlotPos(nodeA, false, outIdx);
					const dx = outPos[0] - inPos[0];
					const dy = outPos[1] - inPos[1];
					inputCandidates.push({
						targetNode: nodeB,
						inIdx,
						sourceNode: nodeA,
						outIdx,
						tier,
						dist: dx * dx + dy * dy,
					});
				}
			}

			for (const c of inputCandidates) {
				c.hasExact = hasExact;
			}
			candidates.push(...inputCandidates);
		}
	}

	// Filter out non-exact matches for inputs that have an exact candidate
	const filtered = candidates.filter(c => c.tier === 2 || !c.hasExact);

	// Sort globally: higher tier first, then closest pair
	filtered.sort((x, y) => (y.tier - x.tier) || (x.dist - y.dist));

	// Greedy global assignment — closest pairs first
	const planned = [];
	const usedInputs = new Set();
	const usedOutputs = new Set();

	for (const c of filtered) {
		const inKey = `${c.targetNode.id}:${c.inIdx}`;
		if (usedInputs.has(inKey)) continue;
		const outKey = `${c.sourceNode.id}:${c.outIdx}`;
		if (usedOutputs.has(outKey)) continue;

		planned.push({
			sourceNode: c.sourceNode,
			outIdx: c.outIdx,
			targetNode: c.targetNode,
			inIdx: c.inIdx,
		});
		usedInputs.add(inKey);
		usedOutputs.add(outKey);
	}

	return planned;
}

function connectPlanned(planned, graph) {
	for (const p of planned) {
		p.sourceNode.connect(p.outIdx, p.targetNode, p.inIdx);
	}
	graph.change();
}

function fillConnectSelected() {
	if (!app.ui.settings.getSettingValue("KJNodes.fillConnectEnabled")) return;

	const canvas = app.canvas;
	const graph = canvas.graph;
	const nodes = Object.values(canvas.selected_nodes || {});
	if (nodes.length < 2) return;

	const ordered = orderNodes(nodes, graph);
	const planned = planConnections(ordered);

	if (planned.length > 0) {
		connectPlanned(planned, graph);
	}

	if (app.extensionManager?.toast) {
		app.extensionManager.toast.add({
			severity: planned.length > 0 ? "info" : "warn",
			summary: planned.length > 0
				? `Connected ${planned.length} link${planned.length > 1 ? "s" : ""}`
				: "No compatible connections found",
			life: 2000,
		});
	}
}

app.registerExtension({
	name: "KJNodes.FillConnect",

	settings: [
		{
			id: "KJNodes.fillConnectEnabled",
			name: "Enable Fill-Connect hotkey",
			category: ["KJNodes", "Fill-Connect"],
			tooltip: "Fill-connect selected nodes by compatible types",
			type: "boolean",
			defaultValue: true,
		},
	],

	commands: [
		{
			id: "KJNodes.fillConnectSelected",
			label: "Fill-Connect Selected Nodes",
			function: fillConnectSelected,
		},
	],

	keybindings: [
		{
			commandId: "KJNodes.fillConnectSelected",
			combo: { key: "f" },
			targetElementId: "graph-canvas",
		},
	],
});
