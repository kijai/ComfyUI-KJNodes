const { app } = window.comfyAPI.app;
import { typesCompatible, clientToCanvas, getNodeAtPoint } from "./utility.js";

let swapTargetNode = null;
let swapDraggedNode = null;
let swapDragStartPos = null;
let swapAnimating = false;
let swapHasMoved = false;
let swapKeyDown = false;

/** Find the topmost node whose bounding box overlaps with draggedNode */
function getOverlappingNode(graph, draggedNode) {
	const ax = draggedNode.pos[0];
	const ay = draggedNode.pos[1];
	const aw = draggedNode.size[0] || 100;
	const ah = draggedNode.size[1] || 60;

	for (let i = graph._nodes.length - 1; i >= 0; i--) {
		const n = graph._nodes[i];
		if (n === draggedNode) continue;
		const bx = n.pos[0];
		const by = n.pos[1];
		if (ax < bx + (n.size[0] || 100) && ax + aw > bx &&
			ay < by + (n.size[1] || 60) && ay + ah > by) {
			return n;
		}
	}
	return null;
}


function startHighlightAnim(lgCanvas) {
	if (swapAnimating) return;
	swapAnimating = true;

	function tick() {
		if (!swapTargetNode) {
			swapAnimating = false;
			return;
		}
		lgCanvas.setDirty(false, true);
		requestAnimationFrame(tick);
	}
	requestAnimationFrame(tick);
}

function clearSwapState() {
	swapTargetNode = null;
	swapDraggedNode = null;
	swapDragStartPos = null;
	swapHasMoved = false;
}

function snapshotConnections(graph, node) {
	const inputs = [];
	for (let i = 0; i < (node.inputs?.length ?? 0); i++) {
		const inp = node.inputs[i];
		if (inp.link == null) continue;
		const link = graph.getLink(inp.link);
		if (!link) continue;
		inputs.push({
			slotIndex: i,
			type: inp.type,
			originNodeId: link.origin_id,
			originSlot: link.origin_slot,
		});
	}

	const outputs = [];
	for (let o = 0; o < (node.outputs?.length ?? 0); o++) {
		const out = node.outputs[o];
		if (!out.links?.length) continue;
		const targets = out.links
			.map((linkId) => graph.getLink(linkId))
			.filter(Boolean)
			.map((link) => ({ targetNodeId: link.target_id, targetSlot: link.target_slot }));
		if (targets.length > 0) {
			outputs.push({ slotIndex: o, type: out.type, targets });
		}
	}

	return { inputs, outputs };
}

function disconnectAll(node) {
	if (node.inputs) {
		for (let i = node.inputs.length - 1; i >= 0; i--) {
			if (node.inputs[i].link != null) node.disconnectInput(i);
		}
	}
	if (node.outputs) {
		for (let o = node.outputs.length - 1; o >= 0; o--) {
			if (node.outputs[o].links?.length) node.disconnectOutput(o);
		}
	}
}

// Find a compatible slot on node, preferring sameIndex. Returns -1 if none found.
// For inputs, isAvailable checks link == null and usedInputs; for outputs, always true.
function findCompatibleSlot(slots, type, sameIndex, isAvailable) {
	if (!slots) return -1;
	if (sameIndex < slots.length && isAvailable(sameIndex) && typesCompatible(type, slots[sameIndex].type)) {
		return sameIndex;
	}
	for (let s = 0; s < slots.length; s++) {
		if (isAvailable(s) && typesCompatible(type, slots[s].type)) return s;
	}
	return -1;
}

// Reconnect external connections from a snapshot onto newNode.
// Tries same-slot first, then scans for a compatible free slot by type.
function reconnectExternal(graph, snapshot, newNode, otherNodeId) {
	const usedInputs = new Set();
	for (const inp of snapshot.inputs) {
		if (inp.originNodeId === otherNodeId) continue;
		const originNode = graph.getNodeById(inp.originNodeId);
		if (!originNode) continue;
		const slot = findCompatibleSlot(
			newNode.inputs, inp.type, inp.slotIndex,
			(s) => !usedInputs.has(s) && newNode.inputs[s].link == null,
		);
		if (slot === -1) continue;
		originNode.connect(inp.originSlot, newNode, slot);
		usedInputs.add(slot);
	}

	for (const out of snapshot.outputs) {
		const bestSlot = findCompatibleSlot(
			newNode.outputs, out.type, out.slotIndex,
			() => true,
		);
		if (bestSlot === -1) continue;
		for (const tgt of out.targets) {
			if (tgt.targetNodeId === otherNodeId) continue;
			const targetNode = graph.getNodeById(tgt.targetNodeId);
			if (!targetNode) continue;
			newNode.connect(bestSlot, targetNode, tgt.targetSlot);
		}
	}
}

// Reconnect links that were between the two swapped nodes in one direction.
// A→B becomes B→A, if slot types are compatible.
function reconnectInternalOneDirection(snap, fromNodeId, srcNode, dstNode) {
	for (const out of snap.outputs) {
		for (const tgt of out.targets) {
			if (tgt.targetNodeId !== fromNodeId) continue;
			if (!srcNode.outputs || out.slotIndex >= srcNode.outputs.length) continue;
			if (!dstNode.inputs || tgt.targetSlot >= dstNode.inputs.length) continue;
			if (!typesCompatible(srcNode.outputs[out.slotIndex].type, dstNode.inputs[tgt.targetSlot].type)) continue;
			srcNode.connect(out.slotIndex, dstNode, tgt.targetSlot);
		}
	}
}

function getNodeElement(nodeId) {
	return document.querySelector(`[data-node-id="${nodeId}"]`);
}

function animateSwapVue(nodeA, toA, nodeB, toB, duration) {
	const elA = getNodeElement(nodeA.id);
	const elB = getNodeElement(nodeB.id);
	const transition = `transform ${duration}ms cubic-bezier(0.65, 0, 0.35, 1)`;

	if (elA) elA.style.transition = transition;
	if (elB) elB.style.transition = transition;

	nodeA.pos = [toA[0], toA[1]];
	nodeB.pos = [toB[0], toB[1]];
	app.canvas.setDirty(true, true);

	setTimeout(() => {
		if (elA) elA.style.transition = "";
		if (elB) elB.style.transition = "";
	}, duration + 16);
}

function animateSwapCanvas(nodeA, fromA, toA, nodeB, fromB, toB, duration) {
	const start = performance.now();

	function ease(t) {
		return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
	}

	function frame(now) {
		const t = Math.min((now - start) / duration, 1);
		const e = ease(t);
		nodeA.pos = [
			fromA[0] + (toA[0] - fromA[0]) * e,
			fromA[1] + (toA[1] - fromA[1]) * e,
		];
		nodeB.pos = [
			fromB[0] + (toB[0] - fromB[0]) * e,
			fromB[1] + (toB[1] - fromB[1]) * e,
		];
		app.canvas.setDirty(true, true);
		if (t < 1) requestAnimationFrame(frame);
	}

	requestAnimationFrame(frame);
}

function executeNodeSwap(canvas, nodeA, nodeB, originalPosA) {
	const graph = canvas.graph || app.graph;

	const snapA = snapshotConnections(graph, nodeA);
	const snapB = snapshotConnections(graph, nodeB);

	const posA = originalPosA || [nodeA.pos[0], nodeA.pos[1]];
	const posB = [nodeB.pos[0], nodeB.pos[1]];

	disconnectAll(nodeA);
	disconnectAll(nodeB);

	reconnectExternal(graph, snapA, nodeB, nodeA.id);
	reconnectExternal(graph, snapB, nodeA, nodeB.id);

	reconnectInternalOneDirection(snapA, nodeB.id, nodeB, nodeA);
	reconnectInternalOneDirection(snapB, nodeA.id, nodeA, nodeB);

	if (LiteGraph.vueNodesMode) {
		animateSwapVue(nodeA, posB, nodeB, posA, 200);
	} else {
		animateSwapCanvas(nodeA, [nodeA.pos[0], nodeA.pos[1]], posB,
			nodeB, [nodeB.pos[0], nodeB.pos[1]], posA, 200);
	}

	graph.change();
	canvas.setDirty(true, true);
}

app.registerExtension({
	name: "KJNodes.NodeSwap",

	settings: [
		{
			id: "KJNodes.nodeSwapEnabled",
			name: "Enable node swap on drag",
			category: ["KJNodes", "Node Swap", "Enable"],
			tooltip: "Hold swap key (default: S, rebindable in Keybindings) and drag a node onto another to swap their positions and reconnect links",
			type: "boolean",
			defaultValue: true,
		},
	],
	commands: [
		{
			id: "KJNodes.ToggleNodeSwap",
			label: "Toggle node swap on drag",
			active: () => app.ui.settings.getSettingValue("KJNodes.nodeSwapEnabled"),
			function: () => {
				const cur = app.ui.settings.getSettingValue("KJNodes.nodeSwapEnabled");
				app.ui.settings.setSettingValue("KJNodes.nodeSwapEnabled", !cur);
			},
		},
		{
			id: "KJNodes.NodeSwapMode",
			label: "Node swap mode (hold to activate)",
			active: () => swapKeyDown,
			function: () => {
				swapKeyDown = true;
				// If already dragging a node, check for overlap immediately
				// (handles case where node is already on top of another when key is pressed)
				if (swapDraggedNode) {
					const graph = app.canvas?.graph || app.graph;
					if (graph) {
						swapTargetNode = getOverlappingNode(graph, swapDraggedNode);
						if (swapTargetNode) startHighlightAnim(app.canvas);
					}
				}
			},
		},
	],
	keybindings: [
		{
			commandId: "KJNodes.NodeSwapMode",
			combo: { key: "s" },
			targetElementId: "graph-canvas",
		},
	],

	async setup() {
		await new Promise((resolve) => {
			function check() {
				if (app.canvas) return resolve();
				requestAnimationFrame(check);
			}
			check();
		});

		const lgCanvas = app.canvas;
		const canvasEl = lgCanvas.canvas;

		// The command system handles keydown (sets swapKeyDown = true).
		// Any keyup clears it since the user must hold the key.
		document.addEventListener("keyup", () => {
			if (swapKeyDown) {
				swapKeyDown = false;
				if (swapTargetNode) {
					swapTargetNode = null;
					lgCanvas.setDirty(false, true);
				}
			}
		});

		// Pointermove/pointerup handlers — only attached while pointer is down.
		function onPointerMove() {
			swapHasMoved = true;

			if (!swapKeyDown) {
				if (swapTargetNode) {
					swapTargetNode = null;
					lgCanvas.setDirty(false, true);
				}
				return;
			}

			if (!swapDraggedNode) return;

			const graph = lgCanvas.graph || app.graph;
			if (!graph) return;

			const prev = swapTargetNode;
			swapTargetNode = getOverlappingNode(graph, swapDraggedNode);

			if (swapTargetNode && !prev) {
				startHighlightAnim(lgCanvas);
			}
		}

		function onPointerUp(e) {
			if (e.button !== 0) return;

			document.removeEventListener("pointermove", onPointerMove);
			document.removeEventListener("pointerup", onPointerUp, true);

			const targetNode = swapTargetNode;
			const draggedNode = swapDraggedNode;
			const startPos = swapDragStartPos;
			const hasMoved = swapHasMoved;
			clearSwapState();

			if (!hasMoved || !targetNode || !draggedNode || !swapKeyDown) return;

			if (!targetNode.flags?.pinned && !draggedNode.flags?.pinned) {
				executeNodeSwap(lgCanvas, draggedNode, targetNode, startPos);
			}
		}

		// Record clicked node and its original position.
		// Listen on document to catch events from both the canvas and Vue node overlays.
		const graphContainer = canvasEl.closest(".graph-canvas-container") || canvasEl.parentElement;
		document.addEventListener("pointerdown", (e) => {
			if (e.button !== 0) return;
			if (!app.ui.settings.getSettingValue("KJNodes.nodeSwapEnabled")) return;
			if (!graphContainer?.contains(e.target)) return;

			swapHasMoved = false;

			const graph = lgCanvas.graph || app.graph;
			if (!graph) return;

			const [cx, cy] = clientToCanvas(lgCanvas, e.clientX, e.clientY);
			const node = getNodeAtPoint(graph, cx, cy);

			if (node && (node.inputs || node.outputs)) {
				swapDraggedNode = node;
				swapDragStartPos = [node.pos[0], node.pos[1]];
				document.addEventListener("pointermove", onPointerMove);
				document.addEventListener("pointerup", onPointerUp, true);
			}
		}, true);

		// Draw pulsing glow outline on the swap target node
		const origOnDrawBg = lgCanvas.onDrawBackground;
		lgCanvas.onDrawBackground = function (ctx, ...args) {
			origOnDrawBg?.call(this, ctx, ...args);

			const target = swapTargetNode;
			if (!target) return;

			const [x, y] = target.pos;
			const w = target.size[0] || 100;
			const h = target.size[1] || 60;
			const titleH = LiteGraph.NODE_TITLE_HEIGHT || 20;
			const pulse = 0.5 + 0.3 * Math.sin(Date.now() / 200);
			const isPinned = target.flags?.pinned;
			const color = isPinned ? "255, 80, 80" : "100, 200, 255";
			const pad = 5;

			ctx.save();
			ctx.strokeStyle = `rgba(${color}, ${pulse})`;
			ctx.lineWidth = 3;
			ctx.shadowColor = `rgba(${color}, 0.8)`;
			ctx.shadowBlur = 15;
			ctx.beginPath();
			ctx.roundRect(x - pad, y - titleH - pad, w + pad * 2, h + titleH + pad * 2, 8);
			ctx.stroke();
			ctx.fillStyle = `rgba(${color}, ${isPinned ? 0.08 : 0.05})`;
			ctx.fill();
			ctx.restore();
		};
	},
});
