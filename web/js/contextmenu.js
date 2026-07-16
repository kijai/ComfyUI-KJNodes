const { app } = window.comfyAPI.app;

function addNode(name, nextTo, options) {
	options = { side: "left", select: true, shiftY: 0, shiftX: 0, ...(options || {}) };
	const node = LiteGraph.createNode(name);
	const graph = app.canvas?.graph || app.graph;
	graph.add(node);

	node.pos = [
		options.side === "left" ? nextTo.pos[0] - (node.size[0] + options.offset): nextTo.pos[0] + nextTo.size[0] + options.offset,
		nextTo.pos[1] + options.shiftY,
	];

	// Automatically connect nodes
	if (options.side === "left") {
		// New node on left: connect new node's first output to nextTo's first free input
		if (node.outputs && node.outputs.length > 0 && nextTo.inputs && nextTo.inputs.length > 0) {
			for (let i = 0; i < nextTo.inputs.length; i++) {
				if (!nextTo.inputs[i].link) {
					node.pos[1] += i * (node.size[1] + 32);
					node.connect(0, nextTo, i);
					break;
				}
			}
		}
	} else {
		// New node on right: connect nextTo's first free output to new node's first free input
		if (nextTo.outputs && nextTo.outputs.length > 0 && node.inputs && node.inputs.length > 0) {
			for (let o = 0; o < nextTo.outputs.length; o++) {
				if (!nextTo.outputs[o].links || nextTo.outputs[o].links.length === 0) {
					// Offset vertically by slot index so multiple Set nodes don't overlap
					node.pos[1] += o * (node.size[1] + 32);
					for (let i = 0; i < node.inputs.length; i++) {
						if (!node.inputs[i].link) {
							nextTo.connect(o, node, i);
							break;
						}
					}
					break;
				}
			}
		}
	}
	
	if (options.select) {
		app.canvas.selectNode(node, false);
	}
	return node;
}

// Expose addNode for use in setgetnodes.js
window.kjNodes = window.kjNodes || {};
window.kjNodes.addNode = addNode;

app.registerExtension({
	name: "KJNodes.Contextmenu",
	settings: [
		{
			id: "KJNodes.helpPopup",
			name: "Help popups",
			category: ["KJNodes", "General", "Help popups"],
			tooltip: "Show help popups when hovering over KJNodes",
			defaultValue: true,
			type: "boolean",
		},
		{
			id: "KJNodes.showSetGetInConnectionMenu",
			name: "Show Set/Get in connection menu",
			category: ["KJNodes", "Set & Get", "Show Set/Get in connection menu"],
			tooltip: "Add Set/Get entries to the slot connection menu (next to Add Reroute)",
			type: "boolean",
			defaultValue: true,
		},
	],
	getNodeMenuItems(node) {
		try {
			const items = [];

			if (node.inputs && node.inputs.length > 0) {
				const selectedForConvert = window.kjNodes.snapshotSelectedNodes(node);
				items.push(
					{
						content: "Add GetNode",
						callback: () => { addNode("GetNode", node, { side: "left", offset: 30 }); }
					},
					{
						content: "Add SetNode",
						callback: () => { addNode("SetNode", node, { side: "right", offset: 30 }); }
					},
					{
						content: "Add PreviewAsTextNode",
						callback: () => { addNode("PreviewAny", node, { side: "right", offset: 30 }); }
					},
					{
						content: "Convert all outputs to Set/Get",
						callback: () => {
							for (const n of selectedForConvert) window.kjNodes.convertOutputsToSetGet(node.graph, n);
						}
					},
				);
			}

			const cls = node.constructor?.comfyClass || node.comfyClass;
			if (cls && window.kjNodes?.recreateNode) {
				items.push({
					content: "Recreate node",
					has_submenu: true,
					submenu: {
						options: [
							{ content: "Keep widget values", callback: () => window.kjNodes.recreateNode(node, cls, { resetValues: false }) },
							{ content: "Reset widget values", callback: () => window.kjNodes.recreateNode(node, cls, { resetValues: true }) },
						],
					},
				});
			}

			return items;
		} catch (err) {
			console.error("[KJNodes.Contextmenu] getNodeMenuItems failed:", err);
			return [];
		}
	},
	async setup(app) {
		const origShowConnectionMenu = app.canvas.showConnectionMenu.bind(app.canvas);

		// Inject "Add SetNode" / "Add GetNode" into the slot connection dropdown menu.
		app.canvas.showConnectionMenu = function(optPass) {
			const showSetGet = app.ui.settings.getSettingValue("KJNodes.showSetGetInConnectionMenu") ?? true;
			if (!showSetGet) return origShowConnectionMenu(optPass);

			const isFrom = optPass.nodeFrom && optPass.slotFrom != null;
			const nodeType = isFrom ? "SetNode" : "GetNode";
			const label = isFrom ? "Add SetNode" : "Add GetNode";

			const OrigCM = LiteGraph.ContextMenu;
			let interceptActive = true;
			LiteGraph.ContextMenu = function(options, menuOpts) {
				LiteGraph.ContextMenu = OrigCM;
				if (!interceptActive) return new OrigCM(options, menuOpts);
				interceptActive = false;

				const idx = options.indexOf("Add Reroute");
				if (idx !== -1) options.splice(idx + 1, 0, label);

				const origCb = menuOpts.callback;
				menuOpts.callback = function(v, cbOpts, e) {
					if (v === label) {
						const node = LiteGraph.createNode(nodeType);
						if (node) {
							const graph = app.canvas?.graph || app.graph;
							node.pos = [optPass.e?.canvasX ?? 0, optPass.e?.canvasY ?? 0];
							graph.add(node);
							if (isFrom && optPass.nodeFrom) {
								const slotIdx = typeof optPass.slotFrom === 'number'
									? optPass.slotFrom
									: optPass.nodeFrom.findOutputSlot(optPass.slotFrom.name);
								optPass.nodeFrom.connect(slotIdx, node, 0);
							} else if (optPass.nodeTo) {
								const slotIdx = typeof optPass.slotTo === 'number'
									? optPass.slotTo
									: optPass.nodeTo.findInputSlot(optPass.slotTo.name);
								node.connect(0, optPass.nodeTo, slotIdx);
							}
							graph.change();
							app.canvas.setDirty(true, true);
						}
						return;
					}
					return origCb?.call(this, v, cbOpts, e);
				};

				return new OrigCM(options, menuOpts);
			};
			LiteGraph.ContextMenu.prototype = OrigCM.prototype;

			const result = origShowConnectionMenu(optPass);
			LiteGraph.ContextMenu = OrigCM;
			interceptActive = false;
			return result;
		};
	}
});
