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
			id: "KJNodes.nodeAutoColor",
			name: "Auto-color nodes",
			category: ["KJNodes", "Set & Get", "Auto-color nodes"],
			tooltip: "Automatically color Set/Get nodes based on their connection type",
			type: "boolean",
			defaultValue: true,
		},
		{
			id: "KJNodes.disablePrefix",
			name: "Disable Set_/Get_ prefix",
			category: ["KJNodes", "Set & Get", "Disable Set_/Get_ prefix"],
			tooltip: "Prevents automatically adding Set_ and Get_ prefixes to node titles",
			defaultValue: false,
			type: "boolean",
		},
		{
			id: "KJNodes.helpPopup",
			name: "Help popups",
			category: ["KJNodes", "General", "Help popups"],
			tooltip: "Show help popups when hovering over KJNodes",
			defaultValue: true,
			type: "boolean",
		},
		{
			id: "KJNodes.shiftMiddleClickSetGet",
			name: "Shift+middle-click creates Set/Get",
			category: ["KJNodes", "Set & Get", "Shift+middle-click creates Set/Get"],
			tooltip: "Shift+middle-click on a slot creates a connected SetNode (output) or GetNode (input) instead of Reroute",
			type: "boolean",
			defaultValue: true,
		},
		{
			id: "KJNodes.middleClickSetGet",
			name: "Middle-click creates Set/Get",
			category: ["KJNodes", "Set & Get", "Middle-click creates Set/Get"],
			tooltip: "Middle-click on a slot creates Set/Get instead of Reroute (overrides core behavior)",
			type: "boolean",
			defaultValue: false,
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
		if (!node.inputs || node.inputs.length === 0) return [];
		const selectedForConvert = window.kjNodes.snapshotSelectedNodes(node);
		return [
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
		];
	},
	async setup(app) {
			// Patch showConnectionMenu to inject "Add SetNode" / "Add GetNode" entries.
			const origShowConnectionMenu = app.canvas.showConnectionMenu.bind(app.canvas);
			app.canvas.showConnectionMenu = function(optPass) {
				if (!(app.ui.settings.getSettingValue("KJNodes.showSetGetInConnectionMenu") ?? true)) return origShowConnectionMenu(optPass);

				const isFrom = optPass.nodeFrom && optPass.slotFrom;
				const nodeType = isFrom ? "SetNode" : "GetNode";
				const label = isFrom ? "Add SetNode" : "Add GetNode";

				// Capture original canvas position from the event
				const origEvent = optPass.e;

				const OrigCM = LiteGraph.ContextMenu;
				let interceptActive = true;
				LiteGraph.ContextMenu = function(options, menuOpts) {
					// Always restore before doing anything else
					LiteGraph.ContextMenu = OrigCM;

					// If another caller triggered this, pass through untouched
					if (!interceptActive) return new OrigCM(options, menuOpts);
					interceptActive = false;

					// Insert after "Add Reroute"
					const idx = options.indexOf("Add Reroute");
					if (idx !== -1) options.splice(idx + 1, 0, label);

					// Wrap callback to handle our entry
					const origCb = menuOpts.callback;
					menuOpts.callback = function(v, cbOpts, e) {
						if (v === label) {
							const canvas = app.canvas;
							const graph = canvas.graph || app.graph;
							const node = LiteGraph.createNode(nodeType);
							if (!node || !graph) return;

							// Use the original drag event's canvas position
							node.pos = [origEvent?.canvasX ?? 0, origEvent?.canvasY ?? 0];
							graph.add(node);

							if (isFrom && optPass.nodeFrom) {
								const slotObj = optPass.slotFrom;
								const slotIdx = typeof slotObj === 'number'
									? slotObj
									: optPass.nodeFrom.findOutputSlot(slotObj.name);
								optPass.nodeFrom.connect(slotIdx, node, 0);
							} else if (optPass.nodeTo) {
								const slotObj = optPass.slotTo;
								const slotIdx = typeof slotObj === 'number'
									? slotObj
									: optPass.nodeTo.findInputSlot(slotObj.name);
								node.connect(0, optPass.nodeTo, slotIdx);
							}
							graph.change();
							canvas.setDirty(true, true);
							return;
						}
						return origCb?.call(this, v, cbOpts, e);
					};

					return new OrigCM(options, menuOpts);
				};
				LiteGraph.ContextMenu.prototype = OrigCM.prototype;

				const result = origShowConnectionMenu(optPass);
				// Safety: ensure restoration even if origShowConnectionMenu didn't trigger the constructor
				LiteGraph.ContextMenu = OrigCM;
				interceptActive = false;
				return result;
			};

			// Middle-click Set/Get: let the core do slot hit detection, then intercept
			// createDefaultNodeForSlot to create Set/Get instead of reroute.
			let _fromMiddleClick = false;
			let _lastMiddleClickShift = false;
			const origProcessMiddle = LGraphCanvas.prototype._processMiddleButton;
			LGraphCanvas.prototype._processMiddleButton = function(e, node) {
				_fromMiddleClick = true;
				_lastMiddleClickShift = !!e.shiftKey;
				const wantSetGet = _lastMiddleClickShift
					? app.ui.settings.getSettingValue("KJNodes.shiftMiddleClickSetGet") ?? true
					: app.ui.settings.getSettingValue("KJNodes.middleClickSetGet") ?? false;
				const saved = LiteGraph.middle_click_slot_add_default_node;
				if (wantSetGet) {
					LiteGraph.middle_click_slot_add_default_node = true;
				} else {
					// Workaround: core slotDefaults.init() resets this flag to true
					const coreSetting = app.ui.settings.getSettingValue("Comfy.Node.MiddleClickRerouteNode");
					if (coreSetting === false) LiteGraph.middle_click_slot_add_default_node = false;
				}
				const result = origProcessMiddle.call(this, e, node);
				LiteGraph.middle_click_slot_add_default_node = saved;
				return result;
			};
			const origCreateDefault = LGraphCanvas.prototype.createDefaultNodeForSlot;
			LGraphCanvas.prototype.createDefaultNodeForSlot = function(optPass) {
				if (!_fromMiddleClick) return origCreateDefault.call(this, optPass);
				_fromMiddleClick = false;
				const wantSetGet = _lastMiddleClickShift
					? app.ui.settings.getSettingValue("KJNodes.shiftMiddleClickSetGet") ?? true
					: app.ui.settings.getSettingValue("KJNodes.middleClickSetGet") ?? false;
				if (!wantSetGet) return origCreateDefault.call(this, optPass);

				const isFrom = optPass.nodeFrom && optPass.slotFrom !== null;
				const sourceNode = isFrom ? optPass.nodeFrom : optPass.nodeTo;
				const slotIndex = isFrom ? optPass.slotFrom : optPass.slotTo;
				if (!sourceNode || slotIndex == null) return origCreateDefault.call(this, optPass);
				const graph = this.graph || app.graph;
				if (!graph) return origCreateDefault.call(this, optPass);

				const newNode = LiteGraph.createNode(isFrom ? "SetNode" : "GetNode");
				if (!newNode) return false;
				const bounding = sourceNode.getBounding();
				newNode.pos = [
					isFrom ? bounding[0] + bounding[2] + 30 : bounding[0] - 200,
					optPass.position?.[1] ?? bounding[1]
				];
				graph.add(newNode);
				if (isFrom) sourceNode.connect(slotIndex, newNode, 0);
				else newNode.connect(0, sourceNode, slotIndex);
				graph.change();
				this.setDirty(true, true);
				return true;
			};
}
});
