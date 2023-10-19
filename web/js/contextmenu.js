import { app } from "../../../scripts/app.js";

// Adds context menu entries, code from pyssssscustom-scripts

function addMenuHandler(nodeType, cb) {
	const getOpts = nodeType.prototype.getExtraMenuOptions;
	nodeType.prototype.getExtraMenuOptions = function () {
		const r = getOpts.apply(this, arguments);
		cb.apply(this, arguments);
		return r;
	};
}

function addNode(name, nextTo, options) {
	console.log("name:", name);
	console.log("nextTo:", nextTo);
	options = { side: "left", select: true, shiftY: 0, shiftX: 0, ...(options || {}) };
	const node = LiteGraph.createNode(name);
	app.graph.add(node);
	
	node.pos = [
		options.side === "left" ? nextTo.pos[0] - (node.size[0] + options.offset): nextTo.pos[0] + nextTo.size[0] + options.offset,
		
		nextTo.pos[1] + options.shiftY,
	];
	if (options.select) {
		app.canvas.selectNode(node, false);
	}
	return node;
}

app.registerExtension({
	name: "KJNodesContextmenu",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.input && nodeData.input.required) {
			addMenuHandler(nodeType, function (_, options) {
				options.unshift(
					{
					content: "Add GetNode",
					callback: () => {addNode("GetNode", this, { side:"left", offset: 30});}
					},
					{
					content: "Add SetNode",
					callback: () => {addNode("SetNode", this, { side:"right", offset: 30 });
					},
					
				});
			});
				
		}
	},
});
