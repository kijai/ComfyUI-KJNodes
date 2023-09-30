import { app } from "../../../scripts/app.js";
//WIP doesn't do anything
app.registerExtension({
	name: "KJNodes.PlotNode",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		switch (nodeData.name) {
			case "PlotNode":

				nodeType.prototype.onNodeCreated = function () {
					
				this.addWidget("button", "Update", null, () => {
					
					console.log("start x:" + this.pos[0])
					console.log("start y:" +this.pos[1])
					console.log(this.graph.links);
					const toNode = this.graph._nodes.find((otherNode) => otherNode.id == this.graph.links[1].target_id);
					console.log("target x:" + toNode.pos[0])
				const a = this.pos[0]
				const b = toNode.pos[0]
				const distance = Math.abs(a - b);
				const maxDistance = 1000
				const finalDistance = (distance - 0) / (maxDistance - 0);

				this.widgets[0].value = finalDistance; 
				});
			}
			break;
	}
	},
});