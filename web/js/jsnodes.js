import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "KJNodes.ConditioningMultiCombine",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		switch (nodeData.name) {
			case "ConditioningMultiCombine":
				nodeType.prototype.onNodeMoved = function () {
					console.log(this.pos[0])
				}
				nodeType.prototype.onNodeCreated = function () {
				this.inputs_offset = nodeData.name.includes("selective")?1:0
				this.cond_type = "CONDITIONING"
				this.addWidget("button", "Add", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					if (this.inputs.length < 20) {
						const newInputName = `conditioning_${this.inputs.length + 1}`;
						this.addInput(newInputName, this.cond_type);
					}
				});
				this.addWidget("button", "Remove", null, () => {
				if (this.inputs.length > 2) {
					const lastInputIndex = this.inputs.length - 1;
					this.removeInput(lastInputIndex);
					}
				});
			}
			break;
	}
	},
});