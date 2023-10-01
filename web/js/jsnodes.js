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
				//this.inputs_offset = nodeData.name.includes("selective")?1:0
				this.cond_type = "CONDITIONING"
				this.inputs_offset = nodeData.name.includes("selective")?1:0
				this.addWidget("button", "Update inputs", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
					    if(target_number_of_inputs===this.inputs.length)return; // already set, do nothing

					    if(target_number_of_inputs < this.inputs.length){
    						for(let i = this.inputs.length; i>=this.inputs_offset+target_number_of_inputs; i--)
							      this.removeInput(i)
					    }
                        else{
						    for(let i = this.inputs.length+1-this.inputs_offset; i <= target_number_of_inputs; ++i)
						    	this.addInput(`conditioning_${i}`, this.cond_type)
                        }
				});
			}
			break;
	}
	},
});