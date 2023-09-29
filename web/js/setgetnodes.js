import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from '../../../scripts/widgets.js';
//based on diffus3's SetGet: https://github.com/diffus3/ComfyUI-extensions

// Nodes that allow you to tunnel connections for cleaner graphs
app.registerExtension({
	name: "SetNode",
	registerCustomNodes() {
		class SetNode {
			defaultVisibility = true;
			serialize_widgets = true;
			constructor() {
				if (!this.properties) {
					this.properties = {
						"previousName": ""
					};
				}
				this.properties.showOutputText = SetNode.defaultVisibility;

				const node = this;

				
				this.addWidget(
					"text", 
					"Constant", 
					'', 
					(s, t, u, v, x) => {
						node.validateName(node.graph);
						this.title = "Set_" + this.widgets[0].value;
						this.update();
						this.properties.previousName = this.widgets[0].value;
					}, 
					{}
				)
				
				this.addInput("*", "*");
				this.addOutput("*", '*');

				

				this.onConnectionsChange = function(
					slotType,	//1 = input, 2 = output
					slot,
					isChangeConnect,
                    link_info,
                    output
				) {
					//On Disconnect
					if (slotType == 1 && !isChangeConnect) {
						this.inputs[slot].type = '*';
						this.inputs[slot].name = '*';
						this.title = "Set"
					}
					if (slotType == 2 && !isChangeConnect) {
						this.outputs[slot].type = '*';
						this.outputs[slot].name = '*';
					}	
					//On Connect
					if (link_info && node.graph && slotType == 1 && isChangeConnect) {
						
						const fromNode = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
						const type = fromNode.outputs[link_info.origin_slot].type;
						
						if (this.title == "Set"){
							this.title = "Set_" + type;	
						}
						if (this.widgets[0].value == ''){
							this.widgets[0].value = type	
						}
						
						this.validateName(node.graph);
						this.inputs[0].type = type;
						this.inputs[0].name = type;
						
						switch (type) {
							case "MODEL":
								this.color = LGraphCanvas.node_colors.blue.color;
								this.bgcolor = LGraphCanvas.node_colors.blue.bgcolor;
								break;
							case "LATENT":
								this.color = LGraphCanvas.node_colors.purple.color;
								this.bgcolor = LGraphCanvas.node_colors.purple.bgcolor;
								break;
							case "VAE":
								this.color = LGraphCanvas.node_colors.red.color;
								this.bgcolor = LGraphCanvas.node_colors.red.bgcolor;
								break;
							case "CONDITIONING":
								this.color = LGraphCanvas.node_colors.brown.color;
								this.bgcolor = LGraphCanvas.node_colors.brown.bgcolor;
								break;
							case "IMAGE":
								this.color = LGraphCanvas.node_colors.pale_blue.color;
								this.bgcolor = LGraphCanvas.node_colors.pale_blue.bgcolor;
								break;
							case "CLIP":
								this.color = LGraphCanvas.node_colors.yellow.color;
								this.bgcolor = LGraphCanvas.node_colors.yellow.bgcolor;
								break;
							case "INT":
								this.color = LGraphCanvas.node_colors.green.color;
								this.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
								break;
						}
					}
					if (link_info && node.graph && slotType == 2 && isChangeConnect) {
						const fromNode = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
						const type = fromNode.inputs[link_info.origin_slot].type;

						this.outputs[0].type = type;
						this.outputs[0].name = type;
					}
					

					//Update either way
					this.update();
				}

				this.validateName = function(graph) {
					let widgetValue = node.widgets[0].value;
					
					if (widgetValue != '') {
						let tries = 0;
						let collisions = [];
						
						do {
							collisions = graph._nodes.filter((otherNode) => {
								if (otherNode == this) {
									return false;
								}
								if (otherNode.type == 'SetNode' && otherNode.widgets[0].value === widgetValue) {
									return true;
								}
								return false;
							})
							if (collisions.length > 0) {
								widgetValue = node.widgets[0].value + "_" + tries;
							}
							tries++;
						} while (collisions.length > 0)
						node.widgets[0].value = widgetValue;
						this.update();
					}
				}

				this.clone = function () {
					console.log("CLONE");
					const cloned = SetNode.prototype.clone.apply(this);
					cloned.inputs[0].name = '*';
					cloned.inputs[0].type = '*';
					cloned.value = '';
					cloned.properties.previousName = '';
					cloned.size = cloned.computeSize();
					return cloned;
				};

				this.onAdded = function(graph) {
					this.validateName(graph);
				}


				this.update = function() {
					if (node.graph) {
						this.findGetters(node.graph).forEach((getter) => {
							getter.setType(this.inputs[0].type);
						});
						if (this.widgets[0].value) {
							this.findGetters(node.graph, true).forEach((getter) => {
								getter.setName(this.widgets[0].value)
							});
						}

						const allGetters = node.graph._nodes.filter((otherNode) => otherNode.type == "GetNode");
						allGetters.forEach((otherNode) => {
							if (otherNode.setComboValues) {
								otherNode.setComboValues();
							}
						})
					}
				}


				this.findGetters = function(graph, checkForPreviousName) {
					const name = checkForPreviousName ? this.properties.previousName : this.widgets[0].value;
					return graph._nodes.filter((otherNode) => {
						if (otherNode.type == 'GetNode' && otherNode.widgets[0].value === name && name != '') {
							return true;
						}
						return false;
					})
				}


				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}

			onRemoved() {
				console.log("onRemove");
				console.log(this);
				console.log(this.flags);
				const allGetters = this.graph._nodes.filter((otherNode) => otherNode.type == "GetNode");
				allGetters.forEach((otherNode) => {
					if (otherNode.setComboValues) {
						otherNode.setComboValues([this]);
					}
				})
			}
		}


		LiteGraph.registerNodeType(
			"SetNode",
			Object.assign(SetNode, {
				title: "Set",
			})
		);

		SetNode.category = "KJNodes";
	},
});


app.registerExtension({
	name: "GetNode",
	registerCustomNodes() {
		class GetNode {

			defaultVisibility = true;
			serialize_widgets = true;

			constructor() {
				if (!this.properties) {
					this.properties = {};
				}
				this.properties.showOutputText = GetNode.defaultVisibility;
				
				const node = this;
				this.addWidget(
					"combo",
					"Constant",
					"",
					(e) => {
						this.onRename();
					},
					{
						values: () => {
                            const setterNodes = graph._nodes.filter((otherNode) => otherNode.type == 'SetNode');
                            return setterNodes.map((otherNode) => otherNode.widgets[0].value).sort();
                        }
					}
				)


				this.addOutput("*", '*');

				
				this.onConnectionsChange = function(
					slotType,	//0 = output, 1 = input
					slot,	//self-explanatory
					isChangeConnect,
                    link_info,
                    output
				) {
					this.validateLinks();	
				}

				
				this.setName = function(name) {
					node.widgets[0].value = name;
					node.onRename();
					node.serialize();
				}
				

				this.onRename = function() {
					const setter = this.findSetter(node.graph);
					if (setter) {
						let linkType = (setter.inputs[0].type);
						
						this.setType(linkType);
						this.title = "Get_" + setter.widgets[0].value;
						switch (linkType) {
							case "MODEL":
								this.color = LGraphCanvas.node_colors.blue.color;
								this.bgcolor = LGraphCanvas.node_colors.blue.bgcolor;
								break;
							case "LATENT":
								this.color = LGraphCanvas.node_colors.purple.color;
								this.bgcolor = LGraphCanvas.node_colors.purple.bgcolor;
								break;
							case "VAE":
								this.color = LGraphCanvas.node_colors.red.color;
								this.bgcolor = LGraphCanvas.node_colors.red.bgcolor;
								break;
							case "CONDITIONING":
								this.color = LGraphCanvas.node_colors.brown.color;
								this.bgcolor = LGraphCanvas.node_colors.brown.bgcolor;
								break;
							case "IMAGE":
								this.color = LGraphCanvas.node_colors.pale_blue.color;
								this.bgcolor = LGraphCanvas.node_colors.pale_blue.bgcolor;
								break;
							case "CLIP":
								this.color = LGraphCanvas.node_colors.yellow.color;
								this.bgcolor = LGraphCanvas.node_colors.yellow.bgcolor;
								break;
							case "INT":
								this.color = LGraphCanvas.node_colors.green.color;
								this.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
								break;
						}

					} else {
						this.setType('*');
					}
				}

				this.clone = function () {
					const cloned = GetNode.prototype.clone.apply(this);
					cloned.size = cloned.computeSize();
					return cloned;
				};

				this.validateLinks = function() {
					console.log("validating links");
					if (this.outputs[0].type != '*' && this.outputs[0].links) {
						console.log("in");
						this.outputs[0].links.forEach((linkId) => {
							const link = node.graph.links[linkId];
							if (link && link.type != this.outputs[0].type && link.type != '*') {
								console.log("removing link");
								node.graph.removeLink(linkId)
							}
						})
					} 
				}

				this.setType = function(type) {
					this.outputs[0].name = type;
					this.outputs[0].type = type;
					this.validateLinks();
				}

				this.findSetter = function(graph) {
					const name = this.widgets[0].value;
					return graph._nodes.find((otherNode) => {
						if (otherNode.type == 'SetNode' && otherNode.widgets[0].value === name && name != '') {
							return true;
						}
						return false;
					})
				}

				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}


			getInputLink(slot) {
                const setter = this.findSetter(this.graph);
                
				if (setter) {
					const slot_info = setter.inputs[slot];
                    console.log("slot info");
                    console.log(slot_info);
                    console.log(this.graph.links);
                    const link = this.graph.links[ slot_info.link ];
                    console.log("link:");
                    console.log(link);
                    return link;
				} else {
                    console.log(this.widgets[0]);
                    console.log(this.widgets[0].value);
					alert("No SetNode found for " + this.widgets[0].value + "(" + this.type + ")");
					throw new Error("No SetNode found for " + this.widgets[0].value + "(" + this.type + ")");
					
				}

			}
			onAdded(graph) {
				//this.setComboValues();
				//this.validateName(graph);
			}

		}


		LiteGraph.registerNodeType(
			"GetNode",
			Object.assign(GetNode, {
				title: "Get",
			})
		);

		GetNode.category = "KJNodes";
	},
});
