const { app } = window.comfyAPI.app;

//based on diffus3's SetGet: https://github.com/diffus3/ComfyUI-extensions
//
// Fix for ComfyUI frontend 1.35+ compatibility:
// - Added setComboValues() method to GetNode (was called but never defined)
// - Changed combo widget from function-based values to array-based with refresh
// - Added support for Easy-Use setNode type for cross-extension compatibility

// Supported SetNode types (KJNodes + Easy-Use compatibility)
const SETNODE_TYPES = ['SetNode', 'easy setNode'];
const GETNODE_TYPES = ['GetNode', 'easy getNode'];

// Nodes that allow you to tunnel connections for cleaner graphs
function setColorAndBgColor(type) {
    const colorMap = {
		"DEFAULT": LGraphCanvas.node_colors.gray,
        "MODEL": LGraphCanvas.node_colors.blue,
        "LATENT": LGraphCanvas.node_colors.purple,
        "VAE": LGraphCanvas.node_colors.red,
		"WANVAE": LGraphCanvas.node_colors.red,
        "CONDITIONING": LGraphCanvas.node_colors.brown,
        "IMAGE": LGraphCanvas.node_colors.pale_blue,
        "CLIP": LGraphCanvas.node_colors.yellow,
        "FLOAT": LGraphCanvas.node_colors.green,
		"MASK": { color: "#1c5715", bgcolor: "#1f401b"},
		"INT": { color: "#1b4669", bgcolor: "#29699c"},
		"CONTROL_NET": { color: "#156653", bgcolor: "#1c453b"},
		"NOISE": { color: "#2e2e2e", bgcolor: "#242121"},
		"GUIDER": { color: "#3c7878", bgcolor: "#1c453b"},
		"SAMPLER": { color: "#614a4a", bgcolor: "#3b2c2c"},
		"SIGMAS": { color: "#485248", bgcolor: "#272e27"},

    };
	console.log("Setting color for type:", colorMap[type]);
    const colors = colorMap[type];
    if (colors) {
        this.color = colors.color;
        this.bgcolor = colors.bgcolor;
    }
	else{
		// Default color
		this.color = LGraphCanvas.node_colors.gray;
		this.bgcolor = LGraphCanvas.node_colors.gray;
	}
}
let disablePrefix = app.ui.settings.getSettingValue("KJNodes.disablePrefix")
const LGraphNode = LiteGraph.LGraphNode

function showAlert(message) {
  app.extensionManager.toast.add({
    severity: 'warn',
    summary: "KJ Get/Set",
    detail: `${message}. Most likely you're missing custom nodes`,
    life: 5000,
  })
}

// Helper function to get all SetNode values from graph
function getSetNodeValues(graph) {
	if (!graph || !graph._nodes) return [];
	return graph._nodes
		.filter(node => SETNODE_TYPES.includes(node.type))
		.map(node => node.widgets[0].value)
		.filter(value => value !== '')
		.sort();
}

// Helper function to notify all GetNodes to refresh their combo values
function notifyGetNodes(graph) {
	if (!graph || !graph._nodes) return;
	graph._nodes
		.filter(node => GETNODE_TYPES.includes(node.type))
		.forEach(node => {
			if (typeof node.setComboValues === 'function') {
				node.setComboValues();
			}
		});
}

app.registerExtension({
	name: "SetNode",
	registerCustomNodes() {
		class SetNode extends LGraphNode {
			defaultVisibility = true;
			serialize_widgets = true;
			drawConnection = false;
			currentGetters = null;
			slotColor = "#FFF";
			canvas = app.canvas;
			menuEntry = "Show connections";

			constructor(title) {
				super(title)
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
						if(this.widgets[0].value !== ''){
							this.title = (!disablePrefix ? "Set_" : "") + this.widgets[0].value;
						}
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
						if(this.inputs[slot].name === ''){
							this.inputs[slot].type = '*';
							this.inputs[slot].name = '*';
							this.title = "Set"
						}
					}
					if (slotType == 2 && !isChangeConnect) {
						if (this.outputs && this.outputs[slot]) {
							this.outputs[slot].type = '*';
							this.outputs[slot].name = '*';
						}
					}
					//On Connect
					if (link_info && node.graph && slotType == 1 && isChangeConnect) {
						const resolve = link_info.resolve(node.graph)
						const type = (resolve?.subgraphInput ?? resolve?.output)?.type
						if (type) {
							if (this.title === "Set"){
								this.title = (!disablePrefix ? "Set_" : "") + type;
							}
							if (this.widgets[0].value === '*'){
								this.widgets[0].value = type
							}

							this.validateName(node.graph);
							this.inputs[0].type = type;
							this.inputs[0].name = type;

							if (app.ui.settings.getSettingValue("KJNodes.nodeAutoColor")){
								setColorAndBgColor.call(this, type);
							}
						} else {
                showAlert("node input undefined.")
						}
					}
					if (link_info && node.graph && slotType == 2 && isChangeConnect) {
						const fromNode = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);

						if (fromNode && fromNode.inputs && fromNode.inputs[link_info.origin_slot]) {
							const type = fromNode.inputs[link_info.origin_slot].type;

							this.outputs[0].type = type;
							this.outputs[0].name = type;
						} else {
							showAlert('node output undefined');
						}
					}


					//Update either way
					this.update();
				}

				this.validateName = function(graph) {
					let widgetValue = node.widgets[0].value;

					if (widgetValue !== '') {
						let tries = 0;
						const existingValues = new Set();

						graph._nodes.forEach(otherNode => {
							if (otherNode !== this && SETNODE_TYPES.includes(otherNode.type)) {
								existingValues.add(otherNode.widgets[0].value);
							}
						});

						while (existingValues.has(widgetValue)) {
							widgetValue = node.widgets[0].value + "_" + tries;
							tries++;
						}

						node.widgets[0].value = widgetValue;
						this.update();
					}
				}

				this.clone = function () {
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
					if (!node.graph) {
						return;
					}

					const getters = this.findGetters(node.graph);
					getters.forEach(getter => {
						getter.setType(this.inputs[0].type);
					});

					if (this.widgets[0].value) {
						const gettersWithPreviousName = this.findGetters(node.graph, true);
						gettersWithPreviousName.forEach(getter => {
							getter.setName(this.widgets[0].value);
						});
					}

					// Notify all GetNodes to refresh their combo values
					notifyGetNodes(node.graph);
				}


				this.findGetters = function(graph, checkForPreviousName) {
					const name = checkForPreviousName ? this.properties.previousName : this.widgets[0].value;
					return graph._nodes.filter(otherNode =>
						GETNODE_TYPES.includes(otherNode.type) &&
						otherNode.widgets[0].value === name &&
						name !== ''
					);
				}


				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}


			onRemoved() {
				// Notify GetNodes that a SetNode was removed
				notifyGetNodes(this.graph);
			}
			getExtraMenuOptions(_, options) {
				this.menuEntry = this.drawConnection ? "Hide connections" : "Show connections";
				options.unshift(
					{
						content: this.menuEntry,
						callback: () => {
							this.currentGetters = this.findGetters(this.graph);
							if (this.currentGetters.length == 0) return;
							let linkType = (this.currentGetters[0].outputs[0].type);
							this.slotColor = this.canvas.default_connection_color_byType[linkType]
							this.menuEntry = this.drawConnection ? "Hide connections" : "Show connections";
							this.drawConnection = !this.drawConnection;
							this.canvas.setDirty(true, true);

						},
						has_submenu: true,
						submenu: {
							title: "Color",
                            options: [
								{
								content: "Highlight",
								callback: () => {
									this.slotColor = "orange"
									this.canvas.setDirty(true, true);
									}
								}
							],
						},
					},
					{
						content: "Hide all connections",
						callback: () => {
							const allGetters = this.graph._nodes.filter(otherNode =>
								GETNODE_TYPES.includes(otherNode.type) || SETNODE_TYPES.includes(otherNode.type)
							);
							allGetters.forEach(otherNode => {
								otherNode.drawConnection = false;
								console.log(otherNode);
							});

							this.menuEntry = "Show connections";
							this.drawConnection = false
							this.canvas.setDirty(true, true);

						},

					},
				);
				// Dynamically add a submenu for all getters
				this.currentGetters = this.findGetters(this.graph);
				if (this.currentGetters) {

					let gettersSubmenu = this.currentGetters.map(getter => ({

						content: `${getter.title} id: ${getter.id}`,
						callback: () => {
							this.canvas.centerOnNode(getter);
							this.canvas.selectNode(getter, false);
							this.canvas.setDirty(true, true);

						},
					}));

					options.unshift({
						content: "Getters",
						has_submenu: true,
						submenu: {
							title: "GetNodes",
                            options: gettersSubmenu,
						}
					});
				}
			}


			onDrawForeground(ctx, lGraphCanvas) {
				if (this.drawConnection) {
					this._drawVirtualLinks(lGraphCanvas, ctx);
				}
			}
			// onDrawCollapsed(ctx, lGraphCanvas) {
			// 	if (this.drawConnection) {
			// 		this._drawVirtualLinks(lGraphCanvas, ctx);
			// 	}
			// }
			_drawVirtualLinks(lGraphCanvas, ctx) {
				if (!this.currentGetters?.length) return;
				var title = this.getTitle ? this.getTitle() : this.title;
				var title_width = ctx.measureText(title).width;
				if (!this.flags.collapsed) {
					var start_node_slotpos = [
						this.size[0],
						LiteGraph.NODE_TITLE_HEIGHT * 0.5,
						];
				}
				else {

					var start_node_slotpos = [
						title_width + 55,
						-15,

						];
				}
				// Provide a default link object with necessary properties, to avoid errors as link can't be null anymore
				const defaultLink = { type: 'default', color: this.slotColor };

				for (const getter of this.currentGetters) {
					if (!this.flags.collapsed) {
					var end_node_slotpos = this.getConnectionPos(false, 0);
					end_node_slotpos = [
						getter.pos[0] - end_node_slotpos[0] + this.size[0],
						getter.pos[1] - end_node_slotpos[1]
						];
					}
					else {
						var end_node_slotpos = this.getConnectionPos(false, 0);
						end_node_slotpos = [
						getter.pos[0] - end_node_slotpos[0] + title_width + 50,
						getter.pos[1] - end_node_slotpos[1] - 30
						];
					}
					lGraphCanvas.renderLink(
						ctx,
						start_node_slotpos,
						end_node_slotpos,
						defaultLink,
						false,
						null,
						this.slotColor,
						LiteGraph.RIGHT,
						LiteGraph.LEFT
					);
				}
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
		class GetNode extends LGraphNode {

			defaultVisibility = true;
			serialize_widgets = true;
			drawConnection = false;
			slotColor = "#FFF";
			currentSetter = null;
			canvas = app.canvas;
			// Store combo values as instance property for frontend compatibility
			_comboValues = [];

			constructor(title) {
				super(title)
				if (!this.properties) {
					this.properties = {};
				}
				this.properties.showOutputText = GetNode.defaultVisibility;
				const node = this;

				// Initialize combo values array
				this._comboValues = [];

				this.addWidget(
					"combo",
					"Constant",
					"",
					(e) => {
						this.onRename();
					},
					{
						// Use array reference instead of function for frontend 1.35+ compatibility
						// The array is updated by setComboValues() method
						values: this._comboValues
					}
				)

				this.addOutput("*", '*');

				// Method to refresh combo values - called by SetNode.update() and on graph changes
				// This fixes the bug where setComboValues was called but never defined
				this.setComboValues = function(excludeNodes) {
					if (!node.graph || !node.graph._nodes) return;

					let setterNodes = node.graph._nodes.filter(otherNode =>
						SETNODE_TYPES.includes(otherNode.type)
					);

					// Exclude removed nodes if provided
					if (excludeNodes && Array.isArray(excludeNodes)) {
						const excludeIds = excludeNodes.map(n => n.id);
						setterNodes = setterNodes.filter(n => !excludeIds.includes(n.id));
					}

					const values = setterNodes
						.map(otherNode => otherNode.widgets[0].value)
						.filter(v => v !== '')
						.sort();

					// Update the combo values array in-place to maintain reference
					node._comboValues.length = 0;
					node._comboValues.push(...values);

					// Also update widget.options.values directly for maximum compatibility
					if (node.widgets && node.widgets[0] && node.widgets[0].options) {
						node.widgets[0].options.values = node._comboValues;
					}

					// Force canvas redraw
					if (app.canvas) {
						app.canvas.setDirty(true, true);
					}
				};

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
						this.title = (!disablePrefix ? "Get_" : "") + setter.widgets[0].value;

						if (app.ui.settings.getSettingValue("KJNodes.nodeAutoColor")){
							setColorAndBgColor.call(this, linkType);
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
					if (this.outputs[0].type !== '*' && this.outputs[0].links) {
						this.outputs[0].links.filter(linkId => {
							const link = node.graph.links[linkId];
							return link && (!link.type.split(",").includes(this.outputs[0].type) && link.type !== '*');
						}).forEach(linkId => {
							node.graph.removeLink(linkId);
						});
					}
				};

				this.setType = function(type) {
					this.outputs[0].name = type;
					this.outputs[0].type = type;
					this.validateLinks();
				}

				this.findSetter = function(graph) {
					const name = this.widgets[0].value;
					const foundNode = graph._nodes.find(otherNode =>
						SETNODE_TYPES.includes(otherNode.type) &&
						otherNode.widgets[0].value === name &&
						name !== ''
					);
					return foundNode;
				};

				this.goToSetter = function() {
					this.canvas.centerOnNode(this.currentSetter);
					this.canvas.selectNode(this.currentSetter, false);
				};

				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}

			getInputLink(slot) {
				const setter = this.findSetter(this.graph);

				if (setter) {
					const slotInfo = setter.inputs[slot];
					const link = this.graph.links[slotInfo.link];
					return link;
				} else {
					const errorMessage = "No SetNode found for " + this.widgets[0].value + "(" + this.type + ")";
					showAlert(errorMessage);
					//throw new Error(errorMessage);
				}
			}

			onAdded(graph) {
				// Initialize combo values when node is added to graph
				// Use setTimeout to ensure graph is fully ready
				setTimeout(() => {
					if (this.setComboValues) {
						this.setComboValues();
					}
				}, 50);
			}

			getExtraMenuOptions(_, options) {
				let menuEntry = this.drawConnection ? "Hide connections" : "Show connections";
				this.currentSetter = this.findSetter(this.graph)
				if (!this.currentSetter) return
				options.unshift(
					{
						content: "Go to setter",
						callback: () => {
							this.goToSetter();
						},
					},
					{
						content: menuEntry,
						callback: () => {
							let linkType = (this.currentSetter.inputs[0].type);
							this.drawConnection = !this.drawConnection;
							this.slotColor = this.canvas.default_connection_color_byType[linkType]
							this.canvas.setDirty(true, true);
						},
					},
				);
			}

			onDrawForeground(ctx, lGraphCanvas) {
				if (this.drawConnection) {
					this._drawVirtualLink(lGraphCanvas, ctx);
				}
			}
			// onDrawCollapsed(ctx, lGraphCanvas) {
			// 	if (this.drawConnection) {
			// 		this._drawVirtualLink(lGraphCanvas, ctx);
			// 	}
			// }
			_drawVirtualLink(lGraphCanvas, ctx) {
				if (!this.currentSetter) return;

				// Provide a default link object with necessary properties, to avoid errors as link can't be null anymore
				const defaultLink = { type: 'default', color: this.slotColor };

				let start_node_slotpos = this.currentSetter.getConnectionPos(false, 0);
				start_node_slotpos = [
					start_node_slotpos[0] - this.pos[0],
					start_node_slotpos[1] - this.pos[1],
				];
				let end_node_slotpos = [0, -LiteGraph.NODE_TITLE_HEIGHT * 0.5];
				lGraphCanvas.renderLink(
					ctx,
					start_node_slotpos,
					end_node_slotpos,
					defaultLink,
					false,
					null,
					this.slotColor
				);
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
