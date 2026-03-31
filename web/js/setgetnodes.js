const { app } = window.comfyAPI.app;

// originally based on diffus3's SetGet: https://github.com/diffus3/ComfyUI-extensions

// Nodes that allow you to hide connections for cleaner graphs

let _typeColorMap;
function setColorAndBgColor(node, type) {
	if (!_typeColorMap) {
		_typeColorMap = {
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
	}
	const colors = _typeColorMap[type] || LGraphCanvas.node_colors?.gray;
	if (colors) {
		node.color = colors.color;
		node.bgcolor = colors.bgcolor;
	}
}
function getDisablePrefix() {
	return app.ui.settings.getSettingValue("KJNodes.disablePrefix") ?? false;
}
function prefixedTitle(prefix, name) {
	return (getDisablePrefix() ? "" : prefix + "_") + name;
}
function autoColor(node, type) {
	if (!app.ui.settings.getSettingValue("KJNodes.nodeAutoColor")) return;
	if (type === '*') { node.color = null; node.bgcolor = null; }
	else setColorAndBgColor(node, type);
}
function addNodeToSelectedOrCursor(nodeType, side) {
	const selected = Object.values(app.canvas.selected_nodes || {});
	if (selected.length > 0) {
		for (const n of selected) window.kjNodes.addNode(nodeType, n, { side, offset: 30 });
	} else {
		const canvas = app.canvas, graph = canvas.graph || app.graph;
		const node = LiteGraph.createNode(nodeType);
		if (!node) return;
		node.pos = [canvas.graph_mouse[0], canvas.graph_mouse[1]];
		graph.add(node);
		canvas.selectNode(node, false);
		canvas.setDirty(true, true);
	}
}
// Temporary map for paste rename coordination between Set and Get nodes.
// Key: old name, Value: new name. Cleared via setTimeout(0) after each paste cycle.
// This works because both onConfigure calls (Set + Get) fire synchronously within
// the same paste operation, before the timeout clears the entry.
const _pasteRenameMap = new Map();

let _forceShowAllLinks = false;

function getShowLinksMode() {
	if (_forceShowAllLinks) return "always";
	return app.ui.settings.getSettingValue("KJNodes.showSetGetLinks") ?? "never";
}
const LGraphNode = LiteGraph.LGraphNode

// Cross-graph traversal utilities with lexical scoping for subgraph support.
// Set nodes propagate downward: a Set in a parent graph is visible to all descendant subgraphs.
// Get nodes look upward: a Get searches its own graph first, then parent, then grandparent, etc.
// Duplicate names are allowed across unrelated (sibling) subgraphs.

function findRootGraph(graph) {
	if (!graph) return null;
	return graph.rootGraph || graph;
}

// Find which SubgraphNode in parentGraph wraps the given subgraph
function findSubgraphNodeFor(parentGraph, innerNode) {
	if (!parentGraph?._nodes || !innerNode?.graph) return null;
	for (const n of parentGraph._nodes) {
		if (n.subgraph && n.subgraph === innerNode.graph) return n;
	}
	return null;
}

// Walk from a subgraph up to root, returning [graph, parent, grandparent, ..., root]
function getGraphAncestors(graph) {
	if (!graph) return [];
	const root = findRootGraph(graph);
	if (!root || graph === root) return [root];

	const chain = [graph];
	const visited = new Set([graph]);
	let current = graph;
	// Walk up: find which SubgraphNode wraps current, then get its parent graph
	while (current !== root) {
		let found = false;
		// Search root nodes
		for (const n of root._nodes) {
			if (n.subgraph === current) {
				chain.push(root);
				current = root;
				found = true;
				break;
			}
		}
		if (found) break;
		// Search sibling subgraphs (for nested subgraphs)
		const subgraphs = root._subgraphs || root.subgraphs;
		if (subgraphs) {
			for (const sg of subgraphs.values()) {
				if (sg === current || !sg._nodes) continue;
				for (const n of sg._nodes) {
					if (n.subgraph === current) {
						if (visited.has(sg)) { found = false; break; }
						visited.add(sg);
						chain.push(sg);
						current = sg;
						found = true;
						break;
					}
				}
				if (found) break;
			}
		}
		if (!found) {
			// Can't find parent, add root as fallback
			if (!chain.includes(root)) chain.push(root);
			break;
		}
	}
	return chain;
}

// Get all descendant subgraphs of a graph (children, grandchildren, etc.)
function getGraphDescendants(graph, _visited) {
	if (!graph?._nodes) return [];
	const visited = _visited || new Set();
	if (visited.has(graph)) return [];
	visited.add(graph);
	const descendants = [];
	for (const n of graph._nodes) {
		if (n.subgraph && !visited.has(n.subgraph)) {
			descendants.push(n.subgraph);
			descendants.push(...getGraphDescendants(n.subgraph, visited));
		}
	}
	return descendants;
}

// Collect nodes of a type from specific graphs
function collectNodesOfType(graphs, type) {
	const results = [];
	for (const g of graphs) {
		if (!g?._nodes) continue;
		for (const node of g._nodes) {
			if (node.type === type) results.push({ node, graph: g });
		}
	}
	return results;
}

// Find all nodes of type across ALL graphs (root + all subgraphs). Used for global operations.
function findAllNodesOfType(graph, type) {
	const root = findRootGraph(graph);
	if (!root) return [];
	const allGraphs = [root];
	const subgraphs = root._subgraphs || root.subgraphs;
	if (subgraphs) {
		for (const sg of subgraphs.values()) allGraphs.push(sg);
	}
	return collectNodesOfType(allGraphs, type);
}

// Scoped setter lookup: search current graph, then ancestors (look up).
function findSetterByName(graph, name) {
	if (!name) return null;
	for (const g of getGraphAncestors(graph)) {
		if (!g?._nodes) continue;
		for (const node of g._nodes) {
			if (node.type === 'SetNode' && node.widgets[0].value === name) {
				return { node, graph: g };
			}
		}
	}
	return null;
}

// Scoped getter lookup: search current graph + descendants (propagate down).
function findGettersByName(graph, name) {
	if (!name) return [];
	const graphs = [graph, ...getGraphDescendants(graph)];
	return collectNodesOfType(graphs, 'GetNode')
		.filter(entry => entry.node.widgets[0].value === name);
}

// Get all visible SetNode names for a GetNode's combo dropdown.
// Shows names from current graph + ancestors (what's in scope).
// Also builds a source map for getOptionLabel. This module-level variable is safe
// because the values getter and getOptionLabel run in the same synchronous render pass.
let _setNameSourceMap = new Map();

function getVisibleSetNames(graph, filterType) {
	const sourceMap = new Map();
	const ancestors = getGraphAncestors(graph);
	const entries = collectNodesOfType(ancestors, 'SetNode');
	for (const e of entries) {
		const name = e.node.widgets[0].value;
		if (!name) continue;
		if (filterType && filterType !== '*') {
			const setType = e.node.inputs[0]?.type;
			if (setType && setType !== '*') {
				const filterTypes = String(filterType).split(",");
				if (!filterTypes.some(ft => ft === setType || setType.split(",").includes(ft))) continue;
			}
		}
		if (!sourceMap.has(name)) {
			sourceMap.set(name, e.graph === graph ? "local" : "parent");
		}
	}
	_setNameSourceMap = sourceMap;
	return [...sourceMap.keys()].sort();
}

// Exposed globally for use in contextmenu.js
window.kjNodes = window.kjNodes || {};
window.kjNodes.convertOutputsToSetGet = convertOutputsToSetGet;
window.kjNodes.snapshotSelectedNodes = snapshotSelectedNodes;
function convertOutputsToSetGet(graph, node) {
	if (!graph || !node) return;
	for (let slotIdx = 0; slotIdx < node.outputs.length; slotIdx++) {
		const output = node.outputs[slotIdx];
		if (!output.links || output.links.length === 0) continue;

		const linkType = output.type || "*";
		const linkName = output.name || linkType;

		// Collect targets and link IDs to remove, skipping existing Set/Get nodes
		const targets = [];
		const linksToRemove = [];
		for (const linkId of [...output.links]) {
			const link = getLink(graph, linkId);
			if (link) {
				const targetNode = graph.getNodeById(link.target_id);
				if (targetNode && targetNode.type !== 'SetNode' && targetNode.type !== 'GetNode') {
					targets.push({ targetId: link.target_id, targetSlot: link.target_slot });
					linksToRemove.push(linkId);
				}
			}
		}
		if (targets.length === 0) continue;

		// Create Set node
		const setNode = LiteGraph.createNode("SetNode");
		if (!setNode) continue;
		setNode.pos = [
			node.pos[0] + node.size[0] + 30,
			node.pos[1] + slotIdx * 60
		];
		graph.add(setNode);
		setNode.flags.collapsed = true;

		for (const linkId of linksToRemove) graph.removeLink(linkId);

		// Connect source → Set node input
		node.connect(slotIdx, setNode, 0);

		// Set the name widget
		setNode.widgets[0].value = linkName;
		setNode.title = prefixedTitle("Set", linkName);
		setNode.validateName(graph);
		const finalName = setNode.widgets[0].value;
		setNode.properties.previousName = finalName;

		// Create a Get node for each target
		for (const target of targets) {
			const targetNode = graph.getNodeById(target.targetId);
			if (!targetNode) continue;

			const getNode = LiteGraph.createNode("GetNode");
			if (!getNode) continue;
			getNode.pos = [
				targetNode.pos[0] - getNode.size[0] - 30,
				targetNode.pos[1]
			];
			graph.add(getNode);
			getNode.flags.collapsed = true;

			getNode.widgets[0].value = finalName;
			getNode.onRename();

			getNode.connect(0, targetNode, target.targetSlot);
		}
	}
	app.canvas.setDirty(true, true);
}

// Snapshot selection immediately (before right-click changes it)
function snapshotSelectedNodes(node, typeFilter) {
	const selected = Object.values(app.canvas.selected_nodes || {});
	// Always include the right-clicked node, even if selection state is inconsistent
	const nodeSet = new Set(selected);
	nodeSet.add(node);
	let nodes = [...nodeSet];
	if (typeFilter) nodes = nodes.filter(n => n.type === typeFilter);
	return nodes;
}

// Tracks nodes flagged with has_errors so they can be cleared on the next alert.
// Always cleared before repopulating, so it only holds nodes from the most recent alert.
const _errorNodes = new Set();
function showAlert(message, nodes) {
	const nodeList = nodes ? (Array.isArray(nodes) ? nodes : [nodes]) : [];
	const nodeInfo = nodeList.map(n =>
		`${n.title || n.type} [${Math.round(n.pos[0])}, ${Math.round(n.pos[1])}]`
	).join(', ');
	if (nodeList.length) {
		// Clear previous error flags
		for (const n of _errorNodes) n.has_errors = false;
		_errorNodes.clear();
		setTimeout(() => {
			for (const n of nodeList) {
				n.has_errors = true;
				_errorNodes.add(n);
			}
			app.canvas?.setDirty(true, true);
		}, 100);
	}
	app.extensionManager.toast.add({
		severity: 'warn',
		summary: "KJ Set/Get",
		detail: nodeInfo ? `${message} — ${nodeInfo}` : message,
		life: 5000,
	});
}
function convertAllSetGetToLinks(graph) {
	if (!graph) return;
	const rootGraph = findRootGraph(graph);

	// First pass: handle cross-graph pairs (must run before same-graph
	// because converting cross-graph creates real links that replace
	// the Set/Get, and same-graph pass would miss the removed nodes)
	const allSetEntries = findAllNodesOfType(rootGraph, 'SetNode');
	for (const { node: setNode, graph: setGraph } of [...allSetEntries]) {
		const name = setNode.widgets[0].value;
		if (!name) continue;
		const allGetEntries = findGettersByName(rootGraph, name);
		const crossGraphGetters = allGetEntries.filter(e => e.graph !== setGraph);
		if (crossGraphGetters.length > 0) {
			convertCrossGraphSetGet(setNode, setGraph, crossGraphGetters);
		}
	}

	// Second pass: handle remaining same-graph pairs
	for (const { node: setNode, graph: g } of [...findAllNodesOfType(rootGraph, 'SetNode')]) {
		convertSetGetToLinks(g, setNode);
	}
}

// Compat shim: older frontends may lack graph.getLink(), and _links may be a Map or
// plain object depending on the litegraph version. The == null check catches both null
// and undefined intentionally.
function getLink(graph, linkId) {
	if (linkId == null) return null;
	if (graph.getLink) return graph.getLink(linkId);
	return graph._links instanceof Map ? graph._links.get(linkId) : graph._links?.[linkId] ?? null;
}

// Collect {targetId, targetSlot} from an output slot's links
function collectOutputConnections(graph, output) {
	const connections = [];
	if (output?.links) {
		for (const linkId of [...output.links]) {
			const link = getLink(graph, linkId);
			if (link) connections.push({ targetId: link.target_id, targetSlot: link.target_slot });
		}
	}
	return connections;
}

function convertCrossGraphSetGet(setNode, setGraph, crossGraphGetters) {
	// Find what's connected to the SetNode's input
	const setInput = setNode.inputs[0];
	if (setInput.link == null) return;
	const sourceLink = getLink(setGraph, setInput.link);
	if (!sourceLink) return;
	const sourceNode = setGraph.getNodeById(sourceLink.origin_id);
	if (!sourceNode) return;
	const sourceSlot = sourceLink.origin_slot;
	const linkType = setNode.inputs[0].type || '*';

	for (const { node: getter, graph: getterGraph } of crossGraphGetters) {
		// Collect getter's downstream connections before removing it
		const connections = collectOutputConnections(getterGraph, getter.outputs[0]);
		if (connections.length === 0) {
			getterGraph.remove(getter);
			continue;
		}

		// Determine direction and create SubgraphInput or SubgraphOutput
		const rootGraph = findRootGraph(setGraph);
		const sgNodeForGetter = findSubgraphNodeFor(rootGraph, getter);
		const sgNodeForSetter = findSubgraphNodeFor(rootGraph, setNode);

		if (sgNodeForGetter && setGraph === rootGraph) {
			// Set in root, Get in subgraph → create SubgraphInput
			const subgraph = sgNodeForGetter.subgraph;
			const inputName = setNode.widgets[0].value || linkType;
			const newInput = subgraph.addInput(inputName, linkType);
			const inputIndex = subgraph.inputs.indexOf(newInput);

			// Connect source → SubgraphNode's new input in root graph
			sourceNode.connect(sourceSlot, sgNodeForGetter, inputIndex);

			// Inside subgraph: connect SubgraphInput slot → getter's targets
			for (const conn of connections) {
				if (conn.targetId === subgraph.outputNode?.id) {
					// GetNode fed a SubgraphOutput — bypass it and connect source directly in parent graph.
					const sgNodeOutput = sgNodeForGetter.outputs[conn.targetSlot];
					if (sgNodeOutput?.links) {
						for (const parentLinkId of [...sgNodeOutput.links]) {
							const parentLink = getLink(rootGraph, parentLinkId);
							if (parentLink) {
								const parentTarget = rootGraph.getNodeById(parentLink.target_id);
								if (parentTarget) {
									sourceNode.connect(sourceSlot, parentTarget, parentLink.target_slot);
								}
							}
						}
					}
					// Remove the now-unnecessary SubgraphOutput
					const sgOutput = subgraph.outputs[conn.targetSlot];
					if (sgOutput) {
						subgraph.removeOutput(sgOutput);
					}
				} else {
					const targetNode = getterGraph.getNodeById(conn.targetId);
					if (targetNode) {
						newInput.connect(targetNode.inputs[conn.targetSlot], targetNode);
					}
				}
			}

			// If no connections used the SubgraphInput, remove it
			if (newInput.linkIds.length === 0) {
				subgraph.removeInput(newInput);
			}

			getterGraph.remove(getter);

		} else if (sgNodeForSetter && getterGraph === rootGraph) {
			// Set in subgraph, Get in root → create SubgraphOutput
			const subgraph = sgNodeForSetter.subgraph;
			const outputName = setNode.widgets[0].value || linkType;
			const newOutput = subgraph.addOutput(outputName, linkType);
			const outputIndex = subgraph.outputs.indexOf(newOutput);

			// Inside subgraph: connect source output → SubgraphOutput slot
			newOutput.connect(sourceNode.outputs[sourceSlot], sourceNode);

			// In root graph: connect SubgraphNode's new output → getter's targets
			for (const conn of connections) {
				const targetNode = getterGraph.getNodeById(conn.targetId);
				if (targetNode) {
					sgNodeForSetter.connect(outputIndex, targetNode, conn.targetSlot);
				}
			}

			getterGraph.remove(getter);

		} else {
			// Both in different subgraphs (sibling) — would need both input and output
			// Skip for now, too complex
			console.warn(`KJNodes: Cannot convert cross-graph Set/Get "${setNode.widgets[0].value}" between sibling subgraphs`);
		}
	}

	// Remove SetNode if all its getters have been handled (check if any same-graph getters remain)
	const remainingGetters = setGraph._nodes.filter(
		n => n.type === 'GetNode' && n.widgets[0].value === setNode.widgets[0].value
	);
	if (remainingGetters.length === 0) {
		setGraph.remove(setNode);
	}
}

function convertSetGetToLinks(graph, setNode) {
	if (!graph || !setNode) return;

	const name = setNode.widgets[0].value;
	let sameGraphGetters = [];
	if (name) {
		const allGetters = findGettersByName(graph, name);
		const crossGetters = [];
		for (const entry of allGetters) {
			if (entry.graph === graph) sameGraphGetters.push(entry.node);
			else crossGetters.push(entry);
		}
		if (crossGetters.length > 0) {
			convertCrossGraphSetGet(setNode, graph, crossGetters);
		}
	}

	// Find the source connected to the Set node's input
	const setInput = setNode.inputs[0];
	if (setInput.link == null) return;
	const sourceLink = getLink(graph, setInput.link);
	if (!sourceLink) return;
	const sourceNode = graph.getNodeById(sourceLink.origin_id);
	if (!sourceNode) return;
	const sourceSlot = sourceLink.origin_slot;

	// Collect all consumer connections from Get nodes and SetNode's own output passthrough
	const connections = [];
	for (const getter of sameGraphGetters) {
		connections.push(...collectOutputConnections(graph, getter.outputs[0]));
	}
	connections.push(...collectOutputConnections(graph, setNode.outputs[0]));

	// Remove all Get nodes (this also removes their links)
	for (const getter of sameGraphGetters) {
		graph.remove(getter);
	}
	// Remove the Set node
	graph.remove(setNode);

	// Create direct links from source to each consumer
	for (const conn of connections) {
		const targetNode = graph.getNodeById(conn.targetId);
		if (targetNode) {
			sourceNode.connect(sourceSlot, targetNode, conn.targetSlot);
		}
	}

	app.canvas?.setDirty(true, true);
}

// region SetNode
app.registerExtension({
	name: "KJNodes.SetNode",
	registerCustomNodes() {
		class SetNode extends LGraphNode {
			static title = "Set";
			static category = "KJNodes";
			serialize_widgets = true;
			drawConnection = false;
			slotColor = "#FFF";
			canvas = app.canvas;

			constructor(title) {
				super(title)
				if (!this.properties) {
					this.properties = {
						"previousName": ""
					};
				}
				this.properties["Node name for S&R"] = "SetNode";
				this.properties["aux_id"] = "kijai/ComfyUI-KJNodes";
				this.isVirtualNode = true; // This node is purely frontend and does not impact the resulting prompt so should not be serialized

				this.addWidget(
					"text",
					"Constant",
					'',
					() => {
						if (!this.graph || app.configuringGraph) return;
						this.validateName(this.graph);
						if (this.widgets[0].value !== '') {
							this.title = prefixedTitle("Set", this.widgets[0].value);
						}
						this.update();
						this.properties.previousName = this.widgets[0].value;
					},
					{}
				)

				this.addInput("*", "*");
				this.addOutput("*", '*');
			}

			onConnectionsChange(
				slotType,
				slot,
				isChangeConnect,
				link_info
			) {
				// During graph load, slots are restored by configure — skip side effects
				if (app.configuringGraph) return;
				//On Disconnect
				if (slotType === LiteGraph.INPUT && !isChangeConnect) {
					const outputConnected = this.outputs[0]?.links?.length > 0;
					if (outputConnected) {
						this.inputs[slot].type = this.outputs[0].type;
						this.inputs[slot].name = this.outputs[0].name;
					} else {
						this.inputs[slot].type = '*';
						this.inputs[slot].name = '*';
						this.outputs[0].type = '*';
						this.outputs[0].name = '*';
						this.title = "Set";
						this.color = null;
						this.bgcolor = null;
					}
					this.update();
				}
				if (slotType === LiteGraph.OUTPUT && !isChangeConnect) {
					if (this.outputs && this.outputs[slot]) {
						// Keep type if input has a real connection
						const inputConnected = this.inputs[0]?.link != null;
						if (inputConnected) {
							this.outputs[slot].type = this.inputs[0].type;
							this.outputs[slot].name = this.inputs[0].name;
						} else {
							this.inputs[0].type = '*';
							this.inputs[0].name = '*';
							this.outputs[slot].type = '*';
							this.outputs[slot].name = '*';
							this.color = null;
							this.bgcolor = null;
						}
					}
				}
				//On Connect
				if (link_info && this.graph && slotType === LiteGraph.INPUT && isChangeConnect) {
					const resolve = link_info.resolve(this.graph)
					const resolvedSlot = resolve?.subgraphInput ?? resolve?.output;
					const type = resolvedSlot?.type;
					if (type) {
						if (this.title === "Set"){
							this.title = prefixedTitle("Set", type);
						}
						if (this.widgets[0].value === '' || this.widgets[0].value === '*'){
							// Determine the initial widget value based on naming setting
							const namingMode = app.ui.settings.getSettingValue("KJNodes.setGetNaming") ?? "empty";
							if (namingMode !== "empty") {
								const link = getLink(this.graph, this.inputs[0]?.link);
								const sourceNode = link ? this.graph.getNodeById(link.origin_id) : null;
								const slotName = sourceNode?.outputs?.[link?.origin_slot]?.name || type;
								switch (namingMode) {
									case "slot name": this.widgets[0].value = slotName; break;
									case "slot name (lowercase)": this.widgets[0].value = slotName.toLowerCase(); break;
									case "slot name (UPPERCASE)": this.widgets[0].value = slotName.toUpperCase(); break;
								}
							}
						}

						this.validateName(this.graph);
						this.properties.previousName = this.widgets[0].value;
						this.inputs[0].type = type;
						this.inputs[0].name = type;
						this.outputs[0].type = type;
						this.outputs[0].name = type;

						autoColor(this, type);
					} else {
						showAlert(`node ${this.title} input undefined.`, this)
					}
				}
				if (link_info && this.graph && slotType === LiteGraph.OUTPUT && isChangeConnect) {
					const inputType = this.inputs[0]?.type;
					if (inputType && inputType !== '*') {
						this.outputs[0].type = inputType;
						this.outputs[0].name = inputType;
					} else {
						const resolve = link_info.resolve(this.graph);
						const type = resolve?.input?.type;
						if (type && type !== '*') {
							this.inputs[0].type = type;
							this.inputs[0].name = type;
							this.outputs[0].type = type;
							this.outputs[0].name = type;
							autoColor(this, type);
						}
					}
				}

				//Update either way
				this.update();
			}

			// Returns true if the name was changed
			validateName(graph, sameGraphOnly) {
				let widgetValue = this.widgets[0].value;

				if (widgetValue !== '') {
					let tries = 0;
					const existingValues = new Set();

					// sameGraphOnly: only check the immediate graph (for paste/clone)
					// otherwise check full scope (own graph + ancestors)
					const scopeGraphs = sameGraphOnly ? [graph] : getGraphAncestors(graph);
					const scopedSetNodes = collectNodesOfType(scopeGraphs, 'SetNode');
					scopedSetNodes.forEach(entry => {
						if (entry.node !== this) {
							existingValues.add(entry.node.widgets[0].value);
						}
					});

					const originalValue = widgetValue;
					// Only strip _N suffix during paste to avoid FOO_0_1_2 accumulation.
					// For manual renames, keep the full name as base (user may intend FOO_3).
					const baseName = this._justAdded ? widgetValue.replace(/_\d+$/, '') : widgetValue;
					while (existingValues.has(widgetValue)) {
						widgetValue = baseName + "_" + tries;
						tries++;
					}

					this.widgets[0].value = widgetValue;
					this.title = prefixedTitle("Set", widgetValue);
					return widgetValue !== originalValue;
				}
				return false;
			}

			clone() {
				const cloned = super.clone();
				cloned.inputs[0].name = '*';
				cloned.inputs[0].type = '*';
				cloned.properties.previousName = '';
				cloned.size = cloned.computeSize();
				return cloned;
			}

			onAdded() {
				this._justAdded = true;
			}

			onConfigure() {
				// Only run paste logic when actually pasting, not during workflow load
				if (this._justAdded && this.graph && !app.configuringGraph) {
					const oldName = this.widgets[0].value;
					this.validateName(this.graph, true);
					this._justAdded = false;
					const newName = this.widgets[0].value;
					if (newName !== oldName) {
						_pasteRenameMap.set(oldName, newName);
						// Clear the map after this paste cycle
						setTimeout(() => _pasteRenameMap.delete(oldName), 0);
					}
					// Reset type and color on paste — nothing is connected yet
					if (this.inputs[0]?.link == null) {
						this.inputs[0].type = '*';
						this.inputs[0].name = '*';
						this.outputs[0].type = '*';
						this.outputs[0].name = '*';
						this.color = null;
						this.bgcolor = null;
					}
				}
				this._justAdded = false;
			}

			update() {
				if (!this.graph) return;

				const getters = this.findGetters(this.graph);
				getters.forEach(getter => {
					getter.setType(this.inputs[0].type);
				});

				if (this.widgets[0].value && this.properties.previousName) {
					const gettersWithPreviousName = this.findGetters(this.graph, true);
					gettersWithPreviousName.forEach(getter => {
						getter.setName(this.widgets[0].value);
					});
				}
				app.canvas?.setDirty(true, true);
			}

			findGetters(graph, checkForPreviousName) {
				const name = checkForPreviousName ? this.properties.previousName : this.widgets[0].value;
				if (!name || name === '') return [];
				// Scoped: searches own graph + descendant subgraphs
				return findGettersByName(graph, name).map(entry => entry.node);
			}

			getExtraMenuOptions(_, options) {
				options.unshift(
					{
						content: this.drawConnection ? "Hide connections" : "Show connections",
						callback: () => {
							this.drawConnection = !this.drawConnection;
							const linkType = this.inputs[0].type;
							this.slotColor = this.canvas.default_connection_color_byType[linkType];
							this.canvas.setDirty(true, true);
						},
					},
					{
						content: "Hide all connections",
						callback: () => {
							for (const n of this.graph._nodes) {
								if (n.type === "GetNode" || n.type === "SetNode") n.drawConnection = false;
							}
							this.canvas.setDirty(true, true);
						},
					
					},
				);
				options.unshift({
					content: "Convert to links",
					callback: () => {
						const graph = this.graph;
						for (const n of snapshotSelectedNodes(this, 'SetNode')) convertSetGetToLinks(graph, n);
					},
				});
				options.unshift({
					content: "Add paired GetNode",
					callback: () => {
						const graph = this.graph;
						const getNode = LiteGraph.createNode("GetNode");
						if (!getNode) return;
						getNode.pos = [this.pos[0] + this.size[0] + 30, this.pos[1]];
						graph.add(getNode);
						// Set the widget value to match — this drives type, color, and connection
						const name = this.widgets[0].value;
						if (getNode.widgets?.[0]) {
							getNode.widgets[0].value = name;
						}
						getNode.onRename?.();
						app.canvas.selectNode(getNode, false);
						app.canvas.setDirty(true, true);
					},
				});
				// Dynamically add a submenu for all getters
				const allGettersForMenu = this.findGetters(this.graph);
				if (allGettersForMenu.length) {
					let gettersSubmenu = allGettersForMenu.map(getter => {
						const sameGraph = getter.graph === this.graph;
						const sgNode = !sameGraph ? findSubgraphNodeFor(this.graph, getter) : null;
						const label = sameGraph
							? `${getter.title} id: ${getter.id}`
							: `${getter.title} (in subgraph${sgNode ? ': ' + (sgNode.title || sgNode.type) : ''})`;
						return {
							content: label,
							callback: () => {
								if (sameGraph) {
									this.canvas.centerOnNode(getter);
									this.canvas.selectNode(getter, false);
								} else if (sgNode) {
									this.canvas.openSubgraph(sgNode.subgraph, sgNode);
									// Center on the getter after entering the subgraph
									setTimeout(() => {
										this.canvas.centerOnNode(getter);
										this.canvas.selectNode(getter, false);
									}, 0);
								}
								this.canvas.setDirty(true, true);
							},
						};
					});
			
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
		}

		LiteGraph.registerNodeType("SetNode", SetNode);
	},
});

// region GetNode
app.registerExtension({
	name: "KJNodes.GetNode",
	registerCustomNodes() {
		class GetNode extends LGraphNode {
			static title = "Get";
			static category = "KJNodes";
			serialize_widgets = true;
			drawConnection = false;
			currentSetter = null;
			canvas = app.canvas;

			constructor(title) {
				super(title)
				if (!this.properties) {
					this.properties = {};
				}
				this.properties["Node name for S&R"] = "GetNode";
				this.properties["aux_id"] = "kijai/ComfyUI-KJNodes";
				this.isVirtualNode = true; // This node is purely frontend and does not impact the resulting prompt so should not be serialized
				const comboOptions = {
					getOptionLabel: (value) => {
						if (!value) return "";
						const source = _setNameSourceMap.get(value);
						if (!source || source === "local") return value;
						return `${value} (${source})`;
					},
				};
				Object.defineProperty(comboOptions, 'values', {
					get: () => {
						if (!this.graph) return [];
						let filterType = null;
						if (app.ui.settings.getSettingValue("KJNodes.filterGetNodeOptions") !== false
							&& this.outputs[0]?.links?.length) {
							const linkId = this.outputs[0].links[0];
							const link = getLink(this.graph, linkId);
							if (link) {
								const targetNode = this.graph.getNodeById(link.target_id);
								filterType = targetNode?.inputs?.[link.target_slot]?.type || null;
							}
						}
						return getVisibleSetNames(this.graph, filterType);
					},
					enumerable: true,
					configurable: true
				});
				this.addWidget("combo", "Constant", "", () => { if (!app.configuringGraph) this.onRename(); }, comboOptions)
				this.addOutput("*", '*');
			}

			onConnectionsChange() {
				if (app.configuringGraph) return;
				this.validateLinks();
			}

			setName(name) {
				this.widgets[0].value = name;
				this.onRename();
				this.serialize();
			}

			onRename() {
				const setter = this.findSetter(this.graph);
				if (setter) {
					this.setType(setter.inputs[0].type);
					this.title = prefixedTitle("Get", setter.widgets[0].value);
				} else {
					this.setType('*');
					const name = this.widgets[0].value;
					this.title = name ? prefixedTitle("Get", name) : "Get";
				}
				app.canvas?.setDirty(true, true);
			}

			clone() {
				const cloned = super.clone();
				cloned.size = cloned.computeSize();
				return cloned;
			}

			validateLinks() {
				if (this.outputs[0].type !== '*' && this.outputs[0].links && this.graph) {
					this.outputs[0].links.filter(linkId => {
						const link = getLink(this.graph, linkId);
						if (!link || !link.type) return false;
						if (link.type === '*') return false;
						const targetNode = this.graph.getNodeById(link.target_id);
						const targetType = targetNode?.inputs?.[link.target_slot]?.type;
						if (targetType === '*') return false;
						if (targetType) {
							const targetTypes = String(targetType).split(",");
							if (targetTypes.includes(this.outputs[0].type)) return false;
						}
						return !link.type.split(",").includes(this.outputs[0].type);
					}).forEach(linkId => {
						this.graph.removeLink(linkId);
					});
				}
			}

			setType(type) {
				this.outputs[0].name = type;
				this.outputs[0].type = type;
				this.validateLinks();
				autoColor(this, type);
			}

			findSetter(graph) {
				const name = this.widgets[0].value;
				// Scoped: searches own graph first, then ancestors
				const result = findSetterByName(graph, name);
				return result ? result.node : undefined;
			}

			goToSetter() {
				if (!this.currentSetter) return;
				const setterGraph = this.currentSetter.graph;
				if (setterGraph && setterGraph !== this.graph) {
					this.canvas.setGraph(setterGraph);
					setTimeout(() => {
						this.canvas.centerOnNode(this.currentSetter);
						this.canvas.selectNode(this.currentSetter, false);
						this.canvas.setDirty(true, true);
					}, 0);
				} else {
					this.canvas.centerOnNode(this.currentSetter);
					this.canvas.selectNode(this.currentSetter, false);
				}
			}

			onAdded() {
				this._justAdded = true;
			}

			onConfigure() {
				if (this._justAdded && !app.configuringGraph) {
					const name = this.widgets[0].value;
					if (name) {
						// Check if our paired SetNode was renamed during this paste
						const newName = _pasteRenameMap.get(name);
						if (newName) {
							this.widgets[0].value = newName;
						}
						// Restore type/color from setter after paste
						setTimeout(() => this.onRename(), 0);
					}
				}
				this._justAdded = false;
			}

			getInputLink(slot) {
				// Same-graph: standard resolveOutput path.
				// Cross-graph: handled by resolveVirtualOutput.
				const name = this.widgets[0].value;
				if (!name || name === '') return null;
				const setter = this.graph?._nodes?.find(
					n => n.type === 'SetNode' && n.widgets[0].value === name
				);
				if (setter) {
					const slotInfo = setter.inputs[slot];
					if (!slotInfo || slotInfo.link == null) return null;
					return getLink(this.graph, slotInfo.link);
				}
				// Scoped lookup: no setter found in own graph or ancestors
				if (name && !findSetterByName(this.graph, name)) {
					showAlert("No SetNode found for " + name + "(" + this.type + ")", this);
				}
				return null;
			}

			resolveVirtualOutput(slot) {
				const name = this.widgets[0].value;
				// Scoped lookup: own graph first, then ancestors
				const result = findSetterByName(this.graph, name);
				if (!result) return undefined;

				// Same graph — let the standard getInputLink path handle it
				if (result.graph === this.graph) return undefined;

				// Warn if multiple SetNodes with this name exist in scope
				const scopeGraphs = getGraphAncestors(this.graph);
				const scopedSetters = collectNodesOfType(scopeGraphs, 'SetNode')
					.filter(e => e.node.widgets[0].value === name);
				if (scopedSetters.length > 1) {
					showAlert(`Multiple SetNodes named "${name}" found in scope. Rename duplicates or use "Convert to links" to resolve`, [this, ...scopedSetters.map(e => e.node)]);
					return undefined;
				}

				const { node: setter, graph: setterGraph } = result;
				const slotInfo = setter.inputs[slot];
				if (!slotInfo || slotInfo.link == null) return undefined;

				const link = getLink(setterGraph, slotInfo.link);
				if (!link) return undefined;

				const sourceNode = setterGraph.getNodeById(link.origin_id);
				if (!sourceNode) return undefined;

				return { node: sourceNode, slot: link.origin_slot };
			}
			getExtraMenuOptions(_, options) {
				this.currentSetter = this.findSetter(this.graph)
				if (!this.currentSetter) return
				const sameGraph = this.currentSetter.graph === this.graph;
				if (sameGraph) {
					let menuEntry = this.drawConnection ? "Hide connections" : "Show connections";
					options.unshift(
						{
							content: "Convert to links",
							callback: () => {
								const graph = this.graph;
								const setters = new Set(snapshotSelectedNodes(this, 'GetNode')
									.map(n => n.findSetter?.(graph)).filter(Boolean));
								for (const s of setters) convertSetGetToLinks(graph, s);
							},
						},
						{
							content: "Go to setter",
							callback: () => {
								this.goToSetter();
							},
						},
						{
							content: menuEntry,
							callback: () => {
								if (!this.currentSetter) return;
								const linkType = this.currentSetter.inputs[0].type;
								// Toggle on the SetNode so the canvas-level hook draws it
								this.currentSetter.drawConnection = !this.currentSetter.drawConnection;
								this.currentSetter.slotColor = this.canvas.default_connection_color_byType[linkType];
								this.drawConnection = this.currentSetter.drawConnection;
								this.canvas.setDirty(true, true);
							},
						},
					);
				} else {
					// Cross-graph setter — navigate to its graph
					const setterGraph = this.currentSetter.graph;
					const isRoot = setterGraph === findRootGraph(this.graph);
					options.unshift(
						{
							content: `Go to setter (in ${isRoot ? 'parent graph' : 'subgraph'})`,
							callback: () => {
								const canvas = this.canvas;
								canvas.setGraph(setterGraph);
								setTimeout(() => {
									canvas.centerOnNode(this.currentSetter);
									canvas.selectNode(this.currentSetter, false);
									canvas.setDirty(true, true);
								}, 0);
							},
						},
					);
				}
			}
		}

		LiteGraph.registerNodeType("GetNode", GetNode);
	},
});

// region UI: commands, keybindings, canvas menu, draw hook, settings
app.registerExtension({
	name: "KJNodes.SetGetUI",
	settings: [
		{
			id: "KJNodes.setGetNaming",
			name: "Default SetNode widget value",
			category: ["KJNodes", "Set & Get", "Default SetNode widget value"],
			tooltip: "Initial Constant value when a Set node is first connected to a slot",
			type: "combo",
			options: ["empty", "slot name", "slot name (lowercase)", "slot name (UPPERCASE)"],
			defaultValue: "empty",
		},
		{
			id: "KJNodes.showSetGetLinks",
			name: "Show links",
			category: ["KJNodes", "Set & Get", "Show links"],
			tooltip: "When to show virtual links between Set/Get pairs",
			type: "combo",
			options: ["never", "selected", "always"],
			defaultValue: "never",
			onChange: () => app.canvas?.setDirty(true, true),
		},
	],
	commands: [
		{
			id: "KJNodes.AddSetNodeToSelected",
			label: "Add Set node to selected / at cursor",
			function: () => addNodeToSelectedOrCursor("SetNode", "right"),
		},
		{
			id: "KJNodes.AddGetNodeAtCursor",
			label: "Add Get node to selected / at cursor",
			function: () => addNodeToSelectedOrCursor("GetNode", "left"),
		},
		{
			id: "KJNodes.ToggleForceShowSetGetLinks",
			label: "Toggle force-show all Set/Get connections",
			function: () => {
				_forceShowAllLinks = !_forceShowAllLinks;
				app.canvas.setDirty(true, true);
			},
		},
	],
	keybindings: [
		{
			commandId: "KJNodes.AddSetNodeToSelected",
			combo: { key: "s", ctrl: true, shift: true },
			targetElementId: "graph-canvas",
		},
		{
			commandId: "KJNodes.AddGetNodeAtCursor",
			combo: { key: "g", ctrl: true, shift: true },
			targetElementId: "graph-canvas",
		},
		{
			commandId: "KJNodes.ToggleForceShowSetGetLinks",
			combo: { key: "l", ctrl: true, shift: true },
			targetElementId: "graph-canvas",
		},
	],
	getCanvasMenuItems() {
		return [
			{
				content: "KJNodes",
				has_submenu: true,
				submenu: {
					options: [
						{
							content: "Convert outputs on all selected nodes to Set/Get",
							callback: () => {
								const selected = Object.values(app.canvas.selected_nodes || {});
								if (selected.length === 0) return;
								for (const n of selected) convertOutputsToSetGet(app.graph, n);
							},
						},
						{
							content: "Convert selected Set/Get to links",
							callback: () => {
								const graph = app.graph;
								const selected = Object.values(app.canvas.selected_nodes || {});
								const setNodes = selected.filter(n => n.type === 'SetNode');
								const getNodes = selected.filter(n => n.type === 'GetNode');
								for (const n of setNodes) convertSetGetToLinks(graph, n);
								const setters = new Set(getNodes.map(n => n.findSetter?.(graph)).filter(Boolean));
								for (const s of setters) convertSetGetToLinks(graph, s);
							},
						},
					],
				},
			},
		];
	},
	setup() {
		// Double-click GetNode to jump to its SetNode (works in both legacy and Vue modes)
		document.addEventListener("dblclick", (e) => {
			// Don't interfere when a ContextMenu (e.g. combo dropdown) is open
			if (document.querySelector(".litecontextmenu")) return;
			const canvas = app.canvas;
			if (!canvas) return;
			const selected = Object.values(canvas.selected_nodes || {});
			if (selected.length !== 1) return;
			const node = selected[0];
			if (node.type === "GetNode") {
				node.currentSetter = node.findSetter(node.graph);
				if (node.currentSetter) node.goToSetter();
			}
		});
		// Suppress title rename on collapsed GetNodes so double-click navigates instead
		document.addEventListener("litegraph:canvas", (e) => {
			if (e.detail?.subType !== 'node-double-click') return;
			const node = e.detail.node;
			if (node?.type === "GetNode" && node.flags?.collapsed) {
				e.stopImmediatePropagation();
			}
		});

		// Monkey-patch: no extension hook exists for filtering visible nodes.
		// Keep Set/Get nodes with visible virtual connections in the visible set
		// even when offscreen, so the link lines don't get clipped.
		const originalComputeVisibleNodes = LGraphCanvas.prototype.computeVisibleNodes;
		LGraphCanvas.prototype.computeVisibleNodes = function () {
			const visible = originalComputeVisibleNodes.apply(this, arguments);
			for (const node of this.graph._nodes) {
				if ((node.type === "SetNode" || node.type === "GetNode") && node.drawConnection && !visible.includes(node)) {
					visible.push(node);
				}
			}
			return visible;
		};

		// Monkey-patch: no extension hook exists for custom canvas drawing.
		// Draws virtual links between Set/Get pairs.
		// Handles per-node "Show connections", "selected" mode, and "always" mode.
		const origCanvasOnDrawBackground = app.canvas.onDrawBackground;
		app.canvas.onDrawBackground = function(ctx, visibleArea) {
			origCanvasOnDrawBackground?.call(this, ctx, visibleArea);

			const graph = this.graph || app.graph;
			if (!graph?._nodes) return;
			const mode = getShowLinksMode();

			// Build set of selected Set/Get names for "selected" mode
			let selectedNames = null;
			if (mode === "selected") {
				const sel = Object.values(this.selected_nodes || {});
				if (!sel.length) {
					// No selection — still honor per-node drawConnection
					if (!graph._nodes.some(n => n.type === 'SetNode' && n.drawConnection)) return;
				} else {
					selectedNames = new Set();
					for (const n of sel) {
						if (n.type === 'SetNode' || n.type === 'GetNode') {
							selectedNames.add(n.widgets?.[0]?.value ?? n.title);
						}
					}
					if (!selectedNames.size && !graph._nodes.some(n => n.type === 'SetNode' && n.drawConnection)) return;
				}
			} else if (mode === "never") {
				if (!graph._nodes.some(n => n.type === 'SetNode' && n.drawConnection)) return;
			}

			// Collect all GetNodes once across current graph + descendants, grouped by name.
			// This avoids re-discovering descendant graphs and re-scanning nodes per SetNode.
			const gettersByName = new Map();
			const scopeGraphs = [graph, ...getGraphDescendants(graph)];
			for (const g of scopeGraphs) {
				if (!g?._nodes) continue;
				for (const node of g._nodes) {
					if (node.type !== 'GetNode') continue;
					const n = node.widgets?.[0]?.value;
					if (!n) continue;
					let list = gettersByName.get(n);
					if (!list) { list = []; gettersByName.set(n, list); }
					list.push(node);
				}
			}

			for (const setNode of graph._nodes) {
				if (setNode.type !== 'SetNode') continue;
				const name = setNode.widgets?.[0]?.value ?? setNode.title;
				const showByMode = mode === "always" || (mode === "selected" && selectedNames?.has(name));
				if (!showByMode && !setNode.drawConnection) continue;

				const allGetters = gettersByName.get(name) || [];
				const drawTargets = [];
				const seenSubgraphs = new Set();
				for (const getter of allGetters) {
					if (getter.graph === graph) {
						drawTargets.push(getter);
					} else {
						const sgNode = findSubgraphNodeFor(graph, getter);
						if (sgNode && !seenSubgraphs.has(sgNode)) {
							seenSubgraphs.add(sgNode);
							drawTargets.push(sgNode);
						}
					}
				}
				if (!drawTargets.length) continue;

				const linkType = setNode.inputs[0]?.type;
				const slotColor = app.canvas.default_connection_color_byType?.[linkType]
					|| LGraphCanvas.link_type_colors?.[linkType]
					|| setNode.bgcolor
					|| (setNode.slotColor !== "#FFF" ? setNode.slotColor : null) || "#AAA";

				const startPos = setNode.getConnectionPos(false, 0);

				for (const target of drawTargets) {
					const endPos = target.getConnectionPos(true, 0);
					const highlighted = setNode.is_selected || target.is_selected;
					const color = highlighted ? "#FFF" : slotColor;
					app.canvas.renderLink(
						ctx,
						startPos,
						endPos,
						null,
						false,
						null,
						color,
						LiteGraph.RIGHT,
						LiteGraph.LEFT
					);
				}
			}
		};

		app.ui.settings.addSetting({
			id: "KJNodes.filterGetNodeOptions",
			name: "Filter Get node options by type",
			category: ["KJNodes", "Set & Get", "Filter Get node options by type"],
			tooltip: "When a Get node is connected, only show Set nodes with compatible types in the dropdown",
			type: "boolean",
			defaultValue: true,
		});
		app.ui.settings.addSetting({
			id: "KJNodes.convertAllSetGet",
			name: "Convert ALL Set/Get to links",
			category: ["KJNodes", "Set & Get", "Convert ALL Set/Get to links"],
			tooltip: "Replaces all Set/Get node pairs with direct links, including across subgraph boundaries (irreversible)",
			type: () => {
				const btn = document.createElement("button");
				btn.textContent = "Convert ALL Set/Get to links";
				btn.style.cssText = "padding: 6px 12px; cursor: pointer;";
				btn.onclick = () => {
					if (confirm("This will replace ALL Set/Get pairs with direct links. This is irreversible. Continue?")) {
						convertAllSetGetToLinks(app.graph);
						app.canvas?.setDirty(true, true);
						app.extensionManager.toast.add({
							severity: 'info',
							summary: "KJ Set/Get",
							detail: "All Set/Get nodes converted to direct links",
							life: 3000,
						});
					}
				};
				return btn;
			},
			defaultValue: false,
		});
	}
});

// Temporary backwards-compat patch: makes cross-subgraph Set/Get work on frontends
// that don't natively support resolveVirtualOutput. Remove when no longer needed.
app.registerExtension({
	name: "KJNodes.CrossGraphSetGet",
	setup() {
		let patched = false;

		const originalGraphToPrompt = app.graphToPrompt.bind(app);
		app.graphToPrompt = async function(...args) {
			if (!patched) {
				try {
					const subgraphNode = app.graph._nodes.find(n => typeof n.getInnerNodes === 'function');
					if (subgraphNode) {
						const tempMap = new Map();
						const dtos = subgraphNode.getInnerNodes(tempMap, []);
						if (dtos.length > 0) {
							const proto = Object.getPrototypeOf(dtos[0]);
							const DtoClass = proto.constructor;
							const nativeSource = proto.resolveOutput.toString();
							const hasNativeSupport = nativeSource.includes('resolveVirtualOutput');
							console.log(`[KJNodes] Cross-graph Set/Get: frontend native support ${hasNativeSupport ? 'detected, skipping patch' : 'not found, applying patch'}`);
							if (!hasNativeSupport) {
								const origResolveOutput = proto.resolveOutput;
								proto.resolveOutput = function(slot, type, visited) {
									if (typeof this.node?.resolveVirtualOutput === 'function') {
										const virtualSource = this.node.resolveVirtualOutput(slot);
										if (virtualSource) {
											const inputNodeDto = [...this.nodesByExecutionId.values()]
												.find(dto => dto instanceof DtoClass && dto.node === virtualSource.node);
											if (inputNodeDto) {
												return inputNodeDto.resolveOutput(virtualSource.slot, type, visited);
											}
											throw new Error(`KJNodes: No DTO found for cross-graph source node [${virtualSource.node.id}]`);
										}
									}
									return origResolveOutput.call(this, slot, type, visited);
								};
							}
							patched = true;
						}
					}
				} catch (e) {
					console.warn('[KJNodes] Failed to probe ExecutableNodeDTO for cross-graph patch:', e);
				}
			}
			return originalGraphToPrompt(...args);
		};
	}
});

// Monkey-patch: no hook exists to extend the link context menu.
// Appends a "Convert to Set/Get" entry by counting DOM menus before/after
// the original call to find the newly created one. Fragile but unavoidable.
app.registerExtension({
	name: "KJNodes.LinkToSetGet",
	setup() {
		const originalShowLinkMenu = LGraphCanvas.prototype.showLinkMenu;
		LGraphCanvas.prototype.showLinkMenu = function(segment, e) {
			const menusBefore = document.querySelectorAll(".litecontextmenu").length;
			const result = originalShowLinkMenu.call(this, segment, e);

			const graph = this.graph;
			if (!graph) return result;

			// Get the full link object to access target info
			const link = getLink(graph, segment.id);
			if (!link || link.origin_id == null || link.target_id == null) return result;

			// Find the newly created context menu
			const menus = document.querySelectorAll(".litecontextmenu");
			if (menus.length <= menusBefore) return result;
			const lastMenu = menus[menus.length - 1];
			if (!lastMenu) return result;

			const entries = lastMenu.querySelector(".litemenu-entry")?.parentElement;
			if (!entries) return result;

			// Create separator
			const separator = document.createElement("div");
			separator.className = "litemenu-entry separator";
			entries.appendChild(separator);

			// Create menu entry
			const menuItem = document.createElement("div");
			menuItem.className = "litemenu-entry submenu";
			menuItem.textContent = "Convert to Set/Get";
			entries.appendChild(menuItem);

			const canvas = this;
			menuItem.addEventListener("click", function() {
				// Close menu
				lastMenu.remove();

				const originNode = graph.getNodeById(link.origin_id);
				const targetNode = graph.getNodeById(link.target_id);
				if (!originNode || !targetNode) return;

				const outputSlot = originNode.outputs[link.origin_slot];
				const linkType = outputSlot?.type || "*";
				const linkName = outputSlot?.name || linkType;

				// Create Set node near the source
				const setNode = LiteGraph.createNode("SetNode");
				if (!setNode) return;
				setNode.pos = [
					originNode.pos[0] + originNode.size[0] + 30,
					originNode.pos[1]
				];
				graph.add(setNode);

				// Create Get node near the target
				const getNode = LiteGraph.createNode("GetNode");
				if (!getNode) return;
				graph.add(getNode);
				getNode.pos = [
					targetNode.pos[0] - getNode.size[0] - 30,
					targetNode.pos[1]
				];

				// Remove the original link
				graph.removeLink(link.id);

				// Connect source -> Set node input
				originNode.connect(link.origin_slot, setNode, 0);

				// Set the name widget on the Set node
				setNode.widgets[0].value = linkName;
				setNode.title = prefixedTitle("Set", linkName);
				setNode.validateName(graph);
				setNode.properties.previousName = setNode.widgets[0].value;

				// Update Get node to match
				const finalName = setNode.widgets[0].value;
				getNode.widgets[0].value = finalName;
				getNode.onRename();

				// Connect Get node output -> target node input
				getNode.connect(0, targetNode, link.target_slot);

				canvas.setDirty(true, true);
			});

			return result;
		};
	}
});
