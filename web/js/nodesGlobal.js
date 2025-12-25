import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("🔧 [NodesGlobal] Loading...");

const TYPE_COLORS = {
    "IMAGE": "#64B5F6",
    "MASK": "#81C784",
    "LATENT": "#FF8A65",
    "CONDITIONING": "#FFA931",
    "MODEL": "#B39DDB",
    "CLIP": "#FFD500",
    "VAE": "#FF6E6E",
    "CONTROL_NET": "#00D4AA",
    "STRING": "#77DD77",
    "INT": "#7EC8E3",
    "FLOAT": "#CDB4DB",
    "BOOLEAN": "#FFCC80",
    "*": "#AAAAAA",
};

const varTypes = {};

// ═══════════════════════════════════════════════════════════════════════════
// HIDE INTERNAL SLOTS
// ═══════════════════════════════════════════════════════════════════════════
function hideInternalSlots(node) {
    if (node.type === "SetNodeGlobal" && node.outputs) {
        node.outputs = node.outputs.filter((o, idx) => idx === 0);
    }
    
    if (node.type === "GetNodeGlobal" && node.inputs) {
        node.inputs = node.inputs.filter(i => i.name !== "_dep");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════
function getInputTypeFromNode(node) {
    if (!node.inputs) return "*";
    
    for (const input of node.inputs) {
        if (input.link != null && input.name === "value") {
            const link = app.graph.links[input.link];
            if (link) {
                const srcNode = app.graph.getNodeById(link.origin_id);
                if (srcNode?.outputs?.[link.origin_slot]) {
                    const t = srcNode.outputs[link.origin_slot].type;
                    if (t && t !== "*" && !t.startsWith("KJNODES_")) {
                        return t;
                    }
                }
            }
        }
    }
    return "*";
}

function findSetNodeByVar(varName, targetOrder = null) {
    if (!app.graph?._nodes) return null;
    
    let bestMatch = null;
    let bestOrder = -1;
    
    for (const node of app.graph._nodes) {
        if (node.type === "SetNodeGlobal") {
            const nameWidget = node.widgets?.find(w => w.name === "variable_name");
            const orderWidget = node.widgets?.find(w => w.name === "order");
            
            if (nameWidget?.value === varName) {
                const nodeOrder = orderWidget?.value ?? 0;
                
                if (targetOrder === null) {
                    // Return highest order Set
                    if (nodeOrder > bestOrder) {
                        bestOrder = nodeOrder;
                        bestMatch = node;
                    }
                } else {
                    // Return Set with order <= targetOrder, preferring highest
                    if (nodeOrder <= targetOrder && nodeOrder > bestOrder) {
                        bestOrder = nodeOrder;
                        bestMatch = node;
                    }
                }
            }
        }
    }
    return bestMatch;
}

function getVarType(varName, order = null) {
    if (!varName) return "*";
    const setNode = findSetNodeByVar(varName, order);
    if (setNode) {
        const t = getInputTypeFromNode(setNode);
        if (t !== "*") {
            varTypes[varName] = t;
            return t;
        }
    }
    return varTypes[varName] || "*";
}

function updateSlotType(node, slotIdx, typeName) {
    if (!node.outputs?.[slotIdx]) return;
    const slot = node.outputs[slotIdx];
    const oldType = slot.type;
    
    slot.type = typeName;
    slot.name = typeName === "*" ? "value" : typeName;
    slot.color_on = TYPE_COLORS[typeName] || TYPE_COLORS["*"];
    slot.color_off = slot.color_on;
    
    if (oldType !== typeName) {
        node.setDirtyCanvas(true, true);
    }
}

function updateAllGetNodes(varName, typeName) {
    if (!app.graph?._nodes) return;
    for (const node of app.graph._nodes) {
        if (node.type === "GetNodeGlobal") {
            const w = node.widgets?.find(w => w.name === "variable_name");
            if (w?.value === varName) {
                updateSlotType(node, 0, typeName);
            }
        }
    }
}

function refreshAllTypes() {
    if (!app.graph?._nodes) return;
    
    // Collect types from all Set nodes
    const varTypesByOrder = {};  // varName -> {order: type}
    
    for (const node of app.graph._nodes) {
        if (node.type === "SetNodeGlobal") {
            const nameWidget = node.widgets?.find(w => w.name === "variable_name");
            const orderWidget = node.widgets?.find(w => w.name === "order");
            
            if (nameWidget?.value) {
                const t = getInputTypeFromNode(node);
                const order = orderWidget?.value ?? 0;
                
                if (!varTypesByOrder[nameWidget.value]) {
                    varTypesByOrder[nameWidget.value] = {};
                }
                varTypesByOrder[nameWidget.value][order] = t;
                
                updateSlotType(node, 0, t);
            }
        }
    }
    
    // Update varTypes with highest order type
    for (const [varName, orderTypes] of Object.entries(varTypesByOrder)) {
        const maxOrder = Math.max(...Object.keys(orderTypes).map(Number));
        varTypes[varName] = orderTypes[maxOrder];
    }
    
    // Update Get nodes based on their order
    for (const node of app.graph._nodes) {
        if (node.type === "GetNodeGlobal") {
            const nameWidget = node.widgets?.find(w => w.name === "variable_name");
            const orderWidget = node.widgets?.find(w => w.name === "order");
            
            if (nameWidget?.value) {
                const order = orderWidget?.value ?? 0;
                const orderTypes = varTypesByOrder[nameWidget.value] || {};
                
                // Find type from Set with highest order <= this Get's order
                let matchingType = "*";
                let bestOrder = -1;
                for (const [setOrder, setType] of Object.entries(orderTypes)) {
                    const so = Number(setOrder);
                    if (so <= order && so > bestOrder) {
                        bestOrder = so;
                        matchingType = setType;
                    }
                }
                
                updateSlotType(node, 0, matchingType);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PATCH CONNECTION MENU
// ═══════════════════════════════════════════════════════════════════════════
function patchConnectionMenu() {
    const origShowSearchBox = LGraphCanvas.prototype.showSearchBox;
    
    LGraphCanvas.prototype.showSearchBox = function(event, options) {
        if (options?.node_from) {
            const node = options.node_from;
            if (node.type === "SetNodeGlobal" || node.type === "GetNodeGlobal") {
                const nameWidget = node.widgets?.find(w => w.name === "variable_name");
                const orderWidget = node.widgets?.find(w => w.name === "order");
                
                if (nameWidget?.value) {
                    const order = orderWidget?.value ?? 0;
                    const type = getVarType(nameWidget.value, order);
                    if (type !== "*" && options.slot_from) {
                        options.type_filter_in = type;
                        options.slot_from.type = type;
                    }
                }
            }
        }
        return origShowSearchBox.call(this, event, options);
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// NODE SETUP
// ═══════════════════════════════════════════════════════════════════════════
function setupSetNode(node) {
    hideInternalSlots(node);
    
    const nameWidget = node.widgets?.find(w => w.name === "variable_name");
    const orderWidget = node.widgets?.find(w => w.name === "order");
    if (!nameWidget) return;
    
    const origOnConn = node.onConnectionsChange;
    node.onConnectionsChange = function(side, slotIdx, connected, linkInfo, ioSlot) {
        origOnConn?.apply(this, arguments);
        if (side === 1) {
            refreshAllTypes();
        }
    };
    
    // Update on name change
    const origNameCb = nameWidget.callback;
    nameWidget.callback = function(value, ...args) {
        origNameCb?.call(this, value, ...args);
        refreshAllTypes();
    };
    
    // Update on order change
    if (orderWidget) {
        const origOrderCb = orderWidget.callback;
        orderWidget.callback = function(value, ...args) {
            origOrderCb?.call(this, value, ...args);
            refreshAllTypes();
        };
    }
    
    setTimeout(refreshAllTypes, 100);
}

function setupGetNode(node) {
    hideInternalSlots(node);
    
    const nameWidget = node.widgets?.find(w => w.name === "variable_name");
    const orderWidget = node.widgets?.find(w => w.name === "order");
    if (!nameWidget) return;
    
    // Update on name change
    const origNameCb = nameWidget.callback;
    nameWidget.callback = function(value, ...args) {
        origNameCb?.call(this, value, ...args);
        const order = orderWidget?.value ?? 0;
        updateSlotType(node, 0, getVarType(value, order));
    };
    
    // Update on order change
    if (orderWidget) {
        const origOrderCb = orderWidget.callback;
        orderWidget.callback = function(value, ...args) {
            origOrderCb?.call(this, value, ...args);
            updateSlotType(node, 0, getVarType(nameWidget.value, value));
        };
    }
    
    setTimeout(() => {
        const order = orderWidget?.value ?? 0;
        updateSlotType(node, 0, getVarType(nameWidget.value, order));
    }, 100);
}

// ═══════════════════════════════════════════════════════════════════════════
// REGISTER
// ═══════════════════════════════════════════════════════════════════════════
app.registerExtension({
    name: "KJNodes.GlobalVariables",
    
    async setup() {
        patchConnectionMenu();
        
        api.addEventListener("kjnodes.type_update", (e) => {
            const { variable_name, type, order } = e.detail;
            varTypes[variable_name] = type;
            refreshAllTypes();
        });
        
        api.addEventListener("executed", refreshAllTypes);
        setInterval(refreshAllTypes, 2000);
        setTimeout(refreshAllTypes, 500);
    },
    
    nodeCreated(node) {
        if (node.type === "SetNodeGlobal") setupSetNode(node);
        else if (node.type === "GetNodeGlobal") setupGetNode(node);
    },
    
    loadedGraphNode(node) {
        if (node.type === "SetNodeGlobal") setupSetNode(node);
        else if (node.type === "GetNodeGlobal") setupGetNode(node);
    },
});

console.log("✅ [NodesGlobal] Loaded with order support");