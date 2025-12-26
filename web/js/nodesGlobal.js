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
    "CLIP_VISION": "#E6A8D7",
    "STYLE_MODEL": "#C9A0DC",
    "STRING": "#77DD77",
    "INT": "#7EC8E3",
    "FLOAT": "#CDB4DB",
    "BOOLEAN": "#FFCC80",
    "*": "#AAAAAA",
};

const varTypes = {};        // varName -> type
const varTypesByOrder = {}; // varName -> {order: type}

// ═══════════════════════════════════════════════════════════════════════════
// HIDE INTERNAL SLOTS
// ═══════════════════════════════════════════════════════════════════════════
function hideInternalSlots(node) {
    if (node.type === "SetNodeGlobal") {
        if (node.outputs) {
            node.outputs = node.outputs.filter(o => o.name === "value");
        }
        if (node.inputs) {
            node.inputs = node.inputs.filter(i => i.name === "value");
        }
    }
    
    if (node.type === "GetNodeGlobal") {
        if (node.inputs) {
            node.inputs = node.inputs.filter(i => i.name !== "_dep");
        }
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

function getTypeForVar(varName, targetOrder = null) {
    if (!varName) return "*";
    
    const orderTypes = varTypesByOrder[varName] || {};
    const orders = Object.keys(orderTypes).map(Number).sort((a,b) => a-b);
    
    if (orders.length === 0) {
        return varTypes[varName] || "*";
    }
    
    if (targetOrder === null) {
        // Return highest order type
        return orderTypes[orders[orders.length - 1]] || "*";
    }
    
    // Find exact match or highest order < targetOrder
    if (orderTypes[targetOrder] !== undefined) {
        return orderTypes[targetOrder];
    }
    
    const lowerOrders = orders.filter(o => o < targetOrder);
    if (lowerOrders.length > 0) {
        return orderTypes[Math.max(...lowerOrders)];
    }
    
    // Fallback to lowest
    return orderTypes[orders[0]] || "*";
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

// ═══════════════════════════════════════════════════════════════════════════
// REFRESH ALL TYPES
// ═══════════════════════════════════════════════════════════════════════════
function refreshAllTypes() {
    if (!app.graph?._nodes) return;
    
    // Clear order cache
    for (const key of Object.keys(varTypesByOrder)) {
        delete varTypesByOrder[key];
    }
    
    // Collect types from active Set nodes
    for (const node of app.graph._nodes) {
        if (node.type === "SetNodeGlobal" && node.mode !== 4) {
            const nameWidget = node.widgets?.find(w => w.name === "variable_name");
            const orderWidget = node.widgets?.find(w => w.name === "order");
            
            if (nameWidget?.value) {
                const varName = nameWidget.value;
                const order = orderWidget?.value ?? 0;
                const inputType = getInputTypeFromNode(node);
                
                if (!varTypesByOrder[varName]) {
                    varTypesByOrder[varName] = {};
                }
                varTypesByOrder[varName][order] = inputType;
                
                // Update Set node output
                updateSlotType(node, 0, inputType);
            }
        }
    }
    
    // Update Get nodes
    for (const node of app.graph._nodes) {
        if (node.type === "GetNodeGlobal" && node.mode !== 4) {
            const nameWidget = node.widgets?.find(w => w.name === "variable_name");
            const orderWidget = node.widgets?.find(w => w.name === "order");
            
            if (nameWidget?.value) {
                const varName = nameWidget.value;
                const order = orderWidget?.value ?? 0;
                const type = getTypeForVar(varName, order);
                updateSlotType(node, 0, type);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATE CONNECTIONS
// ═══════════════════════════════════════════════════════════════════════════
function validateNode(node) {
    const nameWidget = node.widgets?.find(w => w.name === "variable_name");
    const orderWidget = node.widgets?.find(w => w.name === "order");
    
    if (!nameWidget?.value) {
        node.color = "#663333"; // Dark red - missing variable name
        return;
    }
    
    const varName = nameWidget.value;
    const order = orderWidget?.value ?? 0;
    
    if (node.type === "GetNodeGlobal") {
        // Check if corresponding Set exists
        const sets = varTypesByOrder[varName];
        if (!sets || Object.keys(sets).length === 0) {
            node.color = "#664433"; // Orange-ish - no Set found
            node.title = `🔹 Get (⚠️ No Set for "${varName}")`;
        } else if (sets[order] === undefined) {
            // Has Set but not with this order
            const availableOrders = Object.keys(sets).join(", ");
            node.color = "#665533"; // Yellow-ish - order mismatch
            node.title = `🔹 Get (orders: ${availableOrders})`;
        } else {
            node.color = null;
            node.title = undefined;
        }
    } else {
        node.color = null;
        node.title = undefined;
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
                    const type = getTypeForVar(nameWidget.value, order);
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
function setupNode(node) {
    hideInternalSlots(node);
    
    const nameWidget = node.widgets?.find(w => w.name === "variable_name");
    const orderWidget = node.widgets?.find(w => w.name === "order");
    
    const refresh = () => {
        setTimeout(() => {
            refreshAllTypes();
            validateNode(node);
        }, 50);
    };
    
    // Connection changes
    if (node.type === "SetNodeGlobal") {
        const origOnConn = node.onConnectionsChange;
        node.onConnectionsChange = function(...args) {
            origOnConn?.apply(this, args);
            refresh();
        };
    }
    
    // Widget changes
    if (nameWidget) {
        const origCb = nameWidget.callback;
        nameWidget.callback = function(...args) {
            origCb?.call(this, ...args);
            refresh();
        };
    }
    
    if (orderWidget) {
        const origCb = orderWidget.callback;
        orderWidget.callback = function(...args) {
            origCb?.call(this, ...args);
            refresh();
        };
    }
    
    // Mode change (bypass)
    const origOnModeChange = node.onModeChange;
    node.onModeChange = function(mode) {
        origOnModeChange?.call(this, mode);
        refresh();
    };
    
    refresh();
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
            if (!varTypesByOrder[variable_name]) {
                varTypesByOrder[variable_name] = {};
            }
            varTypesByOrder[variable_name][order] = type;
            refreshAllTypes();
        });
        
        api.addEventListener("executed", refreshAllTypes);
        
        setInterval(refreshAllTypes, 1000);
        setTimeout(refreshAllTypes, 500);
    },
    
    nodeCreated(node) {
        if (node.type === "SetNodeGlobal" || node.type === "GetNodeGlobal") {
            setupNode(node);
        }
    },
    
    loadedGraphNode(node) {
        if (node.type === "SetNodeGlobal" || node.type === "GetNodeGlobal") {
            setupNode(node);
        }
    },
});

console.log("✅ [NodesGlobal] Loaded");
