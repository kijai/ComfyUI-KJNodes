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
    // Hide trigger outputs/inputs
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

function findSetNodesByVar(varName) {
    if (!app.graph?._nodes) return [];
    
    const results = [];
    for (const node of app.graph._nodes) {
        if (node.type === "SetNodeGlobal" && node.mode !== 4) { // mode 4 = bypassed
            const nameWidget = node.widgets?.find(w => w.name === "variable_name");
            const orderWidget = node.widgets?.find(w => w.name === "order");
            if (nameWidget?.value === varName) {
                results.push({
                    node,
                    order: orderWidget?.value ?? 0,
                    type: getInputTypeFromNode(node)
                });
            }
        }
    }
    return results.sort((a, b) => a.order - b.order);
}

function getVarType(varName, targetOrder = null) {
    if (!varName) return "*";
    
    const sets = findSetNodesByVar(varName);
    if (sets.length === 0) return varTypes[varName] || "*";
    
    if (targetOrder === null) {
        // Return type from highest order Set
        const t = sets[sets.length - 1].type;
        if (t !== "*") varTypes[varName] = t;
        return t;
    }
    
    // Find Set with same order or highest order < targetOrder
    let match = null;
    for (const s of sets) {
        if (s.order === targetOrder) {
            match = s;
            break;
        }
        if (s.order < targetOrder) {
            match = s;
        }
    }
    
    if (match) {
        if (match.type !== "*") varTypes[varName] = match.type;
        return match.type;
    }
    
    return sets[0].type || varTypes[varName] || "*";
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

function refreshAllTypes() {
    if (!app.graph?._nodes) return;
    
    // Collect types from active Set nodes
    const varTypesByOrder = {};
    
    for (const node of app.graph._nodes) {
        if (node.type === "SetNodeGlobal" && node.mode !== 4) {
            const nameWidget = node.widgets?.find(w => w.name === "variable_name");
            const orderWidget = node.widgets?.find(w => w.name === "order");
            
            if (nameWidget?.value) {
                const t = getInputTypeFromNode(node);
                const order = orderWidget?.value ?? 0;
                const varName = nameWidget.value;
                
                if (!varTypesByOrder[varName]) {
                    varTypesByOrder[varName] = {};
                }
                varTypesByOrder[varName][order] = t;
                
                updateSlotType(node, 0, t);
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
                const orderTypes = varTypesByOrder[varName] || {};
                
                // Find matching type
                let matchingType = "*";
                if (orderTypes[order] !== undefined) {
                    matchingType = orderTypes[order];
                } else {
                    // Find highest order < this order
                    const orders = Object.keys(orderTypes).map(Number).sort((a,b) => a-b);
                    for (const o of orders) {
                        if (o < order) matchingType = orderTypes[o];
                    }
                    if (matchingType === "*" && orders.length > 0) {
                        matchingType = orderTypes[orders[0]];
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
function setupNode(node) {
    hideInternalSlots(node);
    
    const nameWidget = node.widgets?.find(w => w.name === "variable_name");
    const orderWidget = node.widgets?.find(w => w.name === "order");
    
    // Refresh on any change
    const refresh = () => setTimeout(refreshAllTypes, 50);
    
    if (node.type === "SetNodeGlobal") {
        const origOnConn = node.onConnectionsChange;
        node.onConnectionsChange = function(...args) {
            origOnConn?.apply(this, args);
            refresh();
        };
    }
    
    // Refresh on widget changes
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
    
    // Refresh on mode change (bypass/mute)
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
            const { variable_name, type } = e.detail;
            varTypes[variable_name] = type;
            refreshAllTypes();
        });
        
        api.addEventListener("executed", refreshAllTypes);
        
        // More frequent refresh to catch bypass changes
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
