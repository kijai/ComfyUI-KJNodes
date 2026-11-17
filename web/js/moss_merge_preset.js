import { app } from "../../scripts/app.js";

const NODE_ID = "MOSSMergePresetKJ";
const ADVANCED_WIDGETS = [
    "merge_strength",
    "composition_stability",
    "text_accuracy",
    "detail_strength"
];

function updateAdvancedVisibility(node) {
    const toggle = node.widgets?.find(w => w.name === "show_advanced");
    const showAdvanced = toggle?.value === "Show advanced sliders";

    for (const widgetName of ADVANCED_WIDGETS) {
        const widget = node.widgets?.find(w => w.name === widgetName);
        if (!widget) continue;

        // Use official ComfyUI standard hidden property
        widget.hidden = !showAdvanced;

        // Force zero layout size when hidden
        if (widget.hidden) {
            if (!widget._originalComputeSize) widget._originalComputeSize = widget.computeSize;
            widget.computeSize = () => [0, 0];
        } else if (widget._originalComputeSize) {
            widget.computeSize = widget._originalComputeSize;
        }
    }

    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "KJNodes.MOSSMergePreset",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_ID) return;

        // Wait for full widget initialization
        const applyState = () => updateAdvancedVisibility(node);
        if (node.widgets) {
            applyState();
        } else {
            const waiter = setInterval(() => {
                if (node.widgets) {
                    applyState();
                    clearInterval(waiter);
                }
            }, 50);
        }

        // Bind toggle callback
        const toggle = node.widgets?.find(w => w.name === "show_advanced");
        if (toggle) {
            const originalCallback = toggle.callback;
            toggle.callback = function (...args) {
                originalCallback?.apply(this, args);
                updateAdvancedVisibility(node);
            };
        }

        // Re-apply after node added to graph
        const originalOnAdded = node.onAdded;
        node.onAdded = function (...args) {
            originalOnAdded?.apply(this, args);
            updateAdvancedVisibility(this);
        };
    }
});
