import { app } from "../../scripts/app.js";

const NODE_ID = "MOSSMergePresetKJ";
const ADVANCED_WIDGETS = [
    "merge_strength",
    "composition_stability",
    "text_accuracy",
    "detail_strength"
];

function updateAdvancedVisibility(node) {
    const toggleWidget = node.widgets?.find(w => w.name === "show_advanced");
    const showAdvanced = toggleWidget?.value === "Show advanced sliders";

    for (const widgetName of ADVANCED_WIDGETS) {
        const widget = node.widgets?.find(w => w.name === widgetName);
        if (widget) {
            widget._hidden = !showAdvanced;
        }
    }
    node.setSize?.(node.computeSize());
}

app.registerExtension({
    name: "KJNodes.MOSSMergePreset",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_ID) return;

        // Initial state: hide all advanced sliders
        updateAdvancedVisibility(node);

        const toggleWidget = node.widgets?.find(w => w.name === "show_advanced");
        if (toggleWidget) {
            const originalCallback = toggleWidget.callback;
            toggleWidget.callback = function (...args) {
                originalCallback?.apply(this, args);
                updateAdvancedVisibility(node);
            };
        }
    },
    async getCustomWidgets() {
        return {};
    }
});
