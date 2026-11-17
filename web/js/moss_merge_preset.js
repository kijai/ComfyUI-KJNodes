import { app } from "../../scripts/app.js";

const NODE_ID = "MOSSMergePresetKJ";
const ADVANCED_WIDGETS = [
    "merge_strength",
    "composition_stability",
    "text_accuracy",
    "detail_strength"
];

// Base preset SFT weights, 100% aligned with backend
const BASE_PRESET_WEIGHTS = [
    // Composition Preserve
    { early: 0.097, mid: 0.23, late: 0.47 },
    // Balanced
    { early: 0.171, mid: 0.375, late: 0.66 },
    // Text Priority
    { early: 0.28, mid: 0.575, late: 0.85 }
];

function getCurrentWeights(node) {
    const presetIdx = node.widgets.find(w => w.name === "merge_preset")?.value;
    const presetOrder = ["Composition Preserve", "Balanced (Recommended)", "Detail & Typography Priority"];
    const idx = presetOrder.indexOf(presetIdx);
    const base = BASE_PRESET_WEIGHTS[idx] || BASE_PRESET_WEIGHTS[1];

    const strength = node.widgets.find(w => w.name === "merge_strength")?.value ?? 1;
    const comp = node.widgets.find(w => w.name === "composition_stability")?.value ?? 1;
    const text = node.widgets.find(w => w.name === "text_accuracy")?.value ?? 1;
    const detail = node.widgets.find(w => w.name === "detail_strength")?.value ?? 1;

    // Same scaling logic as backend
    const early = Math.max(0, Math.min(1, base.early * strength / comp));
    const mid = Math.max(0, Math.min(1, base.mid * strength * text));
    const late = Math.max(0, Math.min(1, base.late * strength * detail));

    return { early, mid, late, strength, comp, text, detail, presetName: presetIdx };
}

function buildInfoText(node) {
    const w = getCurrentWeights(node);
    const isZh = app.locale?.startsWith("zh");
    if (isZh) {
        let status = "【状态：标准效果】\n预设推荐的默认强度，安全稳定无额外风险。\n";
        if (w.strength <= 0.01) status = "【状态：纯基础模型】\n完全使用基础模型，无微调效果。\n";
        else if (w.strength >= 1.99) status = "【状态：最大强度】\n效果拉满，注意构图风险。\n";

        return `${status}
融合强度: ${w.strength.toFixed(2)}
SFT占比: 浅层${(w.early*100).toFixed(1)}%，中层${(w.mid*100).toFixed(1)}%，深层${(w.late*100).toFixed(1)}%
构图稳定度: ${w.comp.toFixed(2)}
文字准确度: ${w.text.toFixed(2)}
细节强度: ${w.detail.toFixed(2)}

指引: 变形拉稳定/降强度，文字乱拉文字准确度，质感差拉细节强度。`;
    }
    return `[Status: Standard]
Safe recommended preset, no extra risk.

Strength: ${w.strength.toFixed(2)}
SFT Ratio: Early ${(w.early*100).toFixed(1)}%, Mid ${(w.mid*100).toFixed(1)}%, Late ${(w.late*100).toFixed(1)}%
Composition: ${w.comp.toFixed(2)}
Text Accuracy: ${w.text.toFixed(2)}
Detail: ${w.detail.toFixed(2)}`;
}

function updateAdvancedVisibility(node) {
    const toggle = node.widgets?.find(w => w.name === "show_advanced");
    const showAdvanced = toggle?.value === "Show advanced sliders";

    for (const widgetName of ADVANCED_WIDGETS) {
        const widget = node.widgets?.find(w => w.name === widgetName);
        if (!widget) continue;
        widget.hidden = !showAdvanced;
        if (widget.hidden) {
            widget._originalComputeSize = widget._originalComputeSize || widget.computeSize;
            widget.computeSize = () => [0, 0];
        } else if (widget._originalComputeSize) {
            widget.computeSize = widget._originalComputeSize;
        }
    }
    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
}

function updateLiveInfo(node) {
    if (!node._liveInfoDiv) {
        // Create persistent live info box inside node
        node._liveInfoDiv = document.createElement("div");
        node._liveInfoDiv.style.cssText = `
            padding: 10px; margin: 8px; background: #1a1a1a;
            border: 1px solid #444; border-radius: 4px;
            font-size: 12px; line-height: 1.5; color: #ddd;
            white-space: pre-wrap; pointer-events: none;
        `;
        node.addDOMWidget("live_merge_info", "MOSS Live Info", node._liveInfoDiv, {
            getValue: () => "", setValue: () => {}
        });
    }
    node._liveInfoDiv.textContent = buildInfoText(node);
}

app.registerExtension({
    name: "KJNodes.MOSSMergePreset",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_ID) return;

        const init = () => {
            updateAdvancedVisibility(node);
            updateLiveInfo(node);

            // Bind all widgets for real-time updates
            node.widgets.forEach(w => {
                const origCb = w.callback;
                w.callback = (...args) => {
                    origCb?.apply(w, args);
                    updateAdvancedVisibility(node);
                    updateLiveInfo(node);
                };
            });
        };

        if (node.widgets) init();
        else {
            const wait = setInterval(() => {
                if (node.widgets) {
                    init();
                    clearInterval(wait);
                }
            }, 50);
        }

        const origOnAdded = node.onAdded;
        node.onAdded = (...args) => {
            origOnAdded?.apply(node, args);
            updateAdvancedVisibility(node);
            updateLiveInfo(node);
        };
    }
});
