import { chainCallback } from './utility.js';
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

let execId = null;
api.addEventListener("executing", e => { execId = e.detail ?? null; });

app.registerExtension({
  name: 'KJNodes.FastPreview',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== 'FastPreview') return;
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      this.setSize([550, 550]);
      const nodeRef = this;

      const show = blob => {
        const img = new Image();
        img.onload = () => { nodeRef.imgs = [img]; nodeRef.setDirtyCanvas(true); };
        img.src = URL.createObjectURL(blob);
      };

      const metaHandler = e => {
        const { blob, nodeId, displayNodeId } = e.detail;
        if (String(displayNodeId || nodeId) === String(nodeRef.id)) show(blob);
      };
      const plainHandler = e => {
        if (api.serverSupportsFeature?.("supports_preview_metadata")) return;
        if (String(execId) === String(nodeRef.id)) show(e.detail);
      };

      api.addEventListener("b_preview_with_metadata", metaHandler);
      api.addEventListener("b_preview", plainHandler);

      chainCallback(nodeRef, "onRemoved", () => {
        api.removeEventListener("b_preview_with_metadata", metaHandler);
        api.removeEventListener("b_preview", plainHandler);
      });
    });
  }
});
