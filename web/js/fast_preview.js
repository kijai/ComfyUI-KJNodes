import { chainCallback } from './utility.js';
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

app.registerExtension({
  name: 'KJNodes.FastPreview',

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === 'FastPreview') {
      chainCallback(nodeType.prototype, "onNodeCreated", function () {

        this.setSize([550, 550]);

        const nodeRef = this;
        api.addEventListener("b_preview_with_metadata", function (event) {
          const { blob, nodeId, displayNodeId } = event.detail;
          const targetId = String(displayNodeId || nodeId);
          if (targetId !== String(nodeRef.id)) return;

          const img = new Image();
          img.onload = () => {
            nodeRef.imgs = [img];
            nodeRef.setDirtyCanvas(true);
          };
          img.src = URL.createObjectURL(blob);
        });

      });
    }
  }
});
