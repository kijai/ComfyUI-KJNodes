const { app } = window.comfyAPI.app;

export function makeUUID() {
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}

const create_documentation_stylesheet = () => {
  const tag = 'kj-pointseditor-stylesheet'
  let styleTag = document.head.querySelector(`#${tag}`)
  if (!styleTag) {
    styleTag = document.createElement('style')
    styleTag.type = 'text/css'
    styleTag.id = tag
    styleTag.innerHTML = `
       .points-editor {
        position: absolute;
        font: 12px monospace;
        line-height: 1.5em;
        padding: 10px;
        z-index: 0;
        overflow: hidden;
       }
       .points-editor canvas {
        position: relative;
        z-index: 2;
       }
    `
    document.head.appendChild(styleTag)
  }
}
create_documentation_stylesheet()

function chainCallback(object, property, callback) {
  if (object == undefined) {
    console.error("Tried to add callback to non-existant object")
    return;
  }
  if (property in object) {
    const callback_orig = object[property]
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r
    };
  } else {
    object[property] = callback;
  }
}

app.registerExtension({
  name: 'KJNodes.PointEditor',

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === 'PointsEditor') {
      chainCallback(nodeType.prototype, "onNodeCreated", function () {

        hideWidgetForGood(this, this.widgets.find(w => w.name === "coordinates"))
        hideWidgetForGood(this, this.widgets.find(w => w.name === "neg_coordinates"))
        hideWidgetForGood(this, this.widgets.find(w => w.name === "bboxes"))

        var element = document.createElement("div");
        this.uuid = makeUUID()
        element.id = `points-editor-${this.uuid}`

        this.previewMediaType = 'image'

        this.pointsEditor = this.addDOMWidget(nodeData.name, "PointsEditorWidget", element, {
          serialize: false,
          hideOnZoom: false,
          getMinHeight: () => this.pointsEditorHeight || 550,
          getMaxHeight: () => this.pointsEditorHeight || 550,
          getHeight: () => this.pointsEditorHeight || 550,
        });
        this.pointsEditorHeight = 550;

        // context menu
        this.contextMenu = document.createElement("div");
        this.contextMenu.id = "context-menu";
        this.contextMenu.style.display = "none";
        this.contextMenu.style.position = "absolute";
        this.contextMenu.style.backgroundColor = "#202020";
        this.contextMenu.style.minWidth = "100px";
        this.contextMenu.style.boxShadow = "0px 8px 16px 0px rgba(0,0,0,0.2)";
        this.contextMenu.style.zIndex = "100";
        this.contextMenu.style.padding = "5px";

        function styleMenuItem(menuItem) {
          menuItem.style.display = "block";
          menuItem.style.padding = "5px";
          menuItem.style.color = "#FFF";
          menuItem.style.fontFamily = "Arial, sans-serif";
          menuItem.style.fontSize = "16px";
          menuItem.style.textDecoration = "none";
          menuItem.style.marginBottom = "5px";
        }
        function createMenuItem(id, textContent) {
          let menuItem = document.createElement("a");
          menuItem.href = "#";
          menuItem.id = `menu-item-${id}`;
          menuItem.textContent = textContent;
          styleMenuItem(menuItem);
          return menuItem;
        }

        this.menuItems = [
          createMenuItem(0, "Load Image"),
          createMenuItem(1, "Clear Image"),
        ];

        this.menuItems.forEach(menuItem => {
          menuItem.addEventListener('mouseover', function () {
            this.style.backgroundColor = "gray";
          });
          menuItem.addEventListener('mouseout', function () {
            this.style.backgroundColor = "#202020";
          });
        });

        this.menuItems.forEach(menuItem => {
          this.contextMenu.appendChild(menuItem);
        });

        document.body.appendChild(this.contextMenu);

        this.addWidget("button", "New canvas", null, () => {
          try {
            if (!this.properties || !("points" in this.properties)) {
              this.editor = new PointsEditor(this);
              this.addProperty("points", this.constructor.type, "string");
              this.addProperty("neg_points", this.constructor.type, "string");
            }
            else {
              this.editor = new PointsEditor(this, true);
            }
          } catch (error) {
            console.error("An error occurred while creating the point editor:", error);
          }
        });

        this.setSize([550, 550]);
        this.resizable = false;
        this.pointsEditor.parentEl = document.createElement("div");
        this.pointsEditor.parentEl.className = "points-editor";
        this.pointsEditor.parentEl.id = `points-editor-${this.uuid}`
        element.appendChild(this.pointsEditor.parentEl);

        chainCallback(this, "onConfigure", function () {
          try {
            this.editor = new PointsEditor(this);
          } catch (error) {
            console.error("An error occurred while configuring the editor:", error);
          }
        });
        chainCallback(this, "onExecuted", function (message) {
          let bg_image = message["bg_image"];
          this.properties.imgData = {
            name: "bg_image",
            base64: bg_image
          };
          this.editor.refreshBackgroundImage(this);
        });

      });
    }
  }
})

class PointsEditor {
  setNodeWidth(width) {
    this.node.setSize([width, this.node.size[1]]);
    const nodeEl = document.querySelector(`[data-node-id="${this.node.id}"]`);
    if (nodeEl) {
      nodeEl.style.setProperty('--node-width', `${width}px`);
    }
  }

  constructor(context, reset = false) {
    this.node = context;
    this.reset = reset;
    const self = this;

    this.node.pasteFile = (file) => {
      if (file.type.startsWith("image/")) {
        this.handleImageFile(file);
        return true;
      }
      return false;
    };

    this.node.onDragOver = function (e) {
      if (e.dataTransfer && e.dataTransfer.items) {
        return [...e.dataTransfer.items].some(f => f.kind === "file" && f.type.startsWith("image/"));
      }
      return false;
    };

    this.node.onDragDrop = (e) => {
      let handled = false;
      for (const file of e.dataTransfer.files) {
        if (file.type.startsWith("image/")) {
          this.handleImageFile(file);
          handled = true;
        }
      }
      return handled;
    };

    this.createContextMenu();

    if (reset && context.pointsEditor.element) {
      context.pointsEditor.element.innerHTML = '';
    }
    this.pos_coordWidget = context.widgets.find(w => w.name === "coordinates");
    this.neg_coordWidget = context.widgets.find(w => w.name === "neg_coordinates");
    this.pointsStoreWidget = context.widgets.find(w => w.name === "points_store");
    this.widthWidget = context.widgets.find(w => w.name === "width");
    this.heightWidget = context.widgets.find(w => w.name === "height");
    this.bboxStoreWidget = context.widgets.find(w => w.name === "bbox_store");
    this.bboxWidget = context.widgets.find(w => w.name === "bboxes");

    this.widthWidget.callback = () => {
      this.width = this.widthWidget.value;
      if (this.width > 256) {
        this.setNodeWidth(this.width + 45);
      }
      this.canvas.width = this.width + this.margin * 2;
      this.render();
      this.updateData();
    }
    this.heightWidget.callback = () => {
      this.height = this.heightWidget.value;
      this.canvas.height = this.height + this.margin * 2;
      context.pointsEditorHeight = this.height + 30;
      context.setSize([context.size[0], this.height + 300]);
      if (context.graph) {
        context.arrange?.();
        context.graph.setDirtyCanvas(true, true);
      }
      this.render();
      this.updateData();
    }
    this.pointsStoreWidget.callback = () => {
      this.points = JSON.parse(this.pointsStoreWidget.value).positive;
      this.neg_points = JSON.parse(this.pointsStoreWidget.value).negative;
      this.render();
      this.updateData();
    }
    this.bboxStoreWidget.callback = () => {
      this.bbox = JSON.parse(this.bboxStoreWidget.value);
      this.render();
      this.updateData();
    }

    this.width = this.widthWidget.value;
    this.height = this.heightWidget.value;
    this.margin = 10;
    this.points = [];
    this.neg_points = [];
    this.bbox = [{}];
    this.bgImage = null;
    this.dragIndex = -1;
    this.dragType = null; // 'positive', 'negative', 'bbox'
    this.drawing = false;
    this.selectedIndex = -1;

    const pointRadius = Math.log(Math.min(this.width, this.height)) * 4;
    this.pointRadius = pointRadius;

    // Initialize or reset points
    if (!reset && this.pointsStoreWidget.value != "") {
      this.points = JSON.parse(this.pointsStoreWidget.value).positive;
      this.neg_points = JSON.parse(this.pointsStoreWidget.value).negative;
      this.bbox = JSON.parse(this.bboxStoreWidget.value);
    } else {
      this.points = [
        { x: this.width / 2, y: this.height / 2 }
      ];
      this.neg_points = [];
      const combinedPoints = {
        positive: this.points,
        negative: this.neg_points,
      };
      this.pointsStoreWidget.value = JSON.stringify(combinedPoints);
      this.bboxStoreWidget.value = JSON.stringify(this.bbox);
    }

    // Create canvas
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.width + this.margin * 2;
    this.canvas.height = this.height + this.margin * 2;
    this.ctx = this.canvas.getContext('2d');

    this.node.pointsEditor.element.appendChild(this.canvas);

    if (this.width > 256) {
      this.setNodeWidth(this.width + 45);
    }
    this.node.pointsEditorHeight = this.height + 30;
    this.node.setSize([this.node.size[0], this.height + 300]);

    // Event listeners
    this._onDragMove = (e) => this.onMouseMove(e);
    this._onDragEnd = (e) => this.onMouseUp(e);
    this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    this.render();
    this.updateData();
    this.refreshBackgroundImage();
  }

  getLocalMouse(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX - this.margin,
      y: (e.clientY - rect.top) * scaleY - this.margin
    };
  }

  clampToPanel(x, y) {
    return {
      x: Math.max(0, Math.min(this.width, x)),
      y: Math.max(0, Math.min(this.height, y))
    };
  }

  findPointAt(x, y, pointsArray) {
    const r = this.pointRadius;
    for (let i = pointsArray.length - 1; i >= 0; i--) {
      const p = pointsArray[i];
      const dx = p.x - x;
      const dy = p.y - y;
      if (dx * dx + dy * dy <= r * r) {
        return i;
      }
    }
    return -1;
  }

  normBbox() {
    const b = this.bbox[0];
    if (!b || Object.keys(b).length === 0) return null;
    return {
      x1: Math.min(b.startX, b.endX),
      y1: Math.min(b.startY, b.endY),
      x2: Math.max(b.startX, b.endX),
      y2: Math.max(b.startY, b.endY),
    };
  }

  hitTestBbox(mx, my) {
    const n = this.normBbox();
    if (!n) return null;
    const R = 10;
    const hitCorner = (cx, cy) => Math.abs(mx - cx) < R && Math.abs(my - cy) < R;
    if (hitCorner(n.x1, n.y1)) return "resize-tl";
    if (hitCorner(n.x2, n.y1)) return "resize-tr";
    if (hitCorner(n.x1, n.y2)) return "resize-bl";
    if (hitCorner(n.x2, n.y2)) return "resize-br";
    if (mx >= n.x1 && mx <= n.x2 && my >= n.y1 && my <= n.y2) return "move";
    return null;
  }

  onMouseDown(e) {
    const mouse = this.getLocalMouse(e);
    const clamped = this.clampToPanel(mouse.x, mouse.y);

    // Shift + right-click: add negative point
    if (e.shiftKey && e.button === 2) {
      this.neg_points.push({ x: clamped.x, y: clamped.y });
      this.render();
      this.updateData();
      return;
    }

    // Shift + left-click: add positive point
    if (e.shiftKey && e.button === 0) {
      this.points.push({ x: clamped.x, y: clamped.y });
      this.render();
      this.updateData();
      return;
    }

    // Right-click on point: delete it; right-click on bbox: delete bbox
    if (e.button === 2) {
      const posIdx = this.findPointAt(mouse.x, mouse.y, this.points);
      if (posIdx >= 0) {
        this.points.splice(posIdx, 1);
        this.render();
        this.updateData();
        return;
      }
      const negIdx = this.findPointAt(mouse.x, mouse.y, this.neg_points);
      if (negIdx >= 0) {
        this.neg_points.splice(negIdx, 1);
        this.render();
        this.updateData();
        return;
      }
      // Right-click on bbox: delete it
      const bboxHit = this.hitTestBbox(mouse.x, mouse.y);
      if (bboxHit) {
        this.bbox = [{}];
        this.render();
        this.updateData();
        return;
      }
      // Right-click on background: show context menu
      this.node.contextMenu.style.display = 'block';
      this.node.contextMenu.style.left = `${e.clientX}px`;
      this.node.contextMenu.style.top = `${e.clientY}px`;
      return;
    }

    if (e.button !== 0) return;

    // Ctrl + click: start drawing new bbox
    if (e.ctrlKey) {
      this.drawing = true;
      this.dragType = 'bbox-draw';
      this.bbox[0] = { startX: clamped.x, startY: clamped.y, endX: clamped.x, endY: clamped.y };
      this.render();
      this.startDocumentDrag();
      return;
    }

    // Left-click: check bbox corners/move first, then points
    const bboxHit = this.hitTestBbox(mouse.x, mouse.y);
    if (bboxHit) {
      const n = this.normBbox();
      this.bboxAtDragStart = { startX: n.x1, startY: n.y1, endX: n.x2, endY: n.y2 };
      this.dragStart = { x: mouse.x, y: mouse.y };

      if (bboxHit === "move") {
        this.dragType = 'bbox-move';
      } else {
        this.dragType = bboxHit; // 'resize-tl', 'resize-tr', 'resize-bl', 'resize-br'
      }
      this.dragIndex = 0;
      this.startDocumentDrag();
      return;
    }

    const posIdx = this.findPointAt(mouse.x, mouse.y, this.points);
    if (posIdx >= 0) {
      this.dragType = 'positive';
      this.dragIndex = posIdx;
      this.selectedIndex = posIdx;
      this.render();
      this.startDocumentDrag();
      return;
    }

    const negIdx = this.findPointAt(mouse.x, mouse.y, this.neg_points);
    if (negIdx >= 0) {
      this.dragType = 'negative';
      this.dragIndex = negIdx;
      this.selectedIndex = negIdx;
      this.render();
      this.startDocumentDrag();
      return;
    }
  }

  startDocumentDrag() {
    document.addEventListener('mousemove', this._onDragMove);
    document.addEventListener('mouseup', this._onDragEnd);
  }

  onMouseMove(e) {
    const mouse = this.getLocalMouse(e);
    const clamped = this.clampToPanel(mouse.x, mouse.y);

    // Update cursor when not dragging
    if (!this.drawing && this.dragIndex < 0 && !this.dragType) {
      const bboxHit = this.hitTestBbox(mouse.x, mouse.y);
      if (bboxHit === "move") {
        this.canvas.style.cursor = "move";
      } else if (bboxHit === "resize-tl" || bboxHit === "resize-br") {
        this.canvas.style.cursor = "nwse-resize";
      } else if (bboxHit === "resize-tr" || bboxHit === "resize-bl") {
        this.canvas.style.cursor = "nesw-resize";
      } else if (this.findPointAt(mouse.x, mouse.y, this.points) >= 0 ||
                 this.findPointAt(mouse.x, mouse.y, this.neg_points) >= 0) {
        this.canvas.style.cursor = "move";
      } else {
        this.canvas.style.cursor = "default";
      }
      return;
    }

    if (this.dragType === 'bbox-draw') {
      this.bbox[0].endX = clamped.x;
      this.bbox[0].endY = clamped.y;
      this.render();
      return;
    }

    if (this.dragType === 'bbox-move' && this.bboxAtDragStart) {
      const dx = mouse.x - this.dragStart.x;
      const dy = mouse.y - this.dragStart.y;
      const bs = this.bboxAtDragStart;
      const bw = bs.endX - bs.startX;
      const bh = bs.endY - bs.startY;
      const nx = Math.max(0, Math.min(this.width - bw, bs.startX + dx));
      const ny = Math.max(0, Math.min(this.height - bh, bs.startY + dy));
      this.bbox[0] = { startX: nx, startY: ny, endX: nx + bw, endY: ny + bh };
      this.render();
      return;
    }

    if (this.dragType?.startsWith('resize-') && this.bboxAtDragStart) {
      const bs = this.bboxAtDragStart;
      const offX = this.dragStart.x - (
        this.dragType === 'resize-tl' || this.dragType === 'resize-bl' ? bs.startX : bs.endX
      );
      const offY = this.dragStart.y - (
        this.dragType === 'resize-tl' || this.dragType === 'resize-tr' ? bs.startY : bs.endY
      );
      const cx = Math.max(0, Math.min(this.width, mouse.x - offX));
      const cy = Math.max(0, Math.min(this.height, mouse.y - offY));

      if (this.dragType === 'resize-tl') {
        this.bbox[0] = { startX: bs.endX, startY: bs.endY, endX: cx, endY: cy };
      } else if (this.dragType === 'resize-tr') {
        this.bbox[0] = { startX: bs.startX, startY: bs.endY, endX: cx, endY: cy };
      } else if (this.dragType === 'resize-bl') {
        this.bbox[0] = { startX: bs.endX, startY: bs.startY, endX: cx, endY: cy };
      } else {
        this.bbox[0] = { startX: bs.startX, startY: bs.startY, endX: cx, endY: cy };
      }
      this.render();
      return;
    }

    if (this.dragType === 'positive' && this.dragIndex >= 0 && this.dragIndex < this.points.length) {
      this.points[this.dragIndex] = { x: clamped.x, y: clamped.y };
      this.render();
    } else if (this.dragType === 'negative' && this.dragIndex >= 0 && this.dragIndex < this.neg_points.length) {
      this.neg_points[this.dragIndex] = { x: clamped.x, y: clamped.y };
      this.render();
    }
  }

  onMouseUp(e) {
    if (this.drawing || this.dragType) {
      this.drawing = false;
      this.dragIndex = -1;
      this.dragType = null;
      this.bboxAtDragStart = null;
      this.dragStart = null;
      document.removeEventListener('mousemove', this._onDragMove);
      document.removeEventListener('mouseup', this._onDragEnd);
      this.updateData();
    }
  }

  render() {
    const ctx = this.ctx;
    const m = this.margin;
    const w = this.width;
    const h = this.height;

    // Clear
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    ctx.save();
    ctx.translate(m, m);

    // Background
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'gray';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, w, h);

    // Background image
    if (this.bgImage) {
      ctx.drawImage(this.bgImage, 0, 0, w, h);
    }

    // Bounding box
    const nb = this.normBbox();
    if (nb) {
      const bx = nb.x1;
      const by = nb.y1;
      const bw = nb.x2 - nb.x1;
      const bh = nb.y2 - nb.y1;

      // Semi-transparent fill
      ctx.fillStyle = 'rgba(70, 130, 180, 0.3)';
      ctx.fillRect(bx, by, bw, bh);

      // Border with dark outline for contrast
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.lineWidth = 4;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.strokeStyle = 'steelblue';
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);

      // Center cross
      const cx = (nb.x1 + nb.x2) / 2;
      const cy = (nb.y1 + nb.y2) / 2;
      ctx.strokeStyle = 'steelblue';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - 5, cy);
      ctx.lineTo(cx + 5, cy);
      ctx.moveTo(cx, cy - 5);
      ctx.lineTo(cx, cy + 5);
      ctx.stroke();

      // Corner handles (all 4)
      const hs = 5;
      ctx.fillStyle = 'steelblue';
      for (const [hx, hy] of [[nb.x1, nb.y1], [nb.x2, nb.y1], [nb.x1, nb.y2], [nb.x2, nb.y2]]) {
        ctx.fillRect(hx - hs, hy - hs, hs * 2, hs * 2);
      }
    }

    // Draw positive points
    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      const isSelected = (this.dragType === 'positive' && this.dragIndex === i);
      const r = this.pointRadius;

      // Outer circle
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(100, 100, 100, 0.6)';
      ctx.fill();
      ctx.strokeStyle = isSelected ? '#07f907' : '#139613';
      ctx.lineWidth = 4;
      ctx.stroke();

      // Center dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'red';
      ctx.fill();

      // Label
      ctx.font = '25px sans-serif';
      ctx.fillStyle = '#139613';
      ctx.shadowColor = 'black';
      ctx.shadowBlur = 2;
      ctx.shadowOffsetX = 2;
      ctx.shadowOffsetY = 2;
      const labelX = p.x < w / 2 ? p.x + 30 : p.x - 35;
      const labelY = p.y < h / 2 ? p.y + 25 : p.y - 25;
      ctx.fillText(i.toString(), labelX, labelY);
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
    }

    // Draw negative points
    for (let i = 0; i < this.neg_points.length; i++) {
      const p = this.neg_points[i];
      const isSelected = (this.dragType === 'negative' && this.dragIndex === i);
      const r = this.pointRadius;

      // Outer circle
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(100, 100, 100, 0.6)';
      ctx.fill();
      ctx.strokeStyle = isSelected ? '#f91111' : '#891616';
      ctx.lineWidth = 4;
      ctx.stroke();

      // Center dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'red';
      ctx.fill();

      // Label
      ctx.font = '25px sans-serif';
      ctx.fillStyle = 'red';
      ctx.shadowColor = 'black';
      ctx.shadowBlur = 2;
      ctx.shadowOffsetX = 2;
      ctx.shadowOffsetY = 2;
      const labelX = p.x < w / 2 ? p.x + 30 : p.x - 35;
      const labelY = p.y < h / 2 ? p.y + 25 : p.y - 25;
      ctx.fillText(i.toString(), labelX, labelY);
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
    }

    ctx.restore();
  }

  updateData = () => {
    const combinedPoints = {
      positive: this.points || [],
      negative: this.neg_points || [],
    };
    this.pointsStoreWidget.value = JSON.stringify(combinedPoints);
    this.pos_coordWidget.value = JSON.stringify(this.points || []);
    this.neg_coordWidget.value = JSON.stringify(this.neg_points || []);

    if (this.bbox.length != 0) {
      let bboxString = JSON.stringify(this.bbox);
      this.bboxStoreWidget.value = bboxString;
      this.bboxWidget.value = bboxString;
    }
  };

  handleImageLoad = (img, file, base64String) => {
    this.widthWidget.value = img.width;
    this.heightWidget.value = img.height;

    if (img.width != this.width || img.height != this.height) {
      this.width = img.width;
      this.height = img.height;
      this.canvas.width = this.width + this.margin * 2;
      this.canvas.height = this.height + this.margin * 2;

      if (img.width > 256) {
        this.setNodeWidth(img.width + 45);
      }
      this.node.pointsEditorHeight = img.height + 30;
      this.node.setSize([this.node.size[0], this.height + 300]);
      if (this.node.graph) {
        this.node.arrange?.();
        this.node.graph.setDirtyCanvas(true, true);
      }
    }

    // Load as background image
    const mimeType = this.node.properties.imgData?.type || 'image/png';
    const bgImg = new Image();
    bgImg.src = file ? URL.createObjectURL(file) : `data:${mimeType};base64,${base64String}`;
    bgImg.onload = () => {
      this.bgImage = bgImg;
      this.render();
      this.updateData();
    };
  };

  processImage = (img, file) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const maxWidth = 800;
    const maxHeight = 600;
    let width = img.width;
    let height = img.height;

    if (width > height) {
      if (width > maxWidth) {
        height *= maxWidth / width;
        width = maxWidth;
      }
    } else {
      if (height > maxHeight) {
        width *= maxHeight / height;
        height = maxHeight;
      }
    }

    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    const base64String = canvas.toDataURL('image/jpeg', 0.5).replace('data:', '').replace(/^.+,/, '');

    this.node.properties.imgData = {
      name: file.name,
      lastModified: file.lastModified,
      size: file.size,
      type: file.type,
      base64: base64String
    };
    this.handleImageLoad(img, file, base64String);
  };

  handleImageFile = (file) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const img = new Image();
      img.src = reader.result;
      img.onload = () => this.processImage(img, file);
    };
    reader.readAsDataURL(file);

    const imageUrl = URL.createObjectURL(file);
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => this.handleImageLoad(img, file, null);
  };

  refreshBackgroundImage = () => {
    if (this.node.properties.imgData && this.node.properties.imgData.base64) {
      const base64String = this.node.properties.imgData.base64;
      const mimeType = this.node.properties.imgData.type || 'image/png';
      const imageUrl = `data:${mimeType};base64,${base64String}`;
      const img = new Image();
      img.src = imageUrl;
      img.onload = () => this.handleImageLoad(img, null, base64String);
    }
  };

  createContextMenu = () => {
    const self = this;
    document.addEventListener('contextmenu', function (e) {
      if (e.target.closest(`#points-editor-${self.node.uuid}`) ||
        e.target.closest('#context-menu')) {
        e.preventDefault();
      }
    });

    document.addEventListener('click', function (e) {
      if (!self.node.contextMenu.contains(e.target)) {
        self.node.contextMenu.style.display = 'none';
      }
    });

    this.node.menuItems.forEach((menuItem, index) => {
      menuItem.addEventListener('click', function (e) {
        e.preventDefault();
        switch (index) {
          case 0:
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*';
            fileInput.addEventListener('change', function (event) {
              const file = event.target.files[0];
              if (file) {
                const imageUrl = URL.createObjectURL(file);
                let img = new Image();
                img.src = imageUrl;
                img.onload = () => self.handleImageLoad(img, file, null);
              }
            });
            fileInput.click();
            self.node.contextMenu.style.display = 'none';
            break;
          case 1:
            self.bgImage = null;
            self.render();
            self.node.properties.imgData = null;
            self.node.contextMenu.style.display = 'none';
            break;
        }
      });
    });
  }
}

//from melmass
export function hideWidgetForGood(node, widget, suffix = '') {
  widget.origType = widget.type
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4]
  widget.type = "converted-widget" + suffix

  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidgetForGood(node, w, ':' + widget.name)
    }
  }
}
