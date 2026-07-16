import { chainCallback, rectHitTest, cursorForBboxMode } from '../utility.js';
import { BaseEditorCanvas, createEditorStylesheet } from './editor_base.js';
const { app } = window.comfyAPI.app;

createEditorStylesheet('kj-pointseditor-stylesheet', 'points-editor');

app.registerExtension({
  name: 'KJNodes.PointEditor',

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === 'PointsEditor') {
      chainCallback(nodeType.prototype, "onNodeCreated", function () {
        BaseEditorCanvas.setupNode(this, nodeData, {
          editorClass: PointsEditor,
          editorKey: 'pointsEditor',
          heightKey: 'pointsEditorHeight',
          className: 'points-editor',
          menuItems: {
            "Load Image":  { action: (ed) => ed.openImageFilePicker() },
            "Clear Image":  { action: (ed) => ed.clearBackgroundImage() },
          },
          hiddenWidgets: ["coordinates", "neg_coordinates", "bboxes"],
          initialSize: [550, 550],
          extraProperties: [
            ["points", this.constructor.type, "string"],
            ["neg_points", this.constructor.type, "string"],
          ],
        });
      });
    }
  }
});

class PointsEditor extends BaseEditorCanvas {
  constructor(context, reset = false) {
    super(context, reset);
    this.initEditorPreamble('pointsEditor', 'points-editor');

    this.pos_coordWidget = this.findWidget("coordinates");
    this.neg_coordWidget = this.findWidget("neg_coordinates");
    this.pointsStoreWidget = this.findWidget("points_store");
    this.widthWidget = this.findWidget("width");
    this.heightWidget = this.findWidget("height");
    this.bboxStoreWidget = this.findWidget("bbox_store");
    this.bboxWidget = this.findWidget("bboxes");

    this.setupSizeCallbacks();
    this.pointsStoreWidget.callback = () => {
      try {
        const parsed = JSON.parse(this.pointsStoreWidget.value);
        this.points = parsed.positive || [];
        this.neg_points = parsed.negative || [];
      } catch (e) { console.error("Error parsing points data:", e); }
      this.render();
      this.updateData();
    };
    this.bboxStoreWidget.callback = () => {
      try { this.bbox = JSON.parse(this.bboxStoreWidget.value); }
      catch (e) { console.error("Error parsing bbox data:", e); }
      this.render();
      this.updateData();
    };

    this.initDisplaySize();
    this.points = [];
    this.neg_points = [];
    this.bbox = [{}];
    this.drawing = false;
    this.selectedIndex = -1;
    this.pointRadius = Math.max(4, Math.log(Math.max(1, Math.min(this.coordWidth, this.coordHeight))) * 4);

    if (!reset && this.pointsStoreWidget.value !== "") {
      try {
        const parsed = JSON.parse(this.pointsStoreWidget.value);
        this.points = parsed.positive || [];
        this.neg_points = parsed.negative || [];
        this.bbox = JSON.parse(this.bboxStoreWidget.value);
      } catch (e) {
        console.error("Error parsing stored points:", e);
        this.points = [{ x: this.coordWidth / 2, y: this.coordHeight / 2 }];
        this.neg_points = [];
        this.bbox = [{}];
      }
    } else {
      this.points = [{ x: this.coordWidth / 2, y: this.coordHeight / 2 }];
      this.neg_points = [];
      this.pointsStoreWidget.value = JSON.stringify({ positive: this.points, negative: this.neg_points });
      this.bboxStoreWidget.value = JSON.stringify(this.bbox);
    }

    this.initEditor('pointsEditor', 'pointsEditorHeight', 310);
    this.updateData();
    this.refreshBackgroundImage();
  }

  onDataChanged() { this.updateData(); }

  onCoordSpaceResized(oldWidth, oldHeight) {
    const sx = this.coordWidth / oldWidth, sy = this.coordHeight / oldHeight;
    const clampPt = (p) => { p.x = Math.max(0, Math.min(this.coordWidth, p.x * sx)); p.y = Math.max(0, Math.min(this.coordHeight, p.y * sy)); };
    for (const p of this.points) clampPt(p);
    for (const p of this.neg_points) clampPt(p);
    for (const b of this.bbox) {
      if (b.startX != null) {
        b.startX = Math.max(0, Math.min(this.coordWidth, b.startX * sx));
        b.startY = Math.max(0, Math.min(this.coordHeight, b.startY * sy));
        b.endX = Math.max(0, Math.min(this.coordWidth, b.endX * sx));
        b.endY = Math.max(0, Math.min(this.coordHeight, b.endY * sy));
      }
    }
    this.pointRadius = Math.max(4, Math.log(Math.max(1, Math.min(this.coordWidth, this.coordHeight))) * 4);
  }

  // ─── Hit Testing ───

  findPointAt(x, y, pointsArray) {
    // Convert pixel radius to coord space
    const r = this.pointRadius / Math.min(this.scaleX, this.scaleY);
    for (let i = pointsArray.length - 1; i >= 0; i--) {
      const p = pointsArray[i];
      const dx = p.x - x, dy = p.y - y;
      if (dx * dx + dy * dy <= r * r) return i;
    }
    return -1;
  }

  normBbox() {
    const b = this.bbox[0];
    if (!b || Object.keys(b).length === 0) return null;
    return {
      x1: Math.min(b.startX, b.endX), y1: Math.min(b.startY, b.endY),
      x2: Math.max(b.startX, b.endX), y2: Math.max(b.startY, b.endY),
    };
  }

  hitTestBbox(mx, my) {
    const n = this.normBbox();
    if (!n) return null;
    return rectHitTest(mx, my, n.x1, n.y1, n.x2, n.y2, 10 / Math.min(this.scaleX, this.scaleY));
  }

  // ─── Mouse Handlers ───

  onMouseDown(e) {
    const mouse = this.getLocalMouse(e);
    const clamped = this.clamp(mouse.x, mouse.y);

    if (e.shiftKey && e.button === 2) {
      this.neg_points.push({ x: clamped.x, y: clamped.y });
      this.dragType = 'negative'; this.dragIndex = this.neg_points.length - 1;
      this.selectedIndex = this.dragIndex;
      this.dragOffset = { x: 0, y: 0 };
      this.render(); this.updateData(); this.startDocumentDrag(e); return;
    }
    if (e.shiftKey && e.button === 0) {
      this.points.push({ x: clamped.x, y: clamped.y });
      this.dragType = 'positive'; this.dragIndex = this.points.length - 1;
      this.selectedIndex = this.dragIndex;
      this.dragOffset = { x: 0, y: 0 };
      this.render(); this.updateData(); this.startDocumentDrag(e); return;
    }

    if (e.button === 2) {
      const posIdx = this.findPointAt(mouse.x, mouse.y, this.points);
      if (posIdx >= 0) { this.points.splice(posIdx, 1); this.render(); this.updateData(); return; }
      const negIdx = this.findPointAt(mouse.x, mouse.y, this.neg_points);
      if (negIdx >= 0) { this.neg_points.splice(negIdx, 1); this.render(); this.updateData(); return; }
      const bboxHit = this.hitTestBbox(mouse.x, mouse.y);
      if (bboxHit) { this.bbox = [{}]; this.render(); this.updateData(); return; }
      this.showContextMenu(e);
      return;
    }

    if (e.button !== 0) return;

    if (e.ctrlKey) {
      this.drawing = true;
      this.dragType = 'bbox-draw';
      this.bbox[0] = { startX: clamped.x, startY: clamped.y, endX: clamped.x, endY: clamped.y };
      this.render(); this.startDocumentDrag(e); return;
    }

    const bboxHit = this.hitTestBbox(mouse.x, mouse.y);
    if (bboxHit) {
      const n = this.normBbox();
      this.bboxAtDragStart = { startX: n.x1, startY: n.y1, endX: n.x2, endY: n.y2 };
      this.dragStart = { x: mouse.x, y: mouse.y };
      this.dragType = bboxHit === "move" ? 'bbox-move' : bboxHit;
      this.dragIndex = 0;
      this.startDocumentDrag(e); return;
    }

    const posIdx = this.findPointAt(mouse.x, mouse.y, this.points);
    if (posIdx >= 0) {
      this.dragType = 'positive'; this.dragIndex = posIdx; this.selectedIndex = posIdx;
      this.dragOffset = { x: mouse.x - this.points[posIdx].x, y: mouse.y - this.points[posIdx].y };
      this.render(); this.startDocumentDrag(e); return;
    }

    const negIdx = this.findPointAt(mouse.x, mouse.y, this.neg_points);
    if (negIdx >= 0) {
      this.dragType = 'negative'; this.dragIndex = negIdx; this.selectedIndex = negIdx;
      this.dragOffset = { x: mouse.x - this.neg_points[negIdx].x, y: mouse.y - this.neg_points[negIdx].y };
      this.render(); this.startDocumentDrag(e); return;
    }
  }

  onMouseMove(e) {
    const mouse = this.getLocalMouse(e);
    const clamped = this.clamp(mouse.x, mouse.y);

    if (!this.drawing && this.dragIndex < 0 && !this.dragType) {
      const bboxHit = this.hitTestBbox(mouse.x, mouse.y);
      const bboxCursor = cursorForBboxMode(bboxHit);
      if (bboxCursor) { this.canvas.style.cursor = bboxCursor; }
      else if (this.findPointAt(mouse.x, mouse.y, this.points) >= 0 ||
               this.findPointAt(mouse.x, mouse.y, this.neg_points) >= 0) { this.canvas.style.cursor = "move"; }
      else { this.canvas.style.cursor = "default"; }
      return;
    }

    if (this.dragType === 'bbox-draw') {
      this.bbox[0].endX = clamped.x; this.bbox[0].endY = clamped.y;
      this.render(); return;
    }

    if (this.dragType === 'bbox-move' && this.bboxAtDragStart) {
      const dx = mouse.x - this.dragStart.x, dy = mouse.y - this.dragStart.y;
      const bs = this.bboxAtDragStart;
      const bw = bs.endX - bs.startX, bh = bs.endY - bs.startY;
      const nx = Math.max(0, Math.min(this.coordWidth - bw, bs.startX + dx));
      const ny = Math.max(0, Math.min(this.coordHeight - bh, bs.startY + dy));
      this.bbox[0] = { startX: nx, startY: ny, endX: nx + bw, endY: ny + bh };
      this.render(); return;
    }

    if (this.dragType?.startsWith('resize-') && this.bboxAtDragStart) {
      const bs = this.bboxAtDragStart;
      const offX = this.dragStart.x - (this.dragType === 'resize-tl' || this.dragType === 'resize-bl' ? bs.startX : bs.endX);
      const offY = this.dragStart.y - (this.dragType === 'resize-tl' || this.dragType === 'resize-tr' ? bs.startY : bs.endY);
      const cx = Math.max(0, Math.min(this.coordWidth, mouse.x - offX));
      const cy = Math.max(0, Math.min(this.coordHeight, mouse.y - offY));
      if (this.dragType === 'resize-tl') this.bbox[0] = { startX: bs.endX, startY: bs.endY, endX: cx, endY: cy };
      else if (this.dragType === 'resize-tr') this.bbox[0] = { startX: bs.startX, startY: bs.endY, endX: cx, endY: cy };
      else if (this.dragType === 'resize-bl') this.bbox[0] = { startX: bs.endX, startY: bs.startY, endX: cx, endY: cy };
      else this.bbox[0] = { startX: bs.startX, startY: bs.startY, endX: cx, endY: cy };
      this.render(); return;
    }

    if (this.dragType === 'positive' && this.dragIndex >= 0 && this.dragIndex < this.points.length) {
      const ox = this.dragOffset?.x || 0, oy = this.dragOffset?.y || 0;
      const target = this.clamp(mouse.x - ox, mouse.y - oy);
      this.points[this.dragIndex] = { x: target.x, y: target.y };
      this.render();
    } else if (this.dragType === 'negative' && this.dragIndex >= 0 && this.dragIndex < this.neg_points.length) {
      const ox = this.dragOffset?.x || 0, oy = this.dragOffset?.y || 0;
      const target = this.clamp(mouse.x - ox, mouse.y - oy);
      this.neg_points[this.dragIndex] = { x: target.x, y: target.y };
      this.render();
    }
  }

  onMouseUp() {
    if (this.drawing || this.dragType) {
      this.drawing = false;
      this.bboxAtDragStart = null;
      this.dragStart = null;
      this.endDrag();
      this.updateData();
    }
  }

  // ─── Rendering ───

  _render() {
    const ctx = this.ctx;
    this.beginRender();

    // Bounding box
    const nb = this.normBbox();
    if (nb) {
      const tl = this.toCanvas(nb.x1, nb.y1), br = this.toCanvas(nb.x2, nb.y2);
      const bw = br.x - tl.x, bh = br.y - tl.y;
      ctx.fillStyle = 'rgba(70, 130, 180, 0.3)';
      ctx.fillRect(tl.x, tl.y, bw, bh);
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)'; ctx.lineWidth = 4;
      ctx.strokeRect(tl.x, tl.y, bw, bh);
      ctx.strokeStyle = 'steelblue'; ctx.lineWidth = 2;
      ctx.strokeRect(tl.x, tl.y, bw, bh);
      // Center cross
      const cx = (tl.x + br.x) / 2, cy = (tl.y + br.y) / 2;
      ctx.strokeStyle = 'steelblue'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx - 5, cy); ctx.lineTo(cx + 5, cy);
      ctx.moveTo(cx, cy - 5); ctx.lineTo(cx, cy + 5); ctx.stroke();
      // Corner handles
      const hs = 5;
      ctx.fillStyle = 'steelblue';
      for (const [hx, hy] of [[tl.x, tl.y], [br.x, tl.y], [tl.x, br.y], [br.x, br.y]]) {
        ctx.fillRect(hx - hs, hy - hs, hs * 2, hs * 2);
      }
    }

    // Points
    this._badgePositions = [];
    this._drawPoints(ctx, this.points, '#139613', '#07f907', 'positive');
    // Negative points
    this._drawPoints(ctx, this.neg_points, '#891616', '#f91111', 'negative');

    this.endRender();
  }

  _drawPoints(ctx, points, color, selectedColor, type) {
    const r = this.pointRadius;

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const cp = this.toCanvas(p.x, p.y);
      const isSelected = (this.dragType === type && this.dragIndex === i);
      // Outer circle
      ctx.beginPath(); ctx.arc(cp.x, cp.y, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(100, 100, 100, 0.25)'; ctx.fill();
      ctx.strokeStyle = isSelected ? selectedColor : color; ctx.lineWidth = 2; ctx.stroke();
      // Center dot
      ctx.beginPath(); ctx.arc(cp.x, cp.y, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = 'red'; ctx.fill();
      // Index badge
      const label = i.toString();
      const badgeR = label.length > 1 ? 8 : 7;
      const off = r * 0.7;
      const candidates = [
        { x: cp.x + off, y: cp.y - off },
        { x: cp.x - off, y: cp.y - off },
        { x: cp.x + off, y: cp.y + off },
        { x: cp.x - off, y: cp.y + off },
      ];
      const valid = candidates.filter(c => c.x - badgeR >= 0 && c.x + badgeR <= this.width && c.y - badgeR >= 0 && c.y + badgeR <= this.height);
      const pool = valid.length > 0 ? valid : candidates;
      let badge = pool[0];
      const minDist = badgeR * 2.2;
      for (const c of pool) {
        const overlaps = this._badgePositions.some(b => Math.abs(c.x - b.x) < minDist && Math.abs(c.y - b.y) < minDist);
        if (!overlaps) { badge = c; break; }
      }
      this._badgePositions.push(badge);

      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.beginPath(); ctx.arc(badge.x, badge.y, badgeR, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? selectedColor : color; ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillText(label, badge.x, badge.y + 0.5);
      ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
    }
  }

  // ─── Data ───

  updateData() {
    const combinedPoints = { positive: this.points || [], negative: this.neg_points || [] };
    this.pointsStoreWidget.value = JSON.stringify(combinedPoints);
    this.pos_coordWidget.value = JSON.stringify(this.points || []);
    this.neg_coordWidget.value = JSON.stringify(this.neg_points || []);
    if (this.bbox.length !== 0) {
      let bboxString = JSON.stringify(this.bbox);
      this.bboxStoreWidget.value = bboxString;
      this.bboxWidget.value = bboxString;
    }
  }

}
