import { chainCallback } from '../utility.js';
import { BaseEditorCanvas, createEditorStylesheet } from './editor_base.js';
import { Interpolation } from './interpolation.js';
const { app } = window.comfyAPI.app;

createEditorStylesheet('kj-splineeditor-stylesheet', 'spline-editor');

// ─── Hidden SVG for path sampling (singleton) ───
let _sharedSampler = null;
let _samplerRefCount = 0;
class PathSampler {
  constructor() {
    const ns = 'http://www.w3.org/2000/svg';
    this._svg = document.createElementNS(ns, 'svg');
    this._svg.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden;pointer-events:none;';
    this._svg.setAttribute('aria-hidden', 'true');
    document.body.appendChild(this._svg);
    this._path = document.createElementNS(ns, 'path');
    this._svg.appendChild(this._path);
  }
  static acquire() {
    if (_sharedSampler) { _samplerRefCount++; return _sharedSampler; }
    _sharedSampler = new PathSampler();
    _samplerRefCount = 1;
    return _sharedSampler;
  }
  setPath(d) { this._path.setAttribute('d', d); }
  getTotalLength() { return this._path.getTotalLength(); }
  getPointAtLength(len) { return this._path.getPointAtLength(len); }
  release() {
    _samplerRefCount--;
    if (_samplerRefCount <= 0 && this._svg && this._svg.parentNode) {
      this._svg.parentNode.removeChild(this._svg);
      _sharedSampler = null;
    }
  }
  findPointAtX(targetX, pathLength) {
    let low = 0, high = pathLength, bestPoint = this.getPointAtLength(0);
    const epsilon = 0.5;
    while (high - low > epsilon) {
      const mid = (low + high) / 2, point = this.getPointAtLength(mid);
      if (Math.abs(point.x - targetX) < Math.abs(bestPoint.x - targetX)) bestPoint = point;
      if (Math.abs(point.x - targetX) < 0.5) return point;
      if (point.x < targetX) low = mid; else high = mid;
    }
    return bestPoint;
  }
}

// ─── Extension Registration ───
app.registerExtension({
  name: 'KJNodes.SplineEditor',

  init() {
    app.ui.settings.addSetting({
      id: "KJNodes.splineEditor.showControlLines",
      name: "Show control lines",
      category: ["KJNodes", "Editors", "Show control lines"],
      tooltip: "Display straight lines between control points",
      type: "boolean",
      defaultValue: false,
    });
    app.ui.settings.addSetting({
      id: "KJNodes.splineEditor.showSamplePoints",
      name: "Show sample points",
      category: ["KJNodes", "Editors", "Show sample points"],
      tooltip: "Display the sampled output points as red dots",
      type: "boolean",
      defaultValue: false,
    });
    app.ui.settings.addSetting({
      id: "KJNodes.splineEditor.showArrows",
      name: "Show direction arrows",
      category: ["KJNodes", "Editors", "Show direction arrows"],
      tooltip: "Display chevron arrows on control points showing curve direction",
      type: "boolean",
      defaultValue: true,
    });
    app.ui.settings.addSetting({
      id: "KJNodes.editors.embedBackgroundImage",
      name: "Embed background image in workflow",
      category: ["KJNodes", "Editors", "Embed background image"],
      tooltip: "When enabled, editor background images are embedded as base64 in the workflow file (portable, larger files). When disabled, images are stored as temp files on the server (small workflows, survives refresh, but not portable).",
      type: "boolean",
      defaultValue: false,
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === 'SplineEditor') {
      chainCallback(nodeType.prototype, "onNodeCreated", function () {
        BaseEditorCanvas.setupNode(this, nodeData, {
          editorClass: SplineEditor,
          editorKey: 'splineEditor',
          heightKey: 'splineEditorHeight',
          className: 'spline-editor',
          menuClassName: 'spline-editor-context-menu',
          menuItems: {
            "Display control lines":  { toggle: (ed) => ed.drawHandles, action: (ed) => { ed.drawHandles = !ed.drawHandles; app.ui.settings.setSettingValue("KJNodes.splineEditor.showControlLines", ed.drawHandles); ed.render(); } },
            "Display sample points":  { toggle: (ed) => ed.drawSamplePoints, action: (ed) => { ed.drawSamplePoints = !ed.drawSamplePoints; app.ui.settings.setSettingValue("KJNodes.splineEditor.showSamplePoints", ed.drawSamplePoints); ed.updatePath(); } },
            "Display arrows":         { toggle: (ed) => ed.showChevrons, action: (ed) => { ed.showChevrons = !ed.showChevrons; app.ui.settings.setSettingValue("KJNodes.splineEditor.showArrows", ed.showChevrons); ed.render(); } },
            "Background image":       { action: (ed) => ed.openImageFilePicker() },
            "Invert point order":     { action: (ed) => { ed.splines[ed.activeSplineIndex].points.reverse(); ed.render(); ed.updatePath(); } },
            "Clear Image":            { action: (ed) => { ed.clearBackgroundImage(); ed.drawRuler = true; } },
            "Add new spline":         { action: (ed) => { const idx = ed.splines.length; ed.splines.push({ points: [{ x: 0, y: ed.coordHeight }, { x: ed.coordWidth / 2, y: ed.coordHeight / 2 }, { x: ed.coordWidth, y: 0 }], color: ed.getSplineColor(idx), name: `Spline ${idx + 1}` }); ed.activeSplineIndex = idx; ed.render(); ed.updatePath(); } },
            "Add new single point":   { action: (ed) => { const idx = ed.splines.length; ed.splines.push({ points: [{ x: ed.lastContextMenuPos.x, y: ed.lastContextMenuPos.y }], color: ed.getSplineColor(idx), name: `Spline ${idx + 1}`, isSinglePoint: true }); ed.activeSplineIndex = idx; ed.render(); ed.updatePath(); } },
            "Delete current spline":  { action: (ed) => { if (ed.splines.length > 1) { ed.splines.splice(ed.activeSplineIndex, 1); ed.activeSplineIndex = Math.min(ed.activeSplineIndex, ed.splines.length - 1); ed.render(); ed.updatePath(); } } },
            "Next spline":            { action: (ed) => { ed.activeSplineIndex = (ed.activeSplineIndex + 1) % ed.splines.length; ed.render(); ed.updatePath(); } },
          },
          hiddenWidgets: ["coordinates"],
          initialSize: [550, 1000],
          extraProperties: [
            ["points", this.constructor.type, "string"],
          ],
        });
      });
    }
  }
});

// ─── SplineEditor class ───
class SplineEditor extends BaseEditorCanvas {
  constructor(context, reset = false) {
    super(context, reset);
    this.initEditorPreamble('splineEditor', 'spline-editor');

    this.sampler = PathSampler.acquire();
    this.drawSamplePoints = app.ui.settings.getSettingValue("KJNodes.splineEditor.showSamplePoints") ?? false;
    this.drawHandles = app.ui.settings.getSettingValue("KJNodes.splineEditor.showControlLines") ?? false;
    this.showChevrons = app.ui.settings.getSettingValue("KJNodes.splineEditor.showArrows") ?? true;
    this.drawRuler = true;

    this.coordWidget = this.findWidget("coordinates");
    this.interpolationWidget = this.findWidget("interpolation");
    this.pointsWidget = this.findWidget("points_to_sample");
    this.pointsStoreWidget = this.findWidget("points_store");
    this.tensionWidget = this.findWidget("tension");
    this.samplingMethodWidget = this.findWidget("sampling_method");
    this.widthWidget = this.findWidget("mask_width");
    this.heightWidget = this.findWidget("mask_height");

    this.interpolationWidget.callback = () => this.updatePath(true);
    this.samplingMethodWidget.callback = () => {
      if (this.samplingMethod === "controlpoints") this.drawSamplePoints = true;
      else this.drawSamplePoints = false;
      if (this.samplingMethod === "path" || this.samplingMethod === "speed") this.showChevrons = true;
      this.updatePath(true);
    };
    this.tensionWidget.callback = () => this.updatePath(true);
    this.pointsWidget.callback = () => this.updatePath(true);
    this.setupSizeCallbacks();
    this.pointsStoreWidget.callback = () => {
      this.parseSplineData();
      this.render(); this.updatePath(true);
    };

    this.initDisplaySize();
    this.splines = [];
    this.activeSplineIndex = 0;
    this.hoverSplineIndex = -1;
    this.hoverIndex = -1;
    this.sampledCoords = null;
    this.lastContextMenuPos = { x: this.coordWidth / 2, y: this.coordHeight / 2 };

    if (!reset && this.pointsStoreWidget.value !== "") {
      this.parseSplineData();
    } else {
      this.initializeDefaultSplines();
      this.pointsStoreWidget.value = JSON.stringify(this.splines);
    }

    this._onKeyUp = (e) => {
      if (e.key === 'Control' && this.subdividePreview) { this.subdividePreview = null; this.render(); }
    };
    document.addEventListener('keyup', this._onKeyUp);

    this.initEditor('splineEditor', 'splineEditorHeight', 460);
    this.updatePath();
    this.refreshBackgroundImage();
  }

  // ─── Widget value getters ───
  get interpolation() { return this.interpolationWidget.value; }
  get tension() { return this.tensionWidget.value; }
  get points_to_sample() { return this.pointsWidget.value; }
  get samplingMethod() { return this.samplingMethodWidget.value; }

  // ─── Base class hooks ───

  destroy() {
    super.destroy();
    if (this.sampler) this.sampler.release();
  }

  onDataChanged() { this.updatePath(true); }

  onImageResize() { this.drawRuler = false; }

  onCoordSpaceResized(oldWidth, oldHeight) {
    const sx = this.coordWidth / oldWidth, sy = this.coordHeight / oldHeight;
    const cw = this.coordWidth, ch = this.coordHeight;
    const clamp = (v, max) => Math.max(0, Math.min(max, v));
    for (const spline of this.splines) {
      for (const p of spline.points) {
        p.x = clamp(p.x * sx, cw); p.y = clamp(p.y * sy, ch);
        if (p.h1x !== undefined) { p.h1x = clamp(p.h1x * sx, cw); p.h1y = clamp(p.h1y * sy, ch); }
        if (p.h2x !== undefined) { p.h2x = clamp(p.h2x * sx, cw); p.h2y = clamp(p.h2y * sy, ch); }
      }
    }
  }

  // ─── Spline Helpers ───

  parseSplineData() {
    try {
      const parsed = JSON.parse(this.pointsStoreWidget.value);
      if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].hasOwnProperty('points')) {
        this.splines = parsed;
      } else {
        this.splines = [{ points: parsed, color: "#1f77b4", name: "Spline 1" }];
      }
    } catch (e) {
      console.error("Error parsing spline data:", e);
      this.initializeDefaultSplines();
    }
    if (this.activeSplineIndex >= this.splines.length) {
      this.activeSplineIndex = Math.max(0, this.splines.length - 1);
    }
  }

  initializeDefaultSplines() {
    this.splines = [{
      points: [
        { x: 0, y: this.coordHeight },
        { x: this.coordWidth / 5 * 2, y: 50 + Math.random() * (this.coordHeight - 100) },
        { x: this.coordWidth, y: 0 }
      ],
      color: this.getSplineColor(0),
      name: "Spline 1"
    }];
  }

  getSplineColor(index) {
    const colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
    return colors[index % colors.length];
  }

  // ─── Hit Testing ───

  findPointAt(x, y) {
    const activePoints = this.splines[this.activeSplineIndex]?.points;
    if (!activePoints) return -1;
    const r = 12 / Math.min(this.scaleX, this.scaleY);
    for (let i = activePoints.length - 1; i >= 0; i--) {
      const dx = activePoints[i].x - x, dy = activePoints[i].y - y;
      if (dx * dx + dy * dy <= r * r) return i;
    }
    return -1;
  }

  findHandleAt(x, y) {
    if (this.interpolation !== 'bezier') return null;
    const activePoints = this.splines[this.activeSplineIndex]?.points;
    if (!activePoints) return null;
    const r = 8 / Math.min(this.scaleX, this.scaleY);
    for (let i = activePoints.length - 1; i >= 0; i--) {
      const p = activePoints[i];
      if (i > 0 && p.h1x !== undefined) { const dx = p.h1x - x, dy = p.h1y - y; if (dx * dx + dy * dy <= r * r) return { index: i, type: 'h1' }; }
      if (i < activePoints.length - 1 && p.h2x !== undefined) { const dx = p.h2x - x, dy = p.h2y - y; if (dx * dx + dy * dy <= r * r) return { index: i, type: 'h2' }; }
    }
    return null;
  }

  findSplineAt(x, y) {
    const threshold = 15 / Math.min(this.scaleX, this.scaleY);
    let bestDist = Infinity, bestIdx = -1;
    for (let si = 0; si < this.splines.length; si++) {
      const spline = this.splines[si];
      if (spline.isSinglePoint || (spline.points && spline.points.length === 1)) {
        const p = spline.points[0], d = Math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2);
        if (d < threshold && d < bestDist) { bestDist = d; bestIdx = si; }
        continue;
      }
      const pathD = Interpolation.buildPathD(spline.points, this.interpolation, this.tension);
      this.sampler.setPath(pathD);
      const len = this.sampler.getTotalLength();
      const steps = Math.min(200, Math.max(50, Math.round(len / 3)));
      for (let i = 0; i <= steps; i++) {
        const pt = this.sampler.getPointAtLength(len * i / steps);
        const d = Math.sqrt((pt.x - x) ** 2 + (pt.y - y) ** 2);
        if (d < threshold && d < bestDist) { bestDist = d; bestIdx = si; }
      }
    }
    return bestIdx;
  }

  findClosestPoints(points, clickedPoint) {
    if (points.length < 2) return null;
    let distances = points.map((point, idx) => {
      const dx = clickedPoint.x - point.x, dy = clickedPoint.y - point.y;
      return { index: idx, distance: Math.sqrt(dx * dx + dy * dy) };
    });
    distances.sort((a, b) => a.distance - b.distance);
    let i1 = distances[0].index, i2 = distances[1].index;
    if (i1 > i2) [i1, i2] = [i2, i1];
    return { point1Index: i1, point2Index: i2 };
  }

  // ─── Mouse Handlers ───

  onMouseDown(e) {
    const mouse = this.getLocalMouse(e);
    const clamped = this.clamp(mouse.x, mouse.y);
    const activeSpline = this.splines[this.activeSplineIndex];
    const activePoints = activeSpline?.points;

    if (e.shiftKey && e.button === 0 && activePoints) {
      activePoints.push({ x: clamped.x, y: clamped.y });
      this.render(); this.updatePath(); return;
    }
    if (e.ctrlKey && e.button === 0 && activePoints && activePoints.length >= 2) {
      const closest = this.findClosestPoints(activePoints, clamped);
      if (closest) {
        const p1 = activePoints[closest.point1Index], p2 = activePoints[closest.point2Index];
        activePoints.splice(closest.point2Index, 0, { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 });
        this.render(); this.updatePath();
      }
      return;
    }

    if (e.button === 2) {
      const ptIdx = this.findPointAt(mouse.x, mouse.y);
      if (activePoints && ptIdx > 0 && ptIdx < activePoints.length - 1) {
        activePoints.splice(ptIdx, 1); this.render(); this.updatePath(); return;
      }
      this.lastContextMenuPos = { x: clamped.x, y: clamped.y };
      this.showContextMenu(e);
      return;
    }

    if (e.button !== 0) return;

    // Bezier handle drag
    const handle = this.findHandleAt(mouse.x, mouse.y);
    if (handle && activePoints) {
      this.dragIndex = handle.index; this.dragType = handle.type;
      const p = activePoints[handle.index];
      const hx = handle.type === 'h1' ? p.h1x : p.h2x, hy = handle.type === 'h1' ? p.h1y : p.h2y;
      this.dragOffset = { x: mouse.x - hx, y: mouse.y - hy };
      this.render(); this.startDocumentDrag(e); return;
    }

    // Point drag
    const ptIdx = this.findPointAt(mouse.x, mouse.y);
    if (ptIdx >= 0) {
      this.dragIndex = ptIdx; this.dragType = 'point'; this.hoverIndex = ptIdx;
      const p = activePoints[ptIdx];
      this.dragOffset = { x: mouse.x - p.x, y: mouse.y - p.y };
      this.render(); this.startDocumentDrag(e); return;
    }

    // Spline selection
    const splineIdx = this.findSplineAt(mouse.x, mouse.y);
    if (splineIdx >= 0 && splineIdx !== this.activeSplineIndex) {
      this.activeSplineIndex = splineIdx; this.render(); this.updatePath(); return;
    }
  }

  onMouseMove(e) {
    const mouse = this.getLocalMouse(e);
    const clamped = this.clamp(mouse.x, mouse.y);

    // Bezier handle drag
    if ((this.dragType === 'h1' || this.dragType === 'h2') && this.dragIndex >= 0) {
      const activePoints = this.splines[this.activeSplineIndex]?.points;
      if (activePoints && this.dragIndex < activePoints.length) {
        const p = activePoints[this.dragIndex];
        const clampH = (hx, hy) => ({ x: Math.max(0, Math.min(this.coordWidth, hx)), y: Math.max(0, Math.min(this.coordHeight, hy)) });
        const ox = this.dragOffset?.x || 0, oy = this.dragOffset?.y || 0;
        const prevH1x = p.h1x, prevH1y = p.h1y, prevH2x = p.h2x, prevH2y = p.h2y;
        if (this.dragType === 'h1') {
          const c = clampH(mouse.x - ox, mouse.y - oy);
          p.h1x = c.x; p.h1y = c.y;
          if (!e.altKey) { const dx = p.x - p.h1x, dy = p.y - p.h1y; const mc = clampH(p.x + dx, p.y + dy); p.h2x = mc.x; p.h2y = mc.y; }
        } else {
          const c = clampH(mouse.x - ox, mouse.y - oy);
          p.h2x = c.x; p.h2y = c.y;
          if (!e.altKey) { const dx = p.x - p.h2x, dy = p.y - p.h2y; const mc = clampH(p.x + dx, p.y + dy); p.h1x = mc.x; p.h1y = mc.y; }
        }
        if (p.h1x === prevH1x && p.h1y === prevH1y && p.h2x === prevH2x && p.h2y === prevH2y) return;
        this.render();
      }
      return;
    }

    // Point drag
    if (this.dragType === 'point' && this.dragIndex >= 0) {
      const activePoints = this.splines[this.activeSplineIndex]?.points;
      if (activePoints && this.dragIndex < activePoints.length) {
        const p = activePoints[this.dragIndex];
        const ox = this.dragOffset?.x || 0, oy = this.dragOffset?.y || 0;
        const target = this.clamp(mouse.x - ox, mouse.y - oy);
        if (target.x === p.x && target.y === p.y) return;
        const dx = target.x - p.x, dy = target.y - p.y;
        if (this.interpolation === 'bezier' && p.h1x !== undefined) {
          p.h1x += dx; p.h1y += dy; p.h2x += dx; p.h2y += dy;
        }
        p.x = target.x; p.y = target.y;
        this.render();
      }
      return;
    }

    // Hover detection
    if (!this.dragType) {
      const ptIdx = this.findPointAt(mouse.x, mouse.y);
      const activePoints = this.splines[this.activeSplineIndex]?.points;

      // Ctrl preview
      if (e.ctrlKey && activePoints && activePoints.length >= 2) {
        const closest = this.findClosestPoints(activePoints, clamped);
        if (closest) {
          const p1 = activePoints[closest.point1Index], p2 = activePoints[closest.point2Index];
          this.subdividePreview = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
          this.canvas.style.cursor = 'copy'; this.render(); return;
        }
      } else if (this.subdividePreview) { this.subdividePreview = null; this.render(); }

      // Handle hover
      const handleHit = this.findHandleAt(mouse.x, mouse.y);
      if (handleHit) { this.canvas.style.cursor = 'crosshair'; if (ptIdx !== this.hoverIndex) { this.hoverIndex = ptIdx; this.render(); } return; }

      if (ptIdx !== this.hoverIndex) {
        this.hoverIndex = ptIdx;
        this.canvas.style.cursor = ptIdx >= 0 ? 'move' : 'default';
        this.render();
      }

      // Spline hover
      if (ptIdx < 0 && this.splines.length > 1) {
        const splineIdx = this.findSplineAt(mouse.x, mouse.y);
        const newSplineHover = splineIdx >= 0 && splineIdx !== this.activeSplineIndex ? splineIdx : -1;
        if (newSplineHover !== this.hoverSplineIndex) {
          this.hoverSplineIndex = newSplineHover;
          this.canvas.style.cursor = newSplineHover >= 0 ? 'pointer' : 'default';
          this.render();
        }
      } else if (this.hoverSplineIndex >= 0) { this.hoverSplineIndex = -1; this.render(); }
    }
  }

  onMouseUp() {
    if (this.dragType) {
      this.endDrag();
      this.updatePath();
    }
  }

  // ─── Path Building & Sampling ───

  samplePoints(splineIndex, numSamples, samplingMethod) {
    const spline = this.splines[splineIndex];
    if (!spline || !spline.points || spline.points.length < 2) return [];
    if (spline.isSinglePoint) {
      const point = spline.points[0];
      return Array(numSamples).fill().map(() => ({ x: point.x, y: point.y }));
    }
    if (numSamples < 2) {
      const p = spline.points[0];
      return [{ x: p.x, y: p.y }];
    }
    const pathD = Interpolation.buildPathD(spline.points, this.interpolation, this.tension);
    if (!pathD) return [];

    this.sampler.setPath(pathD);
    const pathLength = this.sampler.getTotalLength();
    const points = [];

    if (samplingMethod === "speed") {
      if (pathLength < 0.001) return [{ x: spline.points[0].x, y: spline.points[0].y }];
      const controlPoints = spline.points;
      // Pair each control point with its path position, then sort together
      const cpWithPos = controlPoints.map(cp => {
        let bestDist = Infinity, bestPos = 0;
        for (let pos = 0; pos <= pathLength; pos += pathLength / 100) {
          const pt = this.sampler.getPointAtLength(pos);
          const dist = Math.sqrt((pt.x - cp.x) ** 2 + (pt.y - cp.y) ** 2);
          if (dist < bestDist) { bestDist = dist; bestPos = pos; }
        }
        return { cp, pos: bestPos };
      });
      cpWithPos.sort((a, b) => a.pos - b.pos);
      const pathPositions = cpWithPos.map(c => c.pos);

      const densities = [];
      let totalWeight = 0;
      for (let i = 0; i < cpWithPos.length - 1; i++) {
        const segLength = pathPositions[i + 1] - pathPositions[i];
        const d = 1 / Math.max(segLength, 0.0001);
        densities.push(d);
        totalWeight += d;
      }
      const cumulativeWeights = [];
      let cum = 0;
      for (let i = 0; i < densities.length; i++) {
        cum += densities[i] / totalWeight;
        cumulativeWeights.push(cum);
      }
      const mapToPath = (t) => {
        if (t === 0) return 0; if (t === 1) return pathLength;
        let segIdx = cumulativeWeights.length - 1;
        for (let i = 0; i < cumulativeWeights.length; i++) { if (t <= cumulativeWeights[i]) { segIdx = i; break; } }
        const segStart = segIdx > 0 ? cumulativeWeights[segIdx - 1] : 0;
        const segT = (t - segStart) / (cumulativeWeights[segIdx] - segStart);
        return pathPositions[segIdx] + segT * (pathPositions[segIdx + 1] - pathPositions[segIdx]);
      };
      for (let i = 0; i < numSamples; i++) {
        const pt = this.sampler.getPointAtLength(mapToPath(i / (numSamples - 1)));
        points.push({ x: pt.x, y: pt.y });
      }
      return points;
    }

    for (let i = 0; i < numSamples; i++) {
      let point;
      if (samplingMethod === "time") {
        point = this.sampler.findPointAtX((this.coordWidth / (numSamples - 1)) * i, pathLength);
      } else {
        point = this.sampler.getPointAtLength((pathLength / (numSamples - 1)) * i);
      }
      points.push({ x: point.x, y: point.y });
    }
    if (points.length > 0 && spline.points.length > 1) {
      points[points.length - 1].y = spline.points[spline.points.length - 1].y;
    }
    return points;
  }

  updatePath(allDirty = false) {
    if (!this.splines || this.splines.length === 0) return;

    // Rebuild cache when spline count changes or full refresh requested
    if (!this._sampledCache || this._sampledCache.length !== this.splines.length || allDirty) {
      this._sampledCache = new Array(this.splines.length).fill(null);
    }

    const method = this.samplingMethod;
    const useControlPoints = method === "controlpoints";
    for (let i = 0; i < this.splines.length; i++) {
      if (useControlPoints) {
        this._sampledCache[i] = this.splines[i].points;
      } else if (i === this.activeSplineIndex || !this._sampledCache[i]) {
        this._sampledCache[i] = this.samplePoints(i, this.points_to_sample, method);
      }
    }
    this.sampledCoords = this._sampledCache[this.activeSplineIndex] || [];
    this.pointsStoreWidget.value = JSON.stringify(this.splines);
    if (this.coordWidget) this.coordWidget.value = JSON.stringify(this._sampledCache);
    this.render();
  }

  // ─── Rendering ───

  _render() {
    const ctx = this.ctx;
    const h = this.coordHeight;
    this.beginRender();

    // Ruler lines
    if (this.drawRuler && !this.bgImage) {
      ctx.strokeStyle = 'rgba(128,128,128,0.3)'; ctx.lineWidth = 1;
      for (let y = 64; y < h; y += 64) {
        const cy = y * this.scaleY;
        ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(this.width, cy); ctx.stroke();
      }
    }

    const tc = (x, y) => this.toCanvas(x, y);

    // Draw all splines — use scale transform for curves, toCanvas for discrete elements
    for (let si = 0; si < this.splines.length; si++) {
      const spline = this.splines[si];
      const isActive = si === this.activeSplineIndex, isHover = si === this.hoverSplineIndex;

      if (spline.isSinglePoint || (spline.points && spline.points.length === 1)) {
        const cp = tc(spline.points[0].x, spline.points[0].y), sz = isActive ? 8 : 6;
        ctx.fillStyle = spline.color; ctx.fillRect(cp.x - sz, cp.y - sz, sz * 2, sz * 2);
        ctx.strokeStyle = 'black'; ctx.lineWidth = 2; ctx.strokeRect(cp.x - sz, cp.y - sz, sz * 2, sz * 2);
        continue;
      }
      if (spline.points.length < 2) continue;

      // Build path once, stroke twice (outline + color) to avoid recomputing interpolation
      const pathD = Interpolation.buildPathD(spline.points, this.interpolation, this.tension);
      const curvePath = new Path2D(pathD);
      ctx.save(); ctx.scale(this.scaleX, this.scaleY);
      ctx.lineWidth = (isActive ? 5 : isHover ? 4 : 3.5) / Math.min(this.scaleX, this.scaleY); ctx.strokeStyle = 'black';
      ctx.stroke(curvePath);
      ctx.lineWidth = (isActive ? 3 : isHover ? 2 : 1.5) / Math.min(this.scaleX, this.scaleY); ctx.strokeStyle = spline.color;
      ctx.stroke(curvePath);
      ctx.restore();
    }

    const activeSpline = this.splines[this.activeSplineIndex];

    // Handle lines
    if (this.drawHandles) {
      const activePoints = this.splines[this.activeSplineIndex]?.points;
      if (activePoints && activePoints.length >= 2) {
        ctx.strokeStyle = '#ff7f0e'; ctx.lineWidth = 1; ctx.beginPath();
        const c0 = tc(activePoints[0].x, activePoints[0].y);
        ctx.moveTo(c0.x, c0.y);
        for (let i = 1; i < activePoints.length; i++) { const ci = tc(activePoints[i].x, activePoints[i].y); ctx.lineTo(ci.x, ci.y); }
        ctx.stroke();
      }
    }

    // Control points for active spline
    if (activeSpline && activeSpline.points && !activeSpline.isSinglePoint) {
      const dotRadius = 12;
      for (let i = 0; i < activeSpline.points.length; i++) {
        const p = activeSpline.points[i];
        const cp = tc(p.x, p.y);
        const isHovered = this.hoverIndex === i;
        ctx.fillStyle = 'rgba(100, 100, 100, 0.3)';
        ctx.strokeStyle = isHovered ? '#ff7f0e' : '#1f77b4'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cp.x, cp.y, dotRadius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

        // Chevron
        const isEndpoint = i === 0 || i === activeSpline.points.length - 1;
        const showChevron = this.showChevrons && (
          (this.samplingMethod === 'path' || this.samplingMethod === 'speed')
          || (this.samplingMethod === 'time' && isEndpoint));

        if (showChevron && activeSpline.points.length > 1) {
          let angle = 0;
          if (i > 0 && i < activeSpline.points.length - 1) { const prev = activeSpline.points[i-1], next = activeSpline.points[i+1]; angle = Math.atan2((next.y - prev.y) * this.scaleY, (next.x - prev.x) * this.scaleX); }
          else if (i === 0) { const next = activeSpline.points[1]; angle = Math.atan2((next.y - p.y) * this.scaleY, (next.x - p.x) * this.scaleX); }
          else { const prev = activeSpline.points[i-1]; angle = Math.atan2((p.y - prev.y) * this.scaleY, (p.x - prev.x) * this.scaleX); }
          ctx.save();
          ctx.translate(cp.x, cp.y);
          ctx.rotate(angle);
          ctx.strokeStyle = isHovered ? '#fff' : 'rgba(255,255,255,0.7)';
          ctx.lineWidth = 2.5; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
          ctx.beginPath();
          ctx.moveTo(-dotRadius * 0.2, -dotRadius * 0.35);
          ctx.lineTo(dotRadius * 0.3, 0);
          ctx.lineTo(-dotRadius * 0.2, dotRadius * 0.35);
          ctx.stroke();
          ctx.lineCap = 'butt'; ctx.lineJoin = 'miter';
          ctx.restore();
        }

        if (isHovered) {
          ctx.font = '11px monospace';
          const label = `${i}: (${Math.round(p.x)}, ${Math.round(p.y)})`;
          ctx.strokeStyle = '#000'; ctx.lineWidth = 3; ctx.strokeText(label, cp.x + dotRadius + 5, cp.y - 5);
          ctx.fillStyle = '#fff'; ctx.fillText(label, cp.x + dotRadius + 5, cp.y - 5);
        }
      }
    }

    // Bezier handles
    if (this.interpolation === 'bezier' && activeSpline && activeSpline.points && !activeSpline.isSinglePoint) {
      const pts = activeSpline.points;
      Interpolation.ensureBezierHandles(pts);
      const hr = 5;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const cp = tc(p.x, p.y);
        if (i > 0 && p.h1x !== undefined) {
          const h1 = tc(p.h1x, p.h1y);
          ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(cp.x, cp.y); ctx.lineTo(h1.x, h1.y); ctx.stroke();
          ctx.fillStyle = (this.dragType === 'h1' && this.dragIndex === i) ? '#fff' : 'rgba(255,255,255,0.8)';
          ctx.strokeStyle = '#1f77b4'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(h1.x, h1.y, hr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        }
        if (i < pts.length - 1 && p.h2x !== undefined) {
          const h2 = tc(p.h2x, p.h2y);
          ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(cp.x, cp.y); ctx.lineTo(h2.x, h2.y); ctx.stroke();
          ctx.fillStyle = (this.dragType === 'h2' && this.dragIndex === i) ? '#fff' : 'rgba(255,255,255,0.8)';
          ctx.strokeStyle = '#ff7f0e'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(h2.x, h2.y, hr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        }
      }
    }

    // Sample points
    if (this.drawSamplePoints && this.sampledCoords) {
      ctx.fillStyle = 'red'; ctx.strokeStyle = 'black'; ctx.lineWidth = 1;
      for (const pt of this.sampledCoords) { const cp = tc(pt.x, pt.y); ctx.beginPath(); ctx.arc(cp.x, cp.y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
    }

    // Subdivide preview
    if (this.subdividePreview) {
      const sp = tc(this.subdividePreview.x, this.subdividePreview.y);
      ctx.beginPath(); ctx.arc(sp.x, sp.y, 10, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0,255,0,0.3)'; ctx.lineWidth = 4; ctx.stroke();
      ctx.beginPath(); ctx.arc(sp.x, sp.y, 8, 0, Math.PI * 2);
      ctx.strokeStyle = '#0f0'; ctx.lineWidth = 2; ctx.stroke();
    }

    this.endRender();
  }

}
