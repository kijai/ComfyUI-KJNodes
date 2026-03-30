import { addMiddleClickPan, chainCallback, makeUUID, watchImageInputs } from '../utility.js';

export function createEditorStylesheet(id, className) {
  let styleTag = document.head.querySelector(`#${id}`)
  if (!styleTag) {
    styleTag = document.createElement('style')
    styleTag.type = 'text/css'
    styleTag.id = id
    styleTag.innerHTML = `
      .${className} {
        position: absolute;
        font: 12px monospace;
        line-height: 1.5em;
        padding: 10px;
        z-index: 0;
        overflow: hidden;
      }
      .${className} canvas {
        position: relative;
        z-index: 2;
      }
    `
    document.head.appendChild(styleTag)
  }
}

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
  menuItem.dataset.menuId = id;
  menuItem.textContent = textContent;
  styleMenuItem(menuItem);
  return menuItem;
}

function setupMenuItems(contextMenu, menuItems) {
  menuItems.forEach(mi => {
    mi.addEventListener('mouseover', function () { this.style.backgroundColor = "gray"; });
    mi.addEventListener('mouseout', function () { this.style.backgroundColor = "#202020"; });
    contextMenu.appendChild(mi);
  });
}

export function createContextMenuElement(className) {
  const menu = document.createElement("div");
  if (className) menu.className = className;
  menu.id = `context-menu-${Math.random().toString(36).slice(2, 10)}`;
  menu.style.display = "none";
  menu.style.position = "absolute";
  menu.style.backgroundColor = "#202020";
  menu.style.minWidth = "100px";
  menu.style.boxShadow = "0px 8px 16px 0px rgba(0,0,0,0.2)";
  menu.style.zIndex = "100";
  menu.style.padding = "5px";
  return menu;
}

// ─── Base Editor Canvas ───

const maxDisplayDim = 1024;

export class BaseEditorCanvas {
  constructor(context, reset = false) {
    this.node = context;
    this.reset = reset;
    this.bgImage = null;
    this.margin = 14;
    this.dragIndex = -1;
    this.dragType = null;
    this.dragOffset = null;

    this._uploadGeneration = 0;
  }

  // ─── Shared Methods ───

  setNodeWidth(width) {
    this.node.setSize([width, this.node.size[1]]);
    const nodeEl = document.querySelector(`[data-node-id="${this.node.id}"]`);
    if (nodeEl) nodeEl.style.setProperty('--node-width', `${width}px`);
  }

  // Scale factors: coord space → canvas space
  get scaleX() { return this.width / this.coordWidth; }
  get scaleY() { return this.height / this.coordHeight; }

  // Returns mouse position in coord space
  getLocalMouse(e) {
    const rect = this.canvas.getBoundingClientRect();
    const canvasScaleX = this.canvas.width / rect.width;
    const canvasScaleY = this.canvas.height / rect.height;
    const canvasX = (e.clientX - rect.left) * canvasScaleX - this.margin;
    const canvasY = (e.clientY - rect.top) * canvasScaleY - this.margin;
    return {
      x: canvasX / this.scaleX,
      y: canvasY / this.scaleY
    };
  }

  // Clamp to coord space
  clamp(x, y) {
    return {
      x: Math.max(0, Math.min(this.coordWidth, x)),
      y: Math.max(0, Math.min(this.coordHeight, y))
    };
  }

  // ─── Canvas Setup ───

  createCanvas(parentElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.width + this.margin * 2;
    this.canvas.height = this.height + this.margin * 2;
    this.ctx = this.canvas.getContext('2d');
    parentElement.appendChild(this.canvas);
  }

  resizeCanvas() {
    this.canvas.width = this.width + this.margin * 2;
    this.canvas.height = this.height + this.margin * 2;
  }

  // ─── Event Listeners ───

  setupEventListeners() {
    this._onDragMove = (e) => this.onMouseMove(e);
    this._onDragEnd = (e) => this.onMouseUp(e);
    this._removeMiddleClickPan = addMiddleClickPan(this.canvas);
    this._onCanvasPointerDown = (e) => { e.stopPropagation(); this.onMouseDown(e); };
    this._onCanvasPointerMove = (e) => this.onMouseMove(e);
    this._onCanvasContextMenu = (e) => { e.preventDefault(); e.stopPropagation(); };
    this.canvas.addEventListener('pointerdown', this._onCanvasPointerDown);
    this.canvas.addEventListener('pointermove', this._onCanvasPointerMove);
    this.canvas.addEventListener('contextmenu', this._onCanvasContextMenu);
  }

  removeEventListeners() {
    if (this.canvas) {
      this.canvas.removeEventListener('pointerdown', this._onCanvasPointerDown);
      this.canvas.removeEventListener('pointermove', this._onCanvasPointerMove);
      this.canvas.removeEventListener('pointermove', this._onDragMove);
      this.canvas.removeEventListener('pointerup', this._onDragEnd);
      this.canvas.removeEventListener('contextmenu', this._onCanvasContextMenu);
    }
    document.removeEventListener('pointermove', this._onDragMove);
    document.removeEventListener('pointerup', this._onDragEnd);
    if (this._removeMiddleClickPan) this._removeMiddleClickPan();
  }

  startDocumentDrag(e) {
    if (e && e.pointerId !== undefined) {
      this.canvas.setPointerCapture(e.pointerId);
      this._capturedPointerId = e.pointerId;
      // With pointer capture, events fire on the canvas, not document
      this.canvas.addEventListener('pointermove', this._onDragMove);
      this.canvas.addEventListener('pointerup', this._onDragEnd);
    } else {
      document.addEventListener('pointermove', this._onDragMove);
      document.addEventListener('pointerup', this._onDragEnd);
    }
  }

  endDrag() {
    this.dragIndex = -1;
    this.dragType = null;
    this.dragOffset = null;
    this.canvas.style.cursor = 'default';
    if (this._capturedPointerId !== undefined) {
      try { this.canvas.releasePointerCapture(this._capturedPointerId); } catch (_) {}
      this._capturedPointerId = undefined;
      this.canvas.removeEventListener('pointermove', this._onDragMove);
      this.canvas.removeEventListener('pointerup', this._onDragEnd);
    } else {
      document.removeEventListener('pointermove', this._onDragMove);
      document.removeEventListener('pointerup', this._onDragEnd);
    }
  }

  // ─── Render Batching ───

  render() {
    if (this._renderPending) return;
    this._renderPending = true;
    requestAnimationFrame(() => {
      this._renderPending = false;
      this._render();
    });
  }

  // Subclasses must implement _render()

  // ─── Common Render Helpers ───

  // Convert coord-space point to canvas-space pixel
  toCanvas(x, y) {
    return { x: x * this.scaleX, y: y * this.scaleY };
  }

  // Call at start of _render() — clears and draws background
  beginRender() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.save();
    ctx.translate(this.margin, this.margin);
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, this.width, this.height);
    ctx.strokeStyle = 'gray';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, this.width, this.height);
    if (this.bgImage) {
      ctx.drawImage(this.bgImage, 0, 0, this.width, this.height);
    }
  }

  endRender() {
    this.ctx.restore();
  }

  // ─── Image Handling ───

  handleImageLoad = (img, downscaledImg) => {
    // Set coord space to image dimensions
    this.coordWidth = img.width;
    this.coordHeight = img.height;
    this.widthWidget.value = img.width;
    this.heightWidget.value = img.height;
    this.onImageResize?.(img);

    // Cap display size, preserving aspect ratio
    let displayW = img.width, displayH = img.height;
    if (displayW > maxDisplayDim || displayH > maxDisplayDim) {
      const scale = maxDisplayDim / Math.max(displayW, displayH);
      displayW = Math.round(displayW * scale);
      displayH = Math.round(displayH * scale);
    }

    if (displayW !== this.width || displayH !== this.height) {
      this.width = displayW;
      this.height = displayH;
      this.resizeCanvas();

      if (displayW > 256) this.setNodeWidth(displayW + 45);
      this.onSizeChanged();
      if (this.node.graph) {
        try { this.node.arrange?.(); } catch (_) {}
        this.node.graph.setDirtyCanvas(true, true);
      }
    }

    // Use downscaled image if available to avoid holding full-res in memory
    this.bgImage = downscaledImg || img;
    this.render();
    this.onDataChanged();
  };

  // resizeCanvas: if true (default), resize editor canvas to match image.
  // If false, just store the image and update the background without resizing.
  processImage = (img, { resize = true } = {}) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    let width = img.width, height = img.height;
    if (width > maxDisplayDim || height > maxDisplayDim) {
      const scale = maxDisplayDim / Math.max(width, height);
      width = Math.round(width * scale);
      height = Math.round(height * scale);
    }
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    const { app } = window.comfyAPI.app;
    const embed = app.ui.settings.getSettingValue("KJNodes.editors.embedBackgroundImage") ?? false;

    // Use the downscaled canvas directly as bgImage — drawImage accepts canvas elements,
    // avoids a data URL round-trip, and is immediately available (no async decode).
    const onStored = () => {
      if (resize) {
        this.handleImageLoad(img, canvas);
      } else {
        this.bgImage = canvas;
        this.render();
      }
    };

    const gen = ++this._uploadGeneration;
    if (embed) {
      const base64String = canvas.toDataURL('image/webp', 0.5).replace(/^data:.+?,/, '');
      if (gen !== this._uploadGeneration) return;
      this.node.properties.imgData = { type: 'image/webp', base64: base64String };
      onStored();
    } else {
      canvas.toBlob((blob) => {
        if (gen !== this._uploadGeneration) return;
        const filename = `editor_bg_${this.node.id}_${Date.now()}.webp`;
        const formData = new FormData();
        formData.append('image', blob, filename);
        formData.append('type', 'temp');
        formData.append('overwrite', 'true');
        fetch('/upload/image', { method: 'POST', body: formData })
          .then(r => r.json())
          .then(result => {
            if (gen !== this._uploadGeneration) return;
            this.node.properties.imgData = { type: 'temp', filename: result.name };
            onStored();
          })
          .catch(e => console.error("Failed to upload editor background:", e));
      }, 'image/webp', 0.5);
    }
  };

  handleImageFile = (file) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); this.processImage(img); };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
  };

  refreshBackgroundImage = () => {
    const imgData = this.node.properties.imgData;
    if (!imgData) return;

    const img = new Image();
    img.onerror = (e) => console.error("Background image failed to load:", e);
    img.onload = () => {
      // Just set the background — don't resize canvas, widget values are already correct from serialization
      this.bgImage = img;
      this.render();
    };

    if (imgData.base64) {
      const mimeType = imgData.type || 'image/png';
      img.src = `data:${mimeType};base64,${imgData.base64}`;
    } else if (imgData.filename) {
      img.src = `/view?filename=${encodeURIComponent(imgData.filename)}&type=temp&no-cache=${Date.now()}`;
    }
  };

  // ─── Context Menu Helpers ───

  // Set up document-level listeners for context menu behavior
  setupContextMenuListeners(editorIdPrefix) {
    this._onContextMenu = (e) => {
      if (e.target.closest(`#${editorIdPrefix}-${this.node.uuid}`) ||
          this.node.contextMenu.contains(e.target)) {
        e.preventDefault();
      }
    };
    this._onDocClick = (e) => {
      if (!this.node.contextMenu.contains(e.target)) {
        this.node.contextMenu.style.display = 'none';
      }
    };
    document.addEventListener('contextmenu', this._onContextMenu);
    document.addEventListener('click', this._onDocClick);
  }

  // Clean up previous editor instance
  cleanupPreviousEditor(context) {
    if (context.editor) {
      context.editor.destroy();
    }
  }

  // Full cleanup — override in subclass to add additional cleanup
  destroy() {
    this._uploadGeneration++;  // Invalidate any pending async image uploads
    this.removeEventListeners();
    if (this._onContextMenu) document.removeEventListener('contextmenu', this._onContextMenu);
    if (this._onDocClick) document.removeEventListener('click', this._onDocClick);
    if (this._onKeyUp) document.removeEventListener('keyup', this._onKeyUp);
  }

  // Common menu actions
  openImageFilePicker() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.addEventListener('change', (event) => {
      const file = event.target.files[0];
      if (file) this.handleImageFile(file);
    });
    fileInput.click();
  }

  clearBackgroundImage() {
    this.bgImage = null;
    this.node.properties.imgData = null;
    this.render();
  }

  // Find a widget by name on the node
  findWidget(name) {
    const w = this.node.widgets.find(w => w.name === name);
    if (!w) console.warn(`${this.constructor.name}: widget "${name}" not found`);
    return w;
  }

  // Show context menu at mouse position
  showContextMenu(e) {
    this._updateMenuToggleStates();
    const menu = this.node.contextMenu;
    menu.style.display = 'block';
    menu.style.left = `${e.clientX}px`;
    menu.style.top = `${e.clientY}px`;
    // Adjust if menu overflows viewport
    const rect = menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) menu.style.left = `${Math.max(0, e.clientX - rect.width)}px`;
    if (rect.bottom > window.innerHeight) menu.style.top = `${Math.max(0, e.clientY - rect.height)}px`;
  }

  // ─── Width/Height Resize ───

  setupSizeCallbacks() {
    this.widthWidget.callback = () => {
      const oldCoordW = this.coordWidth;
      this.coordWidth = this.widthWidget.value;
      this.onCoordSpaceResized(oldCoordW, this.coordHeight);
      this.render();
      this.onDataChanged();
    };
    this.heightWidget.callback = () => {
      const oldCoordH = this.coordHeight;
      this.coordHeight = this.heightWidget.value;
      this.onCoordSpaceResized(this.coordWidth, oldCoordH);
      this.render();
      this.onDataChanged();
    };
  }

  // ─── imgData persistence ───
  // Call once from onNodeCreated to ensure imgData survives save/reload
  static setupImgDataPersistence(node) {
    chainCallback(node, "onSerialize", function (o) {
      if (this.properties.imgData && o.properties) {
        o.properties.imgData = this.properties.imgData;
      }
    });
    chainCallback(node, "onConfigure", function (info) {
      if (info?.properties?.imgData) {
        this.properties.imgData = info.properties.imgData;
      }
    });
  }

  // ─── Node registration helper ───
  // Call from onNodeCreated to set up the common editor scaffolding.
  // config: { editorClass, editorKey, heightKey, className, menuItems, hiddenWidgets, initialSize, extraProperties }
  // editorClass: the editor constructor (e.g. SplineEditor)
  // editorKey: node property name for the DOM widget (e.g. 'splineEditor')
  // heightKey: node property name for editor height (e.g. 'splineEditorHeight')
  // className: CSS class for the editor container
  // menuItems: { id: { label, action(editor), toggle?(editor) }, ... } — context menu definition
  // menuClassName: optional CSS class for the context menu
  // hiddenWidgets: widget names to hide
  // initialSize: [width, height] for the node
  // extraProperties: array of [name, default, type] to register via addProperty on first create
  static setupNode(node, nodeData, config) {
    const { editorClass, editorKey, heightKey, className, menuItems,
            menuClassName, hiddenWidgets, initialSize, extraProperties } = config;

    if (!node.properties) node.properties = {};
    if (node.properties.imgData === undefined) node.properties.imgData = null;

    // File handling — set up once per node, delegate to current editor
    node.pasteFile = (file) => {
      if (node.editor && file.type.startsWith("image/")) {
        node.editor.handleImageFile(file);
        return true;
      }
      return false;
    };
    node.onDragOver = (e) => {
      if (node.editor && e.dataTransfer && e.dataTransfer.items) {
        return [...e.dataTransfer.items].some(f => f.kind === "file" && f.type.startsWith("image/"));
      }
      return false;
    };
    node.onDragDrop = (e) => {
      if (!node.editor) return false;
      let handled = false;
      for (const file of e.dataTransfer.files) {
        if (file.type.startsWith("image/")) {
          node.editor.handleImageFile(file);
          handled = true;
        }
      }
      return handled;
    };

    for (const name of (hiddenWidgets || [])) {
      const w = node.widgets.find(w => w.name === name);
      if (w) w.hidden = true;
    }

    const element = document.createElement("div");
    node.uuid = makeUUID();
    element.id = `${className}-${node.uuid}`;
    node.previewMediaType = 'image';

    node[editorKey] = node.addDOMWidget(nodeData.name, `${editorClass.name}Widget`, element, {
      serialize: false, hideOnZoom: false,
      getMinHeight: () => node[heightKey] || 550,
      getMaxHeight: () => node[heightKey] || 550,
      getHeight: () => node[heightKey] || 550,
    });
    node[heightKey] = 550;

    node.contextMenu = createContextMenuElement(menuClassName);
    node._menuDef = menuItems;
    const menuEls = Object.entries(menuItems).map(([id, def]) => createMenuItem(id, def.label || id));
    setupMenuItems(node.contextMenu, menuEls);
    document.body.appendChild(node.contextMenu);

    node.addWidget("button", "Reset canvas", null, () => {
      try { node.editor = new editorClass(node, true); }
      catch (error) { console.error(`Error creating ${editorClass.name}:`, error); }
    });

    node.setSize(initialSize);
    node[editorKey].parentEl = document.createElement("div");
    node[editorKey].parentEl.className = className;
    node[editorKey].parentEl.id = `${className}-${node.uuid}`;
    element.appendChild(node[editorKey].parentEl);

    // Auto-create for new nodes; onConfigure cancels this for saved workflows
    node._autoCreatePending = setTimeout(() => {
      if (!node.editor) {
        try {
          node.editor = new editorClass(node);
          for (const [name, value, type] of (extraProperties || [])) {
            node.addProperty(name, value, type);
          }
        } catch (error) { console.error(`Error creating ${editorClass.name}:`, error); }
      }
    }, 0);

    BaseEditorCanvas.setupImgDataPersistence(node);

    chainCallback(node, "onResize", function () {
      const editor = this.editor;
      if (!editor) return;
      const newWidth = Math.max(64, Math.round(this.size[0] - 45));
      if (newWidth === editor.width) return;

      // Only change display size — coord space stays the same
      editor.width = newWidth;
      editor.height = Math.round(newWidth * (editor.coordHeight / editor.coordWidth));

      editor.resizeCanvas();
      editor.onSizeChanged();
      editor.render();
    });

    chainCallback(node, "onConfigure", function () {
      clearTimeout(this._autoCreatePending);
      try { this.editor = new editorClass(this); }
      catch (error) { console.error(`Error configuring ${editorClass.name}:`, error); }
      // Allow watchImageInputs to operate only after configure + a delay,
      // so refreshBackgroundImage has time to restore the saved image first.
      node._bgWatchReady = false;
      setTimeout(() => { node._bgWatchReady = true; }, 500);
    });

    chainCallback(node, "onExecuted", function (message) {
      let bg_image = message["bg_image"];
      if (Array.isArray(bg_image)) bg_image = bg_image[0];
      if (bg_image) {
        const img = new Image();
        img.src = `data:image/jpeg;base64,${bg_image}`;
        img.onload = () => {
          if (this.editor) this.editor.processImage(img);
        };
      }
    });

    // Load background image from connected source node (LoadImage, VHS, etc.)
    // _bgWatchReady is false during reload until the editor has restored its saved background.
    // _bgFromConnectedSource tracks whether the current bg came from a connection (vs execution/paste).
    node._bgFromConnectedSource = false;
    node._bgWatchReady = true;  // true for new nodes, set false during onConfigure
    watchImageInputs(node, "bg_image", (sources) => {
      if (!node.editor || !node._bgWatchReady) return;
      const source = sources[0];
      if (source && !source.isVideo) {
        node._bgFromConnectedSource = true;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onerror = (e) => console.error("[Editor] source image load error:", e);
        img.onload = () => { if (node.editor) node.editor.processImage(img); };
        img.src = source.url;
      } else if (!source && node._bgFromConnectedSource) {
        node._bgFromConnectedSource = false;
        node.editor.clearBackgroundImage();
      }
    });

    chainCallback(node, "onRemoved", function () {
      if (this.editor) this.editor.destroy();
      if (this.contextMenu?.parentNode) this.contextMenu.parentNode.removeChild(this.contextMenu);
    });
  }

  // ─── Shared constructor flow ───
  // Call from subclass constructor after setting up widgets and data.
  // editorKey: e.g. 'pointsEditor' or 'splineEditor'
  // heightKey: e.g. 'pointsEditorHeight' or 'splineEditorHeight'
  // heightOffset: pixels added to canvas height for full node height (e.g. 310 or 460)
  initEditor(editorKey, heightKey, heightOffset) {
    this._editorKey = editorKey;
    this._heightKey = heightKey;
    this._heightOffset = heightOffset;

    this.createCanvas(this.node[editorKey].element);

    if (this.width > 256) this.setNodeWidth(this.width + 45);
    this.node[heightKey] = this.height + 40;
    this.node.setSize([this.node.size[0], this.height + heightOffset]);

    this.setupEventListeners();
    this.render();
  }

  // Shared onSizeChanged — uses stored heightKey/heightOffset
  onSizeChanged() {
    this.node[this._heightKey] = this.height + 40;
    this.node.setSize([this.node.size[0], this.height + this._heightOffset]);
    if (this.node.graph) this.node.graph.setDirtyCanvas(true, true);
  }

  // Shared constructor preamble — cleanup, reset, context menu, coord/display init
  initEditorPreamble(editorKey, className) {
    this._className = className;
    this.cleanupPreviousEditor(this.node);
    if (this.reset && this.node[editorKey].element) {
      this.node[editorKey].element.innerHTML = '';
    }
    this.createContextMenu();
  }

  // Shared coord/display size init from widgets + saved node size
  initDisplaySize() {
    this.coordWidth = this.widthWidget.value;
    this.coordHeight = this.heightWidget.value;
    const savedWidth = Math.max(64, Math.round(this.node.size[0] - 45));
    this.width = Math.min(savedWidth, maxDisplayDim);
    this.height = Math.round(this.width * (this.coordHeight / this.coordWidth));
  }

  // Shared context menu creation — clone to clear stale listeners, wire up action handlers
  createContextMenu() {
    const oldMenu = this.node.contextMenu;
    const newMenu = oldMenu.cloneNode(true);
    oldMenu.parentNode.replaceChild(newMenu, oldMenu);
    this.node.contextMenu = newMenu;
    this.setupContextMenuListeners(this._className);

    const self = this;
    newMenu.addEventListener('click', (e) => {
      e.preventDefault();
      if (e.target.tagName !== 'A') return;
      const id = e.target.dataset.menuId;
      const def = this.node._menuDef[id];
      if (def?.action) {
        def.action(self);
        self._updateMenuToggleStates();
      }
      newMenu.style.display = 'none';
    });
  }

  // Update toggle item styling from menu definitions
  _updateMenuToggleStates() {
    const menuDef = this.node._menuDef;
    this.node.contextMenu.querySelectorAll('a').forEach(item => {
      const def = menuDef[item.dataset.menuId];
      if (def?.toggle) {
        const on = def.toggle(this);
        item.style.color = on ? '#4fc3f7' : '#FFF';
        item.style.borderLeft = on ? '3px solid #4fc3f7' : '3px solid transparent';
        item.style.paddingLeft = '8px';
      }
    });
  }

  // ─── Hooks for subclasses ───
  // Override these instead of duplicating logic:

  // Called after data changes (render + update widgets)
  onDataChanged() {}

  // Called on image load for editor-specific state (e.g., drawRuler = false)
  onImageResize() {}

  // Called when coord space changes — override to rescale coordinates
  onCoordSpaceResized(_oldWidth, _oldHeight) {}
}
