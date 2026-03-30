const { app } = window.comfyAPI.app;

// ─── General-purpose helpers shared across KJNodes JS ───

export function makeUUID() {
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}

export function chainCallback(object, property, callback) {
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

// ─── Graph coordinate helpers ───

export function clientToCanvas(lgCanvas, clientX, clientY) {
  const rect = lgCanvas.canvas.getBoundingClientRect();
  return [
    (clientX - rect.left) / lgCanvas.ds.scale - lgCanvas.ds.offset[0],
    (clientY - rect.top) / lgCanvas.ds.scale - lgCanvas.ds.offset[1],
  ];
}

export function getNodeAtPoint(graph, cx, cy) {
  for (let i = graph._nodes.length - 1; i >= 0; i--) {
    if (graph._nodes[i].isPointInside(cx, cy)) return graph._nodes[i];
  }
  return null;
}

export function typesCompatible(a, b) {
  if (a === "*" || b === "*") return true;
  if (a === b) return true;
  if (typeof a === "string" && typeof b === "string" && a.toUpperCase() === b.toUpperCase()) return true;
  return false;
}

// ─── Slot position helper ───
// Uses modern API (DOM-registered positions in Vue mode) with legacy fallback
export function getSlotPos(node, isInput, slotIdx) {
  if (isInput && node.getInputPos) return node.getInputPos(slotIdx);
  if (!isInput && node.getOutputPos) return node.getOutputPos(slotIdx);
  const out = [0, 0];
  node.getConnectionPos(isInput, slotIdx, out);
  return out;
}

// ─── Middle-click pan passthrough for DOM widgets ───
// Allows panning the LiteGraph canvas via middle-click drag on any DOM element
export function addMiddleClickPan(element) {
  const onMouseDown = (e) => {
    if (e.button !== 1) return;
    e.preventDefault();
    const ds = app.canvas?.ds;
    if (!ds) return;
    const startX = e.clientX, startY = e.clientY;
    const startOffsetX = ds.offset[0], startOffsetY = ds.offset[1];
    const onMove = (me) => {
      ds.offset[0] = startOffsetX + (me.clientX - startX);
      ds.offset[1] = startOffsetY + (me.clientY - startY);
      app.canvas.setDirty(true, true);
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };
  element.addEventListener('mousedown', onMouseDown);
  return () => element.removeEventListener('mousedown', onMouseDown);
}

// ─── Resolve preview image/video from a connected source node ───
// Walks the graph link to find what's connected to node.inputs[inputSlot]
// and returns { url, isVideo, videoEl } or null
export function resolveSourcePreview(node, inputSlot) {
  if (!node.graph) return null;
  const input = node.inputs?.[inputSlot];
  if (!input || input.link == null) return null;
  const link = node.graph.links?.[input.link] ?? node.graph.links?.get?.(input.link);
  if (!link) return null;
  const srcNode = node.graph.getNodeById(link.origin_id);
  if (!srcNode) return null;

  // Check for VHS-style video preview widget
  const vpWidget = srcNode.widgets?.find((w) => w.name === "videopreview");
  if (vpWidget?.videoEl?.src) {
    return { isVideo: true, videoEl: vpWidget.videoEl };
  }

  // Look for a LoadImage "image" widget or LoadVideo "video" widget
  const w = srcNode.widgets?.find((w) => w.name === "image" || w.name === "video");
  if (w?.value) {
    let subfolder = "", fname = w.value;
    const lastSlash = fname.lastIndexOf("/");
    if (lastSlash >= 0) { subfolder = fname.substring(0, lastSlash); fname = fname.substring(lastSlash + 1); }
    const isVideo = w.name === "video";
    const url = `/view?filename=${encodeURIComponent(fname)}&type=input&subfolder=${encodeURIComponent(subfolder)}`;
    return { url, isVideo };
  }

  // Fallback: check node.imgs (set after execution)
  if (srcNode.imgs?.length > 0 && srcNode.imgs[0].src) {
    return { url: srcNode.imgs[0].src, isVideo: false };
  }
  return null;
}

// ─── Source input watcher ───
// Watches a named IMAGE input for connection changes and source widget changes.
// Calls onChange(sources) when connections change or the source node's image/video widget changes.
// sources: array of { url, isVideo, videoEl } per connected IMAGE input matching inputName (or all if inputName is null).
// Returns a cleanup function.
export function watchImageInputs(node, inputName, onChange) {
  let watchedWidgets = [];

  function unwatchWidgets() {
    for (const { widget, origCb } of watchedWidgets) widget.callback = origCb;
    watchedWidgets = [];
  }

  function resolve() {
    if (!node.inputs) return [];
    const slots = inputName
      ? node.inputs.map((inp, i) => inp.name === inputName ? i : -1).filter(i => i >= 0)
      : node.inputs.map((inp, i) => inp.type === "IMAGE" ? i : -1).filter(i => i >= 0);
    return slots.map(i => resolveSourcePreview(node, i)).filter(s => s !== null);
  }

  function watch() {
    unwatchWidgets();
    if (!node.inputs || !node.graph) return;
    const slots = inputName
      ? node.inputs.map((inp, i) => inp.name === inputName ? i : -1).filter(i => i >= 0)
      : node.inputs.map((inp, i) => inp.type === "IMAGE" ? i : -1).filter(i => i >= 0);
    for (const slotIdx of slots) {
      const input = node.inputs[slotIdx];
      if (!input || input.link == null) continue;
      const link = node.graph.links?.[input.link] ?? node.graph.links?.get?.(input.link);
      if (!link) continue;
      const srcNode = node.graph.getNodeById(link.origin_id);
      if (!srcNode) continue;
      const w = srcNode.widgets?.find(w => w.name === "image" || w.name === "video");
      if (!w) continue;
      const origCb = w.callback;
      w.callback = function (...args) {
        if (origCb) origCb.apply(this, args);
        setTimeout(() => onChange(resolve()), 100);
      };
      watchedWidgets.push({ widget: w, origCb });
    }
  }

  chainCallback(node, "onConnectionsChange", function () {
    setTimeout(() => {
      watch();
      onChange(resolve());
    }, 100);
  });

  chainCallback(node, "onRemoved", unwatchWidgets);

  return unwatchWidgets;
}

// ─── Bounding box hit test ───
// Tests whether (mx, my) hits a corner handle or the interior of a rect defined by (x1, y1)–(x2, y2).
// Returns "resize-tl", "resize-tr", "resize-bl", "resize-br", "move", or null.
export function rectHitTest(mx, my, x1, y1, x2, y2, radius) {
  const hit = (cx, cy) => Math.abs(mx - cx) < radius && Math.abs(my - cy) < radius;
  if (hit(x1, y1)) return "resize-tl";
  if (hit(x2, y1)) return "resize-tr";
  if (hit(x1, y2)) return "resize-bl";
  if (hit(x2, y2)) return "resize-br";
  if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) return "move";
  return null;
}

// Returns the appropriate CSS cursor for a bbox hit mode string.
export function cursorForBboxMode(mode) {
  if (mode === "move") return "move";
  if (mode === "resize-tl" || mode === "resize-br") return "nwse-resize";
  if (mode === "resize-tr" || mode === "resize-bl") return "nesw-resize";
  return null;
}

// ─── Wheel zoom passthrough for DOM widgets ───
// Re-dispatches wheel events to the LiteGraph canvas for zoom
export function addWheelPassthrough(element) {
  element.addEventListener("wheel", (e) => {
    const gc = document.getElementById("graph-canvas");
    if (gc) {
      gc.dispatchEvent(new WheelEvent(e.type, e));
      e.preventDefault();
    }
  }, { passive: false });
}

