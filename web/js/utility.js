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

