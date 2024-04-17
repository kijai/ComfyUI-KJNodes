import { app } from '../../../scripts/app.js'

//from melmass
export function makeUUID() {
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}

export const loadScript = (
    FILE_URL,
    async = true,
    type = 'text/javascript',
  ) => {
    return new Promise((resolve, reject) => {
      try {
        // Check if the script already exists
        const existingScript = document.querySelector(`script[src="${FILE_URL}"]`)
        if (existingScript) {
          resolve({ status: true, message: 'Script already loaded' })
          return
        }
  
        const scriptEle = document.createElement('script')
        scriptEle.type = type
        scriptEle.async = async
        scriptEle.src = FILE_URL
  
        scriptEle.addEventListener('load', (ev) => {
          resolve({ status: true })
        })
  
        scriptEle.addEventListener('error', (ev) => {
          reject({
            status: false,
            message: `Failed to load the script ${FILE_URL}`,
          })
        })
  
        document.body.appendChild(scriptEle)
      } catch (error) {
        reject(error)
      }
    })
  }
  const create_documentation_stylesheet = () => {
    const tag = 'kj-splineditor-stylesheet'

    let styleTag = document.head.querySelector(tag)

    if (!styleTag) {
      styleTag = document.createElement('style')
      styleTag.type = 'text/css'
      styleTag.id = tag
      styleTag.innerHTML = `
       .spline-editor {
        background: var(--comfy-menu-bg);
        position: absolute;
        color: var(--fg-color);
        font: 12px monospace;
        line-height: 1.5em;
        padding: 10px;
        z-index: 3;
        overflow: hidden;
        border-radius: 10px;
        border-style: solid;
        border-width: medium;
        border-color: var(--border-color);
       }
        `
      document.head.appendChild(styleTag)
    }
  }

loadScript('/kjweb_async/svg-path-properties.min.js').catch((e) => {
    console.log(e)
})
loadScript('/kjweb_async/protovis.min.js').catch((e) => {
  console.log(e)
})
create_documentation_stylesheet()

function chainCallback(object, property, callback) {
  if (object == undefined) {
      //This should not happen.
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
    name: 'KJNodes.curves', 
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name == 'SplineEditor') {
          chainCallback(nodeType.prototype, "onNodeCreated", function () {
            hideWidgetForGood(this, this.widgets.find(w => w.name === "coordinates"))
            var element = document.createElement("div");
            this.uuid = makeUUID()
            element.id = `spline-editor-${this.uuid}`


            var splineEditor = this.addDOMWidget(nodeData.name, "SplineEditorWidget", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
              return element.value;
            },
            setValue(v) {
                element.value = v;
            },
              });

            this.setSize([620, 760])
            splineEditor.parentEl = document.createElement("div");
            splineEditor.parentEl.className = "spline-editor";
            splineEditor.parentEl.id = `spline-editor-${this.uuid}`
            splineEditor.parentEl.style['width'] = "90%"
            splineEditor.parentEl.style['height'] = "544px"
            element.appendChild(splineEditor.parentEl);
         
            this.coordWidget = this.widgets.find(w => w.name === "coordinates");
            this.interpolationWidget = this.widgets.find(w => w.name === "interpolation");
            this.pointsWidget = this.widgets.find(w => w.name === "points_to_sample");
            //addElement(nodeData, nodeType, this)
            var w = 512
            var h = 512
            var i = 3
            var segmented = false
            if (points == null) {
                      var points = pv.range(1, 5).map(i => ({
                        x: i * w / 5,
                        y: 50 + Math.random() * (h - 100)
                      }));
                    }
            let vis = new pv.Panel()
              .width(w)
              .height(h)
              .fillStyle("var(--comfy-menu-bg)")
              .strokeStyle("orange")
              .lineWidth(0)
              .antialias(false)
              .margin(10)
              .event("mousedown", function() {
                if (pv.event.shiftKey) { // Use pv.event to access the event object
                    i = points.push(this.mouse()) - 1;
                    return this;
                }
                })
              .event("mouseup", function() {
                if (this.pathElements !== null) {
                  console.log(this.pathElements) 
                    let coords = samplePoints(this.pathElements[0], this.pointsWidget.value);
                    let coordsString = JSON.stringify(coords);
                    if (this.coordWidget) {
                      this.coordWidget.value = coordsString;
                      console.log(this.coordWidget.value)
                    }
                }
              });

              vis.add(pv.Rule)
              .data(pv.range(0, 8, .5))
              .bottom(d =>  d * 70 + 9.5)
              .strokeStyle("gray")
              .lineWidth(1)

            vis.add(pv.Line)
              .data(() => points)
              .left(d => d.x)
              .top(d => d.y)
              .interpolate(() => this.interpolationWidget.value)
              .segmented(() => segmented)
              .strokeStyle(pv.Colors.category10().by(pv.index))
              .tension(0.5)
              .lineWidth(3)

            vis.add(pv.Dot)
              .data(() => points)
              .left(d => d.x)
              .top(d => d.y)
              .radius(7)
              .cursor("move")
              .strokeStyle(function() { return i == this.index ? "#ff7f0e" : "#1f77b4"; })
              .fillStyle(function() { return "rgba(100, 100, 100, 0.2)"; })
              .event("mousedown", pv.Behavior.drag())
              .event("dragstart", function() {
                  i = this.index;
                  if (pv.event.button === 2) {
                    points.splice(i--, 1);
                    vis.render();
                  }
                  return this;
              })
              .event("drag", vis)
              .anchor("top").add(pv.Label)
                  .font(d => Math.sqrt(d[2]) * 32 + "px sans-serif")
                  //.text(d => `(${Math.round(d.x)}, ${Math.round(d.y)})`)
                  .text(d => {
                    // Normalize y to range 0.0 to 1.0, considering the inverted y-axis
                    var normalizedY = 1.0 - (d.y / h);
                    return `${normalizedY.toFixed(2)}`;
                })
                  .textStyle("orange")
                vis.render();
                var svgElement = vis.canvas();
                svgElement.style['zIndex'] = "7"
                svgElement.style['position'] = "fixed"
                vis.id = `spline-editor-vis-${this.uuid}`
                splineEditor.element.appendChild(svgElement);
                svgElement.id = `spline-editor-svg-${this.uuid}`

                this.pathElements = svgElement.getElementsByTagName('path'); // Get all path elements
            });
              
        }
        //disable context menu on right click
        document.addEventListener('contextmenu', function(e) {
            if (e.button === 2) { // Right mouse button
                e.preventDefault();
                e.stopPropagation();
              }
          })
      }
})

function samplePoints(svgPathElement, numSamples) {
    var pathLength = svgPathElement.getTotalLength();
    var points = [];

    for (var i = 0; i < numSamples; i++) {
        // Calculate the distance along the path for the current sample
        var distance = (pathLength / (numSamples - 1)) * i;

        // Get the point at the current distance
        var point = svgPathElement.getPointAtLength(distance);

        // Add the point to the array of points
        points.push({ x: point.x, y: point.y });
    }
    //console.log(points);
    return points;
}

//from melmass
export function hideWidgetForGood(node, widget, suffix = '') {
  widget.origType = widget.type
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
  widget.type = "converted-widget" + suffix
  // widget.serializeValue = () => {
  //     // Prevent serializing the widget if we have no input linked
  //     const w = node.inputs?.find((i) => i.widget?.name === widget.name);
  //     if (w?.link == null) {
  //         return undefined;
  //     }
  //     return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
  // };

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidgetForGood(node, w, ':' + widget.name)
    }
  }
}