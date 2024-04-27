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

        position: absolute;

        font: 12px monospace;
        line-height: 1.5em;
        padding: 10px;
        z-index: 0;
        overflow: hidden;
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
    name: 'KJNodes.SplineEditor', 
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === 'SplineEditor') {
          chainCallback(nodeType.prototype, "onNodeCreated", function () {
            hideWidgetForGood(this, this.widgets.find(w => w.name === "coordinates"))

            var element = document.createElement("div");
            this.uuid = makeUUID()
            element.id = `spline-editor-${this.uuid}`

            this.splineEditor = this.addDOMWidget(nodeData.name, "SplineEditorWidget", element, {
            serialize: false,
            hideOnZoom: false,
            });

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
            this.menuItem1 = document.createElement("a");
            this.menuItem1.href = "#";
            this.menuItem1.id = "menu-item-1";
            this.menuItem1.textContent = "Toggle handles";
            styleMenuItem(this.menuItem1);

            this.menuItem2 = document.createElement("a");
            this.menuItem2.href = "#";
            this.menuItem2.id = "menu-item-2";
            this.menuItem2.textContent = "Display sample points";
            styleMenuItem(this.menuItem2);
            // Add hover effect to menu items
            this.menuItem1.addEventListener('mouseover', function() {
              this.style.backgroundColor = "gray";
            });
            this.menuItem1.addEventListener('mouseout', function() {
              this.style.backgroundColor = "#202020";
            });

            this.menuItem2.addEventListener('mouseover', function() {
              this.style.backgroundColor = "gray";
            });
            this.menuItem2.addEventListener('mouseout', function() {
              this.style.backgroundColor = "#202020";
            });

            this.contextMenu.appendChild(this.menuItem1);
            this.contextMenu.appendChild(this.menuItem2);

            document.body.appendChild( this.contextMenu);
            this.addWidget("button", "New spline", null, () => {
              
              if (!this.properties || !("points" in this.properties)) {
                createSplineEditor(this)
                this.addProperty("points", this.constructor.type, "string");
              }
              else {
                createSplineEditor(this, true)
              }
            });
            this.setSize([550, 870])
            this.splineEditor.parentEl = document.createElement("div");
            this.splineEditor.parentEl.className = "spline-editor";
            this.splineEditor.parentEl.id = `spline-editor-${this.uuid}`
            element.appendChild(this.splineEditor.parentEl);

            chainCallback(this, "onGraphConfigured", function() {
              console.log('onGraphConfigured');
              createSplineEditor(this)
              this.setSize([550, 870])
              });

            //disable context menu on right click
            // document.addEventListener('contextmenu', function(e) {
            //   if (e.button === 2) { // Right mouse button
            //       e.preventDefault();
            //       e.stopPropagation();
            //     }
            // })

          }); // onAfterGraphConfigured
        }//node created
      } //before register
})//register


function createSplineEditor(context, reset=false) {
  console.log("creatingSplineEditor")

  document.addEventListener('contextmenu', function(e) {
    e.preventDefault();
  });

  document.addEventListener('click', function(e) {
    if (!context.contextMenu.contains(e.target)) {
      context.contextMenu.style.display = 'none';
    }
  });

  context.menuItem1.addEventListener('click', function(e) {
    e.preventDefault();
    if (!drawHandles) {
      drawHandles = true
      vis.add(pv.Line)
      .data(() => points.map((point, index) => ({
          start: point,
          end: [index]
      })))
      .left(d => d.start.x)
      .top(d => d.start.y)
      .interpolate("linear")
      .tension(0) // Straight lines
      .strokeStyle("#ff7f0e") // Same color as control points
      .lineWidth(1)
      .visible(() => drawHandles);
      vis.render();
      
   
    } else {
      drawHandles = false
      vis.render();
    }
    context.contextMenu.style.display = 'none';
  
  });

  context.menuItem2.addEventListener('click', function(e) {
      e.preventDefault();
      drawSamplePoints = !drawSamplePoints;
      
      updatePath();
  });

  var drawSamplePoints = false;

  function updatePath() {
      points_to_sample = pointsWidget.value
      let coords = samplePoints(pathElements[0], points_to_sample);
      if (drawSamplePoints) {
      if (pointsLayer) {
        // Update the data of the existing points layer
        pointsLayer.data(coords);
      } else {
          // Create the points layer if it doesn't exist
          pointsLayer = vis.add(pv.Dot)
              .data(coords)
              .left(function(d) { return d.x; })
              .top(function(d) { return d.y; })
              .radius(5) // Adjust the radius as needed
              .fillStyle("red") // Change the color as needed
              .strokeStyle("black") // Change the stroke color as needed
              .lineWidth(1); // Adjust the line width as needed
          }
      }
      else {
          if (pointsLayer) {
              // Remove the points layer
              pointsLayer.data([]);
              vis.render();
          }
      }
      let coordsString = JSON.stringify(coords);
      pointsStoreWidget.value = JSON.stringify(points);
      if (coordWidget) {
        coordWidget.value = coordsString;
      }
      vis.render();
  }
  
  if (reset && context.splineEditor.element) {
    context.splineEditor.element.innerHTML = ''; // Clear the container
  }
  const coordWidget = context.widgets.find(w => w.name === "coordinates");
  const interpolationWidget = context.widgets.find(w => w.name === "interpolation");
  const pointsWidget = context.widgets.find(w => w.name === "points_to_sample");
  const pointsStoreWidget = context.widgets.find(w => w.name === "points_store");
  const tensionWidget = context.widgets.find(w => w.name === "tension");
  const minValueWidget = context.widgets.find(w => w.name === "min_value");
  const maxValueWidget = context.widgets.find(w => w.name === "max_value");
  //const segmentedWidget = context.widgets.find(w => w.name === "segmented");

  var interpolation = interpolationWidget.value
  var tension = tensionWidget.value
  var points_to_sample = pointsWidget.value
  var rangeMin = minValueWidget.value
  var rangeMax = maxValueWidget.value
  var pointsLayer = null;
  
  interpolationWidget.callback = () => {
    interpolation = interpolationWidget.value
    updatePath();
  }

  tensionWidget.callback = () => {
    tension = tensionWidget.value
    updatePath();
  }

  minValueWidget.callback = () => {
    updatePath();
  }
  maxValueWidget.callback = () => {
    updatePath();
  }
  
 // Initialize or reset points array
 var drawHandles = false
 var w = 512
 var h = 512
 var i = 3
 let points = [];
 
 if (!reset && pointsStoreWidget.value != "") {
    points = JSON.parse(pointsStoreWidget.value);
 } else {
  points = pv.range(1, 4).map((i, index) => {
    if (index === 0) {
      // First point at the bottom-left corner
      return { x: 0, y: h };
    } else if (index === 2) {
      // Last point at the top-right corner
      return { x: w, y: 0 };
    } else {
      // Other points remain as they were
      return {
        x: i * w / 5,
        y: 50 + Math.random() * (h - 100)
      };
    }
  });
    pointsStoreWidget.value = JSON.stringify(points);
 }
  
  var vis = new pv.Panel()
  .width(w)
  .height(h)
  .fillStyle("var(--comfy-menu-bg)")
  .strokeStyle("gray")
  .lineWidth(2)
  .antialias(false)
  .margin(10)
  .event("mousedown", function() {
    if (pv.event.shiftKey) { // Use pv.event to access the event object
        i = points.push(this.mouse()) - 1;
        return this;
    }
    else if (pv.event.button === 2) {
      context.contextMenu.style.display = 'block';
      context.contextMenu.style.left = `${pv.event.clientX}px`;
      context.contextMenu.style.top = `${pv.event.clientY}px`;
    }
    })
  .event("mouseup", function() {
    if (this.pathElements !== null) {
      updatePath();
    }
  });

  vis.add(pv.Rule)
    .data(pv.range(0, 8, .5))
    .bottom(d =>  d * 64 + 0)
    .strokeStyle("gray")
    .lineWidth(2)

  vis.add(pv.Line)
    .data(() => points)
    .left(d => d.x)
    .top(d => d.y)
    .interpolate(() => interpolation)
    .tension(() => tension)
    .segmented(() => false)
    .strokeStyle(pv.Colors.category10().by(pv.index))
    .lineWidth(3)

    var hoverIndex = -1;
    var isDragging
  
  vis.add(pv.Dot)
    .data(() => points)
    .left(d => d.x)
    .top(d => d.y)
    .radius(10)
    .cursor("move")
    .strokeStyle(function() { return i == this.index ? "#ff7f0e" : "#1f77b4"; })
    .fillStyle(function() { return "rgba(100, 100, 100, 0.2)"; })
    .event("mousedown", pv.Behavior.drag())
    .event("dragstart", function() {
        i = this.index;
        hoverIndex = this.index;
        isDragging = true;
        if (pv.event.button === 2 && i !== 0 && i !== points.length - 1) {
          points.splice(i--, 1);
          vis.render();
        }
        return this;
    })
    .event("dragend", function() {
        isDragging = false;
    })
    .event("drag", vis)
    .event("mouseover", function() {
      hoverIndex = this.index; // Set the hover index to the index of the hovered dot
      vis.render(); // Re-render the visualization
    })
    .event("mouseout", function() {
      !isDragging && (hoverIndex = -1); // Reset the hover index when the mouse leaves the dot
      vis.render(); // Re-render the visualization
    })
    .anchor("center")
    .add(pv.Label)
    .visible(function() {
      return hoverIndex === this.index; // Only show the label for the hovered dot
    })
    .left(d => d.x < w / 2 ? d.x + 80 : d.x - 70) // Shift label to right if on left half, otherwise shift to left
    .top(d => d.y < h / 2 ? d.y + 20 : d.y - 20)  // Shift label down if on top half, otherwise shift up
    
        .font(12 + "px sans-serif")
        .text(d => {
          // Normalize y to range 0.0 to 1.0, considering the inverted y-axis
          let normalizedY = (1.0 - (d.y / h) - 0.0) * (rangeMax - rangeMin) + rangeMin;
          let normalizedX = (d.x / w);
          let frame = Math.round((d.x / w) * points_to_sample);
          return `F: ${frame}, X: ${normalizedX.toFixed(2)}, Y: ${normalizedY.toFixed(2)}`;
      })
    .textStyle("orange")
    
  
    vis.render();
    var svgElement = vis.canvas();
    svgElement.style['zIndex'] = "2"
    svgElement.style['position'] = "relative"
    context.splineEditor.element.appendChild(svgElement);
    var pathElements = svgElement.getElementsByTagName('path'); // Get all path elements  
 
}
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