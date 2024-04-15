import { app } from '../../../scripts/app.js'


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
        z-index: 5;
        overflow: hidden;
        border-radius: 10px;
        border-style: solid;
        border-width: medium;
        border-color: var(--border-color);
        height: 544px;
        width: 544px;
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

app.registerExtension({
    name: 'KJNodes.curves', 
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name == 'SplineEditor') {
          addElement(nodeData, nodeType);
        }
    },
})

export const addElement = (nodeData,nodeType) => {
    console.log("Creating spline editor")
    const iconSize = 24
    const iconMargin = 4
    
    let splineEditor = null
    let vis = null
    
    const drawFg = nodeType.prototype.onDrawForeground
    nodeType.prototype.onNodeCreated = function () {
      console.log("Node created")
        this.coordWidget = this.widgets.find(w => w.name === "coordinates");
        this.interpolationWidget = this.widgets.find(w => w.name === "interpolation");
        this.pointsWidget = this.widgets.find(w => w.name === "points_to_sample");
    }
    nodeType.prototype.onRemoved = function () {
      console.log("Node removed")
      if (splineEditor !== null) {
        splineEditor.parentNode.removeChild(splineEditor)
        splineEditor = null
      }
    }
    nodeType.prototype.onDrawForeground = function (ctx) {
      console.log("Drawing foreground")
      const r = drawFg ? drawFg.apply(this, arguments) : undefined
      if (this.flags.collapsed) return r
      
      const x = this.size[0] - iconSize - iconMargin

      if (this.show_doc && splineEditor === null) {
        console.log("Drawing spline editor")
        splineEditor = document.createElement('div');
        splineEditor.classList.add('spline-editor');

        // close button
        const closeButton = document.createElement('div');
        closeButton.textContent = '❌';
        closeButton.style.position = 'absolute';
        closeButton.style.top = '0';
        closeButton.style.right = '0';
        closeButton.style.cursor = 'pointer';
        closeButton.style.padding = '5px';
        closeButton.style.color = 'red';
        closeButton.style.fontSize = '12px';
        closeButton.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            this.show_doc = !this.show_doc
            splineEditor.parentNode.removeChild(splineEditor)
            splineEditor = null
          });
            
        splineEditor.appendChild(closeButton)
        
        var w = 512
        var h = 512
        var i = 3
        
        if (points == null) {
          var points = pv.range(1, 5).map(i => ({
            x: i * w / 5,
            y: 50 + Math.random() * (h - 100)
          }));
        }

        var segmented = false
        vis = new pv.Panel()
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

        //disable context menu on right click     
        document.addEventListener('contextmenu', function(e) {
          if (e.button === 2) { // Right mouse button
              e.preventDefault();
              e.stopPropagation();
            }
        })
        //right click remove dot
        pv.listen(window, "mousedown", () => {
            window.focus();
            if (pv.event.button === 2) {
              points.splice(i--, 1);
              vis.render();
              }
            });
        //send coordinates to node on mouseup
        pv.listen(window, "mouseup", () => {           
            if (pathElements !== null) {
                let coords = samplePoints(pathElements[0], this.pointsWidget.value);
                let coordsString = JSON.stringify(coords);
                if (this.coordWidget) {
                  this.coordWidget.value = coordsString;
                }
            }
        });    
          
        vis.render();
        var svgElement = vis.canvas();
        splineEditor.appendChild(svgElement); 
      //   this.addDOMWidget("videopreview", "preview", splineEditor, {
      //     serialize: false,
      //     hideOnZoom: false,
         
      // });
        document.body.appendChild(splineEditor)
        var pathElements = svgElement.getElementsByTagName('path'); // Get all path elements          
      }
       // close the popup
       else if (!this.show_doc && splineEditor !== null) {
        splineEditor.parentNode.removeChild(splineEditor)
        splineEditor = null
      }
      
      if (this.show_doc && splineEditor !== null && vis !== null) {
          const rect = ctx.canvas.getBoundingClientRect()
          const scaleX = rect.width / ctx.canvas.width
          const scaleY = rect.height / ctx.canvas.height
          
          const transform = new DOMMatrix()
          .scaleSelf(scaleX, scaleY)
          .translateSelf(this.size[0] * scaleX, 0)
          .multiplySelf(ctx.getTransform())
          .translateSelf(10, -32)
          
          const scale = new DOMMatrix()
          .scaleSelf(transform.a, transform.d);

          const styleObject = {
              transformOrigin: '0 0',
              transform: scale,
              left: `${transform.a + transform.e}px`,
              top: `${transform.d + transform.f}px`,
              };
          Object.assign(splineEditor.style, styleObject);
      }
      ctx.save()
      ctx.translate(x - 2, iconSize - 45)
      ctx.scale(iconSize / 32, iconSize / 32)
      ctx.strokeStyle = 'rgba(255,255,255,0.3)'
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.lineWidth = 2.4
      ctx.font = 'bold 36px monospace'
      ctx.fillStyle = 'orange';
      ctx.fillText('📈', 0, 24)
      ctx.restore()
      return r
    }
    // handle clicking of the icon
    const mouseDown = nodeType.prototype.onMouseDown
    nodeType.prototype.onMouseDown = function (e, localPos, canvas) {
      const r = mouseDown ? mouseDown.apply(this, arguments) : undefined
      const iconX = this.size[0] - iconSize - iconMargin
      const iconY = iconSize - 45
      if (
        localPos[0] > iconX &&
        localPos[0] < iconX + iconSize &&
        localPos[1] > iconY &&
        localPos[1] < iconY + iconSize
      ) {
        if (this.show_doc === undefined) {
          this.show_doc = true
        } else {
          this.show_doc = !this.show_doc
        }
        return true;
      }
      return r;
    }
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