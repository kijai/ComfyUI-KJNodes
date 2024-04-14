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
        height: 512px;
        width: 512px;
        max-height: 100%;
        /* Scrollbar styling for Chrome */
        &::-webkit-scrollbar {
           width: 6px;
        }
        &::-webkit-scrollbar-track {
           background: var(--bg-color);
        }
        &::-webkit-scrollbar-thumb {
           background-color: var(--fg-color);
           border-radius: 6px;
           border: 3px solid var(--bg-color);
        }
       
        /* Scrollbar styling for Firefox */
        scrollbar-width: thin;
        scrollbar-color: var(--fg-color) var(--bg-color);
        a {
          color: yellow;
        }
        a:visited {
          color: orange;
        }
        a:hover {
          color: red;
        }
       }
        `
      document.head.appendChild(styleTag)
    }
  }

loadScript('/kjweb_async/svg-path-properties.min.js').catch((e) => {
    console.log(e)
})

class SplineEditorWidget {
    constructor(inputName, defaultValue) {
        //this.name = inputName || "Spline";
        //this._value = defaultValue || [{ x: 0, y: 0 }];
        this.type = "SPLINE";
        //this.selectedPointIndex = null;
        this.resize        
        }
    
    computeSize(width) {
        return [width, 300];
    }

    configure(data) {
        console.log(data)
    }

    value() {
        console.debug('Returning value', this._value)
        return this._value
    }
    setValue(value) {
        console.debug('Setting value', value)
        this._value = value
    }
}

app.registerExtension({
    name: 'KJNodes.curves', 
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name == 'SplineEditor') {
            addElement(nodeData, nodeType);
        }
    },
    // getCustomWidgets: function () {
    //     return {
    //         SPLINE: (node, inputName, inputData, app) => {
    //         console.log(inputName)    
    //         console.log(inputData)
    //         console.log("Registering KJNodes curve widget");
    //             return {
    //             widget: node.addCustomWidget(
    //                 new SplineEditorWidget(inputName, inputData[1]?.default)
    //             ),
    //             }
    //         }
    //     }
    // }
})

export const addElement = (nodeData,nodeType, opts = { icon_size: 24, icon_margin: 4 }) => {
    opts = opts || {}
    const iconSize = opts.icon_size ? opts.icon_size : 14
    const iconMargin = opts.icon_margin ? opts.icon_margin : 4
    console.log("Creating spline editor")
    let splineEditor = null
    let vis = null
    
    const drawFg = nodeType.prototype.onDrawForeground
    nodeType.prototype.onNodeCreated = function () {
        this.coordWidget = this.widgets.find(w => w.name === "coordinates"); 
    }
    nodeType.prototype.onDrawForeground = function (ctx) {
    const r = drawFg ? drawFg.apply(this, arguments) : undefined
    if (this.flags.collapsed) return r
    const x = this.size[0] - iconSize - iconMargin
    if (splineEditor === null) {
        create_documentation_stylesheet()
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
        
        document.body.appendChild(splineEditor)
        
        var w = 512
        var h = 512
        var i = 3    
        loadScript('/kjweb_async/protovis.min.js').then(() => {
            console.log('Protovis loaded successfully');
           

        var points = pv.range(1, 5).map(i => ({
            x: i * w / 5,
            y: 50 + Math.random() * (h - 100)
        }));
        var interpolate = "cardinal"
        var segmented = false
        vis = new pv.Panel()
            .width(w)
            .height(h)
            .fillStyle("var(--comfy-menu-bg)")
            //.strokeStyle("orange")
            .lineWidth(0)
            .antialias(false)
            .margin(2)
            .event("mousedown", function() {
                i = points.push(this.mouse()) - 1;
                return this;
               });
        vis.add(pv.Rule)
               .data(pv.range(0, 8, .5))
               .bottom(d =>  d * 70 + 9.5)
               .strokeStyle("white")
               .lineWidth(2)

        vis.add(pv.Line)
            .data(() => points)
            .left(d => d.x)
            .top(d => d.y)
            .interpolate(() => interpolate)
            .segmented(() => segmented)
            .strokeStyle(pv.Colors.category10().by(pv.index))
            .tension(0.5)
            .lineWidth(3);

        vis.add(pv.Dot)
            .data(() => points)
            .left(d => d.x)
            .top(d => d.y)
            .radius(7)
            .cursor("move")
            .strokeStyle(function() { return i == this.index ? "#ff7f0e" : "#1f77b4"; })
            .fillStyle(function() { return "rgba(100, 100, 100, 0.2)"; })
            //.anchor("center").add(pv.Label)
                //.font(d => Math.sqrt(d[2]) * 20 + "px sans-serif")
              //  .text(d => d[2])
            .event("mousedown", pv.Behavior.drag())
            .event("dragstart", function() {
                i = this.index;
                return this;
            })
            .event("drag", vis)

        pv.listen(window, "mousedown", () => { 
            window.focus();
            //logPathLength();
            });
        pv.listen(window, "mouseup", () => { 
            //logPathLength();
            if (svgPathElement !== null) {
                let coords = samplePoints(svgPathElement, points.length);
                let coordsString = JSON.stringify(coords);
                if (this.coordWidget) this.coordWidget.value = coordsString;
            }
        });    
        pv.listen(window, "keydown", (e) => {
            console.log("key code: " + e.keyCode)
            // code 8 is backspace, code 46 is delete
            if ((e.keyCode == 16 || e.keyCode == 46) && (i >= 0)) {
                points.splice(i--, 1);
                vis.render();
                e.preventDefault();
            }
            })
        vis.render();
        var svgElement = vis.canvas();
        var svgPathElement = document.querySelector('path');
        splineEditor.appendChild(svgElement);
               
        })
        .catch((error) => {
            console.error('Failed to load Protovis:', error);
        });
    }
    if (this.show_doc && splineEditor !== null && vis !== null) {
        const rect = ctx.canvas.getBoundingClientRect()
        const scaleX = rect.width / ctx.canvas.width
        const scaleY = rect.height / ctx.canvas.height
        
        const transform = new DOMMatrix()
        .scaleSelf(scaleX -0.05, scaleY -0.05)
        .multiplySelf(ctx.getTransform())
        .translateSelf(470, -10)
        
        const scale = new DOMMatrix()
        .scaleSelf(transform.a, transform.d);

        const styleObject = {
            transformOrigin: '0 0',
            transform: scale,
            left: `${transform.a + transform.e}px`,
            top: `${transform.d + transform.f}px`,
            };
        Object.assign(splineEditor.style, styleObject);

        //vis.render();
        
        //logPathLength(svgPathElement);
       
        
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

function logPathLength(svgPathElement) {
  //var svgPathElement = document.querySelector('path');
  // If the SVG path element exists, get the path string from the 'd' attribute
  if (svgPathElement) {
      var pathString = svgPathElement.getAttribute('d');

      // Now, you can use the svg-path-properties library on this path string.
      const properties = new svgPathProperties.svgPathProperties(pathString);
      const length = properties.getTotalLength();
      // ...and other properties as needed
      console.log(length);
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