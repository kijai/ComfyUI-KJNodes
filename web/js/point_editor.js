import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

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
    const tag = 'kj-pointseditor-stylesheet'

    let styleTag = document.head.querySelector(tag)

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
    name: 'KJNodes.PointEditor', 
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === 'PointsEditor') {
          chainCallback(nodeType.prototype, "onNodeCreated", function () {
            
            hideWidgetForGood(this, this.widgets.find(w => w.name === "coordinates"))
            hideWidgetForGood(this, this.widgets.find(w => w.name === "bboxes"))

            var element = document.createElement("div");
            this.uuid = makeUUID()
            element.id = `points-editor-${this.uuid}`
            
            // fake image widget to allow copy/paste
            const fakeimagewidget = this.addWidget("COMBO", "image", null, () => {}, {});
            hideWidgetForGood(this, fakeimagewidget)

            this.pointsEditor = this.addDOMWidget(nodeData.name, "PointsEditorWidget", element, {
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
            function createMenuItem(id, textContent) {
              let menuItem = document.createElement("a");
              menuItem.href = "#";
              menuItem.id = `menu-item-${id}`;
              menuItem.textContent = textContent;
              styleMenuItem(menuItem);
              return menuItem;
            }
            
            // Create an array of menu items using the createMenuItem function
            this.menuItems = [
              createMenuItem(0, "Load Image"),
              createMenuItem(1, "Clear Image"),
            ];
            
            // Add mouseover and mouseout event listeners to each menu item for styling
            this.menuItems.forEach(menuItem => {
              menuItem.addEventListener('mouseover', function() {
                this.style.backgroundColor = "gray";
              });
            
              menuItem.addEventListener('mouseout', function() {
                this.style.backgroundColor = "#202020";
              });
            });
            
            // Append each menu item to the context menu
            this.menuItems.forEach(menuItem => {
              this.contextMenu.appendChild(menuItem);
            });

            document.body.appendChild( this.contextMenu);

            this.addWidget("button", "New canvas", null, () => {
              if (!this.properties || !("points" in this.properties)) {
                createPointsEditor(this)
                this.addProperty("points", this.constructor.type, "string");
               
              }
              else {
                createPointsEditor(this, true)
              }
            });
            
            this.setSize([550, 550]);
            this.resizable = false;
            this.pointsEditor.parentEl = document.createElement("div");
            this.pointsEditor.parentEl.className = "points-editor";
            this.pointsEditor.parentEl.id = `points-editor-${this.uuid}`
            element.appendChild(this.pointsEditor.parentEl);
            
            chainCallback(this, "onConfigure", function() {
              createPointsEditor(this);
              });
              
          }); // onAfterGraphConfigured
        }//node created
      } //before register
})//register


function createPointsEditor(context, reset=false) {
  console.log("creatingPointEditor")

  // context menu
  function createContextMenu() {
    document.addEventListener('contextmenu', function(e) {
      e.preventDefault();
    });

    document.addEventListener('click', function(e) {
      if (!context.contextMenu.contains(e.target)) {
        context.contextMenu.style.display = 'none';
      }
    });

    context.pasteFile = function(file) {
      console.log(file);
      if (file.type.startsWith("image/")) {
        handleImageFile(file);
        return true;
      }
      return false;
    };
    
    context.onDragOver = function(e) {
      if (e.dataTransfer && e.dataTransfer.items) {
        return [...e.dataTransfer.items].some(f => f.kind === "file" && f.type.startsWith("image/"));
      }
      return false;
    };
    
    // On drop upload files
    context.onDragDrop = function(e) {
      console.log("onDragDrop called");
      let handled = false;
      for (const file of e.dataTransfer.files) {
        if (file.type.startsWith("image/")) {
          handleImageFile(file);
          handled = true;
        }
      }
      return handled;
    };
    
    function handleImageFile(file) {
      const reader = new FileReader();
    
      reader.onloadend = function() {
        const img = new Image();
        img.src = reader.result;
    
        img.onload = function() {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
    
          const maxWidth = 800; // maximum width
          const maxHeight = 600; // maximum height
          let width = img.width;
          let height = img.height;
    
          // Calculate the new dimensions while preserving the aspect ratio
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
    
          // Set the canvas to the new dimensions
          canvas.width = width;
          canvas.height = height;
    
          // Draw the image to the canvas
          ctx.drawImage(img, 0, 0, width, height);
    
          // Get the compressed image data as a Base64 string
          const base64String = canvas.toDataURL('image/jpeg', 0.5).replace('data:', '').replace(/^.+,/, ''); // 0.7 is the quality from 0 to 1
    
          context.properties.imgData = {
            name: file.name,
            lastModified: file.lastModified,
            size: file.size,
            type: file.type,
            base64: base64String
          };
    
          console.log('Compressed Image Data:', context.properties.imgData);
        };
      };
    
      reader.readAsDataURL(file);
    
      const imageUrl = URL.createObjectURL(file);
      const img = new Image();
      img.src = imageUrl;
    
      img.onload = function() {
        console.log(this.width, this.height); // Access width and height here
        const w = this.width;
        const h = this.height;
        widthWidget.value = w;
        heightWidget.value = h;
        
        if (w > 256) {
          context.setSize([w + 45, context.size[1]]);
        }
        context.setSize([context.size[0], h + 300]);
        vis.width(w);
        vis.height(h);
        updateData();
        backgroundImage.url(imageUrl).visible(true).root.render();
      };
    }

    context.menuItems.forEach((menuItem, index) => {
      menuItem.addEventListener('click', function(e) {
        e.preventDefault();
        // Logic specific to each menu item based on its index or id
        switch (index) {
          case 0:
            // Create file input element
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*'; // Accept only image files

            // Listen for file selection
            fileInput.addEventListener('change', function(event) {
              const file = event.target.files[0]; // Get the selected file

              if (file) {
                // Create a URL for the selected file
                const imageUrl = URL.createObjectURL(file);

                // Create an Image object
                let img = new Image();

                // Set the source of the image object
                img.src = imageUrl;

                img.onload = function() {
                  console.log(this.width, this.height); // Access width and height here
                  widthWidget.value = this.width;
                  heightWidget.value = this.height;
                  w = widthWidget.value;
                  h = heightWidget.value;
                  if (this.width > 256) {
                    context.setSize([this.width + 45, context.size[1]]);
                  }
                  context.setSize([context.size[0], this.height + 230]);
                  vis.width(w);
                  vis.height(h);
                  updateData();
                  backgroundImage.url(imageUrl).visible(true).root.render();
                };
              }
            });

            // If the backgroundImage is already visible, hide it. Otherwise, show file input.
            if (backgroundImage.visible()) {
              backgroundImage.visible(false).root.render();
            } else {
              // Trigger the file input dialog
              fileInput.click();
            }
            context.contextMenu.style.display = 'none';
            break;

          case 1:
            backgroundImage.visible(false).root.render();
            context.properties.imgData = null;
            context.contextMenu.style.display = 'none';
            break;
        }
      });
    });
  }
  
  createContextMenu();
  function updateData() {
      if (points.length == 0) {
        console.log("no points")
        return
      }
      let coords = points
      let coordsString = JSON.stringify(coords);
      let bbox = calculateBBox(box_startX, box_startY, box_endX, box_endY)
      let bboxString = JSON.stringify(bbox);
      pointsStoreWidget.value = JSON.stringify(points);
      bboxStoreWidget.value = JSON.stringify(bboxString);
      if (coordWidget) {
        coordWidget.value = coordsString;
        }
      if (bboxWidget) {
        bboxWidget.value = bboxString;
      }
      vis.render();
  }
  
  if (reset && context.pointsEditor.element) {
    context.pointsEditor.element.innerHTML = ''; // Clear the container
  }
  const coordWidget = context.widgets.find(w => w.name === "coordinates");
  const pointsStoreWidget = context.widgets.find(w => w.name === "points_store");
  const widthWidget = context.widgets.find(w => w.name === "width");
  const heightWidget = context.widgets.find(w => w.name === "height");
  const bboxStoreWidget = context.widgets.find(w => w.name === "bbox_store");
  const bboxWidget = context.widgets.find(w => w.name === "bboxes");
  
  widthWidget.callback = () => {
    w = widthWidget.value;
    if (w > 256) {
        context.setSize([w + 45, context.size[1]]);
    }
    vis.width(w);
    updateData();
}
  heightWidget.callback = () => {
    h = heightWidget.value
    vis.height(h)
    context.setSize([context.size[0], h + 300]);
    updateData();
  }
  pointsStoreWidget.callback = () => {
    points = JSON.parse(pointsStoreWidget.value);
    updateData();
  }
  
 // Initialize or reset points array
 var hoverIndex = -1;
 var isDragging = false;
 var w = widthWidget.value;
 var h = heightWidget.value;
 var i = 3;
 let points = [];
 let bbox = [];
 var box_startX, box_startY, box_endX, box_endY;
 var drawing = false;

 if (!reset && pointsStoreWidget.value != "") {
  points = JSON.parse(pointsStoreWidget.value);
  bbox = JSON.parse(bboxStoreWidget.value);
  console.log(context);    
 } else {
  points = [
    {
      x: w / 2, // Middle point horizontally centered
      y: h / 2 // Middle point vertically centered
    }
  ];
    pointsStoreWidget.value = JSON.stringify(points);
    bboxStoreWidget.value = JSON.stringify(bbox);
 }
 
  var vis = new pv.Panel()
  .width(w)
  .height(h)
  .fillStyle("#222")
  .strokeStyle("gray")
  .lineWidth(2)
  .antialias(false)
  .margin(10)
  .event("mousedown", function() {
    if (pv.event.shiftKey) { // Use pv.event to access the event object
        let scaledMouse = {
        x: this.mouse().x / app.canvas.ds.scale,
        y: this.mouse().y / app.canvas.ds.scale
        };
        i = points.push(scaledMouse) - 1;
        updateData();
        return this;
    }
    else if (pv.event.ctrlKey) {
      console.log("start drawing at " + this.mouse().x / app.canvas.ds.scale + ", " + this.mouse().y / app.canvas.ds.scale);
      drawing = true;
      box_startX = this.mouse().x / app.canvas.ds.scale;
      box_startY = this.mouse().y / app.canvas.ds.scale;
    }
    else if (pv.event.button === 2) {
      context.contextMenu.style.display = 'block';
      context.contextMenu.style.left = `${pv.event.clientX}px`;
      context.contextMenu.style.top = `${pv.event.clientY}px`;
      }
    })
    .event("mousemove", function() {
      if (drawing) {
          box_endX = this.mouse().x / app.canvas.ds.scale;
          box_endY = this.mouse().y / app.canvas.ds.scale;
          vis.render();
      }
  })
    .event("mouseup", function() {
      console.log("end drawing at " + this.mouse().x / app.canvas.ds.scale + ", " + this.mouse().y / app.canvas.ds.scale);
      drawing = false;
      updateData();
      });

    var backgroundImage = vis.add(pv.Image)
      .visible(false)

    vis.add(pv.Area)
        .data(function() {
            return drawing || bbox ? [box_startX, box_endX] : [];
        })
        .bottom(function() {
            return h - Math.max(box_startY, box_endY);
        })
        .left(function(d) {
            return d;
        })
        .height(function() {
            return Math.abs(box_startY - box_endY);
        })
        .fillStyle("rgba(70, 130, 180, 0.5)")
        .strokeStyle("steelblue");

  vis.add(pv.Dot)
    .data(() => points)
    .left(d => d.x)
    .top(d => d.y)
    .radius(Math.log(Math.min(w, h)) * 4)
    .shape("circle")
    .cursor("move")
    .strokeStyle(function() { return i == this.index ? "#ff7f0e" : "#00FFFF"; })
    .lineWidth(4)  // Set the stroke thickness to 2
    .fillStyle(function() { return "rgba(100, 100, 100, 0.6)"; })
    .event("mousedown", pv.Behavior.drag())
    .event("dragstart", function() {
        i = this.index;
        console.log(i)
        hoverIndex = this.index;
        isDragging = true;
        
    })
    .event("dragend", function() {
      if (pv.event.button === 2 && i !== 0 && i !== points.length - 1) {
        this.index = i;
        points.splice(i--, 1);
        
      }
        updateData();
          isDragging = false; 
        
    })
    .event("drag", function() {
      let adjustedX = this.mouse().x / app.canvas.ds.scale; // Adjust the new X position by the inverse of the scale factor
      let adjustedY = this.mouse().y / app.canvas.ds.scale; // Adjust the new Y position by the inverse of the scale factor
       // Determine the bounds of the vis.Panel
      const panelWidth = vis.width();
      const panelHeight = vis.height();

      // Adjust the new position if it would place the dot outside the bounds of the vis.Panel
      adjustedX = Math.max(0, Math.min(panelWidth, adjustedX));
      adjustedY = Math.max(0, Math.min(panelHeight, adjustedY));
      points[this.index] = { x: adjustedX, y: adjustedY }; // Update the point's position
      vis.render(); // Re-render the visualization to reflect the new position
   })
    // .event("mouseover", function() {
    //   hoverIndex = this.index; // Set the hover index to the index of the hovered dot
    //   vis.render(); // Re-render the visualization
    // })
    // .event("mouseout", function() {
    //   !isDragging && (hoverIndex = -1); // Reset the hover index when the mouse leaves the dot
    //   vis.render(); // Re-render the visualization
    // })
    .anchor("center")
    .add(pv.Label)
    // .visible(function() {
    //   return hoverIndex === this.index; // Only show the label for the hovered dot
    // })
    .left(d => d.x < w / 2 ? d.x + 30 : d.x - 35) // Shift label to right if on left half, otherwise shift to left
    .top(d => d.y < h / 2 ? d.y + 25 : d.y - 25)  // Shift label down if on top half, otherwise shift up
    .font(25 + "px sans-serif")
    // .text(d => {
    //     return `X: ${Math.round(d.x)}, Y: ${Math.round(d.y)}`;
    //   })
    .text(d => {
      return points.indexOf(d);
  })
    .textStyle("cyan")
    .textShadow("2px 2px 2px black")
    
    .add(pv.Dot) // Add smaller point in the center
      .data(() => points)
      .left(d => d.x)
      .top(d => d.y)
      .radius(2)  // Smaller radius for the center point
      .shape("circle")
      .fillStyle("red")  // Color for the center point
      .lineWidth(1);  // Stroke thickness for the center point
      
    if (points.length != 0) {
      vis.render();
    }
    
    var svgElement = vis.canvas();
    svgElement.style['zIndex'] = "2"
    svgElement.style['position'] = "relative"
    context.pointsEditor.element.appendChild(svgElement);

    if (w > 256) {
      context.setSize([w + 45, context.size[1]]);
    }
    context.setSize([context.size[0], h + 300]);
    updateData();

    if (context.properties.imgData && context.properties.imgData.base64) {
      const base64String = context.properties.imgData.base64;
      const imageUrl = `data:${context.properties.imgData.type};base64,${base64String}`;
      
      backgroundImage.url(imageUrl).visible(true).root.render();
    }

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

function calculateBBox(x1, y1, x2, y2) {
  var x = Math.min(x1, x2);
  var y = Math.min(y1, y2);
  var width = Math.abs(x2 - x1);
  var height = Math.abs(y2 - y1);
  return { x: x, y: y, width: width, height: height };
}