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
        const fakeimagewidget = this.addWidget("COMBO", "image", null, () => { }, {});
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
          menuItem.addEventListener('mouseover', function () {
            this.style.backgroundColor = "gray";
          });

          menuItem.addEventListener('mouseout', function () {
            this.style.backgroundColor = "#202020";
          });
        });

        // Append each menu item to the context menu
        this.menuItems.forEach(menuItem => {
          this.contextMenu.appendChild(menuItem);
        });

        document.body.appendChild(this.contextMenu);

        this.addWidget("button", "New canvas", null, () => {
          if (!this.properties || !("points" in this.properties)) {
            this.editor = new PointsEditor(this);
            this.addProperty("points", this.constructor.type, "string");

          }
          else {
            this.editor = new PointsEditor(this, true);
          }
        });

        this.setSize([550, 550]);
        this.resizable = false;
        this.pointsEditor.parentEl = document.createElement("div");
        this.pointsEditor.parentEl.className = "points-editor";
        this.pointsEditor.parentEl.id = `points-editor-${this.uuid}`
        element.appendChild(this.pointsEditor.parentEl);

        chainCallback(this, "onConfigure", function () {
          this.editor = new PointsEditor(this);
        });
        chainCallback(this, "onExecuted", function (message) {
          let bg_image = message["bg_image"];
          this.properties.imgData = {
            name: "bg_image",
            base64: bg_image
          };
          this.editor.refreshBackgroundImage(this);
        });

      }); // onAfterGraphConfigured
    }//node created
  } //before register
})//register

class PointsEditor {
  constructor(context, reset = false) {
    this.node = context;
    this.reset = reset;

    console.log("creatingPointEditor")

    this.node.pasteFile = (file) => {
      if (file.type.startsWith("image/")) {
        this.handleImageFile(file);
        return true;
      }
      return false;
    };

    this.node.onDragOver = function (e) {
      if (e.dataTransfer && e.dataTransfer.items) {
        return [...e.dataTransfer.items].some(f => f.kind === "file" && f.type.startsWith("image/"));
      }
      return false;
    };

    // On drop upload files
    this.node.onDragDrop = (e) => {
      console.log("onDragDrop called");
      let handled = false;
      for (const file of e.dataTransfer.files) {
        if (file.type.startsWith("image/")) {
          this.handleImageFile(file);
          handled = true;
        }
      }
      return handled;
    };

    // context menu
    this.createContextMenu();


    if (reset && context.pointsEditor.element) {
      context.pointsEditor.element.innerHTML = ''; // Clear the container
    }
    this.coordWidget = context.widgets.find(w => w.name === "coordinates");
    this.pointsStoreWidget = context.widgets.find(w => w.name === "points_store");
    this.widthWidget = context.widgets.find(w => w.name === "width");
    this.heightWidget = context.widgets.find(w => w.name === "height");
    this.bboxStoreWidget = context.widgets.find(w => w.name === "bbox_store");
    this.bboxWidget = context.widgets.find(w => w.name === "bboxes");

    //widget callbacks
    this.widthWidget.callback = () => {
      w = this.widthWidget.value;
      if (w > 256) {
        context.setSize([w + 45, context.size[1]]);
      }
      this.vis.width(w);
      this.updateData();
    }
    this.heightWidget.callback = () => {
      h = this.heightWidget.value
      this.vis.height(h)
      context.setSize([context.size[0], h + 300]);
      this.updateData();
    }
    this.pointsStoreWidget.callback = () => {
      this.points = JSON.parse(pointsStoreWidget.value);
      this.updateData();
    }

    var w = this.widthWidget.value;
    var h = this.heightWidget.value;
    var i = 3;
    this.points = [];
    this.bbox = [];
    var drawing = false;

    // Initialize or reset points array
    if (!reset && this.pointsStoreWidget.value != "") {
      this.points = JSON.parse(this.pointsStoreWidget.value);
      this.bbox = JSON.parse(this.bboxStoreWidget.value);
    } else {
      this.points = [
        {
          x: w / 2, // Middle point horizontally centered
          y: h / 2 // Middle point vertically centered
        }
      ];
      this.pointsStoreWidget.value = JSON.stringify(this.points);
      this.bboxStoreWidget.value = JSON.stringify(this.bbox);
    }
    const self = this; // Keep a reference to the main class context
    this.vis = new pv.Panel()
      .width(w)
      .height(h)
      .fillStyle("#222")
      .strokeStyle("gray")
      .lineWidth(2)
      .antialias(false)
      .margin(10)
      .event("mousedown", function () {
        if (pv.event.shiftKey) { // Use pv.event to access the event object
          let scaledMouse = {
            x: this.mouse().x / app.canvas.ds.scale,
            y: this.mouse().y / app.canvas.ds.scale
          };
          i = self.points.push(scaledMouse) - 1;
          self.updateData();
          return this;
        }
        else if (pv.event.ctrlKey) {
          console.log("start drawing at " + this.mouse().x / app.canvas.ds.scale + ", " + this.mouse().y / app.canvas.ds.scale);
          drawing = true;
          self.box_startX = this.mouse().x / app.canvas.ds.scale;
          self.box_startY = this.mouse().y / app.canvas.ds.scale;
        }
        else if (pv.event.button === 2) {
          self.node.contextMenu.style.display = 'block';
          self.node.contextMenu.style.left = `${pv.event.clientX}px`;
          self.node.contextMenu.style.top = `${pv.event.clientY}px`;
        }
      })
      .event("mousemove", function () {
        if (drawing) {
          self.box_endX = this.mouse().x / app.canvas.ds.scale;
          self.box_endY = this.mouse().y / app.canvas.ds.scale;
          self.vis.render();
        }
      })
      .event("mouseup", function () {
        console.log("end drawing at " + this.mouse().x / app.canvas.ds.scale + ", " + this.mouse().y / app.canvas.ds.scale);
        drawing = false;
        self.updateData();
      });

    this.backgroundImage = this.vis.add(pv.Image).visible(false)

    this.vis.add(pv.Area)
      .data(function () {return drawing || self.bbox ? [self.box_startX, self.box_endX] : []; })
      .bottom(function () {return h - Math.max(self.box_startY, self.box_endY); })
      .left(function (d) {return d; })
      .height(function () {return Math.abs(self.box_startY - self.box_endY);})
      .fillStyle("rgba(70, 130, 180, 0.5)")
      .strokeStyle("steelblue");

    this.vis.add(pv.Dot)
      .data(() => this.points)
      .left(d => d.x)
      .top(d => d.y)
      .radius(Math.log(Math.min(w, h)) * 4)
      .shape("circle")
      .cursor("move")
      .strokeStyle(function () { return i == this.index ? "#ff7f0e" : "#00FFFF"; })
      .lineWidth(4)
      .fillStyle(function () { return "rgba(100, 100, 100, 0.6)"; })
      .event("mousedown", pv.Behavior.drag())
      .event("dragstart", function () {
        i = this.index;
      })
      .event("dragend", function () {
        if (pv.event.button === 2 && i !== 0 && i !== self.points.length - 1) {
          this.index = i;
          self.points.splice(i--, 1);
        }
        self.updateData();

      })
      .event("drag", function () {
        let adjustedX = this.mouse().x / app.canvas.ds.scale; // Adjust the new X position by the inverse of the scale factor
        let adjustedY = this.mouse().y / app.canvas.ds.scale; // Adjust the new Y position by the inverse of the scale factor
        // Determine the bounds of the vis.Panel
        const panelWidth = self.vis.width();
        const panelHeight = self.vis.height();

        // Adjust the new position if it would place the dot outside the bounds of the vis.Panel
        adjustedX = Math.max(0, Math.min(panelWidth, adjustedX));
        adjustedY = Math.max(0, Math.min(panelHeight, adjustedY));
        self.points[this.index] = { x: adjustedX, y: adjustedY }; // Update the point's position
        self.vis.render(); // Re-render the visualization to reflect the new position
      })

      .anchor("center")
      .add(pv.Label)
      .left(d => d.x < w / 2 ? d.x + 30 : d.x - 35) // Shift label to right if on left half, otherwise shift to left
      .top(d => d.y < h / 2 ? d.y + 25 : d.y - 25)  // Shift label down if on top half, otherwise shift up
      .font(25 + "px sans-serif")
      .text(d => {return this.points.indexOf(d); })
      .textStyle("cyan")
      .textShadow("2px 2px 2px black")
      .add(pv.Dot) // Add smaller point in the center
      .data(() => this.points)
      .left(d => d.x)
      .top(d => d.y)
      .radius(2)  // Smaller radius for the center point
      .shape("circle")
      .fillStyle("red")  // Color for the center point
      .lineWidth(1);  // Stroke thickness for the center point

    if (this.points.length != 0) {
      this.vis.render();
    }

    var svgElement = this.vis.canvas();
    svgElement.style['zIndex'] = "2"
    svgElement.style['position'] = "relative"
    this.node.pointsEditor.element.appendChild(svgElement);

    if (w > 256) {
      this.node.setSize([w + 45, this.node.size[1]]);
    }
    this.node.setSize([this.node.size[0], h + 300]);
    this.updateData();
    this.refreshBackgroundImage();

  }//end constructor

  updateData = () => {
    if (this.points.length == 0) {
      console.log("no points")
      return
    }
    let coordsString = JSON.stringify(this.points);
    let bbox = calculateBBox(this.box_startX, this.box_startY, this.box_endX, this.box_endY);
    let bboxString = JSON.stringify(bbox);
    this.pointsStoreWidget.value = JSON.stringify(this.points);
    this.bboxStoreWidget.value = JSON.stringify(bboxString);
    this.coordWidget.value = coordsString;
    this.bboxWidget.value = bboxString;
    this.vis.render();
    };

  handleImageLoad = (img, file, base64String) => {
    console.log(img.width, img.height); // Access width and height here
    this.widthWidget.value = img.width;
    this.heightWidget.value = img.height;

    if (img.width > 256) {
      this.node.setSize([img.width + 45, this.node.size[1]]);
    }
    this.node.setSize([this.node.size[0], img.height + 300]);
    this.vis.width(img.width);
    this.vis.height(img.height);
    this.updateData();
    this.backgroundImage.url(file ? URL.createObjectURL(file) : `data:${this.node.properties.imgData.type};base64,${base64String}`).visible(true).root.render();
    };

  processImage = (img, file) => {
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

    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    // Get the compressed image data as a Base64 string
    const base64String = canvas.toDataURL('image/jpeg', 0.5).replace('data:', '').replace(/^.+,/, ''); // 0.5 is the quality from 0 to 1

    this.node.properties.imgData = {
      name: file.name,
      lastModified: file.lastModified,
      size: file.size,
      type: file.type,
      base64: base64String
    };
    handleImageLoad(img, file, base64String);
};

  handleImageFile = (file) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const img = new Image();
      img.src = reader.result;
      img.onload = () => processImage(img, file);
    };
    reader.readAsDataURL(file);

    const imageUrl = URL.createObjectURL(file);
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => this.handleImageLoad(img, file, null);
  };

  refreshBackgroundImage = () => {
    if (this.node.properties.imgData && this.node.properties.imgData.base64) {
      const base64String = this.node.properties.imgData.base64;
      const imageUrl = `data:${this.node.properties.imgData.type};base64,${base64String}`;
      const img = new Image();
      img.src = imageUrl;
      img.onload = () => this.handleImageLoad(img, null, base64String);
    }
  };

  createContextMenu = () => {
    self = this;
    document.addEventListener('contextmenu', function (e) {
      e.preventDefault();
    });

    document.addEventListener('click', function (e) {
      if (!self.node.contextMenu.contains(e.target)) {
        self.node.contextMenu.style.display = 'none';
      }
    });

    this.node.menuItems.forEach((menuItem, index) => {
      self = this;
      menuItem.addEventListener('click', function (e) {
        e.preventDefault();
        switch (index) {
          case 0:
            // Create file input element
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*'; // Accept only image files

            // Listen for file selection
            fileInput.addEventListener('change', function (event) {
              const file = event.target.files[0]; // Get the selected file

              if (file) {
                const imageUrl = URL.createObjectURL(file);
                let img = new Image();
                img.src = imageUrl;
                img.onload = () => self.handleImageLoad(img, file, null);
              }
            });

            fileInput.click();
            
            self.node.contextMenu.style.display = 'none';
            break;
          case 1:
            self.backgroundImage.visible(false).root.render();
            self.node.properties.imgData = null;
            self.node.contextMenu.style.display = 'none';
            break;
        }
      });
    });
  }//end createContextMenu
}//end class


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