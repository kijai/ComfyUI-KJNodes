const { app } = window.comfyAPI.app;

// code based on mtb nodes by Mel Massadian https://github.com/melMass/comfy_mtb/
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

loadScript('kjweb_async/marked.min.js').catch((e) => {
  console.error(e)
})
loadScript('kjweb_async/purify.min.js').catch((e) => {
  console.error(e)
})

const categories = ["KJNodes", "SUPIR", "VoiceCraft", "Marigold", "IC-Light", "WanVideoWrapper"];
const nodeDescriptions = new Map();

function isHelpPopupEnabled() {
  return app.ui.settings.getSettingValue("KJNodes.helpPopup") !== false;
}

app.registerExtension({
	name: "KJNodes.HelpPopup",
	async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isHelpPopupEnabled()) return;
		try {
			categories.forEach(category => {
        if (nodeData?.category?.startsWith(category)) {
            if (nodeData.description) {
              nodeDescriptions.set(nodeData.name, nodeData.description);
            }
            addDocumentation(nodeData, nodeType);
        }
        else return
    });
		} catch (error) {
			console.error("Error in registering KJNodes.HelpPopup", error);
		}
	},
	nodeCreated(node) {
    if (!isHelpPopupEnabled()) return;
    const description = nodeDescriptions.get(node.type) || nodeDescriptions.get(node.comfyClass);
    if (!description) return;
    node._kjHelpDescription = description;
  },
  setup() {
    if (!isHelpPopupEnabled()) return;
    setupHelpObserver();
  },
});

const create_documentation_stylesheet = () => {
    const tag = 'kj-documentation-stylesheet'

    let styleTag = document.getElementById(tag)

    if (!styleTag) {
      styleTag = document.createElement('style')
      styleTag.type = 'text/css'
      styleTag.id = tag
      styleTag.innerHTML = `
      .kj-documentation-popup {
        background: var(--comfy-menu-bg);
        position: absolute;
        color: var(--fg-color);
        font: 12px monospace;
        line-height: 1.5em;
        padding: 10px;
        border-radius: 10px;
        border-style: solid;
        border-width: medium;
        border-color: var(--border-color);
        z-index: 5;
        overflow: hidden;
       }
       .content-wrapper {
        overflow: auto;
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

/**
 * Creates the documentation popup DOM and wires up resize/close interactions.
 * Returns { docElement, contentWrapper }.
 * @param {string} description - Markdown description text
 * @param {AbortSignal} signal - Signal to clean up event listeners
 * @param {() => void} onClose - Called when the close button is clicked
 * @param {{ scaleResize: boolean }} opts - If scaleResize is true, resize deltas are divided by canvas scale
 */
function createDocPopup(description, signal, onClose, opts = {}) {
  create_documentation_stylesheet()

  const docElement = document.createElement('div')
  const contentWrapper = document.createElement('div')
  docElement.appendChild(contentWrapper)

  contentWrapper.classList.add('content-wrapper')
  docElement.classList.add('kj-documentation-popup')
  // Try ComfyUI's built-in markdown renderer first (available after frontend PR #10700)
  if (app.extensionManager?.renderMarkdownToHtml) {
    contentWrapper.innerHTML = app.extensionManager.renderMarkdownToHtml(description)
  } else if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
    contentWrapper.innerHTML = DOMPurify.sanitize(marked.parse(description))
  } else {
    // Fallback: convert markdown links to <a> tags, auto-link bare URLs, preserve line breaks
    const escaped = description
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
      .replace(/(^|[^"'])(https?:\/\/[^\s<]+)/g, '$1<a href="$2" target="_blank">$2</a>')
      .replace(/\n/g, '<br>')
    contentWrapper.innerHTML = escaped
  }

  // resize handle
  const resizeHandle = document.createElement('div')
  resizeHandle.style.width = '0'
  resizeHandle.style.height = '0'
  resizeHandle.style.position = 'absolute'
  resizeHandle.style.bottom = '0'
  resizeHandle.style.right = '0'
  resizeHandle.style.cursor = 'se-resize'
  const borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim()
  resizeHandle.style.borderTop = '10px solid transparent'
  resizeHandle.style.borderLeft = '10px solid transparent'
  resizeHandle.style.borderBottom = `10px solid ${borderColor}`
  resizeHandle.style.borderRight = `10px solid ${borderColor}`
  docElement.appendChild(resizeHandle)

  let isResizing = false
  let startX, startY, startWidth, startHeight
  resizeHandle.addEventListener('mousedown', (e) => {
    e.preventDefault()
    e.stopPropagation()
    isResizing = true
    startX = e.clientX
    startY = e.clientY
    startWidth = parseInt(document.defaultView.getComputedStyle(docElement).width, 10)
    startHeight = parseInt(document.defaultView.getComputedStyle(docElement).height, 10)
  }, { signal })

  document.addEventListener('mousemove', (e) => {
    if (!isResizing) return
    const scaleFactor = opts.scaleResize ? app.canvas.ds.scale : 1
    const newWidth = startWidth + (e.clientX - startX) / scaleFactor
    const newHeight = startHeight + (e.clientY - startY) / scaleFactor
    docElement.style.width = `${newWidth}px`
    docElement.style.height = `${newHeight}px`
  }, { signal })

  document.addEventListener('mouseup', () => {
    isResizing = false
  }, { signal })

  // close button
  const closeButton = document.createElement('div')
  closeButton.textContent = '❌'
  closeButton.style.position = 'absolute'
  closeButton.style.top = '0'
  closeButton.style.right = '0'
  closeButton.style.cursor = 'pointer'
  closeButton.style.padding = '5px'
  closeButton.style.color = 'red'
  closeButton.style.fontSize = '12px'
  docElement.appendChild(closeButton)

  closeButton.addEventListener('mousedown', (e) => {
    e.stopPropagation()
    onClose()
  }, { signal })

  document.body.appendChild(docElement)
  return { docElement, contentWrapper }
}

// ─── Legacy canvas mode (onDrawForeground) ───

/** Add documentation widget to the selected node (legacy canvas rendering) */
export const addDocumentation = (
  nodeData,
  nodeType,
  opts = { icon_size: 14, icon_margin: 4 },) => {

  opts = opts || {}
  const iconSize = opts.icon_size ? opts.icon_size : 14
  const iconMargin = opts.icon_margin ? opts.icon_margin : 4
  let docElement = null
  let contentWrapper = null

  if (!nodeData.description) {
    return
  }

  const drawFg = nodeType.prototype.onDrawForeground
  nodeType.prototype.onDrawForeground = function (ctx) {
    const r = drawFg ? drawFg.apply(this, arguments) : undefined
    if (this.flags.collapsed) return r

    const x = this.size[0] - iconSize - iconMargin

    // create the popup
    if (this.show_doc && docElement === null) {
      const popup = createDocPopup(
        nodeData.description,
        this.docCtrl.signal,
        () => {
          this.show_doc = !this.show_doc
          docElement.parentNode.removeChild(docElement)
          docElement = null
          contentWrapper = null
        },
        { scaleResize: true }
      )
      docElement = popup.docElement
      contentWrapper = popup.contentWrapper
    }
    // close the popup
    else if (!this.show_doc && docElement !== null) {
      docElement.parentNode.removeChild(docElement)
      docElement = null
    }
    // update position of the popup
    if (this.show_doc && docElement !== null) {
      const rect = ctx.canvas.getBoundingClientRect()
      const scaleX = rect.width / ctx.canvas.width
      const scaleY = rect.height / ctx.canvas.height

      const transform = new DOMMatrix()
      .scaleSelf(scaleX, scaleY)
      .multiplySelf(ctx.getTransform())
      .translateSelf(this.size[0] * scaleX * Math.max(1.0,window.devicePixelRatio) , 0)
      .translateSelf(10, -32)

      const scale = new DOMMatrix()
      .scaleSelf(transform.a, transform.d);
      const bcr = app.canvas.canvas.getBoundingClientRect()

      const styleObject = {
        transformOrigin: '0 0',
        transform: scale,
        left: `${transform.a + bcr.x + transform.e}px`,
        top: `${transform.d + bcr.y + transform.f}px`,
       };
      Object.assign(docElement.style, styleObject);
    }

    ctx.save()
    ctx.translate(x - 2, iconSize - 34)
    ctx.scale(iconSize / 32, iconSize / 32)
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.lineWidth = 2.4
    ctx.font = 'bold 36px monospace'
    ctx.fillStyle = 'orange';
    ctx.fillText('?', 0, 24)
    ctx.restore()
    return r
  }

  // handle clicking of the icon
  const mouseDown = nodeType.prototype.onMouseDown
  nodeType.prototype.onMouseDown = function (e, localPos, canvas) {
    const r = mouseDown ? mouseDown.apply(this, arguments) : undefined
    const iconX = this.size[0] - iconSize - iconMargin
    const iconY = iconSize - 34
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
      if (this.show_doc) {
        this.docCtrl = new AbortController()
      } else {
        this.docCtrl.abort()
      }
      return true;
    }
    return r;
  }

  const onRem = nodeType.prototype.onRemoved
  nodeType.prototype.onRemoved = function () {
    const r = onRem ? onRem.apply(this, []) : undefined
    if (docElement) {
      docElement.remove()
      docElement = null
      contentWrapper = null
    }
    return r
  }
}

// ─── Vue nodes mode (DOM injection via MutationObserver) ───

/** Per-node popup state, keyed by node ID */
const popupState = new Map()

function getNodeById(nodeId) {
  return app.graph?.getNodeById(nodeId)
}

function closeNodePopup(nodeId) {
  const state = popupState.get(nodeId)
  if (!state) return
  if (state.docElement) {
    state.docElement.remove()
  }
  if (state.abortCtrl) {
    state.abortCtrl.abort()
  }
  if (state.animFrame) {
    cancelAnimationFrame(state.animFrame)
  }
  popupState.delete(nodeId)
}

function openNodePopup(nodeId, description) {
  closeNodePopup(nodeId)
  const state = popupState.get(nodeId) || {}
  popupState.set(nodeId, state)

  state.abortCtrl = new AbortController()
  const popup = createDocPopup(
    description,
    state.abortCtrl.signal,
    () => closeNodePopup(nodeId),
    { scaleResize: false }
  )
  state.docElement = popup.docElement

  function updatePosition() {
    if (!state.docElement || !state.docElement.parentNode) return
    const nodeEl = document.querySelector(`[data-node-id="${nodeId}"]`)
    if (nodeEl) {
      const rect = nodeEl.getBoundingClientRect()
      state.docElement.style.left = `${rect.right + 10}px`
      state.docElement.style.top = `${rect.top}px`
    }
    state.animFrame = requestAnimationFrame(updatePosition)
  }
  state.animFrame = requestAnimationFrame(updatePosition)
}

/** Try to inject a "?" button into a Vue node header */
function tryInjectHelpButton(header) {
  if (header.querySelector('.kj-help-btn')) return

  const nodeEl = header.closest('[data-node-id]')
  if (!nodeEl) return
  const nodeId = nodeEl.dataset.nodeId
  const node = getNodeById(nodeId)
  if (!node) return

  const description = node._kjHelpDescription
  if (!description) return

  const flexContainer = header.querySelector(':scope > div')
  if (!flexContainer) return

  const helpBtn = document.createElement('span')
  helpBtn.className = 'kj-help-btn'
  helpBtn.textContent = '?'
  helpBtn.style.cssText = `
    color: orange;
    font-weight: bold;
    font-size: 14px;
    cursor: pointer;
    flex-shrink: 0;
    padding: 0 4px;
    line-height: 1;
    user-select: none;
  `
  helpBtn.title = 'Show help'
  helpBtn.addEventListener('click', (e) => {
    e.stopPropagation()
    const state = popupState.get(nodeId)
    if (state?.docElement) {
      closeNodePopup(nodeId)
    } else {
      openNodePopup(nodeId, description)
    }
  })
  flexContainer.appendChild(helpBtn)
}

/** Observe the DOM for Vue node headers appearing and inject help buttons */
function setupHelpObserver() {
  // Inject into any headers already in the DOM
  document.querySelectorAll('.lg-node-header').forEach(tryInjectHelpButton)

  let pending = false
  const observer = new MutationObserver(() => {
    if (pending) return
    pending = true
    requestAnimationFrame(() => {
      pending = false
      document.querySelectorAll('.lg-node-header:not(:has(.kj-help-btn))').forEach(tryInjectHelpButton)
    })
  })
  observer.observe(document.body, { childList: true, subtree: true })
}
