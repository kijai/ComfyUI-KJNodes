/**
 * Utility functions for protovis in ComfyUI.
 */

/**
 * Get correct local coordinates for protovis in transformed containers.
 * Uses getBoundingClientRect() which properly accounts for CSS transforms.
 *
 * This fixes coordinate calculation issues when protovis widgets are rendered
 * inside ComfyUI's vueNodes mode, which uses CSS transforms for panning/zooming.
 *
 * @param {pv.Mark} mark - The protovis mark instance
 * @returns {{x: number, y: number}} Local coordinates relative to the canvas
 */
export function getLocalMouse(mark) {
  const e = pv.event
  if (!e) return { x: 0, y: 0 }
  const canvas = mark.root.canvas()
  if (!canvas) return { x: 0, y: 0 }
  const rect = canvas.getBoundingClientRect()
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top
  }
}
