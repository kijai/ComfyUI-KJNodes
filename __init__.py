"""Ideogram 4 Prompt Builder KJ V2 — standalone custom node.

A freehand-drawing upgrade of KJNodes' "Ideogram 4 Prompt Builder KJ":
draw shapes by hand (brush, line, rectangle, ellipse), bucket-fill closed
areas like in Paint, and the node auto-outlines each drawn shape into its
own bbox region for the Ideogram 4 JSON caption. Also supports importing a
local image as a background layer with adjustable opacity.

Same inputs/outputs as the original node, so it connects to the same
workflow nodes. Drop this folder into ComfyUI/custom_nodes/ and restart.
"""

from .nodes.ideogram4_v2_nodes import Ideogram4PromptBuilderKJV2

NODE_CLASS_MAPPINGS = {
    "Ideogram4PromptBuilderKJV2": Ideogram4PromptBuilderKJV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogram4PromptBuilderKJV2": "Ideogram 4 Prompt Builder KJ V2",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
