# KJNodes for ComfyUI

At this point pretty random collection of utility, model optimization and QoL nodes, while keeping dependencies at minimum.

Documentation is mostly in the node descriptions and tooltips.

# Installation
1. Clone this repo into `custom_nodes` folder.
2. Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in ComfyUI_windows_portable -folder:

  `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-KJNodes\requirements.txt`
   
### Set/Get

# UPDATE March 19th 2026

Complete (backwards compatible as far as I'm aware currently) rewrite of Set/Get nodes.

# Changelog March 19th 2026

## New Features

- **Nodes 2.0 support** - Works with nodes 2.0 as well as legacy.

- **Subgraph support** — Set/Get now works across subgraph boundaries. A Set in a parent graph is visible to all child subgraphs, and Get nodes search upward through ancestors. Cross-graph connections are resolved during prompt execution.

- **Convert link to Set/Get** — Right-click any link midpoint to convert it into a Set/Get pair.

- **Convert to links** — Right-click a Set or Get node to convert the pair back to direct links.

- **Convert outputs to Set/Get** — Batch operation to convert all outputs of selected nodes into Set/Get pairs at once.

- **Canvas right-click menu** — "Convert outputs on all selected nodes to Set/Get" and "Convert selected Set/Get to links".

- **Keyboard shortcuts** — Ctrl+Shift+S (add Set to selected / at cursor), Ctrl+Shift+G (add Get at cursor), Ctrl+Shift+L (force-show all connections toggle).

- **"Convert ALL Set/Get to links" setting** — One-click button in settings to replace every Set/Get pair with direct links, including across subgraph boundaries.

- **Show links setting** — Combo setting (never / selected / always) to control when virtual links between Set/Get pairs are drawn. "Selected" mode shows links only for currently selected Set/Get nodes. Ctrl+Shift+L temporarily forces all links visible.

- **Add Set/Get from connection menu** — When dragging from a slot, "Add SetNode" and "Add GetNode" entries appear next to "Add Reroute" in the connection menu.

- **Shift+middle-click creates Set/Get** — Shift+middle-click on an output slot creates a connected SetNode; on an input slot creates a connected GetNode. Optional setting to override normal middle-click to also create Set/Get instead of Reroute.

- **Add paired GetNode** — Right-click a SetNode to add a paired GetNode pre-configured with the same name.

- **Double-click Get to jump to Set** — Double-click a GetNode to center and select its paired SetNode.

- **Default widget value setting** — Controls the initial Constant value when a Set node is first connected: empty (default), slot name, slot name (lowercase), or slot name (UPPERCASE).

- **Type inference from output** — If a SetNode's input is unconnected but its output is connected to a typed input, the Set adopts that type and recolors accordingly.

- **Getters submenu navigates into subgraphs** — SetNode's getter list now includes cross-subgraph GetNodes and can navigate into subgraphs to focus them.

## Fixes

- **Paste rename coordination** — When pasting Set+Get pairs, GetNodes now correctly follow their SetNode's rename (e.g., `MODEL` -> `MODEL_0`).

- **Type/color reset on disconnect** — Disconnecting input or output now correctly resets type and color, but only when neither side is still connected.

- **Suffix stripping only on paste** — `validateName` no longer strips intentional `_N` suffixes from user-typed names. Only strips during paste to avoid `FOO_0_1_2` accumulation.

- **Connection drawing moved to canvas level** — Visualization links remain visible even when nodes are off-screen or collapsed.

## Settings

- Settings are now organized under **KJNodes > Set & Get** and **KJNodes > General** categories in the settings panel.


---

