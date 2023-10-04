# KJNodes for ComfyUI

Various quality of life -nodes made by combining functionality of existing nodes for ComfyUI, mostly just visual stuff to improve usability.

## Javascript

### browserstatus.js
Sets the favicon to green circle when not processing anything, sets it to red when processing and shows progress percentage and the lenghth of your queue. Might clash with other scripts that affect the page title, delete this file to disable (until I figure out how to add options).

## Nodes:

### Set/Get

Javascript nodes to set and get constants to reduce unnecessary lines. Takes in and returns anything, purely visual nodes.
Could still be buggy, especially when loading workflow with missing nodes, use with precaution.

### ColorToMask

RBG color value to mask, works with batches and AnimateDiff

### ConditioningMultiCombine

Combine any number of conditions, saves space

### ConditioningSetMaskAndCombine

Mask and combine two sets of conditions, saves space.

### GrowMaskWithBlur

Grows or shrinks (with negative values) mask, option to invert input, returns mask and inverted mask. Additionally Blurs the mask, this is a slow operation especially with big batches.

# Installation
1. Clone this repo into `custom_nodes` folder.
