import numpy as np
import torch
import random
import math
import os
import json

from PIL import Image

from comfy.utils import common_upscale
from comfy_api.latest import io
import folder_paths
from nodes import MAX_RESOLUTION

from ..utility.utility import string_to_color

def _upscale_mask(mask, width, height, method, crop):
    if method == "lanczos":
        return common_upscale(mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, method, crop).movedim(1, -1)[:, :, :, 0]
    return common_upscale(mask.unsqueeze(1), width, height, method, crop).squeeze(1)


def _resize_single_channel(tensor, width, height):
    """Resize a 3D (B,H,W) tensor using bilinear interpolation."""
    return common_upscale(tensor.unsqueeze(1), width, height, "bilinear", "disabled").squeeze(1)


def _pad_inputs():
    """Shared pad_top/bottom/left/right input definitions for extra_padding options."""
    return [
        io.Int.Input("pad_top", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Padding pixels on top."),
        io.Int.Input("pad_bottom", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Padding pixels on bottom."),
        io.Int.Input("pad_left", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Padding pixels on left."),
        io.Int.Input("pad_right", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Padding pixels on right."),
    ]


def _apply_padding(tensor, pad_top, pad_bottom, pad_left, pad_right, mode, edge_mode="clamp", fill_rgb=None):
    """Apply padding to a BHWC tensor. Returns the padded tensor.
    mode: 'color' or 'edge'
    edge_mode: 'clamp', 'repeat', 'mirror' (only used when mode='edge')
    fill_rgb: list of [r, g, b] float values 0-1 (only used when mode='color')
    """
    h, w = tensor.shape[1], tensor.shape[2]
    new_h = h + pad_top + pad_bottom
    new_w = w + pad_left + pad_right

    if mode == "color":
        fill = fill_rgb or [0.0, 0.0, 0.0]
        padded = torch.zeros(tensor.shape[0], new_h, new_w, tensor.shape[3], device=tensor.device, dtype=tensor.dtype)
        for c in range(min(3, tensor.shape[3])):
            padded[:, :, :, c] = fill[c]
        padded[:, pad_top:pad_top+h, pad_left:pad_left+w, :] = tensor
        return padded

    # mode == "edge"
    if edge_mode == "clamp":
        padded = torch.zeros(tensor.shape[0], new_h, new_w, tensor.shape[3], device=tensor.device, dtype=tensor.dtype)
        padded[:, pad_top:pad_top+h, pad_left:pad_left+w, :] = tensor
        if pad_top > 0:
            padded[:, :pad_top, pad_left:pad_left+w, :] = tensor[:, 0:1, :, :].expand(-1, pad_top, -1, -1)
        if pad_bottom > 0:
            padded[:, pad_top+h:, pad_left:pad_left+w, :] = tensor[:, -1:, :, :].expand(-1, pad_bottom, -1, -1)
        if pad_left > 0:
            padded[:, :, :pad_left, :] = padded[:, :, pad_left:pad_left+1, :].expand(-1, -1, pad_left, -1)
        if pad_right > 0:
            padded[:, :, pad_left+w:, :] = padded[:, :, pad_left+w-1:pad_left+w, :].expand(-1, -1, pad_right, -1)
        return padded
    elif edge_mode == "repeat":
        tiles_x = (new_w + w - 1) // w + 1
        tiles_y = (new_h + h - 1) // h + 1
        tiled = tensor.repeat(1, tiles_y, tiles_x, 1)
        # Offset so original content lands at (pad_top, pad_left) in output
        off_x = (w - pad_left % w) % w
        off_y = (h - pad_top % h) % h
        return tiled[:, off_y:off_y+new_h, off_x:off_x+new_w, :]
    elif edge_mode == "mirror":
        flipped_h = tensor.flip(2)
        flipped_v = tensor.flip(1)
        flipped_hv = tensor.flip(1).flip(2)
        mirror_block = torch.cat([
            torch.cat([tensor, flipped_h], dim=2),
            torch.cat([flipped_v, flipped_hv], dim=2),
        ], dim=1)
        mb_h, mb_w = mirror_block.shape[1], mirror_block.shape[2]
        tiles_x = (new_w + mb_w - 1) // mb_w + 1
        tiles_y = (new_h + mb_h - 1) // mb_h + 1
        tiled = mirror_block.repeat(1, tiles_y, tiles_x, 1)
        # Offset so original content lands at (pad_top, pad_left) in output
        off_x = (mb_w - pad_left % mb_w) % mb_w
        off_y = (mb_h - pad_top % mb_h) % mb_h
        return tiled[:, off_y:off_y+new_h, off_x:off_x+new_w, :]
    return tensor


class ImageTransformKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageTransformKJ",
            display_name="Image Transform KJ",
            category="KJNodes/image",
            search_aliases=["resize", "crop", "pad", "upscale", "keep proportion", "bbox", "bounding box", "transform", "rotate", "mirror"],
            is_experimental=True,
            description="""
Interactive image transform node: crop, resize, pad, and rotate.  
Connect an image input — the preview appears automatically.  

Cropping:  
Click + drag to draw a crop region.  
Drag inside to move, drag edges/corners to resize.  
Right-click to delete a region.  
Ctrl to snap to grid.  
Shift + resize to constrain aspect ratio.  
Alt + resize to resize symmetrically.  

Padding:  
Shift + drag to adjust padding position.

Rotate button enables rotation cross (drag to rotate, right-click to reset).  
Set target_width/height to resize output (0 = keep original).  
Use keep_proportion to control how the image fits the target.  
Use extra_padding to add padding with color or edge fill (clamp/repeat/mirror).""",
            inputs=[
                io.MatchType.Input("image", io.MatchType.Template("img_or_mask", [io.Image, io.Mask]), tooltip="The image or mask to transform."),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask to transform alongside the image."),
                io.Int.Input("target_width", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Target output width. 0 = keep original dimensions."),
                io.Int.Input("target_height", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Target output height. 0 = keep original dimensions."),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos", tooltip="Interpolation method for resizing."),
                io.DynamicCombo.Input("keep_proportion", options=[
                    io.DynamicCombo.Option(key="keep_long_edge", inputs=[]),
                    io.DynamicCombo.Option(key="keep_short_edge", inputs=[]),
                    io.DynamicCombo.Option(key="total_pixels", inputs=[]),
                    io.DynamicCombo.Option(key="stretch", inputs=[]),
                    io.DynamicCombo.Option(key="crop", inputs=[]),
                    io.DynamicCombo.Option(key="pad_color", inputs=[
                        io.Float.Input("pad_x", default=0.5, min=0.0, max=1.0, step=0.01,
                            tooltip="Horizontal position of content within padding (0=left, 0.5=center, 1=right). Shift+drag content in preview to adjust."),
                        io.Float.Input("pad_y", default=0.5, min=0.0, max=1.0, step=0.01,
                            tooltip="Vertical position of content within padding (0=top, 0.5=center, 1=bottom). Shift+drag content in preview to adjust."),
                    ]),
                    io.DynamicCombo.Option(key="pad_edge", inputs=[
                        io.Combo.Input("edge_mode", options=["clamp", "repeat", "mirror"], default="clamp",
                            tooltip="clamp: extend edge pixels. repeat: tile the image. mirror: tile with mirroring."),
                        io.Float.Input("pad_x", default=0.5, min=0.0, max=1.0, step=0.01,
                            tooltip="Horizontal position of content within padding (0=left, 0.5=center, 1=right). Shift+drag content in preview to adjust."),
                        io.Float.Input("pad_y", default=0.5, min=0.0, max=1.0, step=0.01,
                            tooltip="Vertical position of content within padding (0=top, 0.5=center, 1=bottom). Shift+drag content in preview to adjust."),
                    ]),
                    io.DynamicCombo.Option(key="multiplier", inputs=[
                        io.Float.Input("width_mult", default=1.0, min=0.01, max=16.0, step=0.05,
                            tooltip="Multiply the crop width by this factor."),
                        io.Float.Input("height_mult", default=1.0, min=0.01, max=16.0, step=0.05,
                            tooltip="Multiply the crop height by this factor."),
                    ]),
                ]),
                io.Int.Input("divisible_by", default=2, min=0, max=512, step=1),
                io.DynamicCombo.Input("extra_padding", options=[
                    io.DynamicCombo.Option(key="disabled", inputs=[]),
                    io.DynamicCombo.Option(key="pad_color", inputs=_pad_inputs()),
                    io.DynamicCombo.Option(key="pad_edge", inputs=_pad_inputs() + [
                        io.Combo.Input("edge_mode", options=["clamp", "repeat", "mirror"], default="clamp",
                            tooltip="clamp: extend edge pixels. repeat: tile the image. mirror: tile with mirroring."),
                    ]),
                    io.DynamicCombo.Option(key="pad_crop_color", inputs=_pad_inputs()),
                    io.DynamicCombo.Option(key="pad_crop_edge", inputs=_pad_inputs() + [
                        io.Combo.Input("edge_mode", options=["clamp", "repeat", "mirror"], default="clamp",
                            tooltip="clamp: extend edge pixels. repeat: tile the image. mirror: tile with mirroring."),
                    ]),
                ]),
                io.DynamicCombo.Input("invert_crop", options=[
                    io.DynamicCombo.Option(key="disabled", inputs=[]),
                    io.DynamicCombo.Option(key="enabled", inputs=[]),
                ]),
                io.String.Input("bboxes", default="", socketless=True, advanced=True),
            ],
            outputs=[
                io.MatchType.Output(io.MatchType.Template("img_or_mask", [io.Image, io.Mask]), id="cropped", display_name="output", is_output_list=True),
                io.Mask.Output("cropped_mask", display_name="output_mask", is_output_list=True),
                io.BBOX.Output("bbox", display_name="bbox", is_output_list=True),
                io.Mask.Output("bbox_mask", display_name="bbox_mask", is_output_list=True),
                io.Int.Output("width", display_name="width", tooltip="Width of the output image."),
                io.Int.Output("height", display_name="height", tooltip="Height of the output image."),
            ],
        )


    @classmethod
    def execute(cls, image, target_width, target_height, upscale_method, keep_proportion, divisible_by,
                extra_padding, invert_crop, bboxes, mask=None):
        # Unpack DynamicCombos
        edge_mode = keep_proportion.get("edge_mode", "clamp")
        pad_x = keep_proportion.get("pad_x", 0.5)
        pad_y = keep_proportion.get("pad_y", 0.5)
        width_mult = keep_proportion.get("width_mult", 1.0)
        height_mult = keep_proportion.get("height_mult", 1.0)
        keep_proportion = keep_proportion["keep_proportion"]
        extra_top = extra_padding.get("pad_top", 0)
        extra_bottom = extra_padding.get("pad_bottom", 0)
        extra_left = extra_padding.get("pad_left", 0)
        extra_right = extra_padding.get("pad_right", 0)
        extra_edge_mode = extra_padding.get("edge_mode", "clamp")
        extra_pad_mode = extra_padding.get("extra_padding", "disabled")
        invert_crop = invert_crop["invert_crop"]


        # Parse fill color from bboxes JSON (shared color picker)
        fill_color_rgb = [0, 0, 0]
        if bboxes:
            try:
                _parsed_tmp = json.loads(bboxes)
                if isinstance(_parsed_tmp, dict) and "fillColor" in _parsed_tmp:
                    fill_color_rgb = string_to_color(_parsed_tmp["fillColor"])
            except (json.JSONDecodeError, Exception):
                pass
        fill_rgb = [c / 255.0 for c in fill_color_rgb[:3]]

        # Handle mask input (3D) by converting to image-like 4D tensor
        input_is_mask = image.ndim == 3
        if input_is_mask:
            image = image.unsqueeze(-1).repeat(1, 1, 1, 3)

        # Save input image as temp preview file for JS canvas
        temp_dir = folder_paths.get_temp_directory()
        pil_img = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        preview_filename = f"crop_preview_{random.randint(0, 0xFFFFFF):06x}.webp"
        pil_img.save(os.path.join(temp_dir, preview_filename), format="WEBP", quality=80)
        preview_ui = {"preview_filename": [preview_filename]}

        img_height = image.shape[1]
        img_width = image.shape[2]

        # Parse bboxes and rotation
        bbox_list = []
        rotation = 0.0
        if bboxes:
            try:
                parsed = json.loads(bboxes)
                # New format: { bboxes: [...], rotation: N }
                if isinstance(parsed, dict):
                    bbox_list = [b for b in parsed.get("bboxes", []) if b and all(k in b for k in ("startX", "startY", "endX", "endY"))]
                    rotation = parsed.get("rotation", 0.0)
                # Legacy format: [bbox, bbox, ...]
                elif isinstance(parsed, list):
                    bbox_list = [b for b in parsed if b and all(k in b for k in ("startX", "startY", "endX", "endY"))]
            except json.JSONDecodeError:
                pass

        # Content mask tracks which pixels are actual image content (1=content, 0=fill)
        content_mask = torch.ones(1, img_height, img_width, device=image.device)

        # Apply rotation before cropping
        if rotation != 0:
            from torchvision.transforms.functional import rotate as tv_rotate
            import torch.nn.functional as F
            # Use shared fill color for rotation corners (unless edge mode)
            rot_fill = fill_rgb

            is_edge_mode = extra_pad_mode in ("pad_edge", "pad_crop_edge") or keep_proportion == "pad_edge"
            if is_edge_mode:
                h, w = image.shape[1], image.shape[2]
                pad_amt = max(h, w)
                img_chw = image.movedim(-1, 1)
                img_padded = F.pad(img_chw, [pad_amt, pad_amt, pad_amt, pad_amt], mode='replicate')
                img_rotated = tv_rotate(img_padded, -rotation, expand=False, fill=rot_fill)
                ch, cw = img_rotated.shape[2], img_rotated.shape[3]
                cy, cx = ch // 2, cw // 2
                image = img_rotated[:, :, cy - h // 2:cy - h // 2 + h, cx - w // 2:cx - w // 2 + w].movedim(1, -1)
                if mask is not None:
                    mask_padded = F.pad(mask.unsqueeze(1), [pad_amt, pad_amt, pad_amt, pad_amt], mode='replicate')
                    mask_rotated = tv_rotate(mask_padded, -rotation, expand=False, fill=[0.0])
                    mask = mask_rotated[:, :, cy - h // 2:cy - h // 2 + h, cx - w // 2:cx - w // 2 + w].squeeze(1)
                # Content mask: rotate the same way (no padding — just rotate and crop)
                cm_padded = F.pad(content_mask.unsqueeze(1), [pad_amt, pad_amt, pad_amt, pad_amt], mode='constant', value=0)
                cm_rotated = tv_rotate(cm_padded, -rotation, expand=False, fill=[0.0])
                content_mask = cm_rotated[:, :, cy - h // 2:cy - h // 2 + h, cx - w // 2:cx - w // 2 + w].squeeze(1)
            else:
                image = tv_rotate(image.movedim(-1, 1), -rotation, expand=True, fill=rot_fill).movedim(1, -1)
                if mask is not None:
                    mask = tv_rotate(mask.unsqueeze(1), -rotation, expand=True, fill=[0.0]).squeeze(1)
                # Content mask: rotate with expand, fill=0
                content_mask = tv_rotate(content_mask.unsqueeze(1), -rotation, expand=True, fill=[0.0]).squeeze(1)
            img_height = image.shape[1]
            img_width = image.shape[2]

        # Normalize mask dimensions to match image
        if mask is not None:
            if mask.shape[-2] != img_height or mask.shape[-1] != img_width:
                if mask.shape[-2] == img_width and mask.shape[-1] == img_height:
                    mask = mask.transpose(-2, -1)
                else:
                    mask = _resize_single_channel(mask, img_width, img_height)

        # "Pad first" modes: apply extra padding to the full image before cropping
        # Skip for keep_proportion pad modes — those handle extra padding via target subtraction
        is_pad_first = extra_pad_mode in ("pad_color", "pad_edge")
        kp_is_pad_mode = keep_proportion in ("pad_color", "pad_edge")
        if is_pad_first and not kp_is_pad_mode and (extra_top > 0 or extra_bottom > 0 or extra_left > 0 or extra_right > 0):
            pad_mode = "color" if extra_pad_mode == "pad_color" else "edge"
            padded_img = _apply_padding(image, extra_top, extra_bottom, extra_left, extra_right, pad_mode, extra_edge_mode, fill_rgb)

            image = padded_img
            img_height = image.shape[1]
            img_width = image.shape[2]
            # Expand content mask and user mask
            cm_new = torch.zeros(1, img_height, img_width, device=content_mask.device)
            cm_new[:, extra_top:extra_top+content_mask.shape[1], extra_left:extra_left+content_mask.shape[2]] = content_mask
            content_mask = cm_new
            if mask is not None:
                m_new = torch.zeros(mask.shape[0], img_height, img_width, device=mask.device, dtype=mask.dtype)
                m_new[:, extra_top:extra_top+mask.shape[1], extra_left:extra_left+mask.shape[2]] = mask
                mask = m_new

        # If no bboxes, treat the full image as a single bbox
        if not bbox_list:
            bbox_list = [None]

        all_cropped = []
        all_cropped_masks = []
        all_bbox_tuples = []
        all_bbox_masks = []

        for bbox_data in bbox_list:
            has_bbox = bbox_data is not None

            if has_bbox:
                preview_width = bbox_data.get("previewWidth", 0)
                preview_height = bbox_data.get("previewHeight", 0)
                sx = img_width / preview_width if preview_width > 0 else 1.0
                sy = img_height / preview_height if preview_height > 0 else 1.0

                x_min = int(min(bbox_data["startX"], bbox_data["endX"]) * sx)
                y_min = int(min(bbox_data["startY"], bbox_data["endY"]) * sy)
                x_max = int(max(bbox_data["startX"], bbox_data["endX"]) * sx)
                y_max = int(max(bbox_data["startY"], bbox_data["endY"]) * sy)

                x_min = max(0, min(x_min, img_width - 1))
                y_min = max(0, min(y_min, img_height - 1))
                x_max = max(x_min + 1, min(x_max, img_width))
                y_max = max(y_min + 1, min(y_max, img_height))

                cropped = image[:, y_min:y_max, x_min:x_max, :]
                cropped_content_mask = content_mask[:, y_min:y_max, x_min:x_max]
                all_bbox_tuples.append((x_min, y_min, x_max - x_min, y_max - y_min))
                bm = torch.zeros(1, img_height, img_width)
                bm[0, y_min:y_max, x_min:x_max] = 1.0
                all_bbox_masks.append(bm)
                cropped_mask = mask[:, y_min:y_max, x_min:x_max] if mask is not None else None
            else:
                cropped = image
                cropped_content_mask = content_mask
                all_bbox_tuples.append((0, 0, img_width, img_height))
                all_bbox_masks.append(torch.ones(1, img_height, img_width))
                cropped_mask = mask
                x_min, y_min, x_max, y_max = 0, 0, img_width, img_height

            # Multiplier mode: compute target from crop dims * multiplier
            if keep_proportion == "multiplier":
                crop_h, crop_w = cropped.shape[1], cropped.shape[2]
                tw = round(crop_w * width_mult)
                th = round(crop_h * height_mult)
                target_width = tw
                target_height = th

            # Resize cropped image if target dimensions are set
            if target_width > 0 or target_height > 0:
                crop_h, crop_w = cropped.shape[1], cropped.shape[2]
                tw = target_width if target_width > 0 else crop_w
                th = target_height if target_height > 0 else crop_h

                # Subtract extra padding from target so content + padding = original target
                # For pad-first + non-pad keep_proportion, padding is on the source (don't subtract)
                # For pad modes or pad-crop, subtract so padding is in the output
                has_extra = extra_top > 0 or extra_bottom > 0 or extra_left > 0 or extra_right > 0
                kp_is_pad = keep_proportion in ("pad_color", "pad_edge")
                if has_extra and (kp_is_pad or not is_pad_first):
                    if target_width > 0:
                        tw = max(1, tw - extra_left - extra_right)
                    if target_height > 0:
                        th = max(1, th - extra_top - extra_bottom)

                if keep_proportion == "keep_long_edge":
                    ratio = min(tw / crop_w, th / crop_h)
                    tw = round(crop_w * ratio)
                    th = round(crop_h * ratio)
                elif keep_proportion == "keep_short_edge":
                    ratio = max(tw / crop_w, th / crop_h)
                    tw = round(crop_w * ratio)
                    th = round(crop_h * ratio)
                elif keep_proportion == "total_pixels":
                    total_pixels = tw * th
                    aspect_ratio = crop_w / crop_h
                    th = int(math.sqrt(total_pixels / aspect_ratio))
                    tw = int(math.sqrt(total_pixels * aspect_ratio))
                elif keep_proportion == "crop":
                    ratio = max(tw / crop_w, th / crop_h)
                    scale_w = round(crop_w * ratio)
                    scale_h = round(crop_h * ratio)
                    samples = common_upscale(cropped.movedim(-1, 1), scale_w, scale_h, upscale_method, "center")
                    cropped = samples.movedim(1, -1)
                    if cropped_mask is not None:
                        cropped_mask = _upscale_mask(cropped_mask, scale_w, scale_h, upscale_method, "center")
                    cropped_content_mask = _resize_single_channel(cropped_content_mask, scale_w, scale_h)
                    cx = (scale_w - tw) // 2
                    cy = (scale_h - th) // 2
                    cropped = cropped[:, cy:cy+th, cx:cx+tw, :]
                    if cropped_mask is not None:
                        cropped_mask = cropped_mask[:, cy:cy+th, cx:cx+tw]
                    cropped_content_mask = cropped_content_mask[:, cy:cy+th, cx:cx+tw]
                elif keep_proportion in ("pad_color", "pad_edge"):
                    ratio = min(tw / crop_w, th / crop_h)
                    scale_w = round(crop_w * ratio)
                    scale_h = round(crop_h * ratio)
                    samples = common_upscale(cropped.movedim(-1, 1), scale_w, scale_h, upscale_method, "disabled")
                    resized = samples.movedim(1, -1)
                    # pad_x/pad_y position across full target (not just content area)
                    full_tw = target_width if target_width > 0 else crop_w
                    full_th = target_height if target_height > 0 else crop_h
                    pad_left = round((full_tw - scale_w) * pad_x)
                    pad_top = round((full_th - scale_h) * pad_y)
                    pad_right = full_tw - pad_left - scale_w
                    pad_bottom = full_th - pad_top - scale_h
                    tw = full_tw
                    th = full_th

                    pad_mode = "edge" if keep_proportion == "pad_edge" else "color"
                    cropped = _apply_padding(resized, pad_top, pad_bottom, pad_left, pad_right, pad_mode, edge_mode, fill_rgb)
                    if cropped_mask is not None:
                        mask_resized = _upscale_mask(cropped_mask, scale_w, scale_h, upscale_method, "disabled")
                        mask_padded = torch.zeros(mask_resized.shape[0], th, tw, device=mask_resized.device, dtype=mask_resized.dtype)
                        mask_padded[:, pad_top:pad_top+scale_h, pad_left:pad_left+scale_w] = mask_resized
                        cropped_mask = mask_padded
                    # Update content mask for padding area
                    cm_resized = _resize_single_channel(cropped_content_mask, scale_w, scale_h)
                    cm_padded = torch.zeros(1, th, tw, device=cropped_content_mask.device)
                    cm_padded[:, pad_top:pad_top+scale_h, pad_left:pad_left+scale_w] = cm_resized
                    cropped_content_mask = cm_padded

                if divisible_by > 1:
                    tw = tw - (tw % divisible_by)
                    th = th - (th % divisible_by)

                if tw > 0 and th > 0:
                    if keep_proportion in ("stretch", "keep_long_edge", "keep_short_edge", "total_pixels", "multiplier"):
                        cropped = common_upscale(cropped.movedim(-1, 1), tw, th, upscale_method, "disabled").movedim(1, -1)
                        if cropped_mask is not None:
                            cropped_mask = _upscale_mask(cropped_mask, tw, th, upscale_method, "disabled")
                        cropped_content_mask = _resize_single_channel(cropped_content_mask, tw, th)
                    else:
                        cropped = cropped[:, :th, :tw, :]
                        if cropped_mask is not None:
                            cropped_mask = cropped_mask[:, :th, :tw]
                        cropped_content_mask = cropped_content_mask[:, :th, :tw]

            # Enforce divisible_by even when no target dimensions are set
            elif divisible_by > 1:
                final_w = cropped.shape[2] - (cropped.shape[2] % divisible_by)
                final_h = cropped.shape[1] - (cropped.shape[1] % divisible_by)
                if final_w != cropped.shape[2] or final_h != cropped.shape[1]:
                    cropped = cropped[:, :final_h, :final_w, :]
                    if cropped_mask is not None:
                        cropped_mask = cropped_mask[:, :final_h, :final_w]
                    cropped_content_mask = cropped_content_mask[:, :final_h, :final_w]

            # Apply extra padding (skip for pad-first and keep_proportion pad modes which handle it above)
            kp_handles_ep = keep_proportion in ("pad_color", "pad_edge")
            if not is_pad_first and not kp_handles_ep and (extra_top > 0 or extra_bottom > 0 or extra_left > 0 or extra_right > 0):
                h_cur, w_cur = cropped.shape[1], cropped.shape[2]
                pad_mode = "edge" if extra_pad_mode == "pad_crop_edge" else "color"
                cropped = _apply_padding(cropped, extra_top, extra_bottom, extra_left, extra_right, pad_mode, extra_edge_mode, fill_rgb)
                new_h, new_w = cropped.shape[1], cropped.shape[2]
                if cropped_mask is not None:
                    padded_mask = torch.zeros(cropped_mask.shape[0], new_h, new_w, device=cropped_mask.device, dtype=cropped_mask.dtype)
                    padded_mask[:, extra_top:extra_top+h_cur, extra_left:extra_left+w_cur] = cropped_mask
                    cropped_mask = padded_mask
                cm_h, cm_w = cropped_content_mask.shape[-2], cropped_content_mask.shape[-1]
                if cm_h != h_cur or cm_w != w_cur:
                    cropped_content_mask = _resize_single_channel(cropped_content_mask, w_cur, h_cur)
                cm_ep = torch.zeros(1, new_h, new_w, device=cropped_content_mask.device)
                cm_ep[:, extra_top:extra_top+h_cur, extra_left:extra_left+w_cur] = cropped_content_mask
                cropped_content_mask = cm_ep

            # If no mask was provided, output a zeros mask matching the cropped image
            if cropped_mask is None:
                cropped_mask = torch.zeros(1, cropped.shape[1], cropped.shape[2])

            # Apply fill mask — marks filled/padded areas as 1 in the output mask
            # Combines with incoming mask: 1 where either input mask is 1 OR area is filled
            if cropped_content_mask is not None:
                out_h, out_w = cropped_mask.shape[1], cropped_mask.shape[2]
                cm_h, cm_w = cropped_content_mask.shape[1], cropped_content_mask.shape[2]
                if cm_h != out_h or cm_w != out_w:
                    cropped_content_mask = _resize_single_channel(cropped_content_mask, out_w, out_h)
                # fill_mask: 1 where filled, 0 where content
                fill_mask = 1.0 - cropped_content_mask.clamp(0, 1)
                # Combine: output mask is max of incoming mask and fill mask
                cropped_mask = torch.max(cropped_mask, fill_mask)

            # Invert crop: output area outside the bbox instead of inside
            if invert_crop == "enabled" and has_bbox:
                inverted = image.clone()
                for c in range(min(3, inverted.shape[3])):
                    inverted[:, y_min:y_max, x_min:x_max, c] = fill_rgb[c]
                cropped = inverted

            # Convert back to mask if input was a mask
            if input_is_mask:
                cropped = cropped[:, :, :, 0]

            all_cropped.append(cropped)
            all_cropped_masks.append(cropped_mask)

        width, height = all_cropped[0].shape[2], all_cropped[0].shape[1]

        return io.NodeOutput(all_cropped, all_cropped_masks, all_bbox_tuples, all_bbox_masks, width, height, ui=preview_ui)
