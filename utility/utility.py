import torch
import numpy as np
from PIL import Image, ImageColor
from typing import Union, List
import logging

# Utility functions from mtb nodes: https://github.com/melMass/comfy_mtb
def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)

    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)


def tensor2np(tensor: torch.Tensor):
    if len(tensor.shape) == 3:  # Single image
        return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    else:  # Batch of images
        return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]

def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def string_to_color(color_string: str) -> List[int]:
    color_list = [0, 0, 0]  # Default fallback (black)

    if ',' in color_string:
        # Handle CSV format (e.g., "255, 0, 0" or "255, 0, 0, 128" or "1.0, 0.5, 0.0")
        try:
            values = [float(channel.strip()) for channel in color_string.split(',')]
            # Convert to 0-255 range if values are in 0-1 range
            if all(0 <= v <= 1 for v in values):
                color_list = [int(v * 255) for v in values]
            else:
                color_list = [int(v) for v in values]
        except ValueError:
            logging.warning(f"Invalid color format: {color_string}. Using default black.")
    elif color_string.lstrip('#').isalnum() and not color_string.lstrip('#').replace('.', '', 1).isdigit():
        # Could be Hex format or color name
        color_string_stripped = color_string.lstrip('#')
        # Try hex first
        if len(color_string_stripped) in [6, 8] and all(c in '0123456789ABCDEFabcdef' for c in color_string_stripped):
            if len(color_string_stripped) == 6:  # #RRGGBB
                color_list = [int(color_string_stripped[i:i+2], 16) for i in (0, 2, 4)]
            elif len(color_string_stripped) == 8:  # #RRGGBBAA
                color_list = [int(color_string_stripped[i:i+2], 16) for i in (0, 2, 4, 6)]
        else:
            # Try color name (e.g., "red", "blue", "cyan")
            try:
                rgb = ImageColor.getrgb(color_string)
                color_list = list(rgb)
            except ValueError:
                logging.warning(f"Invalid color name or hex format: {color_string}. Using default black.")
    else:
        # Handle single value (grayscale) - can be int or float
        try:
            value = float(color_string.strip())
            # Convert to 0-255 range if it's a float between 0-1
            if 0 <= value <= 1:
                value = int(value * 255)
            else:
                value = int(value)
            color_list = [value, value, value]
        except ValueError:
            logging.warning(f"Invalid color format: {color_string}. Using default black.")

    # Clip values to valid range
    color_list = np.clip(color_list, 0, 255).tolist()

    return color_list
