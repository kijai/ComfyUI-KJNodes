import nodes
import torch
import torch.nn.functional as F
import scipy.ndimage
import numpy as np

from nodes import MAX_RESOLUTION

class INTConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"

    CATEGORY = "KJNodes"

    def get_value(self, value):
        return (value,)
    
def gaussian_kernel(kernel_size: int, sigma: float, device=None):
        x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size, device=device), torch.linspace(-1, 1, kernel_size, device=device), indexing="ij")
        d = torch.sqrt(x * x + y * y)
        g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
        return g / g.sum()

class GrowMaskWithBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }
    
    CATEGORY = "KJNodes"

    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"
    
    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, sigma):
        if( flip_input ):
            mask = 1.0 - mask
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in growmask:
            output = m.numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            output = torch.from_numpy(output)
            out.append(output)
        
        blurred = torch.stack(out, dim=0).reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        batch_size, height, width, channels = blurred.shape
        if blur_radius != 0:
            blurkernel_size = blur_radius * 2 + 1
            blurkernel = gaussian_kernel(blurkernel_size, sigma, device=blurred.device).repeat(channels, 1, 1).unsqueeze(1)
            blurred = blurred.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
            padded_image = F.pad(blurred, (blur_radius,blur_radius,blur_radius,blur_radius), 'reflect')
            blurred = F.conv2d(padded_image, blurkernel, padding=blurkernel_size // 2, groups=channels)[:,:,blur_radius:-blur_radius, blur_radius:-blur_radius]
            blurred = blurred.permute(0, 2, 3, 1)
            blurred = blurred[:, :, :, 0]        
        
        return (blurred, 1.0 - blurred,)
           
        
        
class PlotNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "start": ("FLOAT", {"default": 0.5, "min": 0.5, "max": 1.0}),
            "max_frames": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("FLOAT", "INT",)
    FUNCTION = "plot"
    CATEGORY = "KJNodes"

    def plot(self, start, max_frames):
        result = start + max_frames
        return (result,)    
    
class ConditioningMultiCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 20, "step": 1}),
                "conditioning_1": ("CONDITIONING", ),
                "conditioning_2": ("CONDITIONING", ),
            },
        
    }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("combined", "inputcount")
    FUNCTION = "combine"
    CATEGORY = "KJNodes"

    def combine(self, inputcount, **kwargs):
        cond_combine_node = nodes.ConditioningCombine()
        cond = kwargs["conditioning_1"]
        for c in range(1, inputcount):
            new_cond = kwargs[f"conditioning_{c + 1}"]
            cond = cond_combine_node.combine(new_cond, cond)[0]
        return (cond, inputcount,)
  
class ConditioningSetMaskAndCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", ),
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes"

    def append(self, positive_1, negative_1, positive_2, negative_2, mask_1, mask_2, set_cond_area, strength):
        c = []
        c2 = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask_1.shape) < 3:
            mask_1 = mask_1.unsqueeze(0)
        if len(mask_2.shape) < 3:
            mask_2 = mask_2.unsqueeze(0)
        for t in positive_1:
            n = [t[0], t[1].copy()]
            _, h, w = mask_1.shape
            n[1]['mask'] = mask_1
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c.append(n)
        for t in positive_2:
            n = [t[0], t[1].copy()]
            _, h, w = mask_2.shape
            n[1]['mask'] = mask_2
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c.append(n)
        for t in negative_1:
            n = [t[0], t[1].copy()]
            _, h, w = mask_1.shape
            n[1]['mask'] = mask_1
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c2.append(n)
        for t in negative_2:
            n = [t[0], t[1].copy()]
            _, h, w = mask_2.shape
            n[1]['mask'] = mask_2
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c2.append(n)
        return (c, c2)
    

  
NODE_CLASS_MAPPINGS = {
    "INTConstant": INTConstant,
    "ConditioningMultiCombine": ConditioningMultiCombine,
    "ConditioningSetMaskAndCombine": ConditioningSetMaskAndCombine,
    "GrowMaskWithBlur": GrowMaskWithBlur,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INTConstant": "INT Constant",
    "ConditioningMultiCombine": "Conditioning Multi Combine",
    "ConditioningSetMaskAndCombine": "ConditioningSetMaskAndCombine",
    "GrowMaskWithBlur": "GrowMaskWithBlur",
}