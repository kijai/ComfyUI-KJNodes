import nodes
import torch
import torch.nn.functional as F
import scipy.ndimage
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont
import os
import librosa

from nodes import MAX_RESOLUTION

script_dir = os.path.dirname(os.path.abspath(__file__))
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

class CreateAudioMask:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "createaudiomask"
    CATEGORY = "KJNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "scale": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 2.0, "step": 0.01}),
                 "audio_path": ("STRING", {"default": "audio.wav"}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
        },
    } 

    def createaudiomask(self, frames, width, height, invert, audio_path, scale):
             # Define the number of images in the batch
        batch_size = frames
        out = []
        masks = []
        if audio_path == "audio.wav": #I don't know why relative path won't work otherwise...
            audio_path = os.path.join(script_dir, audio_path)
        audio, sr = librosa.load(audio_path)
        spectrogram = np.abs(librosa.stft(audio))
        #normalized_spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
        
        # Generate the text
        for i in range(batch_size):
           image = Image.new("RGB", (width, height), "black")
           draw = ImageDraw.Draw(image)
           frame = spectrogram[:, i]
           circle_radius = int(height * np.mean(frame))
           circle_radius *= scale
           circle_center = (width // 2, height // 2)  # Calculate the center of the image


           draw.ellipse([(circle_center[0] - circle_radius, circle_center[1] - circle_radius),
                      (circle_center[0] + circle_radius, circle_center[1] + circle_radius)],
                      fill='white')
          
           
           image = np.array(image).astype(np.float32) / 255.0
           image = torch.from_numpy(image)[None,]
           mask = image[:, :, :, 0] 
           masks.append(mask)
           out.append(image)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),torch.cat(masks, dim=0),)
       

    
class CreateGradientMask:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "createmask"
    CATEGORY = "KJNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
        },
    } 

    def createmask(self, frames, width, height, invert):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        # Create an empty array to store the image batch
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        # Generate the black to white gradient for each image
        for i in range(batch_size):
            gradient = np.linspace(1.0, 0.0, width, dtype=np.float32)
            time = i / frames  # Calculate the time variable
            offset_gradient = gradient - time  # Offset the gradient values based on time
            image_batch[i] = offset_gradient.reshape(1, -1)
        output = torch.from_numpy(image_batch)
        mask = output
        print("gradientmaskshape")
        print(mask.shape)
        out.append(mask)
        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),)

class CreateTextMask:
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "createtextmask"
    CATEGORY = "KJNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                 "text_x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "text_y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "font_size": ("INT", {"default": 32,"min": 8, "max": 4096, "step": 1}),
                 "text": ("STRING", {"default": "HELLO!"}),
                 "font_path": ("STRING", {"default": "fonts\\TTNorms-Black.otf"}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "start_rotation": ("INT", {"default": 0,"min": 0, "max": 359, "step": 1}),
                 "end_rotation": ("INT", {"default": 359,"min": -359, "max": 359, "step": 1}),
        },
    } 

    def createtextmask(self, frames, width, height, invert, text_x, text_y, text, font_size, font_path, start_rotation, end_rotation):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        masks = []
        rotation = start_rotation
        if frames > 1:
            rotation_increment = (end_rotation - start_rotation) / (batch_size - 1)
        if font_path == "fonts\\TTNorms-Black.otf": #I don't know why relative path won't work otherwise...
            font_path = os.path.join(script_dir, font_path)
        # Generate the text
        for i in range(batch_size):
           image = Image.new("RGB", (width, height), "black")
           draw = ImageDraw.Draw(image)
           font = ImageFont.truetype(font_path, font_size)
           text_width, text_height = draw.textsize(text, font=font)
           text_center_x = text_x + text_width / 2
           text_center_y = text_y + text_height / 2
           draw.text((text_x, text_y), text, font=font, fill="white")
           image = image.rotate(rotation, center=(text_center_x, text_center_y))
           image = np.array(image).astype(np.float32) / 255.0
           image = torch.from_numpy(image)[None,]
           mask = image[:, :, :, 0] 
           masks.append(mask)
           out.append(image)
           rotation += rotation_increment
        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),torch.cat(masks, dim=0),)
    
class GrowMaskWithBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("INT", {
                    "default": 0,
                    "min": 0,
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
    
    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, sigma, incremental_expandrate):
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
            if expand < 0:
                expand -= abs(incremental_expandrate)  # Use abs(growrate) to ensure positive change
            else:
                expand += abs(incremental_expandrate)  # Use abs(growrate) to ensure positive change
            output = torch.from_numpy(output)
            print(output.shape)
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
        return (torch.stack(out, dim=0), 1.0 -torch.stack(out, dim=0),)
           
        
        
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

class ColorToMask:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "clip"
    CATEGORY = "KJNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "invert": ("BOOLEAN", {"default": False}),
                 "red": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "green": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "blue": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "threshold": ("INT", {"default": 10,"min": 0, "max": 255, "step": 1}),
        },
    } 

    def clip(self, images, red, green, blue, threshold, invert):
        color = np.array([red, green, blue])
        images = 255. * images.cpu().numpy()
        images = np.clip(images, 0, 255).astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
        images = [np.array(image) for image in images]

        black = [0, 0, 0]
        white = [255, 255, 255]
        if invert:
             black, white = white, black

        new_images = []
        for image in images:
            new_image = np.full_like(image, black)

            color_distances = np.linalg.norm(image - color, axis=-1)
            complement_indexes = color_distances <= threshold

            new_image[complement_indexes] = white

            new_images.append(new_image)

        new_images = np.array(new_images).astype(np.float32) / 255.0
        new_images = torch.from_numpy(new_images).permute(3, 0, 1, 2)
        return new_images
      
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
    "ColorToMask": ColorToMask,
    "CreateGradientMask": CreateGradientMask,
    "CreateTextMask": CreateTextMask,
    "CreateAudioMask": CreateAudioMask
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INTConstant": "INT Constant",
    "ConditioningMultiCombine": "Conditioning Multi Combine",
    "ConditioningSetMaskAndCombine": "ConditioningSetMaskAndCombine",
    "GrowMaskWithBlur": "GrowMaskWithBlur",
    "ColorToMask": "ColorToMask",
    "CreateGradientMask": "CreateGradientMask",
    "CreateTextMask" : "CreateTextMask",
    "CreateAudioMask": "CreateAudioMask"
}