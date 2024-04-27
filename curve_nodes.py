import torch
import json
from PIL import Image, ImageDraw
import numpy as np
from .utility import pil2tensor

class SplineEditor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points_store": ("STRING", {"multiline": False}),
                "coordinates": ("STRING", {"multiline": False}),
                "mask_width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "mask_height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "points_to_sample": ("INT", {"default": 16, "min": 2, "max": 1000, "step": 1}),
                "interpolation": (
                [   
                    'cardinal',
                    'monotone',
                    'basis',
                    'linear',
                    'step-before',
                    'step-after',
                    'polar',
                    'polar-reverse',
                ],
                {
                "default": 'cardinal'
                    }),
                "tension": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_output": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "float_output_type": (
                [   
                    'list',
                    'list of lists',
                    'pandas series',
                    'tensor',
                ],
                {
                    "default": 'list'
                }),
            },
            "optional": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "FLOAT")
    FUNCTION = "splinedata"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
# WORK IN PROGRESS  
Do not count on this as part of your workflow yet,  
probably contains lots of bugs and stability is not  
guaranteed!!  
  
## Graphical editor to create values for various   
## schedules and/or mask batches.  

Shift + click to add control points.  
Right click to delete control points.  
Note that you can't delete from start/end.  

**points_to_sample** value sets the number of samples  
returned from the **drawn spline itself**, this is independent from the  
actual control points, so the interpolation type matters.  

Changing interpolation type and tension value takes effect on  
interaction with the graph.  

output types:
 - mask batch  
        example compatible nodes: anything that takes masks
 - list of floats
        example compatible nodes: IPAdapter weights  
 - list of lists  
        example compatible nodes: unknown
 - pandas series
        example compatible nodes: anything that takes Fizz'  
        nodes Batch Value Schedule  
 - torch tensor  
        example compatible nodes: unknown
"""

    def splinedata(self, mask_width, mask_height, coordinates, float_output_type, interpolation, 
                   points_to_sample, points_store, tension, repeat_output, min_value=0.0, max_value=1.0):
        
        coordinates = json.loads(coordinates)
        for coord in coordinates:
            coord['x'] = int(round(coord['x']))
            coord['y'] = int(round(coord['y']))
        normalized_y_values = [
            (1.0 - (point['y'] / 512) - 0.0) * (max_value - min_value) + min_value
            for point in coordinates
        ]
        if float_output_type == 'list':
            out_floats = normalized_y_values * repeat_output
        elif float_output_type == 'list of lists':
            out_floats = ([[value] for value in normalized_y_values] * repeat_output),
        elif float_output_type == 'pandas series':
            try:
                import pandas as pd
            except:
                raise Exception("MaskOrImageToWeight: pandas is not installed. Please install pandas to use this output_type")
            out_floats = pd.Series(normalized_y_values * repeat_output),
        elif float_output_type == 'tensor':
            out_floats = torch.tensor(normalized_y_values * repeat_output, dtype=torch.float32)
        # Create a color map for grayscale intensities
        color_map = lambda y: torch.full((mask_height, mask_width, 3), y, dtype=torch.float32)

        # Create image tensors for each normalized y value
        mask_tensors = [color_map(y) for y in normalized_y_values]
        masks_out = torch.stack(mask_tensors)
        masks_out = masks_out.repeat(repeat_output, 1, 1, 1)
        masks_out = masks_out.mean(dim=-1)
        return (masks_out, str(coordinates), out_floats,)

class CreateShapeMaskOnPath:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Creates a mask or batch of masks with the specified shape.  
Locations are center locations.  
Grow value is the amount to grow the shape on each frame, creating animated masks.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": (
            [   'circle',
                'square',
                'triangle',
            ],
            {
            "default": 'circle'
             }),
                "coordinates": ("STRING", {"forceInput": True}),
                "grow": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
        },
    } 

    def createshapemask(self, coordinates, frame_width, frame_height, shape_width, shape_height, grow, shape):
        # Define the number of images in the batch
        coordinates = coordinates.replace("'", '"')
        coordinates = json.loads(coordinates)
        for coord in coordinates:
            print(coord)

        batch_size = len(coordinates)
        print(batch_size)
        out = []
        color = "white"
        
        for i, coord in enumerate(coordinates):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0
            current_width = max(0, shape_width + i*grow)
            current_height = max(0, shape_height + i*grow)

            location_x = coord['x']
            location_y = coord['y']

            if shape == 'circle' or shape == 'square':
                # Define the bounding box for the shape
                left_up_point = (location_x - current_width // 2, location_y - current_height // 2)
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2)
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=color)
                    
            elif shape == 'triangle':
                # Define the points for the triangle
                left_up_point = (location_x - current_width // 2, location_y + current_height // 2) # bottom left
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2) # bottom right
                top_point = (location_x, location_y - current_height // 2) # top point
                draw.polygon([top_point, left_up_point, right_down_point], fill=color)

            image = pil2tensor(image)
            mask = image[:, :, :, 0]
            out.append(mask)
        outstack = torch.cat(out, dim=0)
        return (outstack, 1.0 - outstack,)
    
class MaskOrImageToWeight:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_type": (
                [   
                    'list',
                    'list of lists',
                    'pandas series',
                    'tensor',
                ],
                {
                "default": 'list'
                    }),
             },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),                
            },

        }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes"
    DESCRIPTION = """
Gets the mean values from mask or image batch  
and returns that as the selected output type.   
"""

    def execute(self, output_type, images=None, masks=None):
        mean_values = []
        if masks is not None and images is None:
            for mask in masks:
                mean_values.append(mask.mean().item())
        elif masks is None and images is not None:
            for image in images:
                mean_values.append(image.mean().item())
        elif masks is not None and images is not None:
            raise Exception("MaskOrImageToWeight: Use either mask or image input only.")
                  
        # Convert mean_values to the specified output_type
        if output_type == 'list':
            return mean_values,
        elif output_type == 'list of lists':
            return [[value] for value in mean_values],
        elif output_type == 'pandas series':
            try:
                import pandas as pd
            except:
                raise Exception("MaskOrImageToWeight: pandas is not installed. Please install pandas to use this output_type")
            return pd.Series(mean_values),
        elif output_type == 'tensor':
            return torch.tensor(mean_values, dtype=torch.float32),
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

class WeightScheduleConvert:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_values": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "output_type": (
                [   
                    'match_input',
                    'list',
                    'list of lists',
                    'pandas series',
                    'tensor',
                ],
                {
                "default": 'list'
                    }),
                "invert": ("BOOLEAN", {"default": False}),
                "repeat": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
             },
             "optional": {
                "remap_to_frames": ("INT", {"default": 0}),
                "interpolation_curve": ("FLOAT", {"forceInput": True}),
             },
             
        }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes"
    DESCRIPTION = """
Converts different value lists/series to another type.  
"""

    def detect_input_type(self, input_values):
        import pandas as pd
        if isinstance(input_values, list):
            return 'list'
        elif isinstance(input_values, pd.Series):
            return 'pandas series'
        elif isinstance(input_values, torch.Tensor):
            return 'tensor'
        elif isinstance(input_values, list) and all(isinstance(sub, list) for sub in input_values):
            return 'list of lists'
        else:
            raise ValueError("Unsupported input type")

    def execute(self, input_values, output_type, invert, repeat, remap_to_frames=0, interpolation_curve=None):
        import pandas as pd
        input_type = self.detect_input_type(input_values)

        # Convert input_values to a list of floats
        if input_type == 'list of lists':
            float_values = [item for sublist in input_values for item in sublist]
        elif input_type == 'pandas series':
            float_values = input_values.tolist()
        elif input_type == 'tensor':
            float_values = input_values
        else:
            float_values = input_values

        if invert:
            float_values = [1 - value for value in float_values]

        if interpolation_curve is not None:
            interpolated_pattern = []
            orig_float_values = float_values
            for value in interpolation_curve:
                min_val = min(orig_float_values)
                max_val = max(orig_float_values)
                # Normalize the values to [0, 1]
                normalized_values = [(value - min_val) / (max_val - min_val) for value in orig_float_values]
                # Interpolate the normalized values to the new frame count
                remapped_float_values = np.interp(np.linspace(0, 1, int(remap_to_frames * value)), np.linspace(0, 1, len(normalized_values)), normalized_values).tolist()
                interpolated_pattern.extend(remapped_float_values)
            float_values = interpolated_pattern
        else:
            # Remap float_values to match target_frame_amount
            if remap_to_frames > 0 and remap_to_frames != len(float_values):
                min_val = min(float_values)
                max_val = max(float_values)
                # Normalize the values to [0, 1]
                normalized_values = [(value - min_val) / (max_val - min_val) for value in float_values]
                # Interpolate the normalized values to the new frame count
                float_values = np.interp(np.linspace(0, 1, remap_to_frames), np.linspace(0, 1, len(normalized_values)), normalized_values).tolist()
       
            float_values = float_values * repeat

        if output_type == 'list':
            return float_values,
        elif output_type == 'list of lists':
            return [[value] for value in float_values],
        elif output_type == 'pandas series':
            return pd.Series(float_values),
        elif output_type == 'tensor':
            if input_type == 'pandas series':
                return torch.tensor(input_values.values, dtype=torch.float32),
        elif output_type == 'match_input':
            return float_values,
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

class FloatToMask:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_values": ("FLOAT", {"forceInput": True, "default": 0}),
                "width": ("INT", {"default": 100, "min": 1}),
                "height": ("INT", {"default": 100, "min": 1}),
            },
        }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes"
    DESCRIPTION = """
Generates a batch of masks based on the input float values.
The batch size is determined by the length of the input float values.
Each mask is generated with the specified width and height.
"""

    def execute(self, input_values, width, height):
        import pandas as pd
        # Ensure input_values is a list
        if isinstance(input_values, (float, int)):
            input_values = [input_values]
        elif isinstance(input_values, pd.Series):
            input_values = input_values.tolist()
        elif isinstance(input_values, list) and all(isinstance(item, list) for item in input_values):
            input_values = [item for sublist in input_values for item in sublist]

        # Generate a batch of masks based on the input_values
        masks = []
        for value in input_values:
            # Assuming value is a float between 0 and 1 representing the mask's intensity
            mask = torch.ones((height, width), dtype=torch.float32) * value
            masks.append(mask)
        masks_out = torch.stack(masks, dim=0)
    
        return(masks_out,)
class WeightScheduleExtend:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_values_1": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "input_values_2": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "output_type": (
                [   
                    'match_input',
                    'list',
                    'list of lists',
                    'pandas series',
                    'tensor',
                ],
                {
                "default": 'match_input'
                    }),
             },
             
        }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes"
    DESCRIPTION = """
Converts different value lists/series to another type.  
"""

    def detect_input_type(self, input_values):
        import pandas as pd
        if isinstance(input_values, list):
            return 'list'
        elif isinstance(input_values, pd.Series):
            return 'pandas series'
        elif isinstance(input_values, torch.Tensor):
            return 'tensor'
        elif isinstance(input_values, list) and all(isinstance(sub, list) for sub in input_values):
            return 'list of lists'
        else:
            raise ValueError("Unsupported input type")

    def execute(self, input_values_1, input_values_2, output_type):
        import pandas as pd
        input_type_1 = self.detect_input_type(input_values_1)
        input_type_2 = self.detect_input_type(input_values_2)
        # Convert input_values_2 to the same format as input_values_1 if they do not match
        if not input_type_1 == input_type_2:
            print("Converting input_values_2 to the same format as input_values_1")
            if input_type_1 == 'list of lists':
                # Assuming input_values_2 is a flat list, convert it to a list of lists
                float_values_2 = [[item] for item in input_values_2]
            elif input_type_1 == 'pandas series':
                # Convert input_values_2 to a pandas Series
                float_values_2 = pd.Series(input_values_2)
            elif input_type_1 == 'tensor':
                # Convert input_values_2 to a tensor
                float_values_2 = torch.tensor(input_values_2, dtype=torch.float32)
        else:
            print("Input types match, no conversion needed")
            # If the types match, no conversion is needed
            float_values_2 = input_values_2
     
        float_values = input_values_1 + float_values_2
 
        if output_type == 'list':
            return float_values,
        elif output_type == 'list of lists':
            return [[value] for value in float_values],
        elif output_type == 'pandas series':
            return pd.Series(float_values),
        elif output_type == 'tensor':
            if input_type_1 == 'pandas series':
                return torch.tensor(input_values_1.values, dtype=torch.float32),
        elif output_type == 'match_input':
            return float_values,
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")