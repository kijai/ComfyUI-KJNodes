import torch
from torchvision import transforms
import json
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter, ImageChops
import numpy as np
import math
from ..utility.utility import pil2tensor, tensor2pil
import folder_paths
import io
import base64
        
from comfy.utils import common_upscale

def parse_color(color):
    if isinstance(color, str) and ',' in color:
        return tuple(int(c.strip()) for c in color.split(','))
    return color

def parse_json_tracks(tracks):
    tracks_data = []
    try:
        # If tracks is a string, try to parse it as JSON
        if isinstance(tracks, str):
            parsed = json.loads(tracks.replace("'", '"'))
            tracks_data.extend(parsed)
        else:
            # If tracks is a list of strings, parse each one
            for track_str in tracks:
                parsed = json.loads(track_str.replace("'", '"'))
                tracks_data.append(parsed)
        
        # Check if we have a single track (dict with x,y) or a list of tracks
        if tracks_data and isinstance(tracks_data[0], dict) and 'x' in tracks_data[0]:
            # Single track detected, wrap it in a list
            tracks_data = [tracks_data]
        elif tracks_data and isinstance(tracks_data[0], list) and tracks_data[0] and isinstance(tracks_data[0][0], dict) and 'x' in tracks_data[0][0]:
            # Already a list of tracks, nothing to do
            pass
        else:
            # Unexpected format
            print(f"Warning: Unexpected track format: {type(tracks_data[0])}")
            
    except json.JSONDecodeError as e:
        print(f"Error parsing tracks JSON: {e}")
        tracks_data = []

    return tracks_data

def plot_coordinates_to_tensor(coordinates, height, width, bbox_height, bbox_width, size_multiplier, prompt):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        text_color = '#999999'
        bg_color = '#353535'
        matplotlib.pyplot.rcParams['text.color'] = text_color
        fig, ax = matplotlib.pyplot.subplots(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.grid(color=text_color, linestyle='-', linewidth=0.5)
        ax.set_xlabel('x', color=text_color)
        ax.set_ylabel('y', color=text_color)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color(text_color)
        ax.set_title('position for: ' + prompt)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        #ax.legend().remove()
        ax.set_xlim(0, width) # Set the x-axis to match the input latent width
        ax.set_ylim(height, 0) # Set the y-axis to match the input latent height, with (0,0) at top-left
        # Adjust the margins of the subplot
        matplotlib.pyplot.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)

        cmap = matplotlib.pyplot.get_cmap('rainbow')
        image_batch = []
        canvas = FigureCanvas(fig)
        width, height = fig.get_size_inches() * fig.get_dpi()
        # Draw a box at each coordinate
        for i, ((x, y), size) in enumerate(zip(coordinates, size_multiplier)):
            color_index = i / (len(coordinates) - 1)
            color = cmap(color_index)
            draw_height = bbox_height * size
            draw_width = bbox_width * size
            rect = matplotlib.patches.Rectangle((x - draw_width/2, y - draw_height/2), draw_width, draw_height,
                                            linewidth=1, edgecolor=color, facecolor='none', alpha=0.5)
            ax.add_patch(rect)

            # Check if there is a next coordinate to draw an arrow to
            if i < len(coordinates) - 1:
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[i + 1]
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->",
                                            linestyle="-",
                                            lw=1,
                                            color=color,
                                            mutation_scale=20))
            canvas.draw()
            image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3).copy()
            image_tensor = torch.from_numpy(image_np).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            image_batch.append(image_tensor)
            
        matplotlib.pyplot.close(fig)
        image_batch_tensor = torch.cat(image_batch, dim=0)

        return image_batch_tensor

class PlotCoordinates:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                              "coordinates": ("STRING", {"forceInput": True}),
                              "text": ("STRING", {"default": 'title', "multiline": False}),
                              "width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                              "bbox_width": ("INT", {"default": 128, "min": 8, "max": 4096, "step": 8}),
                              "bbox_height": ("INT", {"default": 128, "min": 8, "max": 4096, "step": 8}),
                            },
                "optional": {"size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True})},
                }
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("images", "width", "height", "bbox_width", "bbox_height",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
Plots coordinates to sequence of images using Matplotlib.  

"""

    def append(self, coordinates, text, width, height, bbox_width, bbox_height, size_multiplier=[1.0]):
        coordinates = json.loads(coordinates.replace("'", '"'))
        coordinates = [(coord['x'], coord['y']) for coord in coordinates]
        batch_size = len(coordinates)
        if not size_multiplier or len(size_multiplier) != batch_size:
            size_multiplier = [0] * batch_size
        else:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]

        plot_image_tensor = plot_coordinates_to_tensor(coordinates, height, width, bbox_height, bbox_width, size_multiplier, text)
        
        return (plot_image_tensor, width, height, bbox_width, bbox_height)
    
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
                "sampling_method": (
                [   
                    'path',
                    'time',
                    'controlpoints',
                    'speed'
                ],
                {
                    "default": 'time'
                }),
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
                "bg_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("mask", "coord_str", "float", "count", "normalized_str",)
    FUNCTION = "splinedata"
    CATEGORY = "KJNodes/weights"
    DESCRIPTION = """
# WORK IN PROGRESS  
Do not count on this as part of your workflow yet,  
probably contains lots of bugs and stability is not  
guaranteed!!  
  
## Graphical editor to create values for various   
## schedules and/or mask batches.  

**Shift + click** to add control point at end.
**Ctrl + click** to add control point (subdivide) between two points.  
**Right click on a point** to delete it.    
Note that you can't delete from start/end.  
  
Right click on canvas for context menu:  
NEW!:
- Add new spline
    - Creates a new spline on same canvas, currently these paths are only outputed  
      as coordinates.
- Add single point
    - Creates a single point that only returns it's current position coords  
- Delete spline
    - Deletes the currently selected spline, you can select a spline by clicking on   
    it's path, or cycle through them with the 'Next spline' -option.  

These are purely visual options, doesn't affect the output:  
 - Toggle handles visibility
 - Display sample points: display the points to be returned.  

**points_to_sample** value sets the number of samples  
returned from the **drawn spline itself**, this is independent from the  
actual control points, so the interpolation type matters.  
sampling_method: 
 - time: samples along the time axis, used for schedules  
 - path: samples along the path itself, useful for coordinates  
 - controlpoints: samples only the control points themselves  

output types:
 - mask batch  
        example compatible nodes: anything that takes masks  
 - list of floats
        example compatible nodes: IPAdapter weights  
 - pandas series
        example compatible nodes: anything that takes Fizz'  
        nodes Batch Value Schedule  
 - torch tensor  
        example compatible nodes: unknown
"""

    def splinedata(self, mask_width, mask_height, coordinates, float_output_type, interpolation, 
               points_to_sample, sampling_method, points_store, tension, repeat_output, 
               min_value=0.0, max_value=1.0, bg_image=None):
    
        coordinates = json.loads(coordinates)
        
        # Handle nested list structure if present
        all_normalized = []
        all_normalized_y_values = []
        
        # Check if we have a nested list structure
        if isinstance(coordinates, list) and len(coordinates) > 0 and isinstance(coordinates[0], list):
            # Process each list of coordinates in the nested structure
            coordinate_sets = coordinates
        else:
            # If not nested, treat as a single list of coordinates
            coordinate_sets = [coordinates]
        
        # Process each set of coordinates
        for coord_set in coordinate_sets:
            normalized = []
            normalized_y_values = []
            
            for coord in coord_set:
                coord['x'] = int(round(coord['x']))
                coord['y'] = int(round(coord['y']))
                norm_x = (1.0 - (coord['x'] / mask_height) - 0.0) * (max_value - min_value) + min_value
                norm_y = (1.0 - (coord['y'] / mask_height) - 0.0) * (max_value - min_value) + min_value
                normalized_y_values.append(norm_y)
                normalized.append({'x':norm_x, 'y':norm_y})
            
            all_normalized.extend(normalized)
            all_normalized_y_values.extend(normalized_y_values)
        
        # Use the combined normalized values for output
        if float_output_type == 'list':
            out_floats = all_normalized_y_values * repeat_output
        elif float_output_type == 'pandas series':
            try:
                import pandas as pd
            except:
                raise Exception("MaskOrImageToWeight: pandas is not installed. Please install pandas to use this output_type")
            out_floats = pd.Series(all_normalized_y_values * repeat_output),
        elif float_output_type == 'tensor':
            out_floats = torch.tensor(all_normalized_y_values * repeat_output, dtype=torch.float32)
        
        # Create a color map for grayscale intensities
        color_map = lambda y: torch.full((mask_height, mask_width, 3), y, dtype=torch.float32)

        # Create image tensors for each normalized y value
        mask_tensors = [color_map(y) for y in all_normalized_y_values]
        masks_out = torch.stack(mask_tensors)
        masks_out = masks_out.repeat(repeat_output, 1, 1, 1)
        masks_out = masks_out.mean(dim=-1)
        
        if bg_image is None:
            return (masks_out, json.dumps(coordinates if len(coordinates) > 1 else coordinates[0]), out_floats, len(out_floats), json.dumps(all_normalized))
        else:
            transform = transforms.ToPILImage()
            image = transform(bg_image[0].permute(2, 0, 1))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=75)

            # Encode the image bytes to a Base64 string
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return {
                "ui": {"bg_image": [img_base64]},
                "result": (masks_out, json.dumps(coordinates if len(coordinates) > 1 else coordinates[0]), out_floats, len(out_floats), json.dumps(all_normalized))
            }
     

class CreateShapeMaskOnPath:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Creates a mask or batch of masks with the specified shape.  
Locations are center locations.  
"""
    DEPRECATED = True

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
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
        },
        "optional": {
            "size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True}),
        }
    } 

    def createshapemask(self, coordinates, frame_width, frame_height, shape_width, shape_height, shape, size_multiplier=[1.0]):
        # Define the number of images in the batch
        coordinates = coordinates.replace("'", '"')
        coordinates = json.loads(coordinates)

        batch_size = len(coordinates)
        out = []
        color = "white"
        if not size_multiplier or len(size_multiplier) != batch_size:
            size_multiplier = [0] * batch_size
        else:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]
        for i, coord in enumerate(coordinates):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0
            current_width = max(0, shape_width + i * size_multiplier[i])
            current_height = max(0, shape_height + i * size_multiplier[i])

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



class CreateShapeImageOnPath:
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image","mask", )
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates an image or batch of images with the specified shape.  
Locations are center locations.  
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
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 2, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 2, "max": 4096, "step": 1}),
                "shape_color": ("STRING", {"default": 'white'}),
                "bg_color": ("STRING", {"default": 'black'}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
        },
        "optional": {
            "size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True}),
            "trailing": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "border_width": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "border_color": ("STRING", {"default": 'black'}),
        }
    } 

    def createshapemask(self, coordinates, frame_width, frame_height, shape_width, shape_height, shape_color, 
                        bg_color, blur_radius, shape, intensity, size_multiplier=[1.0], trailing=1.0, border_width=0, border_color='black'):

        shape_color = parse_color(shape_color)
        border_color = parse_color(border_color)
        bg_color = parse_color(bg_color)
        coords_list = parse_json_tracks(coordinates)

        batch_size = len(coords_list[0])
        images_list = []
        masks_list = []

        if not size_multiplier or len(size_multiplier) != batch_size:
            size_multiplier = [1] * batch_size
        else:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]

        previous_output = None

        for i in range(batch_size):
            image = Image.new("RGB", (frame_width, frame_height), bg_color)
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0
            current_width = shape_width * size_multiplier[i]
            current_height = shape_height * size_multiplier[i]
            
            for coords in coords_list:
                location_x = coords[i]['x']
                location_y = coords[i]['y']
            
                if shape == 'circle' or shape == 'square':
                    # Define the bounding box for the shape
                    left_up_point = (location_x - current_width // 2, location_y - current_height // 2)
                    right_down_point = (location_x + current_width // 2, location_y + current_height // 2)
                    two_points = [left_up_point, right_down_point]

                    if shape == 'circle':
                        if border_width > 0:
                            draw.ellipse(two_points, fill=shape_color, outline=border_color, width=border_width)
                        else:
                            draw.ellipse(two_points, fill=shape_color)
                    elif shape == 'square':
                        if border_width > 0:
                            draw.rectangle(two_points, fill=shape_color, outline=border_color, width=border_width)
                        else:
                            draw.rectangle(two_points, fill=shape_color)
                        
                elif shape == 'triangle':
                    # Define the points for the triangle
                    left_up_point = (location_x - current_width // 2, location_y + current_height // 2) # bottom left
                    right_down_point = (location_x + current_width // 2, location_y + current_height // 2) # bottom right
                    top_point = (location_x, location_y - current_height // 2) # top point
                    
                    if border_width > 0:
                        draw.polygon([top_point, left_up_point, right_down_point], fill=shape_color, outline=border_color, width=border_width)
                    else:
                        draw.polygon([top_point, left_up_point, right_down_point], fill=shape_color)

            if blur_radius != 0:
                    image = image.filter(ImageFilter.GaussianBlur(blur_radius))
            # Blend the current image with the accumulated image
            
            image = pil2tensor(image)
            if trailing != 1.0 and previous_output is not None:
                # Add the decayed previous output to the current frame
                image += trailing * previous_output
                image = image / image.max()
            previous_output = image
            image = image * intensity
            mask = image[:, :, :, 0]
            masks_list.append(mask)
            images_list.append(image)
        out_images = torch.cat(images_list, dim=0).cpu().float()
        out_masks = torch.cat(masks_list, dim=0)
        return (out_images, out_masks)
    
class CreateTextOnPath:
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK",)
    RETURN_NAMES = ("image", "mask", "mask_inverted",)
    FUNCTION = "createtextmask"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Creates a mask or batch of masks with the specified text.  
Locations are center locations.  
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "text": ("STRING", {"default": 'text', "multiline": True}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
                "font_size": ("INT", {"default": 42}),
                 "alignment": (
                [   'left',
                    'center',
                    'right'
                ],
                {"default": 'center'}
                ),
                "text_color": ("STRING", {"default": 'white'}),
        },
        "optional": {
            "size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True}),
        }
    } 

    def createtextmask(self, coordinates, frame_width, frame_height, font, font_size, text, text_color, alignment, size_multiplier=[1.0]):
        coordinates = coordinates.replace("'", '"')
        coordinates = json.loads(coordinates)

        batch_size = len(coordinates)
        mask_list = []
        image_list = []
        color = parse_color(text_color)
        font_path = folder_paths.get_full_path("kjnodes_fonts", font)

        if len(size_multiplier) != batch_size:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]
        
        for i, coord in enumerate(coordinates):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)
            lines = text.split('\n')  # Split the text into lines
            # Apply the size multiplier to the font size for this iteration
            current_font_size = int(font_size * size_multiplier[i])
            current_font = ImageFont.truetype(font_path, current_font_size)
            line_heights = [current_font.getbbox(line)[3] for line in lines]  # List of line heights
            total_text_height = sum(line_heights)  # Total height of text block

            # Calculate the starting Y position to center the block of text
            start_y = coord['y'] - total_text_height // 2
            for j, line in enumerate(lines):
                text_width, text_height = current_font.getbbox(line)[2], line_heights[j]
                if alignment == 'left':
                    location_x = coord['x']
                elif alignment == 'center':
                    location_x = int(coord['x'] - text_width // 2)
                elif alignment == 'right':
                    location_x = int(coord['x'] - text_width)
                
                location_y = int(start_y + sum(line_heights[:j]))
                text_position = (location_x, location_y)
                # Draw the text
                try:
                    draw.text(text_position, line, fill=color, font=current_font, features=['-liga'])
                except:
                    draw.text(text_position, line, fill=color, font=current_font)
            
            image = pil2tensor(image)
            non_black_pixels = (image > 0).any(dim=-1)
            mask = non_black_pixels.to(image.dtype)
            mask_list.append(mask)
            image_list.append(image)

        out_images = torch.cat(image_list, dim=0).cpu().float()
        out_masks = torch.cat(mask_list, dim=0)
        return (out_images, out_masks, 1.0 - out_masks,)

class CreateGradientFromCoords:
    
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "generate"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates a gradient image from coordinates.    
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "start_color": ("STRING", {"default": 'white'}),
                "end_color": ("STRING", {"default": 'black'}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
        },
    } 
    
    def generate(self, coordinates, frame_width, frame_height, start_color, end_color, multiplier):
        # Parse the coordinates
        coordinates = json.loads(coordinates.replace("'", '"'))

        # Create an image
        image = Image.new("RGB", (frame_width, frame_height))
        draw = ImageDraw.Draw(image)

        # Extract start and end points for the gradient
        start_coord = coordinates[0]
        end_coord = coordinates[1]

        start_color = parse_color(start_color)
        end_color = parse_color(end_color)

        # Calculate the gradient direction (vector)
        gradient_direction = (end_coord['x'] - start_coord['x'], end_coord['y'] - start_coord['y'])
        gradient_length = (gradient_direction[0] ** 2 + gradient_direction[1] ** 2) ** 0.5

        # Iterate over each pixel in the image
        for y in range(frame_height):
            for x in range(frame_width):
                # Calculate the projection of the point on the gradient line
                point_vector = (x - start_coord['x'], y - start_coord['y'])
                projection = (point_vector[0] * gradient_direction[0] + point_vector[1] * gradient_direction[1]) / gradient_length
                projection = max(min(projection, gradient_length), 0)  # Clamp the projection value

                # Calculate the blend factor for the current pixel
                blend = projection * multiplier / gradient_length 

                # Determine the color of the current pixel
                color = (
                    int(start_color[0] + (end_color[0] - start_color[0]) * blend),
                    int(start_color[1] + (end_color[1] - start_color[1]) * blend),
                    int(start_color[2] + (end_color[2] - start_color[2]) * blend)
                )

                # Set the pixel color
                draw.point((x, y), fill=color)

        # Convert the PIL image to a tensor (assuming such a function exists in your context)
        image_tensor = pil2tensor(image)

        return (image_tensor,)

class GradientToFloat:
    
    RETURN_TYPES = ("FLOAT", "FLOAT",)
    RETURN_NAMES = ("float_x", "float_y", )
    FUNCTION = "sample"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Calculates list of floats from image.    
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "steps": ("INT", {"default": 10, "min": 2, "max": 10000, "step": 1}),
        },
    } 
    
    def sample(self, image, steps):
        # Assuming image is a tensor with shape [B, H, W, C]
        B, H, W, C = image.shape

        # Sample along the width axis (W)
        w_intervals = torch.linspace(0, W - 1, steps=steps, dtype=torch.int64)
        # Assuming we're sampling from the first batch and the first channel
        w_sampled = image[0, :, w_intervals, 0]

        # Sample along the height axis (H)
        h_intervals = torch.linspace(0, H - 1, steps=steps, dtype=torch.int64)
        # Assuming we're sampling from the first batch and the first channel
        h_sampled = image[0, h_intervals, :, 0]

        # Taking the mean across the height for width sampling, and across the width for height sampling
        w_values = w_sampled.mean(dim=0).tolist()
        h_values = h_sampled.mean(dim=1).tolist()

        return (w_values, h_values)
    
class MaskOrImageToWeight:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_type": (
                [   
                    'list',
                    'pandas series',
                    'tensor',
                    'string'
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
    RETURN_TYPES = ("FLOAT", "STRING",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes/weights"
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
            out = mean_values
        elif output_type == 'pandas series':
            try:
                import pandas as pd
            except:
                raise Exception("MaskOrImageToWeight: pandas is not installed. Please install pandas to use this output_type")
            out = pd.Series(mean_values),
        elif output_type == 'tensor':
            out = torch.tensor(mean_values, dtype=torch.float32),
        return (out, [str(value) for value in mean_values],)
    
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
                "remap_values": ("BOOLEAN", {"default": False}),
                "remap_min": ("FLOAT", {"default": 0.0, "min": -100000, "max": 100000.0, "step": 0.01}),
                "remap_max": ("FLOAT", {"default": 1.0, "min": -100000, "max": 100000.0, "step": 0.01}),
             },
             
        }
    RETURN_TYPES = ("FLOAT", "STRING", "INT",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes/weights"
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
        else:
            raise ValueError("Unsupported input type")

    def execute(self, input_values, output_type, invert, repeat, remap_to_frames=0, interpolation_curve=None, remap_min=0.0, remap_max=1.0, remap_values=False):
        import pandas as pd
        input_type = self.detect_input_type(input_values)

        if input_type == 'pandas series':
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
            if remap_values:
                float_values = self.remap_values(float_values, remap_min, remap_max)

        if output_type == 'list':
            out = float_values,
        elif output_type == 'pandas series':
            out = pd.Series(float_values),
        elif output_type == 'tensor':
            if input_type == 'pandas series':
                out = torch.tensor(float_values.values, dtype=torch.float32),
            else:   
                out = torch.tensor(float_values, dtype=torch.float32),
        elif output_type == 'match_input':
            out = float_values,
        return (out, [str(value) for value in float_values], [int(value) for value in float_values])
    
    def remap_values(self, values, target_min, target_max):
        # Determine the current range
        current_min = min(values)
        current_max = max(values)
        current_range = current_max - current_min
        
        # Determine the target range
        target_range = target_max - target_min
        
        # Perform the linear interpolation for each value
        remapped_values = [(value - current_min) / current_range * target_range + target_min for value in values]
        
        return remapped_values
        

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
    CATEGORY = "KJNodes/masking/generate"
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
    CATEGORY = "KJNodes/weights"
    DESCRIPTION = """
Extends, and converts if needed, different value lists/series  
"""

    def detect_input_type(self, input_values):
        import pandas as pd
        if isinstance(input_values, list):
            return 'list'
        elif isinstance(input_values, pd.Series):
            return 'pandas series'
        elif isinstance(input_values, torch.Tensor):
            return 'tensor'
        else:
            raise ValueError("Unsupported input type")

    def execute(self, input_values_1, input_values_2, output_type):
        import pandas as pd
        input_type_1 = self.detect_input_type(input_values_1)
        input_type_2 = self.detect_input_type(input_values_2)
        # Convert input_values_2 to the same format as input_values_1 if they do not match
        if not input_type_1 == input_type_2:
            print("Converting input_values_2 to the same format as input_values_1")
            if input_type_1 == 'pandas series':
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
        elif output_type == 'pandas series':
            return pd.Series(float_values),
        elif output_type == 'tensor':
            if input_type_1 == 'pandas series':
                return torch.tensor(float_values.values, dtype=torch.float32),
            else:
                return torch.tensor(float_values, dtype=torch.float32),
        elif output_type == 'match_input':
            return float_values,
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
        
class FloatToSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "float_list": ("FLOAT", {"default": 0.0, "forceInput": True}),
                     }
                }
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    CATEGORY = "KJNodes/noise"
    FUNCTION = "customsigmas"
    DESCRIPTION = """
Creates a sigmas tensor from list of float values.  

"""
    def customsigmas(self, float_list):
        return torch.tensor(float_list, dtype=torch.float32),

class SigmasToFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "sigmas": ("SIGMAS",),
                     }
                }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    CATEGORY = "KJNodes/noise"
    FUNCTION = "customsigmas"
    DESCRIPTION = """
Creates a float list from sigmas tensors.  

"""
    def customsigmas(self, sigmas):
        return sigmas.tolist(),

class GLIGENTextBoxApplyBatchCoords:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ),
                              "latents": ("LATENT", ),
                              "clip": ("CLIP", ),
                              "gligen_textbox_model": ("GLIGEN", ),
                              "coordinates": ("STRING", {"forceInput": True}),
                              "text": ("STRING", {"multiline": True}),
                              "width": ("INT", {"default": 128, "min": 8, "max": 4096, "step": 8}),
                              "height": ("INT", {"default": 128, "min": 8, "max": 4096, "step": 8}),
                            },
                "optional": {"size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True})},
                }
    RETURN_TYPES = ("CONDITIONING", "IMAGE", )
    RETURN_NAMES = ("conditioning", "coord_preview", )
    FUNCTION = "append"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
This node allows scheduling GLIGEN text box positions in a batch,  
to be used with AnimateDiff-Evolved. Intended to pair with the  
Spline Editor -node.  

GLIGEN model can be downloaded through the Manage's "Install Models" menu.  
Or directly from here:  
https://huggingface.co/comfyanonymous/GLIGEN_pruned_safetensors/tree/main  
  
Inputs:  
- **latents** input is used to calculate batch size  
- **clip** is your standard text encoder, use same as for the main prompt  
- **gligen_textbox_model** connects to GLIGEN Loader  
- **coordinates** takes a json string of points, directly compatible  
with the spline editor node.
- **text** is the part of the prompt to set position for  
- **width** and **height** are the size of the GLIGEN bounding box  
  
Outputs:
- **conditioning** goes between to clip text encode and the sampler  
- **coord_preview** is an optional preview of the coordinates and  
bounding boxes.

"""

    def append(self, latents, coordinates, conditioning_to, clip, gligen_textbox_model, text, width, height, size_multiplier=[1.0]):
        coordinates = json.loads(coordinates.replace("'", '"'))
        coordinates = [(coord['x'], coord['y']) for coord in coordinates]

        batch_size = sum(tensor.size(0) for tensor in latents.values())
        if len(coordinates) != batch_size:
            print("GLIGENTextBoxApplyBatchCoords WARNING: The number of coordinates does not match the number of latents")

        c = []
        _, cond_pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True)

        for t in conditioning_to:
            n = [t[0], t[1].copy()]
            
            position_params_batch = [[] for _ in range(batch_size)]  # Initialize a list of empty lists for each batch item
            if len(size_multiplier) != batch_size:
                size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]

            for i in range(batch_size):
                x_position, y_position = coordinates[i]
                position_param = (cond_pooled, int((height // 8) * size_multiplier[i]), int((width // 8) * size_multiplier[i]), (y_position - height // 2) // 8, (x_position - width // 2) // 8)
                position_params_batch[i].append(position_param)  # Append position_param to the correct sublist

            prev = []
            if "gligen" in n[1]:
                prev = n[1]['gligen'][2]
            else:
                prev = [[] for _ in range(batch_size)]
            # Concatenate prev and position_params_batch, ensuring both are lists of lists
            # and each sublist corresponds to a batch item
            combined_position_params = [prev_item + batch_item for prev_item, batch_item in zip(prev, position_params_batch)]
            n[1]['gligen'] = ("position_batched", gligen_textbox_model, combined_position_params)
            c.append(n)

        image_height = latents['samples'].shape[-2] * 8
        image_width = latents['samples'].shape[-1] * 8
        plot_image_tensor = plot_coordinates_to_tensor(coordinates, image_height, image_width, height, width, size_multiplier, text)
        
        return (c, plot_image_tensor,)
    
class CreateInstanceDiffusionTracking:
    
    RETURN_TYPES = ("TRACKING", "STRING", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("tracking", "prompt", "width", "height", "bbox_width", "bbox_height",)
    FUNCTION = "tracking"
    CATEGORY = "KJNodes/InstanceDiffusion"
    DESCRIPTION = """
Creates tracking data to be used with InstanceDiffusion:  
https://github.com/logtd/ComfyUI-InstanceDiffusion  
  
InstanceDiffusion prompt format:  
"class_id.class_name": "prompt",  
for example:  
"1.head": "((head))",  
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "bbox_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "bbox_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "class_name": ("STRING", {"default": "class_name"}),
                "class_id": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                "prompt": ("STRING", {"default": "prompt", "multiline": True}),
        },
        "optional": {
            "size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True}),
            "fit_in_frame": ("BOOLEAN", {"default": True}),
        }
    } 

    def tracking(self, coordinates, class_name, class_id, width, height, bbox_width, bbox_height, prompt, size_multiplier=[1.0], fit_in_frame=True):
        # Define the number of images in the batch
        coordinates = coordinates.replace("'", '"')
        coordinates = json.loads(coordinates)

        tracked = {}
        tracked[class_name] = {}
        batch_size = len(coordinates)
        # Initialize a list to hold the coordinates for the current ID
        id_coordinates = []
        if not size_multiplier or len(size_multiplier) != batch_size:
            size_multiplier = [0] * batch_size
        else:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]
        for i, coord in enumerate(coordinates):
            x = coord['x']
            y = coord['y']
            adjusted_bbox_width = bbox_width * size_multiplier[i]
            adjusted_bbox_height = bbox_height * size_multiplier[i]
            # Calculate the top left and bottom right coordinates
            top_left_x = x - adjusted_bbox_width // 2
            top_left_y = y - adjusted_bbox_height // 2
            bottom_right_x = x + adjusted_bbox_width // 2
            bottom_right_y = y + adjusted_bbox_height // 2

            if fit_in_frame:
                # Clip the coordinates to the frame boundaries
                top_left_x = max(0, top_left_x)
                top_left_y = max(0, top_left_y)
                bottom_right_x = min(width, bottom_right_x)
                bottom_right_y = min(height, bottom_right_y)
                # Ensure width and height are positive
                adjusted_bbox_width = max(1, bottom_right_x - top_left_x)
                adjusted_bbox_height = max(1, bottom_right_y - top_left_y)

                # Update the coordinates with the new width and height
                bottom_right_x = top_left_x + adjusted_bbox_width
                bottom_right_y = top_left_y + adjusted_bbox_height

            # Append the top left and bottom right coordinates to the list for the current ID
            id_coordinates.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y, width, height])
        
        class_id = int(class_id)
        # Assign the list of coordinates to the specified ID within the class_id dictionary
        tracked[class_name][class_id] = id_coordinates

        prompt_string = ""
        for class_name, class_data in tracked.items():
            for class_id in class_data.keys():
                class_id_str = str(class_id)
                # Use the incoming prompt for each class name and ID
                prompt_string += f'"{class_id_str}.{class_name}": "({prompt})",\n'

        # Remove the last comma and newline
        prompt_string = prompt_string.rstrip(",\n")

        return (tracked, prompt_string, width, height, bbox_width, bbox_height)

class AppendInstanceDiffusionTracking:
    
    RETURN_TYPES = ("TRACKING", "STRING",)
    RETURN_NAMES = ("tracking", "prompt",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/InstanceDiffusion"
    DESCRIPTION = """
Appends tracking data to be used with InstanceDiffusion:  
https://github.com/logtd/ComfyUI-InstanceDiffusion  

"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tracking_1": ("TRACKING", {"forceInput": True}),
                "tracking_2": ("TRACKING", {"forceInput": True}),
        },
        "optional": {
            "prompt_1": ("STRING", {"default": "", "forceInput": True}),
            "prompt_2": ("STRING", {"default": "", "forceInput": True}),
        }
    } 

    def append(self, tracking_1, tracking_2, prompt_1="", prompt_2=""):
        tracking_copy = tracking_1.copy()
        # Check for existing class names and class IDs, and raise an error if they exist
        for class_name, class_data in tracking_2.items():
            if class_name not in tracking_copy:
                tracking_copy[class_name] = class_data
            else:
                # If the class name exists, merge the class data from tracking_2 into tracking_copy
                # This will add new class IDs under the same class name without raising an error
                tracking_copy[class_name].update(class_data)
        prompt_string = prompt_1 + "," + prompt_2
        return (tracking_copy, prompt_string)
        
class InterpolateCoords:
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coordinates",)
    FUNCTION = "interpolate"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
Interpolates coordinates based on a curve.   
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "interpolation_curve": ("FLOAT", {"forceInput": True}),
                
        },
    } 

    def interpolate(self, coordinates, interpolation_curve):
        # Parse the JSON string to get the list of coordinates
        coordinates = json.loads(coordinates.replace("'", '"'))

        # Convert the list of dictionaries to a list of (x, y) tuples for easier processing
        coordinates = [(coord['x'], coord['y']) for coord in coordinates]

        # Calculate the total length of the original path
        path_length = sum(np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[i-1])) 
                        for i in range(1, len(coordinates)))

        # Initialize variables for interpolation
        interpolated_coords = []
        current_length = 0
        current_index = 0

        # Iterate over the normalized curve
        for normalized_length in interpolation_curve:
            target_length = normalized_length * path_length # Convert to the original scale
            while current_index < len(coordinates) - 1:
                segment_start, segment_end = np.array(coordinates[current_index]), np.array(coordinates[current_index + 1])
                segment_length = np.linalg.norm(segment_end - segment_start)
                if current_length + segment_length >= target_length:
                    break
                current_length += segment_length
                current_index += 1

            # Interpolate between the last two points
            if current_index < len(coordinates) - 1:
                p1, p2 = np.array(coordinates[current_index]), np.array(coordinates[current_index + 1])
                segment_length = np.linalg.norm(p2 - p1)
                if segment_length > 0:
                    t = (target_length - current_length) / segment_length
                    interpolated_point = p1 + t * (p2 - p1)
                    interpolated_coords.append(interpolated_point.tolist())
                else:
                    interpolated_coords.append(p1.tolist())
            else:
                # If the target_length is at or beyond the end of the path, add the last coordinate
                interpolated_coords.append(coordinates[-1])

        # Convert back to string format if necessary
        interpolated_coords_str = "[" + ", ".join([f"{{'x': {round(coord[0])}, 'y': {round(coord[1])}}}" for coord in interpolated_coords]) + "]"
        print(interpolated_coords_str)

        return (interpolated_coords_str,)
    
class DrawInstanceDiffusionTracking:
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "draw"
    CATEGORY = "KJNodes/InstanceDiffusion"
    DESCRIPTION = """
Draws the tracking data from  
CreateInstanceDiffusionTracking -node.

"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "tracking": ("TRACKING", {"forceInput": True}),
                "box_line_width": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "draw_text": ("BOOLEAN", {"default": True}),
                "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
                "font_size": ("INT", {"default": 20}),
        },
    } 

    def draw(self, image, tracking, box_line_width, draw_text, font, font_size):
        import matplotlib.cm as cm

        modified_images = []
        
        colormap = cm.get_cmap('rainbow', len(tracking))
        if draw_text:
            font_path = folder_paths.get_full_path("kjnodes_fonts", font)
            font = ImageFont.truetype(font_path, font_size)

        # Iterate over each image in the batch
        for i in range(image.shape[0]):
            # Extract the current image and convert it to a PIL image
            current_image = image[i, :, :, :].permute(2, 0, 1)
            pil_image = transforms.ToPILImage()(current_image)
            
            draw = ImageDraw.Draw(pil_image)
            
            # Iterate over the bounding boxes for the current image
            for j, (class_name, class_data) in enumerate(tracking.items()):
                for class_id, bbox_list in class_data.items():
                    # Check if the current index is within the bounds of the bbox_list
                    if i < len(bbox_list):
                        bbox = bbox_list[i]
                        # Ensure bbox is a list or tuple before unpacking
                        if isinstance(bbox, (list, tuple)):
                            x1, y1, x2, y2, _, _ = bbox
                            # Convert coordinates to integers
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            # Generate a color from the rainbow colormap
                            color = tuple(int(255 * x) for x in colormap(j / len(tracking)))[:3]
                            # Draw the bounding box on the image with the generated color
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_line_width)
                            if draw_text:
                                # Draw the class name and ID as text above the box with the generated color
                                text = f"{class_id}.{class_name}"
                                # Calculate the width and height of the text
                                _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)
                                # Position the text above the top-left corner of the box
                                text_position = (x1, y1 - text_height)
                                draw.text(text_position, text, fill=color, font=font)
                        else:
                            print(f"Unexpected data type for bbox: {type(bbox)}")
            
            # Convert the drawn image back to a torch tensor and adjust back to (H, W, C)
            modified_image_tensor = transforms.ToTensor()(pil_image).permute(1, 2, 0)
            modified_images.append(modified_image_tensor)
        
        # Stack the modified images back into a batch
        image_tensor_batch = torch.stack(modified_images).cpu().float()
        
        return image_tensor_batch,

class PointsEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points_store": ("STRING", {"multiline": False}),
                "coordinates": ("STRING", {"multiline": False}),
                "neg_coordinates": ("STRING", {"multiline": False}),
                "bbox_store": ("STRING", {"multiline": False}),
                "bboxes": ("STRING", {"multiline": False}),
                "bbox_format": (
                [   
                    'xyxy',
                    'xywh',
                ],
                ),
                "width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "normalize": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "bg_image": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "BBOX", "MASK", "IMAGE")
    RETURN_NAMES = ("positive_coords", "negative_coords", "bbox", "bbox_mask", "cropped_image")
    FUNCTION = "pointdata"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
# WORK IN PROGRESS  
Do not count on this as part of your workflow yet,  
probably contains lots of bugs and stability is not  
guaranteed!!  
  
## Graphical editor to create coordinates

**Shift + click** to add a positive (green) point.
**Shift + right click** to add a negative (red) point.
**Ctrl + click** to draw a box.  
**Right click on a point** to delete it.    
Note that you can't delete from start/end of the points array.  
  
To add an image select the node and copy/paste or drag in the image.  
Or from the bg_image input on queue (first frame of the batch).  

**THE IMAGE IS SAVED TO THE NODE AND WORKFLOW METADATA**  
you can clear the image from the context menu by right clicking on the canvas  

"""

    def pointdata(self, points_store, bbox_store, width, height, coordinates, neg_coordinates, normalize, bboxes, bbox_format="xyxy", bg_image=None):
        coordinates = json.loads(coordinates)
        pos_coordinates = []
        for coord in coordinates:
            coord['x'] = int(round(coord['x']))
            coord['y'] = int(round(coord['y']))
            if normalize:
                norm_x = coord['x'] / width
                norm_y = coord['y'] / height
                pos_coordinates.append({'x': norm_x, 'y': norm_y})
            else:
                pos_coordinates.append({'x': coord['x'], 'y': coord['y']})

        if neg_coordinates:
            coordinates = json.loads(neg_coordinates)
            neg_coordinates = []
            for coord in coordinates:
                coord['x'] = int(round(coord['x']))
                coord['y'] = int(round(coord['y']))
                if normalize:
                    norm_x = coord['x'] / width
                    norm_y = coord['y'] / height
                    neg_coordinates.append({'x': norm_x, 'y': norm_y})
                else:
                    neg_coordinates.append({'x': coord['x'], 'y': coord['y']})

        # Create a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = json.loads(bboxes)
        print(bboxes)
        valid_bboxes = []
        for bbox in bboxes:
            if (bbox.get("startX") is None or
                bbox.get("startY") is None or
                bbox.get("endX") is None or
                bbox.get("endY") is None):
                continue  # Skip this bounding box if any value is None
            else:                
                # Ensure that endX and endY are greater than startX and startY
                x_min = min(int(bbox["startX"]), int(bbox["endX"]))
                y_min = min(int(bbox["startY"]), int(bbox["endY"]))
                x_max = max(int(bbox["startX"]), int(bbox["endX"]))
                y_max = max(int(bbox["startY"]), int(bbox["endY"]))
                
                valid_bboxes.append((x_min, y_min, x_max, y_max))

            bboxes_xyxy = []
            for bbox in valid_bboxes:
                x_min, y_min, x_max, y_max = bbox
                bboxes_xyxy.append((x_min, y_min, x_max, y_max))
                mask[y_min:y_max, x_min:x_max] = 1  # Fill the bounding box area with 1s

            if bbox_format == "xywh":
                bboxes_xywh = []
                for bbox in valid_bboxes:
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    bboxes_xywh.append((x_min, y_min, width, height))
                bboxes = bboxes_xywh
            else:
                bboxes = bboxes_xyxy           

        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.unsqueeze(0).float().cpu()

        if bg_image is not None and len(valid_bboxes) > 0:
            x_min, y_min, x_max, y_max = bboxes[0]
            cropped_image = bg_image[:, y_min:y_max, x_min:x_max, :]

        elif bg_image is not None:
            cropped_image = bg_image

        if bg_image is None:
            return (json.dumps(pos_coordinates), json.dumps(neg_coordinates), bboxes, mask_tensor)
        else:
            transform = transforms.ToPILImage()
            image = transform(bg_image[0].permute(2, 0, 1))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=75)

            # Step 3: Encode the image bytes to a Base64 string
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
            return {
                "ui": {"bg_image": [img_base64]}, 
                "result": (json.dumps(pos_coordinates), json.dumps(neg_coordinates), bboxes, mask_tensor, cropped_image)
            }

class CutAndDragOnPath:
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image","mask", )
    FUNCTION = "cutanddrag" 
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Cuts the masked area from the image, and drags it along the path. If inpaint is enabled, and no bg_image is provided, the cut area is filled using cv2 TELEA algorithm.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "coordinates": ("STRING", {"forceInput": True}),
                "mask": ("MASK",),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "inpaint": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "bg_image": ("IMAGE",),
        }
    }

    def cutanddrag(self, image, coordinates, mask, frame_width, frame_height, inpaint, bg_image=None):
        # Parse coordinates
        coords_list = parse_json_tracks(coordinates)

        batch_size = len(coords_list[0])
        images_list = []
        masks_list = []

        # Convert input image and mask to PIL
        input_image = tensor2pil(image)[0]
        input_mask = tensor2pil(mask)[0]

        # Find masked region bounds
        mask_array = np.array(input_mask)
        y_indices, x_indices = np.where(mask_array > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return (image, mask)
            
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # Cut out the masked region
        cut_width = x_max - x_min
        cut_height = y_max - y_min
        cut_image = input_image.crop((x_min, y_min, x_max, y_max))
        cut_mask = input_mask.crop((x_min, y_min, x_max, y_max))
        
        # Create inpainted background
        if bg_image is None:
            background = input_image.copy()
            # Inpaint the cut area
            if inpaint:
                import cv2
                border = 5 # Create small border around cut area for better inpainting
                fill_mask = Image.new("L", background.size, 0)
                draw = ImageDraw.Draw(fill_mask)
                draw.rectangle([x_min-border, y_min-border, x_max+border, y_max+border], fill=255)
                background = cv2.inpaint(
                    np.array(background), 
                    np.array(fill_mask), 
                    inpaintRadius=3, 
                    flags=cv2.INPAINT_TELEA
                )
                background = Image.fromarray(background)
        else:
            background = tensor2pil(bg_image)[0]
        
        # Create batch of images with cut region at different positions
        for i in range(batch_size):
            # Create new image
            new_image = background.copy()
            new_mask = Image.new("L", (frame_width, frame_height), 0)

            # Get target position from coordinates
            for coords in coords_list:
                target_x = int(coords[i]['x'] - cut_width/2)
                target_y = int(coords[i]['y'] - cut_height/2)

                # Paste cut region at new position
                new_image.paste(cut_image, (target_x, target_y), cut_mask)
                new_mask.paste(cut_mask, (target_x, target_y))

            # Convert to tensor and append
            image_tensor = pil2tensor(new_image)
            mask_tensor = pil2tensor(new_mask)
            
            images_list.append(image_tensor)
            masks_list.append(mask_tensor)

        # Stack tensors into batches
        out_images = torch.cat(images_list, dim=0).cpu().float()
        out_masks = torch.cat(masks_list, dim=0)

        return (out_images, out_masks)

class CreateShapeJointOnPath:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create"
    CATEGORY = "KJNodes/image/generate"
    DESCRIPTION = """
The width is controlled by shape_width, and the length is the distance between the first and second points.
If pivot_coordinates are provided:
  - relative=True: The pivot movement offsets the entire shape from its path-defined position.
  - relative=False: The pivot replaces the starting point of the shape for positioning.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"multiline": True, "default": '[{"x":100,"y":100},{"x":400,"y":400}]'}),
                "frame_width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "frame_height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "total_frames": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "scaling_enabled": ("BOOLEAN", {"default": True}),
                "shape_width": ("INT", {"default": 20, "min": 1, "max": 4096, "step": 1}),
                "shape_width_end": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 0}),
                "bg_color": ("STRING", {"default": "black"}),
                "fill_color": ("STRING", {"default": "white"}),
                "easing_function": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "trailing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}), # Changed default/max from original node
                "bounce_between": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": { # Make pivot_coordinates optional
                "pivot_coordinates": ("STRING", {"multiline": True}),
                "relative": ("BOOLEAN", {"default": True}), # Added relative input
            }
        }

    def create(self, coordinates, frame_width, frame_height, shape_width, shape_width_end, fill_color, bg_color, scaling_enabled, total_frames, easing_function, blur_radius, intensity, trailing, bounce_between, pivot_coordinates=None, relative=True): # Added relative param
        # --- Standardize coordinates input ---
        if isinstance(coordinates, str):
            # Try parsing as a list of lists first, if it looks like it
            try:
                potential_list = json.loads(coordinates.replace("'", '"'))
                if isinstance(potential_list, list) and all(isinstance(item, list) for item in potential_list):
                    # It's likely a string representation of a list of paths
                    # Re-dump each inner list to treat them as separate coord strings
                    coord_strings = [json.dumps(path) for path in potential_list]
                    print(f"Interpreted single string input as {len(coord_strings)} paths.")
                elif isinstance(potential_list, list) and all(isinstance(item, dict) for item in potential_list):
                     # It's a single path represented as a string
                     coord_strings = [coordinates]
                else:
                     # Fallback: treat as single path string if format is unexpected
                     print("Warning: Unexpected format in single coordinate string. Treating as one path.")
                     coord_strings = [coordinates]
            except Exception as e:
                 print(f"Warning: Could not parse single coordinate string as JSON list. Treating as one path string. Error: {e}")
                 coord_strings = [coordinates] # Treat as a single path string if parsing fails
        elif isinstance(coordinates, list) and all(isinstance(item, str) for item in coordinates):
            coord_strings = coordinates # Already a list of strings
        else:
            print(f"Error: Invalid coordinates input type: {type(coordinates)}. Expected string or list of strings.")
            img = Image.new('RGB', (frame_width, frame_height), color=bg_color)
            return (pil2tensor(img),)

        all_paths_control_points = []
        all_paths_original_p0 = []
        all_paths_initial_p1 = []
        valid_paths_found = False
        for i, coord_string in enumerate(coord_strings):
            try:
                coords = json.loads(coord_string.replace("'", '"'))
                if not isinstance(coords, list) or len(coords) < 2:
                    print(f"Warning: Path {i+1} has < 2 points or invalid format. Skipping.")
                    all_paths_control_points.append(None) # Placeholder for skipped path
                    all_paths_original_p0.append(None)
                    all_paths_initial_p1.append(None)
                    continue
                points = [(c['x'], c['y']) for c in coords]
                control_points = [np.array(p) for p in points]
                all_paths_control_points.append(control_points)
                all_paths_original_p0.append(control_points[0])
                all_paths_initial_p1.append(control_points[1])
                valid_paths_found = True
            except Exception as e:
                print(f"Error parsing coordinates for path {i+1}: {e}. Skipping path.")
                all_paths_control_points.append(None) # Placeholder for skipped path
                all_paths_original_p0.append(None)
                all_paths_initial_p1.append(None)
                continue

        if not valid_paths_found:
             print("Error: No valid coordinate paths found.")
             img = Image.new('RGB', (frame_width, frame_height), color=bg_color)
             return (pil2tensor(img),)

        output_images = []
        previous_frame_tensor = None

        # --- Parse and Adjust Pivot Coordinates ---
        # (Applies the *same* pivot motion to *all* paths if provided)
        pivot_points_adjusted = None
        use_dynamic_pivot = False
        static_pivot_point = None # Used if pivot_coordinates is None or invalid

        if pivot_coordinates and pivot_coordinates.strip() and pivot_coordinates.strip() != '[]':
            try:
                pivot_coords_raw = json.loads(pivot_coordinates.replace("'", '"'))
                if isinstance(pivot_coords_raw, list) and len(pivot_coords_raw) > 0:
                    pivot_points_raw = [np.array((c['x'], c['y'])) for c in pivot_coords_raw]
                    current_len = len(pivot_points_raw)
                    if current_len < total_frames:
                        last_point = pivot_points_raw[-1]
                        padding = [last_point] * (total_frames - current_len)
                        pivot_points_adjusted = pivot_points_raw + padding
                    elif current_len > total_frames:
                        pivot_points_adjusted = pivot_points_raw[:total_frames]
                    else:
                        pivot_points_adjusted = pivot_points_raw

                    if pivot_points_adjusted:
                         use_dynamic_pivot = True
                         # print(f"Using dynamic pivot points. Adjusted count: {len(pivot_points_adjusted)}")
            except Exception as e:
                print(f"Warning: Error parsing pivot_coordinates: {e}. Using static p0 for each path.")
                use_dynamic_pivot = False
        # else: use_dynamic_pivot remains False

        # --- Pre-calculate fixed length and direction for paths if needed ---
        all_paths_fixed_length = []
        all_paths_fixed_v_normalized = []
        for i in range(len(all_paths_control_points)):
             if all_paths_control_points[i] is None:
                 all_paths_fixed_length.append(0)
                 all_paths_fixed_v_normalized.append(None)
                 continue

             p0_orig = all_paths_original_p0[i]
             p1_init = all_paths_initial_p1[i]
             fixed_v = p1_init - p0_orig
             fixed_len = np.linalg.norm(fixed_v)
             fixed_v_norm = None
             if fixed_len > 0 and not scaling_enabled:
                 fixed_v_norm = fixed_v / fixed_len
             elif fixed_len == 0 and not scaling_enabled:
                 print(f"Warning: Path {i+1} initial control points p0 and p1 are identical. Fixed length is 0.")

             all_paths_fixed_length.append(fixed_len)
             all_paths_fixed_v_normalized.append(fixed_v_norm)


        try:
            fill_rgb = ImageColor.getrgb(fill_color)
        except ValueError:
            print(f"Warning: Invalid fill_color '{fill_color}'. Defaulting to white.")
            fill_rgb = (255, 255, 255)

        # --- Easing function definitions ---
        def ease_in(t): return t * t
        def ease_out(t): return 1 - (1 - t) * (1 - t)
        def ease_in_out(t): return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2
        easing_map = {"linear": lambda t: t, "ease_in": ease_in, "ease_out": ease_out, "ease_in_out": ease_in_out}
        apply_easing = easing_map.get(easing_function, lambda t: t)

        # --- Loop through frames ---
        for frame_index in range(total_frames):
            img_frame = Image.new('RGB', (frame_width, frame_height), color=bg_color)
            draw_frame = ImageDraw.Draw(img_frame)

            # --- Loop through paths for the current frame ---
            for path_idx, control_points in enumerate(all_paths_control_points):
                if control_points is None: # Skip invalid/skipped paths
                    continue

                p0_original = all_paths_original_p0[path_idx]
                p1_initial = all_paths_initial_p1[path_idx]
                fixed_length = all_paths_fixed_length[path_idx]
                fixed_v_normalized = all_paths_fixed_v_normalized[path_idx]

                # --- Determine current pivot for this frame ---
                # If dynamic pivot is used, all paths use the same pivot point for this frame.
                # Otherwise, each path uses its own original p0 as the static pivot.
                current_pivot = p0_original # Default to path's own p0 if no dynamic pivot
                if use_dynamic_pivot and pivot_points_adjusted:
                    current_pivot = pivot_points_adjusted[frame_index]

                # --- Determine target point for this path and frame (relative to its p0_original) ---
                target_point_relative_to_p0 = None
                num_control_points = len(control_points)

                if frame_index == 0:
                    target_point_relative_to_p0 = p1_initial - p0_original
                elif num_control_points >= 3:
                    # --- Interpolate to find target point ---
                    num_animation_segments = num_control_points - 2 # Segments p1->p2, p2->p3, ...
                    num_animation_frames = total_frames - 1 # Frames 1 to total_frames-1

                    if num_animation_segments > 0 and num_animation_frames > 0:
                        # Find which segment and t value corresponds to frame_index
                        target_frame_in_animation = frame_index # (since frame_index starts at 0, frame 1 is target 1)
                        cumulative_frames = 0
                        segment_found = False
                        for k in range(num_animation_segments):
                            base_frames_per_segment = num_animation_frames // num_animation_segments
                            remainder_frames = num_animation_frames % num_animation_segments
                            num_steps_this_segment = base_frames_per_segment + (1 if k < remainder_frames else 0)
                            
                            if num_steps_this_segment == 0: continue

                            if target_frame_in_animation <= cumulative_frames + num_steps_this_segment:
                                # Target frame falls within this segment (k)
                                p_segment_start = control_points[k + 1]
                                p_segment_end = control_points[k + 2]
                                frame_within_segment = target_frame_in_animation - cumulative_frames
                                
                                t = frame_within_segment / num_steps_this_segment # Linear t [slightly > 0 to 1.0]
                                eased_t = 0.0

                                if bounce_between > 0.0 and num_steps_this_segment > 1: # Need at least 2 steps to bounce
                                    target_t_near = 1.0 - bounce_between * 0.5
                                    target_t_near = max(0.01, target_t_near) # Avoid division by zero or extreme scaling

                                    # Scale t based on reaching target_t_near
                                    current_t_scaled_for_ease = min(1.0, t / target_t_near) if target_t_near > 0 else 1.0
                                    eased_t_part1 = ease_out(current_t_scaled_for_ease)

                                    # Calculate bounce phase if t is past target_t_near
                                    bounce_phase_t = max(0.0, (t - target_t_near) / (1.0 - target_t_near)) if (1.0 - target_t_near) > 1e-6 else 0.0
                                    
                                    # Apply bounce curve (cosine based)
                                    # Change overshoot factor to directly use bounce_between for amplitude
                                    bounce_curve = 0.5 * (1 - np.cos(bounce_phase_t * np.pi))
                                    
                                    # Blend between eased approach and bounce motion
                                    # Target amplitude of bounce relative to segment length
                                    target_overshoot_displacement = (p_segment_end - p_segment_start) * bounce_between 
                                    
                                    # Calculate position based on eased part and add bounce displacement
                                    position_at_eased_t = p_segment_start + (p_segment_end - p_segment_start) * eased_t_part1
                                    
                                    # Direction of bounce is typically along the segment direction
                                    segment_vector = p_segment_end - p_segment_start
                                    segment_direction = segment_vector / np.linalg.norm(segment_vector) if np.linalg.norm(segment_vector) > 0 else np.array([0,0])

                                    # Apply bounce displacement along the segment direction
                                    bounce_displacement_vector = segment_direction * bounce_curve * np.linalg.norm(target_overshoot_displacement) * bounce_phase_t
                                    p_interpolated = position_at_eased_t + bounce_displacement_vector
                                else: # No bounce or not enough steps for bounce
                                    eased_t = apply_easing(t)
                                    p_interpolated = p_segment_start + (p_segment_end - p_segment_start) * eased_t

                                target_point_relative_to_p0 = p_interpolated - p0_original
                                segment_found = True
                                break # Found the segment for this frame

                            cumulative_frames += num_steps_this_segment

                        if not segment_found:
                            # Should not happen if logic is correct, but fallback to last point relative to p0
                            target_point_relative_to_p0 = control_points[-1] - p0_original
                            print(f"Warning: Segment not found for frame {frame_index}, path {path_idx}. Using last point.")
                    else:
                        # Not enough segments/frames for animation beyond frame 0
                        target_point_relative_to_p0 = p1_initial - p0_original # Stay at initial target
                else:
                     # Only 2 control points, target stays at p1 relative to p0
                     target_point_relative_to_p0 = p1_initial - p0_original

                if target_point_relative_to_p0 is None:
                    print(f"Warning: Could not determine target point for frame {frame_index}, path {path_idx}. Skipping draw.")
                    continue

                # --- Apply Relative vs Absolute Pivot Logic ---
                draw_start_point = None
                draw_end_point = None
                length_for_draw = 0
                normalized_v_for_draw = None

                if relative:
                    # 1. Calculate the shape's geometry based *only* on its own path, originating at p0_original
                    #    (p0_calc, pn_calc, length_calc, normalized_v_calc)
                    p0_calc = p0_original
                    target_calc = p0_calc + target_point_relative_to_p0
                    v_dir_calc = target_calc - p0_calc
                    dir_length_calc = np.linalg.norm(v_dir_calc)
                    pn_calc = p0_calc # Default end point is start
                    length_calc = 0
                    normalized_v_calc = None

                    if scaling_enabled:
                        if dir_length_calc > 0:
                            pn_calc = target_calc
                            length_calc = dir_length_calc
                            normalized_v_calc = v_dir_calc / length_calc
                    else: # Fixed Length
                        if fixed_length > 0:
                            length_calc = fixed_length
                            if dir_length_calc > 0:
                                normalized_v_calc = v_dir_calc / dir_length_calc
                                pn_calc = p0_calc + normalized_v_calc * length_calc
                            elif fixed_v_normalized is not None:
                                normalized_v_calc = fixed_v_normalized
                                pn_calc = p0_calc + normalized_v_calc * length_calc

                    # 2. Determine the initial offset (once per path, could be cached outside frame loop if performance needed)
                    initial_pivot_point = p0_original # Default if no dynamic pivot used for frame 0
                    if use_dynamic_pivot and pivot_points_adjusted:
                        initial_pivot_point = pivot_points_adjusted[0]
                    initial_offset_vector = p0_original - initial_pivot_point

                    # 3. Apply the *initial* offset to the *current* pivot point to get the draw start point
                    frame_pivot_point = current_pivot # Already determined for this frame
                    draw_start_point = frame_pivot_point + initial_offset_vector

                    # 4. Calculate the draw end point by applying the shape's calculated vector to the draw start point
                    shape_vector = pn_calc - p0_calc
                    draw_end_point = draw_start_point + shape_vector

                    # 5. Set draw parameters
                    length_for_draw = length_calc
                    normalized_v_for_draw = normalized_v_calc

                else: # Absolute pivot positioning (previous logic)
                    # Calculate offset target based on current pivot
                    offset_target = current_pivot + target_point_relative_to_p0

                    # Determine vector, length, end point (pn) based on current_pivot, offset_target, scaling
                    v_dir = offset_target - current_pivot
                    dir_length = np.linalg.norm(v_dir)
                    pn = current_pivot # Default end point is the pivot itself
                    length = 0
                    normalized_v = None

                    if scaling_enabled:
                        if dir_length > 0:
                            pn = offset_target
                            length = dir_length
                            normalized_v = v_dir / length
                    else: # Fixed Length
                        if fixed_length > 0:
                            length = fixed_length
                            if dir_length > 0:
                                normalized_v = v_dir / dir_length
                                pn = current_pivot + normalized_v * length
                            elif fixed_v_normalized is not None:
                                normalized_v = fixed_v_normalized
                                pn = current_pivot + normalized_v * length

                    # Set draw parameters
                    draw_start_point = current_pivot
                    draw_end_point = pn
                    length_for_draw = length
                    normalized_v_for_draw = normalized_v

                # --- Draw the polygon for this path using calculated/offset points ---
                if length_for_draw > 0 and normalized_v_for_draw is not None and draw_start_point is not None and draw_end_point is not None:
                    perp_v = np.array([-normalized_v_for_draw[1], normalized_v_for_draw[0]])
                    
                    # Calculate half-widths for start and end based on inputs
                    half_w_start = perp_v * (shape_width / 2.0)
                    
                    # Use shape_width_end if > 0, otherwise use shape_width
                    end_width = shape_width_end if shape_width_end > 0 else shape_width
                    half_w_end = perp_v * (end_width / 2.0)
                    
                    # Use draw_start_point and draw_end_point for corners with respective widths
                    c1 = tuple((draw_start_point - half_w_start).astype(int))
                    c2 = tuple((draw_start_point + half_w_start).astype(int))
                    c3 = tuple((draw_end_point + half_w_end).astype(int)) # Use end width at the end point
                    c4 = tuple((draw_end_point - half_w_end).astype(int)) # Use end width at the end point

                    draw_frame.polygon([c1, c2, c3, c4], fill=fill_rgb)

            # --- Post-processing for the completed frame ---
            if blur_radius > 0.0:
                img_frame = img_frame.filter(ImageFilter.GaussianBlur(blur_radius))

            current_frame_tensor = pil2tensor(img_frame)

            if trailing > 0.0 and previous_frame_tensor is not None:
                current_frame_tensor = current_frame_tensor + trailing * previous_frame_tensor
                # Normalize after adding trailing to prevent exceeding 1.0 (or clamp)
                max_val = torch.max(current_frame_tensor)
                if max_val > 1.0:
                    current_frame_tensor = current_frame_tensor / max_val # Normalize
                # Alternative: Clamping
                # current_frame_tensor = torch.clamp(current_frame_tensor, 0.0, 1.0)

            previous_frame_tensor = current_frame_tensor.clone() # Store state before intensity multiplication

            current_frame_tensor = current_frame_tensor * intensity
            # Optional: Clamp again after intensity if intensity > 1.0
            # current_frame_tensor = torch.clamp(current_frame_tensor, 0.0) # Clamp min at 0?

            output_images.append(current_frame_tensor)

        # --- Final Output ---
        if not output_images:
            print("Warning: No frames generated. Returning a single blank image.")
            img = Image.new('RGB', (frame_width, frame_height), color=bg_color)
            return (pil2tensor(img),)

        batch_output = torch.cat(output_images, dim=0)
        return (batch_output,)

class DriverOffsetCoordinates:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "driver_coords": ("STRING", {"multiline": False, "forceInput": True}),
                "driven_coords": ("STRING", {"multiline": False, "forceInput": True}),
                "smooth_out": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "delay": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "rotate": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_coords",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes/coords"
    DESCRIPTION = """Applies rotated, smoothed, and delayed offsets from driver coordinates to driven coordinates."""

    def execute(self, driver_coords, driven_coords, smooth_out, delay, rotate):
        try:
            # Use replace("'", '"') for potentially malformed JSON strings
            D_orig = json.loads(driver_coords.replace("'", '"'))
            Dn = json.loads(driven_coords.replace("'", '"'))
        except json.JSONDecodeError as e:
            print(f"DriverOffsetCoordinates Error: Invalid JSON input - {e}")
            return (driven_coords,)
        except Exception as e:
            print(f"DriverOffsetCoordinates Error: Could not parse coordinates - {e}")
            return (driven_coords,)

        len_dn = len(Dn)
        if len_dn == 0:
            return (driven_coords,)

        len_d_orig = len(D_orig)
        if len_d_orig == 0:
            print("DriverOffsetCoordinates Warning: Driver coordinates are empty. Returning driven coordinates unchanged.")
            return (driven_coords,)

        # --- Rotation Step --- (Operate on a copy)
        D = [coord.copy() for coord in D_orig] # Work on a copy for rotation
        len_d = len(D)

        if rotate != 0 and len_d >= 2:
            try:
                pivot_x = float(D[0]['x'])
                pivot_y = float(D[0]['y'])
                angle_rad = math.radians(rotate)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                for j in range(1, len_d):
                    px = float(D[j]['x'])
                    py = float(D[j]['y'])

                    rel_x = px - pivot_x
                    rel_y = py - pivot_y

                    new_rel_x = rel_x * cos_a - rel_y * sin_a
                    new_rel_y = rel_x * sin_a + rel_y * cos_a

                    D[j]['x'] = new_rel_x + pivot_x
                    D[j]['y'] = new_rel_y + pivot_y
            except KeyError as e:
                print(f"DriverOffsetCoordinates Error: Missing 'x' or 'y' key during rotation - {e}")
                return (driven_coords,) # Abort if keys are missing
            except ValueError as e:
                print(f"DriverOffsetCoordinates Error: Cannot convert coordinate to float during rotation - {e}")
                return (driven_coords,) # Abort if conversion fails

        # --- Padding Step --- (Use potentially rotated D)
        if len_d < len_dn:
            print(f"DriverOffsetCoordinates Info: Driver coords shorter ({len_d}) than driven ({len_dn}). Padding driver with last coordinate.")
            if len_d > 0:
                 last_driver_coord = D[-1]
                 D.extend([last_driver_coord.copy() for _ in range(len_dn - len_d)])
            else:
                 # This case should be impossible now due to earlier check
                 print("DriverOffsetCoordinates Warning: Driver coords empty after rotation (should not happen), padding with {'x':0, 'y':0}.")
                 D.extend([{'x':0.0, 'y':0.0}] * len_dn)
            len_d = len(D)

        # --- Smoothing Step --- (Use potentially rotated and padded D)
        SmoothD = [None] * len_d
        if len_d > 0:
            try:
                # Ensure coords are floats for calculation
                SmoothD[0] = {'x': float(D[0]['x']), 'y': float(D[0]['y'])}
                alpha = 1.0 - smooth_out
                for j in range(1, len_d):
                    prev_smooth_x = SmoothD[j-1]['x']
                    prev_smooth_y = SmoothD[j-1]['y']
                    # Ensure current coords are floats
                    current_x = float(D[j]['x'])
                    current_y = float(D[j]['y'])
                    smooth_x = alpha * current_x + (1.0 - alpha) * prev_smooth_x
                    smooth_y = alpha * current_y + (1.0 - alpha) * prev_smooth_y
                    SmoothD[j] = {'x': smooth_x, 'y': smooth_y}
            except KeyError as e:
                print(f"DriverOffsetCoordinates Error: Missing 'x' or 'y' key during smoothing - {e}")
                return (driven_coords,)
            except ValueError as e:
                print(f"DriverOffsetCoordinates Error: Cannot convert coordinate to float during smoothing - {e}")
                return (driven_coords,)
        else:
             SmoothD = []

        # --- Offset Application Step ---
        OutputCoords = [None] * len_dn
        RefOffsetX = SmoothD[0]['x'] if len(SmoothD) > 0 else 0.0
        RefOffsetY = SmoothD[0]['y'] if len(SmoothD) > 0 else 0.0

        for i in range(len_dn):
            try:
                # Ensure driven coords are floats
                current_driven_x = float(Dn[i]['x'])
                current_driven_y = float(Dn[i]['y'])

                if i < delay:
                    # Keep original driven coord (as float)
                    OutputCoords[i] = {'x': current_driven_x, 'y': current_driven_y}
                else:
                    # Apply offset after delay period
                    driver_idx = i - delay
                    if 0 <= driver_idx < len(SmoothD):
                        driver_smooth_x = SmoothD[driver_idx]['x']
                        driver_smooth_y = SmoothD[driver_idx]['y']
                    elif len(SmoothD) > 0:
                        print(f"DriverOffsetCoordinates Warning: driver_idx {driver_idx} out of bounds for SmoothD (len {len(SmoothD)}). Using last.")
                        driver_smooth_x = SmoothD[-1]['x']
                        driver_smooth_y = SmoothD[-1]['y']
                    else:
                        driver_smooth_x = 0.0
                        driver_smooth_y = 0.0

                    offset_x = driver_smooth_x - RefOffsetX
                    offset_y = driver_smooth_y - RefOffsetY

                    output_x = current_driven_x + offset_x
                    output_y = current_driven_y + offset_y

                    OutputCoords[i] = {'x': output_x, 'y': output_y}
            except KeyError as e:
                print(f"DriverOffsetCoordinates Error: Missing 'x' or 'y' key during offset application - {e}")
                # Return partially processed coords or original driven? Return original for safety.
                return (driven_coords,)
            except ValueError as e:
                print(f"DriverOffsetCoordinates Error: Cannot convert coordinate to float during offset application - {e}")
                return (driven_coords,)

        # Format output as JSON string
        output_json = json.dumps(OutputCoords, separators=(',', ':'))

        return (output_json,)