import numpy as np
import torch
import math
import os
import importlib
import sys
import logging
from io import BytesIO

from PIL import ImageDraw, ImageFont, Image
from comfy.cli_args import args
from comfy.utils import ProgressBar, common_upscale, tiled_scale_multidim
from comfy import model_management
from comfy_api.latest import io, InputImpl, Types, ui
from fractions import Fraction
import folder_paths

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Shared VideoHelperSuite / video-folder helpers
# ============================================================================
_VHS_NODES = None
_VHS_LOAD_VIDEO = None

VIDEO_EXTENSIONS = {".webm", ".mp4", ".mkv", ".gif", ".mov"}

def get_vhs_nodes():
    """
    Locate and return the VideoHelperSuite 'videohelpersuite' module.

    Supports:
      - Standard ComfyUI custom-node installation
      - Alternate lowercase package name
      - Windows / already-loaded-module fallback

    The successfully resolved module is cached.
    """
    global _VHS_NODES

    if _VHS_NODES is not None:
        return _VHS_NODES

    module_names = (
        "ComfyUI-VideoHelperSuite.videohelpersuite",
        "comfyui-videohelpersuite.videohelpersuite",
    )

    # Normal import path.
    for module_name in module_names:
        try:
            _VHS_NODES = importlib.import_module(module_name)
            return _VHS_NODES
        except ImportError:
            continue

    # Windows / already-loaded-module fallback.
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue

        if module_name.endswith("videohelpersuite"):
            _VHS_NODES = module
            return _VHS_NODES

    raise ImportError("This node requires ComfyUI-VideoHelperSuite to be installed.")


def get_vhs_load_video():
    """
    Locate and return VideoHelperSuite's load_video() function.

    This deliberately imports the load_video_nodes submodule directly
    rather than assuming it has already been exposed as an attribute
    of the parent videohelpersuite module.
    """
    global _VHS_LOAD_VIDEO

    if _VHS_LOAD_VIDEO is not None:
        return _VHS_LOAD_VIDEO

    module_names = (
        "ComfyUI-VideoHelperSuite.videohelpersuite.load_video_nodes",
        "comfyui-videohelpersuite.videohelpersuite.load_video_nodes",
    )

    # Normal import path.
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)

            loader = getattr(
                module,
                "load_video",
                None,
            )

            if loader is not None:
                _VHS_LOAD_VIDEO = loader
                return _VHS_LOAD_VIDEO

        except ImportError:
            continue

    # Windows / already-loaded-module fallback.
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue

        if module_name.endswith(
            "videohelpersuite.load_video_nodes"
        ):
            loader = getattr(
                module,
                "load_video",
                None,
            )

            if loader is not None:
                _VHS_LOAD_VIDEO = loader
                return _VHS_LOAD_VIDEO

    raise ImportError("This node requires ComfyUI-VideoHelperSuite with load_video_nodes.load_video().")

def get_videos_from_folder(folder):
    """
    Return supported video files from a folder.
    Each result is:
        (absolute/full filepath, filename)
    Results are sorted by filename to preserve deterministic ordering.
    """
    videos = []

    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)

        if not os.path.isfile(filepath):
            continue

        extension = os.path.splitext(filename)[1].lower()

        if extension in VIDEO_EXTENSIONS:
            videos.append(
                (
                    filepath,
                    filename,
                )
            )
    return videos

def add_video_label(video_tensor, filename):
    """
    Add a filename label above a video tensor.

    Supports both:
        (frames, height, width, channels)
    and:
        (height, width, channels)
    """
    if video_tensor.dim() == 4:
        _, height, width, channels = video_tensor.shape
    else:
        height, width, channels = video_tensor.shape

    label_text = os.path.splitext(filename)[0]

    font_size = max(
        16,
        width // 20,
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            font_size,
        )
    except OSError:
        font = ImageFont.load_default()

    dummy_img = Image.new(
        "RGB",
        (width, 10),
        (0, 0, 0),
    )

    draw = ImageDraw.Draw(dummy_img)

    text_bbox = draw.textbbox(
        (0, 0),
        label_text,
        font=font,
    )

    extra_padding = max(
        12,
        font_size // 2,
    )

    label_height = (
        text_bbox[3]
        - text_bbox[1]
        + extra_padding
    )

    label_img = Image.new(
        "RGB",
        (width, label_height),
        (0, 0, 0),
    )

    draw = ImageDraw.Draw(label_img)

    text_width = text_bbox[2] - text_bbox[0]

    draw.text(
        (
            width // 2 - text_width // 2,
            4,
        ),
        label_text,
        font=font,
        fill=(255, 255, 255),
    )

    label_np = (
        np.asarray(label_img)
        .astype(np.float32)
        / 255.0
    )

    label_tensor = torch.from_numpy(label_np)

    if channels == 1:
        label_tensor = label_tensor.mean(
            dim=2,
            keepdim=True,
        )

    elif channels == 4:
        alpha = torch.ones(
            (
                label_height,
                width,
                1,
            ),
            dtype=label_tensor.dtype,
        )

        label_tensor = torch.cat(
            (
                label_tensor,
                alpha,
            ),
            dim=2,
        )

    if video_tensor.dim() == 4:
        label_tensor = label_tensor.unsqueeze(0).expand(
            video_tensor.shape[0],
            -1,
            -1,
            -1,
        )

        video_tensor = torch.cat(
            (
                label_tensor,
                video_tensor,
            ),
            dim=1,
        )

    else:
        video_tensor = torch.cat(
            (
                label_tensor,
                video_tensor,
            ),
            dim=0,
        )

    return video_tensor

def hash_video_folder(folder):
    """
    Return VideoHelperSuite's hash for a video folder.

    This is shared by LoadVideosFromFolder and LoadVideosFromFolderList
    so that their cache invalidation behavior remains consistent.
    """
    vhs_nodes = get_vhs_nodes()
    return vhs_nodes.utils.hash_path(folder)

class LoadVideosFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    "STRING",
                    {
                        "default": "X://insert/path/",
                    },
                ),

                "force_rate": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 60,
                        "step": 1,
                        "disable": 0,
                    },
                ),

                "custom_width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "disable": 0,
                    },
                ),

                "custom_height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "disable": 0,
                    },
                ),

                "frame_load_cap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "disable": 0,
                    },
                ),

                "skip_first_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                    },
                ),

                "select_every_nth": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                    },
                ),

                "output_type": (
                    [
                        "batch",
                        "grid",
                    ],
                    {
                        "default": "batch",
                    },
                ),

                "grid_max_columns": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "disable": 1,
                    },
                ),

                "add_label": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },

            "hidden": {
                "force_size": "STRING",
                "unique_id": "UNIQUE_ID",
            },
        }

    CATEGORY = "KJNodes/misc"

    RETURN_TYPES = (
        "IMAGE",
    )

    RETURN_NAMES = (
        "IMAGE",
    )

    FUNCTION = "load_video"

    def load_video(
        self,
        output_type,
        grid_max_columns,
        add_label=False,
        **kwargs,
    ):
        if (
            kwargs.get("video")
            and not os.path.isabs(kwargs["video"])
            and args.base_directory
        ):
            kwargs["video"] = os.path.join(
                args.base_directory,
                kwargs["video"],
            )

        video_folder = kwargs["video"]

        if not os.path.isdir(video_folder):
            raise ValueError(
                f"Video folder does not exist or is not a directory: "
                f"{video_folder}"
            )

        videos = get_videos_from_folder(
            video_folder,
        )

        if not videos:
            raise ValueError(
                f"No supported video files found in folder: "
                f"{video_folder}"
            )

        vhs_load_video = get_vhs_load_video()

        loaded_videos = []

        # Remove the folder path before forwarding kwargs to VHS.
        kwargs.pop("video")

        for filepath, filename in videos:

            video_tensor = vhs_load_video(
                video=filepath,
                **kwargs,
            )[0]

            if add_label:
                video_tensor = add_video_label(
                    video_tensor,
                    filename,
                )

            loaded_videos.append(
                video_tensor
            )

        if output_type == "batch":

            out_tensor = torch.cat(
                loaded_videos
            )

        elif output_type == "grid":

            rows = (
                len(loaded_videos)
                + grid_max_columns
                - 1
            ) // grid_max_columns

            # Pad the last row if needed.
            total_slots = (
                rows
                * grid_max_columns
            )

            while len(loaded_videos) < total_slots:
                loaded_videos.append(
                    torch.zeros_like(
                        loaded_videos[0]
                    )
                )

            row_tensors = []

            for row_idx in range(rows):

                start_idx = (
                    row_idx
                    * grid_max_columns
                )

                end_idx = (
                    start_idx
                    + grid_max_columns
                )

                row_videos = loaded_videos[
                    start_idx:end_idx
                ]

                # Pad all videos in this row to the same height.
                heights = [
                    v.shape[1]
                    for v in row_videos
                ]

                max_height = max(
                    heights
                )

                padded_row_videos = []

                for v in row_videos:

                    pad_height = (
                        max_height
                        - v.shape[1]
                    )

                    if pad_height > 0:

                        # (frames, H, W, C)
                        # or
                        # (H, W, C)
                        if v.dim() == 4:

                            v = torch.nn.functional.pad(
                                v,
                                (
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    pad_height,
                                    0,
                                    0,
                                ),
                            )

                        else:

                            v = torch.nn.functional.pad(
                                v,
                                (
                                    0,
                                    0,
                                    0,
                                    0,
                                    pad_height,
                                    0,
                                ),
                            )

                    padded_row_videos.append(v)

                row_tensor = torch.cat(
                    padded_row_videos,
                    dim=2,
                )

                row_tensors.append(
                    row_tensor
                )

            out_tensor = torch.cat(
                row_tensors,
                dim=1,
            )

        else:
            raise ValueError(
                f"Unknown output_type: {output_type}"
            )

        return (
            out_tensor,
        )

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return hash_video_folder(video)

class LoadVideosFromFolderList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadVideosFromFolderList",
            display_name="Load Videos From Folder (List)",
            description="Loads all supported video files from a folder and returns each video as a separate IMAGE batch, along with matching audio.",
            category="KJNodes/misc",
            search_aliases=[
                "load videos",
                "load video folder",
                "load videos folder",
                "video folder",
                "video list",
                "video batch list",
            ],

            inputs=[
                io.String.Input(
                    "video",
                    default="X://insert/path/",
                    tooltip="Folder containing the videos to load.",
                ),

                io.DynamicCombo.Input(
                    "if_no_audio",
                    options=[
                        io.DynamicCombo.Option("return empty audio", []),
                        io.DynamicCombo.Option("return None", []),
                    ],
                    tooltip=(
                        "What to return when a video contains no audio stream. "
                        "'return empty audio' provides a valid zero-length AUDIO object; "
                        "'return None' provides None."
                    ),
                ),
                io.Float.Input(
                    "force_rate",
                    default=0,
                    min=0,
                    max=60,
                    step=1,
                    tooltip="Force a specific frame rate. 0 uses the source frame rate.",
                ),
                io.Int.Input(
                    "custom_width",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip="Resize videos to this width. 0 preserves the source width.",
                ),
                io.Int.Input(
                    "custom_height",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip="Resize videos to this height. 0 preserves the source height.",
                ),
                io.Int.Input(
                    "frame_load_cap",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                    tooltip="Maximum number of frames to load from each video. 0 loads all available frames.",
                ),
                io.Int.Input(
                    "skip_first_frames",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                    tooltip="Number of frames to skip at the beginning.",
                ),
                io.Int.Input(
                    "select_every_nth",
                    default=1,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Load every Nth frame.",
                ),
                io.Boolean.Input(
                    "add_label",
                    default=False,
                    tooltip="Add the filename above each video.",
                ),
            ],

            outputs=[
                io.Image.Output(
                    display_name="image_batches_list",
                    is_output_list=True,
                ),

                io.Audio.Output(
                    display_name="audio_list",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        if_no_audio,
        force_rate,
        custom_width,
        custom_height,
        frame_load_cap,
        skip_first_frames,
        select_every_nth,
        add_label,
    ):
        if not os.path.isdir(video):
            raise ValueError(f"Video folder does not exist or is not a directory: {video}")

        videos = get_videos_from_folder(video)

        if not videos:
            raise ValueError(f"No supported video files found in folder: {video}")

        vhs_load_video = get_vhs_load_video()

        loaded_videos = []
        loaded_audio = []

        for filepath, filename in videos:

            result = vhs_load_video(
                video=filepath,
                force_rate=force_rate,
                custom_width=custom_width,
                custom_height=custom_height,
                frame_load_cap=frame_load_cap,
                skip_first_frames=skip_first_frames,
                select_every_nth=select_every_nth,
            )

            video_tensor = result[0]

            try:
                # VHS returns a LazyAudioMap here. Force it to resolve while we still control the exception handling.
                audio = dict(result[2])

            except Exception as e:

                error_text = str(e)

                if (
                    "Output file does not contain any stream"
                    in error_text
                ):

                    if if_no_audio == "return empty audio":

                        audio = {
                            "waveform": torch.zeros(
                                (
                                    1,
                                    2,
                                    0,
                                ),
                                dtype=torch.float32,
                            ),
                            "sample_rate": 44100,
                        }

                    else:
                        audio = None

                else:
                    # Do not hide genuine audio/extraction errors.
                    raise

            if add_label:
                video_tensor = add_video_label(video_tensor, filename)

            loaded_videos.append(video_tensor)
            loaded_audio.append(audio)

        return io.NodeOutput(loaded_videos, loaded_audio)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return hash_video_folder(video)

class EncodeVideoComponents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        position_options = ["center", "top", "bottom", "left", "right"]
        options = [
            io.DynamicCombo.Option(key="stretch", inputs=[]),
            io.DynamicCombo.Option(key="resize", inputs=[]),
            io.DynamicCombo.Option(key="total_pixels", inputs=[]),
            io.DynamicCombo.Option(key="crop", inputs=[
                io.Combo.Input("crop_position", options=position_options, tooltip="Position to crop from."),
            ]),
            io.DynamicCombo.Option(key="pad", inputs=[
                io.String.Input("pad_color", default="0, 0, 0", tooltip="Color to use for padding."),
                io.Combo.Input("pad_position", options=position_options, tooltip="Position to align the image within the padded area."),
            ]),
            io.DynamicCombo.Option(key="pad_edge", inputs=[
                io.Combo.Input("pad_position", options=position_options, tooltip="Position to align the image within the padded area."),
            ]),
            io.DynamicCombo.Option(key="pad_edge_pixel", inputs=[
                io.Combo.Input("pad_position", options=position_options, tooltip="Position to align the image within the padded area."),
            ]),
            io.DynamicCombo.Option(key="pillarbox_blur", inputs=[
                io.Combo.Input("pad_position", options=position_options, tooltip="Position to align the image within the padded area."),
            ]),
        ]
        return io.Schema(
            node_id="EncodeVideoComponents",
            search_aliases=["video to latent", "encode video", "vae encode video"],
            display_name="Encode Video Components",
            category="KJNodes/image",
            description="Extracts video frames, resizes them, and encodes with a VAE directly, avoiding storing the full image tensor.",
            inputs=[
                io.Video.Input("video", tooltip="The video to extract and encode."),
                io.Vae.Input("vae", tooltip="The VAE model to use for encoding."),
                io.Int.Input("width", default=768, min=0, max=16384, step=2, tooltip="Target width for the frames before encoding. 0 = original width."),
                io.Int.Input("height", default=512, min=0, max=16384, step=2, tooltip="Target height for the frames before encoding. 0 = original height."),
                io.Int.Input("max_frames", default=0, min=0, max=999999, step=1, tooltip="Maximum number of frames. 0 = no limit."),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos", tooltip="Interpolation method for resizing."),
                io.DynamicCombo.Input(
                    "keep_proportion",
                    options=options,
                    display_name="Keep Proportion",
                    tooltip="How to handle aspect ratio mismatch when resizing.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="frame_count", tooltip="Number pixel space frames after any possible cropping"),
            ],
        )

    @staticmethod
    def _compute_resize_params(mode, position, width, height, src_w, src_h):
        """Compute target resize dimensions, crop region, and padding from keep_proportion mode."""
        if width == 0:
            width = src_w
        if height == 0:
            height = src_h
        pillarbox_blur = mode == "pillarbox_blur"
        pad_left = pad_right = pad_top = pad_bottom = 0
        crop_region = None  # (x, y, crop_w, crop_h) or None

        if mode in ["resize", "total_pixels"] or mode.startswith("pad") or pillarbox_blur:
            if mode == "total_pixels":
                total_pixels = width * height
                aspect_ratio = src_w / src_h
                new_height = int(math.sqrt(total_pixels / aspect_ratio))
                new_width = int(math.sqrt(total_pixels * aspect_ratio))
            else:
                ratio = min(width / src_w, height / src_h)
                new_width = round(src_w * ratio)
                new_height = round(src_h * ratio)

            if mode.startswith("pad") or pillarbox_blur:
                if position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height

        if mode == "crop":
            old_aspect = src_w / src_h
            new_aspect = width / height
            if old_aspect > new_aspect:
                crop_w = round(src_h * new_aspect)
                crop_h = src_h
            else:
                crop_w = src_w
                crop_h = round(src_w / new_aspect)
            if position == "center":
                x = (src_w - crop_w) // 2
                y = (src_h - crop_h) // 2
            elif position == "top":
                x = (src_w - crop_w) // 2
                y = 0
            elif position == "bottom":
                x = (src_w - crop_w) // 2
                y = src_h - crop_h
            elif position == "left":
                x = 0
                y = (src_h - crop_h) // 2
            elif position == "right":
                x = src_w - crop_w
                y = (src_h - crop_h) // 2
            crop_region = (x, y, crop_w, crop_h)

        return width, height, crop_region, (pad_left, pad_right, pad_top, pad_bottom)

    @classmethod
    def execute(cls, video, vae, width, height, max_frames, upscale_method, keep_proportion) -> io.NodeOutput:
        import av
        import itertools

        mode = keep_proportion["keep_proportion"]
        position = keep_proportion.get("crop_position") or keep_proportion.get("pad_position", "center")
        pad_color = keep_proportion.get("pad_color", "0, 0, 0")
        target_dtype = vae.vae_dtype

        # Access VideoFromFile internals for efficient per-frame decode
        source = video.get_stream_source()
        start_time = getattr(video, '_VideoFromFile__start_time', 0)
        duration = getattr(video, '_VideoFromFile__duration', 0)

        # Get frame count for progress bar, capped by max_frames
        try:
            total_frames = video.get_frame_count()
        except (ValueError, AttributeError):
            total_frames = 0
        if max_frames > 0 and total_frames > 0:
            total_frames = min(total_frames, max_frames)
        pbar = ProgressBar(total_frames) if total_frames > 0 else None

        # Lanczos requires PIL (CPU-only), all other methods use torch on GPU
        use_gpu = upscale_method != "lanczos"
        device = model_management.get_torch_device() if use_gpu else torch.device("cpu")

        # --- Decode video frames with per-frame resize + dtype cast ---
        with av.open(source, mode='r') as container:
            video_stream = container.streams.video[0]
            start_pts = int(start_time / video_stream.time_base)
            end_pts = int((start_time + duration) / video_stream.time_base) if duration else 0
            container.seek(start_pts, stream=video_stream)

            res_w, res_h, crop_region, padding = None, None, None, (0, 0, 0, 0)
            frames = []
            for frame in container.decode(video_stream):
                if frame.pts < start_pts:
                    continue
                if duration and frame.pts >= end_pts:
                    break
                if max_frames > 0 and len(frames) >= max_frames:
                    break

                if res_w is None:
                    src_h, src_w = frame.height, frame.width
                    res_w, res_h, crop_region, padding = cls._compute_resize_params(
                        mode, position, width, height, src_w, src_h
                    )

                # Decode to tensor and normalize
                img = torch.from_numpy(frame.to_ndarray(format='rgb24')).to(device=device, dtype=torch.float32) / 255.0

                # Crop if needed (before resize)
                if crop_region is not None:
                    cx, cy, cw, ch = crop_region
                    img = img[cy:cy+ch, cx:cx+cw, :]

                # Resize (GPU for torch-native methods, CPU/PIL for lanczos)
                img = common_upscale(
                    img.unsqueeze(0).movedim(-1, 1), res_w, res_h, upscale_method, crop="disabled"
                ).movedim(1, -1).squeeze(0).to(dtype=target_dtype, device="cpu")

                frames.append(img)
                if pbar is not None:
                    pbar.update(1)

            frame_rate = video_stream.average_rate if video_stream.average_rate else 1

        s = torch.stack(frames) if frames else torch.zeros(0, height, width, 3, dtype=target_dtype)

        # Pad logic (applied on the full stack since padding modes like pillarbox_blur need all frames)
        pillarbox_blur = mode == "pillarbox_blur"
        pad_left, pad_right, pad_top, pad_bottom = padding
        if (mode.startswith("pad") or pillarbox_blur) and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            pad_mode = (
                "pillarbox_blur" if pillarbox_blur else
                "edge" if mode == "pad_edge" else
                "edge_pixel" if mode == "pad_edge_pixel" else
                "color"
            )
            s, _ = ImagePadKJ.pad(None, s, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, pad_mode)

        # Trim frames to a count valid for the VAE's temporal compression
        try:
            temporal_compress = vae.downscale_ratio[0]
            temporal_decompress = vae.upscale_ratio[0]
            valid_frames = temporal_decompress(temporal_compress(s.shape[0]))
            if valid_frames < s.shape[0]:
                logging.warning(f"[EncodeVideoComponents] Trimming {s.shape[0] - valid_frames} frames ({s.shape[0]} -> {valid_frames}) to match VAE temporal compression ratio")
                s = s[:valid_frames]
        except (TypeError, IndexError):
            pass

        t = vae.encode(s)

        # --- Extract audio in a separate pass ---
        audio = None
        if isinstance(source, BytesIO):
            source.seek(0)
        with av.open(source, mode='r') as container:
            if len(container.streams.audio):
                audio_stream = container.streams.audio[-1]
                if start_time > 0:
                    audio_start_pts = int(start_time / audio_stream.time_base)
                    container.seek(audio_start_pts, stream=audio_stream)
                audio_frames = []
                resample = av.audio.resampler.AudioResampler(format='fltp').resample
                aframes = itertools.chain.from_iterable(
                    map(resample, container.decode(audio_stream))
                )
                has_first_frame = False
                for aframe in aframes:
                    offset_seconds = start_time - aframe.time
                    to_skip = int(offset_seconds * audio_stream.sample_rate)
                    if to_skip < aframe.samples:
                        has_first_frame = True
                        break
                if has_first_frame:
                    audio_frames.append(aframe.to_ndarray()[..., to_skip:])
                    for aframe in aframes:
                        if duration and aframe.time > start_time + duration:
                            break
                        audio_frames.append(aframe.to_ndarray())
                if audio_frames:
                    audio_data = np.concatenate(audio_frames, axis=1)
                    if duration:
                        audio_data = audio_data[..., :int(duration * audio_stream.sample_rate)]
                    audio = {
                        "waveform": torch.from_numpy(audio_data).unsqueeze(0),
                        "sample_rate": int(audio_stream.sample_rate) if audio_stream.sample_rate else 1,
                    }

        return io.NodeOutput({"samples": t}, audio, float(frame_rate), s.shape[0])

class DecodeAndSaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DecodeAndSaveVideo",
            search_aliases=["video to latent", "decode video"],
            display_name="Decode and Save Video",
            category="KJNodes/image",
            description="Decodes video frames and audio from latent representations, combines them, and saves as a video file, without keeping intermediate images in memory.",
            inputs=[
                io.Latent.Input("video_latent", tooltip="The latent representation of the video frames."),
                io.Latent.Input("audio_latent", optional=True, tooltip="The latent representation of the audio frames."),
                io.Float.Input("fps", default=25.0, min=0.0, max=999.0, step=0.01, tooltip="Frame rate for the output video."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto", tooltip="The codec to use for the video."),
                io.Vae.Input("video_vae", tooltip="The VAE model to use for encoding."),
                io.Vae.Input("audio_vae", optional=True, tooltip="The VAE model to use for decoding audio."),
                io.DynamicCombo.Input("tiling", options=[
                    io.DynamicCombo.Option(key="disabled", inputs=[]),
                    io.DynamicCombo.Option(key="enabled", inputs=[
                        io.Int.Input("tile_size", default=512, min=64, max=4096, step=32, tooltip="Size of the tiles to decode. Smaller tiles use less memory but take more time."),
                        io.Int.Input("overlap", default=64, min=0, max=4096, step=32, tooltip="Amount of overlap between tiles. Higher overlap can improve quality at the edges of tiles but uses more memory and takes more time."),
                        io.Int.Input("temporal_size", default=4096, min=8, max=4096, step=4, tooltip="Only used for video VAEs: Amount of frames to decode at a time. Higher value than number of frames = disabled"),
                        io.Int.Input("temporal_overlap", default=16, min=4, max=4096, step=4, tooltip="Only used for video VAEs: Amount of frames to overlap. Higher overlap can improve quality at the edges of temporal tiles but uses more memory and takes more time."),
                    ]),
                ]),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video_latent, video_vae, filename_prefix, format, codec, tiling, audio_latent=None, audio_vae=None, fps=25.0) -> io.NodeOutput:
        if tiling["tiling"] == "enabled":
            tile_size = tiling["tile_size"]
            overlap = tiling["overlap"]
            temporal_size = tiling["temporal_size"]
            temporal_overlap = tiling["temporal_overlap"]

            if tile_size < overlap * 4:
                overlap = tile_size // 4
            if temporal_size < temporal_overlap * 2:
                temporal_overlap = temporal_overlap // 2
            temporal_compression = video_vae.temporal_compression_decode()
            if temporal_compression is not None:
                temporal_size = max(2, temporal_size // temporal_compression)
                temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
            else:
                temporal_size = None
                temporal_overlap = None

            compression = video_vae.spacial_compression_decode()

            images = cls.decode_tiled(video_vae, video_latent["samples"],
                                      tile_t=max(2, temporal_size),
                                      tile_x=tile_size // compression,
                                      tile_y=tile_size // compression,
                                      overlap=(temporal_overlap if temporal_overlap is not None else 1, max(1, overlap // compression), max(1, overlap // compression)),
            ).movedim(1, -1)
            if len(images.shape) == 5: #Combine batches
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            images = cls.decode_video(video_vae, video_latent)

        if audio_latent is not None:
            if audio_vae is None:
                raise ValueError("Audio VAE must be provided if audio latent is provided.")
            audio = cls.decode_audio(audio_latent, audio_vae)
        else:
            audio = None

        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)))
        file, subfolder = cls.save_video(video, filename_prefix, format, codec)

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))

    @classmethod
    def decode_video(cls, vae, samples):
        samples_in = samples["samples"]
        if samples_in.is_nested:
            samples_in = samples_in.unbind()[0]

        vae.throw_exception_if_invalid()
        pixel_samples = None
        do_tile = False
        if vae.latent_dim == 2 and samples_in.ndim == 5:
            samples_in = samples_in[:, :, 0]
        try:
            memory_used = vae.memory_used_decode(samples_in.shape, vae.vae_dtype)
            model_management.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=True)
            free_memory = vae.patcher.get_free_memory(vae.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x:x+batch_number].to(vae.vae_dtype).to(vae.device)
                out = vae.process_output(vae.first_stage_model.decode(samples).to(vae.output_device).to(torch.float16))
                if pixel_samples is None:
                    pixel_samples = torch.empty((samples_in.shape[0],) + tuple(out.shape[1:]), device=vae.output_device, dtype=out.dtype)
                pixel_samples[x:x+batch_number] = out
        except Exception as e:
            model_management.raise_non_oom(e)
            logging.warning("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            do_tile = True

        if do_tile:
            dims = samples_in.ndim - 2
            if dims == 1 or cls.extra_1d_channel is not None:
                pixel_samples = vae.decode_tiled_1d(samples_in)
            elif dims == 2:
                pixel_samples = vae.decode_tiled_2d(samples_in)
            elif dims == 3:
                tile = 256 // vae.spacial_compression_decode()
                overlap = tile // 4
                pixel_samples = vae.decode_tiled_3d(samples_in, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))

        pixel_samples = pixel_samples.to(vae.output_device).movedim(1,-1)

        if len(pixel_samples.shape) == 5: #Combine batches
            pixel_samples = pixel_samples.reshape(-1, pixel_samples.shape[-3], pixel_samples.shape[-2], pixel_samples.shape[-1])
        return pixel_samples

    @classmethod
    def decode_tiled(cls, vae, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
        vae.throw_exception_if_invalid()
        memory_used = vae.memory_used_decode(samples.shape, vae.vae_dtype)
        model_management.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)
        decode_fn = lambda a: vae.first_stage_model.decode(a.to(vae.vae_dtype).to(vae.device)).to(torch.float16)
        return vae.process_output(tiled_scale_multidim(samples, decode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap,
                                                       upscale_amount=vae.upscale_ratio, out_channels=vae.output_channels, index_formulas=vae.upscale_index_formula, output_device=vae.output_device))


    @classmethod
    def decode_audio(cls, samples, audio_vae):
        audio_latent = samples["samples"]
        if audio_latent.is_nested:
            audio_latent = audio_latent.unbind()[-1]
        audio = audio_vae.decode(audio_latent)
        # Post-PR #13486: audio_vae is a comfy.sd.VAE wrapper returning channels-last (BTC).
        # Pre-PR: audio_vae is a raw AudioVAE returning channels-first (BCT).
        if hasattr(audio_vae, "first_stage_model"):
            audio = audio.movedim(-1, 1)
        audio = audio.to(audio_latent.device)
        output_audio_sample_rate = getattr(
            audio_vae,
            "audio_sample_rate_output",
            getattr(audio_vae, "output_sample_rate", None),
        )
        if output_audio_sample_rate is None:
            output_audio_sample_rate = getattr(
                getattr(audio_vae, "first_stage_model", None), "output_sample_rate", 44100
            )
        return {"waveform": audio, "sample_rate": int(output_audio_sample_rate)}

    @classmethod
    def save_video(cls, video, filename_prefix, format, codec) -> io.NodeOutput:
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=saved_metadata
        )
        return file, subfolder
