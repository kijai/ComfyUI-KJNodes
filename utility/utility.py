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


BBOX_TYPES = "BBOX,BOUNDING_BOX"

def normalize_bboxes(bboxes, bbox_format="xywh") -> List[tuple]:
    """Accept KJNodes BBOX tuples and/or native BOUNDING_BOX dicts, single or nested,
    and return a flat list of 4-tuples in bbox_format ("xywh" or "xyxy").
    Tuples are assumed to already be in bbox_format, dicts are converted."""
    if isinstance(bboxes, (torch.Tensor, np.ndarray)):
        bboxes = bboxes.tolist()
    if isinstance(bboxes, dict):
        bboxes = [bboxes]
    elif isinstance(bboxes, (list, tuple)):
        if len(bboxes) == 4 and all(isinstance(v, (int, float)) for v in bboxes):
            bboxes = [bboxes]
    else:
        raise ValueError(f"Unsupported bbox input: {type(bboxes).__name__}")

    out = []
    for bbox in bboxes:
        if isinstance(bbox, (torch.Tensor, np.ndarray)):
            bbox = bbox.tolist()
        if isinstance(bbox, (list, tuple)) and bbox and not all(isinstance(v, (int, float)) for v in bbox):
            out.extend(normalize_bboxes(bbox, bbox_format))  # nested list, e.g. per-frame boxes
            continue
        if isinstance(bbox, dict):
            x, y = int(bbox["x"]), int(bbox["y"])
            w, h = int(bbox["width"]), int(bbox["height"])
            out.append((x, y, x + w, y + h) if bbox_format == "xyxy" else (x, y, w, h))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            out.append(tuple(int(v) for v in bbox[:4]))
        else:
            raise ValueError(f"Invalid bbox: {bbox}")
    return out


def bbox_to_bounding_box(bbox, bbox_format="xywh") -> dict:
    """Convert a single KJNodes BBOX to the native BOUNDING_BOX dict."""
    a, b, c, d = normalize_bboxes(bbox, bbox_format)[0]
    if bbox_format == "xyxy":
        return {"x": min(a, c), "y": min(b, d), "width": abs(c - a), "height": abs(d - b)}
    return {"x": a, "y": b, "width": c, "height": d}


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
    elif color_string.startswith('#') or (color_string.lstrip('#').isalnum() and not color_string.lstrip('#').replace('.', '', 1).isdigit()):
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


class BrowserH264:
    # libx264 mp4 for browser previews, written from rgb24 uint8 frames. Frames are converted and tagged BT.709 as core does
    # for sRGB video, browsers decode untagged HD video as BT.709. One shared reformatter, a fresh SwsContext per frame costs ~3ms
    def __init__(self, path, width, height, fps, options):
        import av
        self._av = av
        self.container = av.open(path, mode="w")
        try:
            self.stream = self.container.add_stream("libx264", rate=fps)
            self.stream.width, self.stream.height, self.stream.pix_fmt, self.stream.options = width, height, "yuv420p", options
            self._to_yuv = None
            try:
                from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc, VideoReformatter
                cc = self.stream.codec_context
                if all(hasattr(cc, a) for a in ("colorspace", "color_range", "color_primaries", "color_trc")):
                    cc.colorspace, cc.color_range, cc.color_primaries, cc.color_trc = 1, ColorRange.MPEG, ColorPrimaries.BT709, ColorTrc.IEC61966_2_1
                    ref = VideoReformatter()
                    self._to_yuv = lambda f: ref.reformat(f, format="yuv420p", dst_colorspace=1, dst_color_range=ColorRange.MPEG)
            except (ImportError, AttributeError):
                pass  # older PyAV, the encoder converts untagged
        except BaseException:
            self.container.close()
            raise

    def write(self, rgb):
        frame = self._av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(self._to_yuv(frame) if self._to_yuv else frame):
            self.container.mux(packet)

    def flush(self):
        for packet in self.stream.encode():
            self.container.mux(packet)

    def close(self):
        self.container.close()


class VideoFileReader:
    # Streams a file-backed VIDEO frame by frame with core's trim window, rotation and crop, instead of materializing a
    # float batch through get_components(). Iterate once for the decoded av frames, rgb() turns one into display pixels
    @staticmethod
    def supports(video):
        # The private crop field fails closed: if core renames it, callers use get_components, which applies the crop
        from comfy_api.latest import InputImpl
        return (isinstance(video, InputImpl.VideoFromFile) and hasattr(video, "get_active_trim_window")
                and getattr(video, "_VideoFromFile__crop", False) is not False)

    def __init__(self, video):
        # Metadata first, BytesIO sources share one read position with the decoder opened below
        try:
            self.count = video.get_frame_count() or None
        except Exception:
            self.count = None  # core's count fallback can fail on odd containers
        self.fps = video.get_frame_rate()
        self._frames = self._decode(video)
        self.first = next(self._frames, None)
        if self.first is None:
            raise ValueError("Video has no decodable frames.")
        f = self.first
        self.rotation = int(round(f.rotation // 90)) % 4 if getattr(f, "rotation", 0) else 0
        w, h = (f.height, f.width) if self.rotation % 2 else (f.width, f.height)
        self.crop = video._VideoFromFile__crop
        if self.crop is not None:
            from comfy_api.latest._util import normalize_crop_rect
            self.crop = normalize_crop_rect(*self.crop, w, h)
        self.width, self.height = self.crop[2:] if self.crop else (w, h)
        self._ref = None

    @staticmethod
    def _decode(video):
        import av
        start_time, duration = video.get_active_trim_window()
        with av.open(video.get_stream_source(), mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            start_pts = int(start_time / stream.time_base)
            end_pts = int((start_time + duration) / stream.time_base) if duration else None
            if start_pts:
                container.seek(start_pts, stream=stream)
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    continue
                for frame in frames:
                    if frame.pts is not None:
                        if frame.pts < start_pts:
                            continue
                        if end_pts is not None and frame.pts >= end_pts:
                            return
                    yield frame

    def __iter__(self):
        yield self.first
        yield from self._frames

    def rgb(self, frame):
        # Full-resolution rgb24 in display orientation, rotated then cropped like core. Shared reformatter, see BrowserH264
        if self._ref is None:
            from av.video.reformatter import VideoReformatter
            self._ref = VideoReformatter()
        img = self._ref.reformat(frame, format="rgb24").to_ndarray()
        if self.rotation:
            img = np.rot90(img, k=self.rotation, axes=(0, 1))
        if self.crop:
            x, y, w, h = self.crop
            img = img[y:y + h, x:x + w]
        return np.ascontiguousarray(img)

    def close(self):
        self._frames.close()
