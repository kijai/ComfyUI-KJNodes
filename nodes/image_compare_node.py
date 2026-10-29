import os
import secrets
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, wait
from fractions import Fraction

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy import model_management
from comfy.utils import ProgressBar
from comfy_api.latest import io

from ..utility.utility import BrowserH264, VideoFileReader

# Low crf keeps codec noise out of diff mode, short GOP keeps frame scrubbing responsive
_MP4_OPTIONS = {"crf": "16", "preset": "veryfast", "g": "8"}

# Unchanged upstream outputs come back as the same object on re-runs, so their files are reused.
# id(source) -> (weakref, fingerprint, filename, (width, height, frames)), entries drop out when the source is freed
_reuse = {}


def _fingerprint(src, fps):
    if isinstance(src, torch.Tensor):
        # Sparse sample sum catches in-place edits of a cached tensor, cheap even for long batches
        H, W = src.shape[1], src.shape[2]
        return tuple(src.shape), src.dtype, src.data_ptr(), fps, float(src[:, ::max(1, H // 16), ::max(1, W // 16)].float().sum())
    path = src.get_stream_source()
    stat = os.stat(path) if isinstance(path, str) else None
    return (path, stat.st_mtime_ns, stat.st_size) if stat else id(path), src.get_active_trim_window(), src._VideoFromFile__crop, fps


def _to_uint8(images):
    # Same 255-scale truncation as core's SaveImage, so previews match saved files
    return images[..., :3].mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def _batch_frames(images, chunk=16):
    for start in range(0, images.shape[0], chunk):
        yield from _to_uint8(images[start:start + chunk])


def _save_mp4(frames, width, height, path, fps, progress):
    # yuv420p needs even sides, edge padding keeps every source pixel; the frontend crops it back off
    pad = ((0, height & 1), (0, width & 1), (0, 0))
    out = BrowserH264(path, width + (width & 1), height + (height & 1), Fraction(fps).limit_denominator(1001), _MP4_OPTIONS)
    try:
        count = 0
        for img in frames:
            out.write(np.pad(img, pad, mode="edge") if (width | height) & 1 else img)
            count += 1
            progress(1)
        out.flush()
        return count
    finally:
        out.close()


def _save(src, kind, path, fps, progress):
    # Returns the (width, height, frames) shown for the file
    with torch.inference_mode():  # per thread
        if kind == "stream":
            reader = VideoFileReader(src)
            try:
                return reader.width, reader.height, _save_mp4(map(reader.rgb, reader), reader.width, reader.height, path, fps, progress)
            finally:
                reader.close()
        B, H, W = src.shape[:3]
        if kind == "mask":
            # Gray+alpha PNG of the first frame, CSS masks read alpha; the browser stretches it over the reference frame
            m = src[0].mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            Image.fromarray(np.stack([np.full_like(m, 255), m], axis=-1)).save(path, compress_level=1)
        elif B > 1:
            _save_mp4(_batch_frames(src), W, H, path, fps, progress)
        else:
            # Lossless, lossy previews would show up as false differences in diff mode
            Image.fromarray(_to_uint8(src[:1])[0]).save(path, compress_level=1)
        return W, H, B


def _frames_hint(src, kind):
    if kind == "stream":
        try:
            return src.get_frame_count() or 0
        except Exception:
            return 0
    return src.shape[0] if kind == "image" and src.shape[0] > 1 else 0


class ImageCompareKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageCompareKJ",
            display_name="Image Compare KJ",
            category="KJNodes/image",
            search_aliases=["compare", "before after", "a/b", "slider", "diff", "difference", "video compare"],
            is_output_node=True,
            description="Interactive image and video comparison.\n\n"
                "Modes: **slider** (Shift+drag rotates), **box**, **diff**, **toggle** and **mask**.\n"
                "Up to 4 inputs, the first one is the reference. Connected Load Image and Load Video nodes preview before running.\n"
                "Batches and videos play back in sync, frame by frame, as high quality H.264 with frame scrubbing; single images among them stay still.\n"
                "The optional mask (first frame) reveals the compared image where it is white.",
            inputs=[
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplatePrefix(io.MultiType.Input("image", [io.Image, io.Video]), prefix="image", min=2, max=4),
                    tooltip="Images, image batches or videos to compare. The first input is the reference.",
                ),
                io.Mask.Input("mask", optional=True, tooltip="Mask mode: white shows the compared image, black the reference, gray blends. Only the first frame is used."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.1,
                               tooltip="Playback frame rate for image batches. When a video is connected, the first video's frame rate is used instead."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type, fps=24.0, mask=None) -> io.NodeOutput:
        # Videos held in memory become plain batches, file-backed ones stream frame by frame
        sources, video_fps = [], None
        for _, v in sorted(images.items(), key=lambda kv: int(kv[0].removeprefix("image"))):
            if v is None:
                continue
            if not isinstance(v, torch.Tensor):
                video_fps = video_fps or v.get_frame_rate()
                if VideoFileReader.supports(v):
                    sources.append((v, "stream"))
                    continue
                v = v.get_components().images
            if v.shape[0] > 0:  # empty batches have nothing to show
                sources.append((v, "image"))
        if len(sources) < 2:
            return io.NodeOutput(ui={"compare_data": []})
        # Inputs are compared frame by frame, so everything plays at the first video's rate when there is one
        if video_fps:
            fps = float(video_fps)
        if mask is not None:
            mask = mask[None] if mask.ndim == 2 else mask
            mask = mask if mask.shape[0] > 0 else None

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        # Random names, a restarted server reuses counters and the browser would show cached files
        prefix = f"kjnodes_compare_{secrets.token_hex(5)}"

        # One job per distinct source, sources unchanged since an earlier run keep their files
        targets = sources + ([(mask, "mask")] if mask is not None else [])
        names, meta, jobs = {}, {}, {}
        for idx, (src, kind) in enumerate(targets):
            if id(src) in names:
                continue
            moving = kind == "stream" or (kind == "image" and src.shape[0] > 1)
            fp, e = _fingerprint(src, fps if moving else None), _reuse.get(id(src))
            if e and e[0]() is src and e[1] == fp and os.path.exists(os.path.join(temp_dir, e[2])):
                names[id(src)], meta[id(src)] = e[2], e[3]
            else:
                names[id(src)] = fname = f"{prefix}_{idx}.{'mp4' if moving else 'png'}"
                jobs[fname] = (src, kind, fp)

        # Workers only watch a stop event: taking the interrupt clears the global flag, so a worker raising
        # it would leave the others encoding. The main thread takes it and drives the progress bar
        stop, lock, done = threading.Event(), threading.Lock(), [0]
        pbar = ProgressBar(sum(_frames_hint(src, kind) for src, kind, _ in jobs.values()) or 1)

        def progress(n):
            if stop.is_set():
                raise model_management.InterruptProcessingException()
            with lock:
                done[0] += n

        # Encoders, decoders and zlib release the GIL, sources encode in parallel
        with ThreadPoolExecutor(max(1, len(jobs))) as pool:
            futures = {fname: pool.submit(_save, src, kind, os.path.join(temp_dir, fname), fps, progress) for fname, (src, kind, _) in jobs.items()}
            try:
                while wait(futures.values(), timeout=0.1).not_done:
                    model_management.throw_exception_if_processing_interrupted()
                    pbar.update_absolute(done[0])
                for fname, (src, _, fp) in jobs.items():
                    meta[id(src)] = futures[fname].result()
                    _reuse[id(src)] = (weakref.ref(src, lambda _, k=id(src): _reuse.pop(k, None)), fp, fname, meta[id(src)])
            except BaseException:
                stop.set()
                raise

        entries = []
        for src, _ in sources:
            w, h, n = meta[id(src)]
            entries.append({"filename": names[id(src)], "subfolder": "", "type": "temp", "width": w, "height": h, "frames": n})
        data = {"fps": fps, "images": entries}
        if mask is not None:
            data["mask"] = {"filename": names[id(mask)], "subfolder": "", "type": "temp"}
        return io.NodeOutput(ui={"compare_data": [data]})
