import base64
import fractions
import io as pyio
import logging
import queue
import threading
import time

import numpy as np
import torch

import comfy.model_base
import comfy.model_management
import comfy.utils
import comfy.patcher_extension
try:
    import comfy.model_prefetch
except Exception:
    comfy.model_prefetch = None
import folder_paths
import latent_preview
from comfy_api.latest import io
from PIL import Image, ImageOps

from .tiny_vae import load_tiny_vae_decoder

try:
    from .ltxv_nodes import WrappedPreviewer as _LTXWrappedPreviewer, get_ltx_rgb_factors as _ltx_rgb_factors
except Exception as e:
    logging.warning(f"[KJ PreviewOverride] LTX preview helpers unavailable ({e}); LTX previews disabled.")
    _LTXWrappedPreviewer = None
    _ltx_rgb_factors = None


try:
    from server import PromptServer
except ImportError:
    PromptServer = None

def _suppressed_preview_image(self_, preview_format, x0):
    return None


class _AsyncPreviewEncoder:
    """Off-thread encoder. Bounded FIFO drops-on-full so the sampler never blocks on us."""

    _STOP = object()

    def __init__(self, max_in_flight=2):
        self.q = queue.Queue(maxsize=max_in_flight)
        self.thread = threading.Thread(target=self._run, name="kj_preview_encoder", daemon=True)
        self.thread.start()

    def submit(self, fn):
        try:
            self.q.put_nowait(fn)
            return True
        except queue.Full:
            return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is self._STOP:
                return
            try:
                item()
            except Exception:
                logging.exception("[KJ Preview Override] async encoder error")

    def shutdown(self, drain_timeout=5.0):
        try:
            self.q.put(self._STOP, timeout=drain_timeout)
        except queue.Full:
            pass
        self.thread.join(timeout=drain_timeout)


def _get_core_previewer(load_device, latent_format):
    # Walk past custom-node hooks on get_previewer to reach the unwrapped core function.
    fn = latent_preview.get_previewer
    seen = set()
    while hasattr(fn, "__wrapped__") and id(fn) not in seen:
        seen.add(id(fn))
        fn = fn.__wrapped__
    return fn(load_device, latent_format)


def _decode_video_frames_l2rgb(x0, latent_format, max_frames, stride=1):
    # Bulk-blocking GPU→CPU copy (not per-frame non_blocking) avoids torn frames at high res.
    if x0.ndim != 5:
        return []
    rgb_factors = getattr(latent_format, "latent_rgb_factors", None)
    if rgb_factors is None:
        return []
    try:
        reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
        if reshape is not None:
            x0 = reshape(x0)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        factors = torch.tensor(rgb_factors, device=x0.device, dtype=x0.dtype).transpose(0, 1)
        bias_t = torch.tensor(bias, device=x0.device, dtype=x0.dtype) if bias is not None else None
        x = x0[0]
        if stride > 1:
            x = x[:, ::stride]
        t_total = x.shape[1]
        if max_frames > 0 and max_frames < t_total:
            indices = np.linspace(0, t_total - 1, max_frames).round().astype(int).tolist()
            x = x[:, indices]
        x = x.movedim(0, -1)
        rgb = torch.nn.functional.linear(x, factors, bias=bias_t)
        rgb.add_(1.0).mul_(127.5).clamp_(0, 255)
        rgb_cpu = rgb.to(torch.uint8).cpu().numpy()
        return [Image.fromarray(rgb_cpu[i]) for i in range(rgb_cpu.shape[0])]
    except Exception:
        return []


def _probe_codec(name):
    try:
        import av  # noqa
        av.Codec(name, "w")
        return True
    except Exception:
        return False

# --disable-api-nodes installs a CSP with no media-src
def _csp_blocks_video():
    try:
        from comfy.cli_args import args as _args
        return bool(getattr(_args, "disable_api_nodes", False))
    except Exception:
        return False

# CPU x264 first: at ultrafast it encodes a preview clip in ~0.1s and needs no CUDA context, while
# NVENC's ffmpeg wrapper time-slices against the saturated sampling context and takes seconds per step.
# The candidates are (codec, options, min_w, min_h); NVENC rejects sub-145x49 inputs at avcodec_open2.
_MP4_CANDIDATES = [c for c in (
    ("libx264", {"preset": "ultrafast", "tune": "zerolatency", "crf": "28"}, 2, 2),
    ("h264_nvenc", {"preset": "p1", "rc": "vbr", "cq": "23"}, 145, 49),
    ("h264_nvenc", {"preset": "p1"}, 145, 49),
) if _probe_codec(c[0])]
_HAS_MP4 = bool(_MP4_CANDIDATES)
_MP4_AVAILABLE = _HAS_MP4 and not _csp_blocks_video()
if _HAS_MP4 and not _MP4_AVAILABLE:
    logging.info("[KJ PreviewOverride] --disable-api-nodes blocks blob: video, using WebP for animated previews.")

_mp4_warned = False


def _encode_mp4(frames, fps, max_res, audio=None):
    # Fragmented MP4 so the browser can decode mid-download. Returns (None, 0, 0) when every
    # candidate fails (including too-small frames), so caller falls through to WebP.
    # audio: optional ([2, L] float32 in [-1, 1], sample_rate) muxed as an AAC track; fps may be a
    # Fraction so the clip's duration matches the audio
    global _mp4_warned
    if not frames:
        return None, 0, 0
    try:
        import av
    except Exception:
        return None, 0, 0
    # arrays arrive already at the preview size from _frames_to_arrays and go straight into the
    # encoder plane; PIL frames from the other previewers are resized here
    pil_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
            pil_frames.append(f)
            continue
        pf = f if f.mode == "RGB" else f.convert("RGB")
        if max_res and max_res > 0 and (pf.width > max_res or pf.height > max_res):
            pf = ImageOps.contain(pf, (max_res, max_res), Image.LANCZOS)
        pil_frames.append(pf)
    # yuv420p requires even dimensions.
    f0 = pil_frames[0]
    w0, h0 = (f0.shape[1], f0.shape[0]) if isinstance(f0, np.ndarray) else (f0.width, f0.height)
    out_w, out_h = w0 & ~1, h0 & ~1
    if (out_w, out_h) != (w0, h0):
        pil_frames = [pf[:out_h, :out_w] if isinstance(pf, np.ndarray) else pf.resize((out_w, out_h), Image.LANCZOS)
                      for pf in pil_frames]
    last_err = None
    for codec, opts, min_w, min_h in _MP4_CANDIDATES:
        if out_w < min_w or out_h < min_h:
            continue
        buf = pyio.BytesIO()
        try:
            container = av.open(
                buf, mode="w", format="mp4",
                options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
            )
            rate = fps if isinstance(fps, fractions.Fraction) else int(max(1, fps))
            stream = container.add_stream(codec, rate=rate)
            stream.width = out_w
            stream.height = out_h
            stream.pix_fmt = "yuv420p"
            stream.options = opts
            # every stream must exist before the first mux writes the header
            astream = container.add_stream("aac", rate=int(audio[1]), layout="stereo") if audio is not None else None
            for pf in pil_frames:
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(pf), format="rgb24") if isinstance(pf, np.ndarray) \
                    else av.VideoFrame.from_image(pf)
                for pkt in stream.encode(vf):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            if astream is not None:
                _mux_aac(container, av, astream, audio[0], audio[1])
            container.close()
            return base64.b64encode(buf.getvalue()).decode("ascii"), out_w, out_h
        except Exception as e:
            last_err = e
            continue
    if not _mp4_warned and last_err is not None:
        _mp4_warned = True
        logging.warning(f"[KJ PreviewOverride] MP4 encode failed, using WebP fallback: {last_err}")
    return None, 0, 0


def _mux_aac(container, av, astream, wave, sample_rate):
    # wave [2, L] float32 in [-1, 1]; mirrors core's VideoFromComponents.save_to: one whole-clip
    # frame at pts 0, the codec's own FIFO does the AAC framing
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(wave, dtype=np.float32), format="fltp", layout="stereo")
    frame.sample_rate = int(sample_rate)
    frame.pts = 0
    container.mux(astream.encode(frame))
    container.mux(astream.encode(None))


def _encode_animated_webp(frames, fps, quality, max_res):
    if not frames:
        return None, 0, 0
    pil_frames = []
    for f in frames:
        pf = _as_pil(f)
        if pf.mode != "RGB":
            pf = pf.convert("RGB")
        if max_res and max_res > 0 and (pf.width > max_res or pf.height > max_res):
            pf = ImageOps.contain(pf, (max_res, max_res), Image.LANCZOS)
        pil_frames.append(pf)
    duration_ms = max(1, int(round(1000 / max(1, fps))))
    buf = pyio.BytesIO()
    try:
        pil_frames[0].save(
            buf,
            format="WEBP",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            quality=quality,
            method=4,
        )
    except Exception as e:
        logging.warning(f"Animated WebP encode failed: {e}")
        return None, 0, 0
    return base64.b64encode(buf.getvalue()).decode("ascii"), pil_frames[0].width, pil_frames[0].height


def _interp_db_curve(t, xs, ys):
    # Mirrors sampler_nodes._interp_curve.
    if t <= xs[0]:
        return ys[0]
    if t >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= t <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            if span <= 0:
                return ys[i]
            f = (t - xs[i]) / span
            return ys[i] + f * (ys[i + 1] - ys[i])
    return 0.0


def _detect_detail_boost_curve(sampler, model_patcher, sigmas_list):
    # Amount is already baked into ys by the editor, so peak ys == user-set amount.
    try:
        extra = getattr(sampler, "extra_options", None) or {}
        xs = extra.get("db_curve_xs")
        ys = extra.get("db_curve_ys")
        if "db_wrapped_sampler" not in extra or not xs or not ys or len(xs) != len(ys) or len(xs) < 2:
            return None
        ms = model_patcher.get_model_object("model_sampling")
        start_sigma = float(ms.percent_to_sigma(extra.get("db_start_percent", 0.0)))
        end_sigma = float(ms.percent_to_sigma(extra.get("db_end_percent", 1.0)))
        # None outside the gate so JS can distinguish "inactive" from "active with value 0".
        out = []
        for s in sigmas_list:
            sig = float(s)
            if sig <= 0 or start_sigma <= end_sigma or sig >= start_sigma or sig <= end_sigma:
                out.append(None)
                continue
            t = (start_sigma - sig) / (start_sigma - end_sigma)
            out.append(_interp_db_curve(t, xs, ys))
        return out
    except Exception as e:
        logging.warning(f"[KJ PreviewOverride] DB curve detection failed: {e}")
        return None


def _ltx_decode_to_pil(ltx_previewer, x0_5d, max_frames=None, stride=1):
    # Pre-shape (B, C, T, H, W) → (B*T, C, H, W); WrappedPreviewer adds the sequence-batch dim.
    if ltx_previewer is None or x0_5d.ndim != 5:
        return []
    if stride > 1:
        x0_5d = x0_5d[:, :, ::stride]
    x_moved = x0_5d.movedim(2, 1)  # (B, T, C, H, W) — must take shape AFTER movedim
    x_in = x_moved.reshape((-1,) + x_moved.shape[-3:])
    rgb = ltx_previewer.decode_latent_to_preview(x_in)
    if rgb is None:
        return []
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.ndim != 4:
        return []
    t_total = rgb.shape[0]
    if max_frames is not None and 0 < max_frames < t_total:
        indices = np.linspace(0, t_total - 1, max_frames).round().astype(int).tolist()
        rgb = rgb[indices]
    u8 = (rgb * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(u8[i]) for i in range(u8.shape[0])]


def _ltx_full_vae_decode_to_pil(vae, x0_5d, max_frames=None, stride=1):
    # vae.decode handles device + tiling. Slow vs TAEHV but full quality. Output shape
    # varies by VAE; we accept (B, T, H, W, C) or (T, H, W, C) and normalize.
    if vae is None or x0_5d.ndim != 5:
        return []
    if stride > 1:
        x0_5d = x0_5d[:, :, ::stride]
    try:
        images = vae.decode(x0_5d)
    except Exception as e:
        logging.warning(f"[KJ PreviewOverride] LTX VAE decode failed: {e}")
        return []
    if images.ndim == 5:
        images = images[0]
    if images.ndim != 4:
        return []
    t_total = images.shape[0]
    if max_frames is not None and 0 < max_frames < t_total:
        indices = np.linspace(0, t_total - 1, max_frames).round().astype(int).tolist()
        images = images[indices]
    u8 = (images.float() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(u8[i]) for i in range(u8.shape[0])]


def _frame_to_pil(frame):
    # [3, H, W] on any device, float in [0, 1] or uint8. Copies before the in-place math so a view into
    # a shared batch is never scaled twice; fp16 elementwise on the CPU is slow, so it goes through fp32
    if frame.dtype != torch.uint8:
        frame = frame.to(torch.float32, copy=True).clamp_(0, 1).mul_(255).to(torch.uint8)
    return Image.fromarray(frame.movedim(0, -1).contiguous().cpu().numpy())


def _contain_size(w, h, max_res):
    # ImageOps.contain sizing, rounded down to even so the yuv420p encode never needs a second resize;
    # frames that already fit stay at native size
    if not max_res or max_res <= 0 or (w <= max_res and h <= max_res):
        return w, h
    if w >= h:
        tw, th = max_res, round(h / w * max_res)
    else:
        tw, th = round(w / h * max_res), max_res
    return max(2, tw & ~1), max(2, th & ~1)


def _frames_to_arrays(frames, max_res, budget_bytes=256 << 20):
    # [T, 3, H, W] on the CPU -> [H, W, 3] uint8 arrays at the preview size. One antialiased tensor
    # resize per chunk replaces a LANCZOS pass per frame in PIL, which dominated the preview latency
    # on long clips; the chunk follows a byte budget so peak RAM stays flat across resolutions
    h, w = frames.shape[-2:]
    tw, th = _contain_size(w, h, max_res)
    resize = (tw, th) != (w, h)
    chunk = max(1, budget_bytes // (3 * h * w * 4))
    out = []
    for i in range(0, frames.shape[0], chunk):
        x = frames[i:i + chunk]
        if x.dtype != torch.uint8 or resize:
            # copy: the batch may be an inference tensor from the sampler thread, in-place would raise
            x = x.to(torch.float32, copy=True)
            if frames.dtype == torch.uint8:
                x.div_(255)
            if resize:
                x = torch.nn.functional.interpolate(x, size=(th, tw), mode="bilinear", antialias=True, align_corners=False)
            x = x.clamp_(0, 1).mul_(255).to(torch.uint8)
        u8 = x.permute(0, 2, 3, 1).contiguous().numpy()
        out.extend(u8[j] for j in range(u8.shape[0]))
    return out


def _tiny_vae_decode_frames(decoder, x0, max_frames=None, compile_preview=False):
    # Raises on failure so the caller can disable the decoder instead of retrying every step.
    # Returns the whole [T, 3, H, W] clip on the CPU (uint8 from the 2D decoder, model dtype from
    # TAEHV) or None. The compiler treats GPU allocations it did not plan as rogue and pays for them
    # every step, so the decode joins the sampler thread's allocation graph and every GPU tensor is
    # gone before the scope closes. Needs a core with per-thread graphs; older cores skip it.
    if x0.ndim not in (4, 5):
        return None, None
    compiled = compile_preview and comfy.model_prefetch is not None and comfy.model_prefetch.malloc_graph_enabled(x0.device)
    if compiled:
        comfy.model_prefetch.malloc_graph_begin(x0.device)
    try:
        span = None
        if x0.ndim == 4:
            frames = decoder.decode(x0[:1])
        else:
            indices = list(range(x0.shape[2]))
            if max_frames is not None and 0 < max_frames < len(indices):
                picks = np.linspace(0, len(indices) - 1, max_frames).round().astype(int).tolist()
                indices = [indices[i] for i in picks]
            frames = decoder.decode_video(x0[:1], frame_indices=indices)
            # the temporal decoder chains memblock state, so a partial request decodes the prefix instead
            prefix = getattr(decoder, "decodes_prefix", False) and len(indices) < x0.shape[2]
            span = (0, len(indices) - 1) if prefix else (indices[0], indices[-1])
        frames = None if frames is None or frames.shape[0] == 0 else frames.cpu()
    finally:
        if compiled:
            comfy.model_prefetch.malloc_graph_end()
    return frames, span


def _materialize_frames(frames, max_res=0):
    # a tensor batch comes from the tiny VAE and is converted here on the encoder thread into
    # [H, W, 3] uint8 arrays; the other previewers already hand over PIL
    if isinstance(frames, torch.Tensor):
        return _frames_to_arrays(frames, max_res)
    return list(frames)


def _as_pil(frame):
    return frame if isinstance(frame, Image.Image) else Image.fromarray(frame)


def _tiny_vae_decode_to_pil(decoder, x0, max_frames=None, compile_preview=False):
    frames, _ = _tiny_vae_decode_frames(decoder, x0, max_frames, compile_preview)
    return [] if frames is None else [_as_pil(a) for a in _frames_to_arrays(frames, 0)]


def _is_ltx_latent_format(latent_format):
    return "LTX" in type(latent_format).__name__


def _is_ltx2_diffusion_model(model_patcher):
    # Same probe as ltxv_nodes.OuterSampleCallbackWrapper.
    try:
        dm = model_patcher.model.diffusion_model
        return not getattr(dm, "caption_projection_first_linear", True)
    except Exception:
        return False


def _ltx_num_keyframes(guider):
    try:
        positive = guider.conds.get("positive") if hasattr(guider, "conds") else None
        if positive and len(positive) > 0:
            kf = positive[0].get("keyframe_idxs")
            if kf is not None:
                return int(torch.unique(kf[0, 0, :, 0]).numel())
    except Exception:
        pass
    return 0


def _is_av_model(model):
    # core marks audio+video models by model type; other packed latents (e.g. shape + camera) are not audio
    flow_av = getattr(comfy.model_base.ModelType, "FLOW_AV", None)
    return flow_av is not None and getattr(model, "model_type", None) == flow_av


def _packed_audio_latent(x0, latent_shapes):
    # AV models pack every stream into one [B, 1, N] tensor; the audio stream is the last entry, as
    # core's vae_decode_audio picks it
    if x0.ndim != 3 or not latent_shapes or len(latent_shapes) < 2:
        return None
    try:
        return comfy.utils.unpack_latents(x0, latent_shapes)[-1]
    except Exception:
        return None


def _frames_per_token(latent_format):
    # pixel frames each latent token stands for, cycled: MiniMax H3 codes 17 frames per 5 tokens,
    # the causal VAEs one frame then temporal_downscale_ratio per token
    if type(latent_format).__name__.startswith("MiniMaxH3"):
        return (1, 4, 4, 4, 4)
    return (1, max(1, int(getattr(latent_format, "temporal_downscale_ratio", 1))))


def _token_span_fraction(span, n_tokens, pattern):
    # fraction of the clip covered by latent tokens [first, last]; the audio latent spans the whole clip
    if span is None or n_tokens <= 1:
        return 0.0, 1.0
    def frames(k):
        if len(pattern) == 2:
            return 0 if k <= 0 else 1 + (k - 1) * pattern[1]
        return sum(pattern[i % len(pattern)] for i in range(k))
    first, last = span
    total = frames(n_tokens)
    return frames(first) / total, min(1.0, frames(last + 1) / total)


_SPEC_LUT = None
_HANN = {}


def _spec_colormap():
    # black -> purple -> orange -> yellow -> white, 256 entries
    global _SPEC_LUT
    if _SPEC_LUT is None:
        anchors = np.array([[0, 0, 0], [60, 10, 90], [180, 50, 60], [240, 130, 30], [250, 210, 60], [255, 255, 230]], dtype=np.float32)
        pos = np.linspace(0, 1, len(anchors))
        x = np.linspace(0, 1, 256)
        _SPEC_LUT = np.stack([np.interp(x, pos, anchors[:, c]) for c in range(3)], axis=1).astype(np.uint8)
    return _SPEC_LUT


def _strip_to_png(mag, width=512, height=64):
    # mag [rows, cols] >= 0, low rows = low frequency; drawn bottom-up, normalized per strip
    mag = np.asarray(mag, dtype=np.float32)
    if mag.ndim != 2 or mag.size == 0:
        return None
    hi = float(np.percentile(mag, 99.5))
    norm = np.clip(mag / hi, 0, 1) ** 0.6 if hi > 0 else np.zeros_like(mag)
    img = Image.fromarray(_spec_colormap()[(norm[::-1] * 255).astype(np.uint8)], "RGB")
    img = img.resize((width, height), Image.BILINEAR)
    buf = pyio.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _wave_strip(wave, sample_rate, bands=64):
    # [2, L] -> log-magnitude STFT folded into log-spaced bands
    x = torch.as_tensor(wave, dtype=torch.float32).mean(0)
    n_fft = 1024
    hop = max(256, x.shape[0] // 800)
    window = _HANN.get(n_fft)
    if window is None:
        window = _HANN[n_fft] = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft, hop_length=hop, window=window, return_complex=True).abs()
    edges = np.unique(np.geomspace(1, spec.shape[0] - 1, bands + 1).round().astype(int))
    rows = [spec[edges[i]:max(edges[i + 1], edges[i] + 1)].mean(0) for i in range(len(edges) - 1)]
    mag = torch.log1p(torch.stack(rows) * 10).numpy()
    return _strip_to_png(mag)


def _normalize_packed_x0(x0, latent_shapes, num_keyframes):
    # Restore standard video latents from flattened packs
    if latent_shapes and len(latent_shapes) > 0:
        target = latent_shapes[0]
        if x0.ndim == 3 and len(target) >= 3:
            cut = 1
            for d in target[1:]:
                cut *= int(d)
            x0 = x0[:, :, :cut].reshape([x0.shape[0]] + list(target)[1:])
    if num_keyframes > 0 and x0.ndim == 5:
        x0 = x0[:, :, :-num_keyframes]
    return x0


class _PreviewOverrideWrapper:
    def __init__(self, max_resolution, node_id, jpeg_quality, suppress_default, preview_frames=1, preview_fps=12, vae=None, tiny_vae="none", audio_vae=None):
        self.max_resolution = max_resolution
        self.node_id = str(node_id) if node_id is not None else None
        self.jpeg_quality = jpeg_quality
        self.suppress_default = suppress_default
        self.preview_frames = preview_frames
        self.preview_fps = preview_fps
        self.vae = vae
        self.tiny_vae = tiny_vae
        self.audio_vae = audio_vae
        self.frames = []

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes):
        guider = executor.class_obj
        model_patcher = guider.model_patcher

        is_ltx = _is_ltx_latent_format(model_patcher.model.latent_format)
        is_ltx2 = is_ltx and _is_ltx2_diffusion_model(model_patcher)
        num_keyframes = _ltx_num_keyframes(guider) if is_ltx else 0

        # Tiny VAE from models/vae_approx
        tiny_vae = None
        if self.tiny_vae and self.tiny_vae != "none" and load_tiny_vae_decoder is not None:
            tiny_vae = load_tiny_vae_decoder(self.tiny_vae)
            if tiny_vae is not None and latent_shapes and len(latent_shapes[0]) >= 2:
                channels = int(latent_shapes[0][1])
                if channels != tiny_vae.latent_channels:
                    logging.warning(
                        f"[KJ PreviewOverride] '{self.tiny_vae}' decodes {tiny_vae.latent_channels}-channel "
                        f"latents but this model's are {channels}-channel; ignoring it."
                    )
                    tiny_vae = None

        # LTX reuses the LTX-specific node's WrappedPreviewer; we call decode_latent_to_preview
        # directly per step, bypassing its decode_latent_to_preview_image rate-limiting.
        # If a non-TAEHV VAE is supplied, decode via vae.decode() for full quality (slower).
        ltx_previewer = None
        ltx_full_vae = None
        vae_restore_device = None
        if is_ltx:
            try:
                factors, bias = _ltx_rgb_factors(is_ltx2)
                taeltx = None
                if self.vae is not None:
                    if self.vae.first_stage_model.__class__.__name__ == "TAEHV":
                        # TAEHV-LTX decode needs the VAE on GPU; restored at end of __call__.
                        target_device = comfy.model_management.get_torch_device()
                        try:
                            for p in self.vae.first_stage_model.parameters():
                                vae_restore_device = p.device
                                break
                            self.vae.first_stage_model.to(target_device)
                            taeltx = self.vae
                        except Exception as e:
                            logging.warning(f"[KJ PreviewOverride] Could not move TAEHV-LTX to GPU, skipping: {e}")
                    else:
                        # Comfy VAE.decode manages its own device — no pin-to-GPU needed.
                        ltx_full_vae = self.vae
                ltx_previewer = _LTXWrappedPreviewer(factors, bias, rate=8, taeltx=taeltx)
            except Exception as e:
                logging.warning(f"[KJ PreviewOverride] LTX previewer setup failed: {e}")

        previewer = _get_core_previewer(model_patcher.load_device, model_patcher.model.latent_format)
        # Latent2RGB fallback — used when the active previewer returns a non-PIL result
        # (e.g. TAEHV/TAESD on a 5D latent). LTX skips this and goes through ltx_previewer.
        fallback_previewer = None
        try:
            lf = model_patcher.model.latent_format
            rgb_factors = getattr(lf, "latent_rgb_factors", None)
            if rgb_factors is not None:
                fallback_previewer = latent_preview.Latent2RGBPreviewer(
                    rgb_factors,
                    getattr(lf, "latent_rgb_factors_bias", None),
                    getattr(lf, "latent_rgb_factors_reshape", None),
                )
        except Exception:
            pass

        original_callback = callback
        node_id = self.node_id
        max_res = self.max_resolution
        quality = self.jpeg_quality
        self.frames = []

        # N+1 boundaries for N steps: keep them all so the step marker advances through each.
        sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
        # Pre-seed so step 1 has a measurable Δ (model's first transformation from noise → x0).
        compile_preview = getattr(model_patcher.model.latent_format, "compile_preview", False)
        initial_seed_cpu = None
        try:
            if sigmas is not None and len(sigmas) > 0:
                # sigmas often lives on CPU while noise is on CUDA — align before the multiply.
                s0 = sigmas[0].to(noise.device) if hasattr(sigmas[0], "to") else sigmas[0]
                seeded = _normalize_packed_x0(noise * s0, latent_shapes, num_keyframes)
                initial_seed_cpu = seeded.detach().float().cpu()
        except Exception as e:
            logging.warning(f"[KJ PreviewOverride] initial seed Δ pre-fill failed: {e}")
        state = {"last_x0_cpu": initial_seed_cpu, "last_time": None, "step_ms_window": []}
        total_steps_init = max(0, len(sigmas_list) - 1)

        # Boundary-0 message: sigmas (required by JS hover handler) plus optional noise preview.
        if node_id is not None and PromptServer is not None:
            init_payload = {
                "node_id": node_id,
                "step": 0,
                "total": total_steps_init,
                "sigma": sigmas_list[0] if sigmas_list else None,
                "sigmas": sigmas_list,
            }
            db_curve = _detect_detail_boost_curve(sampler, model_patcher, sigmas_list)
            if db_curve is not None:
                init_payload["db_curve"] = db_curve
            # Use Latent2RGB (or LTX previewer) directly — the model's default previewer (TAEHV)
            # slices to one temporal frame and returns a shape PIL can't render on raw noise.
            try:
                lf = model_patcher.model.latent_format
                rgb_factors = getattr(lf, "latent_rgb_factors", None)
                if sigmas is not None and len(sigmas) > 0:
                    s0 = sigmas[0].to(noise.device) if hasattr(sigmas[0], "to") else sigmas[0]
                    init_latent = noise * s0
                else:
                    init_latent = noise
                init_latent = _normalize_packed_x0(init_latent, latent_shapes, num_keyframes)
                pil_init = None
                if tiny_vae is not None:
                    pil_frames = _tiny_vae_decode_to_pil(tiny_vae, init_latent, max_frames=1, compile_preview=compile_preview)
                    pil_init = pil_frames[0] if pil_frames else None
                elif ltx_previewer is not None and init_latent.ndim == 5:
                    pil_frames = _ltx_decode_to_pil(ltx_previewer, init_latent, max_frames=1)
                    pil_init = pil_frames[0] if pil_frames else None
                elif rgb_factors is not None:
                    noise_previewer = latent_preview.Latent2RGBPreviewer(
                        rgb_factors,
                        getattr(lf, "latent_rgb_factors_bias", None),
                        getattr(lf, "latent_rgb_factors_reshape", None),
                    )
                    out = noise_previewer.decode_latent_to_preview(init_latent)
                    if isinstance(out, Image.Image):
                        pil_init = out
                if pil_init is not None:
                    if pil_init.mode != "RGB":
                        pil_init = pil_init.convert("RGB")
                    if max_res and max_res > 0 and (pil_init.width > max_res or pil_init.height > max_res):
                        pil_init = ImageOps.contain(pil_init, (max_res, max_res), Image.LANCZOS)
                    ibuf = pyio.BytesIO()
                    pil_init.save(ibuf, format="JPEG", quality=quality)
                    init_payload["image"] = base64.b64encode(ibuf.getvalue()).decode("ascii")
                    init_payload["w"] = pil_init.width
                    init_payload["h"] = pil_init.height
            except Exception as e:
                logging.warning(f"Initial noise preview failed (sigmas still sent): {e}")
            PromptServer.instance.send_sync("kj_preview_override", init_payload, PromptServer.instance.client_id)

        encoder = _AsyncPreviewEncoder()
        animate_video = self.preview_frames > 1
        anim_frames = self.preview_frames
        anim_fps = self.preview_fps


        # audio: with an audio VAE and an animated preview the audio latent is decoded each step,
        # muxed into the MP4 in real time, and its spectrogram drawn under the graphs
        has_audio_latent = _is_av_model(model_patcher.model) and _packed_audio_latent(noise, latent_shapes) is not None
        audio_vae = self.audio_vae if (has_audio_latent and animate_video) else None
        audio_rate = 0
        frame_pattern = _frames_per_token(model_patcher.model.latent_format)
        # the sampler carries the audio stream scaled by audio_scale and only process_latent_out undoes it
        audio_scale = getattr(model_patcher.model, "audio_scale", None)
        audio_scale = float(audio_scale()) if callable(audio_scale) else 1.0
        if audio_vae is not None:
            # core's vae_decode_audio prefers the output rate; the LTX audio VAE only sets that one
            audio_rate = int(getattr(audio_vae, "audio_sample_rate_output", None) or getattr(audio_vae, "audio_sample_rate", 0) or 0)
            if audio_rate <= 0:
                logging.warning("[KJ PreviewOverride] audio_vae has no sample rate; audio preview disabled")
                audio_vae = None
        if audio_vae is not None:
            # VAE.decode runs model management on every call (gc sweep, cache flush); load once here and
            # go through the raw model per step instead
            try:
                audio_shape = tuple(_packed_audio_latent(noise, latent_shapes).shape)
                comfy.model_management.load_models_gpu(
                    [audio_vae.patcher], memory_required=audio_vae.memory_used_decode(audio_shape, audio_vae.vae_dtype),
                    force_full_load=getattr(audio_vae, "disable_offload", False))
            except Exception as e:
                logging.warning(f"[KJ PreviewOverride] could not load audio_vae, audio preview disabled: {e}")
                audio_vae = None

        def decode_audio(audio_latent):
            # mirrors VAE.decode minus the batching: [B, 32, 2, T] -> [B, 2, L] -> [2, L] float32 CPU, with
            # core's VAEDecodeAudio level trim
            z = (audio_latent / audio_scale).to(device=audio_vae.device, dtype=audio_vae.vae_dtype)
            wave = audio_vae.process_output(audio_vae.first_stage_model.decode(z))
            wave = wave[:1].float().cpu()
            std = torch.std(wave, dim=[1, 2], keepdim=True) * 5.0
            std[std < 1.0] = 1.0
            return (wave / std)[0].clamp_(-1, 1).numpy()

        def produce_frames(x0_view):
            # (frames, span): a [T, 3, H, W] CPU tensor from the tiny VAE, else a PIL list from
            # whichever previewer applies, else []; span = latent tokens (first, last) the frames cover
            nonlocal tiny_vae
            max_pil = anim_frames if animate_video else 1
            if tiny_vae is not None:
                try:
                    frames, span = _tiny_vae_decode_frames(tiny_vae, x0_view, max_frames=max_pil, compile_preview=compile_preview)
                    if frames is not None:
                        return frames, span
                except Exception as e:
                    # OOM at 16x upscale is the likely cause — drop to the cheap paths for good.
                    logging.warning(f"[KJ PreviewOverride] tiny VAE decode failed, falling back: {e}")
                    tiny_vae = None
            pil_frames = []
            if ltx_full_vae is not None and x0_view.ndim == 5:
                pil_frames = _ltx_full_vae_decode_to_pil(ltx_full_vae, x0_view, max_frames=max_pil)
            if not pil_frames and ltx_previewer is not None and x0_view.ndim == 5:
                try:
                    pil_frames = _ltx_decode_to_pil(ltx_previewer, x0_view, max_frames=max_pil)
                except Exception as e:
                    logging.warning(f"LTX preview decode failed: {e}")
            if not pil_frames and animate_video and x0_view.ndim == 5 and ltx_previewer is None:
                pil_frames = _decode_video_frames_l2rgb(
                    x0_view, model_patcher.model.latent_format, anim_frames,
                )

            if not pil_frames:
                for prev in (previewer, fallback_previewer):
                    if prev is None:
                        continue
                    try:
                        out = prev.decode_latent_to_preview(x0_view)
                    except Exception as e:
                        if prev is previewer:
                            logging.warning(f"Active previewer raised, trying Latent2RGB fallback: {e}")
                        continue
                    if isinstance(out, Image.Image):
                        pil_frames = [out]
                        break
                    elif prev is previewer:
                        logging.warning(
                            f"Preview override: {type(previewer).__name__} returned "
                            f"{type(out).__name__} instead of PIL.Image — falling back to Latent2RGB."
                        )
            return pil_frames, None

        def new_callback(step, x0, x, total_steps_):
            if previewer is not None or fallback_previewer is not None or ltx_previewer is not None or tiny_vae is not None:
                try:
                    # NEVER rebind x0 — the sampler reuses the same tensor downstream
                    # (unpack_latents reshapes it). Preview mutations stay on x0_view.
                    x0_view = _normalize_packed_x0(x0, latent_shapes, num_keyframes)
                    frames, span = produce_frames(x0_view)

                    if isinstance(frames, torch.Tensor):
                        pil_first = _frame_to_pil(frames[0])
                    elif frames:
                        pil_first = frames[0]
                        if pil_first.mode != "RGB":
                            pil_first = pil_first.convert("RGB")
                            frames[0] = pil_first
                    else:
                        if original_callback is not None:
                            original_callback(step, x0, x, total_steps_)
                        return
                    # Consumed by GetPreviewOverrideFramesKJ.
                    self.frames.append(pil_first)

                    if node_id is not None and PromptServer is not None:
                        # x0_view (not x0) so LTX keyframe padding doesn't dampen the Δ norm; copied
                        # off the GPU before the float cast so nothing is allocated on the device
                        x0_cpu_now = x0_view.detach().cpu().float()
                        prev_x0_cpu = state["last_x0_cpu"]
                        state["last_x0_cpu"] = x0_cpu_now

                        audio_wave = None
                        n_tokens = int(x0_view.shape[2]) if x0_view.ndim == 5 else 1
                        if audio_vae is not None:
                            audio_latent = _packed_audio_latent(x0, latent_shapes)
                            if audio_latent is not None:
                                try:
                                    audio_wave = decode_audio(audio_latent)
                                except Exception as e:
                                    logging.warning(f"[KJ PreviewOverride] audio decode failed: {e}")

                        now = time.perf_counter()
                        step_ms = None
                        if state["last_time"] is not None:
                            step_ms = (now - state["last_time"]) * 1000.0
                            w = state["step_ms_window"]
                            w.append(step_ms)
                            if len(w) > 8:
                                w.pop(0)
                        state["last_time"] = now
                        avg_step_ms = (sum(state["step_ms_window"]) / len(state["step_ms_window"])) if state["step_ms_window"] else None
                        sigma_val = sigmas_list[step] if 0 <= step < len(sigmas_list) else None
                        sent_step = step + 1

                        def _encode_and_send(
                            frames=frames, span=span, x0_cpu_now=x0_cpu_now, prev_x0_cpu=prev_x0_cpu,
                            audio_wave=audio_wave, n_tokens=n_tokens,
                            step_ms=step_ms, avg_step_ms=avg_step_ms, sigma_val=sigma_val,
                            sent_step=sent_step, total_steps_=total_steps_,
                        ):
                            frames = _materialize_frames(frames, max_res)
                            frac = _token_span_fraction(span, n_tokens, frame_pattern)
                            audio = None
                            fps_out = anim_fps
                            if audio_wave is not None and len(frames) > 1:
                                total = audio_wave.shape[1]
                                s0 = int(frac[0] * total)
                                s1 = max(int(frac[1] * total), s0 + audio_rate // 10)
                                clip = audio_wave[:, s0:s1]
                                if clip.shape[1] > 0:
                                    audio = (clip, audio_rate)
                                    fps_out = fractions.Fraction(len(frames) * audio_rate, clip.shape[1]).limit_denominator(1000)
                            spec_b64 = None
                            if audio is not None:
                                try:
                                    spec_b64 = _wave_strip(audio[0], audio_rate)
                                except Exception as e:
                                    logging.warning(f"[KJ PreviewOverride] spectrogram failed: {e}")

                            if len(frames) > 1:
                                # MP4 is far faster and smaller than PIL WebP when an encoder is available.
                                b64, w_, h_, mime = None, 0, 0, None
                                if _MP4_AVAILABLE:
                                    b64, w_, h_ = _encode_mp4(frames, fps_out, max_res, audio=audio)
                                    if b64:
                                        mime = "video/mp4"
                                if not b64:
                                    audio = None
                                    fps_out = anim_fps
                                    b64, w_, h_ = _encode_animated_webp(frames, anim_fps, quality, max_res)
                                    mime = "image/webp"
                            else:
                                pil_send = _as_pil(frames[0])
                                if max_res and max_res > 0 and (pil_send.width > max_res or pil_send.height > max_res):
                                    pil_send = ImageOps.contain(pil_send, (max_res, max_res), Image.LANCZOS)
                                buf = pyio.BytesIO()
                                pil_send.save(buf, format="JPEG", quality=quality)
                                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                                w_, h_ = pil_send.width, pil_send.height
                                mime = "image/jpeg"

                            if not b64:
                                return

                            delta_v = None
                            if prev_x0_cpu is not None and prev_x0_cpu.shape == x0_cpu_now.shape:
                                diff = x0_cpu_now - prev_x0_cpu
                                delta_v = (diff.norm() / max(1, diff.numel()) ** 0.5).item()

                            PromptServer.instance.send_sync(
                                "kj_preview_override",
                                {
                                    "node_id": node_id,
                                    "image": b64,
                                    "mime": mime,
                                    "w": w_,
                                    "h": h_,
                                    "step": sent_step,
                                    "total": total_steps_,
                                    "sigma": sigma_val,
                                    "sigmas": None,
                                    "delta": delta_v,
                                    "step_ms": step_ms,
                                    "avg_step_ms": avg_step_ms,
                                    "fps": float(fps_out) if mime in ("video/mp4", "image/webp") else None,
                                    "audio": audio is not None,
                                    "audio_spec": spec_b64,
                                },
                                PromptServer.instance.client_id,
                            )

                        encoder.submit(_encode_and_send)
                except Exception as e:
                    logging.warning(f"Preview override failed: {e}")
            if original_callback is not None:
                original_callback(step, x0, x, total_steps_)

        # Patch every concrete decode_latent_to_preview_image — subclasses like VHS's
        # WrappedPreviewer override it and would otherwise still emit previews of their own.
        prev_methods = []
        if self.suppress_default:
            targets = [latent_preview.LatentPreviewer]
            stack = list(latent_preview.LatentPreviewer.__subclasses__())
            while stack:
                cls = stack.pop()
                targets.append(cls)
                stack.extend(cls.__subclasses__())
            for cls in targets:
                if "decode_latent_to_preview_image" in cls.__dict__:
                    prev_methods.append((cls, cls.__dict__["decode_latent_to_preview_image"]))
                    cls.decode_latent_to_preview_image = _suppressed_preview_image
        try:
            # Seeds step 1's duration measurement (sampling-start → end of step 1).
            state["last_time"] = time.perf_counter()
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            encoder.shutdown(drain_timeout=5.0)
            for cls, prev in prev_methods:
                cls.decode_latent_to_preview_image = prev
            if torch.cuda.is_available():
                # the decode grows torch's pool and the pool keeps it; under dynamic VRAM that reservation
                # displaces weight pages for every later prompt, so hand it back before returning
                torch.cuda.empty_cache()
            if vae_restore_device is not None and self.vae is not None:
                try:
                    self.vae.first_stage_model.to(vae_restore_device)
                except Exception:
                    pass


class ModelPreviewOverrideKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ModelPreviewOverrideKJ",
            display_name="Model Preview Override",
            category="KJNodes/sampling",
            description=(
                "Adds a dedicated live-preview frame on this node, with overridable max resolution. "
                "Default ComfyUI preview caps at 512px; this node sends its own preview straight to a "
                "DOM widget on the node so pixel-space models (Chroma Radiance, ZImage, HiDream-O1, …) "
                "can be previewed at full sampler resolution."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Model to attach the preview override to."),
                io.Int.Input(
                    "max_resolution",
                    default=1024,
                    min=0,
                    max=8192,
                    step=8,
                    tooltip="Max preview side in pixels for the live widget. 0 = full sampler resolution (no downscale).",
                ),
                io.Int.Input(
                    "jpeg_quality",
                    default=80,
                    min=30,
                    max=100,
                    step=1,
                    tooltip="JPEG quality for the live preview transport.",
                ),
                io.Boolean.Input(
                    "suppress_default_preview",
                    default=True,
                    tooltip="Suppress the standard sampler-node preview overlay while sampling, so only this node's frame updates. Progress bar still advances normally.",
                ),
                io.Int.Input(
                    "preview_frames",
                    default=1,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Frames to sample from each video step's latent for animated preview. "
                            "1 = single frame (current behavior, fastest). >1 = animated WebP playing back at preview_fps. "
                            "Only applies to video models (5D latents); ignored for image models.",
                ),
                io.Int.Input(
                    "preview_fps",
                    default=12,
                    min=1,
                    max=60,
                    step=1,
                    tooltip="Playback FPS for the animated preview. Ignored when preview_frames=1, and when an "
                            "audio_vae is connected (the clip then plays in real time with its audio).",
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Optional LTX VAE for true-RGB previews. TAEHV-LTX = fast tiny decode "
                            "(VAE pinned to GPU). Any other LTX VAE = full-quality decode via "
                            "vae.decode() — MUCH slower per step.",
                ),
                io.Combo.Input(
                    "tiny_vae",
                    options=["none"] + folder_paths.get_filename_list("vae_approx"),
                    default="none",
                    optional=True,
                    tooltip="Tiny VAE decoder from models/vae_approx for true-RGB previews. "
                            "Overrides Latent2RGB and the 'vae' input.",
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip="Optional audio VAE for audio+video models (MiniMax H3). With preview_frames > 1 the "
                            "audio latent is decoded each step, muxed into the animated preview played back in "
                            "real time (preview_fps is ignored), and its spectrogram is drawn under the graphs.",
                ),
            ],
            outputs=[io.Model.Output(tooltip="Model with preview override attached.")],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, max_resolution, jpeg_quality, suppress_default_preview, preview_frames, preview_fps, vae=None, tiny_vae="none", audio_vae=None) -> io.NodeOutput:
        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "kj_preview_override",
            _PreviewOverrideWrapper(
                max_resolution, cls.hidden.unique_id, jpeg_quality, suppress_default_preview,
                preview_frames, preview_fps, vae, tiny_vae, audio_vae,
            ),
        )
        return io.NodeOutput(m)


class GetPreviewOverrideFramesKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetPreviewOverrideFramesKJ",
            display_name="Get Preview Override Frames",
            category="KJNodes/sampling",
            description=(
                "Returns the frames captured by Model Preview Override during the most recent sampling. "
                "Wire 'model' from Model Preview Override (the same one feeding the sampler) and 'after_sample' "
                "from after the sampler (LATENT/IMAGE) to enforce correct execution order."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The model output by Model Preview Override (used to locate the captured frames)."),
                io.MultiType.Input(
                    "after_sample",
                    [io.Latent, io.Image],
                    tooltip="Anything from after the sampler (LATENT or IMAGE). The value is ignored — it just forces this node to run after sampling.",
                ),
            ],
            outputs=[io.Image.Output(display_name="frames")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, after_sample) -> io.NodeOutput:
        wrappers = model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "kj_preview_override")
        if not wrappers:
            raise RuntimeError("Get Preview Override Frames: no Model Preview Override wrapper found on this model.")
        frames = wrappers[-1].frames
        if not frames:
            raise RuntimeError("Get Preview Override Frames: no frames captured. Ensure the sampler ran with this model.")
        tensors = []
        for pil in frames:
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
        return io.NodeOutput(torch.stack(tensors, dim=0))
