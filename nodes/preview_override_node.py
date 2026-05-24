import base64
import io as pyio
import logging
import queue
import threading
import time

import numpy as np
import torch

import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import latent_preview
from comfy_api.latest import io
from PIL import Image, ImageOps

# Share factor tables + TAEHV-LTX decode with the LTX-specific node rather than copy them.
from .ltxv_nodes import WrappedPreviewer as _LTXWrappedPreviewer, get_ltx_rgb_factors as _ltx_rgb_factors


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


# PyPI PyAV wheels typically lack NVENC; probe once at import.
def _probe_nvenc():
    try:
        import av  # noqa
        av.Codec("h264_nvenc", "w")
        return True
    except Exception:
        return False

_NVENC_AVAILABLE = _probe_nvenc()

# NVENC H.264 rejects sub-145×49 inputs at avcodec_open2 — fall back to WebP for small frames.
_NVENC_MIN_W = 145
_NVENC_MIN_H = 49

_nvenc_warned = False


def _encode_mp4_nvenc(frames, fps, max_res):
    # Fragmented MP4 so the browser can decode mid-download. Returns (None, 0, 0) on failure
    # (including too-small-for-NVENC), so caller falls through to WebP.
    global _nvenc_warned
    if not frames:
        return None, 0, 0
    try:
        import av
    except Exception:
        return None, 0, 0
    pil_frames = []
    for f in frames:
        pf = f if f.mode == "RGB" else f.convert("RGB")
        if max_res and max_res > 0 and (pf.width > max_res or pf.height > max_res):
            pf = ImageOps.contain(pf, (max_res, max_res), Image.LANCZOS)
        pil_frames.append(pf)
    # yuv420p requires even dimensions.
    w0, h0 = pil_frames[0].width, pil_frames[0].height
    out_w, out_h = w0 & ~1, h0 & ~1
    if (out_w, out_h) != (w0, h0):
        pil_frames = [pf.resize((out_w, out_h), Image.LANCZOS) for pf in pil_frames]
    if out_w < _NVENC_MIN_W or out_h < _NVENC_MIN_H:
        return None, 0, 0
    # Driver/GPU varies what option combos are accepted; bare preset always works.
    option_candidates = [
        {"preset": "p1", "rc": "vbr", "cq": "23"},
        {"preset": "p1"},
    ]
    last_err = None
    for opts in option_candidates:
        buf = pyio.BytesIO()
        try:
            container = av.open(
                buf, mode="w", format="mp4",
                options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
            )
            stream = container.add_stream("h264_nvenc", rate=int(max(1, fps)))
            stream.width = out_w
            stream.height = out_h
            stream.pix_fmt = "yuv420p"
            stream.options = opts
            for pf in pil_frames:
                for pkt in stream.encode(av.VideoFrame.from_image(pf)):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            return base64.b64encode(buf.getvalue()).decode("ascii"), out_w, out_h
        except Exception as e:
            last_err = e
            continue
    if not _nvenc_warned:
        _nvenc_warned = True
        logging.warning(f"[KJ PreviewOverride] NVENC MP4 encode failed, using WebP fallback: {last_err}")
    return None, 0, 0


def _encode_animated_webp(frames, fps, quality, max_res):
    if not frames:
        return None, 0, 0
    pil_frames = []
    for f in frames:
        pf = f
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


def _decode_audio_waveform(audio_first_stage, ax, target_device):
    # Decode on first_stage_model directly (VAE pinned to GPU in __call__) to skip comfy
    # VAE.decode()'s per-step load_models_gpu. first_stage_model.decode gives (B, C, samples).
    if ax is None or ax.numel() == 0:
        return None
    try:
        ax = ax.to(device=target_device, dtype=torch.float32)
        waveform = audio_first_stage.decode(ax)
    except Exception as e:
        logging.warning(f"[KJ PreviewOverride] audio VAE decode failed: {e}")
        return None
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)  # (B, samples) → (B, 1, samples)
    if waveform.ndim != 3:
        return None
    # Mirror comfy_extras.nodes_audio.vae_decode_audio loudness normalization.
    std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    waveform = waveform / std
    return waveform[0].clamp(-1.0, 1.0).detach().float().cpu()


def _encode_wav_b64(waveform_cpu, sample_rate):
    # 16-bit PCM WAV — browser-native via <audio>, no extra deps. waveform_cpu: (C, samples).
    if waveform_cpu is None or waveform_cpu.numel() == 0:
        return None
    try:
        import wave
        channels = waveform_cpu.shape[0]
        interleaved = (waveform_cpu.transpose(0, 1).contiguous().numpy() * 32767.0)
        interleaved = np.clip(interleaved, -32768, 32767).astype("<i2")
        buf = pyio.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(int(channels))
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(interleaved.tobytes())
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logging.warning(f"[KJ PreviewOverride] WAV encode failed: {e}")
        return None


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


def _normalize_ltx_x0(x0, latent_shapes, num_keyframes):
    # LTX flattens spatial+temporal into a token sequence and may append keyframe latents
    # at the tail. Restore 5D and trim so downstream previewers see standard video latents.
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
    def __init__(self, max_resolution, node_id, jpeg_quality, suppress_default, preview_frames=1, preview_fps=12, vae=None, audio_vae=None):
        self.max_resolution = max_resolution
        self.node_id = str(node_id) if node_id is not None else None
        self.jpeg_quality = jpeg_quality
        self.suppress_default = suppress_default
        self.preview_frames = preview_frames
        self.preview_fps = preview_fps
        self.vae = vae
        self.audio_vae = audio_vae
        self.frames = []

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes):
        guider = executor.class_obj
        model_patcher = guider.model_patcher

        is_ltx = _is_ltx_latent_format(model_patcher.model.latent_format)
        is_ltx2 = is_ltx and _is_ltx2_diffusion_model(model_patcher)
        num_keyframes = _ltx_num_keyframes(guider) if is_ltx else 0

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

        # Audio preview (LTXAV): >1 latent_shapes means the sampled latent packs an audio
        # tensor. Pin the audio VAE to GPU for the run; restored in finally.
        audio_first_stage = None
        audio_target_device = None
        audio_restore_device = None
        audio_sample_rate = 44100
        if self.audio_vae is not None and len(latent_shapes) > 1:
            try:
                audio_target_device = comfy.model_management.get_torch_device()
                for p in self.audio_vae.first_stage_model.parameters():
                    audio_restore_device = p.device
                    break
                self.audio_vae.first_stage_model.to(audio_target_device)
                audio_first_stage = self.audio_vae.first_stage_model
                audio_sample_rate = getattr(
                    self.audio_vae, "audio_sample_rate_output",
                    getattr(self.audio_vae, "audio_sample_rate", 44100),
                )
            except Exception as e:
                logging.warning(f"[KJ PreviewOverride] audio VAE setup failed: {e}")
                audio_first_stage = None

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
        initial_seed_cpu = None
        try:
            if sigmas is not None and len(sigmas) > 0:
                # sigmas often lives on CPU while noise is on CUDA — align before the multiply.
                s0 = sigmas[0].to(noise.device) if hasattr(sigmas[0], "to") else sigmas[0]
                seeded = noise * s0
                if is_ltx:
                    seeded = _normalize_ltx_x0(seeded, latent_shapes, num_keyframes)
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
                if is_ltx:
                    init_latent = _normalize_ltx_x0(init_latent, latent_shapes, num_keyframes)
                pil_init = None
                if ltx_previewer is not None and init_latent.ndim == 5:
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


        def new_callback(step, x0, x, total_steps_):
            if previewer is not None or fallback_previewer is not None or ltx_previewer is not None:
                try:
                    # NEVER rebind x0 — the sampler reuses the same tensor downstream
                    # (unpack_latents reshapes it). Preview mutations stay on x0_view.
                    x0_view = x0
                    if is_ltx:
                        x0_view = _normalize_ltx_x0(x0_view, latent_shapes, num_keyframes)

                    pil_frames = []
                    max_pil = anim_frames if animate_video else 1
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

                    if not pil_frames:
                        if original_callback is not None:
                            original_callback(step, x0, x, total_steps_)
                        return

                    pil_first = pil_frames[0]
                    if pil_first.mode != "RGB":
                        pil_first = pil_first.convert("RGB")
                        pil_frames[0] = pil_first
                    # Consumed by GetPreviewOverrideFramesKJ.
                    self.frames.append(pil_first)

                    if node_id is not None and PromptServer is not None:
                        # x0_view (not x0) so LTX keyframe padding doesn't dampen the Δ norm.
                        x0_cpu_now = x0_view.detach().float().cpu()
                        prev_x0_cpu = state["last_x0_cpu"]
                        state["last_x0_cpu"] = x0_cpu_now

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

                        # Decode audio inline from the raw packed x0; WAV/base64 packing is
                        # deferred to the async encoder below, like the image path.
                        audio_wave_cpu = None
                        if audio_first_stage is not None:
                            try:
                                unpacked = comfy.utils.unpack_latents(x0, latent_shapes)
                                if len(unpacked) > 1:
                                    audio_wave_cpu = _decode_audio_waveform(
                                        audio_first_stage, unpacked[1], audio_target_device,
                                    )
                            except Exception as e:
                                logging.warning(f"[KJ PreviewOverride] audio extract failed: {e}")

                        def _encode_and_send(
                            pil_frames=pil_frames, x0_cpu_now=x0_cpu_now, prev_x0_cpu=prev_x0_cpu,
                            step_ms=step_ms, avg_step_ms=avg_step_ms, sigma_val=sigma_val,
                            sent_step=sent_step, total_steps_=total_steps_,
                            audio_wave_cpu=audio_wave_cpu, audio_sample_rate=audio_sample_rate,
                        ):
                            if len(pil_frames) > 1:
                                # NVENC ~8x faster + ~5x smaller than PIL WebP when available.
                                b64, w_, h_, mime = None, 0, 0, None
                                if _NVENC_AVAILABLE:
                                    b64, w_, h_ = _encode_mp4_nvenc(pil_frames, anim_fps, max_res)
                                    if b64:
                                        mime = "video/mp4"
                                if not b64:
                                    b64, w_, h_ = _encode_animated_webp(pil_frames, anim_fps, quality, max_res)
                                    mime = "image/webp"
                            else:
                                pil_send = pil_frames[0]
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

                            payload = {
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
                                "fps": anim_fps if mime in ("video/mp4", "image/webp") else None,
                            }
                            audio_b64 = _encode_wav_b64(audio_wave_cpu, audio_sample_rate)
                            if audio_b64 is not None:
                                payload["audio"] = audio_b64
                                payload["audio_sample_rate"] = int(audio_sample_rate)
                                payload["audio_mime"] = "audio/wav"

                            PromptServer.instance.send_sync(
                                "kj_preview_override", payload, PromptServer.instance.client_id,
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
            if vae_restore_device is not None and self.vae is not None:
                try:
                    self.vae.first_stage_model.to(vae_restore_device)
                except Exception:
                    pass
            if audio_restore_device is not None and self.audio_vae is not None:
                try:
                    self.audio_vae.first_stage_model.to(audio_restore_device)
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
                    tooltip="Playback FPS for the animated WebP preview. Ignored when preview_frames=1.",
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Optional LTX VAE for true-RGB previews. TAEHV-LTX = fast tiny decode "
                            "(VAE pinned to GPU). Any other LTX VAE = full-quality decode via "
                            "vae.decode() — MUCH slower per step.",
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip="Optional LTXAV audio VAE. When connected, the audio is decoded and "
                            "previewed in sync with the video during sampling.",
                ),
            ],
            outputs=[io.Model.Output(tooltip="Model with preview override attached.")],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, max_resolution, jpeg_quality, suppress_default_preview, preview_frames, preview_fps, vae=None, audio_vae=None) -> io.NodeOutput:
        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "kj_preview_override",
            _PreviewOverrideWrapper(
                max_resolution, cls.hidden.unique_id, jpeg_quality, suppress_default_preview,
                preview_frames, preview_fps, vae, audio_vae,
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
