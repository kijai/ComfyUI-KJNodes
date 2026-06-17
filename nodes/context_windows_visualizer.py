import inspect
import math

from comfy_api.latest import io

try:
    import comfy.context_windows as context_windows
except ImportError:
    context_windows = None

SCHEDULE_OPTIONS = ["standard_static", "standard_static_balanced", "standard_uniform", "looped_uniform", "batched", "batched_shifted"]
FUSE_OPTIONS = ["pyramid", "relative", "flat", "overlap-linear", "hann", "gaussian"]


def _ordered_halving(val):
    # low-discrepancy fraction in [0, 1) from bit-reversal of the step index (core's ordered_halving)
    return int(f"{int(val):064b}"[::-1], 2) / (1 << 64)


def create_windows_batched_shifted(num_frames, handler, model_options=None):
    # zero-overlap partition whose seams roll each step (no fixed boundary error); ~1.0x compute
    F = int(num_frames)
    L = int(handler.context_length)
    if F <= L:
        return [list(range(F))]
    offset = round(F * _ordered_halving(handler._step)) % L  # 0 at step 0 (plain batched)
    windows = []
    start = 0
    if offset > 0:
        windows.append(list(range(0, offset)))  # leading partial
        start = offset
    while start < F:
        end = min(start + L, F)
        windows.append(list(range(start, end)))  # trailing partial when unaligned
        start = end
    return windows


def create_windows_static_balanced(num_frames, handler, model_options=None):
    # standard_static window count, but spread evenly so every seam is uniform (±1 frame)
    F = int(num_frames)
    L = int(handler.context_length)
    overlap = int(handler.context_overlap)
    if F <= L:
        return [list(range(F))]
    stride_target = max(1, L - overlap)
    N = -(-(F - L) // stride_target) + 1  # ceil((F-L)/stride_target)+1; stride <= L, no gaps
    windows = []
    for i in range(N):
        start = math.floor(i * (F - L) / (N - 1) + 0.5)
        windows.append(list(range(start, start + L)))
    return windows


def create_weights_hann(length, full_length=None, idxs=None, handler=None, **kwargs):
    # smooth overlap-linear: flat interior, raised-cosine ramps in the overlaps (sum to 1)
    weights = [1.0] * length
    overlap = min(max(int(getattr(handler, "context_overlap", 0) or 0), 0), length)
    if overlap >= 1 and idxs:
        denom = max(overlap - 1, 1)
        if min(idxs) > 0:  # left ramp, skipped on first window
            for j in range(overlap):
                weights[j] = max(0.5 * (1 - math.cos(math.pi * j / denom)), 1e-37)
        if full_length is not None and max(idxs) < full_length - 1:  # right ramp, skipped on last
            for j in range(overlap):
                weights[length - overlap + j] = max(0.5 * (1 - math.cos(math.pi * (denom - j) / denom)), 1e-37)
    return weights


def create_weights_gaussian(length, **kwargs):
    # smooth bell over the window (smooth twin of pyramid)
    if length <= 1:
        return [1.0] * length
    c = (length - 1) / 2.0
    std = max(length / 4.0, 1e-6)
    return [math.exp(-0.5 * ((i - c) / std) ** 2) for i in range(length)]


LOCAL_FUSE = {"hann": create_weights_hann, "gaussian": create_weights_gaussian}
LOCAL_SCHEDULES = {
    "standard_static_balanced": create_windows_static_balanced,
    "batched_shifted": create_windows_batched_shifted,
}


class ContextWindowsVisualizerKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ContextWindowsVisualizerKJ",
            display_name="Context Windows Visualizer (KJ)",
            category="KJNodes/misc",
            description="Interactive visualization of ComfyUI context windows: window placement, overlap, "
                        "fuse weights and per-frame blending, updated live. Optionally applies the settings "
                        "to a model like the core Context Windows (Manual) node.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", optional=True, tooltip="Optional model to apply the context windows to during sampling."),
                io.Combo.Input("frame_units", options=["pixel", "latent"], default="pixel",
                               tooltip="Units for the frame counts below. 'pixel' converts to latent via ((n - 1) // temporal_downscale) + 1."),
                io.Int.Input("dim", default=2, min=0, max=5, tooltip="Latent temporal dimension: 2 for WAN-like, 0 for AnimateDiff-style. Model path only.", advanced=True),
                io.Int.Input("temporal_downscale", default=4, min=1, max=16,
                             tooltip="VAE temporal compression for pixel->latent. WAN/Hunyuan/CogVideoX=4, Mochi=6, LTXV=8."),
                io.Int.Input("num_frames", default=161, min=1, max=100000, tooltip="Total frames to visualize. Ignored on the model path (the real latent length is used)."),
                io.Int.Input("context_length", default=81, min=1, max=100000, tooltip="Window length."),
                io.Int.Input("context_overlap", default=30, min=0, max=100000, tooltip="Overlap between windows."),
                io.Combo.Input("context_schedule", options=SCHEDULE_OPTIONS, tooltip="Window placement. standard_static_balanced: static with windows spread evenly (uniform seams, any frame count). batched_shifted: zero-overlap windows whose seams roll each step (~1.0x compute, pairs with freenoise). Uniform schedules also shift per step."),
                io.Int.Input("context_stride", default=1, min=1, max=32, tooltip="Max stride power for uniform schedules (adds dilated windows: 1, 2, 4...)."),
                io.Boolean.Input("closed_loop", default=False, tooltip="Wrap looped-schedule windows around the end back to frame 0."),
                io.Combo.Input("fuse_method", options=FUSE_OPTIONS, default="pyramid", tooltip="How overlapping windows are blended."),
                io.Boolean.Input("causal_window_fix", default=False, tooltip="Prepend an anchor frame to non-0 windows, stripped after the model call.", advanced=True),
                io.Boolean.Input("freenoise", default=True, tooltip="FreeNoise noise shuffling for smoother blending. Model path only.", advanced=True),
                io.String.Input("cond_retain_index_list", default="", tooltip="Comma-separated window-relative indices that keep their original conditioning, e.g. '0' reuses the start frame's cond per window. No viz effect; model path only.", advanced=True),
            ],
            outputs=[
                io.Model.Output(display_name="model", tooltip="Model with context windows applied; only valid when a model is connected."),
            ],
        )

    @classmethod
    def execute(cls, frame_units, temporal_downscale, num_frames, context_length, context_overlap, context_schedule, context_stride,
                closed_loop, fuse_method, dim, causal_window_fix, freenoise, cond_retain_index_list="", model=None) -> io.NodeOutput:
        if model is None:
            return io.NodeOutput(None)
        if context_windows is None:
            raise Exception("comfy.context_windows not available in this ComfyUI version.")
        if frame_units == "pixel":
            f = max(int(temporal_downscale), 1)
            context_length = max(((context_length - 1) // f) + 1, 1)
            context_overlap = max(((context_overlap - 1) // f) + 1, 0)
        # build local schedules/fuse methods (not in core) as Context* objects
        if context_schedule in LOCAL_SCHEDULES:
            schedule = context_windows.ContextSchedule(context_schedule, LOCAL_SCHEDULES[context_schedule])
        else:
            schedule = context_windows.get_matching_context_schedule(context_schedule)
        if fuse_method in LOCAL_FUSE:
            fuse = context_windows.ContextFuseMethod(fuse_method, LOCAL_FUSE[fuse_method])
        else:
            fuse = context_windows.get_matching_fuse_method(fuse_method)
        model = model.clone()
        handler_kwargs = dict(
            context_schedule=schedule,
            fuse_method=fuse,
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=dim,
            freenoise=freenoise,
            causal_window_fix=causal_window_fix,
            cond_retain_index_list=cond_retain_index_list,
        )
        # drop kwargs this ComfyUI version's handler doesn't accept
        sig = inspect.signature(context_windows.IndexListContextHandler.__init__)
        handler_kwargs = {k: v for k, v in handler_kwargs.items() if k in sig.parameters}
        handler = context_windows.IndexListContextHandler(**handler_kwargs)
        model.model_options["context_handler"] = handler
        # scope the vram estimate to the context window length
        context_windows.create_prepare_sampling_wrapper(model)
        if freenoise and hasattr(context_windows, "create_sampler_sample_wrapper"):
            context_windows.create_sampler_sample_wrapper(model)
        return io.NodeOutput(model)
