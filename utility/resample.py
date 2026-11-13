import math
from functools import lru_cache

import torch
import torch.nn.functional as F

# Image resampling for ImageResizeKJv2. Tensors are [B, C, H, W]; math runs in float32, output keeps the input dtype.

RESAMPLE_METHODS = ["bicubic_antialias", "lanczos_antiring", "ssim_downscale"]

_RADIUS = {"lanczos3": 3.0, "catrom": 2.0}
_LUMA = (0.2126, 0.7152, 0.0722)
_CHUNK_ELEMS = 1 << 25  # input elements per batch chunk, keeps peak memory near 1 GB


def _kernel(name, x):
    if name == "lanczos3":
        return torch.sinc(x) * torch.sinc(x / 3)
    x = x.abs()  # Catmull-Rom
    return torch.where(x < 1, 1.5 * x ** 3 - 2.5 * x ** 2 + 1,
                       torch.where(x < 2, -0.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2, torch.zeros_like(x)))


def _centers(n_in, n_out):
    s = n_in / n_out
    return (torch.arange(n_out, dtype=torch.float64) + 0.5) * s - 0.5, max(s, 1.0)


@lru_cache(maxsize=16)
def _dense(n_in, n_out, kernel):
    # [n_out, n_in] weights, kernel widened by the scale when shrinking, edges replicated
    c, scale = _centers(n_in, n_out)
    r = _RADIUS[kernel] * scale
    idx = torch.floor(c - r).long()[:, None] + torch.arange(2 * math.ceil(r) + 2)
    d = (idx - c[:, None]) / scale
    w = _kernel(kernel, d) * (d.abs() < _RADIUS[kernel])
    w = w / w.sum(1, keepdim=True)
    return torch.zeros(n_out, n_in, dtype=torch.float64).scatter_add_(1, idx.clamp(0, n_in - 1), w).float()


@lru_cache(maxsize=64)
def _lobe(n_in, n_out):
    # first input index and count of inputs within one output pixel of each output's centre (the kernel's main lobe)
    c, scale = _centers(n_in, n_out)
    start = torch.ceil(c - scale + 1e-9).long()
    return start, torch.floor(c + scale - 1e-9).long() - start + 1


def _whole(n_in, n_out):
    return n_in > n_out and n_in % n_out == 0


def _strided(x, dim, n_out, s, left, k, op):
    # op over k-wide windows every s inputs along dim, after replicate-padding `left` inputs: the whole-number shrink shortcut
    B, C, H, W = x.shape
    right = max(0, s * (n_out - 1) + k - left - x.shape[dim])
    x = F.pad(x.reshape(B * C, 1, H, W), (left, right, 0, 0) if dim == -1 else (0, 0, left, right), mode="replicate")
    y = op(x, (1, k) if dim == -1 else (k, 1), (1, s) if dim == -1 else (s, 1))
    y = y[..., :n_out] if dim == -1 else y[..., :n_out, :]
    return y.reshape(B, C, y.shape[-2], y.shape[-1])


def _resample_axis(x, n_out, kernel, dim):
    n_in = x.shape[dim]
    if _whole(n_in, n_out):  # every output uses the same weights shifted by s: one strided convolution
        s = n_in // n_out
        row = _dense(16 * s, 16, kernel)[8]  # an interior row, clear of the edges
        nz = row.nonzero().flatten()
        w = row[nz[0]:nz[-1] + 1].to(x.device, x.dtype)
        return _strided(x, dim, n_out, s, 8 * s - int(nz[0]), w.numel(), lambda t, ks, st: F.conv2d(t, w.view(1, 1, *ks), stride=st))
    M = _dense(n_in, n_out, kernel).to(x.device, x.dtype)
    return torch.matmul(x, M.T) if dim == -1 else torch.matmul(M, x)


def _resample(x, h, w, kernel):
    return _resample_axis(_resample_axis(x, w, kernel, -1), h, kernel, -2)


def _window_max(x, n_out, dim):
    # max over each output's main-lobe window: one strided max-pool for whole-number shrinks, a sparse table otherwise
    n_in = x.shape[dim]
    start, width = _lobe(n_in, n_out)
    if _whole(n_in, n_out):
        return _strided(x, dim, n_out, n_in // n_out, -int(start[0]), int(width[0]), lambda t, ks, st: F.max_pool2d(t, ks, stride=st))
    xs = x.movedim(dim, -1)
    kmax, widths = int(width.max()), torch.unique(width).tolist()
    needed = {1 << (k.bit_length() - 1) for k in widths}
    t, p, levels = F.pad(xs.reshape(-1, 1, n_in), (kmax, kmax), mode="replicate").squeeze(1), 1, {}
    while True:  # level p holds the max of p consecutive inputs
        if p in needed:
            levels[p] = t
        if p == max(needed):
            break
        t, p = torch.maximum(t[:, :-p], t[:, p:]), p * 2
    s0 = (start + kmax).to(x.device)
    out = None
    for k in widths:
        p = 1 << (k.bit_length() - 1)
        r = torch.maximum(levels[p].index_select(-1, s0), levels[p].index_select(-1, s0 + k - p))
        out = r if out is None else torch.where((width == k).to(x.device), r, out)
    return out.reshape(xs.shape[:-1] + (n_out,)).movedim(-1, dim)


def _clamp_lobe(y, x, n_out, dim):
    return torch.clamp(y, -_window_max(-x, n_out, dim), _window_max(x, n_out, dim))


def _lanczos_antiring(x, h, w):
    # Lanczos3 per axis, each pass clamped to the input's min/max under the main lobe (no halos)
    t = _clamp_lobe(_resample_axis(x, w, "lanczos3", -1), x, w, -1)
    return _clamp_lobe(_resample_axis(t, h, "lanczos3", -2), t, h, -2)


def _ssim_downscale(x, h, w):
    # Oztireli & Gross 2015 contrast restore, one luma gain capped at 1.5, clamped to the source's local range
    L = _resample(x, h, w, "lanczos3")
    L2 = _resample(x * x, h, w, "catrom")
    k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 16
    mean = lambda t: F.conv2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), k.repeat(t.shape[1], 1, 1, 1), groups=t.shape[1])
    M = mean(L)
    Sl = (mean(L * L) - M * M).clamp(min=0)
    Sh = (mean(L2) - M * M).clamp(min=0)
    if x.shape[1] == 3:
        lw = torch.tensor(_LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        Sl, Sh = (Sl * lw).sum(1, keepdim=True), (Sh * lw).sum(1, keepdim=True)
    R = torch.sqrt((Sh + 1.5e-4) / (Sl + 1.5e-4)).clamp(max=1.5)
    D = mean(M) + mean(R) * L - mean(R * M)
    lo = -_window_max(_window_max(-x, w, -1), h, -2)
    hi = _window_max(_window_max(x, w, -1), h, -2)
    return torch.clamp(D, lo, hi)


def resize(x, width, height, method):
    # method is one of RESAMPLE_METHODS, or "lanczos" for plain float Lanczos3 clamped to [0, 1]
    H, W = x.shape[-2:]
    if (H, W) == (height, width):
        return x
    per = max(1, _CHUNK_ELEMS // max(1, x[0].numel()))
    if x.shape[0] > per:
        return torch.cat([resize(c, width, height, method) for c in x.split(per)])
    dtype = x.dtype
    x = x.float()
    if method == "bicubic_antialias":
        try:
            out = F.interpolate(x, size=(height, width), mode="bicubic", antialias=True, align_corners=False)
        except (NotImplementedError, RuntimeError):  # MPS lacks it, CUDA rejects very large shrink ratios
            out = F.interpolate(x.cpu(), size=(height, width), mode="bicubic", antialias=True, align_corners=False).to(x.device)
    elif method == "lanczos":
        out = _resample(x, height, width, "lanczos3").clamp(0, 1)
    elif method == "ssim_downscale" and height <= H and width <= W:
        out = _ssim_downscale(x, height, width)
    else:  # lanczos_antiring, and ssim_downscale when any axis grows
        out = _lanczos_antiring(x, height, width)
    return out.to(dtype)
