import math
from functools import lru_cache

import torch
import torch.nn.functional as F

# Image resampling for ImageResizeKJv2. Tensors are [B, C, H, W]; math runs in float32, output keeps the input dtype.

RESAMPLE_METHODS = ["bicubic_antialias", "lanczos_antiring", "ssim_downscale"]

_RADIUS = {"lanczos3": 3.0, "catrom": 2.0}
_LUMA = (0.2126, 0.7152, 0.0722)
_DENSE_MAX = 1 << 25  # axes with more n_in * n_out use the tap loop instead of a dense matrix
_CHUNK_ELEMS = 1 << 25  # input elements per batch chunk, keeps peak memory near 1 GB


def _kernel(name, x):
    if name == "lanczos3":
        return torch.sinc(x) * torch.sinc(x / 3)
    x = x.abs()  # Catmull-Rom
    return torch.where(x < 1, 1.5 * x ** 3 - 2.5 * x ** 2 + 1,
                       torch.where(x < 2, -0.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2, torch.zeros_like(x)))


@lru_cache(maxsize=64)
def _taps(n_in, n_out, kernel):
    # [taps, n_out] input indices and normalized weights; kernel widened by the scale when shrinking, edges replicated
    s = n_in / n_out
    scale = max(s, 1.0)
    radius = _RADIUS[kernel]
    centers = (torch.arange(n_out, dtype=torch.float64) + 0.5) * s - 0.5
    n = 2 * int(math.ceil(radius * scale)) + 2
    idx = torch.floor(centers - radius * scale).long()[:, None] + torch.arange(n)[None, :]
    d = (idx - centers[:, None]) / scale
    w = _kernel(kernel, d) * (d.abs() < radius)
    w = w / w.sum(1, keepdim=True)
    keep = (w != 0).any(0)
    return idx.clamp(0, n_in - 1)[:, keep].T.contiguous(), w[:, keep].T.float().contiguous()


@lru_cache(maxsize=16)
def _dense(n_in, n_out, kernel):
    idx, w = _taps(n_in, n_out, kernel)
    return torch.zeros(n_out, n_in).scatter_add_(1, idx.T.contiguous(), w.T.contiguous())


def _resample_axis(x, n_out, kernel, dim):
    n_in = x.shape[dim]
    if n_in * n_out <= _DENSE_MAX:
        M = _dense(n_in, n_out, kernel).to(x.device, x.dtype)
        return torch.matmul(x, M.T) if dim == -1 else torch.matmul(M, x)
    idx, w = _taps(n_in, n_out, kernel)
    idx, w = idx.to(x.device), w.to(x.device, x.dtype)
    shape = [1] * x.dim()
    shape[dim] = n_out
    out = x.index_select(dim, idx[0]) * w[0].view(shape)
    for t in range(1, idx.shape[0]):
        out.add_(x.index_select(dim, idx[t]) * w[t].view(shape))
    return out


def _resample(x, h, w, kernel):
    return _resample_axis(_resample_axis(x, w, kernel, -1), h, kernel, -2)


@lru_cache(maxsize=64)
def _lobe(n_in, n_out):
    # first input index and count of inputs inside each output's positive main lobe (within one output pixel)
    s = n_in / n_out
    scale = max(s, 1.0)
    c = (torch.arange(n_out, dtype=torch.float64) + 0.5) * s - 0.5
    start = torch.ceil(c - scale + 1e-9).long()
    return start, torch.floor(c + scale - 1e-9).long() - start + 1


def _window_reduce(x, n_out, dim, op):
    # min or max over each output's main-lobe window via a sparse table: level p holds the reduction of p inputs
    f = torch.minimum if op == "min" else torch.maximum
    start, width = _lobe(x.shape[dim], n_out)
    xs = x.movedim(dim, -1)
    lead = xs.shape[:-1]
    kmax = int(width.max())
    widths = torch.unique(width).tolist()
    needed = {1 << (k.bit_length() - 1) for k in widths}
    cur = F.pad(xs.reshape(-1, 1, xs.shape[-1]), (kmax, kmax), mode="replicate").squeeze(1)
    levels = {1: cur} if 1 in needed else {}
    p = 1
    while p < max(needed):
        cur = f(cur[:, :-p], cur[:, p:])
        p *= 2
        if p in needed:
            levels[p] = cur
    s0 = (start + kmax).to(x.device)
    out = None
    for k in widths:
        p = 1 << (k.bit_length() - 1)
        r = f(levels[p].index_select(-1, s0), levels[p].index_select(-1, s0 + (k - p)))
        out = r if out is None else torch.where((width == k).to(x.device), r, out)
    return out.reshape(lead + (n_out,)).movedim(-1, dim)


def _lanczos_antiring(x, h, w):
    # Lanczos3 per axis, each pass clamped to the min/max of the input under its main lobe (no halos)
    t = _resample_axis(x, w, "lanczos3", -1)
    t = torch.clamp(t, _window_reduce(x, w, -1, "min"), _window_reduce(x, w, -1, "max"))
    o = _resample_axis(t, h, "lanczos3", -2)
    return torch.clamp(o, _window_reduce(t, h, -2, "min"), _window_reduce(t, h, -2, "max"))


def _wmean3(x, k):
    C = x.shape[1]
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k.repeat(C, 1, 1, 1), groups=C)


def _ssim_downscale(x, h, w):
    # Oztireli & Gross 2015 contrast restore, one luma gain capped at 1.5, clamped to the source's local range
    L = _resample(x, h, w, "lanczos3")
    L2 = _resample(x * x, h, w, "catrom")
    d = torch.tensor([1.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
    k = 0.5 ** (d[:, None] + d[None, :])
    k = (k / k.sum()).view(1, 1, 3, 3)
    M = _wmean3(L, k)
    Sl = (_wmean3(L * L, k) - M * M).clamp(min=0)
    Sh = (_wmean3(L2, k) - M * M).clamp(min=0)
    if x.shape[1] == 3:
        lw = torch.tensor(_LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        Sl, Sh = (Sl * lw).sum(1, keepdim=True), (Sh * lw).sum(1, keepdim=True)
    R = torch.sqrt((Sh + 1.5e-4) / (Sl + 1.5e-4)).clamp(max=1.5)
    D = _wmean3(M, k) + _wmean3(R, k) * L - _wmean3(R * M, k)
    lo = _window_reduce(_window_reduce(x, w, -1, "min"), h, -2, "min")
    hi = _window_reduce(_window_reduce(x, w, -1, "max"), h, -2, "max")
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
        except NotImplementedError:  # backends without antialiased bicubic, e.g. MPS
            out = F.interpolate(x.cpu(), size=(height, width), mode="bicubic", antialias=True, align_corners=False).to(x.device)
    elif method == "lanczos":
        out = _resample(x, height, width, "lanczos3").clamp(0, 1)
    elif method == "ssim_downscale" and height <= H and width <= W:
        out = _ssim_downscale(x, height, width)
    else:  # lanczos_antiring, and ssim_downscale when any axis grows
        out = _lanczos_antiring(x, height, width)
    return out.to(dtype)
