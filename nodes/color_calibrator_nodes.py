import math
import json

import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndi

from comfy_api.latest import io

_MIN_ANCHOR_PCT = 0.02
_WAVELET_LEVELS = 3
_WAVELET_SIGMA = 2.0
_SCALE_CLAMP = (0.5, 2.0)
_RANGE_CLAMP = (0.3, 3.0)
_RANGE_CLIP_PCT = 0.005
_RANGE_DOWNSCALE = 4
_FEATHER_DOWNSCALE = 4
_WAVELET_DOWNSCALE = 4
_PREVIEW_MAX_SIDE = 512
_PREVIEW_WEBP_QUALITY = 85
_PREVIEW_WEBP_METHOD = 4

# ---- 工作图上限：所有模式都缩到不超过此长边，再算 mask ----
_WORK_MAX_SIDE = 1024

# ---- 自动遮罩增强：内部护栏 ----
_USE_CHROMA = True
_USE_CHROMA_ADAPT = True
_USE_AUTO_THR = True

_CHROMA_W_LUMA = 0.4
_CHROMA_W_CHROMA_MAX = 1.2
_CHROMA_W_CHROMA_MIN = 1.0
_COLOR_SHIFT_RATIO_THR = 1.5
_COLOR_SHIFT_RATIO_CLAMP = (0.0, 4.0)

_DIFF_PREBLUR_SIGMA = 0.8

_AUTO_THR_FRAC_MAX = 0.30
_AUTO_THR_CLAMP_MULT = 2.0
_AUTO_THR_MAD_K = 3.0
_AUTO_THR_MAD_FLOOR = 0.002

# ---- auto 闭环：threshold 拐点扫描（1024² 工作尺度）----
_THR_MIN, _THR_MAX = 0.03, 0.10
_THR_STEP = 0.01

# ---- auto 闭环：open_k（1024² 工作尺度）----
_OPEN_MIN, _OPEN_MAX = 3, 33
_OPEN_STEP = 2
_OPEN_ITERS = 16

# ---- auto 闭环：close_k / merge（1024² 工作尺度）----
_CLOSE_MIN, _CLOSE_MAX = 21, 129
_CLOSE_STEP = 8
_CLOSE_ITERS = 16
_MERGE_DILATE_MIN, _MERGE_DILATE_MAX = 2, 32
_MERGE_DILATE_STEP = 2

# ---- auto 闭环：判据（1024² 工作尺度像素面积）----
_NOISE_AREA_PX = 32
_SMALL_BLOB_AREA_PX = 128
_SMALL_BLOB_TARGET = 0
_HOLE_AREA_PX = 64
_HOLE_TARGET = 1
_FG_BLOB_TARGET = 2

# ---- 阶段 4：逐连通域 expand/feather（1024² 工作尺度）----
_EXPAND_RATIO = 0.15          # auto 基准：sqrt(max_area/π) * 0.15
_EXPAND_MIN, _EXPAND_MAX = 8, 64
_FEATHER_MIN, _FEATHER_MAX = 4, 32

_DEBUG_TUNE = False


# ============================================================
# 基础工具
# ============================================================


def _to_bhwc(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t


def _split_rgb_alpha(img: torch.Tensor):
    c = img.shape[-1]
    if c >= 4:
        return img[..., :3], img[..., 3:4]
    if c == 3:
        return img, None
    if c == 1:
        return img.expand(-1, -1, -1, 3), None
    rgb = img[..., :3]
    if rgb.shape[-1] < 3:
        pad = torch.zeros(*rgb.shape[:-1], 3 - rgb.shape[-1],
                          dtype=rgb.dtype, device=rgb.device)
        rgb = torch.cat([rgb, pad], dim=-1)
    return rgb, None


def _merge_rgb_alpha(rgb: torch.Tensor, alpha):
    if alpha is None:
        return rgb
    return torch.cat([rgb, alpha], dim=-1)


def _mask_to_bhw1(mask: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() == 4:
        if mask.shape[-1] == 1:
            m = mask
        elif mask.shape[1] == 1:
            m = mask.permute(0, 2, 3, 1)
        else:
            m = mask[..., :1]
    else:
        m = mask.unsqueeze(-1)

    if m.shape[1] != h or m.shape[2] != w:
        m = F.interpolate(
            m.permute(0, 3, 1, 2), size=(h, w),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)

    if m.shape[0] == 1 and b > 1:
        m = m.expand(b, -1, -1, -1)
    elif m.shape[0] != b:
        m = m[:1].expand(b, -1, -1, -1)

    return m.clamp(0.0, 1.0)


def _work_size(H: int, W: int, cap: int = _WORK_MAX_SIDE):
    long_edge = max(H, W)
    if long_edge <= cap:
        return H, W, 1.0
    s = cap / long_edge
    hw = max(1, int(round(H * s)))
    ww = max(1, int(round(W * s)))
    return hw, ww, max(H / hw, W / ww)


def _down_to(x_bhwc: torch.Tensor, hw: int, ww: int) -> torch.Tensor:
    return F.interpolate(
        x_bhwc.permute(0, 3, 1, 2), size=(hw, ww),
        mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1)


def _up_to(x_bhw1: torch.Tensor, H: int, W: int) -> torch.Tensor:
    if x_bhw1.shape[1] == H and x_bhw1.shape[2] == W:
        return x_bhw1
    return F.interpolate(
        x_bhw1.permute(0, 3, 1, 2), size=(H, W),
        mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1)


# ============================================================
# 单通道高斯（差异图预模糊）
# ============================================================

def _gaussian_blur_single(x_bhw1: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x_bhw1
    k = int(math.ceil(sigma * 3)) * 2 + 1
    xs = torch.arange(k, dtype=torch.float32, device=x_bhw1.device) - k // 2
    g = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    g = g / g.sum()

    x = x_bhw1.permute(0, 3, 1, 2)
    gx = g.view(1, 1, 1, k)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), gx)
    gy = g.view(1, 1, k, 1)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), gy)
    return x.permute(0, 2, 3, 1)


# ============================================================
# 差异度量
# ============================================================

def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack([y, cb, cr], dim=-1)


def _soft_sat(v: torch.Tensor) -> torch.Tensor:
    return v / (1.0 + v)


def _diff_map(I0: torch.Tensor, I1: torch.Tensor) -> torch.Tensor:
    d_rgb = (I0 - I1).abs()

    if not _USE_CHROMA:
        diff = d_rgb.max(dim=-1, keepdim=True).values
    else:
        a = _rgb_to_ycbcr(I0)
        b = _rgb_to_ycbcr(I1)
        dy = (a[..., 0] - b[..., 0]).abs()
        dcb = (a[..., 1] - b[..., 1]).abs()
        dcr = (a[..., 2] - b[..., 2]).abs()
        chroma_raw = torch.sqrt(dcb * dcb + dcr * dcr)

        w_chroma = _CHROMA_W_CHROMA_MAX
        if _USE_CHROMA_ADAPT:
            med_c = chroma_raw.reshape(-1).median()
            med_l = dy.reshape(-1).median()
            ratio = (med_c / (med_l + 1e-4))
            ratio = ratio.clamp(*_COLOR_SHIFT_RATIO_CLAMP).item()
            if ratio > _COLOR_SHIFT_RATIO_THR:
                drop = min((ratio - _COLOR_SHIFT_RATIO_THR) * 0.2, 0.2)
                w_chroma = max(_CHROMA_W_CHROMA_MIN,
                               _CHROMA_W_CHROMA_MAX - drop)

        chroma = _soft_sat(chroma_raw)
        diff = _CHROMA_W_LUMA * dy + w_chroma * chroma
        diff = diff.unsqueeze(-1)

    if _DIFF_PREBLUR_SIGMA > 0:
        diff = _gaussian_blur_single(diff, _DIFF_PREBLUR_SIGMA)

    return diff


def _mad_threshold(diff: torch.Tensor, user_thr: float) -> float:
    if not _USE_AUTO_THR:
        return float(user_thr)

    d = diff.reshape(-1)
    changed_frac = (d > user_thr).float().mean().item()
    if changed_frac >= _AUTO_THR_FRAC_MAX:
        return float(user_thr)

    med = d.median()
    mad = (d - med).abs().median()
    mad = torch.clamp(mad, min=_AUTO_THR_MAD_FLOOR)
    auto_thr = (med + _AUTO_THR_MAD_K * 1.4826 * mad).item()

    lo = float(user_thr)
    hi = float(user_thr) * _AUTO_THR_CLAMP_MULT
    return float(min(max(auto_thr, lo), hi))


# ============================================================
# 高斯模糊（多通道）
# ============================================================

def _gaussian_blur_bhwc(rgb: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return rgb
    k = int(math.ceil(sigma * 3)) * 2 + 1
    xs = torch.arange(k, dtype=torch.float32, device=rgb.device) - k // 2
    g = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    g = g / g.sum()

    x = rgb.permute(0, 3, 1, 2)
    c = x.shape[1]

    gx = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), gx, groups=c)
    gy = g.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), gy, groups=c)
    return x.permute(0, 2, 3, 1)


# ============================================================
# 形态学
# ============================================================

def _odd(k: int) -> int:
    if k <= 1:
        return k
    return k if (k % 2 == 1) else k + 1


def _morph_open(mask_bhw1: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return mask_bhw1
    k = _odd(k)
    x = mask_bhw1.permute(0, 3, 1, 2)
    x = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x.permute(0, 2, 3, 1)


def _morph_close(mask_bhw1: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return mask_bhw1
    k = _odd(k)
    x = mask_bhw1.permute(0, 3, 1, 2)
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    x = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)
    return x.permute(0, 2, 3, 1)


def _dilate_pool(mask_bhw1: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return mask_bhw1
    k = _odd(k)
    x = mask_bhw1.permute(0, 3, 1, 2)
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x.permute(0, 2, 3, 1)


def _dilate(mask_bhw1: torch.Tensor, k: int) -> torch.Tensor:
    return _dilate_pool(mask_bhw1, k)


def _dilate_edt_half(mask_bhw1: torch.Tensor, k: int, downscale: int = 2) -> torch.Tensor:
    # 保留兼容，当前工作尺度 k 不会超过该分支所需，实际走 _dilate_pool
    if k <= 0:
        return mask_bhw1
    B, H, W, _ = mask_bhw1.shape
    sh = max(H // downscale, 1)
    sw = max(W // downscale, 1)
    m = mask_bhw1.permute(0, 3, 1, 2)
    m_small = F.interpolate(m, size=(sh, sw), mode="bilinear", align_corners=False)
    k_small = max(k // downscale, 1)

    device = m_small.device
    outs = []
    for i in range(B):
        arr = m_small[i, 0].detach().cpu().numpy()
        fg = arr > 0.5
        if not fg.any():
            outs.append(torch.zeros_like(m_small[i, 0]))
            continue
        dist = ndi.distance_transform_edt(~fg)
        outs.append(torch.from_numpy((dist <= k_small).astype(np.float32)))
    out = torch.stack(outs, dim=0).unsqueeze(1).to(device)

    out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    return out.permute(0, 2, 3, 1).clamp(0, 1)


# ============================================================
# 羽化
# ============================================================

def _feather_mask_full(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    sigma = max(radius / 3.0, 0.5)
    xs = torch.arange(k, dtype=torch.float32, device=mask.device) - radius
    g = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    g = g / g.sum()

    x = mask.permute(0, 3, 1, 2)
    gx = g.view(1, 1, 1, k)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), gx)
    gy = g.view(1, 1, k, 1)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), gy)
    return x.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _feather_mask_fast(mask_bhw1: torch.Tensor, radius: int,
                       downscale: int = _FEATHER_DOWNSCALE) -> torch.Tensor:
    if radius <= 0:
        return mask_bhw1

    B, H, W, _ = mask_bhw1.shape
    small_h = max(H // downscale, 1)
    small_w = max(W // downscale, 1)

    if small_h < 4 or small_w < 4:
        return _feather_mask_full(mask_bhw1, radius)

    m = mask_bhw1.permute(0, 3, 1, 2)
    m_small = F.interpolate(m, size=(small_h, small_w),
                            mode="bilinear", align_corners=False)

    small_radius = max(radius // downscale, 1)
    m_small = _feather_mask_full(m_small.permute(0, 2, 3, 1), small_radius)

    m_back = F.interpolate(
        m_small.permute(0, 3, 1, 2), size=(H, W),
        mode="bilinear", align_corners=False,
    )
    return m_back.permute(0, 2, 3, 1).clamp(0, 1)


# ============================================================
# 高斯金字塔 / 仿射 / 统计（工作图尺度，ds=1）
# ============================================================

def _gaussian_pyramid(rgb: torch.Tensor, levels: int, sigma: float):
    pyr = [rgb]
    for _ in range(levels):
        pyr.append(_gaussian_blur_bhwc(pyr[-1], sigma))
    low = pyr[-1]
    highs = [pyr[i] - pyr[i + 1] for i in range(len(pyr) - 1)]
    return low, highs


def _compute_affine_params(I0, I1, A):
    device = I0.device
    scale = torch.ones(3, device=device)
    shift = torch.zeros(3, device=device)

    w = A.reshape(-1)
    w_sum = w.sum()
    if w_sum.item() < 32:
        return scale, shift

    inv_n = 1.0 / w_sum

    for c in range(3):
        s = I1[..., c].reshape(-1)
        d = I0[..., c].reshape(-1)

        mu1 = (w * s).sum() * inv_n
        mu0 = (w * d).sum() * inv_n
        var1 = (w * (s - mu1) ** 2).sum() * inv_n
        var0 = (w * (d - mu0) ** 2).sum() * inv_n
        sd1 = torch.sqrt(var1.clamp_min(1e-12))
        sd0 = torch.sqrt(var0.clamp_min(1e-12))

        if sd1.item() < 1e-4:
            scale[c] = 1.0
            shift[c] = mu0 - mu1
        else:
            sc = (sd0 / sd1).clamp(*_SCALE_CLAMP)
            scale[c] = sc
            shift[c] = mu0 - sc * mu1

    return scale, shift


def _apply_affine(rgb, scale, shift):
    return (rgb * scale.view(1, 1, 1, 3) + shift.view(1, 1, 1, 3)).clamp(0, 1)


def _count_components(mask_bhw1, min_area_px=0):
    B = mask_bhw1.shape[0]
    m = mask_bhw1.permute(0, 3, 1, 2)
    totals = []
    smalls = []
    for i in range(B):
        arr = (m[i, 0].detach().cpu().numpy() > 0.5)
        labeled, n = ndi.label(arr)
        totals.append(int(n))
        if n == 0:
            smalls.append(0)
            continue
        areas = ndi.sum(arr, labeled, index=range(1, n + 1))
        smalls.append(int((areas < max(1, min_area_px)).sum()))
    return totals, smalls


def _count_holes(mask_bhw1, min_area_px=0):
    B = mask_bhw1.shape[0]
    m = mask_bhw1.permute(0, 3, 1, 2)
    holes = []
    for i in range(B):
        fg = (m[i, 0].detach().cpu().numpy() > 0.5)
        bg = ~fg
        labeled, n = ndi.label(bg)
        if n == 0:
            holes.append(0)
            continue

        border_labels = set()
        border_labels.update(labeled[0, :].tolist())
        border_labels.update(labeled[-1, :].tolist())
        border_labels.update(labeled[:, 0].tolist())
        border_labels.update(labeled[:, -1].tolist())
        border_labels.discard(0)

        hole_count = 0
        for lab in range(1, n + 1):
            if lab in border_labels:
                continue
            area = (labeled == lab).sum()
            if area < max(1, min_area_px):
                hole_count += 1
        holes.append(hole_count)
    return holes


def _remove_small_blobs(mask_bhw1, min_area_px):
    B = mask_bhw1.shape[0]
    out = mask_bhw1.clone()
    for i in range(B):
        arr = (out[i, ..., 0].detach().cpu().numpy() > 0.5)
        labeled, n = ndi.label(arr)
        if n == 0:
            continue
        keep = np.zeros(n + 1, dtype=np.float32)
        areas = ndi.sum(arr, labeled, index=range(1, n + 1))
        for lab in range(1, n + 1):
            if areas[lab - 1] >= min_area_px:
                keep[lab] = 1.0
        cleaned = keep[labeled]
        out[i, ..., 0] = torch.from_numpy(cleaned).to(out.device)
    return out.clamp(0, 1)


# ============================================================
# 阶段 1：threshold Kneedle 拐点
# ============================================================

def _pick_threshold_by_knee(diff, thr_min, thr_max, step):
    ts = []
    noises = []
    t = float(thr_min)
    while t <= thr_max + 1e-6:
        ts.append(round(t, 4))
        m = (diff > t).float()
        _, smalls = _count_components(m, min_area_px=_NOISE_AREA_PX)
        noises.append(int(max(smalls)) if smalls else 0)
        t += step

    n = len(ts)
    if n <= 2:
        return ts[0], noises

    noise_max, noise_min = max(noises), min(noises)

    if (noise_max - noise_min) < max(3, int(0.03 * noise_max)):
        if _DEBUG_TUNE:
            print(f"[ColorCalibrator][auto] stage=1 flat curve, "
                  f"thr={ts[0]} noises={noises}")
        return ts[0], noises

    x = [i / (n - 1) for i in range(n)]
    span = max(noise_max - noise_min, 1)
    y = [(v - noise_min) / span for v in noises]
    y0, y1 = y[0], y[-1]
    s = (y1 - y0)
    dist = [abs(yy - (y0 + s * xx)) for xx, yy in zip(x, y)]
    knee = max(1, min(dist.index(max(dist)), n - 2))

    if _DEBUG_TUNE:
        print(f"[ColorCalibrator][auto] stage=1 knee thr={ts[knee]} "
              f"noises={noises} knee_idx={knee}")
    return ts[knee], noises


# ============================================================
# 阶段 4：逐连通域 expand/feather
# ============================================================

def _expand_feather_by_components(mask_bhw1, H, W, mode, ui_expand, ui_feather):
    """
    对每个连通域按面积开方比例独立 expand/feather。
    mode="auto": base_expand = sqrt(max_area/π)*0.15, base_feather = base_expand*0.5
    mode="ui"  : base_expand = ui_expand, base_feather = ui_feather
    返回 (mask_after_expand, mask_after_feather, info)
    info = {"expand_max": int, "feather_max": int, "n_components": int, "max_area": int}
    """
    B = mask_bhw1.shape[0]
    out = torch.zeros_like(mask_bhw1)
    expand_max_ret = _EXPAND_MIN
    feather_max_ret = _FEATHER_MIN
    n_comp_ret = 0
    max_area_ret = 0

    for bi in range(B):
        arr = (mask_bhw1[bi, ..., 0].detach().cpu().numpy() > 0.5)
        labeled, n = ndi.label(arr)
        if n == 0:
            continue
        n_comp_ret = max(n_comp_ret, int(n))
        areas = ndi.sum(arr, labeled, index=range(1, n + 1)).astype(np.float64)
        max_area = float(areas.max()) if len(areas) else 0.0
        max_area_ret = max(max_area_ret, int(max_area))
        if max_area <= 0:
            continue

        if mode == "auto":
            base_expand = math.sqrt(max_area / math.pi) * _EXPAND_RATIO
            base_feather = base_expand * 0.5
        else:
            base_expand = float(ui_expand)
            base_feather = float(ui_feather)

        # 先算每个域的 expand_i
        expand_list = []
        for lab in range(1, n + 1):
            area_i = float(areas[lab - 1])
            if area_i <= 0:
                expand_list.append(0)
                continue
            ei = base_expand * math.sqrt(area_i / max_area)
            ei = int(round(ei))
            ei = max(_EXPAND_MIN, min(ei, _EXPAND_MAX))
            expand_list.append(ei)

        idx_max = int(np.argmax(areas))  # 最大域索引（0-based 在 areas 里）
        expand_of_max = max(expand_list[idx_max], 1)
        expand_max_ret = max(expand_max_ret, expand_of_max)

        # 逐域 dilate + 逐域 feather，取 max 合并
        acc = np.zeros_like(arr, dtype=np.float32)
        for lab in range(1, n + 1):
            if expand_list[lab - 1] <= 0:
                continue
            comp = (labeled == lab).astype(np.float32)
            comp_t = torch.from_numpy(comp).to(mask_bhw1.device).view(1, H, W, 1)
            comp_d = _dilate_pool(comp_t, expand_list[lab - 1])

            fi = int(round(base_feather * expand_list[lab - 1] / expand_of_max))
            fi = max(_FEATHER_MIN, min(fi, _FEATHER_MAX))
            feather_max_ret = max(feather_max_ret, fi)
            comp_f = _feather_mask_full(comp_d, fi) if fi > 0 else comp_d
            comp_np = comp_f[0, ..., 0].detach().cpu().numpy()
            acc = np.maximum(acc, comp_np)

        out[bi, ..., 0] = torch.from_numpy(acc).to(out.device)

    info = {
        "expand_max": int(expand_max_ret),
        "feather_max": int(feather_max_ret),
        "n_components": int(n_comp_ret),
        "max_area": int(max_area_ret),
    }
    return out.clamp(0, 1), info


# ============================================================
# 开环遮罩（default / min / manual，工作图尺度）
# ============================================================

def _auto_detect_mask(I0, I1, threshold, open_k, close_k, H, W):
    diff = _diff_map(I0, I1)
    thr_eff = _mad_threshold(diff, float(threshold))
    mask = (diff > thr_eff).float()

    if open_k > 1:
        mask = _morph_open(mask, open_k)
    if close_k > 1:
        mask = _morph_close(mask, close_k)
    return mask.clamp(0, 1)


# ============================================================
# 主节点
# ============================================================


def _auto_tune_mask(I0, I1, init_thr, init_open, init_close, H, W):
    diff = _diff_map(I0, I1)

    # ---------- 阶段 1：threshold Kneedle ----------
    thr, noises = _pick_threshold_by_knee(
        diff, _THR_MIN, _THR_MAX, _THR_STEP
    )

    # ---------- 阶段 2：open_k 清小色块 ----------
    base_mask = (diff > thr).float()
    open_k = max(int(init_open), _OPEN_MIN)
    open_k = min(open_k, _OPEN_MAX)
    best_open = open_k
    for _ in range(_OPEN_ITERS):
        m = _morph_open(base_mask, best_open) if best_open > 1 else base_mask
        totals, smalls = _count_components(m, min_area_px=_SMALL_BLOB_AREA_PX)
        if _DEBUG_TUNE:
            print(f"[ColorCalibrator][auto] stage=2 open_k={best_open} "
                  f"small={smalls} total={totals}")
        if max(smalls) <= _SMALL_BLOB_TARGET:
            break
        if best_open >= _OPEN_MAX:
            break
        best_open = min(best_open + _OPEN_STEP, _OPEN_MAX)
    open_k = best_open

    mask_after_open = _morph_open(base_mask, open_k) if open_k > 1 else base_mask
    mask_after_open = _remove_small_blobs(mask_after_open, _SMALL_BLOB_AREA_PX)
    if _DEBUG_TUNE:
        t2, s2 = _count_components(mask_after_open, min_area_px=_SMALL_BLOB_AREA_PX)
        print(f"[ColorCalibrator][auto] stage=2 cleanup small={s2} total={t2}")

    # ---------- 阶段 3：close_k + merge_dilate ----------
    close_k = max(int(init_close), _CLOSE_MIN)
    close_k = min(close_k, _CLOSE_MAX)
    best_close = close_k
    best_merge_dilate = 0

    for _ in range(_CLOSE_ITERS):
        m = _morph_close(mask_after_open, best_close) if best_close > 1 else mask_after_open
        if best_merge_dilate > 0:
            m = _dilate_pool(mask_after_open, best_merge_dilate)
            m = _morph_close(m, best_close) if best_close > 1 else m
        holes = _count_holes(m, min_area_px=_HOLE_AREA_PX)
        totals, _ = _count_components(m, min_area_px=0)
        if _DEBUG_TUNE:
            print(f"[ColorCalibrator][auto] stage=3 close_k={best_close} "
                  f"merge_dilate={best_merge_dilate} holes={holes} fg={totals}")
        if max(holes) <= _HOLE_TARGET and max(totals) <= _FG_BLOB_TARGET:
            break
        if best_close < _CLOSE_MAX:
            best_close = min(best_close + _CLOSE_STEP, _CLOSE_MAX)
            continue
        if best_merge_dilate < _MERGE_DILATE_MAX:
            best_merge_dilate = min(
                best_merge_dilate + _MERGE_DILATE_STEP, _MERGE_DILATE_MAX
            )
        else:
            break

    mask_final = mask_after_open
    if best_merge_dilate > 0:
        mask_final = _dilate_pool(mask_final, best_merge_dilate)
    if best_close > 1:
        mask_final = _morph_close(mask_final, best_close)
    mask_final = mask_final.clamp(0, 1)

    # ---------- 阶段 4：逐连通域 expand/feather（auto 基准） ----------
    mask_final, info = _expand_feather_by_components(
        mask_final, H, W, mode="auto", ui_expand=0, ui_feather=0
    )
    if _DEBUG_TUNE:
        print(f"[ColorCalibrator][auto] stage=4 {info}")

    tuned = {
        "threshold": round(float(thr), 4),
        "open_k": int(open_k),
        "close_k": int(best_close),
        "expand_max": int(info["expand_max"]),
        "feather_max": int(info["feather_max"]),
        "n_components": int(info["n_components"]),
        "max_area": int(info["max_area"]),
    }
    print(f"[ColorCalibrator] auto-tune done: {tuned}")
    return mask_final, tuned

# --------------------------------------------------------
# 遮罩外对比测范围拉伸
# --------------------------------------------------------


def _restore_contrast_by_anchor(I0, I1_corr, A):
    B, H, W, _ = I0.shape
    ds = _RANGE_DOWNSCALE
    sh, sw = max(H // ds, 1), max(W // ds, 1)

    def _down(x):
        return F.interpolate(
            x.permute(0, 3, 1, 2), size=(sh, sw),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)

    I0_s = _down(I0)
    I1_s = _down(I1_corr)
    A_s = _down(A)

    def _weighted_q(vals, w, q):
        order = torch.argsort(vals)
        v = vals[order]
        ww = w[order]
        cum = torch.cumsum(ww, 0)
        total = cum[-1]
        if total.item() <= 0:
            return None
        target = total * q
        idx = torch.searchsorted(cum, target).clamp(0, len(v) - 1)
        return v[idx]

    out = I1_corr.clone()
    clip_pct = _RANGE_CLIP_PCT
    for bi in range(B):
        w = A_s[bi, ..., 0].reshape(-1)
        if w.sum().item() < 32:
            continue
        g0 = I0_s[bi].mean(dim=-1).reshape(-1)
        g1 = I1_s[bi].mean(dim=-1).reshape(-1)

        L0 = _weighted_q(g0, w, clip_pct)
        H0 = _weighted_q(g0, w, 1.0 - clip_pct)
        L1 = _weighted_q(g1, w, clip_pct)
        H1 = _weighted_q(g1, w, 1.0 - clip_pct)
        if L0 is None or L1 is None:
            continue

        span0 = H0 - L0
        span1 = H1 - L1
        if span0.item() < 1e-3 or span1.item() < 1e-3:
            continue

        a = (span1 / span0).clamp(*_RANGE_CLAMP)
        out[bi] = ((I1_corr[bi] - L1) / a + L1).clamp(0, 1)

    return out

# --------------------------------------------------------
# 小波融合
# --------------------------------------------------------


def _wavelet_merge(I0, I1_corr, M, feather):
    M_soft = _feather_mask_fast(M, feather) if feather > 0 else M
    B, H, W, _ = I0.shape
    ds = _WAVELET_DOWNSCALE
    sh, sw = max(H // ds, 1), max(W // ds, 1)

    def _down(x):
        return F.interpolate(
            x.permute(0, 3, 1, 2), size=(sh, sw),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)

    def _up(x):
        return F.interpolate(
            x.permute(0, 3, 1, 2), size=(H, W),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)

    low0_s, _ = _gaussian_pyramid(_down(I0), _WAVELET_LEVELS, _WAVELET_SIGMA)
    low1_s, _ = _gaussian_pyramid(_down(I1_corr), _WAVELET_LEVELS, _WAVELET_SIGMA)
    M_soft_s = _down(M_soft)

    delta_s = (1.0 - M_soft_s) * (low0_s - low1_s)
    delta = _up(delta_s)

    out = I1_corr + delta
    return out.clamp(0, 1)


class ColorCalibratorKJ(io.ComfyNode):
    """
    Color Calibrator KJ

    忠实移植自 zhoutingxi/ComfyUI-BlueBox (MIT) 的 ColorCalibratorBlueBox 节点（v1.5.x）。

    所有模式的遮罩都先在 <=1024 长边的工作图上生成，再等比放大回原图。
    阶段 4 的 expand/feather 逐连通域按面积比例处理。
    param_preset: auto = 四阶段闭环自动校正；manual = 用参数框当前值。

    输出 IMAGE / MASK / FLOAT(changed_ratio) / STRING(tuned_params JSON)。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorCalibratorKJ",
            display_name="Color Calibrator KJ",
            category="KJNodes/image",
            search_aliases=["color calibrate", "calibrator", "color restore", "色彩还原", "校色", "色彩校准"],
            description=(
                "Restores original colors to an edited image, region-aware.\n\n"
                "Compares image_original with image_edited: unless a mask is given it auto-detects the "
                "changed region on a <=1024px working image using a chroma-aware YCbCr difference and a "
                "MAD-adaptive threshold. param_preset=auto runs a four-stage closed loop (threshold knee, "
                "open_k speck removal, close_k/merge hole filling, per-component expand/feather). It then "
                "uses the unchanged region as an anchor to (1) estimate a per-channel affine color "
                "correction, (2) recompress contrast around the edit's own black point, and (3) blend/paste "
                "the original outside the edited region (edited = wavelet low-frequency merge, original = "
                "strict paste, corrected = only the contrast-restored edit).\n"
                "Outputs image, mask, changed pixels ratio and the tuned parameters as JSON."
            ),
            inputs=[
                io.Image.Input("image_original", tooltip="Reference image before the edit."),
                io.Image.Input("image_edited", tooltip="Edited image whose colors drift from the original."),
                io.Combo.Input("param_preset", options=["auto", "manual"], default="auto",
                               tooltip="auto = four-stage automatic mask tuning; manual = use the widget values."),
                io.Float.Input("threshold", default=0.05, min=0.0, max=1.0, step=0.01,
                               tooltip="Base pixel difference threshold (raised adaptively by MAD)."),
                io.Int.Input("open_k", default=11, min=1, max=81, step=2,
                             tooltip="Opening kernel for the auto-detected mask (removes specks)."),
                io.Int.Input("close_k", default=81, min=1, max=129, step=2,
                             tooltip="Closing kernel for the auto-detected mask (fills holes / merges blobs)."),
                io.Int.Input("expand", default=64, min=8, max=64, step=1,
                             tooltip="Per-component expansion (manual preset scales blobs by area)."),
                io.Int.Input("feather", default=32, min=4, max=32, step=1,
                             tooltip="Feather radius around the change mask (manual preset)."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="How strongly the affine color correction is applied."),
                io.Combo.Input("outside_mode", options=["edited", "original", "corrected"], default="edited",
                               tooltip="edited = wavelet low-frequency merge; original = strict paste; corrected = only the contrast-restored edit."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional change mask. When connected, auto-detection is skipped."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Float.Output(display_name="changed_ratio"),
                io.String.Output(display_name="tuned_params"),
            ],
        )

    @classmethod
    def execute(cls, image_original, image_edited, param_preset,
                threshold, open_k, close_k, expand, feather,
                strength, outside_mode, mask=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        I0_raw = _to_bhwc(image_original).float().to(device)
        I1_raw = _to_bhwc(image_edited).float().to(device)

        I0_rgb, I0_alpha = _split_rgb_alpha(I0_raw)
        I1_rgb, I1_alpha = _split_rgb_alpha(I1_raw)

        B = max(I0_rgb.shape[0], I1_rgb.shape[0])
        if I0_rgb.shape[0] == 1 and B > 1:
            I0_rgb = I0_rgb.expand(B, -1, -1, -1)
            if I0_alpha is not None:
                I0_alpha = I0_alpha.expand(B, -1, -1, -1)
        if I1_rgb.shape[0] == 1 and B > 1:
            I1_rgb = I1_rgb.expand(B, -1, -1, -1)
            if I1_alpha is not None:
                I1_alpha = I1_alpha.expand(B, -1, -1, -1)

        H, W = I1_rgb.shape[1], I1_rgb.shape[2]
        if I0_rgb.shape[1] != H or I0_rgb.shape[2] != W:
            I0_rgb = F.interpolate(
                I0_rgb.permute(0, 3, 1, 2), size=(H, W),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
            if I0_alpha is not None:
                I0_alpha = F.interpolate(
                    I0_alpha.permute(0, 3, 1, 2), size=(H, W),
                    mode="nearest",
                ).permute(0, 2, 3, 1)

        I0 = I0_rgb
        I1 = I1_rgb

        Hw, Ww, work_scale = _work_size(H, W)
        I0_w = _down_to(I0, Hw, Ww)
        I1_w = _down_to(I1, Hw, Ww)

        used_params = None
        M_w = None

        if mask is None:
            if param_preset == "auto":
                M_w, tuned = _auto_tune_mask(I0_w, I1_w, threshold, open_k, close_k, Hw, Ww)
                used_params = dict(tuned)
                used_params["preset"] = "auto"
                used_params["work_scale"] = round(float(work_scale), 4)
                used_params["expand_full"] = int(round(tuned["expand_max"] * work_scale))
                used_params["feather_full"] = int(round(tuned["feather_max"] * work_scale))
            else:
                M_base = _auto_detect_mask(I0_w, I1_w, threshold, open_k, close_k, Hw, Ww)
                M_exp, info = _expand_feather_by_components(
                    M_base, Hw, Ww, mode="ui",
                    ui_expand=int(expand), ui_feather=int(feather),
                )
                M_w = M_exp
                used_params = {
                    "preset": str(param_preset),
                    "threshold": round(float(threshold), 4),
                    "open_k": int(open_k),
                    "close_k": int(close_k),
                    "expand": int(info["expand_max"]),
                    "feather": int(info["feather_max"]),
                    "expand_ui": int(expand),
                    "feather_ui": int(feather),
                    "n_components": int(info["n_components"]),
                    "strength": round(float(strength), 4),
                    "outside_mode": str(outside_mode),
                    "work_scale": round(float(work_scale), 4),
                    "expand_full": int(round(info["expand_max"] * work_scale)),
                    "feather_full": int(round(info["feather_max"] * work_scale)),
                }
            if M_w.shape[0] == 1 and B > 1:
                M_w = M_w.expand(B, -1, -1, -1)
            M = _up_to(M_w, H, W).clamp(0, 1)
        else:
            M = _mask_to_bhw1(mask.to(device), B, H, W)
            used_params = {
                "preset": str(param_preset),
                "mask_input": True,
                "work_scale": round(float(work_scale), 4),
            }

        M_stat = (M > 0.5).float()
        A = 1.0 - M_stat
        anchor_ratio = A.mean().item()
        changed_ratio = M.mean().item()

        if anchor_ratio < _MIN_ANCHOR_PCT:
            out_rgb = I1.clamp(0, 1)
            out_alpha = I1_alpha if I1_alpha is not None else I0_alpha
            out_final = _merge_rgb_alpha(out_rgb, out_alpha).cpu()
            M_final = M.cpu()
            params_str = json.dumps(used_params, ensure_ascii=False) if used_params else ""
            return io.NodeOutput(out_final, M_final.squeeze(-1), float(changed_ratio), params_str)

        scale, shift = _compute_affine_params(I0, I1, A)
        I1_corr = _apply_affine(I1, scale, shift)
        I1_corr = (I1 * (1.0 - strength) + I1_corr * strength).clamp(0, 1)

        if param_preset == "auto":
            f_use = int(used_params.get("feather_max", _FEATHER_MIN))
        else:
            f_use = int(feather)
        f_use = max(0, f_use)

        if outside_mode == "edited":
            I1_ranged = _restore_contrast_by_anchor(I0, I1_corr, A)
            out_rgb = _wavelet_merge(I0, I1_ranged, M, f_use)
        elif outside_mode == "original":
            M_feather = _feather_mask_fast(M, f_use) if f_use > 0 else M
            I1_ranged = _restore_contrast_by_anchor(I0, I1_corr, A)
            out_rgb = (M_feather * I1_ranged + (1.0 - M_feather) * I0).clamp(0, 1)
        else:
            I1_ranged = _restore_contrast_by_anchor(I0, I1_corr, A)
            out_rgb = I1_ranged

        out_alpha = I1_alpha if I1_alpha is not None else I0_alpha
        out_final = _merge_rgb_alpha(out_rgb, out_alpha).cpu()
        M_final = M.cpu()
        params_str = json.dumps(used_params, ensure_ascii=False) if used_params else ""
        return io.NodeOutput(out_final, M_final.squeeze(-1), float(changed_ratio), params_str)

