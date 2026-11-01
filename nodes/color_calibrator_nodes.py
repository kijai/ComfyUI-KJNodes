import math

import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndi

from comfy_api.latest import io

# ============================================================
# 内部常量（与 BlueBox 参考实现 color_calibrator_bluebox.py 一致）
# ============================================================
_MIN_ANCHOR_PCT = 0.02
_WAVELET_LEVELS = 3
_WAVELET_SIGMA = 2.0
_SCALE_CLAMP = (0.5, 2.0)
_RANGE_CLAMP = (0.3, 3.0)       # 对比度回压倍数 a 的放宽区间
_RANGE_CLIP_PCT = 0.005         # 测黑/白场的剪切百分比（抗噪）
_RANGE_DOWNSCALE = 4            # 测范围在 1/4 分辨率上做
_FEATHER_DOWNSCALE = 4
_WAVELET_DOWNSCALE = 4          # 低频在 1/4 分辨率上算
_EDT_MAX_K = 32                 # 超过此半径才考虑半分辨率 EDT


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


# ============================================================
# 高斯模糊
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
    """强制核大小为奇数，避免 max_pool2d 尺寸漂移。"""
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
    """GPU 方形核膨胀，O(HW)，常数极小。k≤32 时与圆核肉眼无差。"""
    if k <= 0:
        return mask_bhw1
    k = _odd(k)
    x = mask_bhw1.permute(0, 3, 1, 2)
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x.permute(0, 2, 3, 1)


def _dilate_edt_half(mask_bhw1: torch.Tensor, k: int, downscale: int = 2) -> torch.Tensor:
    """半分辨率 EDT 再上采样，误差 <1px，速度 ×4。"""
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


def _dilate(mask_bhw1: torch.Tensor, k: int) -> torch.Tensor:
    """统一入口：小半径用 GPU max_pool，大半径用半分辨率 EDT。"""
    if k <= 0:
        return mask_bhw1
    if k <= _EDT_MAX_K:
        return _dilate_pool(mask_bhw1, k)
    return _dilate_edt_half(mask_bhw1, k, downscale=2)


# ============================================================
# 羽化：降采样 + 小核高斯
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
# 高斯金字塔
# ============================================================

def _gaussian_pyramid(rgb: torch.Tensor, levels: int, sigma: float):
    pyr = [rgb]
    for _ in range(levels):
        pyr.append(_gaussian_blur_bhwc(pyr[-1], sigma))
    low = pyr[-1]
    highs = [pyr[i] - pyr[i + 1] for i in range(len(pyr) - 1)]
    return low, highs


def _pyramid_reconstruct(low, highs):
    out = low
    for h in highs:
        out = out + h
    return out


# ============================================================
# 仿射校正（C: 加权矩统计，避免布尔索引物化）
# ============================================================

def _compute_affine_params(I0, I1, A):
    """
    A: (B,H,W,1) 前景权重（遮罩外=1，遮罩内=0）
    用加权均值/方差，等价于"只在前景上统计"，但完全不物化布尔索引。
    """
    device = I0.device
    scale = torch.ones(3, device=device)
    shift = torch.zeros(3, device=device)

    w = A.reshape(-1)                            # (N,)
    w_sum = w.sum()
    if w_sum.item() < 32:                        # 只同步一次
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

        if sd1.item() < 1e-4:                    # 只同步一次/通道
            scale[c] = 1.0
            shift[c] = mu0 - mu1
        else:
            sc = (sd0 / sd1).clamp(*_SCALE_CLAMP)
            scale[c] = sc
            shift[c] = mu0 - sc * mu1

    return scale, shift


def _apply_affine(rgb, scale, shift):
    return (rgb * scale.view(1, 1, 1, 3) + shift.view(1, 1, 1, 3)).clamp(0, 1)


# ============================================================
# 自动遮罩检测
# ============================================================

def _auto_detect_mask(I0, I1, threshold, expand, open_k, close_k):
    diff = (I0 - I1).abs().max(dim=-1, keepdim=True).values
    mask = (diff > threshold).float()

    if open_k > 1:
        mask = _morph_open(mask, open_k)
    if close_k > 1:
        mask = _morph_close(mask, close_k)
    if expand > 0:
        mask = _dilate(mask, expand)

    return mask.clamp(0, 1)


# ============================================================
# 主节点
# ============================================================

class ColorCalibratorKJ(io.ComfyNode):
    """
    Color Calibrator KJ

    忠实移植自 zhoutingxi/ComfyUI-BlueBox (MIT) 的 ColorCalibratorBlueBox 节点。

    一个节点内完成：
      1. 若 mask 未接 → 自动对比两图检测变化区作为遮罩
      2. 用遮罩外（严格排除遮罩内）逐通道仿射估计色差
      3. 用遮罩外前后对比估计对比度拉伸倍数，围绕编辑图自身黑场
         把整张编辑图的跨度回压（含遮罩内），找回原图对比度
      4. outside_mode == "edited"：遮罩外高斯金字塔融合（低频原图 + 高频编辑图）
         outside_mode == "original"：遮罩外严格回贴原图

    输出 IMAGE / MASK / FLOAT（changed_ratio = 改动区占全图比例）。
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
                "Compares `image_original` with `image_edited`: unless a `mask` is given it auto-detects the "
                "changed region, then uses the unchanged region as an anchor to (1) estimate a per-channel "
                "affine color correction (mean/var matching) and pull the whole edit back to the original "
                "colors, (2) recompress contrast around the edit's own black point, and (3) blend/paste the "
                "original outside the edited region (`edited` = wavelet low-frequency merge, `original` = "
                "strict paste). `strength` controls the overall correction intensity.\n"
                "Outputs the corrected image, the change mask and the fraction of changed pixels."
            ),
            inputs=[
                io.Image.Input("image_original",
                               tooltip="Reference image before the edit."),
                io.Image.Input("image_edited",
                               tooltip="Edited image whose colors drift from the original."),
                io.Float.Input("threshold", default=0.08, min=0.0, max=1.0, step=0.01,
                               tooltip="Per-pixel max-channel difference above which a pixel is flagged as changed. Only used when no mask is connected."),
                io.Int.Input("expand", default=48, min=0, max=128, step=1,
                             tooltip="Dilate the auto-detected change mask by this many pixels."),
                io.Int.Input("open_k", default=13, min=1, max=81, step=2,
                             tooltip="Opening kernel size for the auto-detected mask (removes specks)."),
                io.Int.Input("close_k", default=15, min=1, max=81, step=2,
                             tooltip="Closing kernel size for the auto-detected mask (fills holes)."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="How strongly the affine color correction is applied (0 = keep the edit's colors)."),
                io.Combo.Input("outside_mode", options=["edited", "original"], default="edited",
                               tooltip="How the unchanged region is handled: 'edited' keeps the edit's high frequencies and replaces only low frequency with the original; 'original' strictly pastes the original."),
                io.Int.Input("feather", default=24, min=0, max=128, step=1,
                             tooltip="Feather (soft edge) radius around the change mask."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional change mask. When connected, auto-detection is skipped."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Float.Output(display_name="changed_ratio"),
            ],
        )

    @classmethod
    def execute(cls, image_original, image_edited,
                threshold, expand, open_k, close_k,
                strength, outside_mode, feather, mask=None) -> io.NodeOutput:
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

        if mask is None:
            M = _auto_detect_mask(I0, I1, threshold, expand, open_k, close_k)
            if M.shape[0] == 1 and B > 1:
                M = M.expand(B, -1, -1, -1)
        else:
            M = _mask_to_bhw1(mask.to(device), B, H, W)

        # 用 A.mean() 直接得 anchor_ratio，省一次全图 bool + reshape
        M_stat = (M > 0.5).float()
        A = 1.0 - M_stat
        anchor_ratio = A.mean().item()            # 只同步一次
        changed_ratio = M.mean().item()

        if anchor_ratio < _MIN_ANCHOR_PCT:
            out_rgb = I1.clamp(0, 1)
            out_alpha = I1_alpha if I1_alpha is not None else I0_alpha
        else:
            scale, shift = _compute_affine_params(I0, I1, A)
            I1_corr = _apply_affine(I1, scale, shift)
            I1_corr = (I1 * (1.0 - strength) + I1_corr * strength).clamp(0, 1)

            if outside_mode == "edited":
                I1_ranged = _restore_contrast_by_anchor(I0, I1_corr, A)
                out_rgb = _wavelet_merge(I0, I1_ranged, M, feather)
            else:
                M_feather = _feather_mask_fast(M, feather) if feather > 0 else M
                I1_ranged = _restore_contrast_by_anchor(I0, I1_corr, A)
                out_rgb = (M_feather * I1_ranged + (1.0 - M_feather) * I0).clamp(0, 1)

        out_alpha = I1_alpha if I1_alpha is not None else I0_alpha
        out = _merge_rgb_alpha(out_rgb, out_alpha).cpu()
        M_cpu = M.cpu()

        return io.NodeOutput(out, M_cpu.squeeze(-1), float(changed_ratio))


# ============================================================
# 遮罩外前后对比测范围拉伸倍数 a，围绕编辑图自身黑场反转，作用于全图
# ============================================================

def _restore_contrast_by_anchor(I0, I1_corr, A):
    """
    用遮罩外（可靠区）对比 I0 与 I1_corr 的主体黑/白场，
    估计编辑图相对原图的对比度拉伸倍数 a，
    然后围绕 I1_corr 自身黑场把整张图的跨度缩回 /a。
    不分通道、无外部参数。围绕编辑图自身黑场而非原图黑场，
    避免遮罩内外原始范围不同时被错误钉到原图绝对范围。
    """
    B, H, W, _ = I0.shape
    device = I0.device

    ds = _RANGE_DOWNSCALE
    sh, sw = max(H // ds, 1), max(W // ds, 1)

    def _down(x):
        return F.interpolate(
            x.permute(0, 3, 1, 2), size=(sh, sw),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)

    I0_s = _down(I0)
    I1_s = _down(I1_corr)
    A_s = _down(A)                                  # (B,sh,sw,1)

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
        # 统一（不分通道）：用三通道均值灰度测范围，保证色调关系不变
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

        # 围绕 I1_corr 自身黑场 L1 反转拉伸倍数，作用于全图（含遮罩内）
        out[bi] = ((I1_corr[bi] - L1) / a + L1).clamp(0, 1)

    return out


# ============================================================
# B: 用恒等式 out = I1_corr + (1-M)*(low0 - low1) 在 1/4 分辨率算低频
# ============================================================

def _wavelet_merge(I0, I1_corr, M, feather):
    """
    原算法恒等式：
        out = M*low1 + (1-M)*low0 + Σhighs1
            = I1_corr + (1-M)*(low0 - low1)      # 因为 Σhighs1 = I1_corr - low1
    因此只需在低分辨率上算 low0、low1 两个低频，再做 (low0-low1) 的融合。
    (low0-low1) 是低频差，降采样计算误差极小。
    """
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

    # 低频：1/4 分辨率上算 3 层高斯
    low0_s, _ = _gaussian_pyramid(_down(I0), _WAVELET_LEVELS, _WAVELET_SIGMA)
    low1_s, _ = _gaussian_pyramid(_down(I1_corr), _WAVELET_LEVELS, _WAVELET_SIGMA)
    M_soft_s = _down(M_soft)

    # (1-M) * (low0 - low1)，在 1/4 分辨率上融合，再上采样
    delta_s = (1.0 - M_soft_s) * (low0_s - low1_s)
    delta = _up(delta_s)

    # out = I1_corr + (1-M)*(low0 - low1)
    out = I1_corr + delta
    return out.clamp(0, 1)
