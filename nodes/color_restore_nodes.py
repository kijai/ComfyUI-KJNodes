"""
Color Restore KJ — a standalone color restoration / color grading node.

This node extracts the color-restoration (调色/色彩还原) pipeline from
`zhoutingxi/ComfyUI-BlueBox` (MIT) and exposes it as a backend, GPU-accelerated
ComfyUI node. It reproduces the exact same image processing chain that the
BlueBox editor bakes into its output image:

    1. texture      - mid-frequency detail recovery (small blur - large blur,
                      weighted by an edge-preservation mask)
    2. auto color   - robust auto-levels (0.5%/99.5% percentile black/white
                      points per channel) + gray-world white-balance offset
    3. input levels - black point / white point / gamma
    4. output levels- output black / output white remap
    5. brightness   - additive offset and contrast around mid-gray
    6. color        - per-channel RGB bias + luma-weighted saturation

With all parameters at their defaults the node is an identity pass-through.

Source of the algorithm (MIT licensed):
    https://github.com/zhoutingxi/ComfyUI-BlueBox  (web/mask-editor.html,
    `getFinalImageData` + `computeAutoColorStats` + `computeAutoUniforms`)
"""

import numpy as np
import torch
import torch.nn.functional as F

from comfy import model_management
from comfy_api.latest import io

# ---------------------------------------------------------------------------
# Constants mirrored from the BlueBox reference implementation
# ---------------------------------------------------------------------------
_AUTO_STAT_MAX = 1024          # downscale cap for the auto-color statistics
_AUTO_SAMPLE_STEP = 8          # sample every N-th pixel for histogramming
_TEXTURE_GAIN = 4.0            # texture slider (0-100) is scaled by this
_EDGE_MASK_LOW = 0.05          # gradient below this => full texture (flat area)
_EDGE_MASK_HIGH = 0.2          # gradient above this => no texture (edge)
_BLUR_SMALL_RADIUS = 0.8       # Gaussian radius for the small blur
_BLUR_LARGE_RADIUS = 2.5       # Gaussian radius for the large blur
_LUMA = (0.2126, 0.7152, 0.0722)   # for auto-color mid-luminance selection
_LUMA_SAT = (0.299, 0.587, 0.114)  # for saturation


def _downscale_for_stats(img_np):
    """Scale an HxWx3 float array so its max side fits within AUTO_STAT_MAX."""
    h, w, _ = img_np.shape
    if w > _AUTO_STAT_MAX or h > _AUTO_STAT_MAX:
        r = min(_AUTO_STAT_MAX / w, _AUTO_STAT_MAX / h)
        nw = max(1, int(round(w * r)))
        nh = max(1, int(round(h * r)))
        # nearest-neighbour sampling is sufficient for histogram statistics
        ys = (np.linspace(0, h - 1, nh)).astype(np.int64)
        xs = (np.linspace(0, w - 1, nw)).astype(np.int64)
        img_np = img_np[np.ix_(ys, xs)]
    return img_np


def _auto_color_stats(img_np):
    """Faithful port of BlueBox computeAutoColorStats.

    img_np: HxWx3 float array in [0, 1]. Returns dict with per-channel
    black/white points (0-255 units), ranges, and gray-world offsets (0-1).
    """
    small = _downscale_for_stats(img_np)
    data = (small * 255.0).astype(np.float32)
    h, w, _ = data.shape
    pixel_count = h * w

    # sample every AUTO_SAMPLE_STEP-th pixel (flattened row-major, like JS)
    flat = data.reshape(-1, 3)[:: _AUTO_SAMPLE_STEP]
    sample_count = int(np.ceil(pixel_count / _AUTO_SAMPLE_STEP))

    r = flat[:, 0]
    g = flat[:, 1]
    b = flat[:, 2]

    hist_r = np.bincount(r.astype(np.int64), minlength=256)[:256]
    hist_g = np.bincount(g.astype(np.int64), minlength=256)[:256]
    hist_b = np.bincount(b.astype(np.int64), minlength=256)[:256]

    def pct(hist, p):
        target = sample_count * p
        cum = np.cumsum(hist)
        idx = int(np.searchsorted(cum, target))
        return min(idx, 255)

    r_lo, r_hi = pct(hist_r, 0.005), pct(hist_r, 0.995)
    g_lo, g_hi = pct(hist_g, 0.005), pct(hist_g, 0.995)
    b_lo, b_hi = pct(hist_b, 0.005), pct(hist_b, 0.995)

    r_range = max(r_hi - r_lo, 1)
    g_range = max(g_hi - g_lo, 1)
    b_range = max(b_hi - b_lo, 1)

    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mid = (lum > 25) & (lum < 230)
    count_mid = int(mid.sum())
    offset = np.zeros(3, dtype=np.float32)
    if count_mid > 10:
        sum_r = float(r[mid].sum())
        sum_g = float(g[mid].sum())
        sum_b = float(b[mid].sum())
        avg_r = sum_r / count_mid
        avg_g = sum_g / count_mid
        avg_b = sum_b / count_mid
        str_r = (avg_r - r_lo) / r_range
        str_g = (avg_g - g_lo) / g_range
        str_b = (avg_b - b_lo) / b_range
        avg_lum = (str_r + str_g + str_b) / 3.0
        offset[0] = avg_lum - str_r
        offset[1] = avg_lum - str_g
        offset[2] = avg_lum - str_b

    return {
        "lo": np.array([r_lo, g_lo, b_lo], dtype=np.float32),
        "rng": np.array([r_range, g_range, b_range], dtype=np.float32),
        "offset": offset,
    }


def _auto_uniforms(stats, contrast, color):
    """Faithful port of BlueBox computeAutoUniforms.

    Returns (a, b, offset) 3-vectors operating on 0-1 channels.
    contrast / color are in [0, 1] (auto_contrast/100, auto_color/100).
    """
    if contrast <= 0 and color <= 0:
        return np.ones(3, np.float32), np.zeros(3, np.float32), np.zeros(3, np.float32)
    lo01 = stats["lo"] / 255.0
    rng01 = stats["rng"] / 255.0
    a = 1.0 - contrast + contrast / rng01
    b = -contrast * lo01 / rng01
    offset = stats["offset"] * color
    return a.astype(np.float32), b.astype(np.float32), offset.astype(np.float32)


def _gaussian_kernel_1d(sigma, device, dtype):
    """9-tap Gaussian kernel, same as the BlueBox blur shader."""
    i = torch.arange(-4, 5, device=device, dtype=dtype)
    w = torch.exp(-(i * i) / (2.0 * sigma * sigma))
    return w / w.sum()


def _gaussian_blur(img, radius, device, dtype):
    """Separable 9-tap Gaussian blur on a BCHW tensor (reflect padding)."""
    sigma = max(radius / 2.0, 0.5)
    k = _gaussian_kernel_1d(sigma, device, dtype)
    c = img.shape[1]
    pad = k.shape[0] // 2
    kh = k.view(1, 1, 1, -1).expand(c, -1, -1, -1)
    out = F.conv2d(F.pad(img, (pad, pad, 0, 0), mode="reflect"), kh, groups=c)
    kv = k.view(1, 1, -1, 1).expand(c, -1, -1, -1)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), kv, groups=c)
    return out


def _edge_weight(grad):
    """1 - smoothstep between EDGE_MASK_LOW and EDGE_MASK_HIGH."""
    t = (grad - _EDGE_MASK_LOW) / (_EDGE_MASK_HIGH - _EDGE_MASK_LOW)
    smooth = t * t * (3.0 - 2.0 * t)
    w = torch.where(
        grad <= _EDGE_MASK_LOW,
        torch.ones_like(grad),
        torch.where(grad >= _EDGE_MASK_HIGH, torch.zeros_like(grad), 1.0 - smooth),
    )
    return w


def _apply_texture(img, amount, device, dtype):
    """Mid-frequency detail recovery with an edge-preservation mask."""
    b, c, h, w = img.shape
    small = _gaussian_blur(img, _BLUR_SMALL_RADIUS, device, dtype)
    large = _gaussian_blur(img, _BLUR_LARGE_RADIUS, device, dtype)
    mid = small - large

    padded = F.pad(img, (1, 1, 1, 1), mode="reflect")
    l = padded[:, :, 1 : h + 1, 0:w]
    r = padded[:, :, 1 : h + 1, 2 : w + 2]
    u = padded[:, :, 0:h, 1 : w + 1]
    d = padded[:, :, 2 : h + 2, 1 : w + 1]

    gx = torch.sqrt(((r - l).pow(2)).mean(dim=1, keepdim=True))
    gy = torch.sqrt(((d - u).pow(2)).mean(dim=1, keepdim=True))
    grad = torch.sqrt(gx * gx + gy * gy)

    wmask = _edge_weight(grad)
    out = img + amount * mid * wmask
    return out.clamp(0.0, 1.0)


def _apply_auto_color(img, a, b, offset):
    out = img * torch.tensor(a, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    out = out + torch.tensor(b, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    out = out + torch.tensor(offset, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    return out.clamp(0.0, 1.0)


def _apply_levels(img, black, white, gamma, out_black, out_white):
    black = float(black) / 255.0
    white = float(white) / 255.0
    rng = max(white - black, 1.0 / 255.0)
    gamma = max(0.1, min(10.0, float(gamma)))
    out_black = float(out_black) / 255.0
    out_white = float(out_white) / 255.0
    out_rng = out_white - out_black
    v = (img - black) / rng
    v = v.clamp(0.0, 1.0)
    v = v.pow(1.0 / gamma)
    v = out_black + v * out_rng
    return v.clamp(0.0, 1.0)


def _apply_brightness_contrast(img, brightness, contrast):
    bright_add = (brightness / 100.0) * 0.5
    k = 1.0 + contrast / 100.0
    out = (img + bright_add - 0.5) * k + 0.5
    return out.clamp(0.0, 1.0)


def _apply_color_adjust(img, saturation, bias_rgb):
    bias = torch.tensor(bias_rgb, device=img.device, dtype=img.dtype).view(1, 3, 1, 1) * 0.25
    out = img + bias
    luma = torch.tensor(_LUMA_SAT, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    gray = (out * luma).sum(dim=1, keepdim=True)
    sat = 1.0 + saturation / 100.0
    out = gray + (out - gray) * sat
    return out.clamp(0.0, 1.0)


class ColorRestoreKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorRestoreKJ",
            category="KJNodes/image",
            display_name="Color Restore KJ",
            search_aliases=[
                "color restore", "color restoration", "color grading", "white balance",
                "auto levels", "levels", "color cast", "调色", "色彩还原", "白平衡",
            ],
            description=(
                "Restore / re-grade image colors using the same pipeline as the "
                "BlueBox editor (MIT, zhoutingxi/ComfyUI-BlueBox). Chain of: "
                "mid-frequency texture recovery, auto-levels + gray-world white "
                "balance, input/output levels, brightness/contrast and "
                "saturation + RGB bias. With all defaults it is a pass-through."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("auto_contrast", default=0.0, min=0.0, max=100.0, step=1.0,
                               tooltip="Auto-levels strength (per-channel percentile stretch). 0 = off."),
                io.Float.Input("auto_color", default=0.0, min=0.0, max=100.0, step=1.0,
                               tooltip="Gray-world color-balance offset (on per-channel stretched means). Reduces systematic channel bias. 0 = off."),
                io.Float.Input("levels_black", default=0.0, min=0.0, max=255.0, step=1.0,
                               tooltip="Input levels black point."),
                io.Float.Input("levels_white", default=255.0, min=0.0, max=255.0, step=1.0,
                               tooltip="Input levels white point."),
                io.Float.Input("levels_gamma", default=1.0, min=0.1, max=10.0, step=0.05,
                               tooltip="Input levels gamma."),
                io.Float.Input("levels_output_black", default=0.0, min=0.0, max=255.0, step=1.0,
                               tooltip="Output levels black point."),
                io.Float.Input("levels_output_white", default=255.0, min=0.0, max=255.0, step=1.0,
                               tooltip="Output levels white point."),
                io.Float.Input("brightness", default=0.0, min=-100.0, max=100.0, step=1.0),
                io.Float.Input("contrast", default=0.0, min=-100.0, max=100.0, step=1.0),
                io.Float.Input("saturation", default=0.0, min=-100.0, max=100.0, step=1.0),
                io.Float.Input("bias_red", default=0.0, min=-100.0, max=100.0, step=1.0,
                               tooltip="Additive red color cast."),
                io.Float.Input("bias_green", default=0.0, min=-100.0, max=100.0, step=1.0,
                               tooltip="Additive green color cast."),
                io.Float.Input("bias_blue", default=0.0, min=-100.0, max=100.0, step=1.0,
                               tooltip="Additive blue color cast."),
                io.Float.Input("texture", default=0.0, min=0.0, max=100.0, step=1.0,
                               tooltip="Mid-frequency detail recovery (0-100). 0 = off."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, image, auto_contrast=0.0, auto_color=0.0,
                levels_black=0.0, levels_white=255.0, levels_gamma=1.0,
                levels_output_black=0.0, levels_output_white=255.0,
                brightness=0.0, contrast=0.0, saturation=0.0,
                bias_red=0.0, bias_green=0.0, bias_blue=0.0,
                texture=0.0) -> io.NodeOutput:
        texture_amount = (texture / 100.0) * _TEXTURE_GAIN
        if (auto_contrast <= 0 and auto_color <= 0 and texture_amount <= 1e-6
                and levels_black == 0 and levels_white == 255 and abs(levels_gamma - 1.0) < 1e-6
                and levels_output_black == 0 and levels_output_white == 255
                and brightness == 0 and contrast == 0 and saturation == 0
                and bias_red == 0 and bias_green == 0 and bias_blue == 0):
            return io.NodeOutput(image)

        device = model_management.get_torch_device()
        intermediate_device = model_management.intermediate_device()
        dtype = model_management.intermediate_dtype()

        image = image.to(device=device, dtype=dtype)  # BHWC
        b, h, w, c = image.shape
        if c < 3:
            raise ValueError("ColorRestoreKJ requires an image with at least 3 channels")

        # The color pipeline operates on RGB only; an extra channel (alpha) is
        # passed through untouched.
        rgb = image[..., :3]
        alpha = image[..., 3:] if c > 3 else None

        # Auto color statistics are computed once on a downsampled CPU copy.
        a = None
        if auto_contrast > 0 or auto_color > 0:
            stats = _auto_color_stats(rgb[0].float().cpu().numpy())
            a = _auto_uniforms(stats, auto_contrast / 100.0, auto_color / 100.0)

        results = []
        for i in range(b):
            img = rgb[i : i + 1].permute(0, 3, 1, 2).contiguous()
            if texture_amount > 1e-6:
                img = _apply_texture(img, texture_amount, device, dtype)
            if a is not None:
                img = _apply_auto_color(img, a[0], a[1], a[2])
            if not (levels_black == 0 and levels_white == 255 and abs(levels_gamma - 1.0) < 1e-6
                    and levels_output_black == 0 and levels_output_white == 255):
                img = _apply_levels(img, levels_black, levels_white, levels_gamma,
                                    levels_output_black, levels_output_white)
            if brightness != 0 or contrast != 0:
                img = _apply_brightness_contrast(img, brightness, contrast)
            if saturation != 0 or bias_red != 0 or bias_green != 0 or bias_blue != 0:
                img = _apply_color_adjust(img, saturation,
                                          [bias_red, bias_green, bias_blue])
            out_img = img.permute(0, 2, 3, 1)
            if alpha is not None:
                out_img = torch.cat([out_img, alpha[i : i + 1]], dim=-1)
            results.append(out_img)

        out = torch.cat(results, dim=0).to(device=intermediate_device).to(torch.float32)
        return io.NodeOutput(out)
