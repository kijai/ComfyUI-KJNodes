import torch
import torch.nn.functional as F

from comfy import model_management
from comfy_api.latest import io

class ImageSharpenKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageSharpenKJ",
            category="KJNodes/image",
            display_name="Image Sharpen KJ",
            search_aliases=["sharpen", "unsharp mask", "deconvolution", "cas", "rcas", "high-pass", "postprocessing"],
            description="""GPU-accelerated image sharpening with multiple methods.

**RCAS** — AMD's Robust Contrast-Adaptive Sharpening (from FSR).
Single 5-tap cross filter that adapts to local contrast.
Minimal artifacts, good for general use with little tuning.

**Adaptive USM** — Unsharp mask with local variance modulation.
Sharpens detail-rich areas more, flat/noisy areas less.
More controllable than RCAS via radius and threshold parameters.

**High-Pass** — Extracts high-frequency detail and blends it back.
Gives a "clarity" enhancement feel. Uses radius to control detail scale.

**Deconvolution** — Richardson-Lucy iterative deconvolution.
Can recover actual lost detail from blur, not just enhance edges.
Uses radius as the estimated blur kernel and iterations to control convergence.""",
            inputs=[
                io.MatchType.Input("image", io.MatchType.Template("img_or_mask", [io.Image, io.Mask])),
                io.DynamicCombo.Input("method", options=[
                    io.DynamicCombo.Option(key="rcas", inputs=[
                        io.Float.Input("strength", default=0.8, min=0.0, max=1.0, step=0.01,
                                       tooltip="0 = no sharpening, 1 = full RCAS sharpening."),
                    ]),
                    io.DynamicCombo.Option(key="adaptive_usm", inputs=[
                        io.Float.Input("strength", default=0.5, min=0.0, max=3.0, step=0.01,
                                       tooltip="Sharpening multiplier. Values above 1.0 give aggressive sharpening."),
                        io.Float.Input("radius", default=1.0, min=0.5, max=5.0, step=0.1,
                                       tooltip="Gaussian blur sigma for the unsharp mask. Larger = enhances coarser detail."),
                        io.Float.Input("threshold", default=0.05, min=0.0, max=1.0, step=0.01,
                                       tooltip="Noise gate. Higher = only sharpen areas with more texture/detail. 0 = sharpen everything."),
                    ]),
                    io.DynamicCombo.Option(key="high_pass", inputs=[
                        io.Float.Input("strength", default=0.5, min=0.0, max=3.0, step=0.01,
                                       tooltip="Blend factor for high-frequency detail. Values above 1.0 give a punchier effect."),
                        io.Float.Input("radius", default=1.0, min=0.5, max=5.0, step=0.1,
                                       tooltip="Gaussian blur sigma defining the frequency cutoff. Larger = enhances coarser detail."),
                    ]),
                    io.DynamicCombo.Option(key="deconvolution", inputs=[
                        io.Float.Input("strength", default=0.5, min=0.0, max=1.0, step=0.01,
                                       tooltip="Blend between original (0) and fully deconvolved (1)."),
                        io.Float.Input("radius", default=1.0, min=0.5, max=5.0, step=0.1,
                                       tooltip="Sigma of the assumed Gaussian blur to reverse."),
                        io.Int.Input("iterations", default=10, min=1, max=100, step=1,
                                     tooltip="Richardson-Lucy iterations. More = sharper but slower, diminishing returns past ~20."),
                    ]),
                ]),
            ],
            outputs=[
                io.MatchType.Output(io.MatchType.Template("img_or_mask", [io.Image, io.Mask]), display_name="output"),
            ],
        )

    @classmethod
    def execute(cls, image, method) -> io.NodeOutput:
        selected = method["method"]
        strength = method.get("strength", 0.5)
        if strength == 0:
            return io.NodeOutput(image)

        is_mask = image.ndim == 3  # BHW
        if is_mask:
            image = image.unsqueeze(-1)  # BHW -> BHWC with C=1

        radius = method.get("radius", 1.0)
        threshold = method.get("threshold", 0.05)
        iterations = method.get("iterations", 10)
        if selected == "rcas":
            result = _rcas(image, strength)
        elif selected == "adaptive_usm":
            result = _adaptive_usm(image, strength, radius, threshold)
        elif selected == "high_pass":
            result = _high_pass(image, strength, radius)
        else:
            result = _deconvolution(image, strength, radius, iterations)

        if is_mask:
            result = result.squeeze(-1)  # BHWC -> BHW

        return io.NodeOutput(result)


def _rcas(image: torch.Tensor, strength: float) -> torch.Tensor:
    """AMD FidelityFX RCAS — 5-tap cross filter with contrast-adaptive lobe."""
    device = model_management.get_torch_device()
    intermediate_device = model_management.intermediate_device()
    dtype = model_management.intermediate_dtype()

    B, H, W, C = image.shape
    out = torch.empty(B, H, W, C, device=intermediate_device, dtype=dtype)
    attenuation = strength

    for i in range(B):
        img = image[i:i+1].to(device=device, dtype=dtype).permute(0, 3, 1, 2)
        padded = F.pad(img, (1, 1, 1, 1), mode='reflect')

        n = padded[:, :, 0:H, 1:W+1]
        s = padded[:, :, 2:H+2, 1:W+1]
        w = padded[:, :, 1:H+1, 0:W]
        e = padded[:, :, 1:H+1, 2:W+2]
        center = img

        mn = torch.min(torch.min(torch.min(torch.min(n, s), w), e), center)
        mx = torch.max(torch.max(torch.max(torch.max(n, s), w), e), center)

        # hitMin = -mn / (4*mx),  hitMax = -(1-mx) / (4*(1-mn))
        hit_min = mn.div(mx * 4.0 + 1e-6).neg_()
        hit_max = mx.neg().add_(1.0).div_(mn.neg().add_(1.0).mul_(4.0).add_(1e-6)).neg_()

        torch.max(hit_min, hit_max, out=hit_min)
        lobe = hit_min.min(dim=1, keepdim=True).values

        lobe.mul_(attenuation)
        lobe.clamp_(-0.1875, 0.0)

        # (center + lobe*(n+s+e+w)) / (1 + 4*lobe)
        norm = lobe.mul(4.0).add_(1.0).reciprocal_()
        neighbors = n + s
        neighbors.add_(e).add_(w)
        neighbors.mul_(lobe).add_(center).mul_(norm)
        neighbors.clamp_(0.0, 1.0)

        out[i:i+1] = neighbors.permute(0, 2, 3, 1).to(device=intermediate_device)

    return out


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(int(3.0 * sigma + 0.5), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel.div_(kernel.sum())
    return kernel


def _gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on BCHW tensor."""
    kernel_1d = _gaussian_kernel_1d(sigma, img.device, img.dtype)
    k = kernel_1d.shape[0]
    C = img.shape[1]
    pad = k // 2

    kh = kernel_1d.view(1, 1, 1, k).expand(C, -1, -1, -1)
    blurred = F.conv2d(F.pad(img, (pad, pad, 0, 0), mode='reflect'), kh, groups=C)

    kv = kernel_1d.view(1, 1, k, 1).expand(C, -1, -1, -1)
    blurred = F.conv2d(F.pad(blurred, (0, 0, pad, pad), mode='reflect'), kv, groups=C)

    return blurred


def _adaptive_usm(image: torch.Tensor, strength: float, radius: float, threshold: float) -> torch.Tensor:
    """Unsharp mask modulated by local variance — sharpens texture, skips flat areas."""
    device = model_management.get_torch_device()
    intermediate_device = model_management.intermediate_device()
    dtype = model_management.intermediate_dtype()

    B, H, W, C = image.shape
    out = torch.empty(B, H, W, C, device=intermediate_device, dtype=dtype)
    var_sigma = radius * 1.5

    for i in range(B):
        img = image[i:i+1].to(device=device, dtype=dtype, copy=True).permute(0, 3, 1, 2)

        blurred = _gaussian_blur(img, radius)
        detail = img - blurred

        # Local variance: E[x^2] - E[x]^2
        local_mean = _gaussian_blur(img, var_sigma)
        local_mean_sq = _gaussian_blur(img.square(), var_sigma)
        local_mean_sq.sub_(local_mean * local_mean).clamp_(min=0.0)

        # Convert to std deviation so threshold is in pixel-intensity scale
        modulation = local_mean_sq.mean(dim=1, keepdim=True).sqrt_()

        if threshold > 0:
            modulation.div_(threshold).clamp_(0.0, 1.0)
        else:
            modulation.fill_(1.0)

        img.addcmul_(detail, modulation, value=strength).clamp_(0.0, 1.0)

        out[i:i+1] = img.permute(0, 2, 3, 1).to(device=intermediate_device)

    return out


def _high_pass(image: torch.Tensor, strength: float, radius: float) -> torch.Tensor:
    """High-pass sharpening: original + strength * (original - blur)."""
    device = model_management.get_torch_device()
    intermediate_device = model_management.intermediate_device()
    dtype = model_management.intermediate_dtype()

    B, H, W, C = image.shape
    out = torch.empty(B, H, W, C, device=intermediate_device, dtype=dtype)

    for i in range(B):
        img = image[i:i+1].to(device=device, dtype=dtype).permute(0, 3, 1, 2)

        low_pass = _gaussian_blur(img, radius)
        low_pass.neg_().add_(img).mul_(strength).add_(img).clamp_(0.0, 1.0)

        out[i:i+1] = low_pass.permute(0, 2, 3, 1).to(device=intermediate_device)

    return out


def _deconvolution(image: torch.Tensor, strength: float, radius: float, iterations: int) -> torch.Tensor:
    """Richardson-Lucy deconvolution with Gaussian PSF."""
    device = model_management.get_torch_device()
    intermediate_device = model_management.intermediate_device()
    dtype = model_management.intermediate_dtype()

    B, H, W, C = image.shape
    out = torch.empty(B, H, W, C, device=intermediate_device, dtype=dtype)

    # Gaussian PSF kernel (symmetric, so PSF^T = PSF)
    kernel_1d = _gaussian_kernel_1d(radius, device, dtype)
    k = kernel_1d.shape[0]
    pad = k // 2
    kh = kernel_1d.view(1, 1, 1, k).expand(C, -1, -1, -1)
    kv = kernel_1d.view(1, 1, k, 1).expand(C, -1, -1, -1)

    for i in range(B):
        img = image[i:i+1].to(device=device, dtype=dtype).permute(0, 3, 1, 2)
        estimate = img.clone()

        for _ in range(iterations):
            # estimate *= (observed / (estimate * PSF)) * PSF^T
            blurred = F.conv2d(F.pad(estimate, (pad, pad, 0, 0), mode='reflect'), kh, groups=C)
            blurred = F.conv2d(F.pad(blurred, (0, 0, pad, pad), mode='reflect'), kv, groups=C)

            blurred.add_(1e-6)
            torch.div(img, blurred, out=blurred)

            correction = F.conv2d(F.pad(blurred, (pad, pad, 0, 0), mode='reflect'), kh, groups=C)
            correction = F.conv2d(F.pad(correction, (0, 0, pad, pad), mode='reflect'), kv, groups=C)

            estimate.mul_(correction)

        torch.lerp(img, estimate, strength, out=estimate)
        estimate.clamp_(0.0, 1.0)

        out[i:i+1] = estimate.permute(0, 2, 3, 1).to(device=intermediate_device)

    return out
