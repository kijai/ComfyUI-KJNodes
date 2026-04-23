import os
import random

import torch
from PIL import Image
from comfy_api.latest import io
import comfy.model_management as mm
import folder_paths


# LogC3 constants (ARRI LogC3, EI 800). Kept in sync with the WebGL shader in web/js/hdr_preview.js.
LC_A = 5.555556
LC_B = 0.052272
LC_C = 0.247190
LC_D = 0.385537
LC_E = 5.367655
LC_F = 0.092809
LC_CUT = 0.010591
LC_CUT_LOG = LC_E * LC_CUT + LC_F  # ~0.14966

def _logc3_decompress(logc: torch.Tensor) -> torch.Tensor:
    logc = logc.clamp(0.0, 1.0)
    lin_from_log = (10.0 ** ((logc - LC_D) / LC_C) - LC_B) / LC_A
    lin_from_lin = (logc - LC_F) / LC_E
    return torch.where(logc >= LC_CUT_LOG, lin_from_log, lin_from_lin)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    cutoff = 0.0031308
    return torch.where(
        x <= cutoff,
        12.92 * x,
        1.055 * torch.pow(x.clamp(min=cutoff), 1.0 / 2.4) - 0.055,
    ).clamp_(0.0, 1.0)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    cutoff = 0.04045
    return torch.where(
        x <= cutoff,
        x / 12.92,
        ((x.clamp(min=0.0) + 0.055) / 1.055) ** 2.4,
    )


class HDRPreviewKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HDRPreviewKJ",
            display_name="HDR Preview KJ",
            category="KJNodes/image",
            is_output_node=True,
            is_experimental=True,
            description=(
                "Realtime-exposure preview for HDR-compressed images.\n\n"
                "Input: LogC3-compressed [0,1] image/video batch (e.g. the VAE-decoded output "
                "of an HDR IC-LoRA workflow, prior to HDR decompression).\n\n"
                "Decompression + exposure + saturation + Reinhard tonemap + sRGB runs in a WebGL "
                "fragment shader in the browser for realtime slider feedback, and the same math "
                "runs server-side to produce the baked sRGB IMAGE output. Slider changes update "
                "the preview immediately; the IMAGE output only updates when the workflow is re-queued."
            ),
            inputs=[
                io.Image.Input("image",
                    tooltip="LogC3-compressed HDR image/video in [0,1], or linear HDR if 'input_space' is 'linear'."),
                io.Float.Input("exposure", default=0.0, min=-10.0, max=10.0, step=0.01,
                    tooltip="Exposure in EV stops. 0 = no change, +1 = 2x brighter."),
                io.Float.Input("saturation", default=1.0, min=0.0, max=2.0, step=0.01,
                    tooltip="Saturation multiplier. 0 = grayscale, 1 = unchanged, 2 = 2x."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.1, optional=True,
                    tooltip="Playback frame rate for video (batch) inputs."),
                io.Combo.Input("input_space", options=["logc3", "linear", "srgb"], default="logc3", optional=True,
                    tooltip="Color space of input. 'logc3' = ARRI LogC3 compressed HDR; 'linear' = linear HDR directly; 'srgb' = already-graded sRGB image (skips Reinhard tonemap)."),
            ],
            outputs=[
                io.Image.Output(display_name="image",
                    tooltip="Tonemapped sRGB image, ready for preview/save."),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, exposure: float = 0.0, saturation: float = 1.0, fps: float = 24.0, input_space: str = "logc3") -> io.NodeOutput:
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        prefix = f"hdrprv_{random.randint(0, 0xFFFFFF):06x}"

        B, H, W, _ = image.shape
        device = mm.get_torch_device()
        exposure_mul = 2.0 ** exposure
        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device)

        bytes_per_frame = H * W * 3 * 4
        chunk_size = max(1, min(B, int(1_000_000_000 // max(bytes_per_frame * 10, 1))))

        # For linear input we need the global max across all frames to normalize previews.
        norm_scale = 1.0
        if input_space == "linear":
            max_val = float(image[..., :3].max().item())
            norm_scale = max_val if max_val > 1.0 else 1.0

        filenames = []
        srgb_chunks = []

        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            image_rgb = image[start:end, ..., :3].float().to(device, non_blocking=True)

            # --- Preview frames (8-bit PNG, always in the "raw" pre-exposure space) ---
            if input_space == "linear":
                preview = (image_rgb / norm_scale).clamp_(0.0, 1.0)
            else:
                preview = image_rgb.clamp(0.0, 1.0)

            preview_np = preview.mul_(255.0).add_(0.5).clamp_(0.0, 255.0).to(torch.uint8).cpu().numpy()
            del preview

            for i in range(end - start):
                fname = f"{prefix}_{start + i:05d}.png"
                Image.fromarray(preview_np[i], mode="RGB").save(
                    os.path.join(temp_dir, fname),
                    format="PNG",
                    compress_level=1,
                )
                filenames.append(fname)
            del preview_np

            # --- Baked sRGB output (same math as the shader in hdr_preview.js) ---
            if input_space == "logc3":
                hdr = _logc3_decompress(image_rgb).clamp_(min=0.0)
            elif input_space == "srgb":
                hdr = _srgb_to_linear(image_rgb).clamp_(min=0.0)
            else:
                hdr = image_rgb.clamp(min=0.0)
            del image_rgb

            exposed = hdr.mul_(exposure_mul)
            luma = (exposed * luma_weights.to(exposed.dtype)).sum(dim=-1, keepdim=True)
            saturated = (luma + (exposed - luma) * saturation).clamp_(min=0.0)
            del luma, exposed

            if input_space == "srgb":
                # Already display-ready linear; skip Reinhard, just clip over-exposed highlights.
                tonemapped = saturated.clamp_(0.0, 1.0)
            else:
                tonemapped = saturated / (1.0 + saturated)
                del saturated

            srgb_chunks.append(_linear_to_srgb(tonemapped).cpu())
            del tonemapped

        srgb = torch.cat(srgb_chunks, dim=0)
        del srgb_chunks

        data = {
            "frames": [{"filename": f, "type": "temp"} for f in filenames],
            "width": int(W),
            "height": int(H),
            "fps": float(fps),
            "input_space": input_space,
            "linear_scale": float(norm_scale),
            "frame_count": int(B),
            "exposure": float(exposure),
            "saturation": float(saturation),
        }
        return io.NodeOutput(srgb, ui={"hdr_preview_data": [data]})
