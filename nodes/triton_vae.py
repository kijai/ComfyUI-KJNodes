"""Fused norm+SiLU Triton kernels for VAE decode/encode.

Two architectures are covered, auto-detected by build_object_patches:
- Wan video VAEs: channel-dim RMSNorm (F.normalize(x, dim=1) * sqrt(C) * gamma)
  followed by SiLU, wired as Sequential pairs.
- KL image VAEs (Flux2, SDXL, SD1.5, ...): GroupNorm followed by a shared swish
  module inside ResnetBlock, plus the GN-only norm_out heads.

The eager chains make 4-5 full memory passes per call; the fused kernels do one
read+write with fp32 accumulation. The GroupNorm kernel requires channels_last activations, which is
also what makes channels_last conv layout viable for KL VAEs at all: PyTorch's
own channels_last GroupNorm kernel is ~4x slower than NCHW, ours is ~4x faster.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import triton
    import triton.language as tl
except ImportError as e:
    raise ImportError("PatchTritonVAE requires triton (pip install triton, or triton-windows)") from e

from comfy.ldm.wan.vae import RMS_norm
from comfy.ldm.modules.diffusionmodules.model import ResnetBlock

import copy
import logging

import comfy.model_patcher
from comfy.patcher_extension import CallbacksMP
from comfy_api.latest import io

@triton.jit
def _rms_silu_kernel(x_ptr, g_ptr, out_ptr, C, S, stride_b, eps, scale, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    base = x_ptr + pid_b * stride_b
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        ptrs = base + offs_c[:, None] * S + offs_s[None, :]
        v = tl.load(ptrs, mask=mask_c[:, None] & mask_s[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(v * v, axis=0)
    inv = scale / tl.maximum(tl.sqrt(acc), eps)
    out_base = out_ptr + pid_b * stride_b
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        m = mask_c[:, None] & mask_s[None, :]
        ptrs = base + offs_c[:, None] * S + offs_s[None, :]
        v = tl.load(ptrs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        y = v * inv[None, :] * g[:, None]
        y = y * tl.sigmoid(y)
        tl.store(out_base + offs_c[:, None] * S + offs_s[None, :], y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _rms_silu_cl_kernel(x_ptr, g_ptr, out_ptr, C, ROWS, eps, scale, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = tl.arange(0, BLOCK_C)
    mask_s = offs_s < ROWS
    mask_c = offs_c < C
    m = mask_s[:, None] & mask_c[None, :]
    ptrs = x_ptr + offs_s[:, None] * C + offs_c[None, :]
    v = tl.load(ptrs, mask=m, other=0.0).to(tl.float32)
    inv = scale / tl.maximum(tl.sqrt(tl.sum(v * v, axis=1)), eps)
    g = tl.load(g_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    y = v * inv[:, None] * g[None, :]
    y = y * tl.sigmoid(y)
    tl.store(out_ptr + offs_s[:, None] * C + offs_c[None, :], y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _gn_stats_cl(x_ptr, sum_ptr, sumsq_ptr, S,
                 G: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C)
    acc1 = tl.zeros([G], dtype=tl.float32)
    acc2 = tl.zeros([G], dtype=tl.float32)
    s0 = pid * BLOCK_S
    step = tl.num_programs(0) * BLOCK_S
    while s0 < S:
        offs_s = s0 + tl.arange(0, BLOCK_S)
        m = (offs_s < S)[:, None]
        ptr = x_ptr + (b * S + offs_s[:, None]) * BLOCK_C + offs_c[None, :]
        v = tl.load(ptr, mask=m, other=0.0).to(tl.float32)
        v3 = tl.reshape(v, (BLOCK_S, G, BLOCK_C // G))
        acc1 += tl.sum(tl.sum(v3, axis=2), axis=0)
        acc2 += tl.sum(tl.sum(v3 * v3, axis=2), axis=0)
        s0 += step
    tl.atomic_add(sum_ptr + b * G + tl.arange(0, G), acc1)
    tl.atomic_add(sumsq_ptr + b * G + tl.arange(0, G), acc2)


@triton.jit
def _gn_silu_apply_cl(x_ptr, sum_ptr, sumsq_ptr, w_ptr, bias_ptr, out_ptr, S, count, eps,
                      G: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr, SILU: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    CS: tl.constexpr = BLOCK_C // G
    offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = tl.arange(0, BLOCK_C)
    m = (offs_s < S)[:, None]
    ptr = x_ptr + (b * S + offs_s[:, None]) * BLOCK_C + offs_c[None, :]
    v = tl.load(ptr, mask=m, other=0.0).to(tl.float32)
    mean_g = tl.load(sum_ptr + b * G + tl.arange(0, G)) / count
    var_g = tl.load(sumsq_ptr + b * G + tl.arange(0, G)) / count - mean_g * mean_g
    var_g = tl.maximum(var_g, 0.0)  # E[x²]-mean² can go slightly negative from fp32 cancellation
    rstd_g = 1.0 / tl.sqrt(var_g + eps)
    mean_c = tl.reshape(tl.broadcast_to(mean_g[:, None], (G, CS)), (BLOCK_C,))
    rstd_c = tl.reshape(tl.broadcast_to(rstd_g[:, None], (G, CS)), (BLOCK_C,))
    w = tl.load(w_ptr + offs_c).to(tl.float32)
    bias = tl.load(bias_ptr + offs_c).to(tl.float32)
    y = (v - mean_c[None, :]) * rstd_c[None, :] * w[None, :] + bias[None, :]
    if SILU:
        y = y * tl.sigmoid(y)
    tl.store(out_ptr + (b * S + offs_s[:, None]) * BLOCK_C + offs_c[None, :],
             y.to(out_ptr.dtype.element_ty), mask=m)


# autotuned variants: benchmark configs once per shape key, then cache the fastest.
# BLOCK_C of the cl/GN kernels is structural (must hold all of C) so only BLOCK_S/num_warps are tuned.
_rms_silu_kernel_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK_S": bs, "BLOCK_C": bc}, num_warps=w)
             for bs in (128, 256, 512) for bc in (32, 64) for w in (4, 8)],
    key=["C", "S"])(_rms_silu_kernel)

_rms_silu_cl_kernel_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK_S": bs}, num_warps=w) for bs in (4, 8, 16, 32, 64) for w in (4, 8)],
    key=["C", "ROWS"])(_rms_silu_cl_kernel)

_gn_stats_cl_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK_S": bs}, num_warps=w) for bs in (8, 16, 32, 64) for w in (4, 8)],
    key=["S"], reset_to_zero=["sum_ptr", "sumsq_ptr"])(_gn_stats_cl)

_gn_silu_apply_cl_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK_S": bs}, num_warps=w) for bs in (8, 16, 32, 64) for w in (4, 8)],
    key=["S", "count"])(_gn_silu_apply_cl)


def fused_rms_silu(x, gamma, scale, eps=1e-12, autotune=False):
    B, C = x.shape[0], x.shape[1]
    S = x.numel() // (B * C)
    x = x.contiguous()
    out = torch.empty_like(x)
    if autotune:
        grid = lambda meta: (triton.cdiv(S, meta["BLOCK_S"]), B)
        _rms_silu_kernel_tuned[grid](x, gamma, out, C, S, C * S, eps, scale)
    else:
        _rms_silu_kernel[(triton.cdiv(S, 256), B)](x, gamma, out, C, S, C * S, eps, scale,
                                                   BLOCK_S=256, BLOCK_C=32, num_warps=8)
    return out


def fused_rms_silu_cl(x, gamma, scale, eps=1e-12, autotune=False):
    C = x.shape[1]
    rows = x.numel() // C
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    if autotune:
        grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_S"]),)
        _rms_silu_cl_kernel_tuned[grid](x, gamma, out, C, rows, eps, scale, BLOCK_C=BLOCK_C)
    else:
        BLOCK_S = max(1, 4096 // BLOCK_C)
        _rms_silu_cl_kernel[(triton.cdiv(rows, BLOCK_S),)](x, gamma, out, C, rows, eps, scale,
                                                           BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C, num_warps=4)
    return out


def fused_gn_silu_cl(x, weight, bias, groups, eps, silu=True, autotune=False):
    B, C, H, W = x.shape
    S = H * W
    sums = torch.zeros(2, B * groups, device=x.device, dtype=torch.float32)
    out = torch.empty_like(x)
    count = S * (C // groups)
    if autotune:
        # program count beyond ~1024 only adds same-address atomic contention
        grid_stats = lambda meta: (min(1024, triton.cdiv(S, meta["BLOCK_S"])), B)
        _gn_stats_cl_tuned[grid_stats](x, sums[0], sums[1], S, G=groups, BLOCK_C=C)
        grid_apply = lambda meta: (triton.cdiv(S, meta["BLOCK_S"]), B)
        _gn_silu_apply_cl_tuned[grid_apply](x, sums[0], sums[1], weight, bias, out, S, count, eps,
                                            G=groups, BLOCK_C=C, SILU=silu)
    else:
        BLOCK_S = max(1, 8192 // C)
        nprog = min(1024, triton.cdiv(S, BLOCK_S))
        _gn_stats_cl[(nprog, B)](x, sums[0], sums[1], S, G=groups, BLOCK_S=BLOCK_S, BLOCK_C=C, num_warps=8)
        _gn_silu_apply_cl[(triton.cdiv(S, BLOCK_S), B)](x, sums[0], sums[1], weight, bias, out, S, count, eps,
                                                        G=groups, BLOCK_S=BLOCK_S, BLOCK_C=C, SILU=silu, num_warps=8)
    return out


class FusedRMSSiLU(nn.Module):
    def __init__(self, rms, autotune=False):
        super().__init__()
        self.gamma = rms.gamma
        self.scale = rms.scale
        self.autotune = autotune

    def forward(self, x):
        if not x.is_cuda:
            return F.silu(F.normalize(x, dim=1) * self.scale * self.gamma.to(x))
        gamma = self.gamma.reshape(-1)
        if not gamma.is_contiguous():
            gamma = gamma.contiguous()
        if x.ndim == 5 and x.is_contiguous(memory_format=torch.channels_last_3d) and not x.is_contiguous():
            return fused_rms_silu_cl(x, gamma, self.scale, autotune=self.autotune)
        return fused_rms_silu(x, gamma, self.scale, autotune=self.autotune)


class FusedGNSiLU(nn.Module):
    def __init__(self, gn, silu=True, autotune=False):
        super().__init__()
        self.weight = gn.weight
        self.bias = gn.bias
        self.num_groups = gn.num_groups
        self.eps = gn.eps
        self.silu = silu
        self.autotune = autotune

    def forward(self, x):
        C = x.shape[1]
        G = self.num_groups
        if x.is_cuda and x.ndim == 4 and (C & (C - 1)) == 0 and (G & (G - 1)) == 0 and C % G == 0 \
                and x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous():
            return fused_gn_silu_cl(x, self.weight, self.bias, self.num_groups, self.eps, self.silu,
                                    autotune=self.autotune)
        out = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        if self.silu:
            out = F.silu(out)
        return out


def convert_conv_layout(model, channels_last=True):
    for mod in model.modules():
        if isinstance(mod, nn.Conv3d):
            mod.to(memory_format=torch.channels_last_3d if channels_last else torch.contiguous_format)
        elif isinstance(mod, nn.Conv2d):
            mod.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)


def build_object_patches(model, autotune=False):
    """Object patches for vae.patcher: applied at model load, reverted at unload.
    Also matches already-fused modules so a loaded+patched model rebuilds cleanly."""
    patches = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Sequential):  # Wan-style RMS_norm+SiLU pairs
            for i in range(len(mod) - 1):
                if isinstance(mod[i], RMS_norm) and isinstance(mod[i + 1], nn.SiLU) \
                        and mod[i].gamma.ndim == 4 and mod[i].bias is None:
                    patches[f"{name}.{i}"] = FusedRMSSiLU(mod[i], autotune=autotune)
                    patches[f"{name}.{i + 1}"] = nn.Identity()
                elif isinstance(mod[i], FusedRMSSiLU) and isinstance(mod[i + 1], nn.Identity):
                    mod[i].autotune = autotune
                    patches[f"{name}.{i}"] = mod[i]
                    patches[f"{name}.{i + 1}"] = mod[i + 1]
        elif isinstance(mod, ResnetBlock) and not hasattr(mod, "temb_proj"):  # KL-style
            for norm_name in ("norm1", "norm2"):
                norm = getattr(mod, norm_name)
                if isinstance(norm, nn.GroupNorm):
                    patches[f"{name}.{norm_name}"] = FusedGNSiLU(norm, autotune=autotune)
                elif isinstance(norm, FusedGNSiLU):
                    norm.autotune = autotune
                    patches[f"{name}.{norm_name}"] = norm
            patches[f"{name}.swish"] = nn.Identity()
        elif name.endswith("norm_out"):  # KL encoder/decoder head, SiLU applied separately
            if isinstance(mod, nn.GroupNorm):
                patches[name] = FusedGNSiLU(mod, silu=False, autotune=autotune)
            elif isinstance(mod, FusedGNSiLU):
                mod.autotune = autotune
                patches[name] = mod
    return patches


class PatchTritonVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PatchTritonVAE",
            display_name="Patch Triton VAE",
            category="KJNodes/experimental",
            is_experimental=True,
            description="Speeds up VAE decode/encode with fused Triton norm+SiLU kernels and channels_last conv layout. "
                        "Supported VAEs (auto-detected): Wan 2.1/2.2 video VAEs incl. Qwen-Image (RMSNorm, ~1.4x/1.15x) and KL image VAEs "
                        "such as Flux/Flux2, SDXL and SD1.5 (GroupNorm, ~1.6-1.8x at 2048px). Other architectures are not supported. "
                        "Applied as object patches on a cloned patcher, so it only exists while this VAE is loaded.",
            inputs=[
                io.Vae.Input("vae"),
                io.Boolean.Input("fuse_norm_silu", default=True, tooltip="Replace norm+SiLU chains (RMSNorm for Wan, GroupNorm for KL VAEs) with fused Triton kernels (single pass, fp32 accumulation). Requires triton."),
                io.Boolean.Input("channels_last", default=True, tooltip="Convert conv weights to channels_last memory format, removing cuDNN layout transposes around every conv. Required for the fused GroupNorm kernel to engage on KL VAEs."),
                io.Boolean.Input("autotune", default=False, tooltip="Benchmark several kernel block-size configs on first use of each tensor shape and cache the fastest. Brief stutter per new resolution, usually a few percent faster after warmup."),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
            ],
        )

    @classmethod
    def execute(cls, vae, fuse_norm_silu=True, channels_last=True, autotune=False) -> io.NodeOutput:
        vae = copy.copy(vae)
        model = vae.first_stage_model
        if vae.patcher.is_dynamic():
            # the dynamic-vram patcher rematerializes weights from pinned host buffers per forward, which would discard channels_last layout
            model.to(vae.vae_dtype)
            new_patcher = comfy.model_patcher.ModelPatcher(
                model, load_device=vae.patcher.load_device, offload_device=vae.patcher.offload_device)
            # without a parent, LoadedModel flags is_dead when this patcher is GC'd (leak warning + full gc)
            new_patcher.parent = vae.patcher
            vae.patcher = new_patcher

            def clear_dynamic_cast_flags(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
                # a dynamic load of the shared model sets these without ever resetting them;
                for mod in patcher.model.modules():
                    if hasattr(mod, "comfy_cast_weights") and not hasattr(mod, "prev_comfy_cast_weights"):
                        mod.comfy_cast_weights = False
                        mod._v_signature = None
            vae.patcher.add_callback_with_key(CallbacksMP.ON_LOAD, "wan_vae_fused_clear_cast", clear_dynamic_cast_flags)
        else:
            vae.patcher = vae.patcher.clone()

        if fuse_norm_silu:
            patches = build_object_patches(model, autotune=autotune)
            if not patches:
                raise RuntimeError("No RMS_norm+SiLU pairs or GroupNorm ResnetBlocks found, this node supports Wan video VAEs and KL image VAEs (Flux2/SDXL/SD1.5)")
            for name, obj in patches.items():
                vae.patcher.add_object_patch(name, obj)
            logging.info(f"PatchTritonVAE: registered {len(patches)} fused norm object patches")

        if channels_last:
            # reapply on every load: pinned-host/dynamic-vram weight staging restores the original contiguous layout
            def reapply_channels_last(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
                convert_conv_layout(patcher.model, channels_last=True)
            vae.patcher.add_callback_with_key(CallbacksMP.ON_LOAD, "wan_vae_fused_channels_last", reapply_channels_last)
        else:
            convert_conv_layout(model, channels_last=False)
        return io.NodeOutput(vae)
