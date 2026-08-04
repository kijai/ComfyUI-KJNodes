"""Loader for TAESD-family tiny VAE decoders that ComfyUI core can't build yet.

Core's comfy.taesd.taesd.Decoder hardcodes a 64-wide stack with 3 upsample stages, so
checkpoints that differ fail to load — e.g. taeh3 (MiniMax H3): 24 latent channels, 96
wide for the first half, 4 upsample stages for the VAE's 16x spatial ratio.

These decoders are a flat nn.Sequential whose state dict keys are the positional module
indices, so the exact architecture is recoverable from the checkpoint: indices carrying
`N.conv.0.weight` are Blocks, indices carrying `N.weight` are convs (bias-less ones sit
right after an upsample), and the gaps are the parameterless modules.
"""

import logging

import torch
import torch.nn as nn

import comfy.model_management
import comfy.utils
from comfy.taesd.taesd import Block, Clamp, conv


def build_tae_decoder(sd):
    by_index = {}
    for k, v in sd.items():
        head, _, rest = k.partition(".")
        if not head.isdigit():
            raise ValueError(f"not a flat TAE decoder state dict (unexpected key '{k}')")
        by_index.setdefault(int(head), {})[rest] = v

    modules = []
    for i in range(max(by_index) + 1):
        entry = by_index.get(i)
        if entry is None:
            # index 0 is the input Clamp, 2 the ReLU after the input conv, the rest are upsamples
            modules.append(Clamp() if i == 0 else nn.ReLU() if i == 2 else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            w = entry["conv.0.weight"]
            # only pass the kwarg when it's needed — older ComfyUI has no midblock-GN variant
            if "pool.0.weight" in entry:
                modules.append(Block(w.shape[1], w.shape[0], use_midblock_gn=True))
            else:
                modules.append(Block(w.shape[1], w.shape[0]))
        elif "weight" in entry:
            w = entry["weight"]
            modules.append(conv(w.shape[1], w.shape[0], bias="bias" in entry))
        else:
            raise ValueError(f"unrecognized TAE decoder module at index {i}: {sorted(entry)}")
    return nn.Sequential(*modules)


class TinyVAEDecoder:
    """Decode-only tiny VAE. Output is [0, 1], matching the TAE family's convention."""

    def __init__(self, sd, device=None, dtype=None):
        # keys may carry a "taesd_decoder."/"decoder." prefix; strip whatever is common
        prefix = ""
        first = next(iter(sd))
        if not first.split(".")[0].isdigit():
            prefix = first.split(".")[0] + "."
            sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        self.device = device if device is not None else comfy.model_management.vae_device()
        self.dtype = dtype if dtype is not None else comfy.model_management.vae_dtype(
            self.device, [torch.float16, torch.bfloat16])
        self.model = build_tae_decoder(sd)
        self.model.load_state_dict(sd)
        self.model.eval().to(device=self.device, dtype=self.dtype)

        self.latent_channels = self.model[1].weight.shape[1]
        self.upscale_ratio = 2 ** sum(isinstance(m, nn.Upsample) for m in self.model)

    def decode(self, latent):
        """[B, C, H, W] -> [B, 3, H*ratio, W*ratio], float32 on the input device."""
        out = self.model(latent.to(device=self.device, dtype=self.dtype))
        return out.to(device=latent.device, dtype=torch.float32)

    def decode_video(self, latent, frame_indices=None):
        """[B, C, T, H, W] -> [T, H*ratio, W*ratio, 3]. Decodes one frame at a time —
        at 16x the full-resolution activations are the memory peak, not the weights."""
        x = latent[0]
        indices = range(x.shape[1]) if frame_indices is None else frame_indices
        frames = [self.decode(x[:, t].unsqueeze(0))[0].movedim(0, -1) for t in indices]
        return torch.stack(frames, dim=0)


def load_tiny_vae_decoder(name, device=None, dtype=None):
    """Load by vae_approx filename. Returns None (and logs) if it can't be used."""
    import folder_paths

    path = folder_paths.get_full_path("vae_approx", name)
    if path is None:
        logging.warning(f"[KJ TinyVAE] '{name}' not found in models/vae_approx.")
        return None
    try:
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        return TinyVAEDecoder(sd, device=device, dtype=dtype)
    except Exception as e:
        logging.warning(f"[KJ TinyVAE] Could not load '{name}': {e}")
        return None
