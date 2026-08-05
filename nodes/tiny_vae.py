"""Loaders for tiny VAE decoders that ComfyUI core can't build yet.

TinyVAEDecoder handles flat TAESD-style decoders. Core's Decoder hardcodes a 64-wide stack
with 3 upsamples, so anything else fails to load — the 2D taeh3 is 96 wide with 4. The
architecture is recoverable from the checkpoint: keys are positional module indices, so
`N.conv.0.weight` is a Block, `N.weight` a conv, and the gaps are parameterless modules.

TAEHVDecoder handles the temporal taehv format, which core builds but can't size for H3.
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
        self.model = _place(self.model, self.device, self.dtype)

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


def _place(model, device, dtype):
    model = model.eval().to(device=device, dtype=dtype)
    if torch.device(device).type == "cuda":
        model.to(memory_format=torch.channels_last)
    return model


def is_taehv_state_dict(sd):
    return "decoder.1.weight" in sd and "decoder.22.bias" in sd


class TAEHVDecoder:
    """Temporal tiny VAE (madebyollin/taehv), decode only."""

    def __init__(self, sd, device=None, dtype=None):
        from comfy.taesd.taehv import TAEHV, conv

        latent_channels = sd["decoder.1.weight"].shape[1]
        # final conv emits image_channels * patch_size**2
        patch_size = max(1, int(round((sd["decoder.22.bias"].shape[0] / 3) ** 0.5)))
        model = TAEHV(latent_channels=latent_channels)
        # core derives patch_size from the channel count and has no entry for H3's 24;
        # those two convs are the only parts of the model that depend on it
        if model.patch_size != patch_size:
            model.patch_size = patch_size
            model.encoder[0] = conv(3 * patch_size ** 2, model.encoder[0].out_channels)
            model.decoder[-1] = conv(model.decoder[-1].in_channels, 3 * patch_size ** 2)
        model.load_state_dict(sd)
        del model.encoder  # decode only, so keep it off the device entirely

        self.device = device if device is not None else comfy.model_management.vae_device()
        self.dtype = dtype if dtype is not None else comfy.model_management.vae_dtype(
            self.device, [torch.float16, torch.bfloat16])
        self.model = _place(model, self.device, self.dtype)
        self.latent_channels = latent_channels
        self.upscale_ratio = patch_size * 2 ** sum(isinstance(m, nn.Upsample) for m in model.decoder)
        self.is_h3 = latent_channels == 24 and patch_size == 2

    def _decode(self, latent):
        # [B, C, T, H, W] -> [B, 3, T*t_upscale - trim, H*ratio, W*ratio]
        out = self.model.decode(latent.to(device=self.device, dtype=self.dtype))
        return out.to(device=latent.device, dtype=torch.float32)

    def decode(self, latent):
        """[B, C, H, W] -> [B, 3, H*ratio, W*ratio], decoded as a single frame."""
        return self._decode(latent.unsqueeze(2))[:, :, 0]

    def _decode_h3_full(self, latent):
        """Whole-clip decode in H3's own chunking, so the frame count comes out exact."""
        # H3's VAE codes 17 pixel frames per 5 latent tokens: trim each chunk's prefix
        # rather than once globally, then drop the encoder's 3-token tail pad
        from comfy.taesd.taehv import apply_model_with_memblocks

        m = self.model
        x = m.process_in(latent.to(device=self.device, dtype=self.dtype)).movedim(2, 1)
        x = apply_model_with_memblocks(m.decoder, x, m.parallel, False,
                                       output_device=comfy.model_management.intermediate_device(),
                                       patch_size=m.patch_size, decode=True)
        chunk = 5 * m.t_upscale
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 0, -x.shape[1] % chunk))
        x = x.unflatten(1, (-1, chunk))[:, :, m.frames_to_trim:].flatten(1, 2)
        x = x[:, :-3 * m.t_upscale]
        return x.movedim(2, 1).to(device=latent.device, dtype=torch.float32)

    def decode_video(self, latent, frame_indices=None):
        """[B, C, T, H, W] -> [n, H*ratio, W*ratio, 3]."""
        t_total = latent.shape[2]
        n = t_total if frame_indices is None else max(1, min(len(frame_indices), t_total))
        if n == t_total:
            # whole clip asked for, so decode it properly — this returns every pixel frame,
            # which is more than the latent frame count (17 per 5 tokens for H3)
            if self.is_h3:
                out = self._decode_h3_full(latent[:1])
                if out.shape[2] > 0:
                    return out[0].movedim(0, -1)
            return self._decode(latent[:1])[0].movedim(0, -1)
        # MemBlock state chains forward, so frames can't be sampled across the clip without
        # decoding everything before them — take a prefix to keep the per-step cost bounded
        out = self._decode(latent[:1, :, :n])[0].movedim(0, -1)
        if out.shape[0] > n:
            out = out[torch.linspace(0, out.shape[0] - 1, n).round().long()]
        return out


def load_tiny_vae_decoder(name, device=None, dtype=None):
    """Load by vae_approx filename. Returns None (and logs) if it can't be used."""
    import folder_paths

    path = folder_paths.get_full_path("vae_approx", name)
    if path is None:
        logging.warning(f"[KJ TinyVAE] '{name}' not found in models/vae_approx.")
        return None
    try:
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        if is_taehv_state_dict(sd):
            return TAEHVDecoder(sd, device=device, dtype=dtype)
        return TinyVAEDecoder(sd, device=device, dtype=dtype)
    except Exception as e:
        logging.warning(f"[KJ TinyVAE] Could not load '{name}': {e}")
        return None
