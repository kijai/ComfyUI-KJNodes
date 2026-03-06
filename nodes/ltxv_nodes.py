from comfy_extras.nodes_lt import get_noise_mask, LTXVAddGuide
import types
import math
from typing import Tuple
import comfy
from comfy_api.latest import io
import numpy as np
import torch
import logging
import comfy.model_management as mm
device = mm.get_torch_device()
import latent_preview

class LTXVAddGuideMulti(LTXVAddGuide):

    @classmethod
    def define_schema(cls):
        options = []
        for num_guides in range(1, 21):  # 1 to 20 guides
            guide_inputs = []
            for i in range(1, num_guides + 1):
                guide_inputs.extend([
                    io.Image.Input(f"image_{i}"),
                    io.Int.Input(
                        f"frame_idx_{i}",
                        default=0,
                        min=-9999,
                        max=9999,
                        tooltip=f"Frame index for guide {i}.",
                    ),
                    io.Float.Input(f"strength_{i}", default=1.0, min=0.0, max=1.0, step=0.01, tooltip=f"Strength for guide {i}."),
                ])
            options.append(io.DynamicCombo.Option(
                key=str(num_guides),
                inputs=guide_inputs
            ))

        return io.Schema(
            node_id="LTXVAddGuideMulti",
            category="KJNodes/ltxv",
            description="Add multiple guide images at specified frame indices with strengths, uses DynamicCombo which requires ComfyUI 0.8.1 and frontend 1.33.4 or later.",
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to which guide keyframe info will be added"),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to which guide keyframe info will be added"),
                io.Vae.Input("vae", tooltip="Video VAE used to encode the guide images"),
                io.Latent.Input("latent", tooltip="Video latent, guides are added to the end of this latent"),
                io.DynamicCombo.Input(
                    "num_guides",
                    options=options,
                    display_name="Number of Guides",
                    tooltip="Select how many guide images to use",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Video latent with added guides"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, num_guides) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        # num_guides is a dict containing the inputs from the selected option
        # e.g., {'image_1': tensor, 'frame_idx_1': 0, 'strength_1': 1.0, 'image_2': tensor, 'frame_idx_2': 20, 'strength_2': 0.8, ...}

        image_keys = sorted([k for k in num_guides.keys() if k.startswith('image_')])

        for img_key in image_keys:
            i = img_key.split('_')[1]

            img = num_guides[f"image_{i}"]
            f_idx = num_guides[f"frame_idx_{i}"]
            strength = num_guides[f"strength_{i}"]

            image_1, t = cls.encode(vae, latent_width, latent_height, img, scale_factors)

            frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)
            assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                strength,
                scale_factors,
            )

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

class LTXVAddGuidesFromBatch(LTXVAddGuide):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVAddGuidesFromBatch",
            category="conditioning/ltxv",
            description="Adds multiple guide images from a batch to the latent at corresponding frame indices. Non-black images in the batch are used as guides.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Image.Input("images", tooltip="Batch of images - non-black images will be used as guides"),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, images, strength) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        # Process each image in the batch
        batch_size = images.shape[0]

        for i in range(batch_size):
            img = images[i:i+1]

            # Check if image is not black and use batch index as frame index
            if img.max() > 0.001:
                f_idx = i

                image_1, t = cls.encode(vae, latent_width, latent_height, img, scale_factors)

                frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)

                if latent_idx + t.shape[2] <= latent_length:
                    positive, negative, latent_image, noise_mask = cls.append_keyframe(
                        positive,
                        negative,
                        frame_idx,
                        latent_image,
                        noise_mask,
                        t,
                        strength,
                        scale_factors,
                    )
                else:
                    print(f"Warning: Skipping guide at index {i} - conditioning frames exceed latent sequence length")

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


class LTXVAudioVideoMask(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVAudioVideoMask",
            category="KJNodes/ltxv",
            description="Creates noise masks for video and audio latents based on specified time ranges. New content is generated within these masked regions",
            inputs=[
                io.Latent.Input("video_latent", optional=True),
                io.Latent.Input("audio_latent", optional=True),
                io.Float.Input("video_fps", default=25, min=0.0, max=100.0, step=0.1),
                io.Float.Input("video_start_time", default=0.0, min=0.0, max=10000.0, step=0.1, tooltip="Start time in seconds for the video mask."),
                io.Float.Input("video_end_time", default=5.0, min=0.0, max=10000.0, step=0.1, tooltip="End time in seconds for the video mask."),
                io.Float.Input("audio_start_time", default=0.0, min=0.0, max=10000.0, step=0.1, tooltip="Start time in seconds for the audio mask."),
                io.Float.Input("audio_end_time", default=5.0, min=0.0, max=10000.0, step=0.1, tooltip="End time in seconds for the audio mask."),
                io.Combo.Input(
                    "max_length",
                    options=["truncate", "pad", "partial"],
                    default="truncate",
                    tooltip="'truncate': cut latent to end_time length. 'pad': extend latent to end_time. 'partial': mask range within existing latent.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, video_fps, video_start_time, video_end_time, audio_start_time, audio_end_time, max_length="truncate", video_latent=None, audio_latent=None) -> io.NodeOutput:

        time_scale_factor = 8
        mel_hop_length = 160
        sampling_rate = 16000
        latent_downsample_factor = 4
        audio_latents_per_second = (sampling_rate / mel_hop_length / latent_downsample_factor) # 25

        if video_latent is not None:
            video_latent_frame_count = video_latent["samples"].shape[2]

            video_pixel_frame_start_raw = int(round(video_start_time * video_fps))
            video_pixel_frame_end_raw = int(round(video_end_time * video_fps))

            # Calculate required latent frames based on end time
            required_latent_frames = (video_pixel_frame_end_raw - 1) // time_scale_factor + 1

            # Handle different max_length modes
            if max_length == "pad" and required_latent_frames > video_latent_frame_count:
                # Pad video latent if required frames exceed current length
                pad_frames = required_latent_frames - video_latent_frame_count
                padding = torch.zeros(
                    video_latent["samples"].shape[0],
                    video_latent["samples"].shape[1],
                    pad_frames,
                    video_latent["samples"].shape[3],
                    video_latent["samples"].shape[4],
                    dtype=video_latent["samples"].dtype,
                    device=video_latent["samples"].device
                )
                video_samples = torch.cat([video_latent["samples"], padding], dim=2)
                video_latent_frame_count = video_samples.shape[2]
            elif max_length == "truncate":
                # Truncate to the end_time
                video_samples = video_latent["samples"][:, :, :required_latent_frames]
                video_latent_frame_count = video_samples.shape[2]
            else:  # partial
                video_samples = video_latent["samples"]

            # Now calculate indices based on potentially padded latent
            video_pixel_frame_count = (video_latent_frame_count - 1) * time_scale_factor + 1
            xp = np.array([0] + list(range(1, video_pixel_frame_count + time_scale_factor, time_scale_factor)))

            # video_frame_index_start = index of the value in xp rounding up
            video_latent_frame_index_start = np.searchsorted(xp, video_pixel_frame_start_raw, side="left")
            # video_frame_index_end = index of the value in xp rounding down
            video_latent_frame_index_end = np.searchsorted(xp, video_pixel_frame_end_raw, side="right") - 1

            video_latent_frame_index_start = max(0, video_latent_frame_index_start)
            video_latent_frame_index_end = min(video_latent_frame_index_end, video_latent_frame_count)

            # Get existing noise mask if present, otherwise create new one
            if "noise_mask" in video_latent:
                video_mask = video_latent["noise_mask"].clone()
                # Adjust mask size based on mode
                if max_length == "pad" and video_samples.shape[2] > video_latent["samples"].shape[2]:
                    # Pad the mask if we padded the samples
                    mask_padding = torch.zeros(
                        video_mask.shape[0],
                        video_mask.shape[1],
                        video_samples.shape[2] - video_mask.shape[2],
                        video_mask.shape[3],
                        video_mask.shape[4],
                        dtype=video_mask.dtype,
                        device=video_mask.device
                    )
                    video_mask = torch.cat([video_mask, mask_padding], dim=2)
                elif max_length == "truncate":
                    # Truncate the mask to match truncated samples
                    video_mask = video_mask[:, :, :video_samples.shape[2]]
            else:
                video_mask = torch.zeros_like(video_samples)[:, :1]

            video_mask[:, :, video_latent_frame_index_start:video_latent_frame_index_end] = 1.0
            # ensure all padded frames are also masked
            if max_length == "pad" and video_samples.shape[2] > video_latent["samples"].shape[2]:
                video_mask[:, :, video_latent["samples"].shape[2]:] = 1.0
            video_latent = video_latent.copy()
            video_latent["samples"] = video_samples
            video_latent["noise_mask"] = video_mask

        if audio_latent is not None:
            audio_latent_frame_count = audio_latent["samples"].shape[2]

            audio_latent_frame_index_start = int(round(audio_start_time * audio_latents_per_second))
            audio_latent_frame_index_end = int(round(audio_end_time * audio_latents_per_second)) + 1

            # Handle different max_length modes
            if max_length == "pad" and audio_latent_frame_index_end > audio_latent_frame_count:
                # Pad audio latent if end index exceeds current length
                pad_frames = audio_latent_frame_index_end - audio_latent_frame_count
                padding = torch.zeros(
                    audio_latent["samples"].shape[0],
                    audio_latent["samples"].shape[1],
                    pad_frames,
                    audio_latent["samples"].shape[3],
                    dtype=audio_latent["samples"].dtype,
                    device=audio_latent["samples"].device
                )
                audio_samples = torch.cat([audio_latent["samples"], padding], dim=2)
                audio_latent_frame_count = audio_samples.shape[2]
            elif max_length == "truncate":
                # Truncate to the end_time
                audio_samples = audio_latent["samples"][:, :, :audio_latent_frame_index_end]
                audio_latent_frame_count = audio_samples.shape[2]
            else:  # partial
                audio_samples = audio_latent["samples"]

            audio_latent_frame_index_start = max(0, audio_latent_frame_index_start)
            audio_latent_frame_index_end = min(audio_latent_frame_index_end, audio_latent_frame_count)

            # Get existing noise mask if present, otherwise create new one
            if "noise_mask" in audio_latent:
                audio_mask = audio_latent["noise_mask"].clone()
                # Adjust mask size based on mode
                if max_length == "pad" and audio_samples.shape[2] > audio_latent["samples"].shape[2]:
                    # Pad the mask if we padded the samples
                    mask_padding = torch.zeros(
                        audio_mask.shape[0],
                        audio_mask.shape[1],
                        audio_samples.shape[2] - audio_mask.shape[2],
                        audio_mask.shape[3],
                        dtype=audio_mask.dtype,
                        device=audio_mask.device
                    )
                    audio_mask = torch.cat([audio_mask, mask_padding], dim=2)
                elif max_length == "truncate":
                    # Truncate the mask to match truncated samples
                    audio_mask = audio_mask[:, :, :audio_samples.shape[2]]
            else:
                audio_mask = torch.zeros_like(audio_samples)

            audio_mask[:, :, audio_latent_frame_index_start:audio_latent_frame_index_end] = 1.0
            # ensure all padded frames are also masked
            if max_length == "pad" and audio_samples.shape[2] > audio_latent["samples"].shape[2]:
                audio_mask[:, :, audio_latent["samples"].shape[2]:] = 1.0
            audio_latent = audio_latent.copy()
            audio_latent["samples"] = audio_samples
            audio_latent["noise_mask"] = audio_mask

        return io.NodeOutput(video_latent, audio_latent)

def _compute_attention(self, query, context, attention_precision=None, transformer_options={}):
    """Compute attention and return the result. Cleans up intermediate tensors."""
    k = self.k_norm(self.to_k(context)).to(query.dtype)
    v = self.to_v(context).to(query.dtype)
    x = comfy.ldm.modules.attention.optimized_attention(query, k, v, heads=self.heads, attn_precision=attention_precision, transformer_options=transformer_options).flatten(2)
    del k, v
    return x

def nag_attention(self, query, context_positive, nag_context, attention_precision=None, transformer_options={}):
    x_positive = _compute_attention(self, query, context_positive, attention_precision, transformer_options)
    x_negative = _compute_attention(self, query, nag_context, attention_precision, transformer_options)
    return x_positive, x_negative

def normalized_attention_guidance(self, x_positive, x_negative):
    if self.inplace:
        nag_guidance = x_negative.mul_(self.nag_scale - 1).neg_().add_(x_positive, alpha=self.nag_scale)
    else:
        nag_guidance = x_positive * self.nag_scale - x_negative * (self.nag_scale - 1)

    del x_negative

    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)

    scale = norm_guidance / norm_positive
    torch.nan_to_num_(scale, nan=10.0)
    mask = scale > self.nag_tau
    del scale

    adjustment = (norm_positive * self.nag_tau) / (norm_guidance + 1e-7)
    del norm_positive, norm_guidance

    nag_guidance.mul_(torch.where(mask, adjustment, 1.0))
    del mask, adjustment

    if self.inplace:
        nag_guidance.sub_(x_positive).mul_(self.nag_alpha).add_(x_positive)
    else:
        nag_guidance = nag_guidance * self.nag_alpha + x_positive * (1 - self.nag_alpha)
    del x_positive

    return nag_guidance

#region NAG
def ltxv_crossattn_forward_nag(self, x, context, mask=None, transformer_options={}, **kwargs):

    # Single or [pos, neg] pair
    if context.shape[0] == 1:
        x_pos, context_pos = x, context
        x_neg, context_neg = None, None
    else:
        x_pos, x_neg = torch.chunk(x, 2, dim=0)
        context_pos, context_neg = torch.chunk(context, 2, dim=0)

    # Positive
    q_pos = self.q_norm(self.to_q(x_pos))
    del x_pos

    x_positive, x_negative = nag_attention(self, q_pos, context_pos, self.nag_context, attention_precision=self.attn_precision, transformer_options=transformer_options)
    del context_pos, q_pos

    x_pos_out = normalized_attention_guidance(self, x_positive, x_negative)
    del x_positive, x_negative

    # Negative
    if x_neg is not None and context_neg is not None:
        q_neg = self.q_norm(self.to_q(x_neg))
        k_neg = self.k_norm(self.to_k(context_neg))
        v_neg = self.to_v(context_neg)

        x_neg_out = comfy.ldm.modules.attention.optimized_attention(q_neg, k_neg, v_neg, heads=self.heads, attn_precision=self.attn_precision, transformer_options=transformer_options)
        out = torch.cat([x_pos_out, x_neg_out], dim=0)
    else:
        out = x_pos_out

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)  # (B, T, H)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)  # zero-init -> identity
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)


class LTXVCrossAttentionPatch:
    def __init__(self, context, nag_scale, nag_alpha, nag_tau, inplace=True):
        self.nag_context = context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau
        self.inplace = inplace

    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.nag_context = self.nag_context
            self_module.nag_scale = self.nag_scale
            self_module.nag_alpha = self.nag_alpha
            self_module.nag_tau = self.nag_tau
            self_module.inplace = self.inplace

            return ltxv_crossattn_forward_nag(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)

class LTX2_NAG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2_NAG",
            display_name="LTX2 NAG",
            category="KJNodes/ltxv",
            description="https://github.com/ChenDarYen/Normalized-Attention-Guidance",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("nag_scale", default=11.0, min=0.0, max=100.0, step=0.001, tooltip="Strength of negative guidance effect"),
                io.Float.Input("nag_alpha", default=0.25, min=0.0, max=1.0, step=0.001, tooltip="Mixing coefficient in that controls the balance between the normalized guided representation and the original positive representation."),
                io.Float.Input("nag_tau", default=2.5, min=0.0, max=10.0, step=0.001, tooltip="Clipping threshold that controls how much the guided attention can deviate from the positive attention."),
                io.Conditioning.Input("nag_cond_video", optional=True),
                io.Conditioning.Input("nag_cond_audio", optional=True),
                io.Boolean.Input("inplace", default=True, optional=True, tooltip="If true, modifies tensors in place to save memory. Leads to different numerical results which may change the output slightly."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, nag_scale, nag_alpha, nag_tau, nag_cond_video=None, nag_cond_audio=None, inplace=True) -> io.NodeOutput:
        if nag_scale == 0:
            return io.NodeOutput(model)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = model.model.manual_cast_dtype
        if dtype is None:
            dtype = model.model.diffusion_model.dtype

        model_clone = model.clone()

        diffusion_model = model_clone.get_model_object("diffusion_model")
        img_dim = diffusion_model.inner_dim
        audio_dim = diffusion_model.audio_inner_dim

        context_video = context_audio = None

        if nag_cond_video is not None:
            context_video = nag_cond_video[0][0].to(device, dtype)
            if hasattr(diffusion_model, "preprocess_text_embeds"):
                context_video = diffusion_model.preprocess_text_embeds(context_video.to(device=device, dtype=dtype))
            v_context, _ = torch.split(context_video, int(context_video.shape[-1] / 2), len(context_video.shape) - 1)
            if diffusion_model.caption_proj_before_connector and diffusion_model.caption_projection_first_linear:
                diffusion_model.caption_projection.to(device)
                context_video = diffusion_model.caption_projection(v_context)
                diffusion_model.caption_projection.to(offload_device)
            context_video = context_video.view(1, -1, img_dim)
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                patched_attn2 = LTXVCrossAttentionPatch(context_video, nag_scale, nag_alpha, nag_tau, inplace=inplace).__get__(block.attn2, block.__class__)
                model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn2.forward", patched_attn2)

        if nag_cond_audio is not None and diffusion_model.audio_caption_projection is not None:
            context_audio = nag_cond_audio[0][0].to(device, dtype)
            if hasattr(diffusion_model, "preprocess_text_embeds"):
                context_audio = diffusion_model.preprocess_text_embeds(context_audio.to(device=device, dtype=dtype))
            _, a_context = torch.split(context_audio, int(context_audio.shape[-1] / 2), len(context_audio.shape) - 1)
            if diffusion_model.caption_proj_before_connector and diffusion_model.caption_projection_first_linear:
                diffusion_model.audio_caption_projection.to(device)
                context_audio = diffusion_model.audio_caption_projection(a_context)
                diffusion_model.audio_caption_projection.to(offload_device)
            context_audio = context_audio.view(1, -1, audio_dim)
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                patched_audio_attn2 = LTXVCrossAttentionPatch(context_audio, nag_scale, nag_alpha, nag_tau, inplace=inplace).__get__(block.audio_attn2, block.__class__)
                model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.audio_attn2.forward", patched_audio_attn2)

        return io.NodeOutput(model_clone)


def ffn_chunked_forward(self, x):
    if x.shape[1] > self.dim_threshold:
        chunk_size = x.shape[1] // self.num_chunks
        for i in range(self.num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.num_chunks - 1 else x.shape[1]
            x[:, start_idx:end_idx] = self.net(x[:, start_idx:end_idx])
        return x
    else:
        return self.net(x)

class LTXVffnChunkPatch:
    def __init__(self, num_chunks, dim_threshold=4096):
        self.num_chunks = num_chunks
        self.dim_threshold = dim_threshold

    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.num_chunks = self.num_chunks
            self_module.dim_threshold = self.dim_threshold
            return ffn_chunked_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)

class LTXVChunkFeedForward(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVChunkFeedForward",
            display_name="LTXV Chunk FeedForward",
            category="KJNodes/ltxv",
            description="EXPERIMENTAL AND MAY CHANGE THE MODEL OUTPUT!! Chunks feedforward activations to reduce peak VRAM usage.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("chunks", default=2, min=1, max=100, step=1, tooltip="Number of chunks to split the feedforward activations into to reduce peak VRAM usage."),
                io.Int.Input("dim_threshold", default=4096, min=0, max=16384, step=256, tooltip="Dimension threshold above which to apply chunking."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, chunks, dim_threshold) -> io.NodeOutput:
        if chunks == 1:
            return io.NodeOutput(model)

        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        for idx, block in enumerate(diffusion_model.transformer_blocks):
            patched_attn2 = LTXVffnChunkPatch(chunks, dim_threshold).__get__(block.ff, block.__class__)
            model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.ff.forward", patched_attn2)

        return io.NodeOutput(model_clone)



#borrowed VideoHelperSuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/videohelpersuite/latent_preview.py
import server
from threading import Thread
import torch.nn.functional as F
import time
import struct
from PIL import Image
from io import BytesIO
serv = server.PromptServer.instance

class WrappedPreviewer():
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias, rate=8, taeltx=None):
        self.first_preview = True
        self.taeltx = taeltx
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu") if latent_rgb_factors_bias is not None else None

    def decode_latent_to_preview_image(self, preview_format, x0):
        if x0.ndim == 5:
            #Keep batch major
            x0 = x0.movedim(2,1)
            x0 = x0.reshape((-1,)+x0.shape[-3:])
        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews/self.rate
        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None
        if self.first_preview:
            self.first_preview = False
            serv.send_sync('VHS_latentpreview', {'length':num_images, 'rate': self.rate, 'id': serv.last_node_id})
            self.last_time = new_time + 1/self.rate
        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index:self.c_index + num_previews]
        Thread(target=self.process_previews, args=(x0, self.c_index,
                                                   num_images)).run()
        self.c_index = (self.c_index + num_previews) % num_images
        return None
    def process_previews(self, image_tensor, ind, leng):
        max_size = 512
        min_size = 256
        image_tensor = self.decode_latent_to_preview(image_tensor)

        if image_tensor.size(1) < min_size or image_tensor.size(2) < min_size:
            image_tensor = F.interpolate(image_tensor.movedim(-1,0), scale_factor=4, mode='nearest').movedim(0,-1)

        if image_tensor.size(1) > max_size or image_tensor.size(2) > max_size:
            image_tensor = image_tensor.movedim(-1,0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (max_size * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height,max_size), mode='nearest')
            else:
                width = (max_size * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (max_size, width), mode='nearest')
            image_tensor = image_tensor.movedim(0,-1)

        previews_ubyte = (image_tensor.clamp(0, 1).mul(0xFF)).to(device="cpu", dtype=torch.uint8)

        # Send VHS preview
        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            message = BytesIO()
            message.write((1).to_bytes(length=4, byteorder='big')*2)
            message.write(ind.to_bytes(length=4, byteorder='big'))
            message.write(struct.pack('16p', serv.last_node_id.encode('ascii')))
            i.save(message, format="JPEG", quality=95, compress_level=1)
            #NOTE: send sync already uses call_soon_threadsafe
            serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE,
                           message.getvalue(), serv.client_id)
            if self.taeltx is not None:
                ind = (ind + 1) % ((leng-1) * 8 + 1)
            else:
                ind = (ind + 1) % leng

    def decode_latent_to_preview(self, x0):
        if self.taeltx is not None:
            x0 = x0.unsqueeze(0).to(dtype=self.taeltx.first_stage_model.decoder[1].weight.dtype, device=device)
            x_sample = self.taeltx.first_stage_model.decode(x0)[0].permute(1, 2, 3, 0)
            return x_sample
        else:
            self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)
            latent_image = F.linear(x0.movedim(1, -1), self.latent_rgb_factors,
                                    bias=self.latent_rgb_factors_bias)

            low = latent_image.quantile(0.01)
            high = latent_image.quantile(0.99)
            latent_image = ((latent_image - low) / (high - low)).clamp(0, 1)
            return latent_image


def prepare_callback(model, steps, x0_output_dict=None, shape=None, latent_upscale_model=None, vae=None, rate=8, taeltx=False, num_keyframes=0, is_23=False):
    if not is_23:
        latent_rgb_factors = [
                [ 0.0350,  0.0159,  0.0132],
                [ 0.0025, -0.0021, -0.0003],
                [ 0.0286,  0.0028,  0.0020],
                [ 0.0280, -0.0114, -0.0202],
                [-0.0186,  0.0073,  0.0092],
                [ 0.0027,  0.0097, -0.0113],
                [-0.0069, -0.0032, -0.0024],
                [-0.0323, -0.0370, -0.0457],
                [ 0.0174,  0.0164,  0.0106],
                [-0.0097,  0.0061,  0.0035],
                [-0.0130, -0.0042, -0.0012],
                [-0.0102, -0.0002, -0.0091],
                [-0.0025,  0.0063,  0.0161],
                [ 0.0003,  0.0037,  0.0108],
                [ 0.0152,  0.0082,  0.0143],
                [ 0.0317,  0.0203,  0.0312],
                [-0.0092, -0.0233, -0.0119],
                [-0.0405, -0.0226, -0.0023],
                [ 0.0376,  0.0397,  0.0352],
                [ 0.0171, -0.0043, -0.0095],
                [ 0.0482,  0.0341,  0.0213],
                [ 0.0031, -0.0046, -0.0018],
                [-0.0486, -0.0383, -0.0294],
                [-0.0071, -0.0272, -0.0123],
                [ 0.0320,  0.0218,  0.0289],
                [ 0.0327,  0.0088, -0.0116],
                [-0.0098, -0.0240, -0.0111],
                [ 0.0094, -0.0116,  0.0021],
                [ 0.0309,  0.0092,  0.0165],
                [-0.0065, -0.0077, -0.0107],
                [ 0.0179,  0.0114,  0.0038],
                [-0.0018, -0.0030, -0.0026],
                [-0.0002,  0.0076, -0.0029],
                [-0.0131, -0.0059, -0.0170],
                [ 0.0055,  0.0066, -0.0038],
                [ 0.0154,  0.0063,  0.0090],
                [ 0.0186,  0.0175,  0.0188],
                [-0.0166, -0.0381, -0.0428],
                [ 0.0121,  0.0015, -0.0153],
                [ 0.0118,  0.0050,  0.0019],
                [ 0.0125,  0.0259,  0.0231],
                [ 0.0046,  0.0130,  0.0081],
                [ 0.0271,  0.0250,  0.0250],
                [-0.0054, -0.0347, -0.0326],
                [-0.0438, -0.0262, -0.0228],
                [-0.0191, -0.0256, -0.0173],
                [-0.0205, -0.0058,  0.0042],
                [ 0.0404,  0.0434,  0.0346],
                [-0.0242, -0.0177, -0.0146],
                [ 0.0161,  0.0223,  0.0168],
                [-0.0240, -0.0320, -0.0299],
                [-0.0019,  0.0043,  0.0008],
                [-0.0060, -0.0133, -0.0244],
                [-0.0048, -0.0225, -0.0167],
                [ 0.0267,  0.0133,  0.0152],
                [ 0.0222,  0.0167,  0.0028],
                [ 0.0015, -0.0062,  0.0013],
                [-0.0241, -0.0178, -0.0079],
                [ 0.0040, -0.0081, -0.0097],
                [-0.0064,  0.0133, -0.0011],
                [-0.0204, -0.0231, -0.0304],
                [ 0.0011, -0.0011,  0.0145],
                [-0.0283, -0.0259, -0.0260],
                [ 0.0038,  0.0171, -0.0029],
                [ 0.0637,  0.0424,  0.0409],
                [ 0.0092,  0.0163,  0.0188],
                [ 0.0082,  0.0055, -0.0179],
                [-0.0177, -0.0286, -0.0147],
                [ 0.0171,  0.0242,  0.0398],
                [-0.0129,  0.0095, -0.0071],
                [-0.0154,  0.0036,  0.0128],
                [-0.0081, -0.0009,  0.0118],
                [-0.0067, -0.0178, -0.0230],
                [-0.0022, -0.0125, -0.0003],
                [-0.0032, -0.0039, -0.0022],
                [-0.0005, -0.0127, -0.0131],
                [-0.0143, -0.0157, -0.0165],
                [-0.0262, -0.0263, -0.0270],
                [ 0.0063,  0.0127,  0.0178],
                [ 0.0092,  0.0133,  0.0150],
                [-0.0106, -0.0068,  0.0032],
                [-0.0214, -0.0022,  0.0171],
                [-0.0104, -0.0266, -0.0362],
                [ 0.0021,  0.0048, -0.0005],
                [ 0.0345,  0.0431,  0.0402],
                [-0.0275, -0.0110, -0.0195],
                [ 0.0203,  0.0251,  0.0224],
                [ 0.0016, -0.0037, -0.0094],
                [ 0.0241,  0.0198,  0.0114],
                [-0.0003,  0.0027,  0.0141],
                [ 0.0012, -0.0052, -0.0084],
                [ 0.0057, -0.0028, -0.0163],
                [-0.0488, -0.0545, -0.0509],
                [-0.0076, -0.0025, -0.0014],
                [-0.0249, -0.0142, -0.0367],
                [ 0.0136,  0.0041,  0.0135],
                [ 0.0007,  0.0034, -0.0053],
                [-0.0068, -0.0109,  0.0029],
                [ 0.0006, -0.0237, -0.0094],
                [-0.0149, -0.0177, -0.0131],
                [-0.0105,  0.0039,  0.0216],
                [ 0.0242,  0.0200,  0.0180],
                [-0.0339, -0.0153, -0.0195],
                [ 0.0104,  0.0151,  0.0120],
                [-0.0043,  0.0089,  0.0047],
                [ 0.0157, -0.0030,  0.0008],
                [ 0.0126,  0.0102, -0.0040],
                [ 0.0040,  0.0114,  0.0137],
                [ 0.0423,  0.0473,  0.0436],
                [-0.0128, -0.0066, -0.0152],
                [-0.0337, -0.0087, -0.0026],
                [-0.0052,  0.0235,  0.0291],
                [ 0.0079,  0.0154,  0.0260],
                [-0.0539, -0.0377, -0.0358],
                [-0.0188,  0.0062, -0.0035],
                [-0.0186,  0.0041, -0.0083],
                [ 0.0045, -0.0049,  0.0053],
                [ 0.0172,  0.0071,  0.0042],
                [-0.0003, -0.0078, -0.0096],
                [-0.0209, -0.0132, -0.0135],
                [-0.0074,  0.0017,  0.0099],
                [-0.0038,  0.0070,  0.0014],
                [-0.0013, -0.0017,  0.0073],
                [ 0.0030,  0.0105,  0.0105],
                [ 0.0154, -0.0168, -0.0235],
                [-0.0108, -0.0038,  0.0047],
                [-0.0298, -0.0347, -0.0436],
                [-0.0206, -0.0189, -0.0139]
            ]
        latent_rgb_factors_bias = [0.2796, 0.1101, -0.0047]
    else:
        latent_rgb_factors = [[0.023152293637394905, 0.008659062907099724, 0.007089507766067982], [-0.0017481173854321241, -5.615880945697427e-05, 0.0032591703347861767], [0.016661887988448143, 0.010700271464884281, 0.009281095117330551], [0.030969517305493355, 0.008591392077505589, -0.0007975368062034249], [0.017112644389271736, 0.006124283652752638, 0.016782492399215698], [-0.011142476461827755, -0.0028044464997947216, 0.007351504173129797], [-0.018173379823565483, -0.031134044751524925, -0.03484560549259186], [-0.006836462765932083, -0.0008386926492676139, 0.004300590138882399], [0.011448710225522518, 0.012729380279779434, 0.015028423629701138], [0.0030850328039377928, -0.012077241204679012, -0.021550748497247696], [-0.014377499930560589, -0.01327194832265377, -0.010316358879208565], [-0.008129404857754707, -0.0006633125012740493, 0.0038912466261535883], [0.01535210944712162, 0.013854483142495155, 0.004103456158190966], [0.021925266832113266, 0.026155536994338036, 0.013075708411633968], [0.006744623649865389, 0.005066306795924902, -0.0021272373851388693], [0.03606889396905899, 0.027413709089159966, 0.024750912562012672], [-0.010133359581232071, -0.005365394987165928, -0.0015616178279742599], [0.006176959257572889, 0.012287200428545475, 0.008172821253538132], [-0.0020756004378199577, 0.0010795736452564597, 0.0037552921567112207], [0.0076816026121377945, 0.0007980416994541883, 0.005487715359777212], [-6.147939711809158e-05, -0.003009824315086007, -0.009086601436138153], [-0.003097475739195943, 0.0076638986356556416, 0.009116833098232746], [0.010111968964338303, 0.00905579887330532, 0.016790516674518585], [0.003387401346117258, 0.0013503145892173052, -0.0007817387813702226], [0.012572303414344788, 0.0202279444783926, 0.008845849893987179], [-0.010375721380114555, -0.014635534025728703, -0.00888782273977995], [-0.0031294559594243765, 0.003545328974723816, -0.0020312361884862185], [-0.010668950155377388, -0.0010736528784036636, 0.0064877914264798164], [-0.007589623332023621, -0.00890421774238348, -0.010155928321182728], [-0.015637509524822235, -0.004661687649786472, -0.006087623070925474], [0.0010001378832384944, -0.0007695468957535923, 0.007963811047375202], [0.009555311873555183, 0.0024523024912923574, -0.0028863796032965183], [0.0006326463771983981, 0.002317993436008692, -0.003659574082121253], [0.007543386425822973, 0.000969987828284502, 0.00467210728675127], [0.013601789250969887, 0.002284204587340355, 0.0028190030716359615], [-0.005335876252502203, -0.005909997504204512, -0.0047254981473088264], [0.0019774024840444326, -0.0004714909882750362, -0.0026649204082787037], [0.020652391016483307, 0.010778743773698807, -0.0008758048643358052], [-0.008003628812730312, -0.01763632521033287, -0.013426926918327808], [0.0076448568142950535, 0.0017351864371448755, 0.0023979307152330875], [0.011655328795313835, 0.0007561694947071373, 0.001728428527712822], [-0.027024272829294205, -0.017540089786052704, -0.012272858992218971], [-0.025678347796201706, -0.013488398864865303, -0.009427686221897602], [-0.00126330042257905, -0.006546520162373781, -0.006574796978384256], [-0.015390781685709953, -0.01418320182710886, -0.01651645451784134], [0.042718544602394104, 0.02521219477057457, 0.03512519225478172], [0.005203000735491514, 0.009739041328430176, 0.00971255823969841], [0.018030347302556038, 0.017368938773870468, 0.009983435273170471], [0.017262162640690804, 0.01643703319132328, 0.01940014399588108], [0.01304449699819088, -0.00022519860067404807, -0.0019722182769328356], [-0.012866531498730183, -0.009849924594163895, -0.01101815328001976], [0.027065826579928398, 0.023882873356342316, 0.017460422590374947], [-0.002149017760530114, -0.0048354631289839745, -0.005925675388425589], [0.01126411184668541, 0.017699461430311203, 0.015129977837204933], [-0.004810300190001726, -0.00968858040869236, -0.02082108147442341], [0.006631530821323395, 0.005350326653569937, 0.011687885969877243], [0.003963936120271683, 0.0008817348862066865, -0.0028934204019606113], [0.02986362762749195, 0.044911615550518036, 0.047315891832113266], [-0.014298886060714722, -0.014699430204927921, -0.009611102752387524], [-0.0005055639776401222, 0.001526764128357172, 0.007024736143648624], [-0.0022410389501601458, -0.0014325077645480633, -0.0024764498230069876], [-0.01100095547735691, -0.0082693537697196, -0.005145569331943989], [4.7199311666190624e-05, 0.01007300615310669, 0.01281747780740261], [0.01156687829643488, 0.00037906033685430884, -0.0004753572284244001], [-0.0006763937417417765, 0.01272936724126339, 0.015242298133671284], [-0.026727553457021713, -0.022057197988033295, -0.031014638021588326], [-0.006075659766793251, -0.0016017392044886947, 0.0040812245570123196], [-0.010808146558701992, -0.011887932196259499, -0.007613923866301775], [3.796868259087205e-05, 0.0029933962505310774, 0.005436496809124947], [0.0024653258733451366, -0.0072980718687176704, -0.003344171680510044], [0.013797946274280548, 0.009718045592308044, 0.0032146857120096684], [-0.003777711885049939, -0.001181806088425219, -0.001594073255546391], [-0.006684885825961828, -0.002889942144975066, -0.007792340125888586], [-0.01572795771062374, -0.012930076569318771, -0.00928826630115509], [-0.01874835602939129, -0.0075674704276025295, -0.0016446937806904316], [-0.04274476692080498, -0.05715841427445412, -0.05588245391845703], [-0.01287306472659111, -0.010185712948441505, -0.008599143475294113], [0.02108091674745083, 0.018407778814435005, 0.012230940163135529], [-0.014195051975548267, -0.006500702816992998, -0.005639523267745972], [-0.013165016658604145, -0.000986397615633905, 0.00418048445135355], [-0.009570245631039143, -0.011959787458181381, -0.010289434343576431], [0.016156351193785667, 0.008281491696834564, 0.006084192544221878], [0.042509470134973526, 0.026275595650076866, 0.019993359223008156], [-0.020138340070843697, -0.02178509719669819, -0.02018536999821663], [-0.01688985712826252, -0.008751098066568375, -0.005298278294503689], [0.0021132659167051315, -0.006826960016041994, -0.004156344570219517], [0.0213309396058321, 0.021661635488271713, 0.024688463658094406], [0.015322874300181866, 0.011483061127364635, 0.006787601392716169], [-0.018125122413039207, -0.010336644016206264, -0.003608457278460264], [-0.011396413668990135, -0.0031715675722807646, 0.00013172885519452393], [0.0004603993147611618, -0.005084575619548559, 0.0005643954500555992], [0.01745714247226715, 0.018884073942899704, 0.02073814906179905], [-0.009409075602889061, 0.004540730267763138, 0.0033161805476993322], [0.01393672451376915, 0.0051903557032346725, 0.010769308544695377], [0.030873887240886688, 0.020937038585543633, 0.017154591158032417], [-0.008651122450828552, -0.006343003362417221, -0.0070458208210766315], [0.004786082077771425, 0.02008633501827717, 0.019036758691072464], [-0.027862809598445892, -0.03557579591870308, -0.047106098383665085], [0.009377469308674335, 0.008725658059120178, 0.0004689408524427563], [0.0067672207951545715, 0.0029923454858362675, 0.007426049094647169], [0.024414820596575737, 0.021159106865525246, 0.02155803143978119], [-0.008758587762713432, -0.02325029857456684, -0.024170733988285065], [-0.027606766670942307, -0.020386775955557823, -0.011500534601509571], [0.03842690959572792, 0.05081281438469887, 0.05273008346557617], [0.027395356446504593, 0.013771085999906063, 0.005362789612263441], [0.024438932538032532, 0.019016092643141747, 0.02408655174076557], [-0.01664666458964348, -0.015681801363825798, -0.01325436495244503], [-0.0009208855917677283, 8.95546399988234e-05, 0.006721258629113436], [-0.007715767715126276, -0.009740329347550869, 0.001954615581780672], [0.016358669847249985, 0.02182287909090519, 0.02872852422297001], [-0.005065123084932566, -0.002604612149298191, -0.009478142485022545], [-0.002575461519882083, 0.006894931197166443, 0.009547821246087551], [-0.03662169352173805, -0.028026245534420013, -0.025282494723796844], [-0.001314469613134861, 0.0010471580317243934, 0.0029894462786614895], [0.000399910903070122, -0.0023625714238733053, -0.010908326134085655], [-0.0005835064803250134, 0.0011942405253648758, 0.002311111893504858], [-0.03934014216065407, -0.047908660024404526, -0.052158452570438385], [-0.003979772794991732, -0.0037646314594894648, -0.002668222412467003], [0.010000263340771198, 0.004553306847810745, 0.0005724489456042647], [0.03252670168876648, 0.020110340788960457, 0.017566651105880737], [0.00048384201363660395, 0.0027561988681554794, -0.0007845616200938821], [0.03444176912307739, 0.0528838224709034, 0.05433813855051994], [-0.010707678273320198, -0.012343521229922771, -0.007253612391650677], [-0.0002622777537908405, -0.005775974132120609, -0.006958072539418936], [0.012661139480769634, 0.0076528494246304035, 0.007778205443173647], [-0.02316928096115589, -0.013633089140057564, -0.009908167645335197], [-0.01704544760286808, -0.027270395308732986, -0.033381566405296326], [0.0020333407446742058, 0.002290748991072178, 0.005900708492845297]]
        latent_rgb_factors_bias = [0.21232624351978302, 0.07564208656549454, -0.02137378416955471]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = WrappedPreviewer(latent_rgb_factors, latent_rgb_factors_bias, rate=rate, taeltx=vae if taeltx else None)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0 is not None and shape is not None:
            cut = math.prod(shape[1:])
            x0 = x0[:, :, :cut].reshape([x0.shape[0]] + list(shape)[1:])

        if num_keyframes > 0:
            x0 = x0[:, :, :-num_keyframes]

        if latent_upscale_model is not None:
            x0 = vae.first_stage_model.per_channel_statistics.un_normalize(x0)
            x0 =  latent_upscale_model(x0.to(torch.bfloat16))
            x0 = vae.first_stage_model.per_channel_statistics.normalize(x0)

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

class OuterSampleCallbackWrapper:
    def __init__(self, latent_upscale_model=None, vae=None, preview_rate=8, taeltx=False):
        self.latent_upscale_model = latent_upscale_model
        self.vae = vae
        self.preview_rate = preview_rate
        self.taeltx = taeltx
        self.x0_output = {}

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes):
        guider = executor.class_obj
        diffusion_model = guider.model_patcher.model.diffusion_model
        is_23 = not diffusion_model.caption_projection_first_linear

        original_callback = callback
        if self.latent_upscale_model is not None:
            self.latent_upscale_model.to(device)
        if self.vae is not None and self.taeltx:
            self.vae.first_stage_model.to(device)

        num_keyframes = 0
        if 'positive' in guider.conds and len(guider.conds['positive']) > 0:
            keyframe_idxs = guider.conds['positive'][0].get('keyframe_idxs')
            if keyframe_idxs is not None:
                num_keyframes = len(torch.unique(keyframe_idxs[0, 0, :, 0]))

        new_callback = prepare_callback(guider.model_patcher, len(sigmas) -1, shape=latent_shapes[0] if len(latent_shapes) > 1 else None,
                                        x0_output_dict=self.x0_output, latent_upscale_model=self.latent_upscale_model, vae=self.vae, rate=self.preview_rate, taeltx=self.taeltx, num_keyframes=num_keyframes, is_23=is_23)
        # Wrapper that calls both callbacks
        def combined_callback(step, x0, x, total_steps):
            new_callback(step, x0, x, total_steps)
            if original_callback is not None:
                original_callback(step, x0, x, total_steps)
        out = executor(noise, latent_image, sampler, sigmas, denoise_mask, combined_callback, disable_pbar, seed, latent_shapes=latent_shapes)
        if self.latent_upscale_model is not None:
            self.latent_upscale_model.to(mm.unet_offload_device())
        return out

class LTX2SamplingPreviewOverride(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX2SamplingPreviewOverride",
            display_name="LTX2 Sampling Preview Override",
            description="Overrides the LTX2 preview sampling preview function, temporary measure until previews are in comfy core",
            category="KJNodes/ltxv",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The model to add preview override to."),
                io.Int.Input("preview_rate", default=8, min=1, max=60, step=1, tooltip="Preview frame rate."),
                io.LatentUpscaleModel.Input("latent_upscale_model", optional=True, tooltip="Optional upscale model to use for higher resolution previews."),
                io.Vae.Input("vae", optional=True, tooltip="VAE model to use normalizing the latents for the upscale model."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with Sampling Preview Override."),
            ],
        )

    @classmethod
    def execute(cls, model, preview_rate, latent_upscale_model=None, vae=None) -> io.NodeOutput:
        model = model.clone()
        taeltx = False
        if vae is not None:
            if vae.first_stage_model.__class__.__name__ == "TAEHV":
                taeltx = True
                latent_upscale_model=None
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "sampling_preview", OuterSampleCallbackWrapper(latent_upscale_model, vae, preview_rate, taeltx))
        return io.NodeOutput(model)


# based on https://github.com/Lightricks/ComfyUI-LTXVideo/blob/cd5d371518afb07d6b3641be8012f644f25269fc/easy_samplers.py#L916
class OuterSampleAudioNormalizationWrapper:
    def __init__(self, audio_normalization_factors):
        self.audio_normalization_factors = audio_normalization_factors

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes):
        guider = executor.class_obj
        ltxav = guider.model_patcher.model.diffusion_model

        x0_output = {}
        self.total_steps = sigmas.shape[-1] - 1
        pbar = comfy.utils.ProgressBar(self.total_steps)
        self.full_step = 0

        previewer = latent_preview.get_previewer(guider.model_patcher.load_device, guider.model_patcher.model.latent_format)
        def custom_callback(step, x0, x, total_steps):
            if x0_output is not None:
                x0_output["x0"] = x0

            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
            self.full_step += 1
            pbar.update_absolute(self.full_step, self.total_steps, preview_bytes)

        callback = custom_callback

        audio_normalization_factors = self.audio_normalization_factors.strip().split(",")
        audio_normalization_factors = [float(factor) for factor in audio_normalization_factors]

        # Extend normalization factors to match the length of sigmas
        sigmas_len = self.total_steps
        if len(audio_normalization_factors) < sigmas_len and len(audio_normalization_factors) > 0:
            audio_normalization_factors.extend([audio_normalization_factors[-1]] * (sigmas_len - len(audio_normalization_factors)))

        # Calculate indices where both normalization factors are not 1.0
        sampling_split_indices = [i + 1 for i, a in enumerate(audio_normalization_factors) if a != 1.0]

        # Split sigmas according to sampling_split_indices
        def split_by_indices(arr, indices):
            """
            Splits arr into chunks according to indices (split points).
            Indices are treated as starting a new chunk at each index in the list.
            """
            if not indices:
                return [arr]
            split_points = sorted(set(indices))
            chunks = []
            prev = 0
            for idx in split_points:
                if prev < idx:
                    chunks.append(arr[prev : idx + 1])
                prev = idx
            if prev < len(arr):
                chunks.append(arr[prev:])
            return chunks

        sigmas_chunks = split_by_indices(sigmas, sampling_split_indices)

        i = 0
        for sigmas_chunk in sigmas_chunks:
            i += len(sigmas_chunk) - 1
            latent_image = executor(noise, latent_image, sampler, sigmas_chunk, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

            if "x0" in x0_output:
                latent_image = guider.model_patcher.model.process_latent_out(x0_output["x0"])

            if i - 1 < len(audio_normalization_factors):
                vx, ax = ltxav.separate_audio_and_video_latents(comfy.utils.unpack_latents(latent_image, latent_shapes), None)
                if denoise_mask is not None:
                    audio_mask = ltxav.separate_audio_and_video_latents(comfy.utils.unpack_latents(denoise_mask, latent_shapes), None)[1]
                    ax = ax * audio_mask * audio_normalization_factors[i - 1] + ax * (1 - audio_mask)
                else:
                    ax = ax * audio_normalization_factors[i - 1]
                latent_image = comfy.utils.pack_latents(ltxav.recombine_audio_and_video_latents(vx, ax))[0]

                print("After %d steps, the audio latent was normalized by %f" % (i, audio_normalization_factors[i - 1]))

        return latent_image


class LTX2AudioLatentNormalizingSampling(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX2AudioLatentNormalizingSampling",
            display_name="LTX2 Audio Latent Normalizing Sampling",
            description="Improves LTX2 generated audio quality by normalizing audio latents at specified sampling steps.",
            category="KJNodes/ltxv",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The model to add preview override to."),
                io.String.Input("audio_normalization_factors", default="1,1,0.25,1,1,0.25,1,1", tooltip="Comma-separated list of audio normalization factors to apply at each sampling step. For example, '1,1,0.25,1,1,0.25,1,1' will apply a factor of 0.25 at the 3rd and 6th steps."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with Audio Latent Normalizing Sampling."),
            ],
        )

    @classmethod
    def execute(cls, model, audio_normalization_factors) -> io.NodeOutput:
        model = model.clone()
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "ltx2_audio_normalization", OuterSampleAudioNormalizationWrapper(audio_normalization_factors))
        return io.NodeOutput(model)


class LTXVImgToVideoInplaceKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = []
        for num_images in range(1, 21):  # 1 to 20 images
            image_inputs = []
            for i in range(1, num_images + 1):
                image_inputs.extend([
                    io.Image.Input(f"image_{i}", optional=True, tooltip=f"Image {i} to insert into the video latent."),
                    io.Int.Input(
                        f"index_{i}",
                        default=0,
                        min=-9999,
                        max=9999,
                        step=1,
                        tooltip=f"Frame index for image {i} (in pixel space).",
                        optional=True,
                    ),
                    io.Float.Input(f"strength_{i}", default=1.0, min=0.0, max=1.0, step=0.01, tooltip=f"Strength for image {i}."),
                ])
            options.append(io.DynamicCombo.Option(
                key=str(num_images),
                inputs=image_inputs
            ))

        return io.Schema(
            node_id="LTXVImgToVideoInplaceKJ",
            category="KJNodes/ltxv",
            description="Replaces video latent frames with the encoded input images, uses DynamicCombo which requires ComfyUI 0.8.1 and frontend 1.33.4 or later.",
            inputs=[
                io.Vae.Input("vae", tooltip="Video VAE used to encode the images"),
                io.Latent.Input("latent", tooltip="Video latent to insert images into"),
                io.DynamicCombo.Input(
                    "num_images",
                    options=options,
                    display_name="Number of Images",
                    tooltip="Select how many images to insert",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="The video latent with the images inserted and latent noise mask updated."),
            ],
        )

    @classmethod
    def execute(cls, vae, latent, num_images) -> io.NodeOutput:

        samples = latent["samples"].clone()
        scale_factors = vae.downscale_index_formula
        _, height_scale_factor, width_scale_factor = scale_factors

        batch, _, latent_frames, latent_height, latent_width = samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor

        # Get existing noise mask if present, otherwise create new one
        if "noise_mask" in latent:
            conditioning_latent_frames_mask = latent["noise_mask"].clone()
        else:
            conditioning_latent_frames_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=samples.device,
            )

        # num_images is a dict containing the inputs from the selected option
        # e.g., {'image_1': tensor, 'frame_idx_1': 0, 'strength_1': 1.0, 'image_2': tensor, 'frame_idx_2': 20, 'strength_2': 0.8, ...}

        image_keys = sorted([k for k in num_images.keys() if k.startswith('image_')])

        for img_key in image_keys:
            i = img_key.split('_')[1]

            image = num_images[f"image_{i}"]
            if image is None:
                continue
            index = num_images.get(f"index_{i}")
            if index is None:
                continue
            strength = num_images[f"strength_{i}"]

            if image.shape[1] != height or image.shape[2] != width:
                pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            else:
                pixels = image
            encode_pixels = pixels[:, :, :, :3]
            t = vae.encode(encode_pixels)

            # Convert pixel frame index to latent index
            time_scale_factor = scale_factors[0]

            # Handle negative indexing in pixel space
            pixel_frame_count = (latent_frames - 1) * time_scale_factor + 1
            if index < 0:
                index = pixel_frame_count + index

            # Convert to latent index
            latent_idx = index // time_scale_factor

            # Clamp to valid range
            latent_idx = max(0, min(latent_idx, latent_frames - 1))

            # Calculate end index, ensuring we don't exceed latent_frames
            end_index = min(latent_idx + t.shape[2], latent_frames)

            # Replace samples at the specified index range
            samples[:, :, latent_idx:end_index] = t[:, :, :end_index - latent_idx]

            # Update mask at the specified index range
            conditioning_latent_frames_mask[:, :, latent_idx:end_index] = 1.0 - strength

        return io.NodeOutput({"samples": samples, "noise_mask": conditioning_latent_frames_mask})


def ltx2_forward(
        self, x: Tuple[torch.Tensor, torch.Tensor], v_context=None, a_context=None, attention_mask=None, v_timestep=None, a_timestep=None,
        v_pe=None, a_pe=None, v_cross_pe=None, a_cross_pe=None, v_cross_scale_shift_timestep=None, a_cross_scale_shift_timestep=None,
        v_cross_gate_timestep=None, a_cross_gate_timestep=None, transformer_options=None, self_attention_mask=None,
        v_prompt_timestep=None, a_prompt_timestep=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        run_vx = transformer_options.get("run_vx", True)
        run_ax = transformer_options.get("run_ax", True)
        video_scale = getattr(self, 'video_scale', 1.0)
        audio_scale = getattr(self, 'audio_scale', 1.0)
        audio_to_video_scale = getattr(self, 'audio_to_video_scale', 1.0)
        video_to_audio_scale = getattr(self, 'video_to_audio_scale', 1.0)

        vx, ax = x
        run_ax = run_ax and ax.numel() > 0 and audio_scale != 0.0
        run_a2v = run_vx and transformer_options.get("a2v_cross_attn", True) and ax.numel() > 0 and audio_to_video_scale != 0.0
        run_v2a = run_ax and transformer_options.get("v2a_cross_attn", True) and video_to_audio_scale != 0.0

        # video
        if run_vx:
            # video self-attention
            vshift_msa, vscale_msa = (self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(0, 2)))
            norm_vx = comfy.ldm.common_dit.rms_norm(vx) * (1 + vscale_msa) + vshift_msa
            del vshift_msa, vscale_msa
            attn1_out = self.attn1(norm_vx, pe=v_pe, mask=self_attention_mask, transformer_options=transformer_options)
            del norm_vx
            # video cross-attention
            vgate_msa = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(2, 3))[0]
            vx.addcmul_(attn1_out, vgate_msa, value=video_scale)
            del vgate_msa, attn1_out
            vx.add_(self._apply_text_cross_attention(
                vx, v_context, self.attn2, self.scale_shift_table,
                getattr(self, 'prompt_scale_shift_table', None),
                v_timestep, v_prompt_timestep, attention_mask, transformer_options,),
                alpha=video_scale
            )

        # audio
        if run_ax:
            # audio self-attention
            ashift_msa, ascale_msa = (self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(0, 2)))
            norm_ax = comfy.ldm.common_dit.rms_norm(ax) * (1 + ascale_msa) + ashift_msa
            del ashift_msa, ascale_msa
            attn1_out = self.audio_attn1(norm_ax, pe=a_pe, transformer_options=transformer_options)
            del norm_ax
            # audio cross-attention
            agate_msa = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(2, 3))[0]
            ax.addcmul_(attn1_out, agate_msa, value=audio_scale)
            del agate_msa, attn1_out
            ax.add_(self._apply_text_cross_attention(
                ax, a_context, self.audio_attn2, self.audio_scale_shift_table,
                getattr(self, 'audio_prompt_scale_shift_table', None),
                a_timestep, a_prompt_timestep, attention_mask, transformer_options,),
                alpha=audio_scale
            )

        # video - audio cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = comfy.ldm.common_dit.rms_norm(vx)
            ax_norm3 = comfy.ldm.common_dit.rms_norm(ax)

            # audio to video cross attention
            if run_a2v:
                scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[:2]
                scale_ca_video_hidden_states_a2v_v, shift_ca_video_hidden_states_a2v_v = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[:2]

                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v_v) + shift_ca_video_hidden_states_a2v_v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
                del scale_ca_video_hidden_states_a2v_v, shift_ca_video_hidden_states_a2v_v, scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v

                a2v_out = self.audio_to_video_attn(vx_scaled, context=ax_scaled, pe=v_cross_pe, k_pe=a_cross_pe, transformer_options=transformer_options)
                del vx_scaled, ax_scaled

                gate_out_a2v = self.get_ada_values(self.scale_shift_table_a2v_ca_video[4:, :], vx.shape[0], v_cross_gate_timestep)[0]
                vx.addcmul_(a2v_out, gate_out_a2v, value=audio_to_video_scale)
                del gate_out_a2v, a2v_out

            # video to audio cross attention
            if run_v2a:
                scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[2:4]
                scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[2:4]

                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
                del scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a, scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a
                del vx_norm3, ax_norm3

                v2a_out = self.video_to_audio_attn(ax_scaled, context=vx_scaled, pe=a_cross_pe, k_pe=v_cross_pe, transformer_options=transformer_options)
                del ax_scaled, vx_scaled

                gate_out_v2a = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[4:, :], ax.shape[0], a_cross_gate_timestep)[0]
                ax.addcmul_(v2a_out, gate_out_v2a, value=video_to_audio_scale)
                del gate_out_v2a, v2a_out
            else:
                del vx_norm3, ax_norm3

        # video feedforward
        if run_vx:
            vshift_mlp, vscale_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(3, 5))
            vx_scaled = comfy.ldm.common_dit.rms_norm(vx) * (1 + vscale_mlp) + vshift_mlp
            del vshift_mlp, vscale_mlp

            ff_out = self.ff(vx_scaled)
            del vx_scaled

            vgate_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(5, 6))[0]
            vx.addcmul_(ff_out, vgate_mlp, value=video_scale)
            del vgate_mlp, ff_out

        # audio feedforward
        if run_ax:
            ashift_mlp, ascale_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(3, 5))
            ax_scaled = comfy.ldm.common_dit.rms_norm(ax) * (1 + ascale_mlp) + ashift_mlp
            del ashift_mlp, ascale_mlp

            ff_out = self.audio_ff(ax_scaled)
            del ax_scaled

            agate_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(5, 6))[0]
            ax.addcmul_(ff_out, agate_mlp, value=audio_scale)
            del agate_mlp, ff_out

        return vx, ax

class LTX2ForwardPatch:
    def __init__(self, video, audio, audio_to_video, video_to_audio):
        self.video_scale = video
        self.audio_scale = audio
        self.video_to_audio_scale = video_to_audio
        self.audio_to_video_scale = audio_to_video
    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.video_scale = self.video_scale
            self_module.audio_scale = self.audio_scale
            self_module.video_to_audio_scale = self.video_to_audio_scale
            self_module.audio_to_video_scale = self.audio_to_video_scale
            return ltx2_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)

class LTX2AttentionTunerPatch(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2AttentionTunerPatch",
            display_name="LTX2 Attention Tuner Patch",
            category="KJNodes/ltxv",
            description="EXPERIMENTAL! Custom LTX2 forward pass with attention scaling factors per modality, also reduces peak VRAM usage.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.String.Input("blocks", default="", tooltip="Comma separated list of transformer block indices to apply the patch to. Leave empty to apply to all blocks."),
                io.Float.Input("video_scale", default=1.0, min=0.0, max=100, step=0.01, tooltip="Scaling factor for video attention."),
                io.Float.Input("audio_scale", default=1.0, min=0.0, max=100, step=0.01, tooltip="Scaling factor for audio attention."),
                io.Float.Input("audio_to_video_scale", default=1.0, min=0.0, max=100, step=0.01, tooltip="Scaling factor for video attention."),
                io.Float.Input("video_to_audio_scale", default=1.0, min=0.0, max=100, step=0.01, tooltip="Scaling factor for audio attention."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, blocks, video_scale, audio_scale, audio_to_video_scale, video_to_audio_scale) -> io.NodeOutput:
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        # Parse selected block indices
        if blocks.strip() == "":
            selected_blocks = set(range(len(diffusion_model.transformer_blocks)))
        else:
            selected_blocks = set(int(idx) for idx in blocks.strip().split(","))

        logging.info(f"Applying LTX2 Attention Tuner Patch with custom scales to blocks: {sorted(selected_blocks)}")

        # Apply patch to all blocks, but use 1.0 scales for non-selected blocks
        for idx in range(len(diffusion_model.transformer_blocks)):
            block = diffusion_model.transformer_blocks[idx]
            if idx in selected_blocks:
                patched_forward = LTX2ForwardPatch(video_scale, audio_scale, audio_to_video_scale, video_to_audio_scale).__get__(block, block.__class__)
            else:
                patched_forward = LTX2ForwardPatch(1.0, 1.0, 1.0, 1.0).__get__(block, block.__class__)
            model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.forward", patched_forward)

        return io.NodeOutput(model_clone)

class LTX2MemoryEfficientSageAttentionPatch(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2MemoryEfficientSageAttentionPatch",
            display_name="LTX2 Mem Eff Sage Attention Patch",
            category="KJNodes/ltxv",
            description="EXPERIMENTAL! Activates custom sageattention to reduce peak VRAM usage, overrides the attention mode. Requires latest sageattention version.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        if _cuda_archs is None:
            raise RuntimeError("sageattention is not new enough version or could not determine CUDA architecture, cannot apply LTX2 Memory Efficient Sage Attention Patch.")
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        logging.info("Applying LTX2 Memory Efficient Sage Attention Patch to all transformer blocks")

        # Apply patch to all blocks, but use 1.0 scales for non-selected blocks
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn1.forward", ltx2_sageattn_forward.__get__(block.attn1, block.attn1.__class__))

        return io.NodeOutput(model_clone)


def get_cuda_version():
    try:
        version = torch.version.cuda
        if version is not None:
            major, minor = version.split('.')
            return int(major), int(minor)
        else:
            return 0, 0
    except Exception:
        return 0, 0

sageplus_sm89_available = False
_cuda_archs = None
try:
    from sageattention.core import per_thread_int8_triton, per_warp_int8_cuda, per_block_int8_triton, per_channel_fp8, get_cuda_arch_versions, attn_false
    _cuda_archs = get_cuda_arch_versions()
except:
    pass
try:
    from sageattention.core import _qattn_sm89
    cuda_version = get_cuda_version()
    sageplus_sm89_available = hasattr(_qattn_sm89, 'qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf') and cuda_version >= (12, 8)
except ImportError:
    try:
        from sageattention.core import sm89_compile as _qattn_sm89
    except ImportError:
        _qattn_sm89 = None
try:
    from sageattention.core import _qattn_sm80
except ImportError:
    try:
        from sageattention.core import sm80_compile as _qattn_sm80
    except ImportError:
        _qattn_sm80 = None
try:
    from sageattention.core import  _qattn_sm90
except ImportError:
    try:
        from sageattention.core import sm90_compile as _qattn_sm90
    except ImportError:
        _qattn_sm90 = None

from comfy.ldm.lightricks.model import apply_rotary_emb


def ltx2_sageattn_forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    dtype = x.dtype
    context = x if context is None else context

    # query
    q = self.to_q(x)
    q = self.q_norm(q)
    if pe is not None:
        q = apply_rotary_emb(q, pe)
    # key
    k = self.to_k(context)
    k = self.k_norm(k)
    if pe is not None:
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)
    # value
    v = self.to_v(context)

    # Reshape from [batch, seq_len, total_dim] to [batch, seq_len, num_heads, head_dim]
    batch_size, seq_len, _ = q.shape
    head_dim_og = self.dim_head

    q = q.view(batch_size, seq_len, self.heads, head_dim_og)
    k = k.view(batch_size, k.shape[1], self.heads, head_dim_og)
    v = v.view(batch_size, v.shape[1], self.heads, head_dim_og)

    tensor_layout="NHD"
    _tensor_layout = 0 # NHD
    _is_caual = 0
    _qk_quant_gran = 3
    _return_lse = 0
    sm_scale = head_dim_og**-0.5
    quant_v_scale_max = 448.0

    if _cuda_archs[0] in {"sm80", "sm86"}:
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km=k.mean(dim=1, keepdim=True), tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
        del q, k
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        v_fp16 = v.to(torch.float16).contiguous()
        del v
        _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v_fp16, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif _cuda_archs[0] == "sm75":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, km=k.mean(dim=1, keepdim=True), sm_scale=sm_scale, tensor_layout=tensor_layout)
        del q, k
        o, _ = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, attn_mask=None, return_lse=False)
        del v
    elif _cuda_archs[0] == "sm89":
        if not sageplus_sm89_available:
            pv_accum_dtype = "fp32+fp32"
        else:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km=k.mean(dim=1, keepdim=True), tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
        del q, k
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32+fp16":
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        elif pv_accum_dtype == "fp32+fp32":
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        del v_fp8, v_scale
    elif _cuda_archs[0] == "sm90":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km=k.mean(dim=1, keepdim=True), tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)
        del q, k,
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        del v_fp8, v_scale
    elif _cuda_archs[0] == "sm120":
        if not sageplus_sm89_available:
            pv_accum_dtype = "fp32"
        else:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        _qk_quant_gran = 2 # per warp
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km=k.mean(dim=1, keepdim=True), tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
        del q, k
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32":
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        elif pv_accum_dtype == "fp32+fp16":
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        del v_fp8, v_scale

    del q_int8, q_scale, k_int8, k_scale
    return self.to_out(o.view(batch_size, seq_len, -1))


import folder_paths

class LTX2LoraLoaderAdvanced(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2LoraLoaderAdvanced",
            display_name="LTX2 LoRA Loader Advanced",
            category="KJNodes/ltxv",
            description="Advanced LoRA loader with per-block strength control for LTX2 models",
            is_experimental=True,
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"), tooltip="The name of the LoRA."),
                io.Model.Input("model", tooltip="The diffusion model the LoRA will be applied to."),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01, tooltip="How strongly to modify the diffusion model. This value can be negative."),
                io.String.Input("opt_lora_path", optional=True, force_input=True,tooltip="Absolute path of the LoRA."),
                io.Custom("SELECTEDDITBLOCKS").Input("blocks", optional=True, tooltip="Selected DiT blocks configuration"),
                io.Float.Input("video", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Strength for video attention layers."),
                io.Float.Input("video_to_audio", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Strength for video to audio cross-attention layers."),
                io.Float.Input("audio", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Strength for audio attention layers."),
                io.Float.Input("audio_to_video", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Strength for audio to video cross-attention layers."),
                io.Float.Input("other", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Strength for layers not caught by other layer filters."),
            ],
            outputs=[
                io.Model.Output(display_name="model", tooltip="The modified diffusion model."),
                io.String.Output(display_name="rank", tooltip="Possible rank of the LoRA."),
                io.String.Output(display_name="loaded_keys_info", tooltip="List of loaded keys and their alpha values."),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model, video, video_to_audio, audio, audio_to_video, other, opt_lora_path=None, blocks=None) -> io.NodeOutput:
        from comfy.utils import load_torch_file
        import comfy.lora

        if opt_lora_path:
            lora_path = opt_lora_path
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)

        lora = load_torch_file(lora_path, safe_load=True)

        # Find the first key that ends with "weight"
        rank = "unknown"
        weight_key = next((key for key in lora.keys() if key.endswith('weight')), None)
        # Print the shape of the value corresponding to the key
        if weight_key:
            print(f"Shape of the first 'weight' key ({weight_key}): {lora[weight_key].shape}")
            rank = str(lora[weight_key].shape[0])
        else:
            print("No key ending with 'weight' found.")
            rank = "Couldn't find rank"

        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

        loaded = comfy.lora.load_lora(lora, key_map)

        keys_to_delete = []

        # First apply blocks filtering if provided
        if blocks is not None:
            for block in blocks:
                for key in list(loaded.keys()):
                    match = False
                    if isinstance(key, str) and block in key:
                        match = True
                    elif isinstance(key, tuple):
                        for k in key:
                            if block in k:
                                match = True
                                break

                    if match:
                        ratio = blocks[block]
                        if ratio == 0:
                            keys_to_delete.append(key)
                        else:
                            # Only modify LoRA adapters, skip diff tuples
                            value = loaded[key]
                            if hasattr(value, 'weights'):
                                weights_list = list(value.weights)
                                weights_list[2] = ratio
                                loaded[key].weights = tuple(weights_list)

        # Then apply layer-based attention strength filtering (takes priority)
        for key in list(loaded.keys()):
            if key in keys_to_delete:
                continue

            key_str = key if isinstance(key, str) else (key[0] if isinstance(key, tuple) else str(key))

            # Determine the strength multiplier based on layer name
            # Check more specific patterns first
            strength_multiplier = None

            # Video to audio cross-attention (check first - most specific)
            if "video_to_audio_attn" in key_str:
                strength_multiplier = video_to_audio
            # Audio to video cross-attention
            elif "audio_to_video_attn" in key_str:
                strength_multiplier = audio_to_video
            # Audio layers
            elif "audio_attn" in key_str or "audio_ff.net" in key_str:
                strength_multiplier = audio
            # Video layers (check last - most general)
            elif "attn" in key_str or "ff.net" in key_str:
                strength_multiplier = video
            # Everything else not caught by above filters
            else:
                strength_multiplier = other

            # Apply strength or mark for deletion
            if strength_multiplier is not None:
                if strength_multiplier == 0:
                    keys_to_delete.append(key)
                elif strength_multiplier != 1.0:
                    value = loaded[key]
                    if hasattr(value, 'weights'):
                        weights_list = list(value.weights)
                        # Handle case where alpha (weights[2]) might be None
                        current_alpha = weights_list[2] if weights_list[2] is not None else 1.0
                        weights_list[2] = current_alpha * strength_multiplier
                        loaded[key].weights = tuple(weights_list)

        for key in keys_to_delete:
            if key in loaded:
                del loaded[key]

        # Build list of loaded keys and their alphas
        loaded_keys_list = []
        for key, value in loaded.items():
            if hasattr(value, 'weights'):
                key_str = key if isinstance(key, str) else str(key)
                alpha = value.weights[2] if value.weights[2] is not None else "None"
                loaded_keys_list.append(f"{key_str}: alpha={alpha}")
            else:
                key_str = key if isinstance(key, str) else str(key)
                loaded_keys_list.append(f"{key_str}: type={type(value).__name__}")

        if model is not None:
            new_modelpatcher = model.clone()
            k = new_modelpatcher.add_patches(loaded, strength_model)

        # Add not loaded keys to the info
        k = set(k)
        not_loaded = []
        for x in loaded:
            if x not in k:
                key_str = x if isinstance(x, str) else str(x)
                not_loaded.append(f"NOT LOADED: {key_str}")

        if not_loaded:
            loaded_keys_list.extend(not_loaded)

        loaded_keys_info = "\n".join(loaded_keys_list)

        return io.NodeOutput(new_modelpatcher, rank, loaded_keys_info)
