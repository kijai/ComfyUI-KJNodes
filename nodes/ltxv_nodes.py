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
import comfy.ldm.modules.attention as _comfy_attn
from comfy.ldm.lightricks.model import apply_rotary_emb as _apply_rope
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
                io.Combo.Input("existing_mask_mode", options=["add", "subtract", "overwrite"], optional=True, default="add", tooltip="How to combine with existing noise masks if present. 'add' will take the max of existing and new mask, 'overwrite' will replace with new mask. 'subtract' will set the masked region to 0 instead of 1, effectively unmasking it."),
            ],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, video_fps, video_start_time, video_end_time, audio_start_time, audio_end_time, max_length="truncate", existing_mask_mode="add", video_latent=None, audio_latent=None,) -> io.NodeOutput:

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
            if "noise_mask" in video_latent and existing_mask_mode != "overwrite":
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

            video_mask[:, :, video_latent_frame_index_start:video_latent_frame_index_end] = 1.0 if existing_mask_mode != "subtract" else 0.0
            # ensure all padded frames are also masked
            if max_length == "pad" and video_samples.shape[2] > video_latent["samples"].shape[2]:
                video_mask[:, :, video_latent["samples"].shape[2]:] = 1.0 if existing_mask_mode != "subtract" else 0.0
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

def _compute_attention(self, query, context, attn_precision=None, transformer_options={}):
    """Compute attention and return the result. Cleans up intermediate tensors."""
    k = self.k_norm(self.to_k(context)).to(query.dtype)
    v = self.to_v(context).to(query.dtype)
    x = comfy.ldm.modules.attention.optimized_attention(query, k, v, heads=self.heads, attn_precision=attn_precision, transformer_options=transformer_options).flatten(2)
    del k, v
    return x

def nag_attention(self, query, context_positive, nag_context, attn_precision=None, transformer_options={}):
    x_positive = _compute_attention(self, query, context_positive, attn_precision, transformer_options)
    x_negative = _compute_attention(self, query, nag_context, attn_precision, transformer_options)
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

    x_positive, x_negative = nag_attention(self, q_pos, context_pos, self.nag_context, attn_precision=self.attn_precision, transformer_options=transformer_options)
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
            vid_split = getattr(diffusion_model, "cross_attention_dim", None)
            if vid_split is not None and context_video.shape[-1] == vid_split + diffusion_model.audio_cross_attention_dim:
                context_video = context_video[:, :, :vid_split]
            if diffusion_model.caption_proj_before_connector and diffusion_model.caption_projection_first_linear:
                diffusion_model.caption_projection.to(device)
                context_video = diffusion_model.caption_projection(context_video)
                diffusion_model.caption_projection.to(offload_device)
            if hasattr(diffusion_model, "video_embeddings_connector"):
                diffusion_model.video_embeddings_connector.to(device)
                context_video = diffusion_model.video_embeddings_connector(context_video)[0]
                diffusion_model.video_embeddings_connector.to(offload_device)
            context_video = context_video.view(1, -1, img_dim)
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                patched_attn2 = LTXVCrossAttentionPatch(context_video, nag_scale, nag_alpha, nag_tau, inplace=inplace).__get__(block.attn2, block.__class__)
                model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn2.forward", patched_attn2)

        if nag_cond_audio is not None and diffusion_model.audio_caption_projection is not None:
            context_audio = nag_cond_audio[0][0].to(device, dtype)
            vid_split = getattr(diffusion_model, "cross_attention_dim", None)
            if vid_split is not None and context_audio.shape[-1] == vid_split + diffusion_model.audio_cross_attention_dim:
                context_audio = context_audio[:, :, vid_split:]
            if diffusion_model.caption_proj_before_connector and diffusion_model.caption_projection_first_linear:
                diffusion_model.audio_caption_projection.to(device)
                context_audio = diffusion_model.audio_caption_projection(context_audio)
                diffusion_model.audio_caption_projection.to(offload_device)
            if hasattr(diffusion_model, "audio_embeddings_connector"):
                diffusion_model.audio_embeddings_connector.to(device)
                context_audio = diffusion_model.audio_embeddings_connector(context_audio)[0]
                diffusion_model.audio_embeddings_connector.to(offload_device)
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

            #low = latent_image.quantile(0.01)
            #high = latent_image.quantile(0.99)
            #latent_image = ((latent_image - low) / (high - low)).clamp(0, 1)
            latent_image = torch.sigmoid(latent_image)
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
        latent_rgb_factors = [[0.002269406570121646, -0.02110900916159153, -0.009850316680967808], [-0.016038373112678528, -0.012462412007153034, -0.01112896017730236], [0.025274179875850677, 0.011209743097424507, 0.025426799431443214], [0.04690725728869438, 0.041542328894138336, 0.03568895906209946], [-0.02388044260442257, -0.0018645941745489836, 0.01858334057033062], [0.03720448538661003, 0.0220357533544302, 0.027937663719058037], [-0.07273884862661362, -0.09326262027025223, -0.11579664051532745], [-0.063837431371212, 0.00026216846890747547, 0.03042735904455185], [0.02903873845934868, 0.042082373052835464, 0.030649805441498756], [0.03777873516082764, 0.0322984978556633, -0.005671461578458548], [-0.0075670829974114895, -0.012113905511796474, -0.01638956367969513], [0.026524530723690987, 0.060518112033605576, 0.059549521654844284], [0.10093028098344803, 0.10073262453079224, 0.0505094900727272], [0.03725508227944374, 0.015382086858153343, 0.005786076188087463], [-0.03139607608318329, -0.01690264232456684, -0.0013519978383556008], [-0.027200624346733093, -0.02517341822385788, -0.008874989114701748], [0.024963486939668655, 0.04293748363852501, 0.05582639202475548], [-0.0364827960729599, -0.026975594460964203, -0.021950015798211098], [0.027655167505145073, 0.025136707350611687, 0.043967027217149734], [0.035822272300720215, 0.013104500249028206, 0.01113432738929987], [0.05353763327002525, 0.013606574386358261, -0.018720127642154694], [-0.013587888330221176, -0.01689346879720688, -0.027842802926898003], [0.059415675699710846, 0.03734271228313446, 0.04562298208475113], [-0.02946414425969124, -0.038338612765073776, 0.001805233070626855], [0.03921474143862724, 0.0651894062757492, 0.10681862384080887], [-0.00744189927354455, 0.007951526902616024, 0.020728807896375656], [-0.04038553684949875, -0.05215264856815338, -0.07213657349348068], [-0.004655141849070787, 0.01305423304438591, 0.026104029268026352], [0.03434251993894577, 0.018448110669851303, 0.013096392154693604], [0.0022075253073126078, -0.0011812079465016723, 0.0002940484555438161], [-0.00043441299931146204, 0.02366728149354458, 0.035889431834220886], [-0.030657343566417694, -0.024926183745265007, -0.012355240061879158], [-0.018955843523144722, -0.017360301688313484, -0.008214764297008514], [-0.01113052573055029, -0.01201171800494194, -0.002986249281093478], [0.018902746960520744, 0.01758778840303421, 0.026414571329951286], [-0.019977254793047905, -0.01605399139225483, -0.019136475399136543], [-0.00300968368537724, -0.017609693109989166, -0.013655650429427624], [0.0022096361499279737, 0.017998533323407173, 0.01815750263631344], [0.05186990648508072, 0.03285299986600876, 0.016072165220975876], [0.012626334093511105, 0.0013884707586839795, -0.012077193707227707], [-0.0037861645687371492, -0.013902144506573677, -0.01911942847073078], [-0.014163163490593433, -0.00513274222612381, -0.014303527772426605], [-0.010461323894560337, 0.009658926166594028, 0.01644069515168667], [-0.008665377274155617, 0.002501955023035407, -0.009703717194497585], [-0.03404829278588295, -0.02546044997870922, -0.014914450235664845], [0.04997691139578819, 0.06592527031898499, 0.073111392557621], [0.027394814416766167, 0.024555068463087082, 0.019957970827817917], [-0.027501430362462997, -0.01673700101673603, -0.03089248389005661], [-0.018696032464504242, -0.0020940247923135757, 0.015244065783917904], [-0.0062704551964998245, -0.0067006442695856094, -0.007532030809670687], [0.014871004968881607, 0.009914354421198368, 0.020960720255970955], [0.03662937879562378, 0.04413224756717682, 0.04220828413963318], [-0.011242181062698364, -0.013539309613406658, -0.016438307240605354], [-0.014854325912892818, 0.0038217694964259863, -0.002461288822814822], [-0.014826249331235886, 0.0009719038498587906, -0.012078499421477318], [-0.029396841302514076, -0.01432017982006073, 0.013018904253840446], [0.02755064144730568, 0.028369395062327385, 0.01640605367720127], [0.12049165368080139, 0.1395745575428009, 0.14566579461097717], [0.019721267744898796, 0.009739740751683712, 0.0023876908235251904], [-0.007320966571569443, 0.0065013207495212555, 0.01603059470653534], [0.007391378283500671, -0.0073603675700724125, -0.01770283281803131], [0.02984853833913803, 0.012391146272420883, 0.010563627816736698], [-0.013479884713888168, -0.008637298829853535, -0.013457189314067364], [0.04127075523138046, 0.03032625839114189, 0.024770958349108696], [-0.06524652987718582, -0.012209279462695122, 0.02087211236357689], [-0.1179763451218605, -0.060323599725961685, -0.07592175155878067], [-0.07122819870710373, -0.04385707899928093, -0.022124603390693665], [-0.04682473465800285, -0.022610662505030632, -0.010107148438692093], [-0.0054328180849552155, -0.010368981398642063, -0.008167334832251072], [0.029181398451328278, 0.030588403344154358, 0.028090540319681168], [0.016619984060525894, 0.004931286443024874, -0.006450849585235119], [0.01035264041274786, 0.002237115055322647, 0.0013903985964134336], [-0.04313831403851509, -0.061772625893354416, -0.08946335315704346], [0.0150345079600811, 0.007781678810715675, 0.0011013159528374672], [-0.013585779815912247, 0.008117705583572388, 0.020367907360196114], [-0.172962948679924, -0.16406646370887756, -0.1668281853199005], [0.0083833709359169, 0.0015236001927405596, -0.01731627807021141], [0.021939430385828018, 0.018004458397626877, 0.014768349006772041], [0.008083095774054527, -0.013463049195706844, -0.022061636671423912], [0.024328550323843956, 0.0128010343760252, 0.014966367743909359], [0.05850301682949066, 0.027980001643300056, 0.02225641906261444], [0.09690416604280472, 0.06929530203342438, 0.03253814950585365], [0.048208240419626236, 0.025294817984104156, 0.023508133366703987], [-0.026432134211063385, -0.040383171290159225, -0.03950457274913788], [-0.021598653867840767, -0.017070941627025604, -0.010933087207376957], [0.011645167134702206, 0.002806191798299551, 0.003779367310926318], [0.10478592664003372, 0.08954174816608429, 0.06555330753326416], [0.015151776373386383, -0.016160616651177406, -0.024905217811465263], [0.019659176468849182, 0.008487952873110771, 0.002426224760711193], [-0.05173315480351448, -0.026337839663028717, -0.02127116546034813], [0.016987523064017296, 0.006270893849432468, 0.0015798212261870503], [0.007938026450574398, -0.005250005517154932, -0.020408453419804573], [0.013017759658396244, 0.01654384844005108, 0.04163840040564537], [-0.009886542335152626, -0.026848411187529564, -0.03070281818509102], [0.01108171883970499, 0.01827266439795494, -0.007332107983529568], [-0.0285995751619339, -0.031727731227874756, -0.03370537981390953], [0.005299570970237255, 0.05678633600473404, 0.02825017087161541], [-0.055322226136922836, -0.09084303677082062, -0.12999044358730316], [0.01844066195189953, 0.031044499948620796, 0.021148500964045525], [-0.004471115302294493, 0.005830412730574608, 0.00911418441683054], [-0.04053843766450882, -0.016424428671598434, -0.0010634599020704627], [0.03858831524848938, 0.007309338077902794, -0.005618985276669264], [0.01423253770917654, -0.0055681923404335976, 3.394074519746937e-05], [0.11455483734607697, 0.14653916656970978, 0.1488018035888672], [-0.005231931805610657, -0.0033921014983206987, -0.000995257287286222], [0.01449565589427948, 0.019586293026804924, 0.04565812274813652], [-0.005179048050194979, -0.011201606132090092, -0.0008710073889233172], [-0.015361929312348366, 0.00778581015765667, -0.008238887414336205], [-0.1147838830947876, -0.09109023958444595, -0.050579313188791275], [0.09037500619888306, 0.09597006440162659, 0.10811734944581985], [0.001873677596449852, -0.01772197335958481, -0.07681205868721008], [-0.020383257418870926, -0.016072455793619156, -0.01077069528400898], [-0.060444317758083344, -0.05499502643942833, -0.06153025105595589], [-0.016717270016670227, 0.026493264362215996, 0.021835654973983765], [0.008203534409403801, 0.00418612826615572, 0.013867748901247978], [0.0789225772023201, 0.05467747151851654, 0.016568133607506752], [-0.15149451792240143, -0.1526806503534317, -0.14325062930583954], [0.00538366474211216, 0.010192245244979858, -0.00449327751994133], [-0.004906965419650078, -0.005569908302277327, -0.02096559666097164], [0.024530155584216118, 0.010962833650410175, 0.0034586559049785137], [0.03551010414958, 0.017310436815023422, 0.007064413744956255], [0.11111932247877121, 0.09825586527585983, 0.08827318251132965], [-0.051722846925258636, -0.047595202922821045, -0.03763044252991676], [-0.02975175902247429, -0.02153967320919037, -0.021425534039735794], [-0.03462936729192734, -0.025198571383953094, -0.017322326079010963], [-0.016921017318964005, -0.012419789098203182, -0.0154880927875638], [-0.08035065978765488, -0.08451078832149506, -0.09623870998620987], [-0.03870908170938492, -0.04211008921265602, -0.04383759945631027]]
        latent_rgb_factors_bias = [-0.6957847476005554, -0.7276281118392944, -0.7405748963356018]

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

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _rms_norm_scale_shift_kernel(
        X_ptr, Out_ptr, Scale_ptr, Shift_ptr,
        N, scale_n_rows,   # scale_n_rows: actual rows in scale/shift (handles broadcast)
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_BF16: tl.constexpr,
        HAS_BROADCAST: tl.constexpr,  # True when scale_n_rows < n_rows
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        off = row * N + cols

        x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)

        mean_sq = tl.sum(x * x, axis=0) / N
        rrms = tl.math.rsqrt(mean_sq + eps)
        x_norm = x * rrms

        # Fast path: no broadcast, scale row == x row
        if HAS_BROADCAST:
            scale_row = row % scale_n_rows
            scale_off = scale_row * N + cols
        else:
            scale_off = off
        scale = tl.load(Scale_ptr + scale_off, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(Shift_ptr + scale_off, mask=mask, other=0.0).to(tl.float32)
        out = x_norm * (1.0 + scale) + shift

        if IS_BF16:
            tl.store(Out_ptr + off, out.to(tl.bfloat16), mask=mask)
        else:
            tl.store(Out_ptr + off, out.to(tl.float16), mask=mask)

    @triton.jit
    def _rope_qk_split_kernel(
        Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
        H, T, D,          # D = dim_head (full, even)
        TH,               # T * H, precomputed to avoid runtime multiply in index decomp
        # q/k layout: [B, T, H*D] contiguous → q[b,t,h*D + d]
        # cos/sin layout: [B, H, T, D//2] contiguous
        IS_BF16: tl.constexpr,
        BLOCK_HD: tl.constexpr,   # >= D//2
    ):
        # grid: (B*H*T,)
        bht   = tl.program_id(0)
        t     = bht % T
        h     = (bht // T) % H
        b     = bht // TH

        D_half = D // 2
        cols   = tl.arange(0, BLOCK_HD)
        mask   = cols < D_half

        # offset into q/k tensor: [B, T, H*D]
        qk_base  = (b * T + t) * (H * D) + h * D
        # offset into cos/sin: [B, H, T, D//2]
        cs_base  = (b * H * T + h * T + t) * D_half

        # load both halves of q and k
        q_x = tl.load(Q_ptr + qk_base          + cols, mask=mask, other=0.0).to(tl.float32)
        q_y = tl.load(Q_ptr + qk_base + D_half  + cols, mask=mask, other=0.0).to(tl.float32)
        k_x = tl.load(K_ptr + qk_base          + cols, mask=mask, other=0.0).to(tl.float32)
        k_y = tl.load(K_ptr + qk_base + D_half  + cols, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(Cos_ptr + cs_base         + cols, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(Sin_ptr + cs_base         + cols, mask=mask, other=0.0).to(tl.float32)

        # split RoPE: out_x = x*cos - y*sin,  out_y = y*cos + x*sin
        q_ox = q_x * cos - q_y * sin
        q_oy = q_y * cos + q_x * sin
        k_ox = k_x * cos - k_y * sin
        k_oy = k_y * cos + k_x * sin

        if IS_BF16:
            tl.store(Q_ptr + qk_base          + cols, q_ox.to(tl.bfloat16), mask=mask)
            tl.store(Q_ptr + qk_base + D_half  + cols, q_oy.to(tl.bfloat16), mask=mask)
            tl.store(K_ptr + qk_base          + cols, k_ox.to(tl.bfloat16), mask=mask)
            tl.store(K_ptr + qk_base + D_half  + cols, k_oy.to(tl.bfloat16), mask=mask)
        else:
            tl.store(Q_ptr + qk_base          + cols, q_ox.to(tl.float16), mask=mask)
            tl.store(Q_ptr + qk_base + D_half  + cols, q_oy.to(tl.float16), mask=mask)
            tl.store(K_ptr + qk_base          + cols, k_ox.to(tl.float16), mask=mask)
            tl.store(K_ptr + qk_base + D_half  + cols, k_oy.to(tl.float16), mask=mask)

    @triton.jit
    def _rms_norm_dual_scale_shift_kernel(
        X_ptr, Out1_ptr, Out2_ptr,
        Scale1_ptr, Shift1_ptr, Scale2_ptr, Shift2_ptr,
        N, scale_n_rows,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_BF16: tl.constexpr,
        HAS_BROADCAST: tl.constexpr,
    ):
        """Compute rms_norm(x) once, write two differently-scaled outputs."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        off = row * N + cols

        x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / N
        x_norm = x * tl.math.rsqrt(mean_sq + eps)

        if HAS_BROADCAST:
            scale_row = row % scale_n_rows
            s_off = scale_row * N + cols
        else:
            s_off = off

        scale1 = tl.load(Scale1_ptr + s_off, mask=mask, other=0.0).to(tl.float32)
        shift1 = tl.load(Shift1_ptr + s_off, mask=mask, other=0.0).to(tl.float32)
        scale2 = tl.load(Scale2_ptr + s_off, mask=mask, other=0.0).to(tl.float32)
        shift2 = tl.load(Shift2_ptr + s_off, mask=mask, other=0.0).to(tl.float32)

        out1 = x_norm * (1.0 + scale1) + shift1
        out2 = x_norm * (1.0 + scale2) + shift2

        if IS_BF16:
            tl.store(Out1_ptr + off, out1.to(tl.bfloat16), mask=mask)
            tl.store(Out2_ptr + off, out2.to(tl.bfloat16), mask=mask)
        else:
            tl.store(Out1_ptr + off, out1.to(tl.float16), mask=mask)
            tl.store(Out2_ptr + off, out2.to(tl.float16), mask=mask)


def fused_norm_scale_shift(x, scale, shift, eps=1e-6, use_triton=True):
    if use_triton and HAS_TRITON and x.is_cuda:
        orig_shape = x.shape
        hidden = x.shape[-1]
        x_2d = x.contiguous().reshape(-1, hidden)
        n_rows = x_2d.shape[0]
        # Flatten scale/shift without expanding — kernel handles broadcast via modulo
        scale_2d = scale.contiguous().reshape(-1, hidden)
        shift_2d = shift.contiguous().reshape(-1, hidden)
        scale_n_rows = scale_2d.shape[0]
        out = torch.empty_like(x_2d)
        BLOCK_N = triton.next_power_of_2(hidden)
        num_warps = min(max(BLOCK_N // 256, 1), 16)
        _rms_norm_scale_shift_kernel[(n_rows,)](
            x_2d, out, scale_2d, shift_2d,
            hidden, scale_n_rows,
            eps=eps, BLOCK_N=BLOCK_N,
            IS_BF16=(x.dtype == torch.bfloat16),
            HAS_BROADCAST=(scale_n_rows < n_rows),
            num_warps=num_warps,
        )
        return out.view(orig_shape)
    else:
        return comfy.ldm.common_dit.rms_norm(x, eps=eps) * (1 + scale) + shift


def fused_rope_qk(q, k, freqs_cis, use_triton=True):
    """Apply split RoPE to q and k in one fused kernel pass.
    q, k: [B, T, H*D] contiguous
    freqs_cis: (cos, sin, split_pe) where cos/sin: [B, H, T, D//2]
    Falls back to original apply_rotary_emb if preconditions not met.
    """

    if not (use_triton and HAS_TRITON and q.is_cuda):
        return _apply_rope(q, freqs_cis), _apply_rope(k, freqs_cis)

    cos, sin = freqs_cis[0], freqs_cis[1]
    split_pe = freqs_cis[2] if len(freqs_cis) > 2 else False

    if not split_pe or cos.ndim != 4 or q.ndim != 3:
        return _apply_rope(q, freqs_cis), _apply_rope(k, freqs_cis)

    B_cos, H, T_cos, D_half = cos.shape
    D = D_half * 2
    if q.shape != (B_cos, T_cos, H * D) or k.shape != (B_cos, T_cos, H * D):
        return _apply_rope(q, freqs_cis), _apply_rope(k, freqs_cis)

    q = q.contiguous()
    k = k.contiguous()
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    BLOCK_HD = triton.next_power_of_2(D_half)
    num_warps = min(max(BLOCK_HD // 32, 1), 8)
    _rope_qk_split_kernel[(B_cos * H * T_cos,)](
        q, k, cos_c, sin_c,
        H, T_cos, D,
        T_cos * H,
        IS_BF16=(q.dtype == torch.bfloat16),
        BLOCK_HD=BLOCK_HD,
        num_warps=num_warps,
    )
    return q, k


def fused_norm_dual_scale_shift(x, scale1, shift1, scale2, shift2, eps=1e-6, use_triton=True):
    """RMS-norm x once, return two scaled outputs: (x_norm*(1+s1)+b1, x_norm*(1+s2)+b2)."""
    if use_triton and HAS_TRITON and x.is_cuda:
        orig_shape = x.shape
        hidden = x.shape[-1]
        x_2d     = x.contiguous().reshape(-1, hidden)
        n_rows   = x_2d.shape[0]
        scale1_2d = scale1.contiguous().reshape(-1, hidden)
        shift1_2d = shift1.contiguous().reshape(-1, hidden)
        scale2_2d = scale2.contiguous().reshape(-1, hidden)
        shift2_2d = shift2.contiguous().reshape(-1, hidden)
        scale_n_rows = scale1_2d.shape[0]
        out1 = torch.empty_like(x_2d)
        out2 = torch.empty_like(x_2d)
        BLOCK_N   = triton.next_power_of_2(hidden)
        num_warps = min(max(BLOCK_N // 256, 1), 16)
        _rms_norm_dual_scale_shift_kernel[(n_rows,)](
            x_2d, out1, out2,
            scale1_2d, shift1_2d, scale2_2d, shift2_2d,
            hidden, scale_n_rows,
            eps=eps, BLOCK_N=BLOCK_N,
            IS_BF16=(x.dtype == torch.bfloat16),
            HAS_BROADCAST=(scale_n_rows < n_rows),
            num_warps=num_warps,
        )
        return out1.view(orig_shape), out2.view(orig_shape)
    else:
        x_norm = comfy.ldm.common_dit.rms_norm(x, eps=eps)
        return x_norm * (1 + scale1) + shift1, x_norm * (1 + scale2) + shift2


def _apply_text_cross_attention_patched(
    self, x, context, attn, scale_shift_table, prompt_scale_shift_table,
    timestep, prompt_timestep, attention_mask, transformer_options,
):
    """Drop-in replacement for _apply_text_cross_attention with fused norm+scale+shift.
    Patched onto the block instance so self._apply_text_cross_attention resolves here."""
    if self.cross_attention_adaln:
        shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
        batch_size = x.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        attn_input = fused_norm_scale_shift(x, scale_q, shift_q,
                                            use_triton=getattr(self, 'use_triton_kernels', True))
        del shift_q, scale_q
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
        del scale_kv, shift_kv
        return attn(attn_input, context=encoder_hidden_states,
                    mask=attention_mask, transformer_options=transformer_options) * gate
    return attn(
        comfy.ldm.common_dit.rms_norm(x), context=context,
        mask=attention_mask, transformer_options=transformer_options,
    )


def ltx2_forward(
        self, x: Tuple[torch.Tensor, torch.Tensor], v_context=None, a_context=None, attention_mask=None, v_timestep=None, a_timestep=None,
        v_pe=None, a_pe=None, v_cross_pe=None, a_cross_pe=None, v_cross_scale_shift_timestep=None, a_cross_scale_shift_timestep=None,
        v_cross_gate_timestep=None, a_cross_gate_timestep=None, transformer_options=None, self_attention_mask=None,
        v_prompt_timestep=None, a_prompt_timestep=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        run_vx = transformer_options.get("run_vx", True)
        run_ax = transformer_options.get("run_ax", True)
        use_triton = getattr(self, 'use_triton_kernels', True)
        video_scale = getattr(self, 'video_scale', 1.0)
        audio_scale = getattr(self, 'audio_scale', 1.0)
        audio_to_video_scale = getattr(self, 'audio_to_video_scale', 1.0)
        video_to_audio_scale = getattr(self, 'video_to_audio_scale', 1.0)

        vx, ax = x
        run_ax = run_ax and ax.numel() > 0 and audio_scale != 0.0
        run_a2v = run_vx and transformer_options.get("a2v_cross_attn", True) and ax.numel() > 0 and audio_to_video_scale != 0.0
        run_v2a = run_ax and transformer_options.get("v2a_cross_attn", True) and video_to_audio_scale != 0.0

        if run_vx:
            # video self-attention
            vshift_msa, vscale_msa = (self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(0, 2)))
            norm_vx = fused_norm_scale_shift(vx, vscale_msa, vshift_msa, use_triton=use_triton)
            del vshift_msa, vscale_msa
            # inline attn1 only when triton is active (needed to intercept q/k for fused RoPE)
            if use_triton:
                _a1 = self.attn1
                q = _a1.to_q(norm_vx)
                k = _a1.to_k(norm_vx)
                v = _a1.to_v(norm_vx)

                q = _a1.q_norm(q)
                k = _a1.k_norm(k)

                if v_pe is not None:
                    q, k = fused_rope_qk(q, k, v_pe, use_triton=True)

                if self_attention_mask is None:
                    _sa_out = _comfy_attn.optimized_attention(q, k, v, _a1.heads, attn_precision=_a1.attn_precision, transformer_options=transformer_options)
                else:
                    _sa_out = _comfy_attn.optimized_attention_masked(q, k, v, _a1.heads, self_attention_mask, attn_precision=_a1.attn_precision, transformer_options=transformer_options)
                del q, k, v
                if _a1.to_gate_logits is not None:
                    _gate = _a1.to_gate_logits(norm_vx)
                    _b, _t, _ = _sa_out.shape
                    _sa_out = _sa_out.view(_b, _t, _a1.heads, _a1.dim_head)
                    _sa_out = (_sa_out * (2.0 * torch.sigmoid(_gate)).unsqueeze(-1)).view(_b, _t, _a1.heads * _a1.dim_head)
                    del _gate

                attn1_out = _a1.to_out(_sa_out)
                del _sa_out, norm_vx
            else:
                attn1_out = self.attn1(norm_vx, pe=v_pe, transformer_options=transformer_options)
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
            norm_ax = fused_norm_scale_shift(ax, ascale_msa, ashift_msa, use_triton=use_triton)
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
            if run_a2v and run_v2a:
                # Fetch all 4 scale/shift values per table at once (avoids two get_ada_values calls each)
                # and compute rms_norm once per tensor via dual kernel.
                v_ca_vals = self.get_ada_values(self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)
                a_ca_vals = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)
                vx_a2v, vx_v2a = fused_norm_dual_scale_shift(vx, v_ca_vals[0], v_ca_vals[1], v_ca_vals[2], v_ca_vals[3], use_triton=use_triton)
                ax_a2v, ax_v2a = fused_norm_dual_scale_shift(ax, a_ca_vals[0], a_ca_vals[1], a_ca_vals[2], a_ca_vals[3], use_triton=use_triton)
                del v_ca_vals, a_ca_vals
            elif run_a2v:
                sv, bv = self.get_ada_values(self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[:2]
                sa, ba = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[:2]
                vx_a2v = fused_norm_scale_shift(vx, sv, bv, use_triton=use_triton)
                ax_a2v = fused_norm_scale_shift(ax, sa, ba, use_triton=use_triton)
                del sv, bv, sa, ba
            else:  # only v2a
                sv, bv = self.get_ada_values(self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[2:4]
                sa, ba = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[2:4]
                vx_v2a = fused_norm_scale_shift(vx, sv, bv, use_triton=use_triton)
                ax_v2a = fused_norm_scale_shift(ax, sa, ba, use_triton=use_triton)
                del sv, bv, sa, ba

            # audio to video cross attention
            if run_a2v:
                a2v_out = self.audio_to_video_attn(vx_a2v, context=ax_a2v, pe=v_cross_pe, k_pe=a_cross_pe, transformer_options=transformer_options)
                del vx_a2v, ax_a2v
                gate_out_a2v = self.get_ada_values(self.scale_shift_table_a2v_ca_video[4:, :], vx.shape[0], v_cross_gate_timestep)[0]
                vx.addcmul_(a2v_out, gate_out_a2v, value=audio_to_video_scale)
                del gate_out_a2v, a2v_out

            # video to audio cross attention
            if run_v2a:
                v2a_out = self.video_to_audio_attn(ax_v2a, context=vx_v2a, pe=a_cross_pe, k_pe=v_cross_pe, transformer_options=transformer_options)
                del ax_v2a, vx_v2a
                gate_out_v2a = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[4:, :], ax.shape[0], a_cross_gate_timestep)[0]
                ax.addcmul_(v2a_out, gate_out_v2a, value=video_to_audio_scale)
                del gate_out_v2a, v2a_out

        # video feedforward
        if run_vx:
            vshift_mlp, vscale_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(3, 5))
            vx_scaled = fused_norm_scale_shift(vx, vscale_mlp, vshift_mlp, use_triton=use_triton)
            del vshift_mlp, vscale_mlp

            ff_out = self.ff(vx_scaled)
            del vx_scaled

            vgate_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(5, 6))[0]
            vx.addcmul_(ff_out, vgate_mlp, value=video_scale)
            del vgate_mlp, ff_out

        # audio feedforward
        if run_ax:
            ashift_mlp, ascale_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(3, 5))
            ax_scaled = fused_norm_scale_shift(ax, ascale_mlp, ashift_mlp, use_triton=use_triton)
            del ashift_mlp, ascale_mlp

            ff_out = self.audio_ff(ax_scaled)
            del ax_scaled

            agate_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(5, 6))[0]
            ax.addcmul_(ff_out, agate_mlp, value=audio_scale)
            del agate_mlp, ff_out

        return vx, ax

class LTX2ForwardPatch:
    def __init__(self, video, audio, audio_to_video, video_to_audio, use_triton=True):
        self.video_scale = video
        self.audio_scale = audio
        self.video_to_audio_scale = video_to_audio
        self.audio_to_video_scale = audio_to_video
        self.use_triton_kernels = use_triton
    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.video_scale = self.video_scale
            self_module.audio_scale = self.audio_scale
            self_module.video_to_audio_scale = self.video_to_audio_scale
            self_module.audio_to_video_scale = self.audio_to_video_scale
            self_module.use_triton_kernels = self.use_triton_kernels
            self_module._apply_text_cross_attention = types.MethodType(
                _apply_text_cross_attention_patched, self_module)
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
                io.Boolean.Input("triton_kernels", default=True, tooltip="Use Triton fused kernels for norm+scale+shift and rope application operations, can be very slightly faster."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, blocks, video_scale, audio_scale, audio_to_video_scale, video_to_audio_scale, triton_kernels) -> io.NodeOutput:
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        # Parse selected block indices
        if blocks.strip() == "":
            selected_blocks = set(range(len(diffusion_model.transformer_blocks)))
        else:
            selected_blocks = set(int(idx) for idx in blocks.strip().split(","))

        logging.info(f"Applying LTX2 Attention Tuner Patch with custom scales to blocks: {sorted(selected_blocks)}, triton_kernels={triton_kernels}")

        # Apply patch to all blocks, but use 1.0 scales for non-selected blocks
        for idx in range(len(diffusion_model.transformer_blocks)):
            block = diffusion_model.transformer_blocks[idx]
            if idx in selected_blocks:
                patched_forward = LTX2ForwardPatch(video_scale, audio_scale, audio_to_video_scale, video_to_audio_scale, use_triton=triton_kernels).__get__(block, block.__class__)
            else:
                patched_forward = LTX2ForwardPatch(1.0, 1.0, 1.0, 1.0, use_triton=triton_kernels).__get__(block, block.__class__)
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
                io.Boolean.Input("triton_kernels", default=True, tooltip="Use Triton fused RoPE kernel on the self-attention Q/K. Requires Triton."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, triton_kernels) -> io.NodeOutput:
        if _cuda_archs is None:
            raise RuntimeError("sageattention is not new enough version or could not determine CUDA architecture, cannot apply LTX2 Memory Efficient Sage Attention Patch.")
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        logging.info(f"Applying LTX2 Memory Efficient Sage Attention Patch to all transformer blocks, triton_kernels={triton_kernels}")

        for idx, block in enumerate(diffusion_model.transformer_blocks):
            block.attn1.use_triton_kernels = triton_kernels
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
    # key
    k = self.to_k(context)
    k = self.k_norm(k)
    # apply RoPE — fuse q+k into one kernel when both share the same pe
    if pe is not None:
        use_triton = getattr(self, 'use_triton_kernels', False)
        if k_pe is None:
            q, k = fused_rope_qk(q, k, pe, use_triton=use_triton)
        else:
            q = apply_rotary_emb(q, pe)
            k = apply_rotary_emb(k, k_pe)
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

    o = o.view(batch_size, seq_len, -1)

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)  # (B, T, H)
        b, t, _ = o.shape
        o = o.view(b, t, self.heads, self.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)  # zero-init -> identity
        o = o * gates.unsqueeze(-1)
        o = o.view(b, t, self.heads * self.dim_head)

    return self.to_out(o)


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
