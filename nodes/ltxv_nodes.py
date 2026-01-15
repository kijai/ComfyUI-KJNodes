from comfy_extras.nodes_lt import get_noise_mask, LTXVAddGuide
import types
import comfy
from comfy_api.latest import io
import numpy as np
import torch
import comfy.model_management as mm

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
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
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
                io.Latent.Output(display_name="latent"),
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
                    options=["truncate", "pad"],
                    default="truncate",
                    tooltip="Determines how to handle cases where the specified end time exceeds the latent length.",
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

            # Pad video latent if required frames exceed current length
            if max_length == "pad" and required_latent_frames > video_latent_frame_count:
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
            else:
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

            video_mask = torch.zeros_like(video_samples)
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

            # Pad audio latent if end index exceeds current length
            if max_length == "pad" and audio_latent_frame_index_end > audio_latent_frame_count:
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
            else:
                audio_samples = audio_latent["samples"]

            audio_latent_frame_index_start = max(0, audio_latent_frame_index_start)
            audio_latent_frame_index_end = min(audio_latent_frame_index_end, audio_latent_frame_count)

            audio_mask = torch.zeros_like(audio_samples)
            audio_mask[:, :, audio_latent_frame_index_start:audio_latent_frame_index_end] = 1.0
            # ensure all padded frames are also masked
            if max_length == "pad" and audio_samples.shape[2] > audio_latent["samples"].shape[2]:
                audio_mask[:, :, audio_latent["samples"].shape[2]:] = 1.0
            audio_latent = audio_latent.copy()
            audio_latent["samples"] = audio_samples
            audio_latent["noise_mask"] = audio_mask

        return io.NodeOutput(video_latent, audio_latent)

def normalized_attention_guidance(self, query, context_positive, nag_context, transformer_options={}):
    k_positive = self.k_norm(self.to_k(context_positive)).to(query.dtype)
    v_positive = self.to_v(context_positive).to(query.dtype)
    k_negative = self.k_norm(self.to_k(nag_context)).to(query.dtype)
    v_negative = self.to_v(nag_context).to(query.dtype)

    x_positive = comfy.ldm.modules.attention.optimized_attention(query, k_positive, v_positive, heads=self.heads, transformer_options=transformer_options).flatten(2)
    x_negative = comfy.ldm.modules.attention.optimized_attention(query, k_negative, v_negative, heads=self.heads, transformer_options=transformer_options).flatten(2)

    nag_guidance = x_positive * self.nag_scale - x_negative * (self.nag_scale - 1)

    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True).expand_as(x_positive)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True).expand_as(nag_guidance)

    scale = torch.nan_to_num(norm_guidance / norm_positive, nan=10.0)

    mask = scale > self.nag_tau
    adjustment = (norm_positive * self.nag_tau) / (norm_guidance + 1e-7)
    nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)

    x = nag_guidance * self.nag_alpha + x_positive * (1 - self.nag_alpha)
    del nag_guidance

    return x

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
    nag_context = self.nag_context

    x_pos_out = normalized_attention_guidance(self, q_pos, context_pos, nag_context, transformer_options=transformer_options)

    # Negative
    if x_neg is not None and context_neg is not None:
        q_neg = self.q_norm(self.to_q(x_neg))
        k_neg = self.k_norm(self.to_k(context_neg))
        v_neg = self.to_v(context_neg)

        x_neg_out = comfy.ldm.modules.attention.optimized_attention(q_neg, k_neg, v_neg, heads=self.heads, transformer_options=transformer_options)
        x = torch.cat([x_pos_out, x_neg_out], dim=0)
    else:
        x = x_pos_out

    return self.to_out(x)


class LTXVCrossAttentionPatch:
    def __init__(self, context, nag_scale, nag_alpha, nag_tau):
        self.nag_context = context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau

    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.nag_context = self.nag_context
            self_module.nag_scale = self.nag_scale
            self_module.nag_alpha = self.nag_alpha
            self_module.nag_tau = self.nag_tau

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
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, nag_scale, nag_alpha, nag_tau, nag_cond_video=None, nag_cond_audio=None) -> io.NodeOutput:
        if nag_scale == 0:
            return io.NodeOutput(model)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = model.model.diffusion_model.dtype

        model_clone = model.clone()

        diffusion_model = model_clone.get_model_object("diffusion_model")
        img_dim = diffusion_model.inner_dim
        audio_dim = diffusion_model.audio_inner_dim

        context_video = context_audio = None

        if nag_cond_video is not None:
            diffusion_model.caption_projection.to(device)
            context_video = nag_cond_video[0][0].to(device, dtype)
            v_context, _ = torch.split(context_video, int(context_video.shape[-1] / 2), len(context_video.shape) - 1)
            context_video = diffusion_model.caption_projection(v_context)
            diffusion_model.caption_projection.to(offload_device)
            context_video = context_video.view(1, -1, img_dim)
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                patched_attn2 = LTXVCrossAttentionPatch(context_video, nag_scale, nag_alpha, nag_tau).__get__(block.attn2, block.__class__)
                model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn2.forward", patched_attn2)

        if nag_cond_audio is not None and diffusion_model.audio_caption_projection is not None:
            diffusion_model.audio_caption_projection.to(device)
            context_audio = nag_cond_audio[0][0].to(device, dtype)
            _, a_context = torch.split(context_audio, int(context_audio.shape[-1] / 2), len(context_audio.shape) - 1)
            context_audio = diffusion_model.audio_caption_projection(a_context)
            diffusion_model.audio_caption_projection.to(offload_device)
            context_audio = context_audio.view(1, -1, audio_dim)
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                patched_audio_attn2 = LTXVCrossAttentionPatch(context_audio, nag_scale, nag_alpha, nag_tau).__get__(block.audio_attn2, block.__class__)
                model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.audio_attn2.forward", patched_audio_attn2)

        return io.NodeOutput(model_clone)


def ffn_chunked_forward(self, x):
    if x.shape[1] > self.dim_threshold:
        chunks = torch.chunk(x, self.num_chunks, dim=1)
        output_chunks = []
        for chunk in chunks:
            output_chunks.append(self.net(chunk))
        chunked = torch.cat(output_chunks, dim=1)
        return chunked
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
                io.Int.Input("chunks", default=4, min=1, max=100, step=1, tooltip="Number of chunks to split the feedforward activations into to reduce peak VRAM usage."),
                io.Int.Input("dim_threshold", default=4096, min=1024, max=16384, step=256, tooltip="Dimension threshold above which to apply chunking."),
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
