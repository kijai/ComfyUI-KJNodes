from comfy_extras.nodes_lt import get_noise_mask, LTXVAddGuide
import types
import comfy
from comfy_api.latest import io
import numpy as np
import torch
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

def _compute_attention(self, query, context, transformer_options={}):
    """Compute attention and return the result. Cleans up intermediate tensors."""
    k = self.k_norm(self.to_k(context)).to(query.dtype)
    v = self.to_v(context).to(query.dtype)
    x = comfy.ldm.modules.attention.optimized_attention(query, k, v, heads=self.heads, transformer_options=transformer_options).flatten(2)
    del k, v
    return x

def nag_attention(self, query, context_positive, nag_context, transformer_options={}):
    x_positive = _compute_attention(self, query, context_positive, transformer_options)
    x_negative = _compute_attention(self, query, nag_context, transformer_options)
    return x_positive, x_negative

def normalized_attention_guidance(self, x_positive, x_negative):
    nag_guidance = x_positive * self.nag_scale - x_negative * (self.nag_scale - 1)
    del x_negative

    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True).expand_as(x_positive)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True).expand_as(nag_guidance)

    scale = torch.nan_to_num(norm_guidance / norm_positive, nan=10.0)
    mask = scale > self.nag_tau
    del scale

    adjustment = (norm_positive * self.nag_tau) / (norm_guidance + 1e-7)
    del norm_positive, norm_guidance

    nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)
    del mask, adjustment

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
    del x_pos

    x_positive, x_negative = nag_attention(self, q_pos, context_pos, self.nag_context, transformer_options=transformer_options)
    del context_pos, q_pos

    x_pos_out = normalized_attention_guidance(self, x_positive, x_negative)
    del x_positive, x_negative

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
        dtype = model.model.manual_cast_dtype

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

import math

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
        max_size = 256
        image_tensor = self.decode_latent_to_preview(image_tensor)
        if image_tensor.size(1) > max_size or image_tensor.size(2) > max_size:
            image_tensor = image_tensor.movedim(-1,0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (max_size * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height,max_size), mode='bilinear')
            else:
                width = (max_size * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (max_size, width), mode='bilinear')
            image_tensor = image_tensor.movedim(0,-1)
        previews_ubyte = (image_tensor.clamp(0, 1)
                         .mul(0xFF)  # to 0..255
                         ).to(device="cpu", dtype=torch.uint8)

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
            x0 = x0.unsqueeze(0).to(dtype=self.taeltx.vae_dtype, device=device)
            x_sample = self.taeltx.first_stage_model.decode(x0)[0].permute(1, 2, 3, 0)
            return x_sample
        else:
            self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)
            latent_image = F.linear(x0.movedim(1, -1), self.latent_rgb_factors,
                                    bias=self.latent_rgb_factors_bias)
            latent_image = (latent_image + 1.0) / 2.0
            return latent_image


def prepare_callback(model, steps, x0_output_dict=None, shape=None, latent_upscale_model=None, vae=None, rate=8, taeltx=False):
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
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = WrappedPreviewer(latent_rgb_factors, latent_rgb_factors_bias, rate=rate, taeltx=vae if taeltx else None)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0 is not None and shape is not None:
            cut = math.prod(shape[1:])
            x0 = x0[:, :, :cut].reshape([x0.shape[0]] + list(shape)[1:])

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
        original_callback = callback
        if self.latent_upscale_model is not None:
            self.latent_upscale_model.to(device)
        if self.vae is not None and self.taeltx:
            self.vae.first_stage_model.to(device)
        new_callback = prepare_callback(guider.model_patcher, len(sigmas) -1, shape=latent_shapes[0] if len(latent_shapes) > 1 else latent_shapes, x0_output_dict=self.x0_output, latent_upscale_model=self.latent_upscale_model, vae=self.vae, rate=self.preview_rate, taeltx=self.taeltx)
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
            category="KJNodes/experimental",
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
            category="KJNodes/experimental",
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
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.DynamicCombo.Input(
                    "num_images",
                    options=options,
                    display_name="Number of Images",
                    tooltip="Select how many images to insert",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
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
