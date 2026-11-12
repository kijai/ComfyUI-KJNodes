import inspect
import logging
import re
import types

import torch

import comfy.model_management as mm
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.minimax.model import _mod_scale_shift, _mod_gate, PackedLayout

from comfy_api.latest import io, ui
from nodes import MAX_RESOLUTION
from .model_optimization_nodes import _prompt_token_weights, _QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END

try:
    from server import PromptServer
except Exception:
    PromptServer = None


def minimax_mlp_chunked_forward(self, x):
    # x is packed (S, hidden); rows are independent and int8 activation quant is per-row, so token chunking is exact
    if x.shape[0] > self.kj_seq_threshold and self.kj_num_chunks > 1:
        out = torch.empty_like(x)
        offset = 0
        for c in torch.chunk(x, self.kj_num_chunks, dim=0):
            out[offset:offset + c.shape[0]] = comfy.ops.linear_input_act(self.fc2, self.fc1(c), "swiglu")
            offset += c.shape[0]
        return out
    return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")


class MiniMaxFFNChunkPatch:
    def __init__(self, num_chunks, seq_threshold):
        self.num_chunks = num_chunks
        self.seq_threshold = seq_threshold

    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.kj_num_chunks = self.num_chunks
            self_module.kj_seq_threshold = self.seq_threshold
            return minimax_mlp_chunked_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)


class MiniMaxChunkFeedForward(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxChunkFeedForward",
            display_name="MiniMax H3 Chunk FeedForward",
            category="KJNodes/experimental",
            description="Chunks the MiniMax H3 feedforward (SwiGLU) over the packed token dim to reduce peak VRAM usage. "
                        "Each chunk still goes through ComfyUI's fused INT8 kernel when the model is quantized, and since "
                        "activations are quantized per-token the output matches the unchunked model.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("chunks", default=2, min=1, max=64, step=1, tooltip="Number of chunks to split the feedforward tokens into. More chunks = lower peak VRAM, slightly more overhead."),
                io.Int.Input("seq_threshold", default=4096, min=256, max=262144, step=256, tooltip="Only chunk when the packed token count exceeds this (skips tiny sequences)."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, chunks, seq_threshold) -> io.NodeOutput:
        if chunks == 1:
            return io.NodeOutput(model)

        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")

        blocks = getattr(diffusion_model, "blocks", None)
        if not blocks or not hasattr(blocks[0], "mlp") or not hasattr(blocks[0].mlp, "fc1"):
            logging.warning("MiniMaxChunkFeedForward: model does not look like MiniMax H3 "
                            "(expected diffusion_model.blocks[*].mlp.fc1/fc2); returning model unchanged.")
            return io.NodeOutput(model)

        for idx, block in enumerate(blocks):
            patched = MiniMaxFFNChunkPatch(chunks, seq_threshold).__get__(block.mlp, block.mlp.__class__)
            m.add_object_patch(f"diffusion_model.blocks.{idx}.mlp.forward", patched)

        return io.NodeOutput(m)


def minimax_attn_lowmem_forward(self, x, rope_freqs=None, transformer_options={}):
    # Attention.forward restructured to free the normed h right after the qkv GEMM
    # and the fused (S, 3*inner) qkv buffer before out_proj allocates
    if isinstance(x, list):
        x = x.pop()
    s = x.shape[0]
    device = x.device
    dtype = x.dtype
    qkv = self.qkv_proj(x)
    del x
    q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = mm.cast_to(self.q_norm.weight, device=device)
        kw = mm.cast_to(self.k_norm.weight, device=device)
        rot = rope_freqs.shape[-3] * 2
        if mm.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    n = min(transformer_options.get("minimax_head_chunks", 1), self.heads) if isinstance(transformer_options, dict) else 1
    if n <= 1:
        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options).squeeze(0)
    else:
        # heads are independent so per-group attention is exact; the kernel's internal temps scale with heads per call
        out = torch.empty((s, self.heads * self.head_dim), dtype=dtype, device=device)
        hs = 0
        for size in ([self.heads // n + (1 if i < self.heads % n else 0) for i in range(n)]):
            he = hs + size
            o = optimized_attention(q[:, hs:he], k[:, hs:he], v[:, hs:he], size, mask=None, skip_reshape=True, transformer_options=transformer_options)
            out[:, hs * self.head_dim:he * self.head_dim] = o.squeeze(0)
            hs = he
    del q, k, v, qkv
    return self.out_proj(out)


# attention overrides (e.g. Sol-Attn) compose via optimized_attention instead of wrapping this forward
minimax_attn_lowmem_forward._uses_optimized_attention = True


def minimax_block_lowmem_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
    # DiTBlock.forward, but hands h to attn in a list so attn can free it after the qkv GEMM
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
    if attention is None:
        attention = self.attn
        h = [h]
    x = _mod_gate(x, gate_msa, attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
    h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


class MiniMaxLowVRAMAttention(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxLowVRAMAttention",
            display_name="MiniMax H3 Low VRAM Attention",
            category="KJNodes/experimental",
            description="Reduces peak VRAM of the MiniMax H3 attention without changing the math. Splits the attention "
                        "call into head groups so the kernel's internal transients (int8 q/k copies, fp32 accumulator) "
                        "shrink by the chunk count while the fused qkv buffer is alive, and frees the qkv buffer / "
                        "normed block input as soon as they're consumed. Output is identical to the unpatched model.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("head_chunks", default=4, min=1, max=56, step=1, tooltip="Number of head groups to split the attention call into. Kernel-internal transients scale down with this; heads are independent so the result is unchanged."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, head_chunks) -> io.NodeOutput:
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")

        blocks = getattr(diffusion_model, "blocks", None)
        if not blocks or not hasattr(blocks[0], "attn") or not hasattr(blocks[0].attn, "qkv_proj"):
            logging.warning("MiniMaxLowVRAMAttention: model does not look like MiniMax H3 "
                            "(expected diffusion_model.blocks[*].attn.qkv_proj); returning model unchanged.")
            return io.NodeOutput(model)

        # the attn forwards read this per-branch, so head slicing follows the node in either chain order
        if head_chunks > 1:
            m.model_options["transformer_options"]["minimax_head_chunks"] = head_chunks
        # Sol-Attn's compose gate uses this instead of the stock forward, keeping the low-VRAM path on sparse calls
        m.model_options["transformer_options"]["sol_take_forward"] = minimax_attn_lowmem_forward

        composed = False
        for idx, block in enumerate(blocks):
            m.add_object_patch(f"diffusion_model.blocks.{idx}.forward", types.MethodType(minimax_block_lowmem_forward, block))
            attn_key = f"diffusion_model.blocks.{idx}.attn.forward"
            if attn_key in m.object_patches:
                # another attention patch (e.g. mem eff sage) owns this key; keep it, it picks up head_chunks from transformer_options
                composed = True
                continue
            m.add_object_patch(attn_key, types.MethodType(minimax_attn_lowmem_forward, block.attn))

        if composed:
            logging.info("MiniMaxLowVRAMAttention: composing with an existing attention patch; "
                         "keeping its forward, adding the block-level h release and passing head_chunks through.")

        return io.NodeOutput(m)


class MiniMaxH3TokenCounter(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TokenCounter",
            display_name="MiniMax H3 Token Counter",
            category="KJNodes/misc",
            description="Counts the packed sequence length a MiniMax H3 generation will run at, without sampling. "
                        "Builds the same [text | keyframes/references | audio | video] layout the model uses, from the "
                        "AV latent and the conditioning (text length, keyframe and reference blocks included).",
            inputs=[
                io.Latent.Input("samples", tooltip="The AV latent that would be sampled (video+audio pair from the H3 latent nodes)."),
                io.Conditioning.Input("conditioning", tooltip="Positive conditioning; text length and any keyframe/reference blocks are read from it."),
            ],
            outputs=[
                io.Latent.Output(display_name="samples", tooltip="The input latent, unchanged."),
                io.Conditioning.Output(display_name="conditioning", tooltip="The input conditioning, unchanged."),
                io.Int.Output(display_name="tokens"),
                io.String.Output(display_name="breakdown"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, samples, conditioning) -> io.NodeOutput:
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            video, audio = latent.unbind()[:2]
            audio_t = audio.shape[-1]
        else:
            video, audio_t = latent, 0
        if video.ndim != 5:
            raise ValueError(f"MiniMaxH3TokenCounter: expected a video latent of shape [B, C, T, H, W], got {tuple(video.shape)}")

        # mirror model_base.MiniMaxH3.extra_conds: h/w rounded up to the DiT's 2x2 patch
        latent_t = video.shape[2]
        lat_h = (video.shape[3] + 1) // 2 * 2
        lat_w = (video.shape[4] + 1) // 2 * 2

        # older comfy versions have a PackedLayout without some kwargs; only fail if the conditioning actually uses one
        supported = set(inspect.signature(PackedLayout.__init__).parameters)

        def build_layout(cond, cond_dict):
            kwargs = {"keyframes": cond_dict.get("minimax_keyframes"),
                      "refs": cond_dict.get("minimax_refs"),
                      "frame_count": cond_dict.get("minimax_frame_count")}
            missing = [k for k, v in kwargs.items() if v is not None and k not in supported]
            if missing:
                raise RuntimeError(f"MiniMaxH3TokenCounter: this ComfyUI version's PackedLayout does not support {missing}, "
                                   "update ComfyUI to count this conditioning correctly.")
            return PackedLayout(cond.shape[1], latent_t, lat_h, lat_w, audio_t,
                                **{k: v for k, v in kwargs.items() if v is not None and k in supported})

        # scheduled conds can differ in text length; report the largest sequence
        layout = max((build_layout(cond, cond_dict) for cond, cond_dict in conditioning),
                     key=lambda l: l.seq_len)

        rows = {}
        seg_count = {}
        for a, b, kind in layout.segments:
            rows[kind] = rows.get(kind, 0) + (b - a)
            seg_count[kind] = seg_count.get(kind, 0) + 1

        parts = [("total", str(layout.seq_len))]
        # int32 offset limit of the sageattention quant/attention kernels at H3's 56x128 heads
        if layout.seq_len * 7168 >= 2**31:
            parts.append(("WARNING", f"over the int32-safe attention range ({2**31 // 7168} tokens), sageattention kernels may overflow"))
        parts.append(("text", str(rows.get("text", 0))))
        if "cond" in rows:
            parts.append(("keyframes", f"{rows['cond']} ({seg_count['cond']} frame{'s' if seg_count['cond'] > 1 else ''})"))
        if "ref_img" in rows:
            parts.append(("image/video refs", f"{rows['ref_img']} ({seg_count['ref_img']} block{'s' if seg_count['ref_img'] > 1 else ''})"))
        if "ref_audio" in rows:
            parts.append(("audio refs", f"{rows['ref_audio']} ({seg_count['ref_audio']} block{'s' if seg_count['ref_audio'] > 1 else ''})"))
        parts.append(("audio", str(rows.get("audio", 0))))
        parts.append(("video", f"{rows.get('video', 0)} ({latent_t}x{lat_h // 2}x{lat_w // 2} patches)"))

        breakdown = "\n".join(f"{k}: {v}" for k, v in parts)
        # inline display on the node; the frontend text preview strips all tags except <br>/<a>, so plain text only
        unique_id = getattr(cls.hidden, "unique_id", None)
        if PromptServer is not None and unique_id:
            try:
                PromptServer.instance.send_progress_text(breakdown.replace("WARNING:", "⚠"), unique_id)
            except Exception:
                pass
        return io.NodeOutput(samples, conditioning, layout.seq_len, breakdown, ui=ui.PreviewText(breakdown))


# MiniMax H3 Image prompt weighting, same mechanism as Krea2 and MOSS-Image. Text rows come first in the packed sequence
# and the prompt is the last part of the text, behind any picture blocks.
def _minimax_weight_wrapper(pairs, prompt_len, name):
    v_pos, v_fac = [p for p, vf, _ in pairs if vf != 1.0], [vf for _, vf, _ in pairs if vf != 1.0]
    k_pos, k_bias = [p for p, _, kb in pairs if kb != 0.0], [kb for _, _, kb in pairs if kb != 0.0]
    warned = []
    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        offset = context.shape[1] - prompt_len
        if transformer_options.get("cond_or_uncond") == [0]:    # positive cond only
            if offset >= 0:
                dev = context.device
                transformer_options["minimax_token_weights"] = (
                    torch.tensor(v_pos, device=dev) + offset if v_pos else None,
                    torch.tensor(v_fac, device=dev)[:, None] if v_pos else None,
                    torch.tensor(k_pos, device=dev) + offset if k_pos else None,
                    torch.tensor(k_bias, device=dev) if k_pos else None)
            elif not warned:
                warned.append(True)
                logging.warning(f"{name}: conditioning is shorter than the prompt, weighting skipped")
        return executor(x, timestep, context, transformer_options, **kwargs)
    return wrapper

def _minimax_attn_override(weights, comfy_attention, prev):
    v_idx, v_fac, k_idx, k_bias = weights
    def override(func, q, k, v, heads, mask=None, **kwargs):
        if v_idx is not None:
            v[..., v_idx, :] = v[..., v_idx, :] * v_fac.to(v.dtype)
        if k_idx is not None and mask is None:
            mask = q.new_zeros(1, 1, 1, k.shape[-2])
            mask[..., k_idx] = k_bias.to(q.dtype)
        if prev is not None:
            return prev(func, q, k, v, heads, mask=mask, **kwargs)
        # keep the block's own attention backend
        attention = comfy_attention.function if comfy_attention is not None else None
        return (attention or func)(q, k, v, heads, mask=mask, **kwargs)
    return override

def _minimax_block_patch(comfy_attention, prev_patch):
    def patch(args, extra):
        to = args["transformer_options"]
        weights = to.get("minimax_token_weights")
        if weights is not None:
            to = {**to, "optimized_attention_override": _minimax_attn_override(weights, comfy_attention, to.get("optimized_attention_override"))}
            args = {**args, "transformer_options": to}
        return prev_patch(args, extra) if prev_patch is not None else extra["original_block"](args)
    return patch


class MiniMaxH3PromptWeight(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptWeight",
            display_name="MiniMax H3 Prompt Weight",
            category="KJNodes/experimental",
            is_experimental=True,
            description="MiniMax H3 Image Conditioning with per-token prompt weights, (word:-1) removes a concept, (word:1.5) emphasizes it.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True, default="", tooltip="Prompt, weights as (word:weight)."),
                io.Float.Input("strength", default=1.0, min=-4.0, max=4.0, step=0.05, tooltip="Multiplier on the weighting effect."),
                io.Int.Input("width", default=0, min=0, max=MAX_RESOLUTION, step=32, tooltip="0 uses image_1's size, or 2048 without images."),
                io.Int.Input("height", default=0, min=0, max=MAX_RESOLUTION, step=32),
                io.DynamicCombo.Input("mode", tooltip="text with empty image adds a black picture's vision tokens, which softens the hard text-to-image rendering.", options=[
                    io.DynamicCombo.Option("text only", []),
                    io.DynamicCombo.Option("text with empty image", [
                        io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, tooltip="Start of the sampling range with the vision tokens."),
                        io.Float.Input("end_percent", default=0.25, min=0.0, max=1.0, step=0.001, tooltip="End of the range. 0.25 is the first two of 6 steps."),
                    ]),
                    io.DynamicCombo.Option("edit", [
                        io.Vae.Input("vae", optional=True),
                        io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(input=io.Image.Input("image"), prefix="image_", min=0, max=100)),
                    ]),
                ]),
            ],
            outputs=[io.Model.Output(display_name="model"), io.Conditioning.Output(display_name="positive"), io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, model, clip, text, strength, width, height, mode) -> io.NodeOutput:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning    # not in older ComfyUI
        clean_text = re.sub(r"\(([^():]+):(-?\d*\.?\d+)\)", r"\1", text)    # as _prompt_token_weights strips them
        cond, latent = MiniMaxH3ImageConditioning.execute(clip, clean_text, width, height, vae=mode.get("vae"), images=mode.get("images")).result
        _, pairs = _prompt_token_weights(clip, text, strength, (_QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END), "MiniMaxH3PromptWeight", encode=False)
        if mode["mode"] == "text with empty image" and mode["end_percent"] > mode["start_percent"]:
            start_percent, end_percent = mode["start_percent"], mode["end_percent"]
            # the encoder is causal, so the picture block is the same in front of any prompt
            picture, picture_dict = clip.encode_from_tokens_scheduled(
                clip.tokenize("", images=[torch.zeros(1, 512, 512, 3)], picture_label="Picture %d: "))[0]
            with_picture = []
            for c, d in cond:
                tags = torch.cat([picture_dict["minimax_token_tags"], d["minimax_token_tags"]])
                with_picture.append([torch.cat([picture, c], dim=1), {**d, "minimax_token_tags": tags, "start_percent": start_percent, "end_percent": end_percent}])
                if start_percent > 0.0:
                    with_picture.append([c, {**d, "start_percent": 0.0, "end_percent": start_percent}])
                if end_percent < 1.0:
                    with_picture.append([c, {**d, "start_percent": end_percent, "end_percent": 1.0}])
            cond = with_picture
        if not pairs:
            return io.NodeOutput(model, cond, latent)

        prompt_len = len(next(iter(clip.tokenize(clean_text).values()))[0])
        diffusion_model = model.get_model_object("diffusion_model")
        model_clone = model.clone()
        model_clone.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "MiniMaxH3PromptWeight",
                                         _minimax_weight_wrapper(pairs, prompt_len, "MiniMaxH3PromptWeight"))
        existing = model_clone.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for idx, block in enumerate(diffusion_model.blocks):
            model_clone.set_model_patch_replace(_minimax_block_patch(getattr(block.attn, "comfy_attention", None), existing.get(("double_block", idx))),
                                                "dit", "double_block", idx)
        return io.NodeOutput(model_clone, cond, latent)
