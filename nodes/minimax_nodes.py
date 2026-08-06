import logging
import types

import torch

import comfy.model_management as mm
import comfy.ops
import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.minimax.model import _mod_scale_shift, _mod_gate

from comfy_api.latest import io


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
    n = min(getattr(self, "kj_head_chunks", 1), self.heads)
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


def minimax_block_lowmem_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
    # DiTBlock.forward, but hands h to attn in a list so attn can free it after the qkv GEMM
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = [_mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)]
    x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
    h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


class MiniMaxAttnPatch:
    def __init__(self, head_chunks):
        self.head_chunks = head_chunks

    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.kj_head_chunks = self.head_chunks
            return minimax_attn_lowmem_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)


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

        for idx, block in enumerate(blocks):
            m.add_object_patch(f"diffusion_model.blocks.{idx}.forward", types.MethodType(minimax_block_lowmem_forward, block))
            patched_attn = MiniMaxAttnPatch(head_chunks).__get__(block.attn, block.attn.__class__)
            m.add_object_patch(f"diffusion_model.blocks.{idx}.attn.forward", patched_attn)

        return io.NodeOutput(m)
