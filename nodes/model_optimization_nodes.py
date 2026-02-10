import os
from comfy.ldm.modules import attention as comfy_attention
import logging
import torch
import importlib
import math
import datetime

import folder_paths
import comfy.model_management as mm
from comfy.cli_args import args
from comfy.ldm.modules.attention import wrap_attn, optimized_attention
import comfy.model_patcher
import comfy.utils
import comfy.sd


try:
    from comfy_api.latest import io
    v3_available = True
except ImportError:
    v3_available = False
    logging.warning("ComfyUI v3 node API not available, please update ComfyUI to access latest v3 nodes.")

_all_sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]

try:
    import sageattention  # noqa: F401
    _sageattention_available = True
except ImportError:
    _sageattention_available = False
    logging.info("sageattention not available (requires triton/CUDA), sage attention modes will be disabled")

sageattn_modes = _all_sageattn_modes if _sageattention_available else ["disabled"]

_initialized = False
_original_functions = {}

if not _initialized:
    _original_functions["orig_attention"] = comfy_attention.optimized_attention
    _original_functions["original_patch_model"] = comfy.model_patcher.ModelPatcher.patch_model
    _original_functions["original_load_lora_for_models"] = comfy.sd.load_lora_for_models
    try:
        _original_functions["original_qwen_forward"] = comfy.ldm.qwen_image.model.Attention.forward
    except:
        pass
    _initialized = True


def get_sage_func(sage_attention, allow_compile=False):
    logging.info(f"Using sage attention mode: {sage_attention}")
    if not _sageattention_available:
        raise RuntimeError(
            f"sage_attention mode '{sage_attention}' requires the sageattention package "
            "(which depends on triton/CUDA). Install with: pip install sageattention "
            "or set sage_attention to 'disabled'."
        )
    from sageattention import sageattn
    if sage_attention == "auto":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
    elif "sageattn3" in sage_attention:
        from sageattn3 import sageattn3_blackwell
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(sage_attention == "sageattn3_per_block_mean"))
            return out.transpose(1, 2) if tensor_layout == "NHD" else out

    if not allow_compile:
        sage_func = torch.compiler.disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout="HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head),
                (q, k, v),
            )
            tensor_layout="NHD"
        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = (
                    out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                )
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out
    return attention_sage


from comfy.patcher_extension import CallbacksMP
class PathchSageAttentionKJ():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."}),
        },
        "optional": {
            "allow_compile": ("BOOLEAN", {"default": False, "tooltip": "Allow the use of torch.compile for the sage attention function, requires latest sageattn 2.2.0 or higher."})
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"
    DESCRIPTION = "Experimental node for patching attention mode. This doesn't use the model patching system and thus can't be disabled without running the node again with 'disabled' option."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, model, sage_attention, allow_compile=False):
        if sage_attention == "disabled":
            return model,

        model_clone = model.clone()

        new_attention = get_sage_func(sage_attention, allow_compile=allow_compile)
        def attention_override_sage(func, *args, **kwargs):
            return new_attention.__wrapped__(*args, **kwargs)

        # attention override
        model_clone.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage

        return model_clone,


class CheckpointLoaderKJ():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"],),
            "compute_dtype": (["default", "fp16", "bf16", "fp32"], {"default": "default", "tooltip": "The compute dtype to use for the model."}),
            "patch_cublaslinear": ("BOOLEAN", {"default": False, "tooltip": "Enable or disable the cublas_ops arg"}),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Patch comfy attention to use sageattn."}),
            "enable_fp16_accumulation": ("BOOLEAN", {"default": False, "tooltip": "Enable torch.backends.cuda.matmul.allow_fp16_accumulation, required minimum pytorch version 2.7.1"}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    DESCRIPTION = "Experimental node for patching torch.nn.Linear with CublasLinear."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def load(self, ckpt_name, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation):
        DTYPE_MAP = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        model_options = {}
        if dtype := DTYPE_MAP.get(weight_dtype):
            model_options["dtype"] = dtype
            logging.info(f"Setting {ckpt_name} weight dtype to {dtype}")

        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True

        if patch_cublaslinear:
            args.fast.add("cublas_ops")
        else:
            args.fast.discard("cublas_ops")

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)

        model, clip, vae, _ = comfy.sd.load_state_dict_guess_config(
            sd,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            metadata=metadata,
            model_options=model_options)

        if dtype := DTYPE_MAP.get(compute_dtype):
            model.set_model_compute_dtype(dtype)
            model.force_cast_weights = False
            logging.info(f"Setting {ckpt_name} compute dtype to {dtype}")

        if enable_fp16_accumulation:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise RuntimeError("Failed to set fp16 accumulation, requires pytorch version 2.7.1 or higher")
        else:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = False

        if sage_attention != "disabled":
            new_attention = get_sage_func(sage_attention)
            def attention_override_sage(func, *args, **kwargs):
                return new_attention.__wrapped__(*args, **kwargs)

            # attention override
            model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage

        return model, clip, vae


class DiffusionModelSelector():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The name of the checkpoint (model) to load."}),
        },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_path"
    DESCRIPTION = "Returns the path to the model as a string."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def get_path(self, model_name):
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return (model_path,)

class DiffusionModelLoaderKJ():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The name of the checkpoint (model) to load."}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"],),
            "compute_dtype": (["default", "fp16", "bf16", "fp32"], {"default": "default", "tooltip": "The compute dtype to use for the model."}),
            "patch_cublaslinear": ("BOOLEAN", {"default": False, "tooltip": "Enable or disable the cublas_ops arg"}),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Patch comfy attention to use sageattn."}),
            "enable_fp16_accumulation": ("BOOLEAN", {"default": False, "tooltip": "Enable torch.backends.cuda.matmul.allow_fp16_accumulation, requires pytorch 2.7.0 nightly."}),
        },
        "optional": {
            "extra_state_dict": ("STRING", {"forceInput": True, "tooltip": "The full path to an additional state dict to load, this will be merged with the main state dict. Useful for example to add VACE module to a WanVideoModel. You can use DiffusionModelSelector to easily get the path."}),
        }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_and_load"
    DESCRIPTION = "Node for patching torch.nn.Linear with CublasLinear."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch_and_load(self, model_name, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation, extra_state_dict=None):
        DTYPE_MAP = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        model_options = {}
        if dtype := DTYPE_MAP.get(weight_dtype):
            model_options["dtype"] = dtype
            logging.info(f"Setting {model_name} weight dtype to {dtype}")

        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True

        if enable_fp16_accumulation:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.1 or higher")
        else:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = False

        if patch_cublaslinear:
            args.fast.add("cublas_ops")
        else:
            args.fast.discard("cublas_ops")

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        if extra_state_dict is not None:
            # If the model is a checkpoint, strip additional non-diffusion model entries before adding extra state dict
            from comfy import model_detection
            diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
            if diffusion_model_prefix == "model.diffusion_model.":
                temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
                if len(temp_sd) > 0:
                    sd = temp_sd

            extra_sd = comfy.utils.load_torch_file(extra_state_dict)
            sd.update(extra_sd)
            del extra_sd

        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
        if dtype := DTYPE_MAP.get(compute_dtype):
            model.set_model_compute_dtype(dtype)
            model.force_cast_weights = False
            logging.info(f"Setting {model_name} compute dtype to {dtype}")

        if sage_attention != "disabled":
            new_attention = get_sage_func(sage_attention)
            def attention_override_sage(func, *args, **kwargs):
                return new_attention.__wrapped__(*args, **kwargs)

            # attention override
            model.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage

        return (model,)

class ModelPatchTorchSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "enable_fp16_accumulation": ("BOOLEAN", {"default": False, "tooltip": "Enable torch.backends.cuda.matmul.allow_fp16_accumulation, requires pytorch 2.7.0 nightly."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = "Adds callbacks to model to set torch settings before and after running the model."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, model, enable_fp16_accumulation):        
        model_clone = model.clone()

        def patch_enable_fp16_accum(model):
            logging.info("Patching torch settings: torch.backends.cuda.matmul.allow_fp16_accumulation = True")
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        def patch_disable_fp16_accum(model):
            logging.info("Patching torch settings: torch.backends.cuda.matmul.allow_fp16_accumulation = False")
            torch.backends.cuda.matmul.allow_fp16_accumulation = False

        if enable_fp16_accumulation:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_enable_fp16_accum)
                model_clone.add_callback(CallbacksMP.ON_CLEANUP, patch_disable_fp16_accum)
            else:
                raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.1 or higher")
        else:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_disable_fp16_accum)
            else:
                raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.1 or higher")

        return (model_clone,)


class PatchModelPatcherOrder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "patch_order": (["object_patch_first", "weight_patch_first"], {"default": "weight_patch_first", "tooltip": "Patch the comfy patch_model function to load weight patches (LoRAs) before compiling the model"}),
                    "full_load": (["enabled", "disabled", "auto"], {"default": "auto", "tooltip": "Disabling may help with memory issues when loading large models, when changing this you should probably force model reload to avoid issues!"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "NO LONGER NECESSARY OR FUNCTIONAL, keeping node for backwards compatibility. Use the TorchCompileModelAdvanced to use LoRA with torch.compile."
    DEPRECATED = True

    def patch(self, model, patch_order, full_load):
        return model,


class TorchCompileModelFluxAdvancedV2:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
                    "single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                },
                "optional": {
                    "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                    "force_parameter_static_shapes": ("BOOLEAN", {"default": True, "tooltip": "torch._dynamo.config.force_parameter_static_shapes"}),
                }
                }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True
    DEPRECATED = True
    DESCRIPTION = "Deprecated, use TorchCompileModelAdvanced instead."

    def patch(self, model, backend, mode, fullgraph, single_blocks, double_blocks, dynamic, dynamo_cache_size_limit=64, force_parameter_static_shapes=True):
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        torch._dynamo.config.force_parameter_static_shapes = force_parameter_static_shapes

        compile_key_list = []

        try:
            if double_blocks:
                for i, block in enumerate(diffusion_model.double_blocks):
                    print("Adding double block to compile list", i)
                    compile_key_list.append(f"diffusion_model.double_blocks.{i}")
            if single_blocks:
                for i, block in enumerate(diffusion_model.single_blocks):
                    compile_key_list.append(f"diffusion_model.single_blocks.{i}")

            set_torch_compile_wrapper(model=m, keys=compile_key_list, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)           
        except:
            raise RuntimeError("Failed to compile model")

        return (m, )


class TorchCompileModelWanVideoV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": True, "tooltip": "Compile only transformer blocks, faster compile and less error prone"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),

            },
            "optional": {
                "force_parameter_static_shapes": ("BOOLEAN", {"default": True, "tooltip": "torch._dynamo.config.force_parameter_static_shapes"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True
    DEPRECATED = True
    DESCRIPTION = "Deprecated, use TorchCompileModelAdvanced instead."

    def patch(self, model, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks_only, force_parameter_static_shapes=True):
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        m = model.clone()

        # Skip torch.compile on MPS — inductor compilation is 10-100x slower than
        # CUDA, autotuning is unsupported, and dynamic shapes in video models cause
        # recompilation storms that make inference slower, not faster.
        if mm.is_device_mps(mm.get_torch_device()):
            logging.warning("TorchCompileModelWanVideoV2: Skipping torch.compile on MPS (not beneficial for video models)")
            return (m, )

        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        torch._dynamo.config.force_parameter_static_shapes = force_parameter_static_shapes
        try:
            if compile_transformer_blocks_only:
                compile_key_list = []
                for i, block in enumerate(diffusion_model.blocks):
                    compile_key_list.append(f"diffusion_model.blocks.{i}")
            else:
                compile_key_list =["diffusion_model"]

            set_torch_compile_wrapper(model=m, keys=compile_key_list, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
        except:
            raise RuntimeError("Failed to compile model")

        return (m, )


class TorchCompileModelAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": (
                    ["auto", "true", "false"],
                    {"default": "auto", "tooltip": "Use dynamic shape tracing."},
                ),
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": True, "tooltip": "Compile only transformer blocks, faster compile and less error prone"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "debug_compile_keys": ("BOOLEAN", {"default": False, "tooltip": "Print the compile keys used for torch.compile"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/torchcompile"
    DESCRIPTION = "Advanced torch.compile patching for diffusion models."
    EXPERIMENTAL = True

    def patch(self, model, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks_only, debug_compile_keys):
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit   

        try:
            if compile_transformer_blocks_only:
                layer_types = ["double_blocks", "single_blocks", "layers", "transformer_blocks", "blocks", "visual_transformer_blocks", "text_transformer_blocks"]
                compile_key_list = []
                for layer_name in layer_types:
                    if hasattr(diffusion_model, layer_name):
                        blocks = getattr(diffusion_model, layer_name)
                        for i in range(len(blocks)):
                            compile_key_list.append(f"diffusion_model.{layer_name}.{i}")
                if not compile_key_list:
                    logging.warning("No known transformer blocks found to compile, compiling entire diffusion model instead")
                elif debug_compile_keys:
                    logging.info("TorchCompileModelAdvanced: Compile key list:")
                    for key in compile_key_list:
                        logging.info(f" - {key}")
            if not compile_key_list:
                compile_key_list =["diffusion_model"]

            dynamic_kv = {"true": True, "false": False, "auto": None}
            try:
                dynamic = dynamic_kv[dynamic]
            except KeyError:
                raise ValueError(f"Invalid dynamic arg {dynamic}")

            set_torch_compile_wrapper(model=m, keys=compile_key_list, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)           
        except:
            raise RuntimeError("Failed to compile model")

        return (m, )


class TorchCompileModelQwenImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": True, "tooltip": "Compile only transformer blocks, faster compile and less error prone"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True
    DEPRECATED = True
    DESCRIPTION = "Deprecated, use TorchCompileModelAdvanced instead."

    def patch(self, model, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks_only):
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        try:
            if compile_transformer_blocks_only:
                compile_key_list = []
                for i, block in enumerate(diffusion_model.transformer_blocks):
                    compile_key_list.append(f"diffusion_model.transformer_blocks.{i}")
            else:
                compile_key_list =["diffusion_model"]

            set_torch_compile_wrapper(model=m, keys=compile_key_list, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
        except:
            raise RuntimeError("Failed to compile model")

        return (m, )

class TorchCompileVAE:
    def __init__(self):
        self._compiled_encoder = False
        self._compiled_decoder = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "vae": ("VAE",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "compile_encoder": ("BOOLEAN", {"default": True, "tooltip": "Compile encoder"}),
                    "compile_decoder": ("BOOLEAN", {"default": True, "tooltip": "Compile decoder"}),
                }}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "compile"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def compile(self, vae, backend, mode, fullgraph, compile_encoder, compile_decoder):
        if compile_encoder:
            if not self._compiled_encoder:
                encoder_name = "encoder"
                if hasattr(vae.first_stage_model, "taesd_encoder"):
                    encoder_name = "taesd_encoder"

                try:
                    setattr(
                        vae.first_stage_model,
                        encoder_name,
                        torch.compile(
                            getattr(vae.first_stage_model, encoder_name),
                            mode=mode,
                            fullgraph=fullgraph,
                            backend=backend,
                        ),
                    )
                    self._compiled_encoder = True
                except:
                    raise RuntimeError("Failed to compile model")
        if compile_decoder:
            if not self._compiled_decoder:
                decoder_name = "decoder"
                if hasattr(vae.first_stage_model, "taesd_decoder"):
                    decoder_name = "taesd_decoder"

                try:
                    setattr(
                        vae.first_stage_model,
                        decoder_name,
                        torch.compile(
                            getattr(vae.first_stage_model, decoder_name),
                            mode=mode,
                            fullgraph=fullgraph,
                            backend=backend,
                        ),
                    )
                    self._compiled_decoder = True
                except:
                    raise RuntimeError("Failed to compile model")
        return (vae, )

class TorchCompileControlNet:
    def __init__(self):
        self._compiled= False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "controlnet": ("CONTROL_NET",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                }}
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "compile"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def compile(self, controlnet, backend, mode, fullgraph):
        if not self._compiled:
            try:
                # for i, block in enumerate(controlnet.control_model.double_blocks):
                #     print("Compiling controlnet double_block", i)
                #     controlnet.control_model.double_blocks[i] = torch.compile(block, mode=mode, fullgraph=fullgraph, backend=backend)
                controlnet.control_model = torch.compile(controlnet.control_model, mode=mode, fullgraph=fullgraph, backend=backend)
                self._compiled = True
            except:
                self._compiled = False
                raise RuntimeError("Failed to compile model")

        return (controlnet, )


#teacache

try:
    from comfy.ldm.wan.model import sinusoidal_embedding_1d
except:
    pass
from einops import repeat
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np

def relative_l1_distance(last_tensor, current_tensor):
    l1_distance = torch.abs(last_tensor - current_tensor).mean()
    norm = torch.abs(last_tensor).mean()
    relative_l1_distance = l1_distance / norm
    return relative_l1_distance.to(torch.float32)

@torch.compiler.disable()
def tea_cache(self, x, e0, e, transformer_options):
    #teacache for cond and uncond separately
    rel_l1_thresh = transformer_options["rel_l1_thresh"]
    
    is_cond = True if transformer_options["cond_or_uncond"] == [0] else False

    should_calc = True
    suffix = "cond" if is_cond else "uncond"

    # Init cache dict if not exists
    if not hasattr(self, 'teacache_state'):
        self.teacache_state = {
            'cond': {'accumulated_rel_l1_distance': 0, 'prev_input': None, 
                    'teacache_skipped_steps': 0, 'previous_residual': None},
            'uncond': {'accumulated_rel_l1_distance': 0, 'prev_input': None,
                    'teacache_skipped_steps': 0, 'previous_residual': None}
        }
        logging.info("\nTeaCache: Initialized")

    cache = self.teacache_state[suffix]

    if cache['prev_input'] is not None:
        if transformer_options["coefficients"] == []:
            temb_relative_l1 = relative_l1_distance(cache['prev_input'], e0)
            curr_acc_dist = cache['accumulated_rel_l1_distance'] + temb_relative_l1
        else:
            rescale_func = np.poly1d(transformer_options["coefficients"])
            curr_acc_dist = cache['accumulated_rel_l1_distance'] + rescale_func(((e-cache['prev_input']).abs().mean() / cache['prev_input'].abs().mean()).cpu().item())
        try:
            if curr_acc_dist < rel_l1_thresh:
                should_calc = False
                cache['accumulated_rel_l1_distance'] = curr_acc_dist
            else:
                should_calc = True
                cache['accumulated_rel_l1_distance'] = 0
        except:
            should_calc = True
            cache['accumulated_rel_l1_distance'] = 0

    if transformer_options["coefficients"] == []:
        cache['prev_input'] = e0.clone().detach()
    else:
        cache['prev_input'] = e.clone().detach()

    if not should_calc:
        x += cache['previous_residual'].to(x.device)
        cache['teacache_skipped_steps'] += 1
        #print(f"TeaCache: Skipping {suffix} step")
    return should_calc, cache

def teacache_wanvideo_vace_forward_orig(self, x, t, context, vace_context, vace_strength, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        orig_shape = list(vace_context.shape)
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
        c = c.flatten(2).transpose(1, 2)
        c = list(c.split(orig_shape[0], dim=0))

        if not transformer_options:
            raise RuntimeError("Can't access transformer_options, this requires ComfyUI nightly version from Mar 14, 2025 or later")

        teacache_enabled = transformer_options.get("teacache_enabled", False)
        if not teacache_enabled:
            should_calc = True
        else:
            should_calc, cache = tea_cache(self, x, e0, e, transformer_options)
        
        if should_calc:
            original_x = x.clone().detach()
            patches_replace = transformer_options.get("patches_replace", {})
            blocks_replace = patches_replace.get("dit", {})
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

                ii = self.vace_layers_mapping.get(i, None)
                if ii is not None:
                    for iii in range(len(c)):
                        c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=original_x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
                        x += c_skip * vace_strength[iii]
                    del c_skip

            if teacache_enabled:
                cache['previous_residual']  = (x - original_x).to(transformer_options["teacache_device"])
          
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

def teacache_wanvideo_forward_orig(self, x, t, context, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]


        teacache_enabled = transformer_options.get("teacache_enabled", False)
        if not teacache_enabled:
            should_calc = True
        else:
            should_calc, cache = tea_cache(self, x, e0, e, transformer_options)
        
        if should_calc:
            original_x = x.clone().detach()
            patches_replace = transformer_options.get("patches_replace", {})
            blocks_replace = patches_replace.get("dit", {})
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

            if teacache_enabled:
                cache['previous_residual']  = (x - original_x).to(transformer_options["teacache_device"])
          
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

class WanVideoTeaCacheKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.275, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Threshold for to determine when to apply the cache, compromise between speed and accuracy. When using coefficients a good value range is something between 0.2-0.4 for all but 1.3B model, which should be about 10 times smaller, same as when not using coefficients."}),
                "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps to use with TeaCache."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps to use with TeaCache."}),
                "cache_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Device to cache to"}),
                "coefficients": (["disabled", "1.3B", "14B", "i2v_480", "i2v_720"], {"default": "i2v_480", "tooltip": "Coefficients for rescaling the relative l1 distance, if disabled the threshold value should be about 10 times smaller than the value used with coefficients."}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_teacache"
    CATEGORY = "KJNodes/teacache"
    DEPRECATED = True
    DESCRIPTION = """
Patch WanVideo model to use TeaCache. Speeds up inference by caching the output and  
applying it instead of doing the step.  Best results are achieved by choosing the  
appropriate coefficients for the model. Early steps should never be skipped, with too  
aggressive values this can happen and the motion suffers. Starting later can help with that too.   
When NOT using coefficients, the threshold value should be  
about 10 times smaller than the value used with coefficients.  

Official recommended values https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4Wan2.1
"""
    EXPERIMENTAL = True

    def patch_teacache(self, model, rel_l1_thresh, start_percent, end_percent, cache_device, coefficients):
        if rel_l1_thresh == 0:
            return (model,)

        if coefficients == "disabled" and rel_l1_thresh > 0.1:
            logging.warning("Threshold value is too high for TeaCache without coefficients, consider using coefficients for better results.")
        if coefficients != "disabled" and rel_l1_thresh < 0.1 and "1.3B" not in coefficients:
            logging.warning("Threshold value is too low for TeaCache with coefficients, consider using higher threshold value for better results.")
        
        # type_str = str(type(model.model.model_config).__name__)
        #if model.model.diffusion_model.dim == 1536:
        #    model_type ="1.3B"
        # else:
        #     if "WAN21_T2V" in type_str:
        #         model_type = "14B"
        #     elif "WAN21_I2V" in type_str:
        #         model_type = "i2v_480"
        #     else:
        #         model_type = "i2v_720" #how to detect this?
  
       
        teacache_coefficients_map = {
            "disabled": [],
            "1.3B": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
            "14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
            "i2v_480": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
            "i2v_720": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
        }
        coefficients = teacache_coefficients_map[coefficients]
        
        teacache_device = mm.get_torch_device() if cache_device == "main_device" else mm.unet_offload_device()

        model_clone = model.clone()
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}
        model_clone.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        model_clone.model_options["transformer_options"]["teacache_device"] = teacache_device
        model_clone.model_options["transformer_options"]["coefficients"] = coefficients
        diffusion_model = model_clone.get_model_object("diffusion_model")
                
        def outer_wrapper(start_percent, end_percent):        
            def unet_wrapper_function(model_function, kwargs):
                input = kwargs["input"]
                timestep = kwargs["timestep"]
                c = kwargs["c"]
                sigmas = c["transformer_options"]["sample_sigmas"]
                cond_or_uncond = kwargs["cond_or_uncond"]
                last_step = (len(sigmas) - 1)
             
                matched_step_index = (sigmas == timestep[0] ).nonzero()
                if len(matched_step_index) > 0:
                    current_step_index = matched_step_index.item()
                else:
                    for i in range(len(sigmas) - 1):
                        # walk from beginning of steps until crossing the timestep
                        if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                            current_step_index = i
                            break
                    else:
                        current_step_index = 0

                if current_step_index == 0:
                    if (len(cond_or_uncond) == 1 and cond_or_uncond[0] == 1) or len(cond_or_uncond) == 2:
                        if hasattr(diffusion_model, "teacache_state"):
                            delattr(diffusion_model, "teacache_state")
                            logging.info("\nResetting TeaCache state")
                
                current_percent = current_step_index / (len(sigmas) - 1)
                c["transformer_options"]["current_percent"] = current_percent
                if start_percent <= current_percent <= end_percent:
                    c["transformer_options"]["teacache_enabled"] = True
                
                forward_function = teacache_wanvideo_vace_forward_orig if hasattr(diffusion_model, "vace_layers") else teacache_wanvideo_forward_orig
                context = patch.multiple(
                    diffusion_model, 
                    forward_orig=forward_function.__get__(diffusion_model, diffusion_model.__class__)
                )

                with context:
                    out = model_function(input, timestep, **c)
                    if current_step_index+1 == last_step and hasattr(diffusion_model, "teacache_state"):
                        if len(cond_or_uncond) == 1 and cond_or_uncond[0] == 0:
                            skipped_steps_cond = diffusion_model.teacache_state["cond"]["teacache_skipped_steps"]
                            skipped_steps_uncond = diffusion_model.teacache_state["uncond"]["teacache_skipped_steps"]
                            logging.info("-----------------------------------")
                            logging.info(f"TeaCache skipped:")
                            logging.info(f"{skipped_steps_cond} cond steps")
                            logging.info(f"{skipped_steps_uncond} uncond step")
                            logging.info(f"out of {last_step} steps")
                            logging.info("-----------------------------------")
                        elif len(cond_or_uncond) == 2:
                            skipped_steps_cond = diffusion_model.teacache_state["uncond"]["teacache_skipped_steps"]
                            logging.info("-----------------------------------")
                            logging.info(f"TeaCache skipped:")
                            logging.info(f"{skipped_steps_cond} cond steps")
                            logging.info(f"out of {last_step} steps")
                            logging.info("-----------------------------------")
                        
                    return out
            return unet_wrapper_function

        model_clone.set_model_unet_function_wrapper(outer_wrapper(start_percent=start_percent, end_percent=end_percent))

        return (model_clone,)




from comfy.ldm.flux.math import apply_rope

def modified_wan_self_attention_forward(self, x, freqs, transformer_options={}):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n * d)
        return q, k, v

    q, k, v = qkv_fn(x)

    q, k = apply_rope(q, k, freqs)

    feta_scores = get_feta_scores(q, k, self.num_frames, self.enhance_weight)

    try:
        x = comfy.ldm.modules.attention.optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            v,
            heads=self.num_heads,
            transformer_options=transformer_options,
        )
    except:
        # backward compatibility for now
        x = comfy.ldm.modules.attention.attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            v,
            heads=self.num_heads,
        )

    x = self.o(x)

    x *= feta_scores

    return x

from einops import rearrange
def get_feta_scores(query, key, num_frames, enhance_weight, num_heads=12):
    img_q, img_k = query, key #torch.Size([2, 9216, 12, 128])

    if img_q.ndim == 4:
        B, ST, num_heads, head_dim = img_q.shape
    elif img_q.ndim == 3:
        B, ST, hidden_dim = img_q.shape
        head_dim = hidden_dim // num_heads

        # Reshape from [B, ST, hidden_dim] to [B, ST, num_heads, head_dim]
        img_q = img_q.view(B, ST, num_heads, head_dim)
        img_k = img_k.view(B, ST, num_heads, head_dim)

    spatial_dim = ST // num_frames

    query_image = rearrange(
        img_q, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
    )
    key_image = rearrange(
        img_k, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
    )

    return feta_score(query_image, key_image, head_dim, num_frames, enhance_weight)

def feta_score(query_image, key_image, head_dim, num_frames, enhance_weight):
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)  # translate attn to float32
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)

    # Reshape to [batch_size * num_tokens, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Create a mask for diagonal elements
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

    # Zero out diagonal elements
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    # Calculate mean for each token's attention matrix
    # Number of off-diagonal elements per matrix is n*n - n
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    enhance_scores = mean_scores.mean() * (num_frames + enhance_weight)
    enhance_scores = enhance_scores.clamp(min=1)
    return enhance_scores

import types
class WanAttentionPatch:
    def __init__(self, num_frames, weight):
        self.num_frames = num_frames
        self.enhance_weight = weight

    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.num_frames = self.num_frames
            self_module.enhance_weight = self.enhance_weight
            return modified_wan_self_attention_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)

class WanVideoEnhanceAVideoKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT", {"tooltip": "Only used to get the latent count"}),
                "weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of the enhance effect"}),
           }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "enhance"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"
    EXPERIMENTAL = True

    def enhance(self, model, weight, latent):
        if weight == 0:
            return (model,)

        num_frames = latent["samples"].shape[2]

        model_clone = model.clone()
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}
        model_clone.model_options["transformer_options"]["enhance_weight"] = weight
        diffusion_model = model_clone.get_model_object("diffusion_model")

        compile_settings = getattr(model.model, "compile_settings", None)
        for idx, block in enumerate(diffusion_model.blocks):
            patched_attn = WanAttentionPatch(num_frames, weight).__get__(block.self_attn, block.__class__)
            if compile_settings is not None:
                patched_attn = torch.compile(patched_attn, mode=compile_settings["mode"], dynamic=compile_settings["dynamic"], fullgraph=compile_settings["fullgraph"], backend=compile_settings["backend"])

            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.self_attn.forward", patched_attn)

        return (model_clone,)

try:
    from comfy.ldm.lightricks.model import apply_rotary_emb
except:
    apply_rotary_emb = None


def ltxv_feta_forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q = self.q_norm(q)
    k = self.k_norm(k)

    if pe is not None:
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

    feta_scores = get_feta_scores(q, k, self.num_frames, self.enhance_weight, self.heads)

    if mask is None:
        out = comfy.ldm.modules.attention.optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision, transformer_options=transformer_options)
    else:
        out = comfy.ldm.modules.attention.optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision, transformer_options=transformer_options)
    return self.to_out(out * feta_scores)


class LTXCrossAttentionPatch:
    def __init__(self, num_frames, weight):
        self.num_frames = num_frames
        self.enhance_weight = weight

    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.num_frames = self.num_frames
            self_module.enhance_weight = self.enhance_weight
            return ltxv_feta_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)

class LTXVEnhanceAVideoKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT", {"tooltip": "Only used to get the latent count"}),
                "weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of the enhance effect"}),
           }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "enhance"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"
    EXPERIMENTAL = True

    def enhance(self, model, weight, latent):
        if weight == 0:
            return (model,)

        num_frames = latent["samples"].shape[2]

        model_clone = model.clone()
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}
        model_clone.model_options["transformer_options"]["enhance_weight"] = weight
        diffusion_model = model_clone.get_model_object("diffusion_model")

        for idx, block in enumerate(diffusion_model.transformer_blocks):
            patched_attn1 = LTXCrossAttentionPatch(num_frames, weight).__get__(block.attn1, block.__class__)
            model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn1.forward", patched_attn1)
        return (model_clone,)

def normalized_attention_guidance(self, query, context_positive, context_negative, transformer_options={}):
    k_positive = self.norm_k(self.k(context_positive))
    v_positive = self.v(context_positive)
    k_negative = self.norm_k(self.k(context_negative))
    v_negative = self.v(context_negative)

    try:
        x_positive = comfy.ldm.modules.attention.optimized_attention(query, k_positive, v_positive, heads=self.num_heads, transformer_options=transformer_options).flatten(2)
        x_negative = comfy.ldm.modules.attention.optimized_attention(query, k_negative, v_negative, heads=self.num_heads, transformer_options=transformer_options).flatten(2)
    except: #backwards compatibility for now
        x_positive = comfy.ldm.modules.attention.optimized_attention(query, k_positive, v_positive, heads=self.num_heads).flatten(2)
        x_negative = comfy.ldm.modules.attention.optimized_attention(query, k_negative, v_negative, heads=self.num_heads).flatten(2)

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
def wan_crossattn_forward_nag(self, x, context, transformer_options={}, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
    # Determine batch splitting and context handling
    if self.input_type == "default":
        # Single or [pos, neg] pair
        if context.shape[0] == 1:
            x_pos, context_pos = x, context
            x_neg, context_neg = None, None
        else:
            x_pos, x_neg = torch.chunk(x, 2, dim=0)
            context_pos, context_neg = torch.chunk(context, 2, dim=0)
    elif self.input_type == "batch":
        # Standard batch, no CFG
        x_pos, context_pos = x, context
        x_neg, context_neg = None, None

    # Positive branch
    q_pos = self.norm_q(self.q(x_pos))
    nag_context = self.nag_context
    if self.input_type == "batch":
        nag_context = nag_context.repeat(x_pos.shape[0], 1, 1)
    try:
        x_pos_out = normalized_attention_guidance(self, q_pos, context_pos, nag_context, transformer_options=transformer_options)
    except: #backwards compatibility for now
        x_pos_out = normalized_attention_guidance(self, q_pos, context_pos, nag_context)

    # Negative branch
    if x_neg is not None and context_neg is not None:
        q_neg = self.norm_q(self.q(x_neg))
        k_neg = self.norm_k(self.k(context_neg))
        v_neg = self.v(context_neg)
        try:
            x_neg_out = comfy.ldm.modules.attention.optimized_attention(q_neg, k_neg, v_neg, heads=self.num_heads, transformer_options=transformer_options)
        except: #backwards compatibility for now
            x_neg_out = comfy.ldm.modules.attention.optimized_attention(q_neg, k_neg, v_neg, heads=self.num_heads)
        x = torch.cat([x_pos_out, x_neg_out], dim=0)
    else:
        x = x_pos_out

    return self.o(x)


def wan_i2v_crossattn_forward_nag(self, x, context, context_img_len, transformer_options={}, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
    context_img = context[:, :context_img_len]
    context = context[:, context_img_len:]

    q_img = self.norm_q(self.q(x))    
    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    try:
        img_x = comfy.ldm.modules.attention.optimized_attention(q_img, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options)
    except: #backwards compatibility for now
        img_x = comfy.ldm.modules.attention.optimized_attention(q_img, k_img, v_img, heads=self.num_heads)

    if context.shape[0] == 2:
        x, x_real_negative = torch.chunk(x, 2, dim=0)
        context_positive, context_negative = torch.chunk(context, 2, dim=0)
    else:
        context_positive = context
        context_negative = None
    
    q = self.norm_q(self.q(x))

    x = normalized_attention_guidance(self, q, context_positive, self.nag_context, transformer_options=transformer_options)

    if context_negative is not None:
        q_real_negative = self.norm_q(self.q(x_real_negative))
        k_real_negative = self.norm_k(self.k(context_negative))
        v_real_negative = self.v(context_negative)
        try:
            x_real_negative = comfy.ldm.modules.attention.optimized_attention(q_real_negative, k_real_negative, v_real_negative, heads=self.num_heads, transformer_options=transformer_options)
        except: #backwards compatibility for now
            x_real_negative = comfy.ldm.modules.attention.optimized_attention(q_real_negative, k_real_negative, v_real_negative, heads=self.num_heads)
        x = torch.cat([x, x_real_negative], dim=0)

    # output
    x = x + img_x
    x = self.o(x)
    return x

class WanCrossAttentionPatch:
    def __init__(self, context, nag_scale, nag_alpha, nag_tau, i2v=False, input_type="default"):
        self.nag_context = context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau
        self.i2v = i2v
        self.input_type = input_type
    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.nag_context = self.nag_context
            self_module.nag_scale = self.nag_scale
            self_module.nag_alpha = self.nag_alpha
            self_module.nag_tau = self.nag_tau
            self_module.input_type = self.input_type
            if self.i2v:
                return wan_i2v_crossattn_forward_nag(self_module, *args, **kwargs)
            else:
                return wan_crossattn_forward_nag(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)
    
class WanVideoNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.001, "tooltip": "Strength of negative guidance effect"}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Mixing coefficient in that controls the balance between the normalized guided representation and the original positive representation."}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Clipping threshold that controls how much the guided attention can deviate from the positive attention."}),
           },
           "optional": {
                "input_type": (["default", "batch"], {"tooltip": "Type of the model input"}),
           },
                                                 
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "https://github.com/ChenDarYen/Normalized-Attention-Guidance"
    EXPERIMENTAL = True

    def patch(self, model, conditioning, nag_scale, nag_alpha, nag_tau, input_type="default"):
        if nag_scale == 0:
            return (model,)
        
        device = mm.get_torch_device()
        dtype = mm.unet_dtype()

        model_clone = model.clone()

        diffusion_model = model_clone.get_model_object("diffusion_model")

        diffusion_model.text_embedding.to(device)
        context = diffusion_model.text_embedding(conditioning[0][0].to(device, dtype))

        type_str = str(type(model.model.model_config).__name__)
        i2v = True if "WAN21_I2V" in type_str else False
    
        for idx, block in enumerate(diffusion_model.blocks):
            patched_attn = WanCrossAttentionPatch(context, nag_scale, nag_alpha, nag_tau, i2v, input_type=input_type).__get__(block.cross_attn, block.__class__)
          
            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.cross_attn.forward", patched_attn)
            
        return (model_clone,)
    
class SkipLayerGuidanceWanVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             "blocks": ("STRING", {"default": "10", "multiline": False}),
                             "start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "slg"
    EXPERIMENTAL = True
    DESCRIPTION = "Simplified skip layer guidance that only skips the uncond on selected blocks"
    DEPRECATED = True
    CATEGORY = "advanced/guidance"

    def slg(self, model, start_percent, end_percent, blocks):
        def skip(args, extra_args):
            transformer_options = extra_args.get("transformer_options", {})
            original_block = extra_args["original_block"]

            if not transformer_options:
                raise ValueError("transformer_options not found in extra_args, currently SkipLayerGuidanceWanVideo only works with TeaCacheKJ")
            if start_percent <= transformer_options["current_percent"] <= end_percent:
                if args["img"].shape[0] == 2:
                    prev_img_uncond = args["img"][0].unsqueeze(0)

                    new_args = {
                        "img": args["img"][1].unsqueeze(0),
                        "txt": args["txt"][1].unsqueeze(0),
                        "vec": args["vec"][1].unsqueeze(0),
                        "pe": args["pe"][1].unsqueeze(0)
                    }
                    
                    block_out = original_block(new_args)

                    out = {
                        "img": torch.cat([prev_img_uncond, block_out["img"]], dim=0),
                        "txt": args["txt"],
                        "vec": args["vec"],
                        "pe": args["pe"]
                    }
                else:
                    if transformer_options.get("cond_or_uncond") == [0]:
                        out = original_block(args)
                    else:
                        out = args
            else:
                out = original_block(args)
            return out

        block_list = [int(x.strip()) for x in blocks.split(",")]
        blocks = [int(i) for i in block_list]
        logging.info(f"Selected blocks to skip uncond on: {blocks}")

        m = model.clone()

        for b in blocks:
            #m.set_model_patch_replace(skip, "dit", "double_block", b)
            model_options = m.model_options["transformer_options"].copy()
            if "patches_replace" not in model_options:
                model_options["patches_replace"] = {}
            else:
                model_options["patches_replace"] = model_options["patches_replace"].copy()

            if "dit" not in model_options["patches_replace"]:
                model_options["patches_replace"]["dit"] = {}
            else:
                model_options["patches_replace"]["dit"] = model_options["patches_replace"]["dit"].copy()

            block = ("double_block", b)

            model_options["patches_replace"]["dit"][block] = skip
            m.model_options["transformer_options"] = model_options
            

        return (m, )

class CFGZeroStarAndInit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "use_zero_init": ("BOOLEAN", {"default": True}),
                    "zero_init_steps": ("INT", {"default": 0, "min": 0, "tooltip": "for zero init, starts from 0 so first step is always zeroed out if use_zero_init enabled"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = "https://github.com/WeichenFan/CFG-Zero-star"
    CATEGORY = "KJNodes/experimental"
    EXPERIMENTAL = True

    def patch(self, model, use_zero_init, zero_init_steps):
        def cfg_zerostar(args):
            #zero init
            cond = args["cond"]
            timestep = args["timestep"]
            sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                for i in range(len(sigmas) - 1):
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
                else:
                    current_step_index = 0

            if (current_step_index <= zero_init_steps) and use_zero_init:
                return cond * 0
                        
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
                
            batch_size = cond.shape[0]

            positive_flat = cond.view(batch_size, -1)
            negative_flat = uncond.view(batch_size, -1)

            dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
            squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
            alpha = dot_product / squared_norm
            alpha = alpha.view(batch_size, *([1] * (len(cond.shape) - 1)))

            noise_pred = uncond * alpha + cond_scale * (cond - uncond * alpha)
            return noise_pred

        m = model.clone()
        m.set_model_sampler_cfg_function(cfg_zerostar)
        return (m, )

class GGUFLoaderKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Get GGUF models safely, fallback to empty list if unet_gguf folder doesn't exist
        try:
            gguf_models = folder_paths.get_filename_list("unet_gguf")
        except KeyError:
            gguf_models = []

        return io.Schema(
            node_id="GGUFLoaderKJ",
            category="KJNodes/experimental",
            description="Loads a GGUF model with advanced options, requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) to be installed.",
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_name", options=gguf_models),
                io.Combo.Input("extra_model_name", options=gguf_models + ["none"], default="none", tooltip="An extra gguf model to load and merge into the main model, for example VACE module"),
                io.Combo.Input("dequant_dtype", options=["default", "target", "float32", "float16", "bfloat16"], default="default"),
                io.Combo.Input("patch_dtype", options=["default", "target", "float32", "float16", "bfloat16"], default="default"),
                io.Boolean.Input("patch_on_device", default=False),
                io.Boolean.Input("enable_fp16_accumulation", default=False, tooltip="Enable torch.backends.cuda.matmul.allow_fp16_accumulation, required minimum pytorch version 2.7.1"),
                io.Combo.Input("attention_override", options=["none", "sdpa", "sageattn", "xformers", "flashattn"], default="none", tooltip="Overrides the used attention implementation, requires the respective library to be installed"),

            ],
            outputs=[io.Model.Output(),],
        )

    def attention_override_pytorch(func, *args, **kwargs):
        new_attention = comfy.ldm.modules.attention.attention_pytorch
        return new_attention.__wrapped__(*args, **kwargs)
    def attention_override_sage(func, *args, **kwargs):
        new_attention = comfy.ldm.modules.attention.attention_sage
        return new_attention.__wrapped__(*args, **kwargs)
    def attention_override_xformers(func, *args, **kwargs):
        new_attention = comfy.ldm.modules.attention.attention_xformers
        return new_attention.__wrapped__(*args, **kwargs)
    def attention_override_flash(func, *args, **kwargs):
        new_attention = comfy.ldm.modules.attention.attention_flash
        return new_attention.__wrapped__(*args, **kwargs)

    ATTENTION_OVERRIDES = {
        "sdpa": attention_override_pytorch,
        "sageattn": attention_override_sage,
        "xformers": attention_override_xformers,
        "flashattn": attention_override_flash,
    }


    @classmethod
    def _get_gguf_module(cls):
        gguf_path = os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0], "ComfyUI-GGUF")
        """Import GGUF module with version validation"""
        for module_name in ["ComfyUI-GGUF", "custom_nodes.ComfyUI-GGUF", "comfyui-gguf", "custom_nodes.comfyui-gguf", gguf_path, gguf_path.lower()]:
            try:
                module = importlib.import_module(module_name)
                return module
            except ImportError:
                continue

        raise ImportError(
            "Compatible ComfyUI-GGUF not found. "
            "Please install/update from: https://github.com/city96/ComfyUI-GGUF"
        )

    @classmethod
    def execute(cls, model_name, extra_model_name, dequant_dtype, patch_dtype, patch_on_device, attention_override, enable_fp16_accumulation):
        gguf_nodes = cls._get_gguf_module()
        ops = gguf_nodes.ops.GGMLOps()

        def set_linear_dtype(attr, value):
            if value == "default":
                setattr(ops.Linear, attr, None)
            elif value == "target":
                setattr(ops.Linear, attr, value)
            else:
                setattr(ops.Linear, attr, getattr(torch, value))

        set_linear_dtype("dequant_dtype", dequant_dtype)
        set_linear_dtype("patch_dtype", patch_dtype)

        # init model
        extra = {}
        model_path = folder_paths.get_full_path("unet", model_name)
        try:
            sd, extra = gguf_nodes.loader.gguf_sd_loader(model_path)
        except:
            sd = gguf_nodes.loader.gguf_sd_loader(model_path)

        if extra_model_name is not None and extra_model_name != "none":
            if not extra_model_name.endswith(".gguf"):
                raise ValueError("Extra model must also be a .gguf file")
            extra_model_full_path = folder_paths.get_full_path("unet", extra_model_name)
            extra_model = gguf_nodes.loader.gguf_sd_loader(extra_model_full_path)
            sd.update(extra_model)

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, metadata=extra.get("metadata", {})
        )
        if model is None:
            raise RuntimeError(f"ERROR: Could not detect model type of: {model_path}")

        model = gguf_nodes.nodes.GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device

        # attention override
        if attention_override in cls.ATTENTION_OVERRIDES:
            model.model_options["transformer_options"]["optimized_attention_override"] = cls.ATTENTION_OVERRIDES[attention_override]

        if enable_fp16_accumulation:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise RuntimeError("Failed to set fp16 accumulation, requires pytorch version 2.7.1 or higher")
        else:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = False

        return io.NodeOutput(model,)

try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
except:
    flex_attention = None
    BlockMask = None

class NABLA_AttentionKJ():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "latent": ("LATENT", {"tooltip": "Only used to get the latent shape"}),
            "window_time": ("INT", {"default": 11, "min": 1, "tooltip": "Temporal attention window size"}),
            "window_width": ("INT", {"default": 3, "min": 1, "tooltip": "Spatial attention window size"}),
            "window_height": ("INT", {"default": 3, "min": 1, "tooltip": "Spatial attention window size"}),
            "sparsity": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "torch_compile": ("BOOLEAN", {"default": True, "tooltip": "Most likely required for reasonable memory usage"})
        },
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"
    DESCRIPTION = "Experimental node for patching attention mode to use NABLA sparse attention for video models, currently only works with Kadinsky5"
    CATEGORY = "KJNodes/experimental"

    def patch(self, model, latent, window_time, window_width, window_height, sparsity, torch_compile):
        if flex_attention is None or BlockMask is None:
            raise RuntimeError("can't import flex_attention from torch.nn.attention, requires newer pytorch version")

        model_clone = model.clone()
        samples = latent["samples"]

        sparse_params = get_sparse_params(samples, window_time, window_height, window_width, sparsity)
        nabla_attention = NABLA_Attention(sparse_params)

        def attention_override_nabla(func, *args, **kwargs):
            return nabla_attention(*args, **kwargs)

        if torch_compile:
            attention_override_nabla = torch.compile(attention_override_nabla, mode="max-autotune-no-cudagraphs", dynamic=True)

        # attention override
        model_clone.model_options["transformer_options"]["optimized_attention_override"] = attention_override_nabla

        return model_clone,


class NABLA_Attention():
    def __init__(self, sparse_params):
        self.sparse_params = sparse_params

    def __call__(self, q, k, v, heads, **kwargs):
        if q.shape[-2] < 3000 or k.shape[-2] < 3000:
            return optimized_attention(q, k, v, heads, **kwargs)
        block_mask = self.nablaT_v2(q, k, self.sparse_params["sta_mask"], thr=self.sparse_params["P"])
        out = flex_attention(q, k, v, block_mask=block_mask).transpose(1, 2).contiguous().flatten(-2, -1)
        return out

    def nablaT_v2(self, q, k, sta, thr=0.9):
        # Map estimation
        BLOCK_SIZE = 64
        B, h, S, D = q.shape
        s1 = S // BLOCK_SIZE
        qa = q.reshape(B, h, s1, BLOCK_SIZE, D).mean(-2)
        ka = k.reshape(B, h, s1, BLOCK_SIZE, D).mean(-2).transpose(-2, -1)
        map = qa @ ka

        map = torch.softmax(map / math.sqrt(D), dim=-1)
        # Map binarization
        vals, inds = map.sort(-1)
        cvals = vals.cumsum_(-1)
        mask = (cvals >= 1 - thr).int()
        mask = mask.gather(-1, inds.argsort(-1))

        mask = torch.logical_or(mask, sta)

        # BlockMask creation
        kv_nb = mask.sum(-1).to(torch.int32)
        kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
        return BlockMask.from_kv_blocks(torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=BLOCK_SIZE, mask_mod=None)

def fast_sta_nabla(T, H, W, wT=3, wH=3, wW=3):
    l = torch.Tensor([T, H, W]).amax()
    r = torch.arange(0, l, 1, dtype=torch.int16, device=mm.get_torch_device())
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
    sta_t, sta_h, sta_w = (
        mat[:T, :T].flatten(),
        mat[:H, :H].flatten(),
        mat[:W, :W].flatten(),
    )
    sta_t = sta_t <= wT // 2
    sta_h = sta_h <= wH // 2
    sta_w = sta_w <= wW // 2
    sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
    sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H * W, H * W).transpose(1, 2)
    return sta.reshape(T * H * W, T * H * W)


def get_sparse_params(x, wT, wH, wW, sparsity=0.9):
    B, C, T, H, W = x.shape
    #print("x shape:", x.shape)
    patch_size = (1, 2, 2)
    T, H, W = (
        T // patch_size[0],
        H // patch_size[1],
        W // patch_size[2],
    )
    sta_mask = fast_sta_nabla(T, H // 8, W // 8, wT, wH, wW)
    sparse_params = {
        "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
        "to_fractal": True,
        "P": sparsity,
        "wT": wT,
        "wH": wH,
        "wW": wW,
        "add_sta": True,
        "visual_shape": (T, H, W),
        "method": "topcdf",
    }

    return sparse_params

from comfy.comfy_types.node_typing import IO
class StartRecordCUDAMemoryHistory():
    # @classmethod
    # def IS_CHANGED(s):
    #     return True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": (IO.ANY,),
                "enabled": (["all", "state", "None"], {"default": "all", "tooltip": "None: disable, 'state': keep info for allocated memory, 'all': keep history of all alloc/free calls"}),
                "context": (["all", "state", "alloc", "None"], {"default": "all", "tooltip": "None: no tracebacks, 'state': tracebacks for allocated memory, 'alloc': for alloc calls, 'all': for free calls"}),
                "stacks": (["python", "all"], {"default": "all", "tooltip": "'python': Python/TorchScript/inductor frames, 'all': also C++ frames"}),
                "max_entries": ("INT", {"default": 100000, "min": 1000, "max": 10000000, "tooltip": "Maximum number of entries to record"}),
            },
        }

    RETURN_TYPES = (IO.ANY, )
    RETURN_NAMES = ("input", "output_path",)
    FUNCTION = "start"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "THIS NODE ALWAYS RUNS. Starts recording CUDA memory allocation history, can be ended and saved with EndRecordCUDAMemoryHistory. "

    def start(self, input, enabled, context, stacks, max_entries):
        mm.soft_empty_cache()
        torch.cuda.reset_peak_memory_stats(mm.get_torch_device())
        torch.cuda.memory._record_memory_history(
            max_entries=max_entries,
            enabled=enabled if enabled != "None" else None,
            context=context if context != "None" else None,
            stacks=stacks
        )
        return input,

class EndRecordCUDAMemoryHistory():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input": (IO.ANY,),
            "output_path": ("STRING", {"default": "comfy_cuda_memory_history"}, "Base path for saving the CUDA memory history file, timestamp and .pt extension will be added"),
        },
        }

    RETURN_TYPES = (IO.ANY, "STRING",)
    RETURN_NAMES = ("input", "output_path",)
    FUNCTION = "end"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "Records CUDA memory allocation history between start and end, saves to a file that can be analyzed here: https://docs.pytorch.org/memory_viz or with VisualizeCUDAMemoryHistory node"

    def end(self, input, output_path):
        mm.soft_empty_cache()
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_path}{time}.pt"
        torch.cuda.memory._dump_snapshot(output_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        return input, output_path


try:
    from server import PromptServer
except:
    PromptServer = None

class VisualizeCUDAMemoryHistory():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "snapshot_path": ("STRING", ),
        },
         "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "visualize"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "Visualizes a CUDA memory allocation history file, opens in browser"
    OUTPUT_NODE = True

    def visualize(self, snapshot_path, unique_id):
        import pickle
        from torch.cuda import _memory_viz
        import uuid

        from folder_paths import get_output_directory
        output_dir = get_output_directory()

        with open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)

        html = _memory_viz.trace_plot(snapshot)
        html_filename = f"cuda_memory_history_{uuid.uuid4().hex}.html"
        output_path = os.path.join(output_dir, "memory_history", html_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        api_url = f"http://localhost:8188/api/view?type=output&filename={html_filename}&subfolder=memory_history"

        # Progress UI
        if unique_id and PromptServer is not None:
            try:
                PromptServer.instance.send_progress_text(
                    api_url,
                    unique_id
                )
            except:
                pass

        return api_url,


class ModelMemoryUseReportPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = "Adds callbacks to model to report memory usage during after sampling"
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, model):
        model_clone = model.clone()
        device = mm.get_torch_device()

        def reset_mem_usage(model):
            torch.cuda.reset_peak_memory_stats(device)
        def report_mem_usage(model):
            max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            logging.info(f"Sampling max allocated memory: {max_memory=:.3f} GB")
            logging.info(f"Sampling max reserved memory: {max_reserved=:.3f} GB")

        model_clone.add_callback(CallbacksMP.ON_PRE_RUN, reset_mem_usage)
        model_clone.add_callback(CallbacksMP.ON_CLEANUP, report_mem_usage)

        return (model_clone,)


class MemoryUsageFactorAdjustWrapper:
    def __init__(self, memory_usage_factor, original_factor):
        self.memory_usage_factor = memory_usage_factor
        self.original_factor = original_factor

    def __call__(self, executor, model, noise_shape: torch.Tensor, *args, **kwargs):
        m = model.clone()
        m.model.memory_usage_factor = self.memory_usage_factor
        logging.info(f"Temporarily set memory usage factor to {self.memory_usage_factor}")
        try:
            result = executor(m, noise_shape, *args, **kwargs)
        finally:
            logging.info(f"Model memory usage calculated, restoring original memory usage factor: {self.original_factor}")
            m.model.memory_usage_factor = self.original_factor
        return result

class ModelMemoryUsageFactorOverride:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "memory_usage_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.001}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = "Overrides the memory usage factor of the model during sampling."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, model, memory_usage_factor):
        model_clone = model.clone()
        original_memory_usage_factor = model_clone.model.memory_usage_factor
        logging.info(f"Original memory usage factor: {original_memory_usage_factor}")

        wrapper = MemoryUsageFactorAdjustWrapper(memory_usage_factor, original_memory_usage_factor)
        model_clone.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            "memory_usage_factor_adjust_prepare_sampling",
            wrapper
        )
        return (model_clone,)

def wan_ffn_chunked_forward(self, x):
    if x.shape[1] > self.dim_threshold:
        chunks = torch.chunk(x, self.num_chunks, dim=1)
        output_chunks = []
        for chunk in chunks:
            output_chunks.append(torch.nn.Sequential.forward(self, chunk))
        chunked = torch.cat(output_chunks, dim=1)
        return chunked
    else:
        return torch.nn.Sequential.forward(self, x)

class WanffnChunkPatch:
    def __init__(self, num_chunks, dim_threshold=4096):
        self.num_chunks = num_chunks
        self.dim_threshold = dim_threshold

    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.num_chunks = self.num_chunks
            self_module.dim_threshold = self.dim_threshold
            return wan_ffn_chunked_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)

class WanChunkFeedForward(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanChunkFeedForward",
            display_name="Wan Chunk FeedForward",
            category="KJNodes/wan",
            description="EXPERIMENTAL AND MAY CHANGE THE MODEL OUTPUT!! Chunks feedforward activations to reduce peak VRAM usage.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("chunks", default=2, min=1, max=100, step=1, tooltip="Number of chunks to split the feedforward activations into to reduce peak VRAM usage."),
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

        for idx, block in enumerate(diffusion_model.blocks):
            patched_ffn = WanffnChunkPatch(chunks, dim_threshold).__get__(block.ffn, block.__class__)
            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.ffn.forward", patched_ffn)

        return io.NodeOutput(model_clone)

from comfy.samplers import KSAMPLER
from comfy.k_diffusion.sampling import to_d
from tqdm import tqdm
def sample_selfrefinevideo(model, x, sigmas, stochastic_step_map, certain_percentage=0.999, uncertainty_threshold=0.25, extra_args=None, callback=None, disable=None, verbose=False, video_shape=None, seed=None):
    extra_args = {} if extra_args is None else extra_args
    sigma_in = x.new_ones([x.shape[0]])

    if seed is not None:
        generator = torch.Generator(torch.device("cpu")).manual_seed(seed)

    pbar = tqdm(total=len(sigmas) - 1, disable=disable, desc="Sampling")

    for i in range(len(sigmas) - 1):

        # Get stochastic steps for this noise level
        current_num_anneal_steps = stochastic_step_map.get(i, 0)
        use_stochastic = current_num_anneal_steps > 0
        m = current_num_anneal_steps + 1 if use_stochastic else 1

        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        prev_certain_mask = None
        prev_denoised = None
        prev_denoised_full = None
        prev_x_next = None
        prev_x_next_video = None
        is_certain = False

        for ii in range(m):
            if m > 1:
                pbar.set_description(f"Step {i}/{len(sigmas)-1} (substep {ii+1}/{m})")
            # Early exit if certain threshold reached
            if is_certain:
                x = prev_x_next
                break

            # Determine input
            noise = torch.randn(x.shape, device=torch.device("cpu"), generator=generator).to(x)
            x_in = x if ii == 0 else (1.0 - sigma) * prev_denoised_full + sigma * noise
            if ii > 0:
                x = x_in

            denoised = model(x_in, sigmas[i] * sigma_in, **extra_args)

            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

            # Compute next latents
            d = to_d(x, sigma, denoised)
            x_next = x + (sigma_next - sigma) * d

            # Separate video and audio if joint model
            if d.ndim == 3 and video_shape is not None:
                cut = math.prod(video_shape[1:])
                denoised_video = denoised[:, :, :cut].reshape([denoised.shape[0]] + list(video_shape)[1:])
                x_next_video = x_next[:, :, :cut].reshape([denoised.shape[0]] + list(video_shape)[1:])
                denoised_audio = denoised[:, :, cut:]
                x_next_audio = x_next[:, :, cut:]
                if verbose:
                    tqdm.write(f"Video shape: {denoised_video.shape}, Audio shape: {denoised_audio.shape}")
            else:
                denoised_video = denoised
                x_next_video = x_next
                denoised_audio = None
                x_next_audio = None

            # Stochastic sampling with uncertainty masking
            if use_stochastic and prev_denoised is not None:
                # Compute uncertainty and masking on video part
                diff = denoised_video - prev_denoised
                uncertainty = torch.sqrt(torch.sum(diff ** 2, dim=1)) / denoised_video.shape[1]
                certain_mask = uncertainty < uncertainty_threshold

                if verbose:
                    tqdm.write(f"Step {i}/{len(sigmas)-1} substep {ii+1}/{m}:")
                    tqdm.write(f"Uncertainty: min {uncertainty.min():.4f}, max {uncertainty.max():.4f}, threshold {uncertainty_threshold}")
                    tqdm.write(f"Certain pixels: {certain_mask.sum()}/{certain_mask.numel()} = {certain_mask.sum()/certain_mask.numel():.4f}")

                # Update certain mask (union with previous)
                if prev_certain_mask is not None:
                    certain_mask = certain_mask | prev_certain_mask

                # Check certainty threshold
                if certain_mask.sum() / certain_mask.numel() > certain_percentage:
                    is_certain = True
                    if verbose:
                        tqdm.write(f"{ii}/{current_num_anneal_steps}: Certain region is more than {certain_percentage}, we are certain")

                # Apply masking to video
                certain_mask_float = certain_mask.float().unsqueeze(1)
                x_next_video = certain_mask_float * prev_x_next_video + (1.0 - certain_mask_float) * x_next_video
                denoised_video = certain_mask_float * prev_denoised + (1.0 - certain_mask_float) * denoised_video

                # Reconstruct full latents by replacing the video portion
                if x_next_audio is not None:
                    # Flatten masked video back to match original format and replace video portion
                    x_next = x_next.clone()
                    x_next[:, :, :cut] = x_next_video.reshape([x_next_video.shape[0], x_next.shape[1], -1])
                    # Also reconstruct full denoised for next iteration input
                    denoised_full = denoised.clone()
                    denoised_full[:, :, :cut] = denoised_video.reshape([denoised_video.shape[0], denoised.shape[1], -1])
                else:
                    # No audio separation
                    x_next = x_next_video
                    denoised_full = denoised_video

                prev_certain_mask = certain_mask
                prev_denoised = denoised_video
                prev_denoised_full = denoised_full
                prev_x_next_video = x_next_video
                prev_x_next = x_next
            elif use_stochastic:
                # For first stochastic step, create denoised_full if we have audio
                if x_next_audio is not None:
                    denoised_full = denoised.clone()
                    denoised_full[:, :, :cut] = denoised_video.reshape([denoised_video.shape[0], denoised.shape[1], -1])
                else:
                    denoised_full = denoised_video

                prev_certain_mask = None
                prev_denoised = denoised_video
                prev_denoised_full = denoised_full
                prev_x_next_video = x_next_video
                prev_x_next = x_next

            # Update x for final step
            if use_stochastic and ii == m - 1:
                x = prev_x_next
            elif not use_stochastic:
                x = x_next

        pbar.update(1)
        if m == 1:
            pbar.set_description("Sampling")
    pbar.close()
    return x

class SamplerSelfRefineVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        default_ranges = [
            (2, 5, 3),   # Range 1
            (6, 14, 1),  # Range 2
        ]

        options = []

        # Option 1: 2 ranges
        range_inputs_2 = []
        for i in range(1, 3):
            start_default, end_default, steps_default = default_ranges[i - 1]
            range_inputs_2.extend([
                io.Int.Input(f"start_step{i}", default=start_default, min=0, max=999, step=1, tooltip=f"Start step for range {i}"),
                io.Int.Input(f"end_step{i}", default=end_default, min=0, max=999, step=1, tooltip=f"End step for range {i}"),
                io.Int.Input(f"steps_{i}", default=steps_default, min=1, max=100, step=1, tooltip=f"Number of P&P steps for range {i}"),
            ])
        options.append(io.DynamicCombo.Option(key="2 ranges", inputs=range_inputs_2))

        # Option 2: 1 range
        range_inputs_1 = []
        for i in range(1, 2):
            start_default, end_default, steps_default = default_ranges[i - 1]
            range_inputs_1.extend([
                io.Int.Input(f"start_step{i}", default=start_default, min=0, max=999, step=1, tooltip=f"Start step for range {i}"),
                io.Int.Input(f"end_step{i}", default=end_default, min=0, max=999, step=1, tooltip=f"End step for range {i}"),
                io.Int.Input(f"steps_{i}", default=steps_default, min=1, max=100, step=1, tooltip=f"Number of P&P steps for range {i}"),
            ])
        options.append(io.DynamicCombo.Option(key="1 range", inputs=range_inputs_1))

        # Option 3: Manual string input
        options.append(io.DynamicCombo.Option(
            key="from_string",
            inputs=[
                io.String.Input(
                    "stochastic_plan",
                    default="2-5:3,6-14:1",
                    multiline=True,
                    tooltip="Format: 'start-end:steps,start-end:steps' e.g. '2-5:3,6-14:1'"
                )
            ]
        ))
        return io.Schema(
            node_id="SamplerSelfRefineVideo",
            category="KJNodes/samplers",
            description="Attempt to implement https://github.com/agwmon/self-refine-video, for testing only, MAY NOT WORK AS INTENDED.",
            is_experimental=True,
            inputs=[
                io.DynamicCombo.Input("input_mode", options=options, tooltip="How to configure the step plan"),
                io.Float.Input("certain_percentage", default=0.999, min=0.0, max=1.0, step=0.001, round=False, tooltip="Percentage of certain pixels to consider the frame as certain and skip further refinement"),
                io.Float.Input("uncertainty_threshold", default=0.2, min=0.0, max=1.0, step=0.01, round=False, tooltip="Threshold of uncertainty to consider a pixel uncertain"),
                io.Boolean.Input("verbose", default=False, tooltip="Enable verbose logging during sampling"),
                io.Latent.Input("latent", optional=True, tooltip="Optional latent input to get input shape for LTX2 audio/video separation"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, step=1, tooltip="Seed for stochastic sampling"),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, input_mode, certain_percentage, uncertainty_threshold, seed, verbose, latent=None) -> io.NodeOutput:
        video_shape = None
        if latent is not None:
            video_shape = latent["samples"].shape

        range_keys = sorted([k for k in input_mode.keys() if k.startswith('start_step')])
        stochastic_step_map = {}
        if "stochastic_plan" in input_mode:
            # Parse manual string format: "2-5:3,6-14:1"
            plan_str = input_mode["stochastic_plan"]
            ranges = plan_str.split(",")
            for range_spec in ranges:
                range_spec = range_spec.strip()
                if not range_spec:
                    continue
                try:
                    range_part, steps_part = range_spec.split(":")
                    start, end = range_part.split("-")
                    start, end, steps = int(start), int(end), int(steps_part)
                    for idx in range(start, end + 1):
                        stochastic_step_map[idx] = steps
                except ValueError:
                    raise ValueError(f"Invalid format in stochastic_plan: '{range_spec}'. Expected format: 'start-end:steps'")
        else:
            range_keys = [k for k in input_mode.keys() if k.startswith('start_step')]
            for start_key in range_keys:
                i = start_key.replace('start_step', '')
                start = input_mode.get(f"start_step{i}")
                end = input_mode.get(f"end_step{i}")
                steps = input_mode.get(f"steps_{i}")

                if start is not None and end is not None and steps is not None:
                    for idx in range(start, end + 1):
                        stochastic_step_map[idx] = steps

        sampler = KSAMPLER(sample_selfrefinevideo, {
            "stochastic_step_map": stochastic_step_map,
            "certain_percentage": certain_percentage,
            "uncertainty_threshold": uncertainty_threshold,
            "verbose": verbose,
            "video_shape": video_shape,
            "seed": seed,
        })
        return io.NodeOutput(sampler)
