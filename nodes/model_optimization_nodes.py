from comfy.ldm.modules import attention as comfy_attention
import comfy.model_patcher
import comfy.utils
import comfy.sd
import torch
import folder_paths
orig_attention = comfy_attention.optimized_attention

class BaseLoaderKJ:
    original_linear = None
    cublas_patched = False

    def _patch_modules(self, patch_cublaslinear, sage_attention):
        from comfy.ops import disable_weight_init, CastWeightBiasOp, cast_bias_weight

        global orig_attention
        if 'orig_attention' not in globals():
            orig_attention = comfy_attention.optimized_attention

        if sage_attention != "disabled":
            from sageattention import sageattn
            def set_sage_func(sage_attention):
                if sage_attention == "auto":
                    def func(q, k, v, is_causal=False, attn_mask=None):
                        return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
                    from sageattention import sageattn_qk_int8_pv_fp16_cuda
                    def func(q, k, v, is_causal=False, attn_mask=None):
                        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32")
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
                    from sageattention import sageattn_qk_int8_pv_fp16_triton
                    def func(q, k, v, is_causal=False, attn_mask=None):
                        return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
                    from sageattention import sageattn_qk_int8_pv_fp8_cuda
                    def func(q, k, v, is_causal=False, attn_mask=None):
                        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32")
                    return func
                else:
                    def func(q, k, v, is_causal=False, attn_mask=None):
                        return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask)
                    return func

            sage_func = set_sage_func(sage_attention)

            @torch.compiler.disable()
            def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
                if skip_reshape:
                    b, _, _, dim_head = q.shape
                else:
                    b, _, dim_head = q.shape
                    dim_head //= heads
                if dim_head not in (64, 96, 128) or not (k.shape == q.shape and v.shape == q.shape):
                    return orig_attention(q, k, v, heads, mask, attn_precision, skip_reshape)
                if not skip_reshape:
                    q, k, v = map(
                        lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
                        (q, k, v),
                    )
                return (
                    sage_func(q, k, v, is_causal=False, attn_mask=mask)
                    .transpose(1, 2)
                    .reshape(b, -1, heads * dim_head)
                )

            comfy_attention.optimized_attention = attention_sage
        else:
            comfy_attention.optimized_attention = orig_attention

        if patch_cublaslinear:
            if not BaseLoaderKJ.cublas_patched:
                BaseLoaderKJ.original_linear = disable_weight_init.Linear
                try:
                    from cublas_ops import CublasLinear
                except ImportError:
                    raise Exception("Can't import 'torch-cublas-hgemm', install it from here https://github.com/aredden/torch-cublas-hgemm")

                class PatchedLinear(CublasLinear, CastWeightBiasOp):
                    def reset_parameters(self):
                        pass

                    def forward_comfy_cast_weights(self, input):
                        weight, bias = cast_bias_weight(self, input)
                        return torch.nn.functional.linear(input, weight, bias)

                    def forward(self, *args, **kwargs):
                        if self.comfy_cast_weights:
                            return self.forward_comfy_cast_weights(*args, **kwargs)
                        else:
                            return super().forward(*args, **kwargs)

                disable_weight_init.Linear = PatchedLinear
                BaseLoaderKJ.cublas_patched = True
        else:
            if BaseLoaderKJ.cublas_patched:
                disable_weight_init.Linear = BaseLoaderKJ.original_linear
                BaseLoaderKJ.cublas_patched = False

class CheckpointLoaderKJ(BaseLoaderKJ):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            "patch_cublaslinear": ("BOOLEAN", {"default": True, "tooltip": "Enable or disable the patching, won't take effect on already loaded models!"}),
            "sage_attention": ("BOOLEAN", {"default": False, "tooltip": "Patch comfy attention to use sageattn."}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "patch"
    OUTPUT_NODE = True
    DESCRIPTION = "Experimental node for patching torch.nn.Linear with CublasLinear."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, ckpt_name, patch_cublaslinear, sage_attention):
        self._patch_modules(patch_cublaslinear, sage_attention)
        from nodes import CheckpointLoaderSimple
        model, clip, vae = CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)
        return model, clip, vae

class DiffusionModelLoaderKJ(BaseLoaderKJ):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The name of the checkpoint (model) to load."}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
            "patch_cublaslinear": ("BOOLEAN", {"default": True, "tooltip": "Enable or disable the patching, won't take effect on already loaded models!"}),
            "sage_attention": (["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"], {"default": False, "tooltip": "Patch comfy attention to use sageattn."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_and_load"
    OUTPUT_NODE = True
    DESCRIPTION = "Node for patching torch.nn.Linear with CublasLinear."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch_and_load(self, ckpt_name, weight_dtype, patch_cublaslinear, sage_attention):
        self._patch_modules(patch_cublaslinear, sage_attention)
        from nodes import UNETLoader
        model, = UNETLoader.load_unet(self, ckpt_name, weight_dtype)
        return (model,)

    

original_patch_model = comfy.model_patcher.ModelPatcher.patch_model
original_load_lora_for_models = comfy.sd.load_lora_for_models

def patched_patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
    
    if lowvram_model_memory == 0:
        full_load = True
    else:
        full_load = False

    if load_weights:
        self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
    for k in self.object_patches:
        old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
        if k not in self.object_patches_backup:
            self.object_patches_backup[k] = old
    
    return self.model

def patched_load_lora_for_models(model, clip, lora, strength_model, strength_clip):

    patch_keys = list(model.object_patches_backup.keys())
    for k in patch_keys:
        #print("backing up object patch: ", k)
        comfy.utils.set_attr(model.model, k, model.object_patches_backup[k])

    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded = comfy.lora.load_lora(lora, key_map)
    #print(temp_object_patches_backup)
   
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.add_patches(loaded, strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED {}".format(x))

    if patch_keys:
        if hasattr(model.model, "compile_settings"):
            compile_settings = getattr(model.model, "compile_settings")
            print("compile_settings: ", compile_settings)
            for k in patch_keys:
                if "diffusion_model." in k:
                    # Remove the prefix to get the attribute path
                    key = k.replace('diffusion_model.', '')
                    attributes = key.split('.')
                    # Start with the diffusion_model object
                    block = model.get_model_object("diffusion_model")
                    # Navigate through the attributes to get to the block
                    for attr in attributes:
                        if attr.isdigit():
                            block = block[int(attr)]
                        else:
                            block = getattr(block, attr)
                    # Compile the block
                    compiled_block = torch.compile(block, mode=compile_settings["mode"], dynamic=compile_settings["dynamic"], fullgraph=compile_settings["fullgraph"], backend=compile_settings["backend"])
                    # Add the compiled block back as an object patch
                    model.add_object_patch(k, compiled_block)
    return (new_modelpatcher, new_clip)

class PatchModelPatcherOrder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "patch_order": (["object_patch_first", "weight_patch_first"], {"default": "weight_patch_first", "tooltip": "Patch the comfy patch_model function to load weight patches (LoRAs) before compiling the model"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    DESCTIPTION = "Patch the comfy patch_model function patching order, useful for torch.compile (used as object_patch) as it should come last if you want to use LoRAs with compile"
    EXPERIMENTAL = True

    def patch(self, model, patch_order):
        comfy.model_patcher.ModelPatcher.temp_object_patches_backup = {}
        if patch_order == "weight_patch_first":            
            comfy.model_patcher.ModelPatcher.patch_model = patched_patch_model
            comfy.sd.load_lora_for_models = patched_load_lora_for_models
        else:
            comfy.model_patcher.ModelPatcher.patch_model = original_patch_model
            comfy.sd.load_lora_for_models = original_load_lora_for_models
        
        return model,

class TorchCompileModelFluxAdvanced:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "double_blocks": ("STRING", {"default": "0-18", "multiline": True}),
                    "single_blocks": ("STRING", {"default": "0-37", "multiline": True}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/experimental"
    EXPERIMENTAL = True

    def parse_blocks(self, blocks_str):
        blocks = []
        for part in blocks_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                blocks.extend(range(start, end + 1))
            else:
                blocks.append(int(part))
        return blocks

    def patch(self, model, backend, mode, fullgraph, single_blocks, double_blocks, dynamic):
        single_block_list = self.parse_blocks(single_blocks)
        double_block_list = self.parse_blocks(double_blocks)
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        
        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.double_blocks):
                    if i in double_block_list:
                        #print("Compiling double_block", i)
                        m.add_object_patch(f"diffusion_model.double_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                for i, block in enumerate(diffusion_model.single_blocks):
                    if i in single_block_list:
                        #print("Compiling single block", i)
                        m.add_object_patch(f"diffusion_model.single_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model")
        
        return (m, )
        # rest of the layers that are not patched
        # diffusion_model.final_layer = torch.compile(diffusion_model.final_layer, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.guidance_in = torch.compile(diffusion_model.guidance_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.img_in = torch.compile(diffusion_model.img_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.time_in = torch.compile(diffusion_model.time_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.txt_in = torch.compile(diffusion_model.txt_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.vector_in = torch.compile(diffusion_model.vector_in, mode=mode, fullgraph=fullgraph, backend=backend)

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

    CATEGORY = "KJNodes/experimental"
    EXPERIMENTAL = True

    def compile(self, vae, backend, mode, fullgraph, compile_encoder, compile_decoder):
        if compile_encoder:
            if not self._compiled_encoder:
                try:
                    vae.first_stage_model.encoder = torch.compile(vae.first_stage_model.encoder, mode=mode, fullgraph=fullgraph, backend=backend)
                    self._compiled_encoder = True
                except:
                    raise RuntimeError("Failed to compile model")
        if compile_decoder:
            if not self._compiled_decoder:
                try:
                    vae.first_stage_model.decoder = torch.compile(vae.first_stage_model.decoder, mode=mode, fullgraph=fullgraph, backend=backend)
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

    CATEGORY = "KJNodes/experimental"
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

class TorchCompileLTXModel:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/experimental"
    EXPERIMENTAL = True

    def patch(self, model, backend, mode, fullgraph, dynamic):
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        
        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.transformer_blocks):
                        compiled_block = torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend)
                        m.add_object_patch(f"diffusion_model.transformer_blocks.{i}", compiled_block)
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
               
            except:
                raise RuntimeError("Failed to compile model")           
        
        return (m, )