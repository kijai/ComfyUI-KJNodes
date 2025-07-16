import torch
import comfy.model_management
import comfy.utils
import folder_paths
import os
import logging
from enum import Enum
from tqdm import tqdm

CLAMP_QUANTILE = 0.99

def extract_lora(diff, rank, algorithm, lowrank_iters=7):
    conv2d = (len(diff.shape) == 4)
    kernel_size = None if not conv2d else diff.size()[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]
    rank = min(rank, in_dim, out_dim)

    if conv2d:
        if conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()

    diff_float = diff.float()

    if algorithm == "svd_lowrank":
        U, S, V = torch.svd_lowrank(diff_float, q=rank, niter=lowrank_iters)
        U = U @ torch.diag(S)
        Vh = V.t()
    else:
        U, S, Vh = torch.linalg.svd(diff_float)
        U = U[:, :rank]
        S = S[:rank]
        U = U @ torch.diag(S)
        Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)
    if conv2d:
        U = U.reshape(out_dim, rank, 1, 1)
        Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
    return (U, Vh)

class LORAType(Enum):
    STANDARD = 0
    FULL_DIFF = 1

LORA_TYPES = {"standard": LORAType.STANDARD,
              "full_diff": LORAType.FULL_DIFF}

def calc_lora_model(model_diff, rank, prefix_model, prefix_lora, output_sd, lora_type, algorithm, lowrank_iters, out_dtype, bias_diff=False):
    comfy.model_management.load_models_gpu([model_diff], force_patch_weights=True)
    sd = model_diff.model_state_dict(filter_prefix=prefix_model)

    # Get total number of keys to process for progress bar
    total_keys = len([k for k in sd if k.endswith(".weight") or (bias_diff and k.endswith(".bias"))])
    
    # Create progress bar
    progress_bar = tqdm(total=total_keys, desc=f"Extracting LoRA ({prefix_lora.strip('.')})")
    comfy_pbar = comfy.utils.ProgressBar(total_keys)

    for k in sd:
        if k.endswith(".weight"):
            weight_diff = sd[k]
            if weight_diff.ndim == 5:
                logging.info(f"Skipping 5D tensor for key {k}") #skip patch embed
                progress_bar.update(1)
                comfy_pbar.update(1)
                continue
            if lora_type == LORAType.STANDARD:
                if weight_diff.ndim < 2:
                    if bias_diff:
                        output_sd["{}{}.diff".format(prefix_lora, k[len(prefix_model):-7])] = weight_diff.contiguous().to(out_dtype).cpu()
                    progress_bar.update(1)
                    comfy_pbar.update(1)
                    continue
                try:
                    out = extract_lora(weight_diff, rank, algorithm, lowrank_iters)
                    output_sd["{}{}.lora_up.weight".format(prefix_lora, k[len(prefix_model):-7])] = out[0].contiguous().to(out_dtype).cpu()
                    output_sd["{}{}.lora_down.weight".format(prefix_lora, k[len(prefix_model):-7])] = out[1].contiguous().to(out_dtype).cpu()
                except:
                    logging.warning("Could not generate lora weights for key {}, is the weight difference a zero?".format(k))
            elif lora_type == LORAType.FULL_DIFF:
                output_sd["{}{}.diff".format(prefix_lora, k[len(prefix_model):-7])] = weight_diff.contiguous().to(out_dtype).cpu()
            
            progress_bar.update(1)
            comfy_pbar.update(1)

        elif bias_diff and k.endswith(".bias"):
            output_sd["{}{}.diff_b".format(prefix_lora, k[len(prefix_model):-5])] = sd[k].contiguous().to(out_dtype).cpu()
            progress_bar.update(1)
            comfy_pbar.update(1)
    progress_bar.close()
    return output_sd

class LoraExtractKJ:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "finetuned_model": ("MODEL",),
                    "original_model": ("MODEL",),
                    "filename_prefix": ("STRING", {"default": "loras/ComfyUI_extracted_lora"}),
                    "rank": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
                    "lora_type": (tuple(LORA_TYPES.keys()),),
                    "algorithm": (["svd_linalg", "svd_lowrank"], {"default": "svd", "tooltip": "SVD algorithm to use, svd_lowrank is faster but less accurate."}),
                    "lowrank_iters": ("INT", {"default": 7, "min": 1, "max": 100, "step": 1, "tooltip": "The number of subspace iterations for lowrank SVD algorithm."}),
                    "output_dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                    "bias_diff": ("BOOLEAN", {"default": True}),
                },

    }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "KJNodes/lora"

    def save(self, finetuned_model, original_model, filename_prefix, rank, lora_type, algorithm, lowrank_iters, output_dtype, bias_diff):
        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[output_dtype]
        m = finetuned_model.clone()
        kp = original_model.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, - 1.0, 1.0)
        model_diff = m

        lora_type = LORA_TYPES.get(lora_type)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        output_sd = {}
        if model_diff is not None:
            output_sd = calc_lora_model(model_diff, rank, "diffusion_model.", "diffusion_model.", output_sd, lora_type, algorithm, lowrank_iters, dtype, bias_diff=bias_diff)

        output_checkpoint = f"{filename}_rank{rank}_{output_dtype}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=None)
        return {}

NODE_CLASS_MAPPINGS = {
    "LoraExtractKJ": LoraExtractKJ
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraExtractKJ": "LoraExtractKJ"
}
