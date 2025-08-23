import torch
import comfy.model_management
import comfy.utils
import folder_paths
import os
import logging
from tqdm import tqdm

device = comfy.model_management.get_torch_device()

CLAMP_QUANTILE = 0.99

def extract_lora(diff, key, rank, algorithm, lora_type, lowrank_iters=7, adaptive_param=1.0, clamp_quantile=True):
    """
    Extracts LoRA weights from a weight difference tensor using SVD.
    """
    conv2d = (len(diff.shape) == 4)
    kernel_size = None if not conv2d else diff.size()[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]

    if conv2d:
        if conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()

    diff_float = diff.float()
    if algorithm == "svd_lowrank":
        U, S, V = torch.svd_lowrank(diff_float, q=min(rank, in_dim, out_dim), niter=lowrank_iters)
        U = U @ torch.diag(S)
        Vh = V.t()
    else:
        #torch.linalg.svdvals() 
        U, S, Vh = torch.linalg.svd(diff_float)
        # Flexible rank selection logic like locon: https://github.com/KohakuBlueleaf/LyCORIS/blob/main/tools/extract_locon.py
        if "adaptive" in lora_type:
            if lora_type == "adaptive_ratio":
                min_s = torch.max(S) * adaptive_param
                lora_rank = torch.sum(S > min_s).item()
            elif lora_type == "adaptive_energy":
                energy = torch.cumsum(S**2, dim=0)
                total_energy = torch.sum(S**2)
                threshold = adaptive_param * total_energy  # e.g., adaptive_param=0.95 for 95%
                lora_rank = torch.sum(energy < threshold).item() + 1
            elif lora_type == "adaptive_quantile":
                s_cum = torch.cumsum(S, dim=0)
                min_cum_sum = adaptive_param * torch.sum(S)
                lora_rank = torch.sum(s_cum < min_cum_sum).item()
            print(f"{key} Extracted LoRA rank: {lora_rank}")
        else:
            lora_rank = rank

        lora_rank = max(1, lora_rank)
        lora_rank = min(out_dim, in_dim, lora_rank)
        
        U = U[:, :lora_rank]
        S = S[:lora_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:lora_rank, :]

    if clamp_quantile:
        dist = torch.cat([U.flatten(), Vh.flatten()])
        if dist.numel() > 100_000:
            # Sample 100,000 elements for quantile estimation
            idx = torch.randperm(dist.numel(), device=dist.device)[:100_000]
            dist_sample = dist[idx]
            hi_val = torch.quantile(dist_sample, CLAMP_QUANTILE)
        else:
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val

        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)
    if conv2d:
        U = U.reshape(out_dim, lora_rank, 1, 1)
        Vh = Vh.reshape(lora_rank, in_dim, kernel_size[0], kernel_size[1])
    return (U, Vh)


def calc_lora_model(model_diff, rank, prefix_model, prefix_lora, output_sd, lora_type, algorithm, lowrank_iters, out_dtype, bias_diff=False, adaptive_param=1.0, clamp_quantile=True):
    comfy.model_management.load_models_gpu([model_diff], force_patch_weights=True)
    model_diff.model.diffusion_model.cpu()
    sd = model_diff.model_state_dict(filter_prefix=prefix_model)
    del model_diff
    comfy.model_management.soft_empty_cache()
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            sd[k] = v.cpu()

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
            if lora_type != "full":
                if weight_diff.ndim < 2:
                    if bias_diff:
                        output_sd["{}{}.diff".format(prefix_lora, k[len(prefix_model):-7])] = weight_diff.contiguous().to(out_dtype).cpu()
                    progress_bar.update(1)
                    comfy_pbar.update(1)
                    continue
                try:
                    out = extract_lora(weight_diff.to(device), k, rank, algorithm, lora_type, lowrank_iters=lowrank_iters, adaptive_param=adaptive_param, clamp_quantile=clamp_quantile)
                    output_sd["{}{}.lora_up.weight".format(prefix_lora, k[len(prefix_model):-7])] = out[0].contiguous().to(out_dtype).cpu()
                    output_sd["{}{}.lora_down.weight".format(prefix_lora, k[len(prefix_model):-7])] = out[1].contiguous().to(out_dtype).cpu()
                except Exception as e:
                    logging.warning(f"Could not generate lora weights for key {k}, error {e}")
            else:
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
                    "lora_type": (["standard", "full", "adaptive_ratio", "adaptive_quantile", "adaptive_energy"],),
                    "algorithm": (["svd_linalg", "svd_lowrank"], {"default": "svd_linalg", "tooltip": "SVD algorithm to use, svd_lowrank is faster but less accurate."}),
                    "lowrank_iters": ("INT", {"default": 7, "min": 1, "max": 100, "step": 1, "tooltip": "The number of subspace iterations for lowrank SVD algorithm."}),
                    "output_dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                    "bias_diff": ("BOOLEAN", {"default": True}),
                    "adaptive_param": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "For ratio mode, this is the ratio of the maximum singular value. For quantile mode, this is the quantile of the singular values."}),
                    "clamp_quantile": ("BOOLEAN", {"default": True}),
                },

    }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "KJNodes/lora"

    def save(self, finetuned_model, original_model, filename_prefix, rank, lora_type, algorithm, lowrank_iters, output_dtype, bias_diff, adaptive_param, clamp_quantile):
        if algorithm == "svd_lowrank" and lora_type != "standard":
            raise ValueError("svd_lowrank algorithm is only supported for standard LoRA extraction.")

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[output_dtype]
        m = finetuned_model.clone()
        kp = original_model.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, - 1.0, 1.0)
        model_diff = m

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        output_sd = {}
        if model_diff is not None:
            output_sd = calc_lora_model(model_diff, rank, "diffusion_model.", "diffusion_model.", output_sd, lora_type, algorithm, lowrank_iters, dtype, bias_diff=bias_diff, adaptive_param=adaptive_param, clamp_quantile=clamp_quantile)
        if "adaptive" in lora_type:
            rank_str = f"{lora_type}_{adaptive_param:.2f}"
        else:
            rank_str = rank
        output_checkpoint = f"{filename}_rank_{rank_str}_{output_dtype}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=None)
        return {}

NODE_CLASS_MAPPINGS = {
    "LoraExtractKJ": LoraExtractKJ
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraExtractKJ": "LoraExtractKJ"
}
