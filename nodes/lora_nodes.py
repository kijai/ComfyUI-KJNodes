import torch
import comfy.model_management
import comfy.utils
import folder_paths
import os
import logging
from tqdm import tqdm
import numpy as np

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

class LoraReduceRank:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                    "new_rank": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1, "tooltip": "The new rank to resize the LoRA. Acts as max rank when using dynamic_method."}),
                    "dynamic_method": (["disabled", "sv_ratio", "sv_cumulative", "sv_fro"], {"default": "disabled", "tooltip": "Method to use for dynamically determining new alphas and dims"}),
                    "dynamic_param": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Method to use for dynamically determining new alphas and dims"}),
                    "output_dtype": (["match_original", "fp16", "bf16", "fp32"], {"default": "match_original", "tooltip": "Data type to save the LoRA as."}),
                    "verbose": ("BOOLEAN", {"default": True}),
                },

    }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    EXPERIMENTAL = True
    DESCRIPTION = "Resize a LoRA model by reducing it's rank. Based on kohya's sd-scripts: https://github.com/kohya-ss/sd-scripts/blob/main/networks/resize_lora.py"

    CATEGORY = "KJNodes/lora"

    def save(self, lora_name, new_rank, output_dtype, dynamic_method, dynamic_param, verbose):

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_sd, metadata = comfy.utils.load_torch_file(lora_path, return_metadata=True)

        if output_dtype == "fp16":
            save_dtype = torch.float16
        elif output_dtype == "bf16":
            save_dtype = torch.bfloat16
        elif output_dtype == "fp32":
            save_dtype = torch.float32
        elif output_dtype == "match_original":
            first_weight_key = next(k for k in lora_sd if k.endswith(".weight") and isinstance(lora_sd[k], torch.Tensor))
            save_dtype = lora_sd[first_weight_key].dtype

        new_lora_sd = {}
        for k, v in lora_sd.items():
            new_lora_sd[k.replace(".default", "")] = v
        del lora_sd
        print("Resizing Lora...")
        output_sd, old_dim, new_alpha, rank_list = resize_lora_model(new_lora_sd, new_rank, save_dtype, device, dynamic_method, dynamic_param, verbose)

        # update metadata
        if metadata is None:
            metadata = {}

        comment = metadata.get("ss_training_comment", "")

        if dynamic_method == "disabled":
            metadata["ss_training_comment"] = f"dimension is resized from {old_dim} to {new_rank}; {comment}"
            metadata["ss_network_dim"] = str(new_rank)
            metadata["ss_network_alpha"] = str(new_alpha)
        else:
            metadata["ss_training_comment"] = f"Dynamic resize with {dynamic_method}: {dynamic_param} from {old_dim}; {comment}"
            metadata["ss_network_dim"] = "Dynamic"
            metadata["ss_network_alpha"] = "Dynamic"

        # cast to save_dtype before calculating hashes
        for key in list(output_sd.keys()):
            value = output_sd[key]
            if type(value) == torch.Tensor and value.dtype.is_floating_point and value.dtype != save_dtype:
                output_sd[key] = value.to(save_dtype)

        output_filename_prefix = "loras/" + lora_name

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(output_filename_prefix, self.output_dir)
        output_dtype_str = f"_{output_dtype}" if output_dtype != "match_original" else ""
        average_rank = str(int(np.mean(rank_list)))
        rank_str = new_rank if dynamic_method == "disabled" else f"dynamic_{average_rank}"
        output_checkpoint = f"{filename.replace('.safetensors', '')}_resized_from_{old_dim}_to_{rank_str}{output_dtype_str}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        print(f"Saving resized LoRA to {output_checkpoint}")

        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=metadata)
        return {}

NODE_CLASS_MAPPINGS = {
    "LoraExtractKJ": LoraExtractKJ
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraExtractKJ": "LoraExtractKJ"
}

# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo

# This version is based on
# https://github.com/kohya-ss/sd-scripts/blob/main/networks/resize_lora.py

MIN_SV = 1e-6

LORA_DOWN_UP_FORMATS = [
    ("lora_down", "lora_up"),  # sd-scripts LoRA
    ("lora_A", "lora_B"),  # PEFT LoRA
    ("down", "up"),  # ControlLoRA
]

# Indexing functions
def index_sv_cumulative(S, target):
    original_sum = float(torch.sum(S))
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def index_sv_fro(S, target):
    S_squared = S.pow(2)
    S_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def index_sv_ratio(S, target):
    max_sv = S[0]
    min_sv = max_sv / target
    index = int(torch.sum(S > min_sv).item())
    index = max(1, min(index, len(S) - 1))

    return index


# Modified from Kohaku-blueleaf's extract/merge functions
def extract_conv(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size, kernel_size, _ = weight.size()
    if weight.dtype != torch.float32:
        weight = weight.to(torch.float32)
    U, S, Vh = torch.linalg.svd(weight.reshape(out_size, -1).to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size, kernel_size, kernel_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank, 1, 1).cpu()
    del U, S, Vh, weight
    return param_dict


def extract_linear(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size = weight.size()

    if weight.dtype != torch.float32:
        weight = weight.to(torch.float32)
    U, S, Vh = torch.linalg.svd(weight.to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank).cpu()
    del U, S, Vh, weight
    return param_dict


def merge_conv(lora_down, lora_up, device):
    in_rank, in_size, kernel_size, k_ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    assert in_rank == out_rank and kernel_size == k_, f"rank {in_rank} {out_rank} or kernel {kernel_size} {k_} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    merged = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
    weight = merged.reshape(out_size, in_size, kernel_size, kernel_size)
    del lora_up, lora_down
    return weight


def merge_linear(lora_down, lora_up, device):
    in_rank, in_size = lora_down.shape
    out_size, out_rank = lora_up.shape
    assert in_rank == out_rank, f"rank {in_rank} {out_rank} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    weight = lora_up @ lora_down
    del lora_up, lora_down
    return weight


# Calculate new rank


def rank_resize(S, rank, dynamic_method, dynamic_param, scale=1):
    param_dict = {}

    if dynamic_method == "sv_ratio":
        # Calculate new dim and alpha based off ratio
        new_rank = index_sv_ratio(S, dynamic_param) + 1
        new_alpha = float(scale * new_rank)

    elif dynamic_method == "sv_cumulative":
        # Calculate new dim and alpha based off cumulative sum
        new_rank = index_sv_cumulative(S, dynamic_param) + 1
        new_alpha = float(scale * new_rank)

    elif dynamic_method == "sv_fro":
        # Calculate new dim and alpha based off sqrt sum of squares
        new_rank = index_sv_fro(S, dynamic_param) + 1
        new_alpha = float(scale * new_rank)
    else:
        new_rank = rank
        new_alpha = float(scale * new_rank)

    if S[0] <= MIN_SV:  # Zero matrix, set dim to 1
        new_rank = 1
        new_alpha = float(scale * new_rank)
    elif new_rank > rank:  # cap max rank at rank
        new_rank = rank
        new_alpha = float(scale * new_rank)

    # Calculate resize info
    s_sum = torch.sum(torch.abs(S))
    s_rank = torch.sum(torch.abs(S[:new_rank]))

    S_squared = S.pow(2)
    s_fro = torch.sqrt(torch.sum(S_squared))
    s_red_fro = torch.sqrt(torch.sum(S_squared[:new_rank]))
    fro_percent = float(s_red_fro / s_fro)

    param_dict["new_rank"] = new_rank
    param_dict["new_alpha"] = new_alpha
    param_dict["sum_retained"] = (s_rank) / s_sum
    param_dict["fro_retained"] = fro_percent
    param_dict["max_ratio"] = S[0] / S[new_rank - 1]

    return param_dict


def resize_lora_model(lora_sd, new_rank, save_dtype, device, dynamic_method, dynamic_param, verbose):
    max_old_rank = None
    new_alpha = None
    verbose_str = "\n"
    fro_list = []
    rank_list = []

    if dynamic_method:
        print(f"Dynamically determining new alphas and dims based off {dynamic_method}: {dynamic_param}, max rank is {new_rank}")

    lora_down_weight = None
    lora_up_weight = None

    o_lora_sd = lora_sd.copy()
    block_down_name = None
    block_up_name = None

    total_keys = len([k for k in lora_sd if k.endswith(".weight")])

    pbar = comfy.utils.ProgressBar(total_keys)
    for key, value in tqdm(lora_sd.items(), leave=True, desc="Resizing LoRA weights"):
        key_parts = key.split(".")
        block_down_name = None
        for _format in LORA_DOWN_UP_FORMATS:
            # Currently we only match lora_down_name in the last two parts of key
            # because ("down", "up") are general words and may appear in block_down_name
            if len(key_parts) >= 2 and _format[0] == key_parts[-2]:
                block_down_name = ".".join(key_parts[:-2])
                lora_down_name = "." + _format[0]
                lora_up_name = "." + _format[1]
                weight_name = "." + key_parts[-1]
                break
            if len(key_parts) >= 1 and _format[0] == key_parts[-1]:
                block_down_name = ".".join(key_parts[:-1])
                lora_down_name = "." + _format[0]
                lora_up_name = "." + _format[1]
                weight_name = ""
                break

        if block_down_name is None:
            # This parameter is not lora_down
            continue

        # Now weight_name can be ".weight" or ""
        # Find corresponding lora_up and alpha
        block_up_name = block_down_name
        lora_down_weight = value
        lora_up_weight = lora_sd.get(block_up_name + lora_up_name + weight_name, None)
        lora_alpha = lora_sd.get(block_down_name + ".alpha", None)

        weights_loaded = lora_down_weight is not None and lora_up_weight is not None

        if weights_loaded:

            conv2d = len(lora_down_weight.size()) == 4
            old_rank = lora_down_weight.size()[0]
            max_old_rank = max(max_old_rank or 0, old_rank)
            

            if lora_alpha is None:
                scale = 1.0
            else:
                scale = lora_alpha / old_rank

            if conv2d:
                full_weight_matrix = merge_conv(lora_down_weight, lora_up_weight, device)
                param_dict = extract_conv(full_weight_matrix, new_rank, dynamic_method, dynamic_param, device, scale)
            else:
                full_weight_matrix = merge_linear(lora_down_weight, lora_up_weight, device)
                param_dict = extract_linear(full_weight_matrix, new_rank, dynamic_method, dynamic_param, device, scale)

            if verbose:
                max_ratio = param_dict["max_ratio"]
                sum_retained = param_dict["sum_retained"]
                fro_retained = param_dict["fro_retained"]
                if not np.isnan(fro_retained):
                    fro_list.append(float(fro_retained))
                log_str = f"{block_down_name:75} | sum(S) retained: {sum_retained:.1%}, fro retained: {fro_retained:.1%}, max(S) ratio: {max_ratio:0.1f}"
                tqdm.write(log_str)
                verbose_str += log_str
            
            if verbose and dynamic_method:
                verbose_str += f", dynamic | dim: {param_dict['new_rank']}, alpha: {param_dict['new_alpha']}\n"
            else:
                verbose_str += "\n"

            new_alpha = param_dict["new_alpha"]
            o_lora_sd[block_down_name + lora_down_name + weight_name] = param_dict["lora_down"].to(save_dtype).contiguous()
            o_lora_sd[block_up_name + lora_up_name + weight_name] = param_dict["lora_up"].to(save_dtype).contiguous()
            o_lora_sd[block_down_name + ".alpha"] = torch.tensor(param_dict["new_alpha"]).to(save_dtype)

            block_down_name = None
            block_up_name = None
            lora_down_weight = None
            lora_up_weight = None
            weights_loaded = False
            rank_list.append(param_dict["new_rank"])
            del param_dict
        pbar.update(1)

    if verbose:
        print(f"Average Frobenius norm retention: {np.mean(fro_list):.2%} | std: {np.std(fro_list):0.3f}")
    return o_lora_sd, max_old_rank, new_alpha, rank_list
