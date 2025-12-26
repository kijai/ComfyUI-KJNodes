import folder_paths
from server import PromptServer
from aiohttp import web

# --- GLOBAL STORAGE ---
if not hasattr(folder_paths, "_global_store"):
    folder_paths._global_store = {}
GLOBAL_STORE = folder_paths._global_store

if not hasattr(folder_paths, "_global_types"):
    folder_paths._global_types = {}
GLOBAL_TYPES = folder_paths._global_types


# --- IMPROVED TYPE DETECTION ---
def get_comfy_type(value):
    """Determine ComfyUI type with better accuracy."""
    if value is None:
        return "*"
    
    cls_name = type(value).__name__
    
    # Check class name first (most reliable for model types)
    type_by_class = {
        'ModelPatcher': 'MODEL',
        'CLIP': 'CLIP',
        'VAE': 'VAE',
        'ControlNet': 'CONTROL_NET',
        'ControlLora': 'CONTROL_NET',
        'T2IAdapter': 'CONTROL_NET',
        'CLIPVision': 'CLIP_VISION',
        'CLIPVisionModel': 'CLIP_VISION',
        'StyleModel': 'STYLE_MODEL',
        'InsightFace': 'INSIGHTFACE',
    }
    
    for pattern, comfy_type in type_by_class.items():
        if pattern in cls_name:
            return comfy_type
    
    # Check for common methods (for wrapped objects)
    if hasattr(value, 'tokenize') and hasattr(value, 'encode_from_tokens'):
        return "CLIP"
    if hasattr(value, 'patcher') and hasattr(value, 'model'):
        return "MODEL"
    if hasattr(value, 'encode') and hasattr(value, 'decode') and hasattr(value, 'first_stage_model'):
        return "VAE"
    
    # Tensor-like
    if hasattr(value, 'shape'):
        shape = value.shape
        ndim = len(shape)
        
        if ndim == 4:
            # BHWC (batch, height, width, channels)
            if shape[-1] in [1, 3, 4]:
                return "IMAGE"
            # BCHW (batch, channels, height, width) - latent
            if shape[1] in [4, 8, 16]:
                return "LATENT"
            return "IMAGE"
        
        if ndim == 3:
            # BHW - mask
            return "MASK"
        
        if ndim == 2:
            return "MASK"
    
    # Dict types
    if isinstance(value, dict):
        if 'samples' in value:
            return "LATENT"
        if 'noise_mask' in value:
            return "LATENT"
        return "*"
    
    # CONDITIONING is list of [tensor, dict] pairs
    if isinstance(value, list) and len(value) > 0:
        first = value[0]
        # Conditioning is list of tuples/lists: [(tensor, {"pooled_output": ...}), ...]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            if hasattr(first[0], 'shape') and isinstance(first[1], dict):
                return "CONDITIONING"
        # Not conditioning, just a list
        return "*"
    
    # Primitives
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    
    return "*"


# --- ANY TYPE ---
class AnyType(str):
    def __eq__(self, other):
        return True
    def __ne__(self, other):
        return False
    def __hash__(self):
        return hash("*")

any_type = AnyType("*")

INTERNAL_TRIGGER = "KJNODES_INTERNAL_TRIGGER"


class SetNodeGlobal:
    """Sets a global variable by name with order support."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variable_name": ("STRING", {"default": "my_variable"}),
                "order": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "value": (any_type, {}),
                "_prev": (INTERNAL_TRIGGER, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (any_type, INTERNAL_TRIGGER)
    RETURN_NAMES = ("value", "_next")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "KJNodes/Variables"

    def execute(self, variable_name, order=0, unique_id=None, value=None, _prev=None):
        if value is not None:
            detected_type = get_comfy_type(value)
            GLOBAL_STORE[variable_name] = value
            GLOBAL_TYPES[variable_name] = detected_type
            
            # Store with order for debugging
            key_with_order = f"{variable_name}@{order}"
            GLOBAL_STORE[key_with_order] = value
            GLOBAL_TYPES[key_with_order] = detected_type
            
            if hasattr(PromptServer, "instance"):
                PromptServer.instance.send_sync("kjnodes.type_update", {
                    "variable_name": variable_name,
                    "type": detected_type,
                    "order": order,
                    "node_id": unique_id,
                    "class_name": type(value).__name__,
                })
            
            print(f"✅ [Set] '{variable_name}' order={order} → {detected_type} ({type(value).__name__})")
        else:
            print(f"⚠️ [Set] '{variable_name}' order={order} → value is None!")
        
        return (value, True)


class GetNodeGlobal:
    """Gets a global variable by name with order support."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variable_name": ("STRING", {"default": "my_variable"}),
                "order": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "_dep": (INTERNAL_TRIGGER, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes/Variables"

    def execute(self, variable_name, order=0, unique_id=None, _dep=None):
        # Try exact order first
        key_with_order = f"{variable_name}@{order}"
        
        if key_with_order in GLOBAL_STORE:
            value = GLOBAL_STORE[key_with_order]
            var_type = GLOBAL_TYPES.get(key_with_order, "*")
            print(f"📥 [Get] '{variable_name}' order={order} → {var_type} ({type(value).__name__})")
            return (value,)
        
        # Fallback to variable without order
        if variable_name in GLOBAL_STORE:
            value = GLOBAL_STORE[variable_name]
            var_type = GLOBAL_TYPES.get(variable_name, "*")
            print(f"📥 [Get] '{variable_name}' (fallback) → {var_type} ({type(value).__name__})")
            return (value,)
        
        # List available variables
        available = [k for k in GLOBAL_STORE.keys() if not '@' in k]
        available_with_order = [k for k in GLOBAL_STORE.keys() if k.startswith(variable_name + '@')]
        
        error_msg = f"Variable '{variable_name}' with order={order} not found!\n"
        error_msg += f"Available variables: {available}\n"
        error_msg += f"Available orders for '{variable_name}': {available_with_order}\n"
        error_msg += "Make sure Set node with matching order executes before this Get node."
        
        raise ValueError(error_msg)


# --- API ENDPOINTS ---
if hasattr(PromptServer, "instance"):
    
    @PromptServer.instance.routes.get("/kjnodes/global_types")
    async def get_all_types(request):
        # Return only base variables (without @order suffix)
        result = {k: v for k, v in GLOBAL_TYPES.items() if '@' not in k}
        return web.json_response(result)
    
    @PromptServer.instance.routes.get("/kjnodes/global_type/{name}")
    async def get_var_type(request):
        name = request.match_info.get('name', '')
        return web.json_response({
            "name": name,
            "type": GLOBAL_TYPES.get(name, "*"),
            "exists": name in GLOBAL_STORE
        })
    
    @PromptServer.instance.routes.get("/kjnodes/global_debug")
    async def get_debug_info(request):
        """Debug endpoint to see all stored variables."""
        result = {}
        for k, v in GLOBAL_STORE.items():
            result[k] = {
                "type": GLOBAL_TYPES.get(k, "*"),
                "class": type(v).__name__ if v is not None else "None",
            }
        return web.json_response(result)


# --- PROMPT HANDLER ---
def auto_connect_globals(json_data):
    """
    Auto-connect Set→Get and Set→Set nodes by variable name AND order.
    
    Rules:
    1. Get(var, order=N) connects to Set(var, order=N) with SAME order
    2. If no Set with same order, connect to Set with highest order < N
    3. Set(var, order=N) connects to Set(var, order=M) where M is max order < N
    """
    try:
        prompt = json_data.get("prompt", json_data)
        if not isinstance(prompt, dict):
            return json_data
        
        # Clear old data for fresh execution
        # Note: We keep the store as cache, but prompt handler ensures correct order
        
        # Collect nodes by variable
        sets_by_var = {}  # var_name -> {order: [(node_id, node_data), ...]}
        gets_by_var = {}
        
        for node_id, node_data in prompt.items():
            if not isinstance(node_data, dict):
                continue
            
            class_type = node_data.get("class_type")
            inputs = node_data.get("inputs", {})
            var_name = inputs.get("variable_name", "")
            
            if not var_name:
                continue
            
            order_val = inputs.get("order", 0)
            if isinstance(order_val, list):
                order_val = 0
            
            if class_type == "SetNodeGlobal":
                sets_by_var.setdefault(var_name, {}).setdefault(order_val, []).append((node_id, node_data))
            elif class_type == "GetNodeGlobal":
                gets_by_var.setdefault(var_name, {}).setdefault(order_val, []).append((node_id, node_data))
        
        # Debug output
        for var_name in sets_by_var:
            orders = sorted(sets_by_var[var_name].keys())
            print(f"🔍 [AutoConnect] '{var_name}' Set orders: {orders}")
        for var_name in gets_by_var:
            orders = sorted(gets_by_var[var_name].keys())
            print(f"🔍 [AutoConnect] '{var_name}' Get orders: {orders}")
        
        # Process each variable
        all_vars = set(list(sets_by_var.keys()) + list(gets_by_var.keys()))
        
        for var_name in all_vars:
            sets = sets_by_var.get(var_name, {})
            gets = gets_by_var.get(var_name, {})
            
            if not sets:
                print(f"⚠️ [AutoConnect] No Set found for '{var_name}'")
                continue
            
            set_orders = sorted(sets.keys())
            
            # 1. Chain Set nodes by order
            for i, current_order in enumerate(set_orders):
                if i > 0:
                    prev_order = set_orders[i - 1]
                    for set_node_id, set_node_data in sets[current_order]:
                        prev_set_id = sets[prev_order][0][0]
                        set_node_data["inputs"]["_prev"] = [prev_set_id, 1]
                        print(f"🔗 [Chain] '{var_name}': Set[{prev_set_id}] order={prev_order} → Set[{set_node_id}] order={current_order}")
            
            # 2. Connect Get nodes
            for get_order, get_nodes in gets.items():
                # Find Set with same order first
                if get_order in sets:
                    target_order = get_order
                else:
                    # Find highest order < get_order
                    lower_orders = [o for o in set_orders if o < get_order]
                    if lower_orders:
                        target_order = max(lower_orders)
                    else:
                        # Use lowest available
                        target_order = set_orders[0]
                
                set_node_id = sets[target_order][0][0]
                
                for get_node_id, get_node_data in get_nodes:
                    get_node_data["inputs"]["_dep"] = [set_node_id, 1]
                    print(f"🔗 [Connect] '{var_name}': Set[{set_node_id}] order={target_order} → Get[{get_node_id}] order={get_order}")
        
        return json_data
        
    except Exception as e:
        import traceback
        print(f"⚠️ [AutoConnect] Error: {e}")
        traceback.print_exc()
        return json_data


if hasattr(PromptServer, "instance"):
    PromptServer.instance.add_on_prompt_handler(auto_connect_globals)
    print("✅ [NodesGlobal] Initialized")


NODE_CLASS_MAPPINGS = {
    "SetNodeGlobal": SetNodeGlobal,
    "GetNodeGlobal": GetNodeGlobal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SetNodeGlobal": "🔹 Set Global Variable",
    "GetNodeGlobal": "🔹 Get Global Variable",
}
