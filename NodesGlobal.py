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
    
    # Check class name first (most reliable)
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
        'FluxClipModel': 'CLIP',
    }
    
    for pattern, comfy_type in type_by_class.items():
        if pattern in cls_name:
            return comfy_type
    
    # Check methods
    if hasattr(value, 'tokenize') and hasattr(value, 'encode_from_tokens'):
        return "CLIP"
    if hasattr(value, 'patcher') and hasattr(value, 'model'):
        return "MODEL"
    if hasattr(value, 'encode') and hasattr(value, 'decode') and hasattr(value, 'first_stage_model'):
        return "VAE"
    
    # Tensor
    if hasattr(value, 'shape'):
        shape = value.shape
        ndim = len(shape)
        
        if ndim == 4:
            if shape[-1] in [1, 3, 4]:
                return "IMAGE"
            if shape[1] in [4, 8, 16]:
                return "LATENT"
            return "IMAGE"
        if ndim == 3 or ndim == 2:
            return "MASK"
    
    # Dict
    if isinstance(value, dict):
        if 'samples' in value:
            return "LATENT"
        return "*"
    
    # CONDITIONING - list of (tensor, dict) tuples
    if isinstance(value, list) and len(value) > 0:
        first = value[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            if hasattr(first[0], 'shape') and isinstance(first[1], dict):
                return "CONDITIONING"
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
            
            # Unique key for this variable+order
            key = f"{variable_name}@{order}"
            
            GLOBAL_STORE[key] = value
            GLOBAL_TYPES[key] = detected_type
            
            # Also store latest value without order
            GLOBAL_STORE[variable_name] = value
            GLOBAL_TYPES[variable_name] = detected_type
            
            # Check if type changed from previous order
            prev_key = f"{variable_name}@{order-1}" if order > 0 else None
            if prev_key and prev_key in GLOBAL_TYPES:
                prev_type = GLOBAL_TYPES[prev_key]
                if prev_type != detected_type:
                    print(f"⚠️ [Set] '{variable_name}' TYPE CHANGED: order={order-1} was {prev_type}, order={order} is {detected_type}")
            
            if hasattr(PromptServer, "instance"):
                PromptServer.instance.send_sync("kjnodes.type_update", {
                    "variable_name": variable_name,
                    "type": detected_type,
                    "order": order,
                    "node_id": unique_id,
                    "class_name": type(value).__name__,
                })
            
            print(f"✅ [Set] '{variable_name}@{order}' → {detected_type} ({type(value).__name__})")
        
        return (value, True)


class GetNodeGlobal:
    """Gets a global variable by name with order support and type validation."""
    
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
        key = f"{variable_name}@{order}"
        
        # Try exact key first
        if key in GLOBAL_STORE:
            value = GLOBAL_STORE[key]
            var_type = GLOBAL_TYPES.get(key, "*")
            print(f"📥 [Get] '{key}' → {var_type} ({type(value).__name__})")
            return (value,)
        
        # Try to find lower order
        for o in range(order - 1, -1, -1):
            fallback_key = f"{variable_name}@{o}"
            if fallback_key in GLOBAL_STORE:
                value = GLOBAL_STORE[fallback_key]
                var_type = GLOBAL_TYPES.get(fallback_key, "*")
                print(f"📥 [Get] '{variable_name}@{order}' (fallback to @{o}) → {var_type} ({type(value).__name__})")
                return (value,)
        
        # Try variable without order
        if variable_name in GLOBAL_STORE:
            value = GLOBAL_STORE[variable_name]
            var_type = GLOBAL_TYPES.get(variable_name, "*")
            print(f"📥 [Get] '{variable_name}' (no order) → {var_type} ({type(value).__name__})")
            return (value,)
        
        # Error with helpful info
        available_keys = [k for k in GLOBAL_STORE.keys() if k.startswith(variable_name)]
        raise ValueError(
            f"Variable '{variable_name}@{order}' not found!\n"
            f"Available: {available_keys}\n"
            f"All variables: {list(GLOBAL_STORE.keys())}"
        )


# --- API ---
if hasattr(PromptServer, "instance"):
    
    @PromptServer.instance.routes.get("/kjnodes/global_types")
    async def get_all_types(request):
        return web.json_response(GLOBAL_TYPES)
    
    @PromptServer.instance.routes.get("/kjnodes/global_debug")
    async def get_debug(request):
        result = {}
        for k, v in GLOBAL_STORE.items():
            result[k] = {
                "type": GLOBAL_TYPES.get(k, "*"),
                "class": type(v).__name__,
            }
        return web.json_response(result)


# --- PROMPT HANDLER ---
def auto_connect_globals(json_data):
    """Auto-connect Set→Get nodes by variable name AND order."""
    try:
        prompt = json_data.get("prompt", json_data)
        if not isinstance(prompt, dict):
            return json_data
        
        # Collect nodes
        sets_by_var = {}
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
        
        # Debug
        for var_name, orders in sets_by_var.items():
            print(f"🔍 [AutoConnect] '{var_name}' Set orders: {sorted(orders.keys())}")
        for var_name, orders in gets_by_var.items():
            print(f"🔍 [AutoConnect] '{var_name}' Get orders: {sorted(orders.keys())}")
        
        # Process
        for var_name in set(list(sets_by_var.keys()) + list(gets_by_var.keys())):
            sets = sets_by_var.get(var_name, {})
            gets = gets_by_var.get(var_name, {})
            
            if not sets:
                print(f"⚠️ [AutoConnect] No Set for '{var_name}'")
                continue
            
            set_orders = sorted(sets.keys())
            
            # Chain Sets
            for i, order in enumerate(set_orders):
                if i > 0:
                    prev_order = set_orders[i - 1]
                    for set_id, set_data in sets[order]:
                        prev_set_id = sets[prev_order][0][0]
                        set_data["inputs"]["_prev"] = [prev_set_id, 1]
                        print(f"🔗 [Chain] '{var_name}': Set@{prev_order} → Set@{order}")
            
            # Connect Gets to matching Sets
            for get_order, get_nodes in gets.items():
                # Find Set with SAME order
                if get_order in sets:
                    target_order = get_order
                else:
                    # Find highest order < get_order
                    lower = [o for o in set_orders if o < get_order]
                    target_order = max(lower) if lower else set_orders[0]
                
                set_id = sets[target_order][0][0]
                
                for get_id, get_data in get_nodes:
                    get_data["inputs"]["_dep"] = [set_id, 1]
                    print(f"🔗 [Connect] '{var_name}': Set@{target_order}[{set_id}] → Get@{get_order}[{get_id}]")
        
        return json_data
        
    except Exception as e:
        import traceback
        print(f"⚠️ [AutoConnect] {e}")
        traceback.print_exc()
        return json_data


if hasattr(PromptServer, "instance"):
    PromptServer.instance.add_on_prompt_handler(auto_connect_globals)
    print("✅ [NodesGlobal] Ready")


NODE_CLASS_MAPPINGS = {
    "SetNodeGlobal": SetNodeGlobal,
    "GetNodeGlobal": GetNodeGlobal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SetNodeGlobal": "🔹 Set Global Variable",
    "GetNodeGlobal": "🔹 Get Global Variable",
}
