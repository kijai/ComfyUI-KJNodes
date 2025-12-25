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


# --- TYPE DETECTION ---
def get_comfy_type(value):
    if value is None:
        return "*"
    
    if hasattr(value, 'shape'):
        shape = value.shape
        if len(shape) == 4:
            if shape[-1] in [1, 3, 4]:
                return "IMAGE"
            if shape[1] in [4, 8, 16]:
                return "LATENT"
        if len(shape) == 3 or len(shape) == 2:
            return "MASK"
    
    if isinstance(value, dict):
        if 'samples' in value:
            return "LATENT"
        return "*"
    
    if isinstance(value, list):
        if len(value) > 0:
            item = value[0]
            if isinstance(item, dict) and ('pooled_output' in item or 'cond' in item):
                return "CONDITIONING"
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                return "CONDITIONING"
        return "*"
    
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    
    cls_name = type(value).__name__
    model_types = {
        'ModelPatcher': 'MODEL',
        'CLIP': 'CLIP',
        'VAE': 'VAE',
        'ControlNet': 'CONTROL_NET',
    }
    for key, val in model_types.items():
        if key in cls_name:
            return val
    
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

# Internal trigger type
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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (any_type, INTERNAL_TRIGGER)
    RETURN_NAMES = ("value", "_trigger")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "KJNodes/Variables"

    def execute(self, variable_name, order=0, unique_id=None, value=None):
        if value is not None:
            detected_type = get_comfy_type(value)
            # Store with order suffix internally for debugging
            GLOBAL_STORE[variable_name] = value
            GLOBAL_TYPES[variable_name] = detected_type
            
            if hasattr(PromptServer, "instance"):
                PromptServer.instance.send_sync("kjnodes.type_update", {
                    "variable_name": variable_name,
                    "type": detected_type,
                    "order": order,
                    "node_id": unique_id,
                })
            
            print(f"✅ [Set] '{variable_name}' (order={order}) = {detected_type}")
        
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
        if variable_name in GLOBAL_STORE:
            value = GLOBAL_STORE[variable_name]
            var_type = GLOBAL_TYPES.get(variable_name, "*")
            print(f"📥 [Get] '{variable_name}' (order={order}) : {var_type}")
            return (value,)
        else:
            available = list(GLOBAL_STORE.keys())
            raise ValueError(f"Variable '{variable_name}' not found. Available: {available}")


# --- API ENDPOINTS ---
if hasattr(PromptServer, "instance"):
    
    @PromptServer.instance.routes.get("/kjnodes/global_types")
    async def get_all_types(request):
        return web.json_response(GLOBAL_TYPES)
    
    @PromptServer.instance.routes.get("/kjnodes/global_type/{name}")
    async def get_var_type(request):
        name = request.match_info.get('name', '')
        return web.json_response({
            "name": name,
            "type": GLOBAL_TYPES.get(name, "*"),
            "exists": name in GLOBAL_STORE
        })


# --- PROMPT HANDLER WITH ORDER SUPPORT ---
def auto_connect_globals(json_data):
    """
    Auto-connect Set→Get nodes by variable name AND order.
    
    Logic:
    - Get(order=N) connects to Set with highest order <= N
    - This allows chaining: Set(0) → Get(0) → [process] → Set(1) → Get(1)
    """
    try:
        prompt = json_data.get("prompt", json_data)
        if not isinstance(prompt, dict):
            return json_data
        
        # Collect all Set and Get nodes by variable name
        var_sets = {}  # var_name -> [(order, node_id), ...]
        var_gets = {}  # var_name -> [(order, node_id), ...]
        
        for node_id, node_data in prompt.items():
            if not isinstance(node_data, dict):
                continue
            
            class_type = node_data.get("class_type")
            inputs = node_data.get("inputs", {})
            var_name = inputs.get("variable_name", "")
            
            # Get order value (handle both direct value and node connection)
            order_val = inputs.get("order", 0)
            if isinstance(order_val, list):
                # It's a connection, use 0 as fallback
                order_val = 0
            
            if class_type == "SetNodeGlobal" and var_name:
                var_sets.setdefault(var_name, []).append((order_val, node_id))
            elif class_type == "GetNodeGlobal" and var_name:
                var_gets.setdefault(var_name, []).append((order_val, node_id))
        
        # Connect each Get to the appropriate Set based on order
        for var_name, gets in var_gets.items():
            sets = var_sets.get(var_name, [])
            if not sets:
                print(f"⚠️ [AutoConnect] No Set found for variable '{var_name}'")
                continue
            
            # Sort sets by order
            sets_sorted = sorted(sets, key=lambda x: x[0])
            
            for get_order, get_node_id in gets:
                # Find Set with highest order <= get_order
                matching_set = None
                for set_order, set_node_id in reversed(sets_sorted):
                    if set_order <= get_order:
                        matching_set = (set_order, set_node_id)
                        break
                
                # If no matching found, use the first Set
                if matching_set is None:
                    matching_set = sets_sorted[0]
                
                set_order, set_node_id = matching_set
                
                # Connect Get's _dep input to Set's _trigger output (index 1)
                prompt[get_node_id]["inputs"]["_dep"] = [set_node_id, 1]
                print(f"🔗 [AutoConnect] '{var_name}': Set[{set_node_id}](order={set_order}) → Get[{get_node_id}](order={get_order})")
        
        return json_data
    except Exception as e:
        import traceback
        print(f"⚠️ [AutoConnect] Error: {e}")
        traceback.print_exc()
        return json_data


if hasattr(PromptServer, "instance"):
    PromptServer.instance.add_on_prompt_handler(auto_connect_globals)
    print("✅ [NodesGlobal] Initialized with order support")


NODE_CLASS_MAPPINGS = {
    "SetNodeGlobal": SetNodeGlobal,
    "GetNodeGlobal": GetNodeGlobal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SetNodeGlobal": "🔹 Set Global Variable",
    "GetNodeGlobal": "🔹 Get Global Variable",
}
