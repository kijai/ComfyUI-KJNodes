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
                # Hidden dependency input for chaining Set nodes
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
            
            if hasattr(PromptServer, "instance"):
                PromptServer.instance.send_sync("kjnodes.type_update", {
                    "variable_name": variable_name,
                    "type": detected_type,
                    "order": order,
                    "node_id": unique_id,
                })
            
            print(f"✅ [Set] '{variable_name}' order={order} → {detected_type}")
        
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
            print(f"📥 [Get] '{variable_name}' order={order} → {var_type}")
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


# --- PROMPT HANDLER ---
def auto_connect_globals(json_data):
    """
    Auto-connect Set→Get and Set→Set nodes by variable name AND order.
    
    Rules:
    1. Get(var, order=N) connects to Set(var, order=N) with SAME order
    2. If no Set with same order, connect to Set with highest order < N
    3. Set(var, order=N) connects to Set(var, order=N-1) for chaining
    """
    try:
        prompt = json_data.get("prompt", json_data)
        if not isinstance(prompt, dict):
            return json_data
        
        # Collect all Set and Get nodes grouped by variable name
        # var_name -> { order -> [(node_id, node_data), ...] }
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
            
            # Get order value
            order_val = inputs.get("order", 0)
            if isinstance(order_val, list):
                order_val = 0
            
            if class_type == "SetNodeGlobal":
                if var_name not in sets_by_var:
                    sets_by_var[var_name] = {}
                if order_val not in sets_by_var[var_name]:
                    sets_by_var[var_name][order_val] = []
                sets_by_var[var_name][order_val].append((node_id, node_data))
                
            elif class_type == "GetNodeGlobal":
                if var_name not in gets_by_var:
                    gets_by_var[var_name] = {}
                if order_val not in gets_by_var[var_name]:
                    gets_by_var[var_name][order_val] = []
                gets_by_var[var_name][order_val].append((node_id, node_data))
        
        print(f"🔍 [AutoConnect] Found Sets: {[(v, list(o.keys())) for v,o in sets_by_var.items()]}")
        print(f"🔍 [AutoConnect] Found Gets: {[(v, list(o.keys())) for v,o in gets_by_var.items()]}")
        
        # Process each variable
        for var_name in set(list(sets_by_var.keys()) + list(gets_by_var.keys())):
            sets = sets_by_var.get(var_name, {})
            gets = gets_by_var.get(var_name, {})
            
            if not sets:
                print(f"⚠️ [AutoConnect] No Set found for variable '{var_name}'")
                continue
            
            # Get sorted order list for Sets
            set_orders = sorted(sets.keys())
            
            # 1. Chain Set nodes: Set(order=N) depends on Set(order=N-1)
            for i, current_order in enumerate(set_orders):
                if i > 0:
                    prev_order = set_orders[i - 1]
                    # Connect current Set's _prev to previous Set's _next
                    for set_node_id, set_node_data in sets[current_order]:
                        # Find one Set from previous order
                        prev_set_id = sets[prev_order][0][0]
                        set_node_data["inputs"]["_prev"] = [prev_set_id, 1]
                        print(f"🔗 [Chain] '{var_name}': Set[{prev_set_id}](order={prev_order}) → Set[{set_node_id}](order={current_order})")
            
            # 2. Connect Get nodes to Set nodes with SAME order
            for get_order, get_nodes in gets.items():
                # Find Set with same order
                if get_order in sets:
                    set_node_id = sets[get_order][0][0]  # First Set with this order
                    source_order = get_order
                else:
                    # Fallback: find Set with highest order < get_order
                    matching_orders = [o for o in set_orders if o < get_order]
                    if matching_orders:
                        source_order = max(matching_orders)
                        set_node_id = sets[source_order][0][0]
                    else:
                        # Last resort: use lowest order Set
                        source_order = set_orders[0]
                        set_node_id = sets[source_order][0][0]
                
                # Connect all Get nodes with this order
                for get_node_id, get_node_data in get_nodes:
                    get_node_data["inputs"]["_dep"] = [set_node_id, 1]
                    print(f"🔗 [Connect] '{var_name}': Set[{set_node_id}](order={source_order}) → Get[{get_node_id}](order={get_order})")
        
        return json_data
        
    except Exception as e:
        import traceback
        print(f"⚠️ [AutoConnect] Error: {e}")
        traceback.print_exc()
        return json_data


if hasattr(PromptServer, "instance"):
    PromptServer.instance.add_on_prompt_handler(auto_connect_globals)
    print("✅ [NodesGlobal] Initialized with order chaining")


NODE_CLASS_MAPPINGS = {
    "SetNodeGlobal": SetNodeGlobal,
    "GetNodeGlobal": GetNodeGlobal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SetNodeGlobal": "🔹 Set Global Variable",
    "GetNodeGlobal": "🔹 Get Global Variable",
}
