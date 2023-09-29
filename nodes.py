import nodes

class INTConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"

    CATEGORY = "KJNodes"

    def get_value(self, value):
        return (value,)

class ConditioningMultiCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning_1": ("CONDITIONING", ),
            "conditioning_2": ("CONDITIONING", ),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "KJNodes"

    def combine(self, combine, **kwargs):
        cond_combine_node = nodes.ConditioningCombine()
        cond = kwargs["c1"]
        for c in range(1, combine):
            new_cond = kwargs[f"c{c + 1}"]
            cond = cond_combine_node.combine(new_cond, cond)[0]
        return (cond,)
  
class ConditioningSetMaskAndCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", ),
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes"

    def append(self, positive_1, negative_1, positive_2, negative_2, mask_1, mask_2, set_cond_area, strength):
        c = []
        c2 = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask_1.shape) < 3:
            mask_1 = mask_1.unsqueeze(0)
        if len(mask_2.shape) < 3:
            mask_2 = mask_2.unsqueeze(0)
        for t in positive_1:
            n = [t[0], t[1].copy()]
            _, h, w = mask_1.shape
            n[1]['mask'] = mask_1
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c.append(n)
        for t in positive_2:
            n = [t[0], t[1].copy()]
            _, h, w = mask_2.shape
            n[1]['mask'] = mask_2
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c.append(n)
        for t in negative_1:
            n = [t[0], t[1].copy()]
            _, h, w = mask_1.shape
            n[1]['mask'] = mask_1
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c2.append(n)
        for t in negative_2:
            n = [t[0], t[1].copy()]
            _, h, w = mask_2.shape
            n[1]['mask'] = mask_2
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c2.append(n)
        return (c, c2)

NODE_CLASS_MAPPINGS = {
    "INTConstant": INTConstant,
    "ConditioningMultiCombine": ConditioningMultiCombine,
    "ConditioningSetMaskAndCombine": ConditioningSetMaskAndCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INTConstant": "INT Constant",
    "ConditioningMultiCombine": "Conditioning Multi Combine",
    "ConditioningSetMaskAndCombine": "ConditioningSetMaskAndCombine",
}