# SimpleCalculatorKJ 動態輸入擴展修改文檔

## 修改概述

此修改讓 `SimpleCalculatorKJ` 節點支援動態輸入功能：
- 預設只顯示 `a` 和 `b` 兩個輸入
- 當 `a` 和 `b` 都連接後，自動新增 `x` 輸入
- 依序類推：`a → b → x → y → var1 → var2 → ... → var10`（最多14個輸入）
- 斷開連接時會移除多餘的輸入插槽
- 未連接的變數在表達式中使用時會報錯（保持原始行為）

---

## 需要修改的檔案

### 1. Python 後端：`nodes/nodes.py`

找到 `class SimpleCalculatorKJ:` 類別，將整個類別替換為以下內容：

```python
class SimpleCalculatorKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression": ("STRING", {"default": "a + b", "multiline": True}),
            },
            "optional": {
                "a": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "b": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "x": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "y": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var1": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var2": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var3": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var4": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var5": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var6": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var7": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var8": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var9": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
                "var10": (IO.ANY, {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT",)
    FUNCTION = "calculate"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = "Calculator node that evaluates a mathematical expression. Supports variables: a, b, x, y, var1-var10. Inputs appear dynamically when connected."

    def calculate(self, expression, a=None, b=None, x=None, y=None, **kwargs):

        import ast
        import operator
        import math

        # Allowed operations
        allowed_operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,  ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.LShift: operator.lshift, ast.RShift: operator.rshift,
        }

        # Allowed functions
        allowed_functions = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'pow': pow, 'sqrt': math.sqrt, 'sin': math.sin,
            'cos': math.cos, 'tan': math.tan, 'log': math.log,
            'log10': math.log10, 'exp': math.exp, 'floor': math.floor,
            'ceil': math.ceil
        }

        # Allowed constants
        allowed_names = {'pi': math.pi, 'e': math.e}
        # Add connected variables only (not None)
        if a is not None:
            allowed_names['a'] = a
        if b is not None:
            allowed_names['b'] = b
        if x is not None:
            allowed_names['x'] = x
        if y is not None:
            allowed_names['y'] = y
        # Add var1-var10 from kwargs
        for key, value in kwargs.items():
            if key.startswith('var') and value is not None:
                allowed_names[key] = value

        def eval_node(node):
            if isinstance(node, ast.Constant):  # Numbers
                return node.value
            elif isinstance(node, ast.Name):  # Variables
                if node.id in allowed_names:
                    return allowed_names[node.id]
                raise ValueError(f"Name '{node.id}' is not allowed")
            elif isinstance(node, ast.BinOp):  # Binary operations
                if type(node.op) not in allowed_operators:
                    raise ValueError(f"Operator {type(node.op).__name__} is not allowed")
                left = eval_node(node.left)
                right = eval_node(node.right)
                return allowed_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):  # Unary operations
                if type(node.op) not in allowed_operators:
                    raise ValueError(f"Operator {type(node.op).__name__} is not allowed")
                operand = eval_node(node.operand)
                return allowed_operators[type(node.op)](operand)
            elif isinstance(node, ast.Call):  # Function calls
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only simple function calls are allowed")
                if node.func.id not in allowed_functions:
                    raise ValueError(f"Function '{node.func.id}' is not allowed")
                args = [eval_node(arg) for arg in node.args]
                return allowed_functions[node.func.id](*args)
            else:
                raise ValueError(f"Node type {type(node).__name__} is not allowed")

        try:
            tree = ast.parse(expression, mode='eval')
            result = eval_node(tree.body)
            return (float(result), int(result))
        except Exception as e:
            print(f"CalculatorKJ Error: {str(e)}")
            return (0.0, 0)
```

---

### 2. JavaScript 前端：`web/js/jsnodes.js`

在 `switch (nodeData.name)` 區塊內，找到 `case "SoundReactive":` 結束的 `break;` 後面，在 `case "SaveImageKJ":` 之前，加入以下程式碼：

```javascript
		case "SimpleCalculatorKJ":
			nodeType.prototype.onNodeCreated = function () {
				// Variable order: a, b, x, y, var1, var2, ... var10
				this._varOrder = ['a', 'b', 'x', 'y', 'var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10'];
				
				// Remove inputs beyond a and b on node creation
				if (this.inputs) {
					const toRemove = [];
					for (let i = 0; i < this.inputs.length; i++) {
						const name = this.inputs[i].name;
						const varIndex = this._varOrder.indexOf(name);
						if (varIndex > 1) { // Keep only a (0) and b (1)
							toRemove.push(i);
						}
					}
					// Remove in reverse order to avoid index shifting
					for (let i = toRemove.length - 1; i >= 0; i--) {
						this.removeInput(toRemove[i]);
					}
				}
			};
			
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
				// type: 1 = input, 2 = output
				if (type !== 1) return;
				
				// Avoid issues during graph loading
				const stackTrace = new Error().stack;
				if (stackTrace.includes('loadGraphData') || stackTrace.includes('configure')) {
					return;
				}
				
				if (!this.inputs) return;
				
				// Get current variable inputs count
				const varInputs = this.inputs.filter(inp => this._varOrder.includes(inp.name));
				const currentCount = varInputs.length;
				
				// On disconnect: remove the input if we have more than 2
				if (!connected && currentCount > 2) {
					// Find the input that was disconnected
					const inputName = this.inputs[index]?.name;
					const varIndex = this._varOrder.indexOf(inputName);
					
					// Only remove if it's a variable input (not expression)
					if (varIndex >= 0) {
						this.removeInput(index);
						
						// Rename remaining variable inputs to maintain order
						let slot_i = 0;
						for (let i = 0; i < this.inputs.length; i++) {
							if (this._varOrder.includes(this.inputs[i].name)) {
								this.inputs[i].name = this._varOrder[slot_i];
								slot_i++;
							}
						}
					}
					return;
				}
				
				// On connect: add next input if all current variable inputs are connected
				if (connected) {
					// Check if all current variable inputs are connected
					let allConnected = true;
					for (let i = 0; i < this.inputs.length; i++) {
						if (this._varOrder.includes(this.inputs[i].name) && this.inputs[i].link === null) {
							allConnected = false;
							break;
						}
					}
					
					// Add next input if all are connected and we haven't reached the max
					if (allConnected && currentCount < this._varOrder.length) {
						const nextVarName = this._varOrder[currentCount];
						this.addInput(nextVarName, "*");
					}
				}
			};
			break;
```

---

## 還原到原版

如果要還原到原版，執行以下 Git 命令：

```bash
cd d:\Desktop\work\AItool\NovelAi\ComfyUI_windows_portable_nvidia\ComfyUI\custom_nodes\ComfyUI-KJNodes
git checkout nodes/nodes.py
git checkout web/js/jsnodes.js
```

或者使用：
```bash
git checkout .
```

---

## 功能說明

### 變數順序
| 順序 | 變數名 |
|------|--------|
| 1 | a |
| 2 | b |
| 3 | x |
| 4 | y |
| 5-14 | var1 ~ var10 |

### 行為
1. **新建節點**：只顯示 `a` 和 `b` 兩個輸入
2. **連接輸入**：當所有可見的輸入都被連接時，自動新增下一個輸入
3. **斷開連接**：移除空的輸入插槽（但至少保留 `a` 和 `b`）
4. **錯誤處理**：使用未連接的變數會報錯 `Name 'xxx' is not allowed`

### 支援的運算
- 運算符：`+`, `-`, `*`, `/`, `**` (次方), `<<`, `>>`
- 函數：`abs`, `round`, `min`, `max`, `pow`, `sqrt`, `sin`, `cos`, `tan`, `log`, `log10`, `exp`, `floor`, `ceil`
- 常數：`pi`, `e`

---

## 參考

此實作參考了 [Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) 的 `MakeImageList` 節點的動態輸入模式。

---

## 建立日期
2026-01-26

---

## 🔧 可選的防禦性改進（待 PR 回覆後考慮）

以下是額外的防禦性檢查，**非致命錯誤**，可在 PR 被接受後追加：

### 改進 1: 檢查 `_varOrder` 是否存在

**位置：** `web/js/jsnodes.js` - `onConnectionsChange` 函數

**原始代碼：**
```javascript
if (!this.inputs) return;
```

**改進為：**
```javascript
if (!this.inputs || !this._varOrder) return;
```

**原因：** 防止極端情況下 `onConnectionsChange` 在 `onNodeCreated` 之前被調用。

---

### 改進 2: 驗證 index 邊界

**位置：** `web/js/jsnodes.js` - `onConnectionsChange` 函數，斷開連接區塊

**原始代碼：**
```javascript
// On disconnect: remove the input if we have more than 2
if (!connected && currentCount > 2) {
    // Find the input that was disconnected
    const inputName = this.inputs[index]?.name;
```

**改進為：**
```javascript
// On disconnect: remove the input if we have more than 2
if (!connected && currentCount > 2) {
    // Validate index
    if (index < 0 || index >= this.inputs.length) return;
    
    // Find the input that was disconnected
    const inputName = this.inputs[index]?.name;
```

**原因：** 防止無效 index 導致的潛在問題。

---

### 評估

| 改進 | 嚴重程度 | 正常使用會觸發？ |
|------|----------|------------------|
| `_varOrder` 檢查 | 🟡 低 | 否 |
| `index` 邊界檢查 | 🟡 低 | 否 |

**結論：** 這些是預防性修復，原始代碼在正常使用下能正常運作。可等 PR 被接受後再追加。
