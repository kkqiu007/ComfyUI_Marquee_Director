# 自检报告 · 2026-10-17

对象：`custom_nodes/ComfyUI_Marquee_Director`（`marquee_director/` 共 23 个模块、20,010 行）
范围：静态排查 → 白盒（结构/符号）→ 黑盒（注册/schema/路由/工作流/边界）→ 回归

---

## 一、结论

| 层 | 方法 | 结果 |
| --- | --- | --- |
| 静态 | ruff（F/E/B/SIM/RET/PIE）+ 自写 AST 作用域扫描 | **1 个 P1 运行期崩溃**、1 个 P2 路径缺陷 |
| 白盒 | 23 个模块逐个导入、跨模块符号闭环 | 23/23 通过，无循环导入 |
| 黑盒 · 节点 | 经 `MarqueeDirectorExtension` 枚举 8 节点，校验 id/命名/取值域/选项/类型 | 7 项契约全过 |
| 黑盒 · 路由 | 11 条 `/h3/*` 的 async/`request` 契约 | 全过 |
| 黑盒 · 工作流 | `tools/check_workflow.py` + 独立控件值对账 | 干净（差异仅为前端 DOM 控件） |
| 黑盒 · 安全 | 对抗性 zip（zip-slip / 扩展名 / 炸弹）、路径穿越、设备名 | zip 全拦；**设备名有缺口（已修）** |
| 回归 | `tests/` 19 个 + `tools/test_*.py` 9 个 | **19/19、9/9 全绿** |

修复后新增两个常驻测试：`tests/test_node_contracts.py`、`tests/test_path_safety.py`。

---

## 二、P1 · 翻译路径必然崩溃（已修）

**位置** `marquee_director/translate_nodes.py:1074`

```python
def _translate_single(idx):
    nonlocal n_done, n_fail, n_retry, n_salv      # ← 少了 n_skip
    ...
    n_skip += 1                                   # ← 读-改-写
```

`+=` 是读-改-写。`n_skip` 没有出现在 `nonlocal` 里，Python 就把它判成**本函数局部变量**，
走到这一行时它还没被赋值 —— 直接 `UnboundLocalError`。

**触发条件**：块里有结构行（PACK 头/尾、双语气泡标签）且挖掉占位符后**没有待译中文**，
即「原样透传」分支。也就是每个 PACK 的头部块都会命中，属于必现路径。

**取证**（不靠推理，看字节码对象）：

```
_translate_single  co_varnames 含 'n_skip'   → 是局部变量
                   co_freevars 不含 'n_skip' → 不是闭包变量
```

**修法**：把 `n_skip` 补进 `nonlocal` 声明。修复后复验：
`n_skip 局部? False | 闭包? True`。

---

## 三、P2 · Windows 设备名可让整条链写不进去（已修）

**位置** `marquee_director/session.py:sanitize()`

`sanitize()` 已挡住路径穿越（`../`、`/abs`、`C:/`、`\`），但没有处理 Windows 设备名。
实测（本机）：

| 会话名 | `makedirs` | `isdir` | 写 `seg_00.mp4` |
| --- | --- | --- | --- |
| `CON` | 不报错 | True | 成功 |
| `COM1` / `aux` / `CON.txt` | 不报错 | True | 成功 |
| **`NUL`** | **不报错** | **False** | **FileNotFoundError** |

`NUL` 这一格才是真正的坑：目录**根本没建出来**，`makedirs` 却一声不响，
之后每一次写段都变成 `FileNotFoundError`，报错点离根因很远。

**修法**：净化末尾加设备名兜底，只看第一个点之前的部分（`CON.txt` 同样是设备名），
命中就加 `_` 前缀。选前缀而不是替换，是为了保持可读且不与别的会话撞名。

复验：`CON→_CON`、`NUL→_NUL`、`nul.png→_nul.png`、`CON.txt→_CON.txt`；
中文片名与普通名不受影响；对 20 组敌意输入逐一**真实建目录 + 写文件，失败数 0**。

---

## 四、P2 · 契约测试失效（上一轮改动引入，已修）

**位置** `tools/test_refine.py`

该测试原本断言 refine 包**恰好**是 6 个键，注释写着「有人顺手加一个键，所有既有会话就重渲一遍」。
上一轮加入的 `clear_vram` 让它红了 —— 测试是对的，改动也是对的：`clear_vram` 是显存策略，
**有意不进缓存键**。

**修法**：期望集补上 `clear_vram`，并新增一条把结论变成可执行断言的用例：

```python
check("clear_vram does not change the cache key",
      S.segment_key(base_settings(refine=dict(pack, clear_vram=True)), ...) == key_ref)
```

这样保护强度反而更高：以前只防「多一个键」，现在还防「这个键混进缓存键」。

---

## 五、P3 · 判定为非缺陷的静态告警（未改）

| 位置 | 告警 | 判定 |
| --- | --- | --- |
| `common.py:42` `FRAME_RESCALE` | F401 未使用 | **误报**：`engine.py` 从 `common` 再导出并用于 `engine.py:263` |
| `face_qc.py:183` `import insightface` | F401 未使用 | **误报**：它是 `try` 里的**能力探针**，导入本身就是检测 |
| `prompt_pack.py:3174` `tail_out` | F841 赋值未用 | 死赋值，但注释说明了「不再计入生成时长」的历史，保留可读性 |
| `prompt_pack.py:5093` `total` | F841 | 死赋值，无副作用 |
| `wardrobe.py:650` `_sub` | B023 闭包晚绑定 | **误报**：`_rule`/`_benign` 已用默认参数绑定，是正确写法 |
| 其余 11 处 | F401/F841/B904/B905/SIM | 纯风格，无行为影响 |

全包 **无 F821（未定义名）、无裸 `except:`、无可变默认参数**。

---

## 六、黑盒覆盖明细

**节点契约**（`tests/test_node_contracts.py`，常驻）
① 8 节点、id 唯一、`H3`+大驼峰命名 ② 显示名非空、至少一个输出
③ 输入 id 唯一且为合法标识符 ④ 数值 `default ∈ [min,max]` 且 `min ≤ max`
⑤ 下拉 `default ∈ options`、options 非空无重复 ⑥ 控件类型不串味（bool 不许带 min/max）
⑦ `execute` 可调用

> 说明：`H3RenderSegment` / `H3RepairSegment` 定义了 node_id 但**不进注册表**，
> 它们是 Director 内部调用的执行器（`director.py` 直接调 `execute`）。与 README
> 的「8 节点」表一致，不是漏注册。

**路由**：11 条全部 `async def(request)`，符合 ComfyUI 处理器契约。

**工作流**：`workflows/h3_director_unlimited_storyboard.json` 通过 checker；
独立对账确认 `widgets_values` 与 schema 控件数的差异**全部**来自前端 DOM 控件
（`control_after_generate`、`h3_director_panels`），非错位。

**路径安全**（`tests/test_path_safety.py`，常驻）
穿越切断、设备名可写、`sanitize` 幂等、中文保留、zip-slip 五向量全拒 + 正常成员通过。

**纯函数抗崩**：`generation_length` / `guide_length_up` 对 `-100…1000` 全部落在 17k+5 网格；
`_parse_run_segments` 对 `''`/`None`/`abc`/`-5`/`99`/`3-1`/`1,,2`/`1;2` 均不抛异常。

---

## 七、未覆盖（诚实声明）

以下**无法离线验证**，需要真机跑图，本次未做：

1. 实际采样/解码/拼接的画面与音频正确性（需 GPU 与模型权重）。
2. `te_prefetch` shim 在 ComfyUI 升级后的兼容性（依赖运行时版本）。
3. 前端面板（`web/js/h3_director.js`，440 KB）的交互行为 —— 只做了静态存在性检查，
   未做浏览器内实测。
4. `clear_vram` 的实际显存峰值收益 —— 逻辑与缓存键契约已测，但「峰值降多少」需实测。

---

## 八、如何复验

```bash
PY=<ComfyUI>/.venv/Scripts/python.exe

$PY tests/test_node_contracts.py      # 节点注册与 schema 契约
$PY tests/test_path_safety.py         # 会话名净化 + zip-slip
$PY tools/test_refine.py              # 二采契约（含 clear_vram 不进缓存键）
$PY tools/check_workflow.py           # 随包工作流

# 全量
for f in tests/test_*.py; do $PY "$f" || echo "FAIL $f"; done
```
