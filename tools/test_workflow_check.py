# -*- coding: utf-8 -*-
"""Offline regression tests for ``tools/check_workflow.py``.

    python tools/test_workflow_check.py

No ComfyUI server needed —— ``check_file`` 吃的是 schemas 字典，所以这里用**手搓的
schema**（不是包里的真节点）。真节点的控件表一直在长，把它们写进断言就等于每周
改一次测试；这里要钉住的是检查器自己的三条规矩：

1. **旧 ID 只在"这是本包的图"时才报。** ``MinimaxH3LoadJson`` 同时也是
   ``Comfyui-MinimaxH3-Ref2VA-Auto-Video`` 的现役节点。图里一张本包节点都没有，
   那个 ID 就是别人家的，报了会让人去改一张好图，``--fix`` 更是直接改坏。
2. **补控件插回 schema 的那一格，不追加到尾巴。** 数组末尾可能挂着前端 DOM 槽，
   追加会把插入点之后的全部值推移一格。
3. **种子控件占两格**（值 + ``control_after_generate``），按"控件个数 == 数组长度"
   比对会假报移位。
"""

import json
import os
import sys
import tempfile
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_workflow as cw          # noqa: E402  (needs HERE on sys.path)

Widget = namedtuple("Widget", "id io_type control_after_generate force_input default")
Socket = namedtuple("Socket", "id io_type")

FAILS = []


def check(label, cond, extra=""):
    if not cond:
        FAILS.append(label)
    print("%-52s %s %s" % (label, "PASS" if cond else "FAIL", extra))


def w(name, default=None, seed=False):
    return Widget(name, "STRING", seed, False, default)


def schema(widgets=(), sockets=(), outputs=()):
    return {"display": "test", "widgets": list(widgets),
            "sockets": list(sockets), "outputs": list(outputs)}


def write(graph):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, ensure_ascii=False)
    return path


def node(nid, ntype, values=None, named=None, title=None, **kw):
    out = {"id": nid, "type": ntype, "inputs": [], "outputs": [],
           "widgets_values": values if values is not None else [],
           "widgets_values_named": named if named is not None else {}}
    if title:
        out["title"] = title
        out["properties"] = {"Node name for S&R": ntype}
    out.update(kw)
    return out


ABC = schema([w("a", 1), w("b", 2), w("c", 3)])
SCHEMAS = {"H3Foo": ABC}

# --- 1. retired ids: ours vs theirs ------------------------------------------
print("--- 旧 ID 归属 ---")
legacy = node(44, "MinimaxH3LoadJson", ["x.json"], {"filename": "x.json"},
              title="⑭-D 读取存档 JSON")

foreign = write({"nodes": [legacy], "links": []})
problems, fixed = cw.check_file(foreign, SCHEMAS, fix=True)
check("别人的图：旧 ID 不报", not problems, str(problems))
check("别人的图：--fix 不动它", not fixed, str(fixed))
after = json.load(open(foreign, encoding="utf-8"))["nodes"][0]
check("别人的图：类型仍是 MinimaxH3LoadJson",
      after["type"] == "MinimaxH3LoadJson", after["type"])
os.remove(foreign)

good = node(1, "H3Foo", [1, 2, 3], {"a": 1, "b": 2, "c": 3})
ours = write({"nodes": [good, legacy], "links": []})
problems, fixed = cw.check_file(ours, SCHEMAS, fix=True)
check("本包的图：旧 ID 报出来", any("改名" in p for p in problems), str(problems))
check("本包的图：--fix 改类型",
      any("MinimaxH3LoadJson -> H3StoryboardLoad" in f for f in fixed), str(fixed))
after = json.load(open(ours, encoding="utf-8"))["nodes"][1]
check("本包的图：类型已换成 H3StoryboardLoad",
      after["type"] == "H3StoryboardLoad", after["type"])
check("本包的图：S&R 属性跟着换",
      after["properties"]["Node name for S&R"] == "H3StoryboardLoad")
os.remove(ours)

# --- 2. missing widget goes back to its schema slot --------------------------
print("--- 补控件的落点 ---")
# b 是后来加进 schema 的：文件里 a、c 有值，末尾还挂着前端 DOM 槽（真包里是面板
# 展开状态 h3_director_panels）。
PANEL = "h3_director_panels"
stale = node(7, "H3Foo", [1, 3, "expanded"], {"a": 1, "c": 3, PANEL: "expanded"})
path = write({"nodes": [stale], "links": []})
problems, fixed = cw.check_file(path, SCHEMAS, fix=True)
check("报 array predates b", any("predates b" in p for p in problems), str(problems))
check("报 named copy is missing b",
      any("missing b" in p for p in problems), str(problems))
after = json.load(open(path, encoding="utf-8"))["nodes"][0]
check("数组：b 插在 a 之后而不是末尾",
      after["widgets_values"] == [1, 2, 3, "expanded"], str(after["widgets_values"]))
check("命名副本：b 按名字补上", after["widgets_values_named"].get("b") == 2,
      str(after["widgets_values_named"]))
check("已有值一个没动",
      [after["widgets_values"][0], after["widgets_values"][2]] == [1, 3])
problems, fixed = cw.check_file(path, SCHEMAS)
check("修完再查是干净的", not problems, str(problems))
os.remove(path)

# 纯缺值、没有 DOM 槽的图：仍然补到末尾，等价于旧行为。
tail_only = node(8, "H3Foo", [1, 2], {"a": 1, "b": 2})
path = write({"nodes": [tail_only], "links": []})
cw.check_file(path, SCHEMAS, fix=True)
check("无 DOM 槽时 c 追加到末尾",
      json.load(open(path, encoding="utf-8"))["nodes"][0]["widgets_values"] == [1, 2, 3])
os.remove(path)

# --- 3. seed widgets occupy two slots ----------------------------------------
print("--- 种子占两格 ---")
SEED = schema([w("seed", 7, seed=True), w("prompt", "")])
seeded = node(9, "H3Bar", [7, "fixed", "hi"],
              {"seed": 7, "control_after_generate": "fixed", "prompt": "hi"})
path = write({"nodes": [seeded], "links": []})
problems, _ = cw.check_file(path, {"H3Bar": SEED})
check("两格对齐时不报移位", not problems, str(problems))

# 数组只有 [值, 提示词] 两格、命名副本却带齐三件：这就是把种子当成"一个控件"
# 时会出现的错位 —— 提示词顶到了 control_after_generate 那一格。
short = node(10, "H3Bar", [7, "hi"],
             {"seed": 7, "control_after_generate": "fixed", "prompt": "hi"})
path2 = write({"nodes": [short], "links": []})
problems, _ = cw.check_file(path2, {"H3Bar": SEED})
check("种子漏格会报出 disagrees，而不是假的 missing",
      any("control_after_generate disagrees" in p for p in problems), str(problems))
os.remove(path)
os.remove(path2)

print("\n%s" % ("ALL PASS" if not FAILS else "FAILED: %s" % ", ".join(FAILS)))
raise SystemExit(1 if FAILS else 0)
