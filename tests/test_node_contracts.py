# -*- coding: utf-8 -*-
"""黑盒契约测试：只通过公开入口（MarqueeDirectorExtension）观察节点，不碰内部实现。

覆盖 ComfyUI 前端/执行器真正依赖的那几件事：
  ① 注册表：8 个节点、node_id 唯一、命名规范 ``H3`` + 大驼峰；
  ② 每个节点 ``define_schema()`` 可构建，显示名非空，至少一个输出；
  ③ 输入口 id 唯一且是合法标识符（前端把它当控件名，带空格/重名会串值）；
  ④ 数值控件的 default 必须落在 [min, max] 内，且 min <= max；
  ⑤ 下拉控件的 default 必须在 options 里，options 非空且无重复；
  ⑥ 布尔控件 default 必须是 bool（否则前端开关初值不确定）；
  ⑦ 每个注册节点都有可调用的 ``execute``。

跑法（必须用 ComfyUI 的 venv）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_node_contracts.py
"""
import asyncio
import os
import re
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

from marquee_director import MarqueeDirectorExtension                  # noqa: E402

EXPECTED = {
    "H3ChainSettings", "H3Director", "H3PromptPackParser", "H3ScriptTranslate",
    "H3StoryboardSave", "H3StoryboardLoad", "H3Refine", "H3FaceRefine",
}

NODE_ID_RE = re.compile(r"^H3[A-Z][A-Za-z0-9]*$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_fails = []


def check(cond, msg):
    if not cond:
        _fails.append(msg)
        print("  FAIL " + msg)


def schemas():
    nodes = asyncio.run(MarqueeDirectorExtension().get_node_list())
    return nodes, [n.define_schema() for n in nodes]


def test_registry(nodes, schemas_):
    """① 注册表完整、唯一、命名合规。"""
    ids = [s.node_id for s in schemas_]
    check(len(ids) == len(set(ids)), "node_id 有重复：%s" % ids)
    check(set(ids) == EXPECTED,
          "注册节点与预期不符：多 %s / 少 %s"
          % (sorted(set(ids) - EXPECTED), sorted(EXPECTED - set(ids))))
    for s in schemas_:
        check(bool(NODE_ID_RE.match(s.node_id)), "%s 不符合 H3 大驼峰命名" % s.node_id)
        check(bool(str(s.display_name or "").strip()), "%s 缺 display_name" % s.node_id)
    print("PASS ① 注册表 %d 个节点，id 唯一且命名合规" % len(ids))


def test_outputs(schemas_):
    """② 每个节点至少一个输出，输出 id/display_name 不空。"""
    for s in schemas_:
        check(len(s.outputs) > 0, "%s 没有输出口" % s.node_id)
        for o in s.outputs:
            name = getattr(o, "display_name", None) or getattr(o, "id", None)
            check(bool(name), "%s 有输出口缺名字" % s.node_id)
    print("PASS ② 输出口齐备")


def test_input_ids(schemas_):
    """③ 输入 id 唯一、是合法标识符（前端按名字取值）。"""
    for s in schemas_:
        seen = set()
        for i in s.inputs:
            check(bool(IDENT_RE.match(i.id)),
                  "%s 输入口 id 非法：%r" % (s.node_id, i.id))
            check(i.id not in seen, "%s 输入口 id 重复：%s" % (s.node_id, i.id))
            seen.add(i.id)
        check(len(s.inputs) > 0, "%s 没有任何输入口" % s.node_id)
    print("PASS ③ 输入口 id 唯一且合法")


def test_numeric_ranges(schemas_):
    """④ 数值控件 default 落在 [min, max]，且 min <= max。"""
    for s in schemas_:
        for i in s.inputs:
            lo, hi = getattr(i, "min", None), getattr(i, "max", None)
            if lo is None and hi is None:
                continue
            if lo is not None and hi is not None:
                check(lo <= hi, "%s.%s min(%s) > max(%s)" % (s.node_id, i.id, lo, hi))
            d = getattr(i, "default", None)
            if d is None or not isinstance(d, (int, float)) or isinstance(d, bool):
                continue
            if lo is not None:
                check(d >= lo, "%s.%s default %s < min %s" % (s.node_id, i.id, d, lo))
            if hi is not None:
                check(d <= hi, "%s.%s default %s > max %s" % (s.node_id, i.id, d, hi))
    print("PASS ④ 数值控件取值域自洽")


def test_combos(schemas_):
    """⑤ 下拉控件 default 在 options 内，options 非空且无重复。"""
    for s in schemas_:
        for i in s.inputs:
            opts = getattr(i, "options", None)
            if opts is None:
                continue
            opts = list(opts)
            check(len(opts) > 0, "%s.%s options 为空" % (s.node_id, i.id))
            check(len(opts) == len(set(opts)),
                  "%s.%s options 有重复" % (s.node_id, i.id))
            d = getattr(i, "default", None)
            if d is not None:
                check(d in opts,
                      "%s.%s default %r 不在 options 内" % (s.node_id, i.id, d))
    print("PASS ⑤ 下拉控件选项自洽")


def test_no_mixed_type_controls(schemas_):
    """⑥ 控件类型不串味：布尔默认值不许落在数值槽，数值默认值不许落在下拉槽。

    前端按控件种类渲染（数字框 / 开关 / 下拉），一个 bool 默认值配上 min/max
    会让 ComfyUI 把它当数字框画，初值直接错乱 —— 这类错误 define_schema 不会报。
    """
    for s in schemas_:
        for i in s.inputs:
            d = getattr(i, "default", None)
            lo, hi = getattr(i, "min", None), getattr(i, "max", None)
            opts = getattr(i, "options", None)
            if isinstance(d, bool):
                check(lo is None and hi is None,
                      "%s.%s 是开关却有 min/max" % (s.node_id, i.id))
                check(opts is None,
                      "%s.%s 是开关却带 options" % (s.node_id, i.id))
            elif isinstance(d, (int, float)) and opts is not None:
                check(False, "%s.%s 是下拉却给了数值默认值" % (s.node_id, i.id))
    print("PASS ⑥ 控件类型与默认值不串味")


def test_execute_callable(nodes):
    """⑦ 每个注册节点都有可调用的 execute。"""
    for n in nodes:
        s = n.define_schema()
        fn = getattr(n, "execute", None)
        check(callable(fn), "%s 没有可调用的 execute" % s.node_id)
    print("PASS ⑦ execute 可调用")


if __name__ == "__main__":
    nodes, schemas_ = schemas()
    test_registry(nodes, schemas_)
    test_outputs(schemas_)
    test_input_ids(schemas_)
    test_numeric_ranges(schemas_)
    test_combos(schemas_)
    test_no_mixed_type_controls(schemas_)
    test_execute_callable(nodes)
    if _fails:
        print("\n发现 %d 处契约问题：" % len(_fails))
        for f in _fails:
            print("  - " + f)
        sys.exit(1)
    print("\n全部通过：%d 个节点的注册与 schema 契约成立。" % len(schemas_))
