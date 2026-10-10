# -*- coding: utf-8 -*-
"""Check a workflow JSON against this pack's **live** node schemas.

    python tools/check_workflow.py                     # every workflows/*.json
    python tools/check_workflow.py path/to/wf.json     # one file
    python tools/check_workflow.py --fix wf.json       # append missing widgets

Why this exists: the pack's nodes change (inputs added, widgets renamed), and a
saved workflow does not change with them. The failure is silent and looks like
something else entirely — a widget that reads 1.625 where its schema now caps at
1.0, a prompt that vanishes, a value that silently reverts on reload. The two
copies of every widget value (``widgets_values`` by position, ``widgets_values_named``
by name) make it worse: they can disagree with each other, and which one wins
depends on a browser setting.

So this checks the graph the way the server will read it:

1. **Links.** Every link's two ends resolve to a node and a slot that exist, the
   slot types line up, no socket is fed twice, no output index runs off the end.
2. **Widgets.** The positional array and the named object are compared against
   the schema *and against each other*.
3. **Sockets.** Every wired input on one of this pack's nodes matches the type
   that node declares.

``--fix`` fills missing widget defaults into both copies: into the **named** copy
by name, and into the positional array at the slot the schema gives that widget
(its real slot, not simply the tail -- the array may end with a front-end DOM
widget, and appending after it shifts everything below the insertion point). It
never reorders and never touches existing values, because rewriting a saved
value is the one edit a workflow cannot survive silently.
"""

import argparse
import asyncio
import json
import os
import sys

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

# 本包**以前**注册过、现在已改名的节点 ID。2026-10-10 统一命名时不留别名，
# 所以旧图里的 ``type`` 会直接变成红色 missing node —— 而更坏的情况是它并不红：
# 另一个已安装的包 ``Comfyui-MinimaxH3-Ref2VA-Auto-Video`` 恰好也注册这两个 ID，
# 于是节点"看起来在"，实际是别人家的实现，槽位完全不同、连线静默错位。
#
# 但反过来也成立：那张 Ref2VA 自己的看板图里，``MinimaxH3LoadJson`` **就是**
# 它自己的节点，不是我们的旧节点，改它等于把好端端的图改坏。离线检查拿不到
# 全机注册表，所以判据换成图级的：同一张图里还有本包在用的其它节点类型，那个
# 旧 ID 才是我们的（旧 Director 图必然整串都用本包）；一张都不剩，就是别人家的
# 图，闭嘴。
RETIRED_IDS = {
    "MinimaxH3SaveJson": "H3StoryboardSave",
    "MinimaxH3LoadJson": "H3StoryboardLoad",
}


def _comfy_root():
    # tools/ lives at <ComfyUI>/custom_nodes/ComfyUI_Marquee_Director/tools/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_schemas():
    """Import the pack the way ComfyUI does and read its node schemas."""
    comfy = _comfy_root()
    sys.path.insert(0, comfy)
    sys.path.insert(0, os.path.join(comfy, "custom_nodes"))
    pkg = __import__("ComfyUI_Marquee_Director")
    classes = asyncio.run(pkg.MarqueeDirectorExtension().get_node_list())

    out = {}
    for cls in classes:
        schema = cls.define_schema()
        widgets, sockets = [], []
        for item in schema.inputs:
            io_type = getattr(item, "io_type", "")
            if io_type in WIDGET_TYPES and not getattr(item, "force_input", False):
                widgets.append(item)
            else:
                sockets.append(item)
        out[schema.node_id] = {
            "display": schema.display_name,
            "widgets": widgets,
            "sockets": sockets,
            "outputs": list(schema.outputs or []),
        }
    return out


def widget_default(item):
    d = getattr(item, "default", None)
    return d


def _default_for_slot(widgets, slot_name):
    """槽位名 → 默认值。``control_after_generate`` 不是 widget，没有 default。"""
    if slot_name == "control_after_generate":
        return "fixed"
    for item in widgets:
        if item.id == slot_name:
            return widget_default(item)
    return None


def widget_slots(item):
    """这一项在前端 ``widgets_values`` 里占**几个**槽位。

    ★ 2026-09-26 修正：种子类 Int 控件（``control_after_generate=True``）在
    数组式 ``widgets_values`` 里是**两个**槽位 —— 值 + 一个
    ``control_after_generate`` 的下拉（"fixed" / "randomize" / …）。
    旧代码拿「widget 个数」去比「数组长度」，于是任何带种子的节点都报
    「N+1 widget values for N widgets」，并把后面的数组值与命名副本逐项
    错位比较，一口气刷出 8 条假的 "disagrees"。

    DOM widget（本包的 ``h3_director_panels``）与手加的虚拟控件不在 schema 里，
    它们只出现在数组末尾 —— 单独按「额外槽位」报，不判成移位。
    """
    slots = [item.id]
    if getattr(item, "control_after_generate", False):
        slots.append("control_after_generate")
    return slots


def check_file(path, schemas, fix=False):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    links = data.get("links", [])
    problems = []
    fixed = []

    # 这张图是不是本包的图（见 RETIRED_IDS 那段）：只要还有任一在用的类型是本包的，
    # 图里残留的旧 ID 就归我们管。全是别的包的类型 —— 那是别人的图，不改。
    is_ours = any(n.get("type") in schemas for n in data.get("nodes", []))

    # ---- links ----------------------------------------------------------
    seen_targets = {}
    for link in links:
        # [id, origin_node, origin_slot, target_node, target_slot, type]
        if len(link) < 5:
            problems.append("link %s is malformed" % (link,))
            continue
        _lid, o_node, o_slot, t_node, t_slot = link[:5]
        ltype = link[5] if len(link) > 5 else "?"
        if o_node not in nodes:
            problems.append("link %s: origin node %s does not exist" % (_lid, o_node))
            continue
        if t_node not in nodes:
            problems.append("link %s: target node %s does not exist" % (_lid, t_node))
            continue
        key = (t_node, t_slot)
        if key in seen_targets:
            problems.append("node %s (%s) input slot %s is fed twice (links %s and %s)"
                            % (t_node, nodes[t_node].get("type"), t_slot,
                               seen_targets[key], _lid))
        seen_targets[key] = _lid

        o_schema = schemas.get(nodes[o_node].get("type"))
        if o_schema is not None and o_slot >= len(o_schema["outputs"]):
            problems.append(
                "link %s: %s has %d outputs, link reads output %d"
                % (_lid, nodes[o_node].get("type"), len(o_schema["outputs"]), o_slot))

    # ---- nodes ----------------------------------------------------------
    for node in data.get("nodes", []):
        ntype = node.get("type")
        if ntype in RETIRED_IDS and is_ours:
            new = RETIRED_IDS[ntype]
            problems.append(
                "node %s: type %r 已被本包改名为 %r —— 图里这个现在解析到"
                "**别的包**的同名节点（槽位不同），或直接是红色 missing node"
                % (node.get("id"), ntype, new))
            if fix:
                node["type"] = new
                props = node.get("properties") or {}
                if props.get("Node name for S&R") == ntype:
                    props["Node name for S&R"] = new
                fixed.append("node %s: type %s -> %s"
                             % (node.get("id"), ntype, new))
                ntype = new
        schema = schemas.get(ntype)
        if schema is None:
            continue  # a node from another pack; nothing to compare against

        # socket types, matched by name because slot order is not stable
        declared = {s.id: s for s in schema["sockets"]}
        widget_names = {w.id for w in schema["widgets"]}
        for slot in node.get("inputs") or []:
            name = str(slot.get("name") or "")
            base = name.split(".")[0]     # autogrow slots: ref_images.ref_image_0
            got = slot.get("type")
            if base in declared:
                want = declared[base].io_type
                if want != "COMFY_AUTOGROW_V3" and got != want:
                    problems.append(
                        "node %s (%s): input %r is typed %s, schema says %s"
                        % (node.get("id"), ntype, name, got, want))
            elif base in widget_names:
                # A widget promoted to an input (right-click → Convert to input).
                # Legal, and common: ResolutionSelector driving width/height.
                # The value stays in widgets_values as the unwired fallback.
                pass
            elif slot.get("link") is not None and name != "h3_director_panels":
                problems.append(
                    "node %s (%s): wired input %r is not in the schema"
                    % (node.get("id"), ntype, name))

        # widgets: positional array vs schema
        # 命名副本先读 —— 数组是按位置存的，只有命名副本带名字，数组的真实槽位
        # 表要靠它来还原（见下面 slots）。
        names = [n for w in schema["widgets"] for n in widget_slots(w)]
        named = node.get("widgets_values_named")
        named = named if isinstance(named, dict) else None
        values = node.get("widgets_values")
        if values is None:
            values = []

        # 数组里排在 schema 控件**之后**的那几格是前端自己挂的 DOM widget
        # （本包是 ``h3_director_panels``：面板展开状态）。它不在 schema 里，
        # 只在命名副本里有名字，所以按位置比对的真实槽位表是 ``names + 那些
        # 认不出来的键``。以前把"多出来的格子"当成异常单独判断，结果是一个控件
        # 缺席 + 一个 DOM 槽 = 数组长度刚好"对得上"，两条副本互相打掩护。
        dom_keys = [k for k in (named or {}) if k not in names]
        slots = names + dom_keys

        # 文件是在某个控件被加进 schema **之前**存的：那个控件在数组和命名副本
        # 里都缺席。补默认值必须插回它在 schema 里的那一格 —— 追加会插到 DOM 槽
        # 之后，从那里起整条数组全部错位，而 ComfyUI 读的就是数组这一份。
        absent = [n for n in names if named is not None and n not in named]
        if absent and len(values) == len(names) - len(absent) + len(dom_keys):
            problems.append(
                "node %s (%s): array predates %s -- %d values for %d slots "
                "(%d of them DOM keys)"
                % (node.get("id"), ntype, ", ".join(absent), len(values),
                   len(slots), len(dom_keys)))
            if fix:
                for name in absent:
                    at = min(names.index(name), len(values) - len(dom_keys))
                    values.insert(at, _default_for_slot(schema["widgets"], name))
                node["widgets_values"] = values
                fixed.append("node %s (%s): inserted %s at their schema slot"
                             % (node.get("id"), ntype, ", ".join(absent)))
        elif len(values) > len(slots):
            problems.append(
                "node %s (%s): %d widget values for %d slots -- %d extra "
                "(%d explained as DOM keys); values from here on may be shifted"
                % (node.get("id"), ntype, len(values), len(slots),
                   len(values) - len(slots), len(dom_keys)))
        elif len(values) < len(slots):
            missing = [n for n in slots[len(values):] if n in names]
            if missing:
                problems.append(
                    "node %s (%s): %d widget values for %d slots -- missing %s "
                    "(front-end falls back to schema defaults)"
                    % (node.get("id"), ntype, len(values), len(slots),
                       ", ".join(missing)))
                if fix:
                    for name in missing:
                        values.append(_default_for_slot(schema["widgets"], name))
                    node["widgets_values"] = values
                    fixed.append("node %s (%s): appended %s"
                                 % (node.get("id"), ntype, ", ".join(missing)))

        # widgets: named copy vs schema, and vs the positional copy
        if named is not None:
            stale = [k for k in named if k not in names and k != "h3_director_panels"]
            if stale:
                problems.append(
                    "node %s (%s): named copy has stale keys %s"
                    % (node.get("id"), ntype, ", ".join(map(str, stale))))
            if absent:
                problems.append(
                    "node %s (%s): named copy is missing %s"
                    % (node.get("id"), ntype, ", ".join(absent)))
                if fix:
                    for name in absent:
                        named[name] = _default_for_slot(schema["widgets"], name)
                    fixed.append("node %s (%s): named += %s"
                                 % (node.get("id"), ntype, ", ".join(absent)))
            # the two copies must agree where both are present -- walked through
            # ``slots`` so a DOM key never borrows a schema widget's array value
            for i, name in enumerate(slots):
                if name not in names:
                    continue
                if i < len(values) and name in named and named[name] != values[i]:
                    problems.append(
                        "node %s (%s): %s disagrees -- array %r vs named %r"
                        % (node.get("id"), ntype, name, values[i], named[name]))

    if fix and fixed:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return problems, fixed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="workflow JSON(s); default workflows/*.json")
    ap.add_argument("--fix", action="store_true",
                    help="append missing widget defaults to both copies")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    paths = args.paths or [
        os.path.join(here, "..", "workflows", f)
        for f in sorted(os.listdir(os.path.join(here, "..", "workflows")))
        if f.endswith(".json")
    ]

    schemas = load_schemas()
    print("pack nodes: %s\n" % ", ".join(sorted(schemas)))

    bad = 0
    for path in paths:
        path = os.path.abspath(path)
        problems, fixed = check_file(path, schemas, fix=args.fix)
        print("== %s" % os.path.basename(path))
        for line in problems:
            print("   ! %s" % line)
        for line in fixed:
            print("   + %s" % line)
        if not problems and not fixed:
            print("   ok")
        bad += len(problems)
    print("\n%s" % ("all clean" if not bad else "%d problem(s)" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
