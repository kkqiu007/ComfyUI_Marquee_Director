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

``--fix`` appends missing widget defaults at the **end** of both copies. It never
reorders and never touches existing values, because appending is the only edit
that a saved workflow survives without shifting.
"""

import argparse
import asyncio
import json
import os
import sys

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


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
        names = [n for w in schema["widgets"] for n in widget_slots(w)]
        values = node.get("widgets_values")
        if values is None:
            values = []
        if len(values) > len(names):
            # 前端的 DOM widget（本包是 ``h3_director_panels``：面板展开状态）
            # 不在 schema 里，但会在数组末尾占一个槽位。数组多出来的个数
            # 正好等于命名副本里那些"认不出来"的键时，它就是 DOM widget，
            # 不是移位 —— 不报问题。对不上才说明真的有值被推后。
            dom_keys = [k for k in (node.get("widgets_values_named") or {})
                        if k not in names]
            extra = len(values) - len(names)
            if extra != len(dom_keys):
                problems.append(
                    "node %s (%s): %d widget values for %d slots -- %d extra "
                    "(%d unexplained DOM/virtual keys); values from here on may "
                    "be shifted"
                    % (node.get("id"), ntype, len(values), len(names), extra,
                       len(dom_keys)))
        elif len(values) < len(names):
            missing = names[len(values):]
            problems.append(
                "node %s (%s): %d widget values for %d slots -- missing %s "
                "(front-end falls back to schema defaults)"
                % (node.get("id"), ntype, len(values), len(names), ", ".join(missing)))
            if fix:
                for name in missing:
                    values.append(_default_for_slot(schema["widgets"], name))
                node["widgets_values"] = values
                fixed.append("node %s (%s): appended %s"
                             % (node.get("id"), ntype, ", ".join(missing)))

        # widgets: named copy vs schema, and vs the positional copy
        named = node.get("widgets_values_named")
        if isinstance(named, dict):
            extra = [k for k in named if k not in names and k != "h3_director_panels"]
            if extra:
                problems.append(
                    "node %s (%s): named copy has stale keys %s"
                    % (node.get("id"), ntype, ", ".join(map(str, extra))))
            absent = [n for n in names if n not in named]
            if absent:
                problems.append(
                    "node %s (%s): named copy is missing %s"
                    % (node.get("id"), ntype, ", ".join(absent)))
                if fix:
                    for name in absent:
                        named[name] = _default_for_slot(schema["widgets"], name)
                    fixed.append("node %s (%s): named += %s"
                                 % (node.get("id"), ntype, ", ".join(absent)))
            # the two copies must agree where both are present
            for i, name in enumerate(names):
                if name in named and i < len(values):
                    if named[name] != values[i]:
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
