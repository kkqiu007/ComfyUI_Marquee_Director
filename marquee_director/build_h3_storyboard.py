#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_h3_storyboard.py
========================
一键把「MiniMax-H3-五分镜.json」模板升级成「任意分镜数量」的 ComfyUI 工作流。

设计思路（借鉴 h3_ref2va_auto_script_board.json 的"脚本→分镜资产JSON"逻辑）：
  1. h3_ref2va_auto_script_board.json 里的 MinimaxH3ScriptComplete /
     MinimaxH3AssetExtract / MinimaxH3ShotSplit / MinimaxH3ScriptBoard
     这几个节点，负责把一段完整剧本，通过大模型自动拆成 N 个分镜（N 不固定），
     并把结果收纳进一份 JSON（分镜资产结果JSON）里 —— "分镜数量" 从此变成
     JSON 数组的长度，而不是写死的节点个数。
  2. 但 ComfyUI 的工作流本质是静态 DAG，没有原生"跑 N 次循环"的能力；
     五镜模板正是把 N=5 写死成 5 个手工复制的 H3RenderSegment 节点。
  3. 本脚本就是把第 1 步产出的"任意 N 条分镜"，按第 2 步的连接方式，
     自动展开成 N 个正确连线的 H3RenderSegment 节点（settings / chain_state /
     参考图 0-3 全部照抄模板的接法），首尾自动接上"整片拼接"和"单段修复"
     两个开关分组 —— 生成完直接在 ComfyUI 里点一次 Queue Prompt，
     就是"一键"跑完全部分镜，实现真正的"无限分镜"。
  4. 单段修复开关（① 启用单段修复 + ② 要重做的分镜号）保持原有 Switch 逻辑
     不变，只是把提示文字从"(1-5)"自动改成"(1-N)"，天然支持任意分镜数。

用法：
    python build_h3_storyboard.py \
        --template "MiniMax-H3-五分镜.json" \
        --shots shots.json \
        --out my_workflow_N.json

shots.json 格式（数组，长度任意，即"无限分镜"）：
[
  {
    "title": "分镜1：xxx",       // 可选，仅用于节点标题，方便你在 ComfyUI 里认领
    "prompt": "完整的 H3 分镜提示词……",
    "seconds": 10,               // 本镜时长（秒）
    "handoff_seconds": 1.625,    // 与上一镜的重叠交接秒数，第 1 镜通常填 0
    "seed_override": 0,          // 可选，默认 0（不覆盖）
    "unload_models_after": true  // 可选，默认 true
  },
  ...
]

可以手工写，也可以用同目录的 assets_json_to_shots.py 从
"分镜资产结果JSON"（h3_ref2va_auto_script_board 工作流跑出来的那份 JSON）
自动转换出来。
"""

import argparse
import copy
import json
import sys

# ------------------------------------------------------------------
# 模板里固定不变的部分（① 公共设置 / 参考图 / ③④⑤ 开关分组）会原样保留，
# 这里只记录"跟分镜生成强相关"的节点 id，用来定位和改写。
# ------------------------------------------------------------------
OLD_SEGMENT_IDS = [12, 15, 16, 45, 46]          # 五镜模板里手工写死的 5 个分镜节点
CHAIN_SETTINGS_ID = 9                            # H3ChainSettings，settings 的来源
REF_IMAGE_IDS = {                                # images.image_0..3 <- 这几张参考图
    "images.image_0": 13,
    "images.image_1": 14,
    "images.image_2": 32,
    "images.image_3": 51,
}
MODE_SWITCH_ID = 50          # LazySwitch：模式切换：整片 / 单段修复
PREVIEW_SWITCH_ID = 52       # LazySwitch：预览切换：整片摘要 / 单段修复摘要
SEGNUM_CONST_ID = 49         # INTConstant：② 要重做的分镜号
GROUP2_TITLE_PREFIX = "② 分镜生成：按顺序续接，每段在此独立生成"

DEFAULT_HANDOFF = 1.625


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def index_nodes(wf):
    return {n["id"]: n for n in wf["nodes"]}


def next_id_counters(wf):
    node_id = wf.get("last_node_id", max(n["id"] for n in wf["nodes"])) 
    link_id = wf.get("last_link_id", max(l[0] for l in wf["links"]))
    return {"node": node_id, "link": link_id}


def new_id(counter, key):
    counter[key] += 1
    return counter[key]


def make_link(counter, origin_id, origin_slot, target_id, target_slot, ltype):
    lid = new_id(counter, "link")
    return [lid, origin_id, origin_slot, target_id, target_slot, ltype]


def build_segment_node(template_node, node_id, pos, shot, order):
    """基于模板的 H3RenderSegment 节点克隆出一个新的分镜节点（先不连线）。"""
    node = copy.deepcopy(template_node)
    node["id"] = node_id
    node["pos"] = pos
    node["order"] = order
    node["mode"] = 0  # 强制启用，避免继承模板里可能残留的旁路(mode=4)状态
    node["title"] = shot.get("title") or f"分镜 {order}"

    # 清空所有输入的 link，稍后统一重新连线
    for i in node["inputs"]:
        i["link"] = None
    for o in node["outputs"]:
        o["links"] = None

    widgets_named = {
        "resume": True,
        "prompt": shot.get("prompt", ""),
        "seconds": shot.get("seconds", 10),
        "handoff_seconds": shot.get("handoff_seconds", DEFAULT_HANDOFF),
        "seed_override": shot.get("seed_override", 0),
        "unload_models_after": shot.get("unload_models_after", True),
    }
    node["widgets_values_named"] = widgets_named
    # widgets_values 数组顺序跟模板保持一致
    node["widgets_values"] = [
        widgets_named["resume"],
        widgets_named["prompt"],
        widgets_named["seconds"],
        widgets_named["handoff_seconds"],
        widgets_named["seed_override"],
        widgets_named["unload_models_after"],
    ]
    return node


def rewire_segment_inputs(node, counter, chain_settings_id, prev_chain_output, ref_image_ids, nodes_by_id):
    """把一个新分镜节点的 settings / chain_state / 参考图 输入连好。"""
    links_out = []
    for i in node["inputs"]:
        if i["name"] == "settings":
            lk = make_link(counter, chain_settings_id, 0, node["id"], _slot_index(node, "settings"), "H3_SETTINGS")
            i["link"] = lk[0]
            links_out.append(lk)
            nodes_by_id[chain_settings_id]["outputs"][0].setdefault("links", [])
            (nodes_by_id[chain_settings_id]["outputs"][0]["links"] or []).append(lk[0])
        elif i["name"] == "chain_state" and prev_chain_output is not None:
            prev_id, prev_slot = prev_chain_output
            lk = make_link(counter, prev_id, prev_slot, node["id"], _slot_index(node, "chain_state"), "H3_CHAIN")
            i["link"] = lk[0]
            links_out.append(lk)
            out_node = nodes_by_id[prev_id]
            out_node["outputs"][prev_slot].setdefault("links", [])
            (out_node["outputs"][prev_slot]["links"] or []).append(lk[0])
        elif i["name"] in ref_image_ids:
            src_id = ref_image_ids[i["name"]]
            lk = make_link(counter, src_id, 0, node["id"], _slot_index(node, i["name"]), "IMAGE")
            i["link"] = lk[0]
            links_out.append(lk)
            nodes_by_id[src_id]["outputs"][0].setdefault("links", [])
            (nodes_by_id[src_id]["outputs"][0]["links"] or []).append(lk[0])
    return links_out


def _slot_index(node, name):
    for idx, i in enumerate(node["inputs"]):
        if i["name"] == name:
            return idx
    raise KeyError(name)


def _out_slot_index(node, name):
    for idx, o in enumerate(node["outputs"]):
        if o["name"] == name:
            return idx
    raise KeyError(name)


def rewrite_final_links(wf, nodes_by_id, last_seg_id, counter):
    """把 last_seg 的 chain_state / summary 接到③组的两个 LazySwitch 上，
    替换掉原来指向旧第 5 镜(id=46) 的连线。"""
    mode_switch = nodes_by_id[MODE_SWITCH_ID]
    preview_switch = nodes_by_id[PREVIEW_SWITCH_ID]
    last_seg = nodes_by_id[last_seg_id]

    chain_slot = _out_slot_index(last_seg, "chain_state")
    summary_slot = _out_slot_index(last_seg, "summary")

    new_links = []

    # on_false (整片模式) <- last_seg.chain_state
    for i in mode_switch["inputs"]:
        if i["name"] == "on_false":
            lk = make_link(counter, last_seg_id, chain_slot, MODE_SWITCH_ID, _slot_index(mode_switch, "on_false"), "*")
            i["link"] = lk[0]
            new_links.append(lk)
            last_seg["outputs"][chain_slot].setdefault("links", [])
            (last_seg["outputs"][chain_slot]["links"] or []).append(lk[0])

    # on_false (整片摘要) <- last_seg.summary
    for i in preview_switch["inputs"]:
        if i["name"] == "on_false":
            lk = make_link(counter, last_seg_id, summary_slot, PREVIEW_SWITCH_ID, _slot_index(preview_switch, "on_false"), "*")
            i["link"] = lk[0]
            new_links.append(lk)
            last_seg["outputs"][summary_slot].setdefault("links", [])
            (last_seg["outputs"][summary_slot]["links"] or []).append(lk[0])

    return new_links


def remove_old_segments_and_their_links(wf, old_ids):
    old_ids = set(old_ids)

    # 先找出所有即将被删除的 link id（起点或终点是旧分镜节点的）
    removed_link_ids = set()
    for l in wf["links"]:
        _id, o, oslot, t, tslot, typ = l
        if o in old_ids or t in old_ids:
            removed_link_ids.add(_id)

    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in old_ids]

    # 清理幸存节点里残留的 link 引用（比如参考图/H3ChainSettings 的 outputs.links
    # 数组里还挂着指向旧分镜节点的 link id，以及个别 input.link 字段）
    for n in wf["nodes"]:
        for i in n.get("inputs", []):
            if i.get("link") in removed_link_ids:
                i["link"] = None
        for o in n.get("outputs", []):
            if o.get("links"):
                o["links"] = [lk for lk in o["links"] if lk not in removed_link_ids] or None

    wf["links"] = [l for l in wf["links"] if l[0] not in removed_link_ids]


def resize_group2(wf, n_shots, node_width=420, gap=60, left_margin=45):
    for g in wf["groups"]:
        if g["title"].startswith(GROUP2_TITLE_PREFIX) or g["title"].startswith("② 分镜生成"):
            x, y, w, h = g["bounding"]
            new_w = max(w, left_margin + n_shots * (node_width + gap) + 60)
            g["bounding"] = [x, y, new_w, h]
            g["title"] = f"② 分镜生成：按顺序续接，共 {n_shots} 镜，每段在此独立生成"


def relabel_segment_number_hint(wf, n_shots):
    for n in wf["nodes"]:
        if n["id"] == SEGNUM_CONST_ID:
            n["title"] = f"② 要重做的分镜号（1-{n_shots}）"
        if n["type"] == "MarkdownNote" and "分镜逻辑顺序" in n.get("title", ""):
            n["title"] = f"分镜逻辑顺序 1→{n_shots}（剧情/镜头/接缝）"


def main():
    ap = argparse.ArgumentParser(description="把五分镜模板升级为任意分镜数量的一键工作流")
    ap.add_argument("--template", required=True, help="MiniMax-H3-五分镜.json 路径")
    ap.add_argument("--shots", required=True, help="shots.json 路径（任意长度的分镜数组）")
    ap.add_argument("--out", required=True, help="输出的新工作流 json 路径")
    args = ap.parse_args()

    wf = load_json(args.template)
    shots = load_json(args.shots)
    if not isinstance(shots, list) or len(shots) == 0:
        print("shots.json 必须是一个非空数组", file=sys.stderr)
        sys.exit(1)

    nodes_by_id = index_nodes(wf)
    template_seg = nodes_by_id[OLD_SEGMENT_IDS[0]]  # 拿第一个分镜节点当克隆模板
    group2 = next(g for g in wf["groups"] if g["title"].startswith("②"))
    gx, gy = group2["bounding"][0], group2["bounding"][1]

    # 先把旧的 5 个分镜节点和它们的连线全部摘掉，稍后重新生成
    remove_old_segments_and_their_links(wf, OLD_SEGMENT_IDS)
    nodes_by_id = index_nodes(wf)  # 摘除后重新建索引

    counter = next_id_counters(wf)

    new_nodes = []
    prev_chain_out = None  # (node_id, slot_index)
    x = gx + 60
    y = gy + 260
    for order, shot in enumerate(shots, start=1):
        nid = new_id(counter, "node")
        node = build_segment_node(template_seg, nid, [x, y], shot, order)
        new_nodes.append(node)
        nodes_by_id[nid] = node
        x += 420 + 60

    # 统一插入 wf["nodes"]，再重新连线（连线时依赖 nodes_by_id 已包含所有新节点）
    wf["nodes"].extend(new_nodes)

    all_new_links = []
    prev_chain_out = None
    for node in new_nodes:
        links_out = rewire_segment_inputs(
            node, counter, CHAIN_SETTINGS_ID, prev_chain_out, REF_IMAGE_IDS, nodes_by_id
        )
        all_new_links.extend(links_out)
        chain_slot = _out_slot_index(node, "chain_state")
        prev_chain_out = (node["id"], chain_slot)

    final_links = rewrite_final_links(wf, nodes_by_id, new_nodes[-1]["id"], counter)
    all_new_links.extend(final_links)

    wf["links"].extend(all_new_links)

    n_shots = len(shots)
    resize_group2(wf, n_shots)
    relabel_segment_number_hint(wf, n_shots)

    wf["last_node_id"] = counter["node"]
    wf["last_link_id"] = counter["link"]

    save_json(wf, args.out)
    print(f"✅ 已生成 {n_shots} 镜的一键工作流：{args.out}")
    print(f"   新分镜节点 id：{[n['id'] for n in new_nodes]}")
    print(f"   单段修复分镜号范围已自动更新为 1-{n_shots}")


if __name__ == "__main__":
    main()
