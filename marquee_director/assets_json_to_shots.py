#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assets_json_to_shots.py
==========================
把"脚本转分镜"节点（MinimaxH3ShotSplit / MinimaxH3ScriptBoard /
MinimaxH3ScriptConverter，来自 h3_ref2va_auto_script_board.json）
输出的"分镜资产结果JSON"，转换成 build_h3_storyboard.py 需要的
shots.json（数组，每个元素一个分镜）。

⚠️ 重要提示：
  MinimaxH3ScriptBoard 节点的 edit_json 默认模板长这样：
    {
      "script": "...", "script_name": "...", "duration": 10,
      "width": 864, "height": 480, "shots_prompt": "...",
      "global": {"roles": [], "prop": [], "scene": [],
                 "background_audio": "...", "global_prompt": "..."},
      "shots_info": []
    }
  但真正跑完剧本后，"shots_info" 数组里每一条分镜对象具体用什么字段名
  （比如是叫 prompt 还是 生视频提示词，是叫 duration 还是 seconds），
  取决于你安装的这套 H3 自定义节点包的实际实现，我这边拿到的两份工作流
  JSON 里 shots_info 都是空数组，没有真实样例可以照抄。

  所以这个脚本按"常见候选字段名"做了自动匹配（见 CANDIDATE_KEYS），
  如果匹配不上，会把这一条分镜的全部原始字段打印出来，方便你确认后
  改一下 CANDIDATE_KEYS 或者手动改 shots.json 里的对应字段。

用法：
    python assets_json_to_shots.py --asset 分镜资产结果JSON.json --out shots.json
"""

import argparse
import json
import sys

CANDIDATE_KEYS = {
    "prompt": ["prompt", "video_prompt", "生视频提示词", "H3剧本提示词", "shot_prompt", "text"],
    "seconds": ["seconds", "duration", "时长", "shot_duration"],
    "title": ["title", "name", "标题", "shot_name"],
    "handoff_seconds": ["handoff_seconds", "overlap_seconds", "接缝秒数"],
    "seed_override": ["seed_override", "seed"],
}


def pick(d, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, help="分镜资产结果JSON 文件路径")
    ap.add_argument("--out", required=True, help="输出 shots.json 路径")
    ap.add_argument("--default-handoff", type=float, default=1.625)
    args = ap.parse_args()

    with open(args.asset, "r", encoding="utf-8") as f:
        raw = json.load(f)

    shots_info = raw.get("shots_info") if isinstance(raw, dict) else raw
    if not shots_info:
        print("⚠️ 没有在输入 JSON 里找到非空的 shots_info 数组。", file=sys.stderr)
        print("   顶层字段有：", list(raw.keys()) if isinstance(raw, dict) else type(raw), file=sys.stderr)
        sys.exit(1)

    shots = []
    for idx, item in enumerate(shots_info, start=1):
        if not isinstance(item, dict):
            print(f"⚠️ 第 {idx} 条分镜不是对象，原样跳过：{item}", file=sys.stderr)
            continue
        prompt = pick(item, CANDIDATE_KEYS["prompt"])
        if prompt is None:
            print(f"⚠️ 第 {idx} 条分镜没匹配到 prompt 字段，原始字段：{list(item.keys())}", file=sys.stderr)
            print("   请打开脚本顶部 CANDIDATE_KEYS 加上正确的字段名，或手动填 shots.json。", file=sys.stderr)
            prompt = ""
        shots.append({
            "title": pick(item, CANDIDATE_KEYS["title"], f"分镜 {idx}"),
            "prompt": prompt,
            "seconds": pick(item, CANDIDATE_KEYS["seconds"], 10),
            "handoff_seconds": pick(item, CANDIDATE_KEYS["handoff_seconds"], 0 if idx == 1 else args.default_handoff),
            "seed_override": pick(item, CANDIDATE_KEYS["seed_override"], 0),
            "unload_models_after": True,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=1)

    print(f"✅ 已转换 {len(shots)} 条分镜 -> {args.out}")
    print("   请务必打开 shots.json 检查 prompt 字段是否完整，再喂给 build_h3_storyboard.py。")


if __name__ == "__main__":
    main()
