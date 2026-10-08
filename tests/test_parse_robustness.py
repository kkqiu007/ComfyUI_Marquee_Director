# -*- coding: utf-8 -*-
"""解析/注入层防回归（2026-09-30 四层链路排查产物，换任何脚本都必须成立）。

跑法：  python tests/test_parse_robustness.py   （全绿退出码 0）

覆盖 2026-09-30 修复的 8 类故障，全部写成「节点代码必须自动处理」的硬断言
—— 判据与具体剧本内容无关，换脚本同样有效：

  R1  接续段段头漏写 +1.6：replay_in 必须按网格回放窗补齐（与渲染层同口径），
      回放句自动补、回放区内的台词自动平移出窗。
  R2  段头无斜杠（``### S01 ###``）：按分段 PACK 解析（缺省时长），不得静默
      落进整段式路径。
  R3  段头语言槽位写 hard cut：按硬切段处理 + fixes 留痕。
  R4  台词窗时间码缺毫秒（``from 00:01 to 00:03``）：交付提示词里补成 .000。
  R5  整段式中文稿的行外散文被镜头行过滤丢弃时：必须响亮报 issue（不许静默）。
  R6  BOM（\ufeff）开头：解析结果与无 BOM 完全一致。
  R7  en-pack 分段路径必须带全部逐句纪律（LIPSYNC / DELIVERY 只说一次 /
      AUDIO-ONLY 防字幕）+ 说话人自动补全留痕；重复解析幂等。
  R8  en-pack 不做行首 hoist / 条目时间码改写（已规范化的行结构必须原样保留）。
"""
import importlib.machinery
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

ld = importlib.machinery.SourceFileLoader(
    "md_prompt_pack", os.path.join(PKG, "marquee_director", "prompt_pack.py"))
sp = importlib.util.spec_from_loader("md_prompt_pack", ld)
pp = importlib.util.module_from_spec(sp)
ld.exec_module(pp)

FAILS = []


def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        FAILS.append(msg)


def seg_head(tag, dur=None, lang="EN"):
    extra = (" / %s" % dur if dur else "") + (" / %s" % lang if lang else "")
    return "########## %s%s ##########" % (tag, extra)


def mini_pack(segs, project="防回归压测"):
    head = ("==========================================================\n"
            "MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包\n"
            "Project / 项目         : %s\n"
            "Mode / 模式            : Ref2VA (full reference)\n"
            "Total duration / 总时长 : 60s\n"
            "Segments / 段数         : %d\n"
            "Aspect / 画幅           : 9:16 vertical\n"
            "==========================================================\n"
            % (project, len(segs)))
    return head + "\n".join(segs)


def seg(tag, body, dur=None, lang="EN"):
    return seg_head(tag, dur, lang) + "\n" + body


SUBJECT = ("subject_definitions:\n"
           "<Subject 1> is the man in <Picture 1>.\n"
           "<Subject 2> is the woman in <Picture 2>.\n"
           "<Picture 1> is the reference image for Subject 1.\n"
           "<Picture 2> is the reference image for Subject 2.\n")
SUMMARY = "summary:\n[reference generation] A robustness fixture.\n"
RETENTION = ("retention_analysis:\n"
             "<Subject 1> (appears in [Shot 1]): fully_preserved.\n"
             "<Picture 1>: fully_preserved.\n")
SOUND = "overall_soundscape:\nRoom tone only.\n"
MUSIC = "non_diegetic_music:\nNo music.\n"
NOTEXT = ("No on-screen text, subtitles, captions, timecodes, or any graphics "
          "overlays anywhere in the frame. ")
STYLE = ("Style: cinematic test drama, warm candlelight, 2K, 16:9, 24 fps. "
         "Camera grammar: locked-off static camera, 50 mm, eye level. "
         "The appearance of every person and object follows the reference "
         "images exactly and nothing is invented.")


def detail(shots, first_no_text=True):
    head = (NOTEXT + STYLE) if first_no_text else STYLE
    return ("detailed_description:\n" + head + "\n" + shots + "\n"
            + SOUND + "\n" + MUSIC)


def parse(text):
    return pp.parse_pack(text, language="auto", default_duration=8.0,
                         auto_fix=True)


def delivered_shot(segments, index=0):
    return pp.to_shots_json(segments, session_name="t")["shots_info"][index]["shot"]


def dlg(spk, text, a, b):
    return "(S%d) <d>%s</d> from %s to %s." % (spk, text, a, b)


print("=" * 72)
print("R1 接续段漏写 +1.6：回放窗对齐 + 回放区台词平移（与渲染层同口径）")
print("=" * 72)
segs, meta, issues = parse(mini_pack([
    seg("S01", SUBJECT + SUMMARY + RETENTION +
        detail("[Shot 1] 00:00.000 A man waits in the candlelight. "
               + dlg(1, "第一句", "00:02.000", "00:04.000"))),
    seg("S02", SUBJECT + SUMMARY + RETENTION +
        detail("[Shot 1] 00:00.000 replay. [Shot 2] At 00:01.600 new.\n"
               + dlg(1, "回放里说话", "00:00.800", "00:01.500") + "\n"
               + dlg(1, "正常", "00:03.000", "00:05.000"))),
]))
s2 = next(s for s in segs if s["id"] == "S02")
check(abs(float(s2["replay_in"]) - pp.grid_seconds(pp.HANDOFF_SECONDS)) < 1e-9,
      "R1a 段头无 +1.6 时 replay_in=%.3f（= 网格回放窗 1.625）"
      % float(s2["replay_in"]))
fix2 = " ".join(s2.get("fixes") or ())
check("回放区台词后移" in fix2, "R1b 回放区台词自动平移（fixes: %s)" % fix2[:60])
shot2 = delivered_shot(segs, 1)
_ln = next((ln for ln in shot2.splitlines() if "回放里说话" in ln), "")
m = re.search(r"from\s*([\d:.]+)\s*to\s*([\d:.]+)", _ln) if _ln else None
def _t(x):
    mm, ss = x.split(":")
    return int(mm) * 60 + float(ss)
check(bool(m) and _t(m.group(1)) >= float(s2["replay_in"]) - 1e-3,
      "R1c 交付提示词里该句起点 %s ≥ 回放窗（不在会被回放的区域）"
      % (m.group(1) if m else "未找到"))
del m

print()
print("=" * 72)
print("R2 段头无斜杠（### S01 ###）按分段 PACK 解析")
print("=" * 72)
segs, meta, issues = parse(mini_pack([
    seg("S01", SUBJECT + SUMMARY + RETENTION +
        detail("[Shot 1] 00:00.000 A. " + dlg(1, "甲", "00:02.000", "00:04.000")),
        dur=None, lang=""),
], project="R2"))
check(len(segs) == 1 and meta.get("layout") != pp.SHOT_SCRIPT_LAYOUT,
      "R2a 解析为分段（layout=%s，段数=%d）" % (meta.get("layout"), len(segs)))
check(abs(float(segs[0]["new_seconds"]) - 8.0) < 1e-9,
      "R2b 缺省时长生效（new=%.2f，不再从镜内时间码反推）"
      % float(segs[0]["new_seconds"]))

print()
print("=" * 72)
print("R3 段头语言槽位 hard cut → 硬切段")
print("=" * 72)
segs, meta, issues = parse(mini_pack([
    seg("S01", SUBJECT + SUMMARY + RETENTION +
        detail("[Shot 1] 00:00.000 A. " + dlg(1, "甲", "00:02.000", "00:04.000"))),
    seg("S02", SUBJECT + SUMMARY + RETENTION +
        detail("[Shot 1] 00:00.000 B. " + dlg(1, "乙", "00:02.000", "00:04.000")),
        dur="8s", lang="hard cut"),
]))
s2 = next(s for s in segs if s["id"] == "S02")
check(s2.get("hard_cut") is True, "R3a hard_cut=True（以前静默丢失）")
check(any("hard cut" in (f or "") for f in (s2.get("fixes") or ())),
      "R3b fixes 留痕（用户可见，不静默生效）")

print()
print("=" * 72)
print("R4 台词窗时间码缺毫秒 → 交付提示词补齐 .000")
print("=" * 72)
segs, meta, issues = parse(mini_pack([
    seg("S01", SUBJECT + SUMMARY + RETENTION +
        detail("[Shot 1] 00:00.000 A. " + dlg(1, "甲", "00:01", "00:03"))),
]))
shot1 = delivered_shot(segs, 0)
check("from 00:01.000 to 00:03.000" in shot1,
      "R4 from/to 补成 .000（幂等，已有毫秒位不动）")

print()
print("=" * 72)
print("R5 整段式行外散文被丢弃 → 必须响亮报 issue")
print("=" * 72)
zh_raw = ("detailed_description:\n"
          "这里是一段没有时间码的叙事散文，讲的是烛光下的卧房与两个人的对峙，"
          "按整段式契约它应该写进镜头行内，否则拆段时会被镜头行过滤丢掉，"
          "所以解析节点必须把这种损失响亮地报出来。\n"
          "[Shot 1] 00:00.000 他开口质问。(S1) <d>信呢？</d> from 00:02.000 to 00:04.000.\n")
segs, meta, issues = parse(zh_raw)
joined = "\n".join(issues)
check(meta.get("layout") == pp.SHOT_SCRIPT_LAYOUT and
      any("块外汉字" in i and "只保留" in i for i in issues),
      "R5 静默丢正文闸门触发（issues 提示散文写进镜头行）")

print()
print("=" * 72)
print("R6 BOM 开头与无 BOM 解析完全一致")
print("=" * 72)
base = mini_pack([
    seg("S01", SUBJECT + SUMMARY + RETENTION +
        detail("[Shot 1] 00:00.000 A. " + dlg(1, "甲", "00:02.000", "00:04.000"))),
])
a_segs, a_meta, a_issues = parse(base)
b_segs, b_meta, b_issues = parse("\ufeff" + base)
check((len(a_segs), len(a_issues)) == (len(b_segs), len(b_issues)) and
      a_segs[0]["new_seconds"] == b_segs[0]["new_seconds"],
      "R6 \\ufeff 不再卡死行首正则（段数/issue 数/时长一致）")

print()
print("=" * 72)
print("R7 en-pack 分段路径带全部逐句纪律 + 说话人补全 + 幂等")
print("=" * 72)
segs, meta, issues = parse(mini_pack([
    seg("S01", SUBJECT + SUMMARY + RETENTION +
        detail("[Shot 1] 00:00.000 <Subject 1> waits by the bed. <d>谁在说话</d> "
               "from 00:02.000 to 00:04.000.\n"
               "[Shot 2] At 00:05.000 <Subject 1> turns. <d>另一句</d> "
               "from 00:05.000 to 00:07.000.")),
]))
shot1 = delivered_shot(segs, 0)
check(shot1.count(pp.DIALOGUE_NO_TEXT_EN) == 2,
      "R7a 每句台词 1 条 AUDIO-ONLY 防字幕禁令（台词 2 块）")
check("DELIVERY: Speak this line EXACTLY ONCE" in shot1,
      "R7b DELIVERY 只说一次纪律在 en-pack 交付里（此前只有整段式路径有）")
check("ip-sync" in shot1, "R7c LIPSYNC 口型纪律在 en-pack 交付里")
check(shot1.count("(S1)") + shot1.count("(S2)") >= 2,
      "R7d 说话人编号自动补全（无 (Sx) 的台词由上下文推断）")
check(any("推断" in i for i in issues),
      "R7e 推断结果写进 issues（用户可复核，不静默）")
# 幂等：把交付文本再喂回去解析一遍（无段头 → 整段式路径，可能拆成多段），
# 纪律句按**全部段合计**不得叠加
re_segs, _m, _i = parse(shot1)
re_shots = pp.to_shots_json(re_segs, session_name="t")["shots_info"]
tot_ban = sum(sh["shot"].count(pp.DIALOGUE_NO_TEXT_EN) for sh in re_shots)
tot_dlv = sum(sh["shot"].count("DELIVERY: Speak this line EXACTLY ONCE")
              for sh in re_shots)
check(tot_ban == 2 and tot_dlv == 2,
      "R7f 重复解析幂等（合计 ban=%d dlv=%d，不叠加）" % (tot_ban, tot_dlv))

print()
print("=" * 72)
print("R8 en-pack 不做行首 hoist / 条目改写（行结构原样保留）")
print("=" * 72)
line5 = "[Shot 5] At 00:02.900 a handheld close-up of <Subject 1>. " + \
    dlg(1, "台词原文", "00:02.900", "00:08.400") + " slow and clenched."
segs, meta, issues = parse(mini_pack([
    seg("S01", SUBJECT + SUMMARY + RETENTION + detail(line5)),
]))
shot1 = delivered_shot(segs, 0)
check(re.search(r"^\[Shot 5\] At 00:02\.900", shot1, re.M) is not None,
      "R8a 镜头行行首原样（无 At 前缀 hoist 打乱）")
check("from 00:02.900" in shot1,
      "R8b 台词窗 from/to 原样保留（hoist=False 不拆窗）")
first_sent = re.search(r"detailed_description:\s*\n([^\n]+)", shot1)
check(bool(first_sent) and "No on-screen text" in first_sent.group(1),
      "R8c 首句仍是 no-text 声明（validate_chain 全链一致性的前提）")

print()
print("=" * 72)
print("FAILS:", FAILS if FAILS else "无 —— 全部通过")
sys.exit(1 if FAILS else 0)
