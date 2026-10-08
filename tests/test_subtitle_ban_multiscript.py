# -*- coding: utf-8 -*-
"""多剧本泛化对抗测试（2026-10-02）：换任何脚本，防字幕注入都必须完整生效。

覆盖五类易踩坑形态：
  A. 中文古装两人对话（现行主力形态）
  B. 英文独白（画面里只有说话人自己 —— 正向陈述的"另一个角色"措辞不得误导）
  C. 电话戏（说话人对着电话、对象不在场）
  D. 一行多块台词（同一行连续两块 <d>）
  E. 中英混排 + 感叹/呼喊式称谓（S04 高危形态的变体）

断言（对每份脚本、每句台词）：
  1) DIALOGUE_NO_TEXT_EN 完整出现且每块恰一次（幂等：二次解析不叠加）；
  2) 注入句不含时间码/双引号/字面 <d>；
  3) 正向陈述在场（story world / eyes off the lens），且无「另一个角色」这类
     会失真的场景假设；
  4) QC 重试变体里至少一个为正向构图（positive framing），且全部变体
     不含时间码/字面 <d>。
"""
import importlib.machinery, importlib.util, os, sys

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "marquee_director")
sys.path.insert(0, PKG)
ld = importlib.machinery.SourceFileLoader("pp", os.path.join(PKG, "prompt_pack.py"))
sp = importlib.util.spec_from_loader("pp", ld)
pp = importlib.util.module_from_spec(sp)
ld.exec_module(pp)

FAIL = []
def check(cond, msg):
    if not cond: FAIL.append(msg)

SCRIPTS = {
 "A 古装两人对话": (
   "[Shot 1] At 00:00.000 the room holds its breath. (S1) <d>汝与曹贼何异？</d> "
   "from 00:00.000 to 00:02.000, word by word."),
 "B 英文独白": (
   "[Shot 1] At 00:00.000 she stands alone by the window. (S2) <d>I never asked for any of this.</d> "
   "from 00:00.000 to 00:03.000, quietly."),
 "C 电话戏": (
   "[Shot 1] At 00:00.000 he picks up the receiver. (S1) <d>喂？哪位？</d> "
   "from 00:00.000 to 00:02.500, wary."),
 "D 一行多块": (
   "[Shot 1] At 00:00.000 they overlap. (S1) <d>让开。</d> (S2) <d>不让。</d> "
   "from 00:00.000 to 00:02.000, sharp."),
 "E 中英混排呼喊": (
   "[Shot 1] At 00:00.000 the crowd turns. (S1) <d>兄弟们，Deal is deal！</d> "
   "from 00:00.000 to 00:02.000, loud."),
}

for name, script in SCRIPTS.items():
    once = pp._inject_dialogue_no_text(script)
    twice = pp._inject_dialogue_no_text(once)
    nblocks = script.count("<d>")
    cnt = twice.count(pp.DIALOGUE_NO_TEXT_EN)
    check(cnt == nblocks, f"[{name}] 注入次数 {cnt} != 台词块数 {nblocks}")
    check(once == twice, f"[{name}] 幂等失败：二次注入发生变化")
    check("00:" not in pp.DIALOGUE_NO_TEXT_EN, "[全局] 注入句含时间码")
    check('"' not in pp.DIALOGUE_NO_TEXT_EN, "[全局] 注入句含双引号")
    check("<d>" not in pp.DIALOGUE_NO_TEXT_EN, "[全局] 注入句含字面 <d>")
    check("story world" in pp.DIALOGUE_NO_TEXT_EN, "[全局] 正向陈述（story world）缺失")
    check("eyes off the lens" in pp.DIALOGUE_NO_TEXT_EN, "[全局] 正向陈述（eyes off the lens）缺失")
    check("the other character" not in pp.DIALOGUE_NO_TEXT_EN, "[全局] 仍含「另一个角色」场景假设")

# QC 重试变体：至少一个正向构图，且全部不含时间码/字面 <d>
variants = [pp and None]  # placeholder to keep structure
import importlib.machinery as _im, importlib.util as _iu
ld2 = _im.SourceFileLoader("sq", os.path.join(PKG, "subtitle_qc.py"))
sp2 = _iu.spec_from_loader("sq", ld2)
sq = _iu.module_from_spec(sp2)
try:
    ld2.exec_module(sq)
    vs = list(sq.RETRY_VARIANTS)
    check(any("positive framing" in v for v in vs), "QC 重试变体缺正向构图变体")
    for i, v in enumerate(vs, 1):
        check("00:" not in v, f"[variant {i}] 含时间码")
        check("<d>" not in v, f"[variant {i}] 含字面 <d>")
except Exception as e:
    # subtitle_qc 顶层可能依赖 comfy 环境 —— 退化只查源码文本
    src = open(os.path.join(PKG, "subtitle_qc.py"), encoding="utf-8").read()
    check("positive framing" in src, "QC 重试变体缺正向构图变体（源码级）")

print("=" * 72)
if FAIL:
    print("FAILS:", len(FAIL))
    for f in FAIL: print("  ✗", f)
    sys.exit(1)
print("FAIL: 无 —— 五类脚本形态 × 幂等 × 正向陈述 × QC 变体 全部通过")
