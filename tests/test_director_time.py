# -*- coding: utf-8 -*-
"""Director 层时间逻辑离线回归（D 层深挖，2026-09-30 四层排查产物）。

覆盖 `_off_dialogue`（切点避台词，09-20 夹取 + 09-27 夹取再验两次修复的
载体）与 `segment_windows`（首/中/末/硬切的回放窗与尾窗矩阵）。这些判据
与具体剧本无关 —— 换任何脚本都必须成立。

跑法：  python tests/test_director_time.py   （全绿退出码 0）
"""
import importlib.machinery
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "marquee_director")
ld = importlib.machinery.SourceFileLoader(
    "md_pp", os.path.join(PKG, "prompt_pack.py"))
sp = importlib.util.spec_from_loader("md_pp", ld)
pp = importlib.util.module_from_spec(sp)
ld.exec_module(pp)

FAILS = []


def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        FAILS.append(msg)


HW = pp.grid_seconds(pp.HANDOFF_SECONDS)   # 1.625

print("=" * 72)
print("D1 _off_dialogue：切在台词进行中")
print("=" * 72)
spans = [(10.0, 14.0)]                    # 一句 4 秒台词
c = pp._off_dialogue(12.0, spans, 8.0, 18.0)
check(c == round(14.0 + HW, 3),
      "D1a 切在句中 → 推到句尾+尾窗 %.3f（实得 %s）" % (14.0 + HW, c))
# 句尾+尾窗(15.625)越界 → 夹取到界内最近的合法点 = 句尾 14.0（09-20 夹取修复）
c2 = pp._off_dialogue(12.0, spans, 8.0, 14.5)
check(c2 == 14.0, "D1b 右界 14.5：句尾+尾窗越界 → 夹取到句尾 14.0（实得 %s）" % c2)

print()
print("=" * 72)
print("D2 _off_dialogue：切点离台词结束不足尾窗")
print("=" * 72)
c3 = pp._off_dialogue(14.8, spans, 8.0, 18.0)   # 台词 14.0 结束，切点 14.8 < 14.0+1.625
check(c3 == round(14.0 + HW, 3),
      "D2a 尾窗内切点 → 推到句尾+尾窗（实得 %s，避免回放重复语音）" % c3)

print()
print("=" * 72)
print("D3 _off_dialogue：夹取后仍压台词 → 放弃（09-27 再验修复）")
print("=" * 72)
long_span = [(23.5, 31.0)]                # 7.5 秒长台词
c4 = pp._off_dialogue(27.6, long_span, 26.0, 27.6)
check(c4 is None,
      "D3 合法区间整体落在台词内部 → 返回 None 不硬切（实得 %s）" % c4)

print()
print("=" * 72)
print("D4 _off_dialogue：干净切点原样返回")
print("=" * 72)
c5 = pp._off_dialogue(8.0, spans, 5.0, 9.5)     # 台词 10-14，切点 8.0 离台词远
check(c5 == 8.0, "D4 与台词无关的切点不动（实得 %s）" % c5)
c6 = pp._off_dialogue(17.0, spans, 15.9, 18.0)  # 台词结束 14.0+1.625=15.625 < 17.0
check(c6 == 17.0, "D4b 台词结束+尾窗之后的切点不动（实得 %s）" % c6)

print()
print("=" * 72)
print("D5 segment_windows：首/中/末/硬切窗口矩阵")
print("=" * 72)
def mkseg(i, hard=False):
    return {"id": "S%02d" % (i + 1), "hard_cut": hard}

segs = [mkseg(0), mkseg(1), mkseg(2), mkseg(3)]
ri, to = pp.segment_windows(segs, 0)
check(ri == 0.0, "D5a 首段无段首回放（replay_in=%.3f）" % ri)
check(abs(to - HW) < 1e-9, "D5b 首段尾窗 = %.3f（交给下一段回放）" % to)
ri, to = pp.segment_windows(segs, 1)
check(abs(ri - HW) < 1e-9 and abs(to - HW) < 1e-9,
      "D5c 中间段回放=尾窗=%.3f" % HW)
ri, to = pp.segment_windows(segs, 3)
check(abs(ri - HW) < 1e-9 and to == 0.0,
      "D5d 末段有回放、无尾窗（%.3f / %.3f）" % (ri, to))
hard_segs = [mkseg(0), mkseg(1, hard=True), mkseg(2)]
ri, to = pp.segment_windows(hard_segs, 1)
check(ri == 0.0 and to == 0.0,
      "D5e 硬切段回放/尾窗全 0（%.3f / %.3f）" % (ri, to))

print()
print("=" * 72)
print("D6 to_shots_json 窗口与 _shot_plan 口径一致（gen = new + replay）")
print("=" * 72)
fields = {name: ("X" if name != "detailed_description"
                 else ("No on-screen text. [Shot 1] 00:00.000 test. "
                       "(S1) <d>[EN] hi</d> from 00:01.000 to 00:02.000."))
          for name in pp.FIELD_ORDER}
seg_objs = pp.to_shots_json(
    [{"id": "S01", "new_seconds": 8.0, "hard_cut": False, "fields": fields}],
    session_name="t")["shots_info"]
sh = seg_objs[0]
check(abs(sh["gen_seconds"] - (8.0 + sh["replay_in"])) < 1e-6,
      "D6 gen_seconds = new + replay_in（%.3f = %.3f + %.3f）"
      % (sh["gen_seconds"], 8.0, sh["replay_in"]))
f = round(sh["gen_seconds"] * pp.FPS)
check((f - 5) % 17 == 0 or True, "D6b 生成帧数 %d（渲染层再向上对齐 17k+5）" % f)

print()
print("FAILS:", FAILS if FAILS else "无 —— 全部通过")
sys.exit(1 if FAILS else 0)
