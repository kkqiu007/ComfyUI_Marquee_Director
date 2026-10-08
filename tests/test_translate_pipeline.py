# -*- coding: utf-8 -*-
"""H3ScriptTranslate 文本管线离线回归（B 层深挖，2026-09-30 四层排查产物）。

不碰网络/API，只测**纯文本机器**：分块守恒、台词/结构行保护还原 round-trip、
自查闸门（validate）、事实项丢失检测（lost_facts）、残留中文判据。这些是
「中文稿换任何脚本都要能走通英译」的基础保证 —— API 质量问题不可测，
管线自身的丢数据必须在这里拦住。

跑法：  python tests/test_translate_pipeline.py   （全绿退出码 0）
"""
import importlib.machinery
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "marquee_director")

_stubs = {}
for name in ("comfy_api", "comfy_api.latest"):
    m = __import__("types").ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
_io = __import__("types").ModuleType("comfy_api.latest.io")


class _Any:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, item):
        return _Any


class _IO:
    def __getattr__(self, item):
        return _Any


_io.ComfyNode = type("ComfyNode", (), {})
_io.Schema = type("Schema", (), {})
_io.NodeOutput = type("NodeOutput", (), {"__init__": lambda s, *a: None})
_io.Clip = _io.String = _io.Boolean = _io.Combo = _io.Int = _io.Float = _IO()
sys.modules["comfy_api.latest.io"] = _io

# 包上下文（translate_nodes 用相对导入 `from . import prompt_pack`）
_pkg = __import__("types").ModuleType("marquee_director")
_pkg.__path__ = [os.path.dirname(PKG)]
sys.modules["marquee_director"] = _pkg
_c = __import__("types").ModuleType("marquee_director.common")
_c.log = lambda *a, **k: None
sys.modules["marquee_director.common"] = _c
_pkg.common = _c


def _load(path, name):
    ld = importlib.machinery.SourceFileLoader(name, path)
    sp = importlib.util.spec_from_loader(name, ld)
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    ld.exec_module(mod)
    return mod


pp = _load(PKG + os.sep + "prompt_pack.py", "marquee_director.prompt_pack")
tn = _load(PKG + os.sep + "translate_nodes.py", "marquee_director.translate_nodes")

FAILS = []


def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        FAILS.append(msg)


ZH_PACK = """==========================================================
MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包
Project / 项目         : 管线压测稿
Mode / 模式            : Ref2VA (full reference)
########## S01 / 8s / EN ##########
subject_definitions:
<Subject 1> 是 <Picture 1> 里的男人，布衣束发，眉目沉静。
<Subject 2> 是 <Picture 2> 里的女子，素裙挽髻，神色清冷。

summary:
[reference generation] 古卧房夜戏，烛光下两人对峙。

retention_analysis:
<Subject 1>（出现于 [Shot 1]）：fully_preserved，身份与画左位置不变。

detailed_description:
No on-screen text, subtitles, captions, timecodes, or any graphics overlays anywhere in the frame.
[Shot 1] 00:00.000 他握着信纸的手微微收紧，烛焰在风里晃了半拍，她的影子投在窗纸上。(S1) <d>信呢？</d> from 00:02.000 to 00:04.000.
[Shot 2] At 00:05.000，她转身，目光落在他脸上，一字一顿。(S2) <d>烧了。</d> from 00:05.000 to 00:06.200.

overall_soundscape:
烛芯噼啪，远处更鼓。

non_diegetic_music:
全片配乐：低音域京剧打击与弦乐垫底，音量克制。
"""

print("=" * 72)
print("T1 台词保护/还原 round-trip 逐字守恒")
print("=" * 72)
body = ZH_PACK.split("########## S01 / 8s / EN ##########", 1)[1]
prot, dstore = tn.protect_dialogue(body)
check(len(dstore) == 2, "T1a 两句台词入保护 store（实得 %d）" % len(dstore))
check("[[D1]]" in prot and "[[D2]]" in prot and "<d>" not in prot,
      "T1b 台词块全部替换为占位符")
back, miss_d = tn.restore_dialogue(prot, dstore)
norm1 = tn._norm_dialogue("<d>信呢？</d>")
norm2 = tn._norm_dialogue("<d>烧了。</d>")
expect = body.replace("<d>信呢？</d>", norm1).replace("<d>烧了。</d>", norm2)
check(back == expect and not miss_d, "T1c protect→restore = 原文 + 台词规范化（设计行为：补 [Chinese]）")

print()
print("=" * 72)
print("T2 结构行保护/还原 round-trip（双语气泡不送模型）")
print("=" * 72)
prot2, sstore = tn.protect_structure(ZH_PACK)
check(len(sstore) >= 3, "T2a 头部结构行入保护（实得 %d）" % len(sstore))
back2, missing = tn.restore_structure(prot2, sstore)
check(back2 == ZH_PACK and not missing, "T2b protect→restore 逐字等于原文、无丢失")

print()
print("=" * 72)
print("T3 mock 英译全管线：validate 通过 + lost_facts 空 + 还原台词逐字")
print("=" * 72)
# 模拟「守规矩的模型」：把中文句换成英文，但**保留**全部时间码 / <Picture N> /
# <Subject N> / 占位符 / 英文既有句 —— 这是自查闸门要求的最小契约。
def mock_translate(text):
    out_lines = []
    for ln in text.split("\n"):
        if tn._needs_translation(ln):
            keep = (re.findall(r"\[Shot \d+\]", ln)
                    + re.findall(r"\d{2}:\d{2}(?:\.\d{1,3})?", ln)
                    + re.findall(r"<(?:Picture|Subject|Video|Audio) \d+>", ln)
                    + re.findall(r"\[\[D\d+\]\]", ln))
            eng = ("The candle flickers and their shadows cross the paper "
                   "window in the ancient bedroom.")
            out_lines.append(" ".join([eng] + keep))
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)

prot3, dstore3 = tn.protect_dialogue(ZH_PACK)
prot3, sstore3 = tn.protect_structure(prot3)
visible = len(re.findall(r"\[\[D\d+\]\]", prot3))
ok, reason = tn.validate(prot3, mock_translate(prot3), visible)
check(ok, "T3a 自查闸门通过（reason=%s）" % reason)
translated = mock_translate(prot3)
translated, miss_s = tn.restore_structure(translated, sstore3)
restored, miss_d3 = tn.restore_dialogue(translated, dstore3)
check("<d>[Chinese] 信呢？</d>" in restored and "<d>[Chinese] 烧了。</d>" in restored,
      "T3b 台词逐字还原（_norm_dialogue 规范化：补 [Chinese]）")
check(tn.lost_facts(prot3, restored) == [], "T3c lost_facts 空（时间码/标签全保住）")

print()
print("=" * 72)
print("T4 自查闸门的四类拒绝")
print("=" * 72)
ok1, r1 = tn.validate(prot3, "", visible)
check(not ok1 and "为空" in r1, "T4a 空输出拒绝（%s）" % r1)
bad = mock_translate(prot3).replace("[[D1]]", "")
ok2, r2 = tn.validate(prot3, bad, visible)
check(not ok2 and "数量不符" in r2, "T4b 丢台词占位符拒绝（%s）" % r2)
ok3, r3 = tn.validate(prot3, "[[D1]] x [[D2]]", visible)
check(not ok3 and "过短" in r3, "T4c 输出过短拒绝（%s）" % r3)
leak = mock_translate(prot3).replace("[[D1]]", "literal [[Dn]] token [[D1]]")
ok4, r4 = tn.validate(prot3, leak, visible)
check(not ok4 and "泄漏" in r4, "T4d 未编号占位符泄漏拒绝（%s）" % r4)

print()
print("=" * 72)
print("T5 lost_facts 逮住真实丢失")
print("=" * 75)
no_clock = re.sub(r"\d{2}:\d{2}(?:\.\d{1,3})?", "", mock_translate(prot3))
lost = tn.lost_facts(prot3, tn.restore_dialogue(no_clock, dstore3)[0])
check(any("时间码" in l for l in lost), "T5a 时间码丢失被检出（%s）" % lost[:1])
no_pic = mock_translate(prot3).replace("<Picture 1>", "").replace("<Picture 2>", "")
lost2 = tn.lost_facts(prot3, tn.restore_dialogue(no_pic, dstore3)[0])
check(any("Picture" in l or "丢失" in l for l in lost2),
      "T5b <Picture N> 丢失被检出（%s）" % lost2[:1])

print()
print("=" * 72)
print("T6 分块守恒（split_parts / split_oversized）")
print("=" * 75)
parts = tn.split_parts(ZH_PACK)
check("".join(b + s for b, s in parts) == ZH_PACK, "T6a split_parts 拼回 == 原文")
big = ZH_PACK.replace(
    "他握着信纸的手微微收紧，烛焰在风里晃了半拍，她的影子投在窗纸上。",
    "他握着信纸的手微微收紧，烛焰在风里晃了半拍，她的影子投在窗纸上，" * 12)
pieces = tn.split_oversized(big, 300)
check("".join(p for p, _ in pieces) == big, "T6b split_oversized 拼回 == 原文")
check(all(len(p) <= 320 for p, _ in pieces), "T6c 每块 ≤ limit+容差")

print()
print("=" * 72)
print("T7 全角标点/台词规范化")
print("=" * 75)
np = tn.normalize_punct("他停住了，然后离开。<d>信呢？</d>好的，就这样。")
check("，" not in np.replace("<d>信呢？</d>", "") and "他停住了, 然后离开." in np,
      "T7a 块外全角→ASCII，台词块逐字不动")
nd = tn._norm_dialogue("「你到底想怎样？」")
check(nd == "<d>[Chinese] 你到底想怎样？</d>", "T7b 引号对白规范化 + 补 [Chinese]")
nd2 = tn._norm_dialogue("<d>[EN] line</d>")
check(nd2 == "<d>[EN] line</d>", "T7c 已有语言标签原样保留")

print()
print("FAILS:", FAILS if FAILS else "无 —— 全部通过")
sys.exit(1 if FAILS else 0)
