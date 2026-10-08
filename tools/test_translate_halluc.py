# -*- coding: utf-8 -*-
"""Regression: a block with no dialogue must not be thrown away because the
model hallucinated [[Dn]] placeholders.

Run from the ComfyUI root::

    .venv/Scripts/python.exe custom_nodes/ComfyUI_Marquee_Director/tools/test_translate_halluc.py

Background (see the QC skill, "英译『无台词块被模型幻觉占位符判死』"):
the system prompt tells the model to put every ``[[Dn]]`` back where it found
it. On a block that has **no** dialogue at all, ``protect_dialogue()`` returns
an empty store, so the model — following that instruction literally — invents a
run of placeholders. ``validate()`` then reports "expected [], got [[D1]]…" and
the whole block falls back to the Chinese original. On a 96-block script 88
blocks have no dialogue, so this single check was silently reverting most of
the translation.

This test monkeypatches the model call, so it needs no GPU and no server.
"""

import os
import sys

FAILS = []


def check(label, ok, extra=""):
    print("%-62s %s %s" % (label, "PASS" if ok else "FAIL", extra))
    if not ok:
        FAILS.append(label)


def _comfy_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


COMFY = _comfy_root()
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))

import importlib                                                   # noqa: E402

# 必须按**包**导入：模块里有相对 import，spec_from_file_location 直接加载会
# 报 "attempted relative import with no known parent package"。
tn = importlib.import_module(
    "ComfyUI_Marquee_Director.marquee_director.translate_nodes")

CJK = tn._CJK_RE


def _run(body, fake_out):
    """把模型调用换成固定输出，跑一次节点。"""
    tn._clip_generate = lambda *a, **k: fake_out
    res = tn.H3ScriptTranslate.execute(
        clip=object(), text=body, enabled=True, model_source="local",
        chunk_chars=900, max_length=1024, temperature=0.3, seed=0)
    return res.result[0], res.result[1]


ZH_NO_DIALOG = "镜头缓慢推进，烛火摇曳，古卧房的夜色压得很沉。"

# ① 无台词 + 模型幻觉出一串占位符 -> 必须采纳英文，不能回退中文
out, rep = _run(ZH_NO_DIALOG,
                "The camera pushes in slowly, the candle flickers. [[D1]] [[D2]] "
                "The night in the ancient bedroom lies heavy.")
check("no-dialogue block: English adopted", not CJK.search(out), out[:60])
check("no-dialogue block: hallucinated placeholders stripped",
      "[[D" not in out, out[:60])
# ★ 2026-09-27 报告口径改了：不再写「失败保留原文」（**永远不保留中文**），
#   改成「未译出（已置空）」+ 一条硬承诺「交付闸门：台词块之外残留中文 = N」。
check("no-dialogue block: reported as 0 not-translated",
      "未译出（已置空）0" in rep, rep[:90])
check("no-dialogue block: delivery gate reports 0 Chinese",
      "残留中文 = 0" in rep, rep[:90])

# ② 回归：有台词块，占位符必须**保留**（不能被 ① 的剥除逻辑波及）
ZH_WITH_D = ZH_NO_DIALOG + "\n台词：<d>[Chinese] 汝与曹贼何异？</d>"
out2, rep2 = _run(ZH_WITH_D,
                  "The camera pushes in slowly. [[D1]] The night lies heavy.")
check("dialogue block: dialogue restored verbatim",
      "<d>[Chinese] 汝与曹贼何异？</d>" in out2, out2[:80])
check("dialogue block: no placeholder left over", "[[D" not in out2, out2[:80])

# ③ 回归：无台词且输出干净 -> 正常走原分支
out3, rep3 = _run(ZH_NO_DIALOG, "The camera pushes in slowly, the candle flickers.")
check("clean output still translates", not CJK.search(out3), out3[:60])

# ④ 无台词、译文自己也是纯中文 -> **绝不放行中文**（旧行为是"整块保留原文"，
#    那等于把 201 个汉字送进 H3 提示词，模型会当台词念出来）。
#    新契约：剥离/置空，并如实报「未译出」+ 交付闸门 0。
out4, rep4 = _run(ZH_NO_DIALOG, "镜头缓慢推进。[[D1]] 烛火摇曳。")
check("still-Chinese output never leaks", not CJK.search(out4), out4[:60])
check("still-Chinese output reported as not translated",
      "未译出（已置空）1" in rep4, rep4[:90])
check("still-Chinese output: delivery gate reports 0 Chinese",
      "残留中文 = 0" in rep4, rep4[:90])

if FAILS:
    print("\n%d FAILED: %s" % (len(FAILS), "; ".join(FAILS)))
    raise SystemExit(1)
print("\nall translate-hallucination checks passed")
