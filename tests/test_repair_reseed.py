# -*- coding: utf-8 -*-
"""2026-09-30 回归：单段修复必须换种子（否则逐比特复现要修的瑕疵）

背景：段种子是 `_resolve_seed` 的纯函数 `chain_seed + (index+1)*9973`。
修复路径原本传 `seed_override=0` ⇒ 用同一个 chain_seed 再修一次，
**结果与上次逐比特相同**，包括它要修的那个瑕疵。
实测：v25sub 的 seg_04 在 37.75–38.25s 有人脸拖影花屏，
当时用户唯一出路是把 chain_seed 整个换掉 → 连累全部 5 段重新抽签。

跑法（必须用 ComfyUI 的 venv，director 依赖 comfy）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_repair_reseed.py
"""
import importlib.machinery, importlib.util, os, sys

# 本包在 <ComfyUI>/custom_nodes/ 下，所以上两级就是 ComfyUI 根目录。
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

from ComfyUI_Marquee_Director.marquee_director import director as D
from ComfyUI_Marquee_Director.marquee_director import nodes as N

FAIL = []
SETTINGS = {"seed": 307045}


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


# ① 跳变必须真的改变种子（每一段）
for i in range(5):
    old = N._resolve_seed(SETTINGS, {"seed": 0}, i)
    new = D._repair_reseed(SETTINGS, i)
    check(old != new, "seg%d 修复种子未跳变（%d）" % (i + 1, old))

# ② 跳变值必须被 _resolve_seed 采纳（否则 override 形同虚设）
for i in range(5):
    rs = D._repair_reseed(SETTINGS, i)
    check(N._resolve_seed(SETTINGS, {"seed": rs}, i) == rs,
          "seg%d 的 seed_override 未被 _resolve_seed 采纳" % (i + 1))

# ③ 必须与 subtitle_qc 同式（两处跳变不能各写一份，否则日后各自漂移）
_p = os.path.join(PKG, "marquee_director", "subtitle_qc.py")
_ld = importlib.machinery.SourceFileLoader("sq", _p)
_sp = importlib.util.spec_from_file_location("sq", _p, loader=_ld)
sq = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(sq)
_expected = (326991 * 1103515245 + 12345) % (1 << 63)   # v25sub 日志实测值
check(D._repair_reseed(SETTINGS, 1) == _expected,
      "_repair_reseed 与 subtitle_qc 的跳变式不一致（seg2 应为 %d）" % _expected)

# ④ 源码级：reseeded 段必须跳过种子对账，否则修复会在整链重跑时被重渲
_dsrc = open(os.path.join(PKG, "marquee_director", "director.py"), encoding="utf-8").read()
check('if not rec.get("reseeded") and rec.get("seed") != resolved_seed:' in _dsrc,
      "_seed_diff_reason 未跳过 reseeded 段的种子对账")

# ⑤ 源码级：修复记录必须写 reseeded 标记
_nsrc = open(os.path.join(PKG, "marquee_director", "nodes.py"), encoding="utf-8").read()
check('**({"reseeded": True} if seed_override else {}),' in _nsrc,
      "修复记录未写 reseeded 标记")

print("FAIL:", FAIL if FAIL else "无 —— 全部通过")
sys.exit(1 if FAIL else 0)
