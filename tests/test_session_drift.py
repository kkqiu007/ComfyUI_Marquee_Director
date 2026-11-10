# -*- coding: utf-8 -*-
"""2026-10-14 回归：/h3/session 段数漂移探测（用户报「标尺与缩略图不匹配」的根因）。

症状实况：用户改了 PACK（切分点漂移，旧版 S04a/S04b → 新版 S07a/S07b），
磁盘 21 段、面板 20 段。seg_NN 的**渲染顺序号**与面板「本批次分镜序号」
不再同义，而 /h3/session 此前按磁盘号下发视频 URL、前端照单全收 ——
段块视频张冠李戴，doneSet 还多出幽灵段（第 21 段）让进度条虚高。

2026-10-18 追加（用户实况：首渲假阳性）：会话刚开始渲、一次刷新落在第一段
落盘前 → 磁盘 0 段 vs 面板 20 段，横幅喊「PACK 改过？缩略图错位」——可磁盘
上什么都没有，无从错位。修法：drift 判据要求**两边都非空**；渲染途中
1..N vs planned 的前端抑制见 h3_director.js absorbSession 的 _running 门。

2026-10-23 追加（用户实况：中断假阳性）：一次 9 段的渲染在第 5 段被取消，
磁盘留下连续的 1..4，面板计划 9 段 → 旧判据 `planned != disk_count` 立刻喊
「PACK 改过？缩略图与段号可能错位」。可这份 PACK 一个字都没改，真相只是
"没渲完"。前端的 _running 门这时恰恰不生效（已经停了），所以必须在判据这一层
就把两种情况分开：**连续前缀且不超过计划 = 没渲完**；**磁盘超过计划，或段号
缺号（done < disk_count）= 真漂移**。

守八条：
  ① 不带 segs → drift=False、disk_count 照实返回（老调用方无感）；
  ② segs == 磁盘段数 → drift=False（正常路径）；
  ③ 磁盘段数 > 面板计划 → drift=True（旧版多切了几段，段号从此错开）；
  ③' 磁盘是 1..N 连续前缀、N < 面板计划 → drift=False（渲到一半停了）；
  ③'' 磁盘段号缺号（done < disk_count）→ drift=True（段号已对不上）；
  ④ done 仍是「从 1 连续」的磁盘口径（断点信息不能被探测字段带偏）；
  ⑤ 磁盘 0 段（首渲未出片）+ 面板 20 → drift=False（无可错位对象）；
  ⑥ 磁盘 5 段 + 面板 0（PACK 未解析）→ drift=False（无比对对象）；
  ⑦ 脏 segs / 会话不存在 → 原形状不炸。

跑法: python tests/test_session_drift.py
"""
import importlib.machinery
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, "marquee_director")
COMFY_ROOT = os.path.dirname(os.path.dirname(ROOT))
for _p in (COMFY_ROOT, ROOT, PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


# routes.py import aiohttp + common(要 torch)：用最小桩替换，被测逻辑原样跑。
class _FakeWeb:
    def json_response(self, *a, **k):
        return None

    def Response(self, *a, **k):
        return None


class _FakeAiohttp:
    web = _FakeWeb()

    def __init__(self, *a, **k):
        pass


class _FakeCommon:
    @staticmethod
    def log(msg, *args):
        pass

    @staticmethod
    def safe_session_name(*cands):
        for raw in cands:
            text = str(raw or "").strip()
            if text:
                return text[:60]
        return "h3_pack"


sys.modules.setdefault("aiohttp", _FakeAiohttp())

rd_loader = importlib.machinery.SourceFileLoader(
    "marquee_director.routes", os.path.join(PKG, "routes.py"))
rd_spec = importlib.util.spec_from_loader("marquee_director.routes", rd_loader)
routes = importlib.util.module_from_spec(rd_spec)
routes.__package__ = "marquee_director"
_pkg = type(sys)("marquee_director")
_pkg.__path__ = [PKG]
sys.modules["marquee_director"] = _pkg
sys.modules["marquee_director.common"] = _FakeCommon()
rd_loader.exec_module(routes)

# --- fixture：一个带 5 个 seg mp4 的会话目录 + 一个空目录（首渲未出片）--------
TMP = tempfile.mkdtemp(prefix="h3_drift_")
SESS = os.path.join(TMP, "drift_sess")
os.makedirs(SESS)
for i in range(1, 6):
    with open(os.path.join(SESS, "seg_%02d.mp4" % i), "wb") as fh:
        fh.write(b"\0" * 64)
# 空会话目录：只有 manifest 没有 seg_*.mp4 —— 2026-10-18 用户首渲实况
EMPTY = os.path.join(TMP, "drift_empty")
os.makedirs(EMPTY)

_real_root = routes._session_root if hasattr(routes, "_session_root") else None


def _fake_dirs():
    return [TMP]


routes._session_dirs = _fake_dirs
# 跨盘 relpath 会让 _view_subfolder 抛 ValueError（fixture 在 C:、真实 output
# 在 E:）；这个函数与漂移探测无关，替换成常量。
routes._view_subfolder = lambda path: "drift_sess"

# ① 老调用方：不带 planned_segs
r = routes._session_state("drift_sess")
check(r.get("ok") is True, "① 会话应找得到")
check(r.get("disk_count") == 5, "① disk_count 应为 5，实际 %s" % r.get("disk_count"))
check(r.get("drift") is False, "① 不带 segs 时 drift 必须为 False（老调用方无感）")
check(r.get("planned_segs") is None, "① planned_segs 应为 None")
check(r.get("done") == 5, "① done 按连续段数 = 5，实际 %s" % r.get("done"))
check(len(r.get("segments") or []) == 5, "① 仍应返回 5 条段记录")

# ② 一致
r = routes._session_state("drift_sess", 5)
check(r.get("drift") is False, "② segs==disk 时 drift=False")
check(r.get("planned_segs") == 5, "② planned_segs 回显")

# ③ 磁盘段数**超过**面板计划 = 真漂移（用户实况：磁盘 21 vs 面板 20）
r = routes._session_state("drift_sess", 3)
check(r.get("drift") is True, "③ 磁盘 5 段 > 面板 3 段时要报 drift")
check(r.get("disk_count") == 5 and r.get("planned_segs") == 3,
      "③ disk_count/planned_segs 应如实返回两边数字")
check(r.get("done") == 5, "③ done 仍是磁盘连续口径（不能被探测带偏）")

# ③' 磁盘是 1..N 连续前缀、N < 面板计划 = **渲到一半停了**，不是漂移。
#    ★ 2026-10-23 改的判据。旧判据是 `planned != disk_count` 一律报漂移，于是
#    一次 9 段的渲染在第 5 段被取消后，面板对着一份**没改过**的 PACK 喊
#    「磁盘 4 段 ≠ 当前分镜 9 段（PACK 改过？）已渲缩略图与段号可能错位」——
#    把"没渲完"说成"数据错了"，用户会去重跑一个本来只要断点续渲的会话。
#    前端 absorbSession 只在 _running 时抑制这条横幅，而中断之后恰恰不在
#    running，所以抑制不住，必须在判据这一层就分开。
r = routes._session_state("drift_sess", 20)
check(r.get("drift") is False,
      "③' 磁盘 1..5 连续、面板 20 段 = 没渲完/被中断，不许报漂移（实得 %s）"
      % r.get("drift"))
check(r.get("disk_count") == 5 and r.get("planned_segs") == 20,
      "③' 两边数字仍要如实返回（横幅不亮，但 console 与排查要用）")
check(r.get("done") == 5, "③' done 仍是 5，断点续渲从第 6 段接着跑")

# ③'' 磁盘段号缺号（done < disk_count）= 段号已经对不上，仍是真漂移
_gap = os.path.join(SESS, "seg_07.mp4")
open(_gap, "w").close()
try:
    r = routes._session_state("drift_sess", 20)
    check(r.get("done") == 5 and r.get("disk_count") == 6,
          "③'' 缺号会话应读到 done=5 / disk=6，实得 %s/%s"
          % (r.get("done"), r.get("disk_count")))
    check(r.get("drift") is True, "③'' 段号不连续要报 drift")
finally:
    os.remove(_gap)

# ④ 脏输入不炸
r = routes._session_state("drift_sess", "abc")
check(r.get("drift") is False, "④ 非数字 segs 退化为不探测")
check(r.get("planned_segs") is None, "④ 非数字 segs → planned_segs=None")

# ⑤ 空目录 + 面板 20 段：首渲未出片，磁盘 0 段 vs 面板 20 → 不判漂移
#    （2026-10-18 用户实况：刷新落在第一段落盘前，横幅误喊「PACK 改过」）
r = routes._session_state("drift_empty", 20)
check(r.get("ok") is True, "⑤ 空会话应找得到")
check(r.get("disk_count") == 0, "⑤ disk_count 应为 0")
check(r.get("drift") is False,
      "⑤ 磁盘 0 段时 drift 必须为 False（首渲假阳性，无可错位对象）")
check(r.get("segments") == [], "⑤ 空目录无段记录")
check(r.get("done") == 0, "⑤ done=0")

# ⑥ 面板 0 段（PACK 未解析）+ 磁盘 5 段：无比对对象，不判漂移
r = routes._session_state("drift_sess", 0)
check(r.get("drift") is False,
      "⑥ 面板 0 段时 drift=False（PACK 未解析，段数无比对意义）")
check(r.get("planned_segs") == 0, "⑥ planned_segs 回显 0")

# ⑦ 会话不存在时仍带新字段的默认形状
r = routes._session_state("nope")
check(r.get("ok") is False and r.get("segments") == [], "⑦ 不存在会话保持原形状")

shutil.rmtree(TMP, ignore_errors=True)

if FAIL:
    print("FAIL:")
    for m in FAIL:
        print("  -", m)
    sys.exit(1)
print("全部通过：段数漂移探测就位（①老调用无感 ②一致不报 ③双向漂移都报 "
      "④脏输入容错 ⑤磁盘 0 段不误报 ⑥面板 0 段不误报 ⑦done 口径不变）。")
