# -*- coding: utf-8 -*-
"""2026-10-14 回归：/h3/session 段数漂移探测（用户报「标尺与缩略图不匹配」的根因）。

症状实况：用户改了 PACK（切分点漂移，旧版 S04a/S04b → 新版 S07a/S07b），
磁盘 21 段、面板 20 段。seg_NN 的**渲染顺序号**与面板「本批次分镜序号」
不再同义，而 /h3/session 此前按磁盘号下发视频 URL、前端照单全收 ——
段块视频张冠李戴，doneSet 还多出幽灵段（第 21 段）让进度条虚高。

守四条：
  ① 不带 segs → drift=False、disk_count 照实返回（老调用方无感）；
  ② segs == 磁盘段数 → drift=False（正常路径）；
  ③ segs != 磁盘段数 → drift=True 且 disk_count / planned_segs 如实返回；
  ④ done 仍是「从 1 连续」的磁盘口径（断点信息不能被探测字段带偏）。

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

# --- fixture：一个带 5 个 seg mp4 的会话目录 ---------------------------
TMP = tempfile.mkdtemp(prefix="h3_drift_")
SESS = os.path.join(TMP, "drift_sess")
os.makedirs(SESS)
for i in range(1, 6):
    with open(os.path.join(SESS, "seg_%02d.mp4" % i), "wb") as fh:
        fh.write(b"\0" * 64)

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

# ③ 漂移（用户实况：磁盘 21 vs 面板 20）
r = routes._session_state("drift_sess", 20)
check(r.get("drift") is True, "③ segs != disk 时 drift=True")
check(r.get("disk_count") == 5 and r.get("planned_segs") == 20,
      "③ disk_count/planned_segs 应如实返回两边数字")
check(r.get("done") == 5, "③ done 仍是磁盘连续口径（不能被探测带偏）")
# 反向漂移（面板比磁盘多）同样要报
r = routes._session_state("drift_sess", 3)
check(r.get("drift") is True, "③ 面板少于磁盘也要报 drift")

# ④ 脏输入不炸
r = routes._session_state("drift_sess", "abc")
check(r.get("drift") is False, "④ 非数字 segs 退化为不探测")
check(r.get("planned_segs") is None, "④ 非数字 segs → planned_segs=None")

# ⑤ 会话不存在时仍带新字段的默认形状
r = routes._session_state("nope")
check(r.get("ok") is False and r.get("segments") == [], "⑤ 不存在会话保持原形状")

shutil.rmtree(TMP, ignore_errors=True)

if FAIL:
    print("FAIL:")
    for m in FAIL:
        print("  -", m)
    sys.exit(1)
print("全部通过：段数漂移探测就位（①老调用无感 ②一致不报 ③双向漂移都报 "
      "④脏输入容错 ⑤done 口径不变）。")
