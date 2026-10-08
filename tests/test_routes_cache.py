# -*- coding: utf-8 -*-
"""routes 缓存层回归 —— 2026-10-08 性能优化。

背景
----
面板不是「点一次解析一次」，而是一直在问：PACK 文本框每敲一下发一次
/h3/pack_preview、时间线每段完成拉一次 /h3/session、参考图每次重绘拉一次
/h3/input_images。这些端点以前每次都全量重做（两遍 PACK 解析、一次目录树
walk、每段一个 ffprobe 子进程）。于是加了一层缓存。

缓存一旦加错，失败模式全是**静默的**：返回上一份结果 → 面板显示旧数据，
用户以为渲染参数没生效。所以本文件把每一条边界都钉死：

  1) pack 缓存：同文本命中 / 改文本必失效 / 会话名参与键 / 返回值是副本
  2) ffprobe 缓存：同 mtime+size 命中 / 文件重写必失效 / 失败也缓存
  3) panel 元数据：同 mtime 命中 / 重写必失效
  4) LRU 容量有界，且真的会淘汰
  5) _find_session 不再重复调用 _session_dirs（用计数器证明）
  6) 语言守卫仍然拦住「往不存在的版块写回」，且两个端点口径一致

跑法: python tests/test_routes_cache.py   （全绿则退出码 0）
"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, "marquee_director")

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


# ---------------------------------------------------------------------------
# 加载 routes.py：它 import aiohttp + common(要 torch)，离线跑不起，
# 所以把两个重依赖替换成最小桩，其余代码原样执行 —— 被测逻辑是缓存本身。
# ---------------------------------------------------------------------------
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

pp_loader = importlib.machinery.SourceFileLoader(
    "md_pp", os.path.join(PKG, "prompt_pack.py"))
pp_spec = importlib.util.spec_from_loader("md_pp", pp_loader)
pp = importlib.util.module_from_spec(pp_spec)
pp_loader.exec_module(pp)
sys.modules["md_pp"] = pp

rd_loader = importlib.machinery.SourceFileLoader(
    "marquee_director.routes", os.path.join(PKG, "routes.py"))
rd_spec = importlib.util.spec_from_loader("marquee_director.routes", rd_loader)
routes = importlib.util.module_from_spec(rd_spec)
# routes 用相对 import（from . import prompt_pack），必须让它认得自己的包：
# 声明 __package__，并把兄弟模块注册进 sys.modules。
routes.__package__ = "marquee_director"
_pkg = type(sys)("marquee_director")
_pkg.__path__ = [PKG]
_pkg.prompt_pack = pp
_pkg.common = _FakeCommon()
sys.modules["marquee_director"] = _pkg
sys.modules["marquee_director.prompt_pack"] = pp
sys.modules["marquee_director.common"] = _FakeCommon()
rd_loader.exec_module(routes)

# 一份最小可解析的 PACK（英文单版、六字段、分段形态）
PACK = (
    "===== SHOT PROMPT PACK =====\n"
    "Project : bench_project\nMode : narrative\nTotal duration : 16s\n"
    "Segments : 2\nAspect : 16:9\nMusic : N/A\n\n"
    "ENGLISH VERSION\n\n"
    "########## S01 / 8s / EN ##########\n"
    "subject_definitions:\n<Subject 1> Kate <Picture 1>, amber hair, blue coat.\n"
    "summary:\nA woman walks into a lit room and stops.\n"
    "retention_analysis:\nThe reveal of the empty chair lands in second one.\n"
    "detailed_description:\nA medium shot, camera pushes in slowly, no new "
    "spoken word in the last 1.6 seconds.\n"
    "overall_soundscape:\nRoom tone, distant traffic.\n"
    "non_diegetic_music:\nN/A\n\n"
    "########## S02 / 8s / EN ##########\n"
    "subject_definitions:\n<Subject 1> Kate <Picture 1>, amber hair, blue coat.\n"
    "summary:\nShe sits down and speaks.\n"
    "retention_analysis:\nThe line lands hard.\n"
    "detailed_description:\nA close shot, static camera, she says <d>Finally."
    "</d> then no new dialogue in the last 1.6 seconds.\n"
    "overall_soundscape:\nRoom tone.\n"
    "non_diegetic_music:\nN/A\n\n"
    "END OF PACK\n"
)


# ===========================================================================
# 1) pack 缓存
# ===========================================================================
routes._preview_cache.clear()

base = routes._cached_preview(PACK, "sess_a")
check(base.get("ok") is True, "首次 _cached_preview 应返回 ok=True")
check(len(base.get("segments") or []) == 2,
      "应解析出 2 段，实得 %d" % len(base.get("segments") or []))

# 同一请求再打一次 —— 计时上应该几乎为 0（纯查表）
t0 = time.perf_counter()
again = routes._cached_preview(PACK, "sess_a")
hit_ms = (time.perf_counter() - t0) * 1000
check(again == base, "同文本同会话名的第二次调用结果必须与第一次一致")
check(hit_ms < 5.0,
      "缓存命中应在 5ms 内，实得 %.2fms" % hit_ms)

# 会话名参与键：同名不同会话不能互相污染
other = routes._cached_preview(PACK, "sess_b")
check(other is not base, "不同 session_name 必须算出独立结果")

# 文本改动必失效
edited = PACK.replace("A medium shot", "A wide shot")
fresh = routes._cached_preview(edited, "sess_a")
check(fresh != base, "文本改动后必须重新解析，不能返回旧结果")
detail = ((fresh.get("segments") or [{}])[0].get("full_fields") or {})
check("wide shot" in str(detail.get("detailed_description")),
      "改动后的正文应出现在新结果里")

# ★ 返回的是副本：调用方就地修改不得污染缓存
mutated = routes._cached_preview(PACK, "sess_a")
mutated["segments"].append({"id": "BOGUS"})
mutated["issues"].append("BOGUS")
after = routes._cached_preview(PACK, "sess_a")
check(len(after["segments"]) == 2,
      "调用方改坏返回值后，后续命中仍应是 2 段（实得 %d）"
      % len(after["segments"]))
check("BOGUS" not in after["issues"],
      "调用方对 issues 的修改泄漏进了缓存")

# 空 session_name 与 None 必须是不同的键（不能被 _content_key 折叠成一个）
check(routes._content_key("x", None) != routes._content_key("x", ""),
      "_content_key 必须区分 None 与空串")


# ===========================================================================
# 2) LRU 有界
# ===========================================================================
routes._preview_cache.clear()
for i in range(routes._PREVIEW_CACHE_MAX + 12):
    routes._cached_preview(PACK + "\n<!-- %d -->" % i, "s%d" % i)
check(len(routes._preview_cache) <= routes._PREVIEW_CACHE_MAX,
      "缓存条目数应受 _PREVIEW_CACHE_MAX 约束，实得 %d"
      % len(routes._preview_cache))


# ===========================================================================
# 3) ffprobe 缓存：按 (mtime, size) 失效
# ===========================================================================
routes._ffprobe_cache.clear()

tmp_dir = os.path.join(HERE, "_tmp_cache_probe")
os.makedirs(tmp_dir, exist_ok=True)
probe_file = os.path.join(tmp_dir, "seg_01.mp4")

calls = {"n": 0}
_real_run = None


def _fake_run(cmd, **kw):
    calls["n"] += 1
    class R:
        stdout = "12.5"
    return R()


try:
    import subprocess as _sp
    _real_run = _sp.run
    _sp.run = _fake_run

    with open(probe_file, "w", encoding="utf-8") as fh:
        fh.write("v1")

    v1 = routes._ffprobe_seconds(probe_file)
    check(v1 == 12.5, "ffprobe 应返回 12.5，实得 %r" % (v1,))
    check(calls["n"] == 1, "首次应真的起一次子进程")

    v2 = routes._ffprobe_seconds(probe_file)
    check(v2 == 12.5, "命中缓存后返回值不变")
    check(calls["n"] == 1,
          "mtime/size 未变时不应再起子进程（已起 %d 次）" % calls["n"])

    # 重写文件 → mtime/size 变 → 必须重新探测
    time.sleep(0.02)
    with open(probe_file, "w", encoding="utf-8") as fh:
        fh.write("v2-longer-content")
    v3 = routes._ffprobe_seconds(probe_file)
    check(calls["n"] == 2,
          "文件重写后必须重新探测（子进程起了 %d 次）" % calls["n"])
    check(v3 == 12.5, "重新探测后返回值仍应正确")

    # 探测失败也要缓存：ffprobe 缺失时不该每次请求都 fork 再失败一次
    calls["n"] = 0

    def _boom(cmd, **kw):
        calls["n"] += 1
        raise OSError("ffprobe not found")

    _sp.run = _boom
    missing = os.path.join(tmp_dir, "seg_99.mp4")
    with open(missing, "w", encoding="utf-8") as fh:
        fh.write("x")
    r1 = routes._ffprobe_seconds(missing)
    r2 = routes._ffprobe_seconds(missing)
    check(r1 is None and r2 is None, "探测失败应返回 None")
    check(calls["n"] == 1,
          "失败结果同样应被缓存，不该反复 fork（起了 %d 次）" % calls["n"])
finally:
    import subprocess as _sp
    if _real_run is not None:
        _sp.run = _real_run
    for fn in ("seg_01.mp4", "seg_99.mp4"):
        try:
            os.remove(os.path.join(tmp_dir, fn))
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass


# ===========================================================================
# 4) panel 元数据缓存：按 mtime 失效
# ===========================================================================
routes._panel_meta_cache.clear()
sess_dir = os.path.join(HERE, "_tmp_cache_session")
os.makedirs(sess_dir, exist_ok=True)
panel_file = os.path.join(sess_dir, "panel.json")

PAYLOAD = {"shots": [
    {"prompt": "A woman walks into a lit room and stops for a beat.", "duration": 8},
    {"prompt": "She sits down and speaks.", "speakers": ["S1"], "duration": 8},
]}
with open(panel_file, "w", encoding="utf-8") as fh:
    json.dump(PAYLOAD, fh)

meta1 = routes._load_panel_meta(sess_dir)
check(len(meta1) == 2, "panel.json 应解析出 2 段，实得 %d" % len(meta1))
check(meta1[2]["speakers"] == ["S1"], "第 2 段的 speakers 应被读出")

meta2 = routes._load_panel_meta(sess_dir)
check(meta2 == meta1, "命中缓存时结果必须一致")
check(meta2 is not meta1, "返回值必须是副本，调用方改动不得污染缓存")
meta2[1]["prompt"] = "MUTATED"
meta2[2]["speakers"].append("S9")          # 浅拷贝漏掉的正是这一层
check(routes._load_panel_meta(sess_dir)[1]["prompt"] != "MUTATED",
      "调用方对缓存结果的修改泄漏了")
check("S9" not in routes._load_panel_meta(sess_dir)[2]["speakers"],
      "speakers 列表必须是深拷贝：调用方 append 泄漏进了缓存")
# ★ int 键：走 JSON 序列化后必须还能用整数下标取到（json 会把键变成字符串）
fresh_meta = routes._load_panel_meta(sess_dir)
check(isinstance(list(fresh_meta.keys())[0], int),
      "缓存往返后段号键必须仍是 int（json 会把它变成 str）")
check(fresh_meta.get(1) is not None,
      "panel.get(1) 必须能命中（'1' != 1，键类型错了摘要就全空）")

time.sleep(0.02)
PAYLOAD["shots"][0]["prompt"] = "Completely different first shot description."
with open(panel_file, "w", encoding="utf-8") as fh:
    json.dump(PAYLOAD, fh)
meta3 = routes._load_panel_meta(sess_dir)
check("Completely different" in meta3[1]["prompt"],
      "panel.json 重写后必须重新读取（mtime 变了就该失效）")

# 解析不出 rows 的文件也要缓存，且不抛
with open(panel_file, "w", encoding="utf-8") as fh:
    fh.write("{ not json")
check(routes._load_panel_meta(sess_dir) == {},
      "坏 JSON 应被吞掉并返回空，不得抛出")
check(routes._load_panel_meta(sess_dir) == {},
      "坏 JSON 的结果同样走缓存路径，不得抛")

try:
    os.remove(panel_file)
    os.rmdir(sess_dir)
except OSError:
    pass


# ===========================================================================
# 5) _find_session：_session_dirs 只该算一次
# ===========================================================================
calls = {"n": 0}
_real_dirs = routes._session_dirs
tmp_root = os.path.join(HERE, "_tmp_cache_roots", "h3_continuous")
os.makedirs(os.path.join(tmp_root, "my_session"), exist_ok=True)


def _counting_dirs():
    calls["n"] += 1
    return [tmp_root]


routes._session_dirs = _counting_dirs
try:
    calls["n"] = 0
    hit = routes._find_session("my_session")
    check(hit is not None and os.path.isdir(hit),
          "应按名字找到会话目录，实得 %r" % (hit,))
    check(calls["n"] == 1,
          "_session_dirs 应只调用 1 次（原来在候选循环里被调 2~3 次），"
          "实得 %d 次" % calls["n"])
finally:
    routes._session_dirs = _real_dirs
    try:
        os.rmdir(os.path.join(tmp_root, "my_session"))
        os.rmdir(tmp_root)
        os.rmdir(os.path.dirname(tmp_root))
    except OSError:
        pass


# ===========================================================================
# 6) 语言守卫：两个写回端点口径一致，且真的拦住
# ===========================================================================
# 英文单版 PACK 上要求写中文侧 → 必须拒（原文一字不改）
blocked_a = routes._pack_rebuild(PACK, [{"index": 1, "fields": {}}], "zh")
blocked_b = routes._pack_apply_block(PACK, [{"index": 1, "body": "x"}], "zh")
for label, res in (("rebuild", blocked_a), ("apply_block", blocked_b)):
    check(res.get("ok") is False,
          "%s: 英文单版 PACK 上 lang=zh 必须被拒" % label)
    check(res.get("unchanged") is True,
          "%s: 拒绝时必须标记 unchanged" % label)
    check(res.get("text") == PACK,
          "%s: 拒绝时原文必须一字不改" % label)
    check("中文" in str(res.get("warning")),
          "%s: 拒绝必须带可读的 warning" % label)

# 两侧口径必须一致（合成前是两份复制粘贴的代码，最容易分叉）
check(blocked_a["warning"] == blocked_b["warning"],
      "两个写回端点的拒绝文案不一致 —— 守卫已分叉")

# 命中语言侧 → 正常放行
ok_a = routes._pack_rebuild(PACK, [{"index": 1, "fields": {
    "detailed_description": "REPLACED BODY"}}], "en")
check(ok_a.get("ok") is True, "lang=en 在英文 PACK 上必须放行")
check("REPLACED BODY" in ok_a.get("text", ""),
      "放行时改动必须真的写进文本")

# lang=None 一律放行（全篇定位）
ok_none = routes._pack_rebuild(PACK, [{"index": 1, "fields": {
    "detailed_description": "NULL LANG BODY"}}], None)
check(ok_none.get("ok") is True, "lang=None 必须放行")


# ===========================================================================
if FAIL:
    print("FAIL (%d)" % len(FAIL))
    for f in FAIL:
        print("  - " + f)
    sys.exit(1)
print("routes 缓存层回归: 全部通过")