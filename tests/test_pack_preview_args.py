# -*- coding: utf-8 -*-
"""面板后端口径回归 —— 2026-10-23「解析 PACK 不自动联动 / 标尺时间对不上」。

背景
----
用户报的两句现象，根因都在后端这两处，且失败模式全是**静默的**：

  1) ``/h3/pack_preview`` 只读 text + session_name，画布上
     H3PromptPackParser 的 ``default_duration`` / ``language`` / ``auto_fix`` /
     ``pack_override`` 一个都不传给 ``parse_pack`` —— 而节点 ``execute`` 是按
     控件值解析的。实测同一份 PACK 四种入参返回**逐字相同**的结果：面板永远
     按「缺省段长 11 / 自动修正开 / 英文侧 / 无视 override」画时间线，节点却
     按控件值渲染。段标题没写时长的稿子上，改缺省段长后标尺秒数纹丝不动。

  2) ``_find_session`` 的放宽兜底对根下**每一个**剧本夹都调一次
     ``_nested_session(match_parent=True)``，而那个调用压根不看候选名 ——
     夹子底下只要有版本目录就返回最新的一版。于是任何找不到的名字都落到
     "磁盘上最新的那个会话"：实测 name=''、'zzz_nonexistent_xyz'、
     '../../Windows' 全部解析成 EP02v3。时间线于是显示**别的剧本**的缩略图与
     进度，甚至对一份从没渲过的 PACK 报「✓ 完成」。

  3) 缓存键必须把解析参数带上，否则"改了控件"这一维度上的命中会返回旧结果。

  4) 段数漂移判据把「磁盘 1..N 连续、N < 计划段数」也算成漂移 —— 那是渲到
     一半被取消，不是 PACK 改过。实测一次 9 段的渲染在第 5 段中断后，面板对着
     一份没改过的 PACK 喊「磁盘 4 段 ≠ 当前分镜 9 段（PACK 改过？）」。

  5) ``_view_subfolder`` 在 Windows 上跨盘符会抛 ValueError（relpath 的限制），
     而它在 _session_state 的每段循环里被调用 —— 一抛整个 /h3/session 就 500，
     面板一段已渲都拿不到。_session_dirs() 本来就可能返回另一个盘上的历史根。

跑法: python tests/test_pack_preview_args.py   （全绿则退出码 0）
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
# 本包在 <ComfyUI>/custom_nodes/ 下，所以上两级就是 ComfyUI 根目录。
COMFY_ROOT = os.path.dirname(os.path.dirname(ROOT))
for _p in (COMFY_ROOT, ROOT, PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


# ---------------------------------------------------------------------------
# 加载 prompt_pack + routes（routes 要 aiohttp 与 common→torch，离线跑不起，
# 所以把这两个重依赖换成最小桩，其余代码原样执行）。
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
routes.__package__ = "marquee_director"
_pkg = type(sys)("marquee_director")
_pkg.__path__ = [PKG]
_pkg.prompt_pack = pp
_pkg.common = _FakeCommon()
sys.modules["marquee_director"] = _pkg
sys.modules["marquee_director.prompt_pack"] = pp
sys.modules["marquee_director.common"] = _FakeCommon()
rd_loader.exec_module(routes)


# 一份最小可解析的英文单版 PACK：段头**写了**时长（8.000s / 10.625s+1.625）。
PACK = "\n".join([
    "==========================================================",
    "MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包",
    "Project / 项目         : 口径回归",
    "Mode / 模式            : Ref2VA (full reference)",
    "Total duration / 总时长 : 00:18.625",
    "Segments / 段数        : 2",
    "Aspect / 画幅          : 16:9",
    "Music / 配乐           : N/A",
    "",
    "==========================================================",
    "ENGLISH VERSION / 英文版",
    "==========================================================",
    "",
    "########## S01 / 8.000s / EN ##########",
    "",
    "subject_definitions:",
    "<Subject 1> (S1): her face stays consistent with <Picture 1> in all shots.",
    "",
    "summary:",
    "A woman crosses a snowy ridge at dusk.",
    "",
    "retention_analysis:",
    "<Subject 1> persists from <Picture 1>; the ridge matches <Picture 2>.",
    "",
    "detailed_description:",
    "[Shot 1] At 00:00.000, wide static shot, <Subject 1> walks left to right.",
    "[Shot 2] At 00:04.000, slow push in, <Subject 1> turns toward camera.",
    "",
    "overall_soundscape:",
    "Wind across open snow.",
    "",
    "non_diegetic_music:",
    "N/A",
    "",
    "########## S02 / 10.625s+1.625=12.250s / EN ##########",
    "",
    "subject_definitions:",
    "<Subject 1> (S1): unchanged from S01, consistent with <Picture 1>.",
    "",
    "summary:",
    "She reaches the gate and pauses.",
    "",
    "retention_analysis:",
    "<Subject 1> carries the cloak state from S01; the gate matches <Picture 3>.",
    "",
    "detailed_description:",
    "[Shot 1] At 00:00.000, medium tracking shot, <Subject 1> walks to the gate.",
    "[Shot 2] At 00:05.000, static close shot, <Subject 1> stops and looks up.",
    "",
    "overall_soundscape:",
    "Boots on packed snow, low wind.",
    "",
    "non_diegetic_music:",
    "N/A",
    "",
    "END OF PACK / 文件结束",
])

# 同一份稿子，但段头**不写**时长 —— 这时 default_duration 才是唯一来源。
PACK_NO_DUR = (PACK
               .replace("########## S01 / 8.000s / EN ##########",
                        "########## S01 ##########")
               .replace("########## S02 / 10.625s+1.625=12.250s / EN ##########",
                        "########## S02 ##########"))


def durs(result):
    return [(s["new_seconds"], s["gen_seconds"]) for s in result["segments"]]


# ===========================================================================
# 1) default_duration / auto_fix / language / pack_override 必须真的参与解析
# ===========================================================================
base = routes._pack_preview(PACK, "sess")
check(base.get("ok") is True, "基线解析应 ok=True")
check(len(base["segments"]) == 2, "基线应解出 2 段，实得 %d" % len(base["segments"]))
# 标尺/段块的宽度口径。★ 权威值是 gen_seconds，**不是** new+handoff：
#   段头漏写 "+1.6" 时解析层会给接续段补一个 1.625s 的段首回放窗，它进
#   gen_seconds 却不进 handoff_seconds。恒等式是 gen == new + replay_in。
for s in base["segments"]:
    check(abs(s["gen_seconds"] - (s["new_seconds"] + s["replay_in"])) < 1e-6,
          "%s: gen_seconds(%.3f) 必须等于 new+replay_in(%.3f)"
          % (s["id"], s["gen_seconds"], s["new_seconds"] + s["replay_in"]))

# 段头写了时长 → default_duration 不该改变结果（它是"缺省"值）
same = routes._pack_preview(PACK, "sess", default_duration=5.0)
check(durs(same) == durs(base),
      "段头写了时长时 default_duration 不该改变段长（%s vs %s）"
      % (durs(same), durs(base)))

# 段头没写时长 → default_duration 必须是唯一来源，改它就得改段长
d11 = routes._pack_preview(PACK_NO_DUR, "sess", default_duration=11.0)
d5 = routes._pack_preview(PACK_NO_DUR, "sess", default_duration=5.0)
check(durs(d11) != durs(d5),
      "段头没写时长时 default_duration 必须生效（dd=11 %s vs dd=5 %s）"
      % (durs(d11), durs(d5)))
# 具体秒数由双网格吸附决定（chain_seconds：17k+5 帧 ∩ 3 的倍数），不是 dd 原值
# —— 11.0 会被抬到 12.25。这里只钉死标尺真正依赖的那两条不变量。
for tag, res in (("dd=11", d11), ("dd=5", d5)):
    for s in res["segments"]:
        check(abs(s["gen_seconds"] - (s["new_seconds"] + s["replay_in"])) < 1e-6,
              "%s %s: gen_seconds 必须等于 new+replay_in（标尺宽度口径）" % (tag, s["id"]))
    # ★ 回归钉：段头没写 "+1.6" 时，new+handoff 会比真实段长少一个回放窗
    #   （实测 dd=11 的 S02：new+handoff=12.750 而 gen=14.375，差 1.625s）。
    #   前端标尺必须用 gen_seconds，否则每一格都比它下方的缩略图短 1.625s。
    s2 = res["segments"][1]
    check(s2["replay_in"] > 0 and abs(s2["handoff_seconds"]) < 1e-6,
          "%s S02: 段头漏写 +1.6 → replay_in 补齐(%.3f)而 handoff 仍是 0"
          % (tag, s2["replay_in"]))
    check(s2["gen_seconds"] > s2["new_seconds"] + s2["handoff_seconds"] + 1e-6,
          "%s S02: new+handoff 确实少算了一个回放窗（gen=%.3f > new+ho=%.3f）"
          % (tag, s2["gen_seconds"], s2["new_seconds"] + s2["handoff_seconds"]))
check(d11["segments"][0]["gen_seconds"] > d5["segments"][0]["gen_seconds"],
      "缺省段长越大，段越长（dd=11 %.3f 应 > dd=5 %.3f）"
      % (d11["segments"][0]["gen_seconds"], d5["segments"][0]["gen_seconds"]))

# language：与节点 execute 同口径 —— auto 英文版优先，中文版取 zh 侧
check(routes._preview_primary_lang("auto") == "en", "language=auto → 主侧 en")
check(routes._preview_primary_lang("英文版") == "en", "language=英文版 → 主侧 en")
check(routes._preview_primary_lang("中文版") == "zh", "language=中文版 → 主侧 zh")
check(routes._preview_primary_lang("zh") == "zh", "language=zh → 主侧 zh")
check(routes._pack_preview(PACK, "sess", language="中文版")["primary_lang"] == "zh",
      "预览响应里要报出 primary_lang=zh")
check(routes._pack_preview(PACK, "sess")["primary_lang"] == "en",
      "不传 language 时默认主侧 en（与旧行为一致）")

# pack_override：非空且像 PACK → 代替 pack_text（与节点 execute 同口径）
# ★ 改成 12.250s 而不是 6.000s：6.0 会被 chain_seconds 的双网格吸附抬回 8.0
#   （17k+5 帧 ∩ 3 的倍数 → 192 帧 = 8.0s），拿它当判据会误判成"override 没生效"。
ov = PACK.replace("########## S01 / 8.000s / EN ##########",
                  "########## S01 / 12.250s / EN ##########")
r_ov = routes._pack_preview(PACK, "sess", pack_override=ov)
check(r_ov.get("pack_source") == "pack_override",
      "override 生效时 pack_source 应为 pack_override，实得 %r" % r_ov.get("pack_source"))
check(abs(r_ov["segments"][0]["gen_seconds"] - 12.25) < 1e-6,
      "override 的段长应被采用（12.25s），实得 %.3f —— 说明面板画的还是源文"
      % r_ov["segments"][0]["gen_seconds"])
check(abs(base["segments"][0]["gen_seconds"] - 8.0) < 1e-6,
      "不带 override 时仍是源文的 8.0s，实得 %.3f"
      % base["segments"][0]["gen_seconds"])

# widgets_values 位置错位会把 on_issue 的 "warn" 塞进 pack_override ——
# 守卫必须拦下，否则整块面板报「没有解析到任何分镜」。
r_bad = routes._pack_preview(PACK, "sess", pack_override="warn")
check(r_bad.get("pack_source") == "pack_text",
      'pack_override="warn" 应被守卫拦下、退回 pack_text')
check(len(r_bad["segments"]) == 2,
      'pack_override="warn" 时仍应解出 2 段，实得 %d' % len(r_bad["segments"]))

# 守卫与节点共用同一条实现（不许各写一份正则）
check(pp.looks_like_pack("########## S01 / 8s / EN ##########") is True,
      "looks_like_pack 认带斜杠段头")
check(pp.looks_like_pack("########## S01 ##########") is True,
      "looks_like_pack 认无斜杠段头")
check(pp.looks_like_pack("[Shot 1] At 00:00.000, x") is True,
      "looks_like_pack 认整段式 [Shot N]")
check(pp.looks_like_pack("warn") is False, "looks_like_pack 拒绝普通控件值")
check(pp.looks_like_pack("") is False, "looks_like_pack 拒绝空串")

# ===========================================================================
# 2) 缓存键必须含解析参数
# ===========================================================================
routes._preview_cache.clear()
k11 = routes._cached_preview(PACK_NO_DUR, "sess", default_duration=11.0)
n1 = len(routes._preview_cache)
k5 = routes._cached_preview(PACK_NO_DUR, "sess", default_duration=5.0)
check(durs(k11) != durs(k5), "同文本 + 不同 default_duration 不能串缓存")
check(len(routes._preview_cache) == n1 + 1,
      "两种参数各占一个缓存槽（键里带了 default_duration）")
check(durs(routes._cached_preview(PACK_NO_DUR, "sess", default_duration=5.0)) == durs(k5),
      "同参数重复调用命中缓存且结果一致")

routes._preview_cache.clear()
o1 = routes._cached_preview(PACK, "sess")
o2 = routes._cached_preview(PACK, "sess", pack_override=ov)
check(o1.get("pack_source") != o2.get("pack_source"),
      "pack_override 必须参与缓存键（否则带 override 的调用会命中无 override 的结果）")

# 返回值仍是副本：调用方就地改不能污染缓存
mut = routes._cached_preview(PACK, "sess")
mut["segments"][0]["gen_seconds"] = -1
after = routes._cached_preview(PACK, "sess")
check(after["segments"][0]["gen_seconds"] != -1, "缓存返回值必须是副本")

# ===========================================================================
# 3) _find_session：找不到的名字不许落到"磁盘上最新的会话"
# ===========================================================================
tmp = tempfile.mkdtemp()
try:
    os.makedirs(os.path.join(tmp, "剧本A", "EP02v3"))
    open(os.path.join(tmp, "剧本A", "EP02v3", "seg_01.mp4"), "w").close()
    os.makedirs(os.path.join(tmp, "剧本B", "r1"))
    # 刚开始渲、一段都还没落盘的新会话：目录是空的，也必须认得出来是会话目录
    os.makedirs(os.path.join(tmp, "剧本C", "r1"))
    orig = routes._session_dirs
    routes._session_dirs = lambda: [tmp]

    def rel(name):
        p = routes._find_session(name)
        return os.path.relpath(p, tmp).replace(os.sep, "/") if p else None

    got = {n: rel(n) for n in
           ("", "EP02v3", "ep02v3", "r1", "剧本A", "剧本B", "剧本C",
            "zzz_nonexistent_xyz", "../../Windows")}
    routes._session_dirs = orig

    low = lambda s: (s or "").lower()
    check(got["EP02v3"] == "剧本A/EP02v3", "正常版本名仍能找到，实得 %s" % got["EP02v3"])
    # Windows 的 isdir 不分大小写，命中的目录会带着**查询时**的大小写回来，
    # 所以这里只比小写 —— 要判的是"找没找对夹子"，不是字符串逐字节相等。
    check(low(got["ep02v3"]) == "剧本a/ep02v3",
          "大小写放宽仍生效，实得 %s" % got["ep02v3"])
    check(low(got["r1"]) in ("剧本b/r1", "剧本c/r1"),
          "多个剧本下的同名版本取最新的一个，实得 %s" % got["r1"])
    check(got["剧本A"] == "剧本A/EP02v3",
          "★ 传剧本夹名 → 该剧本下最新一版（不是剧本夹本身，否则 done 恒为 0），"
          "实得 %s" % got["剧本A"])
    check(got["剧本B"] == "剧本B/r1",
          "★ 传另一个剧本夹名 → 它自己的版本，实得 %s" % got["剧本B"])
    check(got["剧本C"] == "剧本C/r1",
          "空的新会话目录也要能解析到，实得 %s" % got["剧本C"])
    check(got[""] is None,
          "★ 空名字不许落到最新会话（时间线会显示别的剧本），实得 %s" % got[""])
    check(got["zzz_nonexistent_xyz"] is None,
          "★ 不存在的名字不许落到最新会话，实得 %s" % got["zzz_nonexistent_xyz"])
    check(got["../../Windows"] is None,
          "★ 带 .. 的名字不许落到最新会话，实得 %s" % got["../../Windows"])
finally:
    try:
        routes._session_dirs = orig
    except NameError:
        pass
    shutil.rmtree(tmp, ignore_errors=True)

# 找不到目录时 _session_state 必须如实报错，而不是拿别的会话顶上
st = routes._session_state("绝无此会话_zzz")
check(st.get("ok") is False, "会话找不到时 _session_state 应 ok=False")
check("looked_in" in st, "会话找不到时要报出找过哪些根目录")

# ===========================================================================
# 4) 段数漂移判据：渲到一半停下 ≠ PACK 改过
# ===========================================================================
tmp2 = tempfile.mkdtemp()
try:
    sess = os.path.join(tmp2, "剧本X", "r1")
    os.makedirs(sess)

    def write_segs(n, manifest_seconds=True):
        for f in os.listdir(sess):
            os.remove(os.path.join(sess, f))
        recs = []
        for i in range(1, n + 1):
            open(os.path.join(sess, "seg_%02d.mp4" % i), "w").close()
            recs.append({"index": i - 1, "seconds": 12.25, "key": "k%d" % i})
        if manifest_seconds:
            with open(os.path.join(sess, "manifest.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"version": 1, "session": "r1", "segments": recs}, fh)

    orig2 = routes._session_dirs
    routes._session_dirs = lambda: [tmp2]

    # 9 段的稿子只渲到第 4 段就被取消：连续前缀，不是漂移
    write_segs(4)
    st = routes._session_state("r1", 9)
    check(st.get("done") == 4 and st.get("disk_count") == 4,
          "被中断的会话应读到 done=4 / disk=4，实得 %s/%s"
          % (st.get("done"), st.get("disk_count")))
    check(st.get("drift") is False,
          "★ 磁盘 1..4 连续、计划 9 段 = 渲到一半停了，不许报漂移（实得 drift=%s）"
          % st.get("drift"))

    # 磁盘段数**超过**计划：旧版 PACK 多切了几段，seg_NN 与面板段号错开 —— 真漂移
    write_segs(11)
    check(routes._session_state("r1", 9).get("drift") is True,
          "磁盘 11 段 > 计划 9 段应判漂移")

    # 磁盘段号缺号（有 seg_01、seg_03，没有 seg_02）：done < disk_count —— 真漂移
    for f in os.listdir(sess):
        os.remove(os.path.join(sess, f))
    open(os.path.join(sess, "seg_01.mp4"), "w").close()
    open(os.path.join(sess, "seg_03.mp4"), "w").close()
    st3 = routes._session_state("r1", 9)
    check(st3.get("done") == 1 and st3.get("disk_count") == 2,
          "缺号会话应读到 done=1 / disk=2，实得 %s/%s"
          % (st3.get("done"), st3.get("disk_count")))
    check(st3.get("drift") is True, "段号不连续应判漂移")

    # 一段都还没落盘：无可比对，不判漂移（2026-10-18 那条旧判据仍要成立）
    for f in os.listdir(sess):
        os.remove(os.path.join(sess, f))
    check(routes._session_state("r1", 9).get("drift") is False,
          "磁盘 0 段不判漂移")

    # ★ 跨盘符不许把 /h3/session 整个打挂：tmp 在 C:、真实输出根在别的盘时
    #   os.path.relpath 会抛 ValueError。上面每个 _session_state 调用都已经
    #   走过这条路径，这里再直接钉一次返回值形态。
    write_segs(2)
    sub = routes._view_subfolder(sess)
    check(isinstance(sub, str) and sub != "",
          "_view_subfolder 跨盘符也要给出可用字符串，实得 %r" % sub)
    st4 = routes._session_state("r1", 9)
    check(st4.get("ok") is True and len(st4.get("segments") or []) == 2,
          "跨盘符时 /h3/session 仍应 ok=True 且列出 2 段，实得 %s" % st4.get("ok"))
    check(all((s.get("url") or "").startswith("/view?filename=seg_")
              for s in (st4.get("segments") or [])),
          "每段的 /view 直链都要拼得出来（缩略图靠它）")

    # 全部渲完：不判漂移
    write_segs(9)
    check(routes._session_state("r1", 9).get("drift") is False,
          "磁盘 9 段 == 计划 9 段不判漂移")
    routes._session_dirs = orig2
finally:
    try:
        routes._session_dirs = orig2
    except NameError:
        pass
    shutil.rmtree(tmp2, ignore_errors=True)

# ===========================================================================
print("=" * 62)
if FAIL:
    for m in FAIL:
        print("FAIL:", m)
    print("%d 项未通过" % len(FAIL))
    sys.exit(1)
print("test_pack_preview_args: 全绿")
sys.exit(0)
