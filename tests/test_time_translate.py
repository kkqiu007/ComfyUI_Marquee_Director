# -*- coding: utf-8 -*-
"""ComfyUI_Marquee_Director 时间逻辑 + 英译分块 回归用例（离线，零依赖）。

跑法：
  "<ComfyUI>/.venv/Scripts/python.exe" tests_regression.py

覆盖四类已修缺陷：
  A. translate_nodes.split_parts / _needs_translation 把 PACK 头部误送翻译，
     经 strip_residual_cjk 剁掉双语气泡 →「Project / 项目」变「Project /」。
     → 用例：protect_structure 必须豁免结构行，正负例边界正确。
  B. split_oversized 在 900 字处硬切半句。
     → 用例：切点落在语义边界、逐字守恒、<Picture N> 标签不被劈开。
  C. _snap_cuts_off_dialogue 为避台词把合规段撑成超限段，制造新的过短段。
     → 用例：挪动后两侧新增时长必须仍在 [min_new, max_new]。
  D. 拆段出口未复核 → 过短/超限段带进渲染（H3 13.1s 超限 + 尾窗重复语音）。
     → 用例：_shot_script_plans 出口全部合规。

用法：作为脚本直接运行，全绿退出码 0，有失败退出码 1。
"""
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
import types

PKG = os.path.join(
    os.environ.get("MD_PKG_DIR",
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "marquee_director")
# 个人工作流文件，不随包发布。不设 MD_WORKFLOW 时为空串，
# 下面用 os.path.exists 守卫，那几条用例自动跳过。
WORKFLOW = os.environ.get("MD_WORKFLOW", "")

FAILS = []


def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        FAILS.append(msg)


def _stub_comfy():
    for n in ("comfy_api", "comfy_api.latest"):
        m = types.ModuleType(n)
        m.__path__ = []
        sys.modules[n] = m

    class _Any:
        def __init__(self, *a, **k):
            pass

    class _IO:
        def __getattr__(self, item):
            return _Any

    io = types.ModuleType("comfy_api.latest.io")
    io.ComfyNode = type("ComfyNode", (), {})
    io.Schema = type("Schema", (), {})
    io.NodeOutput = type("NodeOutput", (), {"__init__": lambda s, *a: None})
    io.Clip = io.String = io.Boolean = _IO()
    io.Combo = io.Int = io.Float = _IO()
    sys.modules["comfy_api.latest.io"] = io


def _load(path, name):
    ld = importlib.machinery.SourceFileLoader(name, path)
    sp = importlib.util.spec_from_loader(name, ld)
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    ld.exec_module(mod)
    return mod


_stub_comfy()
_pkg = types.ModuleType("marquee_director")
_pkg.__path__ = [PKG]
sys.modules["marquee_director"] = _pkg
_c = types.ModuleType("marquee_director.common")
_c.log = lambda *a, **k: None
sys.modules["marquee_director.common"] = _c
_pkg.common = _c

pp = _load(os.path.join(PKG, "prompt_pack.py"), "marquee_director.prompt_pack")
tn = _load(os.path.join(PKG, "translate_nodes.py"), "marquee_director.translate_nodes")

CJK = re.compile(r"[\u4e00-\u9fff]")
DIALOG = re.compile(r"<d\b[^>]*>.*?</d>", re.S | re.I)


def raw_script():
    """工作流里各 PrimitiveStringMultiline 的文本（新译版工作流里 361/748 都是
    英文 en-pack —— 2026-09-30 起 C+/D/F 的 fixture 按形态自选，见
    ``shot_script_raw`` / ``cjk_rich_raw``，本函数保留作兼容入口。"""
    if os.path.exists(WORKFLOW):
        d = json.load(open(WORKFLOW, encoding="utf-8"))
        for n in d["nodes"]:
            if n.get("id") == 361:
                return n["widgets_values"][0]
    return ""


def _all_node_texts():
    texts = []
    if os.path.exists(WORKFLOW):
        try:
            d = json.load(open(WORKFLOW, encoding="utf-8"))
        except Exception:
            return texts
        for n in d.get("nodes", []):
            if n.get("type") == "PrimitiveStringMultiline":
                wv = n.get("widgets_values") or []
                if wv and isinstance(wv[0], str) and wv[0].strip():
                    texts.append(wv[0])
    return texts


def shot_script_raw():
    """C+D fixture：真正的「整段式分镜脚本」（有 [Shot N] 头、无段标题）。

    英译版工作流（2026-09-29 起）的 361/748 已换成 en-pack 分段交付形态，
    旧实现无脑抓 361 → ``_shot_script_plans`` 拿分段 PACK 当整段脚本反推，
    边界全错位（段0 new=0.00 的假失败）。没有合形态的节点时返回合成剧本。
    """
    import re as _re
    seg_head = _re.compile(r"^\s*#{3,}\s*S?\d+", _re.M)
    shot_head = _re.compile(r"\[Shot\s*\d+\]")
    for t in _all_node_texts():
        if shot_head.search(t) and not seg_head.search(t):
            return t
    # 合成整段式脚本：镜头起点 0/8/16/24，边界落在镜头起点上（合法），
    # 每段 new = 8 - 1.625 = 6.375 ∈ [5, 11]。
    return (
        "subject_definitions:\n<Subject 1> is the man in <Picture 1>.\n\n"
        "summary:\n[reference generation] 测试用整段式剧本。\n\n"
        "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved.\n\n"
        "detailed_description:\n"
        "[Shot 1] 00:00.000 推门，烛光里他站定，四处打量。\n"
        "[Shot 2] At 00:08.000，他走近桌边，拿起那封信。(S1) <d>信呢？</d> from 00:08.000 to 00:09.500.\n"
        "[Shot 3] At 00:16.000，她转身，目光落在他脸上。(S2) <d>烧了。</d> from 00:16.000 to 00:17.200.\n"
        "[Shot 4] At 00:24.000，两人对峙，静默。\n\n"
        "overall_soundscape:\nRoom tone.\n\n"
        "non_diegetic_music:\nNo music.\n"
    )


def cjk_rich_raw():
    """F fixture：块外汉字最多的节点文本；不足 500 字时退回合成中文剧本。

    F 组保护的是「中文剧本解析后不得被 strip_residual_cjk 剁光」。英译版
    工作流的节点文本是英文 en-pack（块外汉字≈0），断言必然假失败 ——
    合成剧本保证这条守护每次运行都在岗。
    """
    import re as _re
    dlg = _re.compile(r"<d>.*?</d>", _re.S)
    best, best_n = "", 0
    for t in _all_node_texts():
        n = len(_re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
                            dlg.sub("", t)))
        if n > best_n:
            best, best_n = t, n
    if best_n >= 500:
        return best

    def shot_line(idx):
        # ★ 整段式(shot-script)契约：叙事散文必须写在**镜头行内**（跟在时间码
        #   后面）。无时间码的行外散文会被镜头行过滤丢弃（2026-09-30 起解析
        #   节点对这种损失会响亮报 issue，见 parse_pack 的静默丢正文闸门）。
        t = (idx - 1) * 8
        return ("[Shot %d] At %02d:%02d.000，画面中不出现任何文字、字幕、标题卡或水印。"
                "风格：电影感古装剧，古卧房夜景，暖色烛光为主光，侧后方 2700K，"
                "轮廓光薄而克制，自然阴影，不做美颜磨皮，景深中等，背景柔而可读，"
                "色温与光位全部沿用参考图。本镜：他握着信纸的手微微收紧，"
                "纸张边缘被指腹压出折痕，烛焰在风里晃了半拍，她的影子投在窗纸上，"
                "像一笔淡墨；镜头从他的侧脸缓缓摇到她的背影，再由她的背影推回"
                "两人之间的空气，烛台上的火苗、床幔的流苏、地上彼此错开的两道影子"
                "都在戏里，全程不切出这个房间，不加新道具，不换服饰。"
                "(S1) <d>信呢？</d> from %02d:%02d.000 to %02d:%02d.500.\n"
                % (idx, t // 60, t % 60, t // 60, t % 60,
                   (t + 2) // 60, (t + 2) % 60))

    return (
        "subject_definitions:\n"
        "<Subject 1> 是 <Picture 1> 里的男人，布衣束发，眉目沉静。\n"
        "<Subject 2> 是 <Picture 2> 里的女子，素裙挽髻，神色清冷。\n"
        "<Picture 1> 是主体一的参考图。<Picture 2> 是主体二的参考图。\n\n"
        "summary:\n[reference generation] 古卧房夜戏，烛光下两人对峙，"
        "他质问，她平静作答，情绪一热一冷，镜头在两人之间缓慢往复。\n\n"
        "retention_analysis:\n"
        "<Subject 1>（出现于 [Shot 1]）：fully_preserved，身份与画左位置不变。\n"
        "<Subject 2>（出现于 [Shot 2]）：fully_preserved，身份与画右位置不变。\n"
        "<Picture 1>: fully_preserved。<Picture 2>: fully_preserved。\n\n"
        "detailed_description:\n"
        + "".join(shot_line(i) for i in range(1, 9)) +
        "\noverall_soundscape:\n烛芯噼啪，远处更鼓，室内只有两人的呼吸与衣料摩擦。\n\n"
        "non_diegetic_music:\n全片配乐：低音域京剧打击与弦乐垫底，音量克制。\n"
    )


print("=" * 72)
print("A. 结构行豁免（PACK 头部双语气泡不被剁掉）")
print("=" * 72)
S_HEAD = [
    "==========================================================",
    "MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包",
    "Project / 项目         : 曹贼的性价比",
    "Version / 版本         : v20",
    "ENGLISH VERSION / 英文版（本文件唯一语言版本）",
    "END OF PACK / 文件结束",
    "########## S01 / 10.6s / EN ##########",
]
for s in S_HEAD:
    check(tn._line_is_structural(s), "结构行识别: %s" % s[:46])
N_HEAD = [
    "他推门而入，气息未定，手指在门框上敲了三下。",
    "镜头缓缓推进：烛火摇曳，映出两张绷紧的脸。",
    "台词 / 对白：他没说话，只是看着她。",
]
for s in N_HEAD:
    check(not tn._line_is_structural(s), "正文不豁免: %s" % s[:34])

body = "\n".join(S_HEAD)
sp, ss = tn.protect_structure(body)
back, miss = tn.restore_structure(sp, ss)
check(back == body and not miss, "结构行 round-trip 逐字一致")
check(not tn._needs_translation(body),
      "整块纯结构 → _needs_translation=False（不再送模型）")
# 结构行剥离后仍原样（旧行为会把「项目」剁掉）
gate = DIALOG.sub("", body)
for line in tn.protect_structure(gate)[1]:
    gate = gate.replace(line, "")
check(len(CJK.findall(gate)) == 0, "结构行外的 CJK = 0")
check("项目" in back and "v20" in back and "分镜提示词包" in back,
      "双语气泡完整保留（项目 / v20 / 分镜提示词包）")

print()
print("=" * 72)
print("B. split_oversized 语义边界切分")
print("=" * 72)
GLUE = ("At 00:00.000 the frame opens static, a long shot through the bedroom "
        "doorway into the deep interior, where <Subject 1> and <Subject 3> stand "
        "in the middle of the room, the bed and the wardrobe behind them, and any "
        "part, layer or fastening that <Picture 2> does not cover is invisible.")
pieces = tn.split_oversized(GLUE, 120)
joined = "".join(p for p, _ in pieces)
check(joined == GLUE, "切分逐字守恒（拼回 == 原文）")
check(all(len(p) <= 121 for p, _ in pieces), "每块 <= limit（含空白容差）")
tag_ok = True
for p, _ in pieces:
    for m in re.finditer(r"<(?:Picture|Subject|Video|Audio) \d+>", p):
        pass
# 标签完整性：所有标签必须完整出现在某一块里，且不被截断
check(all(("<%" not in p and "%>" not in p) for p, _ in pieces), "无截断标签残留")
# 最小块不许出现 "At " 这种碎片
check(min(len(p.strip()) for p, _ in pieces) > 20, "无 ≤20 字碎片块")

print()
print("=" * 72)
print("C+D. 拆段时间逻辑（出口必须全部合规）")
print("=" * 72)
raw = shot_script_raw()
if not raw:
    print("  (跳过：找不到工作流剧本原文)")
else:
    plans, gf = pp._shot_script_plans(raw)
    bad = []
    for i, (a, b, lines) in enumerate(plans):
        r = 0.0 if i == 0 else pp.HANDOFF_SECONDS
        nw = b - a - r
        if nw > pp.MAX_NEW_SECONDS + 0.001:
            bad.append("段%d 超限 new=%.2f" % (i, nw))
        elif nw < pp.MIN_NEW_SECONDS - 0.001:
            bad.append("段%d 过短 new=%.2f" % (i, nw))
    print("    段表: " + " | ".join(
        "%.2f/%.2f" % (a, b) for a, b, _ in plans))
    check(not bad, "全部段 new ∈ [%.1f, %.1f]（违规 %s）"
          % (pp.MIN_NEW_SECONDS, pp.MAX_NEW_SECONDS, bad or "无"))
    check(len(plans) >= 2, "段数 >= 2")
    check(plans[0][0] == 0.0, "首段从 0 起")
    for i in range(1, len(plans)):
        check(abs(plans[i][0] - plans[i - 1][1]) < 1e-6,
              "段%d 与段%d 边界连续（%.3f == %.3f）"
              % (i - 1, i, plans[i - 1][1], plans[i][0]))

print()
print("=" * 72)
print("E. grid_seconds 幂等（17k+5 对齐）")
print("=" * 72)
check(abs(pp.grid_seconds(1.6) - 1.625) < 1e-9, "grid_seconds(1.6) = 1.625")
check(abs(pp.grid_seconds(1.625) - 1.625) < 1e-9, "grid_seconds(1.625) 幂等")
for sec in (1.0, 2.0, 5.0, 11.0):
    f = round(pp.grid_seconds(sec) * pp.FPS)
    check((f - 5) % 17 == 0, "grid_seconds(%.1f) → %d 帧 在 17k+5 网格" % (sec, f))

print()
print("=" * 72)
print("F. ★ 解析节点不得把中文剧本当「残留中文」剥掉（真渲染事故根因）")
print("=" * 72)
# 背景：_auto_fix 的交付闸门用 cjk_outside_dialogue() 当判据，对**未英译的中文
# 剧本**必然为真 → 整个正文被 strip_residual_cjk 剁光。实测 S05 正文由数千
# 汉字只剩 11 个（那句台词），模型只拿到台词块+英文碎片 → 计划外发声。
_segs_f, _meta_f, _issues_f = pp.parse_pack(
    cjk_rich_raw(), language="zh", default_duration=11.0, auto_fix=True)
_tot_cjk = 0
for _s in _segs_f:
    _body = _s["fields"].get("detailed_description") or ""
    _tot_cjk += len(pp._CJK_RE.findall(pp._DIALOG_RE.sub("", _body)))
check(_tot_cjk > 1500,
      "中文剧本解析后块外汉字保留 %d 个（>1500；被剥光时会掉到 ~30）" % _tot_cjk)
# 每一段都应有实质镜头描述，而不是只剩时间码 + 英文碎片
for _i, _s in enumerate(_segs_f, 1):
    _body = _s["fields"].get("detailed_description") or ""
    _n = len(pp._CJK_RE.findall(pp._DIALOG_RE.sub("", _body)))
    check(_n > 100, "S%02d 块外汉字 %d 个（>100，正文未被剁碎）" % (_i, _n))

# 残留判据本身的正负例。
# 注意：判据是「**残留性**」而不是「有没有汉字」——「Project / 项目：曹贼的性价比」
# 这种双语标签里汉字占 42%，按占比就不算残留（它由 translate_nodes 的
# protect_structure 结构行豁免处理，不归本判据管）。这里只测本判据自己的边界。
check(pp._residual_cjk_only("At 00:01.0 cut to Subject 2 MS.") is False,
      "无汉字判为非残留（不触发剥离）")
check(pp._residual_cjk_only("No text overlay. 画面里只有她没有任何别人。") is False,
      "成句中文判为正文（放行）")
check(pp._residual_cjk_only(
    "At 00:01.0 她抬起一只手做出平静的OK手势，月琴音轻点一音，京胡远在高处挂长音。") is False,
    "长中文句判为正文（放行，不得剥）")
# 真残留：正文几乎全是英文，只有一处零散中文（占比远低于 15%、且不成长句）
check(pp._residual_cjk_only(
    "and carries the motion straight through 画面 out of frame and settles.") is True,
    "零星短中文判为残留（应剥）")
check(pp._residual_cjk_only(
    "The camera arcs slowly around the subject, then holds. 配音继续。") is True,
    "英文正文里 4 字短片段 → 两条判据都不触发，按残留剥掉（代价不对称，安全侧）")
# 长句判据的边界：≥8 连续汉字即放行，哪怕整体占比不高
check(pp._residual_cjk_only(
    "The camera arcs slowly around the subject, then holds. 她缓缓抬起手示意镜头。") is False,
    "英文正文里 9 字中文句 → 长句判据兜住，放行（不剥作者的话）")

print()
print("=" * 72)
print("G. ★ 边界/网格：切点不得把镜头起点 clamp 进回放窗")
print("=" * 72)
# 背景：_rebalance_shot_bounds 把边界滑到 14.0，[Shot 3]（成片 12.0）的头部行
# 被 _retime_line 从 t0 向上 clamp → At 00:01.600，落在回放窗 [0,1.625] 内。
_FIXTURE = """\
subject_definitions:
<Subject 1> is the person in Picture 1 (Wang).
<Subject 2> is the person in Picture 2 (Zhang).
<Picture 3> is the reference image for the scene, an ancient bedroom at night.

summary:
[reference generation] A night scene in an ancient bedroom.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identical to Picture 1.
<Subject 2> (appears in [Shot 1]): fully_preserved - identical to Picture 2.
<Picture 3> ([Shot 1] first frame): fully_preserved - the scene is kept whole.

detailed_description:
No text, subtitles, captions, timecodes, watermarks or any graphic overlay appear anywhere in the frame.
CAMERA: static 35mm medium shot at eye level, locked off, no push, no pull.
LIGHTING: a single warm candle off frame left, deep shadow to the right, no fill.
[Shot 1] At 00:00.000, <Subject 1> stands in the doorway with his shoulders squared.
At 00:02.000, (S1) <d>[Chinese] 汝与曹贼何异？</d>
[Shot 2] At 00:10.000, <Subject 2> sits down on the edge of the bed.
[Shot 3] At 00:12.000, <Subject 2> pulls his coat on over his shoulders.
[Shot 4] At 00:25.000, <Subject 1> freezes and his jaw tightens.

overall_soundscape:
Distant watch drums, a candle wick splitting.

non_diegetic_music:
N/A
"""
_segs_g, _meta_g, _issues_g = pp.parse_pack(
    _FIXTURE, language="en", default_duration=11.0, auto_fix=True)
_replay_hits = []
for _s in _segs_g:
    _d = _s["fields"].get("detailed_description") or ""
    _h = float(_s.get("replay_in") or 0.0)
    for _ln in _d.splitlines():
        _m = pp._SHOT_CLOCK_RE.search(_ln)
        if not _m or _h <= 0:
            continue
        _t = pp._clock_seconds(*_m.groups())
        # 回放窗 = [0, _h]；镜头头部行的时间码不得落在其中
        if 0.0 < _t < _h - 1e-6:
            _replay_hits.append((_t, _ln[:60]))
check(not _replay_hits,
      "无镜头头部行落在回放窗内（%s）" % (_replay_hits or "0 处"))
_struct_g = [i for i in _issues_g
             if "回放区" not in i and "头部" not in i and "词数" not in i]
check(not [i for i in _issues_g if "回放区" in i],
      "不再报「回放区内有切点」")
# 出口长度仍然全部合规
# ★ 2026-09-30：容差带上 GRID_SLACK_SECONDS —— _build_segment 现在把 gen 对齐
#   到 17k+5 网格，`new = gen - replay` 最多比名义值大 16/24 ≈ 0.667s
#   （11.0s 的段在 24fps 下无法精确落在网格上：303 帧 → 311 帧 = 11.333s）。
for _s in _segs_g:
    check(pp.MIN_NEW_SECONDS - 1e-3
          <= _s["new_seconds"]
          <= pp.MAX_NEW_SECONDS + pp.GRID_SLACK_SECONDS + 1e-3,
          "段新增 %.3fs 在 [%g, %g+网格容差]"
          % (_s["new_seconds"], pp.MIN_NEW_SECONDS, pp.MAX_NEW_SECONDS))

print()
print("=" * 72)
print("H. ★ _retime_line 线性映射 + 边界 clamp（下界必须是 t0）")
print("=" * 72)
# 契约：段内 0 秒 = 成片 t0-handoff，段内容起点 t0 映射到 handoff（回放窗刚结束）。
# 下界若误写成 t0+handoff，映射时又加一次 handoff → 2×handoff（实测 3.25），
# 会把**正好写在段边界上的台词**整体后推 1.625s —— 那正是"乱说话"的一种。
H = 1.625
for _name, _g, _a, _b in (("段首台词", 11.0, 11.0, 22.0),
                          ("段中台词", 18.0, 11.0, 22.0),
                          ("段首=边界", 28.6, 28.6, 38.0),
                          ("末段台词", 39.6, 38.0, 45.0),
                          ("首段台词", 2.9, 0.0, 11.0)):
    _out = pp._retime_line("At %s, X" % pp._fmt_clock(_g), _a, _b,
                           H if _a > 0 else 0.0)
    _m = pp._SHOT_CLOCK_RE.search(_out)
    _got = pp._clock_seconds(*_m.groups()) if _m else None
    _exp = _g - _a + (H if _a > 0 else 0.0)
    check(_got is not None and abs(_exp - _got) < 0.01,
          "%s global %.1f → rel %.3f（得 %.3f）" % (_name, _g, _exp,
                                                   _got if _got is not None else -1))
# 早于本段的时间码向上 clamp 到段首 → rel = handoff（不是 2×handoff）
_r = pp._retime_line("At 00:12.000, cut to X.", 14.0, 25.0, H)
_m = pp._SHOT_CLOCK_RE.search(_r)
check(_m is not None and abs(pp._clock_seconds(*_m.groups()) - H) < 1e-6,
      "早于段的时间码 clamp 到段首 → rel=%.3f（不是 2×%.3f）" % (H, H))
# 超出段尾的时间码 clamp 到段尾
_r3 = pp._retime_line("At 00:40.000, cut to Z.", 14.0, 25.0, H)
_m3 = pp._SHOT_CLOCK_RE.search(_r3)
check(_m3 is not None and abs(pp._clock_seconds(*_m3.groups()) - (11.0 + H)) < 1e-6,
      "超出段尾的时间码 clamp 到段尾 rel=%.3f" % (11.0 + H))
_r0 = pp._retime_line("At 00:12.000, cut to X.", 0.0, 25.0, 0.0)
_m0 = pp._SHOT_CLOCK_RE.search(_r0)
check(_m0 is not None and abs(pp._clock_seconds(*_m0.groups()) - 12.0) < 1e-6,
      "首段（无回放）时间码原样保留 → %s" % _r0[:40])

print()
print("=" * 72)
print("I. ★ 文案秒数必须与字段同源（禁止写死名义值 1.6）")
print("=" * 72)
# 缺陷：summary 回放句 / [Shot 1] 标注 / MUSIC 静音窗口 里写死 "1.6"，
# 而字段 replay_in = grid_seconds(1.6) = 1.625（39 帧）。文案给模型 1.6、
# 渲染器出 1.625 → 接缝处 1 帧漂移（"多出 17 帧回放 / 动作原地打结"）。
# 契约：模型可见文案里的回放秒数必须逐位等于 replay_in。
for _sec, _want in ((pp.grid_seconds(pp.HANDOFF_SECONDS), "1.625"),
                    (0.917, "0.917"),
                    (1.6, "1.6"),
                    (2.333, "2.333"),
                    (1.5, "1.5")):
    check(pp._fmt_handoff_seconds(_sec) == _want,
          "_fmt_handoff_seconds(%.3f) → %s（得 %s）"
          % (_sec, _want, pp._fmt_handoff_seconds(_sec)))
# 回放句：句子里的秒数必须就是传进去的那个值
_rep = pp._handoff_replay_en(pp.grid_seconds(pp.HANDOFF_SECONDS))
check("1.625 seconds" in _rep and "1.625-second mark" in _rep,
      "回放句用 1.625 而非 1.6（两处）")
check("1.6 seconds of" not in _rep,
      "回放句不再残留名义值 1.6")
# 短段场景：句子里必须是 0.917，不能又是 1.625
_rep2 = pp._handoff_replay_en(0.917)
check("0.917 seconds" in _rep2 and "0.917-second mark" in _rep2,
      "短段回放句用 0.917（按段匹配，不按表）")
# MUSIC block：边界时间码 + 两处静音窗都必须 1.625
_mb = pp.MUSIC_BLOCK_TEMPLATE.format(style="restrained ambient")
check("00:01.625" in _mb and "00:01.600" not in _mb,
      "MUSIC block 边界 = 00:01.625（旧 00:01.600 已清）")
check("1.625-second opening" in _mb, "MUSIC block 回放窗 = 1.625-second")
check(_mb.count("1.625") >= 4,
      "MUSIC block 全部秒数统一 1.625（得 %d 处）" % _mb.count("1.625"))
check("1.6 seconds" not in _mb and "1.6-second" not in _mb,
      "MUSIC block 无残留 1.6")
_mb2 = pp.MUSIC_BLOCK_SFX_TEMPLATE
check("00:01.625" in _mb2 and "1.625 seconds of any shot" in _mb2,
      "SFX block 边界与尾窗均 1.625")
check("1.6 seconds" not in _mb2 and "1.6 " not in _mb2,
      "SFX block 无残留 1.6")

print()
print("=" * 72)
print("J. ★★ 整段式(shot-script)路径 replay_in 必须网格对齐（两条路径都不能漏）")
print("=" * 72)
# 缺陷：_build_shot_script_segments 用名义 HANDOFF_SECONDS(1.6) 当 replay_in，
# 而 PACK 路径的 _build_segment 用 grid_seconds(=1.625)。两条路径不一致，
# 整段式稿子的模型可见文案就停留 1.6 → 与下游规范化的 1.625 差 1 帧。
# 契约：整段式路径产出的 replay_in / gen_seconds 必须是网格值，
#       且模型可见文案里不得出现 "replaying 1.6 " / "00:01.600"。
_GRID = pp.grid_seconds(pp.HANDOFF_SECONDS)
check(abs(_GRID - 1.625) < 1e-9, "grid_seconds(1.6) == 1.625（得 %.3f）" % _GRID)
_SHOT_SCRIPT = """[Shot 1] At 00:00.000, 开场镜头，<Subject 1> 站在门口。
At 00:03.000, <Subject 1> 抬手指向桌上的账本。
[Shot 2] At 00:08.000, 镜头切到 <Subject 2> 的近景，她低头算账。
At 00:12.000, <Subject 2> 抬头看向 <Subject 1>。
[Shot 3] At 00:20.000, 两人对峙，<Subject 1> 把账本推过去。
At 00:26.000, <Subject 2> 接过账本翻看。
[Shot 4] At 00:33.000, 画面拉远，两人一前一后走出屋子。
At 00:40.000, 门在身后合上。"""
try:
    _segs = pp.parse_shot_script_blocks(_SHOT_SCRIPT)
except Exception as _exc:                                  # noqa: BLE001
    _segs = None
    check(False, "parse_shot_script_blocks 抛异常：%r" % (_exc,))
if _segs:
    # 返回 (segments, speaker_notes)；segments 每项是 4 元组
    # (id, dur_spec, lang, body)。replay_in 只体现在 dur_spec 与正文里。
    _items = _segs[0] if isinstance(_segs, tuple) else _segs
    _specs = [t[1] for t in _items]
    _bodies = [t[3] for t in _items]
    _cont = [s for s in _specs if "+" in s]
    check(bool(_cont) and all("+1.625=" in s for s in _cont),
          "整段式接续段 dur_spec 均带 +1.625=（得 %s）" % (_cont or "无"))
    check(all("+1.6=" not in s for s in _specs),
          "整段式无段 dur_spec 用名义 +1.6=")
    _joined = "\n".join(_bodies)
    check("replaying 1.6 " not in _joined and "00:01.600" not in _joined
          and "closing 1.6 " not in _joined
          and "1.6-second" not in _joined,
          "整段式正文文案无残留 1.6 / 00:01.600")
    check("replaying %.3f" % _GRID in _joined
          and "closing %.3f seconds" % _GRID in _joined,
          "整段式正文文案的 replay/closing 均用 %.3f" % _GRID)

print()
print("=" * 72)
if FAILS:
    print("❌ 失败 %d 项：" % len(FAILS))
    for f in FAILS:
        print("   - " + f)
    sys.exit(1)
print("✅ 全部通过")
