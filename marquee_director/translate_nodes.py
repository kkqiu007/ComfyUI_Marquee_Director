# -*- coding: utf-8 -*-
"""剧本英译（v9）：把任意语言的完整剧本 / 整段式分镜脚本改写成**英文单版**。

★ v9 变更（2026-09-26，删繁就简 + 台词固定格式）：
  - **台词固定 `(Sx) <d>[Chinese] 原文</d>`（编号+<d>块+语言标签+原文，缺一不可）**：
    送模型前代码层把每句台词规范化成 `<d>[语言] 原文</d>`（语言标签缺则补 [Chinese]）；
    还原后逐字填回。`<d>` 块 / [语言] / 原文 三要素由代码 100% 保证；
    `(Sx)` 说话人编号依上下文由模型补（源里已有则保留，不编造、不误伤）。
  - **提示词删繁就简**：砍掉全部冗余条款，只留「翻译成英文 + 台词固定格式 + 输出」三节。
  - 保留 v8 的三级兜底：译后自查 → 定向二次补翻 → 逐行救捞。
  - **schema 与原版完全一致**（17 参数），工作流里 API 配置原样恢复。
"""
from __future__ import annotations

import inspect
import os
import re
import time

from comfy_api.latest import io

from . import prompt_pack as pp
from .common import log

CATEGORY = "ComfyUI_Marquee_Director"

# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------
# 台词块：非贪婪 + DOTALL，台词里可以有换行。
_DIALOG_RE = re.compile(r"<d\b[^>]*>.*?</d>", re.S | re.I)
# 台词语言标签：<d> 块内的 [Word]（英文词），如 [Chinese] / [Japanese]。
_LANG_TAG_RE = re.compile(r"^\s*\[([A-Za-z][A-Za-z0-9-]*)\]\s*([\s\S]*)$")
# 说话人编号 (Sx) + <d> 块：如 "(S2) <d>[Chinese] 快跑！</d>"。
# group(1)=说话人编号 (Sx)，group(2)=完整 <d>…</d>。编号是下游 H3PromptPackParser
# 推断的；翻译节点只在「源里已有编号」时保留它（不编造、不误伤无编号台词）。
_SPEAKER_DIALOG_RE = re.compile(
    r"(\(\s*S\d+\s*\))\s*(<d\b[^>]*>.*?</d>)", re.S | re.I)
# 占位符。ASCII 方括号 tokenizer 切得稳。
_PLACEHOLDER_RE = re.compile(r"\[\[D(\d+)\]\]")
# 未编号 / 字面占位符泄漏（模型把占位符名当成文本写进去，如 [[Dn]] [[DN]] [[D?]]）。
# 这类不能还原成台词，必须判为失败触发二次翻译。
_UNNUMBERED_PH_RE = re.compile(r"\[\[D(?:n|N|\?|[A-Za-z]{1,4}|…)\]\]")
# 中文字符。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# 台词标签行：行首（可带缩进）的 台词/对白/口白/旁白/画外音/VO/Speech，
# 标签后必须紧跟 分隔符(空白 / | / ｜ / : / ：)或行尾 —— 防止把
# 「台词中段」「台词落定」这类镜头/说明段落误判成台词（那些要照翻）。
# 后面是内容（可能是中文台词，也可能已经含 [[Dn]]）。
_LABEL_DIALOG_RE = re.compile(
    r"^[ \t]*(?:台词|对白|口白|旁白|画外音|[Vv][Oo]|[Ss]peech)"
    r"(?=$|[\s|｜:：])(?:[|｜:：]\s*)?(?P<t>[^\n]*)$", re.M)
# 引号内的中文对白（「…」「『…』『“…”‘…'）。
_QUOTED_CJK_RE = re.compile(
    r"[「『\u201c\u2018]([^「」『』\u201c\u201d\u2018\u2019\n]{2,160})[」』\u201d\u2019]")

# ---- 事实项：译文必须逐条保住（丢了不报错，只会让成片悄悄跑偏）-----------
# 这些不是"风格"而是 H3 的硬绑定：<Subject N>/<Picture N> 决定角色与参考图
# 怎么绑，时间码决定镜头节奏与配乐窗口。翻译节点可以改词，但不能把它们改没。
# ★ 2026-09-26 回归修复：v9「删繁就简」时把这一组正则连同校验一起删掉了，
#   实测因此漏检了配乐条目 `00:12.0-00:17.0` 被模型合并成 `…);00:17.0`
#   （区间起点被吃掉）—— 二次翻译兜底没有机会介入。
_FACT_TAGS = (
    ("<Subject N>", re.compile(r"<\s*Subject\s+\d+\s*>", re.I)),
    ("<Picture N>", re.compile(r"<\s*Picture\s+\d+\s*>", re.I)),
    ("<Video N>", re.compile(r"<\s*Video\s+\d+\s*>", re.I)),
    ("<Audio N>", re.compile(r"<\s*Audio\s+\d+\s*>", re.I)),
    ("[Shot N]", re.compile(r"\[\s*Shot\s+\d+\s*\]", re.I)),
    ("(Sx)", re.compile(r"\(\s*S\d+\s*\)")),
)
_CLOCK = r"\d{1,2}:[0-5]\d(?:\.\d{1,3})?"
# ★ 尾部前瞻只排「数字 / 冒号」，**不能把句号排掉** —— `at 00:12.0.` 这种
#   「时间码收在句末」极常见，带上 `.` 会让整个时间码匹配不上（v8 的老缺陷，
#   离线用例抓出来的）。
_CLOCK_RE = re.compile(r"(?<![\d:.])" + _CLOCK + r"(?![\d:])")
# 区间时间码 `A-B`。配乐表按区间写（`00:12.0-00:17.0 三弦接棒…`），模型很
# 容易把两条合并成一条并吃掉区间起点 —— 那时**单个时间码的集合可能没变**
# （起点在别处也出现过），只有比对「区间对」才看得出来。
_CLOCK_RANGE_RE = re.compile(
    r"(?<![\d:.])(" + _CLOCK + r")\s*[-–—~至到]\s*(" + _CLOCK + r")(?![\d:])")

_FULLWIDTH_MAP = str.maketrans({
    **{chr(0xFF01 + i): chr(0x21 + i) for i in range(0x5E)},
    "\u3000": " ",
    "\u3001": ", ", "\u3002": ". ",
    "\uff0c": ", ", "\uff1a": ": ", "\uff1b": "; ",
    "\uff01": "! ", "\uff1f": "? ",
})


def normalize_punct(text):
    """全角标点 → ASCII。台词 `<d>…</d>` 逐字不动（此时台词已是 [[Dn]]，更不会被碰）。"""
    src = str(text or "")
    out = []
    pos = 0
    for m in _DIALOG_RE.finditer(src):
        out.append(src[pos:m.start()].translate(_FULLWIDTH_MAP))
        out.append(m.group(0))
        pos = m.end()
    out.append(src[pos:].translate(_FULLWIDTH_MAP))
    res = "".join(out)
    res = re.sub(r"[ \t]+([,;:.!?])", r"\1", res)
    res = re.sub(r"([,;!?])(?=[A-Za-z])", r"\1 ", res)
    return res


def _norm_dialogue(raw):
    """把一句台词规范化成 `<d>[语言] 原文</d>`（语言标签缺则补 [Chinese]）。

    这是 v9「缺一不可」里 <d>块 + 语言标签 + 原文 三要素的代码层保证：
    剥掉 <d> 外壳 / 引号 / 既有 [语言] 标签，保留台词正文；缺标签时补 Chinese。
    """
    s = str(raw or "").strip()
    m = re.match(r"<d\b[^>]*>([\s\S]*)</d>\s*$", s, re.I)
    inner = m.group(1) if m else s
    inner = inner.strip()
    # 剥行首的分隔符/空白（标签行「台词 | …」里的 | 之类）
    inner = re.sub(r"^[|｜:：\s]+", "", inner)
    # 剥包裹引号（「」『』“”‘' " '）
    if len(inner) >= 2 and inner[0] in "「『“‘\"'" and inner[-1] in "」』”’\"'":
        inner = inner[1:-1].strip()
    lm = _LANG_TAG_RE.match(inner)
    if lm:
        lang, text = lm.group(1), lm.group(2).strip()
    else:
        lang, text = "Chinese", inner
    return "<d>[%s] %s</d>" % (lang, text)


def protect_dialogue(body):
    """台词 → `[[Dn]]` 占位符；store 里按序存**规范化后的台词**（`<d>[语言] 原文</d>`）。

    顺序：先 <d> 块（含 (Sx) 前缀一并纳入），再标签行，再引号对白。
    每种都经 _norm_dialogue 规范化，还原后逐字填回 —— 台词四要素缺一不可由代码保证。
    """
    store = []

    def _add(text):
        store.append(text)
        return "[[D%d]]" % len(store)

    # ① (Sx) 前缀 + <d> 块 一并保护（还原后自动带出说话人编号）
    def _speaker_d(m):
        return _add((m.group(1) + " " if m.group(1) else "") + _norm_dialogue(m.group(2)))

    out = _SPEAKER_DIALOG_RE.sub(_speaker_d, body)

    # ② 裸 <d>…</d>（没被 ① 吃掉、不带 (Sx) 前缀的）
    out = _DIALOG_RE.sub(lambda m: _add(_norm_dialogue(m.group(0))), out)

    # ③ 台词标签行（标签 + 内容 → 保留标签，内容规范化）
    def _label(m):
        t = m.group("t").strip()
        if _CJK_RE.search(t):
            return _add(_norm_dialogue(t))
        return m.group(0)

    out = _LABEL_DIALOG_RE.sub(_label, out)

    # ④ 引号中文对白（统一规范化成 <d> 块，不再保留外层引号）
    def _quote(m):
        if _CJK_RE.search(m.group(1)):
            return _add(_norm_dialogue(m.group(1)))
        return m.group(0)

    out = _QUOTED_CJK_RE.sub(_quote, out)
    return out, store


def restore_dialogue(text, store):
    """占位符逐字填回。多轮替换：外层占位符还原后可能带出内层占位符（嵌套）。"""
    text = str(text or "")
    for _ in range(4):
        if not _PLACEHOLDER_RE.search(text):
            break

        def _sub(m):
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(store):
                return store[idx]
            return m.group(0)

        text = _PLACEHOLDER_RE.sub(_sub, text)
    missing = [int(x) for x in _PLACEHOLDER_RE.findall(text)]
    return text, missing


# 连续中文片段：以 CJK 起、CJK + 中文标点续（不含拉丁字母），整句/整段抓出来。
_CJK_RUN_RE = re.compile(
    r"[\u4e00-\u9fff][\u4e00-\u9fff，。、！？；：（）()「」『』“”‘’—…·《》【】,.;:\s]*")


def residual_cjk_segments(out_text, store):
    """自查③（增强）：把还原后、去掉受保护台词内容之外的**连续中文片段**整句抓出。

    用于二次翻译时把「具体哪段没翻」定向喂给模型（喂片段而不是只说"你失败了"）。
    去重 + 截断，避免把整篇塞回去。"""
    cleaned = str(out_text or "")
    for d in store:
        if d:
            cleaned = cleaned.replace(d, "")
    # ★ 结构行占位符（若还原前）也不算残留。
    cleaned = _STRUCT_PLACEHOLDER_RE.sub("", cleaned)
    seen = []
    for run in _CJK_RUN_RE.findall(cleaned):
        run = run.strip(" \t，。、！？；：")
        if not run:
            continue
        if run not in seen:
            seen.append(run)
    return seen[:40]


# ★ 台词块之外的「中文剥离」实现放在 ``prompt_pack``（零外部依赖），
#   解析节点也要用同一份 —— 两边各写一份就会漂移。见 prompt_pack.strip_residual_cjk。
strip_residual_cjk = pp.strip_residual_cjk

# 六个官方字段名（用于「粘行」断开；字段名清单也从 prompt_pack 取，不另立一份）。
_FIELD_HEAD_RE = re.compile(
    r"(?:^|(?<=[.。!！?？]))[ \t]*(?=(?:"
    + "|".join(re.escape(x) for x in pp.FIELD_ORDER) + r")\s*[:：])", re.I)


def ensure_field_lines(text):
    """保证六个官方字段名各自独占行首。

    ★ 2026-09-27：模型很爱把 ``retention_analysis:`` 粘在上一段末尾（实测
      ``…natural movement.retention_analysis:``）。下游 ``prompt_pack`` 的
      ``_match_field`` 是**行首锚定**的，粘住的字段名一律认不出来 ——
      那一整段 retention_analysis 会被并进 summary，``_seg_retention`` 只能
      退回兜底文案，而兜底会把 **Subject 号当 Picture 号**用
      （``<Subject 3>`` 绑的是 ``<Picture 4>``，兜底却写 ``<Picture 3>`` =
      场景图）→ **角色绑错参考图**。

      这里只加换行、不改任何字符；下游 ``_split_glued_fields`` 也有一份等价
      兜底，两层都做是因为旧 PACK / 手改 PACK 不经过本节点。
    """
    out = []
    for line in str(text or "").splitlines():
        pos = 0
        for m in _FIELD_HEAD_RE.finditer(line):
            if m.start() <= pos:
                continue
            out.append(line[pos:m.start()])
            pos = m.start()
        out.append(line[pos:])
    return "\n".join(out)


def validate(src_protected, out_text, visible):
    """自查①②的轻量部分：非空 + 占位符数量对齐（visible = 送模型前占位符数）。

    ★ 只数**台词占位符** ``[[Dn]]``：结构行占位符 ``[[Sn]]`` 是另一套编号，
      两者不能混着数（``[[S1]]`` 会被 ``\\[\\[D(\\d+)\\]\\]`` 漏掉，混数反而错）。
    """
    if not str(out_text or "").strip():
        return False, "输出为空"
    got_n = len(_PLACEHOLDER_RE.findall(out_text))
    if got_n != visible:
        return False, "台词占位符数量不符（期望 %d，实际 %d）" % (visible, got_n)
    if len(out_text) < 0.2 * max(1, len(src_protected)):
        return False, "输出过短"
    # ★ 查漏补缺②：未编号 / 字面占位符泄漏（模型把 [[Dn]] 这种当成文本写了进去）
    leak = _UNNUMBERED_PH_RE.findall(out_text)
    if leak:
        return False, "占位符泄漏 %s（未编号的字面 [[Dn]]，无法还原台词）" % leak
    return True, ""


def _clock_ms(text):
    """00:12.0 / 00:12.00 / 00:12.000 → 同一毫秒整数。

    归一化后再比，`00:12.0` 与 `00:12.000` 视为同一时刻 —— 避免格式变化
    误报，真正丢失的才拦下来。
    """
    mm, rest = str(text).split(":")
    return int(round((int(mm) * 60 + float(rest)) * 1000))


def _fmt_clock(ms):
    return "%02d:%06.3f" % (ms // 60000, (ms % 60000) / 1000.0)


def lost_facts(src_protected, out_restored):
    """译文相对原文丢掉了哪些事实项。空列表 = 全部保住。

    只查**丢失**（``want - got``）：模型按上下文**新增** ``(Sx)`` 是设计要求
    （源稿常常没标说话人），新增不算错。台词块内容先剥掉 —— 它是逐字还原的，
    里面的中文/时间码不参与比对。
    """
    src = _DIALOG_RE.sub("", str(src_protected or ""))
    out = _DIALOG_RE.sub("", str(out_restored or ""))
    lost = []
    for label, pat in _FACT_TAGS:
        want = {m.group(0).replace(" ", "").lower() for m in pat.finditer(src)}
        got = {m.group(0).replace(" ", "").lower() for m in pat.finditer(out)}
        miss = sorted(want - got)
        if miss:
            lost.append("%s 丢失 %s" % (label, miss[:6]))
    want_tc = {_clock_ms(m.group(0)) for m in _CLOCK_RE.finditer(src)}
    got_tc = {_clock_ms(m.group(0)) for m in _CLOCK_RE.finditer(out)}
    miss_tc = sorted(want_tc - got_tc)
    if miss_tc:
        lost.append("时间码丢失 %s" % [_fmt_clock(t) for t in miss_tc[:6]])
    want_rg = {(_clock_ms(m.group(1)), _clock_ms(m.group(2)))
               for m in _CLOCK_RANGE_RE.finditer(src)}
    got_rg = {(_clock_ms(m.group(1)), _clock_ms(m.group(2)))
              for m in _CLOCK_RANGE_RE.finditer(out)}
    miss_rg = sorted(want_rg - got_rg)
    if miss_rg:
        lost.append("时间码区间丢失 %s" % [
            "%s-%s" % (_fmt_clock(a), _fmt_clock(b)) for a, b in miss_rg[:6]])
    return lost


def split_parts(text):
    """按空行切段，拼回去必须等于原文。"""
    raw = str(text or "")
    if not raw:
        return []
    chunks = re.split(r"(\n[ \t]*\n)", raw)
    parts = []
    for i in range(0, len(chunks), 2):
        body = chunks[i]
        sep = chunks[i + 1] if i + 1 < len(chunks) else ""
        if body == "" and sep:
            if parts:
                parts[-1] = (parts[-1][0], parts[-1][1] + sep)
            else:
                parts.append(("", sep))
            continue
        parts.append((body, sep))
    return parts


def _last_boundary(text, lo, hi):
    """在 ``text[lo:hi]`` 里找最靠后的**语义边界**，返回切点索引（无则 -1）。

    偏好顺序（越靠前越优先，同一级取最靠后者）：
      ① 行尾（``\\n`` 之后）
      ② 句末标点（``. ! ?`` + 空白 / 引号 / 行尾）
      ③ 分句标点（``; ,``）
      ④ 空白
    这样切出来的每一块都尽量落在句子/分句边界上，而不是把
    ``...layer or fastening that <Picture 2> does not`` 这种半句话劈开 ——
    半句话送模型容易被续写跑偏，也是「译文事实项对不上」的入口。
    """
    seg = text[lo:hi]
    # ① 换行
    p = seg.rfind("\n")
    if p > 0:
        return lo + p + 1
    # ② 句末（. ! ? 后跟空白/引号/结尾）
    best = -1
    for m in re.finditer(r"[.!?…](?=[\s\"'”’)]|$)", seg):
        best = max(best, m.end())
    if best > 0:
        return lo + best
    # ③ 分句（; ,）
    for m in re.finditer(r"[;,](?=\s)", seg):
        best = max(best, m.end())
    if best > 0:
        return lo + best
    # ④ 空白
    p = max(seg.rfind(" "), seg.rfind("\t"))
    if p > 0:
        return lo + p + 1
    return -1


def split_oversized(body, limit):
    """超长 body 按**语义边界**切成 ≤ limit 的小块。

    ★ 2026-09-27 改：旧实现先按时间码切，再对仍超限的片段直接
      ``buf[:limit]`` 硬切 —— 实测把
      ``...any part, layer or fastening that <Picture 2> does not`` 劈成两半，
      半句话送模型等于逼它续写，译文事实项极易错位。现在：
        ① 先按时间码切（时间码是天然的镜头边界）；
        ② 仍超限的片段按 ``_last_boundary`` 找最近的语义边界切；
        ③ 找不到任何边界才退回按字符切。
      切点只在**空白/标点之后**，因此 ``<Picture 2>`` / ``<Subject 1>``
      这类带空格的标签不会被劈开。

    ★ 守恒承诺：``"".join(p for p, _ in split_oversized(x, n)) == x``。
      切分**只切、不丢** —— 边界处的空白随**前**一块一起带走（切点取在
      空白**之后**），所以拼回逐字等于原文，``split_parts`` 的「拼回 = 原文」
      契约在 expanded 之后依然成立。
    """
    if len(body) <= limit:
        return [(body, "")]
    out = []
    buf = ""
    # ★ 时间码边界：把紧邻时间码前的 `At ` / `at ` / `@ ` 一并算进**下一块**
    #   （用后顾断言吃掉它），否则 `…locked in a kiss. At ` 的 `At ` 会被留在
    #   前一块尾巴上，凑成 len≈3 的碎片，模型拿到一句「At 」无从下笔。
    pieces = re.split(
        r"(?=(?:(?<=At )|(?<=at )|(?<=@ ))?"
        r"(?<![\d:.])\d{1,2}:[0-5]\d(?:\.\d{1,3})?(?![\d:.]))", body)
    for piece in pieces:
        if buf and len(buf) + len(piece) > limit and len(buf.strip()) >= limit // 3:
            out.append((buf, ""))
            buf = piece
        else:
            buf += piece
        while len(buf) > limit:
            # 边界只找在 [limit//2, limit] 区间内，避免切出过短的头块。
            cut = _last_boundary(buf, max(1, limit // 2), limit)
            if cut <= 0:
                cut = limit          # 无边界 → 只能按字符切
            out.append((buf[:cut], ""))     # ★ 不 rstrip：空白归前一块，守恒
            buf = buf[cut:]                 # ★ 不 lstrip：同上
    if buf:
        out.append((buf, ""))
    return out


def _strip_output(text):
    """去掉 markdown 围栏 + 模型误包的 JSON 数组/对象外壳（如 [ "…", "…" ] / { "d":"…" }）。

    台词此时已是占位符 `[[Dn]]`，剥外壳不会伤到台词内容。"""
    out = str(text or "")
    fence = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?([\s\S]*?)\n?```", out)
    if fence:
        out = fence.group(1)
    out = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n?", "", out)
    out = re.sub(r"\n?```\s*$", "", out)
    out = out.strip()
    # 剥 JSON 数组外壳：[ "a", "b" ] -> a\nb
    while out.startswith("[") and out.endswith("]"):
        inner = out[1:-1].strip()
        # 只有内层是逗号分隔的字符串项才剥，避免误伤正文里的方括号
        items = re.findall(r'"(?:[^"\\]|\\.)*"', inner)
        if len(items) >= 1 and sum(len(x) for x in items) > 0.6 * len(inner):
            joined = "\n".join(x[1:-1] for x in items)
            out = joined
            continue
        break
    # 剥单个 JSON 对象外壳：{ "d": "…" } -> …
    m = re.match(r'^\{\s*"[A-Za-z_]+"\s*:\s*"([\s\S]*)"?\s*\}$', out)
    if m:
        out = m.group(1)
    return out.strip()


# ---------------------------------------------------------------------------
# ★ 2026-09-27：PACK 结构行豁免（banner / 双语气泡标签）
# ---------------------------------------------------------------------------
# PACK 的头部、`ENGLISH VERSION / 英文版`、`END OF PACK / 文件结束` 这些行的中文
# **不是待译文本，而是格式的一部分**：官方 PACK 模板就是「英文 / 中文」双语气泡
# 写法，中文半边是给人看的锚点。
#
# 旧行为把它们当普通正文送模型，模型翻不动（`Project / 项目` 本来就该保留
# `项目`）→ 自查报「残留中文」→ 三级兜底全失败 → 最后 `strip_residual_cjk`
# 把中文半边**逐字符剁掉**，实测产出：
#     `Project / 项目         : 曹贼的性价比`  →  `Project /`
#     `MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包` → `MiniMax H3 SHOT PROMPT PACK / H3`
#     `END OF PACK / 文件结束` → `END OF PACK /`
# 一次翻出 11 处结构损伤，而且每次改稿复跑还会再剁一次。
#
# 判据（逐行，命中即整行豁免，不进模型也不剥离）：
#   ① 分隔线：整行只有 `=` / `-` / `#` / `*` / `~` 等符号；
#   ② 双语气泡标签对：`<ASCII 标签> / <中文标签>` —— 斜杠右侧是中文、
#      左侧不含 CJK，且该中文段**短**（≤ 16 字，是标签不是句子）；
#   ③ 段头/文件尾标记：`###### … ######` / `END OF PACK` 这类整行包装。
_BANNER_RULE_RE = re.compile(r"^\s*([=\-#*~_=·—]{3,})\s*$")
# `Label / 中文` 三态：
#   ① 整行就是气泡标签：`ENGLISH VERSION / 英文版（…）`
#   ② 标签 + 冒号 + 值：`Project / 项目         : 曹贼的性价比`
#   ③ 标签里还夹着英文值：`MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包`
# 统一写法：**行内最后一个 `/` 之后**那段以 CJK 起头（允许前面挂 ≤ 4 个 ASCII
# 字符，如 `H3 `），且该段 CJK 是短标签（≤ _STRUCT_CJK_MAX 且无句读）即豁免。
# 用「最后一个斜杠」是因为 `Label / 中文` 的中文气泡总在行尾。
_BILINGUAL_LABEL_RE = re.compile(
    r"^([\x20-\x7e]{2,72}?)\s*/\s*((?:[\x20-\x7e]{0,4}?)?[\u4e00-\u9fff][^\n]*?)\s*$")
# 整行被符号包起来的标题，如 `########## S01 / 10.6s / EN ##########`。
_WRAPPED_TITLE_RE = re.compile(r"^\s*([#=]{3,})\s*(.*?)\s*\1\s*$")
# 结构行里允许出现的「中文包装符号」（全角括号、顿号、中点）——判定"短标签"用。
_STRUCT_CJK_MAX = 16
# 真·正文句读：出现即判为句子而非标签（用于拒绝豁免）。
_SENTENCE_PUNCT_RE = re.compile(r"[，。！？；：]|(?<=[\u4e00-\u9fff])[,.!?;]")


def _line_is_structural(line):
    """这一行是不是 PACK 结构行（中文属于格式，不该翻译、不该剥离）？"""
    s = str(line or "")
    if not s.strip():
        return False
    if _BANNER_RULE_RE.match(s):
        return True
    m = _BILINGUAL_LABEL_RE.match(s.strip())
    if m:
        # 右半段：CJK 个数受限 且 不是整句中文 → 是标签，不是待译正文。
        right = m.group(2)
        cjk = _CJK_RE.findall(right)
        if cjk and len(cjk) <= _STRUCT_CJK_MAX and not _SENTENCE_PUNCT_RE.search(right):
            return True
    m = _WRAPPED_TITLE_RE.match(s)
    if m:
        inner = m.group(2)
        cjk = _CJK_RE.findall(inner)
        if not cjk:
            return True
        if len(cjk) <= _STRUCT_CJK_MAX and not _SENTENCE_PUNCT_RE.search(inner):
            return True
    return False


def protect_structure(body):
    """把 PACK 结构行替换成占位符 ``[[S…]]``，返回 ``(protected, store)``。

    与 ``protect_dialogue`` 同构：送模型前把**格式自带的**中文行挖走，还原时
    逐字填回。这样结构行的中文既不会被要求翻译，也不会被 ``strip_residual_cjk``
    剁掉 —— 「英文 / 中文」气泡保持原样。

    ★ 与 ``protect_dialogue`` 的分工：台词是 ``[[Dn]]``，结构行是 ``[[S…]]``
      （编号用 ``[[S1]]`` 形式，不会与台词占位符冲突）。
    """
    store = []
    text = str(body or "")
    if not text:
        return text, store
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if _line_is_structural(ln) and _CJK_RE.search(ln):
            store.append(ln)
            lines[i] = "[[S%d]]" % len(store)
    return "\n".join(lines), store


def restore_structure(text, store):
    """把结构行占位符填回本行原文。返回 ``(restored, missing)``。"""
    out = str(text or "")
    missing = []
    for idx, line in enumerate(store or [], 1):
        ph = "[[S%d]]" % idx
        if ph in out:
            out = out.replace(ph, line)
        else:
            missing.append(ph)
    return out, missing


_STRUCT_PLACEHOLDER_RE = re.compile(r"\[\[S(\d+)\]\]")


def _needs_translation(body):
    """台词块 / 结构行之外还有没有中文？"""
    body = protect_structure(body)[0]
    return bool(_CJK_RE.search(_DIALOG_RE.sub("", body or "")))


# ---------------------------------------------------------------------------
# ★ v7 提示词：除台词外**不得残留任何中文**（镜头/音乐/声效/风格总纲/节奏说明全译），
#   并禁止「以下中文保留原文」元注释。台词铁律（沿用原文、不翻译）保持不变。
#   模型若懒翻译漏翻，代码层自查抓到残留片段 → 二次翻译时把未翻片段逐条喂回去定向补翻。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ★ v9 提示词（删繁就简）：台词规则收敛成一条，其余条款全砍。
#   「缺一不可」四要素中 <d>块/[语言]/原文 由代码层 _norm_dialogue 保证，
#   (Sx) 编号靠提示词要求模型按上下文补（逐块无法可靠自动识别全局说话人）。
# ---------------------------------------------------------------------------
TRANSLATE_PROMPT = """\
把下面的剧本片段翻译成**纯英文**。除台词外，最终不得残留任何中文字符。

# 台词（沿用原文，不翻译）
每句台词固定：`(Sx) <d>[Chinese] 原文</d>` = 说话人编号 (Sx) + <d>块 + [Chinese] 标签 + 原文，缺一不可。
原文里的台词已替换成占位符 `[[D1]]` `[[D2]]` …，输出时把每个占位符**原样放回**对应位置，不翻译、不遗漏、不重复，也不要写出 `[[Dn]]` 这种未编号的占位符。

# 输出
只输出译文本身，按原段落顺序；不要解释，不要 markdown / JSON 围栏。
"""

# 三级兜底（逐行救捞）提示词：二翻仍失败时，把该块里含中文的**每一行**单独发给模型翻译
# 注意：{line} 是代码用 .format() 填充的字段。
LINE_SALVAGE_PROMPT = """\
把下面**这一行**译成**纯英文**。
台词占位符 `[[Dn]]` 原样保留、不翻译；除占位符外，这行里不得残留任何中文字符。
只输出译出的这一行本身，不要解释、不要围栏。

# 待译行
{line}
"""

# 二次翻译提示词（自查不过时携原文 + 原因 + 未翻片段 + 上次译文重发）
# 注意：{reason}/{leftover}/{source}/{bad} 是代码用 .format() 填充的字段，必须保留。
RETRY_PROMPT = """\
你上次的输出没通过校验（原因：{reason}），请重新翻译并修正。
- 台词 `<d>[Chinese] 原文</d>`（含 `[[Dn]]` 占位符对应原文）沿用原文，不翻译；占位符逐个原样保留、顺序不变、不增不减。
- 除台词外的**所有**中文必须译成英文，**最终除台词外不得残留任何中文字符**。
只输出译文本身，不要围栏，不要写「以下中文保留原文」之类的元注释。

# 上次未译成英文的中文片段（请逐条译出，对应回原文位置）
{leftover}

# 原文（台词已用占位符保护）
{source}

# 你上次的错误输出
{bad}
"""


# ---------------------------------------------------------------------------
# CLIP（本地模式）调用 —— 保留原版签名兼容逻辑
# ---------------------------------------------------------------------------
def _accepted_kwargs(fn, wanted):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(wanted)
    params = set(sig.parameters)
    params.discard("self")
    params.discard("cls")
    return {k: v for k, v in wanted.items() if k in params}


def _tokenize(clip, prompt, thinking):
    try:
        return clip.tokenize(prompt)
    except Exception:
        pass
    extra = {"skip_template": False, "min_length": 1, "thinking": bool(thinking)}
    try:
        return clip.tokenize(prompt, **_accepted_kwargs(clip.tokenize, extra))
    except Exception:
        pass
    for k, v in extra.items():
        try:
            return clip.tokenize(prompt, **{k: v})
        except Exception:
            continue
    raise RuntimeError("clip.tokenize 调用失败：该 CLIP 不支持文本生成。")


def _generate_ids(clip, tokens, max_length, seed, thinking, temperature):
    wanted = {
        "do_sample": True, "max_length": int(max_length),
        "temperature": float(temperature), "top_k": 64, "top_p": 0.95,
        "seed": int(seed), "thinking": bool(thinking),
    }
    kwargs = _accepted_kwargs(clip.generate, wanted)
    ids = clip.generate(tokens, **kwargs)
    return _strip_output(clip.decode(ids))


def _clip_generate(clip, prompt, max_length, seed, thinking, temperature):
    tokens = _tokenize(clip, prompt, thinking)
    return _generate_ids(clip, tokens, max_length, seed, thinking, temperature)


def _probe_generate(clip, seed=0):
    try:
        _clip_generate(clip, "Reply with the single word: OK.", 16,
                       int(seed), False, 0.0)
    except Exception as exc:
        raise RuntimeError(
            "①-A1 接进来的 CLIP 不支持文本生成（%s）。请把 CLIPLoader 的 "
            "type 改成 boogu 或 ideogram4，或把「模型来源」切到 openai_api。" % exc)


# ---------------------------------------------------------------------------
# API（openai_api 模式）调用 —— 429 限流退避（原版）+ 5xx/网络抖动退避（2026-09-29）
#   实测中转站会间歇性返回 500「请求上游失败，请稍后重试」，网络层也会偶发
#   WinError 121 信号灯超时 —— 这些都是瞬时的，不重试就把整块翻译废掉。
# ---------------------------------------------------------------------------
def _api_generate(base_url, model, api_key, prompt, max_length, temperature, timeout,
                  rate_controller=None):
    import requests
    if rate_controller is not None:
        rate_controller.wait()
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        raise RuntimeError("API 模式需要填写「API Base URL」。")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": str(model or "").strip(),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_length),
    }
    headers = {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    }
    max_429_retries = 6
    max_5xx_retries = 3
    n_429 = n_5xx = 0
    last_status, last_text = 0, ""
    while True:
        try:
            resp = requests.post(url, json=payload, headers=headers,
                                 timeout=float(timeout))
        except Exception as exc:
            if n_5xx < max_5xx_retries:
                n_5xx += 1
                time.sleep(min(float(2 ** n_5xx) * 3.0, 60.0))
                continue
            raise RuntimeError("API 请求失败（%s）：%s" % (url, exc))
        last_status = resp.status_code
        last_text = (resp.text or "")[:500]
        if resp.status_code == 200:
            try:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
            except Exception:
                raise RuntimeError("API 返回结构异常：%s" % last_text)
            if not content or not str(content).strip():
                raise RuntimeError("API 返回空内容。")
            return _strip_output(content)
        if resp.status_code == 429 and n_429 < max_429_retries:
            n_429 += 1
            retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            try:
                wait = float(retry_after)
            except (TypeError, ValueError):
                wait = float(2 ** n_429) * 3.0
            time.sleep(min(wait, 60.0))
            continue
        if resp.status_code >= 500 and n_5xx < max_5xx_retries:
            n_5xx += 1
            time.sleep(min(float(2 ** n_5xx) * 3.0, 60.0))
            continue
        raise RuntimeError("API 返回 HTTP %d：%s" % (last_status, last_text))


def _check_interrupt():
    """生成调用之间的取消检查（借鉴 xiaolibai-sys/ComfyUI-MiniMaxH3 的做法）。

    本节点一次跑几十块、每块最多 3 次调用再加限流间隔，全程**不进** ComfyUI
    的采样器 —— 核心的中断检测只在节点边界生效；不在这里主动查，取消按钮要
    等整份剧本翻完才起作用。
    """
    try:
        from comfy import model_management as _mm
    except ImportError:                     # pragma: no cover - 老核心
        return
    _mm.throw_exception_if_processing_interrupted()


class _RateController:
    """API 调用节流：相邻两次请求至少间隔 interval 秒。"""

    def __init__(self, interval=5.0):
        try:
            self.interval = max(0.0, float(interval or 0.0))
        except (TypeError, ValueError):
            self.interval = 5.0
        self._last_call = None

    def wait(self):
        if self.interval <= 0:
            self._last_call = time.monotonic()
            return
        now = time.monotonic()
        if self._last_call is not None:
            remaining = self.interval - (now - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


# 批量分隔符
_BATCH_SEP = "\n<<<H3SEG_BOUNDARY>>>\n"
_BATCH_PROMPT_NOTE = (
    "\n\n# 批量说明\n"
    "下面一次性给了 {n} 个独立片段，用这个标记隔开：\n"
    "%s\n"
    "这些片段互不相关、各自独立翻译。输出时**原样保留同一个标记**，"
    "在每两段译文之间各插入一次，总数必须还是 {n} 段、{n_sep} 个标记。"
) % _BATCH_SEP.strip("\n")


# ---------------------------------------------------------------------------
# 节点 —— ★ schema 与原版完全一致（参数名 / 顺序 / 默认值不变）
# ---------------------------------------------------------------------------
class H3ScriptTranslate(io.ComfyNode):
    """完整剧本 / 整段式分镜脚本 → 英文单版。v9：台词固定
    `(Sx) <d>[Chinese] 原文</d>`（编号+块+语言标签+原文，缺一不可）；
    <d>块/[语言]/原文 由代码层 _norm_dialogue 规范化保证，(Sx) 编号由模型补；
    提示词删繁就简；保留三级兜底 = 译后自查 → 定向二次补翻 → 逐行救捞。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ScriptTranslate",
            display_name="H3 剧本英译（中文 → 英文单版）",
            category=CATEGORY,
            description=(
                "把剧本改写成英文单版（v9：台词固定 (Sx) <d>[Chinese] 原文</d> 不翻译，"
                "编号+块+语言标签+原文缺一不可；除台词外全译；提示词删繁就简；"
                "保留三级兜底 = 译后自查 → 定向二次补翻 → 逐行救捞，仍不过保留原文并写进报告。"
            ),
            inputs=[
                io.Clip.Input("clip", display_name="文本模型（本地模式）",
                              optional=True,
                              tooltip="本地模式要接支持文本生成的 CLIP"
                                      "（qwen3vl_8b + type=boogu）。"
                                      "openai_api 模式可不接。"),
                io.String.Input("text", display_name="剧本原文",
                                multiline=True, default="", force_input=True,
                                optional=True,
                                tooltip="接 ①-A 的 PACK 文本。留空则原样透传。"),
                io.Boolean.Input("enabled", display_name="启用英译", default=True,
                                 tooltip="关掉 = 原样透传（调试用）。"),
                io.Combo.Input("model_source", options=["local", "openai_api"],
                               display_name="模型来源", default="local",
                               tooltip="local = 走 CLIP 本地生成；"
                                       "openai_api = 走 OpenAI 兼容 API。"),
                io.Boolean.Input("thinking", display_name="深度思考", default=False,
                                 advanced=True,
                                 tooltip="本地模式、且模型支持才有效；翻译通常关掉更快。"),
                io.Int.Input("chunk_chars", display_name="每块字符数",
                             default=900, min=200, max=6000, step=100,
                             advanced=True,
                             tooltip="一次送模型的正文字符上限。"),
                io.Int.Input("max_length", display_name="最大生成长度",
                             default=3072, min=256, max=32768, step=256,
                             advanced=True,
                             tooltip="本地模式 = CLIP 的 max_length；"
                                     "openai_api 模式 = API 的 max_tokens。"),
                io.Float.Input("temperature", display_name="温度", default=0.3,
                               min=0.0, max=2.0, step=0.05, advanced=True),
                io.Int.Input("seed", display_name="种子", default=0, min=0,
                             max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True, advanced=True),
                io.String.Input("api_base_url", display_name="API Base URL",
                                default="", advanced=True,
                                tooltip="OpenAI 兼容端点。"),
                io.String.Input("api_model", display_name="API 模型名",
                                default="gpt-4o", advanced=True),
                io.String.Input("api_key", display_name="API Key（直接填）",
                                default="", advanced=True,
                                tooltip="直接贴 Key（会写进工作流 JSON，别分享给别人）。"
                                        "留空则从环境变量读。"),
                io.String.Input("api_key_env", display_name="API Key 环境变量名（备选）",
                                default="AGNES_API_KEY", advanced=True),
                io.Float.Input("api_timeout", display_name="API 超时(秒)",
                               default=60.0, min=1.0, max=600.0, step=5.0,
                               advanced=True),
                io.Int.Input("api_interval", display_name="API 节流间隔(秒)",
                             default=5, min=1, max=30, step=1, advanced=True,
                             tooltip="免费版建议 ≥5 秒；429 会额外退避。"),
                io.Int.Input("api_batch_chars", display_name="批量合并字数上限",
                             default=3000, min=0, max=8000, step=100, advanced=True,
                             tooltip="★ 提速：相邻块合进一次 API 调用。"
                                     "0 = 关闭批量。默认 3000。"),
                io.String.Input("extra_rules", display_name="追加规则",
                                multiline=True, default="", advanced=True,
                                tooltip="追加到系统提示词末尾的补充要求。"),
            ],
            outputs=[
                io.String.Output("text", display_name="英文单版文本"),
                io.String.Output("report", display_name="英译报告"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """缓存键 = 「会影响译文的那几个输入的摘要」。

        ★ 原来是 `return float("nan")` —— 恒不命中，**每次排队都重翻**。
          对中文剧本喂 H3 来说，这个代价不是几秒钟：实测 2026-09-27 同一份
          剧本、同一个工作流复跑，温度 0.3 的远端模型给出的英文**每次都不
          完全一样**（22935 → 22513 字符），于是下游 segment_key 里的 prompt
          摘要全变 → 4 段全部重渲 → 白烧 60 分钟。用户什么都没改。

          改成「按内容摘要」保留了 nan 的原本意图（剧本或任何翻译参数一变就
          重翻），只是把「没变也重翻」这个副作用消掉：
            · 剧本 / 模型来源 / 模型名 / 端点 / 分块 / 批量 / 温度 / 种子 /
              是否深度思考 / 追加规则 → 进摘要；
            · api_key / api_key_env / api_timeout / api_interval → **不进**摘要
              （它们是身份与传输参数，不影响译文内容；把密钥摘要进缓存键也不合适）。

          代价：远端模型若在**参数与输入完全不变**的情况下给出不同译文，本次会
          复用上一次的译文。这正是我们要的 —— 链条要的是可复现，不是每次重新掷骰子。
        """
        import hashlib
        payload = "\u0000".join(str(x) for x in (
            kwargs.get("text", ""), kwargs.get("enabled", True),
            kwargs.get("model_source", ""), kwargs.get("api_base_url", ""),
            kwargs.get("api_model", ""), kwargs.get("chunk_chars", ""),
            kwargs.get("api_batch_chars", ""), kwargs.get("max_length", ""),
            kwargs.get("temperature", ""), kwargs.get("seed", ""),
            kwargs.get("thinking", ""), kwargs.get("extra_rules", "")))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def execute(cls, clip=None, text="", enabled=True, model_source="local",
                thinking=False, chunk_chars=900, max_length=3072,
                temperature=0.3, seed=0, api_base_url="", api_model="gpt-4o",
                api_key="", api_key_env="AGNES_API_KEY", api_timeout=60.0,
                api_interval=5, api_batch_chars=3000,
                extra_rules=""):
        raw = str(text or "")
        if not enabled:
            return io.NodeOutput(raw, "英译已关闭（启用英译=关）—— 原样透传。")
        if not raw.strip():
            return io.NodeOutput(raw, "剧本原文为空 —— 原样透传。")

        use_api = str(model_source or "local") == "openai_api"
        resolved_key = ""
        if use_api:
            api_base_url = str(api_base_url or "").strip()
            api_model = str(api_model or "").strip()
            resolved_key = str(api_key or "").strip()
            if not resolved_key:
                api_key_env = str(api_key_env or "").strip() or "AGNES_API_KEY"
                resolved_key = os.environ.get(api_key_env, "")
            if not api_base_url:
                raise RuntimeError(
                    "「模型来源」= openai_api 但「API Base URL」为空 —— 无法翻译。")
            if not api_model:
                raise RuntimeError(
                    "「模型来源」= openai_api 但「API 模型名」为空 —— 无法翻译。")
            if not resolved_key:
                raise RuntimeError(
                    "「模型来源」= openai_api 但需要 API Key —— 既没在"
                    "「API Key（直接填）」里填，环境变量 %s 也未设置。"
                    "ComfyUI Desktop 双击启动时请直接把 Key 填进节点。"
                    % api_key_env)
        else:
            if clip is None:
                raise RuntimeError(
                    "「模型来源」= local 但需要连接文本模型 CLIP"
                    "（qwen3vl_8b + type=boogu）。或把「模型来源」切到 openai_api。")

        parts = split_parts(raw)
        expanded = []
        for body, sep in parts:
            if len(body) > int(chunk_chars):
                sub = split_oversized(body, int(chunk_chars))
                for i, (piece, psep) in enumerate(sub):
                    expanded.append((piece, psep if i == len(sub) - 1 else ""))
            else:
                expanded.append((body, sep))

        system = TRANSLATE_PROMPT
        if str(extra_rules or "").strip():
            system = system + "\n# 追加规则\n" + str(extra_rules).strip() + "\n"

        out_parts = []
        report = []
        n_done = 0
        n_skip = 0
        n_fail = 0
        n_retry = 0
        n_salv = 0
        rate_controller = _RateController(interval=float(api_interval)) if use_api else None
        batch_budget = int(api_batch_chars or 0) if use_api else 0

        if not use_api and any(_needs_translation(b) for b, _ in expanded):
            _probe_generate(clip, seed)

        def _needs(idx):
            b, _sep = expanded[idx]
            return bool(b.strip()) and _needs_translation(b)

        def _call(prompt_text):
            """一次生成调用（API 或本地），异常原样抛出由调用方处理。"""
            _check_interrupt()
            if use_api:
                return _api_generate(api_base_url, api_model, resolved_key,
                                     prompt_text, max_length, temperature,
                                     api_timeout, rate_controller)
            return _clip_generate(clip, prompt_text, max_length,
                                  int(seed), thinking, temperature)

        def _check(protected, got, store, struct_store=None):
            """v9 自查：占位符序列 → 逐字还原台词 → 结构行还原 → 事实项保真 → 残留中文扫描。
            返回 (ok, reason, fixed_text, leftover_segments)。

            ★ 占位符要比**序列**不只是数量：数量对而顺序错（模型把 [[D2]] 写在
            [[D1]] 前面）时数量校验照样放行，还原出来的台词就张冠李戴。
            ★ 事实项（<Subject N>/<Picture N>/[Shot N]/(Sx)/时间码）只查丢失 ——
            它们被改没不会报错，只会让成片悄悄跑偏（角色绑错图、配乐窗口错位）。
            ★ 结构行占位符 ``[[Sn]]`` 在事实项比对**之前**先还原：结构行里的
            版本号（v20）、项目名（曹贼的性价比）是 PACK 头部的自带内容，不参与
            「译文丢了什么」的判定。
            """
            visible = len(_PLACEHOLDER_RE.findall(protected))
            ok, reason = validate(protected, got, visible)
            if not ok:
                return False, reason, None, []
            want_seq = ["[[D%d]]" % (i + 1) for i in range(len(store))]
            got_seq = ["[[D%s]]" % n for n in _PLACEHOLDER_RE.findall(got)]
            if got_seq != want_seq:
                return False, "台词占位符序列不符（期望 %s，实际 %s）" % (
                    want_seq[:8], got_seq[:8]), None, []
            fixed, missing = restore_dialogue(normalize_punct(got), store)
            if missing:
                return False, "台词占位符未全部还原（缺 %s）" % missing, None, []
            if struct_store:
                fixed, smissing = restore_structure(fixed, struct_store)
                if smissing:
                    return False, "结构行占位符未还原（缺 %s）" % smissing, None, []
            lost = lost_facts(protected, fixed)
            if lost:
                return False, "事实项丢失：%s" % "；".join(lost), fixed, []
            segs = residual_cjk_segments(fixed, store)
            if segs:
                return False, "残留中文 %d 处（%s…）" % (
                    len(segs), " / ".join(s[:14] for s in segs[:4])), fixed, segs
            return True, "", fixed, []

        def _line_salvage(protected, store, struct_store=None, rounds=3):
            """三级兜底：把 protected 里**含中文的每一行**单独发给模型逐行翻译，
            拼回后还原占位符。台词已是占位符，逐行译不会伤到台词。
            整块重发两次都救不回来（模型逐行偷懒）时，用这个逐行救。

            ★ 结构行占位符 ``[[Sn]]`` 不含中文（它是 ``[[S1]]``），所以逐行筛查
              不会把它们送去翻译 —— 结构行的中文由 ``protect_structure`` 保证。

            ★ 2026-09-27 改**多轮**：以前只跑一轮，只要有一行漏掉一两个字
              （实测块 20 的 `大锣"仓"砸在"棒"字上` 里剩了一个「棒」），整块
              就被判失败并**回退成中文原文** —— 201 个汉字直接进 H3 提示词。
              现在每轮只把**仍含中文的行**再送一次，最多 3 轮；轮内不动已干净
              的行，所以每轮都比上一轮更干净。

            返回 ``(ok, reason, fixed)``：``fixed`` 是**尽力而为**的英文
            （失败时也返回，交给调用方剥离残留中文后采纳），只有占位符还原
            失败才返回 ``None``。
            """
            rejoined = str(protected or "")
            for _round in range(max(1, int(rounds))):
                lines = rejoined.split("\n")
                new_lines = []
                touched = 0
                for ln in lines:
                    # 去掉占位符后这行还有没有中文？没有就不动
                    if not _CJK_RE.search(_PLACEHOLDER_RE.sub(
                            "", _STRUCT_PLACEHOLDER_RE.sub("", ln))):
                        new_lines.append(ln)
                        continue
                    try:
                        new_lines.append(
                            str(_call(LINE_SALVAGE_PROMPT.format(line=ln))).strip())
                        touched += 1
                    except Exception:
                        new_lines.append(ln)   # 该行救捞失败 → 保留（后续会剥离）
                rejoined = "\n".join(new_lines)
                fixed, missing = restore_dialogue(normalize_punct(rejoined), store)
                if missing:
                    return False, "逐行救捞后占位符缺失 %s" % missing, None
                if struct_store:
                    fixed, smissing = restore_structure(fixed, struct_store)
                    if smissing:
                        return False, "逐行救捞后结构行占位符缺失 %s" % smissing, None
                resid = residual_cjk_segments(fixed, store)
                if not resid:
                    return True, "", fixed
                if not touched:
                    # 一轮下来一行都没动（全在抛异常）→ 再跑也是白跑
                    break
            return False, "逐行救捞 %d 轮后仍残留中文 %d 处（%s…）" % (
                max(1, int(rounds)), len(resid),
                " / ".join(s[:12] for s in resid[:3])), fixed

        def _translate_single(idx):
            nonlocal n_done, n_fail, n_retry, n_salv
            body, sep = expanded[idx]
            i = idx + 1
            # ★ 结构行（banner / 双语气泡标签）先挖成 [[Sn]]，不送模型、不剥离。
            struct_protected, struct_store = protect_structure(body)
            protected, store = protect_dialogue(struct_protected)
            prompt = system + "\n# 原文\n" + protected

            # ★ 挖掉结构行后，整块**没有任何待译内容**（纯 PACK 头/尾/banner）→
            #   原样透传。这类块的中文是格式的一部分（`Project / 项目`），送模型
            #   只会被要求"翻掉"再被兜底剥离，反而把格式剁坏。
            if struct_store and not _CJK_RE.search(_DIALOG_RE.sub("", protected)):
                restored, _miss = restore_structure(protected, struct_store)
                out_parts.append(normalize_punct(restored) + sep)
                n_skip += 1
                report.append("  块%-3d %5d 字 → 结构行（%d 行双语气泡）原样保留，未送翻译"
                              % (i, len(body), len(struct_store)))
                return

            # --- 首翻 ---
            try:
                got = _call(prompt)
            except Exception as exc:
                # ★ 调用彻底失败（网络/额度）也不能把中文原文放进去：那是
                #   「模型读中文当台词 → 乱说话」的入口。剥离残留中文后透传。
                #   ★ 结构行先还原（[[Sn]] → `Project / 项目`），再剥离 ——
                #     否则 PACK 头部的双语气泡会被一起剁掉。
                base, _sm = restore_structure(protected, struct_store)
                cleaned, n_strip = strip_residual_cjk(normalize_punct(base), store)
                out_parts.append((cleaned or "") + sep)
                n_fail += 1
                report.append("  块%-3d %5d 字 → **调用失败，已剥离 %d 处中文后透传**"
                              "（%s）" % (i, len(body), n_strip, exc))
                return

            last_got = got
            ok, reason, fixed, leftovers = _check(protected, got, store, struct_store)
            retried = False
            salvaged = False
            halluc = False
            cjk_kept = 0
            # --- 二级：自查不过 → 二次翻译（携原因 + 未翻片段 + 原文 + 上次错误译文）---
            if not ok:
                if leftovers:
                    leftover_text = "\n".join("- " + s for s in leftovers[:20])
                else:
                    leftover_text = "- （占位符/结构问题，非残留中文，请照铁律修正）"
                retry_prompt = RETRY_PROMPT.format(
                    reason=reason, leftover=leftover_text,
                    source=protected[:3000], bad=str(got)[:1500])
                try:
                    got2 = _call(retry_prompt)
                    last_got = got2
                    ok2, reason2, fixed2, leftovers2 = _check(
                        protected, got2, store, struct_store)
                    if ok2:
                        ok, reason, fixed = True, "", fixed2
                        retried = True
                        n_retry += 1
                    else:
                        reason = "二次翻译仍失败：%s" % reason2
                        if fixed2:
                            fixed, leftovers = fixed2, leftovers2
                except Exception as exc2:
                    reason = "二次翻译调用失败：%s" % exc2
            # --- 三级兜底：二翻仍失败 → 逐行救捞（多轮，每轮只补仍含中文的行）---
            if not ok:
                ok3, reason3, fixed3 = _line_salvage(protected, store, struct_store)
                if ok3:
                    ok, reason, fixed = True, "", fixed3
                    salvaged = True
                    n_salv += 1
                else:
                    reason = "逐行救捞仍失败：%s" % reason3
                    if fixed3:
                        fixed = fixed3

            # ★ 本块本来**没有台词**（store 为空）时，模型仍会照着提示词里
            #   「把每个 [[Dn]] 占位符原样放回」那条规则自己编一整串占位符
            #   （实测有块编出 100 个）。validate() 因此报「台词占位符不匹配：
            #   期望 []，实际 […]」→ 整块回退成中文原文。
            #   暴露面极大：一份 96 块的剧本里 88 块是无台词块，全在这个判定下，
            #   实测 20 块因此保留中文 —— 中文再往下走就是「后段计划外发声」。
            #   只要剥掉幻觉占位符后译文本身成立（没有残留中文），就采纳英文。
            if not ok and not store and last_got:
                stripped = _PLACEHOLDER_RE.sub("", str(last_got))
                rest = _DIALOG_RE.sub("", stripped)
                cjk = len(_CJK_RE.findall(rest))
                if stripped.strip() and (cjk == 0
                                         or cjk < max(1, len(rest) * 0.03)):
                    ok, reason, fixed = True, "", normalize_punct(stripped).rstrip()
                    halluc = True

            # ★ 只差「残留中文」这一项时，采纳英文，别整块回退中文（2026-09-27）
            #   实测（第 3 轮真渲染）：一块 342 字的正文，只因为模型把台词里的
            #   「棒」漏抄到英文散句里 —— 一共 2 个残留汉字 —— 被判失败，整块
            #   保留中文。中文重新进入提示词，正是质检技能 §4.4 G 类
            #   「模型读中文当台词 → 计划外发声」的入口；代价远大于这两个字符。
            #   `fixed` 非空即意味着占位符序列、逐字还原、事实项三项都已通过
            #   （见 _check 的返回口径），所以这里采纳的是一份**结构完好**的英文。
            #
            #   ★ 判据数**汉字个数**，不是「残留段数」：`residual_cjk_segments`
            #     返回的是「含中文的行/段」，一段可能装着几十个字。按段数比会
            #     把整块没翻的文本（3 段、上百字）误判成「占比极小」放行 ——
            #     离线回归用例②就是这么被抓出来的。阈值沿用 3%。
            #
            #   ★ 2026-09-27 二改：原来这里「采纳英文」是把残留中文**原样留下**。
            #     实测真跑：块 20 三级兜底全失败 → 整块回退中文 → 201 个汉字
            #     进了 H3 提示词。现在改成**先剥离再采纳**，中文一个都不留。
            if (not ok and fixed and leftovers
                    and "残留中文" in str(reason)):
                cjk_chars = len(_CJK_RE.findall("".join(str(s) for s in leftovers)))
                if cjk_chars / float(max(1, len(fixed))) < 0.03:
                    # ★ 结构行已是**还原后**的中文（`Project / 项目`），剥离前
                    #   先挖回占位符，剥完再填回 —— 只剥真·残留的译漏中文。
                    shielded, shield_store = protect_structure(fixed)
                    cleaned, n_strip = strip_residual_cjk(shielded, store)
                    cleaned, _sm = restore_structure(cleaned, shield_store)
                    if cleaned.strip():
                        ok, reason, fixed = True, "", cleaned
                        cjk_kept = cjk_chars

            # ★ 最后一道防线：任何原因导致的「还没通过」，都**不允许**把中文
            #   放下去。剥离残留中文后采纳英文 —— 宁可丢一小截镜头说明，也
            #   绝不让模型读到中文（§4.4 G 类：中文进提示词 = 计划外发声）。
            #   ★ 剥离前先把结构行占位符还原回上文（若模型抄漏了 [[Sn]]，
            #     这里补回），再只剥离**非结构行**的中文。
            if not ok:
                base = fixed if fixed else normalize_punct(protected)
                base, _sm = restore_structure(base, struct_store)
                cleaned, n_strip = strip_residual_cjk(base, store)
                if cleaned.strip():
                    ok, reason, fixed = True, "", cleaned
                    cjk_kept = cjk_kept or n_strip

            if ok:
                out_parts.append(fixed.rstrip() + sep)
                n_done += 1
                tag = ""
                if halluc:
                    tag = ("（本块无台词，剥掉模型幻觉的 %d 个占位符后采纳）"
                           % len(_PLACEHOLDER_RE.findall(str(last_got))))
                elif cjk_kept:
                    tag = "（剥离残留中文 %d 处后采纳英文）" % cjk_kept
                elif salvaged:
                    tag = "（逐行救捞通过）"
                elif retried:
                    tag = "（二次翻译通过）"
                report.append("  块%-3d %5d 字 → 已英译%s" % (i, len(body), tag))
            else:
                # 走到这里说明剥离后什么都不剩（整块本来就是纯中文且没译出来）。
                # 放中文进去等于让模型乱说话，放空又会让下游分不出段 —— 两者
                # 都不好，但**空**是可诊断的、**中文**是会被念出来的。选空。
                out_parts.append(sep)
                n_fail += 1
                report.append("  块%-3d %5d 字 → **整块未译出，已置空**"
                              "（放中文进提示词会让模型乱说话）" % (i, len(body)))

        def _translate_batch(idxs):
            """批量合并：一次 API 调用翻多块；任何一块自查不过就整批回退逐块
            （逐块会走 自查+二次翻译）。

            ★ 必须**先全部校验、再一次性写入**（原子提交）。旧写法是边校验边
            ``out_parts.append``，中途某块不过就 ``return False`` —— 已经写进去的
            前几块留在 out_parts 里，外层再对整批跑 ``_translate_single``，
            那些块就被写入两遍：最终文本内容重复、``n_done`` 多计
            （实测报告头「共 33 块 —— 英译 34」且块号重复出现）。
            """
            nonlocal n_done
            bodies = [expanded[k][0] for k in idxs]
            protected_list, stores, struct_stores = [], [], []
            for b in bodies:
                sp, ss = protect_structure(b)
                p, s = protect_dialogue(sp)
                protected_list.append(p)
                stores.append(s)
                struct_stores.append(ss)
            note = _BATCH_PROMPT_NOTE.format(n=len(idxs), n_sep=len(idxs) - 1)
            prompt = (system + note + "\n\n# 原文（%d 段）\n" % len(idxs)
                      + _BATCH_SEP.join(protected_list))
            try:
                got = _call(prompt)
            except Exception:
                return False
            pieces = got.split(_BATCH_SEP.strip())
            if len(pieces) != len(idxs):
                pieces = re.split(r"<{2,3}\s*H3SEG_BOUNDARY\s*>{2,3}", got)
            if len(pieces) != len(idxs):
                return False
            pieces = [p.strip("\n") for p in pieces]
            # 阶段一：全部校验，只收集不落盘
            staged = []
            for k, piece, prot, store, sstore in zip(
                    idxs, pieces, protected_list, stores, struct_stores):
                ok, _reason, fixed, _left = _check(prot, piece, store, sstore)
                if not ok:
                    return False
                staged.append((k, fixed))
            # 阶段二：整批通过后一次性写入
            for k, fixed in staged:
                out_parts.append(fixed.rstrip() + expanded[k][1])
                n_done += 1
                report.append("  块%-3d %5d 字 → 已英译（批量：%d 块合 1 次调用）"
                              % (k + 1, len(expanded[k][0]), len(idxs)))
            return True

        i = 0
        n_total = len(expanded)
        while i < n_total:
            if not _needs(i):
                body, sep = expanded[i]
                out_parts.append(normalize_punct(body) + sep)
                n_skip += 1
                i += 1
                continue
            batch = [i]
            if batch_budget > 0:
                total_chars = len(expanded[i][0])
                j = i + 1
                while j < n_total and _needs(j):
                    extra = len(expanded[j][0])
                    if total_chars + extra > batch_budget:
                        break
                    total_chars += extra
                    batch.append(j)
                    j += 1
            if len(batch) > 1:
                if not _translate_batch(batch):
                    for k in batch:
                        _translate_single(k)
            else:
                _translate_single(i)
            i += len(batch)

        result = ensure_field_lines("".join(out_parts))
        head = ("v9：共 %d 块 —— 英译 %d / 已是英文跳过 %d / "
                "未译出（已置空）%d（二次翻译救回 %d、逐行救捞救回 %d）"
                % (len(expanded), n_done, n_skip, n_fail, n_retry, n_salv))
        # ★ 交付闸门：台词块之外**零中文**。这是本节点唯一的硬承诺 ——
        #   中文一旦进 H3 提示词，模型会把它读成台词内容，成片就是
        #   「乱说话、与剧本对不上」（质检技能 §4.4 G 类）。
        #   ★ 结构行（PACK 头 / banner / `Label / 中文` 双语气泡）的 CJK **不计**：
        #     那是官方格式的一部分，剥掉才是损伤（见 protect_structure）。
        gate_text = _DIALOG_RE.sub("", result)
        for _sline in protect_structure(gate_text)[1]:
            gate_text = gate_text.replace(_sline, "")
        leak = len(_CJK_RE.findall(gate_text))
        head += "\n交付闸门：台词块之外残留中文 = %d %s（PACK 结构行已豁免）" % (
            leak, "✅" if leak == 0 else "❌ 不要渲染，先看下面哪块出了问题")
        if n_fail:
            head += ("\n⚠ 有 %d 块没译出来（已置空，**没有**放中文下去）。"
                     "API 模式先检查 Base URL / 模型名 / Key；"
                     "「自查未过」类失败多为台词被翻/残留中文，报告里有原因。"
                     % n_fail)
        log("[H3ScriptTranslate] %s", head.replace("\n", " | "))
        return io.NodeOutput(result, head + "\n" + "\n".join(report))


def register_with_extension(ext):
    """Return node classes for MarqueeDirectorExtension.get_node_list()."""
    return [H3ScriptTranslate]
