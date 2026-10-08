# -*- coding: utf-8 -*-
"""服装锚定兜底 —— 节点内建版（每次渲染自动生效）。

为什么要有这个模块
==================
连续分镜里只要有一段演"穿/脱衣"，模型就容易**自造一件衣裳**：它把提示词里的
``collar`` / ``one piece at a time`` / ``tie it at the waist`` 当成结构断言，
于是渲出来的不是参考图那一件，而是一件它自己编的前开襟外袍。

修法（H3 规范）：外观由参考图决定 —— **删掉冲突断言 + 声明等价**，
绝不能"补一段服装描述"（那等于让模型再发挥一次）。

本模块做的两件事
================
1. **补锚定**（``ensure_wardrobe_anchor``）：给「有穿衣动作、且该动作归属到」
   的那个 ``<Subject N>`` 追加一句结构中立的所有权声明。
2. **告警**（``scan_danger_vocab``）：把"断言了前开襟/多件套/腰带"的词列出来，
   **只报告，不自动改写** —— 这类词是否算错取决于参考服装本身
   （挂颈裙没有领子，但换个剧本的对襟袍可能有），自动删会误伤创作文本。

★ 为什么只补"有穿衣动作的段、且只补动作归属到的那个角色"
--------------------------------------------------------
朴素做法「给每个角色类都补」会在 4 段 × 3 角色的 PACK 上产出 24 条锚定，
其中两个是男性角色 —— 荒谬，而且会压制合理的分层着装。
静态"已穿好"的段照抄参考图不会出错，不需要锚定。

归属算法（``attribute_dressing``）
==================================
对每个 ``DRESS_ACTION`` 命中位置 p，按优先级判定主体：
  ① 动作落在以 ``<Subject N>`` 开头的行内        → 就是该主体（最高优先）
  ② 句内 **动作之前** 最近的 ``<Subject N>``      → 该主体
  ③ 句内所有标签都在动作之后                      → 取离 p 最近的
  ④ 句内无标签                                    → 本段之前最后提到的主体
  ⑤ 全失败                                        → 不归属（只告警，不补）

★ 两条实测踩过的坑，改代码时别踩回去：
  * ``；`` / ``;`` **不是**句界。S01 的长句里 ``<Subject 2>`` 就落在 ``；`` 之后，
    一切开前半句只剩代词「她」，归属会掉到上一个角色（实测错成 ``<Subject 3>``）。
  * 句内标签偏移必须 ``+ s0`` 换算回段内偏移。漏了会把"句末最后一个标签"
    当成"动作前的最近标签"，实测把穿衣动作误归属给 ``<Subject 1>``。
  * 集合里存 **字符串** id：下游比对的是 ``SUBJ_RE`` 的 str 捕获组，
    存 int 会让 ``'2' not in {2}`` 恒真，归属静默失效。

本模块的文案与判定与 ``minimax-h3-shot-segment/scripts/ensure_wardrobe_anchor.py``
保持一致 —— 那边是离线体检/批量修，这边是渲染时的自动兜底，两者不许分叉。
"""

import bisect
import json
import os
import re

# ---------------------------------------------------------------------------
# 判据常量
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ★★ 是否允许 wardrobe 兜底**改写作者的提示词**。默认 **关闭**（只读诊断）。
#
#   2026-09-27 用户裁决：「源剧本 人物服装场景是和参考图一致，禁止私自更换描述」。
#   源剧本自己就写了 ``<Picture N>: fully_preserved - … 外观与参考图一致`` 这套锁，
#   节点再改写作者的文字属于越权；而且它**只在「有穿衣动作」的那一段改** ——
#   实测 iter6 真渲染：S01 被改 15 处措辞 + 追加 4 处句子，S02–S04 一字未动，
#   四段提示词对同一件衣服的描述分叉（违背规范 §4 全链逐字复用），
#   成片表现为同一角色跨段换三套衣服（金 → 粉 → 绿）。
#
#   → 默认只扫描报告；确实需要旧行为时设 ``MARQUEE_WARDROBE_REWRITE=1``。
# ---------------------------------------------------------------------------
_WARDROBE_REWRITE = os.environ.get(
    "MARQUEE_WARDROBE_REWRITE", "0").strip().lower() in ("1", "true", "yes", "on")

PERSON_HINTS = ('person', 'man', 'woman', 'character', 'lawyer',
                '人', '男', '女', '角色', '律师')

# 「已锚定」判据 —— 只认「唯一一件衣物」概念，避免被通用措辞误判
ANCHOR_HINTS = (
    'exactly one garment', 'same single garment', 'owns exactly one',
    '只有一件衣物', '同一件衣物', '唯一一件衣物', '只有这一件',
)

# ★ 不要把 wears / wearing / worn 放进来：锚定声明本身就含 "the one worn in
#   <Picture 2>"，放进来会把所有段都误判成"有穿衣动作"（实测 S04 误报）。
# ★ remove 必须与"衣物名词"同现：裸 remove 会命中 "remove her make-up" 这类比喻。
#   只要**动作**，不要状态。
DRESS_ACTION = (
    r'\bput(?:s|ting)?\b[^.]{0,24}\bon\b',
    r'\btak(?:e|es|ing|en)\b[^.]{0,24}\boff\b',
    r'\b(?:garment|clothes|clothing|robe|gown)\b[^.]{0,24}\bremov(?:e|es|ing|ed)\b',
    r'\bremov(?:e|es|ing|ed)\b[^.]{0,24}\b(?:garment|clothes|clothing|robe|gown)\b',
    r'\bdress(?:es|ing)\b',
    r'\bundress',
    r'穿回|穿上|穿衣|褪下|脱下|系好|着衣|披上',
)

# 已知无害的样板句：四段完全相同、且后段带着它仍然渲对时，判为非诱因。
BENIGN_CONTEXT = (
    (r'taking in shoulders,\s*neck and collar',
     '取景样板句（四段完全相同）—— 后段带着它仍渲对则判为非诱因，保持最小改动'),
    (r'含肩颈与衣襟领口',
     '取景样板句（四段完全相同）—— 同上'),
)

# ★★ 为什么用「凡参考图上没有的部位/层次/系合件一律不加」这种**结构中立**说法，
#    而不是点名枚举 "no collar, no front opening and no wrap"？
#    因为枚举版**假定参考服装没有前襟**。挂颈缎裙确实没有，但换个剧本、
#    参考服装若是带领的对襟袍，枚举版就变成**错的**。
#    中立版在任何服装上都成立，且同样能封死"模型自造结构"。
EN_TAIL = (
    " {subj} owns exactly one garment in the whole film, the one worn in {pic}: "
    "what is worn, what is taken off and what is put back on is always that same "
    "single garment, so no second garment, no substitute and no added layer ever appears, "
    "and it reads as the same garment in every state of dress, off, half on or fully worn, "
    "fastened exactly the way {pic} is fastened and no other way, and no part, layer or "
    "fastening that {pic} does not show is ever added to it."
)
CN_TAIL = (
    " {subj} 全片只有一件衣物，即 {pic} 中所穿的那一件：无论穿在身上、褪下还是重新穿回，"
    "始终是同一件衣物，因此全片不出现第二件衣物、不出现替代款、不叠加任何外披，"
    "无论着装状态是褪下、半掩还是整装，都读作同一件衣物，"
    "系合方式与 {pic} 完全一致、不采用其它方式，"
    "凡 {pic} 上没有的部位、层次或系合件，一律不额外加出。"
)

FABRIC_EN = (
    " fastened exactly the way {pic} is fastened and no other way, and no part, layer or "
    "fastening that {pic} does not show is ever added to it."
)
FABRIC_CN = (
    "，系合方式与 {pic} 完全一致、不采用其它方式，"
    "凡 {pic} 上没有的部位、层次或系合件，一律不额外加出。"
)
# 「已加过禁止新造结构」判据 —— 含枚举版也算已加，避免重复补。
FABRIC_HINTS = (
    'no other way', 'does not show is ever added', 'no collar, no front opening',
    '不采用其它方式', '一律不额外加出', '不额外加出领子',
)

# 着装完成态等价锁：与时间码解耦（不写死 00:07.200），任何剧本都能注入。
EN_LOCK = (
    " From the moment the dressing is finished to the end of the segment, the state of dress "
    "of {subj} is the state of {pic} itself, matched frame for frame: every part fastened is "
    "a part {pic} has, and no part of {pic} is left unfastened, missing or replaced."
)
CN_LOCK = (
    " 自着装完成之时起至本段结束，{subj} 的着装状态即 {pic} 本身的状态，逐帧一致："
    "所系合的每一处都是 {pic} 上有的部分，"
    "{pic} 上的任何部分都不会缺失、不会被替换或未系合。"
)
# ★ 不要用裸 '逐帧一致' —— subject_definitions 里那句
#   "<Picture N> 是 <Subject N> 的外观参考图，全片逐帧一致。" 也含它，会误判成"已有锁"。
LOCK_HINTS = ('matched frame for frame', 'state of dress is the state of',
              '着装状态即', '所系合的每一处')

SEG_MARK_RE = re.compile(r'^#{4,}\s*(.+?)\s*#{4,}\s*$')
SECTION_KEYS = (
    'subject_definitions:', 'summary:', 'retention_analysis:',
    'detailed_description:', 'overall_soundscape:', 'non_diegetic_music:',
)
SUBJ_RE = re.compile(r'^\s*<Subject\s+(\d+)>')
PIC_IN_LINE_RE = re.compile(r'<Picture\s+(\d+)>')
# ★ PACK 里「角色 ↔ 参考图」有两种写法：
#    规范指代  <Picture 2> is the reference image for Subject 2 appearance.
#    裸写      <Subject 2> is the person in Picture 2 (张三，本名；律师)
#   只认尖括号会让裸写那一行被判成「未引用 <Picture N>」而**静默跳过锚定** ——
#   2026-09-19 六段中文 PACK 实测：首段（唯一有穿衣动作、且没有 latent 锚定的一段）
#   因此一条锚定都没吃到，成片首段服装又开始自造。
#   ★ 裸写只在**本行没有尖括号形式**时才启用，避免混排行里读错编号。
PIC_BARE_RE = re.compile(r'(?<!<)\bPicture\s+(\d+)\b(?!>)')
SUBJ_TAG_RE = re.compile(r'<Subject\s+(\d+)>')
# 切句：中文句末标点 + 换行 + 英文句末标点后的空白。★ 不含 `；`（见模块 docstring）。
SENT_BOUND_RE = re.compile(r'[。！？!?]|\n|(?<=[.!?])\s+')

# 断言了「前开襟 / 多件套 / 腰带」的词汇 —— 与挂颈式参考服装冲突。
# 只用于**告警**，不自动改写（是否算错取决于参考服装本身）。
DANGER_VOCAB = (
    (r'\bcollar\b', 'collar —— 断言有领子/前襟'),
    (r'\bfront opening\b|\bopen front\b', 'front opening —— 断言前开襟'),
    (r'\bone piece at a time\b', 'one piece at a time —— 断言多件套'),
    (r'\bthe last piece\b|\blast garment\b', 'the last piece —— 断言多件套'),
    (r'\bstraightens?\b[^.]{0,20}\btie\b|\btie it\b|\btie the\b',
     'straightens/bends to tie it —— 断言需要系带'),
    (r'\bat the waist\b[^.]{0,30}\bcloses\b|\bwaist as (?:the|that) (?:cloth|garment)'
     r'\b[^.]{0,20}\bcloses\b', 'closes at the waist —— 断言腰带/系合'),
    (r'\blifts? a leg\b[^.]{0,20}\barrange\b', 'lifts a leg to arrange —— 断言穿脱动作'),
    (r'领子|前襟|对襟|腰带|系带', '中文：断言领子/前襟/腰带'),
    # ★ 「衣襟」才是中文剧本里最高频的前开襟断言（衣襟半敞 / 掩齐衣襟 /
    #   衣襟领口），比「前襟」常见得多 —— 2026-09-19 六段中文 PACK 里
    #   前襟 0 处、衣襟 9 处，旧表一条都没命中。取景样板句由 BENIGN_CONTEXT 豁免。
    (r'衣襟', '中文：衣襟 —— 断言前开襟'),
    (r'一件件|最后一件', '中文：断言多件套'),
    (r'系好|系上|系紧', '中文：断言需要系带'),
    (r'抬腿[^。]{0,12}(?:理衣|穿衣|整衣|提衣)', '中文：断言穿脱动作'),
)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _sent_spans(text):
    """返回 [(起, 止), …]，覆盖整段文本。"""
    spans, start = [], 0
    for m in SENT_BOUND_RE.finditer(text):
        if m.end() > start:
            spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _line_subject_spans(body):
    """返回 [(行起, 行止, subject_id or None), …]，用于「行首标签」归属。

    ★ 锚定/锁文本自己就含 takes off / puts back on / dressing / 穿衣 / 褪下，
      所以**不能**只靠"行内有没有动作词"判断，必须看行首有没有标签。
    """
    out, pos = [], 0
    for ln in body.split('\n'):
        m = SUBJ_RE.match(ln)
        out.append((pos, pos + len(ln), m.group(1) if m else None))
        pos += len(ln) + 1                       # +1 = 换行符
    return out


def is_chinese_line(line):
    return any('\u4e00' <= ch <= '\u9fff' for ch in line)


def split_segments(text):
    """按 PACK 的 `########## Sxx / … ##########` 标记切段。

    没有标记时整篇当一段 —— 节点里传进来的往往是**已经切好的段级 prompt**
    （没有 `####` 标记），所以这条分支是常态而不是异常。
    """
    lines = text.split('\n')
    head, segs, cur = [], [], None
    for ln in lines:
        m = SEG_MARK_RE.match(ln.strip())
        if m:
            if cur is not None:
                segs.append(cur)
            cur = [ln, m.group(1), []]
            continue
        if cur is None:
            head.append(ln)
        else:
            cur[2].append(ln)
    if cur is not None:
        segs.append(cur)
    if not segs:
        return '', [('', '<全篇>', text)]
    return ('\n'.join(head), [(a, b, '\n'.join(c)) for a, b, c in segs])


def join_segments(head, segs):
    chunks = [head] if head else []
    for marker, _name, body in segs:
        # ★ 无标记的段级 prompt（节点里最常见的形态）marker 是空串 ——
        #   直接 append 会凭空多出一个前导换行，于是每次调用都多一行，
        #   幂等性就这么破了（实测第二次调用文本就对不上）。
        if marker:
            chunks.append(marker)
        if body:
            chunks.append(body)
    return '\n'.join(chunks)


def attribute_dressing(text):
    """把每个「穿衣动作」归属到具体的 `<Subject N>`。

    返回 (per_seg, report)：
      per_seg : {段名: set(subject_id 字符串)}
      report  : [(段名, sid or None, 命中词), …]
    """
    head, segs = split_segments(text)
    per_seg, report = {}, []
    pos = 0
    seed = None

    for _marker, name, body in segs:
        ev = set()
        per_seg[name] = ev
        if not body:
            continue

        off = text.find(body, pos)
        if off < 0:
            off = pos
        else:
            pos = off + len(body)
        pre_tags = SUBJ_TAG_RE.findall(text[:off])
        if pre_tags:
            seed = int(pre_tags[-1])

        spans = _sent_spans(body)
        starts = [s for s, _ in spans]
        lspans = _line_subject_spans(body)
        lstarts = [s for s, _, _ in lspans]

        for pat in DRESS_ACTION:
            for m in re.finditer(pat, body, re.I):
                p = m.start()
                li = max(0, bisect.bisect_right(lstarts, p) - 1)
                l0, l1, line_sid = lspans[li]
                if line_sid is not None and l0 <= p < l1:
                    sid = line_sid                       # ① 行首标签直接归它
                else:
                    k = max(0, bisect.bisect_right(starts, p) - 1)
                    s0, s1 = spans[k]
                    # ★ 必须 + s0 换算回段内偏移
                    tags = [(t.start() + s0, int(t.group(1)))
                            for t in SUBJ_TAG_RE.finditer(body[s0:s1])]
                    if tags:
                        before = [t for t in tags if t[0] <= p]
                        sid = before[-1][1] if before else min(
                            tags, key=lambda t: abs(t[0] - p))[1]
                    else:
                        sid = seed
                report.append((name, sid, m.group(0)))
                if sid is not None:
                    # ★ 存字符串：下游比对的是 SUBJ_RE 的 str 捕获组
                    ev.add(str(sid))

        tail = SUBJ_TAG_RE.findall(body)
        if tail:
            seed = int(tail[-1])

    return per_seg, report


def _fix_line(line, allowed_ids, seg_has_action):
    """返回 (新行, 是否改动, 原因)。

    三种情形：
      * 完全没锚定        → 追加整段锚定（含禁止新造结构）
      * 已锚定但缺禁令     → 只补「禁止新造结构」那半句（升级老 PACK）
      * 已锚定且已带禁令   → 跳过
    """
    m = SUBJ_RE.match(line)
    if not m:
        return line, False, None
    sid = m.group(1)

    low = line.lower()
    if not any(h in low for h in PERSON_HINTS):
        return line, False, 'skip:非角色类'

    # 规范指代优先；没有尖括号时才退回裸写（见 PIC_BARE_RE 的注释）
    pic = PIC_IN_LINE_RE.search(line) or PIC_BARE_RE.search(line)
    if not pic:
        return line, False, 'skip:未引用<Picture N>'
    pic_tag = '<Picture %s>' % pic.group(1)
    subj_tag = '<Subject %s>' % sid

    anchored = any(h in low for h in ANCHOR_HINTS)
    has_fabric = any(h in low for h in FABRIC_HINTS)
    cn = is_chinese_line(line)

    if anchored and has_fabric:
        return line, False, 'skip:已有锚定+禁令'
    if anchored and not has_fabric:
        # 老 PACK 只差禁令半句 —— 与"谁在穿衣"无关，见到就补，避免留下半截锚定
        tail = FABRIC_CN if cn else FABRIC_EN
        return line.rstrip() + tail.format(pic=pic_tag), True, 'upgraded:补禁令'

    if sid not in allowed_ids:
        return line, False, ('skip:本段无穿衣动作归属' if seg_has_action
                             else 'skip:本段无穿衣动作')
    tail = CN_TAIL if cn else EN_TAIL
    return line.rstrip() + tail.format(pic=pic_tag, subj=subj_tag), True, 'added'


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def ensure_wardrobe_anchor(text, want_lock=True):
    """给「有穿衣动作」的段补服装锚定，返回 (新文本, notes)。

    notes 是人类可读的改动/跳过说明，直接喂给 log 即可。

    ★ 幂等：已锚定且已带禁令的行原样返回，重复调用不会叠加。
    ★ 只给**动作归属到的那个角色**补，不是见人就补。
    """
    if not text or not isinstance(text, str):
        return text, []
    head, segs = split_segments(text)
    per_seg, _attr = attribute_dressing(text)
    out_segs, notes = [], []

    for marker, seg_name, body in segs:
        has_action = any(re.search(p, body, re.I) for p in DRESS_ACTION)
        allowed = set(per_seg.get(seg_name, ()))
        pic_map = {}
        lines = body.split('\n')
        out = []
        in_sd = in_ra = False

        for ln in lines:
            s = ln.strip()
            if s.startswith('subject_definitions:'):
                in_sd, in_ra = True, False
                out.append(ln)
                continue
            if s.startswith('retention_analysis:'):
                in_sd, in_ra = False, True
                out.append(ln)
                continue
            if in_sd and any(s.startswith(k) for k in SECTION_KEYS
                             if k != 'subject_definitions:'):
                in_sd = False
            if in_ra and any(s.startswith(k) for k in SECTION_KEYS
                             if k != 'retention_analysis:'):
                in_ra = False

            if in_sd:
                mm0 = SUBJ_RE.match(ln)
                new, changed, why = _fix_line(ln, allowed, has_action)
                out.append(new)
                if mm0:
                    if changed:
                        notes.append('锚定 %s：%s' % (mm0.group(1), why))
                    # 与 _fix_line 同一口径：尖括号优先，退回裸写
                    pic = PIC_IN_LINE_RE.search(new) or PIC_BARE_RE.search(new)
                    if pic and any(h in new.lower() for h in PERSON_HINTS):
                        pic_map[mm0.group(1)] = pic.group(1)
                continue

            if in_ra and has_action and want_lock:
                mm = SUBJ_RE.match(ln)
                if mm and mm.group(1) in pic_map and mm.group(1) in allowed:
                    if any(h in ln.lower() for h in LOCK_HINTS):
                        notes.append('完成态锁 %s：已有，跳过' % mm.group(1))
                    else:
                        pic_tag = '<Picture %s>' % pic_map[mm.group(1)]
                        subj_tag = '<Subject %s>' % mm.group(1)
                        tail = CN_LOCK if is_chinese_line(ln) else EN_LOCK
                        ln = ln.rstrip() + tail.format(pic=pic_tag, subj=subj_tag)
                        notes.append('完成态锁 %s：已注入' % mm.group(1))
                out.append(ln)
                continue

            out.append(ln)

        out_segs.append((marker, seg_name, '\n'.join(out)))

    return join_segments(head, out_segs), notes


# ---------------------------------------------------------------------------
# 转换词规则（提示词转换器）—— **任何剧本通用**
# ---------------------------------------------------------------------------
# 规则 = {id, pattern, repl, note, skip_if?}
#   pattern : 正则（re.I）
#   repl    : **结构中立**的替换文本
#   skip_if : 命中这些正则的**句子**不替换（保护取景样板句等非诱因文本）
#
# ★ 三条铁律（改规则前先读）：
#   1. **删断言，不补描述** —— 补"她穿的是挂颈缎裙"等于让模型再发挥一次，
#      而且换剧本就错。中立说法在任何参考服装上都成立。
#   2. 替换文本必须**结构中立**：不点名"无领子/无前襟/无腰带" ——
#      参考服装究竟有没有前襟**取决于图**，写死枚举在换剧本时就是错的。
#      一律用「凡参考图上没有的部位/层次/系合件不额外加出」这类中立说法。
#   3. 命中 0 次就跳过、**不报错** —— 换剧本后旧规则自然失效，不该打断渲染。
#
# ★ 覆盖不到的剧本特有整句 → 写进 ``data/wardrobe_neutralize.json``。
#   换剧本只改那个 JSON，**不用动 Python**。
DEFAULT_NEUTRALIZE_RULES = (
    {
        "id": "collar",
        # 裸 collar 会误伤取景样板句（"taking in shoulders, neck and collar"
        # 说的是画面取景范围，不是服装结构），用 skip_if 按句子豁免。
        "pattern": r"\bcollar\b",
        "repl": "neckline",
        "note": "collar → neckline（断言有领子/前襟）",
        "skip_if": [r"taking in shoulders,\s*neck and collar"],
    },
    {
        "id": "front-opening",
        "pattern": r"\bfront opening\b|\bopen front\b",
        "repl": "neckline",
        "note": "front opening → neckline（断言前开襟）",
    },
    {
        "id": "one-piece-at-a-time",
        "pattern": r"\bone piece at a time\b",
        "repl": "in one continuous motion",
        "note": "one piece at a time → in one continuous motion（断言多件套）",
    },
    {
        "id": "last-garment",
        "pattern": r"\bthe last garment\b|\bthe last piece\b",
        "repl": "the garment",
        "note": "the last garment/piece → the garment（断言多件套）",
    },
    {
        "id": "straightens-to-tie",
        "pattern": r"\bstraightens?\s+(?:up\s+)?to tie it\b",
        "repl": "straightens, settling the garment",
        "note": "straightens to tie it → settles the garment（断言需要系带）",
    },
    {
        "id": "ties-it",
        "pattern": r"\bties it\b|\bto tie it\b",
        "repl": "settles the garment",
        "note": "tie it → settles the garment（断言需要系带）",
    },
    {
        "id": "cloth-closes",
        # settles 而不是 covers：后面可能接 "at her waist" 补语，
        # covers 会读出 "the garment covers at her waist" 这种别扭英文。
        "pattern": r"\bthe cloth closes\b|\bthe garment closes\b",
        "repl": "the garment settles",
        "note": "cloth/garment closes → garment settles（断言系合）",
    },
    {
        "id": "lifts-leg-to-arrange",
        "pattern": r"\blifts? a leg\b[^.]{0,30}\b(?:arrange|adjust)\b[^.]{0,20}\bgarment\b",
        "repl": "draws the garment up",
        "note": "lifts a leg to arrange → draws the garment up（断言穿脱动作）",
    },
    # ------------------------------------------------------------------
    # 中文通用规则（英文版上面，中文版在下面 —— 结构完全对称）
    #
    # ★ 为什么必须内建中文版：DEFAULT_NEUTRALIZE_RULES 原来只有英文，
    #   中文 PACK 一条都命中不了，于是「任何剧本通用」对中文剧本是假的
    #   （2026-09-19 六段中文 PACK 实测：改写 0 处、告警 0 处）。
    # ★ repl 里**不能写** ``\1`` 这类组引用：``neutralize_danger_vocab`` 用
    #   callable 做替换，Python 不会展开返回值里的反斜杠转义。
    #   要保留名词就把捕获组留在 pattern 外面（如只匹配「最后一件」而非
    #   「最后一件衣裳」），repl 留空即可。
    # ------------------------------------------------------------------
    {
        "id": "cn-one-piece-at-a-time",
        "pattern": r"一件件",
        "repl": "",
        "note": "中文：删「一件件」（断言多件套）",
    },
    {
        "id": "cn-last-garment",
        # 只吃「最后一件」，后面的名词（衣裳/衣物/袍…）自然留下
        "pattern": r"最后一件",
        "repl": "",
        "note": "中文：删「最后一件」（断言多件套）",
    },
    {
        "id": "cn-tie",
        "pattern": r"系(?:好|上|紧)",
        "repl": "整理好",
        "note": "中文：系好/系上/系紧 → 整理好（断言需要系带）",
    },
    {
        "id": "cn-yijin",
        # 衣襟 = 前开襟。替换成「衣料」只留材质、不留闭合方式，结构中立。
        "pattern": r"衣襟",
        "repl": "衣料",
        "note": "中文：衣襟 → 衣料（断言前开襟）",
        "skip_if": [r"含肩颈与衣襟领口"],
    },
)

_NEUTRALIZE_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "wardrobe_neutralize.json")
_neutralize_cache = None


def _load_neutralize_json():
    """读剧本专属词条（exact / regex）。读不到就返回空 —— **绝不能因此打断渲染**。"""
    global _neutralize_cache
    if _neutralize_cache is not None:
        return _neutralize_cache
    out = {"exact": [], "regex": []}
    try:
        with open(_NEUTRALIZE_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            out["exact"] = [r for r in (data.get("exact") or [])
                            if isinstance(r, dict) and r.get("find") and "repl" in r]
            out["regex"] = [r for r in (data.get("regex") or [])
                            if isinstance(r, dict) and r.get("pattern") and "repl" in r]
    except FileNotFoundError:
        pass
    except Exception as exc:
        # ★ 不能静默。这个 JSON 是**所有**服装兜底规则的来源（转换词表 / 锚定句 /
        #   完成态锁）。文件写坏时静默返回空规则 = 服装兜底整体失效，而成片里
        #   的表现只是"某一段的衣服不对" —— 排查时根本想不到是这里。
        #   2026-09-19 的「首段自造服装」事故就是靠这套规则修的。
        from .common import log
        log("wardrobe: 读不了 %s（%s）—— 本次不做服装兜底，成片可能自造服装",
            os.path.basename(_NEUTRALIZE_JSON), exc)
    _neutralize_cache = out
    return out


def _sentence_at(text, pos):
    """取 pos 所在的那一句（用于 skip_if 判断）。"""
    for s, e in _sent_spans(text):
        if s <= pos < e:
            return text[s:e]
    return text


def neutralize_danger_vocab(text):
    """把提示词里「断言前开襟/多件套/腰带」的措辞改成结构中立说法。

    **任何剧本都能用**：内置通用规则（``DEFAULT_NEUTRALIZE_RULES``）+ 剧本专属词条
    （``data/wardrobe_neutralize.json``）。换剧本时旧词条只是匹配不到、自动跳过。

    返回 (新文本, notes)。幂等 —— 改完那些措辞就不存在了，再跑一次无变化。

    ★ 与 ``scan_danger_vocab`` 的分工：那边只报告，这边真的改。改完仍会再扫一次，
      扫不干净的部分照旧告警 —— 那是剧本特有的说法，往 JSON 补一条即可，
      **绝不能硬猜着改创作文本**。
    """
    if not text or not isinstance(text, str):
        return text, []
    notes = []
    out = text

    # ① 剧本专属整句（精确匹配，最具体、优先）
    for rule in _load_neutralize_json()["exact"]:
        find, repl = rule["find"], rule.get("repl", "")
        n = out.count(find)
        if not n:
            continue
        out = out.replace(find, repl)
        notes.append('转换词（词条）×%d：%s' % (n, rule.get("note") or find[:40]))

    # ② 剧本专属正则
    for rule in _load_neutralize_json()["regex"]:
        try:
            out, n = re.subn(rule["pattern"], rule.get("repl", ""), out, flags=re.I)
        except re.error:
            continue                      # 词条写坏了就跳过，不打断渲染
        if n:
            notes.append('转换词（词条正则）×%d：%s' % (n, rule.get("note") or rule["pattern"][:40]))

    # ③ 通用规则（任何剧本）
    for rule in DEFAULT_NEUTRALIZE_RULES:
        try:
            pat = re.compile(rule["pattern"], re.I)
        except re.error:
            continue
        benign = []
        for b in rule.get("skip_if") or ():
            try:
                benign.append(re.compile(b, re.I))
            except re.error:
                continue

        def _sub(m, _rule=rule, _benign=benign):
            if _benign and any(b.search(_sentence_at(out, m.start())) for b in _benign):
                return m.group(0)         # 落在豁免句里 → 原样保留
            return _rule.get("repl", "")

        out, n = pat.subn(_sub, out)
        if n:
            notes.append('转换词（通用）×%d：%s' % (n, rule.get("note") or rule["id"]))

    return out, notes


def process_prompt(prompt):
    """渲染前对 H3 提示词做服装**诊断**；默认**不改写**，返回 ``(new_prompt, notes)``。

    ★★ 2026-09-27 行为变更（用户裁决）：默认**只读**。

      源剧本自己已经声明了外观约束 ——
      ``<Picture 1>: fully_preserved - Subject 1 外观在所有镜头中与参考图一致``
      ``<Picture 2>: fully_preserved - Subject 2 外观在所有镜头中与参考图一致``
      ``<Picture 3>: fully_preserved - 场景在全程与参考图完全一致，无跳变、无新增陈设``
      ``<Picture 4>: fully_preserved - Subject 3 外观在所有镜头中与参考图一致``
      —— 并**禁止私自更换描述**。节点再改写作者的文字就是越权。

      更要命的是它**只在「有穿衣动作」的那一段改**：实测 iter6 真渲染里
      S01 被改了 15 处措辞（``collar → neckline`` / ``front opening → neckline`` …）
      并追加了 2 处锚定 + 2 处完成态锁，而 **S02–S04 一字未动** ——
      四段提示词对同一件衣服的描述就此分叉，直接违背规范 §4「STYLE / 一致性块
      全链逐字复用」。成片表现为**同一角色跨段换三套衣服**（金 → 粉 → 绿），
      而参考图 `<Picture 2>` 是深红纱袍。

      → 现在默认只**扫描并报告**（``scan_danger_vocab``），由作者决定要不要改剧本；
        节点自己一个字都不动。
      → 需要恢复旧的自动改写行为：环境变量 ``MARQUEE_WARDROBE_REWRITE=1``。

    ★ 副作用（正向）：本函数在 ``H3RenderSegmentNode`` 里跑在 ``segment_key``
      **之前**，以前它会改 prompt → 缓存键里存的是被改过的文本；现在 prompt
      原样 → 缓存键就是作者原文，更可复现。
    """
    if not prompt or not isinstance(prompt, str):
        return prompt, []

    notes = []

    if not _WARDROBE_REWRITE:
        # 只读诊断：报出「断言前开襟 / 多件套 / 系带」的词，不改一个字
        try:
            hits = scan_danger_vocab(prompt)
        except Exception as exc:                              # pragma: no cover
            notes.append("诊断跳过（%s）" % exc)
            return prompt, notes
        if hits:
            words = sorted({h[0].lower() for h in hits})
            notes.append(
                "诊断（未改写）：提示词里有 %d 处断言前开襟/多件套/系带的词（%s）。"
                "若参考服装是挂颈式/无前襟，模型可能自造一件 —— 请在**剧本里**"
                "删掉这些断言（不要补服装描述），节点默认不再替作者改稿；"
                "要恢复旧的自动改写行为设 MARQUEE_WARDROBE_REWRITE=1。"
                % (len(hits), ", ".join(words[:8])))
        return prompt, notes

    # ---- 以下为旧行为（需显式 MARQUEE_WARDROBE_REWRITE=1 才走到）-----------
    # ① 转换冲突词
    try:
        prompt, ns = neutralize_danger_vocab(prompt)
    except Exception as exc:                                  # pragma: no cover
        notes.append("转换词跳过（%s）" % exc)
    else:
        notes.extend(ns)

    # ② 补锚定
    try:
        new, ns = ensure_wardrobe_anchor(prompt)
    except Exception as exc:                                  # pragma: no cover
        notes.append("锚定跳过（%s）" % exc)
        new = prompt
    else:
        notes.extend(ns)

    # ③ 复核：还有残留说明转换词表缺条目
    try:
        hits = scan_danger_vocab(new)
        if hits:
            words = sorted({h[0].lower() for h in hits})
            notes.append(
                "仍有 %d 处断言前开襟/多件套/腰带的词（%s）—— "
                "转换词表没覆盖到这些说法。若参考服装是挂颈式/无前襟，模型会自造一件；"
                "请把对应短语补进 wardrobe_neutralize.json（删断言，不要补服装描述），"
                "或直接改提示词。" % (len(hits), ", ".join(words[:8])))
    except Exception as exc:                                  # pragma: no cover
        notes.append("复核跳过（%s）" % exc)

    return new, notes


def scan_danger_vocab(text):
    """列出「断言了前开襟/多件套/腰带」的命中，[(词, 说明, 上下文), …]。

    ``neutralize_danger_vocab`` 改完之后用它复核：还有残留就说明
    ``NEUTRALIZE_TABLE`` 缺条目，需要按本剧本补（或人工改提示词）。
    """
    if not text:
        return []
    hits = []
    for pat, why in DANGER_VOCAB:
        for m in re.finditer(pat, text, re.I):
            ctx = text[max(0, m.start() - 40):m.end() + 40].replace('\n', ' ')
            ctx = re.sub(r'\s+', ' ', ctx).strip()
            # 豁免 skill 自己的取景样板句
            if any(re.search(b, ctx) for b, _ in BENIGN_CONTEXT):
                continue
            hits.append((m.group(0), why, ctx))
    return hits
