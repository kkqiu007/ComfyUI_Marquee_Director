# -*- coding: utf-8 -*-
"""MiniMax H3 分镜提示词 → shots_info 的解析与提示词层规范化。

现行规则 = ``minimax-h3-shot-segment_en_SKILL.md``（**英文单版**）：

* 唯一交付物是**英文版六段**，没有 PACK 头部、没有 ``[1] 中文版`` / ``[2] 英文版``
  分块；文件以 ``END OF PROMPTS`` 收尾。
* 段标题 ``########## S01 / 10s / EN ##########``、接续段
  ``########## S02 / 10s+1.6=11.6 / EN ##########``（新增+回放=生成）。
* 英文版除 ``<d>[Chinese] 原文</d>`` 台词块外**零中文字符**；
  ``detailed_description`` 350–500 英文词，首句是 no-text 声明。
* ``non_diegetic_music`` 按 1.5 判据自动触发 MUSIC block，不默认 ``N/A``。

**向后兼容**：旧的「双语 PACK」（``=====`` 头部 + ``[1] 中文版`` + ``[2] 英文版``）
照样能解析，解析器会把它标成 ``layout="bilingual-pack"`` 并继续按英文版优先取用。
两种形态都支持，不会因为换规范就把老文件判死。

::

    ########## S01 / 10s / EN ##########
    subject_definitions:
    ...
    summary:
    ...
    retention_analysis:
    ...
    detailed_description:
    ...
    overall_soundscape:
    ...
    non_diegetic_music:
    N/A

    ########## S02 / 10s+1.6=11.6 / EN ##########
    ...

    ==========================================================
    END OF PROMPTS
    ==========================================================

这里只做**纯文本**处理：不 import torch / comfy，可以在没有显卡依赖的环境里单测。

Borrowed from ComfyUI_MiniMaxH3_Director
----------------------------------------
* ``lib/task_modes`` — 任务模式（t2v / i2v / fl2v / r2v / v2v / rv2v）与推断口径
* ``lib/task_prompts`` — 官方六段字段名的固定顺序
"""

from __future__ import annotations

import math
import re

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 官方六段，顺序固定（中文版也保留英文字段名）
FIELD_ORDER = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
FIELD_SET = frozenset(FIELD_ORDER)

NO_TEXT_EN = ("No on-screen text, subtitles, captions, timecodes, or any "
              "graphics overlays anywhere in the frame.")
NO_TEXT_ZH = "画面内任何位置都不出现文字、字幕、题字、时间码、水印或任何图形叠加层。"

FPS = 24

# 段首回放区：规范写 1.6s，H3 的 17k+5 网格上落在 1.625s（39 帧）
HANDOFF_SECONDS = 1.6
HANDOFF_GRID_SECONDS = 1.625
# 单段新增时长区间（minimax-h3-shot-segment_en 规范 0.1）：
# 「每段新增固定落在 5–11 秒」——生成最长 12.6 秒（11 + 1.6 回放重叠），
# 首段最长 11 秒。以前这里是 10.0，把规范里的 11s 段判成「超过单段上限」，
# 与样例 PACK（S01 / 11s、S02 / 11s+1.6=12.6）直接打架。
MAX_NEW_SECONDS = 11.0
# 5 秒是硬下限：切镜密度撑不满 5 秒的镜头不单独立段，改写成上一段内的切镜。
MIN_NEW_SECONDS = 5.0

# 现行规则版本（写进解析报告，方便一眼看出跑的是哪一版规范）
RULE_VERSION = "en-single-v2"

# 接续段 [Shot 1] 开头固定原样写的回放句（en 规范 1.4）
HANDOFF_REPLAY_EN = (
    "The segment opens on an exact replay of the closing 1.6 seconds of the "
    "preceding take and carries that motion straight through, so the join is "
    "invisible. The subject holds the exact body pose, head angle, gaze "
    "direction, arm position and facial expression carried in by the replayed "
    "opening and continues out of it without any reset or restart; no new "
    "action, no new spoken word and no new cut begins before the 1.6-second "
    "mark."
)
HANDOFF_REPLAY_ZH = (
    "本段以精确回放上一段结尾 1.6 秒开场，并把该动作直接延续下去，接缝不可见。"
    "主体保持回放带入的身体姿态、头部角度、视线方向、手臂位置与面部表情并由此"
    "继续，没有任何重置或重新开始；在 1.6 秒标记点之前，不开始任何新动作、新"
    "台词或新切镜。"
)

TASK_T2V = "t2v"
TASK_I2V = "i2v"
TASK_FL2V = "fl2v"
TASK_R2V = "r2v"
TASK_V2V = "v2v"
TASK_RV2V = "rv2v"

TASK_LABELS = {
    TASK_T2V: "纯文生视频（无参考）",
    TASK_I2V: "首帧生视频",
    TASK_FL2V: "首尾帧生视频",
    TASK_R2V: "参考图生视频（<Picture N>）",
    TASK_V2V: "视频编辑（<Video N>）",
    TASK_RV2V: "视频编辑 + 参考图",
}

# summary 的官方任务前缀
TASK_PREFIXES = (
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
)

# 1.5 配乐 / 拟音自动判据（中英双语，命中任一即启用 MUSIC block）
MUSIC_WORDS = (
    "配乐", "BGM", "背景音乐", "背景乐", "主题曲", "插曲", "片头曲", "片尾曲",
    "旋律", "鼓点", "节奏", "钢琴", "弦乐", "电子乐",
    "soundtrack", "score", "theme", "beat", "ost", "background music",
    "melody", "piano", "strings", "synth",
)
SFX_WORDS = (
    "拟音", "音效", "SFX", "foley", "stinger", "转场音", "声音设计",
    "sound effect", "sound design",
)
GENRE_WORDS = (
    "演唱会", "舞台表演", "舞蹈", "MV", "广告片", "宣传片", "婚礼", "庆典",
    "仪式", "蒙太奇", "回忆段落", "情绪高潮",
    "concert", "stage performance", "dance", "music video", "wedding",
    "ceremony", "montage", "commercial",
)

# MUSIC block 模板（en 规范 1.5）。<风格> 用 detect_music_style 的结果替换。
MUSIC_BLOCK_TEMPLATE = (
    "A single low, sustained {style} bed at low level. It continues only "
    "through the replayed 1.6-second opening; no new music cue, no BGM swell "
    "and no sound effect starts before 00:01.600. It ducks to clearly below "
    "the voice one second before every spoken line and stays silent for the "
    "entire duration of every line unless the user explicitly asks for music "
    "under dialogue. It stops 1.6 seconds before the end of each shot; the "
    "closing 1.6 seconds of every shot carry no music, no BGM and no sound "
    "effects."
)
# 默认风格（模板里已经有 bed 这个词，这里别再写 bed，否则变成 "bed bed"）
MUSIC_STYLE_DEFAULT = "restrained ambient"

# 只命中拟音/音效、不要音乐床时的专用 block：
# 英文模板是 "sustained <风格> bed"，把 <风格> 填成 "no sustained bed" 会拼出
# "sustained no sustained bed bed"，所以这种情况整句换掉。
MUSIC_BLOCK_SFX_TEMPLATE = (
    "No sustained music bed at any point. A few isolated sound-design hits may "
    "land after 00:01.600; no new music cue and no BGM swell starts before "
    "00:01.600. Every hit ducks to clearly below the voice one second before "
    "each spoken line and stays silent for the entire duration of every line "
    "unless the user explicitly asks for sound under dialogue. Nothing starts "
    "in the closing 1.6 seconds of any shot; the last 1.6 seconds of every "
    "shot carry no music, no BGM and no sound effects."
)

# 从正文里挑 MUSIC block 的 <风格>：取不到就用 restrained ambient bed
STYLE_HINTS = (
    ("钢琴", "piano"), ("piano", "piano"),
    ("弦乐", "strings"), ("strings", "strings"),
    ("电子乐", "electronic"), ("electronic", "electronic"), ("synth", "synth"),
    ("鼓点", "percussive"), ("beat", "percussive"), ("节奏", "percussive"),
    ("民谣", "folk"), ("folk", "folk"),
    ("摇滚", "rock"), ("rock", "rock"),
    ("爵士", "jazz"), ("jazz", "jazz"),
    ("古典", "classical"), ("classical", "classical"),
    ("悬疑", "tense"), ("tense", "tense"),
    ("温暖", "warm"), ("warm", "warm"),
    ("悲伤", "melancholic"), ("sad", "melancholic"),
)

# detailed_description 的英文词数区间（规范 350–500，超出这个宽区间才报）
EN_WORDS_SPEC = (350, 500)
EN_WORDS_TOLERANCE = (200, 700)

# 官方 retention token
RETENTION_TOKENS = (
    "fully_preserved", "partially_preserved", "attribute_transfer",
    "weak_reference", "fully_copy", "partially_copy", "reference",
)

# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------
_SEP_RE = re.compile(r"^\s*={5,}\s*$")
# ########## S02 / 10s+1.6=11.6 / EN ##########
# 语言标记可省（省了就按段标题无标记处理，正文推断语言）
_SEG_RE = re.compile(
    r"^\s*#{3,}\s*(?P<id>S?\d+[A-Za-z]?)\s*/\s*(?P<dur>[^/#]*?)"
    r"(?:\s*/\s*(?P<lang>[^#/]*?))?\s*#{3,}\s*$")
# [1] 中文版 / CHINESE VERSION
_SEC_RE = re.compile(r"^\s*\[(?P<no>[12])\]\s*(?P<name>.*?)\s*$")
_FIELD_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<rest>.*)$")
_PIC_RE = re.compile(r"<\s*Picture\s*(\d+)\s*>", re.I)
_VID_RE = re.compile(r"<\s*Video\s*(\d+)\s*>", re.I)
_SUB_RE = re.compile(r"<\s*Subject\s*(\d+)\s*>", re.I)
_SPEAKER_RE = re.compile(r"\(S(\d+)\)")
_DIALOG_RE = re.compile(r"<d>(.*?)</d>", re.S)
# split 用：奇数位就是完整的 <d>…</d> 台词块（整块跳过，不改写）
_DIALOG_SPLIT_RE = re.compile(r"(<d>.*?</d>)", re.S)
# 前后不能用 \b：Python 里汉字算 \w，"约00:01.600" 的 0 前面紧贴汉字就没有
# 词界，整句时间码会被漏掉（校验失真的老毛病）。改用「前后不能是数字/冒号/点」。
_TIMECODE_RE = re.compile(
    r"(?<![\d:.])(\d{2}):(\d{2})(?:\.(\d{1,3}))?(?![\d:.])")
_CJK_RE = re.compile(r"[一-鿿　-〿＀-￯]")
_QUOTE_RE = re.compile(r"[\"“”]")
# 英文单版文件结尾的 END OF PROMPTS
_ENDPROMPTS_RE = re.compile(r"^\s*END\s+OF\s+PROMPTS\s*$", re.I)
# STYLE / CAMERA / LIGHTING block
_STYLEBLOCK_RE = re.compile(r"STYLE\s*/\s*CAMERA\s*/\s*LIGHTING", re.I)
# 英文词计数
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# ---------------------------------------------------------------------------
# [Shot N] 分镜脚本（六段式 + 镜内时间码，另一种常见交付形态）
# ---------------------------------------------------------------------------
# [Shot 1] At 00:00.000, "破门惊鸳鸯"——节奏：急起、骤顿、转冷。
_SHOT_HEAD_RE = re.compile(
    r"^\s*\[\s*Shot\s*(?P<no>\d+)\s*\]\s*(?P<rest>.*)$", re.I | re.M)
# mm:ss.xxx。比 _TIMECODE_RE 严：前面不能是数字/点/冒号，避免把 "1.2"
# 这类小数或比例误判成时间码。
_SHOT_CLOCK_RE = re.compile(r"(?<![\d.:])(\d{1,2}):([0-5]\d)\.(\d{1,3})(?!\d)")
# 台词行补 (Sx) 用的说话人提示。**必须**紧跟"以/用/开/说"这类动词：
# "S1以中低音区说话" 才是真的在标说话人；"S1 的侧影只作画框边缘存在"里的
# S1 只是个路人，照它补会把张三的台词标成 (S1)。宁可不补，不能补错。
_SPEAKER_HINT_RE = re.compile(
    r"(?<![\w<])S([1-9])(?=(?:的声音|的嗓音|的音色|以|用|开|说|问|答|接|念"
    r"|唱|讲|喊|吐|低声|开口|发声))")


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------
def split_sections(text):
    """把 PACK 切成语言分块。

    返回 ``[(lang, body), ...]``；``lang`` 是 ``"zh"`` / ``"en"`` / ``None``
    （None = 没有 [1]/[2] 分块，逐段按标题自带的语言标记判断）。
    """
    lines = str(text or "").splitlines()
    marks = [(i, _lang_of_section(m.group("name")))
             for i, line in enumerate(lines)
             for m in [_SEC_RE.match(line)] if m]
    if not marks:
        return [(None, "\n".join(lines))]
    out = []
    for idx, (start, lang) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        body = "\n".join(lines[start + 1:end])
        if body.strip():
            out.append((lang, body))
    return out


def _lang_of_section(name):
    text = str(name or "")
    low = text.lower()
    if "中文" in text or "chinese" in low:
        return "zh"
    if "英文" in text or "english" in low or re.search(r"\ben\b", low):
        return "en"
    return None


def _lang_of_tag(tag):
    """段标题里的语言标记：中文 / EN / 英文。"""
    text = str(tag or "")
    low = text.lower()
    if "中文" in text or "chinese" in low:
        return "zh"
    if "英文" in text or "english" in low or re.search(r"\ben\b", low):
        return "en"
    return None


def parse_duration(spec, default=MAX_NEW_SECONDS):
    """``10s`` → (10, 0, 10)；``10s+1.6=11.6`` → (10, 1.6, 11.6)。"""
    text = str(spec or "").strip()
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return float(default), 0.0, float(default)
    new = float(numbers[0])
    if len(numbers) >= 3:
        return new, float(numbers[1]), float(numbers[2])
    if len(numbers) == 2:                      # "10+1.6" 只给了新增+回放
        return new, float(numbers[1]), new + float(numbers[1])
    return new, 0.0, new


def frames_for_seconds(seconds):
    """规范接线表的帧数口径：``帧数 = 生成时长 × fps``（10s=241，11.6s=279）。

    注意这是**规范口径**，不是 H3 真正生成的帧数——H3 只能出 17k+5 的数
    （5/22/39/…/243/…），Director 会把这个值再向上对齐到网格。
    """
    return int(round(float(seconds) * FPS)) + 1


def _infer_lang(body):
    """段标题没写语言标记时，从正文推断：台词块之外有中文就是中文版。"""
    text = _DIALOG_RE.sub("", str(body or ""))
    return "zh" if _CJK_RE.search(text) else "en"


def _word_count(text):
    return len(_WORD_RE.findall(str(text or "")))


def split_fields(body):
    """把一段正文切成 ``{字段名: 内容}``；只认官方六段字段名。"""
    fields = {}
    current = None
    buf = []

    def flush():
        if current is not None:
            fields[current] = "\n".join(buf).strip()

    for line in str(body or "").splitlines():
        stripped = line.strip()
        if _SEG_RE.match(stripped) or _SEP_RE.match(stripped):
            break                                # 下一段 / 分块结束
        match = _FIELD_RE.match(stripped)
        if match and match.group("name") in FIELD_SET:
            flush()
            current = match.group("name")
            buf = [match.group("rest").strip()] if match.group("rest").strip() else []
            continue
        if current is not None:
            buf.append(line)
    flush()
    return fields


def _seg_ordinal(seg_id) -> int | None:
    """把 ``S01`` / ``1`` / ``1a`` 之类段标题 id 归一成从 1 起的整数序号。"""
    m = re.match(r"[^\d]*(\d+)", str(seg_id or ""))
    return int(m.group(1)) if m else None


def _format_segment_fields(fields) -> list:
    """把六字段 dict 渲染成段落正文行（供 rebuild_pack 使用）。

    只输出 fields 里确实存在的字段，顺序跟 FIELD_ORDER 一致；字段间空一行，
    保留官方 ``name:\\n<value>`` 形态。
    """
    lines: list = []
    first = True
    for name in FIELD_ORDER:
        value = fields.get(name)
        if value is None or str(value).strip() == "":
            continue
        if not first:
            lines.append("")
        lines.append("%s:" % name)
        lines.extend(str(value).rstrip("\n").splitlines())
        first = False
    if not first:
        lines.append("")
    return lines


def _lang_block_range(lines, lang):
    """返回 ``[1] 中文版`` / ``[2] 英文版`` 分块的行范围 ``(lo, hi)``。

    ``lo`` 是分块标题行的下一行，``hi`` 是下一个分块标题（或文件末尾）。
    找不到对应语言分块时返回 ``(None, None)``；单语 PACK（无语言分块）也返回
    ``(None, None)`` —— 调用方据此退回「全篇定位段头」的默认行为。
    """
    marks = []  # [(start_line, lang)]
    for i, line in enumerate(lines):
        m = _SEC_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name") or ""
        low = name.lower()
        if "中文" in name:
            marks.append((i, "zh"))
        elif "英文" in name or re.search(r"\ben\b", low):
            marks.append((i, "en"))
    for idx, (start, l) in enumerate(marks):
        if l == lang:
            hi = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
            return (start + 1, hi)
    return (None, None)


def rebuild_pack(original: str, segments, lang=None) -> str:
    """把「编辑后的六字段」回写进原 PACK 文本。

    ``original`` 是工作流里那份完整 PACK（含表头 / ``##########`` 段标题 /
    ``END OF PROMPTS``）。``segments`` 是 ``[{"index": 序号, "fields": {...}}, ...]``
    只改其中命中索引的段，其余段**原样保留**（正文、时间、参考图、台词都不动）。
    返回重拼后的完整 PACK 文本 —— 前端拿它回写外侧 PrimitiveStringMultiline 文本框。

    ``lang`` 控制命中哪一侧语言版块（双语 PACK 才有意义）：

    * ``None``（默认）—— 命中**最后一次出现**的段标题。双语 PACK 里中英同
      序号段头重复出现，取最后一次即英文版；行为与旧版一致，Director 直接吃。
    * ``"en"`` —— 只命中 ``[2] 英文版`` 分块内的段标题。
    * ``"zh"`` —— 只命中 ``[1] 中文版`` 分块内的段标题。

    边界：
    * 表头（第一个 ``##########`` 之前）与结尾 ``END OF PROMPTS`` 原样保留。
    * 被编辑的段会重建其字段块（只保留到下一段/``====``/``END OF PROMPTS``），
      编辑后缺失的字段会在该段里去掉，新增的加在末尾，顺序按 FIELD_ORDER。
    * 双语 PACK 的 ``[1] 中文版`` / ``[2] 英文版`` 分块：``lang`` 指定后只在该
      侧版块内定位段头，另一侧版块完全不动；``lang=None`` 退回最外层逻辑。
    """
    lines = str(original or "").split("\n")
    # 双语 PACK + 指定语言版块：只在该版块行范围内定位段标题
    if lang in ("zh", "en"):
        lo, hi = _lang_block_range(lines, lang)
        if lo is not None:
            heads = [(i, m.group("id"))
                     for i in range(lo, hi)
                     for m in [_SEG_RE.match(lines[i].strip())] if m]
        else:
            # 该语言分块不存在（单语 PACK 被误传 lang）—— 退回全篇定位，不报错
            heads = [(i, m.group("id"))
                     for i, line in enumerate(lines)
                     for m in [_SEG_RE.match(line.strip())] if m]
    else:
        heads = [(i, m.group("id"))
                 for i, line in enumerate(lines)
                 for m in [_SEG_RE.match(line.strip())] if m]
    ordered = []
    for i, seg_id in heads:
        ordinal = _seg_ordinal(seg_id)
        if ordinal is not None:
            ordered.append((ordinal, i))
    if not ordered:
        return original
    # 序号 → 段头行号（段标题可能重复，安全起见取最后一次）
    head_by_ord = {}
    for ordinal, i in ordered:
        head_by_ord[ordinal] = i
    edits = {}
    for seg in segments or []:
        try:
            ordinal = int(seg.get("index"))
        except (TypeError, ValueError):
            continue
        fields = seg.get("fields")
        if isinstance(fields, dict):
            edits.setdefault(ordinal, fields)

    spans = []  # (start, end, new_block_lines)
    ords = sorted(head_by_ord)
    for k, ordinal in enumerate(ords):
        if ordinal not in edits:
            continue
        start = head_by_ord[ordinal] + 1          # 段标题行之后
        end = head_by_ord[ords[k + 1]] if k + 1 < len(ords) else len(lines)
        blk_end = end
        for i in range(start, end):
            stripped = lines[i].strip()
            if _SEP_RE.match(stripped) or _ENDPROMPTS_RE.match(stripped):
                blk_end = i
                break
        spans.append((start, blk_end, _format_segment_fields(edits[ordinal])))

    result = lines[:]
    for start, end, block in sorted(spans, key=lambda s: -s[0]):
        # 用重建块替换原字段区；块内已自带段间空行
        result[start:end] = block
    return "\n".join(result).strip("\n") + "\n"


def rebuild_pack_block(original: str, blocks, lang=None) -> str:
    """把「整段正文」原样替换进原 PACK 文本（不拆字段、不规范化）。

    ``blocks`` 是 ``[{"index": 序号, "body": "整段六段式正文"}, ...]``。

    与 ``rebuild_pack`` 的区别：``rebuild_pack`` 会把六字段拆开、按 FIELD_ORDER
    重排、丢掉空字段；这里**原样替换**段头之后到下一段头 / ``====`` /
    ``END OF PROMPTS`` 之前的内容 —— 用户写成什么样，回写就是什么样。

    适合「一个文本框装整段六段式提示词」的编辑场景：前端把文本框全文当 ``body``
    发上来即可，不必在前端复刻字段切分逻辑，保真度最高。

    ``lang`` 语义与 ``rebuild_pack`` 完全一致（``None``/``"en"``/``"zh"``）。
    """
    lines = str(original or "").split("\n")
    if lang in ("zh", "en"):
        lo, hi = _lang_block_range(lines, lang)
        scan = range(lo, hi) if lo is not None else range(len(lines))
    else:
        scan = range(len(lines))
    heads = []
    for i in scan:
        m = _SEG_RE.match(lines[i].strip())
        if not m:
            continue
        ordinal = _seg_ordinal(m.group("id"))
        if ordinal is not None:
            heads.append((ordinal, i))
    if not heads:
        return original
    # 序号 → 段头行号（重复段头取最后一次，与 rebuild_pack 一致）
    head_by_ord = {}
    for ordinal, i in heads:
        head_by_ord[ordinal] = i
    edits = {}
    for blk in blocks or []:
        try:
            ordinal = int(blk.get("index"))
        except (TypeError, ValueError):
            continue
        body = blk.get("body")
        if isinstance(body, str):
            edits.setdefault(ordinal, body)

    spans = []  # (start, end, new_block_lines)
    ords = sorted(head_by_ord)
    for k, ordinal in enumerate(ords):
        if ordinal not in edits:
            continue
        start = head_by_ord[ordinal] + 1
        end = head_by_ord[ords[k + 1]] if k + 1 < len(ords) else len(lines)
        blk_end = end
        for i in range(start, end):
            stripped = lines[i].strip()
            if _SEP_RE.match(stripped) or _ENDPROMPTS_RE.match(stripped):
                blk_end = i
                break
        body_lines = str(edits[ordinal]).rstrip("\n").splitlines()
        # 前后各留一空行，保持段间视觉分隔（段头行本身不动）
        spans.append((start, blk_end, ["", *body_lines, ""]))

    result = lines[:]
    for start, end, block in sorted(spans, key=lambda s: -s[0]):
        result[start:end] = block
    return "\n".join(result).strip("\n") + "\n"


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def parse_pack(text, language="auto", default_duration=MAX_NEW_SECONDS,
               auto_fix=True):
    """解析整个 PACK。

    返回 ``(segments, meta, issues)``。``segments`` 每项是一个 dict：
    ``id / lang / new_seconds / handoff_seconds / gen_seconds / hard_cut /
    fields / pictures / videos / subjects / speakers / task / issues``。

    ``meta["layout"]`` 标出认出来的文件形态：

    * ``en-single``       — 现行规范：只有英文版六段，无头部、无 [1]/[2]，
                            结尾 ``END OF PROMPTS``。
    * ``bilingual-pack``  — 老的 PACK：有 ``[1] 中文版`` / ``[2] 英文版`` 分块。
    * ``pack-single``     — 有 ``=====`` 头部但没有语言分块。
    """
    raw = str(text or "")
    issues = []
    meta = _parse_header(raw)

    sections = split_sections(raw)
    chosen = _pick_section(sections, language)
    if chosen is None:
        return [], meta, ["没有解析到任何分镜内容：请粘贴分镜提示词文本。"]
    lang, body = chosen

    has_seg_head = any(_SEG_RE.match(line.strip())
                       for line in body.splitlines())
    is_shot_script = bool(_SHOT_HEAD_RE.search(body)) and not has_seg_head
    if any(sec_lang for sec_lang, _ in sections):
        layout = "bilingual-pack"
    elif is_shot_script:
        layout = SHOT_SCRIPT_LAYOUT
    elif meta:
        layout = "pack-single"
    else:
        layout = "en-single"
    meta["layout"] = layout
    meta["rule_version"] = RULE_VERSION

    # 配乐判据的扫描范围有两条硬约束：
    #   1. 不扫 non_diegetic_music 自己写的 MUSIC block（那里必然出现 BGM/music，
    #      扫了会自我命中）；
    #   2. 不扫头部——老 PACK 头部有「Music / 配乐 : N/A」，自带「配乐」两个字，
    #      扫进去会让每一份 PACK 都误判成命中配乐。
    scan = _strip_music_field(body)
    scan_lines = scan.splitlines()
    for i, line in enumerate(scan_lines):                 # 跳到第一个段标题
        if _SEG_RE.match(line.strip()):
            scan_lines = scan_lines[i:]
            break
    cut = len(scan_lines)
    for i, line in enumerate(scan_lines):                 # END OF PROMPTS 之后不算
        if _ENDPROMPTS_RE.match(line.strip()):
            cut = i
            break
    scan = "\n".join(scan_lines[:cut])
    music_hit = list(detect_music_trigger(scan))
    head_music = (meta.get("Music") or meta.get("配乐") or "").strip()
    if head_music and "N/A" not in head_music.upper():
        if "头部声明" not in music_hit:
            music_hit.append("头部声明")
    music_hit = tuple(music_hit)
    meta["music_hit"] = music_hit
    # 风格全片只定一次（规范：启用时全片口径一致）
    music_style = detect_music_style(scan)
    meta["music_style"] = music_style

    blocks = _split_segments(body, lang)
    if not blocks:
        # 没有 ### 段标题：再试 [Shot N] 分镜脚本（时长从镜内时间码推）
        blocks = parse_shot_script_blocks(body, lang, MAX_NEW_SECONDS)
    if not blocks:
        # 都没有：整块当一段（单段提示词直接粘贴的情况）
        fields = split_fields(body)
        if fields:
            blocks = [("S01", "", lang, body, None)]
    if not blocks:
        return [], meta, ["没有找到分镜段。段标题形如 "
                          "「########## S01 / 10s / EN ##########」、"
                          "「########## S02 / 10s+1.6=11.6 / EN ##########」，"
                          "或镜内标记「[Shot 1] At 00:00.000」。"]

    segments = []
    for index, item in enumerate(blocks):
        seg_id, dur_spec, seg_lang, seg_body = item[0], item[1], item[2], item[3]
        replay_in = item[4] if len(item) > 4 else None
        seg = _build_segment(seg_id, dur_spec, seg_lang or lang, seg_body,
                             default_duration, index == 0, replay_in=replay_in)
        if auto_fix:
            seg["issues"].extend(_auto_fix(seg, music_hit=music_hit,
                                           music_style=music_style))
        seg["issues"].extend(validate_segment(seg, is_last=index == len(blocks) - 1,
                                              music_hit=music_hit))
        if layout == SHOT_SCRIPT_LAYOUT:
            seg["music_per_segment"] = True
        seg["task"] = infer_task(seg)
        segments.append(seg)
        issues.extend("%s：%s" % (seg["id"], item) for item in seg["issues"])

    # 跨段一致性（规范 4：no-text 句 / STYLE 块 / MUSIC 块全链逐字复用）
    issues.extend(validate_chain(segments))
    return segments, meta, issues


def _split_segments(body, fallback_lang):
    """按段标题切块 → ``[(id, dur_spec, lang, body, replay_in), ...]``。

    第 5 项 ``replay_in`` 只有 [Shot N] 脚本路径会填（段首回放与尾部窗口要
    分开算）；段标题路径一律 ``None``，表示「按 dur_spec 里的 +1.6 推」。
    """
    lines = str(body or "").splitlines()
    heads = [(i, m) for i, line in enumerate(lines)
             for m in [_SEG_RE.match(line)] if m]
    if not heads:
        return []
    out = []
    for idx, (start, match) in enumerate(heads):
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines)
        out.append((match.group("id").strip(),
                    match.group("dur").strip(),
                    _lang_of_tag(match.group("lang")) or fallback_lang,
                    "\n".join(lines[start + 1:end]),
                    None))
    return out


# ---------------------------------------------------------------------------
# [Shot N] 分镜脚本 → 段
#
# 形态::
#
#     subject_definitions:
#     ...（全局）
#     summary:
#     ...
#     retention_analysis:
#     ...
#     detailed_description:
#     [Shot 1] At 00:00.000, "破门惊鸳鸯"——...
#     ...
#     [Shot 2] At 00:10.000, ...
#     overall_soundscape:
#     ...
#     non_diegetic_music:
#     ...
#
# 与 ``########## S01 / 10s / EN ##########`` 的关键差别：段长**不写在标题
# 里**，要从时间码推。一个 [Shot N] 常常远超单段上限（10s），所以要按镜内
# 出现的时间码再切一刀——这就是以前「整份脚本被当成一段」的根因。
# ---------------------------------------------------------------------------
SHOT_SCRIPT_LAYOUT = "shot-script"
# 切出来的碎段别比这还短，宁可让前一段略微超长
_MIN_SPLIT_SECONDS = 2.0


def _clock_seconds(m, s, frac):
    value = int(m) * 60 + int(s)
    if frac:
        value += int(frac) / (10.0 ** len(frac))
    return value


def _fmt_clock(seconds):
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    frac = int(round((seconds - m * 60 - s) * 1000))
    if frac >= 1000:
        frac -= 1000
        s += 1
        if s >= 60:
            s -= 60
            m += 1
    return "%02d:%02d.%03d" % (m, s, frac)


def _retime_line(line, t0, t1, handoff):
    """把行里的**成片绝对**时间码搬进段内坐标系。

    段开头 ``handoff`` 秒是上一段结尾的回放，所以段内 0 秒对应的成片时刻是
    ``t0 - handoff``；段内容本身占 ``[t0, t1]``，映射后落在
    ``[handoff, handoff + (t1 - t0)]`` —— 上限正好等于本段 gen_seconds，
    校验里那句「时间码超出生成时长」就不会再被误触发。
    落在段外的时间码一律 clamp 到段边界（"必须于 00:17.0 前说完"这种跨段
    的收尾要求，收紧到本段末尾即可，语义不变）。
    """
    def _sub(match):
        value = _clock_seconds(*match.groups())
        value = min(max(value, t0), t1)
        return _fmt_clock(value - t0 + handoff)
    return _SHOT_CLOCK_RE.sub(_sub, line)


def _tag_speakers(detail):
    """台词行缺 (Sx) 编号时，从同一行的 S1/S2 提示补一个。

    规范 1.2 写死了顺序是 ``(S1) <d>[Chinese] 原文</d>``：编号在台词块**前
    面**。以前补在后面（``<d>…</d> (S1)``），模型容易把它读成"这句说完才轮
    到 S1"，而不是"这句是 S1 说的"。
    """
    out = []
    for line in str(detail or "").splitlines():
        if "<d>" in line and not _SPEAKER_RE.search(line):
            head = line[:line.index("<d>")]
            hint = _SPEAKER_HINT_RE.search(head) or _SPEAKER_HINT_RE.search(line)
            if hint:
                line = re.sub(r"(<d>.*?</d>)",
                              r"(S%s) \1" % hint.group(1), line, count=1)
        out.append(line)
    return "\n".join(out)


def _shot_global_fields(text):
    """[Shot N] 之外的全局字段（六段里除 detailed_description 的其余五段）。"""
    fields = {}
    current = None
    in_shots = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if _SHOT_HEAD_RE.match(stripped):
            in_shots = True
            continue
        match = _FIELD_RE.match(stripped)
        if match and match.group("name") in FIELD_SET:
            current = match.group("name")
            # detailed_description 的正文就是那一串 [Shot N]，不进全局
            in_shots = current == "detailed_description"
            fields.setdefault(current, [])
            rest = match.group("rest").strip()
            if rest:
                fields[current].append(rest)
            continue
        if current is None or in_shots:
            continue
        fields[current].append(line)
    return {k: "\n".join(v).strip() for k, v in fields.items()
            if k != "detailed_description"}


def _shot_blocks(text):
    """按 [Shot N] 切块 → ``[(no, rest, lines), ...]``。

    **标题行自带的正文必须收进 lines。** 分镜脚本的写法是
    ``[Shot 1] At 00:00.000, "破门惊鸳鸯"——……开场静帧：ELS 从门外望向卧房
    深处……`` 一整段开场建立描述全压在标题那一行里，只把 ``rest`` 拿去取
    开始时间码、正文行另起收集的话，每个镜头最要紧的开场（同时也是接续段
    回放要继承的姿态来源）会被整段丢掉——表现就是「转换完跟原剧本不一样」。
    """
    blocks = []
    current = None
    in_detail = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        match = _SHOT_HEAD_RE.match(stripped)
        if match:
            rest = match.group("rest").strip()
            current = (int(match.group("no")), rest, [rest] if rest else [])
            blocks.append(current)
            continue
        field = _FIELD_RE.match(stripped)
        if field and field.group("name") in FIELD_SET:
            in_detail = field.group("name") == "detailed_description"
            if not in_detail:
                current = None                # 走出 detailed_description
            continue
        if current is not None and in_detail:
            current[2].append(line)
    return blocks


def _shot_bounds(blocks):
    """每个 Shot 在成片时间线上的 ``[start, end)``。"""
    starts = []
    for _, rest, _lines in blocks:
        match = _SHOT_CLOCK_RE.search(rest or "")
        starts.append(_clock_seconds(*match.groups()) if match else None)

    ends = []
    for i, (_no, _rest, lines) in enumerate(blocks):
        nxt = next((s for s in starts[i + 1:] if s is not None), None)
        if nxt is not None:
            ends.append(nxt)
            continue
        codes = [_clock_seconds(*g)
                 for line in lines for g in _SHOT_CLOCK_RE.findall(line)]
        ends.append(max(codes) if codes else None)

    for i, value in enumerate(starts):
        if value is None:
            starts[i] = (ends[i - 1] if i and ends[i - 1] is not None
                         else max(0.0, (ends[i] or 0.0) - MAX_NEW_SECONDS))
    for i, value in enumerate(ends):
        if value is None:
            ends[i] = starts[i] + MAX_NEW_SECONDS
    return starts, ends


def _talk_spans(lines):
    """每句台词在成片时间线上占的区间 ``[(start, end), ...]``。

    台词行的时间码写法是「自 00:12.0 起说……于 00:17.0 前说完」，取这一行的
    最小/最大时间码就是这句台词占的时间。切段时要用它做两件事：别把缝切在
    一句话的中间（规范 1.4：段尾 1.6 秒必须是无台词区，回放过去就是重复语音）。
    """
    spans = []
    for line in lines or ():
        if "<d>" not in line:
            continue
        codes = [_clock_seconds(*g) for g in _SHOT_CLOCK_RE.findall(line)]
        if codes:
            spans.append((min(codes), max(codes)))
    return spans


def _plan_splits(t0, t1, candidates, max_new=MAX_NEW_SECONDS, talk_spans=(),
                 head_codes=None):
    """把一个超长 Shot 切成 ≤max_new 的若干段，切点尽量落在已有时间码上。

    均分优先（n = ceil(总长 / 上限)），每段理想边界附近挑一个真实存在的时间
    码——落在剧情既有节点上，接缝才不会切在动作中间。

    在此之上再叠两条**软**约束（规范 1.4），代价高但不硬性排除，免得无解：

    * 切点落在一句台词进行中 → 重罚（一句话被劈成两段，两边都说半句）；
    * 切点往前 1.6 秒里还有台词没说完 → 重罚（尾窗要留给下一段回放，
      带台词回放＝重复语音）。规范原话：先定台词落点，再让该段至少晚 1.6
      秒收尾。
    """
    total = t1 - t0
    if total <= max_new + 0.001:
        return [t0, t1]
    n = int(math.ceil(total / float(max_new) - 1e-9))
    inner = sorted({round(c, 3) for c in candidates
                    if t0 + _MIN_SPLIT_SECONDS <= c <= t1 - _MIN_SPLIT_SECONDS})
    spans = [(float(a), float(b)) for a, b in (talk_spans or ())]

    # 切点前后都得有正文行：一个窗口里要是连一行指令都没有，那段时间就会从
    # 成片里凭空消失（40 秒的剧本切完只剩 36 秒就是这么来的）。所以候选只
    # 保留「后面至少还有一行」的那些。
    if head_codes:
        lo, hi = min(head_codes), max(head_codes)
        inner = [c for c in inner if lo + 1e-6 < c <= hi + 1e-6]
    if not inner:
        return [t0, t1]

    def _cost(c, ideal):
        cost = abs(c - ideal)
        for start, end in spans:
            if start + 1e-6 < c < end - 1e-6:
                cost += 100.0                      # 切在台词中间
            if start < c - HANDOFF_SECONDS < end - 1e-6:
                cost += 60.0                       # 尾窗里还有台词
        return cost

    points = [t0]
    for i in range(1, n):
        if not inner:
            break
        ideal = t0 + total * i / n
        cut = min(inner, key=lambda c: (_cost(c, ideal), c))
        inner = [c for c in inner if abs(c - cut) > 1e-6]
        points.append(cut)
    points.append(t1)
    return sorted(set(points))


def _split_music_lines(music_text, t0, t1, handoff):
    """按本段的时间窗切出对应配乐条目，并把时间码搬进段内坐标系。

    全片配乐表是逐条带时间码的（``00:12.0-00:17.0 三弦接棒…``），整份塞进
    每一段等于告诉模型"这一 7 秒里要演完 40 秒的音乐"。没时间码的概述行
    （``全片配乐 = 京剧打击乐…``）描述的是风格，每段都留着。
    """
    out = []
    plain = []
    for line in str(music_text or "").splitlines():
        if not line.strip():
            continue
        codes = [_clock_seconds(*g) for g in _SHOT_CLOCK_RE.findall(line)]
        if not codes:
            # 没有时间码的是全片风格概述（"全片配乐 = 京剧打击乐…"），每段都留
            plain.append(line)
            out.append(line)
            continue
        # 只在本段窗口内留有余量的条目才收；正好卡在边界上的（上一段的收尾
        # "00:08.7-00:10.0"）会被 clamp 成 00:01.6-00:01.6 这种空区间，丢掉
        if max(codes) <= t0 + 1e-6 or min(codes) >= t1 - 1e-6:
            continue
        out.append(_retime_line(line, t0, t1, handoff))
    if not out:
        # 本段窗口里一条带时间码的配乐都没有：宁可只给风格概述，也不要回退成
        # 整份全片配乐表（那样等于告诉模型"这 6 秒里要演完 40 秒的音乐"）。
        out = plain
    return "\n".join(out).strip()


def _shot_seg_body(globals_fields, detail, a, b, idx, total, music="", hard=False):
    """拼回官方六段正文：全局字段复用，detailed_description 用本段切片。"""
    parts = []
    for name in FIELD_ORDER:
        if name == "detailed_description":
            value = detail
        elif name == "non_diegetic_music":
            value = music or (globals_fields.get(name) or "").strip()
        elif name == "summary":
            base = (globals_fields.get("summary") or "").strip()
            span = ("本段为全片第 %d/%d 段，画面覆盖 %s–%s。"
                    % (idx, total, _fmt_clock(a), _fmt_clock(b)))
            if hard:
                # 让 _build_segment 的 HARD CUT 判定能命中：硬切段不继承上一段
                # 的机位与姿态，也不给下一段留回放锚点。
                span = "HARD CUT " + span
            value = (base + "\n" + span).strip() if base else span
        else:
            value = (globals_fields.get(name) or "").strip()
        parts.append("%s:\n%s" % (name, value))
    return "\n\n".join(parts)


def parse_shot_script_blocks(text, fallback_lang=None, max_new=MAX_NEW_SECONDS):
    """[Shot N] 分镜脚本 → ``[(id, dur_spec, lang, body, replay_in), ...]``。

    与 ``_split_segments`` 同构，``_build_segment`` / ``_auto_fix`` /
    ``validate_segment`` 全部原样复用。没有 [Shot N] 时返回 ``[]``，让调用
    方继续退回整块单段的老逻辑。

    **两个方向的手接窗口要分开算**（以前混成一个 ``handoff``，于是首段被
    写成「0 回放」、末段反而多背了 1.6 秒）：

    * ``replay_in`` — 段**首**回放上一段结尾的长度。首段没有上一段可回放
      所以是 0；其余各段（含末段）都是 1.6 秒。它决定段内时间码往哪搬、
      要不要补回放句、以及「回放区里不许有新台词」这条校验。
    * ``tail_out``  — 本段**尾**部留给下一段当锚点的长度。末段没有下一段
      所以是 0；其余各段（含首段）都是 1.6 秒 —— 首段也要留，否则 Director
      渲 seg_01 时不写 tail 文件，seg_02 拿不到 start_anchor，只能把第一张
      参考图当第一帧渲染（表现：分镜 2 开头先闪 1 秒参考图原图）。
    """
    raw = str(text or "")
    blocks = _shot_blocks(raw)
    if not blocks:
        return []
    globals_fields = _shot_global_fields(raw)
    starts, ends = _shot_bounds(blocks)
    lang = fallback_lang or _infer_lang(raw)

    # 先规划出所有段，拿到总数后再拼正文（summary 里要写「第 i/N 段」）
    plans = []
    for (_no, rest, lines), t_start, t_end in zip(blocks, starts, ends):
        if t_end <= t_start + 0.01:
            continue
        candidates = [_clock_seconds(*g) for g in _SHOT_CLOCK_RE.findall(rest or "")]
        head_codes = []
        for line in lines:
            codes = [_clock_seconds(*g) for g in _SHOT_CLOCK_RE.findall(line)]
            candidates.extend(codes)
            if codes:
                # 决定这一行归哪个桶的，是它**行首**那个时间码
                head_codes.append(_SHOT_CLOCK_RE.search(line)
                                  and _clock_seconds(
                                      *_SHOT_CLOCK_RE.search(line).groups()))
        head_codes = [c for c in head_codes if c is not None]
        points = _plan_splits(t_start, t_end, candidates, max_new,
                              talk_spans=_talk_spans(lines),
                              head_codes=head_codes)
        buckets = [[] for _ in range(len(points) - 1)]
        last = 0
        for line in lines:
            match = _SHOT_CLOCK_RE.search(line)
            if match:
                value = _clock_seconds(*match.groups())
                last = len(buckets) - 1
                for i in range(len(buckets)):
                    if value < points[i + 1] - 1e-6:
                        last = i
                        break
            buckets[last].append(line)
        # 空窗口（这段时间剧本没写新指令）必须并进相邻段：语义上是"延续上一段
        # 在做的动作"，直接丢掉会让成片少掉几秒。前一段优先，开头就空窗的话
        # 起点回退到本 Shot 开头，末尾空窗并进最后一段。
        spans_out = []
        for k, seg_lines in enumerate(buckets):
            if not seg_lines:
                if spans_out:
                    spans_out[-1][1] = points[k + 1]
                continue
            start = points[0] if not spans_out else spans_out[-1][1]
            spans_out.append([start, points[k + 1], seg_lines])
        if spans_out:
            spans_out[-1][1] = points[-1]
        for a, b, seg_lines in spans_out:
            plans.append((a, b, seg_lines))

    total = len(plans)
    out = []
    for index, (a, b, seg_lines) in enumerate(plans):
        new = round(b - a, 3)
        is_first = index == 0
        is_last = index == total - 1
        replay_in = 0.0 if is_first else HANDOFF_SECONDS
        tail_out = 0.0 if is_last else HANDOFF_SECONDS
        hard = bool(re.search(r"HARD\s*CUT", "\n".join(seg_lines), re.I))
        if hard:
            replay_in = tail_out = 0.0
        # 生成时长 = 新增 + 段首回放。**尾部锚点 tail_out 不额外占生成时长**：
        # 它就是本段新增内容最后那 1.6 秒（规范 1.4 要求这段无台词），渲完从
        # 结果里切出来给下一段当锚点即可。以前首段在这里多加了一个 tail_out，
        # 等于让模型凭空多编 1.6 秒剧本外的内容 —— 成片表现为首段结尾突然
        # 冒出剧本里没有的动作或声音。
        gen = round(new + replay_in, 3)
        # tail_out 仍然算出来：HARD CUT 判定和段标题语义都还用得上，只是不再
        # 加进生成时长。
        extra = gen - new
        dur_spec = ("%gs" % new) if extra <= 0.001 else (
            "%gs+%.3f=%gs" % (new, extra, gen))
        detail = "\n".join(_retime_line(x, a, b, replay_in)
                           for x in seg_lines).strip()
        music = _split_music_lines(globals_fields.get("non_diegetic_music"),
                                   a, b, replay_in)
        out.append(("S%02d" % (index + 1), dur_spec, lang,
                    _shot_seg_body(globals_fields, _tag_speakers(detail),
                                   a, b, index + 1, total, music=music,
                                   hard=hard),
                    replay_in))
    return out


def _pick_section(sections, language):
    """按 language 选择要用的那一版。auto = 英文版优先（官方六段是英文）。"""
    if not sections:
        return None
    want = {"zh": "zh", "en": "en"}.get(str(language or "").strip().lower())
    if want is None and "中文" in str(language or ""):
        want = "zh"
    if want is None and ("英文" in str(language or "") or "EN" in str(language or "")):
        want = "en"
    if want:
        for lang, body in sections:
            if lang == want:
                return lang, body
    if want is None:                            # auto
        for lang in ("en", "zh", None):
            for sec_lang, body in sections:
                if sec_lang == lang:
                    return sec_lang, body
    return sections[0]


def _parse_header(raw):
    """PACK 头部的 Project / Total duration / Segments / Music 等。

    头部形如::

        ==========================================================
        MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包
        Project / 项目         : 曹贼的性价比
        Total duration / 总时长 : 00:20
        ==========================================================

    即：从第一条分隔线之后开始扫，遇到下一条分隔线 / 段标题结束。
    双语键 ``Project / 项目`` 会同时登记英文与中文两个 key，方便两边取。

    en 单版规范**没有头部**，这时直接返回 ``{}``，项目名由调用方兜底。
    新版头部还可能有 ``Mode`` / ``Aspect`` / ``Version`` / ``Date``，
    走同一套 alias 机制，不需要单独分支。
    """
    head = {}
    lines = str(raw or "").splitlines()[:60]
    start = 0
    for i, line in enumerate(lines):                 # 跳过开场的 ==== 分隔线
        if _SEP_RE.match(line.strip()):
            start = i + 1
            break
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if _ENDPROMPTS_RE.match(stripped):
            break
        if _SEP_RE.match(stripped) or _SEG_RE.match(stripped) or stripped.startswith("#"):
            break
        key, sep, value = _split_kv(line)
        if not sep or not key:
            continue
        # 没有 ===== 头部时（[Shot N] 脚本 / en 单版），前 60 行里就是正文：
        # 官方六段字段名、<Picture N> 占位行、「取景纪律：…」这种全角冒号长
        # 纪律、甚至「00:00.000」这种时间码，都会被 _split_kv 当成
        # key: value 抓进来污染 meta（进而污染会话名与报告）。这里只认真正
        # 的头部键。
        aliases = [a for a in _header_aliases(key) if _is_header_key(a)]
        if not aliases:
            continue
        if len(key) > 40 or len(value) > 200:
            continue
        for alias in aliases:
            head.setdefault(alias, value)
    return head


# 头部键白名单（小写）。只有命中的才算头部 —— 用精确匹配而不是「包含某词」，
# 否则「镜头时长：00:10.000」也会被当成头部键。
_HEADER_KEYS = frozenset((
    "project", "项目", "total duration", "总时长", "总长", "duration", "时长",
    "segments", "段数", "music", "配乐", "mode", "模式", "aspect", "画幅",
    "version", "版本", "date", "日期", "resolution", "分辨率", "fps", "帧率",
    "author", "作者",
))


def _is_header_key(key):
    low = str(key or "").strip().lower()
    if not low or low in FIELD_SET or "<" in low or ">" in low:
        return False
    return low in _HEADER_KEYS


def _split_kv(line):
    """按半角 / 全角冒号拆 ``key: value``，返回 ``(key, sep, value)``。"""
    best = None
    for sep in (":", "："):
        pos = line.find(sep)
        if pos > 0 and (best is None or pos < best[0]):
            best = (pos, sep)
    if best is None:
        return "", "", ""
    pos, sep = best
    return line[:pos].strip(), sep, line[pos + len(sep):].strip()


def _header_aliases(key):
    """``Project / 项目`` → ``["Project", "项目"]``（去掉 ``/`` 与修饰后缀）。"""
    out = []
    for part in re.split(r"[/｜|]", key):
        part = part.strip()
        if part:
            out.append(part)
    return out or [key.strip()]


def _build_segment(seg_id, dur_spec, lang, body, default_duration, is_first,
                   replay_in=None):
    new, handoff, gen = parse_duration(dur_spec, default_duration)
    fields = split_fields(body)
    if not lang:
        # 段标题没写语言标记（en 单版里偶尔省）——从正文推断
        lang = _infer_lang(body)
    text_all = "\n".join(fields.get(name) or "" for name in FIELD_ORDER)
    summary = fields.get("summary") or ""
    hard_cut = bool(re.search(r"HARD\s*CUT", summary, re.I))
    # 旧版本在这里对首段做「handoff=0, gen=new」的覆盖，理由是「首段没有上
    # 一段可回放」——这话没讲错，但少想了一层：handoff 是 **本段尾部交给下
    # 段回放** 的窗口，跟「是不是首段」无关。首段被强行把 handoff 设成 0 后，
    # Director 渲出来的 seg_01 就只有 new 秒（10s）、不写 tail 文件，下一
    # 段 seg_02 的 start_anchor 变成 None，MiniMaxH3 ReferenceToVideo 没有
    # 锚点可用，只好把第一张参考图（P1 = 4-view 拼图）当成第一帧渲出来——
    # 表现为「分镜视频 2 开头就是参考图原图，持续 1s」。
    # 修正：首段也按段标题里的 `10s+1.6=11.6` 正常解析，把最后 1.6s 留给
    # seg_02 当开场锚点；每段对成片的实际贡献仍是 new 秒，总时长不变。
    #
    # 注意 handoff_seconds 在下游（Director / to_shots_json）的含义是「本段
    # 尾部留给下一段的锚点窗口」，而**段首回放**是另一件事 —— 首段没有上一
    # 段可回放，但它的尾巴照样要交给 seg_02。两者分开记在 replay_in 里，
    # 免得回放句被补到首段头上、时间码也整体平移 1.6 秒。
    if replay_in is None:
        replay_in = 0.0 if is_first else handoff
    if hard_cut:
        handoff = 0.0
        gen = new
        replay_in = 0.0
    return {
        "id": seg_id,
        "lang": lang,
        "new_seconds": new,
        "handoff_seconds": handoff,
        "replay_in": replay_in,
        "gen_seconds": gen,
        "spec_frames": frames_for_seconds(gen),
        "hard_cut": hard_cut,
        "fields": fields,
        "pictures": _picture_slots(fields),
        "videos": sorted({int(n) for n in _VID_RE.findall(text_all)}),
        "subjects": sorted({int(n) for n in _SUB_RE.findall(text_all)}),
        "speakers": sorted({int(n) for n in _SPEAKER_RE.findall(
            _DIALOG_RE.sub("", fields.get("detailed_description") or ""))}),
        "dialogues": len(_DIALOG_RE.findall(fields.get("detailed_description") or "")),
        "task": TASK_T2V,
        "issues": [],
    }


def _picture_slots(fields):
    """本段实际要接的 <Picture N> 槽位。

    **口径：提示词里写了 <Picture N>，就必须把第 N 张图接上。** 这是硬约束，
    不是优化项——送进模型的文本里如果出现一个没有对应图像的 ``<Picture N>``，
    H3 不会报错，它会自己"补"一个人出来，于是参考图看起来"没生效"：场景对了，
    人物是另一个人。之前这里只扫 ``retention_analysis`` +
    ``detailed_description``，而 ``subject_definitions``（拼进每一段提示词的
    角色定义块，正是 <Picture 1>/<Picture 2>/<Picture 4> 出现的地方）被排除
    在外，结果每段只绑了一张场景图，角色全靠模型自由发挥。

    所以改成扫**最终会送进模型的全部字段**。多接几张图的代价只是慢一点，
    少接的代价是人物换脸——这两者不对等。
    """
    live = "\n".join(str(v or "") for v in fields.values()
                     if isinstance(v, str))
    slots = {int(n) for n in _PIC_RE.findall(live)}
    return sorted(slots)


def pack_header(meta):
    """把 meta 里的头部键值整理成 PACK 模块要显示的五项 + 附加信息。

    返回统一的小写英文键，前端与 Director 的 pack_info 输出都吃这一份::

        {"project", "mode", "total_duration", "total_duration_s", "segments",
         "aspect", "music", "version", "date", "rule_version", "layout"}

    ``total_duration_s`` 是把 ``00:40`` / ``40s`` / ``40`` 统一成秒的浮点值，
    前端画时间线直接用；解析不出来就是 ``None``。
    """
    meta = meta or {}

    def _pick(*keys):
        for key in keys:
            value = meta.get(key)
            if value:
                return str(value).strip()
        return ""

    duration_raw = _pick("Total duration", "总时长", "总长", "Duration", "时长")
    return {
        "project": _pick("Project", "项目"),
        "mode": _pick("Mode", "模式"),
        "total_duration": duration_raw,
        "total_duration_s": _parse_duration_text(duration_raw),
        "segments": _pick("Segments", "段数"),
        "aspect": _pick("Aspect", "画幅", "Resolution", "分辨率"),
        "music": _pick("Music", "配乐"),
        "version": _pick("Version", "版本"),
        "date": _pick("Date", "日期"),
        "rule_version": meta.get("rule_version") or "",
        "layout": meta.get("layout") or "",
    }


def pack_info_json(segments, meta, extra=None):
    """PACK 模块的负载：头部五项 + 每段一行 + 参考图槽位。

    H3PromptPackParser 把它作为 ``pack_info`` 输出，H3Director 接过去原样转出
    并在运行报告里回显。前端的 PACK 面板读的是同一个结构（走
    ``/h3/pack_preview`` 拿实时解析结果，不用等一次渲染）。
    """
    header = pack_header(meta)
    segs = []
    slots = []
    for seg in segments or []:
        pics = list(seg.get("pictures") or [])
        for num in pics:
            if num not in slots:
                slots.append(num)
        fields = seg.get("fields") or {}
        body = str(fields.get("detailed_description") or "")
        segs.append({
            "id": seg.get("id"),
            "new_seconds": round(float(seg.get("new_seconds") or 0.0), 3),
            "handoff_seconds": round(float(seg.get("handoff_seconds") or 0.0), 3),
            "replay_in": round(float(seg.get(
                "replay_in", seg.get("handoff_seconds")) or 0.0), 3),
            "gen_seconds": round(float(seg.get("gen_seconds") or 0.0), 3),
            "frames": seg.get("spec_frames"),
            "task": seg.get("task"),
            "pictures": pics,
            "words": len(body.split()),
        })
    slots.sort()
    info = {
        "header": header,
        "segments": segs,
        "ref_slots": slots,
        "ref_slot_count": len(slots),
        "total_new_seconds": round(sum(s["new_seconds"] for s in segs), 3),
        "total_gen_seconds": round(sum(s["gen_seconds"] for s in segs), 3),
    }
    info.update(extra or {})
    return info


def _parse_duration_text(text):
    """``00:40`` / ``0:40`` / ``40s`` / ``40`` → 秒（float）；失败返回 None。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    match = re.match(r"^(?:(\d{1,2}):)?([0-5]?\d(?:\.\d+)?)\s*s?$", raw, re.I)
    if match:
        minutes = int(match.group(1) or 0)
        return round(minutes * 60 + float(match.group(2)), 3)
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?|秒)$", raw, re.I)
    if match:
        return round(float(match.group(1)), 3)
    return None


# ---------------------------------------------------------------------------
# 任务模式（borrowed from MiniMaxH3_Director/lib/task_modes.py）
# ---------------------------------------------------------------------------
def infer_task(seg):
    """从本段的参考标签与 summary 前缀推断任务模式。"""
    pictures = len(seg.get("pictures") or [])
    videos = len(seg.get("videos") or [])
    summary = (seg.get("fields") or {}).get("summary") or ""
    low = summary.lower()

    if "[video editing]" in low and pictures:
        return TASK_RV2V
    if videos and pictures:
        return TASK_RV2V
    if videos:
        return TASK_V2V
    if "keyframe completion" in low:
        return TASK_I2V
    if "reference generation" in low or pictures:
        return TASK_R2V
    if "audio reuse" in low or "audio reference" in low:
        return TASK_FL2V
    return TASK_T2V


def _contains_any(text, words):
    """大小写不敏感地找词：ASCII 走 \\b 词界（避免 score/theme 这类常见词误伤），
    中文直接子串匹配。"""
    raw = str(text or "")
    low = raw.lower()
    for word in words:
        w = str(word).lower()
        if w.isascii():
            if re.search(r"\b%s(?:s|es)?\b" % re.escape(w), low):
                return True
        elif w in raw:
            return True
    return False


def detect_music_trigger(text):
    """按规范 1.5 判定是否需要配乐。返回命中的类别列表。"""
    hits = []
    if _contains_any(text, MUSIC_WORDS):
        hits.append("音乐词")
    if _contains_any(text, SFX_WORDS):
        hits.append("拟音/音效词")
    if _contains_any(text, GENRE_WORDS):
        hits.append("剧情需要")
    return hits


def detect_music_style(text):
    """从正文里挑 MUSIC block 的 <风格>；取不到就 restrained ambient bed。"""
    raw = str(text or "")
    low = raw.lower()
    for hint, style in STYLE_HINTS:
        probe = hint.lower()
        if probe.isascii():
            if re.search(r"\b%s\b" % re.escape(probe), low):
                return style
        elif hint in raw:
            return style
    return MUSIC_STYLE_DEFAULT


def build_music_block(text="", sfx_only=False, style=None):
    """规范 1.5 的英文 MUSIC block。

    只命中拟音/音效、不要音乐床时传 ``sfx_only=True``，风格词换成
    ``no sustained bed``，其余静音窗规则不变。

    ``style`` 由调用方显式给时全片共用同一个值 —— 规范要求「启用时全片口径
    一致」，逐段各猜一个风格会出现两段风格打架。
    """
    if sfx_only:
        return MUSIC_BLOCK_SFX_TEMPLATE
    if style is None:
        style = detect_music_style(text)
    return MUSIC_BLOCK_TEMPLATE.format(style=style)


# ---------------------------------------------------------------------------
# 校验（规范第 6 节）
# ---------------------------------------------------------------------------
def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?。！？])\s+", str(text or "").strip())
            if s.strip()]


def _timecodes(text):
    out = []
    for m, s, frac in _TIMECODE_RE.findall(str(text or "")):
        value = int(m) * 60 + int(s)
        if frac:
            value += int(frac) / (10.0 ** len(frac))
        out.append(value)
    return out


def validate_segment(seg, is_last=False, music_hit=()):
    """返回本段的问题列表（字符串）。"""
    issues = []
    fields = seg["fields"]
    for name in FIELD_ORDER:
        if not (fields.get(name) or "").strip():
            issues.append("缺少 %s 字段" % name)

    detail = (fields.get("detailed_description") or "").strip()
    if detail:
        first = _sentences(detail)[:1]
        if first and not re.search(r"on-screen text|不出现文字", first[0], re.I):
            issues.append("detailed_description 首句不是 no-text 声明")

    joined = "\n".join(fields.get(name) or "" for name in FIELD_ORDER)
    if _QUOTE_RE.search(joined):
        issues.append("提示词里出现双引号（规范要求一律不用）")

    for line in (fields.get("detailed_description") or "").splitlines():
        if "<d>" in line and not _SPEAKER_RE.search(line):
            issues.append("台词行缺少 (Sx) 说话人编号：%s" % line.strip()[:40])
            break

    if seg.get("lang") == "en":
        stripped = _DIALOG_RE.sub("", joined)
        if _CJK_RE.search(stripped):
            issues.append("英文版在台词块之外出现中文字符")
        if detail:
            count = _word_count(_DIALOG_RE.sub("", detail))
            low = EN_WORDS_TOLERANCE[0]
            # 只报**过短**。规范写的 350–500 词是**动笔写**提示词的区间；
            # 解析器这边吃的是已经写好的 PACK，写得更详细不是错误 —— 把超长
            # 也报成 issue，会让 on_issue=error 直接中断一次本来没问题的渲染，
            # 也会把真正该看的问题（缺字段、回放区台词）淹在一堆噪音里。
            if count < low:
                issues.append(
                    "detailed_description 英文词数 %d，远低于规范区间 %d–%d"
                    "（多半漏了 STYLE/CAMERA/LIGHTING block 或镜头正文）"
                    % (count, EN_WORDS_SPEC[0], EN_WORDS_SPEC[1]))

    summary = (fields.get("summary") or "").strip()
    if summary and not summary.startswith("["):
        issues.append("summary 没有用官方任务前缀开头（[reference generation] 等）")

    retention = fields.get("retention_analysis") or ""
    if retention.strip():
        if _SPEAKER_RE.search(retention):
            issues.append("retention_analysis 里出现了 (Sx) 说话人编号（规范不允许）")
        if not any(token in retention.lower() for token in RETENTION_TOKENS):
            issues.append("retention_analysis 没有用官方保留标记"
                          "（fully_preserved / partially_preserved / …）")

    # 时间码只活在 detailed_description 里：MUSIC block 也带 00:01.600，
    # 扫全字段会把配乐的静音窗当成时间线的一部分，误报「不是严格递增」。
    codes = _timecodes(detail)
    if codes:
        # 递增分两级看：行内（"自 00:12.0 起说……于 00:17.0 前说完"必须递增）
        # 和行首（镜头顺序）。只按全文铺平了看会误报——一句台词行里带的时间
        # 码本来就会越过下一行的起点，那是正常的叙事倒叙，不是写错。
        for line in detail.splitlines():
            line_codes = _timecodes(line)
            if any(b < a for a, b in zip(line_codes, line_codes[1:])):
                issues.append("时间码不是严格递增：%s" % line.strip()[:40])
                break
        heads = [_timecodes(line)[0] for line in detail.splitlines()
                 if _timecodes(line)]
        if any(b < a for a, b in zip(heads, heads[1:])):
            issues.append("镜头顺序的时间码不是严格递增")
        limit = seg["gen_seconds"] + 0.001
        if max(codes) > limit:
            issues.append("时间码 %.3fs 超出生成时长 %.3fs" % (max(codes), seg["gen_seconds"]))

    if seg["new_seconds"] > MAX_NEW_SECONDS + 0.001:
        issues.append("新增时长 %.2fs 超过单段上限 %.0fs，建议拆段"
                      % (seg["new_seconds"], MAX_NEW_SECONDS))
    elif seg["new_seconds"] < MIN_NEW_SECONDS - 0.001:
        issues.append("新增时长 %.2fs 偏短（<%.0fs），H3 上容易退化成一个近乎"
                      "静止的镜头；可以把切点往前挪，让它并进上一段"
                      % (seg["new_seconds"], MIN_NEW_SECONDS))

    if seg.get("videos"):
        issues.append("提示词用了 <Video %s>，本节点没有接参考视频的口"
                      % seg["videos"][0])

    music_field = (fields.get("non_diegetic_music") or "").strip()
    has_music_block = bool(music_field) and "N/A" not in music_field.upper()
    if music_hit and not has_music_block:
        issues.append("命中配乐判据（%s），但 non_diegetic_music 写了 N/A"
                      % "/".join(music_hit))

    # 回放区校验看的是「本段开头有没有在回放上一段」（replay_in）；
    # 尾窗校验看的是「本段结尾要不要留给下一段」（末段没有下一段，跳过）。
    replay_in = float(seg.get("replay_in", seg.get("handoff_seconds")) or 0.0)
    if replay_in > 0 and not seg.get("hard_cut"):
        if not re.search(r"replay|回放", detail, re.I):
            issues.append("接续段缺少 1.6 秒回放句（[Shot 1] 开头要原样回放上一段结尾）")
        for line in detail.splitlines():
            if "<d>" not in line:
                continue
            line_codes = _timecodes(line)
            if line_codes and min(line_codes) < replay_in - 0.001:
                issues.append("回放区（0–%.1fs）里出现新台词，时间码 %.3fs"
                              % (replay_in, min(line_codes)))
                break

    if not is_last and not seg.get("hard_cut"):
        tail_start = seg["gen_seconds"] - HANDOFF_SECONDS
        talk_codes = []
        for line in (fields.get("detailed_description") or "").splitlines():
            if "<d>" in line:
                talk_codes.extend(_timecodes(line))
        if talk_codes and max(talk_codes) > tail_start:
            issues.append("最后 %.1fs 有台词（%.3fs），会被下一段回放成重复语音"
                          % (HANDOFF_SECONDS, max(talk_codes)))
    return issues


def _strip_music_field(text):
    """去掉每段 ``non_diegetic_music`` 的正文。

    配乐判据要扫的是「用户给的脚本 / 剧情 / 原始提示词」，不能扫我们自己写进
    ``non_diegetic_music`` 的 MUSIC block —— 那里必然出现 BGM / music，
    扫了就会自我命中，导致整份 PACK 永远判成「需要配乐」。
    """
    out = []
    skipping = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if _SEG_RE.match(stripped) or _SEP_RE.match(stripped):
            skipping = False
            out.append(line)
            continue
        match = _FIELD_RE.match(stripped)
        if match and match.group("name") in FIELD_SET:
            skipping = match.group("name") == "non_diegetic_music"
            if skipping:
                continue
        if skipping:
            continue
        out.append(line)
    return "\n".join(out)


def _auto_fix(seg, music_hit=(), music_style=None):
    """能自动补的就补上，并把补了什么记进 issues（自动修过的也算问题提示）。"""
    fixed = []
    fields = seg["fields"]
    detail = (fields.get("detailed_description") or "").strip()
    if detail:
        first = _sentences(detail)[:1]
        if first and not re.search(r"on-screen text|不出现文字", first[0], re.I):
            prefix = NO_TEXT_ZH if seg.get("lang") == "zh" else NO_TEXT_EN
            detail = (prefix + " " + detail).strip()
            fields["detailed_description"] = detail
            fixed.append("已自动补 no-text 首句")

    # 接续段：[Shot 1] 开头原样写回放句（插在 no-text 首句之后）。
    # 判据是「本段开头有没有回放上一段」，不是「本段尾部交不交给下一段」——
    # 用 handoff_seconds 会把首段也算进来，凭空多出一句它根本没有的回放。
    replay_in = float(seg.get("replay_in", seg.get("handoff_seconds")) or 0.0)
    if replay_in > 0 and not seg.get("hard_cut"):
        body = (fields.get("detailed_description") or "").strip()
        if body and not re.search(r"replay|回放", body, re.I):
            replay = HANDOFF_REPLAY_ZH if seg.get("lang") == "zh" else HANDOFF_REPLAY_EN
            sentences = _sentences(body)
            if len(sentences) > 1:
                # 只把回放句插到首句后面，其余原文（含换行/空行分段）原样保留
                # —— 用 " ".join 会把整段压成一行，段落节奏全丢。
                rest = body[len(sentences[0]):].lstrip()
                fields["detailed_description"] = "\n".join(
                    [sentences[0], replay, rest]).strip()
            else:
                fields["detailed_description"] = (body + " " + replay).strip()
            fixed.append("已自动补 1.6 秒回放句")

    # 配乐：命中判据就套 MUSIC block（规范 1.5 明确「不默认 N/A」）
    music_field = (fields.get("non_diegetic_music") or "").strip()
    if music_hit:
        if not music_field or "N/A" in music_field.upper():
            joined = "\n".join(fields.get(name) or "" for name in FIELD_ORDER)
            fields["non_diegetic_music"] = build_music_block(
                joined, sfx_only=tuple(music_hit) == ("拟音/音效词",),
                style=music_style)
            fixed.append("命中配乐判据（%s），已自动套 MUSIC block"
                         % "/".join(music_hit))
    elif not music_field:
        fields["non_diegetic_music"] = "N/A"
        fixed.append("non_diegetic_music 留空，已填 N/A")

    # 双引号：规范 1.1 一律不用（画面里出现文字的头号诱因就是提示词里带引号
    # 的"标题"）。删掉引号字符本身，内容原样保留，语义不变。
    joined = _DIALOG_RE.sub(
        "", "\n".join(str(fields.get(name) or "") for name in FIELD_ORDER))
    if _QUOTE_RE.search(joined):
        for name in FIELD_ORDER:
            value = fields.get(name)
            if isinstance(value, str) and value:
                # 台词块里的引号属于原文，一个字都不动
                fields[name] = "".join(
                    part if part.startswith("<d>") else _QUOTE_RE.sub("", part)
                    for part in _DIALOG_SPLIT_RE.split(value))
        fixed.append("提示词里有双引号，已删除引号字符（台词块内的原文未动）")
    return fixed


# ---------------------------------------------------------------------------
# 跨段一致性（规范 4：全链逐字复用）
# ---------------------------------------------------------------------------
def validate_chain(segments):
    """no-text 句 / STYLE 块 / MUSIC 块必须全片逐字一致。"""
    issues = []
    if len(segments) < 2:
        return issues

    def _first_sentence(seg):
        detail = (seg.get("fields") or {}).get("detailed_description") or ""
        sentences = _sentences(detail)
        return sentences[0].strip() if sentences else ""

    def _style_block(seg):
        detail = (seg.get("fields") or {}).get("detailed_description") or ""
        match = re.search(r"^\s*STYLE\s*/\s*CAMERA\s*/\s*LIGHTING.*$",
                          detail, re.I | re.M)
        return match.group(0).strip() if match else ""

    checks = (
        ("no-text 首句", _first_sentence),
        ("STYLE / CAMERA / LIGHTING block", _style_block),
    )
    for label, getter in checks:
        base_seg, base = segments[0], getter(segments[0])
        if not base:
            continue
        for seg in segments[1:]:
            value = getter(seg)
            if value and value != base:
                issues.append("%s 与 %s 的%s不一致（规范要求全链逐字复用）"
                              % (base_seg["id"], seg["id"], label))
                break

    music_blocks = []
    for seg in segments:
        # [Shot N] 脚本的配乐是逐段按时间窗切出来的，本来就段段不同；
        # 「全片逐字一致」只约束自动套用的统一 MUSIC block。
        if seg.get("music_per_segment"):
            continue
        value = ((seg.get("fields") or {}).get("non_diegetic_music") or "").strip()
        if value and "N/A" not in value.upper():
            music_blocks.append((seg["id"], value))
    if len(music_blocks) >= 2:
        base_id, base = music_blocks[0]
        for seg_id, value in music_blocks[1:]:
            if value != base:
                issues.append("%s 与 %s 的 MUSIC block 不一致（规范要求全片口径一致）"
                              % (base_id, seg_id))
                break
    return issues


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def segment_prompt(fields):
    """按官方顺序拼回六段正文（可直接投喂 H3）。"""
    parts = []
    for name in FIELD_ORDER:
        value = (fields.get(name) or "").strip()
        parts.append("%s:\n%s" % (name, value if value else "N/A"))
    return "\n\n".join(parts)


def segment_windows(segments, index):
    """本段的两个衔接窗口 ``(replay_in, tail_out)``（规范 0.1 / 1.4）。

    这两个方向以前被混成一个 ``handoff``，于是首段被凭空多加了 1.6 秒生成
    （多出来的内容不在剧本里，模型只能自己编 → 成片里"乱说话"），末段反
    而少生成 1.6 秒（提示词时间码写到 8.6s，实际只给 7s，动作被挤压）。

    * ``replay_in`` — 段**首**回放上一段结尾的长度。首段没有上一段可回放
      → 0；其余各段（含末段）都是 1.6 秒。它决定**生成时长**：
      ``gen = new + replay_in``。
    * ``tail_out``  — 本段**尾**部切出来交给下一段当锚点的长度。末段没有下
      一段 → 0；其余各段（含首段）都是 1.6 秒。它**不额外占生成时长**——
      锚点就是本段新增内容最后那 1.6 秒（规范 1.4 要求这段无台词），
      从渲染结果里切出来即可。

    段标题里的 ``+1.6`` 标的是 ``replay_in``（``新增+回放=生成``），跟这里
    算出来的一致；标题没写就以本函数为准，因为"后面还有没有段"这件事只有
    站在整条链上才知道。
    """
    total = len(segments or [])
    seg = (segments or [])[index] if 0 <= index < total else {}
    if seg.get("hard_cut"):
        return 0.0, 0.0
    replay_in = 0.0 if index <= 0 else HANDOFF_SECONDS
    tail_out = 0.0 if index >= total - 1 else HANDOFF_SECONDS
    return replay_in, tail_out


def to_shots_json(segments, session_name="", include_task=True):
    """转成 H3Director 认识的 shots_info。"""
    shots = []
    total = len(segments)
    for index, seg in enumerate(segments):
        shot = {
            "id": seg["id"],
            "shot": segment_prompt(seg["fields"]),
            "duration": round(float(seg["new_seconds"]), 3),
        }
        if seg.get("pictures"):
            shot["ref_images"] = list(seg["pictures"])
        replay_in, tail_out = segment_windows(segments, index)
        # handoff_seconds 保留老语义（尾部锚点窗口），老链路照读不误
        shot["handoff_seconds"] = round(tail_out, 3)
        shot["replay_in"] = round(replay_in, 3)
        shot["tail_out"] = round(tail_out, 3)
        # 生成时长由 replay_in 决定，不再把 tail_out 也算进去
        shot["gen_seconds"] = round(float(seg["new_seconds"]) + replay_in, 3)
        if seg.get("hard_cut"):
            shot["hard_cut"] = True
        if include_task:
            shot["task"] = seg.get("task") or TASK_T2V
        shots.append(shot)
    return {
        "session_name": session_name,
        "global": {"global_prompt": ""},
        "shots_info": shots,
        "source": "prompt_pack",
    }


LAYOUT_LABELS = {
    "en-single": "英文单版（现行规范）",
    "bilingual-pack": "双语 PACK（旧格式，已兼容）",
    "pack-single": "带头部单版",
    SHOT_SCRIPT_LAYOUT: "[Shot N] 分镜脚本（按时长自动切段）",
}

# 头部里这些键值得在报告里回显
_HEADER_ECHO = (
    ("Mode", "模式"), ("Aspect", "画幅"), ("Version", "版本"), ("Date", "日期"),
)


def format_report(segments, meta, issues, music_hit=()):
    """解析报告：头部信息 + 每段一行 + 问题清单。"""
    meta = meta or {}
    total_new = sum(s["new_seconds"] for s in segments)
    total_gen = sum(s["gen_seconds"] for s in segments)
    layout = meta.get("layout") or "en-single"

    def _meta(*keys):
        for key in keys:
            value = meta.get(key)
            if value:
                return str(value)
        return ""

    out = [
        "H3 分镜提示词解析报告",
        "规范   %s · %s" % (LAYOUT_LABELS.get(layout, layout),
                            meta.get("rule_version") or RULE_VERSION),
        "段数   %d" % len(segments),
        "总时长 %.2fs（新增） / %.2fs（生成，含回放重叠）" % (total_new, total_gen),
    ]
    project = _meta("Project", "项目")
    out.append("项目   %s" % (project or "（头部没写项目名，可在会话名里指定）"))
    for key, label in _HEADER_ECHO:
        value = _meta(key, label)
        if value:
            out.append("%-6s %s" % (label, value))
    out.append("配乐   %s" % ("命中 %s" % "/".join(music_hit) if music_hit else "N/A"))
    out.append("")
    out.append("  %-6s %-7s %-7s %-7s %-7s %-7s %-6s %s"
               % ("段", "新增s", "段首回放", "尾窗s", "生成s", "规范帧", "模式",
                  "参考图"))
    out.append("  " + "-" * 70)
    for index, seg in enumerate(segments):
        # 两个窗口以 segment_windows() 为准：段标题里没写 +1.6 不代表本段不留
        # 锚点（首段照样要把尾巴交给下一段），照搬 seg["handoff_seconds"] 会让
        # 报告里的「尾窗」和实际渲出来的 tail 文件对不上。
        replay, tail = segment_windows(segments, index)
        out.append("  %-6s %-7.2f %-7.2f %-7.2f %-7.2f %-7d %-6s %s"
                   % (seg["id"], seg["new_seconds"], replay, tail,
                      seg["gen_seconds"],
                      seg.get("spec_frames")
                      or frames_for_seconds(seg["gen_seconds"]),
                      seg["task"],
                      ",".join("P%d" % p for p in seg["pictures"]) or "-"))
    out.append("")
    out.append("  规范帧 = 生成时长 × %dfps（接线表口径）；实际生成按 H3 的 17k+5"
               % FPS)
    out.append("  网格向上对齐，见 H3 Director 报告里的帧数。")
    if issues:
        out.append("")
        out.append("注意：")
        for item in issues[:30]:
            out.append("  · " + item)

    notes = _writing_notes(segments)
    notes.extend(_cutaway_notes(segments))
    if notes:
        out.append("")
        out.append("写作提示（不影响本次渲染，留给下一次改稿）：")
        for item in notes[:10]:
            out.append("  · " + item)
    return "\n".join(out)


def _writing_notes(segments):
    """规范里属于「怎么写」的那几条：词数超上限、没有显式 STYLE block。

    这些不是结构错误 —— 一份已经写好的 PACK 写得更详细，不代表它不能渲。
    把它们报进 ``issues`` 会让 ``on_issue=error`` 误杀一次本来没问题的渲染，
    所以单独收在报告末尾。
    """
    notes = []
    high = EN_WORDS_SPEC[1]
    longs, no_style = [], []
    for seg in segments or ():
        detail = (seg.get("fields") or {}).get("detailed_description") or ""
        if not detail.strip():
            continue
        count = _word_count(_DIALOG_RE.sub("", detail))
        if count > high:
            longs.append((seg["id"], count))
        if not _STYLEBLOCK_RE.search(detail):
            no_style.append(seg["id"])
    if longs:
        notes.append(
            "detailed_description 超过规范推荐的 %d 词：%s。内容更细不是错误，"
            "但长段会分散模型注意力，下次可以再拆一刀。"
            % (high, "、".join("%s %d 词" % (i, c) for i, c in longs)))
    if no_style:
        notes.append(
            "detailed_description 里没有显式的 STYLE / CAMERA / LIGHTING block"
            "：%s。规范 4 要求全链逐字复用；现在靠每段首句保持一致也能跑，"
            "只是改稿时容易漏改其中一段。" % "、".join(no_style))
    return notes


# 「反打宽度 < HANDOFF_SECONDS（1.6s）」成对视觉故障的元凶。
# 规范 1.4 警告：「反打宽度低于 1.6 秒时，模型会把『任何时点都可切』当成新常态，
# 在对白中间提前触发 → 成片出现单帧闪跳」。本次 v18 S03 的成片 26–27s 闪跳
# 即因此而起。判定条件：同段内 `[Shot N] At 00:XX.XXX... cut to <Subject` 与
# 紧随其后的 `[Shot N+1] At 00:YY.YYY... back to <Subject`，YY-XX < 1.6。
# 不带 re.S：要求正则只在本行内查（PACK 的每个 Shot 标题就是一行）。
_SHOTTIME_CUT_RE = re.compile(
    r"At\s+00:(\d{1,2})\.(\d{1,3})[^.\n]*?"
    r"(?:cut\s+to\s+<Subject|切\s*<Subject)")
_SHOTTIME_BACK_RE = re.compile(
    r"At\s+00:(\d{1,2})\.(\d{1,3})[^.\n]*?"
    r"(?:back\s+to\s+<Subject|回到\s*<Subject)")


def _cutaway_notes(segments):
    """逐段找「cut-to / back-to」配对，宽度 < HANDOFF_SECONDS 的写到提示里。

    不阻塞渲染：写得过短不一定是错误（用户可能就是要 0.6 秒闪白），但要
    显式告知这是已知的闪跳成因，让下一次改稿时主动修。
    """
    notes = []
    for seg in segments or []:
        detail = (seg.get("fields") or {}).get("detailed_description") or ""
        if not detail.strip():
            continue
        cuts, backs = [], []
        for line in detail.splitlines():
            mc = _SHOTTIME_CUT_RE.search(line)
            if mc:
                t = int(mc.group(1)) + int(mc.group(2)) / 1000.0
                cuts.append(t)
            mb = _SHOTTIME_BACK_RE.search(line)
            if mb:
                t = int(mb.group(1)) + int(mb.group(2)) / 1000.0
                backs.append(t)
        if not cuts or not backs:
            continue
        cuts.sort(); backs.sort()
        tight = []
        # 每个 BACK 找它之前最近、且 < HANDOFF_SECONDS 之前的 CUT
        # （不用 zip：CUT 数和 BACK 数不一定对得上，多切无回 / 多回无切都不算）
        for b in backs:
            before = [c for c in cuts if c < b]
            if not before:
                continue
            c = before[-1]
            gap = b - c
            if 0 < gap < HANDOFF_SECONDS:
                tight.append((c, b, gap))
        if not tight:
            continue
        sample = "；".join(
            "%.2fs→%.2fs (Δ=%.2fs)" % (c, b, d) for c, b, d in tight)
        notes.append(
            "%s 里有 %d 处过短反打（< %.1fs）：%s。反打宽度低于 1.6 秒会让"
            "模型把『任何时点都可切』当成新常态，对白中提前触发 → 成片出现"
            "单帧闪跳（v18 S03 即因 05.000→05.600、07.000→07.600 这种模式"
            "在 26–27s 触发了模型自发的 S1→S2 提前切换）。建议把切回往后推"
            "到 ≥ 1.6s，或合并这段反打让镜头持续停在主说话人。"
            % (seg["id"], len(tight), HANDOFF_SECONDS, sample))
    return notes
