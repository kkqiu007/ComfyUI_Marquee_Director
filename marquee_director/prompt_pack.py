# -*- coding: utf-8 -*-
"""MiniMax H3 分镜提示词 PACK → shots_info 的解析与规范化。

规范 = ``minimax-h3-shot-segment`` SKILL（**英文单版**，2026-09-20 版）。

交付形态（SKILL §5）
--------------------
唯一交付物是一份 UTF-8 文本，只有两块：**PACK 头部 + 英文版六段**，以
``END OF PACK`` 收尾；不产中文版（中文版要用户另行明确要求，由 skill 侧
产出，节点不翻译）。::

    ==========================================================
    MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包
    Project / 项目         : <项目名>
    Mode / 模式            : Ref2VA (full reference)
    Total duration / 总时长 : 00:40
    Segments / 段数        : 4
    Aspect / 画幅          : 16:9 / 2K / 24fps
    Music / 配乐           : N/A 或 Per MUSIC block
    Version / 版本         : v1
    Date / 日期            : YYYY-MM-DD
    ==========================================================

    ==========================================================
    ENGLISH VERSION / 英文版（本文件唯一语言版本）
    ==========================================================

    ########## S01 / 11s / EN ##########
    subject_definitions:
    ...
    non_diegetic_music:
    N/A

    ########## S02 / 11s+1.6=12.6 / EN ##########
    ...

    ==========================================================
    END OF PACK / 文件结束
    ==========================================================

本模块**只写英文**：段标题语言标记固定 ``EN``，六段正文按 §1.3 除
``<d>[Chinese] 原文</d>`` 台词块外零中文字符，STYLE / CAMERA / LIGHTING、
DELIVERY、MUSIC 三个 block 全链逐字复用（§4）。

读入是**宽容**的
----------------
输入侧认三种形态，全部归一到「英文六段」输出：

* ``pack-single``      —— 现行规范：PACK 头部 + ``ENGLISH VERSION`` + EN 段。
* ``en-single``        —— 只有英文六段、无头部（skill §0.2 的旧交付形态）。
* ``shot-script``      —— 整段式分镜脚本（``[Shot N] At 00:XX.XXX`` 一路写到
  片尾）。节点能吃下并自动拆段，但拆段器只是**近似**（保证 5–11 秒、不切台词、
  尾窗留白），拿不到镜头表里的创作意图 —— 按 §0.2 仍应交付分段 PACK。
* 旧双语 PACK（``[1] 中文版`` / ``[2] 英文版``）照样能吃，**只取英文版那一侧**。

整段式脚本若写的是中文，节点照旧解析并把它按 §1.3 报出来（不翻译、不改写
创作文本）；交付前应改写成英文单版 PACK。

这里只做**纯文本**处理：不 import torch / comfy，可以在没有显卡依赖的环境里单测。

★ 上下节点的职责边界（2026-09-27 定线，改代码前先看这一段）
------------------------------------------------------------
链路：``PrimitiveStringMultiline(361)`` → ``H3ScriptTranslate(743)``
→ ``H3PromptPackParser(720)`` → ``H3Director(700)``。四层的权限**逐层收窄**：

| 层 | 允许做 | **禁止做** |
|---|---|---|
| 361 剧本输入 | 作者写的一切 | — |
| 743 英译 | ①**语言转换**（中文→英文）②结构保真（字段名独占行首、时间码不许动）③**只**剥离台词块外的中文 | 改写结构、丢时间码、替作者改描述 |
| 720 解析 | ①按镜头边界拆段 ②时间码**重定位**到段内 ③格式规范化（引号/全角/`appears in` 收窄）④补规范句（no-text / 回放 / 静音纪律 / MUSIC block）⑤剥离台词块外中文 | **改写作者的描述文本**（外观锁定那几条尤其） |
| 700 导演 | 按段渲染、拼接、缓存 | 改提示词（除 743/720 已做的之外） |

**两条硬纪律**：

1. **描述文本只读**。作者写的 `subject_definitions` / `retention_analysis`
   里的每一句（尤其 ``<Picture N>: fully_preserved - … 外观与参考图一致``）
   必须**逐字**送到模型；节点只能改"格式"（引号、全角标点、`appears in` 的
   镜头号范围），不能改"内容"。这条由 ``author_text_drift()`` 把关，
   任何新增的改写都会在解析报告里变成 issue。
   > 实测教训：``_seg_retention`` 曾把作者写的
   > "The appearance of Subject N is consistent with the reference image in
   > all shots." 换成 "retained as the appearance reference for `<Subject N>`."
   > —— 把"生成结果要与参考图一致"这条**断言抹掉**了，模型于是跨段换衣服
   > （iter6 成片：金 → 粉 → 绿，参考图却是深红纱袍）。

2. **全段一致**。同一条纪律句要么四段都加、要么都不加。只在"有穿衣动作"的
   那一段改写措辞，会让四段提示词分叉 —— 那正是跨段漂移的温床。
   `wardrobe` 的内容改写因此**默认关闭**（``MARQUEE_WARDROBE_REWRITE=1`` 才开）。

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

# ★ 2026-09-27 加强：iter7 真渲染里 S04 出现了**画面字幕**「记得找我也行」——
#   模型把 `<d>[Chinese] …</d>` 块里的中文当成"要显示在画面上的文字"画了上去。
#   原句只说了 "no on-screen text, subtitles, captions"，没有点明**台词块**这个
#   具体诱因，模型就把台词当字幕渲染了。补两句：
#     ① 台词块只用于口型与配音，**任何情况下都不得出现在画面上**；
#     ② 点名"不要给对话加字幕"（模型对 subtitle 这个词最敏感）。
NO_TEXT_EN = ("No on-screen text, subtitles, captions, timecodes, or any "
              "graphics overlays anywhere in the frame. The spoken lines exist "
              "only as audio for lip-sync and must never be drawn, burned in or "
              "displayed on screen in any form; do not subtitle the dialogue, "
              "and do not render any character's speech as visible text.")

# 零台词段的静音纪律（节点层通用兜底，见 parse_shot_script_blocks）。
# 任何剧本只要某段没分到台词，都必须显式写"这段不说话"——留空会让模型自行
# 即兴发声，成片出现剧本外的含糊人声。
NO_DIALOGUE_EN = (
    "DIALOGUE: This segment contains NO dialogue at all. No character speaks, "
    "whispers, hums, laughs or makes any vocal sound; no mouth movement and no "
    "lip-sync anywhere in this segment. Audio is limited to the score, ambient "
    "room tone and non-vocal sound effects only.")

# 段末留白纪律（通用兜底，见 ``_auto_fix`` 的「台词结束后的留白」）。
# 段里**有**台词、但最后一句说完之后到段尾还空着一大段时补上：模型遇到没有
# 台词指令的空白会即兴发声（成片＝剧本外的含糊音节）。
# ★ 同样不含任何时间码，理由同 NO_DIALOGUE_EN / DIALOGUE_ONCE_EN。
#
# ★ 触发阈值（秒）：最后一句结束 → 段尾的间隔超过它才补纪律句。
#   2026-09-27 从「一个回放窗（1.6s）」收紧到 **0.3s**：实测 iter6 真渲染里
#   S03 的间隔**恰好** 1.6s（台词写到「不晚于 00:11.600」，段生成 13.20s），
#   被原来的严格大于号排除，纪律句没注入 → 成片 31.0–32.6s（= 尾窗）冒出
#   剧本外发声，又被 S04 原样回放成第二次。间隔恰好等于一个回放窗 = 整段
#   剩余时间都是尾窗，恰恰**最需要**静音纪律。0.3s 只防抖，不改语义。
TAIL_SILENCE_MIN_GAP = 0.3
TAIL_SILENCE_EN = (
    "AFTER THE LAST LINE: once the last spoken line above has ended, this "
    "segment contains no further speech of any kind - no character speaks, "
    "whispers, hums, laughs, sighs or makes any vocal sound, and no mouth "
    "movement or lip-sync from that point to the end of the segment. Only the "
    "score, ambient room tone and non-vocal sound effects continue.")

FPS = 24

# H3 只生成 17k+5 帧，且**回放窗口必须向上对齐**（见下面 HANDOFF_SECONDS 的注释）。
_GRID_REMAINDER = 5
_GRID_STEP = 17

# ★ 2026-09-30：网格向上吸附给「新增时长」带来的最大额外量（秒）。
#   ``_build_segment`` 把 gen 对齐到 17k+5 后，``new = gen - replay`` 会比作者
#   写的名义值最多大 ``(_GRID_STEP - 1) / FPS`` = 16/24 ≈ 0.667s（因为向上
#   对齐最多加 16 帧）。所以「单段新增 ≤ MAX_NEW_SECONDS」这条规范判据必须
#   带上这个容差，否则每一段都会误报"超过单段上限，建议拆段"——
#   11.0s 的段在 24fps 下根本无法精确落在网格上（303 帧 → 311 帧 = 11.333s）。
GRID_SLACK_SECONDS = (_GRID_STEP - 1) / float(FPS)


def grid_seconds(seconds):
    """把「秒」向上对齐到 H3 的 17k+5 帧网格，返回**对齐后的秒**。

    ★ 这是本包唯一一处「秒 → 网格秒」的实现，解析层与渲染层必须共用它。

    为什么必须在这里对齐：``HANDOFF_SECONDS = 1.6`` 只是规范写的名义值，
    1.6s × 24fps = **38 帧**，不在 17k+5 网格上。真正能生成的回放窗口是
    **39 帧 = 1.625s**。下游 ``director._shot_plan`` 一直按 ``guide_length_up``
    向上取（39 帧），而解析层（``segment_windows`` / ``_build_segment``）
    却把 1.6 原样写进 ``replay_in`` / ``handoff_seconds`` —— 两边差 **1 帧**。

    1 帧看着小，但它会连着 ``duration`` / ``gen_seconds`` 一起被写进
    ``storyboard.json``，再由 ``_shot_plan`` 反算：文件名里的 1.6 被当成
    「请求值」，``generation_length(round(gen * FPS))`` 向上取 → 每段的生成
    帧数、回放帧数、缓存键（``segment_key`` 把 ``length`` 算进 payload）
    逐段错开一拍。同一条链重跑时这些值在「1.6 与 1.625」之间来回跳，
    表现为**接缝处动作原地打结 / 一段里多出 17 帧回放**的经典症状。

    ``guide_length_up`` 在 ``common.py``（依赖 comfy 的包），本模块是零依赖的
    纯标准库模块 —— 所以这里就地实现一份等价语义，两边口径必须一致：
    网格 = {5, 22, 39, 56, 73, 90, 107, 124, …} = 17k+5。
    """
    try:
        secs = float(seconds or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if secs <= 0:
        return 0.0
    frames = max(5, int(round(secs * FPS)))
    while (frames - _GRID_REMAINDER) % _GRID_STEP != 0:
        frames += 1
    return frames / float(FPS)


# 段首回放区：规范写 1.6s，H3 的 17k+5 网格上落在 1.625s（39 帧）。
# ★ 1.6 是**规范名义值**，落进 shots.json 前一律过 ``grid_seconds()`` 向上对齐。
#   ``segment_windows`` / ``to_shots_json`` 已经在出口处对齐，所以下游读到的
#   永远是 1.625，而不是 1.6。
HANDOFF_SECONDS = 1.6
# 单段新增时长区间（minimax-h3-shot-segment_en 规范 0.1）：
# 「每段新增固定落在 5–11 秒」——生成最长 12.6 秒（11 + 1.6 回放重叠），
# 首段最长 11 秒。以前这里是 10.0，把规范里的 11s 段判成「超过单段上限」，
# 与样例 PACK（S01 / 11s、S02 / 11s+1.6=12.6）直接打架。
MAX_NEW_SECONDS = 11.0
# 5 秒是硬下限：切镜密度撑不满 5 秒的镜头不单独立段，改写成上一段内的切镜。
MIN_NEW_SECONDS = 5.0
# 镜头边界可滑动的范围（``_rebalance_shot_bounds``）。[Shot N] 是叙事锚点，
# 让位太多就会把「Shot 2 的开场」挪进 Shot 1 里；3 秒足够拉平常见的不均匀
# 镜头表（本次 10/12/13/5 → 11/11/11/7 只用了 +1s 与 -2s），又不会串戏。
BOUND_SLACK_SECONDS = 3.0

# 现行规则版本（写进解析报告，方便一眼看出跑的是哪一版规范）
RULE_VERSION = "en-single-v3"

# 接续段 [Shot 1] 开头固定原样写的回放句（SKILL §1.4，英文版唯一交付语言）
#
# ★★ 2026-09-27：句子里**不能写死 1.6 秒**。规范（docs/how-it-works.md:104、
#   README.md:47）的权威值是 **1.625 s = 39 帧**；1.6s 只是"约"的近似说法
#   （规范自己写作 "~1.6"）。且规范明说 handoff「**按段匹配，不按表**」
#   （短段用 0.917 s / 22 帧）。所以这里做成**模板 + 占位符**，由
#   ``_handoff_replay_en(seconds)`` 填入本段实际回放时长，与 ``replay_in``
#   字段严格同源 —— 文本层与字段层用同一个数，模型与渲染器才不会差 1 帧。
HANDOFF_REPLAY_EN_TMPL = (
    "The segment opens on an exact replay of the closing {handoff} seconds of "
    "the preceding take and carries that motion straight through, so the join "
    "is invisible. The subject holds the exact body pose, head angle, gaze "
    "direction, arm position and facial expression carried in by the replayed "
    "opening and continues out of it without any reset or restart; no new "
    "action, no new spoken word and no new cut begins before the {handoff}-"
    "second mark."
)


def _fmt_handoff_seconds(seconds):
    """回放时长的**文案写法**：``1.625`` / ``1.6`` / ``0.917``。

    去掉尾随 0（``1.600`` → ``1.6``），但**保留**有效小数
    （``1.625`` / ``0.917`` 原样）。这样文案里的数字与 ``replay_in``
    逐位一致，不会再出现「文案 1.6 / 字段 1.625」的 1 帧错位。
    """
    text = ("%.3f" % float(seconds)).rstrip("0").rstrip(".")
    return text or "0"


def _handoff_replay_en(seconds):
    """生成接续段 [Shot 1] 的回放句，时长用**本段实际** ``replay_in``。"""
    return HANDOFF_REPLAY_EN_TMPL.format(
        handoff=_fmt_handoff_seconds(seconds))


TASK_T2V = "t2v"
TASK_I2V = "i2v"
TASK_FL2V = "fl2v"
TASK_R2V = "r2v"
TASK_V2V = "v2v"
TASK_RV2V = "rv2v"

# 官方任务前缀（规范：summary 必须以 [xxx] 开头）现在只有前端在认：
# web/js/h3_director.js 的 TASK_PREFIX_RE 是唯一一份登记，后端校验
# （下面「summary 没有用官方任务前缀开头」）只查了行首是不是 `[`。
# 2026-09-20 清理：后端这份 TASK_PREFIXES 常量全包零引用，且与前端那一份是
# 两份真相 —— 删掉，避免以后改了一边忘了另一边。
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
#
# ★★ 2026-09-23 改：由「台词期间完全静音」改为「压低但不停」+ 连续低频底噪。
#
# 改的依据是实测，不是偏好。用一段真实影视交付件（F:\25.mkv a:1 "AAC 2.0 HIFI"）
# 的 1/3 倍频程长期平均谱当基准，量出片子在**台词期间**几乎没有 150 Hz 以下能量：
#
#   * 旧模板强制配乐「stays silent for the entire duration of every line」，
#     而 overall_soundscape 里给的房间声全是高频（烛火、夜虫、窗缝风、更梆），
#     于是台词期间轨道上只剩人声 + 高频细节；
#   * 而模型人声本身 80-150 Hz 只有 0.03%（真对白 2.58%）—— 声码器基本不产生
#     150 Hz 以下的人声能量。两者叠加 ⇒ **台词期间结构性偏薄**，听感就是"不够饱满"。
#   * 母带链对频谱是**中性**的（同一段前后逐 1/3 倍频程差 <2 dB），
#     所以这不是节点能补的：**EQ 加不出信号里没有的能量。**
#
# 两处改动，都在可回滚的文本层面：
#   ① 「静音」→「压低不停」：真人电影在台词下**不停**配乐，只是压低。
#      门槛写死 12 dB，保证不盖人声（原句自相矛盾：「ducks to clearly below」
#      与「stays silent」并存，模型只会执行更强的那句）。
#   ② 增加「连续全频段底噪 + 真实低频能量」要求：低频不掩蔽 300-3k 的语音，
#      所以这条填的是低频缺口，不带来可懂度风险。
#
# ★★ 2026-09-27：块内所有回放窗/静音窗秒数统一到 **1.625 s（39 帧）**，
#   与 ``replay_in`` / ``grid_seconds(HANDOFF_SECONDS)`` 严格同源。
#   旧文案写 1.6（= 38 帧，不在 17k+5 网格上）与 ``00:01.600`` 边界，
#   与下游实际渲染的 39 帧差 1 帧：音乐会侵入被原样回放的最后一帧。
#   规范（docs/how-it-works.md:104）的权威值就是 1.625 s / 39 帧，
#   ``1.6`` 只是规范自己在散文中用的"约"值。
#   ★ 三处一起改：``1.6-second opening`` / ``00:01.600`` / 尾窗两处
#   ``1.6 seconds``（尾窗与回放窗等长，接缝两侧对称）。
MUSIC_BLOCK_TEMPLATE = (
    "A single low, sustained {style} bed at low level, over a continuous "
    "full-range ambience floor. It continues only through the replayed "
    "1.625-second opening; no new music cue, no BGM swell and no sound effect "
    "starts before 00:01.625. It ducks to clearly below the voice one second "
    "before every spoken line and stays at that ducked level -- at least 12 dB "
    "under the voice -- for the entire duration of every line: it never stops "
    "for a line, and it never rises above it. The ambience floor is exempt from "
    "the duck, is continuous under every line, and carries real low-frequency "
    "energy -- a building hum, distant traffic, a low room rumble -- not only "
    "high-frequency detail. It stops 1.625 seconds before the end of each shot; "
    "the closing 1.625 seconds of every shot carry no music, no BGM and no sound "
    "effects."
)
# 默认风格（模板里已经有 bed 这个词，这里别再写 bed，否则变成 "bed bed"）
MUSIC_STYLE_DEFAULT = "restrained ambient"

# 只命中拟音/音效、不要音乐床时的专用 block：
# 英文模板是 "sustained <风格> bed"，把 <风格> 填成 "no sustained bed" 会拼出
# "sustained no sustained bed bed"，所以这种情况整句换掉。
# 台词行后面固定跟的口型同步约束（SKILL §3「对镜台词」）。
LIPSYNC_EN = (
    "Lip-sync is exact: every syllable is clearly articulated, no held open-mouth "
    "smile; the body stays quiet while speaking, only head, shoulders and one hand "
    "may assist; full gestures, blocking and large moves land only after the line "
    "ends and the mouth closes, inside the silent gap. Writing order is fixed: "
    "line, mouth closes, move during silence, settle on a non-speaking pose. "
    "Dialogue and large motion never overlap; no unfinished phonation at the "
    "segment tail."
)

# 台词交付纪律（节点层通用，见 ``_inject_dialogue_once``）。
# 针对「同一句被说两遍 / 提前到上一句的窗口里说」——模型拿到一段多句的 prompt
# 时会自行重排时间码。这是模型侧行为，节点能做的就是**逐句**把纪律钉死：
# 只说一次、严格按该句自己的时间码、不许提前也不许后半段再续。
#
# ★ 本句故意**不含任何 00:MM.m 时间码**：``_talk_spans`` 会按行取时间码，
#   注入句里出现时间码会撑大该句的窗口（见 _TALK_START_RE 处的实测事故）。
DIALOGUE_ONCE_EN = (
    "DELIVERY: Speak this line EXACTLY ONCE, as one continuous utterance at the "
    "timecode stated for it - do not begin it earlier, do not restart or repeat "
    "any part of it, and do not let it resume or echo later in this segment.")

# ★ 2026-09-28 用户故障 #1：画面出现字幕（<d>[Chinese] 台词被模型烧进画面）。
#   EN 分镜的 <d>[Chinese] …</d> 里 "[Chinese]" 标签会被 H3 读成"把中文显示出来"。
#   两条根治：① 剥掉 [Chinese]/[EN] 标签（它们是解析/QC 用的，不是给模型的）；
#   ② 每段 detailed_description 强制追加"台词只出声、严禁上屏"的高优先级禁令。
# ★ 2026-09-30 第三轮：句子里不再出现字面 ``<d>…</d>`` —— 它会作为一个
#   **伪台词块**留在最终提示词里（逐行块扫描、台词计数都把它算进去），
#   模型也看得见一对空的对话括号，反而是"把台词写上屏"的暗示。改成纯文字。
SUBTITLE_BAN_EN = (
    "Hard rule for this whole film: NO text of any kind may appear on screen - "
    "no subtitles, no captions, no on-screen lyrics, no karaoke text, no title "
    "cards, no watermarks, no lettering of any kind in any language. The "
    "bracketed dialogue lines are AUDIO-ONLY lip-sync source; they are heard, "
    "never shown. If a character speaks, only the voice and mouth move — the "
    "frame stays free of any written characters."
)

# ★★ 2026-09-30 第四轮：试过「台词语言标签降级」（`<d>[Chinese] …</d>` →
#   `spoken in Chinese: <d>…</d>`），**已回滚**。留此记录，免得下次再走一遍。
#
#   动机：09-28 曾判定「块内 `[Chinese]` 会被 H3 读成『把这段中文显示出来』→
#   烧字幕」，并据此加过 `_strip_dialogue_lang_tags`。想用合规的写法恢复该效果。
#
#   回滚理由（三条，任一都够）：
#   ① **实验证伪**：v25sub 交付产物块内已无标签（残留 0、`spoken in Chinese` 就位），
#      seg_02 **照样烧出「汝与曹贼何异」**（t=2.75s，conf 0.84）→ 标签既非充分
#      也非必要条件，降级**不修烧字幕**。
#   ② **规范其实是反的**：官方 `base-en.txt §4.4` 原文是「Inside `<d>`, include
#      only the **language tag** and the actual spoken content」—— 语言标签本就
#      **该在块内**。把它挪到块外才是偏离规范。
#   ③ **会打断下游消费者**：`(Sx)` 与 `<d>` 之间插入 `spoken in X: ` 之后，
#      QC 工具的台词正则 `\(S(\d)\)\s*<d>`（`comfyui-h3-video-qc/scripts/h3_qc.py`）
#      直接匹配不到 —— 实测 v25sub 的闸门从「5 句台词对位」退化成
#      **「未解析到台词」**，等于把一条检查关掉了。
#
#   **烧字幕的正确处置**：它是**随机事件**，靠节点包自带的 `subtitle_qc`
#   OCR 兜底 + 换种子重渲修 —— 本轮 seg_02 **第 1 次重渲即通过**
#   （manifest `seed_qc: True`，重渲后 OCR 0 命中）。不要靠改提示词。
#   详见 `subtitle_qc.py` 头部与本包 README 的防字幕一节。


# ★ 2026-09-30 第三轮修复：09-30 渲染《曹贼的性价比》S02/S04 仍烧出逐行画面字幕
#   （「汝与曹贼何异」「没彩礼」…全句卡拉OK式上屏）。NO_TEXT_EN 在**段首**、
#   SUBTITLE_BAN_EN 在**段尾**，离 <d> 台词块隔了整个镜头列表，注意力够不着；
#   同结构 5 段只烧 2 段 = 概率性行为。_inject_lipsync / _inject_dialogue_once
#   证明「逐句紧邻」的位置模型才听得进 —— 防字幕照搬同一范式：**每个台词块
#   紧后面**挂一条只出声、严禁上屏的禁令。节点层通用规则，与剧本内容无关。
DIALOGUE_NO_TEXT_EN = (
    "AUDIO-ONLY LINE: the dialogue written just before this sentence is spoken "
    "sound and mouth movement only - never typeset it, or any of its words, onto "
    "the frame as subtitles, captions, burned-in text or lettering in any "
    "language, and keep the picture completely free of written characters while "
    "it is spoken. The line lives inside the story world - it is spoken "
    "conversationally to whoever it addresses within the scene (or held as a "
    "private aside when alone), eyes off the lens, never addressed to the viewer "
    "or the camera, and the frame holds only the physical scene exactly as the "
    "reference images show it.")


MUSIC_BLOCK_SFX_TEMPLATE = (
    "No sustained music bed at any point. A few isolated sound-design hits may "
    "land after 00:01.625; no new music cue and no BGM swell starts before "
    "00:01.625. Every hit ducks to clearly below the voice one second before "
    "each spoken line and stays at that ducked level -- at least 12 dB under the "
    "voice -- for the entire duration of every line: it never stops for a line. "
    "Under the hits and under every line there is a continuous full-range "
    "ambience floor that carries real low-frequency energy -- a building hum, "
    "distant traffic, a low room rumble -- not only high-frequency detail. "
    "Nothing starts in the closing 1.625 seconds of any shot; the last 1.625 "
    "seconds of every shot carry no music, no BGM and no sound effects."
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
# ★ 2026-09-30 排查补充：段头**连斜杠都没有**的形态（``### S01 ###``）以前
#   不被 _SEG_RE 认 → has_seg_head=False → 整份 PACK 静默落进「整段式分镜
#   脚本」路径，段长/台词全部按错误口径反推（压测 D 实证）。这里加一个
#   **只认 S 前缀 id** 的兜底正则（``### S01 ###``），把这种段头正常收进
#   分段路径、时长走缺省值。裸数字（``### 2026 ###``）不认 —— 那是文档
#   横幅的高危形态，宁可解析不到报错，不可把说明文字切成段。
_SEG_RE_BARE = re.compile(
    r"^\s*#{3,}\s*(?P<id>S\d+[A-Za-z]?)\s*#{3,}\s*$")
# 段头语言槽位里的硬切声明（``### S02 / 8s / hard cut ###``）。hard_cut 的
# 正主判据在 summary 文本里（见 _build_segment），这里只是把**段头写法**
# 也接住 —— 以前这个槽位被当语言标记解析失败后静默丢弃，硬切段被当成
# 连续段（回放窗/尾窗全错位，压测 E 实证）。
_SEG_LANG_HARD_CUT_RE = re.compile(r"hard\s*-?\s*cut|hardcut", re.I)
# [1] 中文版 / CHINESE VERSION
_SEC_RE = re.compile(r"^\s*\[(?P<no>[12])\]\s*(?P<name>.*?)\s*$")
_FIELD_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[:：]\s*(?P<rest>.*)$")
# 段名行也吃 markdown 形态：``## subject_definitions`` / ``**summary**`` /
# ``### retention_analysis：``。两种写法都**必须带 markdown 标记**才认 ——
# 不能写成「可选标记 + 可选冒号」，否则正文里随便一句以 summary 开头的英文
# （``summary of the scene``）都会被当成段名行，整段正文被劈开。
# 名字限定 ``[A-Za-z_]``：``## 取景纪律`` / ``### [Shot 1] …`` 这类中文小标题、
# 镜头标记不会被误判成字段。
_MD_HEAD_FIELD_RE = re.compile(
    r"^\s*#{1,6}\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[:：]?\s*(?P<rest>.*)$")
_MD_BOLD_FIELD_RE = re.compile(
    r"^\s*(?:\*\*|__)\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\*\*|__)\s*[:：]?\s*(?P<rest>.*)$")
# 文档级横幅：``=== 增强点说明（供审阅，正文不包含本段）===``。作者习惯在六段
# 之后附「台词重写对照表 / 时间轴核对表」这类审阅材料并注明不属于正文；
# 不截断的话它们会被并进最后一个字段（non_diegetic_music），污染配乐描述。
#
# ★★ 中间那段必须是「非空白、非 `=`」起头结尾：否则正则回溯之后**纯 `====`
#    分隔线也会命中**（`={2,}` 让出一个 `=`，那个 `=` 又满足 `\S`），
#    于是带 PACK 头部的整段式脚本一进 `_shot_global_fields` 就 break、
#    全局字段全丢（实测：4 段全报「缺少 subject_definitions 字段」）。
#    纯分隔线是 `_SEP_RE` 的职责，这里必须让开。
_BANNER_RE = re.compile(r"^\s*={2,}\s*[^\s=].*[^\s=]\s*={2,}\s*$")


def _match_field(line):
    """段名行 → ``(name, rest)``；只认官方六段字段名，官方 ``name:`` 优先。

    兼容四种写法：规范 ``detailed_description:``、markdown 标题
    ``## detailed_description``、加粗 ``**summary**``，以及中文全角冒号。
    不是六段字段名（含中文小标题、镜头标记）一律返回 ``None``。
    """
    for pattern in (_FIELD_RE, _MD_HEAD_FIELD_RE, _MD_BOLD_FIELD_RE):
        match = pattern.match(line)
        if match and match.group("name") in FIELD_SET:
            return match
    return None
_PIC_RE = re.compile(r"<\s*Picture\s*(\d+)\s*>", re.I)
_VID_RE = re.compile(r"<\s*Video\s*(\d+)\s*>", re.I)
# 一整行**只有**官方任务标签、没有任何实际内容的行（``[reference generation]``
# 单独占一行）。剧本里常见，套进 summary 会让标签出现两次。
_TASK_TAG_ONLY_RE = re.compile(
    r"^\[\s*(?:video\s+continuation\s*\+\s*)?"
    r"(?:reference\s+generation|t2v|i2v|fl2v|l2v|ref2v[a-z]*)\s*\]$", re.I)

_SUB_RE = re.compile(r"<\s*Subject\s*(\d+)\s*>", re.I)
_SPEAKER_RE = re.compile(r"\(S(\d+)\)")
_DIALOG_RE = re.compile(r"<d>(.*?)</d>", re.S)
# split 用：奇数位就是完整的 <d>…</d> 台词块（整块跳过，不改写）
_DIALOG_SPLIT_RE = re.compile(r"(<d>.*?</d>)", re.S)
# 前后不能用 \b：Python 里汉字算 \w，"约00:01.600" 的 0 前面紧贴汉字就没有
# 词界，整句时间码会被漏掉（校验失真的老毛病）。改用「前后不能是数字/冒号/点」。
#
# ★★ 2026-09-27 修尾部前瞻：原来写的是 ``(?![\d:.])``，把紧跟的**冒号**和
#    **句号**也一并排除了。而台词行里时间码最常见的两种收尾恰恰就是这两个：
#      ``From 00:03.000: Her voice maintains…``   （冒号引出说明）
#      ``…finished before 00:08.000.``            （句末句号）
#    于是这些时间码对 ``_timecodes`` **完全不可见**，而 ``_timecodes`` 正是
#    尾窗有台词 / 回放区有台词 / 时间码递增 三条守卫的唯一判据 —— 守卫全部
#    静默放行，台词留在「会被下一段原样回放」的尾窗里 → 成片重复语音；
#    模型也拿不到时间约束 → 自己决定什么时候说 → **乱说话**。
#    实测本机实例：S02 台词行写着 00:03.000 / 00:08.000，``_timecodes`` 返回
#    空列表，尾窗守卫一声不吭，而那一段正是「被切穿的 7 秒台词」所在。
#    现在只排除紧跟的**数字**（避免把 ``00:03.0001`` 这种更长的数截断），
#    与 ``_SHOT_CLOCK_RE`` 的尾部口径保持一致。
_TIMECODE_RE = re.compile(
    r"(?<![\d:.])(\d{2}):(\d{2})(?:\.(\d{1,3}))?(?!\d)")
# ★ 汉字 vs 全角标点必须分开。旧写法 ``[一-鿿　-〿＀-￯]`` 把 U+FF0C（，）
#   也算成「中文字符」，于是英文单版 PACK 里每个 ``At 00:07.200，cut to …``
#   都触发一次「台词块之外出现中文字符」—— 实测本机 4 段全被误报，
#   on_issue=error 时能直接掐掉一次没问题的渲染。
#   §1.3 管的是**汉字**，不是标点；全角标点走 _FW_PUNCT_RE 单独处理。
_CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")          # 汉字（含扩展 A / 兼容）
# ---- 台词块之外的中文剥离（英译节点与解析节点共用）-------------------------
# ★ 为什么放在本模块：``prompt_pack.py`` 是**零外部依赖**（只 import math/re），
#   所以 translate_nodes（要 comfy_api）和 pack_nodes（要 comfy）都能安全地
#   从这里取用，不必各写一份。剥离逻辑只跟文本有关，放这里最合适。
#
# ★ 与 ``residual_cjk_segments``（在 translate_nodes）同源但**不吃换行**：
#   那份的字符类末尾是 ``\s*``，`re.sub` 会把整段中文后面的换行一并吞掉 ——
#   那会把两行粘成一行，段落的条目结构直接毁掉。这里把空白收窄成
#   「空格 / 制表符」，换行原样保留。
_CJK_RUN_STRIP_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff，。、！？；：（）()「」『』"
    r"\u201c\u201d\u2018\u2019—…·《》【】,.;: \t]*")


def cjk_outside_dialogue(text):
    """``<d>`` 台词块**之外**的汉字个数（交付闸门的判据）。"""
    return len(_CJK_RE.findall(_DIALOG_RE.sub("", str(text or ""))))


def _strip_cjk_runs(text):
    """把一段文本里所有连续中文片段换成空格，返回 ``(清理后, 剥掉几处)``。"""
    src = str(text or "")
    runs = _CJK_RUN_STRIP_RE.findall(src)
    if not runs:
        return src, 0
    out = _CJK_RUN_STRIP_RE.sub(" ", src)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+$", "", out, flags=re.M)
    return out, len(runs)


def strip_residual_cjk(text, store=None):
    """把 ``<d>`` 台词块**之外**的连续中文片段整段剥掉 → ``(cleaned, n_runs)``。

    ★ 这是英文单版的**最后一道闸门**，存在的理由是代价不对称：
      · 让中文留在提示词里 → H3 把中文读成台词内容 → 成片**计划外发声 /
        与剧本对不上**（质检技能 §4.4 G 类，本机实测过：201 个汉字漏进去）；
      · 剥掉几个字 → 丢的是一句镜头说明里的一小截，画面质量影响有限。
      两害相权，**宁可剥掉，也绝不放中文进提示词**。

    台词块先单独摘出来逐字保留（它的中文是**规范要求**的
    ``<d>[Chinese] 原文</d>``，剥了就丢台词）。``store`` 参数只为兼容调用方
    签名保留，函数本身按 ``<d>…</d>`` 就地识别，不依赖它。
    """
    src = str(text or "")
    parts = []
    pos = 0
    n = 0
    for m in _DIALOG_RE.finditer(src):
        seg, k = _strip_cjk_runs(src[pos:m.start()])
        parts.append(seg)
        n += k
        parts.append(m.group(0))
        pos = m.end()
    seg, k = _strip_cjk_runs(src[pos:])
    parts.append(seg)
    n += k
    return "".join(parts), n
# 全角标点与全角拉丁（不含汉字块）。英文单版里它是排版残留，不是语言问题。
_FW_PUNCT_RE = re.compile(r"[\u3000-\u303f\uff01-\uff60]")
# 归位表：先按「全角 ASCII → 半角」通铺，再对真正会改变阅读节奏的几个补空格。
# 不收进表的（如 ￠￥）就不会被 _FW_PUNCT_RE 命中，保持原样。
_FW_PUNCT_MAP = {c: chr(c - 0xFEE0) for c in range(0xFF01, 0xFF5F)}
_FW_PUNCT_MAP.update({
    0x3000: " ",            # 表意空格
    0x3001: ", ",           # 、
    0x3002: ". ",           # 。
    0x300C: '"', 0x300E: '"',   # 「 『 —— 交给 _QUOTE_RE 统一删
    0x300D: '"', 0x300F: '"',   # 」 』
    0xFF01: "! ", 0xFF0C: ", ", 0xFF0E: ". ",
    0xFF1A: ": ", 0xFF1B: "; ", 0xFF1F: "? ",
    0xFF08: " (", 0xFF09: ") ",
    0xFF5F: "(", 0xFF60: ")",
})
_FW_PUNCT_MAP = str.maketrans(_FW_PUNCT_MAP)
_QUOTE_RE = re.compile(r"[\"“”]")
# 英文单版文件结尾的 END OF PROMPTS / END OF PACK
_ENDPROMPTS_RE = re.compile(r"^\s*END\s+OF\s+(?:PROMPTS|PACK)\b.*$", re.I)
# SKILL §5 的分区标题：``ENGLISH VERSION / 英文版（本文件唯一语言版本）``
_ENVER_RE = re.compile(r"^\s*ENGLISH\s+VERSION\b", re.I | re.M)
# 文件形态标签：SKILL §5 的现行交付形态（头部 + ENGLISH VERSION + EN 段）
EN_PACK_LAYOUT = "en-pack"
# STYLE / CAMERA / LIGHTING block
_STYLEBLOCK_RE = re.compile(r"STYLE\s*/\s*CAMERA\s*/\s*LIGHTING", re.I)
# 英文词计数
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# ---------------------------------------------------------------------------
# [Shot N] 分镜脚本（六段式 + 镜内时间码，另一种常见交付形态）
# ---------------------------------------------------------------------------
# [Shot 1] At 00:00.000, "破门惊鸳鸯"——节奏：急起、骤顿、转冷。
#
# ★ markdown 容错（2026-09-24）：作者常把整段式分镜脚本写成 markdown，
#   镜头标记带标题前缀 —— ``### [Shot 1] 00:00.0–00:10.0 "碎瓶惊坛"``。
#   原来要求 ``[Shot N]`` 顶在行首，这种稿子一个镜头都认不出来（段数 0），
#   节点直接报「没有找到分镜段」。这里放过标题前缀与列表符号，
#   ``rest`` 仍从 ``]`` 之后开始取（时间码范围取第一个 = 本镜起点）。
_SHOT_HEAD_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[-*>]\s*)?\[\s*Shot\s*(?P<no>\d+)\s*\]\s*(?P<rest>.*)$",
    re.I | re.M)
# 只吃「[Shot N]」这个标记本身（保留后面的正文），段内重编号用。
# ★ 前缀必须一起捕获再原样吐回 —— ``_renumber_shots`` 用的是 sub()，
#   替换的是**整个匹配**，不带上 ``### `` 会把 markdown 标题前缀吃掉。
_SHOT_TAG_RE = re.compile(
    r"^(?P<pre>\s*(?:#{1,6}\s*)?(?:[-*>]\s*)?)\[\s*Shot\s*\d+\s*\]", re.I | re.M)
# mm:ss.xxx。比 _TIMECODE_RE 严：前面不能是数字/点/冒号，避免把 "1.2"
# 这类小数或比例误判成时间码。
_SHOT_CLOCK_RE = re.compile(r"(?<![\d.:])(\d{1,2}):([0-5]\d)\.(\d{1,3})(?!\d)")
# 「条目式」时间码：时间码打头（可带 [Shot N] / 空白前缀）。配乐表的
# "00:12.0-00:17.0 三弦接棒…" 命中；散文里内嵌的 "…（00:06.0-00:07.2）…" 不命中。
_ENTRY_CLOCK_RE = re.compile(
    r"^\s*(?:\[[^\]]*\]\s*)?(?:\d{1,2}:[0-5]\d\.\d{1,3})\s*(?:[-–—~至到]|\S|$)")
# 台词行补 (Sx) 用的说话人提示。**必须**紧跟"以/用/开/说"这类动词：
# "S1以中低音区说话" 才是真的在标说话人；"S1 的侧影只作画框边缘存在"里的
# S1 只是个路人，照它补会把张三的台词标成 (S1)。宁可不补，不能补错。
# ★★ 2026-09-26：这一条原先**只认中文**。而本包的交付形态按 SKILL §1.3/§5
#    就是英文单版 —— 英文脚本里 ``S1 speaks in a mid-low register`` /
#    ``S1's voice`` 一个都匹配不上，于是所有未标编号的台词全部掉到「段内
#    ``<Subject N>`` 众数」那个兜底上。实测本机 5 句台词里 2 句被错标
#    （S1 的台词标成 S2）→ 成片张三说了王总的词，即「乱说话」。
#    英文侧只收**高置信**搭配（S1 + 说话动词 / S1 + 声线名词），
#    宁可不补也不能补错的原则不变。
_SPEAKER_HINT_RE = re.compile(
    r"(?<![\w<])S([1-9])(?="
    r"(?:的声音|的嗓音|的音色|以|用|开|说|问|答|接|念|唱|讲|喊|吐|低声|开口|发声)"
    r"|(?:'|’)?s\s+(?:voice|line|lines|tone|timbre|register|delivery|speech)\b"
    r"|\s+(?:speaks?|says?|asks?|answers?|replies|delivers?|utters?|voices?"
    r"|sings?|shouts?|whispers?|mutters?|murmurs?|cries|calls?)\b"
    r")", re.I)

# ---- 台词「缺一不可」：<d> 块的语言标签补全 -------------------------------
# 每句台词固定 (Sx) <d>[语言] 原文</d>。(Sx) 由 _tag_speakers 补，
# 语言标签由下面 helper 补：已有 [语言] 原样保留，缺则按块内有无中文补
# [Chinese] / [EN]。删繁就简：补全只做「加标签」，不改写台词正文。
_LANG_TAG_IN_D_RE = re.compile(r"^\s*\[([A-Za-z][A-Za-z0-9-]*)\]")


def _ensure_dialog_lang_tag(block):
    """给单个 `<d>…</d>` 块补 [语言] 标签（已有则原样保留，缺则按 CJK 判）。"""
    m = re.match(r"<d>([\s\S]*?)</d>", block)
    if not m:
        return block
    inner = m.group(1).strip()
    lm = _LANG_TAG_IN_D_RE.match(inner)
    if lm:
        lang, body = lm.group(1), inner[lm.end():].strip()
    else:
        lang, body = ("Chinese" if _CJK_RE.search(inner) else "EN"), inner
    return "<d>[%s] %s</d>" % (lang, body)


def _ensure_dialog_lang_tags(text):
    """对整段文本里每个 `<d>…</d>` 块补语言标签。"""
    def _sub(m):
        return _ensure_dialog_lang_tag(m.group(0))
    return re.sub(r"<d>.*?</d>", _sub, str(text or ""), flags=re.S)


# ---- 畸形 `<d>` 扁平化（嵌套 / 重复开闭）------------------------------------
# ★ 2026-09-27 新发现（第 2 轮 QC）：英译节点的「二次翻译救回」路径会产出
#   ``<d>[Chinese] <d>[Chinese] 这叫资源优化配置！…</d>`` 这种**嵌套**块。
#   危害极大且**完全静默**：
#     · ``_inject_lipsync`` / ``_inject_dialogue_once`` 都在 ``</d>`` 后追加纪律句，
#       而嵌套块让**第一个** ``</d>`` 落在错误位置 —— 实测整段
#       ``DELIVERY: … Lip-sync: …``（上百字英文指令）被**吞进台词块内部**，
#       模型把那串指令当成台词内容 → 正是「乱说话 / 与剧本对不上」。
#     · ``_talk_spans`` 取不到干净的时间码 → 台词窗口失真。
#   ``_norm_dialogue`` 用贪婪 ``[\s\S]*`` 匹配，会把畸形**原样固化**（不是制造者
#   也不是修复者），所以必须在规范化的**入口**先扁平化。
#
# ★ 2026-09-27 修正（离线用例抓出）：旧式 ``<d\b[^>]*>\s*(?=<d\b)`` 只能匹配
#   **两个开局直接相邻**（``<d><d>`` / ``<d> <d>``）的情形；而本模块自己的头号
#   示例 —— ``<d>[Chinese] <d>[Chinese] …`` —— 中间夹着 ``[lang]``，旧式**一个
#   都匹配不到**，等于这条防御从未在它真正该管的场景上生效过。
#   现在把「开局 → 可选 [lang] → 下一个开局」整段纳入（group 1 = 外层整段），
#   且只在**下一个开局前没有 ``</d>``** 时才命中（``(?![^<]*</d>)``），避免把
#   ``<d>[Chinese] A</d> … <d>[Chinese] B</d>`` 这种**两个正常块**误判成嵌套。
_MALFORMED_D_RE = re.compile(
    r"(<d\b[^>]*>(?:\s*\[[A-Za-z][A-Za-z0-9-]*\])?\s*)(?=<d\b)(?![^<]*</d>)")


def flatten_dialogue_tags(text):
    """把畸形的嵌套 / 重复开局 `<d>` 块压平成单层 → ``<d>[lang] 原文</d>``。

    只做「结构扁平化」，**不改写台词正文**：

        ``<d>[Chinese] <d>[Chinese] 正文</d>``  →  ``<d>[Chinese] 正文</d>``

    判据（两层都命中才算畸形，避免误伤正常行）：
      1. 某个 `<d>` 之后（可含空白）紧跟另一个 `<d>`；
      2. 或块内出现 `[语言]` 标签重复（``[Chinese] [Chinese]``）。

    返回 ``(flattened_text, n_fixed)``，``n_fixed`` = 扁平化的块数，供调用方
    写进 issues / fixes —— **不能静默**（这正是本轮漏判的原因）。
    """
    src = str(text or "")
    # ① 连续开局：<d>[lang] … <d>[lang] … —— 把**外层**开局连同它的 [lang]
    #   整段删掉，只留内层那个开局（+ 它自己的 [lang]）。
    #   例：``<d>[Chinese] <d>[Chinese] 正文</d>`` → ``<d>[Chinese] 正文</d>``
    #   注意不能替换成 ``<d>``：那会把内层之前的语言标签一起吞掉，
    #   反而产出 ``<d><d>[Chinese] …``（少一层但语言标签错位）。
    def _collapse(m):
        return ""
    fixed = src
    n = 0
    while True:
        new = _MALFORMED_D_RE.sub(_collapse, fixed)
        if new == fixed:
            break
        fixed = new
        n += 1
    # ② 语言标签重复：<d>[Chinese] [Chinese] 正文 —— 只留第一个。
    fixed, k = re.subn(r"(<d>\s*)\[([A-Za-z][A-Za-z0-9-]*)\]\s*\[([A-Za-z][A-Za-z0-9-]*)\]\s*",
                       r"\1[\2] ", fixed)
    n += k
    return fixed, n


def count_malformed_dialogue(text):
    """畸形 `<d>` 块计数（供交付闸门 / issue 用）。"""
    src = str(text or "")
    n = len(_MALFORMED_D_RE.findall(src))
    n += len(re.findall(r"<d>\s*\[[A-Za-z][A-Za-z0-9-]*\]\s*\[[A-Za-z][A-Za-z0-9-]*\]", src))
    return n


def _dialog_missing_lang(line):
    """该行有幾個 `<d>` 块还缺 [语言] 标签（用于「缺一不可」校验）。"""
    n = 0
    for m in _DIALOG_SPLIT_RE.finditer(line):
        body = re.sub(r"</?d>", "", m.group(1)).strip()
        if not _LANG_TAG_IN_D_RE.match(body):
            n += 1
    return n


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
    """规范接线表的帧数口径：``帧数 = 生成时长 × fps``（SKILL §0.1 / §4）。

    24fps 下 SKILL 给的是 **11s = 264 帧、12.6s = 303 帧** —— 即向上取整
    （``ceil``），不是「乘完再加一帧」。旧实现写的是 ``round(...) + 1``，11s
    因此得到 265，与 SKILL 明写的 264 差一帧。

    注意这是**规范口径**，不是 H3 真正生成的帧数——H3 只能出 17k+5 的数
    （5/22/39/…/243/…），Director 会把这个值再向上对齐到网格。
    """
    return int(math.ceil(float(seconds) * FPS - 1e-9))


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
        if (_SEG_RE.match(stripped) or _SEG_RE_BARE.match(stripped)
                or _SEP_RE.match(stripped)):
            break                                # 下一段 / 分块结束
        match = _match_field(stripped)
        if match:
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


def _detail_section_text(lines):
    """PACK 行数组里 ``detailed_description`` 那一节的正文（含 ``[Shot N]`` 行）。

    推语言用：整段式 ``[Shot N]`` 脚本没有段标题，只能靠这一节 —— 绝不能拿整篇，
    PACK 头尾都带中文会把英文单版判成中文版。找不到该节返回 ``""``。
    """
    start = None
    for i, line in enumerate(lines):
        match = _match_field(str(line).strip())
        if match and match.group("name") == "detailed_description":
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for i in range(start, len(lines)):
        if _match_field(str(lines[i]).strip()):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def lang_section_available(text: str, lang) -> bool:
    """这份 PACK 里**真的**存在该语言的内容吗？

    两种 PACK 形态都要答对：

    * 双语 PACK（``[1] 中文版`` / ``[2] 英文版`` 分块）—— 有对应分块才算有。
    * 单语 PACK（现行 ``minimax-h3-shot-segment`` 技能只交付英文版，没有任何
      语言分块）—— ``_lang_block_range`` 对中英两侧都返回 ``(None, None)``，
      此时按正文推断出来的语言算数。

    ``lang`` 传 ``None`` 恒为 True（``None`` 走全篇定位，本来就对）。

    ★ 存在的意义：回写接口要拿它**拦下「往不存在的版块里写」**。以前不拦，
    英文单版 PACK 上误传 ``lang="zh"`` 会退回全篇定位，替换直接打到**英文
    段块**上（实测 PACK 44079 → 31643 字符、S01 英文正文 8469 → 0，界面表现
    就是"提示词莫名消失"）。
    """
    if lang not in ("zh", "en"):
        return True
    lines = str(text or "").split("\n")
    if _lang_block_range(lines, lang)[0] is not None:
        return True
    # 有语言分块标题、但没有这一侧 —— 真的没有。
    if any(_SEC_RE.match(line.strip()) for line in lines):
        return False
    # 单语 PACK（现行技能只交付英文版）：没有分块可依据，只能看**段正文**是什么语言。
    # ★ 千万别拿整篇文本去推断 —— PACK 的头尾都带中文：
    #   头部 `Project / 项目 : 曹贼的性价比`、尾部 `END OF PACK / 文件结束`，
    #   `_infer_lang` 会因此把英文单版判成中文版。实测这样
    #   has_en=False / has_zh=True 正好反了，守卫形同虚设，lang="zh" 照样
    #   把英文段块清空（44079 → 31643 字符）。只取**第一段正文**
    #   （首个段标题行 → 下一个段标题行，或 END OF PACK 之前）。
    heads = [i for i, line in enumerate(lines) if _SEG_RE.match(line.strip())]
    if not heads:
        # 整段式 [Shot N] 脚本没有 ########## 段标题可切。
        # ★ 绝不能拿**整篇**推语言 —— PACK 头尾都带中文（头部 `Project / 项目`、
        #   尾部 `文件结束`），英文单版会被判成中文版：has_en=False / has_zh=True
        #   正好反，守卫形同虚设，面板还会显示一个不该出现的「中文 ZH」标签。
        #   只取 detailed_description 那一节的正文（前端 h3PackLangs 同口径）。
        body = _detail_section_text(lines)
        return _infer_lang(body if body else text) == lang
    start = heads[0]
    end = heads[1] if len(heads) > 1 else len(lines)
    for i in range(start + 1, end):
        if _ENDPROMPTS_RE.match(lines[i].strip()):
            end = i
            break
    return _infer_lang("\n".join(lines[start:end])) == lang


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
    # 段块搜索/切分的**上界**。双语 PACK 指定语言时收窄到该分块末尾，否则整篇。
    # ★ 不收窄的话：双语 PACK 里末段的 span 会一路吃到文件尾，把后面那一侧的
    #   版块整块替换掉（实测 lang="zh" 改末段时 645 字符的 PACK 被打成 94）。
    limit = len(lines)
    # 双语 PACK + 指定语言版块：只在该版块行范围内定位段标题
    if lang in ("zh", "en"):
        lo, hi = _lang_block_range(lines, lang)
        if lo is not None:
            limit = hi
            heads = [(i, m.group("id"))
                     for i in range(lo, hi)
                     for m in [_SEG_RE.match(lines[i].strip())] if m]
        elif not lang_section_available(original, lang):
            # ★ 该语言分块不存在 —— **原样返回，一个字符都不改**。
            #   以前这里退回「全篇定位段头」，于是英文单版 PACK（现行技能只交付
            #   英文版，没有 [1]/[2] 语言分块）上只要前端误传 lang="zh"，替换就会
            #   打到**英文段块**上：实测 PACK 44079 → 31643 字符、S01 英文正文
            #   8469 → 0，界面表现是"提示词莫名消失"。宁可这次编辑不生效，
            #   也不能静默改错版块 —— 调用方看 lang_section_available() 就知道
            #   为什么没生效。
            return original
        else:
            # 单语 PACK + lang 与正文语言一致（例如单语英文 PACK 传 "en"）：
            # 没有分块可依据，退回全篇定位，行为与 lang=None 相同。
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
    edits = {}
    for seg in segments or []:
        try:
            ordinal = int(seg.get("index"))
        except (TypeError, ValueError):
            continue
        fields = seg.get("fields")
        if isinstance(fields, dict):
            edits.setdefault(ordinal, fields)
    if not ordered:
        # ★ 整段式 [Shot N] 脚本没有 ########## 段标题。以前这里直接
        #   ``return original`` —— **静默不生效**（时间线上改了提示词，原文一个
        #   字符都不动，界面还显示"已改"角标）。改用段在原文里的行区间定位。
        #
        #   六字段里只有 ``detailed_description`` 是"每段一段"，其余五段
        #   （subject_definitions / summary / retention_analysis /
        #   overall_soundscape / non_diegetic_music）是**全片共享**的 ——
        #   只把 detailed_description 写进该段区间，绝不把整块六字段塞进去
        #   （那会把全局段重复到正文中间，直接毁稿）。
        ranges = shot_script_segment_ranges(original)
        spans = []
        for ordinal, rng in enumerate(ranges, start=1):
            if ordinal not in edits or not rng:
                continue
            body = str((edits[ordinal] or {}).get("detailed_description") or "")
            spans.append((rng[0], rng[1], body.rstrip("\n").splitlines()))
        if not spans:
            return original
        result = lines[:]
        for start, end, block in sorted(spans, key=lambda s: -s[0]):
            result[start:end] = block
        return "\n".join(result).strip("\n") + "\n"
    # 序号 → 段头行号（段标题可能重复，安全起见取最后一次）
    head_by_ord = {}
    for ordinal, i in ordered:
        head_by_ord[ordinal] = i

    spans = []  # (start, end, new_block_lines)
    ords = sorted(head_by_ord)
    for k, ordinal in enumerate(ords):
        if ordinal not in edits:
            continue
        start = head_by_ord[ordinal] + 1          # 段标题行之后
        end = head_by_ord[ords[k + 1]] if k + 1 < len(ords) else limit
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
    # 同 rebuild_pack：段块切分上界默认整篇，双语指定语言时收窄到该分块末尾。
    limit = len(lines)
    if lang in ("zh", "en"):
        lo, hi = _lang_block_range(lines, lang)
        if lo is None:
            if not lang_section_available(original, lang):
                # ★ 这一侧版块不存在就原样返回，绝不退回全篇定位
                #   （否则英文单版 PACK 上误传 lang="zh" 会整段覆写英文块）。
                return original
            # 单语 PACK + lang 与正文语言一致：没有分块可依据，退回全篇定位。
            scan = range(len(lines))
        else:
            limit = hi
            scan = range(lo, hi)
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
    edits = {}
    for blk in blocks or []:
        try:
            ordinal = int(blk.get("index"))
        except (TypeError, ValueError):
            continue
        body = blk.get("body")
        if isinstance(body, str):
            edits.setdefault(ordinal, body)
    if not heads:
        # ★ 整段式 [Shot N] 脚本没有 ########## 段标题 —— 以前直接
        #   ``return original``：**静默不生效**，前端「分镜时间线」上编辑提示词
        #   写回后原文一个字符都不动。改用段在原文里的行区间定位，与解析时的
        #   分段同源（``shot_script_segment_ranges`` 跑的是同一条流水线）。
        #   抽出什么就写回什么：区间内是**该段那段正文**，不含全片共享的五段。
        ranges = shot_script_segment_ranges(original)
        spans = []
        for ordinal, rng in enumerate(ranges, start=1):
            if ordinal not in edits or not rng:
                continue
            body_lines = str(edits[ordinal]).rstrip("\n").splitlines()
            spans.append((rng[0], rng[1], body_lines))
        if not spans:
            return original
        result = lines[:]
        for start, end, block in sorted(spans, key=lambda s: -s[0]):
            result[start:end] = block
        return "\n".join(result).strip("\n") + "\n"
    # 序号 → 段头行号（重复段头取最后一次，与 rebuild_pack 一致）
    head_by_ord = {}
    for ordinal, i in heads:
        head_by_ord[ordinal] = i

    spans = []  # (start, end, new_block_lines)
    ords = sorted(head_by_ord)
    for k, ordinal in enumerate(ords):
        if ordinal not in edits:
            continue
        start = head_by_ord[ordinal] + 1
        end = head_by_ord[ords[k + 1]] if k + 1 < len(ords) else limit
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
    fields / pictures / videos / subjects / speakers / task / issues / fixes``。

    ``meta["layout"]`` 标出认出来的文件形态：

    * ``en-pack``         — **现行规范**（SKILL §5）：PACK 头部 +
                            ``ENGLISH VERSION`` 分区 + EN 段 + ``END OF PACK``。
    * ``pack-single``     — 有 ``=====`` 头部但没有语言分区（早期单版形态）。
    * ``en-single``       — 只有英文六段，无头部（SKILL §0.2 提到的旧交付形态）。
    * ``shot-script``     — 整段式分镜脚本（``[Shot N]`` 一路写到片尾），
                            段长由镜内时间码反推。
    * ``bilingual-pack``  — 老的 PACK：有 ``[1] 中文版`` / ``[2] 英文版`` 分块，
                            只取英文版那一侧。
    """
    raw = str(text or "")
    # ★ 2026-09-30：剪贴板粘贴的文本常带 BOM（\ufeff）。它不是 \s，会卡死
    #   PACK 头部/段头的一切行首正则（``^`` 之后直接是 \ufeff），整份稿子
    #   静默落进错误形态。入口处剥一次，零成本。
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    issues = []
    meta = _parse_header(raw)
    # ★ 台词句被引号引用在正文里（英文单版特有）—— 必须用**入口 raw**，
    #   因为下游的「双引号自动删除」会把引号先删掉，届时判据必然落空。
    issues.extend(_quoted_line_refs(raw))
    # ★ 畸形 `<d>` 块（嵌套 / 重复开局）**必须在一切规范化之前**压平：
    #   `_norm_dialogue` 用贪婪 `[\s\S]*`，会把畸形原样固化；一旦固化，
    #   `_inject_lipsync` / `_inject_dialogue_once` 追加在 `</d>` 之后的
    #   DELIVERY/Lip-sync 纪律句会被吞进台词块 —— 模型把它当台词念出来，
    #   正是「乱说话 / 与剧本对不上」。两条判据这里**必须都报**（不能静默：
    #   静默正是上一轮漏判的原因）。修改后的文本以 flattened 继续走全流程。
    raw, n_flat = flatten_dialogue_tags(raw)
    if n_flat:
        issues.append(
            "台词块畸形（嵌套 / 重复开局 `<d>`）：已自动压平 %d 处。畸形块会让"
            "追加在 `</d>` 后的台词纪律句被吞进台词内部、被模型当台词念出来"
            "（「乱说话」的直接成因）。请回查上游为什么产出嵌套 —— 常见于"
            "翻译节点的「二次翻译救回」路径。" % n_flat)

    sections = split_sections(raw)
    chosen = _pick_section(sections, language)
    if chosen is None:
        return [], meta, ["没有解析到任何分镜内容：请粘贴分镜提示词文本。"]
    lang, body = chosen

    has_seg_head = any(_SEG_RE.match(line.strip()) or _SEG_RE_BARE.match(line.strip())
                       for line in body.splitlines())
    is_shot_script = bool(_SHOT_HEAD_RE.search(body)) and not has_seg_head
    if any(sec_lang for sec_lang, _ in sections):
        layout = "bilingual-pack"
    elif is_shot_script:
        layout = SHOT_SCRIPT_LAYOUT
    elif meta and _ENVER_RE.search(raw):
        layout = EN_PACK_LAYOUT
    elif meta:
        layout = "pack-single"
    else:
        layout = "en-single"
    meta["layout"] = layout
    meta["rule_version"] = RULE_VERSION

    # 规范 §5 要求交付物以 PACK 头部开头（Project / Mode / 总时长 / 段数 / 画幅 /
    # Music）。整段式分镜脚本（shot-script）天然没有头部 —— 它不是报错，但要
    # 让用户知道：面板头部只能靠反推，且这份交付物与 skill 的分段交付形态不同。
    if layout == SHOT_SCRIPT_LAYOUT and not any(
            meta.get(k) for k in ("Project", "项目", "Total duration", "总时长",
                                  "Segments", "段数")):
        issues.append(
            "缺少 PACK 头部（规范 §5 的 Project / Mode / 总时长 / 段数 / 画幅 / "
            "Music 六行）。本文件是整段式分镜脚本，段数与总时长由节点反推；"
            "按 skill §0「先出镜头表再拆成多个 5–11 秒独立段」交付分段 PACK "
            "（########## S01 / 11s / EN ##########）即可消除本条，也让面板头部"
            "不再显示不全。")

    # 配乐判据的扫描范围有两条硬约束：
    #   1. 不扫 non_diegetic_music 自己写的 MUSIC block（那里必然出现 BGM/music，
    #      扫了会自我命中）；
    #   2. 不扫头部——老 PACK 头部有「Music / 配乐 : N/A」，自带「配乐」两个字，
    #      扫进去会让每一份 PACK 都误判成命中配乐。
    scan = _strip_music_field(body)
    scan_lines = scan.splitlines()
    for i, line in enumerate(scan_lines):                 # 跳到第一个段标题
        if _SEG_RE.match(line.strip()) or _SEG_RE_BARE.match(line.strip()):
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
        blocks, speaker_notes = parse_shot_script_blocks(
            body, lang, MAX_NEW_SECONDS)
        # 台词编号是推断来的（原文没标说话人）—— 必须让用户看得见，能复核
        issues.extend(speaker_notes)
    else:
        speaker_notes = []
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
        header_hard_cut = bool(item[5]) if len(item) > 5 else False
        seg = _build_segment(seg_id, dur_spec, seg_lang or lang, seg_body,
                             default_duration, index == 0, replay_in=replay_in,
                             header_hard_cut=header_hard_cut)
        if auto_fix:
            # 自动修过的项目**单独收在 seg["fixes"]**，不再混入 seg["issues"]
            # —— 两者是两类信号：「已自动补 no-text 首句」是成功日志，「台词行
            # 缺少 (Sx) 说话人编号」是问题。以前混在 issues 里被「注意：」标题
            # 印成一长串，用户分不清哪条是 bug 哪条是修好的。
            seg["fixes"] = _auto_fix(seg, music_hit=music_hit,
                                     music_style=music_style,
                                     is_last=index == len(blocks) - 1)
        seg["issues"].extend(validate_segment(seg, is_last=index == len(blocks) - 1,
                                              music_hit=music_hit))
        if layout == SHOT_SCRIPT_LAYOUT:
            seg["music_per_segment"] = True
        seg["task"] = infer_task(seg)
        segments.append(seg)
        issues.extend("%s：%s" % (seg["id"], item) for item in seg["issues"])

    # 规范 1.2：非说话角色要在 subject_definitions 里明写 never speaks / no
    # dialogue。判据是「全片从没当过 (SN) 说话人」。
    #
    # ★ 必须放到**所有段建好之后**再判：拿未经推断的原始 PACK 去扫 (Sx) 会一个
    #   都扫不到（剧本常常一句 (Sx) 都没写），结果连主角都被标成「全程不说话」
    #   —— 那等于告诉模型闭嘴。seg["speakers"] 是推断**之后**的说话人集合，
    #   用它才准。
    speaking = set()
    for seg in segments:
        speaking |= set(seg.get("speakers") or ())
    for seg in segments:
        sd = seg["fields"].get("subject_definitions") or ""
        if not sd.strip():
            continue
        # ★ ``speaking`` 为空时必须**整段跳过**：那时"从不说话"集合 = 全部角色
        #   （一个 (Sx) 都没推断出来），照着标下去等于命令所有人闭嘴。
        if speaking:
            silent = {int(x) for x in _SUB_RE.findall(sd)} - speaking
            if silent:
                sd = _mark_silent_subjects(sd, silent)
                # retention_analysis 里也要声明一次（SKILL §1.2：定义与段内
                # 状态两处都写）。只写定义不写本段状态，模型在"这一段要不要让
                # 它开口"上仍有自由度。
                seg["fields"]["retention_analysis"] = _append_silent_note(
                    seg["fields"].get("retention_analysis") or "", silent,
                    "; never speaks, no dialogue.")
        seg["fields"]["subject_definitions"] = sd

    # 跨段一致性（规范 4：no-text 句 / STYLE 块 / MUSIC 块全链逐字复用）
    issues.extend(validate_chain(segments))
    # ★ 作者文本保真闸门：节点不得改写作者的描述（只允许格式规范化）。
    #   放在**所有 transform 之后**才有意义 —— 它查的是"最终送出去的东西
    #   还是不是作者写的那句话"，任何一处新增的改写都会在这里现形。
    #
    #   只在**英文单版**上跑：中文稿本来就不是交付形态（正文里已经有
    #   「台词块之外出现中文字符」那条在报），而且此时节点会把中文定义
    #   改写成英文兜底句 —— 那是**翻译**不是改写，逐字比对必然满屏噪音
    #   （实测中文原稿上会多出 44 条）。交付形态是英文单版，闸门也只对它负责。
    if not _CJK_RE.search(_DIALOG_RE.sub("", body)):
        issues.extend(author_text_drift(_shot_global_fields(body), segments))
    # ★ 交付闸门：上面 `flatten_dialogue_tags` 已尽力压平；这里对**最终产物**
    #   再数一次畸形块。若仍有残留（说明压平判据没覆盖到的形态），必须报出来
    #   —— 畸形块的危害是「纪律句被吞进台词」，宁可误报也不能静默放行。
    for seg in segments:
        n_bad = count_malformed_dialogue(segment_prompt(seg["fields"]))
        if n_bad:
            issues.append(
                "%s：最终产物里仍有 %d 处畸形 `<d>` 块（压平未能覆盖）。"
                "这类块会把台词纪律句吞进台词内部并被模型念出来 —— 必须人工"
                "回查。" % (seg["id"], n_bad))
    # ★ 2026-09-30 静默丢正文闸门：整段式路径只保留**带时间码的镜头行**，
    #   [Shot 1] 之前的无时间码叙事散文会被行过滤悄悄丢掉（压测实证：拆段前
    #   数百块外汉字，拆段后只剩个位数，issues 一条不报）。数据丢失必须可见
    #   —— 损失超过 30% 且 ≥50 字时响亮报出，指引作者把散文写进镜头行。
    if layout == SHOT_SCRIPT_LAYOUT:
        _src_hanzi = len(_CJK_RE.findall(_DIALOG_RE.sub("", body)))
        _out_hanzi = sum(
            len(_CJK_RE.findall(_DIALOG_RE.sub(
                "", seg["fields"].get("detailed_description") or "")))
            for seg in segments)
        if _src_hanzi >= 50 and _out_hanzi < _src_hanzi * 0.7:
            issues.append(
                "整段式脚本里有 %d 个块外汉字，拆段后只保留 %d 个 —— 无时间码、"
                "不在任何 [Shot N] 行内的叙事散文会被镜头行过滤丢弃。请把散文"
                "写进镜头行内（跟在时间码后面），或改用分段 PACK 交付。"
                % (_src_hanzi, _out_hanzi))
    return segments, meta, issues


def _split_segments(body, fallback_lang):
    """按段标题切块 → ``[(id, dur_spec, lang, body, replay_in, hard_hdr), ...]``。

    第 5 项 ``replay_in`` 只有 [Shot N] 脚本路径会填（段首回放与尾部窗口要
    分开算）；段标题路径一律 ``None``，表示「按 dur_spec 里的 +1.6 推」。
    第 6 项 ``hard_hdr``：段头语言槽位里写了 hard cut（2026-09-30 起，
    见 ``_SEG_LANG_HARD_CUT_RE``；以前该槽位解析失败后被静默丢弃）。
    """
    lines = str(body or "").splitlines()
    heads = [(i, m) for i, line in enumerate(lines)
             for m in [_SEG_RE.match(line) or _SEG_RE_BARE.match(line)] if m]
    if not heads:
        return []
    out = []
    for idx, (start, match) in enumerate(heads):
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines)
        # _SEG_RE_BARE 没有 dur/lang 组，统一走 groupdict 兜底
        gd = match.groupdict() or {}
        lang_slot = (gd.get("lang") or "").strip()
        out.append((match.group("id").strip(),
                    (gd.get("dur") or "").strip(),
                    _lang_of_tag(lang_slot) or fallback_lang,
                    "\n".join(lines[start + 1:end]),
                    None,
                    bool(_SEG_LANG_HARD_CUT_RE.search(lang_slot))))
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
# 2026-09-26 清理：`_MIN_SPLIT_SECONDS = 2.0` 随 `_plan_splits` 一并删除 ——
# 那套「按 Shot 独立切段」的规划器被 `_split_oversized_segments`（段内切一刀，
# 走 `_off_dialogue` 避台词）取代后，这个常量全包零引用。现在的碎段下限统一
# 由 `MIN_NEW_SECONDS` 表达，`_coalesce_short_segments` 负责把不足的并回去。


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

    ★★ 2026-09-27 回归修复：映射下界用 ``t0``，**不是** ``t0 + handoff``。
       （理由：段内 0 秒对应成片 ``t0 - handoff``，段内容起点 ``t0`` 映射后
       正好落在 ``handoff`` —— 即**回放窗刚结束的那一刻**，正是"本段开头
       即刻执行"的合法落点。若把下界写成 ``t0 + handoff``，映射时又加一次
       ``handoff``，就成了 ``2×handoff``（实测 ``1.625 → 3.25``），把正好
       写在段边界上的台词/镜头整体向后推 1.625 秒。本函数只做线性映射 +
       边界 clamp，不越位兜底。）

      当初误诊的来源：本机 fixture 里 ``[Shot 3]``（成片 12.0）出现在
      ``[14,25]`` 段的正文里，看起来像"被 clamp 进了回放窗"。真因其实是
      上游 ``_rebalance_shot_bounds`` 的**空桶回退把不属于该窗口的行搬了进来**
      （已在 ``_in_window`` 处修掉）。本函数只负责老老实实做线性映射 +
      边界 clamp，不再越位兜底。
    """
    def _sub(match):
        value = _clock_seconds(*match.groups())
        value = min(max(value, t0), t1)
        return _fmt_clock(value - t0 + handoff)
    return _SHOT_CLOCK_RE.sub(_sub, line)


def _tag_speakers(detail, notes=None):
    """台词行补全到「缺一不可」固定格式：``(Sx) <d>[语言] 原文</d>``。

    四个要素：说话人编号 (Sx) + <d> 块 + [语言] 标签 + 原文，缺一不可。
    编号在台词块**前面**（以前补在 ``<d>…</d>`` 后面，模型容易读成"这句说完
    才轮到 S1"）。本函数只补全（缺 Sx 补 Sx、缺 [语言] 补 [语言]），不改写
    台词正文。

    (Sx) 判据按可靠性排序，**前者落空才用后者**：

    判据按可靠性排序，**前者落空才用后者**：

    1. 行内说话人提示（``S1以中低音区说话``）—— 见 ``_SPEAKER_HINT_RE``。
    2. 本行台词块**之前**的 ``<Subject N>``（这句的主语就写在同行前半段）。
    3. 本段出现次数最多的 ``<Subject N>`` —— 戏份最多的角色最可能开口。

    ★ 第 2/3 条是 2026-09-19 补的。只有第 1 条时，六段中文 PACK 有 3 句补不上
    （``<d>[Chinese] 汝与曹贼何异？</d> 自 00:07.200 起说：`` 这行里没有任何
    S 提示，说话人只写在上一行 ``00:07.200 切 <Subject 2> 近景MCU``）。
    编号缺着，模型就得自己猜这句是谁说的 —— 那正是「乱说话」的入口。

    ★★ 为什么第 3 条用「众数」而不是「最近一个标签」：先用"最近"试过，
    S06 被标成了 (S3) —— 那段里 ``<Subject 3> 此时已溜到门口`` 恰好排在台词
    前面，可台词是张三的。离场/插叙这类**一次性次要描述**会盖过真正的主角；
    按出现次数取则不会（主角在整段里反复出现）。

    ★ 推断（而不是原文明示）的结果全部记进 ``notes``，由调用方写进 issues，
    用户能在解析报告里看到并复核 —— 静默补全等于把判断藏起来。
    """
    text = str(detail or "")
    counts = {}
    for sid in _SUB_RE.findall(text):
        counts[sid] = counts.get(sid, 0) + 1
    dominant = max(counts, key=lambda k: counts[k]) if counts else None

    out = []
    for line in text.splitlines():
        if "<d>" in line:
            # ① 缺 (Sx) 说话人编号时按上下文补（行内提示 → 行前 <Subject> → 段内众数）
            if not _SPEAKER_RE.search(line):
                head = line[:line.index("<d>")]
                hint = _SPEAKER_HINT_RE.search(head) or _SPEAKER_HINT_RE.search(line)
                sid = hint.group(1) if hint else None
                if sid is None:
                    subs = _SUB_RE.findall(head)
                    sid = subs[-1] if subs else dominant
                    if sid is not None and notes is not None:
                        # ★ 带上是哪一句：一段里有两句台词时，两条一模一样的
                        #   「(S2) 推断」用户根本分不出该改哪一句。
                        block = _DIALOG_RE.search(line)
                        tip = (block.group(1).strip()[:18] if block
                               else line.strip()[:18])
                        notes.append(
                            "台词「%s」的编号 (S%s) 由 <Subject %s> 推断"
                            "（原文未标说话人，请复核）" % (tip, sid, sid))
                if sid:
                    line = re.sub(r"(<d>.*?</d>)",
                                  r"(S%s) \1" % sid, line, count=1)
            # ② 台词块缺 [语言] 标签时补上 —— 台词固定 (Sx) <d>[语言] 原文</d> 缺一不可
            line = _ensure_dialog_lang_tags(line)
        out.append(line)
    return "\n".join(out)


def _shot_global_fields(text):
    """[Shot N] 之外的全局字段（六段里除 detailed_description 的其余五段）。"""
    fields = {}
    current = None
    in_shots = False
    # ★ 2026-09-27：先把「粘在上一段末尾的字段名」断开。
    #   模型（尤其是英译节点）很爱写 ``…natural movement.retention_analysis:``
    #   —— 字段名紧贴上一段句末，没有换行。而 ``_match_field`` 是**行首锚定**
    #   的，粘住的字段名一律认不出来：那一整段 retention_analysis 会被并进
    #   summary（用户看不见），``_seg_retention`` 只能退回兜底文案 ——
    #   而兜底把 **Subject 号当 Picture 号**用（Subject 3 绑的是 Picture 4，
    #   兜底却写 Picture 3 = 场景图）→ **角色绑错参考图**，成片里人换了个样。
    for line in _split_glued_fields(str(text or "")).splitlines():
        stripped = line.strip()
        if _BANNER_RE.match(stripped):
            break                    # 文档级横幅之后的「审阅材料」不属于正文
        if _SHOT_HEAD_RE.match(stripped):
            in_shots = True
            continue
        match = _match_field(stripped)
        if match:
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


# 六个官方字段名（行首锚定匹配用；也用于「粘行」断开）。
_GLUED_FIELD_HEAD_RE = re.compile(
    r"(?:^|(?<=[.。!！?？]))[ \t]*(?=(?:"
    + "|".join(re.escape(x) for x in FIELD_ORDER) + r")\s*[:：])", re.I)


def _split_glued_fields(text):
    """把「紧贴上一段句末、没有换行」的官方字段名断成独立一行。

    只认**句末标点之后**（``.`` ``。`` ``!`` ``？`` …）或行首出现的字段名，
    正文里提到 ``retention_analysis`` 这个词但后面不是冒号的不会被误断。
    断行只加换行符，不改任何字符 —— 行内容原样保留。
    """
    out = []
    for line in str(text or "").splitlines():
        pos = 0
        for m in _GLUED_FIELD_HEAD_RE.finditer(line):
            if m.start() <= pos:
                continue
            out.append(line[pos:m.start()])
            pos = m.start()
        out.append(line[pos:])
    return "\n".join(out)


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
            # ★ 保留 [Shot N] 标记（2026-09-20）：官方六段式的
            #   detailed_description 就是**按 [Shot N] 组织**的。以前只取 rest
            #   把标记剥掉了，段内退化成一坨没有结构的文本 —— 模型既看不出镜头
            #   边界，「取景纪律 / 节奏对位」也无处附着，retention_analysis 更
            #   没法按镜头声明。段内编号交给 _renumber_shots 从 1 重编。
            head = "[Shot %s]%s" % (match.group("no"),
                                    (" " + rest) if rest else "")
            current = (int(match.group("no")), rest, [head])
            blocks.append(current)
            # shot-script 布局（无 PACK 头部 / 无 detailed_description 字段头，
            # [Shot N] 直接跟在 retention_analysis 等全局字段之后）：原逻辑里
            # in_detail 只认字段头、此处恒 False，每个 [Shot N] block 只收 head
            # 一行、Shot 正文全丢；末镜无 body 又无后继时间码，_shot_bounds 推不出
            # end（end==start），_shot_script_plans 判 t_end<=t_start 跳过 -> 4 镜变 3 段。
            # 遇 [Shot N] 即标志进入正文区；全局字段行仍由下方 field 分支拦截。
            in_detail = True
            continue
        field = _match_field(stripped)
        if field:
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


# 台词「起点 / 终点」的语义锚。只有紧跟在这些词后面的时间码，才算这句台词
# 自己的窗口；行内顺带提到的别处时间码（"the 00:25.0 reverse shot"、"the
# 00:01.2 opening line"）前面没有起止动词，不会被收进来。
#
# ★ 为什么要按语义取，而不是整行取 min/max（2026-09-26 实测事故）：
#   一句 7 秒的台词，只因为在约束里引用了别处的 00:01.2 和 00:28.0，整行
#   min/max 就被撑成 (1.2, 31.0) —— 一句台词变成"占了 29.8 秒"。下游
#   ``_off_dialogue`` 拿这个 span 去避让，算出的避让点远超合法区间、被夹回
#   原处，等于避让彻底失效；结果段边界正好切在这句台词中间，而 ``_rebucket``
#   按行首时间码把整句只归给前一段 —— 后一段拿到 0 条台词，模型即兴发声，
#   成片出现剧本外的含糊人声。改成语义锚后，同一份输入恢复成 (23.5, 31.0)。
#
# 通用性：靠的是「剧本描述台词必然用 from/starting at … before/by … 」这个
# 写法惯例，与具体是哪一本剧本无关；提取不到就退回 min/max，不会变差。
_TALK_START_RE = re.compile(
    r"(?:\bfrom|\bstarting\s+at|\bbeginning\s+at|\bbegins\s+at|\bstarts\s+at"
    r"|\bcommencing\s+at|\bspoken\s+from|\bsays\s+from|\bopens\s+at"
    r"|\bdelivered\s+from)\s*[:(]?\s*(\d{1,2}):([0-5]\d)\.(\d{1,3})", re.I)
# ★ 2026-09-30 补 ``to``：英文交付稿里台词窗最常见的写法是
#   ``from 00:03.200 to 00:05.200, word by word…`` —— 而旧表**没有** ``to``，
#   于是 ``ends`` 恒为空 → ``if starts and ends`` 不成立 → 整行退回 min/max。
#   后果实测（v23full PACK 的 S05）：台词窗 ``from 00:03.200 to 00:06.870``
#   被本行镜头头 ``At 00:01.625`` 污染成 ``(1.625, 6.895)`` —— 起点落进回放窗，
#   ``_auto_fix`` 于是把整行时间码 ``+0.025s`` 平移（本该只动回放区内的那些），
#   且 ``_off_dialogue`` 把 1.625–6.895 整段当成"在说话"，避让点全部失效。
_TALK_END_RE = re.compile(
    r"(?:\bto|\bbefore|\bby|\buntil|\bconcluding|\bconcludes|\bending|\bends\b"
    r"|\bfinished|\bcompletes|\bcompleted|\bdelivered\s+by|\bno\s+later\s+than"
    r"|\bmust\s+be\s+(?:finished|delivered|completed)\s+(?:before|by))"
    r"\s*[:(]?\s*(\d{1,2}):([0-5]\d)\.(\d{1,3})", re.I)


def _talk_spans(lines):
    """每句台词在成片时间线上占的区间 ``[(start, end), ...]``。

    台词行的时间码写法是「自 00:12.0 起说……于 00:17.0 前说完」。

    ★ 优先用**起止语义锚**（``_TALK_START_RE`` / ``_TALK_END_RE``）定位，
    而不是整行 min/max —— 后者会把行内引用的别处时间码也算进这句的窗口，
    把一句台词撑成几十秒，直接破坏切段避让（详见上面正则处的实测事故）。
    只在语义锚缺失或自相矛盾时，才退回整行 min/max。

    切段时要用它做两件事：别把缝切在一句话的中间（规范 1.4：段尾 1.6 秒
    必须是无台词区，回放过去就是重复语音）。
    """
    spans = []
    for line in lines or ():
        if "<d>" not in line:
            continue
        codes = [_clock_seconds(*g) for g in _SHOT_CLOCK_RE.findall(line)]
        if not codes:
            continue
        starts = [_clock_seconds(*g) for g in _TALK_START_RE.findall(line)]
        ends = [_clock_seconds(*g) for g in _TALK_END_RE.findall(line)]
        if starts and ends:
            # ★ 2026-09-30：起止锚必须**成对**取 —— 取 min(starts)/max(ends)
            #   会把行内其它起止对（"up to 00:12.0"、下一句的 "by 00:20.0"）
            #   也算进来，窗口被撑大。改为「第一个起点锚 → 其后第一个终点锚」。
            m_start = _TALK_START_RE.search(line)
            m_end = (_TALK_END_RE.search(line, m_start.end()) if m_start else None)
            if m_start and m_end:
                lo = _clock_seconds(*m_start.groups())
                hi = _clock_seconds(*m_end.groups())
                # 语义锚必须落在这行真实出现过的时间码范围内（别凭空造窗口），
                # 且终点不能早于起点 —— 否则说明是误配，退回 min/max。
                if (min(codes) - 1e-6 <= lo <= hi <= max(codes) + 1e-6):
                    spans.append((lo, hi))
                    continue
        spans.append((min(codes), max(codes)))
    return spans


def _coalesce_short_segments(plans, max_new, min_new=MIN_NEW_SECONDS):
    """对太短的段尝试向相邻段合并 —— 跨 Shot 边界也能合并。

    为什么必须内建（2026-09-19）：``_split_oversized_segments`` 在每个 ``[Shot N]`` 内独立切段，
    Shot 边界是硬墙。长 Shot 被切到 ≤ max_new 时**必然**产生 < min_new 的尾巴 ——
    比如 ``[Shot 2]`` 12s 一刀切成 10+2，2 秒那段渲出来是接近静止的镜头，纯浪费。
    issue 里报"建议合并"也救不了：Shot 边界是脚本意图，**改剧本**才是常规修法；
    但剧本可以不动，由规划器自动把 2 秒尾巴并到下一段（跨 Shot），结果 12 秒一段，
    比 2 秒碎段强太多。

    ★ 策略：宁超 MAX 不产碎段（"超 MAX 上限"有 issue 但渲染影响小，"< min 过短"
      整段浪费 + 退化静止镜头）。合并后允许到 max_new + min_new 才动；超过这个
      上限就别并 —— 该 issue 就 issue，不该默默吞下。

    ★ 安全：第一段（index=0）永远不向"前"并（没有前一段）所以不会被吞掉回放
      区；最后一段永远不向"后"并；合并只动 plans 本身，不重算时间码（lines 在
      prompt 渲染端按 [Shot N] 收口，自带时间码，按行首时间码归桶）。
    """
    if not plans or len(plans) < 2:
        return plans
    cap = max_new + min_new                  # 合并后允许的上限
    out = [list(p) for p in plans]           # 转成可变（每项 [a, b, lines]）
    i = 1
    while i < len(out):
        cur_a, cur_b, cur_lines = out[i]
        if cur_b - cur_a >= min_new - 1e-3:
            i += 1
            continue
        # 优先向左并：上一段"多背一点"比下一段"原地延长"更自然（机位与姿态
        # 沿用前段的回放锚点）。若合并后超 cap，就别动它，去看下一段。
        prev = out[i - 1]
        merged_dur = cur_b - prev[0]
        if merged_dur <= cap + 1e-3:
            out[i - 1] = [prev[0], cur_b, prev[2] + cur_lines]
            del out[i]
            continue                            # 不前移 i：当前位置的新段可能仍太短
        i += 1
    return [tuple(p) for p in out]


# ``_snap_cuts_off_dialogue`` 挪边界时，每段**至少**保留的秒数（见那个函数的
# min_part 说明）。
#
# ★ 2026-09-20 实测扫参（同一份 40 秒 PACK，违规数越少越好）：
#
#   | min_part | 过短 | 超长 | 尾窗违规 | 时间码倒流 | issues |
#   |----------|------|------|----------|------------|--------|
#   | 5.0      | 0    | 0    | **0**    | 2          | **6**  |
#   | 3.0/2.0  | 0    | 0    | 3        | 1          | 8      |
#   | 1.5      | 1    | 0    | 1        | 2          | 10     |
#
#   5.0 反而最好 —— 关键不是放宽下限，而是 ``_off_dialogue`` 的**夹取**：理想
#   位置出界时夹到边界而不是整个放弃。放宽下限只会切出碎段，不解决尾窗。
SNAP_MIN_PART = 5.0


def _inside_any(c, spans):
    """``c`` 是否落在任一 ``(s, e)`` 台词区间**内部**（端点不算）。"""
    return any(s + 1e-6 < c < e - 1e-6 for s, e in spans)


def _off_dialogue(c, spans, lo, hi):
    """把切点 ``c`` 挪开，别切断一句台词（规范 1.4「别把缝切在一句话的中间」）。

    优先推到「台词结束 + 1.6s 尾窗」——整句留在前一段，且段尾 1.6s 无台词，
    下一段回放过去不会变成重复语音；推不动就推到台词结束，再推不动就推到台词
    开始（整句留在后一段）。

    ★ 2026-09-20 加**夹取**：理想位置落在 ``[lo, hi]`` 之外时不再整个放弃，而
    是夹到区间里最接近目标的那个值 —— 挪 0.6 秒也比一动不动强（尾窗里残留的
    台词从 0.7s 降到 0.1s）。只有会往回挪（比当前 ``c`` 还小）时才放弃，那
    种情况返回 ``None``，由调用方决定要不要切这一刀。

    ★ 2026-09-27 夹取**必须再验一次**：夹取后的点可能仍然落在台词内部
    （实测：台词 23.5–31.0、合法区间 [26.0, 27.6]，夹取给出 27.6 —— 正好
    切在这句 7 秒台词的第 4.1 秒上）。切穿台词的代价是**两段各说半句 +
    后段整段零台词（模型即兴发声）**，比"这一段超长"严重得多。所以夹取后
    仍压着台词的，一律返回 ``None``，让调用方**别切这一刀**，超长交给
    ``validate_segment`` 报 issue 兜底（软失败，可读、可改）。
    """
    tail = []
    # ★ 尾窗偏移用 17k+5 对齐的 1.625s（grid_seconds），与 handoff_seconds /
    #   replay_in 同源，避免留下 1 帧残尾被回放。
    _hw = grid_seconds(HANDOFF_SECONDS)
    for s, e in spans:
        if s + 1e-6 < c < e - 1e-6:
            # 切在台词进行中 —— 必须挪开，挪不动就让调用方别切
            for cand in (e + _hw, e, s):
                if lo - 1e-6 <= cand <= hi + 1e-6 and not _inside_any(cand, spans):
                    return round(cand, 3)
            clamped = min(max(e + _hw, lo), hi)
            if clamped > c and not _inside_any(clamped, spans):
                return round(clamped, 3)
            return None
        if e - 1e-6 <= c < e + _hw - 1e-6:
            # 切点离台词结束不足尾窗：这段尾巴要被下一段回放，等于把这句的
            # 尾巴再说一遍。
            tail.append(e + _hw)
    if not tail:
        return round(c, 3)
    target = max(tail)
    if lo - 1e-6 <= target <= hi + 1e-6:
        return round(target, 3)
    clamped = min(max(target, lo), hi)
    return round(clamped, 3) if clamped > c else round(c, 3)


def _rebucket(lines, bounds):
    """按行首时间码把行重新归到 ``bounds`` 划出的各个窗口里。

    没有时间码的续写行跟着上一行走（它描述的就是上一个时间码之后的动作），
    不会跑到别的段去。
    """
    buckets = [[] for _ in range(len(bounds) - 1)]
    cur = 0
    for line in lines:
        head = _SHOT_CLOCK_RE.search(line)
        if head:
            t = _clock_seconds(*head.groups())
            cur = len(buckets) - 1
            for i in range(len(buckets)):
                if t < bounds[i + 1] - 1e-6:
                    cur = i
                    break
        buckets[cur].append(line)
    return buckets


def _snap_cuts_off_dialogue(plans, min_new=MIN_NEW_SECONDS, min_part=None,
                            max_new=MAX_NEW_SECONDS):
    """把已经存在的段边界从台词中间 / 台词尾巴上挪开。

    ``_split_oversized_segments``（段内切一刀）与 ``_coalesce_short_segments``（跨 Shot 合并）
    都可能让边界正好落在一句长台词进行中：两边各说半句，而且尾窗带着半句台词
    被下一段原样回放 → 成片出现重复语音 + 半截句子。这里统一把这类边界推到
    台词外；挪不动就保持原样，交给 issue 去报。

    ★ ``min_part``：挪动后每段**至少**保留的秒数。默认等于 ``min_new``（5s，
    绝对安全但常常挪不动 —— 「尾窗还差 0.6 秒，可下一段就只剩 5 秒」这种僵局
    直接卡死）。传更小的值（如 2.0）能让边界挪到位，挪完产生的短段交给
    ``_coalesce_short_segments`` 去并 —— 两者必须配合才解得开这个死结，所以
    调用方把它们放进同一个迭代里跑。

    ★★ 2026-09-27 **本函数必须"不更坏"**（实测真渲染事故）：本机输入
      ``[0,13][13,25][25,38][38,45]``，① ``_rebalance`` 已给出合规的
      ``[0,11][11,22][22,38][38,45]``（新增 11.0/9.4/14.4/5.4，只有第 3 段超）。
      但本函数为了"避开台词"把 ``[11,22]`` 的边界一路推到 26（挪了 15 秒），
      结果是：

        * 原本合规的 ``[11,22]``（新增 9.4s）被撑成 ``[11,26]``（新增 **13.4s**）；
        * 原本就超的 ``[22,38]`` 变成 ``[26,39.1]``（新增 11.5s，仍超）；
        * 末段被挤成 ``[39.1,45]``（新增 **4.3s**，低于 ``min_new``）。

      **一个超限段被治成了两个超限段 + 一个过短段**，而且 ``[26,39.1]``
      的台词 ``(28.6, 37.5)`` 占了 68% 跨度、两端又都被 ``min_new`` 吃掉，
      ``_split_oversized_segments`` 再也切不动 → 它带着 13.1s（渲染 14.7s，
      超 15s 硬上限的边缘）和"尾窗内有台词（38.9s）→ 下一段回放成重复语音"
      进了渲染 —— 正是用户报的「乱说话、与剧本对不上」+ 质检抓到的重复语音。

      所以挪动前先算**候选边界下的两侧新增时长**，只在「不越 ``max_new`` 且
      不破 ``min_new``」时才接受；避免"为了避 0.3 秒台词，制造 4 秒超限"。
      若两侧都合规而只是尾窗差一点点，仍按原逻辑挪（那是纯收益）。
    """
    if len(plans) < 2:
        return plans
    guard = float(min_part if min_part is not None else min_new)

    def _new_seconds(seg_index, a, b):
        """该段的**新增时长**（首段无回放，其余减一个回放窗）。

        ★ 回放窗用 ``grid_seconds``（= 1.625），与 ``_build_shot_script_segments``
          里写进字段的 ``replay_in`` 同源；用名义 1.6 会让本函数算出的
          「新增时长」比渲染器实际多 1 帧，边界避让随之偏 1 帧。
        """
        replay = 0.0 if seg_index == 0 else grid_seconds(HANDOFF_SECONDS)
        return (b - a) - replay

    out = [list(p) for p in plans]
    for i in range(1, len(out)):
        prev, cur = out[i - 1], out[i]
        if abs(prev[1] - cur[0]) > 1e-6:
            continue
        c = prev[1]
        spans = _talk_spans(prev[2]) + _talk_spans(cur[2])
        moved = _off_dialogue(c, spans, prev[0] + guard, cur[1] - guard)
        if moved is None or abs(moved - c) < 1e-6:
            continue
        moved = round(moved, 3)
        # ★ 挪动后两侧新增时长必须仍然合规 —— 否则宁可不挪（见上面的实测事故）。
        prev_new = _new_seconds(i - 1, prev[0], moved)
        cur_new = _new_seconds(i, moved, cur[1])
        if prev_new > max_new + 0.001 or cur_new > max_new + 0.001:
            continue
        if prev_new < min_new - 0.001 or cur_new < min_new - 0.001:
            continue
        buckets = _rebucket(prev[2] + cur[2], [prev[0], moved, cur[1]])
        # ★ 2026-09-27：边界推后时 ``buckets[1]`` 可能整个为空 —— 那一行都没有
        #   落在新窗口里。空段没有 ``<d>`` 行 = 剧本里这一段没人说话，模型遇到
        #   没有台词指令的空白会**自己即兴发声**（§4.4 E 类）。宁可不挪边界
        #   （把尾窗有台词交给 ``validate_segment`` 报 issue 让人决定），也不要
        #   造出一个零台词段。``_split_oversized_segments`` 里早就有对等保护，
        #   这里之前漏了。
        if not buckets[0] or not buckets[1]:
            continue
        prev[1], cur[0] = moved, moved
        prev[2], cur[2] = buckets[0], buckets[1]
    return [tuple(p) for p in out]


def _shot_start_of(plan):
    """计划段里**第一个 ``[Shot N]`` 头行的成片时间码**；没有则 ``None``。

    段边界滑动必须避让镜头起点（见 ``_rebalance_shot_bounds`` 的禁区说明）。
    镜头的真实起点只写在它自己的头行里（``[Shot 3] At 00:12.000, …``），
    计划元组的 ``a`` 只是"这一段的左界"，两者在滑动后就不再相等 —— 所以
    必须回到行文本里取，不能拿 ``a`` 顶替。
    """
    for line in (plan[2] if len(plan) > 2 else []):
        if not _SHOT_HEAD_RE.search(line):
            continue
        m = _SHOT_CLOCK_RE.search(line)
        if m:
            return _clock_seconds(*m.groups())
        return None
    return None


def _rebalance_shot_bounds(plans, max_new=MAX_NEW_SECONDS,
                           min_new=MIN_NEW_SECONDS,
                           slack=BOUND_SLACK_SECONDS, rounds=8):
    """在**不改段数**的前提下滑动镜头边界，让每段新增落回 ``[min_new, max_new]``。

    ``[Shot N]`` 就是 skill §0 说的镜头表：段数由它决定，不该再切碎。但镜头表
    的时长常常不均匀（本次 10 / 12 / 13 / 5 秒，两个越过 11 秒上限）。这时正确
    做法是**把一秒从长的镜头让给短的邻居**，而不是把长镜头切开 —— 切开会让一个
    叙事单元变成两段，凭空多一条接缝，也多一次回放带来的重复风险。

    做法：每个内部边界可在原镜头边界 ±``slack`` 内滑动；对某个边界而言两侧总长
    ``L`` 是固定的，于是左段的合法取值区间是
    ``[max(min_new, L - max_new), min(max_new, L - min_new)]``，取区间里离原值
    最近的点。相邻边界互相影响，所以扫多轮到稳定。

    ★ 区间为空（这一段对无论怎么滑都排不下）就跳过，留给后面的
    ``_split_oversized_segments`` 兜底 —— 不在明知不可解的地方硬凑。

    实测这份 40 秒 / 4 镜头剧本：``10/12/13/5`` → **``11/11/11/7``**，
    与人工分段逐值一致。
    """
    if len(plans) < 2:
        return plans

    bounds = [float(plans[0][0])] + [float(p[1]) for p in plans]
    orig = list(bounds)
    lo_lim = [max(orig[0], orig[i] - slack) for i in range(len(bounds))]
    hi_lim = [min(orig[-1], orig[i] + slack) for i in range(len(bounds))]
    lo_lim[0] = hi_lim[0] = bounds[0]          # 首尾边界是成片首尾，不动
    lo_lim[-1] = hi_lim[-1] = bounds[-1]

    # ★ 硬约束（两侧都要）：段边界不许落进「镜头起点 s 之后、HANDOFF 秒之内」
    #   （``(s, s+HANDOFF]``）。落在那里，该镜头的 `[Shot N]` 头（以及散文写法的
    #   切镜行）就留在**上一段的尾窗**里，而尾窗会被下一段原样回放 → 接缝处
    #   重演一次切镜（规范 §1.4），且该镜正文里对上一句台词的引用会在同一段里
    #   造成第二处台词文本。
    #   左端开、右端闭：边界正好等于 s（切点落在段首）合法；等于 s+HANDOFF 仍违规。
    #
    # ★★ 2026-09-27：**窗口要按镜头起点建，不是按计划起点建**。原来用
    #   ``p[0]``（计划左界，已被上一轮滑动改过）当锚，等于只在"上一轮的结果"上
    #   自洽；而镜头起点的**原始**位置才是切点必须避让的东西。实测 fixture：
    #   镜头起点 12.0，边界滑到 14.0 —— 14.0 是 ``(14.0, 15.6)`` 的左端点（合法），
    #   可 12.0 这个镜头起点落在 14.0 **之前** 2 秒，属于"留在上一段里但离切点
    #   不足 HANDOFF"，`_retime_line` 会把它 clamp 进回放窗。所以除"切点后"的
    #   禁区外，还要禁"切点前不足 HANDOFF 的镜头起点所在的那一小段"。
    starts = sorted({float(p[0]) for p in plans} |
                    {float(_shot_start_of(p)) for p in plans
                     if _shot_start_of(p) is not None})
    # 禁区宽度 = 一个回放窗，取 ``grid_seconds``（= 1.625）与 ``replay_in`` 同源。
    # 用名义 1.6 会让禁区比真实回放窗短 1 帧，正好放过"离切点 1.625 秒"的镜头
    # 起点 → 它被 clamp 进回放窗 → validate 报「回放区内有切点」。
    forbidden = [(s, s + grid_seconds(HANDOFF_SECONDS)) for s in starts]
    def _legal(value, lo, hi):
        """``[lo, hi]`` 内离 ``value`` 最近的合法边界；没有返回 ``None``。

        合法部分 = ``[lo, hi]`` 去掉若干 ``(s, s+HANDOFF]``。候选只可能在区间端点
        与各禁用区间的左端点 ``s``（它本身合法），枚举即可，不必扫连续量。
        """
        if lo > hi + 1e-9:
            return None
        cands = {lo, hi}
        for a, _c in forbidden:
            if lo - 1e-9 <= a <= hi + 1e-9:
                cands.add(min(max(a, lo), hi))
        best = None
        for x in cands:
            if any(a < x <= c + 1e-9 for a, c in forbidden):
                continue
            if best is None or abs(x - value) < abs(best - value):
                best = x
        return best

    for _ in range(rounds):
        changed = False
        for i in range(1, len(bounds) - 1):
            span = bounds[i + 1] - bounds[i - 1]
            want_lo = max(min_new, span - max_new)
            want_hi = min(max_new, span - min_new)
            if want_lo > want_hi + 1e-6:
                # 这一对排不下（两侧不可能同时落回 [min,max]）。**不能直接跳过**：
                # 跳过等于把边界丢在原地。退让成「先保住左侧 ≤ max_new」，右侧
                # 超出的一点留给邻居边界在后续轮次里匀掉。
                want_lo = want_hi = max(min_new, min(max_new, span - min_new))
            lo_a = max(bounds[i - 1] + want_lo, lo_lim[i])
            hi_a = min(bounds[i - 1] + want_hi, hi_lim[i])
            cand = None
            if lo_a <= hi_a + 1e-6:
                cand = _legal(min(max(bounds[i], lo_a), hi_a), lo_a, hi_a)
            if cand is None:
                # 严格区间整段违规 → 退到「不违规、且不超滑动上限」的最近点：
                # 宁可这一对略微失衡，也不能把切点留在尾窗里。
                cand = _legal(min(max(bounds[i], lo_lim[i]), hi_lim[i]),
                              lo_lim[i], hi_lim[i])
            if cand is None:
                continue
            new_b = round(cand, 3)
            if abs(new_b - bounds[i]) > 1e-6:
                bounds[i] = new_b
                changed = True
        if not changed:
            break

    if all(abs(bounds[i] - orig[i]) <= 1e-6 for i in range(len(bounds))):
        return plans                            # 一个边界都没动，省一次重排

    all_lines = []
    for _a, _b, lines in plans:
        all_lines.extend(lines)
    # 按**全部**边界一次性归桶再取各段：只传一对边界的话所有行都会落进唯一的
    # 那个桶里，等于没重排（无时间码的续写行由 _rebucket 自己跟着上一行走）。
    buckets = _rebucket(all_lines, bounds)

    def _in_window(lines, a, b):
        """只保留时间码确实落在 ``[a, b)`` 里的行。

        ★★ 2026-09-27（本机 fixture 实测的真凶）：空桶回退以前直接
        ``seg_lines = list(plans[i][2])`` —— 把**原计划那一整段的行**搬回来，
        完全不看它们的时间码是否属于新窗口。实测 ``[Shot 3] At 00:12.000``
        本该属于 ``[7,14)``，滑动后桶 2（``[14,25)``）为空，回退就把它塞进了
        段 3；``_retime_line`` 再把 12.0 clamp 到段首 → ``At 00:01.600``，
        落在回放窗内 → 校验报「回放区内有切点」。

        改为按窗口**过滤**：只有时间码在窗内的行才允许回来（它们的语义确实
        是"这段时间的动作"）；一行都不在窗内就让它保持为空 —— 空段由
        ``_coalesce_short_segments`` / 下游校验处理，都好过把别的段的行搬进来。
        无时间码的续写行（``_clock_seconds`` 取不到）跟着上一行走，判据与
        ``_rebucket`` 一致：以"最近一次出现的时间码"为准。
        """
        keep = []
        last_t = None
        for x in lines:
            m = _SHOT_CLOCK_RE.search(x)
            t = _clock_seconds(*m.groups()) if m else last_t
            if t is None:
                keep.append(x)                  # 段首就没有时间码：保留（无法判定）
                continue
            last_t = t
            if a - 1e-6 <= t < b - 1e-6:
                keep.append(x)
        return keep

    out = []
    for i in range(len(bounds) - 1):
        seg_lines = buckets[i] if i < len(buckets) else []
        if not seg_lines and i < len(plans):
            # 窗口空了（这段时间剧本没写新指令）→ 语义上是延续上一段的动作。
            # ★ 只回退**属于本窗口**的行（见 _in_window 说明）。
            seg_lines = _in_window(list(plans[i][2]), bounds[i], bounds[i + 1])
        out.append([bounds[i], bounds[i + 1], seg_lines])
    return out


def _split_oversized_segments(plans, max_new, min_new=MIN_NEW_SECONDS):
    """兜底：滑动 + 避台词 + 合并之后仍 > max_new 的段，才在段内切一刀。

    正常剧本到不了这一步（镜头表会先被 ``_rebalance_shot_bounds`` 拉平）。留着
    是给「镜头表极端不均、滑动量又不够」的输入一个出口，而不是让它带着 20 秒的
    段去渲染。切点走 ``_off_dialogue`` 避开台词，切完两侧都 ≥ min_new 才真切。

    ★ 2026-09-27 **口径修正**：``max_new`` 是**新增内容**的上限
      （``validate_segment`` 也是拿 ``new_seconds`` 去比它），而本函数原来拿
      **span**（含段首 1.6s 回放）去比 —— 于是每段凭空少 1.6s 预算，
      11.6s 的段（新增 10.0s，完全合规）被误判超限并切开。
      实测本机实例：被误切的那一刀正好落在跨段的一句 7 秒台词上，
      造成「前段尾窗带台词（下一段回放成重复语音）+ 后段零台词（模型即兴发声）」
      —— 用户看到的"乱说话、没和剧本对应"。
      首段没有回放（replay = 0），其余段 replay = HANDOFF_SECONDS。
    """
    if not plans:
        return plans
    out = []
    for index, (a, b, lines) in enumerate(plans):
        span = b - a
        # ★ replay 窗口用 17k+5 对齐的 1.625s（grid_seconds），与 handoff_seconds
        #   同源，避免把合规的 11.6s 段（新增 10s）误判超限切开（2026-09-27 口径
        #   修正的延续：这里原本用名义 1.6s，凭空少 1 帧预算）。
        replay = 0.0 if index == 0 else grid_seconds(HANDOFF_SECONDS)
        new_span = span - replay
        if new_span <= max_new + 0.001:
            out.append((a, b, lines))
            continue
        n = int(math.ceil(new_span / float(max_new) - 1e-9))
        if n < 2 or new_span / n < min_new - 1e-3:
            out.append((a, b, lines))           # 切不动：留给 issue，不硬切
            continue
        step = span / n
        spans = _talk_spans(lines)
        bounds = [a]
        for i in range(1, n):
            c = _off_dialogue(a + i * step, spans, a + min_new, b - min_new)
            if c is None or c <= bounds[-1] + min_new - 1e-6:
                bounds = None                   # 挪不开台词 / 会切出碎段：不切
                break
            bounds.append(round(c, 3))
        if bounds is None:
            out.append((a, b, lines))
            continue
        bounds.append(b)
        buckets = _rebucket(lines, bounds)
        for i in range(n):
            if not buckets[i]:
                if out:                         # 空窗口并进前一段
                    pa, _pb, pl = out[-1]
                    out[-1] = (pa, bounds[i + 1], pl)
                continue
            out.append((bounds[i], bounds[i + 1], buckets[i]))
    return out


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
        # ★ 只有**条目式**的行（时间码打头，如配乐表的 "00:12.0-00:17.0 三弦
        #   接棒…"）才按窗口切。散文里**内嵌**时间码的（如声景"…摇镜展示段
        #   （00:06.0-00:07.2）衣料声…"）整段讲的是一条持续规则，按窗切会把它
        #   整条丢掉 —— 实测 S02/S03/S04 的 overall_soundscape 就是这么变空的。
        if not codes or not _ENTRY_CLOCK_RE.match(line):
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


# 「取景纪律 / 视听风格总纲 / 一致性纪律 / 节奏对位」这类成套写的画面约束。
# 它们原本只在 summary 里，而模型逐字读的是 detailed_description —— 段里没有
# 就等于没传达。按**整段**命中收（按句切会把一套纪律拆散）。
_STYLE_HINT_RE = re.compile(
    r"取景纪律|视听风格|一致性纪律|风格总纲|禁止怼脸|180\s*度轴线|节奏对位"
    r"|光色沿用|质感|景深|framing discipline|no on-screen text")


def _music_style_from_text(text):
    """从全片配乐概述里取风格词。

    中文脚本：``全片配乐 = 京剧打击乐（板鼓…）`` → ``京剧打击乐``；
    英文脚本：``Music bed: Peking-opera percussion`` → ``Peking-opera percussion``。

    取不到就返回 ``None``，交给 ``build_music_block`` 走 ``detect_music_style``。
    只取第一个短语（到括号 / 加号 / 顿号 / 逗号为止）—— 整段塞进去会让 MUSIC
    block 变成一句读不通的长串。
    """
    raw = str(text or "")
    for pattern in (
            r"全片配乐\s*[=＝:：]\s*([^（(＋+，,。；;\n]+)",
            r"(?:music\s+bed|background\s+music|music|score|bgm)\s*[:=]\s*"
            r"([^(\n,;.]+)"):
        match = re.search(pattern, raw, re.I)
        if match:
            value = match.group(1).strip().strip("。.")
            if value:
                return value
    return None


# 台词起点写在句中而不是行首（整段式剧本的常见写法）：
#   "…<d>原文</d> … 自 00:01.2 起说：…"
# 规范（skill §1.4 / 交付样本）要求时间码在**行首**：``At 00:01.200，(S1) …``
_DLG_START_RE = re.compile(
    r"自\s*(?P<t>\d{1,2}:[0-5]\d\.\d{1,3})\s*起说\s*[：:]?\s*")


def _hoist_dialogue_clock(line, zh=True):
    """把句中写的台词起点时间码提到行首（规范写法 ``At MM:SS.mmm，``）。

    整段式剧本常把起点塞在句子中间：中文 ``…自 00:01.2 起说：…``，
    英文 ``…</d> From 00:12.0: Her voice…``。不提出来的话，模型读到一个没有
    落点时间的台词行，只能自己决定这句在段内什么时候说 —— 于是出现
    「两句台词抢同一时刻」「尾窗还在说话」这类乱说话。

    ★ 2026-09-27：以前**只认中文** ``自 … 起说``（``_DLG_START_RE``）。而本包
      的交付形态按 SKILL §1.3/§5 就是**英文单版** —— 英文的 ``From 00:12.0:``
      一个都提不出来。实测 S02 的台词行时间码留在句子中间，行首没有落点，
      而同一行还写着「at the timecode stated for it」（自相矛盾）。
      现在中文锚（``自…起说``）优先，落空就退回通用起止锚
      （``_TALK_START_RE``：from / starting at / beginning at / …）。

    只搬**起点**那一个：句尾的「于 00:08.7 前说完」是结束约束，留着有用，
    （交付样本里也是这么写的），不删。
    """
    if "At " in line[:8]:
        return line
    match = _DLG_START_RE.search(line)
    if match:
        start, end, clock_txt = match.start(), match.end(), match.group("t")
    else:
        m2 = _TALK_START_RE.search(line)
        if not m2:
            return line
        start, end, clock_txt = m2.start(), m2.end(), "%s:%s.%s" % m2.groups()
    try:
        mm, ss, ms = re.match(r"(\d{1,2}):([0-5]\d)\.(\d{1,3})", clock_txt).groups()
        clock = _fmt_clock(int(mm) * 60 + int(ss) + int(ms.ljust(3, "0")) / 1000.0)
    except (AttributeError, ValueError):
        return line
    body = (line[:start] + line[end:]).strip()
    return "%s %s%s%s" % ("At", clock, "，" if zh else ", ", body)


# 行首已经是规范写法：``[Shot N] At 00:04.000，…``
_ENTRY_AT_RE = re.compile(r"^\s*(?:\[Shot\s*\d+\]\s*)?At\s+(\d{1,2}):([0-5]\d)\.(\d{1,3})")
# 行首裸时间码（没有 At）：``00:07.200 切 <Subject 2>…``
_ENTRY_BARE_RE = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\.(\d{1,3})\s*(?:[-\u2013~]\s*"
                            r"(?:\d{1,2}):([0-5]\d)\.(\d{1,3})\s*)?[,，、]?\s*")
# 句中括号时间码：``台词中段（约00:03.500），…`` / ``台词落定（00:06.000），…``
# / ``划算静默里（00:06.600 前后），…`` —— 闭括号前允许带"前后 / 左右"这类
# 修饰词，否则最后那种写法会漏掉。
_ENTRY_PAREN_RE = re.compile(
    r"[（(]\s*(?:约|大约|at\s*)?((?:\d{1,2}):(?:[0-5]\d)\.(?:\d{1,3}))"
    r"[^）)]*[)）]\s*[,，]?\s*")
# 段首回放句（skill §1.4 的固定 1.6 秒回放）。源剧本常写成
# ``[Shot 1] [单镜，时长 1.8 秒] 本段以精确回放上一段结尾 1.6 秒开场…`` ——
# 有 Shot 标记、有内容，唯独没有时间码。回放区恒为段首，补 00:00.000。
_REPLAY_LINE_RE = re.compile(r"精确回放|replay(?:s|ed|ing)?\b|原样回放", re.I)
# 一行里塞了两个镜头：``…定格半拍。00:11.600 四击头余音里反切 <Subject 2>…``
# 前一句讲上一个镜头，时间码之后的才是新镜头。不断行的话整行的时间码就是
# 错的（前半段被当成 11.600 才发生）。
_MID_CLOCK_RE = re.compile(
    r"(?P<tail>[。！？；])\s*(?P<t>\d{1,2}:[0-5]\d\.\d{1,3})\s+(?=\S)")
# **结束**时间码不是起点，绝不能提到行首（提到行首等于说这个镜头从片尾开始）：
# ``镜头时长：00:08.600`` / ``持续至 00:12.600`` / ``此镜持续到 00:11.000``
_END_MARK_RE = re.compile(
    r"(?:镜头时长|镜头持续|持续至|持续到|此镜持续|时长[:：]?|至|到|ends?\s+at"
    r"|until|through)\s*[:：]?\s*\d{1,2}:[0-5]\d\.\d{1,3}")


def _normalize_entry_clocks(body, lang=None):
    """把段正文的每一条统一成规范写法 ``At MM:SS.mmm，<内容>``。

    skill §1.4 与交付样本：段内每一条（镜头起幅 / 切镜 / 动作 / 台词）都以
    ``At MM:SS.mmm`` 开头，中文版跟中文逗号。源剧本常见三种不规范写法：

      ① 裸时间码      ``00:07.200 切 <Subject 2>…``
      ② 句中括号时间码 ``台词落定（00:06.000），进入摇镜…``
      ③ 英文逗号      ``[Shot 1] At 00:00.000, 破门…``（中文版应写「，」）

    三者共同的后果是**模型读不到这条发生在第几秒**，只能自己猜 —— 时间码一
    乱，"尾窗有没有台词""接缝会不会闪""切镜和台词重不重叠"全都判不了。

    ★ ④ 完全没有时间码的镜头条：不瞎编秒数，沿用**紧随其后**的那个
      ``[Shot N] At X``（源剧本常把镜头内容先写一段、再用 [Shot N] 标题起
      正式的条目）。后面也没有 [Shot N] 的话原样保留，交给 issue 去报。

    ★ 段首的风格总纲（no-text 句 / 取景纪律 / 视听风格）本来就没有时间码，
      不是"条目"，原样放过。
    """
    zh = str(lang or "").lower().startswith("zh") or "中文" in str(lang or "")
    comma = "，" if zh else ", "
    lines = str(body or "").split("\n")
    out = []

    def _at(clock):
        return "At %s%s" % (clock, comma)

    # 先做一次**断行**：一行里出现句中时间码（且不是"镜头时长/持续至"这类
    # 结束标记）说明作者把两个镜头写进了同一行，前半句属于上一个镜头。
    expanded = []
    for line in lines:
        hit = _MID_CLOCK_RE.search(line)
        if hit and _END_MARK_RE.search(line[hit.start():hit.end() + 20] or ""):
            hit = None
        if hit:
            head = line[:hit.start() + len(hit.group("tail"))].strip()
            if head:
                expanded.append(head)
            expanded.append(_at(hit.group("t")) + line[hit.end():].strip())
        else:
            expanded.append(line)
    lines = expanded

    for idx, line in enumerate(lines):
        if not line.strip():
            out.append(line)
            continue
        head = _ENTRY_AT_RE.match(line)
        if head:
            # ③ 已经是规范写法，只统一逗号（中文版里混进英文逗号很常见）
            out.append(re.sub(r"^(\s*(?:\[Shot\s*\d+\]\s*)?At\s+"
                              r"\d{1,2}:[0-5]\d\.\d{1,3})\s*,\s*",
                              r"\1" + comma, line, count=1))
            continue
        bare = _ENTRY_BARE_RE.match(line)
        if bare:
            # 行首可能是区间（``00:01.000-00:01.200 凝住的半拍里…``）。提到
            # 行首的只能是**起点**，区间本身留在句里（它描述的是这个动作的
            # 持续范围，删掉反而丢信息）。
            clock = "%02d:%02d.%03d" % (int(bare.group(1)), int(bare.group(2)),
                                        int(bare.group(3).ljust(3, "0")))
            out.append(_at(clock) + line[bare.end():])
            continue
        paren = _ENTRY_PAREN_RE.search(line)
        if paren:
            rest = (line[:paren.start()] + line[paren.end():]).strip(" ，,、")
            out.append(_at(paren.group(1)) + rest)
            continue
        # ④ 段首回放句：回放区恒为段首，补 ``At 00:00.000，``
        mshot = re.match(r"^\s*(\[Shot\s*\d+\]\s*)", line)
        if mshot and _REPLAY_LINE_RE.search(line):
            rest = line[mshot.end():]
            # 形如 ``[单镜，时长 1.8 秒]`` 的镜头标注保留在 At 之后
            out.append(mshot.group(1) + _at("00:00.000") + rest)
            continue
        if _CUT_VERB_RE.search(line) and not _STYLE_HINT_RE.search(line):
            # ⑤ 无时间码的镜头条：借用紧随其后的 [Shot N] 的时间码
            nxt = None
            for j in range(idx + 1, min(idx + 4, len(lines))):
                m = _ENTRY_AT_RE.match(lines[j]) or _ENTRY_BARE_RE.match(lines[j])
                if m:
                    nxt = "%02d:%02d.%03d" % (int(m.group(1)), int(m.group(2)),
                                              int(m.group(3).ljust(3, "0")))
                    break
            if nxt:
                out.append(_at(nxt) + line.strip())
                continue
        # ⑥ 剩下的无时间码行（承接上文的动作描写）原样放过 —— 没有可靠依据
        #    去猜它发生在第几秒，硬填一个就是编造。
        out.append(line)
    return "\n".join(out)


def _inject_after_blocks(line, rule, front=False):
    """**每个** ``<d>…</d>`` 块所在行的对应位置挂 ``rule``（幂等：挂过的不再挂）。

    ``front=False``（默认）挂在**本块之后那截文本的末尾**（下一个台词块之前 /
    行尾）；``front=True`` 挂在**紧贴 ``</d>`` 的位置**（那截文本的最前面）。

    ★ 2026-09-30 排查修正：三个逐句注入器（lipsync / dialogue_once /
      dialogue_no_text）原来都是 ``re.sub(..., count=1)`` —— 同一行里有
      **多个台词块**时只有第一块拿到纪律句，后面的块裸奔（健壮性压测
      R7 实证）。这里按块切分、逐块检查「本块到下一个台词块之间」是否已有
      该纪律句，没有才补。

    ★ 幂等判据必须是**区间包含**而不是 ``startswith``：三条纪律句按
      lipsync → once → no_text 的顺序链式挂进同一段后文，第二、三条检查时
      块后紧跟的是先挂的那条 —— startswith 永远失配 → 重复解析时纪律句
      无限叠加（R7f 实证 2→6）。``front=True`` 时纪律句落在 ``</d>`` 与后文
      之间，仍属于 ``parts[i+1]``，所以同一条判据照样成立。

    ★★ 2026-09-30 第二轮：插入点从「紧贴 ``</d>``」改成「本块之后那截文本的
      **末尾**」（下一个台词块之前 / 行尾）。原因：台词自己的时间码窗
      ``from 00:03.200 to 00:05.200, word by word…`` 就写在那里，把三条纪律句
      （合计约 1500 字符）插在中间会把它推离台词块 **1300+ 字符** —— 而
      v23full 实测的失败恰恰是**模型不遵守台词时间码**。

    ★★★ 2026-09-30 第三轮：上面那条改动**把防字幕禁令一起推走了**，造成回归。
      v24fix 实测（对比 v23full 同一份 PACK）：
        v23full  禁令紧贴 ``</d>``、时间码在行尾  → seg1 干净、seg2 干净
        v24fix   时间码紧贴 ``</d>``、禁令在行尾  → **seg1 连烧 2 次、seg2 连烧 3 次**
      两条纪律句对位置的要求是**相反**的：
        · 防字幕（``DIALOGUE_NO_TEXT_EN``）—— 09-30 实测「只有紧邻台词块的
          禁令才压得住」（段首/段尾那份都被无视）；
        · 时间码窗 —— 必须留在台词块旁边，模型才把它当成这句话的时长。
      所以 ``_inject_dialogue_no_text`` 用 ``front=True``（紧贴 ``</d>``），
      lipsync / once 仍挂末尾。结果：
        ``<d>…</d>`` + 防字幕禁令 + 时间码窗 + 口型 + 只说一次
      —— 禁令在 0 字符处，时间码在约 330 字符处（比原来的 995/1500 近得多）。
    """
    parts = re.split(r"(<d>.*?</d>)", str(line))
    out = []
    i = 0
    n = len(parts)
    while i < n:
        part = parts[i]
        if not (part.startswith("<d>") and part.endswith("</d>")):
            out.append(part)
            i += 1
            continue
        out.append(part)
        tail = parts[i + 1] if i + 1 < n else ""
        if rule in tail.split("<d>")[0]:
            out.append(tail)                      # 已挂过，原样放回
        elif front:
            out.append(" " + rule + tail)
        elif tail.strip():
            out.append(tail.rstrip() + " " + rule)
        else:
            out.append(" " + rule)
        i += 2
    return "".join(out)


def _inject_lipsync(detail, hoist=True):
    """每句台词块后面紧跟规范的口型同步约束句（SKILL §3「对镜台词」）。

    样本里每条台词都是 ``(S1) <d>[Chinese] …</d> 口型与台词精确同步：…。 <声音
    描述>`` 这个顺序。缺了中间那句，模型会让人物边说边做大动作、口型糊成一片，
    或者台词还没说完就切走 —— 都是"乱说话"的观感来源。

    幂等：同一行已经挂过就不再挂（重复解析不会叠两遍）。

    ★ 输出固定英文（英文单版是唯一交付语言）。源文本若是中文，逗号按**源文本
      实际语言**走 —— 把中文句子拼上英文逗号会让两边都不合规。

    ★ ``hoist``（2026-09-30）：行首时间码 hoist（``_hoist_dialogue_clock``）是
      整段式剧本的抢救手段 —— 那种稿子的台词起点常埋在句中、行首没有落点。
      en-pack 分段路径的交付稿**行首本来就有 At 时间码**，再 hoist 会把
      "from X" 从台词窗里拆走、行结构被打乱（真实稿实测：4 段全报"时间码
      不是严格递增"）。因此 en-pack 调用时传 ``hoist=False``。
    """
    zh = _infer_lang(detail) == "zh"
    rule = LIPSYNC_EN
    out = []
    for line in str(detail or "").splitlines():
        if "<d>" in line:
            if hoist:
                line = _hoist_dialogue_clock(line, zh)
            # 挂在同一行、台词块紧后面（样本的顺序：台词 → 口型约束 → 声音描述）
            line = _inject_after_blocks(line, rule)
        out.append(line)
    return "\n".join(out)


def _inject_dialogue_once(detail):
    """每句台词块后面紧跟「只说一次、严格按时间码」的交付纪律。

    与 ``_inject_lipsync`` 同一范式（挂在同一行、台词块紧后面、幂等不叠加）。

    为什么要写成节点层的通用规则（2026-09-26 实测）：一段 prompt 里有多句台词
    时，模型会自行重排它们的时间码 —— 成片表现为「第一句说了一半、第二句提前
    插进来、第一句后半段又被补说一遍」，听感就是"重复说话/和剧本不一致"。
    这是模型侧行为，但**每一句**都钉一条交付纪律能有效压住，且与具体是哪一本
    剧本无关：任何剧本的每一句台词都会被自动挂上。

    ★ 幂等：已经挂过就不再挂；纪律句本身不含时间码（否则会污染
      ``_talk_spans``，见 ``DIALOGUE_ONCE_EN`` 处的说明）。
    """
    rule = DIALOGUE_ONCE_EN
    out = []
    for line in str(detail or "").splitlines():
        if "<d>" in line:
            line = _inject_after_blocks(line, rule)
        out.append(line)
    return "\n".join(out)


def _inject_dialogue_no_text(detail):
    """每句台词块后面紧跟「只出声、严禁上屏」的防字幕禁令。

    2026-09-30 渲染实证：no-text 首句（段首）+ SUBTITLE_BAN_EN（段尾）都被模型
    无视，《曹贼的性价比》S02/S04 的中文台词照样逐行烧进画面，而同结构的
    S01/S03/S05 干净 —— 概率性行为，只有**紧邻台词块**的禁令才压得住
    （_inject_lipsync / _inject_dialogue_once 已验证这个位置有效）。

    节点层通用规则：任何剧本的每句台词自动获得，与语言、内容、剧本无关。
    幂等：同一块已挂过就不再挂（见 ``_inject_after_blocks``）；纪律句不含
    时间码（不污染 ``_talk_spans``），不含双引号（过 ``_QUOTE_RE``），
    不含字面 ``<d>``（不干扰逐行块扫描）。

    ★★ 2026-09-30 第三轮：**必须挂在紧贴 ``</d>`` 的位置**（``front=True``）。
      09-30 的实测结论就是「只有紧邻台词块的禁令才压得住 —— 段首/段尾那份都被
      模型无视」。第二轮把三条纪律句统一挪到行尾时，顺带把它也推到了
      **约 1500 字符之外**，代价立刻出现：同一份 PACK，v23full（禁令紧贴）
      seg1/seg2 都不烧，v24fix（禁令在行尾）**seg1 连烧 2 次、seg2 连烧 3 次**，
      烧出来的正是该段台词原文（``汝与曹贼何异``）。
      口型句 / 只说一次那两条与时间码不冲突，仍留在行尾。
    """
    rule = DIALOGUE_NO_TEXT_EN
    out = []
    for line in str(detail or "").splitlines():
        if "<d>" in line:
            line = _inject_after_blocks(line, rule, front=True)
        out.append(line)
    return "\n".join(out)


def _normalize_dialogue_spans(detail):
    """台词窗时间码补毫秒：``from 00:01 to 00:03`` → ``from 00:01.000 to 00:03.000``。

    镜头行首时间码由 ``_normalize_entry_clocks`` 管；这里只补**台词起止窗**
    （from / to / starting at 后面裸 MM:SS 的形态）。不补的话最终提示词里
    同一行混着 ``At 00:01.000`` 与 ``from 00:01`` 两种口径，模型对
    "at the timecode stated for it"（DELIVERY 句）没有一致的时间粒度可遵守。

    ★ 只动含 ``<d>`` 的行、只补缺失的毫秒位，已有 ``.5``/``.000`` 一律不动；
    幂等（再跑一遍无变化）。
    """
    pat = re.compile(
        r"\b([Ff]rom|[Tt]o|[Ss]tarting at|[Bb]eginning at)\s+"
        r"(\d{1,2}:[0-5]\d)(?!\.\d)")
    out = []
    for line in str(detail or "").splitlines():
        if "<d>" in line and pat.search(line):
            line = pat.sub(lambda m: "%s %s.000" % (m.group(1), m.group(2)),
                           line)
        out.append(line)
    return "\n".join(out)


def _style_preamble(globals_fields):
    """每段 detailed_description 开头、全链**逐字一致**的画面纪律总纲。

    skill §4 要求 STYLE 块全链逐字复用。以前段正文里只有一句「不出现文字」，
    取景纪律、180 度轴线、光色沿用、节奏对位全留在 summary 里没进段 —— 模型
    收不到，等于没约束。这里把它们原句搬到每段开头，逐字相同，每段也就能独立
    成立（skill §0「每段是独立的一段 5–11 秒成片」）。
    """
    summary = str(globals_fields.get("summary") or "")
    picked = []
    for para in summary.split("\n"):
        para = para.strip()
        if para and _STYLE_HINT_RE.search(para):
            picked.append(para)
    return "\n".join(picked).strip()


def _renumber_shots(lines, start=1):
    """把段内的 ``[Shot N]`` 按出现顺序重编为 ``start..start+M-1``。

    段是整片的一小截：原剧本第 3 个镜头进了本段，就是本段的第 1 个镜头。留着
    原编号（一段 8 秒的段里冒出 ``[Shot 7]``）会让模型以为前面还有 6 个镜头，
    也会让 retention_analysis 的 ``appears in [Shot N]`` 对不上号。

    接续段传 ``start=2``：段首那 1.6 秒回放镜自己占了 [Shot 1]。
    """
    counter = [start - 1]

    def _sub(_m):
        counter[0] += 1
        # 保留 markdown 标题 / 列表前缀，否则 ``### [Shot 1]`` 会被压成 ``[Shot 1]``
        return "%s[Shot %d]" % (_m.group("pre"), counter[0])

    return [_SHOT_TAG_RE.sub(_sub, line) for line in lines]


def _seg_retention(globals_fields, detail, lang=None):
    """只声明**本段出现**的 Subject / Picture，并标出它们在本段的哪些镜头里。

    全片 retention 照抄进每段的话，一段 8 秒的段里会读到「Shot 1–Shot 4 全部
    角色全部保留」，而其中几个本段根本没出现 —— 声明一个不存在的保留对象，等于
    给模型凭空造人的许可（「多出一个人的声音」就是这么来的）。

    描述文案**优先沿用作者写的**（按 Subject/Picture 号从全局 retention 里
    取），取不到才用通用文案 —— 节点不替作者创作角色描述。

    标记词固定用官方英文 token（SKILL §3：``fully_preserved`` 等两版都写英文
    原样），输出语言固定英文。
    """
    # 兜底文案（作者原文优先，取不到才用这些）
    generic_subj = ("appearance, wardrobe and build stay identical to "
                    "<Picture %s>, unchanged.")
    generic_pic = "retained as the appearance reference for <Subject %s>."
    generic_pic_plain = "retained as the appearance reference, frame by frame."
    generic_scene = ("the scene, colour temperature and light direction are "
                     "fully retained, no jumps and no new props.")

    # 段内镜头 → 该镜头里出现了谁
    shots = []
    n = 0
    cur = None
    for line in str(detail or "").splitlines():
        if _SHOT_HEAD_RE.match(line):
            n += 1
            cur = [n, []]
            shots.append(cur)
        if cur is not None:
            cur[1].append(line)
    if not shots:
        shots = [[1, str(detail or "").splitlines()]]

    seen = {}
    order = []
    for no, lines in shots:
        # 台词块里的 (S1) 只说明"这句是谁说的"，可能是画外音 —— 不作为"人在
        # 画面里"的依据，先剔掉再看。
        blob = _DIALOG_RE.sub("", "\n".join(lines))
        for tok in re.findall(r"<\s*(?:Subject|Picture)\s*\d+\s*>", blob, re.I):
            key = re.sub(r"\s+", "", tok).lower()
            if key not in seen:
                seen[key] = []
                order.append(tok)
            if no not in seen[key]:
                seen[key].append(no)
        # ★ 剧本常常**混用**两种写法：subject_definitions 里写 <Subject 1>，
        #   正文里写 "S1"（本份剧本的 Shot 2 通篇是 "S1 退到画框边缘虚焦"）。
        #   只认尖括号的话，出场的人会被判成"本段不出现"—— 那等于告诉模型
        #   这段里没有他，正是角色凭空消失的入口。裸写 S1 一并认，按 <Subject N>
        #   归一（台词已剔除，(S1) 不会误算成在场）。
        for num in re.findall(r"(?<![\w<])S([1-9])(?![\w>])", blob):
            tok = "<Subject %s>" % num
            key = re.sub(r"\s+", "", tok).lower()
            if key not in seen:
                seen[key] = []
                order.append(tok)
            if no not in seen[key]:
                seen[key].append(no)
    if not order:
        return (globals_fields.get("retention_analysis") or "").strip()

    # 作者写的描述按 token 取用
    authored = {}
    for line in str(globals_fields.get("retention_analysis") or "").splitlines():
        m = re.match(r"^\s*(<[^>]+>)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if m:
            authored[re.sub(r"\s+", "", m.group(1)).lower()] = m.group(2).strip()

    def _rank(tok):
        # Subject 在前、Picture 在后，各自按编号升序 —— 与官方样例 PACK 一致，
        # 也让「谁锁哪张图」一眼可读。
        return (0 if re.search(r"Subject", tok, re.I) else 1,
                int((re.search(r"(\d+)", tok) or [None, "0"])[1]))

    # Picture → (它锁的 Subject 号, 是不是场景图)。标注写法取决于这个：
    # 交付样本里人物参考图写 ``<Picture 1>:``（**不带** appears in），场景图写
    # ``<Picture 3> ([Shot 1] first frame):`` —— 场景是"首帧定住"，不是"出现"。
    pic_meta = {}
    for line in str(globals_fields.get("subject_definitions") or "").splitlines():
        m = re.search(r"<\s*Picture\s*(\d+)\s*>", line, re.I)
        if not m:
            continue
        # 兼容两种写法：中文规范句的 ``<Subject 1>`` 与英文原句的
        # ``Subject 1``（不带尖括号）。只认尖括号会让英文定义里的关联全部
        # 落空，兜底文案就退化成没有主语的"作为外观参考被完整保留"。
        sm = re.search(r"<\s*Subject\s*(\d+)\s*>|Subject\s*(\d+)", line, re.I)
        pic_meta[m.group(1)] = (
            (sm.group(1) or sm.group(2)) if sm else None,
            bool(re.search(r"场景|背景|scene|background", line, re.I)))

    # ★ Subject → 它锁定的 Picture 号（从 subject_definitions 读）。
    #   兜底文案 ``generic_subj`` 里的 ``%s`` 是 **Picture 号**，不是 Subject 号。
    #   本剧本 ``<Subject 3>`` 绑的是 ``<Picture 4>``，直接填 Subject 号会写成
    #   ``<Picture 3>``（= 场景图）—— 等于告诉模型「路人甲的脸和布景一致」，
    #   成片里人换个样。只有读不到绑定关系时才退回同号（旧行为）。
    subj_pic = {}
    for line in str(globals_fields.get("subject_definitions") or "").splitlines():
        sm = re.search(r"<\s*Subject\s*(\d+)\s*>|Subject\s*(\d+)", line, re.I)
        pm = re.search(r"<\s*Picture\s*(\d+)\s*>|Picture\s*(\d+)", line, re.I)
        if sm and pm:
            subj_pic.setdefault(sm.group(1) or sm.group(2),
                                pm.group(1) or pm.group(2))

    items = []
    for tok in sorted(order, key=_rank):
        key = re.sub(r"\s+", "", tok).lower()
        num = (re.search(r"(\d+)", tok) or [None, "1"])[1]
        hits = seen.get(key) or []
        if re.search(r"Subject", tok, re.I):
            label = (" (appears in %s)" % ", ".join("[Shot %d]" % x for x in hits)
                     if hits else "")
            fallback = generic_subj % subj_pic.get(num, num)
        else:
            sub, is_scene = pic_meta.get(num, (None, False))
            # ★★ 2026-09-27 修：**优先沿用作者原文，不再替换**。
            #
            #   以前人物参考图一律套 ``generic_pic``：
            #       "retained as the appearance reference for <Subject N>."
            #   而作者写的是：
            #       "The appearance of Subject N is consistent with the
            #        reference image in all shots."
            #   前者只说「这张图是参考图」，后者才是「**生成结果要与参考图一致**」。
            #   断言被节点悄悄弱化掉，模型于是跨段换衣服 ——
            #   实测 iter6 成片：张三 S01/S02 香槟金 → S03 粉 → S04 浅绿，
            #   而参考图 ``<Picture 2>`` 是深红纱袍。
            #
            #   场景图那条同样是替换：作者写「场景在全程与参考图完全一致，
            #   无跳变、无新增陈设」，被换成 generic_scene，也丢了「一致」二字。
            #
            #   用户裁决：「源剧本 人物服装场景是和参考图一致 禁止私自更换描述」。
            #   → 作者写了就用作者的原话（含 fully_preserved 标记时逐字照搬）；
            #     只有作者**没写**这一条时才退回规范兜底句。
            desc = authored.get(key) or ""
            if not desc:
                desc = (generic_scene if is_scene
                        else ((generic_pic % sub) if sub else generic_pic_plain))
            label = (" ([Shot %d] first frame)" % (hits[0] if hits else 1)
                     if is_scene else "")
            items.append((tok, label,
                          desc if "fully_preserved" in desc.lower()
                          else "fully_preserved - " + desc))
            continue
        desc = authored.get(key) or ""
        if "fully_preserved" not in desc.lower():
            desc = "fully_preserved - " + (desc or fallback)
        items.append((tok, label, desc))

    # 全局定义过、但本段画面里没出现的（参考图锁定是**全片**约束，每段都要
    # 声明；可必须写明"本段不出现"，否则等于告诉模型这段里有这个人 ——
    # 「多出一个人的声音」就是这么来的）。
    present = {re.sub(r"\s+", "", t).lower() for t in order}
    tail = "not in frame in this segment; the lock holds film-wide."
    defined = {}
    for tok in re.findall(r"<\s*(?:Subject|Picture)\s*\d+\s*>",
                          str(globals_fields.get("subject_definitions") or ""),
                          re.I):
        defined.setdefault(re.sub(r"\s+", "", tok).lower(), tok)
    for key, tok in sorted(defined.items(), key=lambda kv: _rank(kv[1])):
        if key in present:
            continue
        num = (re.search(r"(\d+)", tok) or [None, "1"])[1]
        if re.search(r"Subject", tok, re.I):
            desc = authored.get(key) or (generic_subj % num)
            if "fully_preserved" not in desc.lower():
                desc = "fully_preserved - " + desc
            # 人没出现要明确说（否则等于许可模型凭空造人）。参考图不在此列 ——
            # 图是锁定基准，不是画面里的实体，不存在"出不出场"。
            items.append((tok, "", desc + " " + tail))
        else:
            sub, is_scene = pic_meta.get(num, (None, False))
            # ★★ 2026-09-27 修（**这一处才是真凶**）：本段正文里没出现
            #   ``<Picture N>`` 字样时（正文写的是 ``<Subject N>``），这一支
            #   会走到这里 —— 而它以前**完全无视作者原文**，一律套 generic 句。
            #   实测本剧本：``<Picture 1/2/4>`` 全部走这一支，作者写的
            #   "The appearance of Subject N is consistent with the reference
            #    image in all shots." 被换成
            #   "retained as the appearance reference for <Subject N>."
            #   —— 「生成结果要与参考图一致」这条**断言被抹掉**，模型于是跨段
            #   换衣服（iter6 成片：金 → 粉 → 绿，参考图却是深红纱袍）。
            #   用户裁决：「源剧本 人物服装场景是和参考图一致 禁止私自更换描述」。
            desc = authored.get(key) or ""
            if not desc:
                desc = (generic_scene if is_scene
                        else ((generic_pic % sub) if sub else generic_pic_plain))
            items.append((tok, "", desc if "fully_preserved" in desc.lower()
                          else "fully_preserved - " + desc))
    # 统一按 Subject → Picture、各自编号升序排（交付样本的顺序），
    # 不按"本段是否出现"分组 —— 分组会让 Picture 的顺序变成 3,1,2,4。
    items.sort(key=lambda it: _rank(it[0]))
    return "\n".join("%s%s: %s" % it for it in items)


def _shot_seg_body(globals_fields, detail, a, b, idx, total, music="", hard=False,
                   soundscape=None, lang=None, replay_in=0.0, gen_seconds=0.0):
    """拼回官方六段正文 —— **除 detailed_description 外的五段也要按段生成**。

    ★ 2026-09-20 重写。以前除 detailed_description 之外的五段是**整片原文照
      抄**：7 段的 summary / retention_analysis / overall_soundscape 逐字相同，
      每段都带着"全片 40 秒"的剧情与声景。模型拿到"这 8 秒里要演完 40 秒"，
      正是「乱说话 / 重复说话」的入口。现在按段生成：

      * ``summary``            首段 ``[reference generation]``；接续段
                               ``[video continuation + reference generation]``
                               + 从成片哪一段接续 + 本段新增/回放/生成时长
      * ``retention_analysis``  只声明本段出现的对象（见 ``_seg_retention``）
      * ``overall_soundscape``  按本段时间窗切片（``_split_music_lines`` 同逻辑）
      * ``non_diegetic_music``  配乐表按段切 + 全链同一套 MUSIC 硬约束（§4）
      * ``subject_definitions`` 全链一致（本来就该一致）

    ★ 输出固定英文（SKILL §1.3 / §5.1：唯一交付语言）。
    """
    new_seconds = round(b - a, 3)
    parts = []
    for name in FIELD_ORDER:
        if name == "detailed_description":
            value = detail
        elif name == "subject_definitions":
            value = (globals_fields.get(name) or "").strip()
        elif name == "non_diegetic_music":
            # 这一字段**全链逐字一致**，就是 §1.5 的 MUSIC block，不按段切。
            # 配乐的艺术意图（哪一段起什么乐器）属于 detailed_description 与
            # overall_soundscape 要写的 —— 塞进这个字段等于让模型在这 8 秒里
            # 演完 40 秒的配乐表，而且还会违反 §4「MUSIC block 全链逐字复用」。
            value = music or (globals_fields.get(name) or "").strip()
        elif name == "overall_soundscape":
            value = (soundscape if soundscape is not None
                     else (globals_fields.get(name) or "").strip())
        elif name == "retention_analysis":
            value = _seg_retention(globals_fields, detail)
        elif name == "summary":
            base = (globals_fields.get("summary") or "").strip()
            # ★ 剧本常常自己写了任务标签（有的还写成孤零零一行）。这里先剥掉
            #   纯标签行，再看剩下的内容**是不是已经带标签** —— 两种情况都不
            #   能再套一层，否则交付文本里会出现两行 ``[reference generation]``
            #   （实测就出过：首行光秃秃一个标签，第二行才是真正的摘要）。
            _raw = [x for x in base.split("\n")
                    if x.strip() and not _TASK_TAG_ONLY_RE.match(x.strip())]
            base = "\n".join(_raw).strip()
            # 剧情总纲留着：接续段要知道自己处在整个故事的哪一段。但把它和
            # "本段"严格分开写，别让模型以为本段要演完整个故事。
            if base.startswith("["):
                tag = ""                      # 原文已自带标签，不重复套
            else:
                tag = ("[reference generation]" if idx == 1
                       else "[video continuation + reference generation]")
            # 尾部句式（SKILL §3 summary）：
            #   首段    "This is the opening segment with no preceding take to
            #            replay; generated duration 11 seconds."
            #   接续段  "This segment continues from the end of the preceding
            #            take. New duration 11 seconds, replaying 1.625 seconds
            #            at the top, generated duration 12.625 seconds."
            # 中间那段剧情摘要是作者写的，节点不代创作 —— 见模块顶部说明。
            #
            # ★★ 2026-09-27：回放时长必须写 ``replay_in``（已过 ``grid_seconds``
            #   = 1.625），**不能写死名义值 1.6**。1.6s × 24fps = 38 帧不在
            #   17k+5 网格上；下游 ``director._shot_plan`` 按 ``guide_length_up``
            #   实际生成 **39 帧（1.625s）**。文案写 1.6、管线渲 1.625，
            #   模型与渲染器就差了 1 帧 —— 正是 ``grid_seconds`` docstring 里
            #   「接缝处动作原地打结 / 一段里多出 17 帧回放」的同一源头，
            #   只不过这次发生在**文本层**而不是字段层。
            if idx == 1:
                span = ("This is the opening segment with no preceding take "
                        "to replay; generated duration %g seconds."
                        % gen_seconds)
            else:
                span = ("This segment continues from the end of the "
                        "preceding take. New duration %g seconds, replaying "
                        "%g seconds at the top, generated duration %g "
                        "seconds." % (new_seconds, replay_in, gen_seconds))
            if hard:
                # 让 _build_segment 的 HARD CUT 判定能命中：硬切段不继承上一段
                # 的机位与姿态，也不给下一段留回放锚点（SKILL §4）。
                span = "HARD CUT " + span
            # 顺序：任务前缀 → 剧情总纲 → **时长句收尾**（模型读完摘要立刻看到
            # 本段的时间边界）。
            value = ((tag + "\n" if tag else "") + (base + "\n" if base else "")
                     + span).strip()
        else:
            value = (globals_fields.get(name) or "").strip()
        parts.append("%s:\n%s" % (name, value))
    return "\n\n".join(parts)


def _shot_script_plans(raw, max_new=MAX_NEW_SECONDS):
    """整段式 ``[Shot N]`` 脚本 → ``([(a, b, seg_lines), ...], globals_fields)``。

    从 ``parse_shot_script_blocks`` 抽出来单独一份，**回写定位也要用它**：
    段号必须与解析时算出的完全一致，否则时间线编辑会写到别的段上。
    拆段规则（顺序固定，都是通用规则，任何剧本自动生效）：

    ★★ 拆段只沿镜头边界，不切进镜头内部（SKILL §0.2）

    剧本里的 ``[Shot N]`` 就是 §0.2 说的**镜头表**：段边界落在镜头边界上，
    一段可以是一个镜头，也可以是相邻几个镜头（**段数 ≠ 镜头数**），短镜头并入
    邻段、超长镜头走兜底切开。

    旧写法是进到**每个 [Shot N] 内部**再按 max_new 切碎，于是 4 个镜头的剧本
    被切成 7 段还带 2s / 3s 碎段；为了把碎段拼回去又叠了「合并 → 均分 → 避让」
    三层补丁 + 四轮迭代 —— 全是在补救一个**根本不该发生的切碎动作**。已整体删除。

    现在四步：
      ① ``_rebalance_shot_bounds``      边界在镜头边界 ±3s 内滑动，把一秒从长
         的镜头让给短的邻居，使每段落回 5–11s，**段数不变**
         （实测 10/12/13/5 → 11/11/11/7）。
      ② ``_snap_cuts_off_dialogue``     边界不落在台词进行中，且离台词结束 ≥1.6s
         （尾窗无台词）。
      ③ ``_coalesce_short_segments``    滑动后仍 <5s 的向邻镜头并。
      ④ ``_split_oversized_segments``   兜底：前三步之后仍 >11s 才在段内单切一刀
         （正常剧本到不了）。
    """
    blocks = _shot_blocks(raw)
    if not blocks:
        return [], None
    globals_fields = _shot_global_fields(raw)
    starts, ends = _shot_bounds(blocks)
    plans = []
    for (_no, _rest, lines), t_start, t_end in zip(blocks, starts, ends):
        if t_start is None or t_end is None or t_end <= t_start + 0.01:
            continue
        plans.append([float(t_start), float(t_end), list(lines)])
    if not plans:
        return [], globals_fields
    # ① 滑动镜头边界，让每段新增落回 [min_new, max_new]（段数不变）
    plans = _rebalance_shot_bounds(plans, max_new)
    # ② 滑动后的边界可能压在台词上，挪开（§1.4 硬要求，这条一直是对的）
    plans = _snap_cuts_off_dialogue(plans, min_part=SNAP_MIN_PART)
    # ③ 仍有 < min_new 的（镜头本身就短）向邻镜头并 —— 只在这里合并，不再
    #    事后均分。合并后若又超长，交给下面的兜底单切一刀。
    plans = _coalesce_short_segments(plans, max_new)
    # ④ 兜底：滑动 + 避让 + 合并之后还是 > max_new（镜头表极端不均），才在
    #    段内按自然切点切一刀。正常剧本到不了这一步。
    plans = _split_oversized_segments(plans, max_new)
    # ⑤ ★ 2026-09-27 收尾闸门：再并一次过短段 + 再滑一次边界，直到**全部合规**。
    #    为什么必须有这一步：①–④ 各自只在**自己的假设**下成立，前一步的"修复"
    #    可能给后一步造出新违规 —— 实测（本机真渲染）：
    #      ① 已给出合规的 [0,11][11,22][22,38][38,45]；
    #      ② 为了避台词把边界推到 26，把 9.4s 撑成 13.4s、把末段挤成 4.3s；
    #      ③ 只处理 <min_new 的合并，超长的它不管；
    #      ④ 切开了超长的 [11,26]，但 [26,39.1] 因台词占 68% 跨度而切不动；
    #    最终 4.3s 的过短段**一路带进渲染**（H3 上退化成近乎静止镜头），
    #    13.1s 的超限段则带着"尾窗含台词 → 下一段回放成重复语音"进渲染。
    #    ⑤ 在出口处把「并短 → 滑边界 → 再避台词」跑到不动点，并对**无法修复**
    #    的残留违规给出明确 issue（而不是静默带过）。
    for _ in range(4):
        before = [(a, b) for a, b, _ in plans]
        plans = _coalesce_short_segments(plans, max_new)
        plans = _rebalance_shot_bounds(plans, max_new)
        after = [(a, b) for a, b, _ in plans]
        if after == before:
            break
    return plans, globals_fields


_MD_SHOT_PREFIX_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[-*>]\s*)?(?=\[\s*Shot\s*\d+\s*\])", re.I)


def _shot_line_key(line):
    """原文行 → 与 ``_shot_blocks`` 收集行对齐用的键。

    **只**对「镜头头行」剥掉 markdown 前缀 / 列表符号（``### [Shot 1] …`` →
    ``[Shot 1] …``）—— ``_shot_blocks`` 会把这一行重建一遍，原文里还带着前缀。
    正文行原样返回：不能顺手把以 ``-`` / ``*`` 开头的正文行也改了。
    """
    return _MD_SHOT_PREFIX_RE.sub("", str(line).strip())


def shot_script_segment_ranges(text, max_new=MAX_NEW_SECONDS):
    """整段式 ``[Shot N]`` 脚本 → 每段在**原文**里的行区间 ``[(start, end), ...]``。

    段号与 ``parse_shot_script_blocks`` 的 ``S01…`` 一一对应，``end`` 为开区间。
    给「分镜时间线」写回用：`##########` 分段 PACK 靠段标题定位，整段式脚本
    没有段标题，只能靠**行区间**定位。

    ★ 为什么能对回去：``_shot_blocks`` 收集的正文行是原文 ``detailed_description``
      区的一个**有序子序列**，而 ``_rebucket`` / 合并 / 兜底切分都只做"分区"，
      不重排、不丢行 —— 所以把每段的 ``seg_lines`` 展平后，用双指针按顺序扫一遍
      原文即可拿回行号（重复行也不会错位）。
      对不回去（原文被外部改过）就返回 ``[]``，宁可让前端退回"只读兜底"，
      也不能瞎写。
    """
    raw = str(text or "")
    plans, _ = _shot_script_plans(raw, max_new)
    if not plans:
        return []
    lines = raw.split("\n")
    collected = [ln for _a, _b, seg_lines in plans for ln in seg_lines]
    if not collected:
        return []
    idx_of = []
    j = 0
    for i, line in enumerate(lines):
        if j >= len(collected):
            break
        want = collected[j]
        # 正文行原样比；镜头头行要归一化 —— ``_shot_blocks`` 重建过它
        # （``### [Shot 1] …`` → ``[Shot 1] …``），原文里还带着 markdown 前缀。
        if line == want or _shot_line_key(line) == want:
            idx_of.append(i)
            j += 1
    if j != len(collected):
        return []
    ranges = []
    k = 0
    for _a, _b, seg_lines in plans:
        n = len(seg_lines)
        if n == 0:
            ranges.append(None)
            continue
        ranges.append((idx_of[k], idx_of[k + n - 1] + 1))
        k += n
    return ranges


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
    plans, globals_fields = _shot_script_plans(raw, max_new)
    if not plans:
        return [], []
    lang = fallback_lang or _infer_lang(raw)

    total = len(plans)
    out = []
    speaker_notes = []
    carry = []          # 上一段尾窗里截下的「下一镜开头」，原样交给本段
    for index, (a, b, seg_lines) in enumerate(plans):
        new = round(b - a, 3)
        is_first = index == 0
        is_last = index == total - 1
        # ★★ 2026-09-27：这里必须用 ``grid_seconds(HANDOFF_SECONDS)``（= 1.625，
        #   39 帧），**不能**用名义值 ``HANDOFF_SECONDS``（= 1.6，38 帧）。
        #   1.6s 不在 H3 的 17k+5 网格上；下游 ``_shot_plan`` 按 ``guide_length_up``
        #   实际按 39 帧渲。用 1.6 的话，模型可见文案（summary 的
        #   "replaying 1.6 seconds"、段标题 "11s+1.6=12.6s"）与真渲染的
        #   1.625 差 1 帧 —— 正是「接缝处动作原地打结 / 一段多出 17 帧回放」
        #   的文本层成因（见 HANDOFF_REPLAY_EN / grid_seconds 的注释）。
        #   PACK 路径（``_build_segment``）早已走 ``grid_seconds``，本函数是
        #   整段式（shot-script）路径，之前漏了。
        replay_in = 0.0 if is_first else grid_seconds(HANDOFF_SECONDS)
        tail_out = 0.0 if is_last else grid_seconds(HANDOFF_SECONDS)
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
        # 规范 0.1 段标题固定写法「新增+1.6=生成」：重叠固定 1.6 秒，写成
        # 1.600 既不合规也难读，用 %g 去掉多余的零。
        dur_spec = ("%gs" % new) if extra <= 0.001 else (
            "%gs+%g=%gs" % (new, extra, gen))
        # 段内镜头重新编号：段是整片的一小截，编号从 1 起（见 _renumber_shots）。
        # 接续段把 [Shot 1] 让给段首那 1.6 秒回放镜，正文镜头从 2 起。
        start_no = 2 if replay_in > 0 else 1
        # 滑动边界会把镜头标题行留在上一段（标题行的时间码 = 镜头起点，边界滑
        # 过它之后它就归了前一段）。本段剩下的内容于是**没有一个 [Shot] 标记**，
        # 取景纪律无处附着、retention 也没法标 appears in。补一个，写明是承接。
        work_lines = list(carry) + list(seg_lines)
        carry = []
        if work_lines and not any(_SHOT_HEAD_RE.match(x) for x in work_lines):
            marker = ("[Shot %d] (continuing the shot carried over from the "
                      "preceding take)")
            work_lines = [marker % start_no] + work_lines
        # ---- 尾窗里的「下一镜开头」交给下一段（2026-09-27 实测修复）----------
        # 边界滑动（`_rebalance_shot_bounds` 把 10.0 滑到 11.0）会把**下一镜**的
        # `[Shot N]` 头连同它的整段正文留在本段里，而镜头起点正好落在本段的
        # 最后 1.6 秒（尾窗）内。两层后果：
        #
        #   1. 规范 §1.4：尾窗会被下一段原样回放，里面有切点＝接缝处重演一次
        #      切镜（`validate_segment` 早就报这条 issue，但历来只报不改）。
        #   2. ★ 更隐蔽的一层，也是「前 6s 乱说话、没和剧本对应」的真凶：
        #      镜头头后面的正文常常**引用上一镜的台词**。本机实测段 1 的
        #      `[Shot 2]` 正文里就写着 `still frozen in shock from How are you
        #      different from Cao Zhirong?` —— 同一段提示词里于是出现了**第二处**
        #      这句台词的文本，而且压在段尾。模型被往「把这句提前说」的方向拉：
        #      两次独立渲染都是第二句 2~7s 就说了（剧本写 7.2~8.7s）、第一句被
        #      挤到 6~12s（剧本写 1.2~6.0s），段 1 里还出现 2 个镜头头。
        #
        # 处置与 issue 里的原建议一致：「把那个镜头挪到下一段开头当新内容」。
        # 下一段的 `_retime_line` 会把段外时间码 clamp 到段首（＝回放窗结束的
        # 那一刻），所以搬过去正好落在回放窗**之后**，规范 §1.4 两头都满足，
        # 而且一个字都不丢（下一段的桶里本来就没有这一行）。
        # 两条护栏：被搬的部分含台词块就放弃（绝不静默吃掉台词）；末段没有
        # 「下一段」可交，也留着不动。
        #
        # ★ 判据与校验器**共用同一套**（`_SHOT_HEAD_RE` / `_CUT_VERB_RE`，与
        #   `_cut_times_in` 逐字一致）：镜头标题本身就是切点，此外还有散文写法
        #   （``At 00:21.000 … cuts to``、``11.6 切 <Subject 1>``）。只用标题判据
        #   会漏掉后者 —— 实测段 2 的尾窗里就有一个散文切点（最早 11.600s），
        #   改完仍是 issue。共用判据之后，"修复动作"与"校验口径"天然同源：
        #   搬完每个段尾窗与回放区都不再有任何切点。
        if not is_last:
            # 尾窗宽度 = 一个回放窗，取 ``grid_seconds``（= 1.625）与 ``replay_in``
            # 同源；用名义 1.6 会让尾窗比真实回放窗短 1 帧，最后一个被回放的
            # 帧里可能仍留着切点/台词 → 接缝处「重复语音」。
            tail_start = b - grid_seconds(HANDOFF_SECONDS)
            cut_at = None
            for k in range(1, len(work_lines)):
                line = work_lines[k]
                if not (_SHOT_HEAD_RE.search(line) or _CUT_VERB_RE.search(line)):
                    continue
                codes = _timecodes(line)
                if codes and codes[0] >= tail_start - 1e-6:
                    cut_at = k
                    break
            if cut_at is not None and any("<d>" in x for x in work_lines[cut_at:]):
                cut_at = None
            if cut_at is not None:
                carry = work_lines[cut_at:]
                moved = len(carry)
                work_lines = work_lines[:cut_at]
                speaker_notes.append(
                    "尾窗（%gs 之后）里的切点已挪给下一段（%d 行）：留在本段会让"
                    "回放在接缝处重演一次切镜，而该镜正文里对上一句台词的引用会"
                    "在同一段里造成第二处台词文本，把模型往「提前说这句」的方向拉"
                    % (round(tail_start, 3), moved))
        detail = "\n".join(_retime_line(x, a, b, replay_in)
                           for x in _renumber_shots(work_lines, start_no)).strip()
        if replay_in > 0:
            # 回放镜：段首 ``replay_in``（默认 1.625s = 39 帧）是上一段结尾的
            # 原样重演，它是本段的 [Shot 1]，必须显式标出来 —— 否则模型看到的
            # 第一镜是新内容，回放区"不做任何新动作"的约束就没有附着点
            # （SKILL §1.4）。
            #
            # ★ 2026-09-27：这里的秒数一律走 ``_fmt_handoff_seconds``，与
            #   ``replay_in`` 字段同源，禁止再写死 1.6。
            detail = ("[Shot 1] [single continuous shot, %s seconds] %s\n\n%s"
                      % (_fmt_handoff_seconds(replay_in),
                         _handoff_replay_en(replay_in), detail)).strip()
        # ---- 通用兜底：本段一条台词都没分到时，显式写「这段不说话」 --------
        # 零台词段会让模型自行即兴发声（成片＝剧本外的含糊人声/乱说话）。它的
        # 成因与具体剧本无关：只要一句长台词没被完整留在同一段里，后续段就会
        # 空着（2026-09-26 实测：一句 7 秒台词被切穿，后段 0 条台词 → 成片
        # 29s 起「么我才又是惰 / 太太太太」）。改某一本剧本治不了这一类问题，
        # 必须在节点层兜住 —— 空段就明确写成静音段。
        if not any("<d>" in x for x in work_lines):
            detail = (detail + "\n\n" + NO_DIALOGUE_EN).strip()
        # 画面纪律总纲前置（全链逐字一致，见 _style_preamble）。放在
        # _auto_fix 之前：no-text 首句会被补到它前面，顺序正好是
        # 「不出现文字 → 取景纪律/节奏对位 → 正文」。
        preamble = _style_preamble(globals_fields)
        if preamble:
            detail = preamble + "\n\n" + detail
        # non_diegetic_music 全链同一份 MUSIC block（SKILL §1.5 / §4），不按段切。
        # 风格词从全片配乐概述里取（"Music bed = Peking-opera percussion" →
        # "Peking-opera percussion"）。
        music = build_music_block(
            style=_music_style_from_text(
                globals_fields.get("non_diegetic_music")))
        soundscape = _split_music_lines(
            globals_fields.get("overall_soundscape"), a, b, replay_in)
        # 台词编号推断的留痕（见 _tag_speakers 的 notes 注释）：静默补全等于
        # 把判断藏起来，所以每一条都带段号回到调用方，由它写进 issues。
        seg_notes = []
        tagged = _tag_speakers(detail, seg_notes)
        tagged = _inject_lipsync(tagged)
        # 逐句钉死交付纪律（只说一次 / 严格按时间码），通用、与剧本无关
        tagged = _inject_dialogue_once(tagged)
        # 逐句钉死防字幕禁令（只出声、严禁上屏），挂在台词块紧后面 ——
        # 段首/段尾的 no-text 句压不住概率性烧字幕（2026-09-30 实测）
        tagged = _inject_dialogue_no_text(tagged)
        # 统一段内每一条的行首时间码写法（skill §1.4 / 交付样本：
        # ``At MM:SS.mmm，…``）。放在 lipsync/hoist 之后，台词行已经是规范
        # 写法了，这里只负责把**其余**镜头条（裸时间码 / 括号时间码 / 英文
        # 逗号 / 完全没时间码）一起拉齐。
        tagged = _normalize_entry_clocks(tagged, lang)
        # 台词窗（from/to）时间码补齐毫秒位 —— 与行首时间码同一口径
        # （2026-09-30 压测 J：``from 00:01 to 00:03`` 原样进提示词）。
        tagged = _normalize_dialogue_spans(tagged)
        speaker_notes.extend("S%02d：%s" % (index + 1, n) for n in seg_notes)
        out.append(("S%02d" % (index + 1), dur_spec, lang,
                    _shot_seg_body(globals_fields, tagged,
                                   a, b, index + 1, total, music=music,
                                   hard=hard, soundscape=soundscape,
                                   lang=lang, replay_in=replay_in,
                                   gen_seconds=gen),
                    replay_in))
    return out, speaker_notes


def _pick_section(sections, language):
    """选要读的那一版。

    ``auto``（默认）**英文版优先** —— 现行规范的交付物本来就只有英文版，这个
    分支对它是直通。``language`` 只在读**旧双语 PACK** 时才有意义（挑
    ``[1] 中文版`` 还是 ``[2] 英文版`` 那一侧）。

    ★ 输出侧一律按英文单版规范化（段标题 ``EN``、EN 逗号、EN 的 no-text /
    DELIVERY / MUSIC block）。若被显式要求读中文版那一侧，节点不会替它翻译，
    而是照原样解析、由 ``validate_segment`` 按 §1.3 报「台词块之外出现中文」。
    """
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
    if want is None:                            # auto：英文版优先
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
                   replay_in=None, header_hard_cut=False):
    new, handoff, gen = parse_duration(dur_spec, default_duration)
    fields = split_fields(body)
    if not lang:
        # 段标题没写语言标记（en 单版里偶尔省）——从正文推断
        lang = _infer_lang(body)
    # ★★ 2026-09-30 全链排查的最大发现：逐句纪律注入（说话人补全 / LIPSYNC /
    #    DELIVERY 只说一次 / 防字幕 / 时间码规范化）原来**只**在 [Shot N]
    #    整段式路径（parse_shot_script_blocks）里跑 —— en-pack 分段路径
    #    （SKILL §5 现行规范的交付形态！）从来没有过这些注入，交付提示词里
    #    连 "DELIVERY: Speak this line EXACTLY ONCE" 都没有（v22qc manifest
    #    可证）。这正是「同一句被说两遍 / 提前到上一句的窗口里说」的路径级
    #    入口。这里把两条路径拉到同一口径；所有注入器都幂等，双路径叠加无害。
    seg_notes = []
    detail0 = fields.get("detailed_description") or ""
    if detail0.strip():
        # en-pack 只做**纯补全**类注入：说话人/口型句/只说一次/防字幕/毫秒位。
        # _hoist_dialogue_clock（经 hoist=False 关掉）与 _normalize_entry_clocks
        # 是整段式剧本的格式抢救机器 —— en-pack 交付稿行首本就有规范 At 时间码，
        # 再改写只会打乱行结构（2026-09-30 真实稿实测回归，勿加回）。
        detail0 = _tag_speakers(detail0, seg_notes)
        detail0 = _inject_lipsync(detail0, hoist=False)
        detail0 = _inject_dialogue_once(detail0)
        detail0 = _inject_dialogue_no_text(detail0)
        detail0 = _normalize_dialogue_spans(detail0)
        fields["detailed_description"] = detail0
    text_all = "\n".join(fields.get(name) or "" for name in FIELD_ORDER)
    summary = fields.get("summary") or ""
    # ★ 2026-09-30：段头语言槽位写 hard cut（``### S02 / 8s / hard cut ###``）
    #   也算硬切声明 —— 以前这个槽位当语言标记解析失败后静默丢弃，硬切段
    #   被当连续段处理，回放/尾窗全错位（压测 E 实证）。来源记进 marker，
    #   由 _auto_fix 写一条 fixes 说明。
    hard_cut = (bool(re.search(r"HARD\s*CUT", summary, re.I))
                or bool(header_hard_cut))
    hard_cut_from_header = bool(header_hard_cut) and not bool(
        re.search(r"HARD\s*CUT", summary, re.I))
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
        if is_first:
            replay_in = 0.0
        elif handoff:
            replay_in = handoff
        else:
            # ★ 2026-09-30 压测 B/F 实证的漏洞：接续段段头漏写 ``+1.6`` 时，
            #   这里以前落 0 —— 回放句不补、回放区台词/切点不平移；而
            #   ``to_shots_json.segment_windows`` 会给所有非首段**强制**补
            #   1.625s 回放窗。两边口径不一致 = 台词/切点留在会被强制回放的
            #   窗口里，静默进成片（重复语音/闪跳）。解析层与渲染层必须同口径：
            #   非首段、段头没写回放，就按网格回放窗补齐。
            replay_in = grid_seconds(HANDOFF_SECONDS)
    if hard_cut:
        handoff = 0.0
        gen = new
        replay_in = 0.0
    # ★ 回放/锚点窗口一律对齐 17k+5（1.6s → 1.625s = 39 帧）。
    #   ``parse_duration`` 从段标题 ``10s+1.6=11.6`` 里取到的是**名义** 1.6，
    #   38 帧不在网格上；渲染层 ``_shot_plan`` 按向上取用 39 帧，而这里若
    #   原样留 1.6，两处就差 1 帧，``gen_seconds`` 跟着差一帧 → 写进
    #   storyboard 后又被反算一次，每段错开一拍（接缝处多出 17 帧回放的
    #   经典症状）。对齐后再重算 gen，保证 ``new + replay = gen`` 恒成立。
    if replay_in:
        replay_in = grid_seconds(replay_in)
    if handoff:
        handoff = grid_seconds(handoff)
    if not hard_cut:
        gen = round(new + replay_in, 3)
    # ★★ 2026-09-30（第二轮）：把「17k+5 网格对齐」从**渲染层**下沉到**解析层**。
    #
    #   H3 只能生成 17k+5 帧（5/22/39/…/260/…）。渲染层 ``_shot_plan`` 会把
    #   ``gen`` 向上吸附到网格 —— 每段因此多出 **0~16 帧（最多 0.67s）剧本里
    #   根本没有的时间**。而拼接只裁「段首回放」，这些多出来的帧**整段进成片**，
    #   模型则拿它即兴发声：这正是"乱说话 / 和剧本对不上"的结构性来源。
    #
    #   v23full 实测（每段多出帧数 / 秒）：
    #     S01 260-254= 6 帧(0.233s) S02 175-164=11 帧(0.467s)
    #     S03 311-303= 8 帧(0.333s) S04 311-298=13 帧(0.533s)
    #     S05 226-217= 9 帧(0.392s)
    #   5 段合计 47 帧 ≈ **1.96s**：PACK 头部写 Total 00:45，实际成片 46.96s。
    #   而每段提示词里的时间码只写到名义 gen（如 S02 写到 00:06.800），模型
    #   被告知"本段到 6.8s 结束"却必须生成 7.292s —— 多出来的 0.49s 无指令，
    #   实测 ASR 就在那里听到了剧本外的音节（seg_02 的 6.84–7.22s）。
    #
    #   修法：``gen`` 直接取网格帧数 / fps，``new`` 取**真实贡献**
    #   （网格帧 − 回放帧）。这样「提示词时间轴 = shots.json = 实际生成帧数」
    #   三者同源；``_auto_fix`` 再把段末时间码延长到真实终点，模型于是拿到
    #   一段**被完整指定**的时间轴，没有空白可以即兴。
    #   ``grid_seconds`` 与渲染层 ``generation_length`` 是同一算法（见其
    #   docstring），所以这里算出来的帧数与 ``_shot_plan`` 逐帧一致。
    gen = round(grid_seconds(gen), 3)
    new = round(gen - replay_in, 3)
    seg = {
        "id": seg_id,
        "lang": lang,
        "new_seconds": new,
        "handoff_seconds": handoff,
        "replay_in": replay_in,
        "gen_seconds": gen,
        "spec_frames": frames_for_seconds(gen),
        "hard_cut": hard_cut,
        # 段头声明的硬切（summary 里没写）：_auto_fix 据此补一条 fixes 说明，
        # 让用户知道"硬切来自段头槽位"而不是静默生效。
        **({"hard_cut_source": "header"} if hard_cut_from_header else {}),
        "fields": fields,
        "pictures": _picture_slots(fields),
        "videos": sorted({int(n) for n in _VID_RE.findall(text_all)}),
        "subjects": sorted({int(n) for n in _SUB_RE.findall(text_all)}),
        "speakers": sorted({int(n) for n in _SPEAKER_RE.findall(
            _DIALOG_RE.sub("", fields.get("detailed_description") or ""))}),
        "dialogues": len(_DIALOG_RE.findall(fields.get("detailed_description") or "")),
        "task": TASK_T2V,
        "issues": [],
        # 节点自动修过的项目（**不是问题**，是成功日志；以前混在 issues 里被
        # 当作"注意"打印，用户误以为是警告）。报告/前端要把它单独显示。
        "fixes": [],
    }
    # 说话人编号是推断来的（原文没标 (Sx)）—— 必须让用户看得见，能复核
    if seg_notes:
        seg["issues"].extend(seg_notes)
    return seg


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


def pack_header(meta, segments=None):
    """把 meta 里的头部键值整理成 PACK 模块要显示的五项 + 附加信息。

    返回统一的小写英文键，前端与 Director 的 pack_info 输出都吃这一份::

        {"project", "mode", "total_duration", "total_duration_s", "segments",
         "aspect", "music", "version", "date", "rule_version", "layout"}

    ``total_duration_s`` 是把 ``00:40`` / ``40s`` / ``40`` 统一成秒的浮点值，
    前端画时间线直接用；解析不出来就是 ``None``。

    ★ 头部缺失时从 ``segments`` 反推（2026-09-19）：``[Shot N]`` 分镜脚本这类
    输入**没有 PACK 头部区块**，meta 里一个键值都没有，于是面板「项目 / 模式 /
    总时长 / 段数 / 画幅」五项全显示「—」，看上去像没解析出来。
    段数与总时长是**能从解析结果精确算出来的**，这里补上（成片时长 = 各段
    ``new_seconds`` 之和 —— 与前端「N 段」徽标、时间线刻度同一个口径）。
    项目名 / 画幅 PACK 里确实没有，不编，留空交给上游的会话名去显示。
    """
    meta = meta or {}

    def _pick(*keys):
        for key in keys:
            value = meta.get(key)
            if value:
                return str(value).strip()
        return ""

    segs = list(segments or [])
    duration_raw = _pick("Total duration", "总时长", "总长", "Duration", "时长")
    if not duration_raw and segs:
        total_new = sum(float(s.get("new_seconds") or 0.0) for s in segs)
        if total_new > 0:
            duration_raw = "%.1fs" % round(total_new, 3)
    seg_count = _pick("Segments", "段数")
    if not seg_count and segs:
        seg_count = str(len(segs))
    return {
        "project": _pick("Project", "项目"),
        "mode": _pick("Mode", "模式"),
        "total_duration": duration_raw,
        "total_duration_s": _parse_duration_text(duration_raw),
        "segments": seg_count,
        "aspect": _pick("Aspect", "画幅", "Resolution", "分辨率"),
        "music": _pick("Music", "配乐"),
        "version": _pick("Version", "版本"),
        "date": _pick("Date", "日期"),
        "rule_version": meta.get("rule_version") or "",
        "layout": meta.get("layout") or "",
    }


PACK_SEP = "=" * 58
PACK_TITLE = "MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包"
PACK_END = "END OF PACK / 文件结束"
# SKILL §5 的分区标题：英文版是**本文件唯一语言版本**
PACK_EN_HEADING = "ENGLISH VERSION / 英文版（本文件唯一语言版本）"


def _fmt_mmss(seconds):
    """``00:40`` —— PACK 头部的 Total duration 写法（不带毫秒）。"""
    value = float(seconds or 0.0)
    return "%02d:%02d" % (int(value // 60), int(round(value % 60)))


def pack_dur_spec(seg):
    """段标题里的时长写法（SKILL §0.1）：``11s`` / ``11s+1.6=12.6s``。

    ``parse_shot_script_blocks`` 算过一次但没落到段对象上，这里从秒数反推 ——
    段标题是全片唯一一处把「新增 / 回放 / 生成」三个数一起摆出来的地方，渲
    染前扫一眼就能发现规划错了。

    写法与 SKILL §0.1 逐字对齐：``11s`` / ``11s+1.6=12.6`` —— 只有"新增"带
    ``s``，回放量与生成值不带（``########## S02 / 11s+1.6=12.6 / EN ##########``）。
    """
    new = float(seg.get("new_seconds") or 0.0)
    gen = float(seg.get("gen_seconds") or 0.0)
    extra = gen - new
    if extra <= 0.001:
        return "%gs" % new
    return "%gs+%g=%g" % (new, extra, gen)


def pack_text(segments, meta=None, version="v1", date=None, project=None,
              mode=None, aspect=None):
    """把解析结果拼成**可交付的 PACK 文本**（SKILL §5）：头部 + 英文分区 +
    ``########## S01 / 11s / EN ##########`` 段标题 + ``END OF PACK``。

    以前节点只把六段正文交给下游，一个 PACK 该有的头部（项目 / 模式 / 总时长 /
    段数 / 画幅 / 配乐 / 版本 / 日期）和段标题全都没有 —— 而段标题正是 SKILL
    §0.1 用来**声明本段新增多少、回放多少、一共生成多少**的地方，缺了它，下游
    （以及人）只能从正文里数时间码去反推。

    头部缺的项（项目名、模式、画幅）按 ``pack_header`` 的结果走，拿不到就留
    空由调用方补；总时长与段数在 shot-script 这种没有头部的形态下从 segments
    反推（见 ``pack_header``）。
    """
    segs = list(segments or [])
    header = pack_header(meta or {}, segs) if (meta or segs) else {}
    total_s = header.get("total_duration_s")
    if total_s is None and segs:
        total_s = sum(float(s.get("new_seconds") or 0.0) for s in segs)
    lines = [PACK_SEP, PACK_TITLE]

    def _row(label, value):
        # 列宽 21 是交付样本逐字对齐得来的：最长那行 ``Total duration / 总时长``
        # 正好 20 个字符 + 1 空格，冒号落在第 22 列。用别的宽度打开就不是同一
        # 份文件了（diff 会全行飘红，虽然只是空格差异）。
        lines.append("%-21s: %s" % (label, value or ""))

    _row("Project / 项目",
         project or header.get("project"))
    _row("Mode / 模式", mode or header.get("mode"))
    # 头部规范写 ``00:40``。shot-script 这类没有头部的形态里 total_duration 是
    # 反推出的 "40.0s"，统一成 mm:ss 交付。
    _row("Total duration / 总时长", _fmt_mmss(total_s))
    _row("Segments / 段数", header.get("segments") or str(len(segs)))
    _row("Aspect / 画幅", aspect or header.get("aspect"))
    _row("Music / 配乐", header.get("music") or "Per MUSIC block")
    _row("Version / 版本", header.get("version") or version)
    _row("Date / 日期", header.get("date") or (date or ""))
    lines.append(PACK_SEP)
    # SKILL §5 的分区标题。节点只产出**英文单版**（翻译不是节点该干的事，
    # 见模块顶部说明），所以只写英文这一区，不再有 ``[1] 中文版``。
    lines.extend(["", PACK_SEP, PACK_EN_HEADING, PACK_SEP])

    for seg in segs:
        lines.append("")
        lines.append("")
        lines.append("########## %s / %s / %s ##########"
                     % (seg.get("id") or "S??", pack_dur_spec(seg), "EN"))
        fields = seg.get("fields") or {}
        blocks = []
        for name in FIELD_ORDER:
            value = str(fields.get(name) or "").strip()
            if value:
                blocks.append("%s:\n%s" % (name, value))
        lines.append("\n\n".join(blocks))

    lines.extend(["", "", PACK_SEP, PACK_END, PACK_SEP, ""])
    return "\n".join(lines)


def pack_info_json(segments, meta, extra=None):
    """PACK 模块的负载：头部五项 + 每段一行 + 参考图槽位。

    H3PromptPackParser 把它作为 ``pack_info`` 输出，H3Director 接过去原样转出
    并在运行报告里回显。前端的 PACK 面板读的是同一个结构（走
    ``/h3/pack_preview`` 拿实时解析结果，不用等一次渲染）。
    """
    # 传 segments：没有 PACK 头部的输入（[Shot N] 布局）靠它反推段数/总时长
    header = pack_header(meta, segments)
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


def _mark_silent_subjects(sd_text, silent_ids):
    """给**从不说话**的 `<Subject N>` 定义行补「never speaks / no dialogue」。

    规范 1.2：非说话角色不给 ``(Sx)`` 编号，并要在 ``subject_definitions`` 与
    正文里明写 ``never speaks / no dialogue``。剧本常常只写角色关系、不写这句
    声明，模型就默认谁都能开口 —— 这正是「乱说话 / 多出一个人的声音」的入口。
    这里按「全片从没当过 (SN) 说话人」来判定，补在每个角色自己的定义行末尾。

    ★ 幂等：行里已经有声明就不重复补。
    """
    if not sd_text or not silent_ids:
        return sd_text
    return _append_silent_note(
        sd_text, silent_ids,
        "; no speaker number is assigned, never speaks, no dialogue.")


def _append_silent_note(text, silent_ids, tag):
    """给 ``<Subject N>`` 开头的行追加「不说话」声明（sd / retention 共用）。

    两处都要写：``subject_definitions`` 定义这个角色时说一次，
    ``retention_analysis`` 声明它在这一段里的状态时再说一次 —— 只写一处的话，
    模型在"这一段要不要让它开口"这件事上仍有自由度（交付样本两处都写了）。

    ★ 幂等：行里已经有中英文任一种声明就不重复补。
    """
    if not text or not silent_ids:
        return text
    out = []
    for line in str(text).splitlines():
        match = re.match(r"^\s*<Subject\s+(\d+)>", line)
        if match and int(match.group(1)) in silent_ids:
            if not re.search(r"never speaks|no dialogue|不说话|无台词", line, re.I):
                line = line.rstrip().rstrip("。.") + tag
        out.append(line)
    return "\n".join(out)


def build_music_block(text="", sfx_only=False, style=None):
    """SKILL §1.5 的 MUSIC block（英文版，唯一交付语言）。

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


# 行内「节拍标记」的判定：时间码前面必须是行首 / `[Shot N]` / `At ` / `自` / `于`
# / 区间连字符。**正文里的回溯引用不算节拍** ——
# 实测踩过：`…不是 00:01.600 那枚浅笑…` 这类回指会被当成时间码倒退，
# 误报「时间码不是严格递增」。回溯引用前面跟的是"是/的/比"这类词，不匹配。
_BEAT_BEFORE_RE = re.compile(
    r"(?:^|\[\s*Shot\s*\d+\s*\]\s*|\bAt\s+|自\s*|于\s*|[-–—]\s*)$")


def _beat_codes(line):
    """行内**节拍**时间码（用于「严格递增」校验）。

    只取前面是行首 / ``[Shot N]`` / ``At`` / ``自`` / ``于`` / 区间连字符的那些；
    正文里回指早前时刻的时间码（"不是 00:01.600 那枚浅笑"）不算 —— 把它算进来
    必然报"不是严格递增"，但那是正常的叙事回指，不是写错。
    """
    out = []
    for m in _SHOT_CLOCK_RE.finditer(str(line or "")):
        if _BEAT_BEFORE_RE.search(line[:m.start()]):
            out.append(_clock_seconds(*m.groups()))
    return out


# ---------------------------------------------------------------------------
# 切点时间码（规范 1.4：回放区与尾窗必须一镜到底）
# ---------------------------------------------------------------------------
# 回放区（段首 0–1.6s）是拿上一段**最后一帧**当参考图重新生成的：这段里只要有
# 一个切点，下一段开头就只能对上尾帧那一侧，接缝必崩（成片表现＝单帧闪跳）。
# 尾窗（最后 1.6s）同理 —— 它要被下一段原样回放，含切点等于让下一段开头重演
# 一次切镜。规范 1.4 把这条列为「实操最容易漏的一条」，而节点以前完全没查，
# 这正是「接缝闪跳」一直只能靠肉眼找、查不出根因的原因。
#
# 判据是「这一行有时间码 + 有切镜动作」，语言无关：
#   * ``[Shot N] At 00:XX.XXX``  —— 镜头标题本身就是切点
#   * ``At 00:XX.XXX … cuts to / back to / HARD CUT``
#   * ``00:XX.X 切 <Subject N> / 反切 / 切回 / 切给 / 插入 <Subject N>``
_CUT_VERB_RE = re.compile(
    r"(?:cuts?\s+to\b|back\s+to\b|HARD\s*CUT|反切|切回|切给|切换|镜头切"
    r"|插入\s*<|切\s*<)", re.I)


def _cut_times_in(detail):
    """返回段内所有切点的时间码（升序、去重）。语言无关。

    ★ 段首 0.000 的那个镜头起点**不算切点**：它就是回放窗本身，把它算进来
    会让每一个接续段都误报「回放区有切点」。
    """
    out = []
    for line in str(detail or "").splitlines():
        codes = _timecodes(line)
        if not codes:
            continue
        if _SHOT_HEAD_RE.search(line) or _CUT_VERB_RE.search(line):
            if codes[0] >= 0.05:
                out.append(codes[0])
    return sorted(set(out))


def _style_vs_reference_conflicts(segments):
    """风格词与「外观照参考图」冲突的**只读**检测（全片报一次）。

    ★ 2026-09-27 新增。起因：iter7 真渲染里 S03/S04 的角色被画成**京剧花旦**
      （珠翠头面 + 浓重戏曲妆），而 S01/S02 是正常的米白交领袍 —— 同一角色
      跨段换造型，与 ``<Picture 2>``（深红纱袍）也对不上。

      根因不在节点，而在**剧本的风格措辞**：``summary`` 写着
      「古装**戏曲**喜剧质感 / **京剧**锣鼓」「the bed as the **stage**」
      「the opera's **posing**/gong-drum beats」，模型把 "Peking Opera comedy
      texture" 字面理解成"扮上戏妆"；而同一份 PACK 的 ``retention_analysis``
      又写着「外观与参考图一致」。**两条指令互相矛盾**，模型只能二选一 ——
      实测它给 S03/S04 选了戏曲扮相。

      节点**不改写**剧本（用户裁决：禁止私自更换描述），但必须让这个冲突
      **可见** —— 否则只能靠出片后肉眼看出来。

    ★ 判据用**风格短语**而不是「花旦/妆容」这类词：实测 prompt 里根本没有
      "make-up"/"花旦" 字样，是模型自己从 "Peking Opera comedy texture"
      演绎出来的。所以要盯的是**源头**那句风格定位。
    """
    joined = "\n".join(str(s["fields"].get("summary") or "") + "\n" +
                       str(s["fields"].get("detailed_description") or "")
                       for s in segments)
    hit = re.search(
        r"Peking\s+Opera|Beijing\s+Opera|Chinese\s+opera|京剧|戏曲|花旦|头面|脸谱"
        r"|戏妆|扮相|opera\s+make-?up|operatic\s+(?:posing|texture)"
        # ★ 兜底用**裸 opera/operatic**：同一句风格定位在不同译次里可能是
        #   "Peking Opera comedy texture" / "Ancient opera comedy texture" ——
        #   只认前一种会漏（实测 live_translate_out2 就漏了）。这一段本来
        #   只在"外观锁定参考图"同时成立时才报，误报代价低、漏报代价高。
        r"|\boperatic\b|\bopera\b", joined, re.I)
    if not hit:
        return []
    locked = any(re.search(r"fully_preserved|reference image|参考图",
                           str(s["fields"].get("retention_analysis") or ""), re.I)
                 for s in segments)
    if not locked:
        return []
    return [
        "★ 风格词与「外观照参考图」冲突（全片级，只报一次）：正文出现「%s」"
        "这类**戏曲风格定位**，而 retention 又声明外观锁定参考图 —— 模型只能"
        "二选一。实测 iter7：S03/S04 的角色被画成**京剧花旦**（珠翠头面 + 浓妆），"
        "S01/S02 却是正常交领袍，同一角色跨段换造型，且与 <Picture 2> 都对不上。"
        "建议在**剧本里**把风格限定到**配乐 / 节奏 / 镜头**层面，并显式加一句"
        "「人物妆造一律照参考图，不做任何戏曲扮相 / 不上戏曲妆」。"
        "节点不改写剧本，只报这一条。" % hit.group(0)
    ]


# 台词句引用的判据：引号内**像一句可念的话**（多词 + 含谓语/句末标点）。
# 音效拟声（tai / cang / bang - bang）、单字词、技法名（counting-board /
# four-strike / display shot sequence）、镜头标题（Door Kicked Open,
# Lovebirds Caught）都不该命中 —— 实测模型只把**完整句子**念出来。
_QUOTED_SENTENCE_RE = re.compile(
    r"\"([^\"\n]{6,140})\"")
_SENTENCE_LIKE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|am|do|does|did|have|has|had|"
    r"will|would|shall|should|can|could|may|might|must|"
    r"know|knew|come|came|go|went|get|got|make|made|"
    r"different|married|marry|lose|lost|found|find|"
    r"how|why|what|when|where|who)\b", re.I)
# 镜头标题：``[Shot N] At 00:XX.XXX, "标题"`` —— 卷名不是台词句，跳过。
# （``She Is a Professional`` 含 is，会被 _SENTENCE_LIKE_RE 命中，必须靠位置排除）
_SHOT_TITLE_BEFORE_RE = re.compile(
    r"\[Shot\s+\d+\][^\"\n]{0,40},\s*$", re.I)
# 已知安全的引号内容白名单（技法 / 镜头标题 / 音效拟声）—— 命中即跳过。
_QUOTED_SAFE = {
    "tai", "cang", "bang - bang", "ta-ta-ta", "ta ... ta ...",
    "tai, tai", "tai, tai, tai", "tai - ", "cang - ",
    "chaotic hammer", "counting-board", "four-strike", "display shot sequence",
    "loose cut", "cold", "ask", "deal", "worth it", "brothers", "terrific",
    "riffic", "professional", "quoting", "OK", "cools down",
    "the slower the colder",
    "feet to legs to waist to chest",
    "two camera grammars mapping one hot and one cold",
    "hot camera handheld sway / cold camera locked motionless",
    "hot handheld push and whip / cold static slow tilt",
    "the more chaotic the stiller, the hotter the colder",
    "Zhang San",
}


def _quoted_line_refs(raw_text):
    """台词句被**引号引用在正文里**的只读检测（全片报一次）。

    ★ 2026-09-27 新增。起因：英文单版 ``node748_en.txt`` 的 ``summary`` /
      ``detailed_description`` / ``overall_soundscape`` 里写着
      ``delivers the killing line "How are you any different from Cao Cao's
      thieving ilk?"`` —— 模型把这串**英文句子**当成台词念了出来。
      成片 ASR 在段3（29.96–38.4s）转写出 ``How are you any matter for go?``，
      正是那句的念读畸变，听感就是「乱说话 / 与剧本对不上」。

      为什么中文原稿没这个问题、英译版才有：
        · 中文原稿对应位置写的是中文引号句（``"汝与曹贼何异？"``），
          被「零中文闸门」（见 ``_strip_residual_cjk`` 调用处）**整体剥掉**，
          模型看不到 → 不会念；
        · 译成英文后**绕过了中文闸门**，句子合法地留在正文里 → 被念出来。
      这与「中文漏进提示词」是**同一类**代价（模型把正文读成台词内容），
      只是漏进来的语言从中文变成了英文。

    ★★ **必须在 ``parse_pack`` 的入口（``raw``）上调用**：节点自身的
      「双引号自动删除」（见 ``_QUOTE_RE`` 处）会把引号字符先删掉，
      到 ``segments`` 阶段**已经看不到引号**，判据必然落空
      （2026-09-27 实测：在 segments 上检测，改动前/后都报 0 命中）。

      节点**不改写剧本**（用户裁决：禁止私自更换描述），只把风险**报出来**。
      修法（改稿）：把引用的台词句改成**不写原文的指代**，例如
        ``delivers the killing line (the one line that pins the boss as
        Cao Cao's equal in thieving)`` / ``on that phrase`` /
        ``still frozen in the shock of the verdict he has just been handed``。
      音效拟声与技法名不必改（见 ``_QUOTED_SAFE``）。
    """
    raw = str(raw_text or "")
    # 只看 <d> 块**之外**（块内的引号属于台词原文，一个字都不动）
    outside = _DIALOG_RE.sub("", raw)
    hits, seen = [], set()
    for m in _QUOTED_SENTENCE_RE.finditer(outside):
        cand = m.group(1).strip()
        if cand in _QUOTED_SAFE:
            continue
        if len(cand.split()) < 3:
            continue
        # 镜头标题：[Shot N] At 00:XX.XXX, "标题" —— 卷名不是台词句
        if _SHOT_TITLE_BEFORE_RE.search(outside[max(0, m.start() - 60):m.start()]):
            continue
        if not _SENTENCE_LIKE_RE.search(cand):
            continue
        if cand in seen:
            continue
        seen.add(cand)
        hits.append(cand)
    if not hits:
        return []
    lines = ["★ 正文里**引用了完整台词句**（全片级，只报一次）：模型会把这类"
             "英文句子当成台词念出来 → 成片计划外发声 / 与剧本对不上。"
             "实测本机：``summary`` 里的 \"How are you any different from "
             "Cao Cao's thieving ilk?\" 让段3 在 29.96–38.4s 冒出念读畸变 "
             "``How are you any matter for go?``（ASR 实证）。"
             "★ 中文原稿不会踩：中文引号句被「零中文闸门」整体剥掉了；"
             "译成英文后绕过闸门，才留在正文里被念。"]
    for cand in hits[:8]:
        lines.append("    · 「%s」" % cand[:80])
    if len(hits) > 8:
        lines.append("    · …另有 %d 处" % (len(hits) - 8))
    lines.append("  建议在**剧本里**把这些引用改成**不写原文的指代**"
                 "（the killing line / that phrase / the verdict he has just "
                 "been handed 之类）；音效拟声（tai / cang）与技法名"
                 "（counting-board / 镜头标题）不必改。节点不改写剧本，只报这一条。")
    return ["\n".join(lines)]


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
        if "<d>" not in line:
            continue
        missing = []
        if not _SPEAKER_RE.search(line):
            missing.append("(Sx)")
        if _dialog_missing_lang(line):
            missing.append("[语言] 标签")
        if missing:
            issues.append("台词行「缺一不可」缺 %s：%s"
                          % ("、".join(missing), line.strip()[:40]))
            break

    # 台词时间窗重叠 —— 「乱说话」的一种：两句台词占了同一时刻，模型只能挑
    # 一句说（另一句凭空消失），或者把两句叠在一起说。留 0.3s 容差：正常的
    # "尾字接话"本来就有零点几秒交叠，那不是错。
    talk_wins = []
    for line in (fields.get("detailed_description") or "").splitlines():
        if "<d>" not in line:
            continue
        codes = _timecodes(line)
        if codes:
            talk_wins.append((min(codes), max(codes), line.strip()[:24]))
    clash = None
    for a in range(len(talk_wins)):
        for b in range(a + 1, len(talk_wins)):
            lo = max(talk_wins[a][0], talk_wins[b][0])
            hi = min(talk_wins[a][1], talk_wins[b][1])
            if hi - lo > 0.3:
                clash = (hi - lo, talk_wins[a][2], talk_wins[b][2])
                break
        if clash:
            break
    if clash:
        issues.append("台词时间窗重叠 %.2fs（乱说话）：%s ↔ %s" % clash)

    # 切镜与台词**同刻起** —— skill §1.4「台词与大动作不重叠」（§1.4 / 交付
    # 样本的 DELIVERY 纪律）。剧本常常把「切到某人」和「某人开口」写在同一秒
    # （本次 S01 的 00:07.200 就是：同一刻既切 <Subject 2> 又说「汝与曹贼何
    # 异」）。模型只能二选一：要么画面先切完再说话（口型丢了前半句），要么
    # 边切边说（切镜动作吃掉口型）。给的是可直接改的数值。
    for line in (fields.get("detailed_description") or "").splitlines():
        if "<d>" in line or not _CUT_VERB_RE.search(line):
            continue
        cuts = _timecodes(line)
        if not cuts:
            continue
        for lo, hi, tip in talk_wins:
            if abs(min(cuts) - lo) <= 0.05:
                issues.append(
                    "切镜与台词同刻起（%.3fs）：切镜「%s」↔ 台词「%s」。skill "
                    "§1.4 要求台词与大动作不重叠 —— 把切镜提前到 %.3fs 之前（给"
                    "切镜留出落定时间），或把台词起点推到 %.3fs 之后。"
                    % (min(cuts), line.strip()[:24], tip,
                       max(0.0, min(cuts) - 0.4), min(cuts) + 0.4))
                break

    # SKILL §1.3 / §5.1：全文除 ``<d>[Chinese] 原文</d>`` 台词块外零中文字符。
    # 这条对**每一段**都成立（英文单版是唯一交付语言），不再按段标题的语言标记
    # 分情况 —— 源脚本写成中文时，这里就是最直接的定位手段。
    #
    # ★ 正常情况下这条**报不出来**：``_auto_fix`` 的交付闸门已经把台词块之外的
    #   中文剥离掉了（并写进 seg["fixes"]）。能走到这里只剩一种情况 ——
    #   「自动修正」被关掉。此时报 issue 而不是悄悄放过，是对的。
    stripped = _DIALOG_RE.sub("", joined)
    if _CJK_RE.search(stripped):
        issues.append(
            "台词块之外出现中文字符（SKILL §1.3 / §5.1：英文单版除 "
            "<d>[Chinese] 原文</d> 外不得出现任何中文）。本段是整段式中文脚本"
            "或被指定读了旧双语 PACK 的中文侧 —— 节点不翻译，请按 SKILL §5 交付"
            "英文单版 PACK。（「自动修正」打开时节点会就地剥离这些中文，"
            "并在报告里记一条「已自动修复」；本条只在自动修正关闭时出现。）")
    if detail:
        count = _word_count(_DIALOG_RE.sub("", detail))
        low = EN_WORDS_TOLERANCE[0]
        # 只报**过短**。规范写的 350–500 词是**动笔写**提示词的区间；
        # 解析器这边吃的是已经写好的 PACK，写得更详细不是错误 —— 把超长
        # 也报成 issue，会让 on_issue=error 直接中断一次本来没问题的渲染，
        # 也会把真正该看的问题（缺字段、回放区台词）淹在一堆噪音里。
        #
        # ★ 正文还是中文时不报这一条：那说明源脚本没按 §5 交英文单版，
        #   §1.3 那条已经指明了；再叠一条"英文词数偏低"会让用户以为
        #   "既要翻译、又要加词"，其实只要交英文单版、STYLE 块自然会带进来。
        if len(_CJK_RE.findall(_DIALOG_RE.sub("", detail))) < 10 and count < low:
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
        #
        # ★ 行内只比**节拍标记**（见 _beat_codes）：正文里的回溯引用
        #   （"不是 00:01.600 那枚浅笑"）不是节拍，比进去会误报。
        for line in detail.splitlines():
            line_codes = _beat_codes(line)
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

    # 切点纪律（规范 1.4）：回放区与尾窗各必须是**一整段连续镜头**，中间不能有
    # 切点。以前节点只管台词落在这两个窗口里（「重复说话」），切镜一直没人管
    # —— 而切点落在回放区/尾窗，成片表现就是接缝处的单帧闪跳。
    if not seg.get("hard_cut"):
        replay_in = float(seg.get("replay_in") or 0.0)
        cut_times = _cut_times_in(detail)
        if replay_in > 0:
            in_replay = [t for t in cut_times if t < replay_in - 0.001]
            if in_replay:
                issues.append(
                    "回放区（0–%.3fs）内有 %d 个切点（最早 %.3fs）：回放区是拿上"
                    "一段最后一帧当参考图重生成的，含切点时下一段只能对上尾帧那"
                    "一侧，接缝必崩。把该切镜提前到段首 %.3fs 之后，或把那个镜头"
                    "整段挪到 %.3fs 之后当新内容处理"
                    % (replay_in, len(in_replay), in_replay[0], replay_in, replay_in))
        if not is_last:
            # ★ 尾窗必须用本段自己的 ``handoff_seconds``（已对齐 17k+5 = 1.625s
            #   / 39 帧），不能用名义 ``HANDOFF_SECONDS``（1.6s / 38 帧）。用名义
            #   值会让切点/台词被平移到「gen-1.6」，仍比真正的尾窗起点
            #   「gen-1.625」晚 0.025s ≈ 1 帧 —— 这一帧会被下一段原样回放成
            #   重复语音（"乱说话"在接缝处的经典表现）。
            tail_start = seg["gen_seconds"] - float(
                seg.get("handoff_seconds") or HANDOFF_SECONDS)
            in_tail = [t for t in cut_times if t > tail_start + 0.001]
            if in_tail:
                issues.append(
                    "尾窗（最后 %.1fs，即 %.3fs 之后）内有 %d 个切点（最早 "
                    "%.3fs）：这 %.1fs 会被下一段原样回放，含切点会在接缝处重演"
                    "一次切镜 → 单帧闪跳。把该切镜提前到 %.3fs 之前，或把那个镜头"
                    "挪到下一段开头当新内容"
                    % (HANDOFF_SECONDS, tail_start, len(in_tail), in_tail[0],
                       HANDOFF_SECONDS, tail_start))

    if seg["new_seconds"] > MAX_NEW_SECONDS + GRID_SLACK_SECONDS + 0.001:
        issues.append("新增时长 %.2fs 超过单段上限 %.0fs，建议拆段"
                      % (seg["new_seconds"], MAX_NEW_SECONDS))
    elif seg["new_seconds"] < MIN_NEW_SECONDS - 0.001:
        # 2026-09-19：给出可执行的具体数值 —— 切点往前挪多少秒、合并后多长。
        # 单纯说「建议合并」用户还得自己算；算不准的就改不动，等于没说。
        merge_target = max(MIN_NEW_SECONDS, 5.0)
        delta = merge_target - seg["new_seconds"]
        seg_total = float(seg.get("new_seconds") or 0.0)
        issues.append("新增时长 %.2fs 偏短（<%.0fs，H3 上容易退化成近乎静止"
                      "镜头）；建议把这段切点前移 %.2fs 并入上一段，合并后约"
                      " %.2fs（再短就该往前吃掉再前一段了）"
                      % (seg_total, MIN_NEW_SECONDS, round(delta, 2),
                         round(seg_total + delta, 2)))

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
        # ★ 同上：尾窗起点走本段 ``handoff_seconds``（17k+5 对齐的 1.625s），
        #   而非名义 1.6s，否则判据会晚 1 帧放行台词。
        tail_start = seg["gen_seconds"] - float(
            seg.get("handoff_seconds") or HANDOFF_SECONDS)
        talk_codes = []
        for line in (fields.get("detailed_description") or "").splitlines():
            if "<d>" in line:
                talk_codes.extend(_timecodes(line))
        if talk_codes and max(talk_codes) > tail_start:
            # 2026-09-19：把"具体怎么改"也写进 issue —— 把该台词的「必须于 X 前
            # 说完」收紧到 tail_start 之前，避免用户自己对着时间码算。
            issues.append("最后 %.1fs 有台词（%.3fs），会被下一段回放成重复"
                          "语音；把该句的「必须于 ... 前说完」改到 %.3fs 之前、"
                          "或者把台词起点提前到 %.3fs 之前"
                          % (HANDOFF_SECONDS, max(talk_codes), tail_start,
                             tail_start - (max(talk_codes) - tail_start)))
    return issues


def _shift_clocks_where(line, delta, below):
    """只把**小于 ``below`` 的**时间码平移 ``delta``，其余原样保留。

    ★ 2026-09-30：台词行里通常同时挂着两组时间码 —— 本镜头的起点
      （``At 00:01.600 …``）与台词自己的起止窗（``from 00:03.200 to
      00:04.200``）。回放区修正只需要动**落在回放窗内的那一个**；整行平移会把
      窗外的台词窗一起推走。实测（v23full 成片）S03 因此把台词尾从 4.200 推到
      4.225，越过下一个镜头的起点 4.200；S05 同理（6.895 > 6.870）—— 段内时间码
      出现 **1 帧倒流**，模型收到「这句说到 4.225，可下一个镜头 4.200 就开始了」
      这种自相矛盾的指令。
    """
    if abs(delta) < 1e-6:
        return line

    def _sub(match):
        sec = _clock_seconds(*match.groups())
        if sec >= below - 1e-6:
            return match.group(0)
        moved = sec + delta
        return match.group(0) if moved < 0 else _fmt_clock(moved)

    return _SHOT_CLOCK_RE.sub(_sub, line)


def _shift_line_clocks(line, delta):
    """把一行里所有 ``mm:ss.xxx`` 整体平移 ``delta`` 秒（台词原文一个字不动）。

    只改时间码、不改文字，所以**幂等**：第二次进来时时间码已在合法区间，
    delta 算出来是 0，原样返回。挪成负数的时间码保持原样（宁可不挪，也不要
    造出一个 00:0-1.500 这种鬼东西）—— 那种情况留给 issue 去报。
    """
    if abs(delta) < 1e-6:
        return line

    def _sub(match):
        sec = _clock_seconds(*match.groups()) + delta
        if sec < 0:
            return match.group(0)
        return _fmt_clock(sec)

    return _SHOT_CLOCK_RE.sub(_sub, line)


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
        match = _match_field(stripped)
        if match:
            skipping = match.group("name") == "non_diegetic_music"
            if skipping:
                continue
        if skipping:
            continue
        out.append(line)
    return "\n".join(out)


def _extend_segment_end_clock(fields, gen_seconds, tail_seconds=0.0):
    """把段末时间码延长到**真实生成终点**（17k+5 网格对齐后的 gen_seconds）。

    为什么必须做（2026-09-30 实测）：``_build_segment`` 现在让 ``gen_seconds``
    直接等于网格帧数 / fps，但**提示词正文里的时间码是作者写的名义值** ——
    例如 S02 正文只写到 ``00:06.800``，而本段实际要生成 **175 帧 = 7.292s**。
    模型被告知"本段到 6.8s 结束"却必须吐出 7.292s，多出来的 0.49s 没有任何
    指令 → 它自己填词。v23full 的 ASR 就在 seg_02 的 **6.84–7.22s** 听到了
    剧本外的音节，而该段同时被 seg_03 原样回放。

    修法：把 ``detailed_description`` 里**最大**的那个时间码改写成
    ``gen_seconds``，让「提示词时间轴」与「实际生成帧数」严丝合缝。

    四条安全约束（缺一不可）：
      1. **只看不含 ``<d>`` 的行** —— 台词行里的 ``from X to Y`` 是台词的
         起止窗，改它等于改台词时间轴（那是作者的话）。
      2. 只改**行内非首个**的时间码 —— 行首那个是**镜头起点**（``_cut_times_in``
         认它是切点），把它往后挪等于把切镜搬进尾窗：``validate_segment`` 会报
         「尾窗内有切点」，渲染层也会在接缝处重演一次切镜。行内的第二个才是
         ``From X to Y`` 的终点标记，也就是"本段到几点结束"。
      3. **同一个值出现几次就改几次**。★ 2026-09-30 修正：只改第一处会留下
         两个不同的段末值 —— 实测 S05 同一行里 ``by 00:09.000`` 与
         ``(00:07.400 to 00:09.000)`` 各出现一次，只改前者后该行同时声明
         "到 9.417 结束"和"到 9.000 结束"，比不改更糟。
      4. 差不到一帧（``1/fps``）就什么都不做 —— 已经在终点上，别去碰它。

    另外把尾窗声明 ``(A to B)``（``B`` 等于旧段末值、且 ``B-A`` 约等于一个
    回放窗）整体改写为 ``(gen - tail_seconds to gen)`` —— 那句是模型眼里
    **唯一的"这段尾巴要静音"的显式时间范围**，值必须与实际生成时长同源。

    返回 ``(原值文本, 新值文本)``；没改则返回 ``None``。
    """
    body = fields.get("detailed_description") or ""
    if not body or not gen_seconds:
        return None
    gen = float(gen_seconds)
    try:
        tail = max(0.0, float(tail_seconds or 0.0))
    except (TypeError, ValueError):
        tail = 0.0

    lines = body.splitlines()
    best = None                      # (秒, 行号, 文本)
    for i, line in enumerate(lines):
        if "<d>" in line:
            continue
        marks = list(_SHOT_CLOCK_RE.finditer(line))
        for m in marks:
            if m.start() == marks[0].start():
                continue             # 约束 2：行首 = 镜头起点，不动
            secs = _clock_seconds(*m.groups())
            if best is None or secs > best[0]:
                best = (secs, i, m.group(0))
    if best is None:
        return None
    secs, _idx, text = best
    if gen - secs <= 1.0 / FPS + 1e-6:
        return None                  # 约束 4：已对齐（差不到一帧）

    new_text = _fmt_clock(gen)
    changed = False
    for i, line in enumerate(lines):
        if "<d>" in line:
            continue
        # ★ 顺序要紧：**先**改尾窗声明，再改裸时间码。反过来的话尾窗声明的终点
        #   已经被换成 gen，`(B - A) ≈ tail` 这条判据就不成立了（实测 S05 因此
        #   只改到终点、起点停在 7.400，声明变成 "final 1.6 seconds
        #   (00:07.400 to 00:09.417)" —— 宽度 2.017s，与"1.6 seconds"自相矛盾）。
        if tail > 0:
            def _sub(m):
                a = _clock_seconds(*m.group(1, 2, 3))
                b = _clock_seconds(*m.group(4, 5, 6))
                if abs(b - secs) > 0.02 or abs((b - a) - tail) > 0.06:
                    return m.group(0)
                return "(%s to %s)" % (_fmt_clock(gen - tail), _fmt_clock(gen))
            new_line = re.sub(
                r"\(\s*(\d{1,2}):([0-5]\d)\.(\d{1,3})\s+to\s+"
                r"(\d{1,2}):([0-5]\d)\.(\d{1,3})\s*\)", _sub, line)
            if new_line != line:
                line, changed = new_line, True
        if text in line:
            line = line.replace(text, new_text)      # 约束 3：全量替换
            changed = True
        lines[i] = line
    if not changed:
        return None
    fields["detailed_description"] = "\n".join(lines)
    return text, new_text


def _auto_fix(seg, music_hit=(), music_style=None, is_last=False):
    """能自动补的就补上，并把补了什么记进 issues（自动修过的也算问题提示）。"""
    fixed = []
    if seg.get("hard_cut_source") == "header":
        fixed.append("段头语言槽位声明 hard cut，已按硬切段处理"
                     "（无段首回放、无尾窗；summary 里未写 HARD CUT）")
    fields = seg["fields"]
    # ★★ 2026-09-30：**接续段的任务前缀必须声明 ``video continuation``**。
    #
    #   规范（``h3-prompt-writing/references/ref-en.txt``）：
    #   「``video continuation`` = 新内容从既有源视频**延续 / 接续 / 恢复 / 过渡**」，
    #   多重关系用 `` + `` 连接；SKILL §1.2 要求 summary 以官方任务前缀开头。
    #   本段有 ``replay_in``（段首原样回放上一段结尾）—— 那就是"接续"，前缀必须写
    #   ``[video continuation + reference generation]``。
    #
    #   实测工作流 PACK（v23full）：五段**全部**写 ``[reference generation]``，
    #   而同一行正文却写着 "Continuing … from the replayed opening" —— 前缀与正文
    #   自相矛盾；参考交付稿 v20 对 S02–S05 写的正是
    #   ``[video continuation + reference generation]``。而 S02 恰好是唯一
    #   「计划台词没说出来」的段（ASR：12.41–14.41s 无声）。
    #
    #   这里**只补格式**（前缀 token），summary 正文一个字不动；summary 不在
    #   ``_AUTHOR_TEXT_FIELDS`` 之内，不触作者文本保真闸门。
    _summary = fields.get("summary") or ""
    if (float(seg.get("replay_in") or 0.0) > 0 and not seg.get("hard_cut")
            and _summary.lstrip().startswith("[")):
        _m = re.match(r"\s*\[([^\]]+)\]", _summary)
        _token = (_m.group(1) if _m else "").strip()
        if _token and "video continuation" not in _token.lower():
            fields["summary"] = ("[video continuation + %s]%s"
                                 % (_token, _summary[_m.end():]))
            fixed.append("接续段 summary 前缀 `[%s]` → `[video continuation + %s]`"
                         "（本段段首回放上一段结尾，规范要求声明 video continuation）"
                         % (_token, _token))
    detail = (fields.get("detailed_description") or "").strip()
    if detail:
        first = _sentences(detail)[:1]
        if first and not re.search(r"on-screen text|不出现文字", first[0], re.I):
            detail = (NO_TEXT_EN + " " + detail).strip()
            fields["detailed_description"] = detail
            fixed.append("已自动补 no-text 首句（SKILL §1.1 英文版句子）")

    # 接续段：[Shot 1] 开头原样写回放句（插在 no-text 首句之后）。
    # 判据是「本段开头有没有回放上一段」，不是「本段尾部交不交给下一段」——
    # 用 handoff_seconds 会把首段也算进来，凭空多出一句它根本没有的回放。
    replay_in = float(seg.get("replay_in", seg.get("handoff_seconds")) or 0.0)
    if replay_in > 0 and not seg.get("hard_cut"):
        body = (fields.get("detailed_description") or "").strip()
        if body and not re.search(r"replay|回放", body, re.I):
            replay = _handoff_replay_en(replay_in)
            sentences = _sentences(body)
            if len(sentences) > 1:
                # 只把回放句插到首句后面，其余原文（含换行/空行分段）原样保留
                # —— 用 " ".join 会把整段压成一行，段落节奏全丢。
                rest = body[len(sentences[0]):].lstrip()
                fields["detailed_description"] = "\n".join(
                    [sentences[0], replay, rest]).strip()
            else:
                fields["detailed_description"] = (body + " " + replay).strip()
            fixed.append("已自动补 %s 秒回放句"
                         % _fmt_handoff_seconds(replay_in))

    # ---- 说话纪律：把落在回放区 / 尾窗里的台词挪开（防「重复说话」） ----
    #
    # 回放区（段首 0–replay_in）在下一段开头会被**原样回放**：这里写新台词，
    # 等于同一句在接缝处说两遍。尾窗（最后 1.6s）交给下一段当锚点：这里说
    # 台词，下一段回放时又得再说一遍 —— 成片里就是重复语音。
    #
    # 这两条以前都**只报 issue**：剧本照旧渲、成片照旧重复，用户除了自己改
    # 剧本别无他法。这里改成自动平移整行时间码（台词原文一个字不动），
    # 挪不动的（会撞到段的另一头）才保留 issue 让人决定。
    #
    # ★ 为什么平移而不是删台词：删了剧本就缺一句；平移只是让这句在段内换个
    #   时间说，剧情动作顺序不变，代价最小。
    if not seg.get("hard_cut"):
        body_now = (fields.get("detailed_description") or "").strip()
        if body_now:
            gen_seconds = float(seg.get("gen_seconds") or 0.0)
            # ★ 尾窗起点走本段 ``handoff_seconds``（17k+5 对齐的 1.625s），
            #   不用名义 1.6s：否则平移后台词/切点仍落在真正尾窗（gen-1.625）
            #   之内 0.025s ≈ 1 帧，会被下一段回放成重复语音。
            tail_start = gen_seconds - float(
                seg.get("handoff_seconds") or HANDOFF_SECONDS)
            lines = body_now.split("\n")
            moved = []
            # 上一个**带时间码**的行的起始时间码。用来防止把台词挪到它所属镜头
            # 开始之前 —— 那会让段内时间码倒流（"这句比它所在的镜头还早开
            # 始"），模型收到自相矛盾的指令，比"尾窗有台词"更糟。
            prev_head = None
            for i, ln in enumerate(lines):
                codes = _timecodes(ln)
                if "<d>" not in ln:
                    # ★ 非台词行（切镜 / 运镜标记）同样要平移出回放区：作者写的
                    #   ``At 00:01.600`` 是名义 handoff（1.6s / 38 帧），而真正的
                    #   回放窗是 1.625s / 39 帧；落在回放区 1 帧内的切点会被接缝
                    #   原样重演成单帧闪跳（"叠加 17k+5 帧"问题的镜头层表现）。
                    #   这里与台词走同一套平移逻辑（整行时间码一起挪，镜头说明
                    #   文字不动），只是不检查 prev_head —— 镜头标记不需要保持
                    #   "晚于上一句台词"的约束。
                    inside = [c for c in codes if c < replay_in - 0.001]
                    if replay_in > 0 and inside and min(inside) > 0.0:
                        lc, hc = min(inside), max(inside)
                        shift = replay_in - lc
                        if hc + shift <= gen_seconds + 0.001:
                            # ★ 2026-09-30：只挪**落在回放窗内的**时间码。整行平移
                            #   会把同一个镜头说明里那句 ``… to 00:04.200``（窗外）
                            #   一起推到 4.225，越过下一个镜头的起点 4.200 ——
                            #   v23full 实测 S03 段内时间码因此倒流 1 帧。
                            lines[i] = _shift_clocks_where(ln, shift, replay_in)
                            moved.append("%s %+.3fs（原 %.3f–%.3fs）"
                                         % ("回放区镜头后移", shift, lc, hc))
                            prev_head = min(_timecodes(lines[i]) or (replay_in,))
                            continue
                    if codes:
                        prev_head = min(codes)
                    continue
                if not codes:
                    continue
                lo, hi = min(codes), max(codes)
                delta, why = 0.0, ""
                # ① 回放区：把**落在回放窗内的**时间码挪到窗后
                #   ★ 2026-09-30：以前是整行平移，会把窗外的台词起止窗一起推走，
                #     越过下一个镜头的起点（v23full 实测 S03 4.225 > 4.200、
                #     S05 6.895 > 6.870 —— 段内时间码 1 帧倒流）。只挪窗内的。
                inside = [c for c in codes if c < replay_in - 0.001]
                if replay_in > 0 and inside:
                    shift = replay_in - min(inside)
                    if max(inside) + shift <= gen_seconds + 0.001:
                        delta, why = shift, "回放区台词后移"
                # ② 尾窗：整行往前挪（末段没有下一段，不动）
                if not delta and not is_last and hi > tail_start + 0.001:
                    shift = tail_start - hi
                    if lo + shift >= max(0.0, replay_in) - 0.001:
                        # ★ 2026-09-20：挪完不能早于上一行的起始时间码。挪不动
                        #   就**别挪**，让 validate_segment 把"尾窗有台词"报出来
                        #   —— 报出来是给人看的，倒流是给模型看的，后者危害更大。
                        if prev_head is None or lo + shift >= prev_head - 0.001:
                            delta, why = shift, "尾窗台词前移"
                if delta:
                    lines[i] = (_shift_clocks_where(ln, delta, replay_in)
                                if why == "回放区台词后移"
                                else _shift_line_clocks(ln, delta))
                    moved.append("%s %+.3fs（原 %.3f–%.3fs）"
                                 % (why, delta, lo, hi))
                after = _timecodes(lines[i])
                if after:
                    prev_head = min(after)
            if moved:
                fields["detailed_description"] = "\n".join(lines)
                fixed.append("说话纪律：%s" % "；".join(moved))

    # ---- 台词结束后的留白：同样要显式写成"不再有人声" --------------------
    #
    # 上一段只管「台词落在回放区/尾窗」，``NO_DIALOGUE_EN`` 只管「整段零台词」。
    # 但还有第三种情形两者都盖不住：**段里有台词，但最后一句说完之后到段尾还
    # 空着一大段**（物理隔离后很容易出现——一句短台词独占一段，剩下的时间剧本
    # 没写台词）。模型遇到没有台词指令的空白会自己即兴发声，成片就是剧本外的
    # 含糊音节（2026-09-26 实测：一句 6 字台词独占 7.3s 的段，说完还剩 2.85s，
    # 成片 9.4–10.0s 冒出「我出去 / 五」、11–12s 冒出「对」）。
    #
    # 判据：最后一句台词的结束时刻 → 段尾，间隔超过**一个回放窗**（1.6s）就补。
    #
    # ★ 2026-09-27 收紧：原来是「尾窗 1.6s + 1.0s 缓冲」，实测把真实留白漏掉了。
    #   本机实例（英文单版 PACK，4 段）：
    #     S01 段尾自由 2.30s → 旧阈值 2.60 判「不够大」→ 未注入 → 成片 10.23–11.15s
    #         冒出剧本外发声；
    #     S03 段尾自由 2.50s → 同样未注入 → 成片 39.18–40.93s 冒出剧本外发声。
    #   两处都落在「只在阈值下面一点点」的位置上，说明 1.0s 缓冲不是安全带，
    #   而是把 2.3~2.5s 这类**真实**留白放行了。
    #
    #   为什么可以收到 1.6s：段尾那 1.6s 本来就被规范 §1.4 要求「无台词」（留给
    #   下一段当回放锚点）。也就是说**只要最后一句结束后还剩超过一个回放窗，
    #   那段尾巴就必须静音**——这不是审美判断，是硬结构要求。补纪律句只会把
    #   本该成立的事实说出来，不会把有台词的段落误锁（判据以最后一句的结束
    #   时刻为界，界线之前的台词一字不动）。
    #
    # ★ 2026-09-27 再收紧：阈值从「> 1.6s」改成「> 0.3s」。
    #   实测本机 iter6（真渲染）：S03 的最后一句写到「必须于 00:11.100 前说完，
    #   且不晚于 00:11.600」，`last_end = 11.6`、`gen_seconds = 13.20` ——
    #   间隔 **恰好 1.6s**，被严格大于号排除，纪律句没注入；成片 ASR 在
    #   31.0–32.6s（= S03 的尾窗）听出剧本外发声「哼这獒友们」，而且这段尾窗
    #   被 S04 原样回放，等于同一句错话在成片里出现两次（31.05–34.20s 一段）。
    #   「间隔恰好等于一个回放窗」= 整段剩余时间都是尾窗，恰恰是**最需要**
    #   静音纪律的情形，不是不需要。0.3s 只是防抖，不改变语义。
    #
    # ★ 纪律句**不带时间码**：``_talk_spans`` 按行取 min/max，注入句里出现
    #   ``00:MM.m`` 会撑大该句台词的窗口（见 _TALK_START_RE 处的实测事故）。
    #   用「the last line above ends」指代，锚点由台词自身的时间码提供。
    if not seg.get("hard_cut"):
        body_now = (fields.get("detailed_description") or "").strip()
        if body_now and "<d>" in body_now:
            gen_seconds = float(seg.get("gen_seconds") or 0.0)
            last_end = None
            for ln in body_now.split("\n"):
                if "<d>" not in ln:
                    continue
                codes = _timecodes(ln)
                if codes:
                    e = max(codes)
                    last_end = e if last_end is None else max(last_end, e)
            if (last_end is not None and gen_seconds > last_end
                    and gen_seconds - last_end > TAIL_SILENCE_MIN_GAP
                    and TAIL_SILENCE_EN not in body_now):
                fields["detailed_description"] = (
                    body_now.rstrip() + "\n\n" + TAIL_SILENCE_EN)
                fixed.append("台词后留白 %.2fs，已补段末静音纪律"
                             % (gen_seconds - last_end))

    # ★★ 2026-09-30（第二轮）：段末时间码延长到**真实生成终点**。
    #   `_build_segment` 已把 gen_seconds 对齐到 17k+5 网格（= 渲染真正吐出的
    #   帧数 / fps），但正文里的时间码还是作者写的名义值，比真实终点短
    #   0.23–0.56s。那一段没有任何指令的时间就是模型即兴发声的温床，
    #   而且它整段进成片（拼接只裁段首回放）。这里把它补齐。
    #   硬切段不适用（没有"回放"语义，正文本来就是一个镜头）。
    if not seg.get("hard_cut"):
        moved = _extend_segment_end_clock(
            fields, float(seg.get("gen_seconds") or 0.0),
            tail_seconds=grid_seconds(HANDOFF_SECONDS))
        if moved:
            fixed.append("段末时间码 %s → %s（对齐 17k+5 网格后的真实生成终点，"
                         "消除模型在无指令空白里即兴发声的窗口）"
                         % (moved[0], moved[1]))

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
        elif not re.search(r"ducks to clearly below|压低到明显弱于人声",
                           music_field):
            # 整段式脚本的配乐是**按段切**的（整份塞进每段等于告诉模型"这 6
            # 秒里要演完 40 秒的音乐"）。但规范 4 要求 MUSIC block 全链一致 ——
            # 把硬约束句补在切片后面：具体配乐按段走，静音窗 / duck 规则全片一致。
            fields["non_diegetic_music"] = music_field + "\n" + build_music_block(
                music_field, sfx_only=tuple(music_hit) == ("拟音/音效词",),
                style=music_style)
            fixed.append("已补 MUSIC block 硬约束（配乐切片保留，静音窗与 duck"
                         " 规则全链一致）")
    elif not music_field:
        fields["non_diegetic_music"] = "N/A"
        fixed.append("non_diegetic_music 留空，已填 N/A")

    # 全角标点 → ASCII。英文单版里 ``At 00:07.200，cut to …`` 这种「中文输入法
    # 残留的逗号」非常常见：它不是语言问题（§1.3 管的是汉字），但会让模型把
    # 时间码和后面的镜头说明读成一个整体。删繁就简：能自动归位的就自动归位，
    # 只留一条成功日志，别拿它当 issue 去惊动用户。台词块内的原文一个字不动。
    if _FW_PUNCT_RE.search(
            _DIALOG_RE.sub("",
                           "\n".join(str(fields.get(name) or "")
                                     for name in FIELD_ORDER))):
        for name in FIELD_ORDER:
            value = fields.get(name)
            if isinstance(value, str) and value:
                fields[name] = "".join(
                    part if part.startswith("<d>")
                    else part.translate(_FW_PUNCT_MAP)
                    for part in _DIALOG_SPLIT_RE.split(value))
        fixed.append("全角标点已转 ASCII（台词块内的原文未动）")

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

    # ---- ★ 交付闸门：台词块之外零中文（英文单版是唯一交付语言）------------
    # 上游英译节点已经保证过一遍（``translate_nodes`` → ``strip_residual_cjk``），
    # 这里是**最后一道闸门**：只要还有中文漏进来（老 PACK、手改、双语 PACK 读了
    # 中文侧、英译节点被关掉……），就地剥掉，绝不让它进 H3 提示词。
    #
    # 代价不对称，所以选择「剥」而不是「报」：
    #   · 中文进提示词 → 模型把中文读成台词内容 → 成片**计划外发声 / 与剧本
    #     对不上**（质检技能 §4.4 G 类，本机实测 201 个汉字漏进 5 段里）；
    #   · 剥掉几个字 → 只丢一小截镜头说明，画面影响有限。
    #
    # ★ 判据只看 ``<d>`` 块之外（块内的中文是规范要求的台词原文）。
    # ★ 必须放在全角标点归位**之后**：``，`` 之类的全角标点也在剥离正则的
    #   续接字符类里，先归位再剥离，边界更干净。
    #
    # ★★ 2026-09-27 **本闸门只对"残留中文"生效，不对"作者用中文写的剧本"生效**
    #   （本机真渲染事故，用户报「乱说话、没和剧本对应」的最终根因）：
    #   解析节点是**上游**，中文剧本流到这里时**还没经过英译节点**。旧判据只看
    #   "块外有没有汉字"，于是 `cjk_outside_dialogue()` 对中文剧本必然为真，
    #   本闸门把**整个正文**当残留剥掉 —— 实测 S05 正文原有数千汉字，剥完只剩
    #   **11 个**（就是那句台词本身）与几个孤零零的英文碎片（``OK``、
    #   ``arc curving around``）。模型拿到的手里只有台词块和被撕碎的英文，
    #   只能自行脑补 → 计划外发声、与剧本对不上；``"OK"`` 那个碎片还被
    #   `split_oversized` 从中间切开，配上被 clamp 的时间码，就成了
    #   ``At 00:03.200, 00:03.200 OK 00:03.200) , S2)``。
    #
    #   正确的分工：**剥离是英译节点的职责，且只作用于它自己的输出**。解析节点
    #   要么拿到已英译的文本（此时残留确实是零星几个字），要么拿到原始中文
    #   （此时应当原样交给下游英译节点）。判据因此改成「**残留性**」——
    #   只有当汉字在块外**零散且占比很低**时才算残留；成片中文（占比高或出现
    #   长句）一律放行。
    #
    #   阈值取 0.15：真残留是 `Project / 项目：曹贼的性价比` 这种标签里的一两个
    #   词（占比远低于此），而作者写的中文正文占比通常在 0.5 以上。中间留足
    #   安全带，不靠运气。
    gate = "\n".join(str(fields.get(name) or "") for name in FIELD_ORDER)
    if _residual_cjk_only(gate):
        total = 0
        for name in FIELD_ORDER:
            value = fields.get(name)
            if (isinstance(value, str) and value
                    and _residual_cjk_only(value)):
                cleaned, n = strip_residual_cjk(value)
                fields[name] = cleaned
                total += n
        if total:
            fixed.append(
                "台词块之外的中文已剥离 %d 处（英文单版不允许出现中文：模型会把"
                "中文读成台词内容，成片就是剧本外发声）" % total)
    return fixed


_RESIDUAL_CJK_RATIO = 0.15


def _residual_cjk_only(text):
    """块外中文是否**只是残留**（零散几个字），而不是作者用中文写的正文。

    ``True``  → 交给 ``strip_residual_cjk`` 剥掉（英文单版的收尾闸门）。
    ``False`` → 放行（中文剧本，或已含成段中文，应由英译节点处理）。

    两个判据，任一命中即判定为「不是残留」：

    1. **占比**：块外汉字数 / 正文字符数 > ``_RESIDUAL_CJK_RATIO``；
    2. **长句**：块外出现 ≥8 个连续汉字的片段（成句的中文，不可能是残留）。

    第 2 条兜住第 1 条的边界情形：极短的正文里夹了一整句中文（占比可能不高，
    但语义上就是作者的话）。两条都是"有疑问就放行"，绝不误剥作者文本。
    """
    body = _DIALOG_RE.sub("", str(text or ""))
    n_cjk = len(_CJK_RE.findall(body))
    if n_cjk == 0:
        return False
    n_all = len(re.sub(r"\s", "", body))
    if n_all and float(n_cjk) / float(n_all) > _RESIDUAL_CJK_RATIO:
        return False
    if re.search(r"[\u4e00-\u9fff]{8,}", body):
        return False
    return True


# ---------------------------------------------------------------------------
# 跨段一致性（规范 4：全链逐字复用）
# ---------------------------------------------------------------------------
# ★★ 2026-09-27 新增：**作者文本保真闸门**。
#
#   背景（用户裁决）：「源剧本 人物服装场景是和参考图一致，禁止私自更换描述」。
#   但节点链上有好几处会"顺手改一下作者的文字"：
#     · ``_seg_retention`` 的兜底句 —— 曾把作者写的
#       "The appearance of Subject N is consistent with the reference image
#        in all shots." 换成 "retained as the appearance reference for
#        <Subject N>."，**把"生成结果要与参考图一致"这条断言抹掉了**；
#     · ``wardrobe`` 的转换词（``collar → neckline`` …，只在有穿衣动作的段生效）；
#     · ``_auto_fix`` 的格式化（删引号 / 全角转半角 / 补规范句）。
#   改一处看不出问题，但**只在某一段改**就会让四段提示词分叉 ——
#   实测 iter6 成片因此同一角色跨段换三套衣服（金 → 粉 → 绿），
#   而参考图 `<Picture 2>` 是深红纱袍。
#
#   这条闸门把「作者写的话还在不在」变成**可见的 issue**：
#   作者写的每一行（去掉引号 / 全角 / ``appears in (…)`` 之后）必须能在
#   **每一段**的同名字段里原样找到。找不到 = 被节点换成了别的话。
#   新增的改写会立刻暴露，不必等出片。
#
#   只查这两段：``subject_definitions`` / ``retention_analysis``
#   —— 它们是**外观锁定**的载体，也是用户点名的那四条
#   （``<Picture N>: fully_preserved - … 外观与参考图一致``）。
#   不查 ``detailed_description``（按段切片）、``overall_soundscape`` /
#   ``non_diegetic_music``（按时间窗切片）、``summary``（任务前缀与镜头范围
#   都按段重算，整段比对必然误报；它的 STYLE / MUSIC 一致性由
#   ``validate_chain`` 单独把关）。
_AUTHOR_TEXT_FIELDS = ("subject_definitions", "retention_analysis")


def _norm_author_line(line):
    """归一化作者行：去引号、全角转半角、抹掉 ``appears in (…)``、压空白、去句末标点。

    归一化掉的这几项都是**允许**的改动，不算"改写描述"：
      · 引号 —— 规范 §1.1 禁引号（画面出现文字的头号诱因）；
      · 全角标点 —— ``_auto_fix`` 统一转半角；
      · ``appears in (…)`` —— 按段收窄镜头号；
      · **句末标点** —— 节点在作者行尾追加规范句时会把句末 ``.`` 改成 ``;``
        （``…with S2).`` → ``…with S2); never speaks, no dialogue.``）。
        不剥掉它，"作者行是节点行的前缀"这个判据就会误报 —— 实测首版闸门
        在本机剧本上误报 10 条，全是这一条引起的。
    """
    s = str(line or "")
    s = _QUOTE_RE.sub("", s)
    s = s.translate(_FW_PUNCT_MAP)
    s = re.sub(r"\(\s*appears\s+in[^)]*\)", "", s, flags=re.I)
    # 场景图的 ``([Shot 1] first frame)`` 结构标注同理：作者只对 Subject 写了
    # ``appears in``，场景是"首帧定住"由节点补的标注，不是描述改写。
    s = re.sub(r"\(\s*\[Shot\s*\d+\][^)]*\)", "", s, flags=re.I)
    # ``<Subject 2> :`` 与 ``<Subject 2>:`` 是同一件事（作者手写时空格随意）。
    # 只收紧 ``>`` 后面的那个冒号 —— 不能用宽松的 ``\s*:\s*``，
    # 那会把时间码 ``00:03.000`` 拆成 ``00: 03.000``。
    s = re.sub(r">\s*:\s*", ">: ", s)
    # ``summary`` 的**任务前缀**是按段生成的（首段 ``[reference generation]``、
    # 接续段 ``[video continuation + reference generation]``），属于规范要求，
    # 不是改写描述 —— 比较时把行首那个方括号标签剥掉。
    s = re.sub(r"^\s*\[[^\]]*\]\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(".;,。；， ")


def author_text_drift(globals_fields, segments):
    """交付闸门：作者写的描述文本有没有被节点改掉。返回 issue 列表（空 = 全部保真）。"""
    issues = []
    for field in _AUTHOR_TEXT_FIELDS:
        authored = [x for x in str(globals_fields.get(field) or "").splitlines()
                    if x.strip()]
        if not authored:
            continue
        for seg in segments:
            body = _norm_author_line(seg["fields"].get(field) or "")
            for line in authored:
                probe = _norm_author_line(line)
                # 太短的行（``N/A`` / 空标题）没有判别力，跳过
                if len(probe) < 12:
                    continue
                if probe not in body:
                    issues.append(
                        "%s：%s 里作者写的这一行**没被原样保留** ——「%s…」。"
                        "节点只能做格式规范化（删引号 / 全角转半角 / 收窄 "
                        "appears in），**不得改写描述文本**；"
                        "请检查是不是有兜底句把它替换掉了。"
                        % (seg["id"], field, probe[:72]))
    return issues


def validate_chain(segments):
    """no-text 句 / STYLE 块 / MUSIC 块必须全片逐字一致。"""
    issues = []
    # ★ 风格词 vs 参考图锁定 的冲突检测放这里：它是**全片级**的，
    #   放在 validate_segment 里会每段重复报一遍。
    issues.extend(_style_vs_reference_conflicts(segments))
    # 注：台词句引用检测（``_quoted_line_refs``）在 ``parse_pack`` 入口调用 ——
    # 它需要**原始文本**（引号还在），见那里的注释。
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

    ★ 返回值一律**已对齐 17k+5 网格**（``grid_seconds``）：规范写的 1.6s =
      38 帧不在网格上，向上取是 1.625s = 39 帧 —— 与 ``director._shot_plan``
      的 ``guide_length_up`` 同一口径。对齐放在这里（而不是等下游去猜），
      是为了让 ``to_shots_json`` 写出的 ``replay_in`` / ``gen_seconds`` 与
      真正渲染用的帧数**逐帧一致**，否则 storyboard 里那句
      "gen_seconds=9.0 / replay_in=1.6（=38 帧）" 会和实际生成的 39 帧对不上，
      缓存键也跟着漂。
    """
    total = len(segments or [])
    seg = (segments or [])[index] if 0 <= index < total else {}
    if seg.get("hard_cut"):
        return 0.0, 0.0
    window = grid_seconds(HANDOFF_SECONDS)
    replay_in = 0.0 if index <= 0 else window
    tail_out = 0.0 if index >= total - 1 else window
    return replay_in, tail_out


def to_shots_json(segments, session_name="", include_task=True):
    """转成 H3Director 认识的 shots_info。"""
    shots = []
    total = len(segments)
    for index, seg in enumerate(segments):
        shot = {
            "id": seg["id"],
            # ★ 出口处再幂等跑一遍防字幕逐句注入（_inject_dialogue_no_text）：
            #   即使 fields 来自旧版解析产物，喂给 Director 的每句台词也一定
            #   带着「只出声、严禁上屏」禁令（2026-09-30 烧字幕兜底）。
            #
            # ★★ 2026-09-30 第二轮排查撤销 `_strip_dialogue_lang_tags`：
            #   上游 `_tag_speakers`（prompt_pack.py:1603）刚按规范把台词块补成
            #   `(Sx) <d>[语言] 原文</d>`（那里的注释写着"缺一不可"），这里却把
            #   `[语言]` 又剥掉 —— 同一个模块自相矛盾。规范（SKILL §1.3）与
            #   参考交付稿 `H3_曹贼的性价比_分镜提示词_v20_en.txt`（5 处
            #   `<d>[Chinese] …</d>`）都要求保留语言标签：它是模型判断"这句用
            #   哪种语言念"的**唯一**依据。英文单版的正文 95% 是英文，剥掉标签
            #   后模型只能靠猜 —— v23full 实测 S02 的 `汝与曹贼何异？` 被念成
            #   无法对齐剧本的短促音节（ASR: "记事我 / 是觉"），而保留了标签的
            #   S01/S03 中文台词基本可辨。防字幕已由逐句 AUDIO-ONLY 禁令承担，
            #   不需要再牺牲语言标签。
            "shot": _inject_dialogue_no_text(
                segment_prompt(seg["fields"])
            ) + "\n" + SUBTITLE_BAN_EN,
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
    EN_PACK_LAYOUT: "英文单版 PACK（SKILL §5 现行规范）",
    "pack-single": "带头部单版（早期形态）",
    "en-single": "英文单版（无头部）",
    "bilingual-pack": "双语 PACK（旧格式，只取英文侧）",
    SHOT_SCRIPT_LAYOUT: "[Shot N] 分镜脚本（沿镜头边界自动拆段）",
}

# 头部里这些键值得在报告里回显
_HEADER_ECHO = (
    ("Mode", "模式"), ("Aspect", "画幅"), ("Version", "版本"), ("Date", "日期"),
)


def format_report(segments, meta, issues, music_hit=(), fixes=None):
    """解析报告：头部信息 + 每段一行 + **自动修复清单** + 问题清单。

    ``fixes`` 是节点自动修过的项（成功日志，**不是问题**），单独一节「已自动修
    复」展示，与 issues 的「注意」严格分开。以前混在一起时用户分不清哪条是 bug
    哪条是修好的。"""
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

    if fixes:
        out.append("")
        out.append("已自动修复（节点代改，下方对应的就是改后的正文，不是问题）：")
        for item in fixes[:30]:
            out.append("  ✓ " + item)

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
