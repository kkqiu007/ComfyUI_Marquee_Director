# -*- coding: utf-8 -*-
"""渲染后防字幕质检（2026-09-30，通用兜底层）。

为什么存在：提示词层的防字幕禁令（段首 NO_TEXT_EN、段尾 SUBTITLE_BAN_EN、
逐句 DIALOGUE_NO_TEXT_EN）只能**降概率** —— 09-30 实测同结构 5 段仍随机烧出
2 段逐行中文字幕（H3 把 <d> 台词当字幕画进画面）。这里做**确定性兜底**：
每段渲完立刻 OCR 扫底部字幕带，烧了字就由调用方换种子重渲该段，直到干净
或次数用尽。与剧本语言/内容无关 —— 换任何脚本都自动生效。

检测器优先真 OCR（rapidocr-onnxruntime，装在 ComfyUI 的 standalone-env）；
库缺失时**不质检**（返回空命中），质检绝不能连累渲染主链路。

环境变量：
- ``MARQUEE_SUBTITLE_QC``    重渲上限，默认 3（2026-09-30 由 2 校准上调）；
  ``0`` = 整个质检关闭。注意这是**重渲**次数，首渲之外最多再试 3 次。
- ``MARQUEE_SUBTITLE_QC_STEP`` 取样步长（帧），默认 3（8Hz）。
"""

import logging
import os
import re

log = logging.getLogger("marquee.subtitle_qc")

_TEXT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
# 逐字禁令挂进 prompt 后，成片正文里唯一合法的"文字"就是台词配音 —— 画面上
# 出现的任何 CJK/字母数字串都算烧字。置信度与持久度门槛滤掉织物高光误报
# （09-30 校准：真字幕 conf>=0.98 且持续秒级，噪声 conf<=0.77 且单样本）。
_MIN_CONF = 0.70
_MIN_LEN = 2
_PERSIST_S = 0.6          # 同/近似文本须在该窗口内出现 >=2 个样本
_BAND = 0.42              # 底部字幕带高度占比（真字幕在 y≈0.69-0.77，留余量）


def max_attempts():
    # ★ 2026-09-30 实战校准：默认 2→3。《曹贼的性价比》S02 同一句台词
    #   三个种子连续烧字（326991/431720/641178）——对该段近乎确定，样本
    #   多一次是一次的期望。环境变量仍可覆盖（0 = 关闭质检）。
    try:
        return max(0, int(os.environ.get("MARQUEE_SUBTITLE_QC", "3")))
    except (TypeError, ValueError):
        return 3


def _step():
    try:
        return max(1, int(os.environ.get("MARQUEE_SUBTITLE_QC_STEP", "3")))
    except (TypeError, ValueError):
        return 3


# 重试用的**逐级强化禁令**（2026-09-30）：S02 实测同一句台词连烧
# —— 加一条更重的禁令 + 换种子一起上，双保险。
# ⚠️ 2026-09-30 第四轮更正：旧注释写「该失败由提示词诱发，纯换种子去相关不够」
#    —— **这句话是错的**。v25sub 实测：换种子 + 强化禁令后**第 1 次重渲即通过**
#    （manifest `seed_qc: True`，重渲后 OCR 0 命中）。烧字幕是**随机事件**，
#    换种子是有效的，强化禁令只是顺带。所以别把重试次数调得很高去赌"提示词问题"。
# 措辞只加强禁令、不改动构图/剧情描述（保持段间连续性）。
RETRY_VARIANTS = (
    "RENDER NOTE (retry 1): the previous take of this shot was rejected because "
    "typeset text appeared over the picture. The picture must contain zero "
    "lettering - no writing on paper, walls, props, screens or overlays of any "
    "kind; speech is sound only.",
    "RENDER NOTE (retry 2): absolute picture rule - no subtitles, no captions, "
    "no on-screen lyrics, no karaoke text, no watermarks in any shot of this "
    "segment. Show only faces, bodies, costumes and the room; any character, "
    "glyph or caption in the frame is a defect.",
    "RENDER NOTE (retry 3): positive framing - the spoken line is conversational "
    "sound exchanged between the characters on screen: the speaker looks at the "
    "other character and never at the viewer, the viewer is never addressed, "
    "and the frame contains nothing but the physical scene exactly as the "
    "reference images show it - zero lettering anywhere.",
)


def retry_variant(attempt):
    """第 ``attempt`` 次（从 1 起）重渲要追加的强化禁令；超出列表返回空串。"""
    if 1 <= attempt <= len(RETRY_VARIANTS):
        return RETRY_VARIANTS[attempt - 1]
    return ""


def _ocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as e:  # 缺库/坏安装：质检降级为不扫描
        log.info("subtitle QC: rapidocr 不可用（%s），跳过扫描", e)
        return None
    try:
        return RapidOCR()
    except Exception as e:
        log.info("subtitle QC: RapidOCR 初始化失败（%s），跳过扫描", e)
        return None


def scan(path):
    """扫描一个视频的底部字幕带，返回烧字命中 ``[{t, text, conf}]``。

    只读不写、异常自吞（宁可放过不可误伤渲染）。取样默认 8Hz，
    单段 175-311 帧约 60-100 次 OCR ≈ 10-20s，相对 7-13 分钟的渲染可忽略。
    """
    if not path or not os.path.exists(path):
        return []
    ocr = _ocr()
    if ocr is None:
        return []
    try:
        import av
        import numpy as np
    except Exception as e:
        log.info("subtitle QC: av/numpy 不可用（%s），跳过扫描", e)
        return []

    frames = []
    try:
        with av.open(path) as box:
            stream = box.streams.video[0]
            stream.thread_type = "AUTO"
            step = _step()
            for i, frame in enumerate(box.decode(stream)):
                if i % step:
                    continue
                h, w = frame.height, frame.width
                y0 = int(h * (1.0 - _BAND))
                img = frame.to_ndarray(format="bgr24")[y0:h, 0:w]
                frames.append(img)
    except Exception as e:
        log.info("subtitle QC: 解码 %s 失败（%s），跳过扫描", path, e)
        return []

    seen = []  # (t, text, conf)
    for i, img in enumerate(frames):
        try:
            res, _ = ocr(img)
        except Exception:
            continue
        if not res:
            continue
        txt = "".join(str(r[1]) for r in res).strip()
        try:
            conf = min(float(r[2]) for r in res)
        except (TypeError, ValueError):
            conf = 0.0
        if conf >= _MIN_CONF and len(txt) >= _MIN_LEN and _TEXT_RE.search(txt):
            seen.append((i * _step() / 24.0, txt, round(conf, 2)))

    hits = []
    for j, (t, txt, c) in enumerate(seen):
        others = seen[:j] + seen[j + 1:]
        near = any(abs(t - t2) <= _PERSIST_S and
                   (txt == x2 or x2 in txt or txt in x2)
                   for t2, x2, _ in others)
        if not near:
            continue
        # 同一文本只报第一次（持续多帧的字幕去重，日志一条一句）
        if any(h["text"] == txt or h["text"] in txt or txt in h["text"] for h in hits):
            continue
        hits.append({"t": round(t, 2), "text": txt, "conf": c})
    return hits
