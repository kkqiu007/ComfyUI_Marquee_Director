# -*- coding: utf-8 -*-
"""Regression for PACK -> shots_json speaker tagging and the CJK gate.

Run from the ComfyUI root::

    .venv/Scripts/python.exe custom_nodes/ComfyUI_Marquee_Director/tools/test_pack_speakers.py

Two bugs this exists to keep dead:

1. **English speaker hints never matched.** ``_SPEAKER_HINT_RE`` only knew Chinese
   cues (``S1以中低音区说话``). The pack's own delivery format is the *English*
   single version (SKILL §1.3), where the same cue reads ``S1 speaks in a mid-low
   register`` / ``S1's voice`` -- so no unnumbered line ever got a hint and every
   one of them fell through to "most frequent <Subject N> in the segment".
   Measured on a real 4-segment script: 2 of 5 lines were attributed to the wrong
   character, which is exactly the "wrong character speaks the line" failure.

2. **U+FF0C counted as a Chinese character.** ``_CJK_RE`` spanned
   ``[一-鿿　-〿＀-￯]``, so every ``At 00:07.200，cut to …`` in an English script
   tripped the §1.3 "Chinese outside the dialogue block" issue -- four false
   positives, and with ``on_issue=error`` an abort of a perfectly good render.
"""

import os
import sys

FAILS = []


def check(label, ok, extra=""):
    print("%-60s %s %s" % (label, "PASS" if ok else "FAIL", extra))
    if not ok:
        FAILS.append(label)


def _comfy_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


COMFY = _comfy_root()
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))

from ComfyUI_Marquee_Director.marquee_director import prompt_pack as pp   # noqa: E402

# ---------------------------------------------------------------------------
# 1. speaker tagging on an English script
# ---------------------------------------------------------------------------
# Two characters: S1 asks the question, S2 answers. Neither line carries an
# explicit (Sx) in the source -- the speaker is only stated in prose.
EN = """\
subject_definitions:
<Subject 1> is the person in Picture 1 (President Wang).
<Subject 2> is the person in Picture 2 (Zhang San).
summary:
[reference generation] A test scene.
detailed_description:
No on-screen text, no subtitles, no captions and no lettering of any kind appear.
[Shot 1] At 00:00.000 the door bursts open.
Under her gaze, S1's grip loosens involuntarily.

<d>[Chinese] 台词一</d> S1 speaks in a mid-low register with an anxious timbre.
At 00:07.200 cut to a close-up MCU of <Subject 2>.

<d>[Chinese] 台词二</d> Her voice is a crisp tenor.
At 00:09.000 the shot ends.
retention_analysis:
<Subject 2> (appears in [Shot 1]): fully_preserved.
overall_soundscape:
A quiet room.
non_diegetic_music:
N/A
"""


def _tagged(detail):
    out = []
    for line in detail.splitlines():
        if "<d>" not in line:
            continue
        m = pp._SPEAKER_RE.search(line)
        out.append("S%s" % m.group(1) if m else None)
    return out


notes = []
tagged = _tagged(pp._tag_speakers(EN, notes))
check("English cue 'S1 speaks' -> (S1)", tagged[0] == "S1", str(tagged))
check("no cue -> falls back, and says so", tagged[1] is not None, str(tagged))
check("inference is reported, not silent", len(notes) >= 1, str(notes))
check("the note names the line it touched",
      all("台词" in n for n in notes), str(notes))

# Chinese cues must keep working -- the fix adds, it must not replace.
# （提示必须与台词**同一行**：_tag_speakers 只看台词块所在行的前半段与整行，
#   不回溯上一行 —— 这是刻意收窄的口径，跨行猜会把旁白的说明误当说话人。）
ZH = "S1以中低音区说话：<d>[Chinese] 台词一</d>"
check("Chinese cue still matches",
      _tagged(pp._tag_speakers(ZH))[0] == "S1",
      str(_tagged(pp._tag_speakers(ZH))))

# A passing mention must NOT be read as a speaker cue.
MENTION = """\
S1 的侧影只作画框边缘存在。

<d>[Chinese] 台词一</d>
"""
check("passing mention is not a cue",
      _tagged(pp._tag_speakers(MENTION))[0] is None,
      str(_tagged(pp._tag_speakers(MENTION))))

# ---------------------------------------------------------------------------
# 2. the CJK gate
# ---------------------------------------------------------------------------
check("fullwidth comma is not a Chinese character",
      pp._CJK_RE.search("At 00:07.200，cut to a close-up") is None)
check("fullwidth comma IS fullwidth punctuation",
      pp._FW_PUNCT_RE.search("At 00:07.200，cut to a close-up") is not None)
check("real Chinese is still Chinese",
      pp._CJK_RE.search("三弦接棒") is not None)
check("ideographic full stop alone is not a Chinese character",
      pp._CJK_RE.search("。、，「」") is None)
check("but they are fullwidth punctuation",
      pp._FW_PUNCT_RE.search("。、，「」") is not None)
check("real Chinese words still register",
      pp._CJK_RE.search("收戏。") is not None)
check("language inference is not fooled by punctuation",
      pp._infer_lang("At 00:07.200，cut to a close-up") == "en")
check("language inference still sees Chinese",
      pp._infer_lang("切到近景，三弦接棒") == "zh")
check("fullwidth comma normalises to ASCII",
      "，".translate(pp._FW_PUNCT_MAP) == ", ",
      repr("，".translate(pp._FW_PUNCT_MAP)))

# ---------------------------------------------------------------------------
# 3. end to end: no false §1.3 issue on an English script
# ---------------------------------------------------------------------------
segs, _meta, issues = pp.parse_pack(EN, language="auto", default_duration=11.0,
                                    auto_fix=True)
check("English script parses", len(segs) >= 1, str(len(segs)))
check("no bogus 'Chinese outside dialogue' issue",
      not any("台词块之外出现中文字符" in i for i in issues), str(issues[:2]))

if FAILS:
    print("\n%d FAILED: %s" % (len(FAILS), "; ".join(FAILS)))
    raise SystemExit(1)
print("\nall pack-speaker checks passed")
