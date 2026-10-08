# -*- coding: utf-8 -*-
"""Offline unit tests for marquee_director.prompt_pack.

Run from repo root with any Python that has the standard library:
    python tools/test_prompt_pack.py

★ 2026-09-20：固件已从**双语/中文**迁移到 **EN 单版**（RULE_VERSION
``en-single-v3``）。旧固件喂的是「[1] 中文版」双语 PACK，而解析器现在只认
英文单版 —— 除 ``<d>[Chinese] 原文</d>`` 台词块外出现任何中文都会报 issue，
``pack_text`` 也只出 ``/ EN`` 段头。三个用例因此红了很久，根因是固件停在旧规范，
不是解析器坏了。

写固件要守的两条硬规则（都在 prompt_pack 的校验里）：
  * 中文只允许出现在 ``<d>…</d>`` 台词块内；
  * 每段 ``summary`` 必须以官方任务前缀 ``[xxx]`` 开头。
"""
from __future__ import annotations

import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "marquee_director"))

from prompt_pack import (  # noqa: E402
    MUSIC_BLOCK_TEMPLATE,
    parse_pack,
    to_shots_json,
    pack_text,
    pack_dur_spec,
    grid_seconds,
    HANDOFF_SECONDS,
    MAX_NEW_SECONDS,
    MIN_NEW_SECONDS,
    GRID_SLACK_SECONDS,
)

# 写作提示 ≠ 结构性问题。合成固件不可能也不该满足这些「动笔写」的规范：
#   * 缺少 PACK 头部 —— 整段式 [Shot N] 脚本天生没有头部；
#   * 英文词数 —— 350–500 词是给**写提示词的人**的区间，解析器 tolerant 下限
#     是 200 词（EN_WORDS_TOLERANCE），合成固件凑够它只会让断言测的是长度
#     而不是解析。
# 真正的结构性问题（中文混入、缺任务前缀、碎片段、超长段）仍然必须为零。
WRITING_NOTES = ("缺少 PACK 头部", "英文词数")


def structural(issues):
    return [i for i in issues if not any(n in i for n in WRITING_NOTES)]


# The MUSIC block comes from the module, not from a copy in this file.
# It used to be copied verbatim, which made the fixture a second source
# of truth: changing the template then broke two tests that were only
# asserting the old wording back at itself.
_MUSIC_EN = MUSIC_BLOCK_TEMPLATE.format(style="restrained ambient")

SAMPLE_PACK = f"""\
==========================================================
MiniMax H3 SHOT PROMPT PACK / H3 分镜提示词包
Project / 项目         : 曹贼的性价比
Mode / 模式            : Ref2VA (full reference)
Total duration / 总时长 : 00:23
Segments / 段数        : 2
Aspect / 画幅          : 16:9 / 2K / 24fps
Music / 配乐           : Per MUSIC block
Version / 版本         : v1
Date / 日期            : 2026-09-14
==========================================================

==========================================================
ENGLISH VERSION / 英文版（本文件唯一语言版本）
==========================================================


########## S01 / 11s / EN ##########
subject_definitions:
<Subject 1> is the person in <Picture 1> (Wang, S1): his appearance, clothing, build and face are taken solely from <Picture 1> and never change.
<Subject 2> is the person in <Picture 2> (Zhang, S2): his appearance, clothing, build and face are taken solely from <Picture 2> and never change.
<Picture 1> is the appearance reference for <Subject 1>, identical in every frame.
<Picture 2> is the appearance reference for <Subject 2>, identical in every frame.
<Picture 3> is the bedroom scene reference; the set, props, colour temperature and light direction are strictly identical to it and nothing new is added.
summary:
[reference generation] A night scene opens in an ancient bedroom.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identical to <Picture 1>.
<Subject 2> (appears in [Shot 1]): fully_preserved - identical to <Picture 2>.
<Picture 3> ([Shot 1] first frame): fully_preserved - the scene is kept whole.
detailed_description:
No text, subtitles, captions, timecodes, watermarks or any graphic overlay appear anywhere in the frame at any time.
STYLE: naturalistic period drama, handheld medium shot, visible film grain, shallow depth of field, no stylisation.
CAMERA: static 35mm medium shot at eye level, locked off, no push, no pull, no pan, no whip.
LIGHTING: a single warm candle off frame left, deep unlit shadow to the right, no fill light, no motivated second source.
[Shot 1] At 00:00.000, <Subject 1> stands in the doorway with his shoulders squared and stops there.
At 00:02.000, (S1) <d>[Chinese] 汝与曹贼何异？</d>
At 00:06.000, <Subject 2> rises from the bed edge and turns his head toward the doorway.
At 00:09.000, the candle flame bends as the door settles and neither man moves again.
overall_soundscape:
Distant watch drums, a candle wick splitting, cloth shifting.
non_diegetic_music:
{_MUSIC_EN}

########## S02 / 11s+1.6=12.6 / EN ##########
subject_definitions:
<Subject 1> is the person in <Picture 1> (Wang, S1): his appearance, clothing, build and face are taken solely from <Picture 1> and never change.
<Subject 2> is the person in <Picture 2> (Zhang, S2): his appearance, clothing, build and face are taken solely from <Picture 2> and never change.
<Picture 1> is the appearance reference for <Subject 1>, identical in every frame.
<Picture 2> is the appearance reference for <Subject 2>, identical in every frame.
<Picture 3> is the bedroom scene reference; the set, props, colour temperature and light direction are strictly identical to it and nothing new is added.
summary:
[video continuation + reference generation] This segment continues the previous take.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identical to <Picture 1>.
<Subject 2> (appears in [Shot 1]): fully_preserved - identical to <Picture 2>.
<Picture 3> ([Shot 1] first frame): fully_preserved - the scene is kept whole.
detailed_description:
No text, subtitles, captions, timecodes, watermarks or any graphic overlay appear anywhere in the frame at any time.
STYLE: naturalistic period drama, handheld medium shot, visible film grain, shallow depth of field, no stylisation.
CAMERA: static 35mm medium shot at eye level, locked off, no push, no pull, no pan, no whip.
LIGHTING: a single warm candle off frame left, deep unlit shadow to the right, no fill light, no motivated second source.
[Shot 1] At 00:00.000, this segment opens on an exact replay of the closing 1.6 seconds of the preceding take and carries that motion straight through.
At 00:02.000, (S2) <d>[Chinese] 丞相给的实在是多。</d>
At 00:08.000, <Subject 1> freezes and his jaw tightens without turning away.
At 00:10.000, the candle guttering throws both faces into shadow and the beat holds.
overall_soundscape:
Distant watch drums receding, a candle wick splitting.
non_diegetic_music:
{_MUSIC_EN}

==========================================================
END OF PACK / 文件结束
==========================================================
"""


class TestPromptPack(unittest.TestCase):
    def _parse(self):
        return parse_pack(SAMPLE_PACK, language="auto",
                          default_duration=11.0, auto_fix=True)

    def test_parse_en_pack(self):
        segments, meta, issues = self._parse()
        self.assertEqual(len(segments), 2)
        # 现行交付形态：PACK 头部 + ENGLISH VERSION + EN 段。
        # "bilingual-pack" 是**旧格式**（解析器只取英文侧），不再是我们要的。
        self.assertEqual(meta.get("layout"), "en-pack", meta)
        self.assertEqual(len(structural(issues)), 0, structural(issues))

    def test_segment_durations(self):
        segments, _meta, _issues = self._parse()
        # ★ 2026-09-30：`_build_segment` 现在把 gen 对齐到 H3 的 **17k+5 帧网格**，
        #   `new` 取**真实贡献**（网格帧 − 回放帧）。所以「11s 的段」不再等于
        #   11.0s —— 264 帧不在网格上（网格是 …243 / 260 / 277…），向上取是
        #   277 帧 = 11.542s。以前解析层报 11.0、渲染层真生成 11.542，两边差
        #   0.54s 无剧本时间 → 模型即兴发声（"乱说话"）+ 成片比 PACK 头部写的
        #   总时长长出近 2s。判据直接走生产函数 ``grid_seconds()``，别在本文件
        #   里再手搓一份 —— 两份实现会各自漂移。
        handoff = grid_seconds(HANDOFF_SECONDS)
        self.assertEqual(segments[0]["new_seconds"], round(grid_seconds(11.0), 3))
        self.assertEqual(segments[0]["gen_seconds"], round(grid_seconds(11.0), 3))
        self.assertEqual(segments[0]["replay_in"], 0.0)
        self.assertEqual(segments[1]["gen_seconds"],
                         round(grid_seconds(11.0 + handoff), 3))
        self.assertEqual(segments[1]["new_seconds"],
                         round(grid_seconds(11.0 + handoff) - handoff, 3))
        self.assertEqual(segments[1]["replay_in"], handoff)

    def test_to_shots_json(self):
        segments, _meta, _issues = self._parse()
        board = to_shots_json(segments, session_name="test")
        self.assertIn("session_name", board)
        self.assertEqual(len(board["shots_info"]), 2)
        handoff = grid_seconds(HANDOFF_SECONDS)
        first = board["shots_info"][0]
        self.assertEqual(first["duration"], round(grid_seconds(11.0), 3))
        # 回放窗是**网格对齐后**的值（1.625），不是规范名义值 1.6。
        self.assertEqual(first["handoff_seconds"], handoff)
        self.assertEqual(first["replay_in"], 0.0)
        self.assertEqual(first["tail_out"], handoff)
        second = board["shots_info"][1]
        self.assertEqual(second["duration"],
                         round(grid_seconds(11.0 + handoff) - handoff, 3))
        # last segment has no tail to hand off
        self.assertEqual(second["handoff_seconds"], 0.0)
        self.assertEqual(second["replay_in"], handoff)
        self.assertEqual(second["tail_out"], 0.0)

    def test_no_text_first_sentence(self):
        segments, _meta, _issues = self._parse()
        for seg in segments:
            detail = seg["fields"].get("detailed_description") or ""
            first = re.split(r"(?<=[。！？.!?])\s+", detail.strip())[0]
            self.assertIn("text", (first or "").lower())

    def test_speaker_numbers(self):
        segments, _meta, _issues = self._parse()
        for seg in segments:
            detail = seg["fields"].get("detailed_description") or ""
            for line in detail.splitlines():
                if "<d>" in line:
                    self.assertRegex(line, r"\(S\d+\)")

    def test_pack_text_roundtrip(self):
        segments, meta, _issues = self._parse()
        text = pack_text(segments, meta, version="v1", date="2026-09-14",
                         project="曹贼的性价比", mode="Ref2VA (full reference)",
                         aspect="16:9 / 2K / 24fps")
        self.assertIn("Project / 项目", text)
        # EN 单版：段头是 / EN，不再是 / 中文
        # ★ 2026-09-30：段头时长是**网格对齐后**的生成值（11.0s → 277 帧 =
        #   11.542s），与渲染真正吐出的帧数逐帧一致。
        self.assertIn("########## S01 / %gs / EN ##########"
                      % round(grid_seconds(11.0), 3), text)
        # ★ 2026-09-27：段头里的回放窗是**网格对齐后**的 1.625（不是名义 1.6），
        #   与 gen 一致。旧断言写 11s+1.6=12.6，正是「解析层 38 帧 / 渲染层
        #   39 帧」不一致的化石。判据直接走生产格式化器 pack_dur_spec，别在本
        #   文件里再手搓一份 —— 两份实现就会各自漂移。
        _h = grid_seconds(HANDOFF_SECONDS)
        _spec = pack_dur_spec({"new_seconds": 11.0, "gen_seconds": 11.0 + _h})
        self.assertEqual(_spec, "11s+%g=%g" % (_h, 11.0 + _h))
        # 回写出来的段头必须与段对象自己的三个数一致（新增/回放/生成）
        self.assertIn("########## S02 / %s / EN ##########" % pack_dur_spec(segments[1]),
                      text)
        self.assertIn("END OF PACK / 文件结束", text)
        # 台词块是唯一允许出现中文的地方，回写后必须还在
        self.assertIn("<d>[Chinese]", text)


class TestShotScript(unittest.TestCase):
    # [Shot N] 整段式脚本：英文正文 + 台词块内的中文原文
    SCRIPT = """\
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

    def test_split_along_shot_boundary(self):
        segments, meta, issues = parse_pack(self.SCRIPT, language="en",
                                            default_duration=11.0, auto_fix=True)
        # Four shots, uneven durations 10/2/13/5 -> should rebalance to 11/11/11/7
        self.assertGreaterEqual(len(segments), 3)
        for seg in segments:
            self.assertGreaterEqual(seg["new_seconds"], MIN_NEW_SECONDS - 0.001)
            # ★ 2026-09-30：`new_seconds` 现在是网格对齐后的**真实贡献**
            #   （网格帧 − 回放帧），最多比名义值大 16/24 ≈ 0.667s，所以判据要
            #   带上 GRID_SLACK_SECONDS，否则每段都误报"超过单段上限"。
            self.assertLessEqual(seg["new_seconds"],
                                 MAX_NEW_SECONDS + GRID_SLACK_SECONDS + 0.001)
        # 中文混入 / 缺任务前缀 / 碎片段 / 超长段这些**结构性**问题必须为零；
        # 「缺少 PACK 头部」「英文词数」是写作提示，不算（见头部说明）。
        self.assertEqual(len(structural(issues)), 0, structural(issues))


if __name__ == "__main__":
    unittest.main()
