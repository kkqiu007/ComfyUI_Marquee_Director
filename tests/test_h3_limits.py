# -*- coding: utf-8 -*-
"""2026-10-17 回归：H3 三条「核心不拦、渲得出来但结果错」的硬限制。

来源与取证：
  * 核心 ``comfy_extras/nodes_minimax_h3.py`` 只做两件事 —— 帧数吸附到
    17k+5（``align_frame_count``）、画布按 32 对齐（``adapt_canvas``）。它的
    ``length`` 上限是 **3600**，``adapt_canvas`` 接受任意宽高比。
  * diffusers 参考实现 ``modular_pipelines/minimax_h3/modular_pipeline.py``：
    MINIMAX_H3_MIN/MAX_ASPECT_RATIO = 1/4 .. 4。
  * fblissjr/ComfyUI-h3-explorations ``h3_rules.py`` 把这批规则整理成常量；
    ``docs/wiki/decisions.md`` 另记：相接的两个窗口必须**同时**落在 17k+5 与
    3 的倍数上（视频网格 + 音频时钟）。

守四条：
  ① 音频时钟判据：``on_both_clocks`` 只认 17k+5 且为 3 的倍数，合法相接长度
     在 362 以下只有 7 个，最长是 345（362 自己不是 3 的倍数）；
  ② 误差量化：``audio_clock_error_ms`` 对 124/158 给出 ±8.33 ms，对 141/192/345
     给出 0 —— 与 ``audio_latent_frames`` 的四舍五入一致；
  ③ ``nearest_chain_length`` 给出可执行的替代长度（158 → 141，362 → 345）；
  ④ 画幅区间：1:4–4:1 之内静默，之外给出提示（8:1 与 1:8 都触发），且提示
     文案里的横竖屏建议不会被顺手写成越界值；
  ⑤ 段长吸附：``chain_length`` 一律向上吸附到双时钟网格 39+51n，日志里真实
     出现过的 124/311/209/277 帧逐个对上，且全域只向上、幂等、合法；
  ⑥ 越过训练上限仍然闭合：15s(360) → 396，而不是 None；
  ⑦ 解析层与渲染层同源：``prompt_pack.chain_seconds`` 与
     ``common.chain_length`` 对同一秒数给出同一帧数。

跑法（必须用 ComfyUI 的 venv，common 依赖 comfy_extras）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_h3_limits.py
"""
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

from marquee_director import common as C                             # noqa: E402
from marquee_director import director as D                           # noqa: E402


def test_audio_clock_rule():
    """① 17k+5 且 3 的倍数才算能相接的长度。"""
    assert C.on_both_clocks(141) is True
    assert C.on_both_clocks(192) is True
    assert C.on_both_clocks(124) is False       # 17k+5，但 124 % 3 == 1
    assert C.on_both_clocks(158) is False
    assert C.on_both_clocks(362) is False       # 训练上限，但 362 % 3 == 2
    assert C.on_both_clocks(140) is False       # 3 的倍数，但不在 17k+5 网格上

    lengths = C.chain_lengths()
    assert lengths == (39, 90, 141, 192, 243, 294, 345), lengths
    assert max(lengths) == 345, "362 不是 3 的倍数，合法相接长度最大是 345"
    print("PASS ① on_both_clocks / chain_lengths")


def test_audio_clock_error():
    """② 音画错位的量，与 audio_latent_frames 的四舍五入同源。"""
    assert abs(C.audio_clock_error_ms(141)) < 1e-9
    assert abs(C.audio_clock_error_ms(192)) < 1e-9
    assert abs(C.audio_clock_error_ms(124) - 8.3333) < 0.01
    assert abs(C.audio_clock_error_ms(158) + 8.3333) < 0.01
    # 与 audio_latent_frames 自洽：误差 = (落步 - 精确步) / 40 * 1000
    for n in (124, 141, 158, 192, 243, 345):
        exact = n * C.AUDIO_LATENT_FPS / float(C.FPS)
        want = (C.audio_latent_frames(n) - exact) / C.AUDIO_LATENT_FPS * 1000.0
        assert abs(C.audio_clock_error_ms(n) - want) < 1e-9, n
    print("PASS ② audio_clock_error_ms 与 audio_latent_frames 自洽")


def test_nearest_chain_length():
    """③ 给出的替代长度本身必须是合法的，且确实是最近的。"""
    assert C.nearest_chain_length(158) == 141
    assert C.nearest_chain_length(124) == 141
    assert C.nearest_chain_length(362) == 345
    assert C.nearest_chain_length(192) == 192
    for n in range(5, 363):
        pick = C.nearest_chain_length(n)
        assert C.on_both_clocks(pick), (n, pick)
        assert abs(pick - n) <= 51, (n, pick)
    print("PASS ③ nearest_chain_length 合法且最近")


def test_aspect_range():
    """④ 1:4–4:1 内静默，之外提示；文案里的建议画布自身必须合规。"""
    assert D._canvas_range_note({"width": 960, "height": 544}) == ""
    assert D._canvas_range_note({"width": 544, "height": 960}) == ""
    assert D._canvas_range_note({"width": 1024, "height": 288}) == ""   # 3.56:1
    assert D._canvas_range_note({"width": 0, "height": 0}) == ""        # 未设置

    wide = D._canvas_range_note({"width": 1024, "height": 128})         # 8:1
    tall = D._canvas_range_note({"width": 128, "height": 1024})         # 1:8
    assert "8.00:1" in wide and "1:8.00" in tall
    assert "1:4 – 4:1" in wide

    # 文案推荐的两个画布必须落在区间内，否则就是「提示把人带出区间」。
    for w, h in ((960, 544), (544, 960)):
        assert C.MIN_ASPECT_RATIO <= w / float(h) <= C.MAX_ASPECT_RATIO, (w, h)

    assert abs(C.MAX_DURATION - 362 / 24.0) < 1e-9
    print("PASS ④ _canvas_range_note 画幅区间")


def test_chain_length_snaps_up():
    """⑤ ``chain_length``：段长一律向上吸附到双时钟网格。

    这是 2026-10-18 那次修复的核心断言。以前段长只吸附 17k+5，于是实测
    一条 10 段链里 5 段踩中（124 / 311 / 209 / 277 帧），每道接缝差 8.3 ms
    且逐段累积。
    """
    # 日志里真实出现过的四个长度 —— 每一个都必须被抬到合法相接长度上。
    for requested, want in ((124, 141), (311, 345), (209, 243), (277, 294)):
        got = C.chain_length(requested)
        assert got == want, (requested, got, want)
        assert C.on_both_clocks(got), (requested, got)

    # 已经在网格上的不动（243 = 17×14+5 且 243 % 3 == 0）。
    assert C.chain_length(243) == 243
    assert C.chain_length(39) == 39
    # 幂等：吸附两次不产生新的跳变。
    for n in range(5, 400):
        once = C.chain_length(n)
        assert C.chain_length(once) == once, n
    # 只向上，绝不吞掉剧本要的内容。
    for n in range(5, 400):
        assert C.chain_length(n) >= n, n
    # 全域落回合法网格 —— 这是 ⑤ 的真正目的。
    for n in range(5, 400):
        assert C.on_both_clocks(C.chain_length(n)), n
    print("PASS ⑤ chain_length 向上吸附到 39+51n")


def test_chain_length_above_training_ceiling():
    """⑥ 超过训练上限也要有答案：15s(360) → 396，而不是 None。

    ``chain_lengths`` 被 ``MAX_LENGTH`` 截在 345，所以吸附**不能**走枚举，
    否则 15 s 的段会吸附不到、渲染层与解析层就此分叉。
    """
    assert C.chain_length(360) == 396
    assert C.chain_length(345) == 345
    assert C.chain_length(346) == 396
    print("PASS ⑥ chain_length 越过训练上限仍然闭合")


def test_parse_and_render_agree():
    """⑦ 解析层 ``chain_seconds`` 与渲染层 ``chain_length`` 必须逐帧一致。

    两层不同源的后果是「提示词时间码 ≠ 实际生成帧数」：多出来的帧成了剧本
    外的时间，模型只能即兴发声。这条断言就是把那条不变式钉住。
    """
    import marquee_director.prompt_pack as pp
    for secs in (5.0, 5.17, 7.29, 8.0, 8.71, 10.12, 11.0, 11.54, 12.96, 13.2, 15.0):
        parsed = int(round(pp.chain_seconds(secs) * pp.FPS))
        assert parsed == C.chain_length(C.seconds_to_frames(secs)), (secs, parsed)
    print("PASS ⑦ prompt_pack.chain_seconds 与 common.chain_length 同源")


if __name__ == "__main__":
    test_audio_clock_rule()
    test_audio_clock_error()
    test_nearest_chain_length()
    test_aspect_range()
    test_chain_length_snaps_up()
    test_chain_length_above_training_ceiling()
    test_parse_and_render_agree()
    print("\n全部通过：H3 硬限制的判定、吸附与提示就位。")
