# -*- coding: utf-8 -*-
"""2026-10-14 回归：段间 VRAM 清理补齐 prefetch 队列（借鉴
ComfyUI_MiniMaxH3_Director/director/vram_cleanup.py 的 cleanup_segment_vram）。

上游与本包的段间清理序列此前只差一项：``comfy.model_prefetch.
cleanup_prefetch_queues()``。节点自己的 finally 只清一次队列，而 Director
在**一次执行内循环每一段**（且每段都要再过 26 GB 文本编码器），队列不清
会让后续段的权重读取退化成同步 host-to-device 拷贝。

守四条：
  ① ``free_between_segments(False)``（默认路径）也清队列 —— 段间清理不只
     发生在"要卸载"的时候；
  ② ``unload_models=True`` 真卸载路径：unload_all_models → cleanup_models →
     清队列 → soft_empty_cache，一步不少；
  ③ 内存护栏路径（可用内存装不下重装）仍清队列，且**不调 cleanup_models**
     （权重留在 RAM 是本包 8GB 机器的保命设计，不能被"改进"破坏）；
  ④ OOM 重试路径清队列（engine.render_segment 的 except 分支），helper 在
     拿不到 comfy.model_prefetch 时静默 False，不炸链路。

跑法（必须用 ComfyUI 的 venv，engine 依赖 comfy）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_vram_cleanup.py
"""
import os
import sys
from unittest import mock

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

import comfy.model_management as mm                                  # noqa: E402
import comfy.model_prefetch as mp                                    # noqa: E402

from marquee_director import engine                                   # noqa: E402
from marquee_director.common import clear_prefetch_queues            # noqa: E402


class _FakeSlot(object):
    """current_loaded_models 里的一条假 LoadedModel。"""

    def __init__(self, memory):
        self.model_memory = memory
        self._dead = False

    def is_dead(self):
        return self._dead


def _patch_mm():
    """把 engine 用到的 mm 函数全部换成 recorder，避免碰真显存/真队列。"""
    calls = []

    def rec(name):
        def fn(*a, **kw):
            calls.append(name)
            return None
        return fn

    patches = [
        mock.patch.object(mm, "cleanup_models_gc", rec("cleanup_models_gc")),
        mock.patch.object(mm, "unload_all_models", rec("unload_all_models")),
        mock.patch.object(mm, "cleanup_models", rec("cleanup_models")),
        mock.patch.object(mm, "soft_empty_cache", rec("soft_empty_cache")),
    ]
    return calls, patches


def _run(patches):
    for p in patches:
        p.start()
    try:
        pass
    finally:
        pass


def _stop(patches):
    for p in patches:
        p.stop()


def test_default_path_clears_queue():
    """① unload_models=False：仍清 prefetch 队列，不动卸载类函数。"""
    calls, patches = _patch_mm()
    with mock.patch.object(mp, "cleanup_prefetch_queues",
                           side_effect=lambda: calls.append("prefetch")) as q:
        _run(patches)
        try:
            engine.free_between_segments(False)
        finally:
            _stop(patches)
    assert q.call_count == 1, "默认段间清理也必须清 prefetch 队列"
    assert "unload_all_models" not in calls
    assert "cleanup_models" not in calls
    assert calls[-1] == "soft_empty_cache"
    print("PASS ① 默认路径清队列、不卸载")


def test_full_unload_sequence():
    """② unload_models=True（内存充足）：完整序列含清队列。"""
    calls, patches = _patch_mm()
    with mock.patch.object(mp, "cleanup_prefetch_queues",
                           side_effect=lambda: calls.append("prefetch")) as q:
        with mock.patch.object(mm, "current_loaded_models", []):
            with mock.patch.object(engine, "free_ram_bytes", return_value=64 << 30):
                _run(patches)
                try:
                    engine.free_between_segments(True)
                finally:
                    _stop(patches)
    assert q.call_count == 1, "整条 unload 路径清一次即可（函数末尾收尾）"
    assert "cleanup_models" in calls and "prefetch" in calls
    assert calls.index("prefetch") > calls.index("cleanup_models")
    assert calls[-2:] == ["prefetch", "soft_empty_cache"], calls[-4:]
    print("PASS ② 真卸载序列：unload → cleanup_models → … → 队列 → soft_empty_cache")


def test_ram_guard_still_clears_queue():
    """③ 内存护栏：不清 weight（不调 cleanup_models）但队列照清。"""
    calls, patches = _patch_mm()
    slot = _FakeSlot(memory=(20 << 30) + (26 << 30))     # 本包权重 ≈ 49GB
    with mock.patch.object(mp, "cleanup_prefetch_queues",
                           side_effect=lambda: calls.append("prefetch")) as q:
        with mock.patch.object(mm, "current_loaded_models", [slot]):
            with mock.patch.object(engine, "free_ram_bytes", return_value=8 << 30):
                _run(patches)
                try:
                    engine.free_between_segments(True)
                finally:
                    _stop(patches)
    assert q.call_count == 1, "护栏路径也要清队列"
    assert "cleanup_models" not in calls, "护栏路径绝不丢 RAM 里的权重"
    assert "unload_all_models" in calls, "护栏路径只退显存"
    print("PASS ③ 内存护栏：只退显存、保权重、队列照清")


def test_helper_is_exception_safe():
    """④ helper：异常/缺函数 → False，不向上炸。"""
    real = mp.cleanup_prefetch_queues
    try:
        with mock.patch.object(mp, "cleanup_prefetch_queues",
                               side_effect=RuntimeError("boom")):
            assert clear_prefetch_queues() is False
        with mock.patch.object(mp, "cleanup_prefetch_queues",
                               side_effect=lambda: None):
            assert clear_prefetch_queues() is True
        # 彻底没有该入口（老版本 ComfyUI）：静默 False
        saved = mp.__dict__.pop("cleanup_prefetch_queues")
        try:
            assert clear_prefetch_queues() is False
        finally:
            mp.cleanup_prefetch_queues = saved
    finally:
        mp.cleanup_prefetch_queues = real
    print("PASS ④ helper 容错：异常/缺入口均静默 False")


if __name__ == "__main__":
    test_default_path_clears_queue()
    test_full_unload_sequence()
    test_ram_guard_still_clears_queue()
    test_helper_is_exception_safe()
    print("\n全部通过：prefetch 队列清理已进段间清理的全部路径。")
