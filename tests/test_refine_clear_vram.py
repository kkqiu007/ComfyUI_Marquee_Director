# -*- coding: utf-8 -*-
"""2026-10-17 回归：二采前清显存（`clear_vram`）是**显存策略**，不是渲染参数。

来源：AIMixer/ComfyUI_MiniMaxH3_Director `eb9d274`
（``clear_vram_before_refine``）。同一段里一采的 UNET 还驻留着，二采紧接着就开始；
接了另一套 `二采模型` 时两套 UNET 同时在显存里，8GB 卡上峰值就在这里。

守四条：
  ① 控件追加在 `H3Refine` 入参**末尾**（widgets_values 是位置契约，插在中间会让
     已存工作流的取值整体错位），且默认 False —— 旧工作流行为不变；
  ② `execute` 把开关带进 `H3_REFINE` 包，键名 `clear_vram`，布尔化；
  ③ **缓存键不受影响**：`session.segment_key` 逐项枚举 refine 字段，没有这一项。
     这是本次改动唯一可能伤到既有会话的地方 —— 勾一下让整条链重渲是不能接受的；
     同时做反向对照：改 `passes` 必须改变缓存键，否则这个断言等于没测；
  ④ `engine.refine_samples` 只在开关为真时调 `free_between_segments`，且
     未接 refine（`settings` 里没有 `refine`）时一次都不调。

跑法（必须用 ComfyUI 的 venv，engine 依赖 comfy）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_refine_clear_vram.py
"""
import os
import sys
from unittest import mock

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

import torch                                                        # noqa: E402

from marquee_director import engine                                  # noqa: E402
from marquee_director import refine_nodes as R                       # noqa: E402
from marquee_director import session as S                            # noqa: E402


class _DummyModel(object):
    """model_digest 只看类名与 patches，够用。"""

    def __init__(self):
        self.patches = {}


def _settings(clear_vram=False, passes=1, with_refine=True):
    sigmas = torch.linspace(1.0, 0.0, 5)
    out = {
        "width": 960, "height": 544, "ref_image_size": "match",
        "sigmas": sigmas, "sampler": "euler", "model": _DummyModel(),
    }
    if with_refine:
        out["refine"] = {
            "sigmas": sigmas, "model": None, "passes": passes,
            "sampler": None, "seed_mode": "跟随一采", "seed": 0,
            "clear_vram": bool(clear_vram),
        }
    return out


def _segment():
    return {"prompt": "p", "length": 192, "resolved_seed": 7,
            "images": [], "videos": [], "video_audios": [], "audios": []}


def test_widget_is_appended_and_defaults_off():
    """① 末尾追加 + 默认 False。"""
    ids = [i.id for i in R.H3RefineNode.define_schema().inputs]
    assert ids[-1] == "clear_vram", ids
    assert ids[:-1] == ["sigmas", "refine_model", "passes", "sampler",
                        "seed_mode", "refine_seed"], ids
    field = R.H3RefineNode.define_schema().inputs[-1]
    assert field.default is False, field.default
    print("PASS ① clear_vram 追加在末尾，默认关")


def test_pack_carries_the_flag():
    """② 开关进 H3_REFINE 包。"""
    on = R.H3RefineNode.execute(passes=1, clear_vram=True)
    off = R.H3RefineNode.execute(passes=1, clear_vram=False)
    pack_on = on.result[0] if hasattr(on, "result") else on[0]
    pack_off = off.result[0] if hasattr(off, "result") else off[0]
    assert pack_on["clear_vram"] is True, pack_on
    assert pack_off["clear_vram"] is False, pack_off
    # 不传时默认关（旧图没有这个控件，前端不会给值）
    default = R.H3RefineNode.execute(passes=1)
    pack_def = default.result[0] if hasattr(default, "result") else default[0]
    assert pack_def["clear_vram"] is False, pack_def
    print("PASS ② 开关进包，缺省为 False")


def test_cache_key_unaffected():
    """③ 勾与不勾缓存键相同；改 passes 必须变（反向对照）。"""
    seg = _segment()
    k_off = S.segment_key(_settings(clear_vram=False), seg, 39, "prev")
    k_on = S.segment_key(_settings(clear_vram=True), seg, 39, "prev")
    assert k_off == k_on, (k_off, k_on)

    k_more = S.segment_key(_settings(clear_vram=False, passes=2), seg, 39, "prev")
    assert k_more != k_off, "反向对照失败：passes 变化本应改变缓存键"
    print("PASS ③ 缓存键不受 clear_vram 影响（passes 对照有效）")


def test_engine_calls_cleanup_only_when_enabled():
    """④ 只在开关为真时清显存；没接 refine 时一次都不清。"""
    calls = []

    def _fake_free(unload_models=False):
        calls.append(unload_models)

    # 采样本身不该被执行到 —— 用假 sigmas 提前返回不了，所以把采样器与 guider
    # 也替掉（Guider_Basic 要真的 ModelPatcher，Dummy 过不去）。
    with mock.patch.object(engine, "free_between_segments", side_effect=_fake_free), \
         mock.patch.object(engine, "SamplerCustomAdvanced") as sampler, \
         mock.patch.object(engine, "Guider_Basic") as guider, \
         mock.patch.object(engine, "_refine_sampler", return_value=object()):
        guider.return_value.set_conds = lambda *a, **k: None
        sampler.execute.return_value = ({"samples": torch.zeros(1, 24, 2, 4, 4)},)

        # 关：不清
        engine.refine_samples(_settings(clear_vram=False), {"samples": torch.zeros(1)},
                              {}, 1)
        assert calls == [], calls

        # 开：清一次，且是 unload_models=True（要连内存一起清才降得下峰值）
        engine.refine_samples(_settings(clear_vram=True), {"samples": torch.zeros(1)},
                              {}, 1)
        assert calls == [True], calls

        # 没接 refine：settings 里根本没有 refine 键，清 0 次
        calls[:] = []
        engine.refine_samples(_settings(with_refine=False), {"samples": torch.zeros(1)},
                              {}, 1)
        assert calls == [], calls
    print("PASS ④ 仅在 clear_vram 为真时清显存")


if __name__ == "__main__":
    test_widget_is_appended_and_defaults_off()
    test_pack_carries_the_flag()
    test_cache_key_unaffected()
    test_engine_calls_cleanup_only_when_enabled()
    print("\n全部通过：clear_vram 只动显存，不动画面，也不动缓存键。")
