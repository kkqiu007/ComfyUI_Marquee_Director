# -*- coding: utf-8 -*-
"""TeaCache —— H3 采样 loop 的时间步缓存（Timestep-Embedding Aware Cache）。

融合自 **Icyoung/ComfyUI-MiniMaxH3-TeaCache**（MIT，© Icyoung）：
    https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache
论文：Liu et al. 2024, 《Timestep Embedding Aware Cache》, arxiv:2411.19108。

为什么融进本包（2026-10-13 生态调研的结论）
--------------------------------------------
整条链最贵的一步就是每段的采样：20 步 × ~7B 的 DiT。相邻时间步的 block
输出高度相似，TeaCache 在「输入 rel-L1 累积量低于阈值」时直接复用上一步的
输出。上游在同代 H3（fl2va int8_convrot、res_multistep+simple、20 步、
1024×576）实测：总墙钟 306.22s → 102.08s（**-66.7%，3.0×**），A/B 观感无差。

它只走 ComfyUI 官方的
``ModelPatcher.set_model_unet_function_wrapper``（comfy/samplers.py 的
``calc_cond_batch`` 消费它，0.34 起内置），**不改 comfy/ 一行源码** —— 与
本包 ``te_prefetch`` shim 同一种「包内运行时补丁」姿势：ComfyUI 更新不会
冲掉它，上游哪天自己修好这里自动失效。

与上游节点的三点差异（为什么内建而不是让用户外接那个节点）
--------------------------------------------------------
1. **total_steps 由本包从 ``settings["sigmas"]`` 的真实长度推导**，不让用户
   手填。上游节点要求手填步数、且明确要求与调度器一致；本包一次 execute 里
   会用工作流里实际的 sigmas 跑采样，推导比手填可靠（手填不一致时 end_step
   的负偏移会整体算错，缓存打到不该打的步上）。
2. **每次 ``sample()`` 调用新建 state**。上游靠 ComfyUI 每 prompt 重建节点
   实例拿到新 state；本包一次 execute 里会发生**多次** sample()（一采，加
   二采精修的每个 pass 各一次），共享 state 会让第二次采样的步数接着第一次
   往下数。
3. **阈值走 H3 链路设置的控件**（默认 0 = 关），并进 ``segment_key`` ——
   开关与阈值改变成片本身（被跳过的步输出的是复用值），缓存必须跟着失效，
   否则改了参数还拿到旧路径的产物（见 session.segment_key 的同名注释）。

步数守卫
--------
上游默认 start_step=2 / end_step=-2（早步定结构、尾步定细节，都不该跳）。
本包实测工作流有 **8 步（turbo LoRA）和 20 步（精修）** 两条链，所以：

* steps >= 12 ：沿用上游 (2, -2)；
* 6 <= steps < 12 ：收到 (1, -1) —— 8 步的链按 (2,-2) 只剩 4 步可缓存，
  加速比对不起复杂度；
* steps < 6 ：直接不启用（turbo 4 步链上没什么可省的，省一步都是质量风险）。
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import torch

# 上游仓库的 MIT 实现里与本文逐行对应的两个函数（rel_l1 / should_reuse /
# make_wrapper 的决策部分）原样保留，只补了中文注释与文档字符串。

TEACACHE_MAX_THRESH = 0.5
TEACACHE_START_STEP = 2
TEACACHE_END_STEP = -2


@dataclasses.dataclass
class TeaCacheState:
    """Rolling state kept across timesteps for a single sampling run."""

    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    accumulated: float = 0.0
    reuse_count: int = 0
    real_count: int = 0

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.accumulated = 0.0
        self.reuse_count = 0
        self.real_count = 0


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Relative L1 distance between two tensors, on any device.

    Uses float32 accumulation to avoid bf16/fp16 saturation on large tensors.
    """
    diff = (current.float() - previous.float()).abs().mean()
    ref = previous.float().abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


def should_reuse(
    state: TeaCacheState,
    current_input: torch.Tensor,
    thresh: float,
    start_step: int,
    end_step: int,
    total_steps: int,
) -> bool:
    """Decide whether to reuse the cached output at the current step.

    Guard rails:
      * Never reuse before ``start_step`` (structure hasn't stabilized).
      * Never reuse in the final ``-end_step`` steps (fine detail matters).
      * Never reuse on step 0 (nothing cached yet).
      * Never reuse if ``prev_output`` is None (first real run of the session).
    """
    step = state.step_idx
    if state.prev_output is None or state.prev_input is None:
        return False
    if step < start_step:
        return False
    # Support negative end_step: -N means "last N steps must be real"
    effective_end = end_step if end_step >= 0 else total_steps + end_step
    if step >= effective_end:
        return False

    delta = rel_l1(current_input, state.prev_input)
    state.accumulated += delta
    if state.accumulated < thresh:
        return True
    # Cross the threshold: run real forward, reset accumulator
    state.accumulated = 0.0
    return False


def make_wrapper(
    state: TeaCacheState,
    thresh: float,
    start_step: int,
    end_step: int,
    total_steps: int,
) -> Callable[..., torch.Tensor]:
    """Return a ``unet_wrapper_function`` compatible with ModelPatcher.

    ComfyUI's ``ModelPatcher.set_model_unet_function_wrapper(fn)`` calls
    ``fn(apply_model, {"input": x, "timestep": t, "c": conds})`` on every
    sampling step. We wrap it to decide reuse-vs-real.
    """

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]

        if should_reuse(state, x, thresh, start_step, end_step, total_steps):
            state.reuse_count += 1
            state.step_idx += 1
            # Reuse previous output. Note: this assumes the model output is a
            # velocity/noise-prediction that changes slowly across steps -- true
            # for standard v-prediction / rectified-flow DiTs including H3.
            return state.prev_output

        # Real forward pass
        output = apply_model(x, args["timestep"], **args["c"])
        state.prev_input = x.detach()
        state.prev_output = output.detach()
        state.real_count += 1
        state.step_idx += 1
        return output

    return wrapper


def _steps_of(sigmas) -> int:
    """sigma 调度表 → 采样步数。认 tensor（第 0 维）与序列；判不出来给 0。"""
    if sigmas is None:
        return 0
    shape = getattr(sigmas, "shape", None)
    if shape is not None and len(shape) >= 1:
        try:
            return int(shape[0])
        except Exception:
            return 0
    try:
        return len(sigmas)
    except TypeError:
        return 0


def guards_for(steps: int):
    """步数 → (start_step, end_step)；不可用时给 None（= 不启用）。

    >=12 沿用上游 (2,-2)；6..11 收紧到 (1,-1)（turbo 8 步链按 (2,-2) 只剩
    4 步可缓存，加速比对不起这点复杂度）；<6 不启用。
    """
    if steps >= 12:
        return (TEACACHE_START_STEP, TEACACHE_END_STEP)
    if steps >= 6:
        return (1, -1)
    return None


def wrap(model, settings):
    """挂上 TeaCache wrapper。返回 ``(model, state)``。

    关闭（阈值 <= 0 / 超过上限）、步数判不出、或少于 6 步时**原样返回**
    ``(model, None)`` —— 不 clone、不挂 wrapper，一个比特的行为都不变。
    这是默认路径：老工作流没有这个控件、拿到 0，等于这个融合不存在。

    wrapper 只活在返回的 clone 上（``model.clone()`` 后
    ``set_model_unet_function_wrapper``），调用方把 clone 丢掉，钩子跟着没
    —— 与本包 seam_remask 的 ``install_prefix_remask`` 同一个契约，不存在
    跨段残留的全局状态。
    """
    try:
        thresh = float((settings or {}).get("teacache_thresh") or 0.0)
    except (TypeError, ValueError):
        return model, None
    if thresh <= 0.0 or thresh > TEACACHE_MAX_THRESH:
        return model, None

    steps = _steps_of((settings or {}).get("sigmas"))
    guards = guards_for(steps)
    if guards is None:
        return model, None
    start_step, end_step = guards

    state = TeaCacheState()
    state.reset()
    patched = model.clone()
    patched.set_model_unet_function_wrapper(
        make_wrapper(state=state, thresh=thresh,
                     start_step=start_step, end_step=end_step,
                     total_steps=steps))
    # 与上游节点一致：把 state 也塞一份进 model_options，方便以后挂统计面板。
    patched.model_options.setdefault("teacache", {})["state"] = state
    return patched, state


def counts_note(state) -> str:
    """一次采样结束后给人看的一行：真跑了几步、复用了几步。"""
    if state is None:
        return ""
    return "真实 %d 步 + 复用 %d 步（共 %d）" % (
        state.real_count, state.reuse_count, state.step_idx)
