# -*- coding: utf-8 -*-
"""二采精修（Refine）—— 借鉴 ComfyUI_MiniMaxH3_Director 的 ``refine`` 外接形态。

那个包把后处理做成**可选外接节点**：导演台留一个 ``refine`` 口，不接就是原来
的单次采样，接了才多跑一遍。这里照搬同一条契约——``H3 Refine`` 节点把二采配置
打包成一个 ``H3_REFINE`` 对象，接到 H3 Director 的 ``refine`` 口；不接时
``settings`` 里根本没有 ``refine`` 键，``engine.render_segment`` 的分支进不去，
行为与加这个功能之前**逐字节相同**。

没有抄它的实现，只抄了契约。原因：它的 refine 与自己的 ``plan``/``seg`` 体系、
``h3_motion_context`` 关键帧重钉、RTX VSR / 3D latent 放大权重深度耦合，那些
依赖这台机器上没有（``models/latent_upscale_models/`` 里没有 3D 权重、
``nvidia-vfx`` 没装），搬过来是搬一堆跑不了的分支。所以这里只落地**同分辨率
二采**这一条不需要任何外部权重、可以在本机验证的路径。

二采在链条里的位置是这半个功能里唯一需要想清楚的事：

    一采 →【二采】→ latent_tail / latent_signature / VAE 解码

二采必须排在**切尾巴和解码之前**。段间锚点是从采样结果里切出去的，如果继续从
一采结果里切，下一段开头钉住的就是一采画质的帧，而成片里前一段是二采画质——
接缝处会有一记画质跳变，正好毁掉这个包唯一存在的理由。放在解码之前，尾巴、
漂移签名、成片三者天然都来自同一份（二采后的）latent，不需要额外对齐。

段首锁不用重建：一采的 ``positive`` 里带着 ``minimax_keyframes``（上一段尾帧的
钉子），二采复用同一个 ``positive``，所以精修过程中接缝那一段仍被引导锁住，
不会把钉死的开头画飘。这也是 MiniMaxH3_Director ``keep_existing_keyframes``
的同一个决定。
"""

from __future__ import annotations

import comfy.samplers
from comfy_api.latest import io

from .common import (
    REFINE_FOLLOW_SAMPLER,
    REFINE_SEED_FOLLOW,
    REFINE_SEED_MODES,
)
from .nodes import CATEGORY

# 二采配置的载体。Director 的 ``refine`` 口与这个节点的输出口是同一类型。
H3Refine = io.Custom("H3_REFINE")

# 二采次数上限。refine 是**整段重采**，每段每次都要走一遍 UNET，设成 unbounded
# 只会让人误触一次跑一晚上。精修超过 3 次的收益也已经很小。
MAX_REFINE_PASSES = 8

_SIGMAS_TOOLTIP = (
    "二采噪声表。接 BasicScheduler（denoise 调小，如 0.35~0.5）或 ManualSigmas。\n\n"
    "必须带 denoise<1：二采是「在已有画面上再降一次噪」，不是从头生成。接了 "
    "denoise=1 的表等于把一采结果当纯噪声重画一遍，画面会整体偏移、段间直接跳。\n\n"
    "BasicScheduler 的 model 口请接**二采用的那套模型**（接了 refine_model 就接它，"
    "没接就接 H3 Chain Settings 的 model），H3 的 SigmaShift 由模型那侧带上。\n\n"
    "不接这个口 = 关闭二采（即便节点已接到 Director）。"
)

_MODEL_TOOLTIP = (
    "二采用的 UNET。不接则用 H3 Chain Settings 的主模型。\n\n"
    "典型用法：一采挂 Turbo LoRA（步数少、构图快），二采换成不带 LoRA 的原模型或"
    "另一套，用细节换回速度。换模型会进缓存键，相关段会重渲。\n\n"
    "MODEL only——H3 的 LoRA 都在扩散侧，把它塞进 Qwen3-VL 文本编码器不是它被"
    "训练过的用法。"
)

_PASSES_TOOLTIP = (
    "精修遍数。每一遍都用上面同一张噪声表再采一次，画面逐级收敛。\n\n"
    "1 就够用；2~3 在细节上还能看出来，再往上基本是白烧显卡。每段每遍都是完整"
    "的一次 UNET 前向，8 段 × 3 遍 = 24 次采样。"
)

_SAMPLER_TOOLTIP = (
    "二采采样器。「%s」= 跟一采用同一个（H3 Chain Settings 的 sampler 口）。\n\n"
    "官方 H3 模板一采用 res_multistep；二采常用 euler，收敛更稳。\n\n"
    "跟随一采时改上游采样器会连带改二采，且会进缓存键触发重渲 —— 这是对的，"
    "采样器变了产物本来就变了。"
    % REFINE_FOLLOW_SAMPLER
)

_SEED_TOOLTIP = (
    "二采噪声的种子从哪来。\n\n"
    "「跟随一采」= 与本段一采同种子：两次采样落在同一条噪声轨迹上，二采是在"
    "一采的结果上继续收敛，画面最稳，也是默认。\n"
    "「一采+1」= 与本段错开一号：想让二采真正引入一点新变化时用。\n"
    "「独立种子」= 完全用下面那个 seed，与一采无关。\n\n"
    "种子进缓存键，改了会重渲这一段。"
)


class H3RefineNode(io.ComfyNode):
    """Pack a second-sample pass. Wire ``refine`` into H3 Director's ``refine``."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3Refine",
            display_name="H3 Refine（二采精修）",
            category=CATEGORY,
            description=(
                "Optional second sample pass for H3 Director. Connect "
                "`sigmas` (a denoise<1 schedule) and wire `refine` into the "
                "Director's `refine` input; every segment is then sampled once "
                "more before it is decoded, its handoff tail is cut, and the "
                "cut is joined. Unwired (or wired without `sigmas`) the Director "
                "behaves exactly as before: one sample per segment.\n\n"
                "The second pass reuses the first pass' conditioning, so the "
                "pinned seam at a segment's head stays locked while the rest of "
                "the shot is refined. Because the pass runs before the tail is "
                "cut, the next segment anchors on refined frames -- not on "
                "first-pass ones -- so no quality step appears at a join."
            ),
            inputs=[
                io.Sigmas.Input("sigmas", optional=True, tooltip=_SIGMAS_TOOLTIP),
                io.Model.Input("refine_model", display_name="二采模型",
                               optional=True, tooltip=_MODEL_TOOLTIP),
                io.Int.Input("passes", display_name="精修遍数", default=1,
                             min=1, max=MAX_REFINE_PASSES, step=1,
                             tooltip=_PASSES_TOOLTIP),
                io.Combo.Input(
                    "sampler", display_name="采样器",
                    options=[REFINE_FOLLOW_SAMPLER]
                            + list(comfy.samplers.KSampler.SAMPLERS),
                    default=REFINE_FOLLOW_SAMPLER, tooltip=_SAMPLER_TOOLTIP),
                io.Combo.Input("seed_mode", display_name="种子来源",
                               options=list(REFINE_SEED_MODES),
                               default=REFINE_SEED_FOLLOW,
                               tooltip=_SEED_TOOLTIP),
                io.Int.Input(
                    "refine_seed", display_name="二采种子", default=0, min=0,
                    max=0xffffffffffffffff, advanced=True,
                    tooltip="仅「种子来源 = 独立种子」时生效。"),
            ],
            outputs=[H3Refine.Output(display_name="refine")],
        )

    @classmethod
    def execute(cls, passes=1, sampler=REFINE_FOLLOW_SAMPLER,
                seed_mode=REFINE_SEED_FOLLOW, refine_seed=0, sigmas=None,
                refine_model=None) -> io.NodeOutput:
        return io.NodeOutput({
            "sigmas": sigmas,
            "model": refine_model,
            "passes": max(1, int(passes or 1)),
            # None = 跟随一采。存字符串名字而不是 sampler 对象：对象是每次运行
            # 新建的，repr 带内存地址，直接进缓存键会让缓存永远不命中。
            "sampler": None if sampler == REFINE_FOLLOW_SAMPLER else sampler,
            "seed_mode": seed_mode,
            "seed": int(refine_seed or 0),
        })


def register_with_extension(ext):
    """Return node classes for MarqueeDirectorExtension.get_node_list()."""
    return [H3RefineNode]
