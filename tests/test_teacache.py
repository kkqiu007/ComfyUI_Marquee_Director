# -*- coding: utf-8 -*-
"""2026-10-13 回归：TeaCache 时间步缓存（融合 Icyoung/ComfyUI-MiniMaxH3-TeaCache，MIT）

守五条，任何一条破了都会**静默**出问题：
  ① 默认（阈值 0）零成本：wrap() 原样返回同一个 model，不 clone、不挂 wrapper；
  ② 决策逻辑的护栏：第 0 步 / 早于 start_step / 进入尾部 -end_step 步，绝不复用；
     累积量过阈值必须跑真实 forward 并重置累加器；
  ③ wrapper 的语义：真实 forward 的返回原样透出；复用时返回**上一步的输出**；
     每步只调一次 apply_model；
  ④ 步数守卫：≥12 步沿用上游 (2,-2)；6–11 步收紧 (1,-1)；<6 步不启用
     （turbo 4 步链上没什么可省的）；
  ⑤ segment_key 纪律：阈值 0（或缺键）不进 payload —— 否则升级后所有既有会话
     凭空全量重渲；开了之后改阈值必须换键。

跑法（必须用 ComfyUI 的 venv，engine/session 依赖 comfy）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_teacache.py
"""
import os
import sys

import torch

# 本包在 <ComfyUI>/custom_nodes/ 下，所以上两级就是 ComfyUI 根目录。
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

from ComfyUI_Marquee_Director.marquee_director import nodes as N          # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import session as SESS     # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import teacache as TC      # noqa: E402

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  [FAIL]", msg)
    return cond


# ---------------------------------------------------------------- ② 决策逻辑
check(TC.rel_l1(torch.ones(4), torch.ones(4)) == 0.0, "相同张量 rel_l1 应为 0")
check(abs(TC.rel_l1(torch.full((4,), 2.0), torch.ones(4)) - 1.0) < 1e-6,
      "翻倍的张量 rel_l1 应为 1")

st = TC.TeaCacheState()
check(TC.should_reuse(st, torch.ones(4), 1.0, 2, -2, 20) is False,
      "prev_output=None（第 0 步）不该复用")
st.prev_input = torch.ones(4)
st.prev_output = torch.ones(4)
st.step_idx = 1
check(TC.should_reuse(st, torch.ones(4), 1.0, 2, -2, 20) is False,
      "早于 start_step 不该复用")
st.step_idx = 18
check(TC.should_reuse(st, torch.ones(4), 1.0, 2, -2, 20) is False,
      "进入尾部 -end_step 窗口（18 >= 20-2）不该复用")
st.step_idx = 2
check(TC.should_reuse(st, torch.ones(4), 1.0, 2, -2, 20) is True,
      "窗口内、累计量为 0 低于阈值 1.0，应当复用")
check(TC.should_reuse(st, torch.ones(4), 1.0, 2, -2, 20) is True and
      st.accumulated == 0.0,
      "输入不变（rel_l1=0）时累加器不该涨，应持续复用")
# 过阈值：真实 forward 后累加器必须重置
st2 = TC.TeaCacheState()
st2.prev_input = torch.ones(4)
st2.prev_output = torch.ones(4)
st2.step_idx = 2
check(TC.should_reuse(st2, torch.full((4,), 2.0), 0.5, 2, -2, 20) is False,
      "单步 rel_l1=1.0 已过阈值 0.5，应当跑真实步")
check(st2.accumulated == 0.0, "过阈值后累加器没重置")

# ------------------------------------------------------- ③ wrapper 的真实语义
calls = []


def fake_apply(x, t, **c):
    calls.append((float(t),))
    # 输出只跟步号走 —— 这样"复用返回的是上一步输出"可以被直接验出来
    return torch.tensor([float(t)])


st3 = TC.TeaCacheState()
w = TC.make_wrapper(st3, thresh=1.0, start_step=2, end_step=-2, total_steps=20)
args = {"input": torch.ones(4), "timestep": torch.tensor(0.9), "c": {}}
out0 = w(fake_apply, args)
check(len(calls) == 1 and abs(out0.item() - 0.9) < 1e-5,
      "第 0 步必须真实 forward 并原样透出")
# 输入完全不变 ⇒ 每步 rel_l1 = 0 ⇒ 累计永远低于阈值 ⇒ 窗口内全复用。
# 注意 prev_output 在**每次真实 forward 后**更新，所以第 1 步（早于 start_step，
# 仍走真实）之后，复用的返回的是第 1 步的输出（0.89），不是第 0 步的。
for step in range(1, 18):
    out = w(fake_apply, {"input": torch.ones(4),
                         "timestep": torch.tensor(0.9 - 0.01 * step), "c": {}})
    if step < 2:
        check(abs(out.item() - (0.9 - 0.01 * step)) < 1e-5 and len(calls) == step + 1,
              "早于 start_step 的第 %d 步走了缓存" % step)
    else:
        check(abs(out.item() - 0.89) < 1e-5 and len(calls) == 2,
              "第 %d 步应复用第 1 步输出（0.89），实际 %.5f / 调用 %d 次"
              % (step, out.item(), len(calls)))
check(st3.real_count == 2 and st3.reuse_count == 16,
      "20 步调度下应 2 次真实 + 16 次复用，实际 %d/%d"
      % (st3.real_count, st3.reuse_count))

# ------------------------------------------------------------ ④ 步数守卫 + ①
check(TC.guards_for(20) == (2, -2), "20 步应沿用上游 (2,-2)")
check(TC.guards_for(8) == (1, -1), "8 步（turbo 链）应收紧到 (1,-1)")
check(TC.guards_for(4) is None, "4 步不该启用")
check(TC._steps_of(torch.linspace(1, 0, 8)) == 8, "tensor sigmas 步数推导错了")
check(TC._steps_of([1.0, 0.5]) == 2, "序列 sigmas 步数推导错了")
check(TC._steps_of(None) == 0, "sigmas=None 应判 0（= 不启用）")


class _FakeModel:
    """只验契约的最小替身：clone() 返回带标记的新对象。"""

    def __init__(self, tag="orig"):
        self.tag = tag
        self.wrapper = None
        self.model_options = {}

    def clone(self):
        c = _FakeModel(self.tag + ".clone")
        c.model_options = dict(self.model_options)
        return c

    def set_model_unet_function_wrapper(self, fn):
        self.wrapper = fn


m = _FakeModel()
sig8 = torch.linspace(1, 0, 8)
sig4 = torch.linspace(1, 0, 4)

out_m, st_off = TC.wrap(m, {"teacache_thresh": 0.0, "sigmas": sig8})
check(out_m is m and st_off is None,
      "① 阈值 0 必须原样返回同一个 model、state=None（老工作流零成本）")
out_m, st_off = TC.wrap(m, {"sigmas": sig8})
check(out_m is m and st_off is None, "缺键必须等同关闭")
out_m, st_off = TC.wrap(m, {"teacache_thresh": 0.15, "sigmas": sig4})
check(out_m is m and st_off is None, "④ 4 步调度必须不启用")
out_m, st_on = TC.wrap(m, {"teacache_thresh": 0.15, "sigmas": sig8})
check(out_m is not m and st_on is not None,
      "8 步 + 阈值 0.15 必须返回挂上 wrapper 的 clone")
check(out_m.tag == "orig.clone" and out_m.wrapper is not None,
      "clone 上没有挂上 model_function_wrapper")
check(m.wrapper is None, "原 model 被就地改了 —— 必须只改 clone")
check("teacache" in out_m.model_options
      and out_m.model_options["teacache"]["state"] is st_on,
      "state 没塞进 clone 的 model_options")
check("复用" in TC.counts_note(st_on) and TC.counts_note(st_off) == "",
      "counts_note 的开关语义不对")

# --------------------------------------------- ⑤ segment_key 的 payload 纪律
base = {
    "width": 480, "height": 864, "ref_image_size": "match",
    "sigmas": sig8, "sampler": "res_multistep",
    "model": None, "handoff_mode": "latent",
    "seed": 307046,
}
segment = {"prompt": "a", "length": 39, "resolved_seed": 1,
           "images": [], "videos": [], "video_audios": [], "audios": []}
k_off = SESS.segment_key(base, segment, 39, None)
k_zero = SESS.segment_key(dict(base, teacache_thresh=0.0), segment, 39, None)
check(k_off == k_zero,
      "teacache_thresh=0 进了 segment_key —— 升级后既有会话会全量重渲")
k_on = SESS.segment_key(dict(base, teacache_thresh=0.15), segment, 39, None)
k_on2 = SESS.segment_key(dict(base, teacache_thresh=0.2), segment, 39, None)
check(k_off != k_on and k_on != k_on2,
      "开/关与换阈值都必须换键 —— 否则改了参数还拿到旧路径的产物")

# ------------------------------------- 链路设置控件的落点：本工作流真实取值
# 工作流 MiniMax-H3-无限分镜-优化版v4_横向分区.json 的 H3ChainSettings:
#   [480, 864, 307046, "茜色", "match", "latent", 0.05, 0.1, false, ""]
# 末位新增 teacache_thresh；老工作流不带这个键 ⇒ execute 默认 0 = 关。
wf_values = [480, 864, 307046, "茜色", "match", "latent", 0.05, 0.1, False, ""]
settings_wf = N.H3ChainSettingsNode.execute(
    None, None, None, None, "res_multistep", sig8,
    wf_values[0], wf_values[1], wf_values[2], wf_values[3],
    wf_values[4], wf_values[5], wf_values[6], wf_values[7],
    wf_values[8], wf_values[9]).args[0]
check(settings_wf.get("teacache_thresh") == 0.0,
      "老工作流（10 个 widget 值）落下来 teacache_thresh 必须是 0")
k_wf = SESS.segment_key(settings_wf, segment, 39, None)
check(k_wf == k_off,
      "老工作流的 segment_key 变了 —— 升级会让用户的会话凭空重渲")
settings_tc = N.H3ChainSettingsNode.execute(
    None, None, None, None, "res_multistep", sig8,
    480, 864, 307046, "茜色", "match", "latent", 0.05, 0.1, False, "", 0.15).args[0]
check(settings_tc.get("teacache_thresh") == 0.15, "新控件值没进 settings")
m2, st2 = TC.wrap(_FakeModel(), settings_tc)
check(st2 is not None, "工作流取 0.15 + 8 步调度时应启用")

print("FAIL:", FAIL if FAIL else "无 —— 全部通过")
sys.exit(1 if FAIL else 0)
