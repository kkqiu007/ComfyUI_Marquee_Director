# -*- coding: utf-8 -*-
"""2026-10-10 回归：无损段缓存（FFV1 边车）

背景：段落在磁盘上只有 crf10 的 mp4，而拼接成片时要把它**解出来再编一次** crf14
—— 每段带两代有损进片子；单段修复 / 断点续渲的像素锚点同样读它。借鉴
ComfyUI_MiniMaxH3_Director（Apache-2.0）的 frames_ffv1 做法，加一份像素精确的
FFV1 边车；与它不同的是这里用 PyAV 而不是 subprocess 调 ffmpeg，且存 16-bit
平面 RGB（av 的 FFV1 编码器不接受 8-bit rgb24/gbrp，这是前提而不是优化）。

这个文件守的四条，任何一条破了都会**静默**产出错片：
  ① 边车必须逐比特还原写进去的像素（否则「无损」是句空话）；
  ② 重存一段时，陈旧边车必须跟着走（否则新 mp4 配旧像素，接缝钉到被替换的那条 take）；
  ③ 边车读不出来必须回退 mp4（边车是缓存，不是唯一数据源）；
  ④ seg_cache_codec **不进** segment_key（否则升级后所有既有会话凭空全量重渲）。

跑法（必须用 ComfyUI 的 venv，video_io 依赖 comfy）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_frames_cache.py
"""
import os
import shutil
import sys
import tempfile

import torch

# 本包在 <ComfyUI>/custom_nodes/ 下，所以上两级就是 ComfyUI 根目录。
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

from ComfyUI_Marquee_Director.marquee_director import frames_cache as FC   # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import video_io as VI        # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import session as SESS       # noqa: E402

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  [FAIL]", msg)
    return cond


def _quant(images):
    """写侧的量化基准：float [0,1] → 16-bit 栅格 → 回 float。"""
    return (images.float().clamp(0, 1) * 65535.0).round() / 65535.0


tmp = tempfile.mkdtemp(prefix="marquee_ffv1_")
try:
    # ---------------------------------------------------------------- ① 无损
    n, h, w = 6, 64, 96
    g = torch.Generator().manual_seed(7)
    frames = torch.rand(n, h, w, 3, generator=g)
    clip = os.path.join(tmp, "seg_01.mp4")

    check(FC.frames_path(clip) == os.path.join(tmp, "seg_01.frames.mkv"),
          "边车命名不对：%s" % FC.frames_path(clip))
    FC.write(clip, frames)
    check(os.path.exists(FC.frames_path(clip)), "write() 没有落盘")
    back = FC.read(clip)
    check(tuple(back.shape) == (n, h, w, 3), "读回来形状变了：%s" % (tuple(back.shape),))
    check(str(back.dtype) == "torch.float32", "读回来 dtype 变了：%s" % back.dtype)
    check(torch.equal(back, _quant(frames)),
          "边车不是逐比特还原：最大偏差 %.3e" % float((back - _quant(frames)).abs().max()))
    check(float((back - frames).abs().max()) < 2e-5,
          "16-bit 栅格粗于 float32 原值（%.3e）" % float((back - frames).abs().max()))
    os.remove(FC.frames_path(clip))

    # uint8 批次也要能写（别的调用方可能给 uint8）
    u8 = (frames * 255).round().clamp(0, 255).to(torch.uint8)
    FC.write(clip, u8)
    back8 = FC.read(clip)
    check(float((back8 - u8.float() / 255.0).abs().max()) < 1e-6,
          "uint8 批次的 8→16 bit 扩张不精确")
    os.remove(FC.frames_path(clip))

    # ------------------------------------------------------- ②③ save/load 契约
    seg = os.path.join(tmp, "seg_02.mp4")
    VI.save_clip(seg, frames, None, lossless_frames=True)
    check(os.path.exists(FC.frames_path(seg)), "save_clip(lossless_frames=True) 没写边车")
    got, _audio = VI.load_clip(seg)
    check(torch.equal(got, _quant(frames)),
          "load_clip 没有优先读边车（或边车内容与写入不一致）")

    # 与有损 mp4 对照：无损这条路确实少了损失
    sidecar_err = float((got - frames).abs().max())
    os.remove(FC.frames_path(seg))
    lossy, _ = VI.load_clip(seg)
    mp4_err = float((lossy - frames).abs().max())
    check(mp4_err > 2e-5,
          "mp4 往返没有可测损失（%.3e）—— 对照失效，本测试的前提没了" % mp4_err)
    check(sidecar_err * 100.0 < mp4_err,
          "边车相对 crf10 mp4 没有三个数量级的优势：%.1e vs %.1e" % (sidecar_err, mp4_err))
    print("  [对照] ：crf10 mp4 往返误差 %.4f，FFV1 边车 %.1e" % (mp4_err, sidecar_err))

    # ② 关掉开关重存 ⇒ 陈旧边车必须消失
    FC.write(seg, frames)
    VI.save_clip(seg, frames, None, lossless_frames=False)
    check(not os.path.exists(FC.frames_path(seg)),
          "关开关重存后陈旧边车还在 —— 新 mp4 会被旧像素顶替")

    # ③ 边车截断 ⇒ 判为不完整，回退 mp4，不抛异常、不返回少几帧的半成品。
    #    （截断的 Matroska **不报错**，它安静地少解几帧 —— 所以帧数必须对账。）
    VI.save_clip(seg, frames, None, lossless_frames=True)
    sc = FC.frames_path(seg)
    with open(sc, "r+b") as fh:
        fh.truncate(os.path.getsize(sc) // 3)
    short = FC.read_if_present(seg)
    check(short is None or short.shape[0] != n,
          "截断的边车既没报错也没少帧，对账无从谈起")
    if short is not None:
        check(short.shape[0] < n, "截断边车读出 %d 帧（应少于 %d）" % (short.shape[0], n))
    check(FC.read_if_present(seg, expect_frames=n) is None,
          "截断的边车没有被帧数对账判成不可用")
    fallback, _ = VI.load_clip(seg)
    check(tuple(fallback.shape) == (n, h, w, 3),
          "边车坏了以后 load_clip 没有回退到 mp4（读到 %s）" % (tuple(fallback.shape),))
    os.remove(sc)

    # ---------------------------------------------------------------- 开关判定
    check(FC.enabled_for({"seg_cache_codec": "ffv1"}) is True, "ffv1 没被判成开")
    check(FC.enabled_for({"seg_cache_codec": " FFV1 "}) is True, "大小写/空格没归一")
    check(FC.enabled_for({}) is False, "缺键应当是关（老工作流）")
    check(FC.enabled_for({"seg_cache_codec": "h264"}) is False, "h264 应当是关")
    check(FC.enabled_for(None) is False, "settings=None 应当是关")

    count, size = FC.sidecar_summary(tmp)
    check(count == 0 and size == 0, "清空边车后 sidecar_summary 还有货：%s" % ((count, size),))
    FC.write(seg, frames)
    count, size = FC.sidecar_summary(tmp)
    check(count == 1 and size > 0, "sidecar_summary 没数到刚写的边车：%s" % ((count, size),))

    # ------------------------------------------- ④ 缓存键与段缓存开关无关（核心）
    base = {
        "width": 864, "height": 480, "ref_image_size": "match",
        "sigmas": torch.tensor([1.0, 0.5]), "sampler": "dpmpp_2m",
        "model": "unet.safetensors", "handoff_mode": "latent",
        "seed": 307045,
    }
    segment = {"prompt": "a", "length": 39, "resolved_seed": 1,
               "images": [], "videos": [], "video_audios": [], "audios": []}
    k0 = SESS.segment_key(base, segment, 39, None)
    k1 = SESS.segment_key(dict(base, seg_cache_codec="ffv1"), segment, 39, None)
    k2 = SESS.segment_key(dict(base, seg_cache_codec="h264"), segment, 39, None)
    check(k0 == k1 == k2,
          "seg_cache_codec 进了 segment_key —— 切开关会让既有会话全量重渲")
    k3 = SESS.segment_key(base, segment, 39, "PREV")
    k4 = SESS.segment_key(dict(base, seg_cache_codec="ffv1"), segment, 39, "PREV")
    check(k3 == k4, "带前驱键时 seg_cache_codec 仍然进了 payload")

    print("FAIL:", FAIL if FAIL else "无 —— 全部通过")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

sys.exit(1 if FAIL else 0)
