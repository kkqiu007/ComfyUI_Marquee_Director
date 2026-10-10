# -*- coding: utf-8 -*-
"""无损帧缓存：给每一段落盘一份**像素精确**的边车。

要解决的问题
------------
段落在磁盘上是 H.264 crf10 的 mp4（``video_io.save_clip``）。这个文件同时承担
两份工作：

* 给人看 —— 单段预览、时间线缩略图、拖进剪辑软件；
* 给机器读 —— 拼接成片时 ``join`` 把它**解出来再编一次** crf14；单段修复 / 断点
  续渲时 ``_anchor_from_disk`` 也从它取锚点像素。

第二份工作是有代价的：crf10 解码出来的是 yuv420p 8-bit 的重建值，与模型真正吐出的
像素差约 1e-2（实测 40 帧随机噪声样本，crf10 往返最大误差 0.0107，约 39 dB）。
拼接成片因此带着**两代**有损压缩：段一次、成片一次。音频早就有这条问题的解法了
—— ``exact_audio`` 写一份 float-PCM 的 WAV 边车，拼接读边车而不读 AAC —— 视频没有。

这里补的就是视频那一份：**FFV1**（intra-only 无损）装在 Matroska 里。

为什么是 gbrp16le 而不是 8-bit
------------------------------
参考包 ``ComfyUI_MiniMaxH3_Director``（Apache-2.0）的同类做法用 8-bit ``rgb24``。
8-bit 只能保证「编解码器不再损失」，但 ComfyUI 交给我们的是 float32，压到 8-bit
本身就是一次量化。FFV1 level 3 接受 16-bit 平面 RGB（``gbrp16le``），量化误差
1/65535 ≈ 1.5e-5 —— 实测往返最大误差 7.7e-06，比 8-bit 边车好 1000 倍，比 crf10
mp4 好三个数量级，代价约为文件大小的 1.6 倍。

另外两点与参考包不同，是刻意的：

* 用本机已依赖的 **PyAV** 编解码，不 ``subprocess`` 调 ffmpeg 可执行文件 ——
  少一个外部二进制依赖，也少一处命令行拼接。
* ``av`` 的 FFV1 编码器**不接受** ``rgb24`` / ``gbrp``（8-bit 平面），只接受
  ``gbrp16le``；这不是可选优化，是这条路能走通的前提。

默认**关**。打开它改变的是成片本身（少一代损失），而本包的规矩是「改成像的功能
默认关」（见 ``seam_remask``）；而且 8 GB 显存 / 40 GB 内存的机器上，每段要多占
几十到几百 MB 磁盘，这个决定该由用户做。开关在 H3Director 的「段缓存像素」控件。
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .common import FPS, log, warn

# log()/warn() 来自 common：全包只有那一处打前缀（见 common.LOG_PREFIX）。

#: 边车后缀：``seg_03.mp4`` 的无损像素在 ``seg_03.frames.mkv``。
SUFFIX = ".frames.mkv"

#: FFV1 level 3 下 av 实际接受的 16-bit 平面 RGB。
_PIX_FMT = "gbrp16le"
#: 解码回 packed 16-bit RGB；与 _PIX_FMT 之间只是打包/解包，数值不变。
_READ_FMT = "rgb48le"

#: 16-bit 的满量程，写读两侧共用同一个数，避免两边各写一份 65535 而漂开。
_FULL_SCALE = 65535.0


def frames_path(clip_path):
    """一条 mp4 对应的无损边车路径（与 ``_wav_path`` / ``latent_path`` 同一套命名）。"""
    return os.path.splitext(clip_path)[0] + SUFFIX


def enabled_for(settings):
    """本次渲染是否要写无损边车。

    只认 ``seg_cache_codec == "ffv1"``；键不存在（老工作流、别的调用方）就是关。
    这个键**不进** ``session.segment_key`` 的 payload —— 换开关不该让既有会话全部
    重渲：开关只影响**以后真烧显卡重渲的那些段**，复用的段照旧从 mp4 读。
    """
    return str((settings or {}).get("seg_cache_codec") or "h264").strip().lower() == "ffv1"


def _frame_to_u16(frame):
    """单帧 float [0,1]（或 uint8）→ uint16 (H, W, 3) numpy。

    逐帧转换，不一次摊开整段：一段 480×864×277 的 float32 中间量就是 1.4 GB，
    而编码器一帧一帧吃。
    """
    if not isinstance(frame, torch.Tensor) or frame.ndim != 3:
        raise ValueError("需要 (H, W, 3) 的单帧，收到 %s ndim=%s"
                         % (type(frame).__name__, getattr(frame, "ndim", None)))
    x = frame.detach().to("cpu")
    if x.shape[-1] > 3:
        x = x[..., :3]
    if x.dtype == torch.uint8:
        # 8 -> 16 bit 的精确扩张：255 -> 65535。float32 能精确表示这两个数，
        # 所以这一步没有舍入。
        scaled = x.float() * 257.0
    else:
        scaled = x.float().clamp(0.0, 1.0) * _FULL_SCALE
    return np.ascontiguousarray(
        scaled.round().clamp(0, _FULL_SCALE).to(torch.int32).numpy().astype(np.uint16))


def write(clip_path, images, fps=FPS):
    """把 ``images`` 原样写成 ``frames_path(clip_path)``。失败抛异常，由调用方决定响度。

    原子落盘（先 ``.tmp.mkv`` 再 ``os.replace``）：崩在写一半不能留下一个看起来
    完整、实际少几帧的边车 —— ``load_clip`` 会优先信它，那等于把截断的段当成
    完好段接进成片。
    """
    import av

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("需要 (N, H, W, 3) 的 IMAGE 批次，收到 %s ndim=%s"
                         % (type(images).__name__, getattr(images, "ndim", None)))
    n = int(images.shape[0])
    if n <= 0:
        raise ValueError("没有帧可写")
    h, w = int(images.shape[1]), int(images.shape[2])
    target = frames_path(clip_path)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    tmp = target + ".tmp.mkv"
    rate = int(round(float(fps) or FPS))
    container = None
    try:
        container = av.open(tmp, "w", format="matroska")
        stream = container.add_stream("ffv1", rate=rate, options={"level": "3"})
        stream.width, stream.height, stream.pix_fmt = w, h, _PIX_FMT
        for index in range(n):
            frame = av.VideoFrame.from_ndarray(_frame_to_u16(images[index]),
                                               format=_PIX_FMT)
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        os.replace(tmp, target)
    except Exception:
        if container is not None:
            try:
                container.close()
            except Exception:           # 拆容器失败不该盖住真正的错误
                pass
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return target


def read(clip_path):
    """读回 ``(N, H, W, 3)`` float32 [0,1]。

    返回值与 ``VideoFromFile(...).get_components().images`` 同一套约定（float、
    [0,1]、NHWC），所以所有既有调用方不需要知道自己读的是哪个文件。
    """
    import av

    path = frames_path(clip_path)
    frames = []
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(stream):
            packed = torch.from_numpy(np.ascontiguousarray(
                frame.to_ndarray(format=_READ_FMT)))
            frames.append(packed.to(torch.float32).div_(_FULL_SCALE))
    if not frames:
        raise ValueError("边车 %s 解出 0 帧" % os.path.basename(path))
    return torch.stack(frames, dim=0)


def read_if_present(clip_path, expect_frames=None):
    """有边车就读它，否则返回 None；读坏了也返回 None 并大声记录。

    「读坏了」在这里必须是**可恢复**的：边车是缓存，不是唯一数据源，同一段的 mp4
    就在旁边。让一个坏边车打死整条链不值得。

    ``expect_frames`` 是 mp4 的帧数。**必须给**（调用方手上都有），因为截断的
    Matroska 不报错 —— 它安静地少解几帧。测试里实测到的：把边车砍到 1/3 大小，
    ``read()`` 照常返回一个 4 帧的批次，而这一段本该有 6 帧。拼接按 mp4 的帧数算
    裁切点，拿一份短了的像素去拼接 = 成片比剧本短，而且没有任何一处会抱怨。
    少帧与多帧都判失败，回退 mp4。
    """
    path = frames_path(clip_path)
    if not os.path.exists(path):
        return None
    try:
        frames = read(clip_path)
    except Exception as exc:
        warn("无损边车 %s 读不出来（%s: %s）—— 这段回退到有损 mp4",
                    os.path.basename(path), type(exc).__name__, exc)
        return None
    if expect_frames is not None and frames.shape[0] != int(expect_frames):
        warn("无损边车 %s 有 %d 帧，mp4 有 %s 帧 —— 边车不完整"
                    "（截断？磁盘满？），这段回退到有损 mp4",
                    os.path.basename(path), int(frames.shape[0]), int(expect_frames))
        return None
    return frames


def remove(clip_path):
    """删掉一条 mp4 对应的边车（不存在就什么都不做）。"""
    path = frames_path(clip_path)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def sidecar_summary(directory):
    """目录里边车的数量与总字节，给运行报告用。"""
    count = 0
    total = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0, 0
    for name in names:
        if not name.endswith(SUFFIX):
            continue
        try:
            total += os.path.getsize(os.path.join(directory, name))
        except OSError:
            continue
        count += 1
    return count, total
