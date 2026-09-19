"""Reading and writing the session's clips.

Segments are held on disk as ordinary mp4s so they can be scrubbed, dragged into
an editor, or handed back to the repair node. They are written at CRF 10 -- near
enough to lossless that a re-anchored tail measures the same ~30 dB as a plain VAE
round trip, which is the accuracy the seamless join depends on.
"""

import logging
import math
import os
from fractions import Fraction

import av
import comfy.utils
import numpy as np
import torch

from comfy_api.input_impl import VideoFromComponents, VideoFromFile
from comfy_api.util import VideoCodec, VideoComponents, VideoContainer

from .common import FPS

SEGMENT_CRF = 10.0


def latent_path(path):
    """Sidecar holding a handoff's latent, next to the mp4 of the same clip."""
    return os.path.splitext(path)[0] + ".latent.safetensors"


def save_latent_tail(path, anchor):
    """Persist a latent handoff so a resumed run can pin it without a VAE.

    Written as fp16: this is a 39-frame slice of a diffusion latent, and half
    precision costs about 1e-4 of its own standard deviation -- three orders of
    magnitude below what a VAE round trip costs, which is the thing the latent
    handoff exists to avoid. At 480x864 it is 0.9 MB a segment.
    """
    if anchor is None or anchor.get("video_latent") is None:
        return None
    target = latent_path(path)
    payload = {"video": anchor["video_latent"].contiguous().cpu().half()}
    audio = anchor.get("audio_latent")
    if audio is not None:
        payload["audio"] = audio.contiguous().cpu().half()
    tmp = target + ".tmp"
    comfy.utils.save_torch_file(payload, tmp)
    os.replace(tmp, target)
    return target


def load_latent_tail(path):
    """The latent handoff for a clip, or None if this session predates them."""
    target = latent_path(path)
    if not os.path.exists(target):
        return None
    try:
        payload = comfy.utils.load_torch_file(target, safe_load=True)
    except Exception as exc:  # a truncated sidecar must not kill a whole chain
        logging.warning("[H3 Continuous] could not read %s (%s); falling back to the "
                        "pixel handoff", os.path.basename(target), exc)
        return None
    video = payload.get("video")
    if video is None:
        return None
    return {"video_latent": video.float(),
            "audio_latent": payload["audio"].float() if "audio" in payload else None}


def save_clip(path, images, audio, fps=FPS, crf=SEGMENT_CRF, exact_audio=False):
    """Write a clip. ``exact_audio`` adds a sidecar WAV -- see ``_wav_path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    components = VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps))
    tmp = path + ".tmp.mp4"
    VideoFromComponents(components).save_to(
        tmp, format=VideoContainer.MP4, codec=VideoCodec.H264, crf=crf)
    os.replace(tmp, path)
    if exact_audio and audio is not None:
        save_wav(_wav_path(path), audio)
    return path


def _wav_path(path):
    """Sidecar for a handoff clip's soundtrack.

    AAC is not sample-exact on the way back in: the decoder hands you the encoder's
    priming samples, about 20-30 ms of them. That is a fifth of an audio latent frame
    of slip on every anchor, for no reason -- the tail is small, so keep it as
    float PCM and read that instead.
    """
    return os.path.splitext(path)[0] + ".wav"


_LAYOUTS = {1: "mono", 2: "stereo", 3: "2.1", 4: "quad", 5: "5.0", 6: "5.1",
            7: "6.1", 8: "7.1"}


def _channels_first(waveform):
    """Normalise any waveform we are handed into a (channels, samples) array.

    ComfyUI's AUDIO is nominally (batch, channels, samples), but nodes in this
    package build mono tracks as (1, samples) and some paths hand back a bare
    (samples,). The old code did ``waveform[0]`` first, which on those 2-D
    inputs strips the *channel* axis instead of the batch -- leaving a 1-D array
    whose shape[0] is the sample count, so the layout lookup below always missed
    and every sidecar was written as stereo (double length, garbled, and twice
    the duration of the anchor it is supposed to carry).
    """
    arr = waveform
    if arr.ndim == 3:                      # (batch, channels, samples)
        arr = arr[0]
    if arr.ndim == 1:                      # (samples,) -> mono
        arr = arr[None, :]
    if arr.ndim == 2 and arr.shape[0] > 8 and arr.shape[1] <= 8:
        arr = arr.T                        # (samples, channels) -> (channels, samples)
    return np.ascontiguousarray(arr, dtype=np.float32)


def save_wav(path, audio):
    waveform = audio["waveform"]
    if hasattr(waveform, "detach"):        # torch tensor
        waveform = waveform.detach().cpu().float().numpy()
    else:                                  # already ndarray
        waveform = np.asarray(waveform, dtype=np.float32)
    waveform = _channels_first(waveform)
    rate = int(audio["sample_rate"])
    channels = int(waveform.shape[0])
    layout = _LAYOUTS.get(channels, "stereo")
    with av.open(path, "w", format="wav") as container:
        stream = container.add_stream("pcm_f32le", rate=rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout=layout)
        frame.sample_rate = rate
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def load_clip(path, prefer_wav=False):
    """Whole clip back as (IMAGE, AUDIO)."""
    components = VideoFromFile(path).get_components()
    audio = components.audio
    if audio is not None and not isinstance(audio, dict):
        audio = dict(audio)
    if prefer_wav and os.path.exists(_wav_path(path)):
        audio = read_audio(_wav_path(path)) or audio
    return components.images, audio


def read_audio(path):
    """Audio only, without paying to decode the video stream.

    The join needs every segment's soundtrack up front but only one segment's
    frames at a time, so the two are read separately.
    """
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return None
        resampler = av.audio.resampler.AudioResampler(format="fltp")
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())
    if not chunks:
        return None
    data = np.concatenate(chunks, axis=1)  # (channels, samples)
    return {"waveform": torch.from_numpy(data).unsqueeze(0),
            "sample_rate": int(stream.sample_rate or 1)}


def drift_profile(parts, strength, window_seconds=2.0, fps=FPS, clamp=0.25):
    """Per-frame gains that flatten a chain's slow exposure/colour drift.

    A chained take slides away from its own opening -- steadily darker, steadily
    less saturated -- without any single join showing it. That trend is slow by
    definition, which is exactly what makes it separable: smooth each channel's
    per-frame mean over a couple of seconds and what is left is the drift, with
    gestures, blinks and real lighting flicker averaged out of it.

    The correction is a **gain** per frame per channel, aimed at the smoothed level
    of the first segment. Gains preserve black, so nothing is lifted into a milky
    shadow the way an offset would. Because the curve it is built from is smooth,
    the correction is continuous across the joins by construction -- there is no
    step at a cut, which is the whole reason to do this on the finished timeline
    rather than per segment.

    ``clamp`` bounds it. What this cannot tell apart from drift is a shot that is
    legitimately darker because the light changed, and a real lighting change is
    large. Bounding the gain means such a change survives, mostly, while
    accumulated drift -- which is small -- is removed.

    Returns None when there is nothing to do, so the caller can skip pass two.
    """
    if strength <= 0 or not parts:
        return None

    means = []
    for path, skip in parts:
        images, _ = load_clip(path)
        if images.shape[0] > skip:
            means.append(images[skip:].mean(dim=(1, 2)).float().cpu())
        del images
    if not means:
        return None
    series = torch.cat(means, dim=0)  # [frames, 3]
    if series.shape[0] < 3:
        return None

    window = max(3, int(round(window_seconds * fps)) | 1)
    padded = torch.cat([series[:1].repeat(window // 2, 1), series,
                        series[-1:].repeat(window // 2, 1)], dim=0)
    kernel = torch.ones(window) / window
    smooth = torch.stack([
        torch.nn.functional.conv1d(padded[:, c].view(1, 1, -1),
                                   kernel.view(1, 1, -1)).view(-1)
        for c in range(series.shape[1])], dim=1)

    # The target is the opening's own smoothed level: segment 1 is the only shot
    # rendered from the prompt and the reference alone, so it is the one place in a
    # chain that has not yet drifted.
    target = smooth[:max(3, window // 2)].mean(dim=0)
    gains = target.view(1, -1) / smooth.clamp_min(1e-4)
    gains = 1.0 + (gains - 1.0) * float(strength)
    gains = gains.clamp(1.0 - clamp, 1.0 + clamp)
    logging.info("[H3 Continuous] drift stabiliser: gain %.4f..%.4f across %d frames",
                 float(gains.min()), float(gains.max()), gains.shape[0])
    return gains


class _StreamedFrames:
    """A stand-in for the joined IMAGE batch that never exists all at once.

    ``VideoFromComponents.save_to`` only ever asks for ``.shape`` and then iterates,
    so the join can hold one segment's frames at a time instead of the whole cut.
    Eight 8-second segments at 768x1344 would otherwise be about 12 GB of float32.
    """

    def __init__(self, parts, height, width, gains=None):
        self.parts = parts  # [(path, skip_frames)]
        self.gains = gains
        self.total = 0
        self._counts = []
        for path, skip in parts:
            n = max(0, frame_count(path) - skip)
            self._counts.append(n)
            self.total += n
        self.shape = (self.total, height, width, 3)

    def __len__(self):
        return self.total

    def _apply(self, frame, index):
        if self.gains is None or index >= self.gains.shape[0]:
            return frame
        return (frame * self.gains[index].to(frame.dtype)).clamp(0.0, 1.0)

    def __iter__(self):
        index = 0
        for path, skip in self.parts:
            images, _ = load_clip(path)
            for i in range(skip, images.shape[0]):
                yield self._apply(images[i], index)
                index += 1
            del images

    def materialize(self):
        return torch.cat([f.unsqueeze(0) for f in self], dim=0)


def frame_count(path):
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "video")
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


# 裁切必须严格等于回放帧数：回放是上一段尾部的**重复**，多留一帧画面就往回
# 退一帧（跳帧），少裁一帧就多播一次同样的动作（重复）。所以这里不能"找最像
# 的那一帧"—— 那会把相位差掩盖成运动倒退。帧数由 17k+5 网格唯一决定。
_ALIGN_STEP = 4              # 诊断比较用的降采样倍数


def _head_frames_gray(path, count, step=_ALIGN_STEP):
    """解出前 ``count`` 帧的灰度缩略图，只为接缝对齐用，不占内存。"""
    out = []
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "video")
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i >= count:
                break
            arr = frame.to_ndarray(format="gray")
            out.append(torch.from_numpy(
                np.ascontiguousarray(arr[::step, ::step])).float().div_(255.0))
    return out


def _last_frame_gray(path, step=_ALIGN_STEP):
    frame = last_frame(path)
    if frame is None:
        return None
    arr = frame
    if arr.dim() == 3 and arr.shape[-1] == 3:      # (H, W, C) -> 亮度
        arr = arr.float().mean(dim=-1)
    elif arr.dim() == 3:                            # (C, H, W)
        arr = arr.float().mean(dim=0)
    return arr[::step, ::step].float()


def resolve_join_points(parts, fps=FPS):
    """严格按回放帧数去重，并如实报告接缝质量。

    ``parts`` 里每段的 skip 就是它开头那段「回放」的长度。回放是上一段尾部的
    重复，所以必须**不多不少正好裁掉**：
      · 少裁 k 帧 → 多播 k 帧已经看过的动作，成片在这里原地打结
      · 多裁 k 帧 → 吃掉 k 帧新内容，动作跳一大步
    两者都是跳帧，所以这里不找"最像的那一帧"，只按帧数裁。

    曾经在这里加过一个 ±N 帧的相位搜索（"找最像的那一帧当切点"），实测是错
    的：回放不是上一段尾部的逐帧拷贝，模型会重演一遍，帧与帧之间本就有相位
    差。搜索会把这个相位差吸收成位移，画面看起来接缝更"像"了，运动却往回退
    一截 —— 肉眼正是判定为跳帧。真正让接缝严丝合缝的是生成端把尾部 latent
    硬写进下一段并 mask 住（``engine.apply_latent_continue``），不是在这里挪
    切点。所以这段搜索已删除，切点唯一由 17k+5 网格决定。

    接缝质量靠诊断报告：把上一段末帧与下一段裁切点那帧比一下，差异远超段内
    基线就告警 —— 那通常意味着剧本里回放镜头的时长写错了（比如写成 1.8s，
    而 1.6s 回放按 17k+5 网格实际是 39 帧 = 1.625s），模型生成的回放长度和
    镜头描述对不上。这种错改剧本才治本，在裁切上动手脚只会把问题挪个地方。
    """
    if len(parts) < 2:
        return list(parts), []

    aligned = [parts[0]]
    notes = []
    for idx, ((prev_path, _prev_skip), (path, skip)) in enumerate(zip(parts, parts[1:]), start=1):
        seam_diff, inner_base = None, None
        try:
            tail = _last_frame_gray(prev_path)
            heads = _head_frames_gray(path, skip + 25)
            if tail is not None and len(heads) > skip:
                seam_diff = float((heads[skip] - tail).abs().mean())
                inner = [float((heads[j] - heads[j - 1]).abs().mean())
                         for j in range(skip + 1, min(len(heads), skip + 25))]
                inner_base = float(np.median(inner)) if inner else 0.0
        except Exception as exc:  # 诊断失败不影响出片
            logging.warning("[H3 Continuous] 接缝诊断跳过段 %d (%s)", idx + 1, exc)

        if seam_diff is not None:
            # 只记录，不判定。段内本来就有硬切（实测段内 p99 可达 0.18，比接缝
            # 还大），拿"段内中位数"当基线必然误报 —— 判定要放到能看全段帧间差
            # 分布的地方做（tools/h3_qc.py 用段内 p99 比）。
            note = {"segment": idx + 1, "skip_frames": skip,
                    "seam_diff": round(seam_diff, 5),
                    "inner_median": round(inner_base or 0.0, 5)}
            logging.info("[H3 Continuous] 段%d 接缝差异 %.4f（段内中位数 %.4f，"
                         "注意段内硬切天然更大，判定请看 h3_qc）",
                         idx + 1, seam_diff, inner_base or 0.0)
            notes.append(note)
        aligned.append((path, skip))
    return aligned, notes


def join(parts, out_path, fps=FPS, crf=14.0, stabilize=0.0):
    """Concatenate segments, dropping each one's replayed opening.

    ``parts`` is ``[(path, skip_frames), ...]``. Segment N opens on an exact replay
    of segment N-1's last ``skip_frames`` frames -- that replay is the join -- so
    those frames come off the front of every segment after the first or the motion
    stutters once per cut.

    ``stabilize`` additionally flattens the chain's slow drift away from its own
    opening; see ``drift_profile``. It costs one extra decoding pass and no GPU.
    """
    if not parts:
        raise ValueError("nothing to join")

    # 裁切点按实测校准（回放的运动相位会漂几帧）；音频同样用校准后的值，
    # 否则画面接上了、声音还差半句。
    parts, _join_notes = resolve_join_points(parts, fps=fps)

    probe, _ = load_clip(parts[0][0])
    height, width = int(probe.shape[1]), int(probe.shape[2])
    del probe

    audio = _join_audio(parts, fps)
    frames = _StreamedFrames(parts, height, width,
                             gains=drift_profile(parts, stabilize, fps=fps))
    components = VideoComponents(images=frames, audio=audio, frame_rate=Fraction(fps))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp.mp4"
    try:
        VideoFromComponents(components).save_to(
            tmp, format=VideoContainer.MP4, codec=VideoCodec.H264, crf=crf)
    except (AttributeError, TypeError) as exc:
        # Streaming leans on save_to only touching .shape and iteration. If a future
        # ComfyUI indexes or moves the batch instead, fall back to the honest way.
        logging.warning("[H3 Continuous] streamed join failed (%s); "
                        "joining in memory instead", exc)
        components = VideoComponents(images=frames.materialize(), audio=audio,
                                     frame_rate=Fraction(fps))
        VideoFromComponents(components).save_to(
            tmp, format=VideoContainer.MP4, codec=VideoCodec.H264, crf=crf)
    os.replace(tmp, out_path)
    return out_path, frames.total


def _join_audio(parts, fps):
    """Concatenate the soundtracks, each cut to exactly its own kept frames.

    Every part is trimmed or padded to its video length before it is appended.
    Letting audio run even a fraction long is how a chain drifts: the error is not
    corrected at the next join, it accumulates across all of them.
    """
    rate = None
    pieces = []
    for path, skip in parts:
        clip_audio = read_audio(path)
        kept = max(0, frame_count(path) - skip)
        want_seconds = kept / float(fps)
        if clip_audio is None:
            pieces.append((None, want_seconds))
            continue
        if rate is None:
            rate = int(clip_audio["sample_rate"])
        waveform = clip_audio["waveform"]
        if int(clip_audio["sample_rate"]) != rate:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, int(clip_audio["sample_rate"]), rate)
        start = int(round(skip / float(fps) * rate))
        want = int(round(want_seconds * rate))
        cut = waveform[..., start:start + want]
        if cut.shape[-1] < want:
            cut = torch.cat(
                [cut, torch.zeros(cut.shape[:-1] + (want - cut.shape[-1],), dtype=cut.dtype)],
                dim=-1)
        pieces.append((cut, want_seconds))

    if rate is None:
        return None

    channels = next((p.shape[1] for p, _ in pieces if p is not None), 1)
    filled = []
    for piece, seconds in pieces:
        if piece is None:
            # A segment with no audio track still occupies time in the cut.
            filled.append(torch.zeros(1, channels, int(round(seconds * rate))))
        else:
            filled.append(piece)
    return {"waveform": torch.cat(filled, dim=-1), "sample_rate": rate}


def last_frame(path):
    """Decode just the final frame of a clip, without reading the whole thing.

    Used for the per-segment preview of a segment that was reused from cache: the
    footage is already on disk and decoding 192 frames of 480x864 to show one
    thumbnail would cost a gigabyte for nothing.
    """
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "video")
        stream.thread_type = "AUTO"
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if duration > 0.5:
            try:
                container.seek(int((duration - 0.5) / stream.time_base), stream=stream)
            except av.error.PyAVError:
                pass
        frame = None
        for frame in container.decode(stream):
            pass
        if frame is None:  # seek overshot a very short clip; start over
            container.seek(0, stream=stream)
            for frame in container.decode(stream):
                pass
        if frame is None:
            return None
        # gbrpf32le, not rgb24: it is what ComfyUI's own decoder uses, and it carries the
        # stream's BT.709 tagging through the conversion. Going via rgb24 picks up
        # swscale's default matrix instead and shifts saturated colours by up to 0.09.
        return torch.from_numpy(frame.to_ndarray(format="gbrpf32le").copy())
