"""Reading and writing the session's clips.

Segments are held on disk as ordinary mp4s so they can be scrubbed, dragged into
an editor, or handed back to the repair node. They are written at CRF 10 -- near
enough to lossless that a re-anchored tail measures the same ~30 dB as a plain VAE
round trip, which is the accuracy the seamless join depends on.

"Near enough" is doing work in that sentence, though: the join decodes those mp4s
and encodes them *again* at CRF 14, so the finished film carries two generations of
loss per segment while the raw decode only ever saw one. ``lossless_frames`` on
``save_clip`` puts a bit-exact FFV1 sidecar next to the mp4 and ``load_clip`` reads
that instead, dropping the film back to the single encode it was always going to
need. Audio already solved this with the WAV sidecar; this is the video half.
"""

import math
import os
from fractions import Fraction

import av
import comfy.utils
import numpy as np
import torch

from comfy_api.input_impl import VideoFromComponents, VideoFromFile
from comfy_api.util import VideoCodec, VideoComponents, VideoContainer

from . import audio_master
from . import frames_cache
from .common import FPS, log, warn

SEGMENT_CRF = 10.0

# ---------------------------------------------------------------------------
# 接缝音频焊接（2026-09-29，吸收 ComfyUI-H3-Multishot v2.7 的 40 ms seam weld）
# ---------------------------------------------------------------------------
#: 每侧淡出/淡入宽度（毫秒）。默认 20 + 20 = 40 ms 总过渡，与 H3-Multishot 的
#: seam weld 同宽；设 0（或 MARQUEE_SEAM_WELD_MS=0）关闭。
SEAM_WELD_MS = float(os.environ.get("MARQUEE_SEAM_WELD_MS", "20") or 20)


def _weld_seam_fades(pieces, rate):
    """在相邻段的接缝上做**原位**余弦微淡入淡出，杀掉硬拼接的咔哒声。

    H3-Multishot 的原版是 40 ms 等功率交叉淡化（overlap-add，音频总长缩短一个
    焊宽）。这里的接缝音频是按帧**精确**对齐的（``_join_audio`` 的结构性承诺：
    音频哪怕跑长一点，误差也会跨接缝累积），不能缩短，所以改成两侧各
    ``SEAM_WELD_MS`` 的原位淡出/淡入 —— 总过渡宽度相同，总长一个采样都不动。

    段尾 1.6 s 无台词的设计让接缝基本落在呼吸/配乐里；这里处理的是配乐与
    环境声在拼接点的波形相位不连续 —— 那是每道接缝一声「咔」的来源。
    """
    if SEAM_WELD_MS <= 0 or len(pieces) < 2:
        return pieces
    width = int(round(rate * SEAM_WELD_MS / 1000.0))
    if width <= 0:
        return pieces
    welded = 0
    for i, piece in enumerate(pieces):
        if piece is None or not getattr(piece, "is_floating_point", lambda: True)():
            continue
        n = piece.shape[-1]
        # 一侧焊宽最多吃掉本段的 1/4：极短段不被淡成静音。
        w_cap = max(0, n // 4)
        w_in = min(width, w_cap) if i != 0 else 0              # 非首段：开头淡入
        w_out = min(width, w_cap) if i != len(pieces) - 1 else 0  # 非末段：结尾淡出
        if w_in:
            t = torch.arange(w_in, dtype=piece.dtype)
            piece[..., :w_in] = piece[..., :w_in] * (
                0.5 - 0.5 * torch.cos(math.pi * (t + 1) / (w_in + 1)))
            welded += 1
        if w_out:
            t = torch.arange(w_out, dtype=piece.dtype)
            piece[..., n - w_out:] = piece[..., n - w_out:] * (
                0.5 - 0.5 * torch.cos(math.pi * (w_out - t) / (w_out + 1)))
            welded += 1
    if welded:
        log("join audio: seam weld on %d join(s) "
                     "(%.0f ms cosine fade per side)",
                     sum(1 for _ in pieces[1:]), SEAM_WELD_MS)
    return pieces


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
        warn("could not read %s (%s); falling back to the "
                        "pixel handoff", os.path.basename(target), exc)
        return None
    video = payload.get("video")
    if video is None:
        return None
    return {"video_latent": video.float(),
            "audio_latent": payload["audio"].float() if "audio" in payload else None}


def _master_audio(audio):
    """Bring ``audio`` to delivery level, returning ``(audio_or_input, target)``.

    ``target`` is the per-segment loudness this take was mastered to (see
    ``audio_master.profile_target``) -- the caller writes it next to the clip so
    the join reads back the *same* number and never re-classifies the already
    mastered audio. When mastering is skipped, ``target`` is ``None``.

    The soundtrack is mastered **here**, at save time, and not only at the join.
    The finished film is not the reason -- the join handles that either way. The
    reason is the single segment: segments are reviewed one at a time, and a raw
    one measures 7-8 LU below delivery level, which is exactly what reads as
    thin and weak on its own. See ``audio_master.master_segment``.

    Never mutates its input. The caller's ``audio`` stays the raw signal, and
    that matters: ``take_tail`` cuts the handoff from it, and the handoff is the
    next segment's conditioning input rather than an excerpt for listening.
    """
    if not audio_master.ENABLED or not isinstance(audio, dict):
        return audio, None
    waveform = audio.get("waveform")
    rate = int(audio.get("sample_rate") or 0)
    if waveform is None or rate <= 0 or not hasattr(waveform, "ndim"):
        return audio, None
    try:
        target = audio_master.profile_target(waveform, rate)
        mastered = _master_tensor(waveform, rate, audio_master.master_segment,
                                  target_lufs=target)
    except Exception as exc:
        # A finished render is worth more than a loud one: anything unexpected
        # leaves the soundtrack exactly as the model produced it.
        warn("segment mastering skipped (%s: %s)",
                        type(exc).__name__, exc)
        return audio, None
    if mastered is waveform:          # _master_tensor declines rather than raise
        return audio, None
    out = dict(audio)
    out["waveform"] = mastered
    return out, target


def save_clip(path, images, audio, fps=FPS, crf=SEGMENT_CRF, exact_audio=False,
              master_audio=True, lossless_frames=False):
    """Write a clip.

    ``exact_audio`` adds a sidecar WAV -- see ``_wav_path``.

    ``master_audio`` brings the soundtrack to delivery level *before* the mux, so
    the mp4 and the sidecar carry the same signal and a single segment is already
    listenable when it is opened on its own. Pass ``False`` for a handoff tail:
    that file is the next segment's conditioning input, not an excerpt, and
    mastering it would move the condition.

    ``lossless_frames`` adds the FFV1 sidecar next to the mp4 -- see
    ``frames_cache``. The invariant that makes that sidecar safe to read
    preferentially is enforced here: **a clip file and its sidecar are always
    written together or not at all.** Turning the option off re-saves a segment
    without deleting the old sidecar would leave ``load_clip`` serving last
    take's pixels for this take's mp4 -- a silently wrong render, not a
    degraded one.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lufs_target = None
    if audio is not None and master_audio:
        audio, lufs_target = _master_audio(audio)
    components = VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps))
    tmp = path + ".tmp.mp4"
    VideoFromComponents(components).save_to(
        tmp, format=VideoContainer.MP4, codec=VideoCodec.H264, crf=crf)
    os.replace(tmp, path)
    # 把这段的母带响度目标写进 .lufs 伴生文件：拼接时原样读回，save 与 join
    # 永远用同一个目标，避免对已母带化的音频二次分档、重复处理。
    # 本次没母带化（无音频/开关关/跳过）时必须**清掉**上一 take 的残留 ——
    # 与帧边车同一条原则：旧伴生会把上一 take 的目标钉在这一 take 的音频上。
    if lufs_target is not None:
        _write_lufs_target(path, lufs_target)
    else:
        _drop_lufs_target(path)
    if exact_audio and audio is not None:
        save_wav(_wav_path(path), audio)
    if lossless_frames:
        try:
            frames_cache.write(path, images, fps=fps)
        except Exception as exc:
            # The mp4 is on disk and correct, so the segment is fine -- losing
            # the sidecar costs the join one generation of loss, nothing more.
            warn("lossless frame cache skipped for %s "
                            "(%s: %s)", os.path.basename(path),
                            type(exc).__name__, exc)
            frames_cache.remove(path)
    else:
        frames_cache.remove(path)
    return path


def _wav_path(path):
    """Sidecar for a handoff clip's soundtrack.

    AAC is not sample-exact on the way back in: the decoder hands you the encoder's
    priming samples, about 20-30 ms of them. That is a fifth of an audio latent frame
    of slip on every anchor, for no reason -- the tail is small, so keep it as
    float PCM and read that instead.
    """
    return os.path.splitext(path)[0] + ".wav"


def _lufs_path(path):
    """Companion that records the per-segment delivery loudness.

    Written by ``save_clip`` whenever a segment is mastered, read by the join so
    it can match each piece against the *same* target it was saved with. Without
    it the join would re-classify the already-mastered audio -- whose spectrum
    the chain has reshaped -- and could double-process a segment.
    """
    return os.path.splitext(path)[0] + ".lufs"


def _write_lufs_target(path, target_lufs):
    """Persist the per-segment loudness target next to the clip."""
    try:
        with open(_lufs_path(path), "w", encoding="utf-8") as fh:
            fh.write("%.3f" % float(target_lufs))
    except OSError as exc:
        warn("could not write lufs target for %s (%s)",
                        os.path.basename(path), exc)


def _read_lufs_target(path):
    """Read the per-segment loudness target, or None when absent/garbled."""
    p = _lufs_path(path)
    if not os.path.exists(p):
        return None
    try:
        return float(open(p, "r", encoding="utf-8").read().strip())
    except (OSError, ValueError):
        return None


def _drop_lufs_target(path):
    """Remove a stale companion; absence is the normal case, so never raise."""
    try:
        os.remove(_lufs_path(path))
    except OSError:
        pass


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
    """Whole clip back as (IMAGE, AUDIO).

    The lossless FFV1 sidecar wins whenever it is there: it carries the exact
    frames the mp4 was encoded from, so reading it removes the segment's own
    generation of H.264 loss from everything downstream -- the join, and the
    pixel handoff a repair anchors on. Absent, incomplete or unreadable, this is
    the mp4 path unchanged; the frame count is checked against the mp4 because a
    truncated Matroska decodes *shorter* rather than louder.
    """
    frames = None
    if os.path.exists(frames_cache.frames_path(path)):
        frames = frames_cache.read_if_present(path, expect_frames=frame_count(path))
    if frames is not None:
        if prefer_wav and os.path.exists(_wav_path(path)):
            return frames, read_audio(_wav_path(path))
        return frames, read_audio(path)
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
    log("drift stabiliser: gain %.4f..%.4f across %d frames",
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
            warn("接缝诊断跳过段 %d (%s)", idx + 1, exc)

        if seam_diff is not None:
            # 只记录，不判定。段内本来就有硬切（实测段内 p99 可达 0.18，比接缝
            # 还大），拿"段内中位数"当基线必然误报 —— 判定要放到能看全段帧间差
            # 分布的地方做（tools/check_joins.py 用接缝前后局部窗口的中位数比）。
            note = {"segment": idx + 1, "skip_frames": skip,
                    "seam_diff": round(seam_diff, 5),
                    "inner_median": round(inner_base or 0.0, 5)}
            log("段%d 接缝差异 %.4f（段内中位数 %.4f，"
                         "注意段内硬切天然更大，判定请看 tools/check_joins.py）",
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
        warn("streamed join failed (%s); "
                        "joining in memory instead", exc)
        components = VideoComponents(images=frames.materialize(), audio=audio,
                                     frame_rate=Fraction(fps))
        VideoFromComponents(components).save_to(
            tmp, format=VideoContainer.MP4, codec=VideoCodec.H264, crf=crf)
    os.replace(tmp, out_path)
    return out_path, frames.total


def _resample_waveform(waveform, sr_in, sr_out):
    """(batch, ch, samples) float tensor: ``sr_in`` → ``sr_out``，dtype/device 保留。

    ★ scipy 而不是 torchaudio（2026-10-18 第三轮自查，借鉴 ComfyUI 核心
      2026-09-22 提交 b16023b「Remove torchaudio dependency」的方向）：核心已
      把 torchaudio 从依赖里拿掉，本 venv 里它还在，但 requirements 不再钉它
      —— 下一次 Desktop 重建环境它就没了，届时 ``import torchaudio`` 会在
      join 遇到混合采样率段时把整条成片路径打崩，报错点离根因十万八千里。
      scipy 已是 ``audio_master`` 的硬依赖；``resample_poly``（Kaiser 窗
      sinc，多相 FIR）与 ``torchaudio.functional.resample`` 同类，44100→32000
      这种整数比化简后直接可用。
    """
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out)) or 1
    x = waveform.detach().cpu().numpy()
    y = resample_poly(x, int(sr_out) // g, int(sr_in) // g, axis=-1)
    out = torch.from_numpy(np.ascontiguousarray(y))
    return out.to(device=waveform.device, dtype=waveform.dtype)


def _master_tensor(waveform, rate, entry, **kwargs):
    """Run one of ``audio_master``'s numpy entry points on a ComfyUI waveform.

    Takes and returns (batch, channels, samples) -- or (channels, samples), which
    some callers hand over. Returns the input **untouched** if the shapes do not
    line up, and raises only what the caller should see, so the caller can decide
    how loudly to complain.
    """
    batched = waveform.ndim == 3
    signal = waveform[0] if batched else waveform
    out = entry(signal.detach().cpu().numpy(), rate, **kwargs)
    if np.shape(out) != tuple(signal.shape):
        return waveform
    out = torch.from_numpy(np.ascontiguousarray(out))
    return out.unsqueeze(0) if batched else out


def _is_mastered_tensor(waveform, rate, target_lufs=None):
    """True when a ComfyUI waveform already sits at the delivery target.

    ``target_lufs`` is the per-segment target read from the clip's ``.lufs``
    companion (or classified at save time); passing it keeps this check and the
    mastering pass in agreement so a segment is never shaped twice. Same shape
    handling as ``_master_tensor``, and deliberately total: an unmeasurable
    signal answers *no*, so the caller masters it. That is the safe direction --
    the cost of mastering twice is a little shaping, the cost of not mastering is
    a segment 8 LU down in the finished film.
    """
    try:
        batched = waveform.ndim == 3
        signal = waveform[0] if batched else waveform
        return audio_master.is_mastered(signal.detach().cpu().numpy(), rate,
                                        target_lufs=target_lufs)
    except Exception:
        return False


def _join_audio(parts, fps):
    """Concatenate the soundtracks, each cut to exactly its own kept frames.

    Every part is trimmed or padded to its video length before it is appended.
    Letting audio run even a fraction long is how a chain drifts: the error is not
    corrected at the next join, it accumulates across all of them.

    Two things happen here beyond concatenation, both of them structural rather
    than cosmetic -- see ``audio_master`` for the measurements behind them:

    * Each segment is read from its **WAV sidecar** when one exists. The segment
      mp4 carries AAC, and reading that back means the soundtrack is encoded
      twice on its way to the finished film (once per segment, once for the
      join). The sidecars are already written for handoff tails; writing them for
      segments too removes one whole generation.
    * Each segment is loudness-matched **before** it is appended. H3 renders every
      segment in its own pass and the per-segment normaliser in ComfyUI never
      fires on this model, so the segments arrive 7.8 LU apart. A single gain over
      the joined track cannot repair that; it only moves the whole film.
    * Segments that are *already* at the target are left alone. They reach that
      state by being mastered in ``save_clip`` when they were written, which is
      what makes a single segment listenable before the film exists; re-shaping
      them here would spend their headroom a second time. ``is_mastered`` is the
      test, so a segment written by an older build -- still raw -- is still fixed.
    """
    rate = None
    pieces = []
    from_sidecar = 0
    for path, skip in parts:
        # Prefer the float-PCM sidecar: reading the mp4 means a second AAC
        # generation on the way to the finished film.
        sidecar = _wav_path(path)
        clip_audio = read_audio(sidecar) if os.path.exists(sidecar) else None
        if clip_audio is not None:
            from_sidecar += 1
        else:
            clip_audio = read_audio(path)
        kept = max(0, frame_count(path) - skip)
        want_seconds = kept / float(fps)
        if clip_audio is None:
            pieces.append((path, None, want_seconds))
            continue
        if rate is None:
            rate = int(clip_audio["sample_rate"])
        waveform = clip_audio["waveform"]
        if int(clip_audio["sample_rate"]) != rate:
            waveform = _resample_waveform(
                waveform, int(clip_audio["sample_rate"]), rate)
        start = int(round(skip / float(fps) * rate))
        want = int(round(want_seconds * rate))
        cut = waveform[..., start:start + want]
        if cut.shape[-1] < want:
            cut = torch.cat(
                [cut, torch.zeros(cut.shape[:-1] + (want - cut.shape[-1],), dtype=cut.dtype)],
                dim=-1)
        pieces.append((path, cut, want_seconds))

    if rate is None:
        return None

    log("join audio: %d/%d segment(s) from WAV sidecar "
                 "(no second AAC generation), %d from mp4",
                 from_sidecar, len(parts), len(parts) - from_sidecar)

    channels = next((p.shape[1] for _, p, _ in pieces if p is not None), 1)
    filled = []
    matched = 0
    # ★ 三元组带着各自的 ``path``：``.lufs`` 伴生文件是**每段一个**目标。这里
    #   曾直接引用上面循环遗留的 ``path`` 变量 —— 恒为最后一段的路径，于是
    #   全部段都读同一个目标：末段是对白（-16）时，前面的战场段（-8）会被
    #   判成「没母带过」而二次压到 -16，恰好抹平这次改动要引入的动态。
    for path, piece, seconds in pieces:
        if piece is None:
            # A segment with no audio track still occupies time in the cut.
            piece = torch.zeros(1, channels, int(round(seconds * rate)))
        target = _read_lufs_target(path)
        # 有伴生 = 保存时已母带化（已过 40 Hz 高通与 tonal shaping）。
        was_mastered = target is not None
        if target is None and audio_master.ENABLED:
            # 无伴生（旧版本产物，或成片后清理扫掉了伴生再重建音轨）：
            # **先吸附**到最近的规范档位 —— 已母带化的音频离 -8/-13/-16
            # 很近，吸附命中就按「已母带」对待，守卫放行或仅做电平校正；
            # 吸不上（离所有档都远）才是原始段，按音频自身分类走完整处理。
            # ★ 顺序是实测出来的（v17 重建，2026-10-18）：直接对已母带化
            #   音频跑 profile_target 会分错档 —— 一条裁剪后 -14.3 LUFS 的
            #   中性段被判战场档、回来烫 +5.9 dB，三条对白段被拉高 +3.4 dB。
            #   高通与 shaping 已经改过频谱，分类器看到的不是原始信号。
            try:
                np_piece = piece.detach().cpu().numpy()
                target = audio_master.snap_target(np_piece, rate)
                if target is None:
                    target = audio_master.profile_target(np_piece, rate)
                else:
                    was_mastered = True
            except Exception:
                target = None
        if audio_master.ENABLED:
            try:
                # Segments are mastered when they are saved, so most of them are
                # already at the target by the time they get here and re-shaping
                # them would only spend their headroom a second time. The guard
                # is what lets both passes coexist: a segment an older build
                # wrote is still raw and gets fixed, a current one is left alone.
                # ``target`` is the same number ``save_clip`` wrote, so the two
                # passes can never disagree on what "already mastered" means.
                if _is_mastered_tensor(piece, rate, target):
                    matched += 1
                else:
                    # ★ 已母带段只是裁剪后漂出容差时，只做**电平**校正
                    #   （2026-10-18 v17 成片实测发生 1 段）：带 .lufs 伴生的
                    #   段保存时已经过高通与 tonal shaping，再跑一遍 hp/shape
                    #   正是成片趟明文避免的双重处理 —— 两级 2 阶 40 Hz 串联
                    #   把真峰顶高约 1.1 dB，2.5 dB 低频搁架变 5 dB、3.5 dB
                    #   凹陷变 7 dB，那一段的音色平衡会与其他段可闻地不同。
                    #   没有伴生的原始段（旧版本产物）才走完整处理。
                    piece = _master_tensor(piece, rate,
                                           audio_master.master_segment,
                                           target_lufs=target,
                                           hp_hz=None if was_mastered
                                           else audio_master.HIGHPASS_HZ,
                                           shape_tone=not was_mastered)
            except Exception as exc:
                warn("segment mastering skipped (%s: %s)",
                                type(exc).__name__, exc)
        filled.append(piece)
    # 接缝焊接：相邻段交界处的原位微淡入淡出（长度不变，只杀拼接咔哒声）。
    filled = _weld_seam_fades(filled, rate)
    waveform = torch.cat(filled, dim=-1)

    if audio_master.ENABLED and matched:
        log("join audio: %d/%d segment(s) already at "
                     "delivery level (mastered when saved), %d matched here",
                     matched, len(pieces), len(pieces) - matched)

    # Last stop before the muxer, and the only point where the finished
    # soundtrack exists as one signal. The high-pass is skipped because the
    # per-segment pass already did it (two 2nd-order sections at the same corner
    # are a 4th-order high-pass, whose step overshoot measurably *raises* the
    # true peak), the tone shaping is skipped because the per-segment pass
    # already applied it (applying a 2.5 dB shelf and a 3.5 dB dip twice makes a
    # 5 dB shelf and a 7 dB dip), and the pre-limiter is skipped because the
    # segments are already mastered -- applying it again cost 3.6 dB of headroom
    # for nothing.
    if audio_master.ENABLED:
        try:
            # normalize=False: each segment was already brought to *its own*
            # per-segment target when it was saved (see profile_target /
            # master_segment). Re-normalising the whole film to one LUFS would
            # flatten the very dynamics this change introduces, so the join only
            # applies the final true-peak safety clamp and leaves the levels as
            # the segments left them.
            waveform = _master_tensor(waveform, rate, audio_master.master,
                                      hp_hz=None, pre_limit=False,
                                      shape_tone=False, normalize=False)
        except Exception as exc:
            # Losing a finished render to an audio polish step would be a bad
            # trade, so anything unexpected leaves the mix exactly as it was.
            warn("audio mastering skipped (%s: %s)",
                            type(exc).__name__, exc)

    return {"waveform": waveform, "sample_rate": rate}


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
