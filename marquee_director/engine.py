"""Rendering one segment.

The whole seamless-join trick lives in ``render_segment``: the previous segment's
tail is anchored at frame 0 as a ``minimax_keyframes`` entry, so the new video
*opens on those exact frames* rather than being asked to imagine a plausible
continuation.

Why not ``ref_videos``? Because that is reference conditioning. H3 is trained to
continue from it, but it reconstructs the continuation instead of copying it: the
framing and the light carry over, the head pose pops. Measured against the source
clip, ``ref_videos`` joins land at about 17 dB PSNR -- the number you get for
unrelated imagery. A keyframe pins the clip instead: about 30 dB, and it costs no
sequence length, because it writes into frames the latent already has.

Two ways to build that keyframe, and the difference is the point of ``handoff_mode``:

``latent``  slice the tail straight out of the previous segment's *sampled* latent
            and pin it. No VAE anywhere in the handoff. This is the model's own
            representation of its own ending, so the anchor is exact by
            construction, and it saves a 39-frame VAE encode per segment.
``pixel``   what this pack did first: decode, write an mp4, read it back, and let
            ``MiniMaxH3AddGuide`` VAE-encode it again. Kept because it is the only
            path that survives a resolution change mid-session, and because a
            session rendered before latent tails existed has nothing else to
            resume from.
"""

import math

import comfy.model_management
import node_helpers
import torch
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras.nodes_custom_sampler import (
    Guider_Basic,
    Noise_RandomNoise,
    SamplerCustomAdvanced,
)

from .common import (
    FPS,
    FRAME_RESCALE,
    MiniMaxH3AddGuide,
    MiniMaxH3ReferenceToVideo,
    REFINE_SEED_FOLLOW,
    REFINE_SEED_INDEPENDENT,
    REFINE_SEED_OFFSET,
    audio_latent_frames,
    assert_clip_guides_supported,
    evict_dead_loaded_models,
    free_ram_bytes,
    log,
    slice_audio,
    video_latent_t,
)

# 延迟到模块尾部导入：face_refine 在模块层要建 comfy_api 的节点类，而本模块
# 又被 nodes.py 在模块层导入。放在这里能确保 .common / comfy_extras 都已就绪。
from .face_refine import apply_face_refine  # noqa: E402  (见上)


def _reference_conditioning(settings, segment):
    """Hand the segment's references to the core ref2va node in tag order.

    Only ``ref_images`` / ``ref_videos`` / ``ref_audios`` reach the tokenizer, and
    the tags are numbered 1-based per type in wiring order: the first image is
    ``<Picture 1>``, the second ``<Picture 2>``, and so on. That is why the segment
    node keeps its inputs in ordinal order -- the prompt's tag numbers are the
    socket numbers.
    """
    ref_images = {"ref_image_%d" % i: t for i, t in enumerate(segment["images"])}
    ref_videos = {"ref_video_%d" % i: t for i, (_, t) in enumerate(segment["videos"])}
    ref_video_audios = {"ref_video_audio_%d" % i: a
                        for i, (_, a) in enumerate(segment["video_audios"])}
    ref_audios = {"ref_audio_%d" % i: a for i, a in enumerate(segment["audios"])}

    return MiniMaxH3ReferenceToVideo.execute(
        clip=settings["clip"],
        vae=settings["vae"],
        audio_vae=settings["audio_vae"],
        prompt=segment["prompt"],
        width=settings["width"],
        height=settings["height"],
        length=segment["length"],
        ref_image_size=settings["ref_image_size"],
        ref_images=ref_images or None,
        ref_videos=ref_videos or None,
        ref_video_audios=ref_video_audios or None,
        ref_audios=ref_audios or None,
    )


def _av_streams(samples):
    """The (video, audio) pair inside an H3 AV latent, each with a batch axis."""
    parts = samples.unbind() if samples.is_nested else (samples,)
    if len(parts) < 2:
        raise ValueError("not a MiniMax H3 joint AV latent: no audio stream")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    return video, audio


def latent_tail(samples, handoff):
    """The last ``handoff`` frames of a sampled AV latent, as an anchor.

    The slice is always token-aligned. H3's frame grid is 17k+5 frames -> 5k+2
    latent steps, and a valid handoff is 17m+5 -> 5m+2, so the tail starts at
    5(k-m): a multiple of 5, which is cycle position 0 of the 1/4/4/4/4
    frames-per-token pattern. That matters -- start anywhere else and the tail's
    first token claims to cover 4 frames where a fresh encode would cover 1, and
    the anchor lands 3 frames out.
    """
    if handoff <= 0:
        return None
    video, audio = _av_streams(samples)
    tokens = video_latent_t(handoff)
    total = int(video.shape[2])
    if tokens > total:
        raise ValueError("handoff of %d frames needs %d latent steps, segment has %d"
                         % (handoff, tokens, total))
    start = total - tokens
    if start % 5 != 0:
        raise RuntimeError("tail starts at cycle position %d, not 0 -- the frame grid "
                           "and the handoff have gone out of phase" % (start % 5))
    rt = audio_latent_frames(handoff)
    return {
        "video_latent": video[:1, :, start:].clone(),
        "audio_latent": audio[:1, ..., max(0, int(audio.shape[-1]) - rt):].clone(),
    }


def latent_signature(samples):
    """Per-channel mean and std of a segment's whole video latent.

    24 numbers each. This is what "the look of this shot" reduces to: exposure and
    colour balance live in the channel means, contrast in the spreads. Taken over
    the *whole* segment rather than its tail, so that a specific pose, gesture or
    expression at the moment of the handoff averages out and only the shot's
    standing appearance is left.
    """
    video, _ = _av_streams(samples)
    video = video.detach().float()
    return {"mean": video.mean(dim=(0, 2, 3, 4)).cpu(),
            "std": video.std(dim=(0, 2, 3, 4)).cpu()}


def arrest_drift(anchor, reference, current, strength, clamp_sigma=0.35):
    """Pull a handoff's channel means back toward the opening shot's.

    Chained generation drifts because nothing in it is absolute. Each segment is
    told only "continue from this", so wherever segment N ended up *is* the truth
    for segment N+1, and a small bias repeats until the take has walked somewhere
    else entirely -- measurably, on a six-shot chain, about 8 L* of face brightness
    and 0.24 of ArcFace cosine.

    Closing that loop needs an absolute reference, and segment 1 is the only one
    available: it is the shot that was rendered from the prompt and the reference
    image alone, with no inherited anchor to be wrong about. So each handoff is
    shifted back toward segment 1's channel means before it is pinned.

    Two deliberate limits:

    ``strength`` is a fraction, not a reset. Correcting the whole error would put a
    step at every join -- the previous segment really did end where it ended, and
    the cut keeps those frames. A fraction leaves a residual small enough to hide
    under the join while still bounding the walk, which is the difference between
    drift that accumulates and drift that settles.

    ``clamp_sigma`` caps any single channel's correction at a fraction of that
    channel's own spread. It is there for the case this function cannot tell apart
    from drift: a shot that is *legitimately* darker because she walked away from
    the window. A real lighting change is large, and the clamp stops it being
    fought; accumulated drift is small, and passes through untouched.

    Only the video stream is touched. Audio does not drift this way, and shifting
    its latent would detune the voice.
    """
    if anchor is None or reference is None or current is None or strength <= 0:
        return anchor
    video_latent = anchor.get("video_latent")
    if video_latent is None:
        return anchor

    error = (reference["mean"] - current["mean"]) * float(strength)
    limit = reference["std"] * float(clamp_sigma)
    error = torch.clamp(error, -limit, limit)

    shift = error.to(device=video_latent.device, dtype=video_latent.dtype)
    corrected = dict(anchor)
    corrected["video_latent"] = video_latent + shift.view(1, -1, 1, 1, 1)
    corrected["drift_correction"] = float(error.abs().mean())
    return corrected


def anchor_frames(anchor):
    """How many pixel frames an anchor covers, whichever form it is in."""
    if anchor is None:
        return 0
    video_latent = anchor.get("video_latent")
    if video_latent is not None:
        return sum(1 if k % 5 == 0 else 4 for k in range(int(video_latent.shape[2])))
    return int(anchor["images"].shape[0])


def picture_only(anchor):
    """The same anchor with its soundtrack dropped.

    The repair node pins a segment's ending to the clip the *next* segment already
    opens on. That clip's audio belongs to the render being thrown away, so
    re-anchoring it would drag back the very take the repair exists to replace --
    which is how a shot with no dialogue of its own ends up speaking the previous
    shot's lines.
    """
    if anchor is None:
        return None
    if anchor.get("video_latent") is not None:
        return {"video_latent": anchor["video_latent"], "audio_latent": None}
    return {"images": anchor["images"], "audio": None}


def _pin_latent(positive, latent, anchor, frame_idx=0):
    """Pin a pre-encoded tail as a keyframe, the way AddGuide would if it took one.

    This is the one place the pack builds an H3 conditioning entry itself instead
    of calling the core node, because ``MiniMaxH3AddGuide`` only accepts pixels and
    always VAE-encodes them. The dict is exactly the shape ``PackedLayout`` reads:
    a resolved frame index, a video latent, and an audio latent placed on the same
    t-axis.
    """
    target_video, target_audio = _av_streams(latent["samples"])
    frame_count = sum(1 if k % 5 == 0 else 4 for k in range(int(target_video.shape[2])))

    video_latent = anchor["video_latent"]
    if tuple(video_latent.shape[3:]) != tuple(target_video.shape[3:]):
        raise ValueError(
            "latent handoff cannot resize: the anchor is %dx%d but this segment is "
            "%dx%d. Re-render the session at one resolution, or set handoff_mode to "
            "'pixel', which re-encodes through the VAE and can rescale."
            % (int(video_latent.shape[4]) * 16, int(video_latent.shape[3]) * 16,
               int(target_video.shape[4]) * 16, int(target_video.shape[3]) * 16))

    guide_frames = sum(1 if k % 5 == 0 else 4 for k in range(int(video_latent.shape[2])))
    resolved = frame_idx if frame_idx >= 0 else frame_count + frame_idx
    if resolved < 0 or resolved + guide_frames > frame_count:
        raise ValueError("a %d frame anchor at frame_idx %d does not fit in %d frames"
                         % (guide_frames, frame_idx, frame_count))

    keyframe = {
        "resolved_frame_index": resolved,
        "latent": video_latent.to(device=target_video.device, dtype=target_video.dtype),
    }
    audio_latent = anchor.get("audio_latent")
    if audio_latent is not None and audio_latent.shape[-1] > 0:
        # Same clamp AddGuide applies: the guide's audio cannot outrun the target's
        # own remaining track. Letting it would anchor a whole soundtrack into a
        # segment that has room for part of one, which is how a segment ends up
        # replaying the previous segment's dialogue.
        max_rt = math.floor(int(target_audio.shape[-1]) - FRAME_RESCALE * resolved)
        if max_rt >= 1:
            if audio_latent.shape[-1] > max_rt:
                audio_latent = audio_latent[..., :max_rt]
            keyframe["audio_latent"] = audio_latent.to(
                device=target_audio.device, dtype=target_audio.dtype)

    keyframes = list(positive[0][1].get("minimax_keyframes", []))
    keyframes.append(keyframe)
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})


def _pin_pixels(settings, positive, latent, images, audio, frame_idx):
    return MiniMaxH3AddGuide.execute(
        positive=positive,
        latent=latent,
        frame_idx=frame_idx,
        vae=settings["vae"],
        audio_vae=settings["audio_vae"] if audio is not None else None,
        image=images,
        audio=audio,
    )[0]


def _pin(settings, positive, latent, anchor, frame_idx):
    """Anchor whichever form this anchor arrived in."""
    # Both forms pin a clip, and a clip is what a pre-0.34.0 core mis-sizes; the
    # check is here rather than in either branch so neither path can skip it.
    if anchor_frames(anchor) > 1:
        assert_clip_guides_supported()
    if anchor.get("video_latent") is not None:
        return _pin_latent(positive, latent, anchor, frame_idx)
    return _pin_pixels(settings, positive, latent, anchor["images"],
                       anchor.get("audio"), frame_idx)


# ---------------------------------------------------------------------------
# Latent continue: write the previous tail into the samples, not just the conds
#
# ``_pin_latent`` puts the previous tail in *conditioning*. Conditioning is advice:
# H3 is free to re-imagine those frames, and measured against the source clip a
# cond-only handoff drifted by about 7% (cur[38] vs prev[-1] = 0.073, ~22.7 dB)
# rather than the ~30 dB this module's header claims. ComfyUI_MiniMaxH3_Director
# solves the same problem from the other end -- it copies the tail straight into the
# target latent's samples and hands the sampler a ``noise_mask``. A mask is not
# advice; ``sampling_function`` computes
# ``out = out * mask + latent_image * (1 - mask)``, so mask=1 is "redraw this" and
# mask=0 is "leave exactly what I wrote".
#
# The shape of the mask is the subtle part. The replayed head is thrown away at join
# time, so it costs nothing to let the model redraw it (weight 1.0). What has to be
# pinned is the seam -- the last tokens before genuinely new content begins -- so it
# tapers down to ``SEAM_MIN_MASK`` and the model is nearly forced to continue from
# the previous segment's final frame. That is what removes the splice jump.
# ---------------------------------------------------------------------------
SEAM_TAPER_TOKENS = 4
SEAM_MIN_MASK = 0.10
SEAM_MIN_FLOOR = 0.0
SEAM_MIN_CEIL = 0.95
AUDIO_SOFT_RELEASE_TICKS = 8

# 逐 step 重绘掩码（见 ``install_prefix_remask``）在 latent 字典上留的两个标记。
# 采样器只认 ``samples`` / ``noise_mask``，多出来的键会被原样忽略，所以这里
# 是「把这一段的接缝参数带给采样钩子」的唯一通道 —— 否则渲染节点算出的
# seam 值要在 render_segment 里照着公式重算一遍，那就是两处真相。
PREFIX_STEPS_KEY = "_marquee_continue_prefix_steps"
CONTINUE_SEAM_KEY = "_marquee_continue_seam_min"


def clamp_seam_min_mask(value):
    """User-facing「重绘幅度」. Higher = more redraw, less literal copy of the tail."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return SEAM_MIN_MASK
    if not math.isfinite(n):
        return SEAM_MIN_MASK
    return max(SEAM_MIN_FLOOR, min(SEAM_MIN_CEIL, n))


def prefix_token_weights(prefix_steps, taper_steps=SEAM_TAPER_TOKENS, seam_min=None):
    """1.0 across the disposable replay head, tapering to ``seam_min`` at the seam."""
    n = int(prefix_steps)
    if n < 1:
        return ()
    taper = max(1, min(int(taper_steps), n))
    head = n - taper
    floor = clamp_seam_min_mask(SEAM_MIN_MASK if seam_min is None else seam_min)
    weights = [1.0] * head
    weights.extend(1.0 + (floor - 1.0) * (float(i + 1) / float(taper))
                   for i in range(taper))
    return tuple(weights)


def pixel_frames_for_tokens(tokens):
    """Inverse of ``video_latent_t``: how many pixel frames N latent tokens cover.

    H3 packs 1 frame into every fifth token and 4 into the others, so this walks the
    cycle instead of multiplying. Used to assert the pinned prefix really is the
    17k+5 handoff (1.6s -> 39 frames -> 12 tokens) and not an off-grid slice, which
    is the one mistake that shows up as a frame jump at the join.
    """
    return sum(1 if k % 5 == 0 else 4 for k in range(int(tokens)))


def _nested(video, audio, template=None):
    try:
        from comfy.nested_tensor import NestedTensor

        return NestedTensor((video, audio))
    except Exception:
        cls = type(template) if template is not None else None
        if cls is not None:
            return cls((video, audio))
        raise


def _spatial_video_mask(t_steps, prefix_steps, *, height, width, device, dtype,
                        weights=None):
    """[B,1,T,H,W] -- H3 reads its temporal mask laid out spatially."""
    t = int(t_steps)
    h = max(1, int(height))
    w = max(1, int(width))
    mask = torch.ones((1, 1, t, h, w), device=device, dtype=dtype)
    n = max(0, min(int(prefix_steps), t))
    if n < 1:
        return mask
    if weights is None:
        ramp = torch.tensor(prefix_token_weights(n), device=device, dtype=dtype)
    else:
        ramp = weights[:n].to(device=device, dtype=dtype)
    mask[:, :, :n] = ramp.view(1, 1, n, 1, 1)
    return mask


def _soft_audio_mask(like, pin_t, release=AUDIO_SOFT_RELEASE_TICKS):
    """Same shape as the audio stream: 0 = keep the previous tail, 1 = redraw.

    The pinned ticks are hard zero (this is the previous segment's own voice, it
    must not be re-invented) with a cosine release over the last few, so the handoff
    does not click.
    """
    mask = torch.ones_like(like)
    n = max(0, min(int(pin_t), int(like.shape[-1])))
    if n < 1:
        return mask
    mask[..., :n] = 0.0
    rel = min(int(release), n)
    if rel < 1:
        return mask
    idx = torch.arange(1, rel + 1, device=like.device, dtype=mask.dtype)
    ramp = 0.5 - 0.5 * torch.cos(math.pi * idx / float(rel))
    mask[..., n - rel:n] = ramp.reshape((1,) * (mask.ndim - 1) + (rel,))
    return mask


def apply_latent_continue(latent, anchor, seam_min_mask=None):
    """Copy ``anchor``'s tail into ``latent`` samples and mask it against redraw.

    Returns ``(latent, video_tokens, audio_ticks)``.
    """
    video, audio = _av_streams(latent["samples"])
    tail_video = anchor.get("video_latent")
    if tail_video is None:
        raise ValueError("latent continue needs a latent anchor, got pixels")
    if tail_video.ndim == 4:
        tail_video = tail_video.unsqueeze(0)
    if tuple(tail_video.shape[3:]) != tuple(video.shape[3:]):
        raise ValueError(
            "latent handoff cannot resize: the anchor is %dx%d but this segment is "
            "%dx%d. Re-render the session at one resolution, or set handoff_mode to "
            "'pixel', which re-encodes through the VAE and can rescale."
            % (int(tail_video.shape[4]) * 16, int(tail_video.shape[3]) * 16,
               int(video.shape[4]) * 16, int(video.shape[3]) * 16))

    t_tail = min(int(tail_video.shape[2]), int(video.shape[2]) - 1)
    if t_tail < 1:
        raise ValueError("latent continue: empty video prefix")
    covered = pixel_frames_for_tokens(t_tail)
    if covered % 17 != 5:
        # Not fatal -- the join still trims by the handoff the manifest recorded --
        # but it means the prefix is not the 1.6s/39f window and the seam lands
        # between frames instead of on one.
        log("latent continue: prefix is %d frames (%d tokens), off the 17k+5 grid -- "
            "the join's de-dup will still cut %d frames but the seam may sit mid-"
            "token", covered, t_tail, covered)

    patched_video = video.clone()
    patched_video[:, :, :t_tail] = tail_video[:, :, :t_tail].to(
        device=patched_video.device, dtype=patched_video.dtype)

    patched_audio = audio.clone()
    tail_audio = anchor.get("audio_latent")
    audio_pin_t = 0
    if tail_audio is not None and int(tail_audio.shape[-1]) > 0:
        audio_pin_t = min(int(tail_audio.shape[-1]), int(patched_audio.shape[-1]))
        if audio_pin_t > 0:
            patched_audio[..., :audio_pin_t] = tail_audio[..., :audio_pin_t].to(
                device=patched_audio.device, dtype=patched_audio.dtype)

    samples = latent["samples"]
    out = dict(latent)
    out["samples"] = _nested(patched_video, patched_audio, samples)

    seam = clamp_seam_min_mask(SEAM_MIN_MASK if seam_min_mask is None else seam_min_mask)
    weights = prefix_token_weights(t_tail, seam_min=seam)
    video_mask = _spatial_video_mask(
        int(patched_video.shape[2]),
        t_tail,
        height=int(patched_video.shape[3]),
        width=int(patched_video.shape[4]),
        device=patched_video.device,
        dtype=torch.float32,
        weights=torch.tensor(weights, dtype=torch.float32),
    )
    audio_mask = _soft_audio_mask(patched_audio, audio_pin_t)
    out["noise_mask"] = _nested(video_mask, audio_mask, samples)
    out[PREFIX_STEPS_KEY] = int(t_tail)
    out[CONTINUE_SEAM_KEY] = float(seam)
    log("latent continue: pinned %d video tokens (%d frames) + %d audio ticks; "
        "mask head %.2f -> seam %.2f",
        t_tail, covered, audio_pin_t,
        weights[0] if weights else 0.0, weights[-1] if weights else 0.0)
    return out, t_tail, audio_pin_t


# ---------------------------------------------------------------------------
# 逐 step 重绘掩码（借鉴 ComfyUI_MiniMaxH3_Director 的 _PrefixRemask，Apache-2.0）
#
# 静态 noise_mask 的问题不是"锁不锁"，而是"从头锁到尾用的是同一个值"。
# 采样第 1 步时 x 几乎是纯噪声，把前缀硬锁成上一段的真实帧，等于要求模型在
# 一片噪声里先认出一帧画面；采样最后几步时模型已经收敛，此时前缀若还是
# head=1.0（随便重画），模型就会把「上一段的真实结尾」当成可改区域描一遍
# —— 于是接缝那一帧不是被钉住，而是被重新想象了一次。
#
# 改成随 σ 缩放：权重 = 静态权重 × (下一步 σ / 当前步 σ)。采样从高噪走到低噪，
# 这个比值前期约 0.8、末段趋近 0，于是前缀在早期允许重写、在末期整体锁回
# latent_image 里写好的那一段尾巴。接缝那几个 token 由 seam_min 兜底，
# 全程不低于它 —— 无论第几步，接缝永远是"几乎不许动"。
# ---------------------------------------------------------------------------


def _schedule_values(sigmas):
    """Descending, deduped, finite, non-negative sigma values."""
    if torch.is_tensor(sigmas):
        raw = sigmas.detach().float().reshape(-1).cpu().tolist()
    else:
        raw = list(sigmas or ())
    return tuple(sorted({float(v) for v in raw
                         if math.isfinite(float(v)) and float(v) >= 0.0},
                        reverse=True))


def next_sigma_ratio(current, sigmas):
    """``next_sigma / current_sigma`` in [0, 1] -- 1 early in sampling, 0 at the end."""
    cur = float(current)
    if not math.isfinite(cur) or cur <= 0.0:
        return 0.0
    tol = max(1e-7, abs(cur) * 1e-6)
    for cand in _schedule_values(sigmas):
        if cand < cur - tol:
            return max(0.0, min(1.0, cand / cur))
    return 0.0


def live_prefix_weights(prefix_steps, ratio, seam_min=None,
                        taper_steps=SEAM_TAPER_TOKENS):
    """Static taper scaled by the step's ratio, never below ``seam_min``."""
    floor = clamp_seam_min_mask(SEAM_MIN_MASK if seam_min is None else seam_min)
    out = []
    for base in prefix_token_weights(prefix_steps, taper_steps, seam_min=floor):
        value = float(base) * max(0.0, min(1.0, float(ratio)))
        if floor > 0.0:
            value = max(floor, value)
        out.append(max(0.0, min(1.0, value)))
    return tuple(out)


def _unbind_mask(mask):
    """The (video, audio) pair inside a possibly-nested mask."""
    if torch.is_tensor(mask):
        return [mask]
    if hasattr(mask, "unbind"):
        return list(mask.unbind())
    if hasattr(mask, "tensors"):
        return list(mask.tensors)
    if isinstance(mask, (tuple, list)):
        return list(mask)
    return None


class _PrefixRemask:
    """Per-step prefix redraw weights, installed as a model denoise-mask hook.

    Any shape it does not recognise is handed straight back untouched -- a wrong
    guess about the mask layout must degrade to the static mask, never to a
    corrupted frame.
    """

    def __init__(self, prefix_steps, sigmas, video_shape, seam_min=None):
        self.prefix_steps = max(0, int(prefix_steps))
        self.sigmas = _schedule_values(sigmas)
        self.video_shape = tuple(int(x) for x in video_shape)
        self.seam_min = clamp_seam_min_mask(SEAM_MIN_MASK if seam_min is None else seam_min)
        self.steps_seen = 0

    def _weights(self, sigma, extra_options=None):
        schedule = self.sigmas or _schedule_values(
            (extra_options or {}).get("sigmas", ()))
        ratio = next_sigma_ratio(sigma, schedule)
        return torch.tensor(
            live_prefix_weights(self.prefix_steps, ratio, self.seam_min),
            dtype=torch.float32)

    def _write(self, mask, weights):
        """Write along the temporal axis of a 5D [B,C,T,H,W] video mask."""
        n = min(int(weights.numel()), int(mask.shape[2]), self.prefix_steps)
        if n < 1:
            return mask
        out = mask.clone()
        view = [1] * out.ndim
        view[2] = n
        sl = [slice(None)] * out.ndim
        sl[2] = slice(0, n)
        out[tuple(sl)] = weights[:n].to(device=out.device, dtype=out.dtype).view(*view)
        return out

    def denoise_mask_function(self, sigma, denoise_mask, extra_options=None):
        self.steps_seen += 1
        try:
            weights = self._weights(sigma, extra_options)
            # H3 packs the AV latent into [B,1,elems]; the video stream is the
            # first ``prod(video_shape[1:])`` of it.
            if (torch.is_tensor(denoise_mask) and denoise_mask.ndim == 3
                    and len(self.video_shape) == 5):
                elems = int(math.prod(self.video_shape[1:]))
                if int(denoise_mask.shape[-1]) < elems:
                    return denoise_mask
                packed = denoise_mask.clone()
                video = packed[..., :elems].reshape(self.video_shape)
                video = self._write(video, weights)
                packed[..., :elems] = video.reshape(
                    packed.shape[0], 1, elems).to(dtype=packed.dtype)
                return packed
            if torch.is_tensor(denoise_mask) and denoise_mask.ndim == 5:
                return self._write(denoise_mask, weights)
            streams = _unbind_mask(denoise_mask)
            if streams and torch.is_tensor(streams[0]) and streams[0].ndim == 5:
                video = self._write(streams[0], weights)
                rest = list(streams[1:])
                if rest:
                    return _nested(video.to(dtype=streams[0].dtype), rest[0],
                                   denoise_mask)
                return video.to(dtype=streams[0].dtype)
            return denoise_mask
        except Exception as exc:                     # never break a render
            log("seam remask: 本步回退静态掩码（%s）", exc)
            return denoise_mask


def install_prefix_remask(model, prefix_steps, sigmas, video_shape, seam_min=None):
    """Return ``(model_with_hook_or_the_original, state_or_None)``.

    The model is *cloned*: a denoise-mask hook is per-render state, and writing
    it onto the caller's model would leak it into every later segment (and into
    the first segment, which has no prefix at all).
    """
    if model is None or int(prefix_steps or 0) < 1:
        return model, None
    if not callable(getattr(model, "clone", None)):
        return model, None
    try:
        patched = model.clone()
        if not callable(getattr(patched, "set_model_denoise_mask_function", None)):
            return model, None
        state = _PrefixRemask(prefix_steps, sigmas, video_shape, seam_min=seam_min)
        patched.set_model_denoise_mask_function(state.denoise_mask_function)
        setattr(patched, "_marquee_prefix_remask", state)
        log("seam remask: 已安装逐 step 掩码（前缀 %d token，seam_min %.2f）",
            state.prefix_steps, state.seam_min)
        return patched, state
    except Exception as exc:
        log("seam remask: 安装失败，本次用静态掩码（%s）", exc)
        return model, None


def uninstall_prefix_remask(model):
    """Drop the hook so the clone can be garbage collected after the segment."""
    if model is None:
        return
    state = getattr(model, "_marquee_prefix_remask", None)
    if state is not None:
        try:
            state.steps_seen = 0
            delattr(model, "_marquee_prefix_remask")
        except Exception:
            pass
    try:
        options = getattr(model, "model_options", None)
        if isinstance(options, dict):
            options.pop("denoise_mask_function", None)
    except Exception:
        pass


def _refine_seed(pack, first_pass_seed):
    """The seed the refine pass samples on.

    ``跟随一采`` is the default and the safe one: the second pass then walks the
    same noise trajectory as the first, so it reads as *continuing to converge on
    the same image* rather than as a second opinion on it. ``一采+1`` and
    ``独立种子`` deliberately break that, which is what you want when the point of
    the refine pass is to shake a detail loose.
    """
    mode = pack.get("seed_mode") or REFINE_SEED_FOLLOW
    if mode == REFINE_SEED_INDEPENDENT:
        return int(pack.get("seed") or 0)
    if mode == REFINE_SEED_OFFSET:
        return (int(first_pass_seed) + 1) % (1 << 63)
    return int(first_pass_seed)


def _refine_sampler(settings, pack):
    """The SAMPLER object for the refine pass. Follows pass 1 by default."""
    name = pack.get("sampler")
    if not name:
        return settings["sampler"]
    return comfy.samplers.sampler_object(name)


def refine_samples(settings, samples, positive, seed):
    """Optional second sample pass over a segment's AV latent.

    Returns ``(samples, note)`` — ``note`` is empty when refine is off, which is
    the case that has to stay byte-identical to the old behaviour.

    The pass reuses pass 1's conditioning wholesale: it carries the
    ``minimax_keyframes`` pins that lock a segment's opening to the previous
    segment's tail, so refining cannot drift the seam. It deliberately does NOT
    carry pass 1's ``noise_mask`` — that mask existed to protect the *replayed*
    prefix while it was being generated, and by the time we are here that prefix
    is a finished picture, not an input to protect.
    """
    pack = settings.get("refine")
    if not isinstance(pack, dict):
        return samples, ""
    sigmas = pack.get("sigmas")
    if sigmas is None:
        # Wired but no schedule: treat as off rather than sampling with pass 1's
        # sigma table, which would be a full re-generation at denoise=1.
        return samples, ""

    passes = max(1, int(pack.get("passes") or 1))
    model = pack.get("model") or settings["model"]
    sampler = _refine_sampler(settings, pack)
    refine_seed = _refine_seed(pack, seed)

    guider = Guider_Basic(model)
    guider.set_conds(positive)
    try:
        for i in range(passes):
            # A fresh noise object per pass: two passes sharing one seed would
            # start from literally the same noise, which is not "more refining",
            # it is the same step twice.
            out = SamplerCustomAdvanced.execute(
                noise=Noise_RandomNoise(refine_seed + i),
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image={"samples": samples},
            )[0]
            samples = out["samples"]
            del out
    finally:
        del guider

    if hasattr(sigmas, "shape"):
        shape = "x".join(str(int(s)) for s in sigmas.shape)
    else:
        try:
            shape = "%d steps" % len(sigmas)
        except TypeError:                                    # pragma: no cover
            shape = "?"
    note = "%d pass(es), sigmas %s, %s" % (
        passes, shape,
        "跟随一采模型" if pack.get("model") is None else "二采模型")
    return samples, note


def _is_oom(exc):
    """是不是显存/内存耗尽。

    ComfyUI 抛的是 ``torch.OutOfMemoryError``（``RuntimeError`` 的子类），但不同
    torch 版本、以及 CUDA 分配器自己抛的消息措辞不一，所以按关键字判定，别只认
    一个类型名。
    """
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def render_segment(settings, segment, start_anchor=None, end_anchor=None):
    """Render one segment. Returns ``(images, audio, samples)``.

    ``start_anchor`` is the previous segment's tail, pinned at frame 0, in either
    anchor form. ``end_anchor`` is picture only, pinned at the end -- see the repair
    node for why its audio is deliberately dropped.

    ``samples`` is the sampled AV latent, handed back so the caller can cut the next
    handoff out of it without a VAE round trip.

    ★ OOM 自愈（2026-09-26）
    ------------------------
    参考图的视觉 token 要过一次 26 GB 的文本编码器，这是整条链里显存峰值最高、
    也最容易崩的一步：实测 3070 8GB 在**第 5 段**这里 OOM，前 4 段都正常，整条
    链跑了 50 分钟才死，前 4 段全部白烧。

    单段的 OOM 不该毁掉整条链。这里在进入条件编码前先清一次死槽与缓存碎片
    （成本几乎为零），失败后再**强制释放显存并原样重试一次**。

    只退显存、不做 ``cleanup_models()`` 的整包内存清理：本包权重约 49 GB 而对
    象机内存约 40 GB，把权重从内存里丢掉就得从磁盘整包重读，实测会在重载阶段
    100% CPU 空转卡死十几分钟 —— 那是比 OOM 更坏的结局（见
    ``free_between_segments`` 的护栏说明）。重试只此一次，再失败就如实抛出。
    """
    evict_dead_loaded_models()
    comfy.model_management.soft_empty_cache()
    try:
        return _render_segment_once(settings, segment, start_anchor, end_anchor)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        log("OOM：条件编码/采样阶段显存不足（%s）—— 强制释放显存后原样重试一次",
            str(exc).strip().splitlines()[0][:160])
        comfy.model_management.unload_all_models()
        evict_dead_loaded_models()
        comfy.model_management.soft_empty_cache()
        return _render_segment_once(settings, segment, start_anchor, end_anchor)


def _render_segment_once(settings, segment, start_anchor=None, end_anchor=None):
    positive, latent = _reference_conditioning(settings, segment)

    remask_model = None
    if start_anchor is not None:
        positive = _pin(settings, positive, latent, start_anchor, 0)
        # The cond above tells the model what the previous segment ended on; this
        # writes those frames into the samples and masks them, so the model cannot
        # quietly re-imagine them. Only the latent form can be written directly --
        # a pixel anchor would have to round-trip through the VAE first, which is
        # exactly the re-encode ``handoff_mode='latent'`` exists to avoid.
        if (settings.get("handoff_remask", True)
                and start_anchor.get("video_latent") is not None):
            seam = settings.get("seam_redraw")
            if seam is None:
                seam = settings.get("seam_min_mask")      # 旧名兼容
            latent, _pinned_tokens, _pinned_ticks = apply_latent_continue(
                latent, start_anchor, seam_min_mask=seam)
            # 逐 step 掩码是**可选**的（H3 Chain Settings 的 seam_remask）。
            # 默认关：它改变的是成片本身，开着会让既有会话的段全部重渲，
            # 而这不该由一次升级替用户决定。参数从 latent 上读（不是重算），
            # 保证和上面写进 noise_mask 的是同一套。
            if settings.get("seam_remask"):
                _video_stream, _ = _av_streams(latent["samples"])
                remask_model, _remask_state = install_prefix_remask(
                    settings["model"],
                    int(latent.get(PREFIX_STEPS_KEY) or 0),
                    settings["sigmas"], tuple(_video_stream.shape),
                    seam_min=latent.get(CONTINUE_SEAM_KEY))

    if end_anchor is not None:
        positive = _pin(settings, positive, latent, end_anchor,
                        -anchor_frames(end_anchor))

    # 接了逐 step 掩码就用带钩子的那份 clone（它是 model.clone()，权重是同一套，
    # 只是多了一个 model_options 钩子）；没接就是原来的 model，一分不差。
    guider = Guider_Basic(remask_model or settings["model"])
    guider.set_conds(positive)

    try:
        sampled = SamplerCustomAdvanced.execute(
            noise=Noise_RandomNoise(segment["resolved_seed"]),
            guider=guider,
            sampler=settings["sampler"],
            sigmas=settings["sigmas"],
            latent_image=latent,
        )[0]
    finally:
        # 钩子是这一段专属的：不卸掉，这个 clone 就带着指向本段 latent 的闭包
        # 活到下一次 gc，等于每段往内存里留一份模型。
        uninstall_prefix_remask(remask_model)
        remask_model = None

    samples = sampled["samples"]
    # ★ 二采必须排在**切尾巴和解码之前**（借鉴 MiniMaxH3_Director 的 refine 口）。
    #   段间锚点（latent_tail）是从 samples 里切出去的，若继续从一采结果里切，
    #   下一段开头钉的就是一采画质的帧，而成片里前一段是二采画质 —— 接缝处
    #   会出现一记画质跳变，正好毁掉这个包存在的理由。
    #   放在解码之前，尾巴 / 漂移签名 / 成片三者天然同源，不需要额外对齐。
    samples, refine_note = refine_samples(
        settings, samples, positive, segment["resolved_seed"])
    if refine_note:
        log("refine: %s", refine_note)
    video_latent = samples.unbind()[0] if samples.is_nested else samples
    images = settings["vae"].decode(video_latent)
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])

    # ★ 修脸排在**解码之后**：它改的是像素，不是 latent。
    #   段间锚点 latent_tail 依然从上面那份未修的 samples 里切出去 —— 这是刻意的。
    #   下一段开头钉的是「没修过脸的尾帧」，所以本段成片必须在段首/段尾把修脸
    #   结果淡回原图（fade_stitch_at_seams），否则接缝处脸会跳一下。
    #   反过来，若为了让锚点也带上修脸结果而把整段重新编码，代价是每段多一次
    #   VAE encode + 一次重采，且接缝两侧变成两次编码的误差叠加，不划算。
    images, face_note = apply_face_refine(
        settings, segment, images,
        prompt=segment.get("prompt") or "",
        replay_frames=pixel_frames_for_tokens(
            start_anchor["video_latent"].shape[2])
        if (isinstance(start_anchor, dict)
            and start_anchor.get("video_latent") is not None) else 0)
    if face_note:
        log("face refine: %s", face_note)

    audio = vae_decode_audio(settings["audio_vae"], sampled)

    del sampled, video_latent, positive, latent, guider
    return images, audio, samples


def take_tail(images, audio, handoff, fps=FPS):
    """The last ``handoff`` frames and exactly their own audio.

    ``.clone()`` is not optional: a view keeps the whole decoded segment alive, and
    the point of writing each segment to disk is to stop holding it in memory.
    """
    if handoff <= 0:
        return None
    total = int(images.shape[0])
    start = max(0, total - handoff)
    tail_images = images[start:].clone()
    tail_audio = slice_audio(audio, start / float(fps), (total - start) / float(fps))
    return tail_images, tail_audio


def free_between_segments(unload_models=False):
    """段间清理：默认清显存；``unload_models=True`` 时显存 + 内存一起清。

    顺序照 ComfyUI_MiniMaxH3_Director 的 cleanup_segment_vram。

    关键在**死槽必须先清**：``is_dead()`` 的 LoadedModel 会一直占着
    ``current_loaded_models``，而 ``unload_all_models()`` 碰不到它们，被钉住的
    模型权重就留在 RAM 里。这个包跑的是 ~49GB 权重 vs ~40GB 物理内存，每段留
    一个槽就足以把长链推进换页 —— 表现是某段文本编码从 ~70 秒涨到 5-12 分钟，
    而显存明明是空的（GPU 根本没在算，是在等换页）。

    先 ``gc.collect()`` 是让 ModelPatcher 的 weakref 有机会真的死掉，
    否则 ``is_dead()`` 判断不出来、清不掉。

    ★ 内存护栏（2026-09-19 加）
    ----------------------------
    清内存是有代价的：权重被丢掉之后，下一段要**从磁盘整包读回**
    （UNet ~20GB + 文本编码器 ~26GB）。内存本来就装不下这套权重时，
    读回的过程会把 ComfyUI 的动态显存/内存压力缓存推进自旋 —— 实测
    **CPU 100% + 磁盘读 0 MB/s 卡死 15 分钟以上**，链再也走不下去。

    用户的诉求是「既要清显存也要清内存，之后**继续**把未渲的段渲完」——
    关键词是"继续"。所以这里加一道护栏：

      * 可用物理内存 **足够** → 照办，显存 + 内存一起清（真正释放权重）。
      * 可用物理内存 **不够** → 只清显存，**保留内存里的权重**，并打一行告警。

    这样"清内存"在安全时一定会发生，不安全时也不会把整条链拖死 ——
    宁可少清一次内存，也不能让剩下的段渲不完。
    """
    import gc

    mm = comfy.model_management
    gc.collect()
    try:
        mm.cleanup_models_gc()
    except Exception:
        pass
    evict_dead_loaded_models()
    if unload_models:
        # 先算「待会儿重装要多少钱」：当前驻留模型的总大小。
        need = 0
        try:
            for cur in getattr(mm, "current_loaded_models", None) or []:
                need += int(getattr(cur, "model_memory", 0) or 0)
        except Exception:
            need = 0
        avail = free_ram_bytes()
        # 护栏：可用内存得能装下"重装后的峰值"，留 1.25 倍余量给碎片与激活。
        # avail 取不到（None）时不拦 —— 没数据就别自作主张降级。
        if avail is not None and need > 0 and avail < int(need * 1.25):
            log("⚠ 跳过本次内存清理：可用物理内存 %.1f GB，重装这 %.1f GB 权重"
                "会装不下（曾在此处卡死）。本次只清显存，权重留在内存里，"
                "下一段照常继续渲染。", avail / 2 ** 30, need / 2 ** 30)
            # 只退显存。对照 comfy/model_management.py 源码核过：
            #   unload_all_models() -> free_memory(1e30, device) 只把模型挪出显存，
            #   ModelPatcher 对象还在 current_loaded_models 里，权重仍在 RAM；
            #   真正让 RAM 释放的是 cleanup_models()（pop + del 断引用）配合 gc。
            # 所以这里**故意不调** cleanup_models()，内存里的权重得以保留。
            mm.unload_all_models()
            mm.soft_empty_cache()
            return
        # 真的清内存：先退显存，再断引用 + 多轮 gc 把权重从 RAM 放掉。
        mm.unload_all_models()
        mm.cleanup_models()
        evict_dead_loaded_models()
        for _ in range(3):
            gc.collect()
        mm.cleanup_models_gc()
        mm.soft_empty_cache()
        log("已卸载模型：显存 + 内存均已释放（约 %.1f GB 权重），下一段将从磁盘重载。",
            need / 2 ** 30)
    mm.soft_empty_cache()


def unload_every_int(every):
    """把 ``unload_every`` 归一成 int。

    老工作流存的是布尔（``unload_models_after`` 时代）：``True``→1、``False``→0，
    正好等于旧语义。垃圾值（``None`` / 字符串 / 对象）→ 0 = 从不卸载。

    ★ 判定与日志**必须走同一个入口**：以前 ``unload_due`` 里自己 try/int 一次、
    调用点又各写一套，两处口径一旦不一致，就会出现「日志说没到点、其实已经卸了」
    这种自相矛盾的输出。统一到这一个函数，谁也不许再自己 int()。
    """
    try:
        return int(every)
    except (TypeError, ValueError):
        return 0


def unload_due(rendered_since_unload, every, is_last=False):
    """「每 N 段卸载一次模型」的判定。

    ``every`` 语义（2026-09-19 起取代原来的布尔 ``unload_models_after``）：

    ====== ==========================================================
     ``0`` 从不卸载 —— 连链尾也不卸，完全交给 ComfyUI 自己的收尾释放
     ``1`` 每段都卸 = 旧 ``unload_models_after=True`` 的行为
     ``2`` 每渲染满两段卸一次（默认；把加载开销摊薄一半）
     ``N`` 每渲染满 N 段卸一次
    ====== ==========================================================

    ``rendered_since_unload`` 是「距上次卸载已经渲了几段」（含本段）。
    **只数真的烧了显卡的段**：命中磁盘缓存的段不加载模型，不该推进计数器。

    ``is_last=True`` 时无条件卸（``every>0`` 的前提下）—— 这一条是旧行为的保留项：
    以前每段都卸，链尾自然也卸；链尾释放给后面的拼接 / 编码让出显存，不卸会改变
    末段之后那一小段的表现。

    ``every`` 收 bool 也认，归一规则见 ``unload_every_int``。
    """
    every = unload_every_int(every)
    if every <= 0:
        return False
    if is_last:
        return True
    return int(rendered_since_unload) >= every


def describe(segment, index, anchored):
    seconds = segment["length"] / float(FPS)
    return "segment %d: %d frames (%.2fs)%s" % (
        index + 1, segment["length"], seconds,
        ", opening on the previous segment's tail" if anchored else "")


def push_preview(pbar, frame, done, total):
    """Show a finished segment on the node while the chain is still running.

    A chain is one node execution that can run for hours, so without this the canvas
    sits apparently idle from the first segment to the last. ProgressBar's preview
    channel is the same one latent previews use, and the executor resolves the node
    id from the running context, so the image lands on the chain node itself.

    The frame shown is the segment's LAST one, deliberately: that is the frame the
    next segment opens on, so what you are watching is the state being handed
    forward.
    """
    if pbar is None:
        return
    preview = None
    if frame is not None:
        try:
            from PIL import Image
            import latent_preview
            array = (frame.clamp(0, 1) * 255).byte().cpu().numpy()
            preview = ("JPEG", Image.fromarray(array), latent_preview.MAX_PREVIEW_RESOLUTION)
        except Exception:
            preview = None  # a preview is never worth failing a render over
    pbar.update_absolute(done, total, preview)
