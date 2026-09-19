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
    audio_latent_frames,
    assert_clip_guides_supported,
    evict_dead_loaded_models,
    log,
    slice_audio,
    video_latent_t,
)


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
    log("latent continue: pinned %d video tokens (%d frames) + %d audio ticks; "
        "mask head %.2f -> seam %.2f",
        t_tail, covered, audio_pin_t,
        weights[0] if weights else 0.0, weights[-1] if weights else 0.0)
    return out, t_tail, audio_pin_t


def render_segment(settings, segment, start_anchor=None, end_anchor=None):
    """Render one segment. Returns ``(images, audio, samples)``.

    ``start_anchor`` is the previous segment's tail, pinned at frame 0, in either
    anchor form. ``end_anchor`` is picture only, pinned at the end -- see the repair
    node for why its audio is deliberately dropped.

    ``samples`` is the sampled AV latent, handed back so the caller can cut the next
    handoff out of it without a VAE round trip.
    """
    positive, latent = _reference_conditioning(settings, segment)

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

    if end_anchor is not None:
        positive = _pin(settings, positive, latent, end_anchor,
                        -anchor_frames(end_anchor))

    guider = Guider_Basic(settings["model"])
    guider.set_conds(positive)

    sampled = SamplerCustomAdvanced.execute(
        noise=Noise_RandomNoise(segment["resolved_seed"]),
        guider=guider,
        sampler=settings["sampler"],
        sigmas=settings["sigmas"],
        latent_image=latent,
    )[0]

    samples = sampled["samples"]
    video_latent = samples.unbind()[0] if samples.is_nested else samples
    images = settings["vae"].decode(video_latent)
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
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
    """段间清理。顺序照 ComfyUI_MiniMaxH3_Director 的 cleanup_segment_vram。

    关键在**死槽必须先清**：``is_dead()`` 的 LoadedModel 会一直占着
    ``current_loaded_models``，而 ``unload_all_models()`` 碰不到它们，被钉住的
    模型权重就留在 RAM 里。这个包跑的是 ~49GB 权重 vs ~40GB 物理内存，每段留
    一个槽就足以把长链推进换页 —— 表现是某段文本编码从 ~70 秒涨到 5-12 分钟，
    而显存明明是空的（GPU 根本没在算，是在等换页）。

    先 ``gc.collect()`` 是让 ModelPatcher 的 weakref 有机会真的死掉，
    否则 ``is_dead()`` 判断不出来、清不掉。
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
        mm.unload_all_models()
        mm.cleanup_models()
        evict_dead_loaded_models()
    mm.soft_empty_cache()


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
