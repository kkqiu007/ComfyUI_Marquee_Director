"""The five nodes.

    H3 Chain Settings   ->  models, sampler, canvas and seed, bundled once
    H3 Render Segment   ->  one shot: prompt, references and length IN, its rendered
                            video and a chain_state OUT that the next segment's node
                            wires into -- the chain is the graph, not a hidden loop
    H3 Chain to Video   ->  joins a chain_state's segments into one cut
    H3 Repair Segment   ->  re-renders one segment without disturbing its neighbours
    H3 Load Session     ->  re-join an earlier session, or extend it with more
                            H3 Render Segment nodes, without re-rendering what is done

Every H3 Render Segment is its own node execution, so its ``video`` output populates
-- and can feed a Preview Video node, or anything else -- the moment that one segment
finishes, independent of how many more are still to come. That is also what makes
resume robust to a hard kill: the manifest is rewritten after every segment's own
execute() returns, not once at the end of a run that might not reach one.
"""

import json
import os
import shutil

import comfy.utils
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import ComfyExtension, io

from . import board_nodes
from . import prompt_nodes
from . import session as session_mod
from . import video_io
from .common import (
    FPS,
    generation_length,
    guide_length,
    log,
    ordered_autogrow,
    seconds_to_frames,
)
from .engine import (
    anchor_frames,
    arrest_drift,
    describe,
    free_between_segments,
    latent_signature,
    latent_tail,
    picture_only,
    push_preview,
    render_segment,
    take_tail,
)

CATEGORY = "ComfyUI_Marquee_Director"

STABILIZE_TOOLTIP = (
    "Flatten the finished cut's slow drift away from its own opening. 0 is off.\n\n"
    "Chained takes slide steadily darker and less saturated without any one join "
    "showing it. That trend is slow by definition, so it separates cleanly: each "
    "channel's per-frame mean is smoothed over two seconds, and what is left is "
    "the drift. The correction is a gain aimed at segment 1's level -- gains keep "
    "black black -- and because the curve it comes from is smooth it is continuous "
    "across every join, so stabilising cannot introduce a step at a cut.\n\n"
    "Gains are bounded at +/-25%, because the one thing this cannot tell apart "
    "from drift is a shot that is genuinely darker after a real lighting change. "
    "Real changes are large and survive the bound; accumulated drift is small and "
    "does not.\n\n"
    "0.7-1.0 is the useful range. It costs one extra decoding pass and no GPU. "
    "This fixes colour only -- for identity drift use drift_arrest on H3 Chain "
    "Settings, which acts on the generation rather than on the finished file."
)

DRIFT_ARREST_TOOLTIP = (
    "Pull each handoff's colour and exposure back toward SEGMENT 1's before pinning "
    "it, as a fraction of the measured error. 0 is off.\n\n"
    "Chained generation has no absolute reference: each segment is told only to "
    "continue from the last, so wherever segment N ended up becomes the truth for "
    "N+1 and a small bias compounds. Left alone a six-shot chain loses about 8 L* of "
    "face brightness and 0.24 of ArcFace cosine, none of it visible at any one join.\n\n"
    "0.4-0.6 is the useful range. It is deliberately a fraction and not a reset: the "
    "previous segment really did end where it ended and the cut keeps those frames, "
    "so correcting the whole error would put a visible step at every join. Large "
    "per-channel corrections are also clamped, so a shot that is genuinely darker "
    "because the light changed is not fought.\n\n"
    "Needs handoff_mode 'latent' -- the correction is applied to the latent that gets "
    "pinned, and the pixel path has no latent to correct."
)

HANDOFF_MODE_TOOLTIP = (
    "How the previous segment's ending is carried into the next one.\n\n"
    "'latent' (default) slices the tail out of the previous segment's own sampled "
    "latent and pins that. No VAE in the handoff at all, so the anchor is exact by "
    "construction rather than 30 dB of a round trip -- and it skips a 39-frame VAE "
    "encode per segment, so it is also faster.\n\n"
    "'pixel' decodes, writes the tail as an mp4, reads it back and re-encodes it "
    "with MiniMaxH3AddGuide. Slower and lossy, but it is the only path that can "
    "rescale, so use it if you change resolution partway through a session.\n\n"
    "Both tails are always written to disk, so a session can be resumed in either "
    "mode. Sessions rendered before latent tails existed have only the mp4 and fall "
    "back to 'pixel' automatically, with a line in the log."
)

SEAM_REDRAW_TOOLTIP = (
    "接缝处留给模型的重绘余量（对应 Director 的「重绘幅度」continuity_redraw）。\n\n"
    "段间回放区前段 token 的 mask = 1.0（完全交给模型，反正 join 会整段裁掉），"
    "最后 4 个 token 从 1.0 渐变到本值。值越小越锁死：\n"
    "  · 0.10（默认，与 Director 一致）→ 接缝 90% 沿用上一段最后一帧，续接最稳\n"
    "  · 调大 → 接缝处允许模型多改，过渡更自由但可能跳\n\n"
    "仅在 handoff_mode='latent' 时生效（走 apply_latent_continue 的 noise_mask）。\n\n"
    "改动会进缓存键，相关段会重渲。"
)


def _anchor_from_disk(sess, index, handoff_mode):
    """The anchor segment ``index`` hands to ``index+1``, read back off disk.

    Latent first when asked for, but never fatally: a session from before latent
    tails existed has only the mp4, and falling back to it is strictly better than
    refusing to resume.
    """
    tail = sess.tail_path(index)
    if handoff_mode == "latent":
        anchor = video_io.load_latent_tail(tail)
        if anchor is not None:
            return anchor
        log("no latent tail for %s -- resuming this handoff through the VAE instead",
            os.path.basename(tail))
    images, audio = video_io.load_clip(tail, prefer_wav=True)
    return {"images": images, "audio": audio}


def _handoff_anchor_frames(segment):
    """一段的回放锚应该取多少帧（用于 tail 缺失时现场从 seg_NN.mp4 切）。

    ★ segment 里**根本没有 handoff_seconds 这个键** ——
      _segment_from_widgets 生成的是已经算成帧数的 "handoff"
      （= guide_length(seconds_to_frames(handoff_seconds))），磁盘补齐的记录
      （Session.reconcile）也只有 "handoff"。以前代码直接
      ``segment.get("handoff_seconds")`` 恒为 None，一进 tail 缺失的兜底分支
      就 TypeError: float() argument must be ... not 'NoneType'。
      也就是说：**只要 tail 文件缺失，单段修复必崩**。

    取值顺序：现成的帧数 → handoff_seconds → 默认网格值 1.625s（39 帧）。
    用 ``if not frames`` 而不是 ``is None``：补回的记录在 tail 缺失时会写
    handoff=0，0 帧锚等于没锚，必须一并走默认值。
    """
    frames = segment.get("handoff")
    if not frames:
        secs = segment.get("handoff_seconds")
        frames = (guide_length(seconds_to_frames(secs)) if secs
                  else guide_length(seconds_to_frames(1.625)))
    return int(frames)

H3Settings = io.Custom("H3_SETTINGS")
H3Chain = io.Custom("H3_CHAIN")


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
class H3ChainSettingsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ChainSettings",
            display_name="H3 Chain Settings",
            category=CATEGORY,
            description="Everything the chain and the repair node both need: models, "
                        "sampler, canvas and seed. Wire one of these into either.\n\n"
                        "session_name is read by H3 Render Segment (for the first segment "
                        "of a chain; later ones inherit it via chain_state) but NOT by H3 "
                        "Repair Segment, which keeps its own session_name -- a repair "
                        "targets a session by name explicitly and does not have to match "
                        "whatever this settings bundle's chain is currently rendering.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="The H3 *video* VAE."),
                io.Vae.Input("audio_vae", tooltip="The H3 *audio* VAE."),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                # 960x544 = 16:9 @ 0.5 MP，正好是 ResolutionSelector 的预设值。
                # 不要手填非标尺寸：H3 的 VAE 先 /16 再 2x2 patchify，两条边都得是
                # 32 的倍数，且 latent 边长必须是偶数（496px -> 31 会让 patchify 崩）。
                io.Int.Input("width", default=960, min=32, max=16384, step=32,
                             tooltip="Canvas width. Prefer a ResolutionSelector preset "
                                     "(16:9 / 0.5 MP -> 960x544) over a hand-typed size."),
                io.Int.Input("height", default=544, min=32, max=16384, step=32,
                             tooltip="Canvas height. 960x544 is 16:9 at 0.5 MP."),
                # Deliberately not called "seed": the frontend bolts a
                # control_after_generate widget onto any widget with that name, and its
                # default is "randomize" -- which would silently re-roll the seed after
                # every run and re-render the entire chain from scratch each time.
                io.Int.Input("chain_seed", default=0, min=0, max=0xffffffffffffffff,
                             tooltip="Each segment derives its own seed from this one, so "
                                     "changing it re-renders the whole chain."),
                io.String.Input(
                    "session_name", default="my_chain", multiline=False,
                    tooltip="Folder under output/h3_continuous/. Lives here, not on each "
                            "H3 Render Segment, because a chain only ever has one session "
                            "no matter how many segment nodes are wired into it -- the "
                            "first segment (the one with chain_state unconnected) reads it "
                            "from here; every segment after that inherits the session "
                            "object it already resolved, off the chain_state wire."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="'match' scales reference images to the generation's "
                                       "pixel area; 'max' uses a 2048px short edge for the "
                                       "best identity fidelity and is several times slower, "
                                       "because reference tokens ride through every step."),
                io.Combo.Input("handoff_mode", options=["latent", "pixel"], default="latent",
                               advanced=True, tooltip=HANDOFF_MODE_TOOLTIP),
                io.Float.Input("drift_arrest", default=0.0, min=0.0, max=1.0, step=0.05,
                               round=False, advanced=True, tooltip=DRIFT_ARREST_TOOLTIP),
                # 借鉴 ComfyUI_MiniMaxH3_Director 的 continuity_redraw（「重绘幅度」）。
                # 只影响段间回放区的 mask 下限，与任务模式无关，ref2va 线上通用。
                # 必须挂在列表**末尾**：本节点全 required 段，widget 顺序即 schema
                # 顺序，插在中间会让已存工作流的 widgets_values 整体错位。
                io.Float.Input("seam_redraw", default=0.10, min=0.0, max=0.95,
                               step=0.05, round=False, advanced=True,
                               tooltip=SEAM_REDRAW_TOOLTIP),
            ],
            outputs=[H3Settings.Output(display_name="settings")],
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, sampler, sigmas, width, height,
                chain_seed, session_name, ref_image_size, handoff_mode="latent",
                drift_arrest=0.0, seam_redraw=0.10) -> io.NodeOutput:
        return io.NodeOutput({
            "model": model, "clip": clip, "vae": vae, "audio_vae": audio_vae,
            "sampler": sampler, "sigmas": sigmas,
            "width": width, "height": height, "seed": chain_seed,
            "session_name": session_name, "ref_image_size": ref_image_size,
            "handoff_mode": handoff_mode, "drift_arrest": float(drift_arrest),
            "seam_redraw": float(seam_redraw),
        })


# ---------------------------------------------------------------------------
# render one segment
# ---------------------------------------------------------------------------
PROMPT_TOOLTIP = (
    "H3 prompt for this segment. Reference tags are numbered per type in socket "
    "order: image_0 is <Picture 1>, video_0 is <Video 1>, audio_0 is <Audio 1>.\n\n"
    "The clip carried in from the previous segment gets NO tag -- it never reaches "
    "the tokenizer -- so describe it in prose (\"the segment opens on an exact "
    "replay of the preceding shot and carries straight on from it\") and never "
    "label it. A <Video 1> written for it would be an unresolved reference.\n\n"
    "Nothing new can be scheduled inside the replayed opening: with a 1.63s "
    "handoff, an entrance written for \"the one-second mark\" lands in frames that "
    "are already pinned."
)

HANDOFF_TOOLTIP = (
    "How much of THIS segment's ending is replayed as the opening of the NEXT one.\n\n"
    "Snapped down to a valid guide length: 5, 22, 39 or 56 frames (0.21s, 0.92s, "
    "1.63s, 2.33s). 39 is the sweet spot -- guide fidelity peaks exactly at frames "
    "0, 17 and 34 and sags between, which is the latent grouping showing through.\n\n"
    "KEEP IT NEAR A FIFTH OF THE SEGMENT. 1.63s is 20% of an 8s segment and joins "
    "cleanly; the same 1.63s is 31% of a 5s segment, and there the seams measurably "
    "degrade -- too little free runway is left after the pinned region. Shorten the "
    "handoff for short segments, or lengthen the segments.\n\n"
    "A longer handoff also costs you that much new footage: at 8s segments with a "
    "1.63s handoff, each segment after the first contributes 6.37s to the finished cut.\n\n"
    "On the LAST segment of a take, set this to 0 -- there is no next segment to hand "
    "off to, and a nonzero value here just writes a tail clip nothing will ever read."
)


SEGMENT_MODEL_TOOLTIP = (
    "A MODEL for THIS SHOT ONLY, overriding the one on H3 Chain Settings. Leave it "
    "unconnected and the segment renders on the settings model, which is what every "
    "segment did before this input existed.\n\n"
    "This is where a per-shot LoRA goes: wire H3 Chain Settings' own model source "
    "through a LoraLoaderModelOnly (or any LoRA stacker) and into here, and only "
    "this segment sees it. Different LoRAs on different segments is the point -- a "
    "character LoRA that carries the whole take, a motion or style LoRA that comes "
    "in for one shot and leaves again.\n\n"
    "Keep the turbo LoRA (and anything else the chain relies on) in the stack you "
    "wire in: this input REPLACES the settings model rather than adding to it, so a "
    "segment fed a bare checkpoint would sample at 4 steps' worth of sigmas without "
    "the LoRA those sigmas were chosen for.\n\n"
    "MODEL only, by design. H3 LoRAs are diffusion-side, and routing one through "
    "the Qwen3-VL text encoder as well is not something they are trained for.\n\n"
    "Changing the LoRA or its strength re-renders this segment, and every segment "
    "after it, even with resume on -- the cache key sees the patch list."
)


def _segment_schema():
    """The shot fields shared by H3 Render Segment and H3 Repair Segment."""
    return [
        io.String.Input("prompt", multiline=True, dynamic_prompts=False,
                        tooltip=PROMPT_TOOLTIP),
        io.Float.Input(
            "seconds", default=8.0, min=0.25, max=15.0, step=0.25, round=False,
            tooltip="Length of this segment, snapped up to H3's 17k+5 frame grid at "
                    "24 fps (8s -> 192 frames). The trained range is about 5-15s."),
        io.Float.Input("handoff_seconds", default=1.625, min=0.0, max=4.0, step=0.125,
                       round=False, advanced=True, tooltip=HANDOFF_TOOLTIP),
        io.Int.Input(
            "seed_override", default=0, min=0, max=0xffffffffffffffff, advanced=True,
            tooltip="0 = derive this segment's seed from the chain seed. Set anything "
                    "else to re-roll just this segment. (Not named 'seed' on purpose "
                    "-- that name gets an automatic randomise control, which would "
                    "re-render this segment on every run.)"),
        io.Model.Input("model", optional=True, tooltip=SEGMENT_MODEL_TOOLTIP),
        io.Autogrow.Input(
            "images", optional=True,
            tooltip="Reference images -> <Picture 1..9>.",
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input("image", optional=True,
                                     tooltip="Reference image for this segment."),
                prefix="image_", min=1, max=9)),
        io.Autogrow.Input(
            "videos", optional=True,
            tooltip="Reference videos (frames at 24 fps, 2-15s) -> <Video 1..3>.",
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input("video", optional=True,
                                     tooltip="Reference video frames."),
                prefix="video_", min=0, max=3)),
        io.Autogrow.Input(
            "video_audios", optional=True,
            tooltip="Soundtrack of the same-numbered reference video. Each one is "
                    "presented as its own <Audio j>, right before its <Video k>.",
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input("video_audio", optional=True,
                                     tooltip="Soundtrack of the same-numbered video."),
                prefix="video_audio_", min=0, max=3)),
        io.Autogrow.Input(
            "audios", optional=True,
            tooltip="Standalone reference audio -> <Audio 1..3>.",
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input("audio", optional=True,
                                     tooltip="Standalone reference audio."),
                prefix="audio_", min=0, max=3)),
    ]


def _segment_from_widgets(prompt, seconds, handoff_seconds, seed_override,
                          images=None, videos=None, video_audios=None, audios=None):
    length = generation_length(seconds_to_frames(seconds))
    handoff = guide_length(seconds_to_frames(handoff_seconds))
    if handoff >= length:
        raise ValueError(
            "handoff_seconds (%.2fs -> %d frames) must be shorter than the segment "
            "itself (%.2fs -> %d frames), or the next segment would be nothing but "
            "replay." % (handoff_seconds, handoff, seconds, length))
    return {
        "prompt": prompt, "seconds": seconds, "length": length, "handoff": handoff,
        "seed": int(seed_override),
        "images": [t for _, t in ordered_autogrow(images)],
        "videos": ordered_autogrow(videos),
        "video_audios": ordered_autogrow(video_audios),
        "audios": [a for _, a in ordered_autogrow(audios)],
    }


def _with_model(settings, model):
    """The settings bundle this shot renders on, with its own model swapped in.

    A copy, never a mutation: one H3 Chain Settings node feeds every segment in the
    chain, so writing into it would leak this shot's LoRA into all the others -- and
    into their cache keys, re-rendering the lot.
    """
    if model is None:
        return settings
    return dict(settings, model=model)


CHAIN_STATE_TOOLTIP = (
    "Wire in the PREVIOUS H3 Render Segment's chain_state to continue its take. Leave "
    "unconnected on the first segment of a session, or to start a new one.\n\n"
    "Also accepts H3 Load Session's chain output -- wiring that in here appends new "
    "segments onto a session that was already finished, without re-rendering it."
)


def _resolve_seed(settings, segment, index):
    if segment["seed"]:
        return int(segment["seed"])
    # 9973 is prime, so neighbouring segments never collide on a nearby chain seed.
    return (int(settings["seed"]) + (index + 1) * 9973) % (1 << 63)


def _summarize(chain):
    records = chain["segments"]
    total = sum(r["length"] for r in records) - sum(r["handoff"] for r in records[:-1])
    lines = ["%d segments -> %.2fs (%d frames) once the replayed handoffs come off"
             % (len(records), total / float(FPS), total)]
    for r in records:
        notes = "  (repaired)" if r.get("repaired") else ""
        if r.get("drift_correction"):
            notes += "  drift -%.3f" % r["drift_correction"]
        lines.append("  %2d  %5.2fs  handoff %4.2fs  seed %d%s"
                     % (r["index"] + 1, r["seconds"], r["handoff"] / float(FPS), r["seed"],
                        notes))
    lines.append("  in %s" % chain["dir"])
    return "\n".join(lines)


class H3RenderSegmentNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RenderSegment",
            display_name="H3 Render Segment",
            category=CATEGORY,
            description="One shot in a continuous take: a prompt, its references, how long "
                        "it runs -- and it renders right here. Chain shots by wiring this "
                        "node's chain_state output into the next H3 Render Segment's "
                        "chain_state input; each one after the first opens on an exact "
                        "replay of the one before it, anchored with MiniMaxH3AddGuide, so "
                        "the cut is invisible.\n\n"
                        "This node's video output is real the moment THIS segment finishes "
                        "-- you do not wait for the rest of the chain to preview, save or "
                        "post-process it. The segment is also written to "
                        "output/h3_continuous/<session>/ and reused on the next queue, so a "
                        "crash, an interrupt or an edited prompt only costs the segments "
                        "that actually changed.",
            inputs=[
                H3Settings.Input("settings"),
                H3Chain.Input("chain_state", optional=True, tooltip=CHAIN_STATE_TOOLTIP),
                io.Boolean.Input("resume", default=True,
                                 tooltip="Reuse this segment if it is already rendered in the "
                                         "session with unchanged settings. A segment's "
                                         "fingerprint includes the one before it, so editing "
                                         "an earlier segment's prompt still invalidates this "
                                         "one even with resume on."),
                *_segment_schema(),
                io.Boolean.Input("unload_models_after", default=False, advanced=True,
                                 tooltip="Unload models after this segment renders, before the "
                                         "next one starts. OOM-only: it forces a full reload "
                                         "of the UNet and text encoder from disk, which "
                                         "dominates runtime on a small box. Turn it on for "
                                         "just the segment where you actually OOM."),
            ],
            outputs=[io.Video.Output(display_name="video"),
                     H3Chain.Output(display_name="chain_state"),
                     io.String.Output(display_name="summary")],
        )

    @classmethod
    def execute(cls, settings, resume, prompt, seconds, handoff_seconds,
                seed_override, unload_models_after, chain_state=None, model=None,
                images=None, videos=None, video_audios=None,
                audios=None) -> io.NodeOutput:
        # Before segment_key: model_digest reads settings["model"], so swapping it
        # here is what makes changing a shot's LoRA invalidate that shot's cache.
        settings = _with_model(settings, model)
        segment = _segment_from_widgets(prompt, seconds, handoff_seconds, seed_override,
                                        images, videos, video_audios, audios)
        handoff = segment["handoff"]

        if chain_state is None:
            sess = session_mod.Session(settings["session_name"])
            index, previous_key, anchor, records = 0, None, None, []
            reference = None
        else:
            sess = chain_state["sess_obj"]
            index = chain_state["index"] + 1
            previous_key = chain_state["key"]
            anchor = chain_state["anchor"]
            records = list(chain_state["segments"])
            # Segment 1's signature, carried down the whole chain. It is the only
            # absolute reference a chain has: the one shot rendered from the prompt
            # and the reference image alone, with no inherited anchor to be wrong about.
            reference = chain_state.get("reference")

        segment["resolved_seed"] = _resolve_seed(settings, segment, index)
        key = session_mod.segment_key(settings, segment, handoff, previous_key)
        manifest = sess.load() if resume else None
        # 断点续渲：manifest 可能比磁盘落后（少记了已经渲好的段），先按磁盘补齐，
        # 那些段才会被判定为命中而不是重渲。resume 关闭时完全不碰。
        if resume:
            manifest = sess.reconcile(manifest)

        if resume and sess.cached(manifest, index, key, needs_tail=bool(handoff)):
            log("%s -- reusing %s", describe(segment, index, anchor is not None),
                os.path.basename(sess.segment_path(index)))
            rendered_length = video_io.frame_count(sess.segment_path(index))
            new_anchor = (_anchor_from_disk(sess, index, settings["handoff_mode"])
                          if handoff else None)
            signature = session_mod.signature_from_record(manifest["segments"][index])
            # A reused segment still gets a thumbnail, so a resumed run visibly walks
            # through what it is keeping instead of appearing to stall.
            preview_frame = video_io.last_frame(sess.segment_path(index))
        else:
            log("%s", describe(segment, index, anchor is not None))
            # 这一段是真的要烧显卡了 —— 它后面靠"磁盘上有文件"判定命中的段
            # 全部作废，否则会出现新旧尾巴接不上的片子。
            sess.note_render()
            images_out, audio_out, samples = render_segment(
                settings, segment, start_anchor=anchor)
            rendered_length = int(images_out.shape[0])
            video_io.save_clip(sess.segment_path(index), images_out, audio_out)

            # Both forms of the handoff go to disk whatever mode rendered it, so the
            # session can be resumed, repaired or extended in either mode later.
            pixel_tail = take_tail(images_out, audio_out, handoff)
            lat_tail = latent_tail(samples, handoff)
            signature = latent_signature(samples)
            if pixel_tail is not None:
                video_io.save_clip(sess.tail_path(index), pixel_tail[0], pixel_tail[1],
                                   exact_audio=True)
                # Saved uncorrected, deliberately: the file on disk is what this
                # segment actually ended on, which is what the repair node has to pin
                # against. Drift correction is applied when the anchor is consumed.
                video_io.save_latent_tail(sess.tail_path(index), lat_tail)

            if handoff:
                new_anchor = (lat_tail if settings["handoff_mode"] == "latent"
                              else {"images": pixel_tail[0], "audio": pixel_tail[1]})
            else:
                new_anchor = None
            preview_frame = images_out[-1].clone()
            del images_out, audio_out, samples, pixel_tail, lat_tail
            free_between_segments(unload_models_after)

        # Segment 1 seeds the reference and then it rides the chain unchanged: it is
        # the only shot rendered from the prompt and the reference image alone, with
        # no inherited anchor that could already be wrong.
        reference = reference or signature
        correction = 0.0
        if new_anchor is not None and settings.get("drift_arrest"):
            if new_anchor.get("video_latent") is None:
                log("drift_arrest is set but handoff_mode is 'pixel', which has no "
                    "latent to correct -- segment %d's handoff is going out uncorrected",
                    index + 1)
            corrected = arrest_drift(new_anchor, reference, signature,
                                     settings["drift_arrest"])
            correction = corrected.get("drift_correction", 0.0) if corrected else 0.0
            if correction:
                log("segment %d: drift correction %.4f per latent channel applied to "
                    "the handoff", index + 1, correction)
            new_anchor = corrected

        pbar = comfy.utils.ProgressBar(1)
        push_preview(pbar, preview_frame, 1, 1)
        del preview_frame

        record = {
            "index": index, "key": key, "length": rendered_length, "handoff": handoff,
            "seconds": round(rendered_length / float(FPS), 3),
            "seed": segment["resolved_seed"], "prompt": segment["prompt"],
            "file": os.path.basename(sess.segment_path(index)),
            # Kept so a resumed run can still measure how far it has drifted without
            # decoding every segment it is reusing.
            "signature": session_mod.signature_to_record(signature),
        }
        if correction:
            record["drift_correction"] = round(correction, 6)
        # Slicing to `index` rather than appending covers rewiring in a shorter earlier
        # segment upstream of a session that used to be longer.
        records = records[:index] + [record]
        # Written every segment, not once at the end: a hard kill (not just a clean
        # interrupt) still leaves the manifest exactly matching what is on disk, so the
        # next queue's resume never re-renders something that actually finished.
        sess.save(records, extra={"width": settings["width"],
                                  "height": settings["height"], "fps": FPS})

        chain = {"session": sess.name, "dir": sess.dir, "segments": records,
                 "sess_obj": sess, "anchor": new_anchor, "index": index, "key": key,
                 "reference": reference}
        summary = _summarize(chain)
        log("%s", summary.replace("\n", " | "))
        return io.NodeOutput(VideoFromFile(sess.segment_path(index)), chain, summary)


# ---------------------------------------------------------------------------
# JSON script batch rendering
# ---------------------------------------------------------------------------
def _script_asset(shots_json):
    """Parse the Ref2VA Auto board JSON without importing that optional plugin.

    Keeping this parser here makes the continuous pack useful on its own, while the
    payload stays deliberately compatible with ``MinimaxH3ScriptConverter`` and
    ``MinimaxH3ShotSplit``: global.roles/prop/scene plus shots_info/appear.
    """
    raw = str(shots_json or "").strip()
    if not raw:
        raise ValueError("分镜资产结果JSON为空。请连接 H3 剧本转换器或分镜拆解节点。")
    # LLM nodes normally return plain JSON, but accept a fenced/object-prefixed
    # response too so a hand-edited board remains usable.
    start = raw.find("{")
    if start < 0:
        raise ValueError("分镜资产结果JSON没有 JSON 对象。")
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("分镜资产结果JSON无效：%s" % exc) from exc
    if not isinstance(data, dict):
        raise ValueError("分镜资产结果JSON必须是对象。")
    shots = data.get("shots_info") or []
    if not isinstance(shots, list) or not shots:
        raise ValueError("分镜资产结果JSON没有 shots_info；请先完成剧本拆解。")
    return data


def _load_script_image(path, label):
    """Load an asset image saved by Ref2VA Auto into Comfy's IMAGE tensor form."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps, ImageSequence

    path = os.path.abspath(str(path or "").strip())
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError("%s 的参考图不存在：%s" % (label, path))
    image = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(image):
        rgb = ImageOps.exif_transpose(frame).convert("RGB")
        frames.append(torch.from_numpy(np.asarray(rgb).astype(np.float32) / 255.0)[None, ...])
    if not frames:
        raise RuntimeError("无法解码 %s 的参考图：%s" % (label, path))
    return torch.cat(frames, dim=0)


def _images_autogrow(images):
    """Board reference images arrive as a plain list; autogrow inputs want a dict.

    ``_segment_from_widgets`` runs them through ``ordered_autogrow``, which iterates
    ``.items()`` -- passing the raw list raises AttributeError before a single frame
    is rendered.  Ordinals only order the slots, so ``image_0..`` matches the
    ``<Picture N>`` numbering the prompt already assumes.
    """
    if not images:
        return None
    return {"image_%d" % i: t for i, t in enumerate(images)}


def _script_segment(asset, shot, include_global_prompt=True,
                    ref_images_override=None):
    """Resolve one board row into an H3 prompt and ordered <Picture N> tensors.

    Two image sources are supported:
    - The original Ref2VA Auto board: global roles/prop/scene plus shot.appear.
    - Local shot prompts: shot.image_paths, written by H3 Shot Prompt nodes.
    If ``ref_images_override`` is supplied it wins over both.
    """
    images = []
    if ref_images_override:
        images = list(ref_images_override)
    elif shot.get("image_paths"):
        for path in shot["image_paths"]:
            tensor = _load_script_image(path, "local shot ref")
            if tensor is not None:
                images.append(tensor)
    else:
        global_assets = asset.get("global") if isinstance(asset.get("global"), dict) else {}
        by_kind = {}
        for kind in ("roles", "prop", "scene"):
            by_kind[kind] = {
                int(item.get("id")): item
                for item in (global_assets.get(kind) or [])
                if isinstance(item, dict) and str(item.get("id", "")).strip().lstrip("-").isdigit()
            }
        appear = shot.get("appear") if isinstance(shot.get("appear"), dict) else {}
        # This is the same order used by Ref2VA Auto's subject_block_for_shot:
        # roles, props, then scenes.  It preserves the <Picture N> indices in its prompt.
        for kind in ("roles", "prop", "scene"):
            for raw_id in appear.get(kind) or []:
                try:
                    item = by_kind[kind].get(int(raw_id))
                except (TypeError, ValueError):
                    item = None
                if not item:
                    continue
                tensor = _load_script_image(item.get("bing_image_path") or item.get("image_path"),
                                            "%s %s" % (kind, item.get("name") or raw_id))
                if tensor is not None:
                    images.append(tensor)
    prompt = str(shot.get("shot") or "").strip()
    if not prompt:
        raise ValueError("分镜 %s 没有 H3 提示词。" % (shot.get("id") or "?"))
    global_assets = asset.get("global") if isinstance(asset.get("global"), dict) else {}
    global_prompt = str(global_assets.get("global_prompt") or "").strip()
    if include_global_prompt and global_prompt and "overall_soundscape:" not in prompt.lower():
        prompt += "\n\n" + global_prompt
    try:
        seconds = float(shot.get("duration") or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds <= 0:
        raise ValueError("分镜 %s 的时长无效。" % (shot.get("id") or "?"))
    return prompt, seconds, images


def _safe_script_handoff(seconds, requested, is_last):
    """Never hand off a guide as long as the shot itself (short shots are valid)."""
    if is_last:
        return 0.0
    requested = max(0.0, float(requested))
    length = generation_length(seconds_to_frames(seconds))
    handoff = guide_length(seconds_to_frames(requested))
    return requested if handoff < length else 0.0


def _settings_for_script(settings, asset):
    """Let the script board's resolution drive H3, with safe settings fallbacks."""
    effective = dict(settings)
    for key in ("width", "height"):
        try:
            value = int(asset.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value >= 32:
            effective[key] = value
    return effective


class H3ScriptBatchRenderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ScriptBatchRender",
            display_name="H3 Script Batch Render (Unlimited Shots)",
            category=CATEGORY,
            description=(
                "Renders every row in a Ref2VA Auto 分镜资产结果JSON as one H3 continuous "
                "session.  The number of shots comes from shots_info, not from fixed graph "
                "nodes.  Each completed shot is checkpointed, so resume reuses it after an "
                "interrupt and only continues with changed/downstream shots."
            ),
            inputs=[
                H3Settings.Input("settings"),
                io.String.Input("shots_json", display_name="分镜资产结果JSON", force_input=True),
                io.Boolean.Input("resume", default=True),
                io.Float.Input("handoff_seconds", default=1.625, min=0.0, max=4.0,
                               step=0.125, round=False, advanced=True),
                io.Int.Input("max_shots", display_name="最多渲染分镜（0=全部）", default=0,
                             min=0, max=9999),
                io.Boolean.Input("unload_models_after", default=False, advanced=True),
                H3Chain.Input(
                    "chain_state", optional=True,
                    tooltip="续接已完成的会话：把「H3 Load Session」的 chain 接进来，"
                            "本次的 shots_info 会渲染成该会话的第 N+1、N+2… 段，"
                            "而不是新建会话——这就是无限分镜的追加模式。"
                            "留空则从第一段开始新建会话。"),
                io.Boolean.Input(
                    "duration_is_new_content", default=True, advanced=True,
                    tooltip="分镜时长口径。开：JSON 里的 duration 是「新增时长」，"
                            "渲染时自动 +handoff_seconds 作为生成长度"
                            "（新增 10s + 1.6s 回放 = 生成 11.6s），成片总时长等于各段新增之和。"
                            "关：duration 直接当生成长度用。"),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    tooltip="可选：用同一组参考图覆盖 shots_json 里的所有分镜参考图。"
                            "不连时，H3 Script Batch Render 按 JSON 原样加载图片；"
                            "连入后，所有分镜都使用这组 <Picture 1..9>。",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", optional=True),
                        prefix="ref_image_", min=0, max=9)),
            ],
            outputs=[
                io.Video.Output(display_name="最后分镜视频"),
                H3Chain.Output(display_name="chain"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, settings, shots_json, resume=True, handoff_seconds=1.625,
                max_shots=0, unload_models_after=False, chain_state=None,
                duration_is_new_content=True, ref_images=None):
        asset = _script_asset(shots_json)
        shots = [shot for shot in asset.get("shots_info") or [] if isinstance(shot, dict)]
        limit = int(max_shots or 0)
        if limit > 0:
            shots = shots[:limit]
        if not shots:
            raise ValueError("没有可渲染的分镜。")
        settings = _settings_for_script(settings, asset)
        ref_override = [t for _, t in ordered_autogrow(ref_images)] if ref_images else None
        state = chain_state
        last_video = None
        for index, shot in enumerate(shots):
            prompt, seconds, images = _script_segment(
                asset, shot, ref_images_override=ref_override)
            handoff = _safe_script_handoff(seconds, handoff_seconds, index == len(shots) - 1)
            if duration_is_new_content and handoff:
                # duration in the board is the NEW running time; the replayed opening is
                # extra material to generate, not screen time.
                seconds = min(seconds + float(handoff_seconds), 15.0)
            output = H3RenderSegmentNode.execute(
                settings, bool(resume), prompt, seconds, handoff, 0,
                bool(unload_models_after), chain_state=state,
                images=_images_autogrow(images),
            )
            last_video, state = output[0], output[1]
        return io.NodeOutput(last_video, state, _summarize(state))


class H3ScriptRepairSegmentNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ScriptRepairSegment",
            display_name="H3 Script Repair Segment",
            category=CATEGORY,
            description=(
                "Repairs one segment from the same Ref2VA Auto JSON.  It automatically "
                "restores that shot's prompt and ordered references, then pins both joins "
                "when the repaired shot is not the final one."
            ),
            inputs=[
                H3Settings.Input("settings"),
                io.String.Input("session_name", default="my_chain"),
                io.String.Input("shots_json", display_name="分镜资产结果JSON", force_input=True),
                io.Int.Input("segment_number", default=1, min=1, max=9999),
                io.Float.Input("handoff_seconds", default=1.625, min=0.0, max=4.0,
                               step=0.125, round=False, advanced=True),
                io.Boolean.Input("pin_ending", default=True),
                io.Boolean.Input(
                    "duration_is_new_content", default=True, advanced=True,
                    tooltip="与「H3 Script Batch Render」保持同一口径：开=JSON 的 duration "
                            "是新增时长，本段按 duration+handoff 生成，长度才对得上原段，"
                            "pin_ending 才不会因替换段太短而报错。"),
                H3Chain.Input(
                    "chain_state", optional=True,
                    tooltip="把「H3 Script Batch Render」的 chain 接进来：既保证它在本次队列里"
                            "先跑完（会话 manifest 一定存在），也允许对刚渲染完、manifest 还没"
                            "落盘的会话直接修复。不接也能修复磁盘上已存在的会话。"),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    tooltip="可选：覆盖本段分镜的参考图。与 H3 Script Batch Render 的 "
                            "ref_images 行为一致。",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", optional=True),
                        prefix="ref_image_", min=0, max=9)),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                H3Chain.Output(display_name="chain"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, settings, session_name, shots_json, segment_number=1,
                handoff_seconds=1.625, pin_ending=True, duration_is_new_content=True,
                chain_state=None, ref_images=None):
        asset = _script_asset(shots_json)
        settings = _settings_for_script(settings, asset)
        shots = [shot for shot in asset.get("shots_info") or [] if isinstance(shot, dict)]
        index = int(segment_number) - 1
        if index < 0 or index >= len(shots):
            raise ValueError("分镜号 %d 超出 JSON 的 %d 个分镜。" % (segment_number, len(shots)))
        ref_override = [t for _, t in ordered_autogrow(ref_images)] if ref_images else None
        prompt, seconds, images = _script_segment(
            asset, shots[index], ref_images_override=ref_override)
        handoff = _safe_script_handoff(seconds, handoff_seconds, index == len(shots) - 1)
        if duration_is_new_content and handoff:
            seconds = min(seconds + float(handoff_seconds), 15.0)
        return H3RepairSegmentNode.execute(
            settings, session_name, int(segment_number), prompt, seconds, handoff, 0,
            bool(pin_ending), images=_images_autogrow(images), chain_state=chain_state,
        )


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
class H3ChainToVideoNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ChainToVideo",
            display_name="H3 Chain to Video",
            category=CATEGORY,
            description="Joins a session's segments into one cut, dropping each segment's "
                        "replayed opening so the motion does not stutter once per join.",
            inputs=[
                H3Chain.Input("chain"),
                io.Float.Input("crf", default=14.0, min=0.0, max=51.0, step=1.0, advanced=True,
                               tooltip="Quality of the joined file. Lower is better and "
                                       "bigger; 0 is lossless."),
                io.Float.Input("stabilize", default=0.0, min=0.0, max=1.0, step=0.05,
                               round=False, advanced=True, tooltip=STABILIZE_TOOLTIP),
            ],
            outputs=[io.Video.Output(display_name="video"),
                     io.String.Output(display_name="path")],
        )

    @classmethod
    def execute(cls, chain, crf, stabilize=0.0) -> io.NodeOutput:
        records = chain["segments"]
        if not records:
            raise ValueError("this chain has no rendered segments")
        parts = []
        for index, record in enumerate(records):
            path = os.path.join(chain["dir"], record["file"])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "segment %d is missing from the session (%s). Re-queue the chain to "
                    "render it." % (index + 1, path))
            skip = 0 if index == 0 else int(records[index - 1]["handoff"])
            parts.append((path, skip))

        out_path = os.path.join(chain["dir"], "%s.mp4" % chain["session"])
        out_path, frames = video_io.join(parts, out_path, crf=crf,
                                         stabilize=stabilize)
        log("joined %d segments -> %d frames (%.2fs) -> %s",
            len(parts), frames, frames / float(FPS), out_path)
        return io.NodeOutput(VideoFromFile(out_path), out_path)


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------
class H3RepairSegmentNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RepairSegment",
            display_name="H3 Repair Segment",
            category=CATEGORY,
            description="Re-render one segment of a finished session, in place.\n\n"
                        "Normally regenerating segment N changes its ending, so every segment "
                        "after it has to be re-rendered too. This node pins BOTH ends: the "
                        "opening to the previous segment's handoff clip, and the ending to the "
                        "very clip the next segment already opens on. Nothing downstream moves.",
            inputs=[
                H3Settings.Input("settings"),
                io.String.Input("session_name", default="my_chain", multiline=False),
                io.Int.Input("segment_number", default=2, min=1, max=32,
                             tooltip="1-based, matching seg_NN.mp4 in the session folder."),
                *_segment_schema(),
                io.Boolean.Input("pin_ending", default=True,
                                 tooltip="Keep the ending the next segment continues from. Turn "
                                         "this off only for the last segment, or when you mean "
                                         "to re-render everything after this one."),
                H3Chain.Input(
                    "chain_state", optional=True,
                    tooltip="Wire the LAST H3 Render Segment's chain_state in here and the "
                            "chain is guaranteed to have rendered before this node runs -- "
                            "which is what lets one queue render a session and repair a "
                            "segment of it in one go. Without it this node can only repair a "
                            "session whose manifest is already on disk, and it raises if it "
                            "is not. Segments already on disk are still reused (resume), so "
                            "wiring this in costs a model load and nothing else on a session "
                            "that has not changed."),
            ],
            outputs=[io.Video.Output(display_name="video"),
                     H3Chain.Output(display_name="chain"),
                     io.String.Output(display_name="summary")],
        )

    @classmethod
    def execute(cls, settings, session_name, segment_number, prompt, seconds,
                handoff_seconds, seed_override, pin_ending, model=None, images=None,
                videos=None, video_audios=None, audios=None,
                chain_state=None) -> io.NodeOutput:
        settings = _with_model(settings, model)
        segment = _segment_from_widgets(prompt, seconds, handoff_seconds, seed_override,
                                        images, videos, video_audios, audios)
        sess = session_mod.Session(session_name)
        manifest = sess.load()
        # 磁盘上渲好的段可能比 manifest 里记录的多（见 Session.reconcile）。
        # 不补齐的话，修第 3 段会因为 manifest 只有 1 条记录而直接报越界。
        manifest = sess.reconcile(manifest)
        if not manifest and chain_state is not None:
            # The chain rendered earlier in this very run: its segments are on disk and
            # its manifest is in memory, even though nothing has been written yet for a
            # session that has never completed a run before.
            if chain_state.get("session") != sess.name:
                raise ValueError(
                    "chain_state is session '%s' but this repair node targets '%s'. Pick "
                    "one: leave chain_state unconnected and repair a session from disk, or "
                    "name the session the chain is actually rendering."
                    % (chain_state.get("session"), sess.name))
            sess = chain_state["sess_obj"]
            manifest = {"version": session_mod.MANIFEST_VERSION, "session": sess.name,
                        "segments": list(chain_state["segments"]),
                        "width": settings["width"], "height": settings["height"],
                        "fps": FPS}
            log("no manifest on disk yet -- repairing against the chain just rendered "
                "in this run (%d segments)", len(manifest["segments"]))
        if not manifest:
            raise FileNotFoundError(
                "no manifest in %s -- run the chain workflow for this session first, or "
                "wire the last H3 Render Segment's chain_state into this node so both "
                "happen in one queue." % sess.dir)
        records = manifest["segments"]
        index = int(segment_number) - 1
        if index < 0 or index >= len(records):
            raise ValueError("session '%s' has %d segments; segment_number %d is out of range."
                             % (sess.name, len(records), segment_number))

        start_anchor = None
        if index > 0:
            previous_tail = sess.tail_path(index - 1)
            if os.path.exists(previous_tail):
                start_anchor = _anchor_from_disk(sess, index - 1, settings["handoff_mode"])
            else:
                # 尾接片段缺了不等于没法修：段 N-1 的 ``seg_NN.mp4`` 本身通常还在
                # （tail 是额外产物，老会话、被清理过的会话、或只渲了一部分的
                # 会话都可能没有）。这里退化成"用上一段的最后一段像素当锚"，
                # 与 ``_anchor_from_disk`` 的 pixel 回落是同一条路。
                #
                # 以前的写法是直接 raise FileNotFoundError，把"可恢复"当"致命"，
                # 结果是：只要第 1 段的 tail 缺失，修第 2 段就整个跑不动，而错误
                # 信息 ("segment 1 has no handoff clip") 完全没透露其实还能救。
                prev_file = sess.segment_path(index - 1)
                if not os.path.exists(prev_file):
                    raise FileNotFoundError(
                        "segment %d has neither a handoff clip (%s) nor its own render "
                        "(%s), so segment %d has nothing to open on."
                        % (index, previous_tail, prev_file, segment_number))
                images, audio = video_io.load_clip(prev_file, prefer_wav=True)
                if images is None or len(images) == 0:
                    raise FileNotFoundError(
                        "segment %d's render (%s) is unreadable, so segment %d has "
                        "nothing to open on." % (index, prev_file, segment_number))
                # 锚长必须与"正常渲染时 tail 会取多少帧"一致，否则接缝处回放区
                # 对不上。见 _handoff_anchor_frames（第 30 轮：以前这里读
                # segment["handoff_seconds"]，而该键根本不存在 → 恒 None → 崩溃）。
                keep = min(_handoff_anchor_frames(segment), len(images))
                if keep <= 0:
                    raise FileNotFoundError(
                        "segment %d's render (%s) is too short to anchor a handoff "
                        "from." % (index, prev_file))
                log("segment %d has no handoff clip -- anchoring the opening on the "
                    "last %d frame(s) of %s instead",
                    segment_number, keep, os.path.basename(prev_file))
                start_anchor = {"images": images[-keep:], "audio": audio}

        end_anchor = None
        is_last = index == len(records) - 1
        if pin_ending and not is_last:
            own_tail = sess.tail_path(index)
            if os.path.exists(own_tail):
                end_anchor = picture_only(
                    _anchor_from_disk(sess, index, settings["handoff_mode"]))
            else:
                # ★ 第 30 轮：与上面 opening 分支对等 —— tail 的语义就是
                #   "本段视频的最后 N 帧"，既然 seg_NN.mp4 还在，就现场切一段
                #   出来当结尾锚，而不是直接 raise。
                #   以前直接 raise 的后果：老会话 / 产物被清理过的会话
                #   （本机会话连一个 tail 都没有）**根本没法单段重渲**，
                #   而报错只说 "no handoff clip"，完全没透露其实还能救。
                #   这不是降级 —— 数据源语义等价，锚长同样按回放网格取。
                own_file = sess.segment_path(index)
                if not os.path.exists(own_file):
                    raise FileNotFoundError(
                        "segment %d has neither a handoff clip (%s) nor its own "
                        "render (%s), so its ending cannot be pinned."
                        % (segment_number, own_tail, own_file))
                e_images, _ = video_io.load_clip(own_file, prefer_wav=True)
                if e_images is None or len(e_images) == 0:
                    raise FileNotFoundError(
                        "segment %d's render (%s) is unreadable, so its ending "
                        "cannot be pinned."
                        % (segment_number, os.path.basename(own_file)))
                e_keep = min(_handoff_anchor_frames(segment), len(e_images))
                if e_keep <= 0:
                    raise FileNotFoundError(
                        "segment %d's render (%s) is too short to pin an ending "
                        "from." % (segment_number, os.path.basename(own_file)))
                log("segment %d has no handoff clip -- pinning the ending on the "
                    "last %d frame(s) of %s instead",
                    segment_number, e_keep, os.path.basename(own_file))
                end_anchor = picture_only(
                    {"images": e_images[-e_keep:], "audio": None})
            end_frames = anchor_frames(end_anchor)
            if end_frames >= segment["length"]:
                raise ValueError(
                    "the pinned ending is %d frames but the replacement segment is only %d "
                    "long. Give the segment at least its original length."
                    % (end_frames, segment["length"]))

        resolved = dict(segment, resolved_seed=_resolve_seed(settings, segment, index))
        log("repairing segment %d of '%s'%s%s", segment_number, sess.name,
            " (opening pinned)" if start_anchor is not None else "",
            " (ending pinned)" if end_anchor is not None else "")

        pbar = comfy.utils.ProgressBar(1)
        images_out, audio_out, samples = render_segment(
            settings, resolved, start_anchor=start_anchor, end_anchor=end_anchor)
        push_preview(pbar, images_out[-1], 1, 1)

        target = sess.segment_path(index)
        if os.path.exists(target):
            shutil.copyfile(target,
                            os.path.join(sess.dir, "seg_%02d.replaced.mp4" % (index + 1)))
        video_io.save_clip(target, images_out, audio_out)
        length = int(images_out.shape[0])

        # The handoff clip is NOT rewritten while the ending is pinned: the file on disk
        # is what segment N+1 actually opens on, and the new ending was pinned to it.
        if not pin_ending and not is_last:
            handoff = records[index].get("handoff") or 0
            if not handoff and os.path.exists(sess.tail_path(index)):
                # 磁盘补齐的记录没有原始 handoff。用现有尾巴的长度反推，
                # 免得写出去一个 0 帧的衔接片段，把下一段的开头带歪。
                handoff = round(
                    video_io.frame_count(sess.tail_path(index)) / float(FPS), 3)
                records[index]["handoff"] = handoff
            tail = take_tail(images_out, audio_out, handoff)
            if tail is not None:
                video_io.save_clip(sess.tail_path(index), tail[0], tail[1], exact_audio=True)
                video_io.save_latent_tail(sess.tail_path(index),
                                          latent_tail(samples, handoff))
            log("pin_ending was off: segment %d's ending has moved, so segments %d..%d no "
                "longer join and have to be re-rendered.",
                segment_number, segment_number + 1, len(records))

        del images_out, audio_out, samples
        free_between_segments(False)

        records[index].update({
            "length": length,
            "seconds": round(length / float(FPS), 3),
            "seed": resolved["resolved_seed"],
            "prompt": resolved["prompt"],
            "repaired": True,
            "ending_pinned": end_anchor is not None,
        })
        sess.save(records, extra={k: v for k, v in manifest.items()
                                  if k not in ("version", "session", "segments")})

        chain = {"session": sess.name, "dir": sess.dir, "segments": records}
        summary = _summarize(chain)
        log("%s", summary.replace("\n", " | "))
        return io.NodeOutput(VideoFromFile(target), chain, summary)


# ---------------------------------------------------------------------------
# load an existing session
# ---------------------------------------------------------------------------
class H3LoadSessionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadSession",
            display_name="H3 Load Session",
            category=CATEGORY,
            description="Pick up a session already on disk -- to re-join it, feed the repair "
                        "workflow, or extend it -- without re-rendering anything.\n\n"
                        "Its chain output wires into H3 Chain to Video / H3 Repair Segment "
                        "as before, and now also into another H3 Render Segment's "
                        "chain_state, to render more shots onto the end of a session that "
                        "was already finished.",
            inputs=[
                io.String.Input("session_name", default="my_chain", multiline=False),
                io.Combo.Input("handoff_mode", options=["latent", "pixel"],
                               default="latent", advanced=True,
                               tooltip="Which form of the last segment's handoff to load, "
                                       "for extending the session with more H3 Render "
                                       "Segment nodes. Match the chain that will consume "
                                       "it. Falls back to 'pixel' if the session has no "
                                       "latent tail."),
            ],
            outputs=[H3Chain.Output(display_name="chain"),
                     io.String.Output(display_name="summary")],
        )

    @classmethod
    def fingerprint_inputs(cls, session_name, handoff_mode="latent"):
        # The folder changes underneath the graph, so never trust a cached result.
        sess = session_mod.Session(session_name)
        try:
            return "%s:%s" % (os.path.getmtime(sess.manifest_path), handoff_mode)
        except OSError:
            return "missing"

    @classmethod
    def execute(cls, session_name, handoff_mode="latent") -> io.NodeOutput:
        sess = session_mod.Session(session_name)
        manifest = sess.load()
        # 按磁盘补齐缺失的段记录，否则"最后一段"会取到 manifest 里那条旧的，
        # 续接就从第 2 段接起，把磁盘上已经渲好的 2..4 又覆盖一遍。
        manifest = sess.reconcile(manifest)
        if not manifest:
            raise FileNotFoundError("no manifest in %s" % sess.dir)
        records = manifest["segments"]
        last = records[-1]
        anchor = None
        # 有没有东西可续接，看尾巴文件在不在（补齐的记录没有 handoff 字段），
        # 这比问记录里的 handoff 更接近事实。
        if os.path.exists(sess.tail_path(last["index"])):
            anchor = _anchor_from_disk(sess, last["index"], handoff_mode)
        chain = {"session": sess.name, "dir": sess.dir, "segments": records,
                 "sess_obj": sess, "anchor": anchor, "index": last["index"],
                 "key": last["key"],
                 # Segment 1's signature, not the last one's. Extending a session has
                 # to keep measuring drift against the same shot the original chain
                 # did, or the new segments would treat an already-drifted ending as
                 # the reference and lock the drift in instead of correcting it.
                 "reference": session_mod.signature_from_record(records[0])}
        return io.NodeOutput(chain, _summarize(chain))


class MarqueeDirectorExtension(ComfyExtension):
    async def get_node_list(self):
        from .h3_segment_timeline_node import H3SegmentTimelineNode
        # Imported lazily: `director` pulls symbols back out of this module, so a
        # module-level import would be circular.
        from . import director
        from . import json_io_nodes
        from . import pack_nodes
        return [
            H3ChainSettingsNode,
            H3RenderSegmentNode,
            H3ScriptBatchRenderNode,
            H3ScriptRepairSegmentNode,
            H3ChainToVideoNode,
            H3RepairSegmentNode,
            H3LoadSessionNode,
            H3SegmentTimelineNode,
        ] + prompt_nodes.register_with_extension(self) \
            + board_nodes.register_with_extension(self) \
            + director.register_with_extension(self) \
            + pack_nodes.register_with_extension(self) \
            + json_io_nodes.register_with_extension(self)


# ---------------------------------------------------------------------------
# HTTP routes for the Director frontend panels (PACK module, reference-image
# classifier, shot timeline). Registered here because this module is imported
# as part of extension loading -- by then PromptServer.instance exists.
# A failure here must never take the nodes down with it: worst case the panels
# have no live data and everything else still renders.
# ---------------------------------------------------------------------------
try:
    from . import routes as _h3_routes

    _h3_routes.register()
except Exception as _exc:                                      # pragma: no cover
    try:
        from .common import log

        log("H3 routes not registered (%s)", _exc)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 会话目录清理：只留分镜分段视频 + 完整拼接视频，中间产物在关 ComfyUI /
# 重启电脑后自动删掉（install() 里同时挂了 atexit、启动补清、定时巡检）。
# 同样是"失败也不能拖垮节点"。
# ---------------------------------------------------------------------------
try:
    from . import cleanup as _h3_cleanup

    _h3_cleanup.install()
except Exception as _exc:                                      # pragma: no cover
    try:
        from .common import log

        log("H3 cleanup not installed (%s)", _exc)
    except Exception:
        pass
