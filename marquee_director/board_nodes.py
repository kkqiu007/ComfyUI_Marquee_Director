"""Bridge nodes for the Ref2VA Auto board's local (per-shot) video queue.

The board's per-shot queue re-injects its reference media as path loaders
(``MinimaxH3LoadImagePath`` / ``MinimaxH3LoadAudioPath`` / ``MinimaxH3LoadVideoPath``)
and needs a node in the per-shot subgraph that carries the ``ref_images`` /
``ref_audios`` sockets. The core workflow ships ``MiniMaxH3ReferenceToVideo``
for that, but the data-driven H3 continuous workflow has none of those, so
this module adds one self-contained node:

* ``H3ShotRenderer`` -- a standalone per-shot renderer that consumes
  ``ref_images`` / ``ref_audios`` (board path-loader slots) and optionally a
  board-row (``shots_json`` + ``shot_number``) so the same node can serve
  both a hand-written prompt and the board's own JSON, takes the previous
  shot's video on an optional VIDEO socket, renders through the H3
  continuous engine (prev tail re-anchored via ``MiniMaxH3AddGuide``),
  writes the clip + handoff tails into the session manifest, AND emits a
  self-filed video (``VideoFromFile``) so no external SaveVideo node is
  ever required.

Nothing in this module depends on any node from another pack: video input
comes in as an optional VIDEO socket (what the board wires into it is the
board's choice), audio/last-frame extraction is done in-process via
``video_io``, and the outgoing video is a plain file-backed VIDEO value.
"""

import json
import os

import comfy.utils

from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import io

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
    free_between_segments,
    latent_signature,
    latent_tail,
    push_preview,
    render_segment,
    take_tail,
)

CATEGORY = "ComfyUI_Marquee_Director"

H3Settings = io.Custom("H3_SETTINGS")
H3Chain = io.Custom("H3_CHAIN")


class H3ShotRendererNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ShotRenderer",
            display_name="H3 Shot Renderer (board self-contained)",
            category=CATEGORY,
            description=(
                "Self-contained per-shot renderer for the Ref2VA Auto board's "
                "local video queue. Exposes the ref_images / ref_audios "
                "autogrow sockets the board's path-loader injection needs "
                "(no MiniMaxH3ReferenceToVideo required), accepts a "
                "prev-shot video on an optional VIDEO socket, optionally "
                "reads one row out of the board's shots_json (shot_number) "
                "instead of a hand-written prompt, renders through the H3 "
                "continuous engine with the prev tail re-anchored via "
                "MiniMaxH3AddGuide, and writes the clip plus handoff tails "
                "into the session manifest. The outgoing video is a "
                "file-backed value the node filed itself -- no external "
                "SaveVideo node is involved."
            ),
            inputs=[
                H3Settings.Input("settings"),
                H3Chain.Input("chain_state", optional=True,
                              tooltip="Optional: wire a previous H3 Render Segment's "
                                     "chain_state to continue its in-memory session. "
                                     "Leave unconnected to resume the named session "
                                     "from disk."),
                io.String.Input("shots_json", display_name="分镜资产结果JSON",
                                optional=True,
                                tooltip="Optional: one row of the board's 分镜资产结果JSON. "
                                        "When connected, this row's prompt / duration / "
                                        "references are used; the free-text prompt below "
                                        "and the ref sockets are still available as "
                                        "overrides for the row's references."),
                io.Int.Input("shot_number", display_name="分镜号", default=1, min=1,
                             max=9999, advanced=True,
                             tooltip="1-based row index inside shots_json."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False,
                                optional=True,
                                tooltip="H3 prompt for this shot. Skipped when shots_json "
                                        "is connected. Reference tags are numbered per "
                                        "type in socket order: <Picture 1..>, "
                                        "<Audio 1..>. The carried prev-clip gets NO tag."),
                io.Float.Input("seconds", default=8.0, min=0.25, max=15.0, step=0.25,
                               round=False, optional=True,
                               tooltip="Length of this shot. Snapped up to H3's 17k+5 "
                                       "frame grid at 24 fps. Skipped when shots_json "
                                       "is connected."),
                io.Float.Input("handoff_seconds", default=1.625, min=0.0, max=4.0,
                               step=0.125, round=False, advanced=True,
                               tooltip="How much of this shot's ending is replayed as "
                                       "the opening of the next one."),
                io.Int.Input("seed_override", default=0,
                             min=0, max=0xffffffffffffffff, advanced=True,
                             tooltip="0 = derive this shot's seed from the chain seed "
                                     "in settings."),
                io.String.Input("session_name", default="my_chain", multiline=False,
                                tooltip="Session folder under output/h3_continuous/. "
                                        "Used when chain_state is unconnected. Must "
                                        "match H3 Chain Settings for the shots to join "
                                        "the same take."),
                io.Boolean.Input(
                    "duration_is_new_content", default=True, advanced=True,
                    tooltip="与「H3 Script Batch Render」保持同一口径：开=JSON 的 "
                            "duration 是新增时长，本段按 duration+handoff 生成；关=直接"
                            "当生成长度。仅在 shots_json 接入时生效。"),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    tooltip="Reference images -> <Picture 1..9> in socket order. "
                            "The board's path loaders plug in here; they override "
                            "shots_json's own image paths.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", optional=True),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    tooltip="Standalone reference audio -> <Audio 1..3>.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", optional=True),
                        prefix="ref_audio_", min=0, max=3)),
                io.Video.Input("prev_video", optional=True,
                               tooltip="The previous shot's finished video (e.g. the "
                                       "board's per-shot clip). Its last handoff window "
                                       "is re-anchored through the H3 sampler. Leave "
                                       "unconnected on the first shot or when the "
                                       "session already has a tail on disk."),
                io.Boolean.Input("save_to_session", default=True,
                                 tooltip="Write this shot into the session manifest so "
                                         "it joins H3 Chain To Video. The VIDEO output "
                                         "is always the self-filed clip; save_to_session "
                                         "only controls manifest membership."),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                H3Chain.Output(display_name="chain_state"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, settings, seconds, handoff_seconds, seed_override,
                session_name, prompt, duration_is_new_content=True,
                save_to_session=True, chain_state=None, shots_json=None,
                shot_number=1, prev_video=None, ref_images=None,
                ref_audios=None) -> io.NodeOutput:
        # -- resolve the segment ---------------------------------------------
        from .nodes import (
            _resolve_seed,
            _script_asset,
            _script_segment,
            _settings_for_script,
            _safe_script_handoff,
            _summarize,
        )

        ref_override = [t for _, t in ordered_autogrow(ref_images)] if ref_images else None
        ref_audios_list = [a for _, a in ordered_autogrow(ref_audios)] if ref_audios else []

        if shots_json:
            asset = _script_asset(shots_json)
            shots = [s for s in (asset.get("shots_info") or []) if isinstance(s, dict)]
            idx = int(shot_number) - 1
            if idx < 0 or idx >= len(shots):
                raise ValueError("分镜号 %d 超出 JSON 的 %d 个分镜。" % (shot_number, len(shots)))
            settings = _settings_for_script(settings, asset)
            seg_prompt, seg_seconds, seg_images = _script_segment(
                asset, shots[idx], ref_images_override=ref_override or None)
            handoff = _safe_script_handoff(seg_seconds, handoff_seconds, idx == len(shots) - 1)
            if duration_is_new_content and handoff:
                seg_seconds = min(seg_seconds + float(handoff_seconds), 15.0)
        else:
            seg_prompt = str(prompt or "").strip()
            if not seg_prompt:
                raise ValueError("H3ShotRenderer 需要 prompt 或 shots_json，两者都为空。")
            seg_seconds = float(seconds)
            handoff = guide_length(seconds_to_frames(handoff_seconds))
            seg_images = ref_override or []
            if handoff >= generation_length(seconds_to_frames(seg_seconds)):
                raise ValueError(
                    "handoff_seconds (%.2fs -> %d frames) must be shorter than the "
                    "shot itself (%.2fs -> %d frames)."
                    % (handoff_seconds, handoff, seg_seconds,
                       generation_length(seconds_to_frames(seg_seconds))))

        length = generation_length(seconds_to_frames(seg_seconds))

        # -- resolve session & index ------------------------------------------
        if chain_state is not None:
            sess = chain_state["sess_obj"]
            index = chain_state["index"] + 1
            previous_key = chain_state["key"]
            records = list(chain_state["segments"])
            start_anchor = chain_state.get("anchor")
            inherited_reference = chain_state.get("reference")
        else:
            sess = session_mod.Session(session_name)
            # 按磁盘补齐缺失的段，否则 index 会停在 manifest 记录数上，续接时
            # 把磁盘上已经渲好的段又覆盖一遍。
            manifest = sess.reconcile(sess.load())
            if manifest:
                records = list(manifest["segments"])
                index = len(records)
                previous_key = None
                last = records[-1] if records else None
                # 有没有东西可续接看尾巴文件在不在（补齐的记录没有 handoff 字段）
                if last and os.path.exists(sess.tail_path(last["index"])):
                    start_anchor = _anchor_from_disk(sess, last["index"],
                                                      settings.get("handoff_mode", "latent"))
                else:
                    start_anchor = None
            else:
                records = []
                index = 0
                previous_key = None
                start_anchor = None
            inherited_reference = None

        # -- prev-clip handoff (board-injected video; engine-internal logic)
        if start_anchor is None and prev_video is not None:
            start_anchor = _anchor_from_video_value(prev_video, settings.get("handoff_mode", "latent"))

        # -- build the segment dict -------------------------------------------
        segment = {
            "prompt": seg_prompt, "seconds": seg_seconds, "length": length, "handoff": handoff,
            "seed": int(seed_override),
            "images": list(seg_images),
            "videos": [],
            "video_audios": [],
            "audios": list(ref_audios_list),
        }

        # -- derive seed --------------------------------------------------------
        segment["resolved_seed"] = _resolve_seed(settings, segment, index)
        key = session_mod.segment_key(settings, segment, handoff, previous_key)

        # -- render -------------------------------------------------------------
        log("H3ShotRenderer shot %d: %d frames, handoff %d, anchor=%s, seed=%d, "
            "refs=%d, audio=%d",
            index + 1, length, handoff, "yes" if start_anchor is not None else "no",
            segment["resolved_seed"], len(segment["images"]), len(segment["audios"]))
        images_out, audio_out, samples = render_segment(
            settings, segment, start_anchor=start_anchor, end_anchor=None)
        rendered_length = int(images_out.shape[0])

        # -- save clip + tails ----------------------------------------------------
        video_io.save_clip(sess.segment_path(index), images_out, audio_out)
        pixel_tail = take_tail(images_out, audio_out, handoff)
        lat_tail = latent_tail(samples, handoff)
        signature = latent_signature(samples)
        if pixel_tail is not None:
            video_io.save_clip(sess.tail_path(index), pixel_tail[0], pixel_tail[1],
                               exact_audio=True)
            video_io.save_latent_tail(sess.tail_path(index), lat_tail)

        new_anchor = None
        if handoff:
            new_anchor = (lat_tail if settings.get("handoff_mode", "latent") == "latent"
                          else {"images": pixel_tail[0], "audio": pixel_tail[1]})

        preview_frame = images_out[-1].clone()
        del images_out, audio_out, samples, pixel_tail, lat_tail
        free_between_segments(False)

        # -- update manifest -------------------------------------------------------
        record = {
            "index": index, "key": key, "length": rendered_length, "handoff": handoff,
            "seconds": round(rendered_length / float(FPS), 3),
            "seed": segment["resolved_seed"], "prompt": seg_prompt,
            "file": os.path.basename(sess.segment_path(index)),
            "signature": session_mod.signature_to_record(signature),
        }
        if save_to_session:
            records = records[:index] + [record]
            sess.save(records, extra={"width": settings["width"],
                                      "height": settings["height"], "fps": FPS})

        chain = {
            "session": sess.name, "dir": sess.dir, "segments": records,
            "sess_obj": sess, "anchor": new_anchor, "index": index, "key": key,
            "reference": (inherited_reference
                           if chain_state is not None and inherited_reference is not None
                           else signature),
        }

        pbar = comfy.utils.ProgressBar(1)
        push_preview(pbar, preview_frame, 1, 1)
        del preview_frame

        summary = _summarize(chain)
        log("H3ShotRenderer %s", summary.replace("\n", " | "))

        video_out = VideoFromFile(sess.segment_path(index))
        return io.NodeOutput(video_out, chain, summary)


def _anchor_from_video_value(video, handoff_mode):
    """Turn a board-injected prev-shot VIDEO into a render start anchor.

    No external nodes: last-frame + tail audio are extracted in-process from
    the video's file path (VIDEO is a file-backed value in this pack), or
    from a ``dict`` of pixel frames (e.g. a GetVideoComponents IMAGE output
    when the board prefers frame arrays).
    """
    # dict form (frame array) -- use it directly
    if isinstance(video, dict):
        images = video.get("images")
        if images is None:
            return None
        return {"images": images, "audio": video.get("audio")}

    # file-backed VIDEO (VideoFromFile): source is str path or BytesIO
    source = getattr(video, "get_stream_source", lambda: None)()
    path = source if isinstance(source, str) else None
    if path is None:
        path = video if isinstance(video, str) else None
    if not path or not os.path.exists(path):
        log("H3ShotRenderer prev video not resolvable (%r) -- rendering cold", video)
        return None

    last = video_io.last_frame(path)
    tail_audio = video_io.read_audio(os.path.splitext(path)[0] + ".wav")
    return {"images": last, "audio": tail_audio}


def _anchor_from_disk(sess, index, handoff_mode):
    """Read the tail clip for segment ``index`` off disk (same logic as
    H3RenderSegmentNode's resume path)."""
    tail = sess.tail_path(index)
    if handoff_mode == "latent":
        anchor = video_io.load_latent_tail(tail)
        if anchor is not None:
            return anchor
        log("no latent tail for %s -- falling back to pixel clip",
            os.path.basename(tail))
    images, audio = video_io.load_clip(tail, prefer_wav=True)
    return {"images": images, "audio": audio}


def register_with_extension(ext):
    """Return the node classes for MarqueeDirectorExtension.get_node_list()."""
    return [H3ShotRendererNode]
