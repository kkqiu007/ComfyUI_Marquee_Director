# -*- coding: utf-8 -*-
"""H3Director — one node for the whole take: 剧本 → 渲染 → 时间线 → 成片。

This node collapses the script-board / batch-render / timeline-preview /
save-json lane (Ref2VA Auto board → H3 Script Batch Render → H3 Segment
Timeline → Save JSON, ~4 nodes and ~6 sockets that all have to be wired
correctly) into a single node, and it folds the smart shot splitter in as a
second *source mode* instead of a second node.

Why the two are one node now
----------------------------
``H3ShotSplit`` produced a ``shots_info`` JSON; ``H3Director`` consumed one.
They were two nodes with one purpose (get a shot list into the render loop),
so the splitter is now the ``剧本来源 = 源视频自动切分`` branch of this node:
it writes the same JSON into the same session and the rest of the pipeline is
byte-for-byte identical. One node, one place to look, and the emitted
``shots_json`` is always *the JSON that was actually rendered* — so you can
read it, edit it, and paste it back into 「分镜JSON」 mode.

Two source modes
----------------
* ``分镜JSON`` — consume an existing Ref2VA-Auto-compatible payload.
* ``源视频自动切分`` — probe a source video, detect hard cuts with
  PySceneDetect's AdaptiveDetector (uniform split when it is missing), and
  render each detected shot. ``dry_run`` then doubles as a split preview: the
  ``时间线`` output is a contact sheet of the source video's shot head frames,
  so you can check the cuts before spending a single GPU second.

Borrowed from ComfyUI_MiniMaxH3_Director
----------------------------------------
* ``vram_cleanup`` — evict dead ``LoadedModel`` slots between segments.
* ``shot_detect`` — two-pass cut merge with a hard ``MIN_SEG_FRAMES`` floor,
  and the ``low/medium/high`` → ``adaptive_threshold`` mapping.
* ``task_modes`` — per-segment task inference (t2v / r2v / v2v / rv2v).
* ``progress`` — phase-level progress reporting (prepare / sample / decode …)
  pushed to the node through ``cls.hidden.unique_id``.
* ``segment_mp4_export`` — timestamped per-segment MP4 export folder.
* ``image_prep`` — reference images are downscaled (never upscaled) to a
  long-edge cap and snapped to the 32px canvas grid, because an odd latent
  width crashes H3 in ``patchify_video``.
* ``ref_images`` — up to 9 slots, and a per-shot ``ref_images`` selector so a
  shot can use a subset of the pool instead of all of it.

It does **not** call H3 Script Batch Render / H3 Chain To Video /
H3 Segment Timeline as subgraphs: those nodes are thin wrappers around
``render_segment`` / ``video_io.join`` / on-disk glob, and re-implementing
them inline drops the node-instantiation cost while reusing the same helpers.
The session manifest on disk stays identical, so ``H3ScriptRepairSegment`` /
``H3ChainToVideo`` / ``H3LoadSession`` / ``H3ShotRenderer`` keep working.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.utils
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import io

from . import routes as routes_mod
from . import session as session_mod
from . import ref_images as ref_images_mod
from . import video_io
from .common import (
    FPS,
    evict_dead_loaded_models,
    generation_length,
    guide_length,
    guide_length_up,
    log,
    seconds_to_frames,
)
from .engine import free_between_segments
from .nodes import (
    H3RenderSegmentNode,
    H3RepairSegmentNode,
    _images_autogrow,
    _load_script_image,
    _resolve_seed,
    _safe_script_handoff,
    _script_asset,
    _script_segment,
    _settings_for_script,
)

CATEGORY = "ComfyUI_Marquee_Director"

H3Settings = io.Custom("H3_SETTINGS")
H3Chain = io.Custom("H3_CHAIN")

# Shortest shot H3 can render: 5 frames is the bottom of the 17k+5 grid.
MIN_SHOT_FRAMES = 5
# Hard floor for a detected shot (borrowed from MiniMaxH3_Director/shot_detect).
MIN_SEG_FRAMES = 4

# Reference images: MiniMaxH3ReferenceToVideo takes at most 9 (<Picture 1..9>).
# 单一来源放在 ref_images 模块（对齐 ComfyUI_MiniMaxH3_Director 的约定）。
MAX_REFERENCE_IMAGES = ref_images_mod.MAX_REFERENCE_IMAGES

# 闪电渲染：步数下拉里的"跟随上游"哨兵值。选它就不碰上游 BasicScheduler 给的
# sigmas —— 老工作流一个字节都不改。
FOLLOW_UPSTREAM = "跟随上游 sigmas"
NO_LORA = "（不加载）"
SCHEDULER_CHOICES = [
    FOLLOW_UPSTREAM, "simple", "normal", "karras", "exponential",
    "sgm_uniform", "ddim_uniform", "beta", "laplace",
]


def _apply_lightning(settings, lightning, lora_name, lora_strength,
                     render_steps, scheduler, notes):
    """闪电渲染 + 步数覆盖。

    两点说明为什么放在 Director 里而不是再加一个节点：

    * **LoRA 必须挂在 Director 用的那个 model 上**。上游 LoraLoaderModelOnly
      已经把 turbo LoRA 打进去了，这里是在它之上再叠一层；叠完必须替换回
      ``settings["model"]``，否则缓存键看不到 patch 列表，改了强度却不重渲。
    * **步数 = sigmas 的长度**。改步数不是给采样器传个参数，而是重算一遍
      sigmas；所以"跟随上游"必须是默认，一改就会让所有段缓存失效
      （这是对的：4 步和 20 步渲出来的不是同一段片子）。

    返回新的 settings（不改原 dict）。
    """
    settings = dict(settings or {})
    steps = int(render_steps or 0)
    sched = str(scheduler or FOLLOW_UPSTREAM).strip()
    name = str(lora_name or NO_LORA).strip()

    if steps > 0 and sched != FOLLOW_UPSTREAM:
        try:
            import comfy.samplers

            model = settings.get("model")
            model_sampling = model.get_model_object("model_sampling")
            sigmas = comfy.samplers.calculate_sigmas(model_sampling, sched, steps)
            sigmas = torch.cat([sigmas.cpu(), torch.zeros(1)])
            settings["sigmas"] = sigmas
            notes.append("步数覆盖：%s · %d 步（%d 个 sigma）"
                         % (sched, steps, sigmas.shape[0]))
            _note_cache_bust(notes, "渲染步数改动")
        except Exception as exc:
            notes.append("⚠ 步数覆盖失败（沿用上游 sigmas）：%s" % exc)
    elif steps > 0:
        notes.append("⚠ 填了步数 %d 但调度器还是「%s」，没生效 —— 两个都要设。"
                     % (steps, FOLLOW_UPSTREAM))

    if lightning and name and name != NO_LORA:
        try:
            import comfy.sd
            import folder_paths

            path = folder_paths.get_full_path("loras", name)
            if not path:
                notes.append("⚠ 找不到闪电 LoRA：%s" % name)
                return settings
            lora = comfy.utils.load_torch_file(path, safe_load=True)
            model = settings.get("model")
            new_model, _ = comfy.sd.load_lora_for_models(
                model, None, lora, float(lora_strength or 1.0), 0.0)
            if new_model is not None:
                settings["model"] = new_model
                notes.append("闪电 LoRA：%s × %.2f（已叠加在上游 LoRA 之上）"
                             % (name, float(lora_strength or 1.0)))
                _note_cache_bust(notes, "闪电 LoRA 改动")
        except Exception as exc:
            notes.append("⚠ 闪电 LoRA 加载失败（已忽略）：%s" % exc)
    return settings


def _note_cache_bust(notes, what):
    notes.append("  ↳ %s 会改变缓存键 → 从这一段起全部重渲（包括 resume 打开时）"
                 % what)


def _load_pack_info(raw, asset, settings):
    """接进来的 pack_info → dict；顺手把画幅和画布对账。"""
    info = None
    text = str(raw or "").strip()
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                info = parsed
        except (TypeError, ValueError):
            info = None
    if info is None:
        info = {"header": {}, "segments": [], "ref_slots": []}
    info.setdefault("header", {})
    info.setdefault("segments", [])
    # 没接 PACK 时至少把段数补上，面板不至于全空
    if not info["header"].get("segments"):
        info["header"]["segments"] = str(len(asset.get("shots_info") or []))
    info["canvas_note"] = _canvas_note(info, settings)
    return info


_ASPECT_RE = re.compile(r"(\d{1,2})\s*[:：xX*]\s*(\d{1,2})")


def _canvas_note(info, settings):
    """PACK 声明的画幅 vs H3 Chain Settings 的画布，不一致就说一句。

    这是 PACK 模块最有用的一个用处：画幅写 16:9、画布却是 416×736（9:16），
    渲出来是一整条竖屏片子，等到导出才发现就白烧一张卡。
    """
    aspect = str((info.get("header") or {}).get("aspect") or "")
    match = _ASPECT_RE.search(aspect)
    if not match:
        return ""
    try:
        want = int(match.group(1)) / float(int(match.group(2)))
    except (ValueError, ZeroDivisionError):
        return ""
    width = int(settings.get("width") or 0)
    height = int(settings.get("height") or 0)
    if width <= 0 or height <= 0:
        return ""
    got = width / float(height)
    if abs(got - want) < 0.02:
        return ""
    return ("PACK 画幅 %s（%.3f）与画布 %d×%d（%.3f）不一致 —— "
            "成片会是后者，要改请改「H3 Chain Settings」的 width/height。"
            % (aspect, want, width, height, got))


def _pool_labels(slots, ref_classify):
    """参考图池的显示名 → **稀疏字典** ``{槽号: 标签}``。

    旧实现返回 list（下标对应**压实后**的池），槽位一跳空标签就整体错位 ——
    接了 0,1,3 时，第 3 张图的标签会贴到 ref_image_3 上。稀疏槽位模型下必须
    用槽号当键。实现已下放到 ``ref_images.slot_labels``。
    """
    return ref_images_mod.slot_labels(slots, ref_classify)


def _check_ref_slots(info, shots, slots, warnings):
    """剧本要的 <Picture N> vs 实际接上的槽位 —— 双向对账，对不上就报红。

    这是"参考图没生效、人物变样"的第一现场，两个方向都要查：

    * ``missing`` —— 剧本写了 ``<Picture 5>``，槽 5 却空着。压实取图的旧版
      会悄悄拿后面的图顶上（人物换脸），稀疏模型下是明确告警。
    * ``unused``  —— 接了图，但剧本里没有任何 ``<Picture N>`` 指向它。H3
      不会把它画进画面，"换图没反应"最难查的一种就是它。
    """
    # 稀疏槽位模型：剧本要哪些 <Picture N>、实际接了哪些槽，两边各自独立对账。
    # 旧版只有一个 pool_len（压实后的张数），既发现不了"引了空槽"，
    # 也只能在"接得比需要的多"时笼统提示，说不出到底是哪几个槽没被引用。
    rec = ref_images_mod.reconcile(slots, shots)
    needed = rec["needed"]
    connected = rec["connected"]
    missing = rec["missing"]
    unused = rec["unused"]

    if needed:
        info = dict(info)
        info["ref_slots"] = needed
        info["ref_slot_count"] = len(needed)
    if not needed:
        info["ref_note"] = ""
        return info

    if not connected:
        info["ref_missing"] = list(needed)
        info["ref_note"] = (
            "提示词要 %d 张参考图（%s），但「参考图覆盖」一张都没接 —— "
            "H3 会自己「补」出角色，成片人物和参考图对不上。"
            % (len(needed), "、".join("Picture %d" % n for n in needed)))
        warnings.append("⚠ " + info["ref_note"])
        return info

    notes = []
    if missing:
        # ★ 旧版发现不了这种情况：剧本写了 <Picture 5>，而槽 5 空着。
        # 压实取图时它会悄悄拿后面的图顶上（人物换脸），稀疏模型下就是明确告警。
        info["ref_missing"] = missing
        notes.append(
            "剧本引用了 %s，但槽位 %s 没接图 —— H3 会自己补出角色，"
            "成片人物和参考图对不上。请在参考图面板给这些槽位选图，"
            "或把提示词里多余的 <Picture N> 删掉。"
            % ("、".join("<Picture %d>" % n for n in missing),
               "、".join(str(n) for n in missing)))
    if unused:
        # H3 Ref2VA 是靠提示词里的 <Picture N> 把参考图绑到角色/场景上的，
        # 槽位 N 只有在剧本里出现过 <Picture N> 时才有意义。多接的图照样会
        # 进条件编码，但**没有任何指令让模型去用它们** —— 表现就是
        # "图接上了、面板也显示了，成片里就是看不到这张脸"。
        info["ref_unused"] = unused
        notes.append(
            "接了 %d 张参考图（槽 %s），但剧本只引用了 %s —— 槽 %s 没有任何 "
            "<Picture N> 指向它，H3 不会把它画进画面。想让它生效，得在 PACK 里"
            "补 <Picture N> 的定义并在分镜里引用；只是不想看到这条提示的话，"
            "把多余的槽位断开即可。"
            % (len(connected), "、".join(str(n) for n in connected),
               "、".join("Picture %d" % n for n in needed),
               "、".join(str(n) for n in unused)))
    info["ref_note"] = "　".join(notes)
    if notes:
        warnings.append("⚠ " + info["ref_note"])
    return info


def _pack_info_block(info):
    """运行报告里的 PACK 段：五项概要 + 画布对账。"""
    if not isinstance(info, dict):
        return []
    header = info.get("header") or {}
    lines = [
        "",
        "── PACK ──────────────────────────────────────────",
        "项目   %s" % (header.get("project") or "（未填写）"),
        "模式   %s" % (header.get("mode") or "（未填写）"),
        "总时长 %s" % (header.get("total_duration") or "（未填写）"),
        "段数   %s" % (header.get("segments") or "（未填写）"),
        "画幅   %s" % (header.get("aspect") or "（未填写）"),
    ]
    if header.get("music"):
        lines.append("配乐   %s" % header["music"])
    if header.get("version") or header.get("date"):
        # 括号别省：`"..." % (a, b).strip()` 里的 .strip() 会作用在元组
        # (a, b) 上（% 优先级低于属性访问），直接 AttributeError。
        lines.append(("版本   %s %s" % (header.get("version") or "",
                                        header.get("date") or "")).strip())
    slots = info.get("ref_slots") or []
    if slots:
        lines.append("参考图 需要 %d 张：%s"
                     % (len(slots), "、".join("Picture %d" % n for n in slots)))
    for key in ("canvas_note", "ref_note"):
        if info.get(key):
            lines.append("⚠ %s" % info[key])
    return lines
# H3: VAE ÷16 spatially, then 2×2 patchify → canvas must be a multiple of 32.
# An odd latent width (e.g. 496px → 31) crashes patchify_video while sampling.
CANVAS_STRIDE = 32

REF_SIZE_MATCH = "match"
# match = follow the output canvas; the rest are long-edge presets (px).
REF_SIZE_OPTIONS = [REF_SIZE_MATCH, "512", "768", "1024", "1536"]

SOURCE_JSON = "分镜JSON"
SOURCE_VIDEO = "源视频自动切分"
SOURCE_OPTIONS = [SOURCE_JSON, SOURCE_VIDEO]


def _source_mode(value) -> str:
    """Normalise the 剧本来源 combo into ``"json"`` / ``"video"``."""
    text = str(value or "").strip()
    if "视频" in text or "video" in text.lower():
        return "video"
    return "json"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    """Trim ```json ... ``` or leading prose so JSON parsers see a clean object."""
    raw = (text or "").strip()
    if not raw:
        return raw
    start = raw.find("{")
    if start < 0:
        return raw
    return raw[start:]


def _clean_shot_json(shots_json):
    """Validate / canonicalise a Ref2VA-Auto-compatible JSON in one place."""
    text = _strip_fences(shots_json)
    if not text:
        return {"duration": 0, "shots_info": []}
    try:
        return _script_asset(text)
    except ValueError:
        pass
    import json
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("分镜资产结果JSON无法解析：%s" % exc) from exc
    if not isinstance(data, dict):
        raise ValueError("分镜资产结果JSON必须是 JSON 对象。")
    return data


def _session_name_from(data, fallback):
    """Widget wins when it is non-empty; otherwise fall back to the JSON."""
    name = str(fallback or "").strip()
    if name:
        return name
    return str(data.get("session_name") or data.get("session") or "my_chain")


def _json_dumps(data) -> str:
    import json
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:                                            # pragma: no cover
        return "{}"


# ---------------------------------------------------------------------------
# Progress reporting (borrowed from MiniMaxH3_Director/director/progress.py)
# ---------------------------------------------------------------------------
_PHASE_LABELS = {
    "plan": "解析剧本 / 加载视频",
    "prepare": "准备片段",
    "context_encode": "H3 条件编码",
    "sample": "采样",
    "decode": "AV 解码",
    "join": "拼接导出",
    "finish": "全部完成",
}


def _node_id(cls):
    """The node's unique id, available on the class clone during execution.

    ``_ComfyNodeBaseInternal.hidden`` is a ``HiddenHolder`` filled in by
    ``execution.py`` right before ``execute`` runs — but only for classes that
    declared ``hidden=[io.Hidden.unique_id]`` in their schema.
    """
    hidden = getattr(cls, "hidden", None)
    return getattr(hidden, "unique_id", None) if hidden is not None else None


def _phase_text(cls, text) -> None:
    """Push a human-readable status line onto the node in the UI."""
    node_id = _node_id(cls)
    if not node_id or not text:
        return
    try:
        from server import PromptServer

        srv = PromptServer.instance
        if srv is not None:
            srv.send_progress_text(str(text), node_id)
    except Exception:                                            # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Task-mode inference (borrowed from MiniMaxH3_Director/director/task_modes.py)
# ---------------------------------------------------------------------------
def _infer_task(images, videos=None) -> str:
    """Which H3 task a segment will actually run as."""
    has_video = bool(videos)
    has_image = bool(images)
    if has_video and has_image:
        return "rv2v"
    if has_video:
        return "v2v"
    if has_image:
        return "r2v"
    return "t2v"


# ---------------------------------------------------------------------------
# VRAM hygiene (borrowed from MiniMaxH3_Director/director/vram_cleanup.py)
# ---------------------------------------------------------------------------
# 实现已下沉到 common：engine 的段间清理也要用它，而 engine 不能 import
# director（会成环）。这里保留旧名，调用点不用改。
_evict_dead_loaded_models = evict_dead_loaded_models


def cleanup_segment_vram(unload_models: bool = False) -> None:
    """Release what a finished segment still holds."""
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        _evict_dead_loaded_models()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
        _evict_dead_loaded_models()
        gc.collect()
        mm.soft_empty_cache()
    except Exception as exc:                                     # pragma: no cover
        log("H3Director: VRAM cleanup skipped (%s)", exc)


# ---------------------------------------------------------------------------
# Frame-grid planning
# ---------------------------------------------------------------------------
def _shot_windows(shot, index, total, default):
    """本段的 ``(replay_in, tail_out)``（秒）。

    PACK 解析出来的 shot 自带 ``replay_in`` / ``tail_out`` 两个字段（见
    ``prompt_pack.segment_windows``），直接照用 —— 那份算法站在整条链上，
    知道谁是首段谁是末段，是唯一权威。手写 JSON 没有这两个字段时退化成
    「首段无回放、末段无锚点」的老规则。

    ``hard_cut`` 两个窗口都归零：硬切段既不回放上一段，也不给下一段留锚点。
    """
    if isinstance(shot, dict) and shot.get("hard_cut"):
        return 0.0, 0.0
    default = max(0.0, float(default or 0.0))

    def _read(key):
        if not isinstance(shot, dict):
            return None
        raw = shot.get(key)
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None

    replay = _read("replay_in")
    tail = _read("tail_out")
    # 兼容只有 handoff_seconds 的老 JSON：它标的是段标题里的 +1.6（= 回放），
    # 首段写 0 就说明没有段首回放，别拿全局缺省值去补。
    if replay is None and tail is None:
        legacy = _read("handoff_seconds")
        if legacy is not None and legacy > 0:
            replay = legacy
    if replay is None:
        replay = 0.0 if index == 0 else default
    if tail is None:
        tail = 0.0 if index >= total - 1 else default
    return max(0.0, replay), max(0.0, tail)


def _shot_plan(seconds, replay_in=0.0, tail_out=0.0,
               duration_is_new_content=True):
    """Snap one shot onto H3's 17k+5 grid and report what that costs.

    Mirrors the arithmetic ``H3 Script Batch Render`` performs so the numbers
    in the report are the numbers the render will actually use.

    两个方向必须分开算（minimax-h3-shot-segment_en 规范 0.1 / 1.4）：

    * ``replay_in`` — 段**首**回放上一段结尾。它**占生成时长**，因为这段画面
      是下一段要重新生成出来的：``gen = new + replay_in``。拼接时这段会被跳过
      （与上一段尾部重叠）。
    * ``tail_out``  — 段**尾**切给下一段当锚点。它**不占生成时长**：锚点就是
      本段新增内容最后那 1.6 秒（规范 1.4 要求这段无台词），渲完从结果里切
      出来即可，不需要多生成。

    以前两者合成一个 ``handoff``，结果首段被多加 1.6 秒生成（多出来的内容
    剧本里没有，模型只能自由发挥 → 成片"乱说话"），末段又因为 ``is_last``
    把回放归零、少生成 1.6 秒（提示词时间码写到 8.6s 却只给 7s，动作被挤压）。
    """
    length = generation_length(seconds_to_frames(seconds))
    # 回放要的是「至少 1.6 秒」，所以向上对齐到 17k+5：1.6s = 38 帧向下取是
    # 22 帧（0.92s），等于把规范要求的回放窗口砍掉一半。向上取到 39 帧
    # （1.625s）才能完整保留，而且这个值传给 H3RenderSegment 后再被
    # guide_length 截断也不会掉帧。
    replay = _safe_script_handoff(seconds, replay_in, False)
    replay_frames = guide_length_up(seconds_to_frames(replay)) if replay else 0
    if replay_frames:
        replay = replay_frames / float(FPS)

    tail = _safe_script_handoff(seconds, tail_out, False) if tail_out else 0.0
    tail_frames = guide_length_up(seconds_to_frames(tail)) if tail else 0
    if tail_frames:
        tail = tail_frames / float(FPS)
    # 锚点不能吃掉整段：留不出新增内容就干脆不留
    if tail_frames >= length:
        tail, tail_frames = 0.0, 0

    gen_seconds = float(seconds)
    if duration_is_new_content and replay:
        gen_seconds = min(gen_seconds + float(replay), 15.0)
    requested = seconds_to_frames(gen_seconds)
    frames = generation_length(requested)
    new_frames = max(0, frames - replay_frames)
    return {
        "replay_in": replay,
        "replay_frames": replay_frames,
        "tail_out": tail,
        "tail_frames": tail_frames,
        # 旧字段名保持可用：报告与缓存判定都还在读它
        "handoff": replay,
        "handoff_frames": replay_frames,
        "handoff_raised": bool(replay_frames) and abs(
            replay - float(replay_in or 0.0)) > 0.001,
        "gen_seconds": gen_seconds,
        "requested_frames": requested,
        "frames": frames,
        "new_frames": new_frames,
        "new_seconds": new_frames / float(FPS),
    }


def _cache_state(sess, manifest, index, settings, prompt, seconds, images,
                 handoff, previous_key):
    """Would this segment be a cache hit? None when it cannot be determined."""
    try:
        segment = {
            "prompt": prompt,
            "seconds": seconds,
            "length": generation_length(seconds_to_frames(seconds)),
            "handoff": guide_length(seconds_to_frames(handoff)) if handoff else 0,
            "seed": 0,
            "images": list(images or []),
            "videos": [],
            "video_audios": [],
            "audios": [],
        }
        segment["resolved_seed"] = _resolve_seed(settings, segment, index)
        key = session_mod.segment_key(settings, segment, segment["handoff"], previous_key)
        hit = bool(sess.cached(manifest, index, key, needs_tail=bool(segment["handoff"])))
        return hit, key
    except Exception as exc:                                     # pragma: no cover
        log("H3Director: cache prediction failed for segment %d (%s)", index + 1, exc)
        return None, previous_key


# ---------------------------------------------------------------------------
# Reference images (borrowed from MiniMaxH3_Director/lib/image_prep.py)
# ---------------------------------------------------------------------------
def _ref_edge_limit(mode, settings):
    """Long-edge cap for reference images, or None when there is nothing to fit.

    ``match`` follows the render canvas (a reference bigger than the output buys
    nothing but VRAM); the presets are plain pixel caps. ``None`` means
    "downscale is off" — the 32px snap below still runs.
    """
    text = str(mode or REF_SIZE_MATCH).strip().lower()
    if text == REF_SIZE_MATCH:
        try:
            return (max(int(settings.get("width") or 0),
                        int(settings.get("height") or 0)) or None)
        except (TypeError, ValueError):
            return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def _fit_ref_image(image, max_edge):
    """Downscale only, then snap H/W to 32. Returns ``(tensor, changed, w×h, w×h)``."""
    # An empty batch is "no reference image at all": leave it alone rather than
    # trying to resize zero frames (and reporting it as a change).
    if image is None or getattr(image, "ndim", 0) != 4 or image.shape[0] == 0:
        return image, False, None, None
    rgb = image[..., :3]
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    scale = 1.0
    if max_edge and max(height, width) > max_edge:
        scale = float(max_edge) / float(max(height, width))
    new_h = max(CANVAS_STRIDE,
                int(round(height * scale / CANVAS_STRIDE) * CANVAS_STRIDE))
    new_w = max(CANVAS_STRIDE,
                int(round(width * scale / CANVAS_STRIDE) * CANVAS_STRIDE))
    if (new_h, new_w) == (height, width):
        return image, False, (width, height), (width, height)
    resized = comfy.utils.common_upscale(
        rgb.movedim(-1, 1), new_w, new_h, "area", "disabled").movedim(1, -1)
    if image.shape[-1] > 3:                       # keep an alpha channel, if any
        alpha = image[..., 3:]
        if alpha.shape[1:3] != resized.shape[1:3]:
            alpha = comfy.utils.common_upscale(
                alpha.movedim(-1, 1), new_w, new_h, "area", "disabled").movedim(1, -1)
        resized = torch.cat([resized, alpha], dim=-1)
    return resized, True, (width, height), (new_w, new_h)


def _prepare_ref_images(images, mode, settings, labels=None):
    """Fit every reference image and describe what happened (for the report).

    ``images`` 可以是**稀疏槽位字典** ``{槽号: tensor}``（节点接进来的参考图，
    见 ``ref_images.collect_slots``），也可以是普通 list（分镜脚本自带的图）。
    进什么出什么：dict 进 dict 出、list 进 list 出，这样调用方不用改。

    ``labels`` 同理：dict 用 ``labels[槽号]``，list 用 ``labels[下标]``。
    """
    max_edge = _ref_edge_limit(mode, settings)
    as_dict = isinstance(images, dict)
    items = sorted((images or {}).items()) if as_dict else list(enumerate(images or []))
    out = {} if as_dict else []
    details = []
    for key, image in items:
        label = None
        if isinstance(labels, dict):
            label = labels.get(key)
        elif isinstance(labels, (list, tuple)) and 0 <= key < len(labels):
            label = labels[key]
        fitted, changed, orig, new = _fit_ref_image(image, max_edge)
        if as_dict:
            out[key] = fitted
        else:
            out.append(fitted)
        details.append({
            "label": label or ref_images_mod.slot_label(key if as_dict else key + 1),
            "orig": ("%dx%d" % orig) if orig else "-",
            "new": ("%dx%d" % new) if new else "-",
            "changed": changed,
        })
    return out, details


def _shot_ref_images(shot, pool, warnings=None):
    """Pick this shot's reference images out of the connected **slot map**.

    ``shot["ref_images"]`` accepts 1-based slot numbers (they line up with the
    ``<Picture N>`` tags in the prompt) or plain image paths. Omitting the key
    keeps the global set; an explicit ``[]`` means "this shot uses none".

    **「一个都没接」不再中断整条流水线**：PACK 解析出来的分镜几乎都会带
    ``<Picture N>``（那是提示词里的角色/场景定义），而「参考图覆盖」槽位默认
    是空的（预置的 LoadImage 全部 mute）。这时候直接报错等于打开工作流点一下
    运行就崩，所以改成记一条警告、该镜按无参考图渲染；接了图但编号越界才是
    真的配置错误，那种仍然报错。
    """
    # ---- 稀疏槽位字典（新版，见 ref_images.collect_slots）----------------
    # 键就是 <Picture N> 的 N：槽 5 没接图就是没图，绝不会拿槽 7 的图来顶。
    # 旧版传的是压实后的 list、用 pool[index-1] 取图，槽位跳空就整体错位。
    if isinstance(pool, dict):
        return ref_images_mod.resolve_shot_refs(
            shot, pool, warnings=warnings,
            loader=lambda path: _load_script_image(path, "分镜参考图"))
    # ---- 旧的 list 路径：保留给任何还按老约定调用的地方 ------------------
    if not isinstance(shot, dict) or "ref_images" not in shot:
        return pool
    raw = shot.get("ref_images")
    if raw is None:
        return pool
    if isinstance(raw, (str, int, float)) and not isinstance(raw, bool):
        raw = [raw]
    if not isinstance(raw, list):
        return pool
    missing = []
    picks = []
    for item in raw:
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            index = int(item)
            if not pool:
                missing.append(index)
                continue
            if index < 1 or index > len(pool):
                raise ValueError(
                    "分镜 %s 的 ref_images 越界：要第 %d 张，实际只接了 %d 张。"
                    % (shot.get("id") or "?", index, len(pool)))
            picks.append(pool[index - 1])
            continue
        path = str(item).strip()
        if not path:
            continue
        tensor = _load_script_image(path, "分镜参考图")
        if tensor is not None:
            picks.append(tensor)
    if missing and warnings is not None:
        warnings.append(
            "分镜 %s 声明了参考图 %s，但「参考图覆盖」一个都没接 → 本镜按无参考图"
            "渲染（在 ⑫ 组选图并 Ctrl+M 取消 mute 后生效）"
            % (shot.get("id") or "?",
               "、".join("第 %d 张" % n for n in missing)))
    return picks


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _display_width(text):
    """Terminal columns taken by ``text``: CJK / full-width glyphs count 2."""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in str(text))


def _cell(value, width, align="right"):
    """Pad ``value`` to ``width`` display columns."""
    text = str(value)
    pad = max(0, width - _display_width(text))
    return " " * pad + text if align == "right" else text + " " * pad


# (#, 新增s, 请求帧, 对齐帧, 回放帧, 累计s, 耗时s, 模式, 参考, 状态)
_REPORT_COLS = ((4, "left"), (8, "right"), (8, "right"), (8, "right"),
                (8, "right"), (9, "right"), (9, "right"), (6, "right"),
                (6, "right"), (10, "left"))
_REPORT_HEAD = ("#", "新增s", "请求帧", "对齐帧", "回放帧", "累计s", "耗时s",
                "模式", "参考", "状态")


def _report_line(values):
    return "  " + "  ".join(_cell(v, w, a)
                            for (w, a), v in zip(_REPORT_COLS, values))


def _report_rule():
    return "  " + "-" * (sum(w for w, _ in _REPORT_COLS)
                         + 2 * (len(_REPORT_COLS) - 1))


# (槽位, 原始尺寸, 缩放后, 处理)
_REF_COLS = ((12, "left"), (13, "right"), (13, "right"), (10, "left"))


def _ref_line(label, orig, new, state):
    return "  " + "  ".join((
        _cell(label, _REF_COLS[0][0], "left"),
        _cell(orig, _REF_COLS[1][0], "right"), "→",
        _cell(new, _REF_COLS[2][0], "right"),
        _cell(state, _REF_COLS[3][0], "left")))


def _format_report(rows, header, warnings, ref_rows=None):
    """Render the run report: a header block, the per-shot table, and notes."""
    out = list(header)
    out.append("")
    out.append(_report_line(_REPORT_HEAD))
    out.append(_report_rule())
    for r in rows:
        out.append(_report_line((
            r["no"], "%.2f" % r["new_seconds"], r["requested_frames"], r["frames"],
            r["handoff_frames"], "%.2f" % r["cumulative"], "%.1f" % r["elapsed"],
            r.get("task", "t2v"), r.get("refs", 0), r["status"])))
    if rows:
        out.append(_report_rule())
        total_new = sum(r["new_seconds"] for r in rows)
        total_frames = sum(r["frames"] for r in rows)
        reused = sum(1 for r in rows if "缓存" in str(r["status"]))
        tail = "  合计 %d 段 · 新增 %.2fs · 生成 %d 帧 · 成片 %.2fs" % (
            len(rows), total_new, total_frames, total_new)
        if reused:
            tail += " · 其中 %d 段走缓存" % reused
        out.append(tail)
    if ref_rows:
        out.append("")
        out.append("参考图：")
        for item in ref_rows:
            out.append(_ref_line(item["label"], item["orig"], item["new"],
                                 "已缩放" if item["changed"] else "原样"))
    if warnings:
        out.append("")
        out.append("注意：")
        for w in warnings[:20]:
            out.append("  · " + w)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Images / timelines
# ---------------------------------------------------------------------------
def _pil_to_tensor(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def _font(size, bold=False):
    for name in ("arial.ttf", "arialbd.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _blank(text):
    img = Image.new("RGB", (480, 240), color=(25, 25, 30))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill=(180, 180, 180), font=_font(14))
    return _pil_to_tensor(img)


def _placeholder_timeline():
    return _blank("尚无任何分镜。")


def _timeline_image(session_dir, records, thumb_width=160, gap=10,
                    show_status=True, show_duration=True):
    """Build the same strip the old H3SegmentTimeline node produced, inline."""
    existing = [r for r in records
                if os.path.exists(os.path.join(session_dir, r["file"]))]
    if not existing:
        return _blank("暂无完成片段")

    thumbs = []
    labels = []
    for r in existing:
        try:
            frame = video_io.last_frame(os.path.join(session_dir, r["file"]))
        except Exception:
            frame = None
        if frame is None:
            continue
        arr = (frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if arr.ndim == 4:
            arr = arr[0]
        pil = Image.fromarray(arr)
        w, h = pil.size
        if w == 0 or h == 0:
            continue
        new_h = max(1, int(thumb_width * h / w))
        thumbs.append(pil.resize((thumb_width, new_h), Image.Resampling.LANCZOS))
        labels.append("%d  %4.2fs" % (r["index"] + 1, r.get("seconds", 0)))

    if not thumbs:
        return _blank("暂无完成片段")

    total_width = sum(t.width for t in thumbs) + gap * (len(thumbs) - 1)
    max_height = max(t.height for t in thumbs)
    status_h = 30 if show_status else 0
    dur_h = 22 if show_duration else 0
    canvas = Image.new("RGB", (total_width, max_height + status_h + dur_h),
                       color=(25, 25, 30))
    draw = ImageDraw.Draw(canvas)

    if show_status:
        canvas.paste(Image.new("RGB", (total_width, status_h), color=(35, 45, 60)),
                     (0, 0))
        done = sum(1 for r in existing
                   if os.path.exists(os.path.join(session_dir, r["file"])))
        draw.text((10, 8),
                  "已完成 %d 段 · 待渲染 %d 段" % (done, max(0, len(records) - done)),
                  fill=(140, 220, 160), font=_font(12))

    x = 0
    for thumb, label in zip(thumbs, labels):
        canvas.paste(thumb, (x, status_h + dur_h))
        if show_duration:
            draw.text((x + 8, max_height + status_h + 4), label,
                      fill=(220, 220, 220), font=_font(12))
        x += thumb.width + gap

    return _pil_to_tensor(canvas)


def _tail_skip_frames(chain_dir, prev_index):
    """兜底：上一段记录没写 handoff 时，用它的尾巴片段长度当回放帧数。

    磁盘补齐的记录（session.Session.reconcile）probe 失败就会是 0。不知道
    回放多长，拼接就不会跳过，每个接口都会多出一段重复的开头 —— 成片看得
    出来是一顿一顿的，宁可多探一次。
    """
    tail = os.path.join(chain_dir, "seg_%02d.tail.mp4" % (prev_index + 1))
    if not os.path.exists(tail):
        return 0
    try:
        return int(video_io.frame_count(tail))
    except Exception:                                            # pragma: no cover
        return 0


def _join_parts(chain):
    """Mirror H3ChainToVideoNode's part list (skip replays past the first)."""
    parts = []
    records = chain["segments"]
    for index, record in enumerate(records):
        path = os.path.join(chain["dir"], record["file"])
        if not os.path.exists(path):
            raise FileNotFoundError(
                "segment %d missing: %s" % (index + 1, path))
        if index == 0:
            skip = 0
        else:
            prev = records[index - 1]
            skip = int(prev.get("handoff") or 0)
            if not skip and prev.get("has_tail"):
                skip = _tail_skip_frames(chain["dir"], int(prev.get("index", index - 1)))
        parts.append((path, skip))
    return parts


# ---------------------------------------------------------------------------
# Source video → shots  (the 智能分镜 branch, folded in)
# ---------------------------------------------------------------------------
def _video_source_path(video):
    """Best-effort filesystem path for an ``io.Video`` input."""
    if video is None:
        return ""
    for attr in ("get_stream_source", "get_path", "get_abspath"):
        fn = getattr(video, attr, None)
        if not callable(fn):
            continue
        try:
            value = fn()
        except Exception:
            continue
        if isinstance(value, str) and value and os.path.exists(value):
            return value
    for attr in ("path", "_path", "filename"):
        value = getattr(video, attr, None)
        if isinstance(value, str) and value and os.path.exists(value):
            return value
    return ""


def _probe_video(path):
    """fps / frame count / seconds / width / height of a video file."""
    import av

    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("文件里没有视频流：%s" % path)
        try:
            rate = stream.average_rate
            fps = float(rate.numerator) / float(rate.denominator) if rate else 24.0
        except Exception:
            fps = 24.0
        if not fps or fps <= 0:
            fps = 24.0
        frames = int(stream.frames or 0)
        if not frames:
            frames = sum(1 for _ in container.decode(stream))
        width = int(getattr(stream.codec_context, "width", 0) or 0)
        height = int(getattr(stream.codec_context, "height", 0) or 0)
    return {"fps": fps, "frames": frames, "seconds": frames / fps,
            "width": width, "height": height}


def _scenedetect_install_hint():
    import sys
    return '"%s" -m pip install "scenedetect>=0.6.4,<0.8"' % (sys.executable or "python")


def _detect_cut_frames(path, sensitivity, min_gap_frames):
    """Interior cut frame indices, or None when PySceneDetect is unavailable."""
    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError:
        return None
    threshold = {"low": 4.5, "medium": 3.0, "high": 2.0}.get(
        str(sensitivity or "medium").strip().lower(), 3.0)
    detector = AdaptiveDetector(adaptive_threshold=threshold,
                                min_scene_len=max(1, int(min_gap_frames)))
    scenes = detect(path, detector, show_progress=False, start_in_scene=True)
    if not scenes:
        return []
    cuts = [int(start.get_frames()) for start, _ in scenes[1:]]
    return sorted({c for c in cuts if c > 0})


def _shot_bounds(cuts, total_frames, min_gap, max_len):
    """Cut list → final shot boundaries (0 … total_frames inclusive).

    Two passes, mirroring MiniMaxH3_Director's ``_merge_close_cuts``:
    the first drops cuts that are closer than ``min_gap`` to the previous
    kept cut, the second guarantees every span is at least
    ``MIN_SEG_FRAMES`` long. Over-long spans are then subdivided so no shot
    exceeds H3's ~15s generation window.
    """
    total = max(0, int(total_frames))
    if total <= 0:
        return [0, 0]
    # NOTE: no early return for "no cuts" — the subdivision pass below still
    # has to run, otherwise a single long source clip (no detected cuts) would
    # come back as one shot longer than H3's generation window.
    ordered = sorted({0, total} | {int(c) for c in (cuts or ()) if 0 < int(c) < total})
    gap = max(MIN_SEG_FRAMES, int(min_gap))
    kept = [ordered[0]]
    for cut in ordered[1:-1]:
        if cut - kept[-1] < gap:
            continue
        kept.append(cut)
    if ordered[-1] - kept[-1] < gap and len(kept) > 1:
        kept.pop()
    kept.append(ordered[-1])

    final = [kept[0]]
    for cut in kept[1:-1]:
        if cut - final[-1] < MIN_SEG_FRAMES:
            continue
        final.append(cut)
    if kept[-1] - final[-1] < MIN_SEG_FRAMES and len(final) > 1:
        final.pop()
    final.append(kept[-1])

    out = [final[0]]
    for cut in final[1:]:
        span = cut - out[-1]
        while span > max_len * 1.5:
            out.append(out[-1] + max_len)
            span = cut - out[-1]
        out.append(cut)
    return out


def _read_frame_at(path, index, fps_hint):
    """Decode one frame as an RGB ndarray, or None. Seeks to the keyframe."""
    import av

    try:
        with av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return None
            rate = stream.average_rate
            fps = float(rate.numerator) / float(rate.denominator) if rate else fps_hint
            if not fps or fps <= 0:
                fps = fps_hint
            target = int(round(float(index) / fps / float(stream.time_base)))
            container.seek(max(0, target), stream=stream, backward=True,
                           any_frame=False)
            for frame in container.decode(stream):
                return frame.to_ndarray(format="rgb24")
    except Exception:                                            # pragma: no cover
        return None
    return None


def _source_preview(path, bounds, fps, max_thumbs=12, thumb_width=150, gap=8):
    """Contact sheet of each detected shot's head frame (dry-run split check)."""
    pairs = []
    for i in range(len(bounds) - 1):
        start_f = int(bounds[i])
        # Head frame, nudged in a little so fade-ins do not dominate.
        pairs.append((i, min(int(bounds[i + 1]) - 1,
                             start_f + max(0, int((bounds[i + 1] - start_f) * 0.15)))))
    pairs = pairs[:max_thumbs]

    thumbs = []
    labels = []
    for i, frame_index in pairs:
        arr = _read_frame_at(path, frame_index, fps)
        if arr is None:
            continue
        pil = Image.fromarray(arr)
        w, h = pil.size
        if w == 0 or h == 0:
            continue
        thumbs.append(pil.resize((thumb_width, int(thumb_width * h / w)),
                                 Image.Resampling.LANCZOS))
        labels.append("#%d  %.2fs→%.2fs" % (i + 1, bounds[i] / fps,
                                            bounds[i + 1] / fps))
    if not thumbs:
        return _placeholder_timeline()

    total_w = sum(t.width for t in thumbs) + gap * (len(thumbs) - 1)
    max_h = max(t.height for t in thumbs)
    canvas = Image.new("RGB", (total_w, max_h + 30 + 22), color=(25, 25, 30))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.new("RGB", (total_w, 30), color=(35, 45, 60)), (0, 0))
    draw.text((10, 8), "源视频分镜预览 · 共 %d 个镜头（试运行，未渲染）"
              % (len(bounds) - 1), fill=(140, 220, 160), font=_font(12))
    x = 0
    for thumb, label in zip(thumbs, labels):
        canvas.paste(thumb, (x, 52))
        draw.text((x + 6, max_h + 34), label, fill=(220, 220, 220), font=_font(12))
        x += thumb.width + gap
    return _pil_to_tensor(canvas)


def _split_source_video(video, video_path, session_name, sensitivity,
                        min_shot_seconds, max_shot_seconds, fallback_prompt,
                        global_prompt, prompt_lines):
    """Detect shots in a source video and build the ``shots_info`` asset."""
    path = _video_source_path(video) or str(video_path or "").strip()
    if not path:
        raise ValueError("没有源视频：请把视频接到 video，或在 video_path 填绝对路径。")
    if not os.path.isfile(path):
        raise FileNotFoundError("源视频不存在：%s" % path)

    info = _probe_video(path)
    fps = info["fps"]
    total_frames = int(info["frames"] or 0)
    if total_frames <= 0:
        raise ValueError("读不到视频帧数：%s" % path)

    min_gap = max(MIN_SEG_FRAMES, int(round(float(min_shot_seconds) * fps)))
    max_len = max(min_gap + 1, int(round(float(max_shot_seconds) * fps)))

    warnings = []
    cuts = _detect_cut_frames(path, sensitivity, min_gap)
    if cuts is None:
        method = "均分（未安装 PySceneDetect）"
        warnings.append("未安装 PySceneDetect，已按时长均分。装了会更准：%s"
                        % _scenedetect_install_hint())
        step = max(min_gap, min(max_len, total_frames))
        bounds = _shot_bounds(list(range(0, total_frames, step)),
                              total_frames, min_gap, max_len)
    else:
        method = "PySceneDetect AdaptiveDetector（%s）" % sensitivity
        bounds = _shot_bounds(cuts, total_frames, min_gap, max_len)
        if len(bounds) - 1 != len(cuts) + 1:
            warnings.append("检测到 %d 个原始切点，合并/限制后确定为 %d 个镜头。"
                            % (len(cuts) + 1, len(bounds) - 1))

    lines = [ln.strip() for ln in str(prompt_lines or "").splitlines()]
    lines = [ln for ln in lines if ln]
    fallback = str(fallback_prompt or "").strip() or "保持画面内容与镜头运动连贯"

    shots_info = []
    for i in range(len(bounds) - 1):
        start_f, end_f = int(bounds[i]), int(bounds[i + 1])
        duration = (end_f - start_f) / fps
        if duration <= 0:
            continue
        prompt = lines[i] if i < len(lines) else ""
        if not prompt:
            prompt = fallback
        shots_info.append({
            "id": i + 1,
            "shot": prompt,
            "duration": round(duration, 3),
            "start": round(start_f / fps, 3),
            "end": round(end_f / fps, 3),
            "appear": {},
            "image_paths": [],
        })

    if not shots_info:
        raise ValueError("没有切出任何分镜。试着调低 min_shot_seconds。")

    asset = {
        "session_name": str(session_name or "my_chain"),
        "duration": round(total_frames / fps, 3),
        "shots_info": shots_info,
        "global": {"global_prompt": str(global_prompt or "").strip()},
        "source_video": path,
        "split": {"method": method, "fps": fps, "total_frames": total_frames},
    }
    if info["width"] >= 32 and info["height"] >= 32:
        asset["width"] = int(info["width"])
        asset["height"] = int(info["height"])

    meta = {"path": path, "fps": fps, "total_frames": total_frames,
            "method": method, "bounds": bounds, "warnings": warnings,
            "width": info["width"], "height": info["height"]}
    return asset, meta


# ---------------------------------------------------------------------------
# Per-segment MP4 export (borrowed from MiniMaxH3_Director segment_mp4_export)
# ---------------------------------------------------------------------------
def _export_segments(chain, label=""):
    """Copy each rendered segment into ``output/h3_seg_export/<timestamp>/``.

    Never raises: an export failure must not lose a finished render.
    """
    import shutil
    from datetime import datetime
    from pathlib import Path

    try:
        import folder_paths

        base = Path(folder_paths.get_output_directory()) / "h3_seg_export"
    except Exception as exc:                                     # pragma: no cover
        log("H3Director: segment export skipped (%s)", exc)
        return None
    try:
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = base / stamp
        n = 1
        while root.exists() and n < 1000:
            root = base / ("%s_%03d" % (stamp, n))
            n += 1
        root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        log("H3Director: segment export dir unavailable (%s)", exc)
        return None

    written = 0
    for index, record in enumerate(chain["segments"]):
        src = os.path.join(chain["dir"], record["file"])
        if not os.path.exists(src):
            continue
        try:
            shutil.copy2(src, str(root / ("seg_%04d.mp4" % (index + 1))))
            written += 1
        except Exception as exc:                                 # pragma: no cover
            log("H3Director: segment %d export failed (%s)", index + 1, exc)
    log("H3Director exported %d segments -> %s", written, root)
    return str(root)


# ---------------------------------------------------------------------------
# H3Director
# ---------------------------------------------------------------------------
def _parse_run_segments(text, total):
    """解析 "1,3,5-7" 形式的段号选择 → 1-based 段号集合；空/非法返回 None（=全部）。

    与前端 buildTimelinePanel 里的 JS parseRunSegments 保持同一套语法，
    便于双向同步。区间按 1..total 裁剪、去重、按出现顺序收集。"""
    if not text or not str(text).strip():
        return None
    out = set()
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            parts = tok.split("-", 1)
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                if 1 <= i <= total:
                    out.add(i)
        else:
            try:
                n = int(tok)
            except ValueError:
                continue
            if 1 <= n <= total:
                out.add(n)
    return out or None


class H3DirectorNode(io.ComfyNode):
    """One node to replace: MinimaxH3ScriptBoard + H3ScriptBatchRender +
    H3ChainToVideo + H3SegmentTimeline + MinimaxH3SaveJson + H3ShotSplit."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3Director",
            display_name="🎬 ComfyUI_Marquee_Director（剧本/源视频 → 渲染 → 时间线 → 成片）",
            category=CATEGORY,
            description=(
                "One node for the whole take. Pick 剧本来源, connect `settings` "
                "(H3_SETTINGS), and the node renders every shot, joins them, "
                "returns the timeline thumbnail and the joined video, saves the "
                "cleaned JSON next to the session manifest, and emits a full run "
                "report including the 17k+5 frame-grid table.\n\n"
                "剧本来源 = 分镜JSON: consume a Ref2VA-Auto-compatible payload "
                "on `shots_json`.\n"
                "剧本来源 = 源视频自动切分: connect a source video instead; "
                "PySceneDetect finds the cuts and each detected shot is "
                "rendered. Combine with `dry_run` to preview the cuts first.\n\n"
                "`chain_state` appends onto an existing session instead of "
                "starting a new one — that is the unlimited-shot loop.\n\n"
                "`dry_run` resolves every shot, snaps it onto H3's 17k+5 grid "
                "and predicts the resume cache, then returns the plan without "
                "sampling."
            ),
            hidden=[io.Hidden.unique_id],
            inputs=[
                # NOTE: every input here is `optional=True` on purpose. ComfyUI
                # splits the widget list into `required` then `optional`
                # (comfy_api._io.create_input_dict_v1), and the frontend fills
                # `widgets_values` in that order — mixing the two sections
                # silently reorders them and desyncs saved workflows (symptom:
                # "Value 1.625 bigger than max of 1.0: stabilize"). With a single
                # section the widget order is exactly the schema order.
                # `settings` is still mandatory: enforced at the top of execute().
                H3Settings.Input("settings", optional=True),
                io.String.Input(
                    "shots_json", display_name="分镜资产结果JSON", multiline=True,
                    optional=True, force_input=True,
                    tooltip="「分镜JSON」模式下使用。Ref2VA-Auto 兼容的"
                            "分镜资产结果JSON；支持 ```json``` 围栏。"),
                io.String.Input(
                    "pack_info", display_name="PACK信息", default="",
                    multiline=False, optional=True, force_input=True,
                    tooltip="接「H3 分镜提示词 PACK」的 pack_info 输出。"
                            "有了它，Director 的 PACK 模块就能显示"
                            "（项目 / 模式 / 总时长 / 段数 / 画幅），"
                            "并把 PACK 声明的画幅和 H3 Chain Settings 的画布对账。"),
                io.Video.Input(
                    "video", display_name="源视频", optional=True,
                    tooltip="「源视频自动切分」模式下使用。留空则用 video_path。"),
                io.String.Input(
                    "global_prompt", display_name="全局提示词", default="",
                    multiline=True, optional=True,
                    tooltip="追加到每个分镜提示词后面（整体风格 / 声音氛围）。"
                            "非空时覆盖 JSON 自带的 global_prompt。"),
                io.String.Input(
                    "prompt_lines", display_name="分镜提示词（一行一个）",
                    default="", multiline=True, optional=True,
                    tooltip="仅「源视频自动切分」生效：一行对应一个镜头。"
                            "行数不够时用缺省提示词顶上。"),
                io.Combo.Input(
                    "sensitivity", display_name="切点灵敏度",
                    options=["low", "medium", "high"], default="medium",
                    optional=True, advanced=True,
                    tooltip="仅「源视频自动切分」生效：high 切得更碎。"),
                io.Float.Input(
                    "min_shot_seconds", display_name="最短镜头（秒）", default=1.0,
                    min=0.2, max=15.0, step=0.1, round=False, optional=True,
                    advanced=True,
                    tooltip="短于这个时长的碎镜头会被合并掉。"),
                io.Float.Input(
                    "max_shot_seconds", display_name="最长镜头（秒）", default=15.0,
                    min=1.0, max=60.0, step=0.5, round=False, optional=True,
                    advanced=True,
                    tooltip="长于这个时长的镜头会被再切开（H3 单次上限约 15s）。"),
                io.String.Input(
                    "fallback_prompt", display_name="缺省提示词",
                    default="保持画面内容与镜头运动连贯", multiline=True,
                    optional=True, advanced=True,
                    tooltip="自动切分出的镜头没有对应提示词时用这句顶上。"),
                H3Chain.Input(
                    "chain_state", display_name="续接会话", optional=True,
                    tooltip="接「H3 Load Session」的 chain，本次的 shots_info 会"
                            "渲染成该会话的第 N+1、N+2… 段，而不是新建会话——"
                            "这就是无限分镜的追加模式。留空则新建会话。"),
                io.Int.Input(
                    "max_shots", display_name="最多渲染分镜（0=全部）", default=0,
                    min=0, max=9999, optional=True, advanced=True),
                io.Float.Input(
                    "handoff_seconds", display_name="回放衔接（秒）", default=1.625,
                    min=0.0, max=4.0, step=0.125, round=False, optional=True,
                    advanced=True),
                io.Boolean.Input(
                    "resume", default=True, optional=True, advanced=True,
                    tooltip="复用上次渲染结果；未变化的分镜不会重渲。"),
                io.Boolean.Input(
                    "duration_is_new_content", display_name="duration=新增时长",
                    default=True, optional=True, advanced=True,
                    tooltip="开=JSON 里的 duration 是新增时长；关=直接当生成长度。"),
                io.Boolean.Input(
                    "unload_models_after", display_name="每段后卸载模型",
                    default=False, optional=True, advanced=True,
                    tooltip="每段渲染完卸载模型并清显存。OOM 时打开；"
                            "默认关以节省加载时间。"),
                io.Float.Input(
                    "crf", default=14.0, min=0.0, max=51.0, step=1.0, optional=True,
                    advanced=True,
                    tooltip="拼接成片的质量；0=无损，越低越大。"),
                io.Float.Input(
                    "stabilize", display_name="亮度稳定", default=0.0, min=0.0,
                    max=1.0, step=0.05, round=False, optional=True,
                    advanced=True,
                    tooltip="修正多段拼接的亮度漂移；0..1 之间的修正强度。"),
                io.Boolean.Input(
                    "build_timeline", display_name="生成时间线", default=True,
                    optional=True, advanced=True,
                    tooltip="出片后生成时间线缩略图。关掉可省一次抽帧。"),
                io.Boolean.Input(
                    "export_segments", display_name="逐段导出 MP4", default=False,
                    optional=True, advanced=True,
                    tooltip="把每段单独复制到 "
                            "output/h3_seg_export/<时间戳>/seg_XXXX.mp4。"),
                io.Combo.Input(
                    "ref_image_size", display_name="参考图尺寸",
                    options=REF_SIZE_OPTIONS, default=REF_SIZE_MATCH,
                    optional=True, advanced=True,
                    tooltip="参考图统一预处理：match=跟随输出画布的最长边，"
                            "其余为最长边像素上限。只缩不放，并把宽高对齐到 32"
                            "（H3 的 VAE ÷16 后再 2×2 patch，非 32 倍数会在"
                            "采样时崩在 patchify）。"),
                # NOTE: Autogrow.Input has no `advanced` kwarg in comfy_api
                # (signature: id, template, display_name, optional, tooltip,
                # lazy, extra_dict). Passing it raises TypeError and takes the
                # whole extension down with "IMPORT FAILED".
                io.String.Input(
                    "ref_classify", display_name="参考图分类", default="",
                    multiline=True, optional=True, advanced=True,
                    tooltip="参考图的角色/场景分类，前端面板写这里，格式 "
                            "JSON 数组：[{\"slot\":1,\"kind\":\"角色\","
                            "\"name\":\"王总\"}, {\"slot\":3,\"kind\":\"场景\","
                            "\"name\":\"古代卧房\"}]。目前只用于报告与面板显示；"
                            "真正绑图看 <Picture N> 编号。"),
                io.Autogrow.Input(
                    "ref_images", display_name="参考图覆盖", optional=True,
                    tooltip="用同一组参考图覆盖所有分镜。留空按 JSON 原样加载。",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", optional=True),
                        # min 必须是 2 而不是 0：min=0 时前端一个槽位都不展开，
                        # 节点上完全看不到"往哪接图"，用户会以为没有这个功能。
                        prefix="ref_image_", min=2, max=9)),
                # 单段修复：放在最后，不挤占既有控件的 widget 顺序（已保存工作流
                # 的 widgets_values 仍按原序对齐，本项用默认值 0 补齐）。
                io.Int.Input(
                    "repair_segment", display_name="单段修复（0=关闭）", default=0,
                    min=0, max=9999, optional=True, advanced=True,
                    tooltip="≥1 = 只重渲该段（首尾锚定，不动后续段），适合修一段坏镜头。"
                            "设完后点时间线里对应段块即可触发；跑完自动复位为 0。"),
                # 选择性渲染：仅重渲指定段，其余段复用磁盘缓存（不重新烧显卡）。
                # 与单段修复互斥：两者同时设时单段修复优先。
                io.String.Input(
                    "run_segments", display_name="选择渲染段（空=全部）", default="",
                    multiline=False, optional=True, advanced=True,
                    tooltip="从「最早的选中段」起重渲到结尾，之前的段复用磁盘缓存并"
                            "照常参与拼接。格式：逗号分隔的段号与区间，如 3 或 1,3,5-7"
                            "（实际取其中最小的段号作为重渲起点）。留空 = 全部渲染。"
                            "注意连续分镜首尾锚定耦合：后段开头是前段尾帧的回放，所以"
                            "重渲第 N 段必然带着其后所有段一起重渲，只渲孤立段号会让"
                            "后续段复用陈旧缓存、接缝跳变。"
                            "（与单段修复 repair_segment 的两段锚定不同。）"),
            ],
            outputs=[
                io.String.Output(
                    "shots_json", display_name="分镜资产结果JSON（实际使用）"),
                H3Chain.Output("chain", display_name="chain"),
                io.Video.Output("video", display_name="成片"),
                io.Image.Output("timeline", display_name="时间线"),
                io.String.Output("progress", display_name="进度摘要"),
                io.String.Output("report", display_name="运行报告"),
                io.String.Output("pack_info", display_name="PACK信息",
                                 tooltip="JSON：项目 / 模式 / 总时长 / 段数 / "
                                         "画幅 + 每段时长 + 参考图槽位 + 画布对账。"),
            ],
        )

    @classmethod
    def execute(cls, source=SOURCE_JSON, settings=None, shots_json="", video=None,
                video_path="", global_prompt="", prompt_lines="",
                sensitivity="medium", min_shot_seconds=1.0,
                max_shot_seconds=15.0,
                fallback_prompt="保持画面内容与镜头运动连贯", chain_state=None,
                session_name="", max_shots=0,
                handoff_seconds=1.625, resume=True, duration_is_new_content=True,
                unload_models_after=False, crf=14.0, stabilize=0.0,
                build_timeline=True, export_segments=False, ref_images=None,
                ref_image_size=REF_SIZE_MATCH, pack_info="",
                ref_classify="", repair_segment=0, run_segments=""):
        mode = _source_mode(source)
        node_id = _node_id(cls)

        if not settings:
            raise ValueError("settings 未连接：请把 H3 Chain Settings 的 "
                             "H3_SETTINGS 接到 settings 口。")
        # Every input is optional now (see the note in define_schema), so an
        # older or hand-edited workflow can hand us None. Fall back to the
        # documented default instead of crashing on float(None).
        sensitivity = sensitivity or "medium"
        if min_shot_seconds is None:
            min_shot_seconds = 1.0
        if max_shot_seconds is None:
            max_shot_seconds = 15.0
        if max_shots is None:
            max_shots = 0
        if handoff_seconds is None:
            handoff_seconds = 1.625
        if crf is None:
            crf = 14.0
        if stabilize is None:
            stabilize = 0.0
        ref_image_size = ref_image_size or REF_SIZE_MATCH

        # 1. Resolve the shot list --------------------------------------------
        _phase_text(cls, "%s：%s" % (_PHASE_LABELS["plan"], "解析剧本"))
        if mode == "video":
            asset, split_meta = _split_source_video(
                video, video_path, session_name or "my_chain", sensitivity,
                min_shot_seconds, max_shot_seconds, fallback_prompt,
                global_prompt, prompt_lines)
            _phase_text(cls, "%s：切出 %d 个镜头"
                        % (_PHASE_LABELS["plan"], len(asset["shots_info"])))
        else:
            asset = _clean_shot_json(shots_json)
            split_meta = None

        # The global-prompt widget wins over whatever the JSON carried: it is
        # the one knob people reach for when they want to restyle a whole take.
        text_global = str(global_prompt or "").strip()
        if text_global:
            block = asset.get("global")
            block = dict(block) if isinstance(block, dict) else {}
            block["global_prompt"] = text_global
            asset["global"] = block

        shots = [s for s in (asset.get("shots_info") or []) if isinstance(s, dict)]
        limit = int(max_shots or 0)
        if limit > 0:
            shots = shots[:limit]
        if not shots:
            hint = ("把源视频接到 video（或在 video_path 填路径）"
                    if mode == "video" else
                    "把分镜资产结果JSON 接到 shots_json，或把「剧本来源」"
                    "切成「源视频自动切分」")
            raise ValueError("没有可渲染的分镜。%s" % hint)

        sess_name = _session_name_from(asset, session_name)
        settings = _settings_for_script(settings or {}, asset)
        # The session name is a director decision, not a graph-topology one:
        # H3RenderSegmentNode reads settings["session_name"] when chain_state is
        # None, so the widget only takes effect if it lands here.
        settings = dict(settings, session_name=session_mod.sanitize(sess_name))

        # PACK 模块的数据源。接了「H3 分镜提示词 PACK」的 pack_info 就用它的
        # （那是真从 PACK 文本里解析出来的）；没接就退化成一个只带段数的壳，
        # 面板照样能显示，只是头部五项为空。
        info = _load_pack_info(pack_info, asset, settings)

        # Append mode: everything downstream continues the session we were handed.
        chain = chain_state
        start_index = int(chain["index"]) + 1 if chain else 0

        # 2. Planning state (also the source of the report) -------------------
        # Reference images are fitted once, up front: every shot that uses them
        # then picks from the same processed pool, so the cache key sees exactly
        # the tensors that will be rendered.
        # ★ 稀疏槽位字典：键 = <Picture N> 的 N，**不压实**。
        # 旧写法是 ordered_autogrow() 丢掉空口后压成 list，再按 pool[index-1]
        # 取图 —— 接了 0,1,2,3,6 时 pool[4] 其实是 ref_image_6 那张，<Picture 5>
        # 会错拿它，后面几张整体绑错槽位、人物直接换脸。
        # 连续接满时两者取图结果完全一致，缓存键不受影响。
        raw_slots = ref_images_mod.collect_slots(ref_images)
        warnings = []
        if len(raw_slots) > MAX_REFERENCE_IMAGES:
            # collect_slots 已经按槽号裁到 9 了，这里只是兜底提示。
            warnings.append("参考图接了 %d 张，超过 H3 上限 %d 张，多余的会被忽略"
                            % (len(raw_slots), MAX_REFERENCE_IMAGES))
        pool_labels = _pool_labels(raw_slots, ref_classify)
        # 闪电渲染 / 步数覆盖：必须参考图预处理之后、渲染之前改 settings，
        # 改完的 model / sigmas 才会进每一段的缓存键。
        # 当前版本已收起这些面板控件（节点上不再有 lightning/render_steps/
        # scheduler/lightning_lora/lightning_lora_strength 这几个 widget），
        # 所以这里直接传关闭状态：保持与未改动时行为完全一致。
        settings = _apply_lightning(
            settings, False, NO_LORA, 1.0, 0, FOLLOW_UPSTREAM, warnings)
        ref_pool, ref_rows = _prepare_ref_images(
            raw_slots, ref_image_size, settings, labels=pool_labels)
        # ref_pool 现在是 dict，必须遍历 values()；写成 `for t in ref_pool`
        # 只会拿到槽号（int），pool_ids 就全错了。
        pool_ids = {id(t) for t in ref_pool.values()}
        ref_seen = {(r["orig"], r["new"]) for r in ref_rows}

        # 参考图对账：提示词要 N 张、实际接了几张，差一张就会有人物换脸。
        info = _check_ref_slots(info, shots, raw_slots, warnings)

        total = len(shots)
        rows = []

        # A5 选择性渲染：解析 run_segments（"1,3,5-7"）→ 段号集合；空/非法=全部。
        # run_set 为 None 表示走原逻辑（resume 由控件决定）；非 None 表示选择性模式，
        # 选中段强制重渲、未选中段走缓存复用（复用现有 resume 缓存机制与首尾锚定）。
        if run_segments is None:
            run_segments = ""
        run_set = _parse_run_segments(run_segments, total)
        run_mode = run_set is not None
        # 连续分镜首尾锚定耦合：后段开头是前段尾帧的回放，所以"重渲第 N 段"
        # 物理上必然要求 N 之后所有段一起重渲。实际重渲区间因此是
        # [最早选中段, 结尾]，而不是字面上的那几个孤立段号 —— 否则后面的段
        # 会复用基于旧第 N 段的陈旧缓存，接缝直接跳变。
        run_from = min(run_set) if run_mode else None

        # 先按磁盘 reconcile 一下，磁盘上已渲好的段数随 plan 一起广播出去。
        # 不带 done_count 的话前端在收到 plan 事件时只能把 doneSet 清零，
        # 进度条瞬间从「已渲 4/6」掉回「已渲 0/6」，要等后续段渲完才会爬回
        # 正确数字（且最终数字还少 4，因为被清掉的 4 段永远不会被加回来）。
        sess = chain["sess_obj"] if chain else session_mod.Session(settings["session_name"])
        manifest = sess.load() if resume else None
        recovered = 0
        if resume:
            manifest = sess.reconcile(manifest)
            recovered = int((manifest or {}).get("recovered") or 0)
            if recovered:
                log("resume: %s 磁盘上发现 %d 段 manifest 未记录的已完成分镜，本次直接复用",
                    sess.name, recovered)
        # 用磁盘上真实存在的「从 seg_01 起连续」的段数，而不是 manifest 的
        # 记录条数。两者会不一致：manifest 有记录但文件被删、或文件在但记录
        # 没补上。用记录条数会让面板的「已渲 X/Y」显示成磁盘上并不存在的数。
        # disk_segments() 与 /h3/session 的 done 口径一致（都只数连续段）。
        done_count = sess.disk_segments()
        # 真实存在的段号列表（含中间缺段时后面的孤立段）。前端拿它建 doneSet，
        # 就跟 /h3/session 的 segments 口径一致了 —— 否则「面板刷新」和
        # 「plan 广播」两条路算出来的「已渲 X/Y」会不一致，进度条跳变。
        done_indices = sess.disk_segment_indices()

        # 时间线模块：先广播一次"要渲几段、每段多长"，前端把格子铺好，
        # 之后每段开始/结束再各广播一次，当前段高亮 + 小马起跑。
        routes_mod.emit_progress(
            node_id, event="plan", total=total, session=sess_name,
            done_count=done_count, done_indices=done_indices,
            recovered=recovered,
            segments=[{"id": s.get("id") or ("S%02d" % (i + 1)),
                       "duration": s.get("duration"),
                       "ref_images": s.get("ref_images") or []}
                      for i, s in enumerate(shots)])
        previous_key = chain["key"] if chain else None

        # ---- 单段修复模式：跳过整条循环，只重渲 repair_segment 这一段 ----
        # 复用 H3RepairSegmentNode 的"首尾锚定"逻辑，写回同一会话的 seg_NN.mp4
        # 并更新 manifest，再走下方统一的拼接/时间线/报告收尾。0 或越界=整条渲染。
        repair_mode = bool(repair_segment) and 1 <= int(repair_segment) <= len(shots)
        if repair_mode:
            ridx = int(repair_segment) - 1
            rshot = shots[ridx]
            r_override = _shot_ref_images(rshot, ref_pool, warnings)
            r_prompt, r_seconds, r_images = _script_segment(
                asset, rshot, ref_images_override=r_override)
            r_from_pool = bool(r_override) and all(id(t) in pool_ids for t in r_override)
            r_images, r_details = _prepare_ref_images(
                r_images, ref_image_size, settings,
                labels=None if r_from_pool else
                ["分镜%d·图%d" % (ridx + 1, k + 1) for k in range(len(r_images))])
            # 修复需要已存在的会话：续接模式直接有 chain_state，否则从磁盘读 manifest
            if chain is None:
                _sess = session_mod.Session(settings["session_name"])
                # 按磁盘补齐：manifest 可能少记了已经渲好的段，不补齐的话
                # 「重渲第 3 段」会因为只有 1 条记录而报越界。
                _man = _sess.reconcile(_sess.load())
                if not _man:
                    raise FileNotFoundError(
                        "单段修复需要先整条渲染一次：会话 %s 在磁盘上找不到 manifest。" % _sess.name)
                chain = {"session": _sess.name, "dir": _sess.dir,
                         "segments": _man["segments"], "sess_obj": _sess,
                         "index": len(_man["segments"]) - 1, "key": None}
            r_replay, r_tail = _shot_windows(rshot, ridx, total, handoff_seconds)
            r_plan = _shot_plan(r_seconds, r_replay, r_tail,
                                duration_is_new_content)
            routes_mod.emit_progress(
                node_id, event="segment_start", index=ridx + 1, total=total,
                shot_id=rshot.get("id") or ("S%02d" % (ridx + 1)),
                frames=r_plan["frames"], gen_seconds=r_plan["gen_seconds"],
                task=_infer_task(r_images), refs=len(r_images), session=sess_name)
            _rep_video, rep_chain, _rep_summary = H3RepairSegmentNode.execute(
                settings, sess_name, int(repair_segment), r_prompt, r_seconds,
                r_tail, 0, True,
                images=_images_autogrow(r_images), chain_state=chain_state)
            routes_mod.emit_progress(
                node_id, event="segment_done", index=ridx + 1, total=total,
                shot_id=rshot.get("id") or ("S%02d" % (ridx + 1)),
                frames=r_plan["frames"], elapsed=0, session=sess_name)
            chain = rep_chain
            rows = [{
                "no": ridx + 1, "new_seconds": r_plan["new_seconds"],
                "requested_frames": r_plan["requested_frames"],
                "frames": r_plan["frames"],
                "handoff_frames": r_plan["handoff_frames"],
                "cumulative": r_plan["new_seconds"], "elapsed": 0.0,
                "task": _infer_task(r_images), "refs": len(r_images),
                "status": "已修复",
            }]
            cumulative = r_plan["new_seconds"]
            handoff_overrides = []

        header = [
            "H3 Director 运行报告",
            "会话   %s" % sess.name,
            "目录   %s" % sess.dir,
            "来源   %s" % ("源视频自动切分" if mode == "video" else "分镜JSON"),
        ]
        if split_meta is not None:
            header.append("源片   %s" % split_meta["path"])
            header.append("切分   %s · %.2ffps · %d 帧 → %d 个镜头"
                          % (split_meta["method"], split_meta["fps"],
                             split_meta["total_frames"], len(shots)))
            header.append("分辨率 %s x %s（跟随源视频）"
                          % (settings.get("width"), settings.get("height")))
        else:
            header.append("分辨率 %s x %s @ %dfps"
                          % (settings.get("width"), settings.get("height"), FPS))
        header.append("模式   %s" % ("续接会话：已有 %d 段，本次从第 %d 段开始"
                                     % (start_index, start_index + 1) if chain
                                     else "新建会话"))
        if recovered:
            header.append("断点   磁盘上另有 %d 段 manifest 未记录，已判定为已完成并复用"
                          % recovered)
        header.append("口径   duration = %s" % ("新增时长（生成时自动 +handoff）"
                                                if duration_is_new_content
                                                else "生成长度"))
        if run_mode:
            header.append(
                "选择渲染 第 %d 段起重渲到结尾（勾选 %s）· 之前各段复用磁盘缓存"
                % (run_from, ",".join(str(i) for i in sorted(run_set))))
        # 回放帧必须和实际生成用同一个方向：向上对齐。用 guide_length（向下）
        # 会把 1.6s(38 帧) 显示成 22 帧(0.917s)，跟报告正文里的 39 帧对不上。
        header.append("回放   %.3fs → %d 帧" % (float(handoff_seconds),
                                                guide_length_up(seconds_to_frames(handoff_seconds))))
        if ref_pool:
            resized = sum(1 for r in ref_rows if r["changed"])
            header.append("参考图 %d 张 · 尺寸 %s（长边 ≤ %s）%s"
                          % (len(ref_pool), ref_image_size,
                             _ref_edge_limit(ref_image_size, settings) or "不限",
                             " · 已缩放 %d 张" % resized if resized else " · 无需缩放"))

        cumulative = 0.0
        handoff_overrides = []
        pbar = comfy.utils.ProgressBar(total, node_id=node_id)

        for index, shot in enumerate(shots):
            if repair_mode:
                break   # 单段修复已在上方面板处理完，跳过整条循环
            shot_no = start_index + index + 1
            override = _shot_ref_images(shot, ref_pool, warnings)
            prompt, seconds, images = _script_segment(
                asset, shot, ref_images_override=override)
            from_pool = bool(override) and all(id(t) in pool_ids for t in override)
            images, details = _prepare_ref_images(
                images, ref_image_size, settings,
                labels=None if from_pool else
                ["分镜%d·图%d" % (shot_no, k + 1) for k in range(len(images))])
            if not from_pool:
                for detail in details:
                    key = (detail["orig"], detail["new"])
                    if key not in ref_seen:
                        ref_seen.add(key)
                        ref_rows.append(detail)
            is_last = index == total - 1
            # 段首回放 / 尾部锚点分开取（规范 0.1）：replay_in 进生成时长，
            # tail_out 只决定渲完切多长的尾巴给下一段。首段 replay_in=0（没有
            # 上一段可回放），末段 tail_out=0（没有下一段要锚点）。
            replay_in, tail_out = _shot_windows(shot, index, total,
                                                handoff_seconds)
            shot_handoff = replay_in
            plan = _shot_plan(seconds, replay_in, tail_out,
                              duration_is_new_content)
            cumulative += plan["new_seconds"]
            task = _infer_task(images)

            if shot.get("hard_cut"):
                warnings.append("分镜 %d：HARD CUT，本段不接上一段尾帧" % (index + 1))
            if abs(shot_handoff - float(handoff_seconds)) > 0.001:
                handoff_overrides.append((index + 1, shot_handoff))
            if plan["requested_frames"] != plan["frames"]:
                warnings.append(
                    "分镜 %d：%.2fs = %d 帧不在 17k+5 网格上，已上调到 %d 帧"
                    % (index + 1, plan["gen_seconds"], plan["requested_frames"],
                       plan["frames"]))
            if shot_handoff and not plan["handoff"] and not is_last:
                warnings.append(
                    "分镜 %d 太短，已自动关闭回放衔接（否则整段都是回放）" % (index + 1))
            if plan.get("handoff_raised"):
                warnings.append(
                    "分镜 %d：回放 %.3fs 不在 17k+5 网格上，已上调到 %.3fs（%d 帧）"
                    % (index + 1, float(shot_handoff), plan["handoff"],
                       plan["handoff_frames"]))
            if plan["frames"] <= MIN_SHOT_FRAMES and seconds > 0:
                warnings.append("分镜 %d 只有 %d 帧，接近 H3 下限，建议加长"
                                % (index + 1, plan["frames"]))
            if len(images) > MAX_REFERENCE_IMAGES:
                warnings.append("分镜 %d 用了 %d 张参考图，H3 只认前 %d 张"
                                % (index + 1, len(images), MAX_REFERENCE_IMAGES))

            if False:  # dry_run widget removed; preview-only path disabled
                hit, previous_key = _cache_state(
                    sess, manifest, start_index + index, settings, prompt, seconds,
                    images, plan["tail_out"], previous_key)
                status = "待定" if hit is None else ("缓存复用" if hit else "将渲染")
                # 先数再删。原来写成 `del images` 之后还 `len(images)`，
                # 任何 dry_run 都会在这一行 UnboundLocalError ——也就是说
                # 「试运行」这个开关从来没真的跑通过，而单段重渲正是靠
                # dry_run 让 Director 只吐 shots_json 不烧显卡的。
                ref_count = len(images)
                del images
                rows.append({
                    "no": start_index + index + 1,
                    "new_seconds": plan["new_seconds"],
                    "requested_frames": plan["requested_frames"],
                    "frames": plan["frames"],
                    "handoff_frames": plan["handoff_frames"],
                    "cumulative": cumulative,
                    "elapsed": 0.0,
                    "task": task,
                    "refs": ref_count,
                    "status": status,
                })
                pbar.update(index + 1)
                continue

            # A5 选择性渲染：从 run_from 起重渲（resume=False，忽略缓存），
            # 之前的段走缓存复用（resume=True，命中则复用磁盘视频并照常衔接）。
            # 区间用 >= run_from 而不是「段号是否在集合里」——因为后段回放前段尾帧，
            # 只渲孤立段号会让后续段复用陈旧缓存、接缝跳变。
            # 之前段若磁盘无缓存（首次整条渲染），会自然回落为渲染。
            selected = (not run_mode) or ((index + 1) >= run_from)
            resume_eff = bool(resume) if not run_mode else (not selected)
            started = time.time()
            # 不再每段推 send_progress_text —— 跟面板里的 setPony 段起始文案
            # 完全重复，去掉这一行后节点标题下方只剩 ProgressBar 百分比条，
            # 面板里的「第 N/M 段 · S0X · N 帧 · task · 参考图 N 张」更全。
            # 其他阶段（plan / context_encode / decode / join / finish）的 _phase_text
            # 保留，那是阶段标签，跟面板状态不撞。
            # 时间线模块：这一段开始跑 → 前端把这一段高亮 + 小马起跑
            routes_mod.emit_progress(
                node_id, event="segment_start", index=index + 1, total=total,
                shot_id=shot.get("id") or ("S%02d" % (index + 1)),
                frames=plan["frames"], gen_seconds=plan["gen_seconds"],
                task=task, refs=len(images), session=sess_name)
            # 第 5 个参数是**尾部锚点**的长度（渲完从结果里切多长给下一段），
            # 不是生成时长 —— 生成时长是第 4 个参数，已经把 replay_in 加进去了。
            # 以前这里传的是回放长度，于是首段被要求"切一条 1.6s 的尾巴"却没有
            # 对应的额外素材，末段则被判成不留锚点。
            output = H3RenderSegmentNode.execute(
                settings, resume_eff, prompt, plan["gen_seconds"],
                plan["tail_out"], 0,
                bool(unload_models_after), chain_state=chain,
                images=_images_autogrow(images),
            )
            last_video_path = output[0]
            chain = output[1]
            if hasattr(last_video_path, "get_stream_source"):
                src = last_video_path.get_stream_source()
                if isinstance(src, str):
                    last_video_path = src
            # Drop the reference cycle so the next segment rebuilds anchors, then
            # sweep whatever the sampler left behind (Comfy leaves dead
            # LoadedModel slots around after every H3 segment).
            free_between_segments(bool(unload_models_after))
            cleanup_segment_vram(bool(unload_models_after))
            rows.append({
                "no": start_index + index + 1,
                "new_seconds": plan["new_seconds"],
                "requested_frames": plan["requested_frames"],
                "frames": plan["frames"],
                "handoff_frames": plan["handoff_frames"],
                "cumulative": cumulative,
                "elapsed": time.time() - started,
                "task": task,
                "refs": len(images),
                "status": ("缓存复用" if (run_mode and not selected) else "完成"),
            })
            routes_mod.emit_progress(
                node_id, event="segment_done", index=index + 1, total=total,
                shot_id=shot.get("id") or ("S%02d" % (index + 1)),
                frames=plan["frames"], elapsed=round(time.time() - started, 2),
                session=sess_name)
            pbar.update(index + 1)

        if handoff_overrides:
            uniq = sorted({round(v, 3) for _, v in handoff_overrides})
            warnings.append(
                "逐镜回放：%d 个分镜用自己的 %s，未跟随全局 %.3fs"
                % (len(handoff_overrides),
                   " / ".join("%.3fs" % v for v in uniq), float(handoff_seconds)))

        # 2b. Dry run: hand back the plan and touch nothing -------------------
        # dry_run widget removed; this preview-only return path is disabled.
        if False:
            timeline_img = _placeholder_timeline()
            if split_meta is not None:
                try:
                    timeline_img = _source_preview(
                        split_meta["path"], split_meta["bounds"], split_meta["fps"])
                except Exception as exc:                         # pragma: no cover
                    log("H3Director: split preview failed (%s)", exc)
            report = _format_report(rows, header + _pack_info_block(info),
                             warnings, ref_rows)
            _phase_text(cls, "%s：%d 个分镜已解析" % (_PHASE_LABELS["finish"], total))
            return io.NodeOutput(
                _json_dumps(asset), chain, VideoFromFile(""), timeline_img,
                "试运行：%d 个分镜已解析，未渲染任何帧。" % total,
                report,
                _json_dumps(info),
            )

        # 3. Join into one cut (same H3ChainToVideo helper) --------------------
        _phase_text(cls, _PHASE_LABELS["join"])
        video_out = VideoFromFile("")
        if chain is not None and chain.get("segments"):
            out_path = os.path.join(chain["dir"], "%s.mp4" % chain["session"])
            try:
                joined_path, _frames = video_io.join(
                    _join_parts(chain), out_path,
                    crf=float(crf), stabilize=float(stabilize))
                video_out = VideoFromFile(joined_path)
                log("H3Director joined %d segments -> %s",
                    len(chain["segments"]), joined_path)
            except (FileNotFoundError, ValueError) as exc:
                warnings.append("拼接跳过：%s" % exc)
                log("H3Director: chain assembled but join skipped (%s)", exc)

        # 4. Save the cleaned JSON next to the session manifest ----------------
        if chain is not None and chain.get("dir"):
            try:
                sess_disk = session_mod.Session(chain["session"])
                meta_path = os.path.join(sess_disk.dir, "shots.json")
                with open(meta_path, "w", encoding="utf-8") as fh:
                    fh.write(_json_dumps(asset))
            except Exception as exc:                             # pragma: no cover
                log("H3Director: shots.json not written (%s)", exc)

        # 5. Optional per-segment export ---------------------------------------
        if export_segments and chain is not None and chain.get("segments"):
            export_dir = _export_segments(chain)
            if export_dir:
                header.append("导出   %s" % export_dir)

        # 6. Timeline thumbnail --------------------------------------------------
        timeline_img = _placeholder_timeline()
        if build_timeline and chain is not None and chain.get("segments"):
            try:
                timeline_img = _timeline_image(
                    chain["dir"], chain["segments"],
                    thumb_width=160, gap=10,
                    show_status=True, show_duration=True)
            except Exception as exc:                             # pragma: no cover
                log("H3Director: timeline preview failed (%s)", exc)

        # 7. Progress + report ----------------------------------------------------
        records = (chain or {}).get("segments") or []
        chain_dir = (chain or {}).get("dir", "")
        done = sum(1 for r in records
                   if os.path.exists(os.path.join(chain_dir, r["file"])))
        progress = "已渲染 %d / %d 段，会话：%s" % (done, total, sess.name)
        if total and done == 0:
            progress += "（本次运行未产出，可能是 JSON 为空或每段都被 resume 跳过）"

        if chain:
            header.append("结果   %s" % _summarize_chain(chain))

        report = _format_report(rows, header + _pack_info_block(info),
                             warnings, ref_rows)
        _phase_text(cls, "%s：%s" % (_PHASE_LABELS["finish"], progress))
        routes_mod.emit_progress(
            node_id, event="finish", total=total, session=sess_name)
        return io.NodeOutput(
            _json_dumps(asset), chain, video_out, timeline_img, progress, report,
            _json_dumps(info),
        )


def _summarize_chain(chain):
    records = chain["segments"]
    total = sum(r["length"] for r in records) - sum(r["handoff"] for r in records[:-1])
    return "%d 段 · 成片 %.2fs（%d 帧）· %s" % (
        len(records), total / float(FPS), total, chain["dir"])


def register_with_extension(ext):
    return [H3DirectorNode]
