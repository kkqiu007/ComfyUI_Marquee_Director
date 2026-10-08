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
The session manifest on disk is unchanged by all of this.

.. note::
   2026-09-20 —— ``H3ScriptBatchRender`` / ``H3ScriptRepairSegment`` /
   ``H3ChainToVideo`` / ``H3LoadSession`` / ``H3SegmentTimeline`` /
   ``H3ShotPrompt`` / ``H3ShotsBoard`` / ``H3ShotRenderer`` 已**删除**
   （全部被本节点取代）。会话目录与 manifest 格式
   没变，所以旧的 output 目录照旧能读。
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
from . import wardrobe as wardrobe_mod
from .common import (
    FPS,
    evict_dead_loaded_models,
    generation_length,
    guide_length_up,
    log,
    seconds_to_frames,
)
from .engine import free_between_segments, unload_due, unload_every_int
# H3_REFINE 类型定义在这里进：Director 的 refine 口与 H3 Refine 节点的输出口
# 必须是同一个 io.Custom 实例，两边各建一个同名字符串也能跑，但那是两份真相。
from .refine_nodes import H3Refine
# 与 refine 同理：Director 的 face_refine 口和 H3 FaceRefine 节点的输出口必须是
# 同一个 io.Custom 实例。face_refine 模块已被 engine 在模块层导入，这里只是取类型。
from .face_refine import H3FaceRefine
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

# 2026-09-26 清理：闪电渲染 / 步数覆盖整套机制已删除。
#   2026-09-20 起节点上的 lightning / render_steps / scheduler / lightning_lora /
#   lightning_lora_strength 这五个面板控件被收起，调用点改成硬编码关闭值，
#   但 `_apply_lightning`（60 行）+ `_note_cache_bust` + 两个哨兵常量
#   （`FOLLOW_UPSTREAM` / `NO_LORA`）一直留在包里 —— 一行空转的代码却是全包
#   唯一还能改 sigmas / model 的暗门，读代码的人会以为它有入口。整套删掉。
#   要恢复这个功能：从 git 历史取回 `_apply_lightning`，并把五个控件加回
#   `H3DirectorNode.define_schema()` 的**末尾**（widget 顺序即 widgets_values
#   的对齐顺序，插在中间会让已存工作流的控件整体错位）。


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
# 单一来源同 MAX_REFERENCE_IMAGES：ref_images 模块。以前这里自己又写了一遍
# ``CANVAS_STRIDE = 32``，两处各改一半就会对不上（2026-09-20 收敛）。
CANVAS_STRIDE = ref_images_mod.CANVAS_STRIDE

REF_SIZE_MATCH = "match"
REF_SIZE_MAX = "max"          # 历史选项：显式关闭缩放（不在下拉里，但老工作流存过）
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
        # ★ 卸载（含"是否连内存一起清"的护栏）统一交给 engine.free_between_segments，
        #   这里只补一段收尾的 gc + soft_empty_cache。
        #   以前这里自己写了一套 mm.unload_all_models() + cleanup_models()，
        #   没有内存护栏 —— 链尾那一次卸载同样可能把机器拖进重装死锁。
        free_between_segments(unload_models)
        import comfy.model_management as mm

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


def _rebase_disk_progress(done_indices, done_count, start_index, total):
    """把**会话全局**段号换算成**本批次**的 1..total 段号。

    ``disk_segments()`` / ``disk_segment_indices()`` 报的是 seg_NN.mp4 里的 NN
    （会话全局），而面板时间线画的是本批次的分镜（1..total）。追加模式
    （chain_state 接着一个已有会话）下两者差一个 start_index 偏移：已有 4 段的
    会话再追加 4 段，全局段号是 5..8，面板上却只有 1..4。直接下发会让前端
    doneSet 塞进 5/6/7/8 —— 面板上一段都不存在，进度条瞬间 100%、轨道一段都
    不亮、小马也无段可跑。

    ``start_index == 0``（普通断点渲染）是恒等变换。返回 ``(indices, count)``。
    """
    indices = sorted(int(i) for i in (done_indices or []))
    count = int(done_count or 0)
    start = int(start_index or 0)
    total = int(total or 0)
    if not start:
        return indices, count
    return ([i - start for i in indices if start < i <= start + total],
            max(0, min(total, count - start)))


def _why_not_reused(sess, manifest, index, plan, prompt, resolved_seed):
    """断点续渲：这一段**为什么**不会被复用？返回一句人话，或 None（应当能复用）。

    ★★ 只用**已经记录在 manifest 里的事实**做对账，不重算缓存键（2026-09-20 第 44 轮）。
       缓存键是 prompt / 参考图 digest / 上一段 key / settings 的哈希，离线复刻不了；
       猜错会让面板撒谎，比不报更糟。而下面这四项**都是缓存键的成分**，
       任何一项对不上，键就必然不同 —— 所以结论可靠：只可能漏报，不会误报。

    为什么需要它：渲染节点只在**命中**时打 `-- reusing seg_NN.mp4`，没命中时日志里
    只有一行普通的 `segment N: ...`。于是「断点渲染没从已有的继续」在日志里完全
    看不出原因 —— 实测那次是 `ref_image_size` 读到非法值 1，参考图被压成 32×32，
    参考图 digest 全变 → 键永远对不上。现在这一段会直接把差异列出来。
    """
    try:
        path = sess.segment_path(index)
        if not os.path.exists(path):
            return "磁盘上没有 %s" % os.path.basename(path)
        records = (manifest or {}).get("segments") or []
        if index >= len(records):
            return "manifest 里没有第 %d 段的记录" % (index + 1)
        rec = records[index]
        if not isinstance(rec, dict) or not rec:
            # 记录位置存在但是空的（手改过 manifest / 老版本残留）——
            # 当成"没有记录"报，别把 None 当成"段长 None"去跟 277 比。
            return "manifest 第 %d 段的记录是空的" % (index + 1)
        if rec.get("recovered"):
            # 磁盘补齐的记录没有原始键，只能靠"链条是否还完整"判定
            return None if sess.resume_intact else (
                "manifest 缺第 %d 段的原始缓存键（靠磁盘补齐），"
                "而本次已经重渲过更早的段 —— 链条断了，后面的段必须跟着重渲"
                % (index + 1))
        diffs = []
        if rec.get("length") != plan["frames"]:
            diffs.append("段长 %s → %s 帧" % (rec.get("length"), plan["frames"]))
        if rec.get("handoff") != plan["tail_frames"]:
            diffs.append("尾部锚点 %s → %s 帧"
                         % (rec.get("handoff"), plan["tail_frames"]))
        # ★ 2026-09-30：修复时主动换过种子的段（``reseeded``）不再按种子对账 ——
        #   否则「单段修复」在整链重跑时会被判成缓存失效，把刚修好的段又重渲一遍，
        #   等于修复白做（而且重渲回去的还是同一个坏结果）。
        if not rec.get("reseeded") and rec.get("seed") != resolved_seed:
            diffs.append("种子 %s → %s" % (rec.get("seed"), resolved_seed))
        old_prompt = str(rec.get("prompt") or "")
        if old_prompt and old_prompt.strip() != str(prompt or "").strip():
            diffs.append("提示词变了（%d → %d 字符）"
                         % (len(old_prompt), len(str(prompt or ""))))
        if diffs:
            return ("缓存键对不上：" + "；".join(diffs)
                    + "（参考图内容/尺寸、画布宽高、采样器等变化也会走这条）")
        return None
    except Exception as exc:
        # ★ 不能静默 return None。本函数的约定是「返回 None = 应当能复用」，
        #   静默兜底等于把"**对账失败**"说成了"**可以复用**" —— 与它
        #   「只可能漏报、不会误报」的设计目的正好相反，用户会看到一段
        #   本该重渲的段被当成可复用。
        log("segment %d: 缓存对账失败（%s）—— 判不出为什么没复用，按不复用处理",
            index + 1, exc)
        return None


def _repair_reseed(settings, index):
    """单段修复时给该段换一个种子（LCG 跳变，与 ``subtitle_qc`` 同式）。

    ★ 2026-09-30 为什么必须有它：
      段种子是 ``_resolve_seed`` 的纯函数 —— ``chain_seed + (index+1)*9973``。
      同一个 ``chain_seed`` 下，「单段修复」会**逐比特复现**上一次的采样结果，
      **包括它本来要修的那个瑕疵**。实测：v25sub 的 seg_04 在 37.75–38.25s
      出现约 0.5s 的人脸拖影花屏（帧 906–918，正落在该段尾部锚点窗内），
      而修复路径原本传的 ``seed_override=0`` ⇒ 再修一次还是同一张花屏。
      用户当时唯一的出路是把 ``chain_seed`` 整个换掉 —— 那会**连累全部 5 段**
      重新抽签（每段都可能抽出新的瑕疵），代价 81 分钟而不是 16 分钟。

      跳变用 ``old * 1103515245 + 12345``（不是加常数）—— 同 ``subtitle_qc``：
      实测加常数会产生高度相关的样本，乘法跳变去相关更有效。
      换过的种子由修复节点记进 manifest（``reseeded: true``），
      于是整链重跑时该段仍被判定为可复用，修复不会白做。
    """
    base = _resolve_seed(settings, {"seed": 0}, index)
    return (base * 1103515245 + 12345) % (1 << 63)


def _ref_edge_limit(mode, settings):
    """Long-edge cap for reference images, or None when there is nothing to fit.

    ``match`` follows the render canvas (a reference bigger than the output buys
    nothing but VRAM); the presets are plain pixel caps. ``None`` means
    "downscale is off" — the 32px snap below still runs.

    ★★ 非法值必须回落到 ``match``，**绝不能**按"抽出数字"硬解析（2026-09-20 实测）：
       历史工作流的 ``widgets_values`` 一旦错位（见 skill 坑 49），
       ``ref_image_size`` 会读到别人的值 ``1`` —— 而旧实现会把 ``str(1)`` 里的
       数字抽出来当成长边上限，于是 ``max_edge = 1``：

         · 参考图被压成 **32×32**，人物/场景参考等于没有，画质直接崩；
         · 参考图的 ``digest`` 全变 → **缓存键永远对不上** →
           「断点渲染」每次都从头重渲，**日志里一条 ``-- reusing`` 都没有**。

       实测对照：09-15 / 09-19 该值为 ``match`` 时缓存能命中（3 段复用）；
       09-20 全天该值错位成 ``1``，缓存命中数 = 0。
       所以这里改成**只认声明过的选项**，其余一律回落 ``match`` 并打警告 ——
       宁可退回默认行为，也不要静默把图压成 1px、把缓存全部作废。
    """
    # 数字形态先归一化：JSON 往返会把 "512" 变成 512.0，布尔 True 会变成 "true"。
    # 归一化只为了让**合法预设**能被认出来；认不出的照旧回落 match。
    if isinstance(mode, bool):
        text = str(mode).lower()
    elif isinstance(mode, (int, float)) and float(mode).is_integer():
        text = str(int(mode))
    else:
        text = str(mode if mode is not None else REF_SIZE_MATCH).strip().lower()
    if text == REF_SIZE_MATCH:
        return _canvas_ref_limit(settings)
    if text == REF_SIZE_MAX:
        return None                      # 历史选项：显式关闭缩放
    if text in [opt.lower() for opt in REF_SIZE_OPTIONS]:
        return int(text)                 # 512 / 768 / 1024 / 1536
    log("ref_image_size=%r 不是合法预设 %s —— 已按 %r 处理。"
        "（继续按旧逻辑抽数字会把它当成 1px 上限：参考图被压成 32×32，"
        "且缓存键与已渲段对不上，断点渲染会一直从头重渲）",
        mode, REF_SIZE_OPTIONS, REF_SIZE_MATCH)
    return _canvas_ref_limit(settings)


def _canvas_ref_limit(settings):
    """``match`` 口径：跟随渲染画布的长边；取不到就返回 None（不缩放）。"""
    try:
        return (max(int(settings.get("width") or 0),
                    int(settings.get("height") or 0)) or None)
    except (TypeError, ValueError):
        return None


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
                "`refine` takes an optional H3 Refine node. When it is wired "
                "every segment is sampled a second time (a denoise<1 schedule) "
                "before it is decoded, and the handoff tail is cut from that "
                "refined result — so the next segment anchors on refined frames "
                "and no quality step lands at a join. Unwired (or wired without "
                "a sigma schedule) one sample per segment, exactly as before."
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
                io.Int.Input(
                    "unload_every", display_name="每 N 段卸载模型",
                    default=2, min=0, max=99, step=1, optional=True, advanced=True,
                    tooltip="每渲染满 N 段卸载一次模型，并清显存。\n"
                            "0 = 从不卸载；1 = 每段都卸（旧「每段后卸载模型」的行为）；"
                            "2 = 每两段卸一次（默认）。\n"
                            "计数器只数真烧了显卡的段，命中磁盘缓存的段不计数；"
                            "链尾无条件卸一次，给拼接/编码让出显存。\n"
                            "★ 卸载会把内存里的权重副本一起丢掉，下一段要从磁盘重载"
                            "（UNet ~20GB + 文本编码器 ~26GB）。内存装不下这套权重时"
                            "（例如 40GB 内存）务必设 0 —— 实测会在重载时 100% CPU 空转卡死。"
                            "设为 0 时每段仍做 gc / 死槽清理 / 清显存，只是不丢权重。"),
                io.Float.Input(
                    "crf", default=14.0, min=0.0, max=51.0, step=1.0, optional=True,
                    advanced=True,
                    tooltip="拼接成片的质量；0=无损，越低越大。"),
                io.Float.Input(
                    "stabilize", display_name="亮度稳定", default=0.0, min=0.0,
                    max=1.0, step=0.05, round=False, optional=True,
                    advanced=True,
                    # 2026-09-20：这段说明原本写在 nodes.STABILIZE_TOOLTIP，
                    # 旧节点下线后成了没人引用的孤儿，Director 只剩一行占位。
                    # 回收到这里（译中文）：限幅、有效区间、和 drift_arrest 的
                    # 分工这三件事只有这儿写着，删了就查不到了。
                    tooltip="把整条成片缓慢偏离自身开场的亮度漂移压平；0 = 关。\n\n"
                            "多段拼接会一路变暗、掉饱和，但**单看任何一处接缝都看不出来**"
                            "——漂移按定义就是慢的，所以能干净地分离出来：每通道的逐帧"
                            "均值做两秒平滑，剩下的就是漂移。校正用的是**增益**（增益不改"
                            "黑位），目标对齐第 1 段的水平；它来自一条平滑曲线，所以跨接缝"
                            "连续，稳定化本身不可能在切点引入台阶。\n\n"
                            "增益限幅 ±25%：唯一和漂移分不清的，是某个镜头因为真实光线变化"
                            "而确实变暗。真实变化幅度大、留得下来；累积漂移幅度小、被限掉。\n\n"
                            "0.7–1.0 是有效区间。代价是多一次解码，不烧显卡。"
                            "**只修颜色** —— 人物/身份漂移要用 H3 Chain Settings 的 "
                            "drift_arrest，那个作用在生成阶段，不是成片。"),
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
                # 选择性渲染：只重渲指定段，其余段完全复用磁盘缓存（不重新烧显卡）。
                # 2026-09-19 起走 H3RepairSegmentNode 的双端锚定，因此可以真正
                # "任意挑几段"，不必再退化成「从最早选中段渲到结尾」。
                # 与单段修复互斥：两者同时设时单段修复优先。
                io.String.Input(
                    "run_segments", display_name="选择渲染段（空=全部）", default="",
                    multiline=False, optional=True, advanced=True,
                    tooltip="只重渲这些段，其余段原样复用磁盘缓存并照常参与拼接。"
                            "格式：逗号分隔的段号与区间，如 3 或 1,3,5-7。留空 = 全部渲染。"
                            "每一段都用首尾双端锚定（段首钉上一段的 handoff、段尾钉住"
                            "下一段已在用的那一帧），所以未选中的段一个都不用动，"
                            "不会出现接缝跳变。"
                            "前提：该会话已整条渲过一次（磁盘上要有相邻段的产物）。"
                            "（与单段修复 repair_segment 是同一条通道，只是支持多段。）"),
                # 二采精修：外接可选节点（借鉴 ComfyUI_MiniMaxH3_Director 的
                # refine 口）。不接 = 一采即成片，与加这个口之前完全一样。
                # ★ 必须挂在**列表末尾**：已存工作流按 slot 下标存连线，插在
                #   中间会把既有连线整体带偏一格。
                H3Refine.Input(
                    "refine", display_name="二采精修", optional=True,
                    tooltip="接「H3 Refine（二采精修）」节点的 refine 输出。"
                            "接上后每一段在一采之后、解码之前再采一遍（噪声表由"
                            "那个节点的 sigmas 口给），成片与段间锚点都来自二采"
                            "结果，所以接缝不会出现画质跳变。\n"
                            "不接 = 不二采，行为与以前一致。"),
                # 剧本来源 / 源视频路径：execute 一直读这两个参数，但 2026-09-20
                # 收敛节点集时把它们从 schema 里删掉了，于是 mode 恒为
                # 「分镜JSON」—— 源视频自动切分那整条链路（PySceneDetect 切镜头）
                # 连同 video 输入口一起变成了永远进不去的死代码。这里补回控件。
                # 同样挂在末尾，理由同上（widget 顺序 = 已存工作流的值顺序）。
                io.Combo.Input(
                    "source", display_name="剧本来源", options=SOURCE_OPTIONS,
                    default=SOURCE_JSON, optional=True,
                    tooltip="分镜JSON = 直接消费 shots_json（Ref2VA-Auto 兼容载荷）。\n"
                            "源视频自动切分 = 接一个源视频到 video（或在下面填绝对路径），"
                            "用 PySceneDetect 找硬切点，每个镜头渲一段。\n"
                            "没装 scenedetect 时退化成按时长均分。"),
                io.String.Input(
                    "video_path", display_name="源视频路径", default="",
                    multiline=False, optional=True, advanced=True,
                    tooltip="「源视频自动切分」模式下使用：源视频的绝对路径。"
                            "接了 video 输入口就不需要填。"),
                # 脸部修复：外接可选节点，与 refine 同一个契约。接上但 sigmas 口
                # 空着 = 不修脸。同样挂在末尾（widget 顺序 = 已存工作流的值顺序）。
                H3FaceRefine.Input(
                    "face_refine", display_name="脸部修复", optional=True,
                    tooltip="接「H3 FaceRefine（脸部修复）」节点的 face_refine 输出。"
                            "接上后每段解码成片会逐帧检测并跟踪人脸，裁成特写画布重采"
                            "一次再贴回；段首/段尾会把修脸结果淡回原图，避免接缝处"
                            "「脸突然变清楚」的跳变。\n"
                            "★ 那个节点自己不接 sigmas = 关闭修脸。\n"
                            "★ 检测不到脸的段原样放过，不报错也不改画面。\n"
                            "需要 ultralytics + models/ultralytics/bbox/ 下的 YOLO 权重。"),
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
                unload_every=2, crf=14.0, stabilize=0.0,
                build_timeline=True, export_segments=False, ref_images=None,
                ref_image_size=REF_SIZE_MATCH, pack_info="",
                ref_classify="", repair_segment=0, run_segments="",
                refine=None, face_refine=None):
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
        # ★ 上下游会话名对账（2026-09-26）：包里有**两个**同名控件 ——
        #   H3 Chain Settings 的 session_name，与 H3 分镜提示词 PACK 的 session_name。
        #   真正生效的是 PACK 那一个：解析器把名字写进 shots_json，Director 从
        #   JSON 里读（`_session_name_from` 的 fallback 参数恒为空，因为 H3Director
        #   自己不暴露 session_name 控件）。实测三个名字互不相同
        #   （Chain Settings=h3_3070_5shot_v2、PACK 控件=…_v10_iter5、
        #   实际落盘=…_v10_iter6）却全程静默 —— 用户会以为改 Chain Settings 就能
        #   换会话，结果断点续渲照样接在 PACK 那个目录上。分歧必须说出来。
        _declared_session = str((settings or {}).get("session_name") or "").strip()
        settings = dict(settings, session_name=session_mod.sanitize(sess_name))
        session_name_note = ""
        if (_declared_session
                and session_mod.sanitize(_declared_session) != settings["session_name"]):
            session_name_note = (
                "会话名不一致：H3 Chain Settings 写的是「%s」，本次实际使用「%s」"
                "（取自 PACK 头部 Project / 分镜 JSON 的 session_name）。"
                "要以 Chain Settings 为准，请改 PACK 头部的 Project。"
                % (_declared_session, settings["session_name"]))
            log("%s", session_name_note)
        # 二采精修（外接 H3 Refine）。只在**接了并且给了噪声表**时才并进
        # settings —— 不并的话 segment_key 的 payload 与旧版字节一致，既有会话
        # 不会因为多了这个功能而全量重渲；并了之后改任何二采参数都会正确重渲。
        # 二采发生在每段一采之后、解码之前，所以段间锚点与成片同源（见
        # engine.render_segment）——接缝不会出现画质跳变。
        if isinstance(refine, dict) and refine.get("sigmas") is not None:
            settings = dict(settings, refine=refine)
        # 同 refine 的规则：只在「真的接了且给了 sigmas」时才并进 settings。
        # 没接 / 接了没给噪声表 -> settings 里没有 face_refine 这个键，
        # apply_face_refine 直接原样返回，缓存键也一字不变。
        if isinstance(face_refine, dict) and face_refine.get("sigmas") is not None:
            settings = dict(settings, face_refine=face_refine)

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
        if session_name_note:
            warnings.append(session_name_note)
        if len(raw_slots) > MAX_REFERENCE_IMAGES:
            # collect_slots 已经按槽号裁到 9 了，这里只是兜底提示。
            warnings.append("参考图接了 %d 张，超过 H3 上限 %d 张，多余的会被忽略"
                            % (len(raw_slots), MAX_REFERENCE_IMAGES))
        pool_labels = _pool_labels(raw_slots, ref_classify)
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
        # (run_from 已废弃：2026-09-19 起「选择渲染段」走双端锚定，逐段只渲选中段，
        #  不再需要「最早选中段」这个下界。接线仍在，只是不再参与判定。)

        # ★ 开工先自报口径。这一行是用户**唯一**能确认「开关到底生效没有」的依据：
        #   unload_every 是可选 widget，老工作流不带这个键时会静默落到 schema 默认值，
        #   日志里不写出来就只能靠猜 —— 2026-09-19 正因缺这一行白查了一整轮。
        #   同时把每段判决也打出来（见 H3RenderSegmentNode.execute），
        #   于是日志里能直接数出「留卸留卸」的节奏。
        _every_n = unload_every_int(unload_every)
        log("unload_every=%d（%s）；本次共 %d 段", _every_n,
            "从不卸载" if _every_n <= 0
            else ("每段都卸" if _every_n == 1 else "每 %d 段卸一次" % _every_n),
            total)

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

        # ★★ 段号口径换算：上面两个都是**会话全局**段号（seg_NN.mp4 里的 NN），
        #    而面板的时间线画的是**本批次**的分镜（1..total）。追加模式
        #    （chain_state 接着一个已有会话）下两者差一个 start_index 的偏移。
        #    换算逻辑见 _rebase_disk_progress（抽成纯函数以便单元测试）。
        done_indices, done_count = _rebase_disk_progress(
            done_indices, done_count, start_index, total)

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

        # ---- 修复 / 选择性重渲模式：跳过整条循环，只重渲指定段 ----
        # 两个入口共用同一条通道：
        #   repair_segment = N     只修第 N 段（老行为）
        #   run_segments = "1,3"   只重渲第 1、3 段（2026-09-19 起，任意挑几段）
        # 都走 H3RepairSegmentNode 的**首尾双端锚定**：段首钉住上一段的 handoff，
        # 段尾钉住下一段已经在用的那一帧。所以没被选中的段一个都不用动，也就不存在
        # 「重渲第 N 段必然连累其后所有段」的接缝问题 —— 这正是不再退化成
        # 「从最早选中段渲到结尾」的依据（老口径见 git 历史与 README 的说明）。
        # 互斥：两者同时设时单段修复优先。越界/空 = 走下方整条渲染。
        repair_mode = bool(repair_segment) and 1 <= int(repair_segment) <= len(shots)
        if repair_mode:
            repair_indices = [int(repair_segment) - 1]
        elif run_mode:
            repair_indices = sorted(i - 1 for i in run_set)
        else:
            repair_indices = None

        if repair_indices is not None:
            rows = []
            handoff_overrides = []
            if run_mode:
                log("选择性重渲：第 %s 段（共 %d 段）—— 逐段双端锚定，"
                    "未选中的段原样复用磁盘缓存",
                    ",".join(str(i + 1) for i in repair_indices), len(repair_indices))
            for ridx in repair_indices:
                rshot = shots[ridx]
                r_override = _shot_ref_images(rshot, ref_pool, warnings)
                r_prompt, r_seconds, r_images = _script_segment(
                    asset, rshot, ref_images_override=r_override)
                r_prompt, r_wnotes = wardrobe_mod.process_prompt(r_prompt)
                for note in r_wnotes:
                    log("wardrobe[S%02d]: %s", ridx + 1, note)
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
                # ★ 传 r_plan["gen_seconds"]（= 新增 + 回放），不是 r_seconds。
                #   修复节点内部按 ``generation_length(seconds_to_frames(seconds))``
                #   定段长；传 r_seconds（只含新增）会让修复段比正常渲染**少一整个
                #   回放窗（39 帧）**，而拼接仍按 39 帧裁 —— 等于吃掉 39 帧正片，
                #   而且这一段的内部时间轴与剧本写的时间码整体错位。
                #   正常路径传的就是 plan["gen_seconds"]（见下方渲染调用），
                #   两条路径必须同口径。
                # ★ 2026-09-30：修复必须换种子，否则逐比特复现上一次（含要修的瑕疵）。
                #   理由与取式见 `_repair_reseed`。
                _rep_seed = _repair_reseed(settings, ridx)
                log("修复第 %d 段：换种子 %d → %d（否则会复现同一结果）",
                    ridx + 1, _resolve_seed(settings, {"seed": 0}, ridx), _rep_seed)
                _rep_video, rep_chain, _rep_summary = H3RepairSegmentNode.execute(
                    settings, sess_name, ridx + 1, r_prompt, r_plan["gen_seconds"],
                    r_tail, _rep_seed, True,
                    images=_images_autogrow(r_images), chain_state=chain_state)
                routes_mod.emit_progress(
                    node_id, event="segment_done", index=ridx + 1, total=total,
                    shot_id=rshot.get("id") or ("S%02d" % (ridx + 1)),
                    frames=r_plan["frames"], elapsed=0, session=sess_name)
                chain = rep_chain
                rows.append({
                    "no": ridx + 1, "new_seconds": r_plan["new_seconds"],
                    "requested_frames": r_plan["requested_frames"],
                    "frames": r_plan["frames"],
                    "handoff_frames": r_plan["handoff_frames"],
                    "cumulative": r_plan["new_seconds"], "elapsed": 0.0,
                    "task": _infer_task(r_images), "refs": len(r_images),
                    "status": "已修复",
                })
            # 累计列要的是**整条片子**的总长，不是只渲的那几段的长度
            # （以前这一列恒为 0.00：上面算完又被后面的 `cumulative = 0.0` 冲掉了）。
            # 按全部分镜重算一遍，纯算术，不碰显卡。
            cumulative = 0.0
            for _i, _shot in enumerate(shots):
                _p, _sec, _img = _script_segment(asset, _shot)
                _rp, _tl = _shot_windows(_shot, _i, total, handoff_seconds)
                cumulative += _shot_plan(_sec, _rp, _tl,
                                         duration_is_new_content)["new_seconds"]
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
            # 2026-09-19 起「选择渲染段」是真的只渲选中段（逐段双端锚定），
            # 不再是「从最早选中段渲到结尾」——文案必须跟着改，否则报告自相矛盾。
            header.append(
                "选择渲染 只重渲第 %s 段（双端锚定）· 其余 %d 段原样复用磁盘缓存"
                % (",".join(str(i) for i in sorted(run_set)), total - len(run_set)))
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
        # 二采进报告：它改变的是成片本身，报告里不说就等于让用户去猜画质从哪来。
        if isinstance(settings.get("refine"), dict):
            _ref = settings["refine"]
            header.append("二采   %d 遍 · %s · %s"
                          % (int(_ref.get("passes") or 1),
                             "跟随一采模型" if _ref.get("model") is None else "二采模型",
                             _ref.get("seed_mode") or "跟随一采"))
        # 修脸同样进报告：它改的是成片像素，而且「检测到脸才修」—— 报告里不说，
        # 用户看到一部分段被修了、一部分没修，只会以为是 bug。
        if isinstance(settings.get("face_refine"), dict):
            _fr = settings["face_refine"]
            header.append("修脸   %s · %s · 阈值 %.2f · %s"
                          % (_fr.get("detector") or "?",
                             _fr.get("paste_region") or "face_only",
                             float(_fr.get("confidence") or 0.0),
                             "接缝淡出" if str(_fr.get("seam_fade")) != "关闭"
                             else "整段修（接缝可能跳）"))

        # 修复 / 选择性重渲上方已经把 rows / cumulative / handoff_overrides 填好了，
        # 这里不要再把它们重置成 0（以前单段修复的累计列因此恒为 0.00）。
        if repair_indices is None:
            cumulative = 0.0
            handoff_overrides = []
        # ★ 本次整条渲染里，有几段是**复用磁盘缓存**、几段是**真烧了显卡**
        #   （2026-09-20 第 44 轮）。以前只有日志里零星的 `-- reusing` 能看出来，
        #   面板完全不知道 —— 用户报「断点渲染没从已有的继续渲染」时，
    #   没有任何一处能回答"到底复用了没有"。现在随 finish 事件一起下发。
        reused_count = 0
        rendered_count = 0
        pbar = comfy.utils.ProgressBar(total, node_id=node_id)

        for index, shot in enumerate(shots):
            if repair_indices is not None:
                break   # 修复 / 选择性重渲已在上面逐段处理完，跳过整条循环
            shot_no = start_index + index + 1
            override = _shot_ref_images(shot, ref_pool, warnings)
            prompt, seconds, images = _script_segment(
                asset, shot, ref_images_override=override)
            prompt, wnotes = wardrobe_mod.process_prompt(prompt)
            for note in wnotes:
                log("wardrobe[S%02d]: %s", shot_no, note)
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


            # 2026-09-19：run_mode（选择渲染段）已经在上方走**双端锚定**逐段处理完，
            # 走到这里时 repair_indices 必为 None —— 也就是只剩下“整条渲染”这一种
            # 情况，所以 resume_eff 直接按整条渲染取值。
            # 以前这里是「从最早选中段起重渲、之前复用缓存」的旧口径（selected /
            # run_from），那段逻辑会让 "1,3" 退化成 1→结尾，现已废弃
            # （见 repair_indices 分支的注释）。
            resume_eff = bool(resume)
            started = time.time()
            # ★ 断点续渲的"为什么没复用"要说清楚（2026-09-20 第 44 轮）。
            #   渲染节点内部才知道缓存到底命中没有，而它只在**命中**时打
            #   `-- reusing seg_NN.mp4`；没命中时日志里只有一行普通的
            #   `segment N: ...`，用户完全看不出"为什么这一段没被复用"。
            #   这里在开渲之前，用**已经记录在 manifest 里的事实**做一次结构化对账：
            #   段长 / 回放帧 / 种子 / 提示词，任何一项与本次规划不同，就说明
            #   入参变了、缓存键必然不同。**只报已证实的差异，不猜哈希。**
            _resume_note = _why_not_reused(
                sess, manifest, start_index + index, plan, prompt,
                _resolve_seed(settings, {"seed": 0}, start_index + index))
            if _resume_note:
                log("segment %d: 本次不复用 —— %s", index + 1, _resume_note)
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
                task=task, refs=len(images), session=sess_name,
                resume_note=_resume_note or "")
            # 第 5 个参数是**尾部锚点**的长度（渲完从结果里切多长给下一段），
            # 不是生成时长 —— 生成时长是第 4 个参数，已经把 replay_in 加进去了。
            # 以前这里传的是回放长度，于是首段被要求"切一条 1.6s 的尾巴"却没有
            # 对应的额外素材，末段则被判成不留锚点。
            routes_mod.set_phase(node_id, "条件编码 → 采样 → AV 解码 → 质检",
                                 index=index + 1, total=total)
            output = H3RenderSegmentNode.execute(
                settings, resume_eff, prompt, plan["gen_seconds"],
                plan["tail_out"], 0,
                unload_every, chain_state=chain,
                images=_images_autogrow(images),
            )
            last_video_path = output[0]
            chain = output[1]
            # ★ 如实统计这一段是"复用"还是"重渲"（2026-09-20 第 44 轮）。
            #   渲染节点把结果放在 chain["reused"] 里（它内部才知道）。
            #   没有这个键（老版本 / 单节点串联）时按"重渲"计 —— 宁可少报复用，
            #   也不要谎报"复用了"然后用户去核对时发现画面变了。
            if chain.get("reused") is True:
                reused_count += 1
                log("segment %d: 复用磁盘缓存（未烧显卡）", index + 1)
            else:
                rendered_count += 1
            if hasattr(last_video_path, "get_stream_source"):
                src = last_video_path.get_stream_source()
                if isinstance(src, str):
                    last_video_path = src
            # Drop the reference cycle so the next segment rebuilds anchors, then
            # sweep whatever the sampler left behind (Comfy leaves dead
            # LoadedModel slots around after every H3 segment).
            # ★ 卸载模型这件事**已经不在这里**：它由上面那次 H3RenderSegmentNode.execute
            #   内部按「每 N 段卸载一次」判定并执行（计数器挂在 chain_state 上，两条路径
            #   同一套语义）。这里只做不需要卸模型的那半截卫生工作，避免重复卸载
            #   ——以前两处都按同一个布尔卸，等于卸两遍。
            free_between_segments(False)
            cleanup_segment_vram(False)
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
                "status": "完成",
            })
            routes_mod.emit_progress(
                node_id, event="segment_done", index=index + 1, total=total,
                shot_id=shot.get("id") or ("S%02d" % (index + 1)),
                frames=plan["frames"], elapsed=round(time.time() - started, 2),
                session=sess_name)
            pbar.update(index + 1)

        # 链尾释放：unload_every>0 时，整条链渲完无条件卸一次模型。
        # 这一条是旧行为的保留项 —— 以前每段都卸，末段自然也卸，而末段之后紧接着
        # 就是拼接 / 编码，显存留给它更稳。unload_every=0（从不卸载）时连这里也不卸。
        if unload_due(1, unload_every, is_last=True):
            log("H3Director: 链尾释放 —— 卸载模型并清显存（unload_every=%d）",
                unload_every_int(unload_every))
            free_between_segments(True)
            cleanup_segment_vram(True)
        else:
            log("H3Director: 链尾不释放 —— unload_every=0，模型交给 ComfyUI 自行回收")

        if handoff_overrides:
            uniq = sorted({round(v, 3) for _, v in handoff_overrides})
            warnings.append(
                "逐镜回放：%d 个分镜用自己的 %s，未跟随全局 %.3fs"
                % (len(handoff_overrides),
                   " / ".join("%.3fs" % v for v in uniq), float(handoff_seconds)))

        # 3. Join into one cut (same H3ChainToVideo helper) --------------------
        _phase_text(cls, _PHASE_LABELS["join"])
        video_out = VideoFromFile("")
        if chain is not None and chain.get("segments"):
            # ★ 2026-09-28 P2 修复：repair/resume 写回 manifest 时
            #   session.reconcile() 因 .tail 被 cleanup 删除只能写 handoff=0，
            #   _join_parts 就不裁 39 帧回放 → 每道接缝重复 39 帧。
            #   join 前用 shots_info 权威 handoff_seconds 回填。
            for _rec in chain["segments"]:
                _ri = int(_rec.get("index", -1))
                if int(_rec.get("handoff") or 0) > 0 or _ri < 0 or _ri >= len(shots):
                    continue
                try:
                    _sec = float(shots[_ri].get("handoff_seconds") or 0.0)
                except (TypeError, ValueError):
                    _sec = 0.0
                if _sec > 0.5:
                    _rec["handoff"] = generation_length(seconds_to_frames(_sec))
                    log("P2 handoff-stamp: seg %d handoff 0 -> %d frames (shots=%.3fs)",
                        _ri + 1, _rec["handoff"], _sec)
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
        # 断点续渲到底复用了没有 —— 放进报告，别再让人去翻日志（第 44 轮）
        header.append("缓存   复用 %d 段 / 重渲 %d 段（共 %d 段）"
                      % (reused_count, rendered_count, total))
        if resume and reused_count == 0 and total > 1:
            header.append("注意   本次一段都没复用。若磁盘上本来就有已渲段，"
                          "往上翻 `本次不复用 —— …` 那几行，那里写了每一段"
                          "缓存键对不上的具体差异。")

        report = _format_report(rows, header + _pack_info_block(info),
                             warnings, ref_rows)
        _phase_text(cls, "%s：%s" % (_PHASE_LABELS["finish"], progress))
        routes_mod.set_phase(node_id, None)
        routes_mod.emit_progress(
            node_id, event="finish", total=total, session=sess_name,
            reused_count=reused_count, rendered_count=rendered_count)
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
