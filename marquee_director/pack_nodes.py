# -*- coding: utf-8 -*-
"""MiniMax H3 分镜提示词 → shots_json。

按 ``minimax-h3-shot-segment_en_SKILL.md``（**英文单版**）解析：一个 txt 里
只有英文版六段，段标题 ``########## S01 / 10s / EN ##########``、接续段
``########## S02 / 10s+1.6=11.6 / EN ##########``，结尾 ``END OF PROMPTS``；
没有 PACK 头部，也没有 ``[1] 中文版`` / ``[2] 英文版`` 分块。

旧的双语 PACK（头部 + 两版分块）**仍然能解析**，向后兼容。

设计要点
--------
* **零样板的直通模式**：``pack_text`` 留空时，节点原样转发 ``shots_json``。
  这样工作流里只需要 **一个** 输入节点，两条路线（手写 JSON / 分镜提示词）
  靠"填没填"自动分流，不用再摆 BOOL 开关和 LazySwitch。
* **逐镜解析**：段里出现的 ``<Picture N>`` 编号写进 ``shot["ref_images"]``，
  由 Director 从「参考图覆盖」槽位里按号取图。
* **提示词层规范化**：六段字段补全并按官方顺序拼回、no-text 首句自动补、
  接续段自动补 1.6 秒回放句、配乐按 1.5 判据自动套 MUSIC block（不默认
  N/A）、任务模式（t2v/i2v/fl2v/r2v/v2v/rv2v）自动推断。

Borrowed from ComfyUI_MiniMaxH3_Director
----------------------------------------
* ``lib/task_modes``  — 任务模式集合与推断口径
* ``lib/task_prompts`` — 官方六段字段与固定顺序
"""

from __future__ import annotations

import json
import re

from comfy_api.latest import io

from . import prompt_pack as pp
from .common import log, safe_session_name

CATEGORY = "ComfyUI_Marquee_Director"

LANG_OPTIONS = ["auto", "中文版", "英文版"]
ISSUE_OPTIONS = ["warn", "error"]

PACK_TOOLTIP = (
    "粘贴整份英文版分镜提示词（只要英文六段，形如\n"
    "########## S01 / 10s / EN ##########\n"
    "########## S02 / 10s+1.6=11.6 / EN ##########\n"
    "结尾 END OF PROMPTS）。旧的「===== 头部 + [1] 中文版 / [2] 英文版」"
    "双语 PACK 也照样吃。留空则原样转发 shots_json —— 一个节点同时吃"
    "「手写 JSON」和「分镜提示词」两条路线。"
)

# 会话名算法搬到 common 了 —— PACK 解析器、Director、/h3/pack_preview
# 三处必须算出同一个结果，否则面板查的会话和落盘的会话对不上（时间线空）。
_safe_session_name = safe_session_name


def _count_shots(text):
    """Best-effort shot count of an already-built shots_json."""
    try:
        data = json.loads(str(text or ""))
    except (TypeError, ValueError):
        return 0
    if isinstance(data, dict):
        shots = data.get("shots_info")
        if isinstance(shots, list):
            return len(shots)
    return 0


def _ref_slot_line(segments):
    """Which <Picture N> slots the whole PACK actually needs."""
    slots = []
    for seg in segments:
        for num in seg.get("pictures") or []:
            if num not in slots:
                slots.append(num)
    if not slots:
        return ""
    slots.sort()
    return "参考图 %s（共 %d 张）→ 按这个顺序接到 H3 Director 的「参考图覆盖」" % (
        "、".join(str(n) for n in slots), len(slots))


def _ref_slots_of_json(raw):
    """透传模式下也把参考图槽位算出来：从每个 shot 的 ref_images 取并集。"""
    try:
        data = json.loads(raw or "")
    except (TypeError, ValueError):
        return []
    shots = (data or {}).get("shots_info") or []
    slots = []
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        for item in shot.get("ref_images") or []:
            try:
                num = int(item)
            except (TypeError, ValueError):
                continue
            if num not in slots:
                slots.append(num)
    slots.sort()
    return slots


def _looks_like_pack(text) -> bool:
    """粗判一段文本是不是分镜 PACK（而不是被 widget 错位塞进来的普通控件值）。

    认两种形态：段标题 ``########## S01 / 10s / EN ##########``，或镜内标记
    ``[Shot 1]``。用于 ``pack_override`` 的健壮性守卫 —— 见 ``execute`` 里的注释。
    """
    s = str(text or "")
    if not s:
        return False
    return bool(re.search(r"#{3,}\s*S?\d+[A-Za-z]?\s*/", s)
                or re.search(r"\[\s*Shot\s*\d+\s*\]", s, re.I))


class H3PromptPackParser(io.ComfyNode):
    """PACK 文本 → shots_json（Direct H3 Director 输入）。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3PromptPackParser",
            display_name="H3 分镜提示词 PACK",
            category=CATEGORY,
            description="把规范格式的 H3 英文版分镜提示词解析成 shots_json，直接接 "
                        "H3 Director 的 shots_json 入口。自动做六段字段规范化、"
                        "no-text 首句补全、1.6 秒回放句补全、配乐 MUSIC block "
                        "自动判定、任务模式推断、<Picture N> 逐镜参考图解析。"
                        "pack_text 留空时原样转发已接入的 shots_json。",
            inputs=[
                # force_input：PACK 是几百行的长文本，走独立的多行文本框
                # （工作流里的 361）比塞进节点 widget 好用得多。
                io.String.Input(
                    "pack_text", display_name="PACK 提示词", default="",
                    force_input=True, optional=True,
                    tooltip=PACK_TOOLTIP),
                io.String.Input(
                    "shots_json", display_name="已有分镜 JSON", default="",
                    multiline=False, optional=True, force_input=True,
                    tooltip="上一路（手写 JSON / 文件加载）的结果。"
                            "pack_text 留空时原样透传，非空时被 PACK 结果覆盖。"),
                io.Combo.Input(
                    "language", display_name="取哪一版",
                    options=LANG_OPTIONS, default="auto", optional=True,
                    tooltip="只对旧的「双语 PACK」生效（英文单版本来就只有英文）。"
                            "auto = 英文版优先（官方六段字段是英文，H3 也吃英文）；"
                            "也可以强制取中文版或英文版。"),
                io.Float.Input(
                    "default_duration", display_name="缺省段长（秒）", default=10.0,
                    min=0.25, max=15.0, step=0.25, round=False, optional=True,
                    tooltip="段标题没写时长（如「10s+1.6=11.6」）时用这个值。"),
                io.Boolean.Input(
                    "auto_fix", display_name="自动修正", default=True,
                    optional=True, advanced=True,
                    tooltip="补全 no-text 首句、接续段的 1.6 秒回放句；命中配乐"
                            "判据时把 N/A 换成 MUSIC block。修过的项会在 report "
                            "里单列。"),
                io.String.Input(
                    "session_name", display_name="会话名", default="",
                    multiline=False, optional=True, advanced=True,
                    tooltip="留空则取 PACK 头部的 Project。"),
                io.Combo.Input(
                    "on_issue", display_name="发现问题",
                    options=ISSUE_OPTIONS, default="warn", optional=True,
                    advanced=True,
                    tooltip="warn = 写进 report 继续跑；error = 直接中断，"
                            "适合把 PACK 规范当硬约束用。"),
                # ★ pack_override 必须留在**最后**：ComfyUI 的 widgets_values 按
                #   widget 位置对齐，早于本输入保存的工作流只有前 5 个值。把它插在
                #   中间会让后续 widget 整体错位（on_issue 的 "warn" 被塞进
                #   pack_override，直接导致 PACK 解析报「没有解析到任何分镜」）。
                #   放末尾则旧工作流前 5 个值照常对齐，本项拿到默认空串。
                io.String.Input(
                    "pack_override", display_name="PACK覆盖(内部)",
                    default="", multiline=True, optional=True, advanced=True,
                    tooltip="由前端「分镜提示词编辑器」写入：非空时用它代替 "
                            "pack_text 参与解析（实时编辑结果直接覆盖 parser），"
                            "留空则照常读 pack_text。手动编辑请勿填。"),
            ],
            outputs=[
                io.String.Output("shots_json", display_name="分镜JSON"),
                io.String.Output("report", display_name="解析报告"),
                io.Int.Output("segment_count", display_name="段数"),
                io.String.Output(
                    "pack_info", display_name="PACK信息",
                    tooltip="JSON：头部（项目/模式/总时长/段数/画幅）+ 每段时长"
                            "+ 需要接的几个参考图槽位。接到 H3 Director 的 "
                            "pack_info 入口，Director 就能在自己的面板上显示"
                            "这份 PACK 的概要，并和画布尺寸对账。"),
            ],
        )

    @classmethod
    def execute(cls, pack_text=None, shots_json=None, language="auto",
                default_duration=10.0, auto_fix=True, session_name="",
                on_issue="warn", pack_override=None) -> io.NodeOutput:
        # 前端「实时编辑器」写入的覆盖优先：非空就以它代替外侧 pack_text 解析。
        # ★ 健壮性守卫：ComfyUI 的 widgets_values 是按 widget **位置**对齐的。
        #   pack_override 是后加到本 schema 的（插在 session_name 与 on_issue 之间），
        #   在它之前保存的工作流只有 5 个 widget 值，加载时位置整体错位 —— 会把
        #   on_issue 的 "warn" 塞进 pack_override。若照单全收，execute 就拿 "warn"
        #   去解析，整条链直接报「没有解析到任何分镜」。所以这里只接受「看起来像
        #   PACK」的覆盖值（含段标题 ########## S01 或镜内标记 [Shot N]），其余
        #   一律忽略、退回 pack_text。
        override = str(pack_override or "").strip()
        if override and not _looks_like_pack(override):
            override = ""
        text = override or str(pack_text or "").strip()
        incoming = str(shots_json or "").strip()

        # ---- Pass-through: no PACK pasted, forward whatever is wired --------
        if not text:
            if not incoming:
                raise ValueError(
                    "H3 分镜提示词 PACK：既没有粘贴 PACK 文本，也没有接入 "
                    "shots_json。请把 PACK 粘进 pack_text，或把分镜 JSON 接到 "
                    "shots_json 入口。")
            count = _count_shots(incoming)
            passthrough_info = {
                "header": pp.pack_header({}),
                "segments": [],
                "ref_slots": _ref_slots_of_json(incoming),
                "passthrough": True,
                "shot_count": count,
            }
            return io.NodeOutput(
                incoming,
                "透传模式：pack_text 为空，原样转发已接入的分镜 JSON（%d 个分镜）。"
                % count,
                count,
                json.dumps(passthrough_info, ensure_ascii=False),
            )

        # ---- Parse ---------------------------------------------------------
        segments, meta, issues = pp.parse_pack(
            text, language=language,
            default_duration=float(default_duration or 10.0),
            auto_fix=bool(auto_fix))
        if not segments:
            raise ValueError("H3 分镜提示词 PACK：没有解析到任何分镜。%s"
                             % ("；".join(issues) if issues else ""))

        name = _safe_session_name(session_name,
                                  meta.get("Project"), meta.get("项目"))
        board = pp.to_shots_json(segments, session_name=name)
        # parse_pack 只在正文上跑配乐判据，并把结果放进 meta（见 prompt_pack）
        music_hit = tuple(meta.pop("music_hit", ()) or ())

        report = pp.format_report(segments, meta, issues, music_hit=music_hit)
        slot_line = _ref_slot_line(segments)
        if slot_line:
            report += "\n" + slot_line + "\n"

        if issues and str(on_issue).strip().lower() == "error":
            raise ValueError("H3 分镜提示词 PACK：%d 处不符合规范\n%s"
                             % (len(issues), "\n".join("  · " + i for i in issues)))

        info = pp.pack_info_json(segments, meta, {
            "session_name": name,
            "issues": list(issues),
            "music_hit": list(music_hit),
        })
        payload = json.dumps(board, ensure_ascii=False, indent=2)
        log("H3PromptPackParser: %d segment(s), session=%s, %d issue(s)",
            len(segments), name, len(issues))
        return io.NodeOutput(payload, report, len(segments),
                             json.dumps(info, ensure_ascii=False))


def register_with_extension(ext):
    """Return node classes for MarqueeDirectorExtension.get_node_list()."""
    return [H3PromptPackParser]
