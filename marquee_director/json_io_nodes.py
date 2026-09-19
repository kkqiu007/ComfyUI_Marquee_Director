# -*- coding: utf-8 -*-
"""分镜 JSON 的落盘 / 读回：`MinimaxH3SaveJson` / `MinimaxH3LoadJson`。

这两个类名是从更早的 H3 工作流里沿留下来的——图里一直挂着 351（② 剧本存档
JSON）和 340（⑭-D 读取存档 JSON），但**全机没有任何已安装的包提供它们**，
ComfyUI 里就是两个红色 missing node，点运行直接失败。与其让用户去找一个早就
不存在的插件包，不如在 H3-Continuous 里按原来的槽位契约实现一份：

===========  ==============  ==========================================
节点         槽位             说明
===========  ==============  ==========================================
SaveJson     入 shots_json    只能接线（`force_input`，不占 widgets_values）
            出 文件名 / 保存路径 / 分镜资产结果JSON
LoadJson     入 filename      下拉（启动时生成）+ path（手填路径兜底）
            出 分镜资产结果JSON / 文件名 / 文件路径
===========  ==============  ==========================================

节点号、槽位名、输出顺序都和老图一致，**旧工作流不用改线**。

存档落在 ``output/h3_continuous/<session>/storyboard.json``——和 Director 的
manifest、分段 mp4 同目录，方便一起打包带走。
"""

from __future__ import annotations

import json
import os

import folder_paths

from comfy_api.latest import io

from .common import log
from .session import sanitize

CATEGORY = "ComfyUI_Marquee_Director"

STORE = "h3_continuous"
ARCHIVE_NAME = "storyboard.json"
NO_FILE = "(暂无 JSON)"
DEFAULT_SESSION = "h3_pack"
MAX_OPTIONS = 80


# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
def _store_root():
    return os.path.join(folder_paths.get_output_directory(), STORE)


def _output_root():
    return folder_paths.get_output_directory()


def list_archives(limit=MAX_OPTIONS):
    """``output/h3_continuous/`` 下所有 .json，按修改时间倒序，返回相对路径。"""
    root = _store_root()
    found = []
    if os.path.isdir(root):
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith(".json"):
                    continue
                full = os.path.join(dirpath, name)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    continue
                rel = os.path.relpath(full, root).replace("\\", "/")
                found.append((mtime, rel))
    found.sort(reverse=True)
    rels = [rel for _m, rel in found[:limit]]
    return rels or [NO_FILE]


def resolve_candidate(filename, path=""):
    """把「下拉选的」+「手填的」合成一个真实路径；都没给就返回 ''。"""
    text = str(path or "").strip()
    if text:
        if os.path.isabs(text):
            return text
        for base in (_output_root(), _store_root()):
            cand = os.path.join(base, text)
            if os.path.exists(cand):
                return cand
        return os.path.join(_store_root(), text)

    name = str(filename or "").strip()
    if not name or name == NO_FILE:
        return ""
    return os.path.join(_store_root(), name.replace("/", os.sep))


def _session_from_payload(text):
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("session_name") or payload.get("session")
               or "").strip()


# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
class MinimaxH3SaveJson(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinimaxH3SaveJson",
            display_name="💾 剧本存档 JSON",
            category=CATEGORY,
            # 纯副作用节点（写盘）。不声明 output node 的话，ComfyUI 会因为
            # 「不通往任何输出节点」而把整条分支剪掉，存档永远不发生。
            is_output_node=True,
            description="把 H3Director 清洗后的分镜 JSON 落到磁盘，供 ⑭-D "
                        "「读取存档 JSON」下次直接读回，也方便把一份定稿剧本"
                        "拷给别人复用。\n\n"
                        "路径：output/h3_continuous/<会话名>/storyboard.json。"
                        "会话名取自 JSON 里的 session_name。",
            inputs=[
                io.String.Input(
                    "shots_json", force_input=True, optional=True,
                    tooltip="接 H3Director / H3PromptPackParser 的 shots_json。"
                            "只能接线——这个槽位是存档的来源，没有默认值。"),
            ],
            outputs=[
                io.String.Output("filename", display_name="文件名"),
                io.String.Output("path", display_name="保存路径"),
                io.String.Output("shots_json", display_name="分镜资产结果JSON"),
            ],
        )

    @classmethod
    def execute(cls, shots_json=None) -> io.NodeOutput:
        text = str(shots_json or "").strip()
        if not text:
            raise ValueError(
                "MinimaxH3SaveJson：shots_json 是空的，没有东西可以存档。")

        name = sanitize(_session_from_payload(text) or DEFAULT_SESSION)
        directory = os.path.join(_store_root(), name)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, ARCHIVE_NAME)
        # 先写 .tmp 再原子替换：写一半崩了不会留下半截 JSON 把 ⑭-D 坑了
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)

        log("MinimaxH3SaveJson: %s (%d chars, session=%s)",
            path, len(text), name)
        return io.NodeOutput(ARCHIVE_NAME, path, text)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
class MinimaxH3LoadJson(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = list_archives()
        return io.Schema(
            node_id="MinimaxH3LoadJson",
            display_name="📂 读取存档 JSON",
            category=CATEGORY,
            description="读回 ② 存档的分镜 JSON（output/h3_continuous/**/"
                        "storyboard.json），接到 610 或 H3Director 的 "
                        "shots_json 上就能原样复跑一份定稿剧本。",
            inputs=[
                io.Combo.Input(
                    "filename", options=options, optional=True,
                    default=options[0] if options else NO_FILE,
                    tooltip="存档下拉。注意：下拉内容是 ComfyUI **启动时**扫的，"
                            "刚存的新文件不会自动出现——要么重启，要么直接把路径"
                            "填到下面的 path 里（path 优先）。"),
                io.String.Input(
                    "path", default="", optional=True, advanced=True,
                    tooltip="手填路径兜底：绝对路径，或相对 output/ 的路径"
                            "（如 h3_continuous/h3_pack/storyboard.json）。"
                            "填了就以它为准。"),
            ],
            outputs=[
                io.String.Output("shots_json", display_name="分镜资产结果JSON"),
                io.String.Output("filename", display_name="文件名"),
                io.String.Output("path", display_name="文件路径"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, filename, path=""):
        # 文件会被别的节点改写，绝不能吃缓存。
        cand = resolve_candidate(filename, path)
        try:
            return "%s:%s" % (cand, os.path.getmtime(cand))
        except OSError:
            return "missing"

    @classmethod
    def execute(cls, filename=NO_FILE, path="") -> io.NodeOutput:
        cand = resolve_candidate(filename, path)
        if not cand:
            raise FileNotFoundError(
                "MinimaxH3LoadJson：还没选文件。先跑一遍 ② 存档节点，"
                "或在 path 里填上 JSON 的路径。")
        if not os.path.exists(cand):
            raise FileNotFoundError(
                "MinimaxH3LoadJson：文件不存在 —— %s" % cand)

        with open(cand, encoding="utf-8") as handle:
            text = handle.read()
        log("MinimaxH3LoadJson: %s (%d chars)", cand, len(text))
        return io.NodeOutput(text, os.path.basename(cand), cand)


def register_with_extension(ext):
    """Return node classes for MarqueeDirectorExtension.get_node_list()."""
    return [MinimaxH3SaveJson, MinimaxH3LoadJson]
