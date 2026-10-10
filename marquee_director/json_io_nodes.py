# -*- coding: utf-8 -*-
"""分镜 JSON 的落盘 / 读回：``H3StoryboardSave`` / ``H3StoryboardLoad``。

这两个节点最初是按更早的 H3 工作流里的槽位契约实现的——图里一直挂着 351
（② 剧本存档 JSON）和 340（⑭-D 读取存档 JSON），而**全机没有任何已安装的包
提供它们**，ComfyUI 里就是两个红色 missing node，点运行直接失败。槽位名与
输出顺序照原样实现，旧图不用改线。

===========  ==============  ==========================================
节点         槽位             说明
===========  ==============  ==========================================
Save         入 shots_json    只能接线（`force_input`，不占 widgets_values）
            出 文件名 / 保存路径 / 分镜资产结果JSON
Load         入 filename      下拉（启动时生成）+ path（手填路径兜底）
            出 分镜资产结果JSON / 文件名 / 文件路径
===========  ==============  ==========================================

★ 2026-10-10 节点 ID 统一成包内一致的 ``H3*`` 风格（原名 ``MinimaxH3SaveJson``
  / ``MinimaxH3LoadJson``）。改名不是好看：另一个已安装的包
  ``Comfyui-MinimaxH3-Ref2VA-Auto-Video`` **注册了同样的 ID**，谁后加载谁赢，
  于是这两个节点在同一个安装里显示成哪个包的哪个名字取决于目录扫描顺序。
  已存工作流里如果用到它们，把类型名换成 ``H3StoryboardSave`` /
  ``H3StoryboardLoad`` 即可，槽位一个没动。

存档落在会话目录里，和 Director 的 manifest、分段 mp4 同目录，方便一起打包
带走：``<输出目录>/h3_continuous/<剧本名>/<版本名>/storyboard.json``。
"""

from __future__ import annotations

import json
import os

from comfy_api.latest import io

from .common import log
# ★ 路径与名字都从 session 拿：**同一个函数**，不再自己拼 ``<output>/h3_continuous``。
#   以前这里独立算了一份根、Director 又算一份，ComfyUI 的 outputDir 一改，
#   存档写新根、渲染留旧根，面板就查不到自己刚存的那份。
from .session import (UNNAMED, output_root, resolve_session_dir, store_root)

CATEGORY = "ComfyUI_Marquee_Director"

ARCHIVE_NAME = "storyboard.json"
NO_FILE = "(暂无 JSON)"
MAX_OPTIONS = 80

# 报错/日志里的人话名字：与 display_name 同源，用户看得见的就这一个词。
SAVE_LABEL = "H3 剧本存档"
LOAD_LABEL = "H3 剧本读档"


# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
def _store_root():
    return store_root()


def _output_root():
    return output_root()


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


def _names_from_payload(text):
    """从分镜 JSON 里取出 ``(会话名, 剧本名)`` —— 与 Director 读的是同两个键。

    ★ 存档必须落到**渲染所在的目录**里。以前这里只认 ``session_name``，兜底
      ``h3_pack``；Director 兜底却是另一个名字，于是同一份剧本的存档和段文件
      分居两个目录，面板查不到、"拷给别人复用"也拷不到东西。现在两边共用
      ``session.resolve_session_dir``，同一个输入必然同一个目录。
    """
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return "", ""
    if not isinstance(payload, dict):
        return "", ""
    return (str(payload.get("session_name") or payload.get("session") or "").strip(),
            str(payload.get("script_name") or payload.get("project") or "").strip())


# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
class H3StoryboardSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3StoryboardSave",
            display_name="H3 剧本存档（分镜 JSON 落盘）",
            category=CATEGORY,
            # 纯副作用节点（写盘）。不声明 output node 的话，ComfyUI 会因为
            # 「不通往任何输出节点」而把整条分支剪掉，存档永远不发生。
            is_output_node=True,
            description="把 H3 导演清洗后的分镜 JSON 落到磁盘，供「H3 剧本读档」"
                        "下次直接读回，也方便把一份定稿剧本拷给别人复用。\n\n"
                        "落点与渲染同目录：``<输出目录>/h3_continuous/<剧本名>/"
                        "<版本名>/storyboard.json``。两个名字都取自 JSON 里的 "
                        "``script_name`` / ``session_name``（解析器写进去的）。",
            inputs=[
                io.String.Input(
                    "shots_json", force_input=True, optional=True,
                    tooltip="接 H3 导演 / H3 分镜 PACK 的 shots_json。"
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
                "%s：shots_json 是空的，没有东西可以存档。" % SAVE_LABEL)

        version, script = _names_from_payload(text)
        directory, _script, name = resolve_session_dir(
            version, script=script or UNNAMED)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, ARCHIVE_NAME)
        # 先写 .tmp 再原子替换：写一半崩了不会留下半截 JSON 把读档坑了
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)

        log("%s: %s (%d chars, 剧本=%s, 会话=%s)",
            SAVE_LABEL, path, len(text), _script or "（单层）", name)
        return io.NodeOutput(ARCHIVE_NAME, path, text)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
class H3StoryboardLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = list_archives()
        return io.Schema(
            node_id="H3StoryboardLoad",
            display_name="H3 剧本读档（分镜 JSON 读回）",
            category=CATEGORY,
            description="读回「H3 剧本存档」写的分镜 JSON（``h3_continuous/**/*.json``，"
                        "两层目录都会扫），接到 H3 导演的 shots_json 上就能原样"
                        "复跑一份定稿剧本。",
            inputs=[
                io.Combo.Input(
                    "filename", options=options, optional=True,
                    default=options[0] if options else NO_FILE,
                    tooltip="存档下拉。注意：下拉内容是 ComfyUI **启动时**扫的，"
                            "刚存的新文件不会自动出现——要么重启，要么直接把路径"
                            "填到下面的 path 里（path 优先）。"),
                io.String.Input(
                    "path", default="", optional=True, advanced=True,
                    display_name="路径兜底",
                    tooltip="手填路径：绝对路径，或相对输出根目录的路径"
                            "（如 h3_continuous/茜色/r1/storyboard.json）。"
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
                "%s：还没选文件。先跑一遍「%s」节点，"
                "或在路径兜底里填上 JSON 的路径。" % (LOAD_LABEL, SAVE_LABEL))
        if not os.path.exists(cand):
            raise FileNotFoundError(
                "%s：文件不存在 —— %s" % (LOAD_LABEL, cand))

        with open(cand, encoding="utf-8") as handle:
            text = handle.read()
        log("%s: %s (%d chars)", LOAD_LABEL, cand, len(text))
        return io.NodeOutput(text, os.path.basename(cand), cand)


def register_with_extension(ext):
    """Return node classes for MarqueeDirectorExtension.get_node_list()."""
    return [H3StoryboardSave, H3StoryboardLoad]
