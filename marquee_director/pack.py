# -*- coding: utf-8 -*-
"""导演包（PACK 剧本 + 参考图）的 zip 导入 / 导出。

借鉴 `ComfyUI_MiniMaxH3_Director/director/pack.py` 的**安全骨架**，不照搬它的
结构：那边的导演台自己管媒体（时间轴 UI 内上传），而本包的参考图是工作流里
的 ``LoadImage`` 节点、分镜 JSON 走 ``H3StoryboardSave`` 落盘 —— 包格式按本包
自己的资产来：

    pack.json            格式 / 版本 / 剧本名 / 会话名 / 段数 / 时间戳 / 槽位表
    storyboard.json      与「H3 剧本存档」同款分镜 JSON（读档节点可直接吃）
    script.txt           PACK 原文（六段格式源文本），可选
    refs/PictureN.png    参考图，槽位号 = 提示词里的 ``<Picture N>``
                         （= Director 的 ``ref_image_(N-1)`` 口）

安全纪律全部来自上游（它踩过的坑，不重踩）：
  * 路径穿越 / 绝对路径 / 非 ASCII 路径：拒绝；
  * zip 条目数、解压总容量、单文件大小：三重上限，防 zip bomb；
  * 扩展名白名单（图片 + json/txt）：别的直接拒；
  * 导出 zip 走调用方 StreamResponse 分块发送（Windows 上 FileResponse 会给
    空 body），临时导出文件按 TTL 清。

导入的参考图复制到 ``input/h3_director_packs/<包名>/``，分镜 JSON 落到它
真正的会话目录（``resolve_session_dir`` 同一个算法），storyboard 与渲染产物
同目录，断点续渲染、面板时间线天然都认它。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

PACK_FORMAT = "marquee-h3-director-pack"
PACK_VERSION = 1
PACK_SUFFIX = ".h3pack.zip"

#: 导入的参考图在 ComfyUI input 下的落点（子目录名）。
PACKS_DIRNAME = "h3_director_packs"

# 三重上限。本包的成员只有图 + 两个文本，条目数给到 2000 足够（图 9 张是
# ref2va 硬上限），总量 8 GB、单文件 2 GB 是图/短视频源素材的量级。
MAX_ZIP_ENTRIES = 2000
MAX_UNCOMPRESSED = 8 * 1024 * 1024 * 1024
MAX_SINGLE_FILE = 2 * 1024 * 1024 * 1024

PACK_EXPORT_TTL_SEC = 60 * 60
ZIP_STREAM_CHUNK = 1024 * 1024

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
              ".mp4", ".mov", ".webm", ".mkv"}       # 源视频模式可挂片段参考
TEXT_EXTS = {".json", ".txt"}
ALLOWED_EXTS = IMAGE_EXTS | TEXT_EXTS

ASCII_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
PICTURE_FILE_RE = re.compile(r"^Picture([1-9])(\.[A-Za-z0-9]{1,8})$", re.I)

META_NAME = "pack.json"
STORYBOARD_NAME = "storyboard.json"
SCRIPT_NAME = "script.txt"
REFS_DIR = "refs"


class PackError(Exception):
    """包格式 / 安全问题。message 直接给人看。"""


# ---------------------------------------------------------------------------
# 路径与名字
# ---------------------------------------------------------------------------
def _input_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_input_directory())
    except Exception:
        return Path("input")


def _export_root() -> Path:
    root = Path(tempfile.gettempdir()) / "h3_pack_export"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def purge_exports(*, keep: str | None = None) -> int:
    """删掉超过 TTL 的导出 zip。正在下载的那个不删。"""
    root = _export_root()
    now = time.time()
    removed = 0
    for path in root.glob("*" + PACK_SUFFIX):
        if keep and path.name == keep:
            continue
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age >= PACK_EXPORT_TTL_SEC:
            _unlink_quiet(path)
            removed += 1
    return removed


def _safe_slug(name: str, fallback: str = "pack") -> str:
    """落盘子目录名：只留安全字符，防路径穿越。"""
    from .common import safe_session_name

    slug = safe_session_name(str(name or ""))
    return slug or fallback


def _is_safe_rel(rel: str) -> bool:
    """zip 成员名必须 ASCII、相对、无 ..。"""
    s = str(rel or "").replace("\\", "/").strip()
    if not s or s.startswith("/") or ".." in s.split("/"):
        return False
    return bool(ASCII_PATH_RE.fullmatch(s))


def _safe_ext(name: str) -> bool:
    return Path(str(name)).suffix.lower() in ALLOWED_EXTS


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------
def storyboard_from_pack_text(text: str, session_name: str = "") -> tuple:
    """用 Parser 的**同一套**解析从 PACK 原文生成 shots_json。

    返回 ``(storyboard_json, script_name, session_name)``。

    为什么不前端拼：H3PromptPackParser.execute 里 ``pp.to_shots_json`` 就是
    存档节点 / 导演节点吃的那个格式（两层落盘名的算法也只此一份）。面板
    /h3/pack_preview 下发的是 segments（给编辑器用），不是 shots_json 原文；
    包里的 storyboard.json 要与读档节点完全同规格，只能在服务端用同一个
    函数生成。
    """
    from . import prompt_pack as pp
    from .common import safe_session_name

    segments, meta, _issues = pp.parse_pack(text or "", language="en")
    if not segments:
        raise PackError("PACK 原文没有解析出任何分镜（storyboard 生成失败）。")
    script = safe_session_name(meta.get("Project"), meta.get("项目"))
    name = safe_session_name(session_name, script)
    board = pp.to_shots_json(segments, session_name=name, script_name=script)
    return (json.dumps(board, ensure_ascii=False, indent=2), script, name)


def build_pack(*, meta: dict, storyboard: str, script: str = "",
               refs: dict | None = None) -> Path:
    """把剧本 + 媒体打成 zip，返回 zip 的临时路径。

    ``refs``: ``{槽位号(int): input 相对路径}`` —— 前端从 LoadImage 节点读到的
    就是 input 相对路径（widgets_values 存的就是它）。槽位号 1 起，对应
    ``<Picture N>``。
    """
    refs = dict(refs or {})
    meta = dict(meta or {})
    if storyboard and not meta.get("script_name") and not meta.get("session_name"):
        # 前端没给名字时从 storyboard 里补（与存档节点同款键名）
        session_name, script_name = _names_from_storyboard(storyboard)
        meta.setdefault("session_name", session_name)
        meta.setdefault("script_name", script_name)
    meta.update({
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "refs": {str(k): v for k, v in refs.items()},
    })

    purge_exports()
    name = "%s_%s%s" % (
        _safe_slug(meta.get("script_name") or meta.get("session_name") or "pack"),
        "%08x" % (int(time.time()) & 0xffffffff), PACK_SUFFIX)
    zip_path = _export_root() / name

    input_root = _input_root()
    written = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(META_NAME, json.dumps(meta, ensure_ascii=False, indent=2))
        zf.writestr(STORYBOARD_NAME, storyboard or "")
        if script and str(script).strip():
            zf.writestr(SCRIPT_NAME, script)
        for slot, rel in sorted(refs.items(), key=lambda kv: int(kv[0])):
            slot = int(slot)
            if slot < 1 or slot > 9:
                raise PackError("参考图槽位 %d 超出 1-9（ref2va 最多 9 张）。" % slot)
            src = Path(str(rel))
            if not src.is_absolute():
                src = input_root / src
            if not src.is_file():
                raise PackError(
                    "参考图 <Picture %d> 找不到文件：%s" % (slot, rel))
            ext = src.suffix.lower() or ".png"
            zf.writestr("%s/Picture%d%s" % (REFS_DIR, slot, ext), src.read_bytes())
            written.append(slot)
    return zip_path


# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
def _validate_entry(name: str, info: zipfile.ZipInfo,
                    uncompressed_total: int) -> int:
    """返回该成员解压后的大小；不合格直接抛。"""
    if len(name) > 200 or not _is_safe_rel(name):
        raise PackError("包内路径不合法：%r" % name[:80])
    if not _safe_ext(name):
        raise PackError("包内文件类型不允许：%r" % name[:80])
    size = int(info.file_size)
    if size > MAX_SINGLE_FILE:
        raise PackError("包内单文件过大（%.1f MB > 上限 %.1f MB）：%s"
                        % (size / 1e6, MAX_SINGLE_FILE / 1e6, name[:80]))
    if uncompressed_total + size > MAX_UNCOMPRESSED:
        raise PackError("解压总容量超过上限（%.1f GB）：%s"
                        % (MAX_UNCOMPRESSED / 2 ** 30, name[:80]))
    return size


def extract_pack_zip(zip_path, dest: Path) -> None:
    """安全解压到 ``dest``。逐条目校验 + 边写边计数（zip bomb 也拦得住）。"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise PackError("包内条目过多（%d > %d）" % (len(infos), MAX_ZIP_ENTRIES))
        total = 0
        for info in infos:
            name = info.filename
            if name.endswith("/"):                       # 目录条目
                if not _is_safe_rel(name):
                    raise PackError("包内路径不合法：%r" % name[:80])
                continue
            total += _validate_entry(name, info, total)
            target = dest / name
            # 双保险：解析后的真实路径必须仍在 dest 内
            try:
                target.resolve().relative_to(dest.resolve())
            except ValueError:
                raise PackError("包内路径越界：%r" % name[:80])
            # zip 常没有显式目录条目（writestr 直接写 refs/PictureN.png）
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)


def import_pack(zip_path) -> dict:
    """导入一个 .hpack.zip，返回 ``{meta, storyboard, script, slots, pack_dir}``。

    副作用（按顺序）：
      1. 参考图复制到 ``input/h3_director_packs/<包名>/PictureN.ext``；
      2. storyboard.json 写到它真正的会话目录（故事名/会话名取自 JSON 本身，
         算法与存档节点共用 ``resolve_session_dir``）。

    ``slots``: ``[{slot, path}]``，``path`` 是 input 相对路径 —— 前端拿它去
    改写 LoadImage 节点的 widgets_values。
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise PackError("找不到上传的包文件。")

    staging = Path(tempfile.mkdtemp(prefix="h3_pack_import_"))
    try:
        extract_pack_zip(zip_path, staging)
        return _assemble_pack(staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _assemble_pack(root: Path) -> dict:
    meta_path = root / META_NAME
    if not meta_path.is_file():
        raise PackError("这个 zip 不是 Marquee 导演包（缺 pack.json）。")
    try:
        meta = json.loads(_read_text(meta_path))
    except ValueError:
        raise PackError("pack.json 不是合法 JSON。")
    if not isinstance(meta, dict) or meta.get("format") != PACK_FORMAT:
        raise PackError("包格式不是 %s（version=%s）。" % (PACK_FORMAT,
                                                          meta.get("version")))

    storyboard = _read_text(root / STORYBOARD_NAME)
    if not storyboard.strip():
        raise PackError("包里没有 storyboard.json（或它是空的）。")
    script = _read_text(root / SCRIPT_NAME)

    # 剧本名/会话名从 storyboard 里取（和存档节点读的是同两个键）
    session_name, script_name = _names_from_storyboard(storyboard)
    session_name = str(meta.get("session_name") or session_name or "").strip()
    script_name = str(meta.get("script_name") or script_name or "").strip()

    # 1) 参考图 → input/h3_director_packs/<包名>/
    input_root = _input_root()
    pack_dir = input_root / PACKS_DIRNAME / _safe_slug(
        script_name or session_name or "pack")
    pack_dir.mkdir(parents=True, exist_ok=True)
    slots = []
    refs_dir = root / REFS_DIR
    if refs_dir.is_dir():
        for entry in sorted(refs_dir.iterdir()):
            match = PICTURE_FILE_RE.match(entry.name)
            if not entry.is_file() or not match:
                continue                     # 非 PictureN 的杂件忽略
            slot = int(match.group(1))
            dest = pack_dir / entry.name
            shutil.copyfile(entry, dest)
            # path 必须是 **input 相对、含包名层**：LoadImage 的
            # widgets_values 认的就是这个形式；少了包名层的话，两个不同包
            # 各自的 Picture1.png 会指向同一个文件。
            try:
                rel = dest.resolve().relative_to(input_root.resolve()).as_posix()
            except ValueError:                                 # pragma: no cover
                rel = dest.name
            slots.append({"slot": slot, "path": rel})

    # 2) storyboard.json → 会话目录（与渲染同目录，读档/断点续渲染都认）
    storyboard_path = ""
    try:
        from .session import resolve_session_dir

        directory, _script, _version = resolve_session_dir(
            session_name, script=script_name)
        os.makedirs(directory, exist_ok=True)
        target = os.path.join(directory, STORYBOARD_NAME)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(storyboard)
        os.replace(tmp, target)
        storyboard_path = target
    except Exception as exc:                                     # pragma: no cover
        storyboard_path = "（写入会话目录失败：%s）" % exc

    return {
        "meta": meta,
        "storyboard": storyboard,
        "script": script,
        "slots": slots,
        "pack_dir": str(pack_dir),
        "storyboard_path": storyboard_path,
        "session_name": session_name,
        "script_name": script_name,
    }


def _names_from_storyboard(text: str):
    """storyboard.json 里的 (会话名, 剧本名)。与 json_io_nodes 同款键名。"""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return "", ""
    if not isinstance(payload, dict):
        return "", ""
    return (str(payload.get("session_name") or payload.get("session") or "").strip(),
            str(payload.get("script_name") or payload.get("project") or "").strip())
