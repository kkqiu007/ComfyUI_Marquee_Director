# -*- coding: utf-8 -*-
"""HTTP glue for the H3 Director frontend panels.

为什么要有这一层
----------------
PACK 头部（项目 / 模式 / 总时长 / 段数 / 画幅）、参考图分类、时间线进度，
这些信息以前只存在于**运行报告的文本里**——要真跑完一遍才知道对不对。
而"跑一遍"在 H3 上是几十分钟。前端在 Director 节点上开一块面板，边填边读，
配错了当场就能看见，不用拿一次渲染去试。

暴露的东西都是只读的（除了解析文本），不改任何渲染状态。
"""

from __future__ import annotations

import json
import os
import re
import time

from aiohttp import web

from . import prompt_pack as pp
from .common import safe_session_name

# 不再硬编码任何输出根。输出目录一律跟 folder_paths.get_output_directory()，
# 用户换实例/换配置都能自动对上。保留这个常量只是为了让老引用不炸。
EXTRA_OUTPUT_ROOTS = []


# ---------------------------------------------------------------------------
# /h3/pack_preview —— 给 PACK 模块用：不渲染，只解析
# ---------------------------------------------------------------------------
def _pack_preview(text: str, session_name: str = "") -> dict:
    # 双语 PACK 同时解两版：英文版给 Director 真正渲染用（默认 auto 取的那侧），
    # 中文版给前端「六段实时编辑」面板做对照源 / 编辑目标。两段按 index 对齐。
    en_segments, en_meta, issues = pp.parse_pack(text or "", language="en")
    zh_segments, _, _ = pp.parse_pack(text or "", language="zh")
    header = pp.pack_header(en_meta)
    segs = []
    for idx, seg in enumerate(en_segments):
        fields = seg.get("fields") or {}
        body = str(fields.get("detailed_description") or "")
        zh_fields = (zh_segments[idx].get("fields") or {}) if idx < len(zh_segments) else {}
        # 末段不要求无台词尾巴，其余每段最后 1.6s 必须没有新台词
        tail_clean = bool(re.search(
            r"(?i)no\s+new\s+(?:spoken\s+word|dialogue|line)", body[-400:]))
        segs.append({
            "id": seg["id"],
            "new_seconds": round(float(seg["new_seconds"]), 3),
            "handoff_seconds": round(float(seg["handoff_seconds"]), 3),
            "gen_seconds": round(float(seg["gen_seconds"]), 3),
            "frames": seg.get("spec_frames"),
            "task": seg.get("task"),
            "pictures": list(seg.get("pictures") or []),
            "subjects": list(seg.get("subjects") or []),
            "speakers": list(seg.get("speakers") or []),
            "dialogues": seg.get("dialogues") or 0,
            "words": len(body.split()),
            "dialogue_free_tail": tail_clean,
            # 完整六字段（英文版，Director 实际渲染用）：给前端「分镜提示词编辑器」
            # 做逐段实时显示/编辑用，顺序固定为 FIELD_ORDER。
            "full_fields": {k: str(fields.get(k) or "") for k in pp.FIELD_ORDER},
            # 完整六字段（中文版）：对照源 + 可编辑；写回时传 lang="zh"。
            "full_fields_zh": {k: str(zh_fields.get(k) or "") for k in pp.FIELD_ORDER},
            # 每段真正进模型的正文：subject_definitions 也会拼进首段提示词，
            # 所以取详细描述 + 角色定义拼合后的全文更贴近渲染实际。
            "prompt": body,
        })
        # ★ 第 30 轮：整段正文为空是**致命**的（渲染出来全是默认运镜），
        #   但以前面板一个字都不报 —— 用户只看到空提示词框，分不清是
        #   "本来就没内容"还是"数据坏了"。实测踩过：工作流 JSON 里 PACK
        #   存了两份（widgets_values / widgets_values_named），
        #   ComfyUI 加载了 S01 英文块被清空的那一份，界面完全静默。
        #   这里补一条明确指路的 issue，前端会红字显示。
        if not body.strip() and not list(seg.get("pictures") or []):
            issues = list(issues) + [
                "%s：整段正文为空（六个字段一个都没解析到）。最常见原因："
                "工作流 JSON 里同一个 PACK 存了两份"
                "（widgets_values / widgets_values_named），"
                "ComfyUI 加载到了被清空的那一侧 —— 把两份同步即可。"
                % seg["id"]]
    total_new = sum(s["new_seconds"] for s in segs)
    total_gen = sum(s["gen_seconds"] for s in segs)
    declared = None
    try:
        declared = int(str(header.get("segments") or "").strip())
    except (TypeError, ValueError):
        declared = None
    if declared is not None and declared != len(segs):
        issues = list(issues) + [
            "头部写的段数是 %d，实际解析出 %d 段 —— 按实际段数渲染，"
            "成片时长会和头部声明对不上。" % (declared, len(segs))]
    if header.get("total_duration_s") is not None:
        gap = abs(float(header["total_duration_s"]) - total_new)
        if gap > 0.35:
            issues = list(issues) + [
                "头部总时长 %s（%.1fs）与实际新增时长之和 %.1fs 差 %.1fs。"
                % (header["total_duration"], header["total_duration_s"],
                   total_new, gap)]
    return {
        "ok": True,
        "header": header,
        "segments": segs,
        "total_new_seconds": round(total_new, 3),
        "total_gen_seconds": round(total_gen, 3),
        # 会话名由后端算好下发：H3Director 就是拿它当 chain 的 session 落盘的。
        # 面板以前自己猜（读 session_name 控件，空着就退 my_chain），猜错就
        # 查到一个不存在的会话 —— 时间线全空、hover 预览没视频。
        # 候选顺序必须和 H3PromptPackParser 一模一样（控件 → Project），
        # 否则解析器和预览会算出两个名字。
        "session_name": safe_session_name(session_name, header.get("project")),
        "issues": list(issues),
    }


# ---------------------------------------------------------------------------
# /h3/loras —— 闪电渲染下拉框
# ---------------------------------------------------------------------------
def _lora_list() -> list:
    import folder_paths

    out = []
    seen = set()
    for name in folder_paths.get_filename_list("loras") or []:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return sorted(out, key=str.lower)


def _h3_lora_suggest(loras) -> list:
    """把明显是 H3 / turbo / lightning 的 LoRA 排到前面。"""
    hot, cold = [], []
    for name in loras:
        low = name.lower()
        if any(k in low for k in ("h3", "turbo", "lightning", "light", "4step",
                                  "加速", "闪电")):
            hot.append(name)
        else:
            cold.append(name)
    return hot + cold


# ---------------------------------------------------------------------------
# /h3/input_images —— 参考图选择器
# ---------------------------------------------------------------------------
def _input_images(limit=400) -> list:
    import folder_paths

    root = folder_paths.get_input_directory()
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp",
                                        ".bmp", ".gif")):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/")
            try:
                mtime = os.path.getmtime(os.path.join(dirpath, fn))
            except OSError:
                mtime = 0.0
            out.append({"name": rel, "mtime": mtime})
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    out.sort(key=lambda item: -item["mtime"])
    return [{"name": item["name"]} for item in out]


# ---------------------------------------------------------------------------
# /h3/session —— 时间线模块：这个会话已有哪些段、每段多长
# ---------------------------------------------------------------------------
def _session_dirs():
    """会话目录根列表（**查找用**，第一个永远是当前实例的真实落盘根）。

    这里**绝不能硬编码路径**。用户是自定义 ComfyUI，输出目录可能是
    ``<ComfyUI>/output``，也可能是别的；只有 ``folder_paths.get_output_directory()``
    才知道当下这一份实例把东西写到哪。以前这里写死过 ``<ComfyUI>/output`` 和
    ``<另一份安装>/output``，实例一换就指错地方，表现为
    「会话目录不存在」或者反过来「找到一堆别人的旧目录」。

    Session 类（marquee_director/session.py）也是用同一个函数拼目录的，
    所以这里跟着它走就永远和真实落盘位置一致。

    ★ 为什么还要追加第二个根（历史遗留根）
    ---------------------------------------
    ComfyUI Desktop 的 ``--output-directory`` 是**可变的**：同一台机器上它先
    指向 ``<另一份安装>/output``，后来改成 ``<ComfyUI>/output``，
    而更早还用过「不带 --output-directory」的裸 CLI 启动。三段时间各自把
    会话写进了三棵不同的目录树。用户隔天点「断点渲染」时，查找只认当下这
    一个根 → 在当时的 ``<ComfyUI>/output/h3_continuous`` 里找不到昨天那批 seg_NN.mp4 →
    ``done=0`` → 表面上就是「断点渲染又从第 1 段开始」（真实历史在
    ``<base_path>\\output\\h3_continuous`` 下）。

    第二个根 = ``folder_paths.base_path`` + ``output`` —— 这是
    ``folder_paths`` 里 ``output_directory = os.path.join(base_path, "output")``
    的同一条推导式，**同样是配置派生，不是写死**；``base_path`` 又是
    ``--base-directory`` 或 ComfyUI 安装目录本身。没传 ``--output-directory``
    的实例正好落在这里，所以它能覆盖「裸 CLI 时代」的产物。
    """
    try:
        import folder_paths

        roots = [os.path.join(folder_paths.get_output_directory(), "h3_continuous")]
        # 历史遗留根：与当前 output_directory 同源推导（<base_path>/output）。
        base = getattr(folder_paths, "base_path", "") or ""
        if base:
            legacy = os.path.join(base, "output", "h3_continuous")
            if os.path.normcase(legacy) not in [os.path.normcase(r) for r in roots]:
                roots.append(legacy)
    except Exception:
        return []
    return [root for root in roots if os.path.isdir(root)]


def _find_session(name: str):
    """按会话名找目录。

    **必须用 session.sanitize 本尊**。这里原来抄了半份正则（只有
    ``[^A-Za-z0-9._-]+``，缺了后面的 ``strip("._")``），于是中文会话名
    ``曹贼的性价比_v18_en`` 被算成 ``_v18_en``，而 Director 真正建的目录是
    ``v18_en``（前导下划线被 strip 掉了）——差一个字符，永远找不到，
    时间线模块就会一直显示「会话目录不存在」。
    """
    try:
        from .session import sanitize
    except Exception:                                            # pragma: no cover
        def sanitize(text):
            text = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
            text = re.sub(r"\.{2,}", "_", text)
            return text.strip("._") or "session"

    raw = (name or "").strip()
    candidates = []
    for cand in (raw, sanitize(raw)):
        if cand and cand not in candidates:
            candidates.append(cand)
    # 兜底：目录名被截断到 64 字符时，按前缀再找一次
    if raw and len(raw) > 64:
        candidates.append(sanitize(raw)[:64])

    for root in _session_dirs():
        for cand in candidates:
            path = os.path.join(root, cand)
            if os.path.isdir(path):
                return path
        # 最后再放宽：不区分大小写 + 去下划线比一遍
        flat = sanitize(raw).lower().replace("_", "")
        if flat:
            try:
                for entry in os.listdir(root):
                    if not os.path.isdir(os.path.join(root, entry)):
                        continue
                    if entry.lower().replace("_", "") == flat:
                        return os.path.join(root, entry)
            except OSError:
                pass
    return None


def _ffprobe_seconds(path):
    try:
        import subprocess

        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=20)
        return round(float(out.stdout.strip()), 3)
    except Exception:
        return None


def _load_panel_meta(path: str) -> dict:
    """读面板元数据：{段号: {"prompt": 摘要, "speakers": [...], "seconds": 秒}}。

    清完目录后只剩视频，时间线就变成一排没有信息的缩略图。这里把每段「讲了
    什么 / 谁在说话」从存档里捞回来 —— 优先 panel.json（精简），退回
    storyboard.json / shots.json。都没有就返回空，面板只显示时长，不会报错。
    """
    for fname in ("panel.json", "storyboard.json", "shots.json"):
        full = os.path.join(path, fname)
        if not os.path.isfile(full):
            continue
        try:
            with open(full, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        rows = None
        if isinstance(data, dict):
            rows = (data.get("shots_info") or data.get("shots")
                    or data.get("segments"))
        elif isinstance(data, list):
            rows = data
        if not rows:
            continue
        out = {}
        for i, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            shot = row.get("shot")
            if isinstance(shot, dict):
                text = (shot.get("prompt") or shot.get("detailed_description")
                        or shot.get("summary") or "")
                speakers = shot.get("speakers") or []
            else:
                text = str(shot or row.get("prompt") or "")
                speakers = row.get("speakers") or []
            # 只留摘要：面板一格放不下整段提示词，取首句即可
            brief = re.sub(r"\s+", " ", str(text)).strip()
            if len(brief) > 120:
                brief = brief[:120] + "…"
            out[i] = {"prompt": brief,
                      "speakers": speakers if isinstance(speakers, list) else [],
                      "seconds": row.get("duration") or row.get("seconds")}
        if out:
            return out
    return {}


def _session_state(name: str) -> dict:
    path = _find_session(name)
    if not path:
        return {"ok": False, "error": "会话目录不存在：%s" % name,
                "segments": [], "looked_in": _session_dirs()}
    manifest = None
    mpath = os.path.join(path, "manifest.json")
    try:
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        pass
    records = (manifest or {}).get("segments") or []
    panel = _load_panel_meta(path)
    segs = []
    for fn in sorted(os.listdir(path)):
        match = re.match(r"^seg_(\d+)\.mp4$", fn)
        if not match:
            continue
        index = int(match.group(1))
        full = os.path.join(path, fn)
        rec = records[index - 1] if index - 1 < len(records) else {}
        segs.append({
            "index": index,
            "file": fn,
            "url": "/view?filename=%s&subfolder=%s&type=output"
                   % (fn, os.path.relpath(path, os.path.dirname(
                       os.path.dirname(path))).replace("\\", "/")),
            "mtime": round(os.path.getmtime(full), 3),
            "size": os.path.getsize(full),
            "seconds": (rec or {}).get("seconds") or _ffprobe_seconds(full),
            # glob 得到 seg_NN.mp4 就说明这一段渲完了。缓存键只是"能不能原样
            # 复用"的判断，跟"渲没渲过"是两件事，别混在前端的显示里。
            "cached": True,
            "recovered": not bool((rec or {}).get("key")),
            # 面板要在时间线上显示这段「讲了什么 / 谁在说话」
            "prompt": (panel.get(index) or {}).get("prompt", ""),
            "speakers": (panel.get(index) or {}).get("speakers", []),
            "planned_seconds": (panel.get(index) or {}).get("seconds"),
        })
    # 断点续渲会从第几段接着跑：只数从 1 开始连续的那几段。
    done = 0
    for seg in segs:
        if seg["index"] == done + 1:
            done += 1
        else:
            break
    joined = os.path.join(path, "%s.mp4" % os.path.basename(path))
    return {
        "ok": True,
        "session": os.path.basename(path),
        "dir": path,
        "segments": segs,
        "done": done,
        "joined": (os.path.basename(joined) if os.path.exists(joined) else ""),
        "manifest_segments": len(records),
        # 会话名归一化是否已收敛到单一实现（session.sanitize is common.safe_session_name）。
        # 这两侧一旦分叉，渲染写 A 目录、面板查 B 目录，表现为「断点渲染又从第 1
        # 段开始」以及修复节点抛 "segment 1 has no handoff clip"。留着这个字段，
        # 下次面板/断点行为异常时一眼就能排除/坐实这个根因。
        "name_homogeneous": _session_name_homogeneous(),
    }


def _session_name_homogeneous() -> bool:
    """True = 落盘名与查找名用同一套规则（见 common._install_session_sanitize_alias）。"""
    try:
        from . import session as _sess
        return getattr(_sess, "_SANITIZE_IS_ALIAS", False)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------
_registered = False


# ---------------------------------------------------------------------------
# /h3/pack_rebuild —— PACK 实时编辑器：把编辑后的六字段回写进整份 PACK
# ---------------------------------------------------------------------------
def _pack_rebuild(text: str, segments: list, lang=None) -> dict:
    """segments=[{"index": n, "fields": {...}}] → 重拼后的 PACK 文本。

    ``lang`` 透传给 ``pp.rebuild_pack``：``"zh"``/``"en"`` 命中对应语言版块，
    ``None`` 命中最后一次出现的段标题（双语 PACK 默认英文版）。
    """
    rebuilt = pp.rebuild_pack(text or "", segments, lang)
    return {"ok": True, "text": rebuilt}


def _pack_apply_block(text: str, blocks: list, lang=None) -> dict:
    """blocks=[{"index": n, "body": "整段六段式正文"}] → 原样整块替换后的 PACK。

    「一个文本框装整段提示词」的编辑通道：不拆字段、不规范化，body 原样写回。
    """
    rebuilt = pp.rebuild_pack_block(text or "", blocks, lang)
    return {"ok": True, "text": rebuilt}


def register() -> bool:
    """注册 HTTP 路由。可重复调用；失败返回 False（不影响节点本身可用）。"""
    global _registered
    if _registered:
        return True
    try:
        from server import PromptServer
    except Exception:
        return False
    srv = PromptServer.instance
    if srv is None:
        return False
    routes = srv.routes

    @routes.post("/h3/pack_preview")
    async def h3_pack_preview(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        text = (payload or {}).get("text") or ""
        # 画布上 H3PromptPackParser 的 session_name 控件值 —— 前端顺带传上来，
        # 好让会话名的候选顺序和那个节点完全一致。
        sess = str((payload or {}).get("session_name") or "")
        try:
            return web.json_response(_pack_preview(text, sess))
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post("/h3/pack_rebuild")
    async def h3_pack_rebuild(request):
        """实时编辑器写回：body={"text": 原PACK, "segments":[{"index":n,"fields":{}}], "lang": "en"|"zh"|null}

        ``lang`` 指定回写哪一侧语言版块（双语 PACK）：``en`` 只改英文版、``zh`` 只改
        中文版、``null``/省略命中最后一次出现的段标题（默认英文版，Director 直接吃）。
        """
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        text = str((payload or {}).get("text") or "")
        segments = (payload or {}).get("segments") or []
        lang = (payload or {}).get("lang") or None
        if lang not in ("zh", "en"):
            lang = None
        try:
            return web.json_response(_pack_rebuild(text, segments, lang))
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post("/h3/pack_apply_block")
    async def h3_pack_apply_block(request):
        """整段正文写回：body={"text": 原PACK, "blocks":[{"index":n,"body":"整段正文"}], "lang":...}

        与 /h3/pack_rebuild 的区别：这里按「整块正文」原样替换（不拆六字段），
        对应前端「一个文本框装整段六段式提示词」的编辑形态。
        """
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        text = str((payload or {}).get("text") or "")
        blocks = (payload or {}).get("blocks") or []
        lang = (payload or {}).get("lang") or None
        if lang not in ("zh", "en"):
            lang = None
        try:
            return web.json_response(_pack_apply_block(text, blocks, lang))
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.get("/h3/loras")
    async def h3_loras(request):
        try:
            names = _lora_list()
            return web.json_response({"ok": True, "loras": names,
                                      "suggested": _h3_lora_suggest(names)})
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc),
                                      "loras": []}, status=500)

    @routes.get("/h3/input_images")
    async def h3_input_images(request):
        try:
            return web.json_response({"ok": True, "images": _input_images()})
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc),
                                      "images": []}, status=500)

    @routes.get("/h3/dump_refs")
    async def h3_dump_refs(request):
        """调试用：把 H3Director 节点当前真实拿到的 ref_image_N 接线还原成文件名。

        **为什么有这个**：用户报"换参考图没用"，但前几轮排查发现，问题往往不是
        没换上去，而是**接到了错的槽位** —— 工作流里 LoadImage 节点的物理编号
        (731/732/...) 与 ref_image_N 的逻辑编号不一致，又没有直观的展示层。
        以前每次都得手动 curl /queue 然后写 Python 反查；这个端点把这件事固化下来。

        数据源（按新鲜度顺序，能找到几条就返回几条）：
            1. /queue 的 queue_running（含正在跑的完整 prompt）
            2. /queue 的 queue_pending（排队的）
            3. /history 最近 10 条（完成的）

        可选 query：
            node_id=700       只看这一个 H3Director 节点（其它 class_type 也认）
            max_history=10    最多翻多少条 history（默认 10，避免一次拉太多）
            session=xxx       只保留 session_name 含该子串的条目

        返回结构：
            {
              ok: true,
              found: N,                     # 命中的 H3 节点总数
              entries: [
                {
                  source: "queue_running" | "queue_pending" | "history",
                  prompt_id: "abc...",
                  node_id: "700",
                  class_type: "H3Director",
                  session_name: "曹贼的性价比_v18_en_v11",
                  refs: {
                    "ref_image_0": { upstream_node: "731", upstream_class: "LoadImage",
                                     filename: "Qwen_4View_sheet_00008_.png",
                                     link_kind: "loadimage" },
                    "ref_image_1": { upstream_node: "732", upstream_class: "LoadImage",
                                     filename: "krea2_identity_edit_00071_.png",
                                     link_kind: "loadimage" },
                    ...
                  }
                }
              ],
              hint: "..."
            }
        """
        import aiohttp                                   # noqa: F401  用现成 import
        node_filter = (request.query.get("node_id") or "").strip()
        session_filter = (request.query.get("session") or "").strip()
        try:
            max_history = int(request.query.get("max_history") or 10)
        except Exception:
            max_history = 10
        max_history = max(1, min(max_history, 50))

        def _ref_for(prompt_dict, ref_value):
            """把 ref_image_N 的 link 还原成 {上游节点id, class_type, 文件名}。
            link 形态（按优先级）：上游 IMAGE 输出 / 直接字符串文件名 / 元组 (filename, ?)。"""
            link = ref_value
            upstream_node = None
            upstream_class = None
            filename = None
            link_kind = "unknown"
            if isinstance(link, list) and link:
                upstream_node = str(link[0])
                upstream = prompt_dict.get(link[0], {}) if isinstance(prompt_dict, dict) else {}
                upstream_class = upstream.get("class_type") if isinstance(upstream, dict) else None
                up_inp = upstream.get("inputs", {}) if isinstance(upstream, dict) else {}
                # LoadImage / LoadImageMask 系列的常见输入键
                for k in ("image", "filename", "image1"):
                    if k in up_inp and isinstance(up_inp[k], str):
                        filename = up_inp[k]; break
                # 复合键：image 是 list [filename, subfolder, ...]
                if not filename and "image" in up_inp and isinstance(up_inp["image"], list) and up_inp["image"]:
                    first = up_inp["image"][0]
                    if isinstance(first, str):
                        filename = first
                link_kind = "loadimage" if upstream_class in ("LoadImage", "LoadImageMask", "VHSLoadVideo", "ImageLoader") or filename else "image_link"
            elif isinstance(link, str):
                filename = link
                link_kind = "filename"
            return {"upstream_node": upstream_node, "upstream_class": upstream_class,
                    "filename": filename, "link_kind": link_kind}

        async def _fetch(session, url):
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                return await r.json(content_type=None)

        entries = []
        try:
            async with aiohttp.ClientSession() as session:
                # 1) queue
                try:
                    qd = await _fetch(session, "/queue")
                except Exception as e:
                    qd = {}
                for src_key in ("queue_running", "queue_pending"):
                    for item in (qd.get(src_key) or []):
                        if not isinstance(item, list) or len(item) < 3:
                            continue
                        prompt_dict = item[2]
                        if not isinstance(prompt_dict, dict):
                            continue
                        for nid, node in prompt_dict.items():
                            ct = node.get("class_type") if isinstance(node, dict) else None
                            if ct != "H3Director":
                                continue
                            if node_filter and str(nid) != node_filter:
                                continue
                            ins = node.get("inputs", {}) if isinstance(node, dict) else {}
                            session_name = str(ins.get("session_name") or "")
                            if session_filter and session_filter not in session_name:
                                continue
                            refs = {}
                            for k, v in ins.items():
                                if "ref_images.ref_image_" in k:
                                    refs[k.replace("ref_images.", "")] = _ref_for(prompt_dict, v)
                            entries.append({"source": src_key, "prompt_id": item[1] if len(item) > 1 else None,
                                            "node_id": str(nid), "class_type": ct,
                                            "session_name": session_name, "refs": refs})
                # 2) history（最多 max_history 条）
                try:
                    hd = await _fetch(session, f"/history?max_items={max_history}")
                except Exception as e:
                    hd = {}
                for hid, h in list((hd or {}).items())[:max_history]:
                    prompt_dict = h.get("prompt", [None, None, {}])[2]
                    if not isinstance(prompt_dict, dict):
                        continue
                    for nid, node in prompt_dict.items():
                        ct = node.get("class_type") if isinstance(node, dict) else None
                        if ct != "H3Director":
                            continue
                        if node_filter and str(nid) != node_filter:
                            continue
                        ins = node.get("inputs", {}) if isinstance(node, dict) else {}
                        session_name = str(ins.get("session_name") or "")
                        if session_filter and session_filter not in session_name:
                            continue
                        refs = {}
                        for k, v in ins.items():
                            if "ref_images.ref_image_" in k:
                                refs[k.replace("ref_images.", "")] = _ref_for(prompt_dict, v)
                        entries.append({"source": "history", "prompt_id": hid,
                                        "node_id": str(nid), "class_type": ct,
                                        "session_name": session_name, "refs": refs})
        except Exception as exc:                          # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc), "entries": []}, status=500)

        hint = ("看每个 ref_image_N 的 filename —— 它就是那张图实际喂给 H3 的来源。"
                "如果看板显示的『图片2 · 角色』和 ref_image_2 不一致，"
                "就是工作流连线把那张图接到了别的 ref_image 槽位。")
        return web.json_response({"ok": True, "found": len(entries),
                                  "entries": entries, "hint": hint,
                                  "filters": {"node_id": node_filter or None,
                                              "session": session_filter or None,
                                              "max_history": max_history}})

    @routes.get("/h3/session")
    async def h3_session(request):
        name = request.query.get("name", "")
        try:
            return web.json_response(_session_state(name))
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc),
                                      "segments": []}, status=500)

    @routes.post("/h3/cleanup")
    async def h3_cleanup(request):
        """清会话目录：只留分镜分段视频 + 完整拼接视频。

        body: {"force": true}   —— force 会无视"正在渲染"的守卫（慎用）
        query: ?dry=1           —— 只列出会删什么，不真删
        """
        try:
            from . import cleanup as cleanup_mod
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        dry = request.query.get("dry", "") in ("1", "true", "yes")
        force = False
        try:
            body = await request.json()
            if isinstance(body, dict):
                force = bool(body.get("force"))
        except Exception:
            force = request.query.get("force", "") in ("1", "true", "yes")

        try:
            if dry:
                return web.json_response(
                    {"ok": True, "dry": True, **cleanup_mod.preview(force)})
            return web.json_response(
                {"ok": True, **cleanup_mod.sweep(force=force)})
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    _registered = True
    return True


# ---------------------------------------------------------------------------
# 进度事件（Director → 前端时间线）
# ---------------------------------------------------------------------------
EVENT_PROGRESS = "h3.director.progress"


def emit_progress(node_id, **payload) -> None:
    """把一段渲染进度广播给所有前端。

    用 send_sync 广播而不是塞进 progress 输出：输出要等节点跑完才拿得到，
    而时间线高亮和小马是"正在跑"的时候才需要的东西。
    """
    try:
        from server import PromptServer

        srv = PromptServer.instance
        if srv is None:
            return
        data = dict(payload)
        data["node_id"] = node_id
        data["ts"] = time.time()
        srv.send_sync(EVENT_PROGRESS, data)
    except Exception:
        pass
