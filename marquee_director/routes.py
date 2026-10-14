# -*- coding: utf-8 -*-
"""HTTP glue for the H3Director frontend panels.

为什么要有这一层
----------------
PACK 头部（项目 / 模式 / 总时长 / 段数 / 画幅）、参考图分类、时间线进度，
这些信息以前只存在于**运行报告的文本里**——要真跑完一遍才知道对不对。
而"跑一遍"在 H3 上是几十分钟。前端在 Director 节点上开一块面板，边填边读，
配错了当场就能看见，不用拿一次渲染去试。

暴露的东西都是只读的（除了解析文本），不改任何渲染状态。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

from aiohttp import web

from . import prompt_pack as pp
from .common import log, safe_session_name

# 输出目录一律跟 folder_paths.get_output_directory()，不硬编码任何输出根。
# （2026-09-20 清理：这里的 EXTRA_OUTPUT_ROOTS = [] 是「怕老引用炸」留下的空壳，
#   实测全包零引用 —— 老引用早就没了。）


# ---------------------------------------------------------------------------
# 缓存层
# ---------------------------------------------------------------------------
# 面板不是「点一次解析一次」，而是**一直在问**：PACK 文本框每敲一下就发一次
# /h3/pack_preview（前端 debounce 1200ms）、改语言标签再发一次、点刷新再发一次，
# 时间线/参考图/功能规划三个版块共用同一份结果。而 _pack_preview 一次要跑
# **两遍**全量解析（en 侧给 Director 渲染用、zh 侧给编辑器对照用）—— 实测一份
# 6 段 PACK 单侧 18ms，两侧 36ms，外加 shot_script_segment_ranges 与
# lang_section_available。文本没变时这些全是白烧的 CPU。
#
# 所以按「内容哈希」缓存：同一个 PACK 文本 + 同一个会话名 → 同一个结果。
# 键里必须带 session_name —— 它参与 safe_session_name()，同名不同值会算出
# 不同会话，漏掉就会串目录。
#
# ★ 为什么用「复制后再返回」而不是直接返回缓存对象：返回的 dict 会被
#   web.json_response 序列化，调用方（_pack_rebuild 等）也可能就地改。共享
#   同一个 dict 的话，一次调用方的误改会污染后续所有命中。深拷贝在这里比重新
#   解析便宜两个数量级。
_PREVIEW_CACHE_MAX = 32
_preview_cache = {}          # key -> (stamp, result)
_preview_lock = threading.Lock()


def _content_key(*parts) -> str:
    """内容寻址的缓存键。空值也参与哈希（None 与 "" 必须是不同的键）。"""
    h = hashlib.sha1()
    for p in parts:
        h.update(b"\x00")
        h.update(str(p if p is not None else "\x00None").encode("utf-8", "replace"))
    return h.hexdigest()


def _cache_get(store, key):
    with _preview_lock:
        hit = store.get(key)
        if hit is None:
            return None
        store[key] = hit          # 重新插入 → LRU 顺序（dict 保序）
        return hit[1]


def _cache_put(store, key, value, maxsize=_PREVIEW_CACHE_MAX):
    with _preview_lock:
        store[key] = (time.time(), value)
        while len(store) > maxsize:
            store.pop(next(iter(store)))     # 丢最久未用的


def _cached_preview(text: str, session_name: str = "") -> dict:
    """/h3/pack_preview 的带缓存版本。键 = PACK 全文 + 会话名。

    缓存里存的是 **JSON 字符串**，返回时再 ``json.loads`` 还原成一份新对象。
    为什么不直接存 dict：返回值会被 web.json_response 序列化，也可能被调用方
    就地修改；共享同一个 dict 的话，一次误改会污染后续所有命中（表现为"改了
    PACK 面板却还是旧内容"，且再也刷不掉）。一次序列化往返比重新解析便宜两个
    数量级（实测命中路径 <1ms，而单侧解析就要 18ms）。
    """
    key = _content_key(text, session_name)
    hit = _cache_get(_preview_cache, key)
    if hit is not None:
        return json.loads(hit)              # 防调用方就地改坏缓存
    result = _pack_preview(text, session_name)
    _cache_put(_preview_cache, key, json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# /h3/pack_preview —— 给 PACK 模块用：不渲染，只解析
# ---------------------------------------------------------------------------
def _pack_preview(text: str, session_name: str = "") -> dict:
    # 双语 PACK 同时解两版：英文版给 Director 真正渲染用（默认 auto 取的那侧），
    # 中文版给前端「六段实时编辑」面板做对照源 / 编辑目标。两段按 index 对齐。
    en_segments, en_meta, issues = pp.parse_pack(text or "", language="en")
    zh_segments, _, _ = pp.parse_pack(text or "", language="zh")
    # 传 en_segments：无 PACK 头部的输入（[Shot N] 布局）靠它反推段数/总时长
    header = pp.pack_header(en_meta, en_segments)
    # ★ 整段式 [Shot N] 脚本**没有** ########## 段标题 —— 前端「分镜时间线」
    #   靠段标题定位原文块（h3LocateSegRange），在这类稿子上一个段都切不出来：
    #   编辑器只能显示后端兜底拼的内容，且 doWrite 会因 fromFallback 拒绝写回
    #   （表现：时间线上改了提示词，什么都没发生）。
    #   所以把每段在**原文里的行区间**一起下发，前端拿它当兜底定位。
    #   分段 PACK（########## 形式）用不到，给 None。
    pack_ranges = []
    if str((en_meta or {}).get("layout") or "") == pp.SHOT_SCRIPT_LAYOUT:
        try:
            pack_ranges = pp.shot_script_segment_ranges(text or "")
        except Exception:                                     # pragma: no cover
            pack_ranges = []
    segs = []
    for idx, seg in enumerate(en_segments):
        fields = seg.get("fields") or {}
        body = str(fields.get("detailed_description") or "")
        zh_fields = (zh_segments[idx].get("fields") or {}) if idx < len(zh_segments) else {}
        rng = pack_ranges[idx] if idx < len(pack_ranges) else None
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
            # ★ 该段在**原文**里的行区间 [start, end)（整段式 [Shot N] 脚本才有；
            #   分段 PACK 为 None）。前端 h3LocateSegRange 的兜底定位用它 ——
            #   没有它，[Shot N] 稿子上的分镜提示词编辑器既切不出内容也写不回去。
            "pack_range": [int(rng[0]), int(rng[1])] if rng else None,
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
    # 节点自动修过的项（不是问题，是成功日志）—— 前端要单独显示，别混进 issues。
    fixes = []
    for seg in en_segments:
        for fix in (seg.get("fixes") or ()):
            fixes.append("%s：%s" % (seg["id"], fix))
    declared = None
    try:
        declared = int(str(header.get("segments") or "").strip())
    except (TypeError, ValueError):
        declared = None
    # ★ 面板拿到的永远是**源文**（①-A 的文本框），而 ①-B 真正吃的是上游
    #   「剧本英译」节点算出来的英文单版。两者语言不同是设计使然，不是故障 ——
    #   但报告里得说清楚，否则用户会以为英译没生效。
    if re.search(r"[\u4e00-\u9fff]", re.sub(r"<d>.*?</d>", "", text or "",
                                            flags=re.S | re.I)):
        issues = list(issues) + [
            "面板这里显示的是**源文**（含中文）。①-A 与 ①-B 之间的「剧本英译」"
            "节点会在注入「H3 导演」前把它改写成英文单版（SKILL §1.3，"
            "台词 <d>…</d> 保留原语言）；英译结果与逐块情况见那个节点的"
            "「英译报告」输出口，以及 ①-B 的「解析报告」。"]
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
        # ★ ref_slots / ref_slot_count **不下发**（2026-09-24 18:23 回退）：
        #   面板按用户定版只显示 6 项头部，不渲染参考图槽；需要槽位信息的
        #   调用方（参考图版块、check_workflow）走 segments[*].pictures 或
        #   prompt_pack.pack_info_json 即可，不改后端保持最小改动面。
        "total_new_seconds": round(total_new, 3),
        "total_gen_seconds": round(total_gen, 3),
        # 会话名由后端算好下发：H3Director 就是拿它当 chain 的 session 落盘的。
        # 面板以前自己猜（读 session_name 控件，空着就退 my_chain），猜错就
        # 查到一个不存在的会话 —— 时间线全空、hover 预览没视频。
        # 候选顺序必须和 H3PromptPackParser 一模一样（控件 → Project），
        # 否则解析器和预览会算出两个名字。
        "session_name": safe_session_name(session_name, header.get("project")),
        "issues": list(issues),
        "fixes": fixes,
        # ★ 这份 PACK 里**真的有**哪几侧内容。现行 minimax-h3-shot-segment 技能
        #   只交付英文单版（没有 [1] 中文版 / [2] 英文版 分块），此时 has_zh 为
        #   False —— 前端据此**藏掉「中文 ZH」标签**。
        #   以前不区分：zh 侧解析在单语 PACK 上会退回英文内容，于是"中文"标签里
        #   显示的是英文；用户在那儿一编辑，lang="zh" 回写还会打到英文段块上
        #   （实测 PACK 44079 → 31643 字符、S01 英文正文清零）。
        "has_en": bool(pp.lang_section_available(text or "", "en")),
        "has_zh": bool(pp.lang_section_available(text or "", "zh")),
        # 完整 PACK 文本（头部 + 段标题 + 六段正文 + END OF PACK）。
        # 面板只会把每段正文送进 H3，但**人**要能核对的东西在头部和段标题里：
        # 段数、总时长、以及 `S02 / 11s+1.6=12.6s` 这种一眼看出规划对不对的
        # 写法。以前这两样根本没产出，只能从正文里数时间码反推。
        "pack_text": pp.pack_text(
            en_segments if en_segments else segs, en_meta,
            project=header.get("project") or session_name or None,
            mode=header.get("mode") or None,
            aspect=header.get("aspect") or None),
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


def _loras_cached(ttl: float = 10.0) -> dict:
    """/h3/loras 的带缓存版本。

    LoRA 列表来自 ``folder_paths.get_filename_list`` —— 那是一次目录扫描，
    而面板打开下拉框、每次重建参考图版块都会重拉一遍。用户在渲染中途往
    loras 目录里丢文件是常事，所以不能永久缓存，用 10s TTL：既压掉同一
    秒内的重复扫描，又不会让人等太久才看到新文件。
    """
    store = _lora_cache
    now = time.time()
    with _preview_lock:
        hit = store.get("v")
        if hit is not None and (now - hit[0]) < ttl:
            return hit[1]
    names = _lora_list()
    payload = {"ok": True, "loras": names, "suggested": _h3_lora_suggest(names)}
    with _preview_lock:
        store["v"] = (now, payload)
    return json.loads(json.dumps(payload))


_lora_cache = {}


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


def _input_images_cached(ttl: float = 8.0) -> list:
    """/h3/input_images 的带缓存版本（8s TTL）。

    这一段是所有只读端点里最贵的：``os.walk`` 整棵 input 树 + **每个文件一次
    ``os.path.getmtime`` 系统调用**，只为拿到一个按新到旧排序的文件名列表。
    input 目录往往有几百上千张图，而面板每重建一次参考图版块就拉一次。
    TTL 取 8s：同一轮编辑里的重复请求全部命中，而用户刚拖进来的新图最多
    8 秒后就会出现在列表里 —— 这个延迟对"挑一张参考图"完全无感。
    """
    now = time.time()
    with _preview_lock:
        hit = _input_cache.get("v")
        if hit is not None and (now - hit[0]) < ttl:
            return [dict(item) for item in hit[1]]
    names = _input_images()
    with _preview_lock:
        _input_cache["v"] = (now, names)
    return [dict(item) for item in names]


_input_cache = {}


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

        from . import session as _sess
        # 当前真实落盘根 = 渲染写它的那个根，同一个函数算的，不会再分叉。
        roots = [_sess.store_root()]
        # 历史遗留根：与当前 output_directory 同源推导（<base_path>/output）。
        base = getattr(folder_paths, "base_path", "") or ""
        if base:
            legacy = os.path.join(base, "output", _sess.ROOT)
            if os.path.normcase(legacy) not in [os.path.normcase(r) for r in roots]:
                roots.append(legacy)
    except Exception:
        return []
    return [root for root in roots if os.path.isdir(root)]


def _find_session(name: str):
    """按会话名找目录。

    ★ **必须用 session.sanitize 本尊**。这里原来抄了半份正则（只有
      ``[^A-Za-z0-9._-]+``，缺了后面的 ``strip("._")``），于是中文会话名
      ``曹贼的性价比_v18_en`` 被算成 ``_v18_en``，而 Director 真正建的目录是
      ``v18_en``（前导下划线被 strip 掉了）——差一个字符，永远找不到，时间线
      模块就会一直显示「会话目录不存在」。

    ★ 直接从 ``session`` 取，不绕 ``common``：``common.sanitize`` 只是同一个对象
      的转发，而 common 顶部要 import torch。面板这条 HTTP 路径没有理由为了一个
      纯字符串函数把整条 torch 依赖挂上（离线测试也因此能单独跑 routes）。
      「只有一套实现」这件事由 :func:`_session_name_homogeneous` 持续守着。
    """
    from .session import sanitize

    raw = (name or "").strip()
    candidates = []
    for cand in (raw, sanitize(raw)):
        if cand and cand not in candidates:
            candidates.append(cand)
    # ★ 这个名字来自 HTTP 查询参数，会直接拼进路径。**不许往上走**。
    #   ``<总根>/<剧本名>/<版本名>`` 之后面板可以填 "剧本/版本" 这种带分隔的
    #   精确名，所以不能一律禁止分隔符，但 ".." 与绝对路径必须挡掉：
    #   _session_state 会把命中目录列出来并回给前端。
    candidates = [c for c in candidates
                  if not os.path.isabs(c)
                  and ".." not in os.path.normpath(c).split(os.sep)]
    # 兜底：目录名被截断到 64 字符时，按前缀再找一次
    if raw and len(raw) > 64:
        candidates.append(sanitize(raw)[:64])
    flat = sanitize(raw).lower().replace("_", "")

    # ★ _session_dirs() 提到循环外：它每次都要 import folder_paths、读配置、
    #   对每个根做 isdir。原来它在 `for cand in candidates` 里面，一个会话名
    #   要重复算 2~3 遍同样的目录列表（面板每次拉 /h3/session 都重来一轮）。
    #   顺带把只在兜底里用到的 flat 也提到循环外 —— 它同样调了一次 sanitize。
    roots = _session_dirs()
    for root in roots:
        for cand in candidates:
            # 1) 老的单层布局：<总根>/<会话名>
            path = os.path.join(root, cand)
            if os.path.isdir(path):
                return path
            # 2) 现行两层：<总根>/<剧本名>/<版本名>。先按版本名找，再放宽到
            #    "用户填的是剧本名" —— 那种情况下取最新的一版，面板才有东西可显示。
            hit = _nested_session(root, cand)
            if hit:
                return hit
        # 最后再放宽：不区分大小写 + 去下划线比一遍（两种布局都试）
        if flat:
            try:
                for entry in os.listdir(root):
                    path = os.path.join(root, entry)
                    if not os.path.isdir(path):
                        continue
                    if entry.lower().replace("_", "") == flat:
                        return path
                    hit = _nested_session(root, entry, match_parent=True)
                    if hit:
                        return hit
            except OSError:
                pass
    return None


def _nested_session(root, name, match_parent=False):
    """在 ``<总根>/<剧本名>/`` 下面找版本目录。

    ``match_parent=False``：``name`` 是版本名，找 ``<root>/<任意剧本>/<name>``，
    多个剧本命中时取修改时间最新的那个（用户同时在两部剧本里用了 ``r1`` 这种
    通用版本名，这是常事）。
    ``match_parent=True``：``name`` 是**剧本夹名**，返回该剧本下最新的一版 ——
    面板收到的是剧本名时也该有东西可显示，而不是"会话目录不存在"。
    """
    try:
        parents = [name] if match_parent else os.listdir(root)
    except OSError:
        return None
    best, best_mtime = None, -1.0
    for parent in parents:
        base = os.path.join(root, parent)
        if not os.path.isdir(base):
            continue
        if match_parent:
            leaves = [os.path.join(base, sub) for sub in sorted(os.listdir(base))
                      if os.path.isdir(os.path.join(base, sub))]
        else:
            leaves = [os.path.join(base, name)]
        for leaf in leaves:
            try:
                if not os.path.isdir(leaf):
                    continue
                m = os.path.getmtime(leaf)
            except OSError:
                continue
            if m > best_mtime:
                best, best_mtime = leaf, m
    return best


def _ffprobe_seconds(path):
    """视频时长（秒）。

    ★ 必须按 (path, mtime, size) 缓存，且要**在起子进程之前**查。
      ffprobe 是一次 fork + 一次进程等待，最快也要几十毫秒；而 /h3/session
      在渲染途中被前端反复拉取（每段完成一次、断点渲染前一次、点刷新一次），
      一条 6 段的会话就是每次 6 个子进程。渲染正忙时这些 fork 还会和采样
      抢 CPU。段文件一旦落盘就不再改动（重渲会换 mtime），所以按 mtime+size
      缓存是安全的 —— 内容变了这两个值必然变。

      缓存无上限地按路径累积也没问题：会话目录里的段数是有限的，且用 LRU
      兜住极端情况。
    """
    try:
        st = os.stat(path)
        stamp = (st.st_mtime, st.st_size)
    except OSError:
        stamp = None
    key = os.path.abspath(path)
    if stamp is not None:
        hit = _cache_get(_ffprobe_cache, key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    try:
        import subprocess

        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=20)
        value = round(float(out.stdout.strip()), 3)
    except Exception:
        value = None
    if stamp is not None:
        # 探测失败（返回 None）也缓存：ffprobe 缺失或文件损坏时，重试一万次
        # 也是同样的结果，没必要每次请求都再 fork 一次然后失败。
        _cache_put(_ffprobe_cache, key, (stamp, value), maxsize=512)
    return value


_ffprobe_cache = {}


def _load_panel_meta(path: str) -> dict:
    """读面板元数据：{段号: {"prompt": 摘要, "speakers": [...], "seconds": 秒}}。

    清完目录后只剩视频，时间线就变成一排没有信息的缩略图。这里把每段「讲了
    什么 / 谁在说话」从存档里捞回来 —— 优先 panel.json（精简），退回
    storyboard.json / shots.json。都没有就返回空，面板只显示时长，不会报错。

    ★ 按 (文件, mtime, size) 缓存：/h3/session 会被反复拉，而这三个存档
      文件在一次渲染过程中只被后端写一次（写完 mtime 就变了，缓存自动失效），
      读侧却要反复 open + json.load。storyboard.json 动辄几百 KB。
    """
    for fname in ("panel.json", "storyboard.json", "shots.json"):
        full = os.path.join(path, fname)
        if not os.path.isfile(full):
            continue
        try:
            st = os.stat(full)
            stamp = (st.st_mtime, st.st_size)
        except OSError:
            continue
        key = full
        hit = _cache_get(_panel_meta_cache, key)
        if hit is not None and hit[0] == stamp:
            rows_out = hit[1]
            if rows_out is not None:
                # ★ 缓存里存的是「键值对列表」的 JSON 字符串，返回时反序列化出
                #   一份全新对象。两个细节都不能省：
                #     · 存 list-of-pairs 而不是 dict —— json 会把 int 键变成
                #       字符串，"1" 永远匹配不上调用方的 panel.get(1)，时间线
                #       的每段摘要会全空。
                #     · 反序列化而不是浅拷贝 —— out[i]["speakers"] 是 list，
                #       浅拷贝下调用方 append 一次说话人就写进缓存本体，
                #       之后所有请求都带着这条脏数据（面板正是读它画徽标）。
                return {int(k): v for k, v in json.loads(rows_out)}
            continue                      # 缓存过一次「解析不出内容」
        try:
            with open(full, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            _cache_put(_panel_meta_cache, key, (stamp, None), maxsize=32)
            continue
        rows = None
        if isinstance(data, dict):
            rows = (data.get("shots_info") or data.get("shots")
                    or data.get("segments"))
        elif isinstance(data, list):
            rows = data
        if not rows:
            _cache_put(_panel_meta_cache, key, (stamp, None), maxsize=32)
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
            _cache_put(_panel_meta_cache, key,
                       (stamp, json.dumps([[i, out[i]] for i in sorted(out)])),
                       maxsize=32)
            return out
    return {}


_panel_meta_cache = {}


def _view_subfolder(path):
    """会话目录相对 **ComfyUI 输出根** 的路径（``/view?subfolder=`` 要的正是它）。

    ★ 以前这里写的是 ``relpath(path, dirname(dirname(path)))`` —— 单层布局下
      分母正好是输出根，两层布局（``<输出根>/h3_continuous/<剧本>/<版本>``）下
      分母变成了 ``h3_continuous``，算出来的 subfolder 少一层，
      于是每一段的预览图 404，面板只剩一排空白缩略图。相对基准只有一个正解：
      ``session.output_root()``。
    """
    try:
        from . import session as _sess

        root = _sess.output_root()
    except Exception:
        root = os.path.dirname(os.path.dirname(os.path.normpath(path)))
    return os.path.relpath(path, root).replace("\\", "/")


def _script_of(session_dir):
    """会话目录所属的剧本夹名；老单层布局或 helper 不可用时返回 ''。"""
    try:
        from . import session as _sess

        return _sess.script_of(session_dir)
    except Exception:
        return ""


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
                   % (fn, _view_subfolder(path)),
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
        # 剧本名（两层布局的上一级夹）；老的单层布局为空字符串。面板用它把
        # 同一部剧本的几个版本归到一起。
        "script": _script_of(path),
        "dir": path,
        "segments": segs,
        "done": done,
        "joined": (os.path.basename(joined) if os.path.exists(joined) else ""),
        "manifest_segments": len(records),
        # 落盘名与查找名是否共用**同一个** sanitize 函数对象。
        # 这两侧一旦分叉，渲染写 A 目录、面板查 B 目录，表现为「断点渲染又从第 1
        # 段开始」以及修复节点抛 "segment 1 has no handoff clip"。以前靠 import
        # 期的 monkey-patch 缝合（顺序相关，随时会解），现在 common 里已经没有
        # 第二套正则，这个字段退化成"哪天有人再抄一份正则"的哨兵。
        "name_homogeneous": _session_name_homogeneous(),
    }


def _session_name_homogeneous() -> bool:
    """True = 只有一个归一化实现，且 common 用的就是它本尊。"""
    try:
        from . import common as _common
        from . import session as _sess

        return getattr(_common, "sanitize", None) is _sess.sanitize
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------
_registered = False


# ---------------------------------------------------------------------------
# /h3/pack_rebuild —— PACK 实时编辑器：把编辑后的六字段回写进整份 PACK
# ---------------------------------------------------------------------------
def _reject_missing_lang(text: str, lang):
    """这份 PACK 没有 ``lang`` 那一侧内容时，返回"不改文本"的拒绝结果。

    两个写回端点（``_pack_rebuild`` / ``_pack_apply_block``）的守卫逐字相同，
    合成一处 —— 拦截判据一旦分叉，就会出现"一个拦了另一个没拦"的静默数据
    损坏（实测 PACK 44079 → 31643 字符、S01 英文正文清零）。
    """
    if pp.lang_section_available(text or "", lang):
        return None
    return {
        "ok": False, "text": text or "", "unchanged": True,
        "warning": "这份 PACK 没有「%s」版块，本次编辑未写回（原文一字未改）。"
                   "现行 minimax-h3-shot-segment 技能只交付英文单版。"
                   % ("中文" if lang == "zh" else "英文"),
    }


def _pack_rebuild(text: str, segments: list, lang=None) -> dict:
    """segments=[{"index": n, "fields": {...}}] → 重拼后的 PACK 文本。

    ``lang`` 透传给 ``pp.rebuild_pack``：``"zh"``/``"en"`` 命中对应语言版块，
    ``None`` 命中最后一次出现的段标题（双语 PACK 默认英文版）。

    ★ 指定了 ``lang`` 但这份 PACK 里没有那一侧内容时，**不改文本**，回一个
    带 ``warning`` 的结果让前端说清楚。见 ``_reject_missing_lang``。
    """
    blocked = _reject_missing_lang(text, lang)
    if blocked is not None:
        return blocked
    rebuilt = pp.rebuild_pack(text or "", segments, lang)
    return {"ok": True, "text": rebuilt}


def _pack_apply_block(text: str, blocks: list, lang=None) -> dict:
    """blocks=[{"index": n, "body": "整段六段式正文"}] → 原样整块替换后的 PACK。

    「一个文本框装整段提示词」的编辑通道：不拆字段、不规范化，body 原样写回。
    ``lang`` 的缺失侧拦截同 ``_pack_rebuild``。
    """
    blocked = _reject_missing_lang(text, lang)
    if blocked is not None:
        return blocked
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
            return web.json_response(_cached_preview(text, sess))
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
            return web.json_response(_loras_cached())
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc),
                                      "loras": []}, status=500)

    @routes.get("/h3/input_images")
    async def h3_input_images(request):
        try:
            return web.json_response({"ok": True, "images": _input_images_cached()})
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

    # ------------------------------------------------------------------
    # 导演包（PACK 剧本 + 参考图）导入 / 导出
    # 借鉴 ComfyUI_MiniMaxH3_Director 的 pack.py：zip 安全 + StreamResponse
    # 分块（Windows 上 FileResponse 会给空 body）+ TTL 清理临时导出。
    # ------------------------------------------------------------------
    @routes.post("/h3/pack_export")
    async def h3_pack_export(request):
        """打包当前剧本与参考图。

        body: {"script": "PACK原文", "session_name": "控件值",
               "storyboard": "分镜JSON文本（可省——省了就用 script 现解析）",
               "meta": {...}, "refs": {"1": "in/m.png", ...}}
        参考图路径是 **input 相对路径**（LoadImage 的 widgets_values 原样）。

        storyboard 由服务端生成（pp.to_shots_json，与存档节点同源）：面板
        /h3/pack_preview 下发的是 segments 而不是 shots_json 原文，前端拼不出
        与读档节点完全同规格的那一份。
        """
        from . import pack as pack_mod

        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            storyboard = str(body.get("storyboard") or "")
            script = str(body.get("script") or "")
            if not storyboard.strip() and script.strip():
                storyboard, _script, _sess = pack_mod.storyboard_from_pack_text(
                    script, str(body.get("session_name") or ""))
            if not storyboard.strip():
                raise pack_mod.PackError(
                    "没有可分镜的内容：PACK 原文为空，storyboard 也没给。")
            path = pack_mod.build_pack(
                meta=body.get("meta") or {},
                storyboard=storyboard,
                script=script,
                refs=body.get("refs") or {})
            return web.json_response({"ok": True, "name": path.name,
                                      "bytes": path.stat().st_size})
        except pack_mod.PackError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @routes.get("/h3/pack_download")
    async def h3_pack_download(request):
        """下载导出的 zip。★ 必须分块 StreamResponse：Windows 上
        FileResponse/sendfile 会 200 + 空 body，存出来是 0 字节损坏包。"""
        from . import pack as pack_mod

        name = request.query.get("name", "")
        if pack_mod.PACK_SUFFIX not in name or "/" in name or "\\" in name:
            return web.Response(status=400, text="Bad pack name.")
        path = pack_mod._export_root() / name
        if not path.is_file():
            return web.Response(status=404, text="Pack not found or expired.")
        size = path.stat().st_size
        if size <= 0:
            return web.Response(status=404, text="Pack is empty.")
        resp = web.StreamResponse(status=200, headers={
            "Content-Type": "application/zip",
            "Content-Disposition": 'attachment; filename="%s"' % name,
            "Content-Length": str(size),
            "Cache-Control": "no-store",
        })
        await resp.prepare(request)
        try:
            with path.open("rb") as fh:
                while True:
                    chunk = fh.read(pack_mod.ZIP_STREAM_CHUNK)
                    if not chunk:
                        break
                    await resp.write(chunk)
            await resp.write_eof()
            return resp
        finally:
            pack_mod.purge_exports(keep=name)

    @routes.post("/h3/pack_import")
    async def h3_pack_import(request):
        """导入 .hpack.zip（multipart，字段名 file）。

        参考图复制进 input/h3_director_packs/<包名>/，storyboard.json 落到
        会话目录；返回槽位表供前端改写 LoadImage。
        """
        from . import pack as pack_mod

        try:
            reader = await request.multipart()
        except Exception:
            reader = None
        if reader is None:
            return web.json_response(
                {"ok": False, "error": "请以 multipart 上传 .hpack.zip。"},
                status=400)

        staging = None
        try:
            async for part in reader:
                if part.name != "file":
                    continue
                staging = pack_mod._export_root() / ("up_%d%s" % (
                    int(time.time() * 1000), pack_mod.PACK_SUFFIX))
                with staging.open("wb") as out:
                    while True:
                        chunk = await part.read_chunk(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                break
            if staging is None or not staging.is_file():
                return web.json_response(
                    {"ok": False, "error": "上传里没有 file 字段。"}, status=400)
            result = pack_mod.import_pack(staging)
            return web.json_response({"ok": True, **result})
        except pack_mod.PackError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:                              # pragma: no cover
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)

    _registered = True
    return True


# ---------------------------------------------------------------------------
# 进度事件（Director → 前端时间线）
# ---------------------------------------------------------------------------
EVENT_PROGRESS = "h3.director.progress"


# ---- 相位心跳（2026-10-02）：渲染途中每 4s 广播一次当前相位 ----
# 小马/状态行靠事件驱动；以前事件只在段完成时发，采样+解码+质检的几分钟里
# 前端完全没有生命迹象（用户报「小马在渲染时不动了」）。现在 director 在每段
# 开始时 set_phase，这个心跳线程按固定间隔把相位+已渲秒数持续广播出去。
_phase = {"node_id": None, "phase": None, "t0": None, "extra": None}
_hb_started = False


def set_phase(node_id, phase, **extra):
    """记录当前渲染相位（phase=None 清除）；心跳线程负责持续广播。"""
    global _hb_started
    if not _hb_started:
        _hb_started = True
        import threading
        threading.Thread(target=_heartbeat_loop, daemon=True,
                         name="h3-phase-heartbeat").start()
    _phase["node_id"] = node_id
    _phase["extra"] = dict(extra) if extra else None
    if phase is not None and phase != _phase["phase"]:
        _phase["phase"] = phase
        _phase["t0"] = time.time()
        emit_progress(node_id, event="phase", phase=phase, elapsed=0.0,
                      **(_phase["extra"] or {}))
    if phase is None:
        _phase["phase"] = None


def _heartbeat_loop():
    while True:
        try:
            if _phase["phase"] is not None and _phase["node_id"] is not None:
                emit_progress(_phase["node_id"], event="phase",
                              phase=_phase["phase"],
                              elapsed=round(time.time() - (_phase["t0"] or time.time()), 1),
                              **(_phase["extra"] or {}))
        except Exception:
            pass
        time.sleep(4)


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
