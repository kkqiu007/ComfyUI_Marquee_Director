"""H3 参考图槽位模型 —— **槽位序号 == 提示词里 ``<Picture N>`` 的 N**。

设计来源
--------
``ComfyUI_MiniMaxH3_Director/lib/ref_images.py`` 用**固定编号输入**
``reference_image_0 … reference_image_8``，槽位序号直接就是 ``<Picture N>``
的 N，中间不留空、不重新编号、不压实。本模块把这套语义搬到
``ComfyUI_Marquee_Director``：

* 节点侧仍然是 ComfyUI 的 Autogrow（``ref_images.ref_image_0..8``），
  **输入布局一个字都不改**，既有工作流照常打开、不用重新接线；
* 但取值改成**稀疏字典** ``{槽号: tensor}``，不再把没接线的口压实丢掉。

为什么必须改
------------
旧实现用 ``common.ordered_autogrow()`` 拿到 ``[(序号, tensor), …]`` 之后只留
tensor 组成 list（压实），再按 ``pool[index - 1]`` 取图。于是：:

    接了 ref_image_0,1,2,3,6   →   pool = [图0, 图1, 图2, 图3, 图6]
    <Picture 5> 取 pool[4]     →   实际拿到的是 **ref_image_6** 那张

面板上标 P7 的图被 ``<Picture 5>`` 引用，后面几张整体错位，人物直接换脸。
改成稀疏字典后，``<Picture 5>`` 只会取槽 5；槽 5 没接图就是**明确告警**，
而不是悄悄拿错图继续渲。

兼容性
------
槽位**连续接满**（0..N-1）时，稀疏字典的取图结果与旧 list **完全一致**，
``segment_key`` 里的 images 摘要不变 → 既有会话缓存不会误失效、不会整片重渲。

本模块刻意不 import torch / comfy，便于离线复现与单元测试
（与 ``prompt_pack.py`` 同样的做法：``importlib`` 直接加载就能跑）。
"""

from __future__ import annotations

# 官方 MiniMaxH3ReferenceToVideo 的上限：最多 9 张参考图。
MAX_REFERENCE_IMAGES = 9

# H3 的 VAE 空间下采样 16 倍、再 2×2 patchify → 画布必须是 32 的倍数。
CANVAS_STRIDE = 32

# 节点输入侧的键名前缀（Autogrow TemplatePrefix）。
AUTOGROW_PREFIX = "ref_image_"
# 下游 / 文案里用的语义前缀（与参照包保持一致）。
REF_IMAGE_KEY_PREFIX = "reference_image_"


def slot_label(index: int) -> str:
    """槽号（1 起）→ 面向用户的标签：图片1 … 图片9。"""
    return "图片%d" % int(index)


def autogrow_ordinal(key) -> int | None:
    """``ref_image_3`` / ``ref_images.ref_image_3`` → 3（**0 起**）。

    不是参考图键就返回 ``None``，调用方据此跳过（Autogrow 组里理论上只有
    参考图，但防御一下没坏处 —— 名字对不上就别硬当槽位用）。
    """
    name = str(key or "")
    tail = name.rsplit(".", 1)[-1]
    if not tail.startswith(AUTOGROW_PREFIX):
        return None
    digits = tail[len(AUTOGROW_PREFIX):]
    if not digits.isdigit():
        return None
    return int(digits)


def collect_slots(ref_images) -> dict:
    """Autogrow 输入 → **稀疏**槽位字典 ``{1 起槽号: tensor}``。

    只收真正连上且非空的口。``None`` 或空 batch 一律视为"这槽没图"，
    **不参与压实** —— 这正是与旧 ``ordered_autogrow()`` 的根本区别。
    """
    slots: dict = {}
    for key, value in (ref_images or {}).items():
        ordinal = autogrow_ordinal(key)
        if ordinal is None:
            continue
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                if int(shape[0]) <= 0:
                    continue          # 空 batch = 没图
            except (TypeError, ValueError, IndexError):
                pass
        slot = ordinal + 1            # 0 起的输入序号 → 1 起的 <Picture N>
        if slot < 1 or slot > MAX_REFERENCE_IMAGES:
            continue
        slots[slot] = value
    return slots


def ordered_slot_tensors(slots) -> list:
    """稀疏字典 → 按槽号升序的 tensor 列表（"没指定就用全部"的默认集）。"""
    return [slots[k] for k in sorted(slots or {})]


def picture_numbers_of_shots(shots) -> list:
    """所有分镜引用过的 ``<Picture N>`` 编号（去重、升序）。

    分镜里的字段就是 ``ref_images``（prompt_pack 解析 ``<Picture N>`` 得来）。
    这里只认数字编号；脚本路径那种字符串条目不参与对账。
    """
    needed: list = []
    for shot in shots or []:
        if not isinstance(shot, dict):
            continue
        for item in (shot.get("ref_images") or []):
            if item is None or isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                num = int(item)
                if num not in needed:
                    needed.append(num)
    needed.sort()
    return needed


def resolve_shot_refs(shot, slots, warnings=None, loader=None) -> list:
    """按 ``<Picture N>`` 的编号从稀疏槽位取这一镜要用的参考图。

    * ``shot`` 没有 ``ref_images`` 键 / 值为 ``None`` → 用全部已接槽位（旧行为）。
    * 显式 ``[]`` → 这一镜不用参考图。
    * 引用了没接图的槽 → **记 warning 并跳过**，不中断整条流水线
      （PACK 解析出的分镜几乎都带 ``<Picture N>``，一上来就 raise 等于
      打开工作流点一下运行就崩）。
    * 编号越界（<1 或 >9）→ 同样记 warning。旧版对这种情况是 raise，
      结果是一片没渲完先崩；这里改成告警更符合"先出片再挑毛病"。

    ``loader`` 用于字符串条目（脚本里直接写图片路径），由调用方注入，
    避免本模块反向依赖 director（会循环导入）。
    """
    if not isinstance(shot, dict) or "ref_images" not in shot:
        return ordered_slot_tensors(slots)

    raw = shot.get("ref_images")
    if raw is None:
        return ordered_slot_tensors(slots)
    if isinstance(raw, (str, int, float)) and not isinstance(raw, bool):
        raw = [raw]
    if not isinstance(raw, list):
        return ordered_slot_tensors(slots)

    picks: list = []
    missing: list = []
    oob: list = []
    for item in raw:
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            num = int(item)
            if num < 1 or num > MAX_REFERENCE_IMAGES:
                oob.append(num)
                continue
            tensor = (slots or {}).get(num)
            if tensor is None:
                missing.append(num)
                continue
            picks.append(tensor)
            continue
        path = str(item).strip()
        if not path:
            continue
        if loader is None:
            continue
        tensor = loader(path)
        if tensor is not None:
            picks.append(tensor)

    if warnings is not None:
        shot_id = shot.get("id") or "?"
        if missing:
            warnings.append(
                "分镜 %s 引用了 %s，但对应槽位没接图 → 本镜按无参考图渲染"
                "（在参考图面板给这些槽位选图后再跑）"
                % (shot_id,
                   "、".join("<Picture %d>" % n for n in missing)))
        if oob:
            warnings.append(
                "分镜 %s 的 ref_images 越界（%s）：H3 只支持 "
                "<Picture 1>~<Picture %d>。"
                % (shot_id, "、".join(str(n) for n in oob),
                   MAX_REFERENCE_IMAGES))
    return picks


def plan_long_edge(height: int, width: int, max_px) -> tuple:
    """算出"只缩不放 + 对齐 32"之后的目标尺寸。纯算术，可离线测。

    返回 ``(new_h, new_w, changed)``。``max_px`` 为 ``None``/0 时只做对齐。
    """
    height, width = int(height), int(width)
    scale = 1.0
    if max_px:
        limit = int(max_px)
        longest = max(height, width)
        if longest > limit:
            scale = float(limit) / float(longest)

    def snap(value: int) -> int:
        if scale == 1.0:
            snapped = int(round(value / CANVAS_STRIDE) * CANVAS_STRIDE)
        else:
            snapped = int(round(value * scale / CANVAS_STRIDE) * CANVAS_STRIDE)
        return max(CANVAS_STRIDE, snapped)

    new_h, new_w = snap(height), snap(width)
    return new_h, new_w, (new_h != height or new_w != width)


def limit_ref_image_dict(images: dict, max_px, fit=None) -> tuple:
    """对稀疏槽位字典逐张做长边缩放（只缩不放 + 对齐 32）。

    ``fit`` 是真正干活的缩放函数，签名 ``(tensor, max_px) -> (tensor, changed,
    orig_wh, new_wh)``，由 ``director._fit_ref_image`` 提供（它要用
    ``comfy.utils.common_upscale``，本模块不能依赖）。不传就原样返回 ——
    这样本模块在没有 torch/comfy 的环境里也能 import 和被测试。
    """
    if not images:
        return images, 0
    if fit is None:
        return images, 0
    changed = 0
    out: dict = {}
    for slot, tensor in images.items():
        fitted, did_change = fit(tensor, max_px)[:2]
        if did_change:
            changed += 1
        out[slot] = fitted
    return out, changed


def slot_labels(slots, ref_classify) -> dict:
    """槽号 → 显示名。有分类就用「角色·王总」，没有就退回「图片 N」。

    旧实现返回 list（按下标对应压实后的池），稀疏模型下必须用字典，
    否则槽位跳空时标签会整体错位。
    """
    labels = {slot: slot_label(slot) for slot in (slots or {})}
    text = str(ref_classify or "").strip()
    if not text:
        return labels
    try:
        import json
        rows = json.loads(text)
    except (TypeError, ValueError):
        return labels
    if not isinstance(rows, list):
        return labels
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            slot = int(row.get("slot"))
        except (TypeError, ValueError):
            continue
        if slot not in labels:
            continue
        kind = str(row.get("kind") or "").strip()
        name = str(row.get("name") or "").strip()
        label = "·".join(part for part in (kind, name) if part)
        if label:
            labels[slot] = label
    return labels


def reconcile(slots, shots) -> dict:
    """参考图对账：剧本要什么、实际接了什么，差在哪。

    返回 ``{"needed", "connected", "missing", "unused"}``：

    * ``missing`` —— 剧本引用了 ``<Picture N>`` 但槽位 N 没接图
      （H3 会自己"补"出角色，成片人物和参考图对不上）
    * ``unused``  —— 接了图但剧本里没有任何 ``<Picture N>`` 指向它
      （H3 不会把它画进画面，接了等于没接）
    """
    needed = picture_numbers_of_shots(shots)
    connected = sorted(slots or {})
    return {
        "needed": needed,
        "connected": connected,
        "missing": [n for n in needed if n not in (slots or {})],
        "unused": [n for n in connected if n not in needed],
    }
