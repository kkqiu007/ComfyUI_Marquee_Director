# -*- coding: utf-8 -*-
"""修脸质量判据 —— 回答"这一遍修完到底是变好了还是变坏了"。

Carasibana/ComfyUI-H3-FaceRefine（MIT）把"修"这条链路做完了：逐帧检测 → 裁剪 →
注入 latent → 低 denoise 重采 → 贴回。整条链路里**没有一步会停下来看结果**——
denoise 给多少就修多少，修完就贴回去。于是 denoise 调高一档，效果从"脸清楚了"
变成"蜡像脸"，而流程不会有任何反应。本模块补的就是这一段：修完之后量一量，
不合格就换种子重来。

★ 判据的参照系是全模块最容易搞错、也最关键的一个决定
------------------------------------------------------------------
量的是「**修后 vs 修前的同一帧**」，不是「修后 vs 参考图」。

理由：修脸要解决的是"H3 把远景小脸渲成糊块"，正确的结果是**这张脸变得更清楚、
更像它本来的那个人**。如果拿参考图当标尺，那么参考图是侧脸、这一帧是正脸时，
正确结果反而会被判成"身份漂移"，于是自动重试永远修不好一张好脸。反过来，
一张已经糊掉的脸与它自己修前的版本相比，ArcFace 余弦本来就低 —— 修好了会**上升**。
所以"修后 - 修前"这个差值，既能认出修好了，也能认出修过头了。

三条判据（都按"越高越好"归一）：

1. **identity** —— InsightFace ArcFace 512-D 嵌入的余弦相似度，帧内比较。
   参考量级：GAN 类修脸会把余弦压低 0.03~0.08，扩散类压得更狠（0.12~0.19）。
   所以**小幅下降是正常的**，我们只拦"掉得离谱"的那种。
2. **detail** —— 人脸区域的高频能量（拉普拉斯方差）。修糊了会**掉**。
   这是识别蜡像脸的主力：ArcFace 对轻微的"变光滑"其实相当宽容，
   而人眼对"塑料感"极其敏感 —— 两者必须同时看。
3. **stability** —— 相邻帧之间的 ArcFace 余弦均值。脸在逐帧"沸腾"
   （每帧微调一点、连起来像水在流）时，均值会明显低于静止镜头。

★ 依赖缺失一律降级，不报错
------------------------------------------------------------------
insightface / onnxruntime 装不上是常态（insightface 在 Windows 上的安装史
相当有名）。缺了就只剩 detail + 两条纯像素判据（帧间差），
功能降级但仍能拦住最常见的"修成蜡像脸"。与本包"缺依赖降级而非报错"的
既有纪律一致（见 ref_images.py / subtitle_qc.py）。
"""

from __future__ import annotations

import numpy as np
import torch

from .common import log

# ArcFace 余弦：同一人通常 >0.5，0.30 是业界公认的通过线。
# 我们的判据不是"是不是同一个人"，而是"有没有比修前掉得离谱"，所以默认很宽松。
IDENTITY_FLOOR = 0.72

# 高频能量比：修后 / 修前。低于这个数就是画面被抹平了。
# ★ 阈值取 0.92 而不是 1.0：**持平（ratio≈1.0）不算失败**。
#   "这一遍几乎没改动"是"白跑一趟"，由上层的重试逻辑去发现和处理
#   （换种子），不是质量失败 —— 否则会把一张本来就够清楚的脸反复重渲，
#   烧掉几次采样却永远过不了。真正的失败只有"越修越糊"这一种。
DETAIL_RATIO_FLOOR = 0.92

# 相邻帧余弦均值：低于这个值判定脸在沸腾（逐帧抖动）。
STABILITY_FLOOR = 0.55

# 参与判定的帧上限。判据只需要一个**有代表性的样本**，不需要逐帧全跑：
# ArcFace 每帧 ~6ms GPU，但 200 帧就是 1.2s，而成片本来就几分钟。
# 取 24 帧、均匀铺开，足以看出趋势，且省掉几十倍时间。
SAMPLE_FRAMES = 24

# 极端情况下的保护：一张脸修到"跟修前一模一样"也不该被判失败
# （说明这一遍白跑了），所以给出相等判据时返回 None（不判）。
_EPS = 1e-6

_ARCFACE = {}


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    """[H,W,3] float 0..1 → HWC uint8 RGB。"""
    arr = (img[..., :3].clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def _to_bgr(arr: np.ndarray) -> np.ndarray:
    return arr[..., ::-1].copy()


def sample_indices(total: int, want: int = SAMPLE_FRAMES) -> list:
    """均匀取样下标，至少两个（identity 与 stability 都需要相邻帧）。"""
    if total <= 0:
        return []
    if total <= want:
        return list(range(total))
    step = total / float(want)
    idx = [int(round(i * step)) for i in range(want)]
    # 去重并夹回范围，同时保证首尾各有一个（首帧常是段首回放，尾帧接下一段）。
    out = sorted({min(total - 1, max(0, i)) for i in idx})
    if out[0] != 0:
        out.insert(0, 0)
    if out[-1] != total - 1:
        out.append(total - 1)
    return out


def _hf_of(frames) -> float:
    """一批 ``(H, W, 3)`` 帧的平均高频能量。

    逐帧走 ``highfreq_energy``（二维契约），不走"整批一次拉普拉斯"——
    后者会把 batch 轴一起当空间轴，算出来的是帧间差而不是清晰度。
    """
    # 入参是**批量**帧：(N, H, W, 3)，所以 ndim 是 4 而不是 3。
    # 这里只校验"最后一维是 3 通道"与"至少一帧"，不校验总维数 ——
    # 早先写成 `frames.ndim != 3` 会把正常的 (N,H,W,3) 整批拒掉、返回 0.0，
    # 于是高频判据恒为"量不出来"，质检静默失效。
    if frames is None:
        return 0.0
    # 接受批量 ndarray 或逐帧 list（judge 里按采样帧逐张裁好再传进来）。
    if isinstance(frames, list):
        if not frames:
            return 0.0
        vals = [highfreq_energy(_gray(np.asarray(f))) for f in frames]
    else:
        if frames.size == 0 or frames.ndim < 3 or frames.shape[-1] != 3:
            return 0.0
        vals = [highfreq_energy(_gray(f)) for f in frames]
    vals = [v for v in vals if v > 0.0]
    return float(np.mean(vals)) if vals else 0.0


def _crop_face_box(frames: np.ndarray, box, pad: float = 0.45) -> np.ndarray:
    """按检测框外扩后裁出人脸区域（外扩是为了把头发/下巴纳入高频统计）。"""
    h, w = frames.shape[1:3]
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    bw, bh = x1 - x0, y1 - y0
    cx, cy = x0 + bw * 0.5, y0 + bh * 0.5
    half_w, half_h = bw * (0.5 + pad), bh * (0.5 + pad)
    a = int(max(0, round(cx - half_w)))
    b = int(min(w, round(cx + half_w)))
    c = int(max(0, round(cy - half_h)))
    d = int(min(h, round(cy + half_h)))
    if b - a < 8 or d - c < 8:            # 脸太小，高频统计不可靠
        return frames
    return frames[:, a:b, c:d]


def highfreq_energy(gray2d: np.ndarray) -> float:
    """拉普拉斯响应的方差 —— 经典的"清晰度"代理指标。

    不用 FFT、不用梯度模：4 邻域拉普拉斯在这个尺寸上足够稳，
    而且比 Sobel 梯度更不受压缩噪点影响（高频噪点会被二阶差分压下去）。

    ★ 参数契约是**二维灰度图**。这里刻意不做"按 ndim 猜语义"的自适应：
      猜错过一次，而且错得很隐蔽。``_gray`` 保留前导 batch 维，单帧时形状是
      ``(1, H, W)``，ndim 与"彩色图 (H, W, C)"**同为 3**。一旦按 ndim==3
      走彩色分支，就会把 W 方向整列平均掉，H=64 的图算出 (1,64)，
      再切片得到空数组，``var()`` = nan；而 nan 参与任何比较都是 False，
      判据会**静默失效**，表现为"质检永远通过"。比抛异常难查得多。
      所以降维责任交给调用方（``judge`` 逐帧传单张二维图），这里只做一次
      isfinite 兜底。
    """
    g = np.asarray(gray2d, dtype=np.float32)
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (-4.0 * g[1:-1, 1:-1]
           + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    v = float(lap.var())
    return v if np.isfinite(v) else 0.0


def _gray(frames: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 亮度，整数运算够用。"""
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def load_arcface(model: str = "buffalo_l"):
    """加载 InsightFace 分析器。不可用时返回 ``None``，由调用方降级。

    ★ 强制 CPU provider：这是社区反复踩的坑。insightface 默认要 CUDA，
      而 ComfyUI 自己也用 onnxruntime —— 两个进程/两个 session 抢同一块卡，
      轻则抢显存导致 OOM，重则**静默退回 CPU**，把一次几秒的判据拖成几分钟，
      而且没有任何报错。Duanyll 的 InsightFace Similarity 节点也是刻意强制 CPU
      的，理由相同。两张图（几十帧）的量级下 CPU 完全够用。
    """
    if model in _ARCFACE:
        return _ARCFACE[model]
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except Exception as exc:                                  # pragma: no cover
        log("face_qc: insightface 不可用（%s），质量判据降级为纯像素模式。"
            "要启用身份判据请在 ComfyUI 的 Python 里 pip install insightface。", exc)
        _ARCFACE[model] = None
        return None
    try:
        app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
    except Exception as exc:                                  # pragma: no cover
        log("face_qc: ArcFace(%s) 初始化失败（%s），质量判据降级为纯像素模式。",
            model, exc)
        _ARCFACE[model] = None
        return None
    _ARCFACE[model] = app
    return app


def _embed(app, img_u8: np.ndarray):
    """取最大人脸的归一化嵌入；检测不到返回 None。"""
    try:
        faces = app.get(_to_bgr(img_u8))
    except Exception:                                         # pragma: no cover
        return None, None
    if not faces:
        return None, None
    face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        raw = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(raw)
        if norm < _EPS:
            return None, face.bbox
        emb = raw / norm
    return emb.astype(np.float32), face.bbox


def _cosine(a, b) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < _EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def judge(before: torch.Tensor, after: torch.Tensor, boxes, *,
          model: str = "buffalo_l", identity_floor: float = IDENTITY_FLOOR,
          detail_floor: float = DETAIL_RATIO_FLOOR,
          stability_floor: float = STABILITY_FLOOR) -> dict:
    """量一次修脸的质量。

    ``before`` / ``after`` 是**同一批**帧（``after`` 长度可能更短，贴回时
    截断过），``boxes`` 是 ``face_refine.track_and_crop`` 记下的每帧检测框。
    返回一个 dict，永远有 ``ok`` / ``reason`` 两个键，调用方不必处理异常。

    判据全过 → ``ok=True``；任一不达标 → ``ok=False`` 且 ``reason`` 说明是哪一条。
    """
    n_before = int(before.shape[0])
    n_after = min(int(after.shape[0]), n_before)
    if n_after < 1 or not boxes:
        return {"ok": True, "reason": "", "degraded": True, "sampled": 0}

    idx = sample_indices(n_before)
    idx = [i for i in idx if i < n_after]
    if len(idx) < 2:
        # 只剩一帧可判：做不了稳定性，也做不了前后对比，退回"不判"。
        return {"ok": True, "reason": "", "degraded": True, "sampled": len(idx)}

    b_u8 = [_to_uint8(before[i]) for i in idx]
    a_u8 = [_to_uint8(after[i]) for i in idx]
    # 裁到人脸区域再量高频：整幅画面里有大量与脸无关的纹理（衣服、背景、
    # 噪点），它们的能量会淹没脸本身的糊与不糊。
    crops_b = [_crop_face_box(f, boxes[i]) for f, i in zip(b_u8, idx)]
    crops_a = [_crop_face_box(f, boxes[i]) for f, i in zip(a_u8, idx)]

    detail_b = _hf_of(crops_b)
    detail_a = _hf_of(crops_a)
    detail_ratio = (detail_a / detail_b) if detail_b > _EPS else 1.0

    # 帧间差：逐帧平均绝对差。修糊了它会掉（画面变平），也能顺手抓住"整段
    # 变成一张静止图"这种病态情况。不用 ArcFace 也能用。
    wob_b = float(np.mean([np.abs(crops_b[i + 1].astype(np.float32)
                                  - crops_b[i].astype(np.float32)).mean()
                           for i in range(len(crops_b) - 1)]))
    wob_a = float(np.mean([np.abs(crops_a[i + 1].astype(np.float32)
                                  - crops_a[i].astype(np.float32)).mean()
                           for i in range(len(crops_a) - 1)]))

    result = {
        "ok": True,
        "reason": "",
        "degraded": True,          # 先乐观假设"只能做像素判据"，拿到 ArcFace 再翻
        "sampled": len(idx),
        "detail_before": detail_b,
        "detail_after": detail_a,
        "detail_ratio": detail_ratio,
        "wobble_before": wob_b,
        "wobble_after": wob_a,
    }

    # --- 判据 2（主判据，不依赖任何第三方包）---
    if detail_ratio < detail_floor:
        result["ok"] = False
        result["reason"] = ("脸被抹平了（高频只剩 %.0f%%）：这一遍 denoise 过高，"
                           "把五官细节一起擦掉了" % (detail_ratio * 100.0))

    # --- 判据 1 + 3（需要 InsightFace）---
    app = load_arcface(model)
    if app is None:
        if detail_ratio < detail_floor:
            return result
        # 没有 ArcFace 时也要拦住最明显的"整段变静"。
        if wob_b > _EPS and wob_a < wob_b * 0.55:
            result["ok"] = False
            result["reason"] = ("画面几乎不动了（帧间差只剩 %.0f%%），"
                               "人物疑似僵住" % (wob_a / wob_b * 100.0))
        return result

    result["degraded"] = False
    emb_b, emb_a = [], []
    for fb, fa in zip(crops_b, crops_a):
        eb, _ = _embed(app, fb)
        ea, _ = _embed(app, fa)
        if eb is None or ea is None:
            continue
        emb_b.append(eb)
        emb_a.append(ea)

    if len(emb_b) < 2:
        # 判据 2 已经跑过了；识别不出来不再额外扣分（可能是侧脸/遮挡）。
        return result

    pair = [_cosine(b, a) for b, a in zip(emb_b, emb_a)]
    stab = float(np.mean([_cosine(emb_a[i], emb_a[i + 1]) for i in range(len(emb_a) - 1)]))
    identity = float(np.mean(pair))
    result.update({"identity": identity, "identity_min": float(np.min(pair)),
                   "stability": stab, "faces_scored": len(emb_b)})

    if identity < identity_floor:
        result["ok"] = False
        result["reason"] = ("身份漂移（修前修后余弦 %.3f < %.2f）：这一遍把人修成了"
                           "另一个人，换低 denoise 或换种子" % (identity, identity_floor))
    elif stab < stability_floor:
        result["ok"] = False
        result["reason"] = ("脸在沸腾（相邻帧余弦 %.3f < %.2f）：逐帧改动太大，"
                           "降 denoise 或加大帧间平滑" % (stab, stability_floor))
    return result


def format_report(result: dict, attempt: int = 1, total: int = 1) -> str:
    """给面板/运行报告用的一行摘要。"""
    if not result:
        return ""
    if result.get("degraded"):
        base = "像素判据"
    else:
        base = "ArcFace 判据"
    if total > 1:
        base += " 第 %d/%d 遍" % (attempt, total)
    if not result.get("faces_scored") and not result.get("degraded"):
        base += "（未检到可比对的人脸，已跳过身份判据）"
    if result.get("ok"):
        tail = "通过"
        if "identity" in result:
            tail += "，身份 %.3f" % result["identity"]
        tail += "，细节 %.0f%%" % (result.get("detail_ratio", 1.0) * 100.0)
        return "修脸质检（%s）：%s" % (base, tail)
    return "修脸质检（%s）：%s" % (base, result.get("reason") or "未通过")
