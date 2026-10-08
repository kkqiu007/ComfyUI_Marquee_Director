# -*- coding: utf-8 -*-
"""脸部修复（FaceRefine）—— 借鉴 ComfyUI_MiniMaxH3_Director 的 ``face_refine`` 外接形态。

那个包的 face_refine 又改编自 ``ComfyUI-H3-FaceRefine``（MIT, Carasibana），
这里照的是同一条链路：

    解码成片 → 逐帧检测并跟踪人脸 → 仿射裁出特写画布 → 把裁图注入 AV latent 的
    视频流 → 低 denoise 重采一次 → 解码 → 按仿射逆变换贴回 → 接缝处淡回原图

做成**可选外接节点**，与 ``H3 Refine``（二采）同一套契约：接到 Director 的
``face_refine`` 口才生效，不接（或接了没给 sigmas）就完全不动成片。

★ 接缝是本模块唯一需要自己想的地方，也是它和别的包不一样的地方。
------------------------------------------------------------------
修脸改的是**像素**，而 Marquee 的段间锚点是**latent**：下一段开头钉的是上一段
采样 latent 的尾帧，那里面没有修过脸。于是如果整段照直修，接缝处就是
「上一段没修过的脸 → 这一段修过的脸」的一记跳变 —— 脸会在每个接口处
呼吸一下。

``fade_stitch_at_seams`` 就是为这个存在的：把段首 / 段尾各 N 帧的修脸结果
lerp 回未修的原图，让接缝两侧回到锚点那一版画面。

MiniMaxH3_Director 那里 N 是固定 12 帧（它的上下文窗口是 22 帧）。这里改成
**跟随本段的回放帧数**——Marquee 的段首是 handoff 秒（默认 1.625s ≈ 39 帧）
的整段回放，钉死在 latent 里，淡出范围必须覆盖它，12 帧不够，接缝照样跳。
这是本模块唯一"比原版更贴合"的改动，其余都是照搬。
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from comfy_extras.nodes_custom_sampler import (
    Guider_Basic,
    Noise_RandomNoise,
    SamplerCustomAdvanced,
)

from . import face_qc
from .common import (
    MiniMaxH3ReferenceToVideo,
    generation_length,
    log,
)

# ★ 与 director.py / pack_nodes.py / json_io_nodes.py 一致：各模块自带 CATEGORY，
#   不从 .nodes 取。engine.py 在模块层导入本文件，而 nodes.py 又在自己的模块层
#   导入 engine.py —— 从 .nodes 拿常量会撞上半初始化的模块（nodes.CATEGORY 定义
#   在 `from .engine import` 之后）。
CATEGORY = "ComfyUI_Marquee_Director"

# 外接可选节点的载体类型。Director 的 face_refine 口与这个节点的输出口是同一个。
H3FaceRefine = None  # 下面用 io.Custom 赋值（io 需要在 comfy_api 可用后导入）

SKIP_NO_FACE = "FaceRefine skipped"

# 接缝淡出的兜底帧数（"跟随回放衔接"取不到 handoff 时用）。
DEFAULT_SEAM_FADE_FRAMES = 39

SEAM_FADE_FOLLOW = "跟随回放衔接"
SEAM_FADE_OFF = "关闭"
SEAM_FADE_CHOICES = [SEAM_FADE_FOLLOW, SEAM_FADE_OFF]

# --- 质检 / 自动重试 -------------------------------------------------
QC_OFF = "关闭"
QC_ON = "开启"
QC_CHOICES = [QC_OFF, QC_ON]

# 自动重试上限。每多一遍就是一次完整的 H3 采样（8s 段在 8 步 turbo 下
# 约 1-2 分钟），而质检只在"明显修坏"时才拦。三遍是"值得再试"与
# "别把显卡烧了"之间的平衡点；超限则保留**最好的一版**而非最后一版。
MAX_QC_ATTEMPTS = 3

# 每次重试的 denoise 折扣。修成蜡像脸 = denoise 过高，所以往回收；
# 逐次减半（0.75 -> 0.56 -> 0.42）而不是一步砍到底 —— 砍太狠会退化成
# "什么都没修"，白跑一趟采样。
QC_DENOISE_DISCOUNT = 0.75

PASTE_CHOICES = ["face_only", "face_ellipse", "full_crop"]
SELECT_CHOICES = ["largest_face", "centre_most"]
CANVAS_CHOICES = ["auto_768", "auto_1344", "manual"]
SEED_FOLLOW = "跟随一采"
SEED_OFFSET = "一采+1"

_DETECTOR_CACHE: dict = {}


def list_detectors():
    """``models/ultralytics/bbox/`` 里可用的 YOLO 权重。拿不到就给个空列表。"""
    import folder_paths

    names = set()
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            names.update(folder_paths.get_filename_list(key) or [])
        except Exception:
            pass
    if not names:
        base = getattr(folder_paths, "models_dir", None)
        if not base:
            return []
        sub = os.path.join(base, "ultralytics", "bbox")
        try:
            names.update(n for n in os.listdir(sub) if n.endswith(".pt"))
        except OSError:
            return []
    out = sorted(n for n in names if n.endswith(".pt"))
    return out or ["face_yolov8m.pt"]


def load_detector(name: str):
    """Load (and cache) a YOLO detector. Cached: it is ~50 MB and runs every frame."""
    if name in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[name]
    import folder_paths

    path = None
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path:
            break
    if path is None:
        base = getattr(folder_paths, "models_dir", "models")
        for sub in ("ultralytics/bbox", "ultralytics", "ultralytics/segm"):
            cand = os.path.join(base, *sub.split("/"), name)
            if os.path.isfile(cand):
                path = cand
                break
    if path is None:
        raise FileNotFoundError(
            "FaceRefine 检测器 '%s' 未找到。请放到 ComfyUI/models/ultralytics/bbox/。"
            % name)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("FaceRefine 需要 ultralytics：pip install ultralytics") from exc
    model = YOLO(path)
    _DETECTOR_CACHE[name] = model
    return model


# --- 检测 / 跟踪 / 裁切（改编自 ComfyUI-H3-FaceRefine，MIT）-----------------

def _to_bgr_u8(img: torch.Tensor) -> np.ndarray:
    arr = (img[..., :3].clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    return arr[..., ::-1].copy()


def _interp_gaps(vals, valid):
    n = len(vals)
    idx = np.arange(n)
    if not np.asarray(valid).any():
        return np.zeros(n, dtype=np.float64)
    return np.interp(idx, idx[valid], vals[valid])


def _smooth(vals, window, method="gaussian"):
    """Gaussian smoothing with reflect padding -- a locked face must not drift."""
    vals = np.asarray(vals, dtype=np.float64)
    if window <= 1 or len(vals) < 3:
        return vals
    window = min(int(window), len(vals))
    if window % 2 == 0:
        window += 1
    if window < 3:
        return vals
    pad = window // 2
    padded = np.pad(vals, pad, mode="reflect")
    x = np.arange(window, dtype=np.float64) - pad
    sigma = max(window / 6.0, 0.5)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(padded, kernel, mode="valid")[: len(vals)]


def affine_crop(img: torch.Tensor, box, cw: int, ch: int) -> torch.Tensor:
    """Sub-pixel crop + resize. img [1,H,W,C] -> [1,ch,cw,C]."""
    x, y, bw, bh = box
    _, height, width, _ = img.shape
    src = img[..., :3].movedim(-1, 1).float()
    theta = torch.tensor(
        [[[bw / width, 0.0, (2.0 * x + bw) / width - 1.0],
          [0.0, bh / height, (2.0 * y + bh) / height - 1.0]]],
        dtype=torch.float32, device=src.device)
    grid = F.affine_grid(theta, (1, 3, int(ch), int(cw)), align_corners=False)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.movedim(1, -1).to(img.dtype)


def _gaussian_blur_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    if feather <= 0:
        return mask
    k = 2 * int(feather) + 1
    shortest = min(mask.shape[-2], mask.shape[-1])
    if shortest <= k:
        k = max(3, int(shortest / 2) | 1)
    sigma = max(k / 6.0, 0.5)
    x = torch.arange(k, device=mask.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(mask.dtype)
    pad = k // 2
    m = F.conv2d(F.pad(mask, (pad, pad, 0, 0), mode="replicate"),
                 g.view(1, 1, 1, k))
    m = F.conv2d(F.pad(m, (0, 0, pad, pad), mode="replicate"),
                 g.view(1, 1, k, 1))
    return m


def face_region_mask(ch: int, cw: int, face_rect, dilation: int, feather: int,
                     shape: str, device, dtype) -> torch.Tensor:
    m = torch.zeros((1, 1, int(ch), int(cw)), device=device, dtype=torch.float32)
    fx, fy, fwd, fhd = face_rect
    fx -= dilation
    fy -= dilation
    fwd += 2 * dilation
    fhd += 2 * dilation
    if shape == "ellipse":
        yy = torch.arange(ch, device=device, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(cw, device=device, dtype=torch.float32).view(1, -1)
        ccx, ccy = fx + fwd / 2.0, fy + fhd / 2.0
        rx, ry = max(fwd / 2.0, 1.0), max(fhd / 2.0, 1.0)
        m[0, 0] = (((xx - ccx) / rx) ** 2 + ((yy - ccy) / ry) ** 2 <= 1.0).float()
    else:
        x0 = max(0, int(round(fx)))
        y0 = max(0, int(round(fy)))
        x1 = min(int(cw), int(round(fx + fwd)))
        y1 = min(int(ch), int(round(fy + fhd)))
        if x1 > x0 and y1 > y0:
            m[0, 0, y0:y1, x0:x1] = 1.0
    return _gaussian_blur_mask(m, feather).clamp(0, 1).to(dtype)


def feather_edge_mask(h: int, w: int, feather: int, device, dtype) -> torch.Tensor:
    m = torch.ones((h, w), device=device, dtype=dtype)
    f = int(max(0, min(feather, min(h, w) // 2 - 1)))
    if f <= 0:
        return m
    ramp = 0.5 - 0.5 * torch.cos(
        torch.linspace(0, math.pi, f + 2, device=device, dtype=dtype)[1:-1])
    m[:f, :] *= ramp.view(-1, 1)
    m[h - f:, :] *= ramp.flip(0).view(-1, 1)
    m[:, :f] *= ramp.view(1, -1)
    m[:, w - f:] *= ramp.flip(0).view(1, -1)
    return m


def _rank_box(box, frame_w: int, frame_h: int, select: str) -> float:
    x0, y0, x1, y1 = box
    h = max(1.0, y1 - y0)
    if select == "centre_most":
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        dx, dy = cx - frame_w * 0.5, cy - frame_h * 0.5
        return -float(dx * dx + dy * dy)
    return h


def track_and_crop(images: torch.Tensor, pack: dict):
    """Return ``(crops [N,ch,cw,3], transform, report)``; no face -> ``(None, None, note)``."""
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("FaceRefine 需要 IMAGE 视频帧 [N,H,W,C]。")
    frames = images[..., :3].contiguous()
    n_frames, height, width, _ = frames.shape
    detector = load_detector(str(pack.get("detector") or "face_yolov8m.pt"))
    conf = float(pack.get("confidence") or 0.35)
    crop_factor = float(pack.get("crop_factor") or 2.5)
    select = str(pack.get("select") or "largest_face")
    canvas_w = int(pack.get("canvas_width") or 768)
    canvas_h = int(pack.get("canvas_height") or 768)
    canvas_mode = str(pack.get("canvas_mode") or "auto_768")

    cx = np.zeros(n_frames, dtype=np.float64)
    cy = np.zeros(n_frames, dtype=np.float64)
    sz = np.zeros(n_frames, dtype=np.float64)
    fw = np.zeros(n_frames, dtype=np.float64)
    valid = np.zeros(n_frames, dtype=bool)

    lock = None
    found = 0
    for i in range(n_frames):
        try:
            res = detector.predict(_to_bgr_u8(frames[i]), conf=conf, verbose=False)[0]
            boxes = res.boxes.xyxy.tolist() if len(res.boxes) else []
        except Exception as exc:
            # common.log 是函数不是 logger（它内部走 logging.info），别写 log.debug。
            log("FaceRefine: frame %d detect failed (%s)", i, exc)
            boxes = []
        if not boxes:
            continue
        if lock is None:
            pick = max(boxes, key=lambda b: _rank_box(b, width, height, select))
        else:
            lx, ly, lh = lock
            reach = 0.75 * max(lh, 8.0)

            def _dist(b):
                pcx, pcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
                return (pcx - lx) ** 2 + (pcy - ly) ** 2

            pick = min(boxes, key=_dist)
            if _dist(pick) ** 0.5 > reach:
                continue           # lost the subject this frame; gap-filled below
        x0, y0, x1, y1 = pick
        pcx, pcy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        ph, pw = max(1.0, y1 - y0), max(1.0, x1 - x0)
        cx[i], cy[i], sz[i], fw[i] = pcx, pcy, ph, pw
        valid[i] = True
        lock = (pcx, pcy, ph)
        found += 1

    if found == 0:
        note = ("%s：%d 帧里没检测到人脸（detector=%s, confidence=%g）。"
                "沿用解码成片，未做修脸；可换检测器或调低 confidence。"
                % (SKIP_NO_FACE, n_frames,
                   str(pack.get("detector") or "face_yolov8m.pt"), conf))
        log(note)
        return None, None, note

    sm_cx = _smooth(_interp_gaps(cx, valid), 21)
    sm_cy = _smooth(_interp_gaps(cy, valid), 21)
    sm_sz = _smooth(_interp_gaps(sz, valid), 51)
    sm_fw = _smooth(_interp_gaps(fw, valid), 51)

    if canvas_mode != "manual":
        need = float(min(sm_sz.max() * crop_factor, height))
        snapped = int(math.ceil(need / 32.0) * 32)
        cap = 768 if canvas_mode == "auto_768" else 1344
        snapped = max(512, min(snapped, cap))
        canvas_w = canvas_h = snapped

    aspect = canvas_w / float(canvas_h)
    boxes = []
    crops = torch.zeros((n_frames, canvas_h, canvas_w, 3), dtype=frames.dtype)
    for i in range(n_frames):
        bh = sm_sz[i] * crop_factor
        bw = bh * aspect
        if bw > width:
            bw, bh = float(width), float(width) / aspect
        if bh > height:
            bh, bw = float(height), float(height) * aspect
        x = min(max(sm_cx[i] - bw / 2.0, 0.0), max(0.0, width - bw))
        y = min(max(sm_cy[i] - bh / 2.0, 0.0), max(0.0, height - bh))
        box = (float(x), float(y), float(bw), float(bh))
        boxes.append(box)
        crops[i:i + 1] = affine_crop(frames[i:i + 1], box, canvas_w, canvas_h).to(crops.dtype)

    weights = np.clip(_smooth(valid.astype(np.float64), 11), 0.0, 1.0)
    # ★ 必须 enumerate 逐帧取：上面那个循环结束后 i 停在 n_frames-1，直接写
    #   sm_fw[i] 会让**每一帧**的脸部矩形都用最后一帧的脸尺寸 —— 镜头推近/拉远
    #   时贴回的框就不跟着动了。（stitch_faces 是按帧 face_rects[i] 取的。）
    face_rect = []
    for i, box in enumerate(boxes):
        bw, bh = max(box[2], 1e-6), max(box[3], 1e-6)
        face_rect.append((
            float(canvas_w) * 0.5 - 0.5 * float(sm_fw[i]) / bw * canvas_w,
            float(canvas_h) * 0.5 - 0.5 * float(sm_sz[i]) / bh * canvas_h,
            float(sm_fw[i]) / bw * canvas_w,
            float(sm_sz[i]) / bh * canvas_h,
        ))
    transform = {
        "boxes": boxes,
        "canvas": (int(canvas_w), int(canvas_h)),
        "src_size": (int(width), int(height)),
        "frames": int(n_frames),
        "source": list(range(n_frames)),
        "weights": [float(w) for w in weights],
        "detected": [bool(v) for v in valid],
        "face_rect": face_rect,
        "crop_factor": float(crop_factor),
    }
    report = ("FaceRefine track: %d/%d 帧有脸, select=%s, canvas=%dx%d (%s), crop×%g"
              % (found, n_frames, select, canvas_w, canvas_h, canvas_mode, crop_factor))
    return crops, transform, report


# --- 贴回 / 接缝 -------------------------------------------------------------

def stitch_faces(base_images, refined_crops, transform, pack):
    """Warp the refined close-ups back onto the source frames through the inverse affine."""
    boxes = transform["boxes"]
    weights = transform.get("weights")
    source = transform.get("source") or list(range(min(len(boxes), refined_crops.shape[0])))
    count = min(len(boxes), refined_crops.shape[0], len(source), base_images.shape[0])
    if count <= 0:
        return base_images[..., :3].clone()
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
    except Exception:
        dev = base_images.device

    paste = str(pack.get("paste_region") or "face_only")
    dilation = int(pack.get("mask_dilation") or 16)
    feather = int(pack.get("feather") or 24)
    colour_match = float(pack.get("colour_match") or 1.0)
    blend = float(pack.get("blend") or 1.0)
    cw, ch = transform["canvas"]
    width, height = transform["src_size"]
    face_rects = transform.get("face_rect")
    dt = base_images.dtype
    out = base_images[..., :3].clone()
    per_frame_mb = (height * width * 3 * 4) / 2 ** 20
    chunk = max(1, min(32, int(1024 / max(per_frame_mb, 1e-6))))

    for c0 in range(0, count, chunk):
        c1 = min(c0 + chunk, count)
        n = c1 - c0
        bh_mid = float(boxes[(c0 + c1 - 1) // 2][3])
        f_can = int(round(feather * (ch / max(bh_mid, 1.0))))
        f_can = max(1, min(f_can, ch // 3))
        if paste == "full_crop":
            one = feather_edge_mask(ch, cw, f_can, dev, torch.float32)
            mask_can = one.view(1, 1, ch, cw).expand(n, 1, ch, cw)
        else:
            mask_can = torch.cat([
                face_region_mask(
                    ch, cw,
                    face_rects[i] if face_rects and i < len(face_rects)
                    else (cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5),
                    dilation, f_can,
                    "ellipse" if paste == "face_ellipse" else "rect",
                    dev, torch.float32)
                for i in range(c0, c1)], dim=0)

        th = torch.empty((n, 2, 3), dtype=torch.float32, device=dev)
        for j, i in enumerate(range(c0, c1)):
            x, y, bw, bh = (float(v) for v in boxes[i])
            th[j, 0, 0] = width / bw
            th[j, 0, 1] = 0.0
            th[j, 0, 2] = (width - 2.0 * x) / bw - 1.0
            th[j, 1, 0] = 0.0
            th[j, 1, 1] = height / bh
            th[j, 1, 2] = (height - 2.0 * y) / bh - 1.0
        grid = F.affine_grid(th, (n, 3, int(height), int(width)), align_corners=False)
        patch_can = refined_crops[c0:c1, ..., :3].to(dev).movedim(-1, 1).float()
        patch = F.grid_sample(patch_can, grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        m = F.grid_sample(mask_can.to(dev), grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False).clamp(0, 1)
        patch = patch.movedim(1, -1)
        m = m.movedim(1, -1)
        dst = torch.as_tensor(source[c0:c1], dtype=torch.long, device=out.device)
        base = out[dst].to(dev).float()
        if colour_match > 0.0:
            # Match the patch's mean/std to the frame it lands on, or the seam
            # shows up as a colour step even when the geometry is perfect.
            wsum = m.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            bmu = (base * m).sum(dim=(1, 2), keepdim=True) / wsum
            pmu = (patch * m).sum(dim=(1, 2), keepdim=True) / wsum
            bsd = (((base - bmu) ** 2 * m).sum(dim=(1, 2), keepdim=True) / wsum).sqrt().clamp_min(1e-6)
            psd = (((patch - pmu) ** 2 * m).sum(dim=(1, 2), keepdim=True) / wsum).sqrt().clamp_min(1e-6)
            adj = (patch - pmu) * (bsd / psd) + bmu
            patch = (patch + (adj - patch) * colour_match).clamp(0, 1)
        wv = torch.full((n, 1, 1, 1), float(blend), device=dev, dtype=torch.float32)
        if weights is not None:
            for j, i in enumerate(range(c0, c1)):
                if i < len(weights):
                    wv[j] *= float(weights[i])
        mm_ = m * wv
        out[dst] = ((1.0 - mm_) * base + mm_ * patch).to(out.device, dt)
    return out


def fade_stitch_at_seams(stitched, base, *, head_frames=0, tail_frames=0):
    """Lerp the paste back to the unstitched clip at continuity edges.

    Head: frame 0 is exactly ``base`` (that is what the next segment's anchor
    carries), ramping to the refined result by ``head_frames``.
    Tail: the mirror -- the last frame is exactly ``base`` again, because it is
    the frame the *next* segment will open on.
    """
    if (stitched is None or base is None or stitched is base
            or not isinstance(stitched, torch.Tensor)
            or not isinstance(base, torch.Tensor)
            or stitched.ndim != 4 or base.ndim != 4):
        return stitched
    n = min(int(stitched.shape[0]), int(base.shape[0]))
    if n < 2:
        return stitched
    head = max(0, min(int(head_frames or 0), n))
    tail = max(0, min(int(tail_frames or 0), n))
    if head + tail > n:
        tail = max(0, n - head)
    if head < 1 and tail < 1:
        return stitched
    out = stitched[:n, ..., :3].clone()
    src = base[:n, ..., :3].to(device=out.device, dtype=out.dtype)
    if head:
        # t=0 -> w=1（第 0 帧**精确等于**未修原图，下一段钉的就是这一版）；
        # t->1 -> w=0（到 head 帧处完全放开成修脸结果）。
        t = torch.arange(head, device=out.device, dtype=out.dtype) / float(head)
        w = (0.5 * (1.0 + torch.cos(t * math.pi))).view(head, 1, 1, 1)
        out[:head] = src[:head] * w + out[:head] * (1.0 - w)
    if tail:
        # 尾部是镜像，但必须取到 t=1：用 arange/tail 的话最后一项是
        # (tail-1)/tail，权重 0.998 而不是 1 —— 最后一帧淡不干净，接缝两侧
        # 会差出那么一点点。这里 (arange+1)/tail 让最后一帧**精确等于**原图。
        t = (torch.arange(tail, device=out.device, dtype=out.dtype) + 1.0) / float(tail)
        w = (0.5 * (1.0 - torch.cos(t * math.pi))).view(tail, 1, 1, 1)
        out[-tail:] = src[-tail:] * w + out[-tail:] * (1.0 - w)
    if n < int(stitched.shape[0]):
        merged = stitched.clone()
        merged[:n] = out.to(device=stitched.device, dtype=stitched.dtype)
        return merged
    return out.to(device=stitched.device, dtype=stitched.dtype)


def inject_video_latent(av_latent: dict, images: torch.Tensor, vae) -> dict:
    """Replace the video stream of an H3 joint AV latent with encoded crops."""
    samples = av_latent.get("samples")
    if samples is None:
        raise KeyError('LATENT is missing "samples".')
    import comfy.nested_tensor

    # ★ 必须显式判 nested：普通 5D 张量也能 .unbind()，但那是沿 batch 维拆开，
    #   会把一段 latent 拆成一堆片段、静默拼出一个错的东西。原版有这个守卫。
    if not (isinstance(samples, comfy.nested_tensor.NestedTensor)
            or getattr(samples, "is_nested", False)):
        raise ValueError(
            "FaceRefine 需要 MiniMax H3 联合 AV latent（NestedTensor），"
            "拿到的是 %s。" % type(samples).__name__)

    members = list(samples.unbind())
    video_tmpl = members[0]
    encoded = vae.encode(images[..., :3])
    if encoded.ndim == 4:
        encoded = encoded.unsqueeze(0).movedim(1, 2)
    tgt_t, tgt_h, tgt_w = video_tmpl.shape[-3], video_tmpl.shape[-2], video_tmpl.shape[-1]
    got_t, got_h, got_w = encoded.shape[-3], encoded.shape[-2], encoded.shape[-1]
    if (got_h, got_w) != (tgt_h, tgt_w):
        raise ValueError("FaceRefine 裁剪画布与 H3 latent 空间不一致：编码 %dx%d，期望 %dx%d。"
                         % (got_h, got_w, tgt_h, tgt_w))
    if got_t != tgt_t:
        if got_t > tgt_t:
            encoded = encoded[..., :tgt_t, :, :]
        else:
            pad = video_tmpl[..., :tgt_t - got_t, :, :].to(encoded.device, encoded.dtype)
            encoded = torch.cat([encoded, pad], dim=-3)
    members[0] = encoded.to(video_tmpl.device, video_tmpl.dtype)
    out = dict(av_latent)
    out["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))
    return out


def _pad_frames(frames: torch.Tensor, length: int) -> torch.Tensor:
    if frames.shape[0] >= length:
        return frames[:length]
    last = frames[-1:].expand(length - frames.shape[0], -1, -1, -1)
    return torch.cat([frames, last], dim=0)


# --- 按脸大小定 denoise ------------------------------------------------------

def _denoise_for_faces(pack, transform) -> float:
    """按脸在画面里的实际大小，给出这一遍该用多少 denoise。

    借鉴 Carasibana/ComfyUI-H3-FaceRefine 的 ``H3 Per-Frame Denoise`` 思路
    （小脸全强度、大脸低强度），但本包取的是**整段一个值**而不是逐帧曲线：
    本包的人脸跟踪本来就按段平滑（``_smooth`` 的高斯窗），一段里脸的大小
    变化有限；而逐帧 denoise 需要改写 noise_mask 并把节点塞进模型路径
    （会动到本包 ``H3ChainSettings`` 的接线），收益不足以承担那个风险。

    为什么必须分大小：H3 在人脸占画面很小时会把它渲成糊块，**这时候没有
    细节可保留**，需要接近全强度的重绘；而特写大脸本身有真实细节，同样的
    denoise 会把它改写成"另一个人的脸"。一个全局值伺候不了两种镜头。
    """
    if not pack.get("adaptive_denoise"):
        return 1.0
    boxes = (transform or {}).get("boxes") or []
    crop_factor = float(pack.get("crop_factor") or 2.5)
    heights = []
    for box in boxes:
        try:
            # transform 存的是裁切框，脸高 = y1 - y，再除以 crop_factor
            # 还原成源像素里的真实脸高。
            heights.append(abs(float(box[3]) - float(box[1]))
                           / max(crop_factor, 1e-3))
        except (TypeError, ValueError, IndexError):
            continue
    if not heights:
        return 1.0
    face_px = float(np.median(heights))
    small_px = float(pack.get("face_px_small") or 30.0)
    large_px = float(pack.get("face_px_large") or 120.0)
    lo = float(pack.get("strength_small_face") or 1.0)
    hi = float(pack.get("strength_large_face") or 0.35)
    if large_px <= small_px:
        return lo
    t = (face_px - small_px) / (large_px - small_px)
    t = float(min(1.0, max(0.0, t)))
    return lo + (hi - lo) * t


def _scaled_sigmas(sigmas, scale: float):
    """把噪声表整体收力，用于自动重试时逐次降低 denoise。

    denoise 不是 sigma：ComfyUI 用 ``s / (s + 1)`` 把 sigma 映射成连续
    denoise 值（BasicScheduler 同一套）。要"把 denoise 压到原来的 75%"，
    必须反解回 sigma 再重算，而不是把 denoise 值直接当 sigma 塞回去 ——
    那样实际强度会远低于预期。scale≈1 时原样返回，不碰对象。
    """
    if abs(float(scale) - 1.0) <= 1e-6:
        return sigmas
    raw = getattr(sigmas, "sigma", None)
    if raw is None:
        return sigmas
    import comfy.samplers as _cs

    def to_sigma(d: float) -> float:
        d = float(min(1.0 - 1e-4, max(0.0, d)))
        return d / max(1.0 - d, 1e-6)

    out = [to_sigma(s / max(s + 1.0, 1e-6) * float(scale)) for s in raw[:-1]]
    out.append(float(raw[-1]))          # 末尾的 0 不动
    return _cs.SigmaSchedule(out)


# --- 主流程 -----------------------------------------------------------------

def apply_face_refine(settings, segment, images, *, prompt="", replay_frames=0):
    """Optional face pass over one segment's decoded frames.

    Returns ``(images, note)`` -- ``note`` is empty when it did nothing, which is
    the case that has to stay byte-identical.
    """  # noqa: D401 -- 真正的实现在 _apply_face_refine_inner
    pack = settings.get("face_refine")
    if not isinstance(pack, dict) or pack.get("sigmas") is None:
        return images, ""

    # ★ 修脸是**美容性**的可选步骤：它自己失败绝不能把整段渲染带崩。
    #   一条 8 段的长链要跑二十分钟，因为「画布尺寸和 latent 空间对不上」这种
    #   配置问题而全丢，代价和收益完全不成比例。所以这里兜住所有异常，日志里
    #   留完整堆栈，画面退回未修的成片 —— 宁可少修一次脸，不要没有成片。
    try:
        return _apply_face_refine_inner(settings, segment, images, pack,
                                        prompt=prompt, replay_frames=replay_frames)
    except Exception:
        import traceback
        log("FaceRefine 失败，本段沿用未修脸的成片（画面不受影响）：\n%s",
            traceback.format_exc())
        return images, ("FaceRefine 出错，已跳过（画面沿用未修脸的成片，"
                        "详见日志堆栈）")


def _sample_faces_once(settings, segment, pack, crops, transform, *,
                       prompt="", seed=0, denoise_scale=1.0):
    """One face pass: sample the crops and hand back the refined close-ups.

    返回解码后的特写帧（未贴回）。``denoise_scale`` 是本次尝试的强度折扣，
    供自动重试逐次收力用。
    """
    canvas_w, canvas_h = transform["canvas"]
    gen_len = generation_length(int(crops.shape[0]))
    crop_in = _pad_frames(crops, gen_len)

    ref_images = {"ref_image_%d" % i: t
                  for i, t in enumerate(segment.get("images") or [])}
    positive, latent = MiniMaxH3ReferenceToVideo.execute(
        clip=settings["clip"],
        vae=settings["vae"],
        audio_vae=settings["audio_vae"],
        prompt=str(prompt or "a person, face close-up"),
        width=int(canvas_w),
        height=int(canvas_h),
        length=int(gen_len),
        ref_image_size=settings.get("ref_image_size", "match"),
        ref_images=ref_images or None,
    )
    latent = inject_video_latent(latent, crop_in, settings["vae"])

    sampler = pack.get("sampler") or settings["sampler"]
    if isinstance(sampler, str):
        import comfy.samplers
        sampler = comfy.samplers.sampler_object(sampler)

    guider = Guider_Basic(settings["model"])
    guider.set_conds(positive)
    try:
        sampled = SamplerCustomAdvanced.execute(
            noise=Noise_RandomNoise(int(seed)),
            guider=guider,
            sampler=sampler,
            sigmas=_scaled_sigmas(pack["sigmas"], denoise_scale),
            latent_image=latent,
        )[0]
    finally:
        # 显式解引用再置空：guider 持有 model 的 patch 引用，不放掉会让
        # 重试时的显存水位比首遍高一截，几遍之后 OOM。
        del guider

    refined = settings["vae"].decode(
        sampled["samples"].unbind()[0]
        if getattr(sampled["samples"], "is_nested", False) else sampled["samples"])
    if refined.ndim == 5:
        refined = refined.reshape(-1, *refined.shape[-3:])
    refined = refined[:int(crops.shape[0]), ..., :3].float().cpu()
    if refined.shape[0] < crops.shape[0]:
        refined = _pad_frames(refined, crops.shape[0])
    del sampled, latent, positive
    return refined


def _apply_face_refine_inner(settings, segment, images, pack, *,
                             prompt="", replay_frames=0):
    base = images[..., :3].contiguous().float().cpu()
    n_src = int(base.shape[0])
    crops, transform, track_note = track_and_crop(base, pack)
    if crops is None or transform is None:
        return images, track_note

    seed = int(segment.get("resolved_seed") or 0)
    if str(pack.get("seed_mode")) == SEED_OFFSET:
        seed = seed + 1

    qc_on = str(pack.get("quality_check") or QC_OFF) != QC_OFF
    attempts = int(pack.get("max_attempts") or MAX_QC_ATTEMPTS) if qc_on else 1
    attempts = max(1, min(attempts, MAX_QC_ATTEMPTS))
    scale = _denoise_for_faces(pack, transform)

    fade = int(replay_frames or 0) or DEFAULT_SEAM_FADE_FRAMES
    do_fade = str(pack.get("seam_fade") or SEAM_FADE_FOLLOW) != SEAM_FADE_OFF

    best = None            # (score, stitched)：保留**最好**的一版，不是最后一版
    notes = []
    for attempt in range(1, attempts + 1):
        try:
            refined = _sample_faces_once(
                settings, segment, pack, crops, transform,
                prompt=prompt, seed=seed, denoise_scale=scale)
        except Exception as exc:                    # 采样本身炸了这一遍
            log("FaceRefine 第 %d/%d 遍采样失败：%s", attempt, attempts, exc)
            if attempt >= attempts:
                break
            seed = (seed * 1103515245 + 12345) % (1 << 63)
            scale *= QC_DENOISE_DISCOUNT
            continue

        stitched = stitch_faces(base, refined, transform, pack)
        if do_fade:
            # 接缝淡回：见模块开头那段说明。head 用本段回放帧数（下一段钉的
            # 就是这段），tail 用同一个数 —— 本段最后一帧会被下一段原样打开。
            stitched = fade_stitch_at_seams(stitched, base,
                                            head_frames=fade, tail_frames=fade)
        stitched = stitched[:n_src]

        if not qc_on:
            return (stitched.contiguous().to(device=images.device,
                                             dtype=images.dtype),
                    "%s; canvas %dx%d" % (track_note, transform["canvas"][0],
                                          transform["canvas"][1]))

        verdict = face_qc.judge(
            base, stitched, transform.get("boxes"),
            identity_floor=float(pack.get("identity_floor")
                                 or face_qc.IDENTITY_FLOOR),
            detail_floor=float(pack.get("detail_floor")
                               or face_qc.DETAIL_RATIO_FLOOR),
            stability_floor=float(pack.get("stability_floor")
                                  or face_qc.STABILITY_FLOOR))
        score = 1.0 if verdict.get("ok") else 0.0
        if best is None or score > best[0]:
            best = (score, stitched)
        line = face_qc.format_report(verdict, attempt, attempts)
        notes.append(line)
        log("FaceRefine %s", line)
        if verdict.get("ok"):
            break
        if attempt < attempts:
            # 不合格就换种子 + 收 denoise 再试。乘法跳变而不是加常数：加常数
            # 实测会产生高度相关的样本（与 director._repair_reseed 同一结论）。
            seed = (seed * 1103515245 + 12345) % (1 << 63)
            scale *= QC_DENOISE_DISCOUNT

    if best is None:                                # 每一遍都炸在采样里
        return images, track_note + "; FaceRefine 全部重试失败，已沿用原片"

    note = "%s; canvas %dx%d" % (track_note, transform["canvas"][0],
                                 transform["canvas"][1])
    if notes:
        note += "; " + (" | ".join(notes) if len(notes) > 1 else notes[0])
    return (best[1].contiguous().to(device=images.device, dtype=images.dtype),
            note)


# --- 节点 -------------------------------------------------------------------

def _build_node():
    from comfy_api.latest import io
    import comfy.samplers

    global H3FaceRefine
    H3FaceRefine = io.Custom("H3_FACE_REFINE")

    class H3FaceRefineNode(io.ComfyNode):
        """Pack a face-refine pass. Wire ``face_refine`` into H3 Director."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="H3FaceRefine",
                display_name="H3 FaceRefine（脸部修复）",
                category=CATEGORY,
                description=(
                    "Optional face pass for H3 Director. Connect `sigmas` (a "
                    "denoise<1 schedule) and wire `face_refine` into the "
                    "Director: every segment's decoded frames are then tracked, "
                    "cropped to a close-up canvas, re-sampled, and stitched back "
                    "with the seam faded to the unrefined frames so the join "
                    "does not step. Unwired (or wired without `sigmas`) the "
                    "Director behaves exactly as before.\n\n"
                    "Needs ultralytics and a YOLO weight in "
                    "ComfyUI/models/ultralytics/bbox/ (face_yolov8m.pt). No "
                    "face detected -> the segment is left untouched."
                ),
                inputs=[
                    io.Sigmas.Input(
                        "sigmas", optional=True,
                        tooltip="修脸噪声表。接 BasicScheduler（denoise 调小，"
                                "0.35~0.5）或 ManualSigmas。\n\n"
                                "★ H3 的 denoise 不能照搬 SDXL 的经验值。H3 是"
                                "flow matching + 大 sigma shift（默认 shift 12），"
                                "两者不是一套刻度：\n"
                                "    denoise 0.02 → 有效 sigma 0.197\n"
                                "    denoise 0.05 → 有效 sigma 0.387\n"
                                "    denoise 0.25 → 有效 sigma 0.800 ← 整帧重写\n"
                                "也就是说 SDXL 常用的 0.25 在 H3 上已经是"
                                "「把这一帧推倒重来」，大脸会被改成另一个人。"
                                "修脸的安全区大致是 0.02~0.15。\n\n"
                                "别接 SplitSigmas：4 步 turbo 调度的最后一个"
                                "分割点已经在 sigma 0.8 附近。\n\n"
                                "不接 = 关闭修脸（即便节点已接到 Director）。"),
                    io.Combo.Input(
                        "detector", display_name="检测器",
                        options=list_detectors(), default="face_yolov8m.pt",
                        tooltip="models/ultralytics/bbox/ 里的 YOLO 权重。"),
                    io.Float.Input("confidence", display_name="检测阈值",
                                   default=0.35, min=0.05, max=0.95, step=0.05,
                                   round=False),
                    io.Combo.Input("select", display_name="选哪张脸",
                                   options=SELECT_CHOICES, default="largest_face"),
                    io.Float.Input("crop_factor", display_name="裁切倍数",
                                   default=2.5, min=1.2, max=6.0, step=0.1,
                                   round=False,
                                   tooltip="以检测框高为基准向外放大多少倍。"),
                    io.Combo.Input("canvas_mode", display_name="特写画布",
                                   options=CANVAS_CHOICES, default="auto_768",
                                   tooltip="auto_768 = 跟随人脸自动定，上限 768；"
                                           "auto_1344 = 上限 1344（更清晰、更慢）；"
                                           "manual = 用下面两个框。"),
                    io.Int.Input("canvas_width", display_name="画布宽",
                                 default=768, min=256, max=2048, step=32,
                                 advanced=True),
                    io.Int.Input("canvas_height", display_name="画布高",
                                 default=768, min=256, max=2048, step=32,
                                 advanced=True),
                    io.Combo.Input(
                        "sampler", display_name="采样器",
                        options=["跟随一采"] + list(comfy.samplers.KSampler.SAMPLERS),
                        default="跟随一采"),
                    io.Combo.Input("seed_mode", display_name="种子",
                                   options=[SEED_FOLLOW, SEED_OFFSET],
                                   default=SEED_FOLLOW),
                    io.Combo.Input("paste_region", display_name="贴回区域",
                                   options=PASTE_CHOICES, default="face_only",
                                   tooltip="face_only = 只贴脸（矩形）；"
                                           "face_ellipse = 椭圆；full_crop = 整块特写。"),
                    io.Int.Input("feather", display_name="羽化", default=24,
                                 min=0, max=128, step=1, advanced=True),
                    io.Float.Input("blend", display_name="混合强度", default=1.0,
                                   min=0.0, max=1.0, step=0.05, round=False,
                                   advanced=True),
                    io.Combo.Input(
                        "seam_fade", display_name="接缝淡出",
                        options=SEAM_FADE_CHOICES, default=SEAM_FADE_FOLLOW,
                        tooltip="★ 别轻易关。段首是上一段尾帧的整段回放（钉在 "
                                "latent 里，没修过脸），段尾会被下一段原样打开。"
                                "修脸结果必须在这两段内淡回原图，否则每个接缝"
                                "都会有一记「脸突然变清楚」的跳变。\n"
                                "跟随回放衔接 = 用本段回放帧数（默认 39 帧）；"
                                "关闭 = 整段都修（接缝可能跳）。"),
                    # ── 质量闭环（2026-10-06 新增）──────────────────────────
                    # ★ 以下全部挂在 schema **末尾**，且一律给默认值。
                    #   本包的 widget 按位置对齐（见 director.py / nodes.py 的
                    #   同款约定）：插在中间会让所有已存工作流的 widgets_values
                    #   整体错位，而修脸节点恰恰是那种"已经存了一堆工作流"的
                    #   节点。末尾追加 + 有默认值 = 老工作流零风险加载。
                    io.Combo.Input(
                        "quality_check", display_name="质量自检",
                        options=QC_CHOICES, default=QC_OFF,
                        tooltip="修完自己看一眼，判不过就换种子重修。\n\n"
                                "判据是**修后 vs 修前的同一帧**（不是对参考图）：\n"
                                "· 高频能量 —— 掉下去就是被抹平成蜡像脸\n"
                                "· ArcFace 身份余弦 —— 掉下去就是修成了另一个人\n"
                                "· 相邻帧余弦 —— 掉下去就是脸在逐帧沸腾\n\n"
                                "★ InsightFace 缺失时自动降级为纯像素判据，"
                                "仍能拦住蜡像脸，只是认不出身份漂移。"),
                    io.Int.Input("max_attempts", display_name="最多修几遍",
                                 default=MAX_QC_ATTEMPTS, min=1,
                                 max=MAX_QC_ATTEMPTS, step=1, advanced=True,
                                 tooltip="判不过时最多重试几遍。每遍是一次完整的"
                                         "H3 采样，所以别开太大。\n\n"
                                         "重试同时做两件事：换种子（乘法跳变，"
                                         "避免抽到高度相关的同一结果）"
                                         "并逐次收低 denoise。\n\n"
                                         "全部遍数都不过时，保留**最好的一版**，"
                                         "不是最后一版。"),
                    io.Float.Input("identity_floor", display_name="身份下限",
                                   default=face_qc.IDENTITY_FLOOR,
                                   min=0.0, max=1.0, step=0.01, round=False,
                                   advanced=True,
                                   tooltip="修前修后 ArcFace 余弦的最低可接受值。"
                                           "默认 0.72 很宽松：正常修脸会让余弦"
                                           "小幅下降（GAN 类 -0.03~-0.08，"
                                           "扩散类更多），只拦掉得离谱的。"),
                    io.Float.Input("detail_floor", display_name="细节下限",
                                   default=face_qc.DETAIL_RATIO_FLOOR,
                                   min=0.1, max=2.0, step=0.01, round=False,
                                   advanced=True,
                                   tooltip="修后/修前 的高频能量比，低于它判定"
                                           "「越修越糊」。默认 0.92 意味着"
                                           "持平不算失败（白跑一趟由重试逻辑"
                                           "处理，不是质量问题）。"),
                    io.Float.Input("stability_floor", display_name="稳定下限",
                                   default=face_qc.STABILITY_FLOOR,
                                   min=0.0, max=1.0, step=0.01, round=False,
                                   advanced=True,
                                   tooltip="相邻帧余弦均值的下限，低于它判定"
                                           "「脸在沸腾」。"),
                    io.Boolean.Input("adaptive_denoise",
                                     display_name="按脸大小定强度", default=True,
                                     tooltip="★ 强烈建议开着。\n\n"
                                             "H3 在人脸占画面很小时会渲成糊块，"
                                             "**这时没有细节可保留**，要接近全强度"
                                             "重绘；而特写大脸本身有真实细节，"
                                             "同样强度会把它改写成另一个人。\n\n"
                                             "一个全局值伺候不了这两种镜头，所以"
                                             "按本段脸的真实像素高度在下面两个"
                                             "强度之间插值。"),
                    io.Float.Input("strength_small_face", display_name="小脸强度",
                                   default=1.0, min=0.0, max=1.5, step=0.05,
                                   round=False, advanced=True,
                                   tooltip="脸最小时（≤ 下面的「小脸像素」）"
                                           "用的 denoise 倍率。1.0 = 保持不变。"),
                    io.Float.Input("strength_large_face", display_name="大脸强度",
                                   default=0.35, min=0.0, max=1.5, step=0.05,
                                   round=False, advanced=True,
                                   tooltip="脸最大时（≥ 「大脸像素」）的倍率。"
                                           "0.35 = 只轻轻收一点，保住大脸已有的"
                                           "真实细节。"),
                    io.Int.Input("face_px_small", display_name="小脸像素",
                                 default=30, min=4, max=400, step=1,
                                 advanced=True,
                                 tooltip="源像素里的脸高低于这个值算「小脸」"
                                         "（≈ 全景/远景）。"),
                    io.Int.Input("face_px_large", display_name="大脸像素",
                                 default=120, min=8, max=800, step=1,
                                 advanced=True,
                                 tooltip="源像素里的脸高高于这个值算「大脸」"
                                         "（≈ 特写/近景）。超过它不再收强度。"),
                ],
                outputs=[H3FaceRefine.Output(display_name="face_refine")],
            )

        @classmethod
        def execute(cls, confidence=0.35, select="largest_face", crop_factor=2.5,
                    canvas_mode="auto_768", canvas_width=768, canvas_height=768,
                    sampler="跟随一采", seed_mode=SEED_FOLLOW,
                    paste_region="face_only", feather=24, blend=1.0,
                    seam_fade=SEAM_FADE_FOLLOW, detector="face_yolov8m.pt",
                    quality_check=QC_OFF, max_attempts=MAX_QC_ATTEMPTS,
                    identity_floor=face_qc.IDENTITY_FLOOR,
                    detail_floor=face_qc.DETAIL_RATIO_FLOOR,
                    stability_floor=face_qc.STABILITY_FLOOR,
                    adaptive_denoise=True, strength_small_face=1.0,
                    strength_large_face=0.35, face_px_small=30,
                    face_px_large=120, sigmas=None) -> "io.NodeOutput":
            return io.NodeOutput({
                "sigmas": sigmas,
                "detector": detector,
                "confidence": float(confidence),
                "select": select,
                "crop_factor": float(crop_factor),
                "canvas_mode": canvas_mode,
                "canvas_width": int(canvas_width),
                "canvas_height": int(canvas_height),
                "sampler": None if sampler == "跟随一采" else sampler,
                "seed_mode": seed_mode,
                "paste_region": paste_region,
                "feather": int(feather),
                "blend": float(blend),
                "seam_fade": seam_fade,
                "mask_dilation": 16,
                "colour_match": 1.0,
                # 质量闭环。★ 这些键全部进 pack，而 pack 参与缓存键 ——
                # 所以改质检阈值会重渲（这是对的：阈值变了判据就变了）。
                "quality_check": quality_check,
                "max_attempts": int(max_attempts),
                "identity_floor": float(identity_floor),
                "detail_floor": float(detail_floor),
                "stability_floor": float(stability_floor),
                "adaptive_denoise": bool(adaptive_denoise),
                "strength_small_face": float(strength_small_face),
                "strength_large_face": float(strength_large_face),
                "face_px_small": int(face_px_small),
                "face_px_large": int(face_px_large),
            })

    return H3FaceRefineNode


H3FaceRefineNode = _build_node()


def register_with_extension(ext):
    """Return node classes for MarqueeDirectorExtension.get_node_list()."""
    return [H3FaceRefineNode]


__all__ = [
    "H3FaceRefineNode",
    "H3FaceRefine",
    "apply_face_refine",
    "fade_stitch_at_seams",
    "register_with_extension",
]
