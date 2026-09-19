"""H3SegmentTimeline - 分镜时间线预览（ComfyUI 1.0 io.ComfyNode 新 API 风格）

与同包 H3ShotRendererNode 一致，必须实现 define_schema()/execute()，
否则整个 ComfyUI_Marquee_Director 包的 comfy_entrypoint 会失败（IMPORT FAILED）。

功能：
  - 扫描会话目录 <output>/h3_continuous/<session>/ 下已完成的 seg_XX.mp4
  - 读 manifest.json 获取每段真实时长与完成状态
  - 生成缩略图时间线（IMAGE）+ 进度摘要（STRING）+ 完成段数（INT）
  - 可选把已完成段回写到看板 metadata JSON 的 shot.video_path，
    让看板成片时间线实时显示（h3_board.py 白名单已含 h3_continuous 会话目录）

取代了原先独立的进度监控节点（功能已并入本节点，旧节点已移除）。
"""

import glob
import json
import os
import re

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from comfy_api.latest import io

CATEGORY = "ComfyUI_Marquee_Director"

def default_output_roots():
    """当前 ComfyUI 实例的 <output>/h3_continuous（**查找用**，可含历史遗留根）。

    **不硬编码**。用户是自定义 ComfyUI，输出目录可能是 <ComfyUI>/output 也可能是
    别的地方，只有 ``folder_paths.get_output_directory()`` 知道当下这份实例
    往哪写。Session 类（同包 session.py）也是这么拼目录的，两边必须一致，
    否则时间线会去扫一个根本不存在的目录。

    ★ 第二个根（历史遗留）与 routes._session_dirs() 保持同一套语义：
    Desktop 的 ``--output-directory`` 是可变的，同一台机器先后用过
    ``<另一份安装>/output`` / ``<自定义目录>/output`` / 以及
    「不带 --output-directory」的裸 CLI（后者落在 ``<base_path>\\output``）。
    只认当下这一个根 → 昨天渲好的段找不到 → 时间线空、断点渲染从头开始。
    ``<base_path>/output`` 正是 folder_paths 里 output_directory 的同一条
    推导式，属配置派生，不是写死。
    """
    try:
        import folder_paths

        roots = [os.path.join(folder_paths.get_output_directory(), "h3_continuous")]
        base = getattr(folder_paths, "base_path", "") or ""
        if base:
            legacy = os.path.join(base, "output", "h3_continuous")
            if os.path.normcase(legacy) not in [os.path.normcase(r) for r in roots]:
                roots.append(legacy)
        return [r for r in roots if os.path.isdir(r)]
    except Exception:
        return []


# 兼容旧引用：以前这里写死过两条 Windows 绝对路径。现在改成走 folder_paths，
# 保留常量名是为了不打断任何还在 import 它的地方。
H3_OUTPUT_ROOTS = default_output_roots()


class H3SegmentTimelineNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SegmentTimeline",
            display_name="📊 分镜时间线预览",
            category=CATEGORY,
            description=(
                "Scans a session dir under <output>/h3_continuous/<session>/, "
                "reads manifest.json for real per-segment duration/status, builds a "
                "thumbnail timeline image, and optionally writes completed segments' "
                "video_path back into the board's metadata JSON so the board's "
                "成片时间线 shows them in real time."
            ),
            inputs=[
                io.String.Input("session_name", default="my_chain",
                                tooltip="Session folder under output/h3_continuous/. "
                                        "Auto-scanned across known H3 output roots."),
                io.Int.Input("thumb_width", default=180, min=80, max=480,
                             step=20,
                             tooltip="Width of each thumbnail in px."),
                io.Int.Input("columns", default=0, min=0, max=12,
                             tooltip="0 = auto (4 cols when <=8 segs, else 4). "
                                     "Otherwise fixed column count."),
                io.Int.Input("gap", default=12, min=0, max=40,
                             tooltip="Gap between thumbnails in px."),
                io.Boolean.Input("show_status", default=True,
                                 tooltip="Show a status dot + label under each thumb."),
                io.Boolean.Input("sync_board", default=True,
                                 tooltip="Write completed segments' video_path back "
                                         "into the board metadata JSON so the board "
                                         "成片时间线 shows them in real time."),
                io.String.Input("output_root", default="", optional=True,
                                tooltip="Leave empty to auto-scan all known H3 "
                                        "Continuous output roots. Set explicitly to "
                                        "point at a custom output dir."),
            ],
            outputs=[
                io.Image.Output(display_name="timeline_image"),
                io.String.Output(display_name="progress_summary"),
                io.Int.Output(display_name="completed_count"),
            ],
        )

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_session_dir(session_name, output_root):
        """在已知 H3 输出根中查找 <root>/<session_name>/。返回 (session_dir, found)。"""
        # 必须用 session.sanitize 本尊。这里原来抄了半份正则
        # （只有 [^A-Za-z0-9._-]+，少了 strip("._")），中文会话名
        # 会被算成 "_v18_en"，而 Director 真正建的目录是 "v18_en"，
        # 差一个前导下划线就永远找不到。
        try:
            from .session import sanitize as _sanitize
        except Exception:
            def _sanitize(text):
                text = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
                text = re.sub(r"\.{2,}", "_", text)
                return text.strip("._") or "session"
        session_name = _sanitize(session_name)
        if not session_name:
            return "", False
        candidates = []
        if output_root:
            candidates.append(os.path.join(output_root, session_name))
        for root in default_output_roots():   # 延迟求值，别用导入期快照
            candidates.append(os.path.join(root, session_name))
        for cand in candidates:
            if os.path.isdir(cand):
                return cand, True
        return (candidates[0] if candidates else ""), False

    @staticmethod
    def _load_manifest(session_dir):
        """读 manifest.json -> {index: record}；失败返回 {}。"""
        mpath = os.path.join(session_dir, "manifest.json")
        if not os.path.isfile(mpath):
            return {}
        try:
            with open(mpath, encoding="utf-8") as f:
                data = json.load(f)
            out = {}
            for rec in (data.get("segments") or []):
                try:
                    out[int(rec.get("index", 0))] = rec
                except (TypeError, ValueError):
                    continue
            return out
        except Exception:
            return {}

    @staticmethod
    def _extract_first_frame(video_path):
        """从 mp4 提取首帧为 PIL Image（PyAV 优先，OpenCV 兜底）。失败返回 None。"""
        if not os.path.isfile(video_path):
            return None
        try:
            import av
        except ImportError:
            av = None
        if av is not None:
            try:
                with av.open(video_path) as c:
                    for frame in c.decode(video=0):
                        return frame.to_image()  # RGB PIL
            except Exception:
                pass
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
        return None

    @staticmethod
    def _font(size, bold=False):
        candidates = ["arialbd.ttf" if bold else "arial.ttf", "msyh.ttc", "arial.ttf"]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _to_tensor(img):
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None]

    @staticmethod
    def _placeholder(msg):
        img = Image.new("RGB", (480, 110), (36, 36, 42))
        dr = ImageDraw.Draw(img)
        dr.text((14, 10), "📊 分镜时间线", fill=(180, 180, 200), font=H3SegmentTimelineNode._font(13, True))
        dr.text((14, 44), msg, fill=(120, 120, 140), font=H3SegmentTimelineNode._font(11))
        return img

    @staticmethod
    def _sync_board(session_dir, manifest, seg_files):
        """把已完成段写回看板 metadata JSON 的 shot[i].video_path。返回写入的 metadata 文件数。"""
        written = 0
        out_root = os.path.dirname(os.path.dirname(session_dir))  # <output>
        data_dir = os.path.join(out_root, "h3_ref2va_auto", "data")
        if not os.path.isdir(data_dir):
            return 0
        session_base = os.path.basename(session_dir.rstrip("/\\"))
        for fn in sorted(os.listdir(data_dir)):
            if not (fn.startswith("metadata-") and fn.endswith(".json")):
                continue
            path = os.path.join(data_dir, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            shots = data.get("shots") or data.get("shots_info") or []
            if not isinstance(shots, list):
                continue
            changed = False
            for i, sp in enumerate(seg_files):
                shot_idx = i + 1
                if shot_idx > len(shots):
                    break
                rec = manifest.get(i, {})
                rel = os.path.join("h3_continuous", session_base, os.path.basename(sp))
                abs_video = os.path.join(out_root, rel)
                if os.path.isfile(abs_video):
                    shot = shots[shot_idx - 1]
                    shot["video_path"] = rel
                    shot["status"] = "rendered"
                    if rec.get("seconds"):
                        shot["duration"] = round(rec["seconds"], 2)
                    changed = True
            if changed:
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
                written += 1
        return written

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    @classmethod
    def execute(cls, session_name, thumb_width, columns, gap, show_status,
                sync_board, output_root="") -> io.NodeOutput:
        session_dir, found = cls._find_session_dir(session_name, output_root)

        if not found:
            img = cls._placeholder("会话目录不存在：%s" % session_name)
            summary = "未找到会话 %s（已扫描：%s）" % (
                session_name, "、".join(default_output_roots()))
            return io.NodeOutput(cls._to_tensor(img), summary, 0)

        manifest = cls._load_manifest(session_dir)
        seg_files = [f for f in sorted(glob.glob(os.path.join(session_dir, "seg_*.mp4")))
                     if ".tail." not in os.path.basename(f)]

        total = len(seg_files)
        if total == 0:
            img = cls._placeholder("暂无完成片段（%s）" % session_dir)
            summary = "会话 %s：暂无渲染片段" % session_name
            return io.NodeOutput(cls._to_tensor(img), summary, 0)

        def seg_index(path):
            m = re.search(r"seg_(\d+)\.mp4$", os.path.basename(path))
            return int(m.group(1)) - 1 if m else -1

        segs = sorted(seg_files, key=seg_index)
        n_cols = columns if columns > 0 else 4
        n_rows = (total + n_cols - 1) // n_cols

        thumbs = []
        statuses = []
        for sp in segs:
            frame = cls._extract_first_frame(sp)
            if frame is None:
                frame = Image.new("RGB", (thumb_width, int(thumb_width * 9 / 16)), (40, 40, 48))
                dr = ImageDraw.Draw(frame)
                dr.text((8, 8), "读取失败", fill=(200, 120, 120), font=cls._font(12))
            else:
                w, h = frame.size
                new_h = int(thumb_width * h / w)
                frame = frame.resize((thumb_width, new_h), Image.Resampling.LANCZOS)
            thumbs.append(frame)
            statuses.append("done" if os.path.isfile(sp) else "pending")

        thumb_h = max(t.height for t in thumbs) if thumbs else thumb_width
        status_h = 22 if show_status else 0
        label_h = 26
        cell_w = thumb_width + gap
        cell_h = thumb_h + label_h + status_h
        canvas_w = n_cols * cell_w - gap + 24
        canvas_h = n_rows * cell_h + 30

        canvas = Image.new("RGB", (canvas_w, canvas_h), (22, 22, 28))
        dr = ImageDraw.Draw(canvas)
        dr.text((12, 6), "📊 %s  (%d 段)" % (session_name, total),
                fill=(230, 230, 230), font=cls._font(13, True))

        for i, (thumb, st) in enumerate(zip(thumbs, statuses)):
            r = i // n_cols
            c = i % n_cols
            x = 12 + c * cell_w
            y = 30 + r * cell_h
            canvas.paste(thumb, (x, y))
            rec = manifest.get(seg_index(segs[i]), {})
            secs = rec.get("seconds", 0)
            label = "段%d  %.1fs" % (i + 1, secs) if secs else "段%d" % (i + 1)
            dr.text((x + 4, y + thumb_h + 4), label, fill=(190, 190, 190), font=cls._font(11))
            if show_status:
                color = (90, 200, 90) if st == "done" else (220, 180, 60)
                dr.ellipse([x + 4, y + thumb_h + label_h - 6,
                            x + 16, y + thumb_h + label_h + 6], fill=color)
                dr.text((x + 20, y + thumb_h + label_h - 4), st, fill=color, font=cls._font(10))

        completed = sum(1 for s in statuses if s == "done")
        total_secs = sum(manifest.get(seg_index(p), {}).get("seconds", 0) for p in segs)
        summary_lines = [
            "会话：%s" % session_name,
            "完成 %d / %d 段，累计 %.1fs" % (completed, total, total_secs),
        ]
        if sync_board and manifest:
            synced = cls._sync_board(session_dir, manifest, segs)
            if synced:
                summary_lines.append("已回写看板 %d 个 metadata，成片时间线将实时显示" % synced)

        summary = "\n".join(summary_lines)
        return io.NodeOutput(cls._to_tensor(canvas), summary, completed)


NODE_CLASS_MAPPINGS = {
    "H3SegmentTimeline": H3SegmentTimelineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SegmentTimeline": "📊 分镜时间线预览",
}
