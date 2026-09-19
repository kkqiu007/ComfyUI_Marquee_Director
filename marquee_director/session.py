"""On-disk state for one chain.

A chain is not a single render, it is N renders that each take minutes. Keeping
them in a session folder is what makes the pack usable:

* **Resume.** Re-queueing after a crash, an OOM or a Ctrl-C reuses every segment
  whose settings did not change and only renders the rest.
* **Edit one segment.** Change segment 4's prompt and segments 1-3 are reused;
  4 onward re-render, because each cache key folds in the key of the segment
  before it.
* **Repair.** The repair node needs the previous segment's tail and the tail the
  *next* segment already continued from. Both are on disk.

Everything lives under ``output/h3_continuous/<session>/``.
"""

import hashlib
import json
import os
import re

import folder_paths

ROOT = "h3_continuous"
MANIFEST_VERSION = 2


def sanitize(name):
    """Normalise a session name into a safe directory name.

    ★ 必须保留 CJK。会话名常带中文项目名（``曹贼的性价比_v18_en_v10``），
      而「落盘目录」走本函数、「面板/查找用的目录名」走
      ``common.safe_session_name`` —— 两处算法不一致时，渲染写进
      ``v18_en_v10``、面板去查 ``曹贼的性价比_v18_en_v10``，于是
      ``done=0``、断点续渲每次都从第 1 段重来。

      曾经靠 ``common`` 在 import 时把本函数替换成 ``safe_session_name``
      来统一（``_install_session_sanitize_alias``）。但那是**导入顺序相关**的补丁：
      任何只导入 ``session`` 而未触发 ``common`` 的路径都会拿到丢 CJK 的旧行为，
      目录再次分叉。所以这里直接在源码层面对齐（截断长度也统一为 60），
      那条补丁退化为无害的幂等保险。
    """
    # \w 在 Python 3 下是 Unicode 的，已包含 CJK；不用额外列 \u4e00-\u9fff。
    name = re.sub(r"[^\w.-]+", "_", (name or "").strip(), flags=re.UNICODE)
    name = re.sub(r"\.{2,}", "_", name)  # never let a session name walk up a directory
    name = name.strip("._") or "session"
    return name[:60]


def digest(value):
    """Stable short digest of anything a segment input can be.

    Tensors are sampled rather than hashed whole: a 2048px reference image is
    50 MB and this runs for every segment on every queue. Shape, dtype and a
    strided slice change whenever the picture does, which is all a cache key needs.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        t = value
        step = max(1, t.shape[0] // 8) if t.ndim > 0 and t.shape[0] > 8 else 1
        sample = t[::step].contiguous().cpu().numpy().tobytes()
        h = hashlib.sha1()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(sample)
        return h.hexdigest()[:16]
    if isinstance(value, dict) and "waveform" in value:
        return "audio:%s:%s" % (digest(value["waveform"]), value.get("sample_rate"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(digest(v) for v in value) + "]"
    return hashlib.sha1(repr(value).encode("utf-8", "replace")).hexdigest()[:16]


def sampler_digest(sampler):
    """Stable fingerprint of a SAMPLER object.

    ``repr()`` will not do: it carries a memory address, so it changes on every
    queue and would invalidate the whole cache every run.
    """
    fn = getattr(sampler, "sampler_function", None)
    name = getattr(fn, "__name__", None) or type(sampler).__name__
    extra = getattr(sampler, "extra_options", None) or {}
    return hashlib.sha1((name + repr(sorted(extra.keys()))).encode()).hexdigest()[:16]


def model_digest(model):
    """Fingerprint the model patcher well enough to notice a LoRA change.

    Weights themselves are never hashed -- far too slow. This sees the model class
    and every patch key with its strengths, which covers adding, removing or
    re-weighting a LoRA. It does *not* see a swapped checkpoint file of the same
    class: rename the session or turn ``resume`` off when you change one.
    """
    try:
        parts = [type(getattr(model, "model", model)).__name__]
        for key, patches in sorted(getattr(model, "patches", {}).items()):
            strengths = []
            for patch in patches:
                strengths += [round(float(v), 6) for v in patch[:1] + patch[2:3]
                              if isinstance(v, (int, float))]
            parts.append("%s:%s" % (key, strengths))
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    except Exception:  # never let a cache key break a render
        return "unknown"


# 渲染管线标识。改动 render_segment 的**渲染逻辑本身**（不是可调参数）时必须
# bump：它进缓存键，bump 会让所有段重渲 —— 这正是目的，旧缓存是旧逻辑的产物。
#
# 历史：
#   v1  cond 引导（_pin_latent 只写 minimax_keyframes）
#   v2  cond 引导 + latent 直写 + noise_mask（apply_latent_continue，2026-09-15）
#
# ⚠️ 已知混合产物：会话 v18_en_v5 的段 1/2 是 v1 渲染的、段 3/4 是 v2 渲染的。
#    尾巴 latent 与渲染路径无关所以成片仍能接上，但不算干净。想让整条链一致，
#    把这个值改成 "h3_render_v2" 即可强制段 1..N 全部重渲（约 8 分钟/段）。
#    平时保持 None：既有会话的 payload 保持字节一致，不会因升级而无故重渲。
RENDER_PIPELINE_ID = None

# seam_redraw 的默认值，必须与 engine.SEAM_MIN_MASK 一致。
# session.py 不 import engine（避免循环依赖），所以在这里再写一遍。
SEAM_REDRAW_DEFAULT = 0.10


def segment_key(settings, segment, handoff, previous_key):
    """Fingerprint of everything that changes what this segment renders.

    ``previous_key`` is in the hash on purpose: a segment is a continuation of the
    one before it, so editing segment 2 must invalidate 3..N even though their own
    prompts are untouched.
    """
    payload = {
        "prev": previous_key,
        "prompt": segment["prompt"].strip(),
        "length": segment["length"],
        "handoff": handoff,
        "seed": segment["resolved_seed"],
        "width": settings["width"],
        "height": settings["height"],
        "ref_image_size": settings["ref_image_size"],
        "sigmas": digest(settings["sigmas"]),
        "sampler": sampler_digest(settings["sampler"]),
        "model": model_digest(settings["model"]),
        "images": [digest(t) for t in segment["images"]],
        "videos": [digest(t) for _, t in segment["videos"]],
        "video_audios": [digest(a) for _, a in segment["video_audios"]],
        "audios": [digest(a) for a in segment["audios"]],
    }
    # Only a segment that actually consumes an anchor cares how that anchor was
    # built, and the key is left byte-identical when it does not. That keeps the
    # first segment of every pre-existing session cached across this upgrade -- the
    # segment renders the same either way, and its own outgoing tail is now written
    # in both forms regardless. Segments 2..N do change, because their handoff
    # genuinely did.
    if previous_key and settings.get("handoff_mode"):
        payload["handoff_mode"] = settings["handoff_mode"]
    if previous_key and settings.get("drift_arrest"):
        payload["drift_arrest"] = round(float(settings["drift_arrest"]), 4)
    # 渲染方式参数：只在**显式设置**时才并入。既有会话的 settings 没有这两个键，
    # payload 就保持字节一致、不会因升级而无故重渲；一旦真设了，改动才会正确
    # 触发重渲 —— 否则改了渲染方式缓存却认为没变，会拿到旧路径渲染的结果。
    if "handoff_remask" in settings:
        payload["handoff_remask"] = bool(settings["handoff_remask"])
    # seam_redraw 默认 0.10（== engine.SEAM_MIN_MASK）。**等于默认值时不并入
    # payload**：H3ChainSettings 现在总是传出这个键，无条件并入会让所有既有
    # 会话的段 2..N 全部重渲。只有真的调过才进键、才触发重渲。
    seam = settings.get("seam_redraw")
    if seam is None:
        seam = settings.get("seam_min_mask")          # 旧名兼容
    if seam is not None and abs(float(seam) - SEAM_REDRAW_DEFAULT) > 1e-6:
        payload["seam_redraw"] = round(float(seam), 4)
    if RENDER_PIPELINE_ID:
        payload["pipe"] = RENDER_PIPELINE_ID
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def signature_to_record(signature):
    """A latent signature as plain JSON -- 24 means and 24 spreads."""
    if not signature:
        return None
    return {"mean": [round(float(v), 6) for v in signature["mean"]],
            "std": [round(float(v), 6) for v in signature["std"]]}


def signature_from_record(record):
    """The inverse, for a segment being reused from cache.

    Returns None for a session rendered before signatures existed, which simply
    means drift arrest has no reference to work from until something re-renders.
    """
    import torch

    payload = (record or {}).get("signature")
    if not payload:
        return None
    return {"mean": torch.tensor(payload["mean"]), "std": torch.tensor(payload["std"])}


class Session:
    def __init__(self, name):
        self.name = sanitize(name)
        self.dir = os.path.join(folder_paths.get_output_directory(), ROOT, self.name)
        os.makedirs(self.dir, exist_ok=True)
        # 断点续渲的"链条完整性"开关。见 ``note_render`` / ``cached``：
        # 磁盘补齐的记录是靠"文件还在"判定命中的，一旦本次运行真的重渲了
        # 某一段，它后面所有补齐的段都不能再算命中 —— 否则会出现"第 3 段
        # 是新渲的、第 4 段却接着老的第 3 段尾巴"这种接不上的片子。
        self.resume_intact = True

    # --- paths -----------------------------------------------------------------
    def segment_path(self, index):
        return os.path.join(self.dir, "seg_%02d.mp4" % (index + 1))

    def tail_path(self, index):
        """The clip segment index+1 opens on. Kept even when a segment is repaired:
        it is the thing downstream already continues from."""
        return os.path.join(self.dir, "seg_%02d.tail.mp4" % (index + 1))

    def joined_path(self):
        return os.path.join(self.dir, "%s.mp4" % self.name)

    @property
    def manifest_path(self):
        return os.path.join(self.dir, "manifest.json")

    # --- manifest --------------------------------------------------------------
    def load(self):
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get("version") != MANIFEST_VERSION:
            return None
        return data

    def save(self, records, extra=None):
        data = {"version": MANIFEST_VERSION, "session": self.name, "segments": records}
        data.update(extra or {})
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.manifest_path)
        return data

    def disk_segments(self, limit=9999):
        """How many segments are already on disk, counting up from the first.

        Only a *contiguous* run counts: a session with 1, 2, 3 and 5 is four
        segments of progress, not five, because 4 is missing and 5 no longer
        continues from anything.
        """
        count = 0
        while count < limit and os.path.exists(self.segment_path(count)):
            count += 1
        return count

    def disk_segment_indices(self):
        """1-based indices that actually have a ``seg_NN.mp4`` on disk.

        Unlike :meth:`disk_segments` this does not stop at the first gap, so an
        orphaned segment (say 5 when 4 is missing) is still reported. That file
        is real output: the timeline shows its preview, so the "rendered" count
        must not silently exclude it and then disagree with the previews.
        Resuming still uses :meth:`disk_segments`, which is the contiguous run.
        """
        out = []
        try:
            for fn in os.listdir(self.dir):
                m = re.match(r"^seg_(\d+)\.mp4$", fn)
                if m:
                    out.append(int(m.group(1)))
        except OSError:
            return []
        return sorted(set(out))

    def probe_tail_frames(self, index):
        """Best-effort frame count of segment ``index``'s handoff clip.

        Only used to fill in the ``handoff`` of a record being recovered from
        disk. Without it the join does not know how many replayed frames to drop
        and every cut stutters. Imported lazily: this module is otherwise
        importable without torch, which keeps it testable on its own.
        """
        try:
            from . import video_io
            return int(video_io.frame_count(self.tail_path(index)))
        except Exception:      # never let a probe break a resume
            return 0

    def probe_segment_frames(self, index):
        """Best-effort frame count of segment ``index``'s own render.

        Needed by ``reconcile()``: a record rebuilt from disk used to carry a
        hard-coded ``length=0``, and ``director`` sums ``length`` to size the
        join (``total = sum(length) - sum(handoff[:-1])``). With every recovered
        record contributing 0, a session restored after its manifest was swept
        joined to <= 0 frames. Probe the real file instead -- it is the same
        lazy-import / never-raise contract as ``probe_tail_frames``.
        """
        try:
            from . import video_io
            return int(video_io.frame_count(self.segment_path(index)))
        except Exception:      # never let a probe break a resume
            return 0

    def reconcile(self, manifest, limit=9999):
        """补齐 manifest 里缺失的段记录：磁盘上有 seg_NN.mp4，manifest 却没记。

        怎么会发生：段是每段跑完就落盘的，manifest 也每段写一次，但一次
        「只渲 1 段」的排队（或画完之后被截断的 shots_json）会把
        ``records[:index] + [record]`` 写成只有一条，而 seg_02..seg_04 早就
        躺在磁盘上了。此时重跑，2..4 会被当成"没渲过"重新烧一遍显卡。

        补齐的记录打 ``recovered`` 标记。原始缓存键已经无从复原（它是
        prompt/参考图/上一段 key 的哈希），所以只要文件还在就认为这一段
        已完成，见 ``cached``。本次运行命中时会带着真实 key 重新写回
        manifest，下一次排队就回到普通的键比对。
        """
        on_disk = self.disk_segments(limit)
        # 既没有 manifest 又一段都没渲：没什么可补的，返回 None，别伪造一个空
        # manifest 出来 —— 调用方（比如修复节点）靠 None 判断"会话还没跑过"。
        if not manifest and not on_disk:
            return None
        if not manifest:
            manifest = {"version": MANIFEST_VERSION, "session": self.name,
                        "segments": [], "recovered": 0}
        segments = list(manifest.get("segments") or [])
        known = set()
        for i, rec in enumerate(segments):
            if isinstance(rec, dict):
                known.add(int(rec.get("index", i)))
        # 录的是帧数（和真实记录一致）。拼接靠它跳过回放帧，写 0 会让
        # 每个接口都多出一段重复的开头；而 director 又拿 sum(length) 定拼接
        # 总长，写 0 会让总长算成 0。所以这里必须真的去探一下文件。
        try:
            _fps = float(manifest.get("fps") or 0) or 0.0
        except (TypeError, ValueError):
            _fps = 0.0
        added = 0
        for i in range(on_disk):
            if i in known:
                continue
            has_tail = os.path.exists(self.tail_path(i))
            _len = self.probe_segment_frames(i)
            segments.append({
                "index": i,
                "key": None,
                "file": os.path.basename(self.segment_path(i)),
                "length": _len,
                "handoff": self.probe_tail_frames(i) if has_tail else 0,
                "seconds": round(_len / _fps, 3) if (_len and _fps) else 0.0,
                "seed": 0,
                "prompt": "", "signature": None,
                "has_tail": has_tail,
                "recovered": True,
            })
            added += 1
        if added:
            segments.sort(key=lambda r: int(r.get("index", 0)) if isinstance(r, dict) else 0)
            manifest = dict(manifest)
            manifest["segments"] = segments
            manifest["recovered"] = int(manifest.get("recovered") or 0) + added
        return manifest

    def note_render(self):
        """这一段是真的重渲了 —— 后面所有"磁盘补齐"的段都不再可信。"""
        self.resume_intact = False

    def cached(self, manifest, index, key, needs_tail):
        """True when segment ``index`` on disk is still exactly what we would render."""
        if not manifest:
            return False
        segments = manifest.get("segments", [])
        if index >= len(segments):
            return False
        record = segments[index]
        if not os.path.exists(self.segment_path(index)):
            return False
        if needs_tail and not os.path.exists(self.tail_path(index)):
            return False
        # 磁盘补齐的记录（见 ``reconcile``）：键无从复原，文件还在就算完成。
        # 但前提是本次运行到这儿为止一段都没重渲过，否则链条已经断了。
        if record.get("recovered"):
            return bool(self.resume_intact)
        if record.get("key") != key:
            return False
        return True
