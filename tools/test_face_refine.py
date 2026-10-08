# -*- coding: utf-8 -*-
"""Offline checks for the H3 FaceRefine port. No GPU, no ComfyUI server.

Run from the ComfyUI root (so ``comfy``/``comfy_extras`` import)::

    .venv/Scripts/python.exe custom_nodes/ComfyUI_Marquee_Director/tools/test_face_refine.py

What it pins down, in the order it matters:

1. **Unwired means untouched.** No pack / pack without sigmas -> ``apply_face_refine``
   returns the *same object* and an empty note. This is the whole "zero regression"
   promise of an optional external node, and it is the thing that would silently
   break if someone later adds an eager branch.
2. **Seam fade endpoints are exact.** Frame 0 and the last frame must equal the
   unrefined base bit-for-bit; that is what the next segment's anchor opens on.
   A 0.998 weight is a visible step at a join.
3. **Cache key contract.** No face pack -> payload byte-identical to the manual
   reproduction of the old key. Wired -> changed. Different detector -> changed.
4. **Per-frame face rect.** The loop-variable bug (sm_fw[i] with a stale i) made
   every frame use the last frame's face size; a growing face must produce a
   growing rect.
5. **Replay sizing.** 12 latent tokens -> 39 pixel frames, so the fade covers the
   whole handoff window.
6. **inject_video_latent** replaces only the video stream of the joint AV latent.
"""

import hashlib
import json
import os
import sys

import torch

def _comfy_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


COMFY = _comfy_root()
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))

from ComfyUI_Marquee_Director.marquee_director import face_refine as FR   # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import session as S        # noqa: E402

FAILS = []


def check(label, ok, extra=""):
    print("%-58s %s %s" % (label, "PASS" if ok else "FAIL", extra))
    if not ok:
        FAILS.append(label)


# ---------------------------------------------------------------------------
# 1. unwired == untouched
# ---------------------------------------------------------------------------
imgs = torch.rand(6, 32, 32, 3)
out, note = FR.apply_face_refine({}, {}, imgs)
check("no pack: same object returned", out is imgs)
check("no pack: empty note", note == "", repr(note))

out, note = FR.apply_face_refine({"face_refine": None}, {}, imgs)
check("pack=None: same object", out is imgs)

out, note = FR.apply_face_refine({"face_refine": {"detector": "face_yolov8m.pt"}},
                                 {}, imgs)
check("pack without sigmas: same object (sigmas is the switch)", out is imgs)
check("pack without sigmas: empty note", note == "", repr(note))

# ---------------------------------------------------------------------------
# 2. seam fade endpoints
# ---------------------------------------------------------------------------
N = 24
base = torch.zeros(N, 8, 8, 3)
refined = torch.ones(N, 8, 8, 3)          # fully "refined" == white
got = FR.fade_stitch_at_seams(refined, base, head_frames=6, tail_frames=6)
check("head frame 0 == base exactly", torch.equal(got[0], base[0]))
check("last frame == base exactly", torch.equal(got[-1], base[-1]))
check("middle frame == refined exactly", torch.equal(got[N // 2], refined[N // 2]))
# ramp strictly increasing across the head window
h = [float(got[i, 0, 0, 0]) for i in range(7)]
check("head ramp monotonic increasing", all(h[i] < h[i + 1] for i in range(6)),
      "->".join("%.3f" % v for v in h[:4]))
t = [float(got[N - 6 + i, 0, 0, 0]) for i in range(6)]
check("tail ramp monotonic decreasing", all(t[i] > t[i + 1] for i in range(5)),
      "->".join("%.3f" % v for v in t[-3:]))

# seam_fade off -> untouched
same = FR.fade_stitch_at_seams(refined, base, head_frames=0, tail_frames=0)
check("fade window 0: unchanged", torch.equal(same, refined))

# head+tail longer than the clip: clamped, no crash, endpoints still exact
short = FR.fade_stitch_at_seams(refined[:5].clone(), base[:5].clone(),
                                head_frames=40, tail_frames=40)
check("window > clip: no crash, shape kept", short.shape == refined[:5].shape)
check("window > clip: frame 0 still base", torch.equal(short[0], base[0]))

# non-tensor / degenerate input passes through untouched
sentinel = object()
check("non-tensor: passed through", FR.fade_stitch_at_seams(sentinel, base) is sentinel)

# ---------------------------------------------------------------------------
# 5. replay sizing
# ---------------------------------------------------------------------------
# ★ 2026-09-26：这里原先测 ``FR.replay_frames_of()`` —— 那个 helper 在
#   face_refine 里**从来不存在**（engine.py:833 早就把它内联成
#   ``pixel_frames_for_tokens(anchor["video_latent"].shape[2])`` 了），
#   所以这个测试跑到第 110 行必挂 AttributeError，等于整份测试没在跑。
#   改成测真正在用的那个函数，回归覆盖才留得住。
from ComfyUI_Marquee_Director.marquee_director import engine as ENG   # noqa: E402

pixel_frames_for_tokens = ENG.pixel_frames_for_tokens

check("12 latent tokens -> 39 pixel frames",
      pixel_frames_for_tokens(12) == 39, str(pixel_frames_for_tokens(12)))
check("0 tokens -> 0 pixel frames", pixel_frames_for_tokens(0) == 0)
check("1 token -> 1 pixel frame", pixel_frames_for_tokens(1) == 1)
# 17k+5 网格：第 0/5/10… 个 token 装 1 帧，其余每个装 4 帧。
check("5 tokens -> 17 pixel frames (1+4*4)",
      pixel_frames_for_tokens(5) == 17, str(pixel_frames_for_tokens(5)))

# ---------------------------------------------------------------------------
# 4. per-frame face rect (the stale-loop-variable bug)
# ---------------------------------------------------------------------------
class _Res:
    def __init__(self, boxes):
        self.boxes = type("B", (), {
            "__len__": lambda self: len(boxes),
            "xyxy": type("T", (), {"tolist": lambda self: boxes})(),
        })()


class _FakeDetector:
    """Face grows from 20px to 60px over the clip."""

    def __init__(self, n):
        self.n = n
        self.i = 0

    def predict(self, img, conf=0.25, verbose=False):
        i = self.i
        self.i += 1
        h = 20.0 + 40.0 * (i / max(self.n - 1, 1))
        cx, cy = 64.0, 64.0
        return [_Res([[cx - h / 2, cy - h / 2, cx + h / 2, cy + h / 2]])]


NFR = 40
clip = torch.rand(NFR, 128, 128, 3)
FR._DETECTOR_CACHE["_test_grow.pt"] = _FakeDetector(NFR)
pack = {
    "detector": "_test_grow.pt", "confidence": 0.25, "select": "largest_face",
    "crop_factor": 2.5, "canvas_mode": "auto_768", "canvas_width": 768,
    "canvas_height": 768,
}
crops, transform, report = FR.track_and_crop(clip, pack)
check("track: crops produced", crops is not None and transform is not None)
rects = transform["face_rect"]
heights = [r[3] for r in rects]
check("face rect varies per frame (not stuck on last frame)",
      (max(heights) - min(heights)) > 1e-3,
      "min=%.3f max=%.3f" % (min(heights), max(heights)))
check("face rect grows with the face", heights[-1] > heights[0],
      "%.3f -> %.3f" % (heights[0], heights[-1]))
check("one rect per frame", len(rects) == NFR, str(len(rects)))

# no face at all -> graceful skip
class _EmptyDetector:
    def predict(self, img, conf=0.25, verbose=False):
        return [_Res([])]


FR._DETECTOR_CACHE["_test_empty.pt"] = _EmptyDetector()
c2, t2, note2 = FR.track_and_crop(clip, dict(pack, detector="_test_empty.pt"))
check("no face: (None, None, note)", c2 is None and t2 is None)
check("no face: note explains why", FR.SKIP_NO_FACE in note2, note2[:60])

# no face detected -> apply_face_refine hands the clip back untouched
FR._DETECTOR_CACHE["_test_empty.pt"] = FR._DETECTOR_CACHE.get(
    "_test_empty.pt") or _EmptyDetector()
out, note = FR.apply_face_refine(
    {"face_refine": dict(pack, detector="_test_empty.pt",
                         sigmas=torch.tensor([0.4, 0.0]))},
    {}, clip)
check("no face: clip returned untouched", out is clip)
check("no face: note says skipped", FR.SKIP_NO_FACE in note, note[:40])

# a broken pass must degrade to the unrefined clip, not kill the render
out, note = FR.apply_face_refine(
    {"face_refine": dict(pack, sigmas=torch.tensor([0.4, 0.0]))},
    {}, clip)          # settings has no clip/vae/model -> must raise internally
check("broken pass: clip survives", out is clip)
check("broken pass: note explains the skip", "跳过" in note, note[:40])

# ---------------------------------------------------------------------------
# 3. cache key contract
# ---------------------------------------------------------------------------
class _Fn:
    def __init__(self, name="euler"):
        self.__name__ = name


class _Sampler:
    def __init__(self):
        self.sampler_function = _Fn()
        self.extra_options = {}


class _Model:
    class model:
        pass

    patches = {}


settings = {
    "width": 960, "height": 544, "seed": 7, "session_name": "x",
    "ref_image_size": "match", "handoff_mode": "latent", "drift_arrest": 0.0,
    "seam_redraw": 0.10, "sigmas": None, "sampler": _Sampler(), "model": _Model(),
}
segment = {
    "prompt": "hi", "length": 192, "seed": 0, "resolved_seed": 12345,
    "images": [], "videos": [], "video_audios": [], "audios": [],
}

key_no_face = S.segment_key(settings, segment, 39, None)

# Manual reproduction of the payload as it stood before FaceRefine existed.
manual = {
    "prev": None, "prompt": "hi", "length": 192, "handoff": 39, "seed": 12345,
    "width": 960, "height": 544, "ref_image_size": "match",
    "sigmas": S.digest(None), "sampler": S.sampler_digest(settings["sampler"]),
    "model": S.model_digest(settings["model"]),
    "images": [], "videos": [], "video_audios": [], "audios": [],
}
expected = hashlib.sha1(json.dumps(manual, sort_keys=True).encode()).hexdigest()[:20]
check("no face pack: key unchanged vs manual payload",
      key_no_face == expected, "%s vs %s" % (key_no_face, expected))

sigmas = torch.tensor([0.5, 0.25, 0.0])
face_pack = {
    "sigmas": sigmas, "detector": "face_yolov8m.pt", "confidence": 0.35,
    "select": "largest_face", "crop_factor": 2.5, "canvas_mode": "auto_768",
    "canvas_width": 768, "canvas_height": 768, "sampler": None,
    "seed_mode": "跟随一采", "paste_region": "face_only", "feather": 24,
    "blend": 1.0, "seam_fade": "跟随回放衔接",
}
key_face = S.segment_key(dict(settings, face_refine=face_pack), segment, 39, None)
check("wired: key changes", key_face != key_no_face)

key_other_det = S.segment_key(
    dict(settings, face_refine=dict(face_pack, detector="other.pt")), segment, 39, None)
check("different detector: key changes", key_other_det != key_face)

key_other_conf = S.segment_key(
    dict(settings, face_refine=dict(face_pack, confidence=0.6)), segment, 39, None)
check("different confidence: key changes", key_other_conf != key_face)

key_no_sigmas = S.segment_key(
    dict(settings, face_refine=dict(face_pack, sigmas=None)), segment, 39, None)
check("wired but no sigmas: key unchanged (sigmas is the switch)",
      key_no_sigmas == key_no_face)

# ---------------------------------------------------------------------------
# 6. inject_video_latent
# ---------------------------------------------------------------------------
try:
    from comfy.nested_tensor import NestedTensor
except Exception:                                          # pragma: no cover
    NestedTensor = None

if NestedTensor is not None:
    video = torch.zeros(1, 16, 12, 4, 4)
    audio = torch.ones(1, 8, 40, 2)
    av = {"samples": NestedTensor((video, audio))}

    TOK = 12      # H3 packed token dim for 39 frames (see pixel_frames_for_tokens)

    class _Vae5D:
        """A video VAE: temporal dim already compressed to latent tokens."""

        def encode(self, x):
            return torch.zeros(1, 16, TOK, 4, 4)

    class _Vae4D:
        """A plain image VAE: [N,C,H,W] -> the 4D branch lifts T to the batch."""

        def __init__(self, t):
            self.t = t

        def encode(self, x):
            return torch.zeros(self.t, 16, 4, 4)

    got_av = FR.inject_video_latent(av, torch.rand(39, 32, 32, 3), _Vae5D())
    parts = list(got_av["samples"].unbind())
    check("inject (5D vae): still two streams", len(parts) == 2)
    check("inject (5D vae): audio stream untouched", torch.equal(parts[1], audio))
    check("inject (5D vae): video shape preserved",
          tuple(parts[0].shape) == tuple(video.shape), str(tuple(parts[0].shape)))
    check("inject: original dict not mutated",
          torch.equal(list(av["samples"].unbind())[0], video))

    got4 = FR.inject_video_latent(av, torch.rand(TOK, 32, 32, 3), _Vae4D(TOK))
    p4 = list(got4["samples"].unbind())[0]
    check("inject (4D vae): lifted to [1,C,T,H,W]",
          tuple(p4.shape) == (1, 16, TOK, 4, 4), str(tuple(p4.shape)))

    # temporal mismatch is trimmed, not crashed on (longer) ...
    got_trim = FR.inject_video_latent(av, torch.rand(39, 32, 32, 3), _Vae4D(39))
    check("inject: longer t trimmed to template",
          tuple(list(got_trim["samples"].unbind())[0].shape) == (1, 16, TOK, 4, 4))
    # ... and padded when shorter
    got_pad = FR.inject_video_latent(av, torch.rand(5, 32, 32, 3), _Vae4D(5))
    check("inject: shorter t padded to template",
          tuple(list(got_pad["samples"].unbind())[0].shape) == (1, 16, TOK, 4, 4))

    # a non-nested latent must be rejected, not silently unbound along batch
    try:
        FR.inject_video_latent({"samples": torch.zeros(2, 16, TOK, 4, 4)},
                               torch.rand(39, 32, 32, 3), _Vae5D())
        check("inject: plain tensor rejected", False, "no raise")
    except ValueError:
        check("inject: plain tensor rejected", True)
else:                                                      # pragma: no cover
    print("%-58s %s" % ("inject_video_latent", "SKIP (no comfy.nested_tensor)"))

# ---------------------------------------------------------------------------
print()
if FAILS:
    print("%d FAILED: %s" % (len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
print("all face-refine checks passed")
