# -*- coding: utf-8 -*-
"""Offline tests for the per-step seam remask (borrowed from MiniMaxH3_Director).

    python tools/test_seam_remask.py

The thing under test is a *shape-sensitive* hook that runs on every sampling
step. Get the layout wrong and you do not get an exception, you get a corrupted
frame. So the tests are deliberately paranoid:

1. **Weights.** Step ratio must fall monotonically as sampling converges, the
   seam tokens must never drop below ``seam_min`` at any ratio, and ratio=1 must
   reproduce the static taper exactly (that is what "off by default, identical
   when it does nothing" means numerically).
2. **Shapes.** All three layouts ComfyUI actually hands the hook — packed
   ``[B,1,elems]``, 5D video, and nested (video, audio) — must get their temporal
   prefix rewritten and everything else left alone.
3. **Degradation.** An unrecognised layout, or anything that raises, must return
   the mask untouched. Static mask, never a broken frame.
4. **Install/uninstall.** The hook must land on a *clone* and come off again,
   or every segment leaks a model copy holding a closure over its own latent.
"""

import math
import os
import sys

import torch

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _comfy_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


COMFY = _comfy_root()
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))

from ComfyUI_Marquee_Director.marquee_director import engine as E  # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import session as S  # noqa: E402

FAILS = []


def check(label, cond, extra=""):
    if not cond:
        FAILS.append(label)
    print("%-58s %s %s" % (label, "PASS" if cond else "FAIL", extra))


def close(a, b, tol=1e-6):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


# --- 1. weights --------------------------------------------------------------
print("--- weights ---")
SIGMAS = [14.6, 12.0, 9.0, 6.0, 3.5, 1.8, 0.9, 0.3, 0.0]

ratios = [E.next_sigma_ratio(s, SIGMAS) for s in SIGMAS[:-1]]
check("ratio starts high (early steps may redraw)", ratios[0] > 0.5, "%.3f" % ratios[0])
check("ratio decays monotonically", all(b <= a + 1e-9 for a, b in zip(ratios, ratios[1:])),
      str(["%.3f" % r for r in ratios]))
check("ratio is 0 on the last step", E.next_sigma_ratio(0.3, SIGMAS) == 0.0)
check("ratio clamped to [0,1]", all(0.0 <= r <= 1.0 for r in ratios))
check("degenerate sigma -> 0", E.next_sigma_ratio(0.0, SIGMAS) == 0.0
      and E.next_sigma_ratio(float("nan"), SIGMAS) == 0.0)

STATIC = E.prefix_token_weights(12, seam_min=0.10)
check("static taper: head 1.0, seam 0.10",
      abs(STATIC[0] - 1.0) < 1e-9 and abs(STATIC[-1] - 0.10) < 1e-9, str(STATIC[-4:]))
check("ratio=1 reproduces the static taper exactly",
      close(E.live_prefix_weights(12, 1.0, 0.10), STATIC))
check("ratio=0 flattens everything to seam_min",
      close(E.live_prefix_weights(12, 0.0, 0.10), [0.10] * 12))

mid = E.live_prefix_weights(12, 0.5, 0.10)
check("ratio=0.5 scales the head", abs(mid[0] - 0.5) < 1e-9, "%.3f" % mid[0])
check("seam never below seam_min at any ratio",
      all(min(E.live_prefix_weights(12, r / 20.0, 0.10)[-4:]) >= 0.10 - 1e-9
          for r in range(21)))
check("weights stay within [0,1]",
      all(0.0 <= v <= 1.0 for r in range(21)
          for v in E.live_prefix_weights(12, r / 20.0, 0.10)))
check("seam_min=0 hard-locks the seam (mask 0 = keep, exactly)",
      E.live_prefix_weights(12, 0.0, 0.0)[-1] == 0.0)

# --- 2. shapes ---------------------------------------------------------------
print("\n--- mask shapes ---")
PREFIX = 12
VSHAPE = (1, 32, 20, 4, 4)          # [B,C,T,H,W]
state = E._PrefixRemask(PREFIX, SIGMAS, VSHAPE, seam_min=0.10)

# (a) packed [B,1,elems] -- what H3 actually hands the hook
elems = int(math.prod(VSHAPE[1:]))
packed = torch.ones((1, 1, elems + 500))
out = state.denoise_mask_function(torch.tensor(14.6), packed)
video = out[..., :elems].reshape(VSHAPE)
check("packed: prefix rewritten",
      abs(float(video[0, 0, 0, 0, 0]) - E.live_prefix_weights(
          PREFIX, E.next_sigma_ratio(14.6, SIGMAS), 0.10)[0]) < 1e-5,
      "%.3f" % float(video[0, 0, 0, 0, 0]))
check("packed: seam token at floor", abs(float(video[0, 0, 11, 0, 0]) - 0.10) < 1e-6,
      "%.3f" % float(video[0, 0, 11, 0, 0]))
check("packed: content past the prefix untouched",
      bool(torch.all(video[0, 0, PREFIX:] == 1.0)))
check("packed: audio tail untouched", bool(torch.all(out[..., elems:] == 1.0)))
check("packed: shape preserved", tuple(out.shape) == (1, 1, elems + 500))

# (b) plain 5D
five = torch.ones(VSHAPE)
out5 = state.denoise_mask_function(torch.tensor(14.6), five)
check("5D: prefix rewritten", bool(torch.all(out5[0, 0, :PREFIX] < 1.0)))
check("5D: rest untouched", bool(torch.all(out5[0, 0, PREFIX:] == 1.0)))

# (c) nested (video, audio)
audio_mask = torch.zeros((1, 1, 1, 64))
nested = E._nested(torch.ones(VSHAPE), audio_mask)
outn = state.denoise_mask_function(torch.tensor(14.6), nested)
streams = E._unbind_mask(outn)
check("nested: still two streams", len(streams) == 2)
check("nested: video prefix rewritten", bool(torch.all(streams[0][0, 0, :PREFIX] < 1.0)))
check("nested: audio stream left alone", bool(torch.all(streams[1] == 0.0)))

# (d) step-to-step: the same hook, later in sampling, must be tighter
late = state.denoise_mask_function(torch.tensor(0.3), torch.ones(VSHAPE))
check("later step locks the prefix harder",
      float(late[0, 0, 0, 0, 0]) < float(out5[0, 0, 0, 0, 0]),
      "%.3f -> %.3f" % (float(out5[0, 0, 0, 0, 0]), float(late[0, 0, 0, 0, 0])))

# --- 3. degradation ----------------------------------------------------------
print("\n--- degradation ---")
check("unknown layout returned untouched",
      state.denoise_mask_function(torch.tensor(14.6), torch.ones(4, 4)) is not None
      and bool(torch.all(state.denoise_mask_function(
          torch.tensor(14.6), torch.ones(4, 4)) == 1.0)))
weird = torch.ones((1, 1, 3))
out_w = state.denoise_mask_function(torch.tensor(14.6), weird)
check("short packed mask returned untouched", bool(torch.all(out_w == 1.0)))
check("None mask survives", state.denoise_mask_function(torch.tensor(14.6), None) is None)


class Boom:
    """A mask object that explodes on contact."""

    @property
    def ndim(self):
        raise RuntimeError("boom")


check("exception -> mask returned, no raise",
      state.denoise_mask_function(torch.tensor(14.6), Boom()) is not None)

# --- 4. install / uninstall --------------------------------------------------
print("\n--- install / uninstall ---")


class FakeModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        new = FakeModel()
        new.model_options = dict(self.model_options)
        return new

    def set_model_denoise_mask_function(self, fn):
        self.model_options["denoise_mask_function"] = fn


base = FakeModel()
patched, st = E.install_prefix_remask(base, PREFIX, SIGMAS, VSHAPE, seam_min=0.10)
check("hook installed on a clone, not the original",
      patched is not base and "denoise_mask_function" in patched.model_options
      and "denoise_mask_function" not in base.model_options)
check("state is retrievable", st is not None and st.prefix_steps == PREFIX)
check("state reachable from the clone",
      getattr(patched, "_marquee_prefix_remask", None) is st)

check("prefix_steps=0 -> not installed",
      E.install_prefix_remask(base, 0, SIGMAS, VSHAPE)[0] is base)
check("model without clone -> untouched",
      E.install_prefix_remask(object(), PREFIX, SIGMAS, VSHAPE)[0] is not None)

E.uninstall_prefix_remask(patched)
check("uninstall drops the hook", "denoise_mask_function" not in patched.model_options)
check("uninstall drops the state", not hasattr(patched, "_marquee_prefix_remask"))
check("uninstall on None is a no-op", E.uninstall_prefix_remask(None) is None)

# an object with no clone() at all -> must not raise, must hand it straight back
plain = object()
m2, s2 = E.install_prefix_remask(plain, PREFIX, SIGMAS, VSHAPE)
check("model without clone(): returned as-is, no raise", m2 is plain and s2 is None)

# --- 5. cache keys -----------------------------------------------------------
print("\n--- cache keys ---")


def settings(**over):
    s = {"width": 960, "height": 544, "seed": 7, "session_name": "x",
         "ref_image_size": "match", "handoff_mode": "latent",
         "drift_arrest": 0.0, "seam_redraw": 0.10,
         "sigmas": None, "sampler": type("F", (), {
             "sampler_function": type("X", (), {"__name__": "euler"})(),
             "extra_options": {}})(), "model": type("M", (), {
                 "model": type("I", (), {}), "patches": {}})()}
    s.update(over)
    return s


seg = {"prompt": "hi", "length": 192, "seed": 0, "resolved_seed": 1,
       "images": [], "videos": [], "video_audios": [], "audios": []}

k_plain = S.segment_key(settings(), seg, 39, None)
k_first = S.segment_key(settings(seam_remask=True), seg, 39, None)
check("seam_remask on segment 1: key unchanged (no anchor to mask)",
      k_plain == k_first)

k_prev = S.segment_key(settings(), seg, 39, "prevkey")
k_prev_on = S.segment_key(settings(seam_remask=True), seg, 39, "prevkey")
check("seam_remask on later segments: key changes", k_prev != k_prev_on)
check("seam_remask=False: later segments unchanged",
      S.segment_key(settings(seam_remask=False), seg, 39, "prevkey") == k_prev)

print("\n%s" % ("ALL PASS" if not FAILS else "FAILED: %s" % ", ".join(FAILS)))
raise SystemExit(1 if FAILS else 0)
