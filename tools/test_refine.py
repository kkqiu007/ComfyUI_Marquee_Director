# -*- coding: utf-8 -*-
"""Offline regression tests for the H3 Refine (second-sample) path.

    python tools/test_refine.py

No GPU and no ComfyUI server needed. Run it with ComfyUI's own python, from
anywhere: ``_comfy_root()`` works out where ComfyUI is.

What it pins down:

1. **Cache keys.** A settings bundle without ``refine`` must hash to exactly
   what it hashed to before this feature existed — otherwise every session on
   disk silently re-renders the next time it is queued. Wiring refine, or
   changing any of its knobs, must change the key.
2. **The call sequence.** ``refine_samples`` must run N passes, feed pass N+1
   the output of pass N, use the right guider model and sampler, and give each
   pass its own noise.
3. **Off means off.** No refine key, or a refine key without a sigma schedule,
   has to return the first-pass latent untouched.
4. **Wardrobe idempotence.** The Director runs ``process_prompt`` and then hands
   the prompt to the segment node, which runs it again. If that were not a no-op
   the second time, prompts would grow every run and the cache reconciliation
   would be comparing different strings.
"""

import hashlib
import json
import os
import sys

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _comfy_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


COMFY = _comfy_root()
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))

from ComfyUI_Marquee_Director.marquee_director import director as D  # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import engine as E  # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import refine_nodes as R  # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import session as S  # noqa: E402
from ComfyUI_Marquee_Director.marquee_director import wardrobe as W  # noqa: E402
from ComfyUI_Marquee_Director.marquee_director.common import (  # noqa: E402
    REFINE_SEED_FOLLOW,
    REFINE_SEED_INDEPENDENT,
    REFINE_SEED_OFFSET,
)

FAILS = []


def check(label, cond, extra=""):
    if not cond:
        FAILS.append(label)
    print("%-56s %s %s" % (label, "PASS" if cond else "FAIL", extra))


class FakeSampler:
    def __init__(self, name="euler"):
        self.sampler_function = type("F", (), {"__name__": name})()
        self.extra_options = {}


class FakeModel:
    class model:
        pass

    patches = {}


def base_settings(**over):
    s = {"width": 960, "height": 544, "seed": 7, "session_name": "x",
         "ref_image_size": "match", "handoff_mode": "latent",
         "drift_arrest": 0.0, "seam_redraw": 0.10,
         "sigmas": None, "sampler": FakeSampler(), "model": FakeModel()}
    s.update(over)
    return s


def segment(**over):
    s = {"prompt": "hi", "length": 192, "seed": 0, "resolved_seed": 12345,
         "images": [], "videos": [], "video_audios": [], "audios": []}
    s.update(over)
    return s


# --- 1. cache keys -----------------------------------------------------------
print("--- cache keys ---")
settings = base_settings()
key_plain = S.segment_key(settings, segment(), 39, None)

# The pre-refine payload, written out by hand. This is the contract: if someone
# adds a key here by accident, every existing session re-renders.
payload = {
    "prev": None, "prompt": "hi", "length": 192, "handoff": 39, "seed": 12345,
    "width": 960, "height": 544, "ref_image_size": "match",
    "sigmas": S.digest(None), "sampler": S.sampler_digest(settings["sampler"]),
    "model": S.model_digest(settings["model"]),
    "images": [], "videos": [], "video_audios": [], "audios": [],
}
key_old = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
check("no refine: key unchanged vs pre-refine", key_plain == key_old,
      "%s vs %s" % (key_plain, key_old))

pack = R.H3RefineNode.execute(
    passes=1, sampler=R.REFINE_FOLLOW_SAMPLER, seed_mode=REFINE_SEED_FOLLOW,
    refine_seed=0, sigmas=[0.9, 0.5, 0.1])[0]
check("node packs what the engine reads",
      set(pack) == {"sigmas", "model", "passes", "sampler", "seed_mode", "seed"}
      and pack["passes"] == 1 and pack["sampler"] is None, str(pack))

key_ref = S.segment_key(base_settings(refine=pack), segment(), 39, None)
check("refine wired: key differs", key_ref != key_plain)
check("passes change: key differs",
      S.segment_key(base_settings(refine=dict(pack, passes=2)), segment(), 39, None)
      != key_ref)
check("seed_mode change: key differs",
      S.segment_key(
          base_settings(refine=dict(pack, seed_mode=REFINE_SEED_INDEPENDENT,
                                    seed=999)), segment(), 39, None) != key_ref)
check("wired but no sigmas: key == plain key",
      S.segment_key(base_settings(refine=dict(pack, sigmas=None)), segment(), 39,
                    None) == key_plain)
check("refine=<non-dict> cannot crash the key",
      S.segment_key(base_settings(refine="junk"), segment(), 39, None) == key_plain)

# --- 2. seed + sampler resolution -------------------------------------------
print("\n--- seed / sampler ---")
check("seed follow", E._refine_seed({"seed_mode": REFINE_SEED_FOLLOW, "seed": 42}, 100) == 100)
check("seed offset", E._refine_seed({"seed_mode": REFINE_SEED_OFFSET, "seed": 42}, 100) == 101)
check("seed independent", E._refine_seed({"seed_mode": REFINE_SEED_INDEPENDENT, "seed": 42}, 100) == 42)
check("seed default follows", E._refine_seed({}, 100) == 100)
sampler = FakeSampler("res_multistep")
check("sampler follows pass 1", E._refine_sampler({"sampler": sampler}, {}) is sampler)
check("sampler by name builds an object", E._refine_sampler({"sampler": {}}, {"sampler": None}) is not None)

# --- 3. source mode (the widget that had gone missing) ----------------------
print("\n--- 剧本来源 ---")
check("json", D._source_mode(D.SOURCE_JSON) == "json")
check("video", D._source_mode(D.SOURCE_VIDEO) == "video")
check("empty -> json", D._source_mode("") == "json")
check("None -> json", D._source_mode(None) == "json")
schema = D.H3DirectorNode.define_schema()
names = [i.id for i in schema.inputs
         if i.io_type in WIDGET_TYPES and not i.force_input]
check("schema exposes source + video_path widgets",
      "source" in names and "video_path" in names)
check("source/video_path are last (saved workflows shift otherwise)",
      names[-2:] == ["source", "video_path"], str(names[-3:]))

# --- 4. refine_samples call sequence (sampler mocked out) -------------------
print("\n--- refine_samples (mocked sampler) ---")
calls = []


def fake_execute(noise=None, guider=None, sampler=None, sigmas=None, latent_image=None):
    calls.append({"seed": getattr(noise, "seed", None),
                  "model": getattr(guider, "model", None),
                  "sampler": sampler,
                  "dict_latent": isinstance(latent_image, dict),
                  "in_samples": latent_image.get("samples") if isinstance(latent_image, dict) else None})
    return ({"samples": ("sampled", len(calls))},)


E.SamplerCustomAdvanced = type("S", (), {"execute": staticmethod(fake_execute)})
E.Guider_Basic = type("G", (), {
    "__init__": lambda self, model: setattr(self, "model", model),
    "set_conds": lambda self, conds: setattr(self, "conds", conds)})
E.Noise_RandomNoise = type("N", (), {"__init__": lambda self, seed: setattr(self, "seed", seed)})

MAIN, REFINE_MODEL, SIGMAS, FIRST = object(), object(), [0.4, 0.2], ("first", 0)
st = {"model": MAIN, "sampler": "sampler-main",
      "refine": {"sigmas": SIGMAS, "model": None, "passes": 2,
                 "sampler": None, "seed_mode": REFINE_SEED_FOLLOW, "seed": 0}}
out, note = E.refine_samples(st, FIRST, "positive", 777)
check("2 passes -> 2 sampler calls", len(calls) == 2, str(len(calls)))
check("latent arrives as a dict", all(c["dict_latent"] for c in calls))
check("pass 2 consumes pass 1's output", calls[1]["in_samples"] == ("sampled", 1))
check("model falls back to the main model", all(c["model"] is MAIN for c in calls))
check("sampler follows pass 1", all(c["sampler"] == "sampler-main" for c in calls))
check("fresh noise per pass", [c["seed"] for c in calls] == [777, 778],
      str([c["seed"] for c in calls]))
check("note reports the passes", "2 pass" in note, note)

calls.clear()
out, note = E.refine_samples({"model": MAIN, "sampler": "s"}, FIRST, "p", 1)
check("no refine key -> untouched", len(calls) == 0 and out is FIRST)
calls.clear()
out, _ = E.refine_samples(
    {"model": MAIN, "sampler": "s",
     "refine": {"sigmas": None, "model": None, "passes": 3, "sampler": None,
                "seed_mode": REFINE_SEED_FOLLOW, "seed": 0}}, FIRST, "p", 1)
check("wired but no sigmas -> off", len(calls) == 0 and out is FIRST)
calls.clear()
E.refine_samples({"model": MAIN, "sampler": "s",
                  "refine": {"sigmas": SIGMAS, "model": REFINE_MODEL, "passes": 1,
                             "sampler": None, "seed_mode": REFINE_SEED_FOLLOW,
                             "seed": 0}}, FIRST, "p", 5)
check("refine_model overrides the main model",
      len(calls) == 1 and calls[0]["model"] is REFINE_MODEL)
check("refine=<junk> cannot crash it",
      E.refine_samples({"model": MAIN, "sampler": "s", "refine": "junk"}, FIRST, "p", 5)[0] is FIRST)

# --- 5. wardrobe idempotence -------------------------------------------------
print("\n--- wardrobe ---")
sample = ("A woman in a red dress walks through the market. "
          "She wears a blue jacket and a green scarf.")
once, _ = W.process_prompt(sample)
twice, notes2 = W.process_prompt(once)
check("second pass is a no-op", once == twice,
      "" if once == twice else "\n  1: %r\n  2: %r" % (once, twice))
check("second pass reports nothing", not notes2, str(notes2))
check("empty input survives", W.process_prompt("")[0] == "")
check("non-str input survives", W.process_prompt(None)[0] is None)

print("\n%s" % ("ALL PASS" if not FAILS else "FAILED: %s" % ", ".join(FAILS)))
raise SystemExit(1 if FAILS else 0)
