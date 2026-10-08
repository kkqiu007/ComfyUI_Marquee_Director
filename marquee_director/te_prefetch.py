"""Put the text encoder's weight prefetch back, without editing ComfyUI.

The problem
-----------
``comfy/text_encoders/llama.py`` builds its layer prefetch queue like this::

    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        list(self.layers), x.device,
        {"prefetch_dynamic_vbars": self.prefetch_dynamic_vbars
                                   and past_key_values is not None})

``Qwen3VL_32BConfig.prefetch_dynamic_vbars`` is ``True`` -- the config *asks* for
prefetch -- but when a prompt is being encoded there is no KV cache yet, so
``past_key_values is None`` and the whole 25.88 GB text encoder is streamed layer
by layer with zero overlap. Measured on this machine, that is the single largest
wasted block in an H3 render: ~9-10 minutes per segment, about 35 minutes of a
1h41 render, with the GPU at 27% of its power limit, the disk at 2-6% of its
bandwidth, and the CPU near idle. py-spy landed in ``model_prefetch.py:184`` --
inside ``if queue is None:`` -- on 3 of 3 samples. ``gemma4.py`` carries the same
gate.

Why this is a shim and not a patch
----------------------------------
Editing ``comfy/**`` does not survive. Tried on 2026-09-22, and on 2026-09-23 the
07:49 restart auto-updated the install (``comfyVersion`` v0.37.0 -> v0.37.1),
restored ``llama.py`` to upstream, and deleted the ``.bak`` next to it. Anything
that has to keep working has to live in this pack.

How it works
------------
Wrap ``comfy.model_prefetch.make_prefetch_queue``. When it arrives with the flag
off, look one frame up the stack, at the caller:

* read the caller's own ``self.prefetch_dynamic_vbars``. If the model's config
  does **not** want prefetch, leave the call alone -- this shim only restores a
  flag the KV-cache gate cleared, it does not override a config decision.
* only act for the two text-encoder files that carry the gate.

So the override is narrow by construction, and it is self-cancelling: if upstream
ever drops the gate, the flag arrives ``True``, the shim does nothing, and
behaviour is upstream's.

Switch it off with ``MARQUEE_TE_PREFETCH=0``.
"""

import logging
import os
import sys

log = logging.getLogger(__name__)

#: ``MARQUEE_TE_PREFETCH=0`` disables the shim entirely.
ENABLED = os.environ.get("MARQUEE_TE_PREFETCH", "1").strip().lower() not in (
    "0", "false", "no", "off")

#: Only these files gate prefetch on the KV cache. Matched on the tail of the
#: path, so it works whatever ComfyUI's install root happens to be.
_GATED_SOURCES = ("text_encoders/llama.py", "text_encoders/gemma4.py")

_installed = False
_reported = False


def _is_gated_source(path):
    return any(str(path).replace("\\", "/").endswith(tail)
               for tail in _GATED_SOURCES)


def should_restore(incoming_flag, wants_prefetch, source_file):
    """Whether to turn the flag back on. Pure, so it can be tested directly.

    All three conditions have to hold, and each one rules out a different
    mistake:

    * ``incoming_flag`` already on -> nothing to do. This is what makes the shim
      self-cancelling once upstream drops the gate.
    * ``wants_prefetch`` false -> the model's config does not want prefetch, and
      that is a decision, not an accident. Overriding it would be this pack
      changing someone else's model behaviour.
    * the caller is not one of the two gated files -> not our business.
    """
    if incoming_flag:
        return False
    if not wants_prefetch:
        return False
    return _is_gated_source(source_file)


def _restore(transformer_options):
    """Return options with the flag restored, or the input unchanged."""
    try:
        caller = sys._getframe(2)           # _restore -> wrapper -> forward
    except ValueError:                      # pragma: no cover - depth changed
        return transformer_options
    model = caller.f_locals.get("self")
    wants = bool(getattr(model, "prefetch_dynamic_vbars", False))
    if not should_restore(transformer_options.get("prefetch_dynamic_vbars",
                                                  False),
                          wants, caller.f_code.co_filename):
        return transformer_options
    out = dict(transformer_options)
    out["prefetch_dynamic_vbars"] = True
    _report_once(wants)
    return out


def _report_once(wants):
    global _reported
    if _reported:
        return
    _reported = True
    log.info("[Marquee] text-encoder prefetch: restored the flag the "
             "KV-cache gate cleared (config asked for prefetch: %s). This is "
             "the shim that replaces the comfy/text_encoders/llama.py patch, "
             "which a ComfyUI update reverts. Set MARQUEE_TE_PREFETCH=0 to "
             "disable.", wants)


def install():
    """Wrap ``make_prefetch_queue`` once. Returns True if the shim is in place."""
    global _installed
    if _installed:
        return True
    if not ENABLED:
        log.info("[Marquee] text-encoder prefetch shim disabled by "
                 "MARQUEE_TE_PREFETCH")
        return False
    try:
        import comfy.model_prefetch as mp
    except Exception as exc:
        log.warning("[Marquee] text-encoder prefetch shim skipped, could not "
                    "import comfy.model_prefetch (%s: %s)",
                    type(exc).__name__, exc)
        return False

    original = getattr(mp, "make_prefetch_queue", None)
    if original is None:
        log.warning("[Marquee] text-encoder prefetch shim skipped: "
                    "comfy.model_prefetch.make_prefetch_queue is gone "
                    "(upstream changed); nothing was patched")
        return False
    if getattr(original, "_marquee_te_prefetch", False):
        _installed = True
        return True

    def make_prefetch_queue(queue, device, transformer_options):
        options = transformer_options
        try:
            if not options.get("prefetch_dynamic_vbars", False):
                options = _restore(options)
        except Exception as exc:
            # A shim that can take down a render is worse than the wasted
            # minutes it saves, so any surprise falls back to upstream exactly.
            log.warning("[Marquee] text-encoder prefetch shim skipped this "
                        "call (%s: %s)", type(exc).__name__, exc)
            options = transformer_options
        return original(queue, device, options)

    make_prefetch_queue._marquee_te_prefetch = True
    mp.make_prefetch_queue = make_prefetch_queue
    _installed = True
    log.info("[Marquee] text-encoder prefetch shim installed "
             "(wraps comfy.model_prefetch.make_prefetch_queue; no ComfyUI file "
             "was modified)")
    return True
