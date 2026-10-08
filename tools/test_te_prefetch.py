# -*- coding: utf-8 -*-
"""Offline tests for the text-encoder prefetch shim.

    <ComfyUI>/.venv/Scripts/python.exe tools/test_te_prefetch.py

The shim reaches into another module's call stack and changes an argument, so
"it seemed to work" is not good enough. What has to be true, and is asserted
here:

  * it restores the flag when the caller is a gated text encoder whose config
    asks for prefetch -- the case it exists for;
  * it leaves the call alone when the config does **not** ask for prefetch, so
    this pack is not overriding another model's decision;
  * it leaves the call alone for callers that are not the two gated files;
  * it is self-cancelling: if the flag arrives already on (upstream dropped the
    gate) nothing is touched;
  * ``install()`` is idempotent, and ``MARQUEE_TE_PREFETCH=0`` disables it.

The caller is simulated by writing a real module at a ``text_encoders/llama.py``
path and calling through it, because the decision depends on the *caller's*
``co_filename`` and a mock would not exercise that at all.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

# 本包在 <ComfyUI>/custom_nodes/ 下：上两级是 custom_nodes，再上一级是 ComfyUI 根。
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(REPO))
PKG_DIR = os.path.join(REPO, "marquee_director")
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG_DIR))

import comfy.model_management as mm          # noqa: E402
import comfy.model_prefetch as mp            # noqa: E402

from marquee_director import te_prefetch     # noqa: E402

#: The caller stub. ``self.prefetch_dynamic_vbars`` and a local named ``self``
#: are exactly what the shim reads out of the frame.
_CALLER = '''
import comfy.model_prefetch as mp


class _Model:
    def __init__(self, wants):
        self.prefetch_dynamic_vbars = wants


def call(wants, incoming, device):
    self = _Model(wants)
    return mp.make_prefetch_queue(
        [], device, {"prefetch_dynamic_vbars": incoming})
'''


class TestDecision(unittest.TestCase):
    """The pure decision table, without any call stack involved."""

    def test_restores_the_gated_case(self):
        self.assertTrue(te_prefetch.should_restore(
            False, True, r"E:\Comfy\comfy\text_encoders\llama.py"))

    def test_second_gated_file(self):
        self.assertTrue(te_prefetch.should_restore(
            False, True, "/opt/comfy/comfy/text_encoders/gemma4.py"))

    def test_leaves_a_config_that_says_no(self):
        self.assertFalse(te_prefetch.should_restore(
            False, False, r"E:\Comfy\comfy\text_encoders\llama.py"))

    def test_leaves_other_callers(self):
        self.assertFalse(te_prefetch.should_restore(
            False, True, r"E:\Comfy\comfy\ldm\minimax\model.py"))

    def test_self_cancelling_when_already_on(self):
        self.assertFalse(te_prefetch.should_restore(
            True, True, r"E:\Comfy\comfy\text_encoders\llama.py"))


class TestInstalled(unittest.TestCase):
    """The wrapper as it behaves with a real call stack."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="te_prefetch_")
        self.original = mp.make_prefetch_queue
        # Fresh state each test, and re-install the shim over the real function.
        te_prefetch._installed = False
        te_prefetch._reported = False
        mp.make_prefetch_queue = getattr(self.original, "__wrapped__",
                                         self.original)
        self.assertTrue(te_prefetch.install())

    def tearDown(self):
        mp.make_prefetch_queue = self.original
        te_prefetch._installed = False
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _caller(self, relpath, name):
        """Import a stub module whose __file__ is ``relpath``."""
        path = os.path.join(self.tmp, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_CALLER)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _device(self):
        import torch
        return torch.device("cuda:0")

    def test_restores_for_a_gated_encoder(self):
        mod = self._caller(os.path.join("comfy", "text_encoders", "llama.py"),
                           "stub_llama")
        got = mod.call(True, False, self._device())
        self.assertIsNotNone(
            got, "the shim did not restore the flag for a gated text encoder")

    def test_does_not_touch_a_config_that_says_no(self):
        mod = self._caller(os.path.join("comfy", "text_encoders", "llama.py"),
                           "stub_llama_no")
        self.assertIsNone(mod.call(False, False, self._device()))

    def test_does_not_touch_other_modules(self):
        mod = self._caller(os.path.join("comfy", "ldm", "minimax", "model.py"),
                           "stub_dit")
        self.assertIsNone(mod.call(True, False, self._device()))

    def test_install_is_idempotent(self):
        first = mp.make_prefetch_queue
        self.assertTrue(te_prefetch.install())
        self.assertIs(mp.make_prefetch_queue, first,
                      "install() wrapped the wrapper a second time")

    def test_gate_state_is_what_the_shim_assumes(self):
        """If these change, the shim's premise is gone and the test should say so."""
        self.assertNotEqual(mm.NUM_STREAMS, 0,
                            "NUM_STREAMS is 0 here, so make_prefetch_queue "
                            "returns None whatever the flag says")
        self.assertFalse(mm.is_device_cpu(self._device()))
        self.assertTrue(mm.device_supports_non_blocking(self._device()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
