# -*- coding: utf-8 -*-
"""2026-10-18 第三轮自查：钉死 video_io 重采样的依赖与数值契约。

守两条：
  ① ``video_io.py`` 不得 **import** torchaudio —— ComfyUI 核心 2026-09-22
    （b16023b）已移除 torchaudio 依赖，requirements 不再钉它；Desktop 下一次
    重建 venv 它就消失，届时 join 遇到混合采样率段会整条打崩，报错点离根因
    很远。docstring 里提及不算违约，只禁 import 语句。
  ② ``_resample_waveform`` 数值正确：440 Hz 正弦 44100 → 32000，长度按比例、
    主频不变、dtype 保留、batch 维不丢；同率调用返回等长副本。

跑法（ComfyUI venv）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_resample.py
"""
import os
import re
import sys

import numpy as np
import torch

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
for _p in (COMFY_ROOT, os.path.dirname(PKG), PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAILS = []


def check(label, cond, got=""):
    print(("  OK   " if cond else "  FAIL ") + label
          + ("" if cond else "  -> %r" % (got,)))
    if not cond:
        FAILS.append(label)


# --- ① 依赖契约 -------------------------------------------------------------
src = open(os.path.join(PKG, "marquee_director", "video_io.py"),
           encoding="utf-8").read()
imports_ta = re.search(r"^\s*(import|from)\s+torchaudio", src, re.M)
check("video_io.py does not import torchaudio", imports_ta is None,
      imports_ta.group(0) if imports_ta else "")

from marquee_director import video_io as V                        # noqa: E402

# --- ② 数值契约 -------------------------------------------------------------
fs_in, fs_out = 44100, 32000
t = np.arange(fs_in * 4) / float(fs_in)
sig = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
wf = torch.from_numpy(np.stack([sig, sig])).unsqueeze(0)      # (1, 2, N)
out = V._resample_waveform(wf, fs_in, fs_out)

want_len = int(round(sig.size * fs_out / float(fs_in)))
check("length follows the rate ratio",
      abs(out.shape[-1] - want_len) <= 2, out.shape[-1])
check("batch/channel shape kept", tuple(out.shape[:2]) == (1, 2),
      tuple(out.shape))
check("dtype kept", out.dtype == wf.dtype, out.dtype)

mono = out[0, 0].numpy().astype(np.float64)
spec = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
freqs = np.fft.rfftfreq(mono.size, 1.0 / fs_out)
peak_hz = float(freqs[int(np.argmax(spec))])
check("440 Hz tone survives 44100->32000", abs(peak_hz - 440.0) <= 1.0,
      peak_hz)
check("amplitude preserved (0.5 +- 5%)",
      0.475 <= float(np.abs(mono).max()) <= 0.525, float(np.abs(mono).max()))

same = V._resample_waveform(wf, 32000, 32000)
check("same-rate call is a length-preserving copy",
      same.shape == wf.shape and bool(torch.allclose(same, wf, atol=1e-6)),
      tuple(same.shape))

print("\n" + ("ALL PASS" if not FAILS else "FAILED: %s" % ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
