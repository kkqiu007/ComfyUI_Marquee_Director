"""Rebuild one finished film's soundtrack through the real join path.

Loads marquee_director.video_io without importing the package __init__ (which
would pull the whole node registry), then runs the *same* functions the render
uses: resolve_join_points -> _join_audio. The result is written as a float-PCM
WAV so the film can be remuxed with -c:v copy, no video re-encode.

Usage:
    python rebuild_film_audio.py <session_dir> [--skips 0,39,39,39]
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import types

import numpy as np

# 本包在 <ComfyUI>/custom_nodes/ 下：上两级是 custom_nodes，再上一级是 ComfyUI 根。
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(REPO))
PKG_DIR = os.path.join(REPO, "marquee_director")


def load_modules():
    """Import the marquee_director modules without running its __init__."""
    sys.path.insert(0, COMFY_ROOT)
    pkg = types.ModuleType("marquee_director")
    pkg.__path__ = [PKG_DIR]
    sys.modules["marquee_director"] = pkg

    def load(name):
        full = "marquee_director." + name
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(PKG_DIR, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    video_io = load("video_io")          # pulls .common and .audio_master
    return video_io, sys.modules["marquee_director.audio_master"]


def measure(audio_master, waveform, rate):
    """Integrated loudness (LUFS) and true peak (**dBTP**) of a waveform.

    ``true_peak`` returns a *linear* amplitude, not decibels -- reading its 0.794
    as "0.79 dBTP" is how a correct -2.0 dBTP master first looked like it was
    clipping. Convert before reporting.
    """
    arr = np.asarray(waveform.detach().cpu()
                     if hasattr(waveform, "detach") else waveform)
    if arr.ndim == 3:
        arr = arr[0]
    lin = float(audio_master.true_peak(arr, rate))
    dbtp = 20.0 * np.log10(max(lin, 1e-12))
    return float(audio_master.measure_lufs(arr, rate)), dbtp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--skips", default=None,
                    help="comma-separated skip_frames, one per segment")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    video_io, audio_master = load_modules()
    session = os.path.abspath(args.session_dir)

    # ``^seg_NN.mp4$`` exactly, the same pattern the panel and the session use.
    # A loose ``startswith("seg_")`` also catches ``seg_01.replaced.mp4`` (written
    # by the repair node) and ``seg_01.remastered.mp4`` (written by
    # ``remaster_segments.py``), and would then join takes that are not the film.
    segs = sorted(f for f in os.listdir(session)
                  if re.match(r"^seg_\d+\.mp4$", f))
    if not segs:
        raise SystemExit("no segments in " + session)

    if args.skips:
        skips = [int(v) for v in args.skips.split(",")]
    else:
        # 17k+5 grid: the replay length the engine asks for, in frames.
        skips = [0] + [39] * (len(segs) - 1)
    if len(skips) != len(segs):
        raise SystemExit("skips (%d) != segments (%d)" % (len(skips), len(segs)))

    parts = [(os.path.join(session, s), k) for s, k in zip(segs, skips)]

    print("== segments ==")
    total_in = 0
    for path, skip in parts:
        n = video_io.frame_count(path)
        total_in += n
        print("  %-12s frames=%-5d skip=%-4d kept=%d"
              % (os.path.basename(path), n, skip, n - skip))
    expect = total_in - sum(skips)
    print("  total kept = %d frames = %.3f s" % (expect, expect / args.fps))

    parts, notes = video_io.resolve_join_points(parts, fps=args.fps)
    print("== join notes ==")
    print(json.dumps(notes, ensure_ascii=False, indent=2))

    print("== per-segment (trimmed to kept frames) ==")
    # ``at target`` is the check ``_join_audio`` itself makes. A segment written
    # by a current build is mastered when it is saved and should read ``yes``;
    # ``no`` means it is still raw -- either an older build's file, or a tail
    # that was deliberately left alone.
    print("  %-12s %8s %9s %9s %9s %9s  %s"
          % ("segment", "src", "as-found", "at target", "if mastered", "gain dB",
             "as-found dBTP"))
    for path, skip in parts:
        sidecar = video_io._wav_path(path)
        clip = video_io.read_audio(sidecar) if os.path.exists(sidecar) else None
        src = "wav" if clip is not None else "mp4"
        if clip is None:
            clip = video_io.read_audio(path)
        rate = int(clip["sample_rate"])
        wf = clip["waveform"]
        if wf.ndim == 3:
            wf = wf[0]
        start = int(round(skip / float(args.fps) * rate))
        kept = video_io.frame_count(path) - skip
        want = int(round(kept / float(args.fps) * rate))
        cut = wf[..., start:start + want]
        raw_lufs, raw_tp = measure(audio_master, cut, rate)
        already = audio_master.is_mastered(cut.detach().cpu().numpy(), rate)
        m = audio_master.master_segment(cut.detach().cpu().numpy(), rate)
        m_lufs, m_tp = measure(audio_master, m, rate)
        print("  %-12s %8s %9.2f %9s %9.2f %+9.2f %9.2f"
              % (os.path.basename(path), src, raw_lufs, "yes" if already else "no",
                 m_lufs, m_lufs - raw_lufs, raw_tp))

    audio = video_io._join_audio(parts, args.fps)
    if audio is None:
        raise SystemExit("no audio produced")

    wf, rate = audio["waveform"], audio["sample_rate"]
    lufs, tp = measure(audio_master, wf, rate)
    samples = wf.shape[-1]
    print("== film soundtrack ==")
    print("  shape=%s rate=%d  duration=%.4f s (video expects %.4f s)"
          % (tuple(wf.shape), rate, samples / rate, expect / args.fps))
    print("  integrated=%.2f LUFS   true peak=%.2f dBTP" % (lufs, tp))
    print("  samples vs expected: %d vs %d (delta %d)"
          % (samples, round(expect / args.fps * rate),
             samples - round(expect / args.fps * rate)))

    out = args.out or os.path.join(session, "_film_audio.wav")
    video_io.save_wav(out, audio)
    print("  written: %s (%d bytes)" % (out, os.path.getsize(out)))


if __name__ == "__main__":
    main()
