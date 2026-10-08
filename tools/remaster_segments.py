# -*- coding: utf-8 -*-
"""Bring an existing session's segments up to delivery level, on disk.

    python tools/remaster_segments.py <session_dir>
    python tools/remaster_segments.py <session_dir> --dry-run
    python tools/remaster_segments.py <session_dir> --no-mp4

Why this exists: the segment pass moved to *save* time, so any segment a new
build writes is already at delivery level and can be judged on its own. Segments
already sitting on disk were written before that change and are still raw --
7-8 LU down -- and they will stay that way until the film is joined, which makes
"the single segment sounds thin" a fair complaint about files that the fix has
not reached yet.

This closes that gap without a re-render: it runs the same ``master_segment``
entry point over each segment's own audio and writes the result **next to** the
original, never over it.

    seg_01.mp4            untouched
    seg_01.remastered.wav the mastered soundtrack, float PCM
    seg_01.remastered.mp4 video copied, new audio muxed at 192k

Segments that are already at the target are skipped, so the tool is safe to
re-run and safe to point at a session a new build produced. Nothing here is
picked up by the render, the panel, or the join: the session's own file list
matches ``^seg_NN.mp4$`` exactly, and the join reads the manifest.

Exit code is 0 unless something could not be written.
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import types

import numpy as np

# 本包在 <ComfyUI>/custom_nodes/ 下：上两级是 custom_nodes，再上一级是 ComfyUI 根。
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(REPO))
PKG_DIR = os.path.join(REPO, "marquee_director")


def load_modules():
    """Import marquee_director's modules without running the package __init__."""
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

    return load("video_io"), sys.modules["marquee_director.audio_master"]


def measure(am, waveform, rate):
    arr = np.asarray(waveform.detach().cpu()
                     if hasattr(waveform, "detach") else waveform)
    if arr.ndim == 3:
        arr = arr[0]
    lin = float(am.true_peak(arr, rate))
    return (float(am.measure_lufs(arr, rate)),
            20.0 * np.log10(max(lin, 1e-12)))


def mux(video_in, wav_in, out_path):
    """Copy the video, replace the audio. Returns None on success, else a reason."""
    tmp = out_path + ".tmp.mp4"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", video_in, "-i", wav_in,
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", tmp],
        capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return (proc.stderr or "ffmpeg failed").strip().splitlines()[-1]
    os.replace(tmp, out_path)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and report, write nothing")
    ap.add_argument("--no-mp4", action="store_true",
                    help="write only the WAV sidecar, skip the mp4 mux")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing .remastered output")
    args = ap.parse_args()

    video_io, am = load_modules()
    session = os.path.abspath(args.session_dir)
    if not os.path.isdir(session):
        raise SystemExit("not a directory: " + session)

    segs = sorted(f for f in os.listdir(session)
                  if re.match(r"^seg_\d+\.mp4$", f))
    if not segs:
        raise SystemExit("no segments in " + session)

    print("session : %s" % session)
    print("target  : %.1f LUFS, ceiling %.1f dBTP%s\n"
          % (am.TARGET_LUFS, am.CEILING_DBTP,
             "   [dry run]" if args.dry_run else ""))

    have_ffmpeg = bool(shutil.which("ffmpeg"))
    if not args.dry_run and not args.no_mp4 and not have_ffmpeg:
        print("ffmpeg not on PATH: writing WAV only\n")

    problems = []
    print("  %-12s %9s %9s %9s %9s  %s"
          % ("segment", "as-found", "mastered", "gain dB", "dBTP", "action"))
    for name in segs:
        path = os.path.join(session, name)
        stem = os.path.splitext(path)[0]
        sidecar = video_io._wav_path(path)
        clip = video_io.read_audio(sidecar) if os.path.exists(sidecar) else None
        src = "wav" if clip is not None else "mp4"
        if clip is None:
            clip = video_io.read_audio(path)
        if clip is None:
            print("  %-12s %9s %9s %9s %9s  no audio track"
                  % (name, "-", "-", "-", "-"))
            continue

        rate = int(clip["sample_rate"])
        whole = clip["waveform"][0].detach().cpu().numpy()
        before, before_tp = measure(am, whole, rate)

        if am.is_mastered(whole, rate):
            print("  %-12s %9.2f %9s %9s %9.2f  already at level, skipped (%s)"
                  % (name, before, "-", "-", before_tp, src))
            continue

        done = am.master_segment(whole, rate)
        after, after_tp = measure(am, done, rate)

        if args.dry_run:
            print("  %-12s %9.2f %9.2f %+9.2f %9.2f  would write (%s)"
                  % (name, before, after, after - before, before_tp, src))
            continue

        import torch
        wav_out = stem + ".remastered.wav"
        if os.path.exists(wav_out) and not args.force:
            print("  %-12s %9.2f %9.2f %+9.2f %9.2f  %s exists, skipped"
                  % (name, before, after, after - before, before_tp,
                     os.path.basename(wav_out)))
            continue
        video_io.save_wav(wav_out, {"waveform": torch.from_numpy(done)[None],
                                    "sample_rate": rate})

        action = "wrote " + os.path.basename(wav_out)
        if not args.no_mp4 and have_ffmpeg:
            mp4_out = stem + ".remastered.mp4"
            if os.path.exists(mp4_out) and not args.force:
                action += "  (%s exists, not remuxed)" % os.path.basename(mp4_out)
            else:
                reason = mux(path, wav_out, mp4_out)
                if reason is None:
                    action += " + " + os.path.basename(mp4_out)
                else:
                    action += "  (mux failed: %s)" % reason
                    problems.append("%s: mux failed -- %s" % (name, reason))
        print("  %-12s %9.2f %9.2f %+9.2f %9.2f  %s"
              % (name, before, after, after - before, before_tp, action))

    print()
    if problems:
        print("%d problem(s):" % len(problems))
        for p in problems:
            print("   ! %s" % p)
        return 1
    print("done -- originals untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
