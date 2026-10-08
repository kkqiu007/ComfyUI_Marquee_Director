# -*- coding: utf-8 -*-
"""Verify ``marquee_director/audio_master.py`` without starting ComfyUI.

    python tools/check_audio_master.py
    python tools/check_audio_master.py --audio <ComfyUI>/output/<run>/<film>.mp4
    python tools/check_audio_master.py --no-ffmpeg        # skip the cross-check

Why this exists: the loudness standard is easy to implement *almost* right, and
the two failure modes are both silent.

**Wrong filter, plausible number.** The obvious way to build the K-weighting
biquads is RBJ's EQ cookbook -- it is the formula everyone knows, it produces a
shelf of the right shape at the right frequency, and it is not the filter
BS.1770-4 specifies. The standard's pre-filter is a bilinear transform of an
analog prototype with ``K = tan(pi*f0/fs)`` and a fixed transition sharpness of
``Vh**0.4996667741545416``, not ``alpha = sin(w0)/(2Q)``. The RBJ version
reproduces the standard's tabulated 48 kHz coefficients only to ~5e-2, which
shows up as a systematic bias in every measurement rather than as an error
message. Brecht De Man's reconstruction (github.com/BrechtDeMan/loudness.py,
shipped by ``pyloudnorm`` as its ``DeMan`` filter class) is the one that matches
the table; test 1 below is the regression guard for that.

**Right number, wrong trade.** A loudness target and a peak ceiling can be
mutually unreachable, and when they are, the way you give way matters. Scaling
the whole track to fit the ceiling costs loudness across the entire render;
limiting costs it only where the peaks are. Test 4 pins the invariant that
separates them.

**Right ceiling, wrong domain.** A ceiling labelled dBTP but evaluated on
``abs(x).max()`` is a sample-peak ceiling, and a sample-peak limiter ships a
track about 1 dB hotter than the number it was given -- silently, because every
other assertion still passes. Test 4 checks the ceiling on the oversampled
signal, and checks the oversampling itself first.

The numbers here are cross-checked against ``ffmpeg -af ebur128`` (test 3),
because agreeing with an independent implementation of the same standard is the
only evidence that "it measures loudness" means anything. ffmpeg is optional --
if it is not on PATH the cross-check is skipped and reported as skipped, not as
a pass.

**Right routine, wrong shape.** ``master()`` used to answer a 3-D input by
returning it untouched -- before measuring anything, and therefore with nothing
in the log. The signal the join assembles is always 3-D, so the join squeezed it
inline, and that workaround is a copy of logic which only has to be forgotten
once to yield a silent no-op. Test 5 pins the contract instead: a batch must come
back mastered, the per-segment pass must collapse the segments' level spread, and
both join shortcuts (``hp_hz=None``, ``pre_limit=False``) must do what they say.

**Right routine, run at the wrong time.** The pass has to happen when a segment
is *saved* and not only when the film is joined, because segments are reviewed
one at a time and a raw one measures 7-8 LU under delivery level -- which is
what reads as thin and weak on its own. Running it in both places is only safe
if the second run can tell it is not needed, so test 6 pins ``is_mastered`` in
both directions: it must fire on a mastered segment and stay off for a raw one,
for a gain-capped one, and for anything it cannot measure.

Exit code is 0 only if every check passes.
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "..", "marquee_director", "audio_master.py")

#: ITU-R BS.1770-4 Table 1, valid at 48 kHz.
TABLE_48K = {
    "b_shelf": [1.53512485958697, -2.69169618940638, 1.19839281085285],
    "a_shelf": [1.00000000000000, -1.69065929318241, 0.73248077421585],
    "b_hpf":   [1.00000000000000, -2.00000000000000, 1.00000000000000],
    "a_hpf":   [1.00000000000000, -1.99004745483398, 0.99007225036621],
}

#: Rates the model and the encoders actually meet.
RATES = (32000, 44100, 48000)

#: ffmpeg prints one decimal, so agreement cannot be asserted tighter than that.
FFMPEG_TOLERANCE_LU = 0.15


def load_audio_master():
    """Import the module by path, not through the package.

    ``marquee_director/__init__.py`` pulls in ``nodes.py``, which imports
    ``comfy.utils`` -- only importable inside a running ComfyUI. Loading the
    single file keeps this check usable offline, which is the whole point.
    ``audio_master`` itself needs nothing but numpy and scipy.
    """
    path = os.path.abspath(MODULE)
    spec = importlib.util.spec_from_file_location("audio_master_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. The filter is the standard's filter
# ---------------------------------------------------------------------------

def check_k_weighting(am):
    problems = []
    (b_shelf, a_shelf), (b_hpf, a_hpf) = am._k_weighting(48000)
    got = {"b_shelf": b_shelf, "a_shelf": a_shelf, "b_hpf": b_hpf, "a_hpf": a_hpf}
    lines = []
    for key in ("b_shelf", "a_shelf", "b_hpf", "a_hpf"):
        ref = np.array(TABLE_48K[key])
        err = float(np.max(np.abs(np.asarray(got[key]) - ref)))
        ok = err < 1e-6
        lines.append("   %-8s max|err| = %.3e  %s" % (key, err, "ok" if ok else "FAIL"))
        if not ok:
            problems.append("%s deviates from the BS.1770-4 table by %.3e "
                            "(got %s, want %s)"
                            % (key, err, np.asarray(got[key]).tolist(), TABLE_48K[key]))
    return problems, lines


# ---------------------------------------------------------------------------
# 2. And it is stable at the rates we use
# ---------------------------------------------------------------------------

def check_stability(am):
    problems = []
    lines = []
    for fs in RATES:
        (_, a_shelf), (_, a_hpf) = am._k_weighting(fs)
        r_shelf = np.abs(np.roots(a_shelf))
        r_hpf = np.abs(np.roots(a_hpf))
        ok = bool(np.all(r_shelf < 1.0) and np.all(r_hpf < 1.0))
        lines.append("   %5d Hz  shelf |z| %.6f  hpf |z| %.6f  %s"
                     % (fs, r_shelf.max(), r_hpf.max(), "ok" if ok else "FAIL"))
        if not ok:
            problems.append("K-weighting is unstable at %d Hz" % fs)
    return problems, lines


# ---------------------------------------------------------------------------
# 3. ffmpeg agrees about how loud the signal is
# ---------------------------------------------------------------------------

def _ffmpeg_loudness(ffmpeg, path):
    """(integrated LUFS, true peak dBTP) as ffmpeg's ebur128 reports them."""
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", path,
         "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace")
    text = proc.stdout + proc.stderr
    i_matches = re.findall(r"\bI:\s*(-?\d+(?:\.\d+)?)\s*LUFS", text)
    p_matches = re.findall(r"\bPeak:\s*(-?\d+(?:\.\d+)?)\s*dBFS", text)
    if not i_matches:
        raise RuntimeError("ffmpeg produced no integrated loudness:\n%s"
                           % text[-2000:])
    lufs = float(i_matches[-1])
    peak = float(p_matches[-1]) if p_matches else float("nan")
    return lufs, peak


def _write_wav(path, waveform, fs):
    from scipy.io import wavfile
    wavfile.write(path, fs, np.ascontiguousarray(waveform.T, dtype=np.float32))


def _probe_signals(fs):
    """Signals whose loudness exercises different parts of the gate."""
    rng = np.random.RandomState(20260922)
    t = np.arange(fs * 10) / float(fs)

    sine = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    sine = np.tile(sine, (2, 1))

    noise = rng.randn(2, fs * 10).astype(np.float64) * 0.3

    # Half-second lines separated by half-second gaps: the absolute and relative
    # gates have to do real work here, which is the case a naive implementation
    # gets wrong by several LU.
    bursts = np.zeros((2, 0), dtype=np.float64)
    for _ in range(20):
        b = 0.5 * np.sin(2.0 * np.pi * 1000.0 * np.arange(fs // 2) / float(fs))
        bursts = np.concatenate([bursts, np.tile(b, (2, 1)),
                                 np.zeros((2, fs // 2))], axis=-1)

    return [("1 kHz sine", sine), ("white noise", noise), ("gated bursts", bursts)]


def check_against_ffmpeg(am, ffmpeg, audio_path, fs=32000):
    problems = []
    lines = []
    tmpdir = tempfile.mkdtemp(prefix="audio_master_check_")
    try:
        cases = list(_probe_signals(fs))
        if audio_path:
            from scipy.io import wavfile
            rate, data = wavfile.read(audio_path)
            arr = data if data.ndim == 2 else data[:, None]
            cases.append((os.path.basename(audio_path), arr.T.astype(np.float64)))
            fs = rate

        for name, waveform in cases:
            path = os.path.join(tmpdir, "probe.wav")
            _write_wav(path, waveform, fs)
            want, want_peak = _ffmpeg_loudness(ffmpeg, path)
            got = float(am.measure_lufs(waveform, fs))
            delta = got - want
            ok = abs(delta) <= FFMPEG_TOLERANCE_LU
            lines.append("   %-22s ours %8.2f  ffmpeg %8.2f  delta %+.2f LU  %s"
                         % (name, got, want, delta, "ok" if ok else "FAIL"))
            if not ok:
                problems.append("%s: measure_lufs %.2f vs ffmpeg %.2f LUFS "
                                "(delta %+.2f, tolerance %.2f)"
                                % (name, got, want, delta, FFMPEG_TOLERANCE_LU))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return problems, lines


# ---------------------------------------------------------------------------
# 4. Mastering hits the target and keeps the ceiling
# ---------------------------------------------------------------------------

def check_master(am):
    problems = []
    lines = []
    fs = 32000
    rng = np.random.RandomState(7)

    # 1 kHz at a level well inside the ceiling: the target must be reached
    # exactly, because nothing forces a compromise.
    t = np.arange(fs * 20) / float(fs)
    quiet = np.tile(0.02 * np.sin(2.0 * np.pi * 1000.0 * t), (2, 1))
    out = am.master(quiet, fs)
    lufs = float(am.measure_lufs(out, fs))
    peak_db = 20.0 * np.log10(float(np.abs(out).max()) + 1e-12)
    ok = abs(lufs - am.TARGET_LUFS) <= 0.3
    lines.append("   target reached       %8.2f LUFS (want %.1f)  %s"
                 % (lufs, am.TARGET_LUFS, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("master() landed at %.2f LUFS, target %.1f"
                        % (lufs, am.TARGET_LUFS))

    ok = peak_db <= am.CEILING_DBTP + 0.01
    lines.append("   sample ceiling held  %8.2f dBFS (ceiling %.1f)  %s"
                 % (peak_db, am.CEILING_DBTP, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("master() peak %.2f dBFS exceeds the ceiling %.1f"
                        % (peak_db, am.CEILING_DBTP))

    # ... and the ceiling has to hold on the *decoded* signal, which is the one
    # a sample-peak limiter passes while still shipping a track ~1 dB hot. The
    # measurement itself is checked first, on a signal built to overshoot:
    # a fs/4 cosine offset by pi/4 lands every sample at +-0.7071 while the
    # ideal reconstruction between them reaches 1.0. FIR oversampling reads
    # ~1.09 here and ffmpeg reads ~1.11 -- both over-read by design -- so the
    # assertion only asks that the detector finds overshoot at all, not that it
    # lands on a particular number.
    probe = np.cos(2.0 * np.pi * 0.25 * np.arange(fs) + np.pi / 4.0)
    probe = np.tile(probe, (2, 1))
    probe_sample = float(np.abs(probe).max())
    probe_true = am.true_peak(probe, fs)
    ok = probe_true > probe_sample * 1.2
    lines.append("   true_peak detector   sample %.4f, true %.4f  %s"
                 % (probe_sample, probe_true, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("true_peak() reported %.4f on a signal whose samples sit "
                        "at %.4f -- it is not detecting intersample peaks"
                        % (probe_true, probe_sample))

    tp_db = 20.0 * np.log10(am.true_peak(out, fs) + 1e-12)
    ok = tp_db <= am.CEILING_DBTP + 0.01
    lines.append("   true ceiling held    %8.2f dBTP (ceiling %.1f)  %s"
                 % (tp_db, am.CEILING_DBTP, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("master() true peak %.2f dBTP exceeds the ceiling %.1f "
                        "(the limiter is tracking sample peaks, not true peaks)"
                        % (tp_db, am.CEILING_DBTP))

    # The hard case, and the one this render actually presented: a quiet
    # dialogue bed with occasional near-full-scale transients. The crest factor
    # is ~40 dB against a ceiling that permits 12, so the target is *unreachable*
    # and the only question is how the code gives way. Whatever it does, the
    # ceiling is absolute and the loudness must not come out below the input --
    # that is what regressing to a whole-track scale-down would violate.
    spiky = np.tile(0.01 * np.sin(2.0 * np.pi * 1000.0 * np.arange(fs * 10) / float(fs)), (2, 1))
    for start in range(fs // 2, fs * 10 - 4, fs):
        spiky[:, start:start + 4] = 0.99 * np.sign(rng.randn(2, 4))
    before = float(am.measure_lufs(spiky, fs))
    out2 = am.master(spiky, fs)
    after = float(am.measure_lufs(out2, fs))
    peak2 = 20.0 * np.log10(float(np.abs(out2).max()) + 1e-12)
    ok = after >= before and peak2 <= am.CEILING_DBTP + 0.01
    lines.append("   high-crest track     %8.2f -> %.2f LUFS, peak %.2f  %s"
                 % (before, after, peak2, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("high-crest track: loudness %.2f -> %.2f, peak %.2f dBFS"
                        % (before, after, peak2))

    # Shape and dtype preservation, and "never raise" on odd input.
    #
    # The batch entry used to read "batch passes through", and that assertion was
    # itself the bug: it made "3-D in, 3-D out, unmastered" look like the
    # intended contract, when ``_join_audio`` is *always* 3-D and so was always
    # left unmastered. Test 5 now pins what a batch must actually do; here only
    # the shape is checked.
    checks = [
        ("stereo", np.zeros((2, fs * 2)), (2, fs * 2)),
        ("mono 1-D", np.zeros(fs * 2), (fs * 2,)),
        ("empty", np.zeros((2, 0)), (2, 0)),
        ("too short to gate", np.zeros((2, 100)), (2, 100)),
        ("batch shape kept", np.zeros((1, 2, fs * 2)), (1, 2, fs * 2)),
        ("NaN input", np.full((2, fs * 2), np.nan), (2, fs * 2)),
    ]
    for name, arr, shape in checks:
        try:
            result = am.master(arr, fs)
            got = tuple(np.shape(result))
            ok = got == shape
            lines.append("   %-20s -> %-16s %s" % (name, got, "ok" if ok else "FAIL"))
            if not ok:
                problems.append("%s: shape %s, expected %s" % (name, got, shape))
        except Exception as exc:
            lines.append("   %-20s raised %s: %s" % (name, type(exc).__name__, exc))
            problems.append("%s raised %s: %s" % (name, type(exc).__name__, exc))

    out32 = am.master(np.zeros((2, fs * 2), dtype=np.float32), fs)
    ok = out32.dtype == np.float32
    lines.append("   dtype preserved      %-16s %s" % (out32.dtype, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("master() returned %s, expected float32" % out32.dtype)

    # The limiter is the mechanism the whole trade-off rests on: it must never
    # hand back a gain above unity, or it would be amplifying rather than
    # limiting. And it must hold the ceiling as a *true* peak.
    loud = np.tile(0.95 * np.sin(2.0 * np.pi * 200.0 * np.arange(fs * 3) / float(fs)), (2, 1))
    limited = am.limiter(loud, fs, 0.2)
    ok = float(np.abs(limited).max()) <= 0.2 + 1e-6
    lines.append("   limiter holds ceil   %8.4f (ceil 0.2)  %s"
                 % (float(np.abs(limited).max()), "ok" if ok else "FAIL"))
    if not ok:
        problems.append("limiter let through %.4f above a 0.2 ceiling"
                        % float(np.abs(limited).max()))

    tp_limited = am.true_peak(limited, fs)
    ok = tp_limited <= 0.2 + 1e-3
    lines.append("   limiter true peak    %8.4f (ceil 0.2)  %s"
                 % (tp_limited, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("limiter true peak %.4f exceeds a 0.2 ceiling"
                        % tp_limited)

    return problems, lines


# ---------------------------------------------------------------------------
# 5. The join path: batch, per-segment match, and both shortcuts
# ---------------------------------------------------------------------------

def _band_energy(x, fs, f0, width=20.0):
    """Energy in a narrow band around ``f0``. Used to watch a filter act."""
    mono = np.asarray(x, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    windowed = mono * np.hanning(len(mono))
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / fs)
    band = (freqs >= f0 - width) & (freqs <= f0 + width)
    return float((spec[band] ** 2).sum())


def check_join_path(am):
    """The half of the story that ``master()`` alone cannot see.

    Test 4 proves one call does the right thing on a clean 2-D signal. This one
    proves the *join* gets the right thing done to it: the signal ``_join_audio``
    assembles is always ``(1, channels, samples)``, a batch must therefore come
    back mastered, and the per-segment pass must actually collapse the level
    spread the render arrives with.
    """
    problems = []
    lines = []
    fs = 32000

    # --- a batch is mastered, not waved through ----------------------------
    # The tone is deliberately not super quiet. The shaping stage costs about
    # 3 dB of the total gain budget (it removes energy, so the loop has to add
    # more back), and a probe sitting near ``_MAX_TOTAL_GAIN_DB`` would be capped
    # and then read as "the batch was not mastered" -- blaming the batch path for
    # a gain-cap boundary. The batch behaviour is what this checks.
    t = np.arange(fs * 12) / float(fs)
    tone = np.tile(0.05 * np.sin(2.0 * np.pi * 700.0 * t), (2, 1))
    batch = np.stack([tone, tone * 0.5])
    before = [float(am.measure_lufs(batch[i], fs)) for i in range(2)]
    out = am.master(batch, fs)
    after = [float(am.measure_lufs(out[i], fs)) for i in range(2)]
    ok = (tuple(np.shape(out)) == tuple(np.shape(batch))
          and all(abs(a - am.TARGET_LUFS) <= 0.4 for a in after))
    lines.append("   3-D batch mastered   %s -> %s LUFS   %s"
                 % (["%.1f" % b for b in before], ["%.1f" % a for a in after],
                    "ok" if ok else "FAIL"))
    if not ok:
        problems.append(
            "a 3-D batch came out at %s LUFS instead of ~%.1f: master() is "
            "handing the batch back untouched, and the join's signal is always "
            "3-D, so that is a silent no-op on the whole film"
            % (["%.1f" % a for a in after], am.TARGET_LUFS))

    # --- the per-segment pass collapses the spread -------------------------
    # H3 renders every segment in its own pass and the normaliser in
    # comfy_extras/nodes_audio.py never fires on this model, so segments arrive
    # up to ~8 LU apart. A single gain over the joined track cannot repair that,
    # it only moves the whole film -- which is why the rejected cut measured
    # -25.8 LUFS with two thirds of it 4-6 dB below the rest.
    level_db = [0.0, 3.2, 6.0, 7.8]
    segs = []
    for k, lv in enumerate(level_db):
        tt = np.arange(fs * 8) / float(fs)
        sig = sum(np.sin(2.0 * np.pi * f * tt + 0.7 * (k + 1) * j)
                  for j, f in enumerate((220.0, 330.0, 440.0, 550.0,
                                         660.0, 880.0)))
        segs.append(np.tile(0.01 * sig * (10.0 ** (-lv / 20.0)), (2, 1)))
    raw = [float(am.measure_lufs(s, fs)) for s in segs]
    done = [am.master_segment(s, fs) for s in segs]
    got = [float(am.measure_lufs(d, fs)) for d in done]
    raw_spread = max(raw) - min(raw)
    spread = max(got) - min(got)
    ok = spread <= 0.5 and all(abs(g - am.TARGET_LUFS) <= 0.4 for g in got)
    lines.append("   segment spread       %5.2f LU raw -> %5.2f LU mastered   %s"
                 % (raw_spread, spread, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("per-segment pass left a %.2f LU spread (raw %.2f LU): %s"
                        % (spread, raw_spread, ["%.2f" % g for g in got]))

    # --- the film pass reaches the target with both shortcuts off ----------
    # ``shape_tone=False`` mirrors what ``_join_audio`` actually passes: the
    # pieces were already shaped by the per-segment pass, and shaping them again
    # at the join would double every gain (a 2.5 dB shelf becomes 5 dB, a 3.5 dB
    # dip becomes 7 dB). The film pass is a *level* pass, not a tone pass.
    joined = np.concatenate(done, axis=-1)
    tp_joined = 20.0 * np.log10(am.true_peak(joined, fs) + 1e-12)
    film = am.master(joined, fs, hp_hz=None, pre_limit=False, shape_tone=False)
    lufs = float(am.measure_lufs(film, fs))
    tp = 20.0 * np.log10(am.true_peak(film, fs) + 1e-12)
    ok = abs(lufs - am.TARGET_LUFS) <= 0.4 and tp <= am.CEILING_DBTP + 0.05
    lines.append("   film pass            %6.2f LUFS, %6.2f dBTP (ceil %.1f)   %s"
                 % (lufs, tp, am.CEILING_DBTP, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("film pass landed at %.2f LUFS / %.2f dBTP" % (lufs, tp))

    # --- hp_hz=None really skips the high-pass -----------------------------
    # It has to: the segments were already high-passed, and two 2nd-order
    # sections at the same corner are a 4th-order Butterworth, whose step
    # overshoot measurably *raises* the true peak.
    #
    # Probed with two tones, not one: mastering normalises loudness, so on a
    # pure low tone the gain simply compensates for the attenuation and the
    # absolute energy barely moves. What the filter actually changes is the
    # *balance*, so that is what gets measured.
    #
    # The probe tone is tied to the corner rather than hard-coded. It used to be
    # a literal 40 Hz, which was half of the old 90 Hz corner -- when the corner
    # moved to 40 Hz that literal sat *on* the corner, where a 2nd-order section
    # only attenuates 3 dB, and the 5x assertion would have failed while the
    # filter was working perfectly. Half the corner keeps the probe in the
    # stopband whatever the corner becomes.
    probe_hz = am.HIGHPASS_HZ / 2.0
    tt = np.arange(fs * 6) / float(fs)
    two = np.tile(0.05 * (np.sin(2.0 * np.pi * probe_hz * tt)
                          + np.sin(2.0 * np.pi * 1000.0 * tt)), (2, 1))
    hp_on = am.master(two, fs)
    hp_off = am.master(two, fs, hp_hz=None)
    ratio_on = _band_energy(hp_on, fs, probe_hz) / _band_energy(hp_on, fs, 1000.0)
    ratio_off = _band_energy(hp_off, fs, probe_hz) / _band_energy(hp_off, fs, 1000.0)
    ok = ratio_off > ratio_on * 5.0
    lines.append("   %.0f Hz kept (corner %.0f)    low/1k energy: hp on %.4f, "
                 "hp off %.4f (%.1fx)   %s"
                 % (probe_hz, am.HIGHPASS_HZ, ratio_on, ratio_off,
                    ratio_off / max(ratio_on, 1e-30), "ok" if ok else "FAIL"))
    if not ok:
        problems.append("hp_hz=None did not skip the high-pass (low/1k energy "
                        "%.4f vs %.4f)" % (ratio_off, ratio_on))

    # --- a segment that is nearly silence is capped, not lifted ------------
    # The gain bound has to leave room for real material (the render needed up to
    # +16.2 dB) while still refusing to amplify a noise floor into the join. A
    # signal ~36 dB below the target is the case that separates the two.
    hush = np.tile(0.0032 * np.sin(2.0 * np.pi * 440.0 * np.arange(fs * 8) / fs),
                   (2, 1))
    raw_hush = float(am.measure_lufs(hush, fs))
    lufs_hush = float(am.measure_lufs(am.master_segment(hush, fs), fs))
    ok = np.isfinite(lufs_hush) and lufs_hush < am.TARGET_LUFS - 3.0
    lines.append("   near-silence capped  %7.2f -> %7.2f LUFS (cap %.0f dB)   %s"
                 % (raw_hush, lufs_hush, am._MAX_TOTAL_GAIN_DB,
                    "ok" if ok else "FAIL"))
    if not ok:
        problems.append("near-silent segment was boosted to %.2f LUFS; the "
                        "%.0f dB bound is not holding"
                        % (lufs_hush, am._MAX_TOTAL_GAIN_DB))

    # --- pre_limit=False is what keeps the film's headroom -----------------
    # The film pass runs on pieces that are already mastered, so the assertion
    # is on the outcome, not on a difference: the pass must hand back the
    # headroom it was given. Synthetic material has a near-constant envelope,
    # and there the pre-limiter's time-varying gain cancels exactly against the
    # loop's uniform gain -- both variants land in the same place, so asserting a
    # gap here would be asserting something this probe cannot show. Real material
    # is what separates them, and that is what the render's own log is for.
    film_on = am.master(joined, fs, hp_hz=None, pre_limit=True, shape_tone=False)
    tp_on = 20.0 * np.log10(am.true_peak(film_on, fs) + 1e-12)
    ok = tp >= tp_joined - 0.1
    lines.append("   headroom kept        in %6.2f -> out %6.2f dBTP "
                 "(pre_limit on: %.2f)   %s" % (tp_joined, tp, tp_on,
                                                "ok" if ok else "FAIL"))
    if not ok:
        problems.append("the film pass lost headroom: %.2f dBTP in, %.2f dBTP out"
                        % (tp_joined, tp))

    return problems, lines


# ---------------------------------------------------------------------------
# 6. The save-time pass and the join guard agree
# ---------------------------------------------------------------------------

def check_save_guard(am):
    """``is_mastered`` is what lets one routine run at two different times.

    ``save_clip`` masters a segment as it is written, so a single segment is
    already at delivery level when it is played on its own -- which is how the
    work is actually reviewed, one segment at a time. ``_join_audio`` then runs
    the same pass over every piece, and this guard is the only thing stopping it
    from shaping an already-shaped signal a second time.

    Both directions fail silently. A guard that never fires re-masters the whole
    film piece by piece; a guard that always fires leaves a segment an older
    build wrote -- still raw, 7-8 LU down -- untouched in the finished film. The
    tests below pin both.
    """
    problems = []
    lines = []
    fs = 32000

    t = np.arange(fs * 10) / float(fs)
    raw = np.tile(0.02 * np.sin(2.0 * np.pi * 800.0 * t), (2, 1))
    done = am.master_segment(raw, fs)

    cases = [
        ("raw segment", raw, False),
        ("mastered segment", done, True),
        ("empty", np.zeros((2, 0)), False),
        ("too short to gate", np.zeros((2, 100)), False),
        ("silence", np.zeros((2, fs * 5)), False),
    ]
    for name, signal, want in cases:
        got = bool(am.is_mastered(signal, fs))
        ok = got is want
        lines.append("   %-20s -> %-5s (want %-5s)   %s"
                     % (name, got, want, "ok" if ok else "FAIL"))
        if not ok:
            problems.append("is_mastered(%s) returned %s, expected %s"
                            % (name, got, want))

    # A second pass must not move the level: that is the property the guard
    # exploits. If re-mastering did move the signal, "skip what is already
    # done" would be the wrong rule and the join would be leaving work undone.
    once = float(am.measure_lufs(done, fs))
    again = float(am.measure_lufs(am.master_segment(done, fs), fs))
    ok = abs(again - once) <= 0.2
    lines.append("   second pass is a no-op  %7.2f -> %7.2f LUFS   %s"
                 % (once, again, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("re-mastering a mastered segment moved it %.2f -> %.2f "
                        "LUFS; the join guard is skipping work that still had "
                        "something to do" % (once, again))

    # ... and the guard must stay *off* for a segment the gain cap could not
    # bring to target. It is not mastered, so the join is right to try again.
    hush = np.tile(0.0032 * np.sin(2.0 * np.pi * 440.0 * np.arange(fs * 8) / fs),
                   (2, 1))
    hush_done = am.master_segment(hush, fs)
    hush_lufs = float(am.measure_lufs(hush_done, fs))
    ok = not am.is_mastered(hush_done, fs)
    lines.append("   capped segment         %7.2f LUFS -> guarded off   %s"
                 % (hush_lufs, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("a gain-capped segment reports as mastered; the join "
                        "would skip a piece that never reached the target")

    return problems, lines


# ---------------------------------------------------------------------------
# 7. The documented escape hatch actually opens
# ---------------------------------------------------------------------------

def _load_with_env(value):
    """A *fresh* copy of audio_master, imported with the env var set."""
    previous = os.environ.get("MARQUEE_AUDIO_MASTER")
    if value is None:
        os.environ.pop("MARQUEE_AUDIO_MASTER", None)
    else:
        os.environ["MARQUEE_AUDIO_MASTER"] = value
    try:
        path = os.path.abspath(MODULE)
        spec = importlib.util.spec_from_file_location(
            "audio_master_env_%s" % value, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("MARQUEE_AUDIO_MASTER", None)
        else:
            os.environ["MARQUEE_AUDIO_MASTER"] = previous


def check_escape_hatch(am):
    """``MARQUEE_AUDIO_MASTER=0`` has to actually stop the mastering.

    The constant's comment promised this switch while no code read it -- the
    module only ever tested the literal ``True`` -- so for a while the
    documented way out of the feature did nothing at all. That is worth a test
    because of who reaches for it: someone who has already decided the mastering
    is in the way. Finding the switch inert at that moment is how a file stops
    being trusted. Both halves are checked -- the flag flips, and the signal
    comes back bit-identical.
    """
    problems = []
    lines = []
    fs = 32000
    t = np.arange(fs * 6) / float(fs)
    raw = np.tile(0.02 * np.sin(2.0 * np.pi * 800.0 * t), (2, 1))

    for value, want in ((None, True), ("1", True), ("0", False),
                        ("false", False), ("off", False)):
        module = _load_with_env(value)
        out = module.master(raw, fs)
        untouched = bool(np.array_equal(out, raw))
        ok = (module.ENABLED is want) and (untouched is (not want))
        lines.append("   %-24s ENABLED=%-5s untouched=%-5s   %s"
                     % ("MARQUEE_AUDIO_MASTER=%s" % value, module.ENABLED,
                        untouched, "ok" if ok else "FAIL"))
        if not ok:
            problems.append("MARQUEE_AUDIO_MASTER=%s gave ENABLED=%s, "
                            "untouched=%s (wanted ENABLED=%s)"
                            % (value, module.ENABLED, untouched, want))
    return problems, lines


# ---------------------------------------------------------------------------
# 8. The tone shaping does what it claims
# ---------------------------------------------------------------------------

def _gain_at(am, fs, f, seconds=4.0):
    """dB gain the shaping stage applies at ``f``, measured, not read off."""
    t = np.arange(int(fs * seconds)) / float(fs)
    x = np.tile(0.1 * np.sin(2.0 * np.pi * f * t), (2, 1))
    y = am.shape(x, fs)
    k = int(fs * 0.5)                     # skip the filter transient
    rms = lambda a: np.sqrt(np.mean(a[..., k:] ** 2))     # noqa: E731
    return 20.0 * np.log10(max(rms(y), 1e-12) / max(rms(x), 1e-12))


def check_shape(am):
    """Three gains, three assertions, plus the flag that turns them off.

    A new DSP stage is exactly the kind of thing that can be wired in and never
    actually run -- the shape constants could be zero, the call could sit behind
    a false branch, or the flag could be ignored. Measuring the response at the
    three frequencies the constants name catches all of that at once, and
    catches it before a render does.
    """
    problems = []
    lines = []
    fs = 32000

    # Measure each filter where it is *asymptotic*, not at its corner. A shelving
    # filter's ``gain_db`` is the gain in the flat part of the shelf; at the
    # corner it is exactly half. Asserting the full gain at the corner made a
    # correct filter look wrong (measured +0.94 dB against a nominal +2.5, and
    # -1.45 against -2.5, which is the shelf behaving exactly as designed).
    expected = [(am.SHAPE_LOW_HZ / 4.0, am.SHAPE_LOW_DB, "low shelf, 2 oct down"),
                (am.SHAPE_DIP_HZ, am.SHAPE_DIP_DB, "peaking, at centre"),
                (am.SHAPE_HIGH_HZ * 6.0, am.SHAPE_HIGH_DB, "high shelf, 2.6 oct up")]
    for f, want, what in expected:
        got = _gain_at(am, fs, f)
        ok = abs(got - want) <= 0.8
        lines.append("   %7.0f Hz  %+6.2f dB (want %+.1f)  %-22s %s"
                     % (f, got, want, what, "ok" if ok else "FAIL"))
        if not ok:
            problems.append("shaping gain at %.0f Hz is %+.2f dB, expected %+.1f"
                            % (f, got, want))

    # 1 kHz is the LTAS reference the film comparison is normalised against, so
    # it should move as little as possible -- but a 2 kHz high shelf cannot leave
    # it perfectly alone, hence a bound rather than an equality.
    got = _gain_at(am, fs, 1000.0)
    ok = abs(got) <= 2.0
    lines.append("   %7.0f Hz  %+6.2f dB (want |x| <= 2, it is the reference)   %s"
                 % (1000.0, got, "ok" if ok else "FAIL"))
    if not ok:
        problems.append("shaping moves the 1 kHz reference by %+.2f dB" % got)

    # The flag has to actually skip it -- the film pass relies on that to avoid
    # doubling a 2 dB shelf into 4 dB and a 3.5 dB dip into 7 dB.
    #
    # The probe sits on the dip's own centre, read from the constant. Hard-coding
    # 630 Hz here (the first value the dip happened to have) made a working stage
    # look broken as soon as the centre moved to 700 Hz -- the same mistake the
    # high-pass probe made, in the same file, one function earlier.
    dip = am.SHAPE_DIP_HZ
    t = np.arange(fs * 6) / float(fs)
    x = np.tile(0.05 * (np.sin(2.0 * np.pi * dip * t)
                        + np.sin(2.0 * np.pi * 1000.0 * t)), (2, 1))
    on = am.master(x, fs)
    off = am.master(x, fs, shape_tone=False)
    r_on = _band_energy(on, fs, dip) / _band_energy(on, fs, 1000.0)
    r_off = _band_energy(off, fs, dip) / _band_energy(off, fs, 1000.0)
    ok = r_off > r_on * 1.5
    lines.append("   shape_tone=False skips it   %.0f/1k energy: on %.4f, "
                 "off %.4f (%.1fx)   %s"
                 % (dip, r_on, r_off, r_off / max(r_on, 1e-30),
                    "ok" if ok else "FAIL"))
    if not ok:
        problems.append("shape_tone=False did not skip the shaping (%.0f/1k "
                        "energy %.4f vs %.4f)" % (dip, r_off, r_on))

    return problems, lines


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audio", metavar="PATH",
                    help="a real render's audio (wav) to cross-check against ffmpeg")
    ap.add_argument("--no-ffmpeg", action="store_true",
                    help="skip the ffmpeg cross-check")
    args = ap.parse_args()

    am = load_audio_master()
    print("module: %s" % os.path.abspath(MODULE))
    print("target %.1f LUFS, ceiling %.1f dBTP, high-pass %.0f Hz, "
          "limiter %.2f / %.0f ms / %.0f ms\n"
          % (am.TARGET_LUFS, am.CEILING_DBTP, am.HIGHPASS_HZ,
             am.LIMITER_CEIL, am.LIMITER_ATTACK_MS, am.LIMITER_RELEASE_MS))

    problems = []

    print("1. K-weighting matches ITU-R BS.1770-4 Table 1 at 48 kHz")
    p, lines = check_k_weighting(am)
    print("\n".join(lines))
    problems += p

    print("\n2. K-weighting is stable at the rates we use")
    p, lines = check_stability(am)
    print("\n".join(lines))
    problems += p

    print("\n3. Integrated loudness agrees with ffmpeg ebur128")
    ffmpeg = None if args.no_ffmpeg else shutil.which("ffmpeg")
    if ffmpeg is None:
        print("   skipped (%s)" % ("--no-ffmpeg" if args.no_ffmpeg else "ffmpeg not on PATH"))
    else:
        p, lines = check_against_ffmpeg(am, ffmpeg, args.audio)
        print("\n".join(lines))
        problems += p

    print("\n4. master() hits the target, holds the ceiling, never raises")
    p, lines = check_master(am)
    print("\n".join(lines))
    problems += p

    print("\n5. the join path: batch, per-segment match, and both shortcuts")
    p, lines = check_join_path(am)
    print("\n".join(lines))
    problems += p

    print("\n6. the save-time pass and the join guard agree")
    p, lines = check_save_guard(am)
    print("\n".join(lines))
    problems += p

    print("\n7. the documented escape hatch opens")
    p, lines = check_escape_hatch(am)
    print("\n".join(lines))
    problems += p

    print("\n8. the tone shaping does what it claims")
    p, lines = check_shape(am)
    print("\n".join(lines))
    problems += p

    print()
    if problems:
        print("%d problem(s):" % len(problems))
        for line in problems:
            print("   ! %s" % line)
        return 1
    print("all clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
