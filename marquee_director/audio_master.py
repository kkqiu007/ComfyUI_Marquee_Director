"""Dialogue mastering for H3 segments and the joined soundtrack.

Where the level is lost -- and where it is not
----------------------------------------------
It is the model, not the node chain. Measured directly: a raw segment written by
this build (``曹贼的性价比_v18_en_v12/seg_01.mp4``) decodes at

    integrated loudness   -21.0 LUFS     LRA 5.9 LU     true peak +0.3 dBFS
    the single loudest instant    t = 0.306-0.330 s    (0.19% of all samples)
    spectrum of that instant      81% of its energy below 150 Hz
    spectrum of the dialogue      73% of its energy at 600-1200 Hz
    dialogue RMS / peak RMS       -20.6 dB
    same take's float-PCM sidecar -20.6 LUFS   (the copy the join reads)

and ``engine.py`` hands the VAE decode straight to ``save_clip``: the only
amplitude operation anywhere in ``marquee_director`` outside this file is none.
So the model emits a **loud, transient low-frequency component** that sits well
above the dialogue, and the segment is not "quiet" on a peak meter -- the
headroom is simply spent on it before the first line of dialogue.

That low end is **content, not an artifact.** This module used to call it a
"front-loaded sub-bass thump" and high-passed it away at 90 Hz. Both halves of
that were wrong, and the second one was audible:

* *Front-loaded* was wrong. Measured on a raw segment, the 20-150 Hz band runs
  15 dB above the 600 Hz-3 kHz band at t ~ 0.3 s, t ~ 3 s **and** t ~ 5 s, then
  falls below -60 dB once the music ducks out from under the dialogue. It is
  transient throughout, not one thump at the start.
* *Artifact* was wrong. The workflow's own prompt asks for it -- the
  ``overall_soundscape`` line requests "the muffled thud of a boot on the door
  panel and the echo of the panel slamming into the wall", and
  ``non_diegetic_music`` requests "a single low, sustained Peking-opera
  percussion and strings bed". Deleting it deletes the sound design, and the
  segment comes back sounding weightless.

Measured cost of the old 90 Hz corner, on that same raw segment: the chain took
the share of energy below 150 Hz from **43.5% to 10.5%**, and the high-pass
alone accounted for 2.1 dB of the 6.2 dB lost. Dropping the corner to 40 Hz
gives 2.1 dB of it back with **no** change to loudness, true peak or crest
factor -- the ceiling limiter simply does the work instead. See
``HIGHPASS_HZ``.

What is *not* fixed here: the voice itself is thin at the source. Measured with
the low band empty (t > 7 s, music ducked), the model's dialogue holds 42% of
its energy at 150-600 Hz where a real film's dialogue holds 67%, and 2.8x the
reference's share at 1.2-6 kHz. Its 80-150 Hz band is 0.03% -- the vocoder
produces essentially no voice energy below 150 Hz, so there is nothing for a
high-pass to preserve and nothing for a gain stage to restore. Fixing that
means EQ, which is a design decision and not a parameter tweak.

That is why no amount of gain fixes the level. Turning the whole segment up hits
the ceiling on the low end; ComfyUI's own ``vae_decode_audio`` does not help: its
normaliser is a *safety* attenuator (``std[std < 1.0] = 1.0``, so it can only
ever divide down, and only when ``5*std > 1``), and H3's decode has std
0.019-0.052, so it never fires at all.

The delivery benchmark is a real film release (47 min, HEVC,
three stereo tracks). Its three audio tracks measure:

    a:0  eac3  "DDP 2.0"         -22.5 LUFS   LRA  8.6 LU   peak  -4.8 dBFS
    a:1  aac   "AAC 2.0 HIFI"    -12.6 LUFS   LRA 15.0 LU   peak  +0.6 dBFS
    a:2  aac   "AAC 2.0"         -14.6 LUFS   LRA 15.1 LU   peak  -0.5 dBFS

(The two loud mixes are a:1 and a:2; an earlier note in this file attributed
-12.6 to a:0, which is the quiet DDP track. The numbers were right, the labels
were swapped.) The loud mixes sit ~1.4 LU above the -14 platform figure and
just under 0 dBFS, which is what the constants below encode: a loudness target
a little above -14, and a lossy master near 0 dBFS true peak being normal rather
than something to avoid.

The chain, in order
-------------------
1. **High-pass at 40 Hz.** Removes DC and sub-audio rumble only. It used to be
   90 Hz, on the mistaken belief that the low end was an artifact -- see above.
   Second order on purpose: a 4th-order Butterworth has enough step overshoot to
   push the true peak *up* (+1.1 dBTP measured), which is the opposite of
   helpful.
2. **Spectral shaping** -- a low shelf, a dip and a high shelf, all sized from a
   measured gap against a real film mix. This stage did not exist until
   2026-09-23: the chain was spectrally *neutral* (before/after differ by under
   2 dB in every 1/3-octave band), so the tone was whatever the model produced,
   and the model produces a voice that is thin and boxy. See ``shape``.
3. **Pre-limiter at -9.1 dBFS.** Tames the low-frequency transients so the gain
   in step 4 has somewhere to go, without applying the old -14 dBFS threshold
   that audibly pumped the dialogue. The delivery ceiling limiter below still
   handles the final true-peak constraint. This pre-stage is deliberately mild:
   it is a headroom aid, not a second audible mastering stage.
4. **Loudness normalisation** to ``TARGET_LUFS`` by *linear* gain, computed from
   an ITU-R BS.1770-4 integrated measurement. Linear, not the gated dynamic mode,
   because time-varying gain pumps on continuous dialogue.
4. **Gain/ceiling iteration.** Measure, apply the linear gain the target asks
   for, then let a limiter at ``CEILING_DBTP`` take back whatever that gain
   pushed past the ceiling, and repeat. This is the step that makes the target
   reachable at all. The material arrives with a ~21 dB crest factor and the
   ceiling only permits 12, so the difference has to come off the peaks
   somewhere; the limiter takes it from the peaks alone, whereas scaling the
   whole track down to fit would have cost 2.19 LU across the entire render.

It runs twice, at two different times
-------------------------------------
``master_segment`` is applied when a **segment is saved**, so that a single
segment is listenable on its own -- which is how the work actually gets
reviewed, one segment at a time, and it used to sound thin until the whole film
was joined. The join then measures each piece and masters only the ones that are
still raw (``is_mastered``), so a segment cached by an older build is repaired
and a current one is not reshaped twice.

Everything is measured with the same K-weighting the broadcast standard defines,
so the numbers here line up with ``ffmpeg -af ebur128`` -- see
``tools/check_audio_master.py`` for the cross-check.

``CEILING_DBTP`` is a **true** peak. The limiter is driven by a 4x-oversampled
envelope (BS.1770-4 Annex 2's factor), so the ceiling holds in the decoded file
and not merely on the sample grid. Driving it from sample peaks instead leaves
the delivered track about 1 dB above the number asked for -- that is how a
"-2 dBTP" master decodes at -1.0 dBTP with no error anywhere to warn you.
"""

import logging
import os

import numpy as np
from scipy.signal import lfilter, resample_poly

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables. Module constants rather than node widgets on purpose: adding an
# input changes every saved workflow's widget order, and a soundtrack that is
# simply *correct* by default does not need a knob in the UI.
# ---------------------------------------------------------------------------

#: Integrated loudness target.
#:
#: -14 LUFS is what YouTube and most short-video platforms normalise playback
#: to, and that was the value here. It was raised to -13.0 after measuring a
#: real film release (47 min, HEVC + 3 stereo tracks) as the
#: delivery benchmark:
#:
#:     a:0 (eac3, DDP 2.0)   -22.5 LUFS   LRA  8.6 LU   true peak -4.8 dBFS
#:     a:1 (aac, HIFI)       -12.6 LUFS   LRA 15.0 LU   true peak +0.6 dBFS
#:     a:2 (aac)             -14.6 LUFS   LRA 15.1 LU   true peak -0.5 dBFS
#:
#: So a professionally finished film sits ~1.4 LU above the platform target, and
#: -14 was the one place in this chain where the output was measurably quieter
#: than the reference. -13.0 takes most of that back while staying a little
#: under the reference, which leaves room for a platform to turn it *down*
#: rather than up. The delivery spec is a floor for loudness normalisation, not
#: a ceiling to sit exactly on.
TARGET_LUFS = -13.0

#: True-peak ceiling.
#:
#: The reference film decodes at **+0.6 dBFS** true peak -- above full scale,
#: which is normal for a lossy consumer master and is not clipping in practice.
#: -2.0 dBTP was therefore 2.6 dB of headroom given away for nothing, and the
#: comment that used to sit here ("raising this to -1.0 costs nothing in
#: loudness on this material") was wrong: with the sub-bass onset now taken off
#: by the high-pass, the ceiling is exactly what bounds the loudness, so every
#: dB of ceiling is a dB of loudness.
#:
#: -1.0 dBTP, with ComfyUI's AAC adding ~+1.1 dB on decode (measured at 128k:
#: -2.0 in, -1.4 out; at 192k: -1.1), delivers around 0 dBFS -- the same place
#: the reference sits. ``add_stream("aac", ...)`` is called with no bitrate,
#: which is libavcodec's default of 128 kbps at 32 kHz stereo.
CEILING_DBTP = -1.0

#: High-pass corner. Removes DC and sub-audio rumble, and nothing else.
#:
#: It was 90 Hz, justified by calling the low end a "front-loaded sub-bass thump"
#: -- a model artifact. That was wrong twice over: the 20-150 Hz band runs 15 dB
#: above the 600 Hz-3 kHz band at t ~ 0.3, 3 **and** 5 s (transient throughout,
#: not one thump), and the workflow's prompt *asks* for it (a boot on the door
#: panel, the panel slamming into the wall, a low Peking-opera percussion bed).
#: So the old corner was deleting sound design.
#:
#: Measured on a raw segment, the whole chain takes the share of energy below
#: 150 Hz from 43.5% to 10.5%, and the high-pass accounts for 2.1 dB of that
#: 6.2 dB. At 40 Hz the chain gives 2.1 dB of it back with **no** change to
#: loudness (-13.18 vs -13.17 LUFS), true peak (-1.00 either way) or crest
#: factor (16.03 vs 16.21 dB) -- the ceiling limiter absorbs the difference.
#:
#: 40 Hz sits well below the lowest speech fundamental (male ~85 Hz) and below
#: the body of a door thud, so the sound design survives, while DC and the
#: sub-audio rumble the model also produces do not.
HIGHPASS_HZ = 40.0

#: Spectral shaping, derived from a measured gap against a real film mix.
#: See ``shape`` for the three measurements behind each number. Set every gain to
#: 0.0 to disable the stage; ``SHAPE_ENABLED`` turns it off in one place.
SHAPE_ENABLED = True
SHAPE_LOW_HZ = 90.0        # low shelf corner -- recovers what the film has
#: Kept at 2.0, not more. The 20-80 Hz gap against the film is large (-11 dB) but
#: it is a *content* gap -- the film's is music, ours is intermittent hits -- and
#: pushing the shelf harder measured an overshoot in the neighbouring 80-160 Hz
#: band (+2.3 dB at +3.0 dB of shelf). EQ cannot add energy that is not there;
#: see the module docstring on the generation-side cause.
SHAPE_LOW_DB = 2.0
SHAPE_DIP_HZ = 700.0       # the boxy region the model puts into its dialogue
SHAPE_DIP_DB = -3.5
#: Narrow on purpose. A wide dip drags the *neighbouring* bands down with it, so
#: the boxiness is only partly removed -- a dip has to be narrower than the thing
#: it is cutting. Measured on the real segment: -4.0 dB at Q 1.4 left the band
#: essentially as boxy as before, -3.5 dB at Q 1.8 does not.
SHAPE_DIP_Q = 1.6
SHAPE_HIGH_HZ = 2000.0     # presence/air shelf -- pulls back the excess
SHAPE_HIGH_DB = -3.5

#: Pre-limiter threshold, linear. 0.35 == -9.1 dBFS.
#:
#: Softened from 0.20 (-14 dBFS) on 2026-09-23. On the four-segment render the old
#: value engaged on 17.4% of samples with up to -12.7 dB of reduction -- the note
#: below already called it "not nearly transparent" -- which pumps the dialogue and
#: reads as an artificial / "weird" voice timbre. 0.35 still tames the
#: low-frequency transients enough for the gain loop to reach TARGET_LUFS (verified
#: on a synthetic raw signal), while leaving far more of the voice's natural
#: dynamics intact. The ceiling limiter at CEILING_DBTP does the final peak work
#: either way, so this only trades a little crest headroom for a less processed
#: sound.
LIMITER_CEIL = 0.35

#: Limiter envelope times.
LIMITER_ATTACK_MS = 5.0
LIMITER_RELEASE_MS = 50.0

#: Gain/ceiling iteration. See ``master`` for why one pass is not enough.
_MAX_GAIN_PASSES = 4
_GAIN_TOLERANCE_LU = 0.2

#: How close to the target a piece has to be before it counts as already
#: mastered and is left alone. See ``is_mastered``.
#:
#: This exists because the segment pass moved to *save* time: ``seg_01.mp4`` is
#: now mastered when it is written, so that a single segment is listenable on
#: its own instead of only sounding right after the join. The join therefore has
#: to tell "already done" from "raw", and it has to be able to do so for a
#: segment that was rendered by an older build and cached on disk.
#:
#: 1.0 LU is far wider than the 0.2 LU the gain loop converges to and far
#: narrower than the 7-9 LU a raw segment arrives at, so the two cases cannot be
#: confused. Re-running the pass on finished material is not free: the high-pass
#: is a 2nd-order section (a second one makes 4th-order, whose step overshoot
#: pushes the true peak *up*), and the pre-limiter is not idempotent.
MASTERED_TOLERANCE_LU = 1.0

#: Largest total correction either pass will apply, in dB. It exists so a segment
#: that is almost entirely silence cannot ask for a +60 dB boost that would drag
#: its noise floor up into the join -- not to trim legitimate material.
#:
#: It was 18.0, and that was too tight to be safe: the four-segment render needed
#: +8.4 to +16.2 dB, leaving under 2 dB of headroom, so any take a little quieter
#: than the quietest one seen would be capped and shipped *below* the rest --
#: which is the exact "the sound is still weak" symptom this module exists to
#: remove. 30 dB covers segments down to about -44 LUFS, comfortably past
#: anything a real render produces, while still refusing to amplify near-silence.
#:
#: Note that this is the budget for the *whole* correction, and the shaping stage
#: spends about 3 dB of it (it removes energy, so the loop must add more back).
#: Measured on the 2026-09-22 render: a segment needing +14.7 dB needs +17.9 with
#: shaping on. Still 12 dB of margin, but the two knobs are not independent.
_MAX_TOTAL_GAIN_DB = 30.0

#: Set MARQUEE_AUDIO_MASTER=0 to disable mastering entirely (escape hatch if a
#: future model already outputs correctly mastered audio, or if a take needs to
#: be heard exactly as the vocoder produced it).
#:
#: Read from the environment at import time. The comment here used to promise
#: this switch while no code read it -- the module only ever tested the constant
#: below -- so the documented escape hatch did nothing. An escape hatch that
#: does not open is worse than none, because it is the thing you reach for when
#: you have already decided the feature is in the way.
ENABLED = os.environ.get("MARQUEE_AUDIO_MASTER", "1").strip().lower() not in (
    "0", "false", "no", "off")

# ---------------------------------------------------------------------------
# Biquads
# ---------------------------------------------------------------------------

# ITU-R BS.1770-4 K-weighting design parameters. The standard tabulates
# coefficients at 48 kHz only; these are the underlying analog prototypes, from
# which the biquads are rebuilt for whatever rate the model actually runs at
# (32 kHz here). check_audio_master.py asserts the 48 kHz rebuild reproduces the
# tabulated values to within float noise.
_SHELF_F0, _SHELF_GAIN_DB, _SHELF_Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
_HPF_F0, _HPF_Q = 38.13547087602444, 0.5003270373238773

#: The shelf's transition-sharpness exponent, fixed by the standard.
_VB_EXP = 0.4996667741545416


def _k_shelf(fs):
    """Stage 1 of BS.1770-4 K-weighting, the high-shelf pre-filter.

    **Not** the RBJ "high shelf" cookbook formula. Both are a shelf, both use a
    4 dB lift near 1.7 kHz, and they disagree in the third decimal of every
    coefficient -- the RBJ version reproduces the standard's 48 kHz table only
    to ~5e-2, which is audible as a systematic error in the measurement. The
    standard's own design is a bilinear transform of an analog prototype with
    ``K = tan(pi*f0/fs)`` and a transition sharpness of ``Vh**0.4996667741545416``
    rather than ``alpha = sin(w0)/(2Q)``. This is Brecht De Man's reconstruction
    (github.com/BrechtDeMan/loudness.py), the same one ``pyloudnorm`` ships as
    its ``DeMan`` filter class; use it, not the cookbook, whenever the number
    has to agree with ``ffmpeg -af ebur128``.
    """
    K = np.tan(np.pi * _SHELF_F0 / fs)
    Vh = 10.0 ** (_SHELF_GAIN_DB / 20.0)
    Vb = Vh ** _VB_EXP
    a0 = 1.0 + K / _SHELF_Q + K * K
    b = np.array([(Vh + Vb * K / _SHELF_Q + K * K) / a0,
                  2.0 * (K * K - Vh) / a0,
                  (Vh - Vb * K / _SHELF_Q + K * K) / a0])
    a = np.array([1.0,
                  2.0 * (K * K - 1.0) / a0,
                  (1.0 - K / _SHELF_Q + K * K) / a0])
    return b, a


def _k_high_pass(fs):
    """Stage 2 of BS.1770-4 K-weighting, the RLB high-pass.

    Numerator is exactly ``[1, -2, 1]`` as tabulated -- the standard leaves it
    unnormalised, so this stage carries a ~+0.04 dB passband lift. That is part
    of the standard, not a bug to correct.
    """
    K = np.tan(np.pi * _HPF_F0 / fs)
    d = 1.0 + K / _HPF_Q + K * K
    return (np.array([1.0, -2.0, 1.0]),
            np.array([1.0,
                      2.0 * (K * K - 1.0) / d,
                      (1.0 - K / _HPF_Q + K * K) / d]))


def _rbj_high_pass(fs, f0, q):
    """RBJ cookbook 2nd-order high-pass, unity gain in the passband.

    Used for the *dialogue* high-pass, where the job is tone shaping rather
    than standards compliance, and exact unity passband gain is what you want.
    Deliberately kept separate from ``_k_high_pass`` so nobody swaps one for
    the other: the K-weighting one is a measurement filter and must match the
    tabulated coefficients, this one must not.
    """
    w0 = 2.0 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b = np.array([(1 + cw) / 2.0, -(1 + cw), (1 + cw) / 2.0])
    a_ = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a_[0], a_ / a_[0]


def _k_weighting(fs):
    """The two cascaded biquads of BS.1770-4, designed for ``fs``."""
    return (_k_shelf(fs), _k_high_pass(fs))


def _rbj_peaking(fs, f0, gain_db, q):
    """RBJ cookbook peaking EQ. Used to take a resonance out, not to voice."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a_ = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    return b / a_[0], a_ / a_[0]


def _rbj_shelf(fs, f0, gain_db, high):
    """RBJ cookbook shelving filter, slope S = 1, unity in the flat region.

    A shelf, not a peaking band, is the right tool for the two ends of a tilt:
    the correction is a *slope* over several octaves, and a wide peaking band
    would leave the extremes untouched while overshooting in the middle.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw = np.cos(w0)
    alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)          # S = 1
    tsa = 2.0 * np.sqrt(A) * alpha
    if high:
        b = np.array([A * ((A + 1) + (A - 1) * cw + tsa),
                      -2 * A * ((A - 1) + (A + 1) * cw),
                      A * ((A + 1) + (A - 1) * cw - tsa)])
        a_ = np.array([(A + 1) - (A - 1) * cw + tsa,
                       2 * ((A - 1) - (A + 1) * cw),
                       (A + 1) - (A - 1) * cw - tsa])
    else:
        b = np.array([A * ((A + 1) - (A - 1) * cw + tsa),
                      2 * A * ((A - 1) - (A + 1) * cw),
                      A * ((A + 1) - (A - 1) * cw - tsa)])
        a_ = np.array([(A + 1) + (A - 1) * cw + tsa,
                       -2 * ((A - 1) + (A + 1) * cw),
                       (A + 1) + (A - 1) * cw - tsa])
    return b / a_[0], a_ / a_[0]


def shape(x, fs, low_db=SHAPE_LOW_DB, dip_db=SHAPE_DIP_DB,
          high_db=SHAPE_HIGH_DB):
    """Tilt the tone toward a real film mix, from a measured gap.

    The chain used to be spectrally **neutral** -- measured, the difference
    between a segment before and after it is within +-2 dB in every 1/3-octave
    band -- so any tonal problem is either in the source or has to be added
    here. It is in the source, and three measurements say so:

    1. Compared with a real film mix's long-term average spectrum, normalised at
       1 kHz, our segment is **-5.7 dB below 80 Hz**, level from 160-600 Hz, and
       **+3.0 dB from 600 Hz to 3 kHz**.
    2. It carries a **narrow peak at 630 Hz that stands 14 dB above its own
       315-1250 Hz trend**. A film's dialogue sits at -1 dB there, i.e. flat. A
       peak like that is a resonance: it is what makes a voice read as hollow or
       "boxy" rather than full, and it is present in the model's raw output
       (+14.1 dB) and after the chain (+14.3 dB) alike.
    3. The model puts essentially no energy below 150 Hz into its dialogue at
       all -- 0.03% in the 80-150 Hz band, against 2.58% for real film dialogue
       -- so the lift in (1) is recovering the *sound design* the prompt asked
       for (a boot on a door, the panel slamming, a percussion bed), not voice.

    Three moves, all conservative, and all on the measured gap:

        low shelf  +2.0 dB @ 90 Hz     -- recover some of the low end the film has
        peaking    -3.5 dB @ 700 Hz    -- take the boxiness out
        high shelf -3.5 dB @ 2 kHz     -- pull back the presence excess

    Measured on the real segment, against the film mix: the sub gap goes -11.3 ->
    -8.0 dB, presence +3.1 -> +2.6, air +2.1 -> +0.3, body -1.3 -> -0.4, with no
    band changing sign (an overshoot would just be the same error mirrored).

    **This stage is a polish, not the cure.** The largest single difference
    against the film is the low end, and it is a *content* difference, not a
    tonal one: during dialogue this chain's output carries almost nothing below
    150 Hz because the generation side was told to keep the music silent under
    every line and the room tone it was given is all high-frequency (candles,
    insects, a draught). EQ cannot add energy that is not in the signal. Fixing
    that means changing what the model is asked to generate -- see
    ``prompt_pack.MUSIC_BLOCK_TEMPLATE`` and the ``overall_soundscape`` spec.

    Runs *before* the gain/ceiling loop, so the loudness normalisation accounts
    for it -- applying EQ after mastering instead leaves the file about 1 dB
    hot, which is enough to make an A/B meaningless.
    """
    if not ENABLED:
        return x
    y = x
    if abs(low_db) > 1e-9:
        y = _apply(y, _rbj_shelf(fs, SHAPE_LOW_HZ, low_db, high=False))
    if abs(dip_db) > 1e-9:
        y = _apply(y, _rbj_peaking(fs, SHAPE_DIP_HZ, dip_db, SHAPE_DIP_Q))
    if abs(high_db) > 1e-9:
        y = _apply(y, _rbj_shelf(fs, SHAPE_HIGH_HZ, high_db, high=True))
    return y


def _apply(x, coeffs):
    """Run a biquad along the last axis of a (channels, samples) array."""
    b, a = coeffs
    return lfilter(b, a, x, axis=-1)


# ---------------------------------------------------------------------------
# True peak
# ---------------------------------------------------------------------------

#: Oversampling factor for true-peak work. BS.1770-4 Annex 2 specifies 4x.
_OVERSAMPLE = 4


def true_peak(x, fs, factor=_OVERSAMPLE):
    """True peak, linear, of a (channels, samples) array.

    The sample peak is not the peak of the signal. Between two samples the
    reconstructed waveform overshoots, and a lossy encoder reproduces that
    overshoot faithfully -- so a track whose samples sit at -2.00 dBFS can
    decode at -1.0 dBTP and clip a downstream converter. Measured on the
    four-segment render, that gap was exactly 1.0 dB. Any ceiling that calls
    itself dBTP has to be evaluated here, not on ``abs(x).max()``.

    The signal is oversampled **before** the absolute value is taken. Doing it
    the other way round -- rectify, then resample -- silently loses the peak:
    ``abs`` of a cosine is a rectified cosine at twice the frequency, and
    interpolating that is not interpolating the original. It reads about 20%
    low on a fs/4 cosine, which is exactly what the detector test in
    ``tools/check_audio_master.py`` is built to catch.

    Accuracy, so nobody "fixes" this later: on a fs/4 cosine whose samples sit
    at +-0.7071 and whose ideal true peak is 1.000, this returns 1.092 and
    ``ffmpeg -af ebur128`` returns 1.109, while Fourier-method resampling gives
    exactly 1.000. FIR-based 4x oversampling -- the method BS.1770-4 Annex 2
    specifies, and what every commercial true-peak meter uses -- over-reads
    intersample peaks by up to ~0.8 dB on such signals. That is a property of
    the method rather than of this code, it is the number the delivery spec is
    checked against, and it errs toward a lower ceiling rather than a clipped
    file. Do not swap in an FFT resampler to chase the ideal: it would make
    this disagree with ffmpeg and with every other meter.
    """
    if x.shape[-1] == 0:
        return 0.0
    if factor <= 1:
        return float(np.abs(x).max())
    return float(np.abs(resample_poly(x, factor, 1, axis=-1)).max())


def _true_peak_envelope(x, fs, factor=_OVERSAMPLE):
    """Per-sample true-peak envelope, for driving the limiter.

    Same oversampling as ``true_peak``, reduced back to the original length by
    taking the max over each group of ``factor`` interpolated samples, so it can
    stand in for ``abs(x).max(axis=0)`` inside the limiter. Falls back to the
    sample envelope if anything about the resampling is off -- a slightly loose
    ceiling beats a failed render.
    """
    mono = np.abs(x).max(axis=0)
    if factor <= 1 or mono.shape[0] < factor * 4:
        return mono
    try:
        # Oversample first, *then* rectify -- see true_peak.
        up = np.abs(resample_poly(x, factor, 1, axis=-1)).max(axis=0)
    except Exception:
        return mono
    usable = (up.shape[0] // factor) * factor
    grouped = up[:usable].reshape(-1, factor).max(axis=1)
    n = mono.shape[0]
    if grouped.shape[0] >= n:
        return grouped[:n]
    return np.concatenate([grouped, np.full(n - grouped.shape[0], grouped[-1])])


# ---------------------------------------------------------------------------
# Loudness (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------

_BLOCK_S = 0.400      # gating block
_BLOCK_STEP_S = 0.100  # 75% overlap
_ABS_GATE = -70.0     # LKFS
_REL_GATE_LU = 10.0


def measure_lufs(waveform, fs):
    """Integrated loudness in LUFS for a (channels, samples) array.

    Gated exactly as the standard specifies: 400 ms blocks at 100 ms hop, an
    absolute -70 LKFS gate, then a relative gate 10 LU below the ungated mean of
    the surviving blocks. Gating is what makes the number insensitive to the
    silence between lines -- without it a sparse dialogue track measures far
    quieter than it sounds.
    """
    x = np.atleast_2d(np.asarray(waveform, dtype=np.float64))
    if x.shape[-1] < int(_BLOCK_S * fs):
        return float("nan")

    for coeffs in _k_weighting(fs):
        x = _apply(x, coeffs)

    block = int(round(_BLOCK_S * fs))
    step = int(round(_BLOCK_STEP_S * fs))
    n = x.shape[-1]
    starts = range(0, n - block + 1, step)
    # z_j = mean square summed over channels (all channel weights are 1.0)
    z = np.array([float(np.mean(np.sum(x[:, s:s + block] ** 2, axis=0)))
                  for s in starts])
    if z.size == 0:
        return float("nan")

    loudness = -0.691 + 10.0 * np.log10(np.maximum(z, 1e-12))

    keep = loudness > _ABS_GATE
    if not keep.any():
        return float("nan")
    rel = (-0.691 + 10.0 * np.log10(np.mean(z[keep]))) - _REL_GATE_LU
    keep &= loudness > rel
    if not keep.any():
        return float("nan")
    return float(-0.691 + 10.0 * np.log10(np.mean(z[keep])))


def is_mastered(waveform, sample_rate, target_lufs=TARGET_LUFS,
                tolerance=MASTERED_TOLERANCE_LU):
    """True when ``waveform`` already sits at the target, within ``tolerance``.

    Used by the join to tell a segment that was mastered when it was saved from
    one that is still raw -- including a segment an older build left on disk.
    A signal that cannot be measured counts as *not* mastered, so the safe
    direction is to master it.
    """
    current = measure_lufs(waveform, sample_rate)
    if not np.isfinite(current):
        return False
    return bool(abs(current - target_lufs) <= tolerance)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def highpass(x, fs, f0=HIGHPASS_HZ):
    return _apply(x, _rbj_high_pass(fs, f0, 0.7071067811865476))


def limiter(x, fs, ceil=LIMITER_CEIL,
            attack_ms=LIMITER_ATTACK_MS, release_ms=LIMITER_RELEASE_MS,
            oversample=_OVERSAMPLE):
    """True-peak limiter with lookahead, driven by a smoothed peak envelope.

    Offline, so the envelope is a *centred* running maximum over the attack
    window -- that is real lookahead, and it is what keeps the gain curve from
    overshooting on a sharp transient. A one-pole release then holds the
    reduction down between peaks so the gain does not chatter.

    The envelope is built from the *oversampled* signal, so ``ceil`` is a true
    peak and not a sample peak. With a sample-peak envelope the output decodes
    about 1 dB above the ceiling; see ``true_peak``.
    """
    if x.shape[-1] == 0:
        return x
    env = _true_peak_envelope(x, fs, oversample) if oversample > 1 \
        else np.abs(x).max(axis=0)

    win = max(1, int(round(attack_ms * 1e-3 * fs)))
    if win > 1:
        # Odd window, so that padding by win//2 on *both* sides leaves exactly
        # ``n`` windows and each one is centred on its sample. With an even
        # window the centred padding is asymmetric (win-1 must be split as
        # win//2-1 and win//2) and forgetting that yields n+1 gains, which then
        # fails to broadcast against the signal.
        if win % 2 == 0:
            win += 1
        # Centred running max == lookahead. np.maximum.accumulate twice would be
        # cheaper but asymmetric; this is a 41 s track, not a realtime path.
        pad = win // 2
        padded = np.pad(env, pad, mode="edge")
        strides = np.lib.stride_tricks.sliding_window_view(padded, win)
        env = strides.max(axis=-1)

    target = np.minimum(1.0, ceil / np.maximum(env, 1e-12))

    # one-pole release: rise instantly, fall with tau = release
    tau = max(1.0, release_ms * 1e-3 * fs)
    alpha = float(np.exp(-1.0 / tau))
    out = np.empty_like(target)
    prev = 1.0
    # np.frompyfunc-free running min-plus-one-pole; vectorised via lfilter would
    # need the min() first, so keep it explicit and readable.
    for i, t in enumerate(target):
        prev = t if t < prev else t + (prev - t) * alpha
        out[i] = prev
    return x * out


def master(waveform, sample_rate, target_lufs=TARGET_LUFS,
           ceiling_dbtp=CEILING_DBTP, hp_hz=HIGHPASS_HZ,
           limiter_ceil=LIMITER_CEIL, pre_limit=True,
           max_gain_db=_MAX_TOTAL_GAIN_DB, shape_tone=True):
    """Return the waveform mastered to ``target_lufs``.

    ``waveform`` is a (channels, samples) float array; ``sample_rate`` an int.
    A (batch, channels, samples) array is accepted too and is mastered per item --
    see the batch note below, it is the shape ``_join_audio`` actually passes.
    Shape is preserved. Never raises on odd input -- a soundtrack that cannot be
    measured is returned untouched, because losing a finished render to an
    audio polish step would be a bad trade.

    Three flags exist so the same routine can serve both passes of the join:

    ``hp_hz=None``
        Skip the high-pass. Pass this on the *join* when the pieces were already
        high-passed by the per-segment pass. Doing it twice is not harmless: two
        2nd-order sections at the same corner are a 4th-order high-pass, and the
        step overshoot of a 4th-order Butterworth was measured pushing the true
        peak *up* to +1.14 dBTP -- the opposite of the point.

    ``shape_tone=False``
        Skip the spectral shaping. Same reason as the high-pass: the per-segment
        pass already applied it, and applying a 2.5 dB low shelf plus a 3.5 dB
        dip twice is a 5 dB shelf and a 7 dB dip. See ``shape``.

    ``pre_limit=False``
        Skip the -9.1 dBFS pre-limiter, which exists to tame the low-frequency
        transients in raw vocoder output so the loudness gain has somewhere to
        go. Applying it to an already-mastered signal still crushes it, so the
        joined-film pass keeps this disabled and only performs the final
        gain/ceiling iteration.
    """
    if not ENABLED:
        return waveform

    x = np.asarray(waveform, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]

    # A batch is mastered item by item rather than handed back. The signal
    # ``_join_audio`` assembles is always 3-D -- it is a ``torch.cat`` over
    # (1, channels, samples) pieces -- and until now this function answered a
    # 3-D input by returning it *untouched*, before measuring anything and
    # therefore without logging. The join worked around that by squeezing
    # inline; the point of handling it here is that the workaround was a second
    # copy of the same logic, and any future caller that forgot it would get a
    # silent no-op instead of an error. A failure with no log line is the one
    # kind worth engineering out.
    if x.ndim == 3:
        if x.shape[0] == 0:
            return waveform
        items = [master(x[i], sample_rate, target_lufs, ceiling_dbtp, hp_hz,
                        limiter_ceil, pre_limit, max_gain_db, shape_tone)
                 for i in range(x.shape[0])]
        return np.stack(items, axis=0).astype(np.float32, copy=False)

    if x.ndim != 2 or x.shape[-1] == 0:
        return waveform

    before = measure_lufs(x, sample_rate)
    if not np.isfinite(before):
        return waveform

    y = x if hp_hz is None else highpass(x, sample_rate, hp_hz)
    # Tone shaping sits between the high-pass and the limiter on purpose: the
    # limiter must see the signal that will actually be shipped, or it will
    # spend its reduction on peaks the shelf was about to remove.
    if shape_tone and SHAPE_ENABLED:
        y = shape(y, sample_rate)
    if pre_limit:
        y = limiter(y, sample_rate, limiter_ceil)

    ceil = 10.0 ** (ceiling_dbtp / 20.0)

    # Gain and ceiling interact, so do it more than once. A single pass has only
    # two outcomes and both are wrong: if the ceiling does not bind, the target
    # is missed; if it does bind, the only way to honour it with one pass is to
    # scale the *whole* track down, which throws away exactly the loudness the
    # gain was there to add. Measured on the four-segment render, one pass cost
    # 2.19 LU that way. Iterating lets the limiter take back only the peaks that
    # poke through, and converges on the loudest the material can be at this
    # ceiling -- 3 rounds in practice, 4 allowed.
    #
    # ``max_gain_db`` bounds the *total* correction so that a segment which is
    # almost entirely silence cannot ask for a +60 dB boost that drags its noise
    # floor up into the join. See ``_MAX_TOTAL_GAIN_DB`` for why it is 30 and not
    # 18 -- at 18 the real render's quietest segment had 1.8 dB of margin.
    applied_db = 0.0
    for _ in range(_MAX_GAIN_PASSES):
        current = measure_lufs(y, sample_rate)
        if not np.isfinite(current):
            break
        error = float(np.clip(target_lufs - current,
                              -max_gain_db - applied_db,
                              max_gain_db - applied_db))
        if abs(error) < _GAIN_TOLERANCE_LU:
            break
        y *= 10.0 ** (error / 20.0)
        applied_db += error
        y = limiter(y, sample_rate, ceil)

    # The limiter guarantees peak <= ceil, so this only ever catches the case
    # where the loop bailed out on a measurement failure or hit the gain cap.
    peak = float(np.abs(y).max())
    if peak > ceil:
        y *= ceil / peak

    after = measure_lufs(y, sample_rate)
    short = "" if not np.isfinite(after) else (
        "" if abs(after - target_lufs) < 1.0 else " [ceiling-bound]")
    # ``hp`` is in the line on purpose: the corner is a constant that gets
    # changed, and without it in the log there is no way to tell from a render
    # whether the running process picked the change up. The pass that runs on
    # the film reports ``hp None``, which also distinguishes the two callers.
    log.info("[H3 Continuous] audio master: %.1f -> %.1f LUFS "
             "(target %.1f, %+.1f dB, peak %.2f dBTP, hp %s, shape %s)%s",
             before, after, target_lufs, applied_db,
             20.0 * np.log10(max(float(np.abs(y).max()), 1e-12)),
             "off" if hp_hz is None else "%.0f Hz" % hp_hz,
             "on" if (shape_tone and SHAPE_ENABLED) else "off", short)

    y = np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)
    return y[0] if squeeze else y


def master_segment(waveform, sample_rate, target_lufs=TARGET_LUFS,
                   hp_hz=HIGHPASS_HZ, ceiling_dbtp=CEILING_DBTP,
                   shape_tone=True):
    """Master **one segment** before the segments are joined.

    Why this has to happen per segment and not once at the end
    ---------------------------------------------------------
    H3 renders each segment in its own sampling pass, and the normaliser in
    ``comfy_extras/nodes_audio.py`` never fires on this model. Its guard is

        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std

    -- the divisor can only be >= 1.0, so it can only attenuate, and it only
    engages once ``5*std > 1``, i.e. above std 0.2. Measured on the 2026-09-22
    four-segment render, every segment decodes at std 0.019-0.052, an order of
    magnitude *below* that threshold:

        seg_01 std 0.0519 (5*std 0.259)   seg_02 std 0.0251 (0.126)
        seg_03 std 0.0278 (0.139)         seg_04 std 0.0188 (0.094)

    So the guard clamps to 1.0 for all four and the audio reaches disk at
    whatever level the vocoder happened to produce. The resulting per-segment
    loudness was:

        seg_01  -22.5 LUFS   seg_02  -27.6   seg_03  -27.5   seg_04  -30.3

    **7.8 LU of spread.** One gain over the joined track -- which is what the
    first version of this module did -- raises the whole film but leaves those
    jumps intact, and the jumps are what make the quieter segments read as thin
    and weak no matter how loud the film is overall. Hence: match the segments,
    then master the join.

    The same measurement shows seg_01 carrying 52% of its energy below 150 Hz
    against 0.2-2.9% for the other three, which is why the high-pass lives here
    rather than only at the end.

    This is ``master`` -- same iteration, same ceiling -- applied to a shorter
    signal. It is a separate name only so the *reason* it runs per segment has
    somewhere to live.

    The shape contract
    ------------------
    ``waveform`` may be (channels, samples) **or** (batch, channels, samples);
    the shape comes back unchanged. Both occur in practice -- ``_join_audio``
    hands over the batch form and ``save_clip`` does too -- so this stays
    shape-agnostic instead of demanding one, and ``master`` already masters a
    batch item by item. A one-item batch is not a special case of anything: it is
    just a segment.

    Where it runs, and where it must not
    ------------------------------------
    It runs **when a segment is saved** (``video_io.save_clip``), so a single
    segment already sits at delivery level when it is played on its own. That is
    the whole point: the work gets reviewed one segment at a time, and a raw
    segment measures 7-8 LU low, which is what reads as thin and weak before the
    film is ever joined. ``_join_audio`` then calls the same entry point per
    piece behind an ``is_mastered`` guard, so a segment an older build left on
    disk still gets repaired while a current one is not shaped twice.

    It must **not** be applied to a handoff tail. The tail is not an excerpt for
    listening; it is the conditioning signal the next segment is rendered
    against, and ``engine.py`` deliberately clamps it so the guide cannot outrun
    its target. Mastering it would move the condition itself, and the following
    segment would then be pinned to audio the film never contained. Both tail
    writes in ``nodes.py`` therefore pass ``master_audio=False``.
    """
    return master(waveform, sample_rate, target_lufs=target_lufs,
                  ceiling_dbtp=ceiling_dbtp, hp_hz=hp_hz,
                  shape_tone=shape_tone)
