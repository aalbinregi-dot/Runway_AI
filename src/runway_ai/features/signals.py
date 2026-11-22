"""Signal-processing primitives shared by the feature extractors.

A walking clip filmed head-on contains two superimposed effects: the periodic
gait signal (~0.8-1.2 Hz) and a slow perspective trend as the subject
approaches the lens. Almost every feature here first removes the trend, then
measures the oscillation - mixing the two is the single easiest way to get
meaningless gait features.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division that degrades to ``default`` instead of raising or returning inf."""
    return float(a / b) if abs(b) > 1e-9 else default


def detrend_poly(x: np.ndarray, order: int = 3) -> np.ndarray:
    """Remove a low-order polynomial trend (the perspective drift)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n <= order + 1:
        return x - x.mean()
    t = np.linspace(-1.0, 1.0, n)
    coef = np.polyfit(t, x, order)
    return x - np.polyval(coef, t)


GAIT_PERIOD_S = 1.15      # nominal stride period
HIGHPASS_HZ = 0.45        # corner: below the ~0.9 Hz stride rate, above any drift


def gait_highpass(x: np.ndarray, fps: float, cutoff: float = HIGHPASS_HZ) -> np.ndarray:
    """Isolate gait-band oscillation by high-passing below the stride rate.

    A polynomial detrend is not good enough: the pelvis traverses several torso
    lengths across the frame as the subject approaches, while the sway we want
    is a few tenths of one, and a cubic fit to that trajectory leaves residuals
    larger than the signal.

    A one-stride moving average was the next attempt and is also not good
    enough. Its stopband rejection is poor, so a strongly curved drift - and
    perspective drift is close to exponential as the subject nears the lens -
    leaves a residual comparable to the sway itself, because the mean of a
    convex function over a window sits above its centre value.

    A zero-phase Butterworth high-pass rejects the drift properly. The corner
    sits at 0.45 Hz, below the ~0.9 Hz stride rate, so gait passes essentially
    untouched (~0.91 amplitude at 0.9 Hz after the forward-backward pass) while
    anything slower is removed. Zero-phase matters because event timing is
    measured downstream.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    nyq = fps / 2.0
    wn = cutoff / nyq
    # filtfilt needs a few times the filter order in samples to be stable.
    if n < 30 or not (0 < wn < 0.99):
        return detrend_poly(x, 2)
    b, a = butter(2, wn, btype="high")
    return filtfilt(b, a, x, padlen=min(3 * 4, n - 1))


def angle_from_vertical(vec: np.ndarray) -> np.ndarray:
    """Signed angle (degrees) of a (T, 2) image-space vector from straight up.

    Image ``y`` grows downward, so "up" is ``-y``. Positive means leaning toward
    +x (the subject's left in a frontal view).
    """
    return np.degrees(np.arctan2(vec[:, 0], -vec[:, 1]))


def angle_from_horizontal(vec: np.ndarray) -> np.ndarray:
    """Signed angle (degrees) of a (T, 2) vector from the horizontal axis."""
    return np.degrees(np.arctan2(-vec[:, 1], vec[:, 0]))


def transverse_tilt(vec: np.ndarray) -> np.ndarray:
    """Tilt of a left-right body segment from horizontal, folded to [-90, 90].

    Shoulder and hip lines are *undirected*: a level shoulder line is level
    whether the vector is measured left-to-right or right-to-left. Using the raw
    angle makes that distinction matter enormously - a subject facing the camera
    puts their anatomical left on the image right, so the vector points the
    other way and a level line reads as 180 deg instead of 0. Taking ``abs()``
    of that measures nothing: perfectly level becomes 180, and a five-degree
    tilt becomes 175 or 185 depending on direction.

    Folding into [-90, 90] gives the deviation from horizontal directly, with
    the sign carrying which side is low, and makes the feature invariant to a
    mirrored view.
    """
    a = np.degrees(np.arctan2(-vec[:, 1], vec[:, 0]))
    a = np.where(a > 90.0, a - 180.0, a)
    return np.where(a < -90.0, a + 180.0, a)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at ``b`` in degrees, for (T, 2) tracks (e.g. elbow flexion)."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.sum(v1 * v2, axis=1) / np.clip(n1 * n2, 1e-9, None)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def amplitude(x: np.ndarray, fps: float) -> float:
    """Robust gait-band oscillation amplitude.

    The 5-95 percentile span is used rather than min-max so one bad frame from
    a dropped landmark cannot dominate.
    """
    d = gait_highpass(x, fps)
    return float(np.percentile(d, 95) - np.percentile(d, 5))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=float)))))


def jerk_index(x: np.ndarray, fps: float) -> float:
    """Mean absolute third derivative, normalised by signal amplitude.

    A standard smoothness measure in motor-control research: hesitant, jerky
    movement has high spectral content in the third derivative.
    """
    x = gait_highpass(np.asarray(x, dtype=float), fps)
    if x.shape[0] < 6:
        return 0.0
    d3 = np.diff(x, n=3) * fps**3
    return safe_div(float(np.mean(np.abs(d3))), float(np.std(x)) + 1e-6)


def spectrum(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of a detrended signal, restricted to plausible gait rates."""
    x = gait_highpass(np.asarray(x, dtype=float), fps)
    nper = int(min(len(x), max(32, len(x) // 2)))
    f, pxx = welch(x, fs=fps, nperseg=nper)
    keep = (f >= 0.3) & (f <= 6.0)
    return f[keep], pxx[keep]


def dominant_frequency(x: np.ndarray, fps: float) -> tuple[float, float]:
    """Return ``(peak_hz, spectral_purity)``.

    ``spectral_purity`` is the fraction of in-band power inside a narrow window
    around the peak. A metronomic walk concentrates its power at one frequency
    (purity -> 1); an irregular one smears it (purity -> 0). This is the core
    rhythm measurement.
    """
    f, pxx = spectrum(x, fps)
    if f.size == 0 or pxx.sum() <= 0:
        return 0.0, 0.0
    i = int(np.argmax(pxx))
    peak = float(f[i])
    band = (f >= peak * 0.75) & (f <= peak * 1.25)
    return peak, safe_div(float(pxx[band].sum()), float(pxx.sum()))


def high_freq_ratio(x: np.ndarray, fps: float, cutoff: float = 2.5) -> float:
    """Fraction of in-band power above ``cutoff`` Hz.

    Gait lives below ~2 Hz, so power above the cutoff is tremor or correction
    wobble - a direct read on instability.
    """
    f, pxx = spectrum(x, fps)
    if pxx.sum() <= 0:
        return 0.0
    return safe_div(float(pxx[f > cutoff].sum()), float(pxx.sum()))


def lowpass(x: np.ndarray, fps: float, cutoff: float = 3.0) -> np.ndarray:
    """Zero-phase Butterworth low-pass. Zero-phase matters: a causal filter
    would shift every detected event in time and corrupt the timing features."""
    x = np.asarray(x, dtype=float)
    nyq = fps / 2.0
    wn = min(cutoff / nyq, 0.95)
    if x.shape[0] < 15 or wn >= 0.95:
        return x
    b, a = butter(2, wn, btype="low")
    return filtfilt(b, a, x)


def gait_phase(x: np.ndarray, fps: float) -> np.ndarray:
    """Unwrapped instantaneous phase of a quasi-periodic gait signal.

    Obtained from the analytic signal (Hilbert transform) of the band-limited
    trace. Phase advances continuously, so cycle boundaries can be located to
    sub-frame precision and a momentarily weak cycle cannot be skipped.
    """
    y = lowpass(gait_highpass(np.asarray(x, dtype=float), fps), fps, cutoff=3.0)
    if y.size < 8 or np.std(y) < 1e-12:
        return np.zeros_like(y)
    return np.unwrap(np.angle(hilbert(y - y.mean())))


def cycle_times(x: np.ndarray, fps: float) -> np.ndarray:
    """Times (seconds) of successive gait-cycle boundaries for one limb.

    Peak-picking was the obvious approach and it is not good enough: a single
    missed or duplicated peak changes one interval by a factor of two and ruins
    every timing statistic built on it, and frame quantisation alone puts a ~3%
    floor under the stride CV at 30 fps. Tracking the *phase* instead and
    interpolating each 2*pi crossing gives one event per cycle by construction,
    with sub-frame resolution.
    """
    ph = gait_phase(x, fps)
    if ph.size < 8 or not np.all(np.isfinite(ph)) or np.ptp(ph) < 2 * np.pi:
        return np.array([])

    two_pi = 2 * np.pi
    ph = np.maximum.accumulate(ph)          # enforce monotonicity for the search

    # Discard a margin at each end. The Hilbert transform assumes a periodic,
    # infinite signal; at a finite boundary its phase estimate is distorted,
    # which alone put a spurious ~0.1 CV on a *perfectly* periodic input and so
    # masked the real timing variability we are trying to measure.
    n = ph.size
    est_period = two_pi / max(np.median(np.diff(ph)), 1e-9)
    margin = int(min(max(0.5 * est_period, 0.15 * n), 0.3 * n))
    lo, hi = margin, n - margin
    if hi - lo < 4:
        lo, hi = 0, n
    ph_c = ph[lo:hi]

    k0 = int(np.ceil(ph_c[0] / two_pi))
    k1 = int(np.floor(ph_c[-1] / two_pi))
    times = []
    for k in range(k0, k1 + 1):
        level = k * two_pi
        i = int(np.searchsorted(ph_c, level))
        if 0 < i < ph_c.size:
            p0, p1 = ph_c[i - 1], ph_c[i]
            frac = (level - p0) / (p1 - p0) if p1 > p0 else 0.0
            times.append((lo + i - 1 + frac) / fps)
    return np.asarray(times)


def step_events(ankle_y: np.ndarray, fps: float) -> np.ndarray:
    """Frame indices of foot strikes for one foot.

    A foot strike is the *lowest* point of the ankle, i.e. a peak in image ``y``
    (which grows downward). Cycle boundaries come from the phase tracker, then
    each is snapped to the true local maximum nearby so that features which
    *index* the pose at strike - leg obliquity, step width - land on the right
    frame rather than a phase-convention offset.
    """
    y = lowpass(gait_highpass(np.asarray(ankle_y, dtype=float), fps), fps, cutoff=3.0)
    if y.size < 8 or np.std(y) < 1e-9:
        return np.array([], dtype=int)

    starts = cycle_times(ankle_y, fps)
    if starts.size == 0:
        return np.array([], dtype=int)

    period = float(np.mean(np.diff(starts)) * fps) if starts.size > 1 else y.size
    half = max(int(round(period / 2)), 1)
    out = []
    for t in starts:
        c = int(round(t * fps))
        lo, hi = max(c - half, 0), min(c + half + 1, y.size)
        if hi > lo:
            out.append(lo + int(np.argmax(y[lo:hi])))
    return np.unique(np.asarray(out, dtype=int))


def interval_cv(times_s: np.ndarray, fps: float = 1.0) -> tuple[float, float]:
    """Return ``(mean_interval_s, coefficient_of_variation)`` for event *times*.

    ``times_s`` is in seconds (as returned by :func:`cycle_times`), not frames.
    The CV of stride time is the standard clinical measure of gait regularity;
    healthy steady walking sits near 0.02-0.04.
    """
    times_s = np.asarray(times_s, dtype=float)
    if times_s.size < 3:
        return 0.0, 0.0
    iv = np.diff(times_s)
    m = float(np.mean(iv))
    return m, safe_div(float(np.std(iv)), m)


def phase_offset(a: np.ndarray, b: np.ndarray, fps: float) -> float:
    """Phase lag between two signals, in cycles of their shared period.

    Used for left/right limb timing: ideal alternating gait gives 0.5.
    """
    a = gait_highpass(a, fps)
    b = gait_highpass(b, fps)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    n = len(a)
    xc = np.correlate(a / a.std(), b / b.std(), mode="full") / n
    lags = np.arange(-n + 1, n)
    period_hz, _ = dominant_frequency(a, fps)
    if period_hz <= 0:
        return 0.0
    period = fps / period_hz
    window = np.abs(lags) <= period
    best = lags[window][int(np.argmax(xc[window]))]
    return float(abs(best) / period)


def symmetry_index(left: float, right: float) -> float:
    """Normalised L/R difference in [0, 1]; 0 = perfectly symmetric.

    This is the Robinson symmetry index used in gait labs, rescaled.
    """
    return safe_div(abs(left - right), 0.5 * (abs(left) + abs(right)), 0.0)
