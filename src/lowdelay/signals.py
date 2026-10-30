"""Synthetic test signals with exact ground truth, noise models and the zero-phase reference.

Synthetic ECG: sum-of-Gaussians beat model (P, Q, R, S, T waves; cf. McSharry et al. 2003,
IEEE TBME 50:289) evaluated analytically at the sample instants, so the clean waveform is
known exactly.  Synthetic PPG: two-Gaussian (systolic + reflected/diastolic) pulse model.
"""
from __future__ import annotations

import numpy as np

from .specs import FilterSpec, ideal_reference

# (name, time offset from R [s], amplitude [mV], width sigma [s]) at HR = 60 bpm
ECG_WAVES = [("P", -0.20, 0.15, 0.025),
             ("Q", -0.028, -0.12, 0.008),
             ("R", 0.0, 1.20, 0.010),
             ("S", 0.028, -0.30, 0.009),
             ("T", 0.28, 0.32, 0.050)]


def synthetic_ecg(fs, duration_s, hr_bpm=70.0, qrs_scale=1.0, hrv_std=0.03, amp_jitter=0.05,
                  rng=None):
    """Return (x [mV], r_peak_indices (float, exact), beat_times [s]).

    qrs_scale scales the Q/R/S widths (smaller -> sharper QRS -> more high-frequency energy).
    """
    rng = np.random.default_rng(rng)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    x = np.zeros(n)
    rr0 = 60.0 / hr_bpm
    beats = []
    tb = 0.4 + rng.uniform(0, 0.2)
    while tb < duration_s - 0.6:
        beats.append(tb)
        tb += rr0 * (1 + hrv_std * rng.standard_normal())
    beats = np.array(beats)
    for tb in beats:
        rr_scale = np.sqrt(rr0)              # Bazett-like QT scaling of the T-wave position
        for name, off, a, s in ECG_WAVES:
            if name in "QRS":
                s = s * qrs_scale
            if name == "T":
                off = off * rr_scale
            a = a * (1 + amp_jitter * rng.standard_normal())
            lo, hi = np.searchsorted(t, [tb + off - 6 * s, tb + off + 6 * s])
            x[lo:hi] += a * np.exp(-0.5 * ((t[lo:hi] - tb - off) / s) ** 2)
    return x, beats * fs, beats


def synthetic_ppg(fs, duration_s, hr_bpm=70.0, sharpness=1.0, hrv_std=0.03, rng=None):
    """Two-Gaussian PPG pulse train. Returns (x [a.u.], systolic-peak indices, beat times)."""
    rng = np.random.default_rng(rng)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    x = np.zeros(n)
    rr0 = 60.0 / hr_bpm
    beats, peaks = [], []
    tb = 0.3 + rng.uniform(0, 0.2)
    while tb < duration_s - 1.2:
        beats.append(tb)
        s1, s2 = 0.055 * sharpness, 0.11 * sharpness
        t1, t2 = tb + 0.18, tb + 0.18 + 0.30 * np.sqrt(rr0)
        for (tc, a, s) in [(t1, 1.0, s1), (t2, 0.45, s2)]:
            lo, hi = np.searchsorted(t, [tc - 6 * s, tc + 6 * s])
            x[lo:hi] += a * np.exp(-0.5 * ((t[lo:hi] - tc) / s) ** 2)
        peaks.append(t1 * fs)
        tb += rr0 * (1 + hrv_std * rng.standard_normal())
    return x, np.array(peaks), np.array(beats)


# --------------------------------------------------------------------------- noise
def powerline(n, fs, f0=60.0, amp=0.1, harmonics=(1.0, 0.3), drift_hz=0.2, rng=None):
    rng = np.random.default_rng(rng)
    t = np.arange(n) / fs
    out = np.zeros(n)
    for k, rel in enumerate(harmonics, start=1):
        fk = k * f0
        if fk >= fs / 2:
            break
        phase = 2 * np.pi * (fk * t + k * drift_hz / (2 * np.pi * 0.1) * np.sin(2 * np.pi * 0.1 * t))
        out += amp * rel * np.cos(phase + rng.uniform(0, 2 * np.pi))
    return out


def emg_noise(n, fs, rms, f_lo=20.0, rng=None):
    """Broadband muscle-like noise: white Gaussian noise high-passed (1st order) above f_lo."""
    rng = np.random.default_rng(rng)
    v = rng.standard_normal(n + 64)
    X = np.fft.rfft(v)
    f = np.fft.rfftfreq(len(v), 1 / fs)
    X *= (f / f_lo) / np.sqrt(1 + (f / f_lo) ** 2)
    v = np.fft.irfft(X, len(v))[64:]
    return rms * v / np.std(v)


# --------------------------------------------------------------------------- reference
def zero_phase_reference(x, spec: FilterSpec, pad=None):
    """Offline zero-phase filtering with the raised-cosine reference magnitude R(w) (FFT based,
    reflect-padded). This is the morphology target, NOT a real-time filter."""
    pad = pad or int(2 * spec.fs)
    xp = np.pad(x, pad, mode="reflect")
    nfft = int(2 ** np.ceil(np.log2(len(xp))))
    X = np.fft.rfft(xp, nfft)
    w = np.linspace(0, np.pi, len(X))
    y = np.fft.irfft(X * ideal_reference(spec, w), nfft)[: len(xp)]
    return y[pad: pad + len(x)]


def frac_delay(x, d):
    """Delay x by d samples (fractional) with an FFT phase ramp (circular, use on padded data)."""
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(n + 2 * int(abs(d) + 8))))
    X = np.fft.rfft(np.pad(x, (0, nfft - n), mode="edge"), nfft)
    w = np.linspace(0, np.pi, len(X))
    return np.fft.irfft(X * np.exp(-1j * w * d), nfft)[:n]
