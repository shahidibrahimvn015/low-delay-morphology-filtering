"""Time-domain / application-level metrics (morphology, timing, overshoot, ST level)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .signals import frac_delay


def prd(y, r):
    """Normalised PRD [%] (mean-removed reference energy in the denominator)."""
    return 100 * np.sqrt(np.sum((y - r) ** 2) / np.sum((r - r.mean()) ** 2))


def aligned_prd(y, r, delay, skip):
    """PRD of y against the reference delayed by `delay` samples, ignoring `skip` edge samples."""
    rd = frac_delay(r, delay)
    sl = slice(skip, len(y) - skip)
    return prd(y[sl], rd[sl])


def best_alignment(y, r, skip, d_hint, span=6.0, search_len=None):
    """Fractional delay minimising the PRD (pure-delay component removed): returns (prd, D*).

    D* is a property of the filter, so it is searched on the first `search_len` samples
    (default: 120 s at the record's rate is plenty) and the PRD is then evaluated once on
    the whole record."""
    n = len(y)
    m = n if search_len is None else min(n, search_len)
    ys, rs = y[:m], r[:m]
    sk = min(skip, m // 4)
    f = lambda d: aligned_prd(ys, rs, d, sk)
    ds = np.arange(max(0.0, d_hint - span), d_hint + span + 1e-9, 0.5)
    e = [f(d) for d in ds]
    d0 = ds[int(np.argmin(e))]
    res = minimize_scalar(f, bounds=(max(0.0, d0 - 0.5), d0 + 0.5), method="bounded",
                          options={"xatol": 1e-3})
    dstar = float(res.x)
    return float(aligned_prd(y, r, dstar, skip) if m < n else res.fun), dstar


def _refine_peak(x, i):
    """Parabolic interpolation of a local extremum at integer index i."""
    if i <= 0 or i >= len(x) - 1:
        return float(i), x[i]
    a, b, c = x[i - 1], x[i], x[i + 1]
    den = a - 2 * b + c
    if den == 0:
        return float(i), b
    p = 0.5 * (a - c) / den
    return i + p, b - 0.25 * (a - c) * p


def peak_metrics(y, r, peaks, fs, delay, search_ms=60.0, skip=0, polarity=1):
    """Per-beat timing and amplitude of the main peak (R wave / PPG systolic peak).

    Reference peak: extremum of r near the given location. Output peak: extremum of y near
    (reference peak + nominal delay). Returns per-beat arrays (timing shift [samples] *relative
    to the reference peak*, i.e. including the delay, and relative amplitude error [%]).
    """
    hw = int(round(search_ms * 1e-3 * fs))
    sh, amp = [], []
    for p in peaks:
        i0 = int(round(p))
        if i0 - hw < skip or i0 + hw + int(delay) + hw >= len(y) - skip:
            continue
        lo = i0 - hw - 1
        seg = polarity * r[lo:i0 + hw + 2]
        ir = 1 + int(np.argmax(seg[1:-1]))
        tr, ar = _refine_peak(seg, ir)
        tr += lo
        c = int(round(tr + delay))
        lo = c - hw - 1
        seg = polarity * y[lo:c + hw + 2]
        if len(seg) < 3:
            continue
        iy = 1 + int(np.argmax(seg[1:-1]))
        ty, ay = _refine_peak(seg, iy)
        ty += lo
        sh.append(ty - tr)
        amp.append(100 * (ay - ar) / abs(ar))
    return np.array(sh), np.array(amp)


def st_error(y, r, peaks, fs, delay, win_ms=(80, 120), skip=0):
    """Mean |ST-level error| [signal units] in a window after the R peak (aligned at `delay`)."""
    rd = frac_delay(r, delay)
    a, b = (int(round(v * 1e-3 * fs)) for v in win_ms)
    errs = []
    for p in peaks:
        i = int(round(p + delay))
        if i - 1 < skip or i + b >= len(y) - skip:
            continue
        errs.append(np.mean(y[i + a:i + b]) - np.mean(rd[i + a:i + b]))
    return np.array(errs)


def step_metrics(filt, fs, n=None):
    """Step response: overshoot [%], undershoot [%], 10-90 rise time [ms], 2 % settling [ms],
    50 % crossing delay [ms]. The input step occurs at sample 0."""
    n = n or int(fs)
    s = filt.apply(np.ones(n))
    final = s[-1]
    t = np.arange(n) * 1000 / fs
    over = 100 * max(0.0, s.max() - final) / abs(final)
    under = 100 * max(0.0, -s.min()) / abs(final)
    i10 = np.argmax(s >= 0.1 * final)
    i90 = np.argmax(s >= 0.9 * final)
    i50 = np.argmax(s >= 0.5 * final)
    out = np.abs(s - final) > 0.02 * abs(final)
    settle = t[np.nonzero(out)[0][-1] + 1] if out.any() else 0.0
    return {"overshoot_pct": float(over), "undershoot_pct": float(under),
            "rise_ms": float(t[i90] - t[i10]), "settle_ms": float(settle), "delay50_ms": float(t[i50]),
            "response": s}


def ringing_energy(filt, fs, tail_ms=(30, 300)):
    """Fraction of impulse-response energy that arrives later than tail_ms[0] after the
    impulse-response energy centroid (a ringing/'tail' indicator)."""
    n = int(tail_ms[1] * 1e-3 * fs) + 512
    h = filt.impulse(n)
    e = h ** 2
    c = np.sum(np.arange(n) * e) / e.sum()
    k = int(c + tail_ms[0] * 1e-3 * fs)
    return float(e[k:].sum() / e.sum())
