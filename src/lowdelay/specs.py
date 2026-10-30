"""Filtering specification and frequency-grid conventions.

Conventions (kept explicit everywhere in the project):
    f   [Hz]          physical frequency
    fs  [Hz]          sampling rate
    w   [rad/sample]  digital angular frequency, w = 2*pi*f/fs, 0 <= w <= pi
    nu  [cycles/sample] normalized frequency nu = f/fs (only used for printing)
Delays are carried in samples and converted to ms with 1000/fs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FilterSpec:
    name: str
    fs: float            # sampling rate [Hz]
    fp_hz: float         # passband edge [Hz]
    fst_hz: float        # stopband edge [Hz]
    ap_db: float         # max peak-to-peak passband ripple [dB], centred on unity gain
    as_db: float         # min stopband attenuation [dB] relative to unity gain
    latency_ms: float    # latency budget [ms]
    n_grid: int = 1024   # grid points on [0, pi]

    # ---- derived quantities -------------------------------------------------
    @property
    def wp(self) -> float:
        return 2 * np.pi * self.fp_hz / self.fs

    @property
    def ws(self) -> float:
        return 2 * np.pi * self.fst_hz / self.fs

    @property
    def delta_p(self) -> float:
        """Linear passband tolerance: |H| in [1-dp, 1+dp] <=> ripple_pp <= ap_db."""
        g = 10 ** (self.ap_db / 20)
        return (g - 1) / (g + 1)

    @property
    def delta_s(self) -> float:
        return 10 ** (-self.as_db / 20)

    @property
    def latency_samples(self) -> float:
        return self.latency_ms * 1e-3 * self.fs

    def ms(self, samples):
        return np.asarray(samples) * 1000.0 / self.fs

    def grid(self):
        """Dense uniform grid on [0, pi] plus boolean masks for the bands."""
        w = np.linspace(0, np.pi, self.n_grid)
        pb = w <= self.wp + 1e-12
        sb = w >= self.ws - 1e-12
        tb = ~(pb | sb)
        return w, pb, tb, sb

    def with_(self, **kw) -> "FilterSpec":
        d = asdict(self)
        d.update(kw)
        return FilterSpec(**d)

    def describe(self) -> str:
        return (f"{self.name}: fs={self.fs:g} Hz, passband 0-{self.fp_hz:g} Hz "
                f"(wp={self.wp:.4f} rad/sample), stopband >= {self.fst_hz:g} Hz "
                f"(ws={self.ws:.4f} rad/sample), ripple <= {self.ap_db:g} dB p-p "
                f"(dp={self.delta_p:.4f}), attenuation >= {self.as_db:g} dB "
                f"(ds={self.delta_s:.4g}), latency <= {self.latency_ms:g} ms "
                f"({self.latency_samples:.2f} samples)")


def load_spec(path) -> FilterSpec:
    d = json.loads(Path(path).read_text())
    d = {k: v for k, v in d.items() if k in FilterSpec.__dataclass_fields__}
    return FilterSpec(**d)


def ideal_reference(spec: FilterSpec, w):
    """Zero-phase reference magnitude: 1 in passband, raised-cosine transition, 0 in stopband.

    This is the 'morphology target' used by every comparison: every matched-spec
    filter approximates this magnitude, so the reference does not favour any
    particular architecture (in particular it is *not* the linear-phase FIR itself).
    """
    w = np.asarray(w)
    r = np.zeros_like(w, dtype=float)
    r[w <= spec.wp] = 1.0
    t = (w > spec.wp) & (w < spec.ws)
    x = (w[t] - spec.wp) / (spec.ws - spec.wp)
    r[t] = 0.5 * (1 + np.cos(np.pi * x))
    return r
