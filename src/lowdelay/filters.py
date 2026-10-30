"""A small, uniform container for FIR and IIR (second-order-section) filters.

Frequency response and group delay are evaluated *analytically* from the
coefficients:  for a polynomial P(e^{jw}) = sum_k p_k e^{-jwk}

    tau_P(w) = Re{ sum_k k p_k e^{-jwk} / P(e^{jw}) }

and tau_{B/A} = tau_B - tau_A.  This avoids numerical differentiation; the
result is cross-checked against the derivative of the *unwrapped* phase in
analysis.check_group_delay().
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


def _poly_resp(p, w):
    p = np.asarray(p, dtype=float)
    k = np.arange(len(p))
    E = np.exp(-1j * np.outer(w, k))
    return E @ p, E @ (k * p)


def poly_group_delay(p, w):
    P, dP = _poly_resp(p, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.real(dP / P)


@dataclass
class Filter:
    name: str
    family: str                      # 'LP-FIR', 'MP-FIR', 'IIR', 'IIR+AP', 'LD-FIR', ...
    b: np.ndarray | None = None      # FIR taps (kind == 'fir')
    sos: np.ndarray | None = None    # second-order sections (kind == 'sos')
    meta: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "fir" if self.b is not None else "sos"

    # ---- responses ----------------------------------------------------------
    def freq(self, w):
        w = np.asarray(w, dtype=float)
        if self.kind == "fir":
            return _poly_resp(self.b, w)[0]
        H = np.ones_like(w, dtype=complex)
        for s in self.sos:
            H *= _poly_resp(s[:3], w)[0] / _poly_resp(s[3:], w)[0]
        return H

    def group_delay(self, w):
        w = np.asarray(w, dtype=float)
        if self.kind == "fir":
            return poly_group_delay(self.b, w)
        g = np.zeros_like(w)
        for s in self.sos:
            g += poly_group_delay(s[:3], w) - poly_group_delay(s[3:], w)
        return g

    def apply(self, x, axis=-1):
        if self.kind == "fir":
            return signal.lfilter(self.b, [1.0], x, axis=axis)
        return signal.sosfilt(self.sos, x, axis=axis)

    def impulse(self, n):
        d = np.zeros(n)
        d[0] = 1.0
        return self.apply(d)

    # ---- structure ----------------------------------------------------------
    def poles_zeros(self):
        if self.kind == "fir":
            return np.roots(self.b) if len(self.b) > 1 else np.array([]), np.zeros(0)
        z, p, _ = signal.sos2zpk(self.sos)
        return z, p

    def is_stable(self) -> bool:
        if self.kind == "fir":
            return True
        _, p = self.poles_zeros()
        return bool(np.all(np.abs(p) < 1))

    def max_pole_radius(self) -> float:
        if self.kind == "fir":
            return 0.0
        return float(np.max(np.abs(self.poles_zeros()[1])))

    def ops(self) -> dict:
        """Arithmetic cost per output sample (direct-form realisations)."""
        if self.kind == "fir":
            n = len(self.b)
            sym = bool(self.meta.get("linear_phase", False))
            mult = (n + 1) // 2 if sym else n
            return {"mult": mult, "add": n - 1, "states": n - 1, "order": n - 1}
        mult = add = order = states = 0
        for s in self.sos:
            dden = 2 if abs(s[5]) > 1e-15 else (1 if abs(s[4]) > 1e-15 else 0)
            dnum = 2 if abs(s[2]) > 1e-15 else (1 if abs(s[1]) > 1e-15 else 0)
            order += dden                     # filter order = number of poles
            states += max(dden, dnum)         # delay elements of a direct-form-II section
            num = [c for c in s[:3] if abs(c) > 1e-15]
            den = [c for c in s[4:] if abs(c) > 1e-15]
            # unity coefficients cost no multiplication
            mult += sum(abs(abs(c) - 1) > 1e-12 for c in num) + len(den)
            add += len(num) + len(den) - 1
        return {"mult": int(mult), "add": int(add), "states": int(states), "order": int(order)}
