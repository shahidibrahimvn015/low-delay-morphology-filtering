"""Frequency-domain analysis: matched-spec check, group-delay metrics and the J objective.

Morphology-error model (derived in the report, Sec. 3)
-----------------------------------------------------
For a wide-sense-stationary input x with PSD S(w) and a target system
R(w) e^{-jwD} (zero-phase reference magnitude R, pure delay D), the expected
squared error of y = h * x is

    E|y[n] - r[n-D]|^2 = (1/pi) int_0^pi S(w) |H(w) - R(w) e^{-jwD}|^2 dw .

Normalising by the reference power gives the *predicted* PRD^2.  In the passband
(R = 1) with phase error phi_e(w) = arg H(w) + wD:

    |H - e^{-jwD}|^2 = (|H|-1)^2 + 2|H|(1 - cos phi_e) ~= (|H|-1)^2 + phi_e^2 ,
    phi_e(w) = - int_0^w (tau(v) - D) dv .

Cauchy-Schwarz gives phi_e(w)^2 <= w int_0^w (tau - D)^2 dv, hence

    E_phi <= E_tau := (1/Z) int_0^wp K(v) (tau(v) - D)^2 dv ,
    K(v)  = int_v^wp  w S(w) dw              (morphology group-delay kernel)

so  PRD^2 ~ E_mag + E_phi <= E_mag + E_tau.  This motivates
J = alpha*E_mag + beta*E_tau + gamma*D with alpha = beta = 1 and gamma the
price of latency; K(v) is decreasing, i.e. group-delay error at *low*
frequency is the most damaging because its phase error accumulates over every
higher frequency.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import minimize_scalar

from .specs import FilterSpec, ideal_reference


# --------------------------------------------------------------------------- weights
def psd_on_grid(spec: FilterSpec, f_hz, pxx, floor=1e-3):
    """Interpolate a one-sided PSD (Hz axis) to the w-grid, normalise to max 1, add a floor."""
    w, *_ = spec.grid()
    f_grid = w * spec.fs / (2 * np.pi)
    s = np.interp(f_grid, f_hz, pxx, left=pxx[0], right=pxx[-1])
    s = s / s.max()
    return s + floor


def flat_weight(spec: FilterSpec):
    w, *_ = spec.grid()
    return np.ones_like(w)


def gd_kernel(spec: FilterSpec, S):
    """K(v) = int_v^wp w S(w) dw on the passband part of the grid."""
    w, pb, *_ = spec.grid()
    wpb, spb = w[pb], S[pb]
    c = cumulative_trapezoid(wpb * spb, wpb, initial=0.0)
    return c[-1] - c


# --------------------------------------------------------------------------- metrics
def predicted_prd(H, spec: FilterSpec, S, D=None):
    """Predicted PRD [%] and optimal alignment delay D [samples] (Sec. 3 error model)."""
    w, *_ = spec.grid()
    R = ideal_reference(spec, w)
    Z = trapezoid(S * R ** 2, w)

    def err(d):
        return trapezoid(S * np.abs(H - R * np.exp(-1j * w * d)) ** 2, w) / Z

    if D is None:
        # coarse scan then refine (objective is multimodal for long delays)
        ds = np.linspace(0, 400, 1601)
        e = [err(d) for d in ds]
        d0 = ds[int(np.argmin(e))]
        r = minimize_scalar(err, bounds=(max(0, d0 - 0.5), d0 + 0.5), method="bounded")
        D = float(r.x)
    return 100 * np.sqrt(err(D)), D


def freq_metrics(filt, spec: FilterSpec, S=None) -> dict:
    """All frequency-domain metrics of the evaluation framework for one filter."""
    w, pb, tb, sb = spec.grid()
    if S is None:
        S = np.ones_like(w)
    H = filt.freq(w)
    mag = np.abs(H)
    tau = filt.group_delay(w)

    m = {}
    m["dev_p"] = float(np.max(np.abs(mag[pb] - 1)))
    m["ripple_pp_db"] = float(20 * np.log10(mag[pb].max() / mag[pb].min()))
    m["atten_db"] = float(-20 * np.log10(mag[sb].max()))
    m["trans_peak"] = float(mag[tb].max()) if tb.any() else float("nan")
    m["meets_spec"] = bool(m["dev_p"] <= spec.delta_p * 1.002 and m["atten_db"] >= spec.as_db - 0.02
                           and (not tb.any() or m["trans_peak"] <= 1 + spec.delta_p * 1.002))

    # ----- group delay (passband only - stopband GD is not signal delay)
    wpb, tpb, spb = w[pb], tau[pb], S[pb]
    K = gd_kernel(spec, S)
    m["tau_dc"] = float(tau[0])
    m["tau_mean"] = float(trapezoid(tpb, wpb) / trapezoid(np.ones_like(wpb), wpb))
    m["tau_W"] = float(trapezoid(spb * tpb, wpb) / trapezoid(spb, wpb))
    m["tau_K"] = float(trapezoid(K * tpb, wpb) / trapezoid(K, wpb))   # latency D
    m["tau_max_pb"] = float(tpb.max())
    m["dtau_max"] = float(np.max(np.abs(tpb - m["tau_mean"])))
    m["dtau_pp"] = float(tpb.max() - tpb.min())
    m["dtau_rms"] = float(np.sqrt(trapezoid((tpb - m["tau_mean"]) ** 2, wpb) / wpb[-1]))
    m["dtau_rms_W"] = float(np.sqrt(trapezoid(spb * (tpb - m["tau_W"]) ** 2, wpb) / trapezoid(spb, wpb)))

    # ----- J terms (normalised by reference power Z, see module docstring)
    R = ideal_reference(spec, w)
    Z = trapezoid(S * R ** 2, w)
    D = m["tau_K"]
    phase_e = np.unwrap(np.angle(H[pb])) + wpb * D
    m["E_mag"] = float(trapezoid(spb * (mag[pb] - 1) ** 2, wpb) / Z)
    m["E_phi"] = float(trapezoid(spb * phase_e ** 2, wpb) / Z)
    m["E_tau"] = float(trapezoid(K * (tpb - D) ** 2, wpb) / Z)
    m["prd_pred"], m["D_opt"] = predicted_prd(H, spec, S)
    m["prd_pred_at_tauK"] = predicted_prd(H, spec, S, D=D)[0]

    # ----- physical units
    for k in ["tau_dc", "tau_mean", "tau_W", "tau_K", "tau_max_pb", "dtau_max", "dtau_pp",
              "dtau_rms", "dtau_rms_W", "D_opt"]:
        m[k + "_ms"] = float(spec.ms(m[k]))
    # 0.5 % tolerance: designs constrained to D <= budget land on it up to optimiser precision
    m["latency_ok"] = bool(m["tau_K"] <= spec.latency_samples * 1.005)
    m.update({"ops_" + k: v for k, v in filt.ops().items()})
    m["stable"] = filt.is_stable()
    m["max_pole_radius"] = filt.max_pole_radius()
    return m


def J_value(m: dict, alpha=1.0, beta=1.0, gamma=0.0):
    """J = alpha*E_mag + beta*E_tau + gamma*D   (D in samples)."""
    return alpha * m["E_mag"] + beta * m["E_tau"] + gamma * m["tau_K"]


def check_group_delay(filt, spec: FilterSpec, n=8192):
    """Cross-check analytic GD against -d(unwrapped phase)/dw in the passband."""
    w = np.linspace(0, spec.wp, n)
    H = filt.freq(w)
    ph = np.unwrap(np.angle(H))
    gd_num = -np.gradient(ph, w)
    gd_ana = filt.group_delay(w)
    return w, ph, gd_num, gd_ana, float(np.max(np.abs(gd_num[2:-2] - gd_ana[2:-2])))
