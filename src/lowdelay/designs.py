"""Filter designs: literature baselines and the proposed morphology-aware low-delay designs.

Baselines (reproduced)
    lp_fir            Parks-McClellan equiripple linear-phase FIR, minimum length meeting spec
    mp_fir            minimum-phase FIR obtained by exact spectral factorisation (zero reflection)
                      of the LP-FIR -> identical magnitude, minimum delay
    iir_classic       elliptic / Butterworth (bilinear-transform) IIR, minimum order meeting spec
    iir_ap_flat       IIR + 2nd-order all-pass cascade minimising the *unweighted* passband
                      group-delay variance (classical equaliser criterion, e.g. Quelhas et al.)
    ld_fir(flat)      low-delay FIR, complex least squares to e^{-jwD} with uniform weighting
                      and Chebyshev band constraints (the classical constrained low-delay FIR family)

Proposed
    ld_fir(psd)       morphology-optimal constrained low-delay FIR (MO-LDFIR):
                      minimise the *signal-PSD-weighted* complex error, i.e. the expected
                      output-morphology MSE, subject to SOC constraints that guarantee the
                      matched magnitude spec;  convex (SOCP) for every fixed latency D
    iir_ap_morph      energy-weighted, latency-penalised all-pass equaliser (EW-APE):
                      minimise J = E_tau(K-kernel) + gamma * D
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, signal

from .analysis import freq_metrics, gd_kernel
from .filters import Filter
from .specs import FilterSpec, ideal_reference


# =========================================================================== FIR baselines
def lp_fir(spec: FilterSpec, numtaps=None, max_taps=801) -> Filter:
    """Equiripple linear-phase FIR (Parks-McClellan). If numtaps is None the minimum
    length (odd or even) that meets the spec is searched."""
    bands = [0, spec.fp_hz, spec.fst_hz, spec.fs / 2]
    wt = [1.0, spec.delta_p / spec.delta_s]

    def design(n):
        h = signal.remez(n, bands, [1, 0], weight=wt, fs=spec.fs, maxiter=200)
        return Filter(f"LP-FIR (N={n})", "LP-FIR", b=h, meta={"linear_phase": True, "numtaps": n})

    if numtaps is not None:
        return design(numtaps)
    # Kaiser/Herrmann-style starting estimate
    dw = (spec.fst_hz - spec.fp_hz) / spec.fs
    n0 = max(3, int((-20 * np.log10(np.sqrt(spec.delta_p * spec.delta_s)) - 13) / (14.6 * dw)) - 6)
    for n in range(n0, max_taps):
        f = design(n)
        if freq_metrics(f, spec)["meets_spec"]:
            return f
    raise RuntimeError("LP-FIR: spec not met")


def to_minimum_phase(f: Filter) -> Filter:
    """Exact minimum-phase factor with the *same* magnitude: reflect zeros outside |z|=1."""
    h = f.b
    z = np.roots(h)
    out = np.abs(z) > 1 + 1e-9
    z_new = z.copy()
    z_new[out] = 1 / np.conj(z[out])
    hm = np.real(np.poly(z_new))
    # |1 - zk/z| = |zk| |1 - z^-1 / conj(zk)|: magnitudes differ only by a constant, which is
    # restored on a grid (robust against root round-off)
    w = np.linspace(0, np.pi, 4096)
    from .filters import _poly_resp
    s = np.median(np.abs(_poly_resp(h, w)[0]) / np.maximum(np.abs(_poly_resp(hm, w)[0]), 1e-300))
    hm = hm * s
    if np.sum(hm) * np.sum(h) < 0:
        hm = -hm
    return Filter(f"MP-FIR (N={len(h)})", "MP-FIR", b=hm, meta={"numtaps": len(h), "from": f.name})


def mp_fir(spec: FilterSpec, numtaps=None) -> Filter:
    return to_minimum_phase(lp_fir(spec, numtaps))


# =========================================================================== IIR baselines
def _center_gain(sos, spec: FilterSpec):
    """Scale an IIR so its passband magnitude is centred on unity (matched-spec convention)."""
    f = Filter("tmp", "IIR", sos=sos)
    w, pb, *_ = spec.grid()
    m = np.abs(f.freq(w[pb]))
    sos = sos.copy()
    sos[0, :3] *= 2 / (m.max() + m.min())
    return sos


def iir_classic(spec: FilterSpec, ftype="ellip", order=None) -> Filter:
    fp, fst, fs = spec.fp_hz, spec.fst_hz, spec.fs
    # design for slightly tighter tolerances; the centring step costs ~ap/2 dB of attenuation
    rp, rs = spec.ap_db, spec.as_db + spec.ap_db / 2 + 0.05
    if ftype == "ellip":
        n, wn = signal.ellipord(fp, fst, rp, rs, fs=fs)
        n = order or n
        sos = signal.ellip(n, rp, rs, wn, output="sos", fs=fs)
    elif ftype == "butter":
        n, wn = signal.buttord(fp, fst, rp, rs, fs=fs)
        n = order or n
        sos = signal.butter(n, wn, output="sos", fs=fs)
    elif ftype == "cheby1":
        n, wn = signal.cheb1ord(fp, fst, rp, rs, fs=fs)
        n = order or n
        sos = signal.cheby1(n, rp, wn, output="sos", fs=fs)
    else:
        raise ValueError(ftype)
    sos = _center_gain(sos, spec)
    name = {"ellip": "Elliptic", "butter": "Butterworth", "cheby1": "Chebyshev-I"}[ftype]
    return Filter(f"{name} IIR (n={n})", "IIR", sos=sos, meta={"ftype": ftype, "order": int(n)})


def bessel_min_order(spec: FilterSpec, max_order=24):
    """Bessel IIR (maximally-flat delay prototype): search order/cutoff meeting the spec.
    Returns (Filter or None, log). The bilinear transform warps the flat delay, and the
    gentle Bessel roll-off usually makes a tight transition band infeasible."""
    log = []
    for n in range(2, max_order + 1):
        # largest cutoff that still meets attenuation at fst, then check the passband
        def atten_margin(fc):
            sos = signal.bessel(n, fc, output="sos", fs=spec.fs, norm="mag")
            f = Filter("b", "IIR", sos=_center_gain(sos, spec))
            return freq_metrics(f, spec)
        lo, hi = spec.fp_hz * 0.5, min(spec.fst_hz * 1.5, spec.fs / 2 * 0.999)
        best = None
        for fc in np.linspace(lo, hi, 60):
            m = atten_margin(fc)
            if m["atten_db"] >= spec.as_db:
                best = (fc, m)
        if best is not None:
            log.append((n, best[0], best[1]["dev_p"], best[1]["meets_spec"]))
            if best[1]["meets_spec"]:
                sos = _center_gain(signal.bessel(n, best[0], output="sos", fs=spec.fs, norm="mag"), spec)
                return Filter(f"Bessel IIR (n={n})", "IIR", sos=sos, meta={"ftype": "bessel", "order": n}), log
    return None, log


# =========================================================================== all-pass equaliser
def allpass_sos(r, th):
    """Second-order all-pass sections with poles r e^{+-j th}."""
    r, th = np.atleast_1d(r), np.atleast_1d(th)
    a1, a2 = -2 * r * np.cos(th), r ** 2
    return np.column_stack([a2, a1, np.ones_like(r), np.ones_like(r), a1, a2])


def allpass_gd(r, th, w):
    g = np.zeros_like(w)
    for rk, tk in zip(np.atleast_1d(r), np.atleast_1d(th)):
        for s in (+1, -1):
            g += (1 - rk ** 2) / (1 - 2 * rk * np.cos(w - s * tk) + rk ** 2)
    return g


def _fit_allpass(spec, base: Filter, K, weight, gamma, r_max, n_starts, seed, minimax=False):
    w, pb, *_ = spec.grid()
    wpb = w[pb]
    tau0 = base.group_delay(wpb)
    wt = weight / np.trapezoid(weight, wpb)
    rng = np.random.default_rng(seed)

    def unpack(x):
        return x[:K], x[K:]

    def cost(x):
        r, th = unpack(x)
        tau = tau0 + allpass_gd(r, th, wpb)
        D = np.trapezoid(wt * tau, wpb)
        e = (tau - D) ** 2
        if minimax:  # smooth approximation of max|tau - D| (p = 16 norm)
            return np.trapezoid(e ** 8, wpb) ** (1 / 16) / wpb[-1] ** (1 / 16) + gamma * D
        return np.trapezoid(wt * e, wpb) + gamma * D

    bounds = [(0.05, r_max)] * K + [(1e-3, np.pi - 1e-3)] * K
    best = None
    for _ in range(n_starts):
        # initial poles placed where the base GD is low (classical heuristic) + jitter
        th0 = np.sort(rng.uniform(0.2, 1.2, K) * spec.wp)
        r0 = rng.uniform(0.4, min(0.9, r_max), K)
        res = optimize.minimize(cost, np.r_[r0, th0], method="L-BFGS-B", bounds=bounds,
                                options={"maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    return unpack(best.x), best


def allpass_resp(r, th, w):
    A = np.ones_like(w, dtype=complex)
    z1 = np.exp(-1j * w)
    for rk, tk in zip(np.atleast_1d(r), np.atleast_1d(th)):
        a1, a2 = -2 * rk * np.cos(tk), rk ** 2
        A *= (a2 + a1 * z1 + z1 ** 2) / (1 + a1 * z1 + a2 * z1 ** 2)
    return A


def _fit_allpass_morph(spec, base: Filter, K, S, gamma, r_max, n_starts, seed, d_max=None):
    """Proposed EW-APE: minimise the exact predicted morphology error + gamma*D over (r, theta, D).

        J(r, th, D) = (1/Z) int_0^pi S |H_IIR A_{r,th} - R e^{-jwD}|^2 dw  +  gamma * D
        (optionally with the hard latency constraint 0 <= D <= d_max)
    """
    w, *_ = spec.grid()
    Hb = base.freq(w)
    R = ideal_reference(spec, w)
    Z = np.trapezoid(S * R ** 2, w)
    rng = np.random.default_rng(seed)

    def cost(x):
        r, th, D = x[:K], x[K:2 * K], x[-1]
        H = Hb * allpass_resp(r, th, w)
        return np.trapezoid(S * np.abs(H - R * np.exp(-1j * w * D)) ** 2, w) / Z + gamma * D

    bounds = [(0.05, r_max)] * K + [(1e-3, np.pi - 1e-3)] * K + [(0.0, d_max or 400.0)]
    best = None
    for _ in range(n_starts):
        th0 = np.sort(rng.uniform(0.2, 1.2, K) * spec.wp)
        r0 = rng.uniform(0.05 if d_max else 0.3, min(0.9, r_max), K)
        tau = base.group_delay(w[w <= spec.wp]) + allpass_gd(r0, th0, w[w <= spec.wp])
        x0 = np.r_[r0, th0, min(np.mean(tau), d_max or np.inf)]
        res = optimize.minimize(cost, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
        if best is None or res.fun < best.fun:
            best = res
    return (best.x[:K], best.x[K:2 * K]), best


def iir_ap(spec: FilterSpec, base: Filter, K: int, criterion="flat", gamma=0.0,
           S=None, r_max=0.97, n_starts=12, seed=0, d_max=None) -> Filter:
    """IIR + K second-order all-pass sections.

    criterion='flat'    classical equaliser: uniform passband GD-variance, gamma = 0 (baseline)
    criterion='minimax' classical minimax-flat GD (p-norm approximation), gamma = 0 (baseline)
    criterion='morph'   proposed EW-APE: exact PSD-weighted morphology error + latency price gamma
    criterion='kernel'  ablation: GD surrogate with the derived kernel K(v) + gamma
    criterion='psd'     ablation: GD variance weighted by the PSD instead of K(v) + gamma
    """
    w, pb, *_ = spec.grid()
    if K == 0:
        return base
    if criterion == "morph":
        (r, th), res = _fit_allpass_morph(spec, base, K, S, gamma, r_max, n_starts, seed, d_max)
    else:
        if criterion in ("flat", "minimax"):
            weight, gamma_eff = np.ones(pb.sum()), 0.0
        elif criterion == "kernel":
            weight, gamma_eff = gd_kernel(spec, S) + 1e-6, gamma
        elif criterion == "psd":
            weight, gamma_eff = S[pb], gamma
        else:
            raise ValueError(criterion)
        (r, th), res = _fit_allpass(spec, base, K, weight, gamma_eff, r_max, n_starts, seed,
                                    minimax=criterion == "minimax")
    sos = np.vstack([base.sos, allpass_sos(r, th)])
    tag = {"flat": "AP-flat", "minimax": "AP-minimax", "morph": "EW-APE", "kernel": "AP-kernel",
           "psd": "AP-psd"}[criterion]
    extra = (f", g={gamma:.3g}" if gamma else "") + (f", D<={d_max:.3g}" if d_max else "")
    return Filter(f"{base.name.split(' (')[0]} + {tag} (K={K}{extra})",
                  "EW-APE" if criterion == "morph" else "IIR+AP", sos=sos,
                  meta={"K": K, "criterion": criterion, "gamma": gamma, "d_max": d_max,
                        "r": list(map(float, r)),
                        "theta": list(map(float, th)), "base": base.name, "opt_cost": float(res.fun)})


# =========================================================================== low-delay FIR (SOCP)
def ld_fir(spec: FilterSpec, N: int, D: float, S=None, weighting="psd", eps_c=None,
           solver="CLARABEL", stop_margin=0.98, verbose=False):
    """Constrained low-delay FIR via second-order-cone programming.

        minimise    int W(w) |H(w) - R(w) e^{-jwD}|^2 dw
        subject to  |H(w) - e^{-jwD}| <= eps_c        w in passband   (=> magnitude spec)
                    |H(w)|            <= delta_s      w in stopband
                    |H(w)|            <= 1 + delta_p  w in transition band (no peaking)

    weighting='psd'  -> W = S (signal PSD): the objective is the expected morphology MSE (proposed)
    weighting='flat' -> W = 1                 (classical low-delay complex-LS baseline)
    Returns Filter or None if infeasible.
    """
    import cvxpy as cp

    w, pb, tb, sb = spec.grid()
    if eps_c is None:
        eps_c = spec.delta_p
    W = np.ones_like(w) if (weighting == "flat" or S is None) else S
    n = np.arange(N)
    C, Sn = np.cos(np.outer(w, n)), np.sin(np.outer(w, n))
    R = ideal_reference(spec, w)
    Tr, Ti = R * np.cos(w * D), -R * np.sin(w * D)
    dw = np.gradient(w)
    sw = np.sqrt(W * dw)

    h = cp.Variable(N)
    Hr, Hi = C @ h, -Sn @ h
    obj = cp.sum_squares(cp.multiply(sw, Hr - Tr)) + cp.sum_squares(cp.multiply(sw, Hi - Ti))
    cons = [cp.norm(cp.vstack([Hr[pb] - Tr[pb], Hi[pb] - Ti[pb]]), axis=0) <= eps_c,
            cp.norm(cp.vstack([Hr[sb], Hi[sb]]), axis=0) <= spec.delta_s * stop_margin]
    if tb.any():
        cons.append(cp.norm(cp.vstack([Hr[tb], Hi[tb]]), axis=0) <= 1 + spec.delta_p)
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate") or h.value is None:
        return None
    tag = "MO-LDFIR" if weighting == "psd" else "LD-FIR-flat"
    return Filter(f"{tag} (N={N}, D={D:g})", "LD-FIR" if weighting == "flat" else "MO-LDFIR",
                  b=np.asarray(h.value).ravel(),
                  meta={"numtaps": N, "D": D, "weighting": weighting, "eps_c": eps_c,
                        "status": prob.status, "objective": float(prob.value)})


def ld_fir_min_length(spec, D, S=None, weighting="psd", n_lo=None, n_hi=400, **kw):
    """Smallest N for which the constrained low-delay FIR with latency D is feasible."""
    n_lo = n_lo or int(np.ceil(2 * D + 1))
    n_lo = max(n_lo, 3)
    lo, hi, best = n_lo, n_lo, None
    # exponential search then bisection on N
    while hi <= n_hi:
        f = ld_fir(spec, hi, D, S, weighting, **kw)
        if f is not None and freq_metrics(f, spec)["meets_spec"]:
            best = f
            break
        lo, hi = hi, int(hi * 1.3) + 2
    if best is None:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f = ld_fir(spec, mid, D, S, weighting, **kw)
        if f is not None and freq_metrics(f, spec)["meets_spec"]:
            hi, best = mid, f
        else:
            lo = mid
    return best
