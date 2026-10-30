"""Matched-specification filter bank, persistence and per-record evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import designs as d
from . import metrics as mt
from .analysis import freq_metrics
from .filters import Filter
from .specs import FilterSpec

# display order / short labels used in every table and figure
ORDER = ["LP-FIR", "MP-FIR", "Elliptic", "Butterworth", "Ell+AP-flat", "LD-FIR-flat",
         "MO-LDFIR", "EW-APE"]
PROPOSED = {"MO-LDFIR", "EW-APE"}


def d_budget(spec: FilterSpec) -> float:
    """Largest integer design latency that respects the budget."""
    return float(np.floor(spec.latency_samples + 1e-9))


def ld_min_length(spec: FilterSpec, S) -> int:
    f = d.ld_fir_min_length(spec, d_budget(spec), S, "psd")
    if f is None:
        raise RuntimeError("no feasible low-delay FIR at the latency budget")
    return len(f.b)


def build_bank(spec: FilterSpec, S, hp: dict, log=print) -> dict[str, Filter]:
    """Design the matched-spec set. hp: selected hyper-parameters (from the dev set):
        hp['ldfir_N']   FIR length used by both low-delay FIRs (flat and MO)
        hp['ewape_K']   number of all-pass sections of EW-APE under the latency budget
        hp['apflat_K']  number of sections of the classical (latency-unconstrained) equaliser
    """
    D = d_budget(spec)
    bank = {}
    bank["LP-FIR"] = d.lp_fir(spec)
    bank["MP-FIR"] = d.to_minimum_phase(bank["LP-FIR"])
    ell = d.iir_classic(spec, "ellip")
    bank["Elliptic"] = ell
    bank["Butterworth"] = d.iir_classic(spec, "butter")
    bank["Ell+AP-flat"] = d.iir_ap(spec, ell, hp["apflat_K"], "flat", seed=hp.get("seed", 0))
    N = hp.get("ldfir_N")
    if N is None:   # transferable rule: minimum feasible length at the budget + selected slack
        N = ld_min_length(spec, S) + hp["ldfir_N_slack"]
    bank["LD-FIR-flat"] = d.ld_fir(spec, N, D, S, "flat")
    bank["MO-LDFIR"] = d.ld_fir(spec, N, D, S, "psd")
    bank["EW-APE"] = d.iir_ap(spec, ell, hp["ewape_K"], "morph", S=S, d_max=spec.latency_samples,
                              seed=hp.get("seed", 0), n_starts=hp.get("n_starts", 12))
    for k, f in bank.items():
        if f is None:
            raise RuntimeError(f"design {k} infeasible")
        log(f"  {k:12s} -> {f.name}")
    return bank


def save_bank(bank: dict[str, Filter], path):
    out = {}
    for k, f in bank.items():
        out[k] = {"name": f.name, "family": f.family, "meta": f.meta,
                  "b": None if f.b is None else np.asarray(f.b).tolist(),
                  "sos": None if f.sos is None else np.asarray(f.sos).tolist()}
    Path(path).write_text(json.dumps(out, indent=1))


def load_bank(path) -> dict[str, Filter]:
    raw = json.loads(Path(path).read_text())
    return {k: Filter(v["name"], v["family"],
                      b=None if v["b"] is None else np.array(v["b"]),
                      sos=None if v["sos"] is None else np.array(v["sos"]), meta=v["meta"])
            for k, v in raw.items()}


def bank_table(bank, spec, S) -> list[dict]:
    rows = []
    for k in ORDER:
        if k in bank:
            m = freq_metrics(bank[k], spec, S)
            m["key"], m["name"] = k, bank[k].name
            rows.append(m)
    return rows


def evaluate_record(bank, fmet, x, r, peaks, fs, spec, polarity=1, skip_s=1.0, noise_free_r=None):
    """Apply every filter to x and compare with the zero-phase reference r.

    fmet: dict key -> freq_metrics (supplies the nominal latency tau_K used for alignment).
    Returns dict key -> metrics. Two PRDs are reported:
        prd_nom  reference delayed by the filter's *nominal* latency tau_K (what a real-time
                 system that compensates its known delay would see)
        prd_opt  reference delayed by the best fractional delay (pure-delay effect removed:
                 isolates morphology/shape distortion)
    """
    skip = int(skip_s * fs)
    out = {}
    for k, f in bank.items():
        y = f.apply(x)
        D = fmet[k]["tau_K"]
        m = {"prd_nom": mt.aligned_prd(y, r, D, skip)}
        m["prd_opt"], m["D_star"] = mt.best_alignment(y, r, skip, D, search_len=int(120 * fs))
        sh, amp = mt.peak_metrics(y, r, peaks, fs, D, skip=skip, polarity=polarity)
        st = mt.st_error(y, r, peaks, fs, D, skip=skip)
        if len(sh):
            m["r_shift_ms"] = float(np.mean(sh) * 1000 / fs)
            m["r_jitter_ms"] = float(np.std(sh) * 1000 / fs)
            m["r_timing_err_ms"] = float(np.mean(np.abs(sh - D)) * 1000 / fs)
            m["r_amp_err_pct"] = float(np.mean(np.abs(amp)))
            m["r_amp_bias_pct"] = float(np.mean(amp))
        if len(st):
            m["st_err_uV"] = float(np.mean(np.abs(st)) * 1000)
        m["n_beats"] = int(len(sh))
        out[k] = m
    return out
