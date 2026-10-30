"""Stage 6: robustness, ablation and sensitivity.

A) Coefficient quantisation (B-bit fixed point) versus the float64 reference implementation:
   spec compliance, attenuation, passband deviation, predicted PRD, stability, pole radius.
B) Ablation of the proposed designs (frequency-domain model + measured on MIT-BIH DS2, clean):
   MO-LDFIR: PSD weight vs flat weight; matched vs mismatched PSD (synthetic ECG PSD);
             no length slack; PSD floor 1e-4 / 1e-2.
   EW-APE:   exact morphology criterion vs GD-kernel surrogate vs PSD-weighted GD vs flat GD
             (K = 2, latency unconstrained) - isolates the value of the derived objective.
C) Latency-budget sweep at the base spec: best achievable predicted PRD per architecture.
Outputs: results/robust_*.json / csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from lowdelay import designs as d
from lowdelay import metrics as mt
from lowdelay import signals as sg
from lowdelay.analysis import freq_metrics, psd_on_grid
from lowdelay.bank import ORDER, d_budget, load_bank
from lowdelay.data import DS2, load_mitdb, remove_baseline, welch_psd
from lowdelay.filters import Filter
from lowdelay.specs import load_spec
from lowdelay.util import CONFIG, RESULTS, load_json, load_params, save_json

P = load_params()
hp = load_json("hparams.json")
spec = load_spec(CONFIG / "spec_ecg.json")
bank = load_bank(RESULTS / "bank_ecg360.json")
z = np.load(RESULTS / "psd_mitdb_ds1.npz")
S = psd_on_grid(spec, z["f"], z["pxx"], P["psd_floor"])
t0 = time.time()


# ------------------------------------------------------------------ A) quantisation
def quantise(f: Filter, bits: int) -> Filter:
    if f.kind == "fir":
        e = np.ceil(np.log2(np.max(np.abs(f.b))))           # shared exponent (block floating)
        q = 2.0 ** (bits - 1 - e)
        return Filter(f.name, f.family, b=np.round(f.b * q) / q, meta=f.meta)
    sos = f.sos.copy()
    q = 2.0 ** (bits - 2)                                    # Q2.(B-2): |a1| < 2 representable
    for s in sos:
        g = 2.0 ** np.ceil(np.log2(max(np.max(np.abs(s[:3])), 1e-12)))   # per-section num scale
        s[:3] = np.round(s[:3] / g * q) / q * g
        s[4:] = np.round(s[4:] * q) / q
    return Filter(f.name, f.family, sos=sos, meta=f.meta)


qrows = []
for k in ORDER:
    for bits in [64] + P["quant_bits"]:
        fq = bank[k] if bits == 64 else quantise(bank[k], bits)
        m = freq_metrics(fq, spec, S) if fq.is_stable() else {"stable": False}
        qrows.append({"filter": k, "bits": bits, **{c: m.get(c) for c in
                      ["meets_spec", "atten_db", "dev_p", "prd_pred", "tau_K_ms", "stable", "max_pole_radius"]}})
q = pd.DataFrame(qrows)
q.to_csv(RESULTS / "robust_quantisation.csv", index=False)
print(q.pivot(index="filter", columns="bits", values="atten_db").loc[ORDER].round(2))
print(f"A done {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ B) ablations
D = d_budget(spec)
N = len(bank["MO-LDFIR"].b)
n_min = N - hp["ldfir_N_slack"]
ps = []
for j in range(30):
    x, _, _ = sg.synthetic_ecg(spec.fs, 60, 75, 1.0, rng=40_000 + j)
    f_, p_ = welch_psd(x - x.mean(), spec.fs)
    ps.append(p_ / np.trapezoid(p_, f_))
S_syn = psd_on_grid(spec, f_, np.median(ps, axis=0), P["psd_floor"])
abl = {
    "MO-LDFIR (proposed)": bank["MO-LDFIR"],
    "- PSD weight (flat)": bank["LD-FIR-flat"],
    "- matched PSD (synthetic-ECG PSD)": d.ld_fir(spec, N, D, S_syn, "psd"),
    "- length slack (N = N_min)": d.ld_fir(spec, n_min, D, S, "psd"),
    "PSD floor 1e-4": d.ld_fir(spec, N, D, psd_on_grid(spec, z["f"], z["pxx"], 1e-4), "psd"),
    "PSD floor 1e-2": d.ld_fir(spec, N, D, psd_on_grid(spec, z["f"], z["pxx"], 1e-2), "psd"),
}
ell = d.iir_classic(spec, "ellip")
for crit, lab in [("morph", "EW-APE exact criterion (proposed)"), ("kernel", "AP: GD-kernel surrogate K(v)"),
                  ("psd", "AP: PSD-weighted GD variance"), ("flat", "AP: flat GD variance (classical)"),
                  ("minimax", "AP: minimax GD (classical)")]:
    abl[lab + " K=2"] = d.iir_ap(spec, ell, 2, crit, S=S, seed=hp["seed"])

# measure on DS2 (clean protocol, first 10 min per record to bound run time)
data = []
for rec in DS2:
    dd = load_mitdb(rec, seconds=600)
    x = remove_baseline(dd["x"], dd["fs"])
    data.append((x, sg.zero_phase_reference(x, spec)))
arows = []
for lab, f in abl.items():
    if f is None:
        arows.append({"variant": lab, "feasible": False})
        continue
    m = freq_metrics(f, spec, S)
    prds = []
    for x, r in data:
        y = f.apply(x)
        prds.append(mt.best_alignment(y, r, int(spec.fs), m["tau_K"])[0])
    arows.append({"variant": lab, "feasible": True, "meets_spec": m["meets_spec"], "N_or_order": m["ops_order"],
                  "tau_K_ms": m["tau_K_ms"], "dtau_max_ms": m["dtau_max_ms"], "E_mag": m["E_mag"],
                  "E_phi": m["E_phi"], "E_tau": m["E_tau"], "prd_pred": m["prd_pred"],
                  "prd_ds2_median": float(np.median(prds)),
                  "prd_ds2_q1": float(np.percentile(prds, 25)), "prd_ds2_q3": float(np.percentile(prds, 75))})
    print(f"  {lab:42s} PRDpred={m['prd_pred']:.3f}  DS2 median PRD_opt={np.median(prds):.3f}  "
          f"lat={m['tau_K_ms']:.2f} ms")
save_json(arows, "robust_ablation.json")
print(f"B done {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ C) latency-budget sweep
lp = bank["LP-FIR"]
brows = []
for lat_ms in [5, 8, 10, 12.5, 15, 20, 25, 30, 40, 45, 60, 80]:
    sp = spec.with_(latency_ms=float(lat_ms))
    Db = d_budget(sp)
    best = {}
    # FIR families
    if Db >= (len(lp.b) - 1) / 2:
        best["LP-FIR"] = freq_metrics(lp, sp, S)["prd_pred"]
    mp = bank["MP-FIR"]
    if freq_metrics(mp, sp, S)["latency_ok"]:
        best["MP-FIR"] = freq_metrics(mp, sp, S)["prd_pred"]
    fmin = d.ld_fir_min_length(sp, Db, S, "psd") if Db >= 2 else None
    if fmin is not None:
        f = d.ld_fir(sp, len(fmin.b) + hp["ldfir_N_slack"], Db, S, "psd")
        if f is not None:
            m = freq_metrics(f, sp, S)
            best["MO-LDFIR"] = m["prd_pred"]
            best["MO-LDFIR_N"] = len(f.b)
    # IIR families
    for n in (ell.meta["order"], ell.meta["order"] + 1):
        fe = d.iir_classic(sp, "ellip", n)
        m = freq_metrics(fe, sp, S)
        if m["latency_ok"]:
            best["Elliptic"] = min(best.get("Elliptic", 1e9), m["prd_pred"])
    for K in (1, 2, 3):
        f = d.iir_ap(sp, ell, K, "morph", S=S, d_max=sp.latency_samples, seed=hp["seed"], n_starts=8)
        m = freq_metrics(f, sp, S)
        if m["latency_ok"]:
            if m["prd_pred"] < best.get("EW-APE", 1e9):
                best["EW-APE"], best["EW-APE_K"] = m["prd_pred"], K
    brows.append({"latency_ms": lat_ms, **best})
    print(f"  budget {lat_ms:5.1f} ms: " + ", ".join(f"{k}={v:.2f}" for k, v in best.items()), flush=True)
pd.DataFrame(brows).to_csv(RESULTS / "robust_budget_sweep.csv", index=False)
print(f"done {time.time()-t0:.0f}s")
