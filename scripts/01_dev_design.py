"""Stage 1 (development set only): PSD weight, hyper-parameter selection, final filter bank.

Everything tuned here uses MIT-BIH DS1 exclusively. DS2 and PTB-XL are never touched.
Outputs
    results/psd_mitdb_ds1.npz         ECG PSD (Hz axis) used as the morphology weight S(w)
    results/dev_selection.json        DS1 scores for every candidate hyper-parameter
    results/hparams.json              locked hyper-parameters
    results/bank_ecg360.json          final matched-spec filter bank (coefficients)
    results/freq_metrics_ecg360.json  frequency-domain metrics of the bank
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from lowdelay import designs as d
from lowdelay.analysis import freq_metrics, psd_on_grid
from lowdelay.bank import build_bank, d_budget, ld_min_length, save_bank, bank_table
from lowdelay.data import DS1, load_mitdb, remove_baseline, welch_psd
from lowdelay.signals import zero_phase_reference
from lowdelay import metrics as mt
from lowdelay.specs import load_spec
from lowdelay.util import CONFIG, RESULTS, load_params, provenance, save_json

P = load_params()
spec = load_spec(CONFIG / "spec_ecg.json")
print(spec.describe())
t0 = time.time()

# ---------------------------------------------------------------- 1. PSD weight from DS1
psds, f = [], None
dev = {}
for rec in DS1:
    r = load_mitdb(rec, seconds=P["dev_seconds_per_record"])
    x = remove_baseline(r["x"], r["fs"])
    f, p = welch_psd(x, r["fs"])
    psds.append(p / np.trapezoid(p, f))           # unit-power normalisation per record
    dev[rec] = (x, r["beats"])
pxx = np.median(psds, axis=0)
np.savez(RESULTS / "psd_mitdb_ds1.npz", f=f, pxx=pxx, records=DS1)
S = psd_on_grid(spec, f, pxx, P["psd_floor"])
print(f"PSD from {len(DS1)} DS1 records; {100*np.trapezoid(pxx[f<=spec.fp_hz], f[f<=spec.fp_hz]):.1f}% "
      f"of power below fp")


def dev_score(filt):
    """Median DS1 PRD (nominal alignment) and median prd_opt for one filter."""
    fm = freq_metrics(filt, spec, S)
    a, b = [], []
    for rec, (x, beats) in dev.items():
        ref = zero_phase_reference(x, spec)
        y = filt.apply(x)
        sk = int(spec.fs)
        a.append(mt.aligned_prd(y, ref, fm["tau_K"], sk))
        b.append(mt.best_alignment(y, ref, sk, fm["tau_K"])[0])
    return float(np.median(a)), float(np.median(b)), fm


# ---------------------------------------------------------------- 2. MO-LDFIR length
D = d_budget(spec)
n_min = ld_min_length(spec, S)
print(f"latency budget {spec.latency_ms} ms -> design delay D = {D:g} samples; minimum feasible N = {n_min}")
sel = {"ldfir": [], "ewape": []}
for slack in P["ldfir_N_slack_candidates"]:
    filt = d.ld_fir(spec, n_min + slack, D, S, "psd")
    pn, po, fm = dev_score(filt)
    sel["ldfir"].append({"slack": slack, "N": n_min + slack, "prd_nom_med": pn, "prd_opt_med": po,
                         "prd_pred": fm["prd_pred"], "tau_K_ms": fm["tau_K_ms"]})
    print(f"  MO-LDFIR N={n_min+slack:3d}: DS1 median PRD_nom={pn:.3f}%  PRD_opt={po:.3f}%")
best = min(s["prd_nom_med"] for s in sel["ldfir"])
slack = min(s["slack"] for s in sel["ldfir"] if s["prd_nom_med"] <= best * (1 + P["ldfir_parsimony_rel_tol"]))

# ---------------------------------------------------------------- 3. EW-APE sections under budget
from lowdelay.designs import iir_classic, iir_ap
ell = iir_classic(spec, "ellip")
for K in P["ewape_K_candidates"]:
    filt = iir_ap(spec, ell, K, "morph", S=S, d_max=spec.latency_samples, seed=P["seed"],
                  n_starts=P["n_starts"], r_max=P["allpass_r_max"])
    pn, po, fm = dev_score(filt)
    sel["ewape"].append({"K": K, "prd_nom_med": pn, "prd_opt_med": po, "prd_pred": fm["prd_pred"],
                         "tau_K_ms": fm["tau_K_ms"], "latency_ok": fm["latency_ok"]})
    print(f"  EW-APE K={K}: DS1 median PRD_nom={pn:.3f}%  PRD_opt={po:.3f}%  tau_K={fm['tau_K_ms']:.2f} ms")
ok = [s for s in sel["ewape"] if s["latency_ok"]]
bestK = min(ok, key=lambda s: s["prd_nom_med"])["K"]

hp = {"ldfir_N_slack": int(slack), "ewape_K": int(bestK), "apflat_K": P["apflat_K"],
      "seed": P["seed"], "n_starts": P["n_starts"], "selected_on": "MIT-BIH DS1 (first %d s/record)"
      % P["dev_seconds_per_record"]}
print("locked hyper-parameters:", hp)
save_json(sel, "dev_selection.json")
save_json(hp, "hparams.json")

# ---------------------------------------------------------------- 4. final bank
bank = build_bank(spec, S, hp)
save_bank(bank, RESULTS / "bank_ecg360.json")
rows = bank_table(bank, spec, S)
save_json({"spec": spec.__dict__, "rows": rows, "provenance": provenance()}, "freq_metrics_ecg360.json")
for m in rows:
    print(f"  {m['key']:12s} spec={m['meets_spec']!s:5s} latency={m['tau_K_ms']:6.2f} ms  "
          f"dtau_max={m['dtau_max_ms']:6.2f} ms  PRD_pred={m['prd_pred']:6.2f}%  mult={m['ops_mult']}")
print(f"done in {time.time()-t0:.0f} s")
