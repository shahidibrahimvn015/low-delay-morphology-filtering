"""Stage 3: controlled synthetic experiments with exact ground truth.

A) ECG Monte-Carlo (bank from stage 1, unchanged): random HR, QRS sharpness, amplitudes,
   power-line (drifting 60 Hz + 120 Hz) and EMG-like noise.  Because every filter is linear,
   y = h*(x+v) = h*x + h*v, so morphology distortion (h*x vs reference) and residual noise
   (h*v) are measured separately and exactly.
B) QRS-sharpness sweep (noise-free) and noise-level sweep.
C) Error-model validation: predicted PRD (frequency-domain model with the synthetic PSD)
   versus measured PRD.
D) Transient tests: step response (overshoot, settling, 50 % delay) and impulse ringing.
E) PPG case study: separate spec (config/spec_ppg.json), bank designed with a PSD estimated
   from *development* seeds only, evaluated on disjoint test seeds.
Outputs: results/synthetic_*.csv/json
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from lowdelay import metrics as mt
from lowdelay import signals as sg
from lowdelay.analysis import freq_metrics, predicted_prd, psd_on_grid
from lowdelay.bank import ORDER, bank_table, build_bank, load_bank, save_bank
from lowdelay.data import welch_psd
from lowdelay.specs import load_spec
from lowdelay.util import CONFIG, RESULTS, load_json, load_params, save_json

P = load_params()
spec = load_spec(CONFIG / "spec_ecg.json")
bank = load_bank(RESULTS / "bank_ecg360.json")
z = np.load(RESULTS / "psd_mitdb_ds1.npz")
S_mit = psd_on_grid(spec, z["f"], z["pxx"], P["psd_floor"])
fm = {k: freq_metrics(f, spec, S_mit) for k, f in bank.items()}
fs = spec.fs
DUR = 20.0
t0 = time.time()


def draw(rng):
    return dict(hr=rng.uniform(50, 120), qrs=rng.uniform(0.6, 1.6),
                pli=rng.uniform(0.02, 0.2), emg=rng.uniform(0.005, 0.05))


def run_trial(seed, par, noise=True, bank_=bank, fm_=fm, spec_=spec, kind="ecg"):
    rng = np.random.default_rng(seed)
    if kind == "ecg":
        x, pk, _ = sg.synthetic_ecg(spec_.fs, DUR, par["hr"], par["qrs"], rng=rng)
        pol = 1
    else:
        x, pk, _ = sg.synthetic_ppg(spec_.fs, DUR, par["hr"], par["sharp"], rng=rng)
        pol = 1
    r = sg.zero_phase_reference(x, spec_)
    n = len(x)
    v = np.zeros(n)
    if noise:
        v = sg.powerline(n, spec_.fs, 60.0 if kind == "ecg" else 0, par["pli"], rng=rng) if kind == "ecg" else v
        v = v + sg.emg_noise(n, spec_.fs, par["emg"], f_lo=20.0 if kind == "ecg" else 5.0, rng=rng)
    skip = int(spec_.fs)
    sl = slice(skip, n - skip)
    rows = []
    for k, f in bank_.items():
        D = fm_[k]["tau_K"]
        y = f.apply(x)
        m = {"filter": k, "seed": seed, **par}
        m["prd_nom"] = mt.aligned_prd(y, r, D, skip)
        m["prd_opt"], m["D_star"] = mt.best_alignment(y, r, skip, D)
        sh, amp = mt.peak_metrics(y, r, pk, spec_.fs, D, skip=skip, polarity=pol)
        m["timing_err_ms"] = float(np.mean(np.abs(sh - D)) * 1000 / spec_.fs)
        m["peak_shift_ms"] = float(np.mean(sh) * 1000 / spec_.fs)
        m["peak_amp_err_pct"] = float(np.mean(np.abs(amp)))
        if kind == "ecg":
            m["st_err_uV"] = float(np.mean(np.abs(mt.st_error(y, r, pk, spec_.fs, D, skip=skip))) * 1000)
        if noise:
            yv = f.apply(v)
            m["snr_in_db"] = 10 * np.log10(np.sum(r[sl] ** 2) / np.sum(v[sl] ** 2))
            m["noise_red_db"] = 10 * np.log10(np.sum(v[sl] ** 2) / np.sum(yv[sl] ** 2))
            m["prd_total_nom"] = mt.aligned_prd(y + yv, r, D, skip)
        rows.append(m)
    return rows


# ------------------------------------------------------------------ A) Monte-Carlo
rows = []
for i in range(P["mc_trials"]):
    seed = 10_000 + i
    rows += run_trial(seed, draw(np.random.default_rng(seed + 7)))
mc = pd.DataFrame(rows)
mc.to_csv(RESULTS / "synthetic_mc.csv", index=False)
print(f"MC done ({len(mc)} rows) {time.time()-t0:.0f}s")
print(mc.groupby("filter")[["prd_opt", "prd_nom", "timing_err_ms", "peak_amp_err_pct", "st_err_uV",
                            "noise_red_db", "prd_total_nom"]].median().loc[ORDER].round(3))

# ------------------------------------------------------------------ B) sweeps
sw = []
for q in [0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2.0]:
    for j in range(20):
        par = dict(hr=70.0, qrs=q, pli=0.0, emg=0.0)
        sw += run_trial(20_000 + j, par, noise=False)
pd.DataFrame(sw).to_csv(RESULTS / "synthetic_qrs_sweep.csv", index=False)
nz = []
for lvl in [0.003, 0.01, 0.03, 0.1, 0.3]:
    for j in range(20):
        par = dict(hr=70.0, qrs=1.0, pli=lvl * 2, emg=lvl)
        nz += run_trial(30_000 + j, par, noise=True)
pd.DataFrame(nz).to_csv(RESULTS / "synthetic_noise_sweep.csv", index=False)
print(f"sweeps done {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ C) error-model validation
ps = []
for j in range(30):
    x, _, _ = sg.synthetic_ecg(fs, 60, 75, 1.0, rng=40_000 + j)
    f_, p_ = welch_psd(x - x.mean(), fs)
    ps.append(p_ / np.trapezoid(p_, f_))
S_syn = psd_on_grid(spec, f_, np.median(ps, axis=0), floor=0.0) + 1e-12
w, *_ = spec.grid()
val = []
qmeas = pd.DataFrame(sw)
qmeas = qmeas[qmeas.qrs == 1.1]
for k, f in bank.items():
    pred, Dp = predicted_prd(f.freq(w), spec, S_syn)
    val.append({"filter": k, "prd_pred_synPSD": pred, "D_pred": Dp,
                "prd_pred_mitPSD": fm[k]["prd_pred"],
                "prd_meas_median": float(qmeas[qmeas["filter"] == k].prd_opt.median())})
save_json(val, "synthetic_model_validation.json")
print(pd.DataFrame(val).round(3))

# ------------------------------------------------------------------ D) transients
tr = []
for k, f in bank.items():
    s = mt.step_metrics(f, fs)
    s.pop("response")
    s["ringing_frac"] = mt.ringing_energy(f, fs)
    s["filter"] = k
    tr.append(s)
save_json(tr, "synthetic_transients.json")
print(pd.DataFrame(tr).set_index("filter").loc[ORDER].round(3))

# ------------------------------------------------------------------ E) PPG case study
spec_p = load_spec(CONFIG / "spec_ppg.json")
print(spec_p.describe())
hp = load_json("hparams.json")
ps = []
for j in range(50):  # development seeds
    rng = np.random.default_rng(50_000 + j)
    x, _, _ = sg.synthetic_ppg(spec_p.fs, 60, rng.uniform(50, 110), rng.uniform(0.6, 1.4), rng=rng)
    f_, p_ = welch_psd(x - x.mean(), spec_p.fs)
    ps.append(p_ / np.trapezoid(p_, f_))
pxx_p = np.median(ps, axis=0)
np.savez(RESULTS / "psd_ppg_dev.npz", f=f_, pxx=pxx_p)
S_p = psd_on_grid(spec_p, f_, pxx_p, P["psd_floor"])
bank_p = build_bank(spec_p, S_p, hp)
save_bank(bank_p, RESULTS / "bank_ppg125.json")
rows_p = bank_table(bank_p, spec_p, S_p)
save_json({"spec": spec_p.__dict__, "rows": rows_p}, "freq_metrics_ppg125.json")
fm_p = {r["key"]: r for r in rows_p}
rp = []
for i in range(P["mc_trials"]):
    rng = np.random.default_rng(60_000 + i)
    par = dict(hr=rng.uniform(50, 110), sharp=rng.uniform(0.6, 1.4), emg=rng.uniform(0.005, 0.05), pli=0)
    rp += run_trial(60_000 + i, par, True, bank_p, fm_p, spec_p, kind="ppg")
ppg = pd.DataFrame(rp)
ppg.to_csv(RESULTS / "synthetic_ppg_mc.csv", index=False)
print(ppg.groupby("filter")[["prd_opt", "prd_nom", "timing_err_ms", "peak_amp_err_pct",
                             "noise_red_db"]].median().loc[ORDER].round(3))
for r in rows_p:
    print(f"  {r['key']:12s} {r['name']:40s} lat={r['tau_K_ms']:7.2f} ms PRDpred={r['prd_pred']:6.2f} "
          f"mult={r['ops_mult']} spec={r['meets_spec']}")
print(f"done {time.time()-t0:.0f}s")
