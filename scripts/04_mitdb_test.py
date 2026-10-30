"""Stage 4: held-out test on MIT-BIH DS2 (22 records, inter-patient; never used for tuning).

Two protocols per record (full 30-min MLII lead, baseline-removed by the common pre-stage):
  clean      filters applied to the recorded ECG; target = zero-phase reference of the same ECG
  stress     + synthetic drifting 60/120-Hz power-line (0.1 mV) and EMG-like noise (0.03 mV rms)
             added to the input only; target unchanged  -> denoising + distortion
Beat positions: reference annotations (atr), refined on the reference signal.
Output: results/mitdb_ds2.csv (one row per record x filter x protocol)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from lowdelay import signals as sg
from lowdelay.analysis import freq_metrics, psd_on_grid
from lowdelay.bank import ORDER, evaluate_record, load_bank
from lowdelay.data import DS2, load_mitdb, peak_polarity, remove_baseline
from lowdelay.specs import load_spec
from lowdelay.util import CONFIG, RESULTS, load_params

P = load_params()
spec = load_spec(CONFIG / "spec_ecg.json")
bank = load_bank(RESULTS / "bank_ecg360.json")
z = np.load(RESULTS / "psd_mitdb_ds1.npz")
S = psd_on_grid(spec, z["f"], z["pxx"], P["psd_floor"])
fm = {k: freq_metrics(f, spec, S) for k, f in bank.items()}

rows = []
t0 = time.time()
for i, rec in enumerate(DS2):
    d = load_mitdb(rec)
    x = remove_baseline(d["x"], d["fs"])
    r = sg.zero_phase_reference(x, spec)
    pol = peak_polarity(r, d["beats"])
    rng = np.random.default_rng(P["seed"] + i)
    v = sg.powerline(len(x), d["fs"], 60.0, 0.1, rng=rng) + sg.emg_noise(len(x), d["fs"], 0.03, rng=rng)
    for proto, xin in (("clean", x), ("stress", x + v)):
        res = evaluate_record(bank, fm, xin, r, d["beats"], d["fs"], spec, polarity=pol)
        for k, m in res.items():
            rows.append({"record": rec, "protocol": proto, "filter": k, **m})
    print(f"[{time.time()-t0:5.0f}s] {rec}: " + "  ".join(
        f"{k}={rows[-len(bank)*2 + j]['prd_opt']:.2f}" for j, k in enumerate(bank)), flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "mitdb_ds2.csv", index=False)
for proto in ("clean", "stress"):
    print(f"\n== {proto}: median over DS2 records")
    print(df[df.protocol == proto].groupby("filter")[["prd_nom", "prd_opt", "r_timing_err_ms", "r_jitter_ms",
                                                      "r_amp_err_pct", "st_err_uV"]].median().loc[ORDER].round(3))
