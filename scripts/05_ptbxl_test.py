"""Stage 5: independent secondary validation on PTB-XL (500 Hz, lead II, recommended test fold 10).

Nothing is tuned here: the physical specification (40/60 Hz, 0.5 dB, 40 dB, 20 ms) and the
hyper-parameter *rules* locked on MIT-BIH DS1 (N = N_min + slack, K) are re-applied at
fs = 500 Hz, and the morphology weight is the MIT-BIH DS1 PSD (Hz axis) - i.e. a
cross-dataset, cross-sampling-rate transfer test.
Outputs: results/bank_ecg500.json, results/freq_metrics_ecg500.json, results/ptbxl.csv
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from lowdelay import signals as sg
from lowdelay.analysis import freq_metrics, psd_on_grid
from lowdelay.bank import ORDER, bank_table, build_bank, evaluate_record, load_bank, save_bank
from lowdelay.data import detect_r_peaks, load_ptbxl, peak_polarity, ptbxl_records, remove_baseline
from lowdelay.specs import load_spec
from lowdelay.util import CONFIG, RESULTS, load_json, load_params, save_json

P = load_params()
hp = load_json("hparams.json")
spec = load_spec(CONFIG / "spec_ecg.json").with_(fs=500.0, name="ECG low-pass (PTB-XL, 500 Hz)")
print(spec.describe())
z = np.load(RESULTS / "psd_mitdb_ds1.npz")
S = psd_on_grid(spec, z["f"], z["pxx"], P["psd_floor"])
path = RESULTS / "bank_ecg500.json"
if path.exists():
    bank = load_bank(path)
else:
    bank = build_bank(spec, S, hp)
    save_bank(bank, path)
rows_f = bank_table(bank, spec, S)
save_json({"spec": spec.__dict__, "rows": rows_f}, "freq_metrics_ecg500.json")
fm = {r["key"]: r for r in rows_f}
for r in rows_f:
    print(f"  {r['key']:12s} {r['name']:42s} lat={r['tau_K_ms']:6.2f} ms PRDpred={r['prd_pred']:5.2f}% "
          f"mult={r['ops_mult']} spec={r['meets_spec']}")

rows = []
t0 = time.time()
recs = ptbxl_records()
for i, row in enumerate(recs):
    d = load_ptbxl(row)
    x = remove_baseline(d["x"], d["fs"])
    if not np.all(np.isfinite(x)) or np.std(x) < 1e-3:
        continue
    r = sg.zero_phase_reference(x, spec)
    pk = detect_r_peaks(r, d["fs"])
    pol = peak_polarity(r, pk)
    res = evaluate_record(bank, fm, x, r, pk, d["fs"], spec, polarity=pol)
    scp = json.dumps(row.get("scp_codes", ""))
    for k, m in res.items():
        rows.append({"record": d["id"], "filter": k, "scp": scp, **m})
    if i % 25 == 0:
        print(f"[{time.time()-t0:5.0f}s] {i+1}/{len(recs)}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "ptbxl.csv", index=False)
print(f"\n== PTB-XL ({df.record.nunique()} records): median")
print(df.groupby("filter")[["prd_nom", "prd_opt", "r_timing_err_ms", "r_jitter_ms", "r_amp_err_pct",
                            "st_err_uV"]].median().loc[ORDER].round(3))
