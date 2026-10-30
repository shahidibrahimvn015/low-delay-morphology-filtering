"""Stage 7: statistics. Paired, record-wise (or trial-wise) comparisons of MO-LDFIR against every
other filter: median [IQR], median paired difference with a bootstrap 95 % CI (resampling
records), Wilcoxon signed-rank test, Holm correction within each (dataset, metric) family.
Outputs: results/stats_summary.csv, results/stats_tests.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lowdelay.bank import ORDER
from lowdelay.util import RESULTS, load_params

P = load_params()
rng = np.random.default_rng(P["seed"])
METRICS = ["prd_nom", "prd_opt", "r_timing_err_ms", "r_jitter_ms", "r_amp_err_pct", "st_err_uV"]

sets = {}
d = pd.read_csv(RESULTS / "mitdb_ds2.csv")
sets["MIT-BIH DS2 (clean)"] = d[d.protocol == "clean"]
sets["MIT-BIH DS2 (stress)"] = d[d.protocol == "stress"]
if (RESULTS / "ptbxl.csv").exists():
    sets["PTB-XL fold 10"] = pd.read_csv(RESULTS / "ptbxl.csv")
mc = pd.read_csv(RESULTS / "synthetic_mc.csv").rename(
    columns={"seed": "record", "timing_err_ms": "r_timing_err_ms", "peak_amp_err_pct": "r_amp_err_pct"})
sets["Synthetic ECG MC"] = mc
if (RESULTS / "synthetic_ppg_mc.csv").exists():
    sets["Synthetic PPG MC"] = pd.read_csv(RESULTS / "synthetic_ppg_mc.csv").rename(
        columns={"seed": "record", "timing_err_ms": "r_timing_err_ms", "peak_amp_err_pct": "r_amp_err_pct"})


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


summ, tests = [], []
for name, df in sets.items():
    for met in METRICS:
        if met not in df.columns:
            continue
        piv = df.pivot_table(index="record", columns="filter", values=met)
        for k in ORDER:
            if k in piv:
                v = piv[k].dropna()
                summ.append({"dataset": name, "metric": met, "filter": k, "n": len(v), "median": v.median(),
                             "q1": v.quantile(0.25), "q3": v.quantile(0.75), "mean": v.mean()})
        fam = []
        for k in ORDER:
            if k == "MO-LDFIR" or k not in piv:
                continue
            pair = piv[["MO-LDFIR", k]].dropna()
            diff = (pair["MO-LDFIR"] - pair[k]).to_numpy()
            if len(diff) < 5 or np.allclose(diff, 0):
                continue
            boots = [np.median(rng.choice(diff, len(diff))) for _ in range(2000)]
            p = wilcoxon(pair["MO-LDFIR"], pair[k]).pvalue
            fam.append({"dataset": name, "metric": met, "comparator": k, "n": len(diff),
                        "median_diff": float(np.median(diff)), "ci_lo": float(np.percentile(boots, 2.5)),
                        "ci_hi": float(np.percentile(boots, 97.5)),
                        "frac_records_better": float(np.mean(diff < 0)), "p": float(p)})
        if fam:
            for f_, pa in zip(fam, holm([f["p"] for f in fam])):
                f_["p_holm"] = float(pa)
            tests += fam

pd.DataFrame(summ).to_csv(RESULTS / "stats_summary.csv", index=False)
t = pd.DataFrame(tests)
t.to_csv(RESULTS / "stats_tests.csv", index=False)
pd.set_option("display.width", 200)
print(t[t.metric.isin(["prd_opt", "prd_nom", "st_err_uV"])].round(4).to_string(index=False))
