"""Stage 8: all figures of the report (figures/*.png). Each figure explains a DSP mechanism."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lowdelay import signals as sg
from lowdelay.analysis import check_group_delay, freq_metrics, gd_kernel, psd_on_grid
from lowdelay.bank import ORDER, PROPOSED, load_bank
from lowdelay.data import load_mitdb, remove_baseline
from lowdelay.metrics import step_metrics
from lowdelay.specs import ideal_reference, load_spec
from lowdelay.util import COLORS, CONFIG, MARKERS, RESULTS, load_params, plot_style, savefig

plot_style()
P = load_params()
spec = load_spec(CONFIG / "spec_ecg.json")
bank = load_bank(RESULTS / "bank_ecg360.json")
z = np.load(RESULTS / "psd_mitdb_ds1.npz")
S = psd_on_grid(spec, z["f"], z["pxx"], P["psd_floor"])
w, pb, tb, sb = spec.grid()
fhz = w * spec.fs / (2 * np.pi)
fm = {k: freq_metrics(f, spec, S) for k, f in bank.items()}
LAB = {k: f"{k}{' (proposed)' if k in PROPOSED else ''}" for k in ORDER}


def lw(k):
    return 2.0 if k in PROPOSED else 1.2


# ------------------------------------------------------------------ F1 magnitude
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
wd = np.linspace(0, np.pi, 4096)
fd = wd * spec.fs / (2 * np.pi)
for k in ORDER:
    H = np.abs(bank[k].freq(wd))
    ax[0].plot(fd, 20 * np.log10(np.maximum(H, 1e-8)), color=COLORS[k], lw=lw(k), label=LAB[k])
    m = fd <= spec.fp_hz + 3
    ax[1].plot(fd[m], 20 * np.log10(H[m]), color=COLORS[k], lw=lw(k))
for a in ax:
    a.axvspan(spec.fp_hz, spec.fst_hz, color="0.85", alpha=0.5, lw=0)
ax[0].hlines(-spec.as_db, spec.fst_hz, spec.fs / 2, color="k", ls="--", lw=1)
ax[0].set(ylim=(-90, 5), xlabel="Frequency (Hz)", ylabel="|H| (dB)", title="(a) Magnitude, matched specification")
dp = 20 * np.log10(1 + spec.delta_p), 20 * np.log10(1 - spec.delta_p)
ax[1].hlines(dp, 0, spec.fp_hz, color="k", ls="--", lw=1)
ax[1].set(ylim=(-0.6, 0.6), xlabel="Frequency (Hz)", ylabel="|H| (dB)", title="(b) Passband detail (0.5 dB p-p mask)")
ax[0].legend(ncol=2, loc="lower left")
savefig(fig, "F01_magnitude")

# ------------------------------------------------------------------ F2 phase / GD
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
m = w <= spec.ws
for k in ORDER:
    H = bank[k].freq(w[m])
    ph = np.unwrap(np.angle(H))
    ax[0].plot(fhz[m], ph, color=COLORS[k], lw=lw(k), label=LAB[k])
    pe = ph + w[m] * fm[k]["tau_K"]
    ax[1].plot(fhz[m], np.degrees(pe), color=COLORS[k], lw=lw(k))
    ax[2].plot(fhz[pb], spec.ms(bank[k].group_delay(w[pb])), color=COLORS[k], lw=lw(k))
for a in ax[:2]:
    a.axvspan(spec.fp_hz, spec.fst_hz, color="0.85", alpha=0.5, lw=0)
ax[0].set(xlabel="Frequency (Hz)", ylabel="Unwrapped phase (rad)", title="(a) Unwrapped phase")
ax[1].set(xlabel="Frequency (Hz)", ylabel="Phase error (deg)", ylim=(-60, 60),
          title=r"(b) Deviation from linear phase $\phi(\omega)+\omega\tau_K$")
ax[2].axhline(spec.latency_ms, color="k", ls="--", lw=1)
ax[2].text(1, spec.latency_ms + 2, "latency budget 20 ms", fontsize=7.5)
ax[2].set(xlabel="Frequency (Hz)", ylabel="Group delay (ms)", ylim=(0, 100),
          title="(c) Passband group delay (stopband omitted)")
ax[0].legend(ncol=1, fontsize=6.5)
savefig(fig, "F02_phase_groupdelay")

# ------------------------------------------------------------------ F3 GD validation + kernel
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ww, ph, gnum, gana, err = check_group_delay(bank["Elliptic"], spec)
ax[0].plot(ww * spec.fs / 2 / np.pi, gana, color=COLORS["Elliptic"], lw=2.5, label="analytic Re{-z dH/dz / H}")
ax[0].plot(ww * spec.fs / 2 / np.pi, gnum, "k--", lw=1, label=r"$-d(\mathrm{unwrap}\,\phi)/d\omega$")
ax[0].set(xlabel="Frequency (Hz)", ylabel="Group delay (samples)",
          title=f"(a) GD check, elliptic IIR (max |diff| = {err:.1e} samples)")
ax[0].legend()
K = gd_kernel(spec, S)
ax[1].plot(fhz, S / S.max(), color=COLORS["LP-FIR"], lw=1.5, label="ECG PSD S(ω) (MIT-BIH DS1)")
ax[1].plot(fhz[pb], K / K.max(), color=COLORS["MO-LDFIR"], lw=2, label="GD kernel K(ν) = ∫ω S dω")
ax[1].axvspan(spec.fp_hz, spec.fst_hz, color="0.85", alpha=0.5, lw=0)
ax[1].set(xlim=(0, 80), xlabel="Frequency (Hz)", ylabel="normalised",
          title="(b) Signal-energy weight and derived GD kernel")
ax[1].legend()
savefig(fig, "F03_gdcheck_kernel")

# ------------------------------------------------------------------ F4 pole-zero
sel = ["LP-FIR", "MP-FIR", "MO-LDFIR", "Elliptic", "Ell+AP-flat", "EW-APE"]
fig, ax = plt.subplots(2, 3, figsize=(10, 6.6))
t = np.linspace(0, 2 * np.pi, 400)
for a, k in zip(ax.ravel(), sel):
    zz, pp = bank[k].poles_zeros()
    a.plot(np.cos(t), np.sin(t), "k", lw=0.6)
    a.plot(zz.real, zz.imag, "o", mfc="none", mec=COLORS[k], ms=5, label="zeros")
    if len(pp):
        a.plot(pp.real, pp.imag, "x", color="k", ms=6, label="poles")
    for ang in (spec.wp, spec.ws):
        a.plot([0, 1.15 * np.cos(ang)], [0, 1.15 * np.sin(ang)], color="0.6", lw=0.6, ls=":")
    lim = 1.6 if k not in ("MO-LDFIR", "LD-FIR-flat") else 2.2
    far = zz[np.abs(zz) > lim]
    groups = {}
    for zf in far:   # zeros outside the view are pointed to (merged if repeated), not plotted off-scale
        key = (round(zf.real, 1), round(abs(zf.imag), 1))
        groups[key] = groups.get(key, 0) + 1
    for (re_, im_), cnt in groups.items():
        xe = np.sign(re_) * lim * 0.97
        val = f"{re_:.1f}" if im_ == 0 else f"{re_:.1f} ± {im_:.1f}j"
        a.annotate(f"{cnt} zeros at {val}" if cnt > 1 else f"zero at {val}",
                   xy=(xe, 0), xytext=(xe * 0.35, 0.55 * lim), fontsize=7,
                   arrowprops=dict(arrowstyle="->", lw=0.7))
    a.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
          title=f"{k}  (max |p| = {bank[k].max_pole_radius():.3f})" if len(pp) else k)
    a.legend(loc="lower left", fontsize=6.5)
savefig(fig, "F04_polezero")

# ------------------------------------------------------------------ F5 impulse/step
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
n = int(0.2 * spec.fs)
tt = np.arange(n) * 1000 / spec.fs
for k in ORDER:
    ax[0].plot(tt, bank[k].impulse(n), color=COLORS[k], lw=lw(k), label=LAB[k])
    s = step_metrics(bank[k], spec.fs, n)["response"]
    ax[1].plot(tt, s, color=COLORS[k], lw=lw(k))
ax[0].set(xlabel="Time (ms)", ylabel="h[n]", title="(a) Impulse responses (coefficient patterns)")
ax[1].axvline(spec.latency_ms, color="k", ls="--", lw=1)
ax[1].set(xlabel="Time (ms)", ylabel="Step response", title="(b) Step responses (overshoot / settling)")
ax[0].legend(fontsize=6.5)
savefig(fig, "F05_impulse_step")

# ------------------------------------------------------------------ F6 Pareto
par = json.load(open(RESULTS / "pareto.json"))["points"]
pdf = pd.DataFrame([p for p in par if "error" not in p])
pdf = pdf[pdf.meets_spec == True]
archs = ["LP-FIR", "MP-FIR", "Elliptic", "Butterworth", "Chebyshev-I", "Ell+AP-flat", "LD-FIR-flat", "EW-APE",
         "MO-LDFIR"]
mk = dict(MARKERS, **{"Chebyshev-I": "<"})


def pareto_front(df, x, y):
    df = df.sort_values([x, y])
    best, keep = np.inf, []
    for i, r in df.iterrows():
        if r[y] < best:
            best = r[y]
            keep.append(i)
    return df.loc[keep]


for kind, vals, fname, ttl in (("atten", [30, 40, 50, 60], "F06_pareto_attenuation", "As"),
                               ("trans", [5, 10, 40], "F07_pareto_transition", "TW")):
    vv = [v for v in vals if ((pdf.variant_kind == kind) & (pdf.variant_val == v)).any()]
    if kind == "trans":
        vv = sorted(vv + [20])
    fig, ax = plt.subplots(1, len(vv), figsize=(3.3 * len(vv), 3.4), sharey=True)
    for a, v in zip(np.atleast_1d(ax), vv):
        sub = pdf[(pdf.variant_kind == kind) & (pdf.variant_val == v)] if not (kind == "trans" and v == 20) else \
            pdf[(pdf.variant_kind == "atten") & (pdf.variant_val == 40)]
        for ar in archs:
            s = sub[sub.arch == ar]
            if s.empty:
                continue
            a.scatter(s.tau_K_ms, s.prd_pred, s=28 if ar in PROPOSED else 18, marker=mk[ar], color=COLORS[ar],
                      label=ar, zorder=3, edgecolors="white", linewidths=0.4)
            if ar in ("MO-LDFIR", "EW-APE", "Ell+AP-flat", "LD-FIR-flat"):
                f_ = pareto_front(s, "tau_K_ms", "prd_pred")
                a.plot(f_.tau_K_ms, f_.prd_pred, color=COLORS[ar], lw=1)
        a.axvline(spec.latency_ms, color="k", ls="--", lw=0.8)
        a.set(xscale="log", yscale="log", xlabel="Latency τ_K (ms)",
              title=f"{ttl} = {v} {'dB' if kind == 'atten' else 'Hz'}")
    np.atleast_1d(ax)[0].set_ylabel("Predicted morphology PRD (%)")
    np.atleast_1d(ax)[-1].legend(fontsize=6.3, loc="upper right")
    savefig(fig, fname)

# GD variation vs latency vs attenuation (the 3-way trade-off)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for ar in ["MO-LDFIR", "EW-APE", "Ell+AP-flat", "Elliptic", "MP-FIR", "LP-FIR"]:
    xs, ys, ms = [], [], []
    for a_ in [30, 40, 50, 60]:
        s = pdf[(pdf.variant_kind == "atten") & (pdf.variant_val == a_) & (pdf.arch == ar)]
        s = s[s.tau_K_ms <= spec.latency_ms * 1.005]
        if len(s):
            b = s.loc[s.prd_pred.idxmin()]
            xs.append(a_), ys.append(b.prd_pred), ms.append(b.dtau_max_ms)
    if xs:
        ax[0].plot(xs, ys, marker=mk[ar], color=COLORS[ar], lw=lw(ar), label=ar)
        ax[1].plot(xs, ms, marker=mk[ar], color=COLORS[ar], lw=lw(ar), label=ar)
ax[0].set(xlabel="Stopband attenuation As (dB)", ylabel="Best predicted PRD (%)", yscale="log",
          title="(a) Morphology error under the 20-ms budget")
ax[1].set(xlabel="Stopband attenuation As (dB)", ylabel="max passband |Δτ| (ms)", yscale="log",
          title="(b) Group-delay variation of the same designs")
ax[0].legend(fontsize=7)
savefig(fig, "F08_attenuation_tradeoff")

# ------------------------------------------------------------------ F9 budget sweep
b = pd.read_csv(RESULTS / "robust_budget_sweep.csv")
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for ar in ["LP-FIR", "MP-FIR", "Elliptic", "EW-APE", "MO-LDFIR"]:
    if ar in b:
        s = b[["latency_ms", ar]].dropna()
        ax.plot(s.latency_ms, s[ar], marker=MARKERS[ar], color=COLORS[ar], lw=lw(ar), label=ar)
ax.axvline(spec.latency_ms, color="k", ls="--", lw=0.8)
ax.set(xscale="log", yscale="log", xlabel="Latency budget (ms)", ylabel="Best predicted PRD (%)",
       title="Best achievable morphology error vs latency budget")
ax.legend()
savefig(fig, "F09_budget_sweep")

# ------------------------------------------------------------------ F10 synthetic sweeps / model validation
q = pd.read_csv(RESULTS / "synthetic_qrs_sweep.csv")
val = pd.DataFrame(json.load(open(RESULTS / "synthetic_model_validation.json")))
ds2 = pd.read_csv(RESULTS / "mitdb_ds2.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for k in ORDER:
    s = q[q["filter"] == k].groupby("qrs").prd_opt.median()
    ax[0].plot(s.index, s.values, marker=MARKERS[k], color=COLORS[k], lw=lw(k), label=LAB[k], ms=4)
ax[0].set(yscale="log", xlabel="QRS width scale (1 = σ_R 10 ms)", ylabel="PRD_opt (%) (noise-free)",
          title="(a) Synthetic ECG: sensitivity to QRS sharpness")
ax[0].legend(fontsize=6.3, ncol=2)
meas = ds2[ds2.protocol == "clean"].groupby("filter").prd_opt.median()
for k in ORDER:
    ax[1].scatter(fm[k]["prd_pred"], meas[k], marker=MARKERS[k], color=COLORS[k], s=40, label=k,
                  edgecolors="white", linewidths=0.4)
    v = val[val["filter"] == k].iloc[0]
    ax[1].scatter(v.prd_pred_synPSD, v.prd_meas_median, marker=MARKERS[k], facecolors="none",
                  edgecolors=COLORS[k], s=40)
lim = [0.8, 15]
ax[1].plot(lim, lim, "k:", lw=1)
ax[1].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Predicted PRD (%) (frequency-domain model)",
          ylabel="Measured median PRD_opt (%)", title="(b) Error-model validation (filled: MIT DS2, open: synthetic)")
savefig(fig, "F10_synthetic_validation")

# ------------------------------------------------------------------ F11 DS2 / PTB-XL boxplots
sets = [("MIT-BIH DS2", ds2[ds2.protocol == "clean"])]
if (RESULTS / "ptbxl.csv").exists():
    sets.append(("PTB-XL", pd.read_csv(RESULTS / "ptbxl.csv")))
mets = [("prd_opt", "PRD_opt (%)"), ("r_timing_err_ms", "|R-peak timing error| (ms)"), ("st_err_uV", "|ST-level error| (µV)")]
fig, ax = plt.subplots(len(sets), 3, figsize=(12, 3.3 * len(sets)), squeeze=False)
for i, (nm, df) in enumerate(sets):
    for j, (mc, yl) in enumerate(mets):
        data = [df[df["filter"] == k][mc].dropna().values for k in ORDER]
        bp = ax[i, j].boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, k in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COLORS[k])
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("k")
        ax[i, j].set_xticks(range(1, len(ORDER) + 1), ORDER, rotation=45, ha="right", fontsize=7)
        ax[i, j].set(ylabel=yl, title=f"{nm}: {yl}")
        if mc == "prd_opt":
            ax[i, j].set_yscale("log")
savefig(fig, "F11_dataset_boxplots")

# ------------------------------------------------------------------ F12 waveform examples
fig, ax = plt.subplots(2, 2, figsize=(11, 6))
for row, (rec, sym) in enumerate((("100", "N"), ("200", "V"))):
    d = load_mitdb(rec, seconds=300)
    x = remove_baseline(d["x"], d["fs"])
    r = sg.zero_phase_reference(x, spec)
    idx = [b for b, s_ in zip(d["beats"], d["symbols"]) if s_ == sym and b > 2 * spec.fs][0]
    lo, hi = idx - int(0.25 * spec.fs), idx + int(0.45 * spec.fs)
    tt = (np.arange(lo, hi) - idx) * 1000 / spec.fs
    ax[row, 0].plot(tt, r[lo:hi], color=COLORS["Reference"], lw=2.4, label="zero-phase reference")
    for k in ["MP-FIR", "Elliptic", "Ell+AP-flat", "LP-FIR", "EW-APE", "MO-LDFIR"]:
        y = bank[k].apply(x)
        D = fm[k]["tau_K"]
        yy = np.interp(np.arange(lo, hi) + D, np.arange(len(y)), y)   # remove nominal latency
        ax[row, 0].plot(tt, yy, color=COLORS[k], lw=lw(k), label=f"{k} (−{fm[k]['tau_K_ms']:.1f} ms)")
        ax[row, 1].plot(tt, (yy - r[lo:hi]) * 1000, color=COLORS[k], lw=lw(k), label=k)
    ax[row, 0].set(xlabel="Time from R peak (ms)", ylabel="mV",
                   title=f"Record {rec}, {'normal' if sym == 'N' else 'ventricular'} beat, latency-compensated")
    ax[row, 1].set(xlabel="Time from R peak (ms)", ylabel="error (µV)", title="Output − reference")
    ax[row, 0].legend(fontsize=6.3)
savefig(fig, "F12_waveforms")

# ------------------------------------------------------------------ F13 quantisation
qd = pd.read_csv(RESULTS / "robust_quantisation.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
for k in ORDER:
    s = qd[(qd["filter"] == k) & (qd.bits < 64)].sort_values("bits")
    ax[0].plot(s.bits, s.atten_db, marker=MARKERS[k], color=COLORS[k], lw=lw(k), label=k)
    ax[1].plot(s.bits, s.prd_pred, marker=MARKERS[k], color=COLORS[k], lw=lw(k), label=k)
ax[0].axhline(spec.as_db, color="k", ls="--", lw=0.8)
ax[0].set(xlabel="Coefficient word length (bits)", ylabel="Stopband attenuation (dB)",
          title="(a) Coefficient quantisation: attenuation")
ax[1].set(xlabel="Coefficient word length (bits)", ylabel="Predicted PRD (%)", yscale="log",
          title="(b) Coefficient quantisation: morphology error")
ax[0].legend(fontsize=6.5)
savefig(fig, "F14_quantisation")

# ------------------------------------------------------------------ F14 PPG
spec_p = load_spec(CONFIG / "spec_ppg.json")
bank_p = load_bank(RESULTS / "bank_ppg125.json")
zp = np.load(RESULTS / "psd_ppg_dev.npz")
Sp = psd_on_grid(spec_p, zp["f"], zp["pxx"], P["psd_floor"])
fmp = {k: freq_metrics(f, spec_p, Sp) for k, f in bank_p.items()}
x, pk, _ = sg.synthetic_ppg(spec_p.fs, 12, 72, 0.8, rng=99_999)
r = sg.zero_phase_reference(x, spec_p)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
lo, hi = int(4 * spec_p.fs), int(6.5 * spec_p.fs)
tt = np.arange(lo, hi) / spec_p.fs
ax[0].plot(tt, r[lo:hi], color="k", lw=2.4, label="reference")
wp_ = np.linspace(0, spec_p.wp, 400)
for k in ["LP-FIR", "MP-FIR", "Elliptic", "EW-APE", "MO-LDFIR"]:
    y = bank_p[k].apply(x)
    D = fmp[k]["tau_K"]
    ax[0].plot(tt, np.interp(np.arange(lo, hi) + D, np.arange(len(y)), y), color=COLORS[k], lw=lw(k),
               label=f"{k} (−{fmp[k]['tau_K_ms']:.0f} ms)")
    ax[1].plot(wp_ * spec_p.fs / 2 / np.pi, spec_p.ms(bank_p[k].group_delay(wp_)), color=COLORS[k], lw=lw(k), label=k)
ax[1].axhline(spec_p.latency_ms, color="k", ls="--", lw=0.8)
ax[0].set(xlabel="Time (s)", ylabel="a.u.", title="(a) Synthetic PPG, latency-compensated outputs")
ax[1].set(xlabel="Frequency (Hz)", ylabel="Group delay (ms)", ylim=(0, 400), title="(b) PPG-spec passband group delay")
ax[0].legend(fontsize=6.5)
savefig(fig, "F13_ppg")
print("figures written")
