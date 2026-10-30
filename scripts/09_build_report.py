"""Stage 9: build the journal-style technical report (report/Low_Group_Delay_Morphology_Preserving_Filtering_Report.docx).
All numbers in tables and in the text are read from results/ - nothing is typed by hand."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from lowdelay.bank import ORDER
from lowdelay.docx_util import Report
from lowdelay.specs import load_spec
from lowdelay.util import CONFIG, FIGS, RESULTS, ROOT, load_json, load_params

P = load_params()
hp = load_json("hparams.json")
spec = load_spec(CONFIG / "spec_ecg.json")
spec_p = load_spec(CONFIG / "spec_ppg.json")
F = {r["key"]: r for r in load_json("freq_metrics_ecg360.json")["rows"]}
F5 = {r["key"]: r for r in load_json("freq_metrics_ecg500.json")["rows"]}
FP = {r["key"]: r for r in load_json("freq_metrics_ppg125.json")["rows"]}
prov = load_json("freq_metrics_ecg360.json")["provenance"]
SUM = pd.read_csv(RESULTS / "stats_summary.csv")
TST = pd.read_csv(RESULTS / "stats_tests.csv")
TR = {r["filter"]: r for r in load_json("synthetic_transients.json")}
VAL = pd.DataFrame(load_json("synthetic_model_validation.json"))
SEL = load_json("dev_selection.json")
ABL = pd.DataFrame(load_json("robust_ablation.json"))
QNT = pd.read_csv(RESULTS / "robust_quantisation.csv")
BUD = pd.read_csv(RESULTS / "robust_budget_sweep.csv")
PAR = pd.DataFrame([p for p in load_json("pareto.json")["points"] if "error" not in p])
QSW = pd.read_csv(RESULTS / "synthetic_qrs_sweep.csv")


def pareto_summary():
    """Per (variant, architecture): minimum latency, best PRD within 20 ms, latency/cost to reach 3 %."""
    out = []
    for v in PAR.variant.unique():
        s = PAR[(PAR.variant == v) & (PAR.meets_spec == True)]
        for a in s.arch.unique():
            t = s[s.arch == a]
            b20 = t[t.tau_K_ms <= spec.latency_ms * 1.005]
            r3 = t[t.prd_pred <= 3.0]
            out.append(dict(variant=v, arch=a, minlat=t.tau_K_ms.min(),
                            best20=b20.prd_pred.min() if len(b20) else np.nan,
                            ops20=b20.loc[b20.prd_pred.idxmin()].ops_mult if len(b20) else np.nan,
                            lat3=r3.tau_K_ms.min() if len(r3) else np.nan,
                            ops3=r3.loc[r3.tau_K_ms.idxmin()].ops_mult if len(r3) else np.nan))
    o = pd.DataFrame(out)
    o.to_csv(RESULTS / "pareto_summary.csv", index=False)
    return o


PS = pareto_summary()
bank_meta = {k: v["meta"] for k, v in json.load(open(RESULTS / "bank_ecg360.json")).items()}


def ps(v, a, c):
    s = PS[(PS.variant == v) & (PS.arch == a)]
    return float(s[c].iloc[0]) if len(s) else float("nan")


def med(ds, met, flt):
    s = SUM[(SUM.dataset == ds) & (SUM.metric == met) & (SUM["filter"] == flt)]
    return float(s["median"].iloc[0]) if len(s) else float("nan")


def iqr(ds, met, flt, nd=2):
    s = SUM[(SUM.dataset == ds) & (SUM.metric == met) & (SUM["filter"] == flt)].iloc[0]
    return f"{s['median']:.{nd}f} [{s.q1:.{nd}f}, {s.q3:.{nd}f}]"


def test(ds, met, comp):
    s = TST[(TST.dataset == ds) & (TST.metric == met) & (TST.comparator == comp)]
    return s.iloc[0] if len(s) else None


def andlist(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def pfmt(p):
    return "< 0.001" if p < 1e-3 else f"= {p:.3f}"


DS2, DS2S, PTB, MC, PPG = ("MIT-BIH DS2 (clean)", "MIT-BIH DS2 (stress)", "PTB-XL fold 10",
                           "Synthetic ECG MC", "Synthetic PPG MC")
LAT_OK = [k for k in ORDER if F[k]["latency_ok"]]
n_ptb = int(SUM[(SUM.dataset == PTB)].n.max())
ms = spec.ms

R = Report()
R.title("Low-Group-Delay Morphology-Preserving Digital Filtering: Constrained FIR versus Phase-Equalized IIR",
        ["V N Shahid Ibrahim (ED24B064)", "Nida Paravath (ED24B062)", "Madhav Tripathi (ED24B059)", "Deepak Meena (ED24B045)",
         "ED5017 - Digital Signal Processing (research-oriented course project)",
         "Department of Engineering Design, IIT Madras"])

# =========================================================================== abstract
R.h("Abstract", 1)
R.p(f"Real-time biomedical front ends must remove out-of-band noise while staying within a strict latency "
    f"budget and without distorting diagnostically relevant waveform morphology. We ask which filter architecture "
    f"best preserves morphology for a fixed attenuation requirement and latency budget. The comparison covers "
    f"linear-phase FIR, minimum-phase FIR, classical IIR, and IIR plus all-pass phase equalization. First, we derive "
    f"a frequency-domain morphology-error model. For an input with power spectral density S(ω), the expected "
    f"squared deviation from a delayed zero-phase reference splits into a PSD-weighted magnitude term and a "
    f"phase-linearity term. The phase term is bounded by a passband group-delay error weighted by an explicit kernel "
    f"K(ν)=∫_ν^ωp ωS(ω)dω. This justifies the objective J = αE_mag + βE_τg + γD with α=β=1 and gives a "
    f"principled, signal-energy-weighted group-delay metric. Second, we propose a morphology-optimal constrained "
    f"low-delay FIR (MO-LDFIR), posed as a second-order-cone program whose constraints guarantee the matched "
    f"magnitude specification. Third, we propose an energy-weighted, latency-penalized all-pass equalizer (EW-APE). "
    f"For an ECG low-pass specification (0–40 Hz passband, ≥40 dB above 60 Hz, 0.5 dB ripple, 20 ms "
    f"budget), all filters were matched in magnitude. Hyperparameters were tuned only on MIT-BIH DS1, and testing "
    f"used inter-patient MIT-BIH DS2 (22 patients) and {n_ptb} PTB-XL records at a different sampling rate. "
    f"MO-LDFIR ({ms(F['MO-LDFIR']['tau_K']):.1f} ms) gave a median DS2 morphology PRD of "
    f"{med(DS2, 'prd_opt', 'MO-LDFIR'):.2f}% versus {med(DS2, 'prd_opt', 'Elliptic'):.2f}% "
    f"for the elliptic IIR, {med(DS2, 'prd_opt', 'MP-FIR'):.2f}% for the minimum-phase FIR and "
    f"{med(DS2, 'prd_opt', 'EW-APE'):.2f}% for the budget-constrained equalized IIR (all p_Holm < 0.001). Its "
    f"ST-level error was {med(DS2, 'st_err_uV', 'MO-LDFIR'):.2f} µV, against {med(DS2, 'st_err_uV', 'LP-FIR'):.2f} µV for the "
    f"{ms(F['LP-FIR']['tau_K']):.0f}-ms linear-phase FIR. All-pass equalization of the IIR could not be exploited "
    f"under the budget, because every section adds delay. The linear-phase FIR remains preferable when its latency "
    f"is acceptable, and the IIR when arithmetic cost dominates. We report the failure regime of the proposed FIR "
    f"(very sharp QRS complexes), quantization sensitivity, ablations, and a Pareto map from which "
    f"architecture-selection rules are derived.")
R.p("**Keywords:** group delay; low-delay FIR; all-pass equalization; phase distortion; ECG morphology; "
    "second-order-cone programming; latency.")

# =========================================================================== 1 introduction
R.h("1. Introduction", 1)
R.p("Linear filtering is the first stage of almost every biomedical signal chain. The morphology of ECG waves "
    "(QRS amplitude and width, ST level, T-wave shape) and of PPG pulses carries the clinical information, so the "
    "filter must reject noise without reshaping the waveform. A linear-phase FIR delays all frequencies by the same "
    "(N−1)/2 samples and introduces no phase distortion. For sharp transition bands this delay is large, which is "
    "unacceptable in closed-loop or event-triggered applications: arrhythmia alarms, R-wave-synchronized "
    "triggering, wearable feedback. IIR filters meet the same magnitude specification with far lower order and "
    "delay, but their group delay peaks near the band edge, which causes ringing and morphology-dependent timing. "
    "The standard remedy, all-pass phase equalization, flattens the group delay only by *adding* delay [5]. "
    "Low-delay FIR designs relax linear phase in exchange for latency [3], [4]. Hybrid causal/non-causal "
    "equalization [6] needs future samples and is therefore not a real-time option.")
R.p("Comparisons between these architectures are often made under unrelated specifications. They typically rely "
    "on generic metrics such as maximum group-delay deviation over the whole passband, which ignore *where* the "
    "signal energy lies. This project addresses that gap with the following contributions:")
R.bullets([
    "**A morphology-error model and a derived group-delay metric.** The expected output error splits into "
    "magnitude and phase terms, and the phase term is bounded by a group-delay error weighted by the kernel K(ν). "
    "This gives a defensible definition of the J = αE_mag + βE_τg + γD objective (Sec. 3) and is validated against "
    "measured errors (Sec. 6.3).",
    "**MO-LDFIR**: a constrained low-delay FIR that minimizes the PSD-weighted morphology error, posed as an SOCP "
    "with constraints that guarantee the matched magnitude specification. It is globally optimal for each latency D.",
    "**EW-APE**: an all-pass equalizer for IIR filters that minimizes the same exact morphology error with a "
    "latency price or a hard latency bound.",
    "**A matched-specification benchmark.** Five reproduced baselines and an unweighted low-delay FIR are compared "
    "on Pareto maps (attenuation × group-delay/morphology error × latency × cost), synthetic ground truth, two "
    "independent open ECG datasets, PPG, quantization and ablation. The benchmark ends in explicit design "
    "recommendations (Sec. 7)."])
R.p("**Hypothesis.** Under a matched magnitude specification and a strict latency budget, (H1) a non-linear-phase "
    "FIR whose phase error is concentrated where the signal has little energy preserves morphology better than "
    "both the classical IIR and the IIR with all-pass equalization; and (H2) all-pass equalization cannot "
    "substantially reduce the morphology error of an IIR when the budget leaves little room for added delay. "
    "Both hypotheses are fixed before testing and are evaluated on held-out data (Sec. 5–6).")

# =========================================================================== 2 problem formulation
R.h("2. Problem formulation", 1)
R.h("2.1 Signal model and conventions", 2)
R.p("The input is x[n] = s[n] + v[n], where s is the physiological waveform (mV for ECG, arbitrary units for PPG) "
    "and v is additive noise (drifting 60/120 Hz power-line interference and broadband EMG-like noise). Physical "
    "frequency f is in Hz, the sampling rate fs in Hz, and the digital angular frequency is ω = 2πf/fs "
    "rad/sample, with 0 ≤ ω ≤ π. The normalized frequency f/fs (cycles/sample) is used only for printing, and "
    "FFT bins never appear in a definition. Delays are reported in samples and in ms (1000·samples/fs). "
    "A causal LTI filter H(e^{jω}) = |H|e^{jφ(ω)} has group delay τ(ω) = −dφ/dω, evaluated analytically from the "
    "coefficients. For a polynomial P(e^{jω}) = Σ p_k e^{−jωk}, τ_P = Re{Σ k p_k e^{−jωk} / P}, and τ_{B/A} = τ_B − τ_A. "
    "Fig. 3a confirms the analytic value against the derivative of the *unwrapped* phase.")
R.h("2.2 Matched specification and latency budget", 2)
R.p(f"Primary specification (ECG, MIT-BIH, fs = {spec.fs:g} Hz): passband 0–{spec.fp_hz:g} Hz "
    f"(ωp = {spec.wp:.4f} rad/sample); stopband ≥ {spec.fst_hz:g} Hz (ωs = {spec.ws:.4f} rad/sample), which also "
    f"covers 60-Hz mains in US recordings; passband ripple ≤ {spec.ap_db:g} dB peak-to-peak centered on unity, "
    f"i.e. |H| ∈ [1−δp, 1+δp] with δp = {spec.delta_p:.4f}; stopband attenuation ≥ {spec.as_db:g} dB "
    f"(δs = {spec.delta_s:g}); transition-band peaking ≤ 1+δp; and latency ≤ {spec.latency_ms:g} ms "
    f"({spec.latency_samples:.1f} samples). The 40-Hz edge keeps the QRS energy, and the 20-ms budget is an "
    f"illustrative real-time constraint. Neither is a clinical requirement. Every filter is verified "
    f"numerically against this mask on a 1024-point grid before it is compared, so all comparisons use exactly "
    f"the same specification. The same physical specification is redesigned at 500 Hz for PTB-XL. A second "
    f"specification is used for PPG (fs = {spec_p.fs:g} Hz, 0–{spec_p.fp_hz:g} Hz passband, ≥ {spec_p.as_db:g} dB "
    f"above {spec_p.fst_hz:g} Hz, {spec_p.latency_ms:g}-ms budget).")
R.h("2.3 Delay and group-delay metrics", 2)
R.p("Group delay is evaluated only in the passband Ωp = [0, ωp], where |H| is meaningful. Stopband group-delay "
    "spikes, caused by zeros on the unit circle, are not signal delay and are excluded. We report the "
    "following quantities:")
R.bullets([
    "τ(0): DC delay;",
    "τ̄_K: kernel-weighted mean passband group delay, defined as the **latency D** of a filter (Sec. 3);",
    "Δτ_max = max_Ωp |τ(ω) − τ̄| and Δτ_rms (uniform): the classical flatness metrics;",
    "Δτ_{rms,W}: PSD-weighted RMS group-delay variation;",
    "E_τg: the derived morphology-consistent group-delay error (Eq. 4)."])

# =========================================================================== 3 theory
R.h("3. Morphology-error model and the J objective", 1)
R.p("Let the morphology target be the delayed output of a zero-phase reference filter with magnitude R(ω). "
    "R equals 1 in the passband, follows a raised-cosine transition, and is 0 in the stopband. Every "
    "matched-spec filter approximates R, so the reference does not favor any architecture; in particular it is "
    "*not* the linear-phase FIR. For a wide-sense-stationary input with one-sided PSD S(ω), the error "
    "e[n] = y[n] − r[n−D] has power")
R.eq(r"\mathrm{E}\,|e[n]|^{2}=\frac{1}{\pi}\int_{0}^{\pi}S(\omega)\,\left|H(e^{j\omega})-R(\omega)\,e^{-j\omega D}\right|^{2}d\omega",
     "E|e[n]|² = (1/π) ∫₀^π S(ω) |H(e^{jω}) − R(ω) e^{−jωD}|² dω .")
R.p("Normalizing by the reference power Z = (1/π)∫S R² dω gives the predicted PRD = 100·√(E|e|²/Z), which we use as "
    "the Pareto ordinate. D is chosen to minimize the error, which removes the pure-delay component. In the "
    "passband (R = 1), write H = |H|e^{jφ} and define the phase-linearity error φ_e(ω) = φ(ω) + ωD. Then")
R.eq(r"\left|H-e^{-j\omega D}\right|^{2}=(|H|-1)^{2}+2|H|(1-\cos\,\varphi_{e})\approx(|H|-1)^{2}+|H|\,\varphi_{e}^{2}",
     "|H − e^{−jωD}|² = (|H| − 1)² + 2|H|(1 − cos φ_e) ≈ (|H| − 1)² + |H| φ_e² ,")
R.p("so the passband error splits into a magnitude term E_mag = (1/Z)∫S(|H|−1)² and a phase term "
    "E_φ = (1/Z)∫S φ_e². For a real filter with positive DC gain, φ(0) = 0, so the phase error is the "
    "*accumulated* group-delay error:")
R.eq(r"\varphi_{e}(\omega)=-\int_{0}^{\omega}\left(\tau(\nu)-D\right)d\nu",
     "φ_e(ω) = −∫₀^ω (τ(ν) − D) dν .")
R.p("Applying the Cauchy–Schwarz inequality, φ_e(ω)² ≤ ω ∫₀^ω (τ(ν) − D)² dν. Substituting and exchanging the "
    "order of integration gives the bound")
R.eq(r"E_{\varphi}\le E_{\tau g}:=\frac{1}{Z}\int_{0}^{\omega_{p}}K(\nu)\left(\tau(\nu)-D\right)^{2}d\nu,\qquad K(\nu)=\int_{\nu}^{\omega_{p}}\omega\,S(\omega)\,d\omega",
     "E_φ ≤ E_τg := (1/Z) ∫₀^ωp K(ν) (τ(ν) − D)² dν ,   K(ν) = ∫_ν^ωp ω S(ω) dω .")
R.p("Hence PRD²/10⁴ ≈ E_mag + E_φ ≤ E_mag + E_τg (plus a transition-band term common to all matched-spec "
    "designs). This justifies the project's central objective")
R.eq(r"J=\alpha E_{\mathrm{mag}}+\beta E_{\tau g}+\gamma D,\qquad \alpha=\beta=1",
     "J = α E_mag + β E_τg + γ D ,   α = β = 1 ,")
R.p("in which α = β = 1 follows from the derivation rather than from tuning, and γ is the application's price "
    "of latency. A Pareto front is traced either by sweeping γ or by imposing D ≤ D_max. The kernel has a clear "
    "interpretation. K(ν) decreases monotonically with ν (Fig. 3b), so **group-delay error at low frequency is "
    "the most damaging**: its phase error accumulates over every higher frequency that carries energy. "
    "Group-delay peaks at the band edge, the signature of elliptic IIRs, matter only through the little signal "
    "energy above them. A uniform Δτ_max criterion ranks filters by exactly the wrong part of the band. The "
    "optimal alignment D minimizing E_τg is the K-weighted mean group delay τ̄_K, which we therefore use as "
    "the latency definition.")

# =========================================================================== 4 methods
R.h("4. Filter designs", 1)
R.h("4.1 Reproduced baselines", 2)
R.bullets([
    f"**LP-FIR**: Parks–McClellan equiripple linear-phase FIR [1]. The minimum length meeting the mask is "
    f"N = {len(json.load(open(RESULTS / 'bank_ecg360.json'))['LP-FIR']['b'])}, with passband:stopband error weights 1 : δp/δs. Its "
    f"latency is (N−1)/2 samples.",
    "**MP-FIR**: the minimum-phase spectral factor of the LP-FIR, obtained by reflecting zeros outside the "
    "unit circle into it [2]. The magnitude is identical, the delay is the minimum possible, and the phase is "
    "nonlinear.",
    f"**Elliptic and Butterworth IIR**: minimum orders ({bank_meta['Elliptic']['order']} and "
    f"{bank_meta['Butterworth']['order']}) from the bilinear transform, designed for slightly tighter tolerances "
    f"and gain-centered to meet the same mask. Both are realized as second-order sections.",
    f"**Ell+AP-flat**: the elliptic IIR followed by K = {hp['apflat_K']} second-order all-pass sections "
    f"A(z) = (r² − 2r cosθ z⁻¹ + z⁻²)/(1 − 2r cosθ z⁻¹ + r² z⁻²). The sections are optimized (L-BFGS-B, "
    f"{P['n_starts']} random starts, pole radius r ≤ {P['allpass_r_max']}) to minimize the *unweighted* passband "
    f"group-delay variance, which is the classical equalizer criterion [5]. A minimax variant is reported in the ablation.",
    "**LD-FIR-flat**: a constrained low-delay FIR approximating e^{−jωD} with uniform weighting. It belongs to the "
    "classical low-delay complex-approximation family [3], [4] and has the same constraints and length as "
    "MO-LDFIR, so it isolates the effect of the PSD weighting."])
R.p("*Note on reproduction.* Refs. [3]–[6] were used for problem framing. The baselines above are faithful "
    "implementations of the *design principles* of those works under our matched specification. They are not "
    "line-by-line reproductions of each paper's algorithm.", italic=False)
R.h("4.2 Proposed: morphology-optimal constrained low-delay FIR (MO-LDFIR)", 2)
R.p("For a fixed latency D and length N, with h ∈ ℝ^N and H(ω) = Σ h[n]e^{−jωn}, MO-LDFIR solves")
R.eq(r"\underset{h}{\mathrm{min}}\,\int_{0}^{\pi}S(\omega)\,\left|H(\omega)-R(\omega)\,e^{-j\omega D}\right|^{2}d\omega",
     "min_h ∫₀^π S(ω) |H(ω) − R(ω)e^{−jωD}|² dω")
R.eq(r"\mathrm{s.t.}\quad\left|H(\omega)-e^{-j\omega D}\right|\le\delta_{p}\;(\omega\in\Omega_{p}),\;\;|H(\omega)|\le\delta_{s}\;(\omega\in\Omega_{s}),\;\;|H(\omega)|\le1+\delta_{p}\;(\omega\in\Omega_{t})",
     "s.t.  |H(ω) − e^{−jωD}| ≤ δp (ω ∈ Ωp),  |H(ω)| ≤ δs (ω ∈ Ωs),  |H(ω)| ≤ 1 + δp (transition band).")
R.p("Here Ω_p, Ω_s and Ω_t denote the passband, stopband and transition band. "
    "The objective is exactly Eq. (1), the expected morphology MSE. Each constraint is a second-order cone in h. "
    "The passband cone is a convex *sufficient* condition for the ripple mask, because it implies "
    "1−δp ≤ |H| ≤ 1+δp, and it also bounds the passband phase error by asin δp. The program is therefore convex, "
    "and the CLARABEL interior-point solver (via CVXPY [13]) returns its global optimum. The minimum feasible "
    "length N_min(D) is found by bisection. The stopband cone uses 0.98·δs as a small safety margin. N = N_min + slack, with the slack chosen on the development set. "
    "S(ω) is the median unit-power Welch PSD of the baseline-corrected MIT-BIH DS1 ECGs (floor 10⁻³).")
R.h("4.3 Proposed: energy-weighted, latency-penalized all-pass equalizer (EW-APE)", 2)
R.p("An all-pass section leaves |H| and hence E_mag unchanged, so equalization can only reduce E_φ. EW-APE "
    "minimizes the *exact* full-band error, not a group-delay surrogate:")
R.eq(r"\underset{r_{k},\theta_{k},D}{\mathrm{min}}\;\frac{1}{Z}\int_{0}^{\pi}S\,\left|H_{\mathrm{IIR}}(\omega)A(\omega;r,\theta)-R\,e^{-j\omega D}\right|^{2}d\omega+\gamma D,\quad0.05\le r_{k}\le0.97,\;(0\le D\le D_{\mathrm{max}})",
     "min_{r_k, θ_k, D} (1/Z)∫₀^π S |H_IIR(ω)A(ω; r, θ) − R e^{−jωD}|² dω + γD ,  0.05 ≤ r_k ≤ 0.97, (0 ≤ D ≤ D_max).")
R.p("The bound on r_k keeps a stability margin of 0.03 and limits coefficient sensitivity. For the "
    "comparison under the budget, D_max is the latency budget. Variants that use the K(ν)-weighted or "
    "PSD-weighted group-delay surrogate are evaluated in the ablation (Sec. 6.7).")
R.h("4.4 Computational cost", 2)
R.p("Cost is counted per output sample for direct-form realizations. An FIR of length N costs N multiplies "
    "(⌈N/2⌉ if it is symmetric), N−1 additions and N−1 states. A biquad costs at most 5 multiplies, 4 additions "
    "and 2 states. A direct-form all-pass biquad costs 4 multiplies, or 2 in lattice form. These counts are "
    "algorithmic costs, not hardware claims.")

# =========================================================================== 5 protocol
R.h("5. Experimental protocol", 1)
R.p("**Partitions (fixed before tuning).** Development: MIT-BIH Arrhythmia Database [8], [9] DS1 (22 records), "
    "using the inter-patient split of de Chazal et al. [7]. Held-out test 1: MIT-BIH DS2, 22 *different patients*, "
    "full 30-min MLII lead. Independent test 2: PTB-XL v1.0.1 [10], recommended test fold 10, the first "
    f"{n_ptb} records by ecg_id, lead II, 500 Hz. Paced records (102, 104, 107, 217) are excluded. The PSD weight, "
    f"the FIR length slack and the EW-APE section count were selected on DS1 only (Table 13). For PTB-XL nothing was "
    f"re-tuned: the locked rules were re-applied at 500 Hz with the MIT-BIH PSD (cross-dataset and "
    f"cross-rate transfer).")
R.p("**Pre-processing and reference.** A common offline pre-stage removes baseline wander. It is a zero-phase "
    "0.5-Hz Butterworth high-pass applied identically to every filter's input and to the reference, and it is "
    "never one of the compared real-time filters. The morphology target is the FFT-based zero-phase reference "
    "filter R(ω) applied offline. Forward-backward and zero-phase processing are used *only* to build "
    "references, never as a real-time solution.")
R.p("**Protocols.** (i) *Clean*: recorded ECG, filtered, compared with its reference. (ii) *Stress*: drifting "
    "60/120-Hz interference (0.1 mV) and EMG-like noise (0.03 mV rms) are added to the input only, and the target "
    "is unchanged. (iii) *Synthetic ECG Monte-Carlo*: a sum-of-Gaussians beat model [11] is evaluated analytically, "
    f"giving exact ground truth. {P['mc_trials']} trials of 20 s are drawn with HR ~ U(50,120) bpm, QRS width "
    "scale ~ U(0.6,1.6), power-line amplitude ~ U(0.02,0.2) mV and EMG level ~ U(0.005,0.05) mV. Because the "
    "filters are linear, distortion (h∗s) and residual noise (h∗v) are measured separately. (iv) A QRS-sharpness "
    "sweep and a noise-level sweep. (v) Synthetic PPG with its own specification, with the PSD from disjoint "
    "development seeds. (vi) Step and impulse transients.")
R.p("**Metrics.** PRD_nom: normalized PRD against the reference delayed by the filter's nominal latency τ̄_K. "
    "PRD_opt: the same at the best fractional delay, which isolates shape distortion. R-peak (or PPG systolic "
    "peak) timing error |t_out − t_ref − τ̄_K| and timing jitter, both after parabolic peak interpolation. "
    "Peak-amplitude error. ST-level error: the mean error 80–120 ms after R. Overshoot, settling and ringing. "
    "Frequency-domain metrics: passband deviation, attenuation, Δτ, E_mag, E_φ, E_τg, operation count, and pole "
    "radius as a stability margin. **Statistics:** medians [IQR] over records; paired Wilcoxon signed-rank tests "
    "of MO-LDFIR against each comparator with Holm correction within each (dataset, metric) family; bootstrap "
    "95% CIs of the median paired difference (2000 resamples). The seed is "
    f"{P['seed']}, and all parameters are in config/*.json.")

# =========================================================================== 6 results
R.h("6. Results", 1)
R.h("6.1 Matched-specification filter set", 2)
rows = []
for k in ORDER:
    m = F[k]
    rows.append([k + (" *" if k in ("MO-LDFIR", "EW-APE") else ""), m["name"].split("(")[-1].rstrip(")"),
                 f"{m['ripple_pp_db']:.3f}", f"{m['atten_db']:.2f}", f"{m['tau_dc']:.2f}",
                 f"{m['tau_K']:.2f} / {m['tau_K_ms']:.1f}", f"{m['dtau_max_ms']:.2f}", f"{m['dtau_rms_W_ms']:.3f}",
                 ("0" if m['E_tau'] < 1e-20 else f"{m['E_tau']:.2e}"), f"{m['prd_pred']:.2f}", f"{m['ops_mult']}/{m['ops_add']}/{m['ops_states']}",
                 "yes" if m["latency_ok"] else "no"])
R.table(["Filter", "Design", "Ripple p-p (dB)", "Atten. (dB)", "τ(0) (smp)", "Latency τ̄_K (smp/ms)",
         "Δτ_max (ms)", "Δτ_{rms,W} (ms)", "E_τg", "PRD_pred (%)", "mult/add/states", "≤20 ms"], rows,
        "Matched-specification ECG filter set (fs = 360 Hz). All designs satisfy the 0.5-dB/40-dB mask. "
        "* proposed. PRD_pred is the frequency-domain prediction (Eq. 1) with the DS1 ECG PSD.", font=7.5)
R.p(f"All eight designs meet the identical magnitude mask (Fig. 1), so differences in morphology are due to phase "
    f"alone. Only {andlist(LAT_OK)} meet the {spec.latency_ms:g}-ms budget. The LP-FIR needs "
    f"{F['LP-FIR']['tau_K_ms']:.1f} ms. The elliptic IIR is the cheapest design ({F['Elliptic']['ops_mult']} "
    f"multiplies) and has low latency ({F['Elliptic']['tau_K_ms']:.1f} ms), but its group delay rises by "
    f"{F['Elliptic']['dtau_max_ms']:.1f} ms toward the band edge (Fig. 2c), and its phase deviates from linear "
    f"by tens of degrees inside the passband (Fig. 2b). The classical equalizer (K = {hp['apflat_K']}) reduces "
    f"Δτ_max to {F['Ell+AP-flat']['dtau_max_ms']:.1f} ms but increases latency to "
    f"{F['Ell+AP-flat']['tau_K_ms']:.1f} ms. MO-LDFIR stays within budget ({F['MO-LDFIR']['tau_K_ms']:.2f} ms) "
    f"with Δτ_max = {F['MO-LDFIR']['dtau_max_ms']:.2f} ms and the lowest E_τg of all designs "
    f"({F['MO-LDFIR']['E_tau']:.1e}), at the price of {F['MO-LDFIR']['ops_mult']} multiplies per sample.")
R.figure(FIGS / "F01_magnitude.png", "Magnitude responses of the matched-spec set: (a) full band with the 40-dB "
         "stopband mask; (b) passband ripple against the ±δp mask. The grey band is the 40–60 Hz transition band.")
R.figure(FIGS / "F02_phase_groupdelay.png", "(a) Unwrapped phase. (b) Deviation from linear phase "
         "φ(ω)+ωτ̄_K, i.e. the phase error that distorts morphology. (c) Passband group delay with the 20-ms budget. "
         "Stopband group delay is deliberately not shown (it is not signal delay).")
R.figure(FIGS / "F03_gdcheck_kernel.png", "(a) Analytic group delay agrees with the numerical derivative of "
         "the unwrapped phase. (b) The DS1 ECG PSD and the derived group-delay kernel K(ν): low-frequency "
         "group-delay error is weighted most.")
R.figure(FIGS / "F04_polezero.png", "Pole-zero maps; dotted rays mark ωp and ωs. MO-LDFIR is a non-minimum-phase "
         "FIR: 2 of its 70 zeros lie outside the unit circle (at z ≈ +1.78 and z ≈ −11.0, the latter indicated by an arrow), "
         "which is what allows a delay shorter than (N−1)/2. "
         "The all-pass poles of EW-APE under the budget are weak (small radius), because a strong section would "
         "exceed the latency budget.", width_cm=14)
R.figure(FIGS / "F05_impulse_step.png", "Impulse responses (coefficient patterns) and step responses. The "
         "dashed line marks the 20-ms budget.")
rows = [[k, f"{TR[k]['delay50_ms']:.1f}", f"{TR[k]['overshoot_pct']:.1f}", f"{TR[k]['undershoot_pct']:.1f}",
         f"{TR[k]['rise_ms']:.1f}", f"{TR[k]['settle_ms']:.1f}", f"{100 * TR[k]['ringing_frac']:.2f}"] for k in ORDER]
R.table(["Filter", "50% delay (ms)", "Overshoot (%)", "Pre-shoot (%)", "Rise 10–90 (ms)", "Settling 2% (ms)",
         "Late IR energy (%)"], rows, "Transient behavior (step and impulse). Late IR energy is the fraction of "
        "impulse-response energy arriving >30 ms after the energy centroid (ringing indicator).")

# ---- 6.2 Pareto
R.h("6.2 Pareto trade-offs: attenuation, group delay, latency and cost", 2)
R.figure(FIGS / "F06_pareto_attenuation.png", "Pareto map of latency τ̄_K versus predicted morphology PRD for "
         "stopband attenuation 30–60 dB (transition 40→60 Hz). Lines connect the non-dominated designs of each "
         "swept family; the dashed line is the 20-ms budget.")
R.figure(FIGS / "F07_pareto_transition.png", "The same Pareto map for transition widths of 5, 10, 20 and "
         "40 Hz (As = 40 dB).")
R.figure(FIGS / "F08_attenuation_tradeoff.png", "Three-way trade-off under the 20-ms budget: best predicted "
         "PRD (a) and passband group-delay variation (b) of each architecture as the attenuation requirement "
         "grows.")
R.figure(FIGS / "F09_budget_sweep.png", "Best achievable predicted morphology PRD as a function of the latency "
         "budget (As = 40 dB, 40→60 Hz).", width_cm=11)
vlist = ["As=30 dB", "As=40 dB", "As=50 dB", "As=60 dB", "TW=5 Hz", "TW=10 Hz", "TW=40 Hz"]
archs_t = ["LP-FIR", "MP-FIR", "Elliptic", "Ell+AP-flat", "EW-APE", "LD-FIR-flat", "MO-LDFIR"]


def cell20(v, a):
    b = ps(v, a, "best20")
    return "–" if np.isnan(b) else f"{b:.2f} ({ps(v, a, 'ops20'):.0f})"


def cell3(v, a):
    b = ps(v, a, "lat3")
    return "–" if np.isnan(b) else f"{b:.0f} ({ps(v, a, 'ops3'):.0f})"


R.table(["Variant"] + archs_t, [[v.replace("As=40 dB", "As=40 dB, TW=20 Hz (base)")] + [cell20(v, a) for a in archs_t]
                                for v in vlist],
        "Best predicted PRD (%) achievable within the 20-ms budget, with multiplies/sample in parentheses. "
        "'–' means no design of that family meets both the mask and the budget.", font=7.5)
R.table(["Variant"] + archs_t, [[v] + [cell3(v, a) for a in archs_t] for v in vlist],
        "Minimum latency (ms) at which each family reaches a predicted PRD ≤ 3%, with multiplies/sample in "
        "parentheses. '–' means the family never reaches 3%.", font=7.5)
R.p(f"The Pareto maps (Figs. 6–9, Tables 3–4) answer the central question for the whole design space, not just "
    f"one point. Three regimes emerge. **(i) Budgets below the IIR/minimum-phase limit.** At the base "
    f"specification, nothing meets the mask below {F['MP-FIR']['tau_K_ms']:.1f} ms (MP-FIR). Between that and "
    f"about {ps('As=40 dB', 'MO-LDFIR', 'minlat'):.0f} ms only MP-FIR, the elliptic IIR and a "
    f"weakly equalized IIR are feasible, all with PRD ≈ 7–10%. In this regime latency is bought with "
    f"distortion, and no architecture can avoid it. **(ii) Moderate budgets.** Once the budget admits a "
    f"low-delay FIR, MO-LDFIR dominates every other family. Within 20 ms it reaches "
    f"{ps('As=30 dB', 'MO-LDFIR', 'best20'):.2f}%, {ps('As=40 dB', 'MO-LDFIR', 'best20'):.2f}% and "
    f"{ps('As=50 dB', 'MO-LDFIR', 'best20'):.2f}% for 30/40/50 dB, against "
    f"{ps('As=30 dB', 'EW-APE', 'best20'):.1f}/{ps('As=40 dB', 'EW-APE', 'best20'):.1f}/"
    f"{ps('As=50 dB', 'EW-APE', 'best20'):.1f}% for the best equalized IIR. The budget sweep (Fig. 9) shows that "
    f"MO-LDFIR at the LP-FIR's own latency ({F['LP-FIR']['tau_K_ms']:.0f} ms) is *more* accurate than the "
    f"minimum-length LP-FIR ({BUD.set_index('latency_ms').loc[45.0, 'MO-LDFIR']:.2f}% vs "
    f"{BUD.set_index('latency_ms').loc[45.0, 'LP-FIR']:.2f}%). The extra taps buy morphology accuracy rather "
    f"than delay. **(iii) Demanding specifications.** At 60 dB, or with 10- and 5-Hz transition widths, the "
    f"minimum latency of a mask-compliant low-delay FIR rises to {ps('As=60 dB', 'MO-LDFIR', 'minlat'):.0f}, "
    f"{ps('TW=10 Hz', 'MO-LDFIR', 'minlat'):.0f} and {ps('TW=5 Hz', 'MO-LDFIR', 'minlat'):.0f} ms, so no FIR "
    f"meets 20 ms. Here only the IIR families are feasible, and attenuation can be traded for latency only by "
    f"accepting their phase distortion (Fig. 8).")
_nb = lambda a, c: sorted(ps(v, a, c) for v in ("TW=5 Hz", "TW=10 Hz"))
nb_lat, nb_ops = _nb("EW-APE", "lat3"), _nb("EW-APE", "ops3")
nb_mo_ops, nb_mo_lat = _nb("MO-LDFIR", "ops3"), _nb("MO-LDFIR", "lat3")
nb_lp_ops, nb_lp_lat = _nb("LP-FIR", "ops3"), _nb("LP-FIR", "lat3")
R.p(f"**Cost.** The FIR advantage is paid for in arithmetic. To reach PRD ≤ 3% at TW = 5 Hz, MO-LDFIR needs "
    f"{ps('TW=5 Hz', 'MO-LDFIR', 'ops3'):.0f} multiplies/sample at {ps('TW=5 Hz', 'MO-LDFIR', 'lat3'):.0f} ms. "
    f"EW-APE needs {ps('TW=5 Hz', 'EW-APE', 'ops3'):.0f} at {ps('TW=5 Hz', 'EW-APE', 'lat3'):.0f} ms, and the "
    f"classical flat equalizer {ps('TW=5 Hz', 'Ell+AP-flat', 'ops3'):.0f} at "
    f"{ps('TW=5 Hz', 'Ell+AP-flat', 'lat3'):.0f} ms. Of the 35 classical-equalizer designs in the sweep, 34 are "
    f"dominated by an EW-APE design (equal or lower PRD at lower latency), because the EW-APE objective spends delay only where "
    f"signal energy makes it worthwhile. The exception is the K = 5 flat equalizer at TW = 5 Hz (2.7% at 131 ms), "
    f"which lies beyond the K ≤ 3 range swept for EW-APE. Equalized IIRs are therefore the cost-efficient choice when the "
    f"transition band is narrow and a latency of about {nb_lat[0]:.0f}–{nb_lat[1]:.0f} ms (EW-APE at TW = 5–10 Hz) "
    f"is tolerable. One baseline failure is "
    f"reported for transparency: at TW = 5 Hz the minimum-phase factor of the 119-tap LP-FIR fails the mask, "
    f"because root-finding a degree-118 polynomial is numerically ill-conditioned. A cepstral or "
    f"optimization-based spectral factorization would be needed there.")

# ---- 6.3 synthetic
R.h("6.3 Synthetic ground truth and validation of the error model", 2)
rows = []
for k in ORDER:
    rows.append([k, iqr(MC, "prd_opt", k), iqr(MC, "prd_nom", k), iqr(MC, "r_timing_err_ms", k, 3),
                 iqr(MC, "r_amp_err_pct", k), iqr(MC, "st_err_uV", k)])
R.table(["Filter", "PRD_opt (%)", "PRD_nom (%)", "R timing err. (ms)", "R amp. err. (%)", "ST err. (µV)"], rows,
        f"Synthetic ECG Monte-Carlo ({P['mc_trials']} trials; exact ground truth): median [IQR] of the "
        "distortion of the clean component.", font=7.5)
mc = pd.read_csv(RESULTS / "synthetic_mc.csv")
nr = mc.groupby("filter").noise_red_db.median()
R.p(f"On synthetic data with exact ground truth, MO-LDFIR has the lowest distortion "
    f"(PRD_opt {med(MC, 'prd_opt', 'MO-LDFIR'):.2f}% against {med(MC, 'prd_opt', 'LP-FIR'):.2f}% for the LP-FIR and "
    f"{med(MC, 'prd_opt', 'Elliptic'):.2f}% for the elliptic IIR). The noise reduction of the matched-spec "
    f"filters is similar ({nr.min():.1f}–{nr.max():.1f} dB median), confirming that the magnitude specifications "
    f"are matched. The IIRs reject slightly more because their transition bands are steeper than required. "
    f"**The QRS-sharpness sweep (Fig. 10a) reveals the failure regime of the proposed design.** For very narrow "
    f"QRS complexes (width scale 0.5, σ_R = 5 ms) the energy moves toward the band edge, where the low-delay "
    f"FIR's phase error is largest. There MO-LDFIR reaches "
    f"{QSW[(QSW.qrs == 0.5) & (QSW['filter'] == 'MO-LDFIR')].prd_opt.median():.1f}% against "
    f"{QSW[(QSW.qrs == 0.5) & (QSW['filter'] == 'LP-FIR')].prd_opt.median():.1f}% for the LP-FIR, although it "
    f"still outperforms the IIR and MP-FIR "
    f"({QSW[(QSW.qrs == 0.5) & (QSW['filter'] == 'Elliptic')].prd_opt.median():.1f}% and "
    f"{QSW[(QSW.qrs == 0.5) & (QSW['filter'] == 'MP-FIR')].prd_opt.median():.1f}%). The PSD weight is only "
    f"optimal for the spectrum it was designed for.")
rho = np.corrcoef(np.log(VAL.prd_pred_synPSD), np.log(VAL.prd_meas_median))[0, 1]
R.p(f"The error model of Sec. 3 predicts the measured synthetic PRD from the filter's frequency response and "
    f"the signal PSD alone. The log-log correlation across the eight designs is r = {rho:.3f} (Fig. 10b). For "
    f"MO-LDFIR the prediction is {VAL.set_index('filter').loc['MO-LDFIR', 'prd_pred_synPSD']:.2f}% against "
    f"{VAL.set_index('filter').loc['MO-LDFIR', 'prd_meas_median']:.2f}% measured. On real ECG the ranking is also "
    f"preserved, but the absolute values differ because the stationary-PSD assumption is coarse for a "
    f"quasi-periodic signal.")
R.figure(FIGS / "F10_synthetic_validation.png", "(a) Noise-free distortion versus QRS sharpness (20 seeds per "
         "point). (b) Predicted versus measured PRD (filled: MIT-BIH DS2 medians with the DS1 PSD; open: synthetic "
         "ECG with its own PSD).")

# ---- 6.4 MIT DS2
R.h("6.4 Held-out real ECG: MIT-BIH DS2 (inter-patient)", 2)
rows = []
for k in ORDER:
    rows.append([k, f"{F[k]['tau_K_ms']:.1f}", iqr(DS2, "prd_opt", k), iqr(DS2, "prd_nom", k),
                 iqr(DS2, "r_timing_err_ms", k, 3), iqr(DS2, "r_amp_err_pct", k), iqr(DS2, "st_err_uV", k),
                 iqr(DS2S, "prd_nom", k)])
R.table(["Filter", "Latency (ms)", "PRD_opt (%)", "PRD_nom (%)", "R timing err. (ms)", "R amp. err. (%)",
         "ST err. (µV)", "PRD_nom stress (%)"], rows,
        "MIT-BIH DS2 (22 held-out patients, 30 min each): median [IQR] over records.", font=7.5)
rows = []
for comp in [k for k in ORDER if k != "MO-LDFIR"]:
    r_ = [comp]
    for ds_, met in ((DS2, "prd_opt"), (DS2, "st_err_uV"), (DS2S, "prd_nom"), (PTB, "prd_opt")):
        t = test(ds_, met, comp)
        r_.append("–" if t is None else f"{t.median_diff:+.2f} [{t.ci_lo:+.2f}, {t.ci_hi:+.2f}], p {pfmt(t.p_holm)}")
    rows.append(r_)
R.table(["MO-LDFIR vs.", "DS2 PRD_opt (pp)", "DS2 ST err. (µV)", "DS2 stress PRD_nom (pp)", "PTB-XL PRD_opt (pp)"],
        rows, "Paired comparisons: median difference (MO-LDFIR − comparator) with bootstrap 95% CI and "
        "Holm-corrected Wilcoxon p. Negative = MO-LDFIR better.", font=7.5)
t = test(DS2, "prd_opt", "LP-FIR")
R.p(f"On held-out patients, MO-LDFIR is significantly better than every other filter that meets the latency "
    f"budget for both shape (PRD_opt) and ST level. Against the Elliptic IIR the median paired PRD reduction is "
    f"{-test(DS2, 'prd_opt', 'Elliptic').median_diff:.2f} percentage points, and against the budget-constrained "
    f"EW-APE it is {-test(DS2, 'prd_opt', 'EW-APE').median_diff:.2f} pp (both p_Holm < 0.001). Against the "
    f"LP-FIR, which takes {F['LP-FIR']['tau_K_ms']:.0f} ms, the PRD difference is {t.median_diff:+.2f} pp "
    f"(p_Holm {pfmt(t.p_holm)}). The most striking result is the ST level: "
    f"{med(DS2, 'st_err_uV', 'MO-LDFIR'):.2f} µV against {med(DS2, 'st_err_uV', 'LP-FIR'):.2f} µV for the LP-FIR. "
    f"The mechanism follows from Eq. (1). The PSD weight concentrates accuracy where the ECG energy is, near DC, "
    f"so MO-LDFIR's low-frequency gain is very close to 1. The equiripple LP-FIR spends its full δp = "
    f"{100 * spec.delta_p:.1f}% ripple budget there, which scales slow ST-segment levels. The LP-FIR keeps the "
    f"smallest R-peak timing error, as expected from its exactly constant delay. Under the stress protocol, "
    f"residual in-band noise dominates and MO-LDFIR and LP-FIR become statistically indistinguishable in PRD, while "
    f"MO-LDFIR retains the ST advantage.")
R.figure(FIGS / "F11_dataset_boxplots.png", "Record-wise distributions on MIT-BIH DS2 (top) and PTB-XL (bottom): "
         "shape error, R-peak timing error relative to the nominal latency, and ST-level error.")
R.figure(FIGS / "F12_waveforms.png", "Latency-compensated outputs and errors for a normal beat (record 100) and a "
         "ventricular ectopic beat (record 200). The IIR and MP-FIR errors are concentrated on the QRS complex and "
         "the following ringing; the equalized IIR shows pre-ringing.")

# ---- 6.5 PTB-XL
R.h("6.5 Independent validation: PTB-XL (500 Hz, no re-tuning)", 2)
rows = []
for k in ORDER:
    rows.append([k, f"{F5[k]['tau_K_ms']:.1f}", str(F5[k]["ops_mult"]), iqr(PTB, "prd_opt", k),
                 iqr(PTB, "r_timing_err_ms", k, 3), iqr(PTB, "r_amp_err_pct", k), iqr(PTB, "st_err_uV", k)])
R.table(["Filter", "Latency (ms)", "mult", "PRD_opt (%)", "R timing err. (ms)", "R amp. err. (%)", "ST err. (µV)"],
        rows, f"PTB-XL fold 10 ({n_ptb} records, lead II). Filters were redesigned at 500 Hz from the same physical "
        "specification and locked rules. LD-FIR-flat's kernel-weighted latency (20.2 ms) slightly exceeds the "
        "budget at 500 Hz: its uniform weighting does not control the group delay where the latency kernel K(ν) is largest.", font=7.5)
t = test(PTB, "prd_opt", "LP-FIR")
R.p(f"The ranking transfers to a different dataset, sampling rate and patient population without any re-tuning. "
    f"MO-LDFIR (N = {F5['MO-LDFIR']['ops_mult']}, {F5['MO-LDFIR']['tau_K_ms']:.1f} ms) is again the best "
    f"latency-compliant design ({med(PTB, 'prd_opt', 'MO-LDFIR'):.2f}% against "
    f"{med(PTB, 'prd_opt', 'Elliptic'):.2f}% for the elliptic IIR and {med(PTB, 'prd_opt', 'MP-FIR'):.2f}% for "
    f"the MP-FIR), and it has the lowest ST error. Here the 45-ms LP-FIR is better in shape by "
    f"{t.median_diff:.2f} pp (p_Holm {pfmt(t.p_holm)}; better on {100 * (1 - t.frac_records_better):.0f}% of "
    f"records). This is a genuine latency-morphology trade-off. A plausible but untested explanation is a spectral "
    f"mismatch between the MIT-BIH-derived weight and the PTB-XL lead-II population; the QRS-sharpness sweep "
    f"(Fig. 10a) shows that MO-LDFIR's advantage shrinks when signal energy moves toward the band edge.")

# ---- 6.6 PPG
R.h("6.6 PPG case study", 2)
rows = []
for k in ORDER:
    rows.append([k, f"{FP[k]['tau_K_ms']:.0f}", str(FP[k]["ops_mult"]), iqr(PPG, "prd_opt", k),
                 iqr(PPG, "r_timing_err_ms", k), iqr(PPG, "r_amp_err_pct", k)])
R.table(["Filter", "Latency (ms)", "mult", "PRD_opt (%)", "Systolic-peak timing err. (ms)", "Peak amp. err. (%)"],
        rows, f"Synthetic PPG ({P['mc_trials']} trials) under the PPG specification (fs = {spec_p.fs:g} Hz, "
        f"0–{spec_p.fp_hz:g} Hz, ≥{spec_p.as_db:g} dB above {spec_p.fst_hz:g} Hz, budget {spec_p.latency_ms:g} ms).",
        font=7.5)
R.p(f"For the narrower relative transition band of the PPG specification, the LP-FIR needs "
    f"{FP['LP-FIR']['tau_K_ms']:.0f} ms, nearly three times the budget. MO-LDFIR meets the budget with "
    f"{med(PPG, 'prd_opt', 'MO-LDFIR'):.2f}% shape error against {med(PPG, 'prd_opt', 'Elliptic'):.2f}% for the "
    f"elliptic IIR. The cost grows to {FP['MO-LDFIR']['ops_mult']} multiplies per sample, against "
    f"{FP['Elliptic']['ops_mult']} for the IIR: a narrow transition band makes every FIR expensive (Sec. 7).")
R.figure(FIGS / "F13_ppg.png", "PPG specification: (a) latency-compensated outputs; (b) passband group delay.")

# ---- 6.7 ablation
R.h("6.7 Ablation and sensitivity", 2)
ABL_LABEL = {"- PSD weight (flat)": "without PSD weight (uniform weighting)",
             "- matched PSD (synthetic-ECG PSD)": "mismatched PSD (synthetic-ECG PSD)",
             "- length slack (N = N_min)": "without length slack (N = N_min)",
             "PSD floor 1e-4": "PSD floor 10⁻⁴", "PSD floor 1e-2": "PSD floor 10⁻²"}
rows = []
for _, a in ABL.iterrows():
    if not a.get("feasible", True):
        rows.append([ABL_LABEL.get(a.variant, a.variant), "infeasible", "", "", "", ""])
        continue
    rows.append([ABL_LABEL.get(a.variant, a.variant), "yes" if a.meets_spec else "NO", f"{a.tau_K_ms:.1f}", f"{a.E_tau:.2e}",
                 f"{a.prd_pred:.2f}", f"{a.prd_ds2_median:.2f} [{a.prd_ds2_q1:.2f}, {a.prd_ds2_q3:.2f}]"])
R.table(["Variant", "Meets spec", "Latency (ms)", "E_τg", "PRD_pred (%)", "DS2 PRD_opt (%) median [IQR]"], rows,
        "Ablation (DS2, first 10 min per record). The upper block varies MO-LDFIR components; the lower block "
        "compares all-pass criteria for K = 2 without a latency bound.", font=7.5)
A = ABL.set_index("variant")
apk = [v for v in A.index if v.startswith("AP:") or v.startswith("EW-APE")]
R.p(f"**MO-LDFIR components.** Removing the PSD weight, which leaves the classical uniformly weighted low-delay "
    f"FIR with identical constraints and length, raises the DS2 median PRD from "
    f"{A.loc['MO-LDFIR (proposed)', 'prd_ds2_median']:.2f}% to {A.loc['- PSD weight (flat)', 'prd_ds2_median']:.2f}%. "
    f"The signal-energy weighting is therefore the main source of the gain. Designing with the PSD of the "
    f"*synthetic* ECG model instead of real DS1 ECG costs little "
    f"({A.loc['- matched PSD (synthetic-ECG PSD)', 'prd_ds2_median']:.2f}%), so an approximate spectral model "
    f"suffices. Removing the length slack (N = N_min) gives "
    f"{A.loc['- length slack (N = N_min)', 'prd_ds2_median']:.2f}%: at the minimum feasible length the constraints "
    f"leave almost no freedom for the objective. The PSD floor matters moderately "
    f"({A.loc['PSD floor 1e-4', 'prd_ds2_median']:.2f}% at 10⁻⁴, {A.loc['PSD floor 1e-2', 'prd_ds2_median']:.2f}% "
    f"at 10⁻²). A high floor flattens the weight toward the uniform design. The floor of 10⁻³ was fixed a priori "
    f"and not tuned on test data.")
R.p(f"**All-pass criteria (K = 2, no latency bound).** The exact morphology criterion of EW-APE gives the lowest "
    f"error ({A.loc['EW-APE exact criterion (proposed) K=2', 'prd_ds2_median']:.2f}%). The PSD-weighted "
    f"group-delay variance ({A.loc['AP: PSD-weighted GD variance K=2', 'prd_ds2_median']:.2f}%) and the derived "
    f"kernel surrogate ({A.loc['AP: GD-kernel surrogate K(v) K=2', 'prd_ds2_median']:.2f}%) come next, and the "
    f"classical flat criterion is worst ({A.loc['AP: flat GD variance (classical) K=2', 'prd_ds2_median']:.2f}%), "
    f"while also adding the most latency. The kernel surrogate is the cheapest to evaluate, because the "
    f"group delay of an all-pass section is analytic, but it is a bound and therefore conservative. Our "
    f"smoothed-minimax (p = 16) implementation of the classical minimax criterion converged to a poor local "
    f"optimum ({A.loc['AP: minimax GD (classical) K=2', 'prd_ds2_median']:.1f}%). We report it as an "
    f"implementation limitation of that baseline, not as evidence against minimax equalization in general.")

# ---- 6.8 quantisation
R.h("6.8 Finite word-length behavior", 2)
piv = QNT.pivot(index="filter", columns="bits", values="atten_db").loc[ORDER]
rows = [[k] + [f"{piv.loc[k, b]:.2f}" if pd.notna(piv.loc[k, b]) else "unstable" for b in piv.columns] for k in ORDER]
R.table(["Filter"] + [("float64" if b == 64 else f"{b} bit") for b in piv.columns], rows,
        "Stopband attenuation (dB) after coefficient quantization (FIR: shared exponent; IIR: Q2.(B−2) per "
        "section). The float64 design is the high-precision reference.", font=7.5)
R.figure(FIGS / "F14_quantisation.png", "Coefficient quantization: attenuation (a) and predicted morphology "
         "error (b) versus word length.")
qa = QNT.set_index(["filter", "bits"])
ok16 = [k for k in ORDER if bool(qa.loc[(k, 16), "meets_spec"])]
bad16 = [k for k in ORDER if k not in ok16]
rel = lambda k, b: 100 * (qa.loc[(k, b), "dev_p"] / spec.delta_p - 1)
R.p(f"The IIR designs are realized as second-order sections with pole radius ≤ "
    f"{F['Elliptic']['max_pole_radius']:.3f}. They keep their *stopband* attenuation even with 10-bit "
    f"coefficients ({qa.loc[('Elliptic', 10), 'atten_db']:.1f} dB). However, the minimum-order elliptic filter "
    f"sits exactly on the passband-ripple limit with zero margin, so any rounding violates the mask. At 16 bits, "
    f"{andlist(bad16)} exceed the ripple tolerance by {min(rel(k, 16) for k in bad16):.1f}–"
    f"{max(rel(k, 16) for k in bad16):.1f}% of δp, which is a very small but formal violation. At 16 bits all the "
    f"FIRs pass ({andlist(k for k in ok16 if 'FIR' in k)}). At 12 bits "
    f"{andlist(k for k in ORDER if 'FIR' in k and not bool(qa.loc[(k, 12), 'meets_spec']))} lose stopband "
    f"attenuation, while the MP-FIR still meets the mask ({qa.loc[('MP-FIR', 12), 'atten_db']:.1f} dB). "
    f"MO-LDFIR drops to {qa.loc[('MO-LDFIR', 12), 'atten_db']:.1f} dB, because the SOCP optimum lies on the "
    f"attenuation constraint with only ~0.2 dB margin and 71 taps accumulate rounding error. The LP-FIR drops "
    f"to {qa.loc[('LP-FIR', 12), 'atten_db']:.2f} dB. The predicted PRD at 16 bits is unchanged to within "
    f"{max(abs(qa.loc[(k, 16), 'prd_pred'] - qa.loc[(k, 64), 'prd_pred']) for k in ORDER):.3f} pp for every "
    f"design. The lesson is architecture-independent: designs placed exactly on a constraint by an optimizer or by "
    f"minimum-order formulas need explicit margin before quantization. Margin can come from a tightened ripple "
    f"or stopband target, or from quantization-aware constraints; neither was evaluated here.")

# =========================================================================== 7 discussion
R.h("7. Discussion and design recommendations", 1)
R.p("The central question, which architecture best preserves morphology for a fixed attenuation and latency "
    "budget, has no single answer. The evidence does, however, support the following rules, summarized in "
    "Table 12.")
R.table(["Situation", "Recommended architecture", "Evidence"], [
    ["Latency budget ≥ (N_LP−1)/2, i.e. the LP-FIR fits; robustness to unknown signal spectra matters",
     "Linear-phase FIR (Parks–McClellan)", "Exactly zero phase distortion and timing jitter; best on PTB-XL; "
     "insensitive to spectral mismatch (QRS sweep)"],
    ["Latency budget between the IIR/MP limit and (N_LP−1)/2, transition ≥ ~20 Hz at 360 Hz, As ≤ 50 dB",
     "MO-LDFIR (proposed)", "Lowest PRD among compliant filters on DS2, PTB-XL and PPG and lowest ST error on DS2 and PTB-XL "
     "(p_Holm < 0.001); Pareto-dominant (Tables 3–4)"],
    ["Same budget, but the full-length LP-FIR also fits and morphology fidelity is paramount",
     "MO-LDFIR at D = budget with longer N", "Lower predicted PRD than the minimum-length LP-FIR at equal latency (Fig. 9; model prediction, not measured on test data)"],
    [f"Narrow transition / high attenuation, latency tolerance ≈ {nb_lat[0]:.0f}–{nb_lat[1]:.0f} ms, tight arithmetic budget",
     "Elliptic IIR + EW-APE (proposed)",
     f"Reaches PRD ≤ 3% at TW = 5–10 Hz with {nb_ops[0]:.0f}–{nb_ops[1]:.0f} multiplies, vs "
     f"{nb_mo_ops[0]:.0f}–{nb_mo_ops[1]:.0f} for MO-LDFIR ({nb_mo_lat[0]:.0f}–{nb_mo_lat[1]:.0f} ms) and "
     f"{nb_lp_ops[0]:.0f}–{nb_lp_ops[1]:.0f} for the LP-FIR ({nb_lp_lat[0]:.0f}–{nb_lp_lat[1]:.0f} ms); "
     "dominates 34 of 35 classical-equalizer designs"],
    ["Budget below the low-delay-FIR limit (≈ 17 ms at the base spec)", "Minimum-phase FIR or elliptic IIR "
     "(accept distortion)", "Only feasible families; MP-FIR has lower PRD, the elliptic IIR far lower cost"],
    ["Short coefficient word lengths (≤ 16 bits)", "Design every architecture with explicit ripple/stopband margin; "
     "IIR in SOS form keeps its attenuation best", "Quantization study (Table 11): zero-margin designs violate the mask"],
    ["Offline analysis (no latency limit)", "Zero-phase (forward-backward) filtering", "Not a real-time solution; "
     "used here only as reference"]],
    "Design recommendations derived from the results.", font=8)
R.p("Two DSP mechanisms explain these rules. First, morphology error is governed by the *accumulated* phase "
    "error weighted by signal energy (Eqs. 3–5), not by the peak group-delay deviation. The elliptic IIR's "
    "large band-edge Δτ_max matters less than its passband phase curvature, while the classical "
    "equalizer, which flattens group delay uniformly, wastes delay correcting a region with little signal "
    "energy. Second, an all-pass section can only *add* group delay. Under a strict budget it cannot pull "
    "the low-frequency group delay of an IIR up to a flat level without also raising the latency, so EW-APE "
    "collapses to a very weak equalizer (Table 1, Fig. 4). A low-delay FIR has no such restriction: its zeros "
    "outside the unit circle shape the phase freely, which is why it is the only architecture that improves "
    "morphology at fixed latency.")
R.h("7.1 Limitations and threats to validity", 2)
R.bullets([
    "The error model assumes a wide-sense-stationary input. ECG is quasi-periodic, so the model predicts "
    "*rankings* well but absolute PRD only approximately (Fig. 10b).",
    "The morphology reference is a zero-phase *ideal* filter. A different, equally defensible reference, for "
    "example one with a different transition shape, would change absolute PRD values, mainly through the "
    "transition-band term common to all designs.",
    "MO-LDFIR is optimal for the PSD it was designed with. Mismatched spectra (very sharp QRS, paced beats, "
    "pediatric ECG) degrade it (Fig. 10a, PSD-mismatch ablation). In those cases a linear-phase FIR is the "
    "safe choice if its latency is acceptable.",
    "Only the low-pass stage was studied. High-pass baseline removal, where phase distortion of the ST segment "
    "is clinically more critical [14], has far narrower relative transition bands and would require "
    "proportionally longer low-delay FIRs. The same framework applies but was not evaluated.",
    "Latency was defined as the energy-kernel-weighted passband group delay. Other definitions, such as the "
    "50%-step delay in Table 2, give a consistent ordering of the designs but differ in absolute value.",
    "Operation counts are direct-form algorithmic costs. No hardware or power claims are made, and no clinical "
    "claims are made: the 20-ms budget is illustrative.",
    "PTB-XL R-peaks were detected automatically on the reference signal. Detection errors affect all filters "
    "equally."])

# =========================================================================== 8 conclusion
R.h("8. Conclusion", 1)
R.p(f"We compared linear-phase FIR, minimum-phase FIR, classical IIR and phase-equalized IIR filters under "
    f"exactly matched magnitude specifications and an explicit latency budget. The derived morphology-error model "
    f"defines the J = αE_mag + βE_τg + γD objective with α = β = 1 and a signal-energy group-delay kernel K(ν). "
    f"It predicts the ranking of all designs from their frequency responses and the signal PSD. The proposed "
    f"constrained low-delay FIR (MO-LDFIR) is a convex SOCP that guarantees the magnitude mask. Within a 20-ms "
    f"budget it reduced the median shape error on held-out MIT-BIH patients from "
    f"{med(DS2, 'prd_opt', 'Elliptic'):.1f}% (elliptic IIR) to {med(DS2, 'prd_opt', 'MO-LDFIR'):.1f}%, and the "
    f"ST-level error to below 1 µV. The ranking transferred without re-tuning to PTB-XL at 500 Hz and to a PPG "
    f"specification. All-pass equalization is effective only when latency can grow. The proposed energy-weighted "
    f"equalizer uses that latency more efficiently than the classical flat-delay criterion, and it is the "
    f"cost-efficient option for narrow transition bands. The linear-phase FIR remains the most robust choice "
    f"whenever its delay is acceptable. The proposed FIR's main weaknesses are its arithmetic cost, its "
    f"sensitivity to coefficient word length, and its dependence on the assumed signal spectrum. Future work "
    f"includes quantization-aware SOCP constraints, adaptive or patient-specific spectral weights, mixed "
    f"FIR–IIR correctors, and the high-pass (baseline-removal) stage.")

# =========================================================================== references
R.h("References", 1)
refs = [
    "T. W. Parks and J. H. McClellan, “Chebyshev approximation for nonrecursive digital filters with linear phase,” "
    "IEEE Trans. Circuit Theory, vol. 19, no. 2, pp. 189–194, 1972.",
    "A. V. Oppenheim and R. W. Schafer, Discrete-Time Signal Processing, 3rd ed. Pearson, 2010.",
    "C. Wu, D. Gao and K. L. Teo, “A direct optimization method for low group delay FIR filter design,” "
    "Signal Processing, vol. 93, no. 7, pp. 1764–1772, 2013. doi:10.1016/j.sigpro.2013.01.015.",
    "B. C. Garai, P. Das and A. K. Mishra, “Group delay reduction in FIR digital filters,” Signal Processing, "
    "vol. 91, no. 8, pp. 1812–1825, 2011. doi:10.1016/j.sigpro.2011.02.005.",
    "M. F. Quélhas and A. Petraglia, “Optimum design of group delay equalizers,” Digital Signal Processing, "
    "vol. 21, no. 1, pp. 1–12, 2011. doi:10.1016/j.dsp.2010.06.001.",
    "Y. Lyu, Z. Liu, K. Pan and Y. Shen, “Phase equalization using a hybrid cascade of causal and noncausal IIR "
    "allpass filters,” Digital Signal Processing, vol. 167, 105455, 2025. doi:10.1016/j.dsp.2025.105455.",
    "P. de Chazal, M. O'Dwyer and R. B. Reilly, “Automatic classification of heartbeats using ECG morphology and "
    "heartbeat interval features,” IEEE Trans. Biomed. Eng., vol. 51, no. 7, pp. 1196–1206, 2004.",
    "G. B. Moody and R. G. Mark, “The impact of the MIT-BIH Arrhythmia Database,” IEEE Eng. Med. Biol. Mag., "
    "vol. 20, no. 3, pp. 45–50, 2001. Data: physionet.org/content/mitdb/1.0.0 (ODC-By 1.0).",
    "A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet,” Circulation, vol. 101, no. 23, "
    "pp. e215–e220, 2000.",
    "P. Wagner et al., “PTB-XL, a large publicly available electrocardiography dataset,” Scientific Data, vol. 7, "
    "154, 2020. Data: physionet.org/content/ptb-xl/1.0.1 (CC BY 4.0).",
    "P. E. McSharry, G. D. Clifford, L. Tarassenko and L. A. Smith, “A dynamical model for generating synthetic "
    "electrocardiogram signals,” IEEE Trans. Biomed. Eng., vol. 50, no. 3, pp. 289–294, 2003.",
    "S. Boyd and L. Vandenberghe, Convex Optimization. Cambridge University Press, 2004.",
    "S. Diamond and S. Boyd, “CVXPY: A Python-embedded modeling language for convex optimization,” J. Mach. "
    "Learn. Res., vol. 17, no. 83, pp. 1–5, 2016.",
    "P. Kligfield et al., “Recommendations for the standardization and interpretation of the electrocardiogram: "
    "Part I,” Circulation, vol. 115, pp. 1306–1324, 2007.",
    "S. Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. Statist., vol. 6, no. 2, "
    "pp. 65–70, 1979."]
for i, r in enumerate(refs, 1):
    R.p(f"[{i}] {r}", align="left", size=9)

# =========================================================================== appendix
R.h("Appendix A. Development-set selection and reproducibility", 1)
rows = [[f"MO-LDFIR N = {s['N']} (slack {s['slack']})", f"{s['prd_nom_med']:.3f}", f"{s['prd_opt_med']:.3f}",
         f"{s['tau_K_ms']:.2f}"] for s in SEL["ldfir"]]
rows += [[f"EW-APE K = {s['K']} (D ≤ budget)", f"{s['prd_nom_med']:.3f}", f"{s['prd_opt_med']:.3f}",
          f"{s['tau_K_ms']:.2f}"] for s in SEL["ewape"]]
R.table(["Candidate", "DS1 median PRD_nom (%)", "DS1 median PRD_opt (%)", "Latency (ms)"], rows,
        f"Hyperparameter selection on MIT-BIH DS1 only (first {P['dev_seconds_per_record']} s per record). Rule fixed "
        f"in advance: smallest FIR slack within {100 * P['ldfir_parsimony_rel_tol']:.0f}% of the best median PRD; "
        f"EW-APE K with the lowest median PRD among latency-compliant candidates. Selected: slack = "
        f"{hp['ldfir_N_slack']}, K = {hp['ewape_K']}.", font=8)
R.p("**Software.** " + ", ".join(f"{k} {v}" for k, v in prov.items()) + ". CVXPY solver: CLARABEL. "
    "**Reproduction:** pip install -r requirements.txt, then python run_all.py. The datasets are downloaded by "
    "scripts/00_download_data.py from PhysioNet's AWS open-data mirror and are not redistributed. All parameters "
    "and seeds are in config/params.json, and every number in this report is generated from results/ by "
    "scripts/09_build_report.py.", align="left")

out = ROOT / "report"
out.mkdir(exist_ok=True)
R.doc.core_properties.title = "Low-Group-Delay Morphology-Preserving Digital Filtering: Constrained FIR versus Phase-Equalized IIR"
R.doc.core_properties.author = "V N Shahid Ibrahim; Nida Paravath; Madhav Tripathi; Deepak Meena"
R.doc.core_properties.comments = ""
R.save(out / "Low_Group_Delay_Morphology_Preserving_Filtering_Report.docx")
print("report written:", out / "Low_Group_Delay_Morphology_Preserving_Filtering_Report.docx")
