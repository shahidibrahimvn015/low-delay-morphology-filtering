# Low-Group-Delay Morphology-Preserving Digital Filtering: Constrained FIR versus Phase-Equalized IIR

**Authors:** V N Shahid Ibrahim (ED24B064), Nida Paravath (ED24B062), Madhav Tripathi (ED24B059), Deepak Meena (ED24B045) - Department of Engineering Design, IIT Madras.

ED5017 DSP course research project. Reproducible Python code, parameter files, dataset instructions and
the generated report (`report/`).
Repository: https://github.com/shahidibrahimvn015/low-delay-morphology-filtering

## Research question
For a fixed attenuation requirement and a strict latency budget, which filter architecture best preserves
waveform morphology?

## Contribution (separated from the baselines)
1. **Morphology error model.** For an input with PSD S(ω), expected output error w.r.t. a delayed zero-phase
   reference is  `(1/π)∫S|H − R e^{-jωD}|²dω`. In the passband this splits into a magnitude term plus a phase
   term, and by Cauchy–Schwarz the phase term is bounded by a **group-delay error weighted by the kernel
   K(ν)=∫_ν^{ωp} ω S(ω) dω**. This gives `J = αE_mag + βE_τg + γD` with α=β=1 and an explicit, derived
   passband group-delay metric (`src/lowdelay/analysis.py`).
2. **MO-LDFIR**: a constrained low-delay FIR that minimises the PSD-weighted morphology error. Second-order-cone
   constraints guarantee the matched magnitude spec. The problem is convex (SOCP) for every latency D.
3. **EW-APE**: an all-pass equaliser for the IIR that minimises the same exact morphology error, with a latency
   price γ or a hard latency bound.
4. A matched-spec benchmark against five reproduced baselines (Parks–McClellan LP-FIR, minimum-phase FIR,
   elliptic and Butterworth IIR, IIR plus classical flat-delay all-pass equaliser) and the uniform-weight
   low-delay FIR. It includes Pareto fronts (attenuation × group-delay/morphology error × latency × cost),
   synthetic ground truth, two open ECG datasets, ablation, quantisation and statistics.

## Layout
```
config/            spec_ecg.json, spec_ppg.json (filter specs), params.json (all hyper-parameters, seeds)
src/lowdelay/      specs, filters (analytic GD), analysis (metrics, J), designs, signals, metrics, data, bank
scripts/00..09     pipeline stages (see below)
results/           JSON/CSV outputs and logs        figures/   all figures (PNG)
report/            generated journal-style report (.docx)
```

## Reproduce
```
pip install -r requirements.txt
python run_all.py                 # ~1.5-2 h on a laptop; or run the stages individually:
python scripts/00_download_data.py   # MIT-BIH (48 records) + PTB-XL fold-10 subset (300 records), ~120 MB
python scripts/01_dev_design.py      # PSD weight + hyper-parameter selection on MIT-BIH DS1 ONLY; final bank
python scripts/02_pareto.py          # Pareto sweeps (attenuation 30-60 dB, transition 5-40 Hz)
python scripts/03_synthetic.py       # Monte-Carlo with exact ground truth, sweeps, transients, PPG case
python scripts/04_mitdb_test.py      # held-out MIT-BIH DS2 (inter-patient) test
python scripts/05_ptbxl_test.py      # independent PTB-XL test (500 Hz, no re-tuning)
python scripts/06_robustness.py      # quantisation, ablation, latency-budget sweep
python scripts/07_stats.py           # Wilcoxon signed-rank + Holm, bootstrap CIs
python scripts/08_figures.py
python scripts/09_build_report.py
```

## Data partitions (fixed before any tuning)
* **Development:** MIT-BIH DS1 (de Chazal et al. 2004; 22 records). Used for the PSD weight, the FIR length
  slack and the EW-APE section count.
* **Test 1:** MIT-BIH DS2 (22 different patients). Never used for tuning.
* **Test 2:** PTB-XL v1.0.1, recommended test fold 10, first 300 records by ecg_id, lead II, 500 Hz. Filters
  are redesigned from the same physical spec and locked rules. Nothing is tuned on this set.
* **Synthetic:** disjoint seed ranges for development (PSD) and Monte-Carlo testing (see `params.json`).

## Datasets and licences
* MIT-BIH Arrhythmia Database v1.0.0. Moody & Mark, IEEE EMB Mag 20(3):45-50, 2001; Goldberger et al.,
  Circulation 101(23):e215-e220, 2000. Licence ODC-By 1.0.
* PTB-XL v1.0.1. Wagner et al., Scientific Data 7:154, 2020. Licence CC BY 4.0.

Files are fetched from PhysioNet's AWS open-data mirror (`physionet-open.s3.amazonaws.com`), which holds
byte-identical copies of physionet.org. The data are **not** redistributed with this package.

## Conventions
f [Hz], fs [Hz], ω = 2πf/fs [rad/sample]. Delays are reported in samples and ms. Group delay is computed
analytically and cross-checked against the derivative of the unwrapped phase (Fig. 3a). It is evaluated
only in the passband. Forward-backward (zero-phase) filtering is used only to build the offline reference
and the common baseline pre-stage; it is never one of the compared real-time filters.
