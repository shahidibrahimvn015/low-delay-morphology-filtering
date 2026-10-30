# Low-Group-Delay Morphology-Preserving Digital Filtering: Constrained FIR versus Phase-Equalized IIR

**Authors:** V N Shahid Ibrahim (ED24B064), Nida Paravath (ED24B062), Madhav Tripathi (ED24B059), Deepak Meena (ED24B045) - Department of Engineering Design, IIT Madras.

ED5017 Digital Signal Processing - research-oriented course project.
This repository contains the Python code, the parameter files, the dataset instructions, all results and
figures, and the journal-style technical report (`report/`).

Repository: https://github.com/shahidibrahimvn015/low-delay-morphology-filtering

---

## 1. What this project is about

Biomedical signals such as the ECG and PPG are filtered before they are analysed, to remove noise. The
*shape* of the waveform (QRS amplitude and width, ST level, T wave, pulse shape) is what carries the
information, so the filter must remove noise **without changing the shape**. Many applications (arrhythmia
alarms, R-wave triggering, wearables) also need the filtered signal **quickly**, i.e. with a small delay.

These two goals conflict:

| Architecture | Shape | Delay |
|---|---|---|
| Linear-phase FIR | perfect (constant group delay) | large |
| Minimum-phase FIR / classical IIR | distorted (group delay varies with frequency) | small |
| IIR + all-pass phase equalizer | flatter group delay | equalizer *adds* delay |

**Research question:** for a fixed attenuation requirement and a strict latency budget, which filter
architecture best preserves waveform morphology?

All filters are compared under the **same specification** (ECG, fs = 360 Hz): passband 0-40 Hz with
0.5 dB ripple, stopband from 60 Hz with 40 dB attenuation, latency budget 20 ms.

## 2. Main idea and contributions

1. **Morphology-error model.** For an input with power spectral density S(ω), the expected error between the
   filter output and a delayed, undistorted reference is `(1/π)∫ S(ω) |H(e^jω) − R(ω) e^(−jωD)|² dω`.
   This splits into a magnitude term and a phase term, and the phase term is bounded by a **group-delay
   error weighted by the kernel K(ν) = ∫_ν^ωp ω S(ω) dω**. Group-delay error therefore matters most at low
   frequencies, where ECG energy lies. This derives the objective `J = α·E_mag + β·E_τg + γ·D` (α = β = 1)
   and an explicit passband group-delay metric (`src/lowdelay/analysis.py`).
2. **MO-LDFIR (proposed).** A low-delay FIR that minimises the PSD-weighted morphology error. It is posed
   as a convex second-order-cone program (SOCP); its constraints guarantee the matched magnitude
   specification, so it is globally optimal for each latency D.
3. **EW-APE (proposed).** An all-pass equalizer for the IIR that minimises the same morphology error, with a
   latency price γ or a hard latency bound.
4. **Matched-specification benchmark** against five reproduced baselines, with Pareto trade-offs,
   synthetic ground truth, two open ECG datasets, a PPG case, quantization, ablation and statistics.

### Filters compared

| Filter | Type | Role |
|---|---|---|
| LP-FIR | linear-phase FIR (Parks-McClellan), N = 33 | baseline |
| MP-FIR | minimum-phase FIR (zero reflection), N = 33 | baseline |
| Elliptic | elliptic IIR, order 5 | baseline |
| Butterworth | Butterworth IIR, order 13 | baseline |
| Ell+AP-flat | elliptic + classical flat-delay all-pass equalizer, K = 3 sections | baseline |
| LD-FIR-flat | low-delay FIR with uniform (unweighted) error, N = 71 | ablation |
| **MO-LDFIR** | morphology-optimal low-delay FIR, N = 71, D = 7 samples | **proposed** |
| **EW-APE** | elliptic + energy-weighted all-pass equalizer, K = 1 | **proposed** |

### Headline results

Median over records. Latency = energy-weighted group delay τ̄_K. PRD = morphology error after optimal
alignment (lower is better). Full results, statistics and discussion are in the report.

| Filter | Latency (ms) | Within 20 ms? | Multiplications / sample | PRD MIT-BIH DS2 (%) | PRD PTB-XL (%) |
|---|---|---|---|---|---|
| LP-FIR | 44.4 | no | 17 | 2.01 | 1.89 |
| MP-FIR | 9.5 | yes | 33 | 5.14 | 7.77 |
| Elliptic | 12.1 | yes | 9 | 6.94 | 11.18 |
| Butterworth | 30.3 | no | 21 | 7.93 | 11.98 |
| Ell+AP-flat | 79.1 | no | 21 | 3.10 | 4.37 |
| LD-FIR-flat | 19.7 | yes | 71 | 2.73 | 3.08 |
| **MO-LDFIR** | **19.5** | **yes** | 71 | **1.51** | **2.54** |
| EW-APE | 20.0 | yes | 13 | 6.63 | 10.69 |

**In short:** within the 20 ms budget, MO-LDFIR preserves morphology best (about 4.6× lower PRD than the
elliptic IIR on held-out MIT-BIH patients). All-pass equalization cannot help much under a tight budget,
because flattening group delay requires adding delay. The IIR remains preferable when arithmetic cost
dominates. Section 7 of the report gives the full design recommendations, and Section 7.1 the limitations
(e.g. very sharp QRS complexes).

---

## 3. Repository layout

```
config/
  spec_ecg.json        ECG filter specification (fs, band edges, ripple, attenuation, latency budget)
  spec_ppg.json        PPG filter specification
  params.json          all hyper-parameters and the random seed
src/lowdelay/          the Python package (imported by the scripts)
  specs.py             loads a spec, converts dB ripple/attenuation to δp, δs, frequency grids
  filters.py           FIR/IIR/all-pass filter classes, analytic group delay, filtering
  designs.py           all filter designs (baselines, MO-LDFIR SOCP, EW-APE, AP-flat)
  analysis.py          frequency-domain metrics, kernel K(ν), E_mag, E_τg, J, predicted PRD
  signals.py           synthetic ECG/PPG generator, noise, fractional delay
  metrics.py           PRD, R-peak timing/amplitude, ST error, step/impulse metrics
  data.py              MIT-BIH and PTB-XL loading, DS1/DS2 split, pre-processing
  bank.py              builds and saves the final filter bank
  util.py              paths, JSON I/O, software-version recording
  docx_util.py         helpers used to write the Word report
scripts/               the pipeline, stages 00-09 (see Section 5)
results/               every numerical output (JSON/CSV) and one log file per stage
figures/               all figures of the report (PNG, F01-F14)
report/                the technical report (.docx)
run_all.py             runs every stage in order
requirements.txt       exact package versions
```

## 4. How to run

### 4.1 Requirements
* Python 3.11 or newer recommended (developed and tested with Python 3.13.1 on Windows 11).
* About 150 MB of free disk space for the datasets, and an internet connection for the first run.
* Optional: Microsoft Word on Windows. The report builder uses Office's equation converter to write native
  Word equations; without it the report is still built, with plain-text equations.

### 4.2 Install
```
git clone https://github.com/shahidibrahimvn015/low-delay-morphology-filtering.git
cd low-delay-morphology-filtering
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

### 4.3 Run everything
```
python run_all.py                  # all stages, about 1.5-2 h on a laptop
python run_all.py --skip-download  # same, but reuse data/ already downloaded
```
`run_all.py` runs stages 00-09 in order, writes each stage's output to `results/log_<stage>.txt`, prints
the end of each log, and stops at the first stage that fails.

### 4.4 Run one stage at a time
Each stage reads the outputs of the previous ones, so run them in this order the first time. After that,
any single stage can be re-run on its own (e.g. only `09_build_report.py` after editing the report text).

```
python scripts/00_download_data.py           # optional: --ptbxl 300  (number of PTB-XL records)
python scripts/01_dev_design.py
python scripts/02_pareto.py
python scripts/03_synthetic.py
python scripts/04_mitdb_test.py
python scripts/05_ptbxl_test.py
python scripts/06_robustness.py
python scripts/07_stats.py
python scripts/08_figures.py
```

## 5. What each stage does

| Stage | What happens | Main outputs |
|---|---|---|
| `00_download_data` | Downloads MIT-BIH (48 records) and a fixed PTB-XL subset (300 records of test fold 10) from PhysioNet's open-data mirror (about 120 MB). Skips files already present. | `data/mitdb/`, `data/ptbxl/` |
| `01_dev_design` | Estimates the ECG power spectrum S(ω) on **MIT-BIH DS1 only**, selects the hyper-parameters (FIR length slack, EW-APE sections) on DS1, then designs the final filter bank and checks every filter against the specification. | `psd_mitdb_ds1.npz`, `hparams.json`, `dev_selection.json`, `bank_*.json`, `freq_metrics_*.json` |
| `02_pareto` | Redesigns every architecture for stopband attenuation 30-60 dB and transition width 5-40 Hz, giving the latency / group-delay error / morphology error / cost trade-off. | `pareto.json`, `pareto_summary.csv` |
| `03_synthetic` | Monte-Carlo tests on synthetic ECG and PPG with exact ground truth; checks the error model's predicted PRD against the measured PRD; noise and QRS-sharpness sweeps; step and impulse transients. | `synthetic_*.csv`, `synthetic_*.json` |
| `04_mitdb_test` | Held-out test on **MIT-BIH DS2** (22 patients never used for tuning), on clean and noise-stressed signals: PRD, R-peak timing and amplitude, ST-level error. | `mitdb_ds2.csv` |
| `05_ptbxl_test` | Independent test on PTB-XL at 500 Hz. Filters are redesigned from the same physical specification with the locked rules; nothing is re-tuned. | `ptbxl.csv` |
| `06_robustness` | Coefficient quantization (24/16/12/10 bits), ablation of each design choice, latency-budget sweep. | `robust_*.csv`, `robust_ablation.json` |
| `07_stats` | Medians and IQRs per record; paired Wilcoxon signed-rank tests (MO-LDFIR vs each filter) with Holm correction; bootstrap confidence intervals (2000 resamples). | `stats_summary.csv`, `stats_tests.csv` |
| `08_figures` | Draws all figures F01-F14. | `figures/*.png` |
| `09_build_report` | Writes the Word report. Every number and table is read from `results/`, so the report always matches the code. | `report/*.docx` |

### Figures
| Figure | Shows |
|---|---|
| F01 | Magnitude responses of all filters against the specification mask |
| F02 | Unwrapped phase and passband group delay |
| F03 | Analytic vs numerical group delay check; the kernel K(ν) |
| F04 | Pole-zero maps |
| F05 | Impulse and step responses |
| F06-F07 | Pareto maps vs stopband attenuation and transition width |
| F08 | Attenuation vs morphology/latency trade-off |
| F09 | Latency-budget sweep |
| F10 | Synthetic validation: predicted vs measured PRD, sweeps |
| F11 | Per-record box plots on MIT-BIH DS2 and PTB-XL |
| F12 | Example ECG waveforms after each filter |
| F13 | PPG case study |
| F14 | Quantization (finite word-length) behaviour |

## 6. Changing the settings

* **Filter specification:** edit `config/spec_ecg.json` (`fs`, `fp_hz` passband edge, `fst_hz` stopband edge,
  `ap_db` ripple, `as_db` attenuation, `latency_ms` budget) or `config/spec_ppg.json`.
* **Hyper-parameters** in `config/params.json`:

| Key | Meaning |
|---|---|
| `seed` | random seed for all synthetic signals and optimizer starts (2026) |
| `psd_floor` | floor added to the normalized PSD weight S(ω) |
| `dev_seconds_per_record` | seconds of each DS1 record used for development |
| `ldfir_N_slack_candidates` | extra FIR taps tried above the minimum feasible length |
| `ldfir_parsimony_rel_tol` | choose the shortest FIR within this relative PRD tolerance |
| `ewape_K_candidates` | numbers of all-pass sections tried for EW-APE |
| `apflat_K` | sections of the classical flat-delay equalizer |
| `n_starts` | random restarts of the all-pass optimizer |
| `allpass_r_max` | maximum all-pass pole radius (stability margin) |
| `mc_trials` | Monte-Carlo trials per synthetic condition |
| `quant_bits` | word lengths tested for coefficient quantization |

After changing a setting, re-run from `01_dev_design.py` onwards
(`python run_all.py --skip-download`).

## 7. Data partitions (fixed before any tuning)
* **Development:** MIT-BIH DS1 (de Chazal et al. 2004; 22 records). Used only for the PSD weight, the FIR
  length slack and the EW-APE section count.
* **Test 1:** MIT-BIH DS2 (22 different patients). Never used for tuning.
* **Test 2:** PTB-XL v1.0.1, recommended test fold 10, first 300 records by ecg_id, lead II, 500 Hz.
  Filters are redesigned from the same physical specification and locked rules. Nothing is tuned on this set.
* **Synthetic:** disjoint seed ranges for development (PSD) and Monte-Carlo testing (see `params.json`).

## 8. Datasets and licences
* **MIT-BIH Arrhythmia Database v1.0.0.** Moody & Mark, IEEE EMB Mag 20(3):45-50, 2001; Goldberger et al.,
  Circulation 101(23):e215-e220, 2000. Licence ODC-By 1.0. https://physionet.org/content/mitdb/1.0.0/
* **PTB-XL v1.0.1.** Wagner et al., Scientific Data 7:154, 2020. Licence CC BY 4.0.
  https://physionet.org/content/ptb-xl/1.0.1/

`scripts/00_download_data.py` fetches the files from PhysioNet's AWS open-data mirror
(`physionet-open.s3.amazonaws.com`), which holds byte-identical copies of physionet.org. The data are **not**
redistributed in this repository (`data/` is excluded by `.gitignore`).

## 9. Conventions
* f [Hz], fs [Hz], ω = 2πf/fs [rad/sample], 0 ≤ ω ≤ π. Delays are reported in samples and in ms.
* Group delay τ(ω) = −dφ/dω is computed analytically and cross-checked against the derivative of the
  unwrapped phase (Fig. 3a). It is evaluated **only in the passband**; stopband group-delay spikes are not
  treated as signal delay.
* Forward-backward (zero-phase) filtering is used only to build the offline reference and the common
  pre-processing stage. It is never one of the compared real-time filters.

## 10. Troubleshooting
| Problem | Fix |
|---|---|
| `PermissionError` when building the report | The .docx is open in Word. Close it and re-run `09_build_report.py`. |
| Download fails or stops | Check the internet connection and re-run `00_download_data.py`; finished files are skipped. |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| Equations appear as plain text in the report | Microsoft Word / Office is not installed; the report is still complete. |
| A stage fails | Read `results/log_<stage>.txt`; run the earlier stages first if their outputs are missing. |
