"""Dataset access (MIT-BIH Arrhythmia, PTB-XL) and the common pre-processing stage."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Inter-patient split of de Chazal et al., IEEE TBME 51(7):1196-1206, 2004 (paced records
# 102, 104, 107, 217 excluded).  DS1 = development (all tuning), DS2 = held-out test.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
       "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
       "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


def remove_baseline(x, fs, fc=0.5):
    """Common offline pre-stage applied identically to every filter's input *and* to the
    reference: zero-phase 2nd-order Butterworth high-pass (forward-backward). It removes
    baseline wander so that the low-pass study is not dominated by sub-Hz drift. It is
    NOT part of any real-time filter under comparison."""
    sos = signal.butter(2, fc, "highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def load_mitdb(record: str, seconds=None, channel="MLII"):
    import wfdb
    path = str(DATA / "mitdb" / record)
    rec = wfdb.rdrecord(path, sampto=None if seconds is None else int(seconds * 360))
    ch = rec.sig_name.index(channel) if channel in rec.sig_name else 0
    x = rec.p_signal[:, ch].astype(float)
    ann = wfdb.rdann(path, "atr", sampto=len(x))
    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    return {"x": x, "fs": float(rec.fs), "beats": np.asarray(ann.sample)[keep],
            "symbols": np.asarray(ann.symbol)[keep], "lead": rec.sig_name[ch], "id": record}


def ptbxl_records(n=None):
    import csv
    rows = list(csv.DictReader(open(DATA / "ptbxl" / "ptbxl_database.csv", encoding="utf-8")))
    rows = [r for r in rows if r["strat_fold"] == "10"]
    rows.sort(key=lambda r: int(float(r["ecg_id"])))
    rows = [r for r in rows if (DATA / "ptbxl" / (r["filename_hr"] + ".dat")).exists()]
    return rows[:n] if n else rows


def load_ptbxl(row, lead="II"):
    import wfdb
    rec = wfdb.rdrecord(str(DATA / "ptbxl" / row["filename_hr"]))
    ch = rec.sig_name.index(lead)
    return {"x": rec.p_signal[:, ch].astype(float), "fs": float(rec.fs), "lead": lead,
            "id": str(int(float(row["ecg_id"]))), "scp": row.get("scp_codes", "")}


def detect_r_peaks(x, fs):
    """Simple energy-based QRS detector (band-pass 5-20 Hz, squared, 120-ms integration,
    adaptive threshold), then the largest |x| within +-60 ms. Used only on the *reference*
    signal for datasets without beat annotations (PTB-XL)."""
    sos = signal.butter(2, [5, 20], "bandpass", fs=fs, output="sos")
    e = signal.sosfiltfilt(sos, x) ** 2
    e = np.convolve(e, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same")
    pk, _ = signal.find_peaks(e, distance=int(0.3 * fs), height=0.25 * np.percentile(e, 99))
    hw = int(0.06 * fs)
    out = []
    for p in pk:
        lo, hi = max(0, p - hw), min(len(x), p + hw + 1)
        out.append(lo + int(np.argmax(np.abs(x[lo:hi]))))
    return np.unique(np.array(out, dtype=int))


def peak_polarity(x, peaks):
    if len(peaks) == 0:
        return 1
    return 1 if np.median(x[peaks]) >= 0 else -1


def welch_psd(x, fs, nperseg=None):
    nperseg = nperseg or int(2 ** np.round(np.log2(10 * fs)))
    return signal.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
