"""Download the open datasets used in this study (run once, ~120 MB).

Source: PhysioNet's AWS open-data mirror https://physionet-open.s3.amazonaws.com, which hosts
byte-identical copies of https://physionet.org/files/<db>/<version>/ .

  MIT-BIH Arrhythmia Database v1.0.0   (Moody & Mark 2001; Goldberger et al. 2000)
      licence: Open Data Commons Attribution License v1.0 (ODC-By 1.0)       -> data/mitdb
  PTB-XL v1.0.1  (Wagner et al., Sci. Data 7:154, 2020)
      licence: Creative Commons Attribution 4.0 (CC BY 4.0)                  -> data/ptbxl
      only the 500-Hz records of the recommended test fold (strat_fold = 10), first
      N_PTBXL records in ecg_id order, are downloaded (deterministic subset).

Usage:  python scripts/00_download_data.py [--ptbxl 300]
"""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com"
ROOT = Path(__file__).resolve().parents[1] / "data"


def fetch(url: str, dest: Path, retries: int = 4) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            return
        except Exception as e:  # pragma: no cover - network
            if k == retries - 1:
                raise RuntimeError(f"failed: {url}: {e}")


def text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", type=int, default=300, help="number of PTB-XL fold-10 records")
    a = ap.parse_args()
    jobs = []

    # ---- MIT-BIH
    mit = f"{BASE}/mitdb/1.0.0"
    recs = text(f"{mit}/RECORDS").split()
    (ROOT / "mitdb").mkdir(parents=True, exist_ok=True)
    (ROOT / "mitdb" / "RECORDS").write_text("\n".join(recs) + "\n")
    for r in recs:
        for ext in ("hea", "dat", "atr"):
            jobs.append((f"{mit}/{r}.{ext}", ROOT / "mitdb" / f"{r}.{ext}"))

    # ---- PTB-XL (fold 10 subset)
    ptb = f"{BASE}/ptb-xl/1.0.1"
    meta = ROOT / "ptbxl" / "ptbxl_database.csv"
    fetch(f"{ptb}/ptbxl_database.csv", meta)
    fetch(f"{ptb}/scp_statements.csv", ROOT / "ptbxl" / "scp_statements.csv")
    fetch(f"{ptb}/LICENSE.txt", ROOT / "ptbxl" / "LICENSE.txt")
    rows = list(csv.DictReader(io.StringIO(meta.read_text(encoding="utf-8"))))
    rows = [r for r in rows if r["strat_fold"] == "10"]
    rows.sort(key=lambda r: int(float(r["ecg_id"])))
    for r in rows[: a.ptbxl]:
        stem = r["filename_hr"]
        for ext in ("hea", "dat"):
            jobs.append((f"{ptb}/{stem}.{ext}", ROOT / "ptbxl" / f"{stem}.{ext}"))

    print(f"downloading {len(jobs)} files ...")
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))
    print("done:", sum(p.stat().st_size for _, p in jobs) / 1e6, "MB")


if __name__ == "__main__":
    main()
