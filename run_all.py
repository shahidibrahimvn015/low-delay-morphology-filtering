"""Reproduce every result, table and figure of the report:  python run_all.py [--skip-download]"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["00_download_data.py", "01_dev_design.py", "02_pareto.py", "03_synthetic.py", "04_mitdb_test.py",
          "05_ptbxl_test.py", "06_robustness.py", "07_stats.py", "08_figures.py"]

if __name__ == "__main__":
    for s in STAGES:
        if s.startswith("00") and "--skip-download" in sys.argv:
            continue
        print(f"\n===== {s} =====", flush=True)
        log = ROOT / "results" / f"log_{s[:-3]}.txt"
        log.parent.mkdir(exist_ok=True)
        with open(log, "w") as fh:
            r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / s)], stdout=fh, stderr=subprocess.STDOUT)
        print(log.read_text()[-1500:])
        if r.returncode:
            sys.exit(f"stage {s} failed (see {log})")
