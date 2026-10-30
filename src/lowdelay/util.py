"""Paths, persistence, provenance and plotting style."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
CONFIG = ROOT / "config"
for p in (RESULTS, FIGS):
    p.mkdir(exist_ok=True)


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def save_json(obj, name):
    (RESULTS / name).write_text(json.dumps(obj, indent=1, cls=_Enc))


def load_json(name):
    return json.loads((RESULTS / name).read_text())


def load_params():
    return json.loads((CONFIG / "params.json").read_text())


def provenance() -> dict:
    import scipy, matplotlib
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
         "matplotlib": matplotlib.__version__, "platform": platform.platform()}
    for mod in ("cvxpy", "wfdb", "pandas"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    return v


# consistent colours per architecture: fixed categorical order of a CVD-validated palette
# (adjacent CVD dE >= 9.1, normal-vision dE >= 19.6); markers give a second, colour-free cue
COLORS = {"LP-FIR": "#2a78d6", "MP-FIR": "#eb6834", "Elliptic": "#1baf7a", "Butterworth": "#eda100",
          "Ell+AP-flat": "#e87ba4", "LD-FIR-flat": "#008300", "EW-APE": "#6250d6",
          "MO-LDFIR": "#e34948", "Chebyshev-I": "#8a8a86", "Reference": "#0b0b0b"}
MARKERS = {"LP-FIR": "s", "MP-FIR": "v", "Elliptic": "o", "Butterworth": "D", "Ell+AP-flat": "P",
           "LD-FIR-flat": "x", "MO-LDFIR": "*", "EW-APE": "h"}


def plot_style():
    import matplotlib as mpl
    mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
                         "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                         "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight",
                         "figure.constrained_layout.use": True})


def _plain_log_ticks(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "log":
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(NullFormatter())


def savefig(fig, name):
    for ax in fig.axes:
        _plain_log_ticks(ax)
    fig.savefig(FIGS / f"{name}.png")
    import matplotlib.pyplot as plt
    plt.close(fig)
