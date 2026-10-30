"""Stage 2: Pareto trade-off (attenuation x group-delay/morphology error x latency x cost).

For several matched specifications (stopband attenuation 30-60 dB; transition width 5-40 Hz)
every architecture is swept over its own design knob and evaluated with the frequency-domain
error model (PSD weight from MIT-BIH DS1 only):
    LP-FIR / MP-FIR       minimum length                     (1 point each)
    classical IIR         elliptic order n_min..n_min+2, Butterworth, Chebyshev-I
    Ell+AP-flat           K = 1..5 all-pass sections, flat-GD criterion
    EW-APE                K = 1..3, latency constraint D <= d_max over a grid; and gamma = 0
    MO-LDFIR / LD-flat    design delay D over a grid, N = N_min(D) + selected slack
Output: results/pareto.json
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from lowdelay import designs as d
from lowdelay.analysis import freq_metrics, psd_on_grid
from lowdelay.specs import load_spec
from lowdelay.util import CONFIG, RESULTS, load_json, load_params, save_json

KEYS = ["tau_K", "tau_K_ms", "prd_pred", "dtau_max_ms", "dtau_rms_W_ms", "atten_db", "dev_p",
        "ops_mult", "meets_spec", "E_mag", "E_tau", "E_phi", "max_pole_radius"]


def variants(base):
    out = []
    for a in (30, 40, 50, 60):
        out.append(("atten", a, base.with_(as_db=float(a), name=f"As={a} dB")))
    for tw in (5, 10, 20, 40):
        if tw == 20:
            continue  # identical to atten=40 variant
        out.append(("trans", tw, base.with_(fst_hz=base.fp_hz + tw, name=f"TW={tw} Hz")))
    return out


def _pt(spec, filt, arch, S, **par):
    m = freq_metrics(filt, spec, S)
    r = {k: m[k] for k in KEYS}
    r.update(arch=arch, label=filt.name, **par)
    return r


def job(args):
    """One (variant, architecture) sweep - executed in a worker process."""
    kind, val, spec, arch, S, hp = args
    pts = []
    try:
        if arch == "fixed":
            lp = d.lp_fir(spec)
            pts.append(_pt(spec, lp, "LP-FIR", S))
            pts.append(_pt(spec, d.to_minimum_phase(lp), "MP-FIR", S))
            ell = d.iir_classic(spec, "ellip")
            n0 = ell.meta["order"]
            for n in (n0, n0 + 1, n0 + 2):
                pts.append(_pt(spec, d.iir_classic(spec, "ellip", n), "Elliptic", S, order=n))
            pts.append(_pt(spec, d.iir_classic(spec, "butter"), "Butterworth", S))
            pts.append(_pt(spec, d.iir_classic(spec, "cheby1"), "Chebyshev-I", S))
        elif arch == "apflat":
            ell = d.iir_classic(spec, "ellip")
            for K in range(1, 6):
                pts.append(_pt(spec, d.iir_ap(spec, ell, K, "flat", seed=hp["seed"]), "Ell+AP-flat", S, K=K))
        elif arch == "ewape":
            ell = d.iir_classic(spec, "ellip")
            lat = spec.latency_samples
            for K in (1, 2, 3):
                pts.append(_pt(spec, d.iir_ap(spec, ell, K, "morph", S=S, seed=hp["seed"]), "EW-APE", S, K=K,
                               d_max=None))
                for dm in np.array([0.7, 1.0, 1.5, 2.0, 3.0, 4.5]) * lat:
                    f = d.iir_ap(spec, ell, K, "morph", S=S, d_max=float(dm), seed=hp["seed"], n_starts=8)
                    pts.append(_pt(spec, f, "EW-APE", S, K=K, d_max=float(dm)))
        elif arch.startswith("ldfir"):
            wt = "psd" if arch == "ldfir_psd" else "flat"
            name = "MO-LDFIR" if wt == "psd" else "LD-FIR-flat"
            lp = d.lp_fir(spec)
            d_lp = (len(lp.b) - 1) / 2
            grid = sorted({int(round(v)) for v in np.linspace(2, d_lp, 9)} | {int(spec.latency_samples)})
            for D in grid:
                fmin = d.ld_fir_min_length(spec, D, S, "psd")
                if fmin is None:
                    continue
                N = len(fmin.b) + hp["ldfir_N_slack"]
                f = d.ld_fir(spec, N, D, S, wt)
                if f is not None:
                    pts.append(_pt(spec, f, name, S, D=D, N=N, N_min=len(fmin.b)))
    except Exception as e:  # keep the sweep going; record the failure
        pts.append({"arch": arch, "error": repr(e)})
    for p in pts:
        p.update(variant_kind=kind, variant_val=val, variant=spec.name)
    return pts


def main():
    P = load_params()
    hp = load_json("hparams.json")
    base = load_spec(CONFIG / "spec_ecg.json")
    z = np.load(RESULTS / "psd_mitdb_ds1.npz")
    S = psd_on_grid(base, z["f"], z["pxx"], P["psd_floor"])
    tasks = [(k, v, s, a, S, hp) for (k, v, s) in variants(base)
             for a in ("fixed", "apflat", "ewape", "ldfir_psd", "ldfir_flat")]
    t0 = time.time()
    pts = []
    # the SOCPs of narrow-transition variants are memory hungry: keep the pool small
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(job, tasks):
            pts.extend(res)
            print(f"[{time.time()-t0:5.0f}s] {res[0]['variant'] if res else '?'} {res[0]['arch'] if res else ''}"
                  f" -> {len(res)} pts", flush=True)
            save_json({"spec": base.__dict__, "points": pts, "complete": False}, "pareto.json")
    save_json({"spec": base.__dict__, "points": pts, "complete": True}, "pareto.json")
    print("done", len(pts), "points")


if __name__ == "__main__":
    main()
