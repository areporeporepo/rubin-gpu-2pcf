"""CPU-vs-GPU scaling benchmark for the w(theta) pair-counting engine.

Honest measurement: we time the *same brute-force algorithm* on CPU (NumPy) and
GPU (CuPy) across a range of catalog sizes, plus TreeCorr (smart CPU tree) as the
real-world reference. The interesting, honest result is the crossover -- GPU
brute force beats the clever CPU tree above some N, and that N is exactly where
LSST data volumes land.
"""

from __future__ import annotations

import time

import numpy as np

from . import baseline
from .catalog import make_clustered_catalog, make_randoms
from .paircount import has_gpu, log_edges, w_theta


def _time(fn, repeat=3):
    """Best-of-`repeat` wall time (seconds). Warms up once first."""
    fn()  # warm-up (JIT/allocation/transfer)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run(sizes=(2000, 5000, 10000, 20000, 50000), min_sep=0.003, max_sep=1.0, nbins=12,
        rand_factor=1, include_treecorr=True):
    """Benchmark across catalog sizes. Returns a list of per-size dicts."""
    edges = log_edges(min_sep, max_sep, nbins)
    have_gpu = has_gpu()
    rows = []
    for n in sizes:
        ra_d, dec_d = make_clustered_catalog(n_target=n, seed=0)
        ra_r, dec_r = make_randoms(n * rand_factor, seed=1)

        t_cpu = _time(lambda: w_theta(ra_d, dec_d, ra_r, dec_r, edges, backend="cpu"))
        t_gpu = (
            _time(lambda: w_theta(ra_d, dec_d, ra_r, dec_r, edges, backend="gpu"))
            if have_gpu
            else None
        )
        t_tc = None
        if include_treecorr:
            try:
                t_tc = _time(
                    lambda: baseline.w_theta_treecorr(
                        ra_d, dec_d, ra_r, dec_r, min_sep, max_sep, nbins
                    )
                )
            except Exception as exc:  # treecorr missing -> skip gracefully
                print(f"  [treecorr skipped: {exc}]")

        speedup = (t_cpu / t_gpu) if t_gpu else None
        rows.append(
            dict(n=len(ra_d), cpu_s=t_cpu, gpu_s=t_gpu, treecorr_s=t_tc, gpu_speedup=speedup)
        )
        msg = f"N={len(ra_d):>7}  cpu={t_cpu:8.3f}s"
        if t_gpu:
            msg += f"  gpu={t_gpu:8.3f}s  speedup={speedup:6.1f}x"
        if t_tc:
            msg += f"  treecorr={t_tc:8.3f}s"
        print(msg)
    return rows


def plot(rows, path="benchmark_scaling.png"):
    """Log-log time-vs-N plot of CPU / GPU / TreeCorr."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n, [r["cpu_s"] for r in rows], "o-", label="CPU brute force (NumPy)")
    if any(r["gpu_s"] for r in rows):
        ax.plot(n, [r["gpu_s"] for r in rows], "s-", label="GPU brute force (CuPy)")
    if any(r["treecorr_s"] for r in rows):
        ax.plot(n, [r["treecorr_s"] for r in rows], "^-", label="TreeCorr (CPU tree)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("catalog size N")
    ax.set_ylabel("wall time [s]")
    ax.set_title("Angular 2PCF pair counting: CPU vs GPU scaling")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    rows = run()
    plot(rows)
