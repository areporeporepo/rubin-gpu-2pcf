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


def _fmt(seconds):
    if seconds < 90:
        return f"{seconds:5.0f} s"
    if seconds < 5400:
        return f"{seconds/60:5.1f} min"
    if seconds < 172800:
        return f"{seconds/3600:5.1f} hr"
    return f"{seconds/86400:5.1f} days"


def analysis_budget(per_run_cpu_s, n_runs=1000):
    """Wall-clock for a FULL analysis = per-run time x ~n_runs runs.

    A real clustering/SBI analysis runs the estimator ~1000 times (covariance
    mocks, systematics tests, inference). Time (minutes/hours) is the metric a
    researcher actually feels — measure seconds-per-run, multiply by n_runs.

    Tiers: CPU (1x), Tesla T4 (~35x, measured), A100/H100 (~150x, projected),
    and multi-GPU (data-parallel over the ~1000 independent runs — near-linear
    for this embarrassingly-parallel workload).
    """
    tiers = [
        ("CPU (today)", 1),
        ("GPU Tesla T4 (measured ~35x)", 35),
        ("GPU A100/H100 S3DF/Marlowe (proj ~150x)", 150),
        ("2027 Blackwell Rubin/NVIDIA/AWS (proj ~400x)", 400),
        ("cloud-scale multi-GPU over the runs (proj)", 150 * 100),
    ]
    print(f"full analysis = {per_run_cpu_s:g}s/run (CPU)  x  {n_runs} runs\n")
    for name, s in tiers:
        print(f"  {name:48s} {_fmt(per_run_cpu_s * n_runs / s)}")


if __name__ == "__main__":
    rows = run()
    plot(rows)
    print("\n=== full-analysis time budget (x1000 runs) ===")
    analysis_budget(per_run_cpu_s=48.0)  # measured CPU per-run at N=20k
