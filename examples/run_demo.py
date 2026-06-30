"""End-to-end demo: measure w(theta) on a synthetic Rubin-like field, validate
the GPU result against the CPU and TreeCorr, and save the figure.

    python examples/run_demo.py

Produces:
    wtheta_demo.png        -- w(theta) from CPU / GPU / TreeCorr, overlaid
    (run twopcf/bench.py separately for the scaling plot)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twopcf import baseline  # noqa: E402
from twopcf.catalog import make_clustered_catalog, make_randoms  # noqa: E402
from twopcf.paircount import bin_centers, has_gpu, log_edges, w_theta  # noqa: E402

MIN_SEP, MAX_SEP, NBINS = 0.003, 1.0, 12  # degrees


def main():
    ra_d, dec_d = make_clustered_catalog(n_target=20000, seed=0)
    ra_r, dec_r = make_randoms(60000, seed=1)
    edges = log_edges(MIN_SEP, MAX_SEP, NBINS)
    theta = bin_centers(edges)

    w_cpu, _ = w_theta(ra_d, dec_d, ra_r, dec_r, edges, backend="cpu")
    print("CPU  w(theta):", np.array2string(w_cpu, precision=4))

    w_gpu = None
    if has_gpu():
        w_gpu, _ = w_theta(ra_d, dec_d, ra_r, dec_r, edges, backend="gpu")
        max_rel = np.nanmax(np.abs(w_gpu - w_cpu) / (np.abs(w_cpu) + 1e-12))
        print(f"GPU vs CPU  max relative diff: {max_rel:.2e}  (identical algorithm)")
    else:
        print("GPU  : no CUDA device -> skipped (will run on the A100 box)")

    th_tc = w_tc = None
    try:
        th_tc, w_tc = baseline.w_theta_treecorr(
            ra_d, dec_d, ra_r, dec_r, MIN_SEP, MAX_SEP, NBINS
        )
        # compare on shared bins
        rel = np.nanmedian(np.abs(w_tc - w_cpu) / (np.abs(w_cpu) + 1e-12))
        print(f"TreeCorr vs CPU  median relative diff: {rel:.2e}")
    except Exception as exc:
        print(f"TreeCorr: skipped ({exc})")

    _plot(theta, w_cpu, w_gpu, th_tc, w_tc)


def _plot(theta, w_cpu, w_gpu, th_tc, w_tc, path="wtheta_demo.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(theta, w_cpu, "o-", label="CPU brute force (NumPy)")
    if w_gpu is not None:
        ax.plot(theta, w_gpu, "s--", label="GPU brute force (CuPy)")
    if w_tc is not None:
        ax.plot(th_tc, w_tc, "x:", label="TreeCorr (reference)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$w(\theta)$")
    ax.set_title(r"Angular two-point correlation function $w(\theta)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
