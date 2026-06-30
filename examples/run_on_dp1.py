"""STEP 2 — run on a GPU box (Colab/GCP) AFTER downloading the DP1 CSV from the RSP.

Measures w(theta) on the REAL DP1 catalog with the GPU engine, against
cone-matched randoms, and saves a figure.

    python examples/run_on_dp1.py dp1_ECDFS.csv 53.13 -28.10 1.0

Args: <csv> <ra0> <dec0> <radius_deg>   (the field center/radius used to extract)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twopcf.catalog import load_catalog_csv, make_randoms_cone  # noqa: E402
from twopcf.grid import w_theta_grid  # noqa: E402
from twopcf.paircount import bin_centers, has_gpu, log_edges  # noqa: E402


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "dp1_ECDFS.csv"
    ra0 = float(sys.argv[2]) if len(sys.argv) > 2 else 53.13
    dec0 = float(sys.argv[3]) if len(sys.argv) > 3 else -28.10
    radius = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    max_sep = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5  # deg
    nsub = int(sys.argv[6]) if len(sys.argv) > 6 else 0  # 0 = use all (GPU)

    ra, dec = load_catalog_csv(csv)
    backend = "gpu" if has_gpu() else "cpu"
    if nsub and len(ra) > nsub:  # keep CPU runs tractable; w(theta) unbiased under random subsampling
        idx = np.random.default_rng(0).choice(len(ra), nsub, replace=False)
        ra, dec = ra[idx], dec[idx]
        print(f"subsampled to {nsub} of the full catalog")
    print(f"DP1 objects used: {len(ra)}   backend: {backend}   max_sep: {max_sep} deg")

    # cone-matched randoms (3x data); see make_randoms_cone caveat re: real mask
    ra_r, dec_r = make_randoms_cone(ra0, dec0, radius, 3 * len(ra), seed=1)

    edges = log_edges(0.002, max_sep, 12)
    theta = bin_centers(edges)
    w, _ = w_theta_grid(ra, dec, ra_r, dec_r, edges, backend=backend,
                        block=4096, ncells=128)

    print(" theta(deg)   w(theta)")
    for t, wi in zip(theta, w):
        print(f"  {t:8.4f}  {wi:9.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(theta, w, "o-", color="C2", label=f"DP1 (real) — {len(ra)} objects")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$w(\theta)$")
    ax.set_title(f"Angular clustering of real Rubin DP1 galaxies ({backend.upper()})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("dp1_wtheta.png", dpi=140)
    print("wrote dp1_wtheta.png")


if __name__ == "__main__":
    main()
