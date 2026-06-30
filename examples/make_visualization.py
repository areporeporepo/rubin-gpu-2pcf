"""Event-ready 'decode the universe' figure: GPU-rendered galaxy density field
(left) + the validated angular clustering w(theta) (right).

    python examples/make_visualization.py   ->  visualization.png

The density field uses a multi-scale mock (large + medium + small structure)
so it looks like real large-scale structure; w(theta) is measured by the same
GPU engine. Density binning/smoothing runs on the GPU when one is present.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twopcf.catalog import make_clustered_catalog, make_randoms  # noqa: E402
from twopcf.grid import w_theta_grid  # noqa: E402
from twopcf.paircount import bin_centers, has_gpu, log_edges  # noqa: E402
from twopcf.visualize import render  # noqa: E402

BOX, CENTER = (3.0, 3.0), (60.0, -30.0)


def rich_field():
    """Multi-scale structure: superpose large/medium/small clustering + a field."""
    comps = [
        make_clustered_catalog(n_target=45000, box_deg=BOX, center=CENTER,
                               n_parents=35, scatter_deg=0.16, frac_clustered=1.0, seed=1),
        make_clustered_catalog(n_target=45000, box_deg=BOX, center=CENTER,
                               n_parents=400, scatter_deg=0.05, frac_clustered=1.0, seed=2),
        make_clustered_catalog(n_target=25000, box_deg=BOX, center=CENTER,
                               n_parents=2500, scatter_deg=0.015, frac_clustered=1.0, seed=3),
        make_randoms(20000, box_deg=BOX, center=CENTER, seed=4),
    ]
    ra = np.concatenate([c[0] for c in comps])
    dec = np.concatenate([c[1] for c in comps])
    return ra, dec


def main():
    ra, dec = rich_field()
    backend = "gpu" if has_gpu() else "cpu"
    print(f"galaxies: {len(ra)}   backend: {backend}")

    # w(theta) on a subsample (keeps the figure fast); density map uses all points
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ra), size=12000, replace=False)
    ra_r, dec_r = make_randoms(24000, box_deg=BOX, center=CENTER, seed=99)
    edges = log_edges(0.004, 0.3, 12)
    theta = bin_centers(edges)
    w, _ = w_theta_grid(ra[idx], dec[idx], ra_r, dec_r, edges, backend=backend,
                        block=2048, ncells=96)

    render(ra, dec, theta, w, w_theory=None, density_backend=backend,
           n_pix=600, sigma_pix=4.0, path="visualization.png")


if __name__ == "__main__":
    main()
