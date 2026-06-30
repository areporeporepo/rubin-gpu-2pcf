"""Tomographic angular clustering -- the Rubin analog of DESI's 3D clustering.

DESI has spectroscopic redshifts -> 3D clustering. Rubin has imaging -> you split
galaxies into photometric-redshift (photo-z) bins and measure the angular
correlation w(theta) in each bin. The set of per-bin w(theta) is the
density-density leg of a tomographic 3x2pt analysis (what TXPipe computes).

This module:
- make_tomographic_mock : galaxies in N photo-z bins, each with its own (known)
  clustering amplitude, so per-bin measurements can be validated against theory.
- w_theta_tomographic   : measure w(theta) per photo-z bin with the GPU grid counter.

Randoms share the angular footprint across bins (selection is angular-uniform
here), so one random catalog serves every bin.
"""

from __future__ import annotations

import numpy as np

from .catalog import make_clustered_catalog
from .grid import w_theta_grid


def make_tomographic_mock(n_bins=3, per_bin_n=6000, box_deg=(3.0, 3.0),
                          center=(60.0, -30.0), parents_per_bin=(150, 300, 600),
                          scatter_deg=0.03, frac_clustered=0.6, seed=0):
    """Galaxies in `n_bins` photo-z bins with distinct clustering amplitudes.

    Fewer parents -> higher clustering (Thomas xi ~ 1/parent_density), so
    parents_per_bin=(150,300,600) gives high->low amplitude across bins, like
    real tomographic samples. Returns (ra, dec, zbin, photoz, parents_per_bin).
    """
    if len(parents_per_bin) < n_bins:
        parents_per_bin = tuple(parents_per_bin) + (parents_per_bin[-1],) * n_bins
    z_centers = np.linspace(0.3, 0.3 + 0.4 * (n_bins - 1), n_bins)
    ras, decs, zbins, photoz = [], [], [], []
    for i in range(n_bins):
        ra, dec = make_clustered_catalog(
            n_target=per_bin_n, box_deg=box_deg, center=center,
            n_parents=parents_per_bin[i], scatter_deg=scatter_deg,
            frac_clustered=frac_clustered, seed=seed + i,
        )
        rng = np.random.default_rng(seed + 100 + i)
        ras.append(ra)
        decs.append(dec)
        zbins.append(np.full(len(ra), i, dtype=int))
        photoz.append(rng.normal(z_centers[i], 0.05, len(ra)))
    return (np.concatenate(ras), np.concatenate(decs),
            np.concatenate(zbins), np.concatenate(photoz),
            tuple(parents_per_bin[:n_bins]))


def w_theta_tomographic(ra, dec, zbin, ra_r, dec_r, edges_deg, backend="cpu",
                        block=2048, ncells=96):
    """Per-photo-z-bin auto w(theta). Returns {bin_index: w_array}.

    The diagonal of the tomographic 3x2pt density-density matrix.
    """
    out = {}
    for b in np.unique(zbin):
        m = zbin == b
        w, _ = w_theta_grid(ra[m], dec[m], ra_r, dec_r, edges_deg,
                            backend=backend, block=block, ncells=ncells)
        out[int(b)] = w
    return out
