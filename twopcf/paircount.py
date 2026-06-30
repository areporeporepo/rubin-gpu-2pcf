"""GPU/CPU angular pair counting and the Landy-Szalay estimator.

The angular two-point correlation function w(theta) is the summary statistic
at the heart of galaxy clustering (the "2" in a 3x2pt analysis). We measure it
with the Landy-Szalay estimator:

    w(theta) = (DD - 2 DR + RR) / RR

where DD, DR, RR are normalized counts of data-data, data-random and
random-random pairs in bins of angular separation.

Pair counting is O(N^2) and embarrassingly parallel -- the textbook GPU win.
This module provides two *identical-algorithm* backends so the GPU result can
be validated bit-for-bit against the CPU one:

    backend='cpu'  -> NumPy
    backend='gpu'  -> CuPy   (install cupy-cuda12x on the A100 box)

A separate, independent reference (TreeCorr, tree-based) lives in baseline.py.
Three independent implementations agreeing is our correctness gate.
"""

from __future__ import annotations

import numpy as np

try:  # CuPy is only present on the GPU box; keep the module importable without it.
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover - exercised only off-GPU
    cp = None
    _HAS_CUPY = False


def has_gpu() -> bool:
    """True if CuPy is importable and at least one CUDA device is visible."""
    if not _HAS_CUPY:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover
        return False


def log_edges(min_sep_deg: float, max_sep_deg: float, nbins: int) -> np.ndarray:
    """Log-spaced bin edges in degrees, matching TreeCorr's binning convention."""
    return np.logspace(np.log10(min_sep_deg), np.log10(max_sep_deg), nbins + 1)


def bin_centers(edges: np.ndarray) -> np.ndarray:
    """Geometric bin centers (TreeCorr reports the nominal/rnom this way)."""
    return np.sqrt(edges[:-1] * edges[1:])


def radec_to_unit(ra_deg, dec_deg, xp=np):
    """(RA, Dec) in degrees -> (N, 3) unit vectors on the sphere."""
    ra = xp.deg2rad(xp.asarray(ra_deg, dtype=xp.float64))
    dec = xp.deg2rad(xp.asarray(dec_deg, dtype=xp.float64))
    cd = xp.cos(dec)
    x = cd * xp.cos(ra)
    y = cd * xp.sin(ra)
    z = xp.sin(dec)
    return xp.stack([x, y, z], axis=1)


def _hist_pairs(vecs_a, vecs_b, edges_deg, xp, chunk):
    """Histogram angular separations (deg) over all a-b pairs, in row chunks.

    cos(separation) is just the dot product of unit vectors, so a chunk of rows
    against all of B is a single (chunk x Nb) matmul -- this is what puts the
    work on the GPU's BLAS units.
    """
    edges = xp.asarray(edges_deg, dtype=xp.float64)
    nbins = edges.size - 1
    counts = xp.zeros(nbins, dtype=xp.float64)
    n_a = vecs_a.shape[0]
    for i in range(0, n_a, chunk):
        block = vecs_a[i : i + chunk]            # (c, 3)
        dots = block @ vecs_b.T                  # (c, Nb) cos(separation)
        dots = xp.clip(dots, -1.0, 1.0)
        theta = xp.rad2deg(xp.arccos(dots)).ravel()
        c, _ = xp.histogram(theta, bins=edges)
        counts = counts + c
    return counts


def count_auto(vecs, edges_deg, xp=np, chunk=4096):
    """Auto-pair counts (e.g. DD, RR). Self-pairs sit at theta=0, below the
    smallest bin edge, so the full matrix divided by 2 gives unordered pairs."""
    return _hist_pairs(vecs, vecs, edges_deg, xp, chunk) / 2.0


def count_cross(vecs_a, vecs_b, edges_deg, xp=np, chunk=4096):
    """Cross-pair counts (e.g. DR). No symmetry factor, no self-pairs."""
    return _hist_pairs(vecs_a, vecs_b, edges_deg, xp, chunk)


def landy_szalay(dd, dr, rr, n_d, n_r):
    """Landy-Szalay w(theta) from raw (unnormalized) pair counts per bin."""
    n_d = float(n_d)
    n_r = float(n_r)
    dd_norm = dd / (n_d * (n_d - 1.0) / 2.0)
    rr_norm = rr / (n_r * (n_r - 1.0) / 2.0)
    dr_norm = dr / (n_d * n_r)
    return (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm


def w_theta(ra_d, dec_d, ra_r, dec_r, edges_deg, backend="cpu", chunk=4096):
    """Compute w(theta) via Landy-Szalay on a chosen backend.

    Returns (w, counts) where counts is a dict of raw DD/DR/RR arrays (NumPy).
    """
    if backend == "gpu":
        if not has_gpu():
            raise RuntimeError("backend='gpu' requested but no CUDA device/CuPy found")
        xp = cp
    elif backend == "cpu":
        xp = np
    else:
        raise ValueError(f"unknown backend {backend!r}")

    vd = radec_to_unit(ra_d, dec_d, xp)
    vr = radec_to_unit(ra_r, dec_r, xp)
    dd = count_auto(vd, edges_deg, xp, chunk)
    rr = count_auto(vr, edges_deg, xp, chunk)
    dr = count_cross(vd, vr, edges_deg, xp, chunk)

    if xp is cp:  # bring results back to host
        dd, dr, rr = cp.asnumpy(dd), cp.asnumpy(dr), cp.asnumpy(rr)

    w = landy_szalay(dd, dr, rr, len(ra_d), len(ra_r))
    return w, {"dd": dd, "dr": dr, "rr": rr}
