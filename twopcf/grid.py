"""Block-pruned pair counting -- scales w(theta) past O(N^2) brute force.

The idea, and why it's *provably correct*:

1. Spatially sort points (tangent-plane grid) so consecutive points are near
   each other, then cut into blocks of `block` points.
2. For each block, compute a centroid direction and an angular radius (the
   farthest member from the centroid).
3. By the spherical triangle inequality, the closest possible pair between
   blocks I and J is at least  sep(C_I, C_J) - r_I - r_J. If that lower bound
   exceeds the largest separation bin, NO pair in (I, J) can contribute -> skip
   the whole block-pair. We never skip a block-pair that could contain a real
   pair, so the result is identical to brute force (verified in tests).

This is the same algorithm on CPU (xp=numpy) and GPU (xp=cupy); the GPU version
is validated against this CPU one and against brute force.

Brute force is fine up to ~1e4-1e5; this scales to DP1's ~2.3M objects (run
per field) where O(N^2) is hopeless.
"""

from __future__ import annotations

import numpy as np

from .paircount import landy_szalay, radec_to_unit

try:
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CUPY = False


def _tangent_basis(vecs, xp):
    """Two orthonormal vectors spanning the tangent plane at the mean direction."""
    d = xp.sum(vecs, axis=0)
    d = d / xp.linalg.norm(d)
    ref = xp.asarray([0.0, 0.0, 1.0])
    if abs(float(d[2])) > 0.9:  # mean near a pole -> pick a different reference
        ref = xp.asarray([1.0, 0.0, 0.0])
    e1 = ref - d * xp.dot(ref, d)
    e1 = e1 / xp.linalg.norm(e1)
    e2 = xp.cross(d, e1)
    return e1, e2


def _grid_order(vecs, ncells, xp):
    """argsort that makes consecutive points spatially compact (for tight blocks)."""
    e1, e2 = _tangent_basis(vecs, xp)
    u = vecs @ e1
    v = vecs @ e2

    def cell_idx(a):
        a_min = a.min()
        t = (a - a_min) / (a.max() - a_min + 1e-12)
        return xp.clip((t * ncells).astype(xp.int64), 0, ncells - 1)

    cell = cell_idx(u) * ncells + cell_idx(v)
    return xp.argsort(cell)


def _block_bounds(n, block):
    return [(i, min(i + block, n)) for i in range(0, n, block)]


def _block_centroids(vecs, bounds, xp):
    """Centroid unit vector and angular radius (rad) for each block."""
    nb = len(bounds)
    centroids = xp.zeros((nb, 3))
    radii = xp.zeros(nb)
    for k, (s, e) in enumerate(bounds):
        seg = vecs[s:e]
        c = xp.sum(seg, axis=0)
        c = c / xp.linalg.norm(c)
        centroids[k] = c
        radii[k] = xp.max(xp.arccos(xp.clip(seg @ c, -1.0, 1.0)))
    return centroids, radii


def _surviving_pairs(centroids, radii, max_sep_rad, xp, upper_only):
    """Host-side list of block-pairs that could contain a pair <= max_sep."""
    sep = xp.arccos(xp.clip(centroids @ centroids.T, -1.0, 1.0))
    lower_bound = sep - radii[:, None] - radii[None, :]
    keep = lower_bound <= max_sep_rad
    keep = cp.asnumpy(keep) if xp is cp else np.asarray(keep)
    if upper_only:
        keep = np.triu(keep)
    return np.argwhere(keep)


def _surviving_pairs_cross(ca, ra_, cb, rb, max_sep_rad, xp):
    sep = xp.arccos(xp.clip(ca @ cb.T, -1.0, 1.0))
    lower_bound = sep - ra_[:, None] - rb[None, :]
    keep = lower_bound <= max_sep_rad
    keep = cp.asnumpy(keep) if xp is cp else np.asarray(keep)
    return np.argwhere(keep)


def _hist_block(v_i, v_j, edges, xp):
    dots = xp.clip(v_i @ v_j.T, -1.0, 1.0)
    theta = xp.rad2deg(xp.arccos(dots)).ravel()
    h, _ = xp.histogram(theta, bins=edges)
    return h


def count_auto_grid(vecs, edges_deg, xp=np, block=2048, ncells=128):
    """Unordered auto-pair counts (DD / RR) with block pruning."""
    order = _grid_order(vecs, ncells, xp)
    v = vecs[order]
    bounds = _block_bounds(v.shape[0], block)
    centroids, radii = _block_centroids(v, bounds, xp)
    edges = xp.asarray(edges_deg, dtype=xp.float64)
    counts = xp.zeros(edges.size - 1, dtype=xp.float64)
    max_sep_rad = float(np.deg2rad(edges_deg[-1]))

    for i, j in _surviving_pairs(centroids, radii, max_sep_rad, xp, upper_only=True):
        si, ei = bounds[int(i)]
        sj, ej = bounds[int(j)]
        h = _hist_block(v[si:ei], v[sj:ej], edges, xp)
        # within-block (i==j) double-counts orders + has the theta=0 diagonal
        # (excluded by the bins) -> /2 gives unordered pairs.
        counts = counts + (h / 2.0 if i == j else h)
    return counts


def count_cross_grid(va, vb, edges_deg, xp=np, block=2048, ncells=128):
    """Ordered cross-pair counts (DR) with block pruning."""
    oa = _grid_order(va, ncells, xp)
    ob = _grid_order(vb, ncells, xp)
    a, b = va[oa], vb[ob]
    ba = _block_bounds(a.shape[0], block)
    bb = _block_bounds(b.shape[0], block)
    ca, ra_ = _block_centroids(a, ba, xp)
    cb, rb = _block_centroids(b, bb, xp)
    edges = xp.asarray(edges_deg, dtype=xp.float64)
    counts = xp.zeros(edges.size - 1, dtype=xp.float64)
    max_sep_rad = float(np.deg2rad(edges_deg[-1]))

    for i, j in _surviving_pairs_cross(ca, ra_, cb, rb, max_sep_rad, xp):
        si, ei = ba[int(i)]
        sj, ej = bb[int(j)]
        counts = counts + _hist_block(a[si:ei], b[sj:ej], edges, xp)
    return counts


def w_theta_grid(ra_d, dec_d, ra_r, dec_r, edges_deg, backend="cpu",
                 block=2048, ncells=128):
    """w(theta) via block-pruned counting. Same result as brute force, scales far better."""
    if backend == "gpu":
        if not _HAS_CUPY or cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("backend='gpu' requested but no CuPy/CUDA device")
        xp = cp
    else:
        xp = np

    vd = radec_to_unit(ra_d, dec_d, xp)
    vr = radec_to_unit(ra_r, dec_r, xp)
    dd = count_auto_grid(vd, edges_deg, xp, block, ncells)
    rr = count_auto_grid(vr, edges_deg, xp, block, ncells)
    dr = count_cross_grid(vd, vr, edges_deg, xp, block, ncells)
    if xp is cp:
        dd, dr, rr = cp.asnumpy(dd), cp.asnumpy(dr), cp.asnumpy(rr)
    w = landy_szalay(dd, dr, rr, len(ra_d), len(ra_r))
    return w, {"dd": dd, "dr": dr, "rr": rr}
