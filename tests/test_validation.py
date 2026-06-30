"""Correctness gate: independent implementations must agree.

    pytest -q

- GPU brute force must equal CPU brute force (same algorithm) -> near machine eps.
- TreeCorr (independent tree code) must agree with brute force on w(theta).
- A pure-random catalog must give w(theta) ~ 0 within noise.
"""

import numpy as np
import pytest

from twopcf.catalog import make_clustered_catalog, make_randoms
from twopcf.paircount import has_gpu, log_edges, w_theta

MIN_SEP, MAX_SEP, NBINS = 0.005, 1.0, 10
EDGES = log_edges(MIN_SEP, MAX_SEP, NBINS)


def _catalogs(n=4000, seed=0):
    ra_d, dec_d = make_clustered_catalog(n_target=n, seed=seed)
    ra_r, dec_r = make_randoms(3 * n, seed=seed + 100)
    return ra_d, dec_d, ra_r, dec_r


def test_clustered_signal_is_positive_at_small_scales():
    ra_d, dec_d, ra_r, dec_r = _catalogs()
    w, _ = w_theta(ra_d, dec_d, ra_r, dec_r, EDGES, backend="cpu")
    # smallest bins should show clustering excess
    assert w[0] > 0.5, f"expected strong small-scale clustering, got {w[0]}"


def test_random_catalog_is_consistent_with_zero():
    # data drawn from the SAME uniform distribution as randoms -> w ~ 0
    ra_d, dec_d = make_randoms(6000, seed=7)
    ra_r, dec_r = make_randoms(18000, seed=8)
    w, _ = w_theta(ra_d, dec_d, ra_r, dec_r, EDGES, backend="cpu")
    assert np.nanmax(np.abs(w)) < 0.3, f"random field should be ~0, got {w}"


@pytest.mark.skipif(not has_gpu(), reason="no CUDA device / CuPy")
def test_gpu_matches_cpu_bitwise_close():
    ra_d, dec_d, ra_r, dec_r = _catalogs()
    w_cpu, _ = w_theta(ra_d, dec_d, ra_r, dec_r, EDGES, backend="cpu")
    w_gpu, _ = w_theta(ra_d, dec_d, ra_r, dec_r, EDGES, backend="gpu")
    np.testing.assert_allclose(w_gpu, w_cpu, rtol=1e-6, atol=1e-9)


def test_grid_matches_brute_force_exactly():
    from twopcf.grid import w_theta_grid

    ra_d, dec_d, ra_r, dec_r = _catalogs()
    w_brute, _ = w_theta(ra_d, dec_d, ra_r, dec_r, EDGES, backend="cpu")
    w_grid, _ = w_theta_grid(ra_d, dec_d, ra_r, dec_r, EDGES, backend="cpu",
                             block=1024, ncells=64)
    # block pruning only skips pairs that provably cannot contribute -> identical
    np.testing.assert_allclose(w_grid, w_brute, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not has_gpu(), reason="no CUDA device / CuPy")
def test_grid_gpu_matches_cpu():
    from twopcf.grid import w_theta_grid

    ra_d, dec_d, ra_r, dec_r = _catalogs()
    w_cpu, _ = w_theta_grid(ra_d, dec_d, ra_r, dec_r, EDGES, backend="cpu")
    w_gpu, _ = w_theta_grid(ra_d, dec_d, ra_r, dec_r, EDGES, backend="gpu")
    np.testing.assert_allclose(w_gpu, w_cpu, rtol=1e-6, atol=1e-9)


def test_treecorr_agrees_with_brute_force():
    treecorr = pytest.importorskip("treecorr")
    from twopcf import baseline

    ra_d, dec_d, ra_r, dec_r = _catalogs()
    w_cpu, _ = w_theta(ra_d, dec_d, ra_r, dec_r, EDGES, backend="cpu")
    _, w_tc = baseline.w_theta_treecorr(ra_d, dec_d, ra_r, dec_r, MIN_SEP, MAX_SEP, NBINS)
    # exact pair counts (bin_slop=0) -> should match closely
    np.testing.assert_allclose(w_tc, w_cpu, rtol=5e-2, atol=5e-2)
