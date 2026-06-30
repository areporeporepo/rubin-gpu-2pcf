"""Simulation-based inference (SBI) — the AI / field-level track.

The first rung toward field-level / ML inference, built on the same GPU engine:
use the w(theta) measurement as a *forward model*, generate
(summary statistic -> parameter) pairs from many mock catalogs, then learn the
inverse mapping. Here the "parameter" is the clustering amplitude of the Thomas
mock (parent density — a stand-in for a cosmological parameter); the *workflow*
is the same one used for dark-energy / dark-matter SBI at survey scale.

Why GPU matters here: the forward model (mock -> w(theta)) is evaluated
hundreds–thousands of times to build the training set — exactly the loop that
becomes the bottleneck at DP2/DR1, and where the GPU pays off.

Status: early scaffold. The forward-model data generation is real (GPU engine);
the inference below is a simple ridge baseline — swap in a neural posterior
estimator (sbi / lampe / torch) for production field-level inference.
"""

from __future__ import annotations

import numpy as np

from .catalog import make_clustered_catalog, make_randoms
from .grid import w_theta_grid
from .paircount import log_edges


def forward_model(n_parents, edges, n_target=3000, box=(3.0, 3.0), center=(60.0, -30.0),
                  scatter_deg=0.03, frac=0.6, backend="cpu", seed=0):
    """One mock realization -> its w(theta) summary statistic.

    `n_parents` sets the clustering amplitude (Thomas xi ~ 1/parent_density).
    """
    ra_d, dec_d = make_clustered_catalog(
        n_target=n_target, n_parents=int(n_parents), box_deg=box, center=center,
        scatter_deg=scatter_deg, frac_clustered=frac, seed=seed,
    )
    ra_r, dec_r = make_randoms(2 * n_target, box_deg=box, center=center, seed=seed + 99991)
    w, _ = w_theta_grid(ra_d, dec_d, ra_r, dec_r, edges, backend=backend, block=2048, ncells=96)
    return w


def make_training_set(n_sims=40, param_range=(80, 800), backend="cpu", seed=0,
                      min_sep=0.005, max_sep=0.1, nbins=8, **kw):
    """Build (X=w(theta) vectors, y=parameters) across the parameter range."""
    rng = np.random.default_rng(seed)
    edges = log_edges(min_sep, max_sep, nbins)
    X, y = [], []
    for i in range(n_sims):
        p = float(rng.uniform(*param_range))     # parameter to recover
        w = forward_model(p, edges, backend=backend, seed=1000 + i, **kw)
        X.append(np.nan_to_num(w))
        y.append(p)
    return np.asarray(X), np.asarray(y), edges


def fit_baseline(X, y, lam=1e-2):
    """Minimal inverse model (standardized ridge). Returns a predict() function.

    Production: replace with a neural posterior estimator that returns a full
    posterior, not a point estimate (this is the SBI upgrade path).
    """
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Xn = (X - Xm) / Xs
    A = np.hstack([Xn, np.ones((len(Xn), 1))])
    coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)

    def predict(X_new):
        Xn2 = (np.atleast_2d(X_new) - Xm) / Xs
        A2 = np.hstack([Xn2, np.ones((len(Xn2), 1))])
        return A2 @ coef

    return predict


def infer_from_catalog(ra, dec, ra_r, dec_r, predict, edges, backend="cpu"):
    """Apply a trained SBI inverse model to a REAL catalog (DP1 now, DP2 later).

    Computes the catalog's w(theta) with the GPU engine and returns the model's
    parameter estimate. Point it at `dp1_ECDFS.csv` now, or `dp2_*.csv` the day
    DP2 lands this summer — same call.

    NOTE (honest): meaningful only when the forward model is a *realistic
    cosmological* mock. The Thomas-mock parameter here is a workflow stand-in,
    not a physical parameter; swapping in cosmological mocks (and a neural
    posterior estimator) is the science upgrade for DP2/DR1.
    """
    w, _ = w_theta_grid(ra, dec, ra_r, dec_r, edges, backend=backend, block=2048, ncells=96)
    return float(predict(np.nan_to_num(w))[0]), w
