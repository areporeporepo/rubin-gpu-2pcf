"""Theory curves to validate the measured w(theta) against.

Two references:

1. thomas_w_theta -- the *analytic* angular correlation of our synthetic mock
   (a Thomas / Neyman-Scott cluster process). This lets us close the loop
   locally: measure w(theta) on the mock and check it recovers the known truth,
   no external data or libraries needed.

2. ccl_w_theta -- the *cosmological* prediction from the DESC Core Cosmology
   Library (CCL, github.com/LSSTDESC/CCL). This is the curve you overlay on
   REAL data (DP1/DC2 galaxies). Guarded import; needs `pip install pyccl`.

A Thomas process with parent density kappa (per deg^2) and Gaussian offspring
scatter sigma (deg) has pair correlation
    xi(theta) = 1 / (4 pi kappa sigma^2) * exp(-theta^2 / (4 sigma^2)).
Mixing in a uniform fraction (only a fraction f of points are clustered) dilutes
the excess as f^2, giving the formula below.
"""

from __future__ import annotations

import numpy as np


def thomas_w_theta(theta_deg, n_parents=300, box_deg=(3.0, 3.0),
                   scatter_deg=0.03, frac_clustered=0.6):
    """Analytic w(theta) for the make_clustered_catalog() Thomas mock.

    Pass the SAME parameters used to build the catalog. theta_deg may be scalar
    or array; returns the model correlation at those separations.
    """
    area = box_deg[0] * box_deg[1]          # deg^2
    kappa = n_parents / area                # parent density, per deg^2
    sigma = scatter_deg
    theta = np.asarray(theta_deg, dtype=float)
    amp = frac_clustered**2 / (4.0 * np.pi * kappa * sigma**2)
    return amp * np.exp(-(theta**2) / (4.0 * sigma**2))


def ccl_w_theta(theta_deg, z=None, nz=None, bias=1.5, cosmo=None,
                ell_max=20000, n_ell=400):
    """Cosmological angular correlation w(theta) from DESC CCL.

    Overlay this on real galaxy data. Defaults: vanilla LCDM, a broad Gaussian
    n(z) centered at z=0.5, linear bias=1.5. Tested against CCL's current API;
    adjust if your installed version differs.
    """
    import pyccl as ccl

    if cosmo is None:
        cosmo = ccl.CosmologyVanillaLCDM()
    if z is None:
        z = np.linspace(0.0, 2.0, 256)
    if nz is None:
        nz = np.exp(-0.5 * ((z - 0.5) / 0.25) ** 2)

    tracer = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z, nz), bias=(z, bias * np.ones_like(z))
    )
    ell = np.unique(np.geomspace(2, ell_max, n_ell).astype(int)).astype(float)
    cl = ccl.angular_cl(cosmo, tracer, tracer, ell)
    # real-space angular correlation; theta in degrees, type 'NN' = clustering
    return ccl.correlation(cosmo, ell=ell, C_ell=cl, theta=np.asarray(theta_deg),
                           type="NN")
