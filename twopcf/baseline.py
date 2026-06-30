"""Independent CPU reference using TreeCorr (Jarvis et al.) -- the tree-based
angular correlation code used in DES and trusted across the community.

We run it with bin_slop=0 so its pair counts are *exact* (no tree approximation),
which lets us compare it head-to-head against our brute-force backends.
"""

from __future__ import annotations

import numpy as np


def w_theta_treecorr(ra_d, dec_d, ra_r, dec_r, min_sep_deg, max_sep_deg, nbins):
    """Return (rnom_deg, w) computed by TreeCorr with Landy-Szalay.

    Requires `pip install treecorr`. Uses the great-circle ('Arc') metric and
    exact pair counts (bin_slop=0) so it matches our brute-force edges.
    """
    import treecorr

    cfg = dict(
        min_sep=min_sep_deg,
        max_sep=max_sep_deg,
        nbins=nbins,
        sep_units="deg",
        bin_slop=0.0,
        metric="Arc",
    )
    cat_d = treecorr.Catalog(ra=ra_d, dec=dec_d, ra_units="deg", dec_units="deg")
    cat_r = treecorr.Catalog(ra=ra_r, dec=dec_r, ra_units="deg", dec_units="deg")

    dd = treecorr.NNCorrelation(**cfg)
    rr = treecorr.NNCorrelation(**cfg)
    dr = treecorr.NNCorrelation(**cfg)
    dd.process(cat_d)
    rr.process(cat_r)
    dr.process(cat_d, cat_r)

    w, _ = dd.calculateXi(rr=rr, dr=dr)
    return np.asarray(dd.rnom), np.asarray(w)
