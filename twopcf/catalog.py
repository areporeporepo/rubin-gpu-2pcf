"""Catalogs to run w(theta) on.

Three sources, in order of how "real" they are:

1. make_clustered_catalog  -- synthetic sky with *known* small-scale clustering
   (a Neyman-Scott / Poisson-cluster process). Runs anywhere, no data rights
   needed, lets us validate correctness and benchmark scaling at any N.

2. load_dp1_objects        -- the real Rubin DP1 Object catalog via the Butler,
   inside the Rubin Science Platform. Stub + instructions; fill in once your
   RSP account clears.

3. (later) DP2 / DR1        -- same loader, more area. See ROADMAP.md.

All functions return (ra_deg, dec_deg) NumPy arrays.
"""

from __future__ import annotations

import numpy as np


def _uniform_dec(rng, dec_lo, dec_hi, n):
    """Uniform-on-the-sphere declinations (uniform in sin(dec)), not uniform in dec."""
    s_lo, s_hi = np.sin(np.deg2rad(dec_lo)), np.sin(np.deg2rad(dec_hi))
    s = rng.uniform(s_lo, s_hi, n)
    return np.rad2deg(np.arcsin(s))


def make_clustered_catalog(
    n_target=20000,
    box_deg=(3.0, 3.0),
    center=(60.0, -30.0),
    frac_clustered=0.6,
    n_parents=300,
    scatter_deg=0.03,
    seed=0,
):
    """Synthetic catalog with a measurable positive w(theta) at small scales.

    A fraction `frac_clustered` of objects are scattered (Gaussian, `scatter_deg`)
    around `n_parents` random parent positions; the rest are a uniform field.
    The clustered component produces excess pairs at separations ~scatter_deg.
    """
    rng = np.random.default_rng(seed)
    w, h = box_deg
    ra0, dec0 = center
    ra_lo, ra_hi = ra0 - w / 2, ra0 + w / 2
    dec_lo, dec_hi = dec0 - h / 2, dec0 + h / 2

    n_clustered = int(n_target * frac_clustered)
    n_uniform = n_target - n_clustered

    ra_u = rng.uniform(ra_lo, ra_hi, n_uniform)
    dec_u = _uniform_dec(rng, dec_lo, dec_hi, n_uniform)

    pra = rng.uniform(ra_lo, ra_hi, n_parents)
    pdec = _uniform_dec(rng, dec_lo, dec_hi, n_parents)
    per_parent = rng.multinomial(n_clustered, np.ones(n_parents) / n_parents)
    ra_c = np.repeat(pra, per_parent) + rng.normal(0, scatter_deg, n_clustered)
    dec_c = np.repeat(pdec, per_parent) + rng.normal(0, scatter_deg, n_clustered)

    ra = np.concatenate([ra_u, ra_c])
    dec = np.concatenate([dec_u, dec_c])

    keep = (ra >= ra_lo) & (ra <= ra_hi) & (dec >= dec_lo) & (dec <= dec_hi)
    return ra[keep], dec[keep]


def _ang_sep_deg(ra, dec, ra0, dec0):
    """Great-circle separation (deg) from each (ra,dec) to a center."""
    r1, d1 = np.deg2rad(ra), np.deg2rad(dec)
    r0, d0 = np.deg2rad(ra0), np.deg2rad(dec0)
    cos = np.sin(d1) * np.sin(d0) + np.cos(d1) * np.cos(d0) * np.cos(r1 - r0)
    return np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))


def make_randoms_cone(ra0, dec0, radius_deg, n, seed=1):
    """Uniform-on-sphere randoms inside a cone -- matches a DP1 field's selection.

    NOTE: this is a circular approximation to the real DP1 footprint/mask. A
    rigorous w(theta) needs randoms tracing the actual depth map; this is fine
    for a first look. Returns (ra_deg, dec_deg).
    """
    rng = np.random.default_rng(seed)
    cosd = np.cos(np.deg2rad(dec0))
    ra_lo, ra_hi = ra0 - radius_deg / cosd, ra0 + radius_deg / cosd
    out_ra, out_dec, have = [], [], 0
    while have < n:
        m = max(2 * (n - have), 1000)
        ra = rng.uniform(ra_lo, ra_hi, m)
        dec = _uniform_dec(rng, dec0 - radius_deg, dec0 + radius_deg, m)
        keep = _ang_sep_deg(ra, dec, ra0, dec0) <= radius_deg
        out_ra.append(ra[keep])
        out_dec.append(dec[keep])
        have += int(keep.sum())
    return np.concatenate(out_ra)[:n], np.concatenate(out_dec)[:n]


def load_catalog_csv(path, ra_col="ra", dec_col="dec"):
    """Load (ra_deg, dec_deg) from a CSV with a header row (the DP1 export)."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(data[ra_col], float), np.asarray(data[dec_col], float)


def make_randoms(n, box_deg=(3.0, 3.0), center=(60.0, -30.0), seed=1):
    """Uniform random catalog over the same rectangular footprint.

    For a real survey the randoms must trace the actual mask/depth; for the
    synthetic box a uniform field is the correct unclustered reference.
    """
    rng = np.random.default_rng(seed)
    w, h = box_deg
    ra0, dec0 = center
    ra = rng.uniform(ra0 - w / 2, ra0 + w / 2, n)
    dec = _uniform_dec(rng, dec0 - h / 2, dec0 + h / 2, n)
    return ra, dec


# --- The real DP1 footprint: seven LSSTComCam fields (late 2024) ------------
# Centers (RA, Dec deg) from the DP1 paper (arXiv:2603.23786) / dp1.lsst.io.
# The three deep extragalactic fields are the ones for *galaxy* clustering;
# the Galactic/crowded fields are star-dominated.
DP1_FIELDS = {
    "ECDFS":     dict(center=(53.13, -28.10), kind="extragalactic",    clustering=True),
    "EDFS":      dict(center=(59.10, -48.73), kind="extragalactic",    clustering=True),
    "SV_95_-25": dict(center=(95.00, -25.00), kind="extragalactic",    clustering=True),
    "Fornax":    dict(center=(40.00, -34.45), kind="dwarf_spheroidal", clustering=False),
    "47_Tuc":    dict(center=(6.02,  -72.08), kind="globular_cluster", clustering=False),
    "Seagull":   dict(center=(106.23, -10.51), kind="low_galactic",    clustering=False),
    "SV_38_7":   dict(center=(37.86,   6.98), kind="low_ecliptic",     clustering=False),
}

# DP1 holds ~2.3M objects across the seven ~1 deg^2 fields (DP1 paper).
DP1_FIELD_RADIUS_DEG = 1.0


def make_dp1_like_field(field="ECDFS", n_target=200000, seed=0):
    """Synthetic catalog over a *real* DP1 field footprint -- a faithful
    stand-in until you read the true catalog inside the RSP.

    Uses the field's real center and ~1 deg^2 area. Galaxy-clustering fields get
    injected clustering; the Galactic/crowded fields get a uniform placeholder.
    """
    if field not in DP1_FIELDS:
        raise ValueError(f"unknown DP1 field {field!r}; choose from {list(DP1_FIELDS)}")
    info = DP1_FIELDS[field]
    box = (2 * DP1_FIELD_RADIUS_DEG, 2 * DP1_FIELD_RADIUS_DEG)
    if info["clustering"]:
        return make_clustered_catalog(
            n_target=n_target, box_deg=box, center=info["center"], seed=seed
        )
    return make_randoms(n_target, box_deg=box, center=info["center"], seed=seed)


def load_dp1_objects(butler=None, snr_min=5.0, galaxies_only=False):
    """Load the real DP1 Object catalog (RA, Dec in deg) via the Butler.

    DP1 is NOT a public download -- it lives in the Rubin Science Platform, for
    data-rights holders (US/Chile scientists & students). Run this *inside an
    RSP notebook*; it returns (ra_deg, dec_deg) ready for w(theta).

    Parameters
    ----------
    butler : lsst.daf.butler.Butler, optional
        Existing Butler, or None to construct the DP1 one.
    snr_min : float
        Minimum i-band PSF-flux SNR cut.
    galaxies_only : bool
        Keep only extended sources (i_extendedness == 1), i.e. galaxies.

    See https://dp1.lsst.io/ for the current collection string and schema.
    """
    import numpy as _np
    from lsst.daf.butler import Butler

    if butler is None:
        # Verify the exact collection on dp1.lsst.io -- it is versioned.
        butler = Butler("dp1", collections="LSSTComCam/runs/DRP/DP1/DM-*")

    obj = butler.get("object")  # consolidated Object table (parquet-backed)

    ra = _np.asarray(obj["coord_ra"])    # degrees
    dec = _np.asarray(obj["coord_dec"])  # degrees

    good = _np.asarray(obj["detect_isPrimary"], dtype=bool)
    snr = _np.asarray(obj["i_psfFlux"]) / _np.asarray(obj["i_psfFluxErr"])
    good &= _np.isfinite(snr) & (snr > snr_min)
    if galaxies_only:
        good &= _np.asarray(obj["i_extendedness"]) >= 0.5
    return ra[good], dec[good]
