"""GPU-accelerated angular two-point correlation function for Rubin/LSST.

Phase 1 of a staged demo: a clean, validated GPU w(theta) engine (the 3x2pt
clustering summary statistic), with a roadmap to satellite-galaxy / dark-matter
simulation-based inference. See README.md and ROADMAP.md.
"""

from .paircount import (
    bin_centers,
    has_gpu,
    landy_szalay,
    log_edges,
    w_theta,
)
from .catalog import (
    DP1_FIELDS,
    load_dp1_objects,
    make_clustered_catalog,
    make_dp1_like_field,
    make_randoms,
)
from .grid import w_theta_grid

__all__ = [
    "w_theta",
    "w_theta_grid",
    "log_edges",
    "bin_centers",
    "landy_szalay",
    "has_gpu",
    "make_clustered_catalog",
    "make_randoms",
    "make_dp1_like_field",
    "load_dp1_objects",
    "DP1_FIELDS",
]
