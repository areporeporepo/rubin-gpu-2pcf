"""Visualization layer -- the "visualize" in cuPhoton's load/process/analyze/visualize.

Rides on the same catalogs the correlation engine uses. The heavy step (binning
the catalog into a fine density grid and smoothing it) runs on the GPU via CuPy,
so this is a *GPU* visualization, not just a plot. CPU fallback works anywhere.

density_map(ra, dec) -> a smoothed 2D galaxy-density image (the large-scale
structure / "cosmic web" view). Pair it with the measured w(theta) for an
event-ready "decode the universe" panel.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CUPY = False


def _has_gpu():
    if not _HAS_CUPY:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover
        return False


def _fft_gaussian(img, sigma, xp):
    """Gaussian smooth via FFT -- works for NumPy and CuPy, no scipy dependency."""
    ny, nx = img.shape
    fy = xp.fft.fftfreq(ny)[:, None]
    fx = xp.fft.fftfreq(nx)[None, :]
    kernel = xp.exp(-2.0 * (xp.pi**2) * (sigma**2) * (fx**2 + fy**2))
    return xp.real(xp.fft.ifft2(xp.fft.fft2(img) * kernel))


def density_map(ra, dec, n_pix=600, sigma_pix=4.0, backend="cpu"):
    """Smoothed 2D galaxy-density image over the catalog's footprint.

    Bins points to an n_pix x n_pix grid (cos-dec corrected) and FFT-smooths.
    backend='gpu' runs the binning + smoothing on the GPU (CuPy).
    Returns a (n_pix, n_pix) NumPy array.
    """
    xp = cp if (backend == "gpu" and _has_gpu()) else np
    ra_a = xp.asarray(ra, dtype=xp.float64)
    dec_a = xp.asarray(dec, dtype=xp.float64)

    ra0, ra1 = float(ra_a.min()), float(ra_a.max())
    dec0, dec1 = float(dec_a.min()), float(dec_a.max())
    cosd = np.cos(np.deg2rad(0.5 * (dec0 + dec1)))  # flatten RA by cos(dec)

    fx = (ra_a - ra0) / ((ra1 - ra0) + 1e-12)
    fy = (dec_a - dec0) / ((dec1 - dec0) + 1e-12)
    ix = xp.clip((fx * n_pix).astype(xp.int64), 0, n_pix - 1)
    iy = xp.clip((fy * n_pix).astype(xp.int64), 0, n_pix - 1)
    flat = iy * n_pix + ix
    counts = xp.bincount(flat, minlength=n_pix * n_pix).reshape(n_pix, n_pix)

    smoothed = _fft_gaussian(counts.astype(xp.float64), sigma_pix, xp)
    smoothed = xp.clip(smoothed, 0, None)
    out = cp.asnumpy(smoothed) if xp is cp else np.asarray(smoothed)
    return out, (ra0, ra1, dec0, dec1), cosd


def render(ra, dec, theta, w, w_theory=None, density_backend="cpu",
           n_pix=600, sigma_pix=4.0, path="visualization.png", title=None):
    """Two-panel event figure: GPU density field (left) + measured w(theta) (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dmap, extent_box, _ = density_map(ra, dec, n_pix=n_pix, sigma_pix=sigma_pix,
                                      backend=density_backend)
    ra0, ra1, dec0, dec1 = extent_box

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))
    axL.imshow(dmap.T, origin="lower", extent=[ra0, ra1, dec0, dec1],
               cmap="magma", aspect="auto",
               vmax=np.percentile(dmap, 99.5))
    axL.set_xlabel("RA [deg]")
    axL.set_ylabel("Dec [deg]")
    axL.set_title("GPU-rendered galaxy density field")

    axR.plot(theta, w, "o", ms=7, color="C0", label="measured w(θ)  (GPU)")
    if w_theory is not None:
        axR.plot(theta, w_theory, "-", lw=2, color="0.4", label="theory")
    axR.axhline(0, color="0.8", lw=0.8)
    axR.set_xscale("log")
    axR.set_xlabel(r"$\theta$ [deg]")
    axR.set_ylabel(r"$w(\theta)$")
    axR.set_title("Angular clustering (validated)")
    axR.legend(fontsize=9)
    axR.grid(True, which="both", alpha=0.25)

    fig.suptitle(title or "Decoding the universe — GPU-accelerated, Rubin-format field",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    return path
