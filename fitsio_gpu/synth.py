"""Synthetic LSST-scale images for the FITS ingestion benchmark.

We generate a realistic-ish astronomical image with NumPy (no astropy needed),
so the tiling + decode harness runs anywhere. An optional helper writes real
tile-compressed FITS files (RICE_1 / GZIP_1) via astropy when it's installed,
to anchor the benchmark to genuine FITS I/O.

LSSTCam science sensors are 4096x4096; a calexp is float32. We default to one
CCD-sized image; scale up to mimic a focal-plane stack.
"""

from __future__ import annotations

import numpy as np

# A single LSSTCam CCD. The full focal plane is 189 of these (~3.2 Gpix).
LSST_CCD_SHAPE = (4096, 4096)


def make_image(shape=LSST_CCD_SHAPE, n_sources=4000, sky=1000.0, seed=0, dtype=np.int32):
    """A sky-background + Poisson-noise + Gaussian-sources image.

    Noise-dominated like a real exposure, so it compresses at a realistic ratio
    rather than artificially well. Returns an array of `dtype`.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    img = rng.poisson(sky, size=shape).astype(np.float64)

    # sprinkle point sources (compact Gaussians)
    ys = rng.integers(8, h - 8, n_sources)
    xs = rng.integers(8, w - 8, n_sources)
    fluxes = rng.lognormal(mean=7.0, sigma=1.2, size=n_sources)
    yy, xx = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-(xx**2 + yy**2) / (2 * 1.2**2))
    kernel /= kernel.sum()
    for x, y, f in zip(xs, ys, fluxes):
        img[y - 4 : y + 5, x - 4 : x + 5] += f * kernel

    if np.issubdtype(dtype, np.integer):
        return np.clip(img, 0, np.iinfo(dtype).max).astype(dtype)
    return img.astype(dtype)


def write_fits_variants(img, prefix):
    """Write uncompressed, RICE_1 and GZIP_1 FITS (needs astropy).

    Returns {label: path}. RICE_1 is what LSST actually uses; for float data
    astropy quantizes before Rice (the quantize level is exactly what Bosch's
    cfitsio `reset-quantize-level-on-read` branch concerns).
    """
    from astropy.io import fits

    paths = {}

    p = f"{prefix}_uncompressed.fits"
    fits.PrimaryHDU(img).writeto(p, overwrite=True)
    paths["uncompressed"] = p

    for ctype, label in [("RICE_1", "rice"), ("GZIP_1", "gzip")]:
        p = f"{prefix}_{label}.fits"
        hdu = fits.CompImageHDU(img, compression_type=ctype)
        hdu.writeto(p, overwrite=True)
        paths[label] = p

    return paths
