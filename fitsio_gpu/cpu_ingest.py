"""CPU ingestion: the baseline cuPhoton beats.

Two flavors:
- decode_tiles: our raw-DEFLATE tile decode (stdlib zlib) -- the apples-to-apples
  partner of the GPU nvCOMP path.
- read_fits: a real tile-compressed FITS file via astropy, whose decompression
  is derived from the CFITSIO tile-compression algorithms (the same library
  Jim Bosch forks at github.com/TallJimbo/cfitsio). This anchors the benchmark
  to genuine FITS I/O.
"""

from __future__ import annotations

import numpy as np

from .tiles import decompress_tiles_cpu


def decode_tiles(blobs, meta):
    """CPU reference decode of raw-DEFLATE tiles -> NumPy image."""
    return decompress_tiles_cpu(blobs, meta)


def read_fits(path):
    """Read a (possibly tile-compressed) FITS image via astropy/CFITSIO.

    Accessing `.data` forces decompression. For a CompImageHDU the image lives
    in HDU 1; otherwise HDU 0.
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        hdu = hdul[1] if len(hdul) > 1 and hdul[1].data is not None else hdul[0]
        data = np.array(hdu.data)  # triggers decompression
    return data
