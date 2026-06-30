"""GPU FITS-ingestion benchmark -- the image-level companion to the catalog
clustering demo. Reproduces the *mechanism* behind cuPhoton's headline numbers
(GPUDirect I/O via kvikio + GPU-parallel tile decompression via nvCOMP) on a
single accessible GPU, honestly, with the Rice-codec caveat documented.

See FITS_IO.md.
"""

from .synth import LSST_CCD_SHAPE, make_image, write_fits_variants
from .tiles import compress_tiles, compression_ratio, decompress_tiles_cpu

__all__ = [
    "make_image",
    "write_fits_variants",
    "LSST_CCD_SHAPE",
    "compress_tiles",
    "decompress_tiles_cpu",
    "compression_ratio",
]
