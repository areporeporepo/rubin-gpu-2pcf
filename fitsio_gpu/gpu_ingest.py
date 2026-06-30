"""GPU ingestion: the two mechanisms behind cuPhoton's headline number.

1. kvikio_read_to_gpu   -- GPUDirect Storage: read file bytes NVMe -> GPU,
   bypassing the CPU. Stable kvikio.CuFile API. Falls back to a POSIX +
   bounce-buffer "compatibility mode" if real GDS isn't configured on the box
   (still works, smaller win).

2. nvcomp_decode_tiles  -- GPU-parallel decompression of the raw-DEFLATE tiles
   produced by tiles.compress_tiles(). nvCOMP's Python API has churned across
   versions, so we try the standalone batched API first, then kvikio's codec,
   and raise a clear message if neither matches the installed version.

Together these are the mechanism; the 14,900x headline adds a GB200 NVL72 rack +
their CPU baseline on top. We measure the mechanism on one GPU, honestly.

Everything here is import-guarded so the package imports without CUDA.
"""

from __future__ import annotations

import os

import numpy as np


def has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def has_kvikio() -> bool:
    try:
        import kvikio  # noqa: F401

        return True
    except Exception:
        return False


def kvikio_read_to_gpu(path):
    """Read the whole file directly into a CuPy uint8 buffer via kvikio.

    Returns the device buffer. Uses GPUDirect Storage when available.
    """
    import cupy as cp
    import kvikio

    size = os.path.getsize(path)
    buf = cp.empty(size, dtype=cp.uint8)
    with kvikio.CuFile(path, "r") as f:
        nread = f.read(buf)
    assert nread == size, f"short read: {nread} != {size}"
    return buf


def cpu_read_bytes(path):
    """Plain host read of the file bytes (the I/O baseline kvikio competes with)."""
    with open(path, "rb") as f:
        return f.read()


def nvcomp_decode_tiles(blobs, meta):
    """GPU-decompress raw-DEFLATE tiles and reassemble into a CuPy image.

    Tries, in order:
      (a) standalone `nvidia.nvcomp` batched decode  (best throughput)
      (b) `kvikio.nvcomp_codec.NvCompBatchCodec`     (per-tile)
    Adjust the algorithm string if your wheel names it differently.
    """
    import cupy as cp

    dtype = np.dtype(meta["dtype"])
    out = cp.empty(meta["shape"], dtype=dtype)

    raws = _nvcomp_decompress_batch(blobs, meta["raw_sizes"])
    for raw, (y0, y1, x0, x1) in zip(raws, meta["boxes"]):
        tile = cp.frombuffer(raw, dtype=dtype).reshape(y1 - y0, x1 - x0)
        out[y0:y1, x0:x1] = tile
    return out


def _nvcomp_decompress_batch(blobs, raw_sizes):
    """Return a list of decompressed device byte-buffers, one per blob."""
    # (a) standalone nvcomp batched API
    try:
        from nvidia import nvcomp  # type: ignore

        codec = nvcomp.Codec(algorithm="Deflate")
        comp = [nvcomp.as_array(memoryview(b)) for b in blobs]
        decomp = codec.decode(comp)  # batched on the GPU
        return [d.view() for d in decomp]
    except Exception:
        pass

    # (b) kvikio's numcodecs-style codec (per-tile)
    try:
        import cupy as cp
        from kvikio.nvcomp_codec import NvCompBatchCodec  # type: ignore

        codec = NvCompBatchCodec("deflate")
        return [cp.asarray(bytearray(codec.decode(b))) for b in blobs]
    except Exception as exc:  # pragma: no cover
        raise NotImplementedError(
            "Could not find a working nvCOMP Deflate decode API. Install a "
            "matching wheel (e.g. `pip install nvidia-nvcomp-cu12`) and adjust "
            f"the algorithm name to your version. Underlying error: {exc}"
        )
