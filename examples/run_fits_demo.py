"""End-to-end FITS-ingestion demo.

    python examples/run_fits_demo.py

- builds a synthetic LSST CCD-scale image
- tile-compresses it (raw DEFLATE)
- CPU-decodes (stdlib zlib) and checks the round-trip is lossless
- if a GPU is present: kvikio reads the bytes + nvCOMP decodes, and we verify
  the GPU image matches the CPU image exactly
- if astropy is present: writes/reads a real RICE_1 FITS as an anchor
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fitsio_gpu import cpu_ingest, gpu_ingest  # noqa: E402
from fitsio_gpu.synth import make_image  # noqa: E402
from fitsio_gpu.tiles import compress_tiles, compression_ratio  # noqa: E402


def main(side=2048):
    print(f"synthetic image: {side}x{side} int32 ({side*side/1e6:.1f} Mpix)")
    img = make_image(shape=(side, side), n_sources=side // 2)
    blobs, meta = compress_tiles(img, tile=(256, 256))
    print(f"tiles: {len(blobs)}   compression ratio: {compression_ratio(blobs, meta):.2f}x")

    t0 = time.perf_counter()
    img_cpu = cpu_ingest.decode_tiles(blobs, meta)
    t_cpu = time.perf_counter() - t0
    assert np.array_equal(img_cpu, img), "CPU round-trip not lossless!"
    print(f"CPU decode: {t_cpu:.3f}s  (round-trip lossless: OK)")

    if gpu_ingest.has_cuda():
        import cupy as cp

        t0 = time.perf_counter()
        img_gpu = gpu_ingest.nvcomp_decode_tiles(blobs, meta)
        cp.cuda.runtime.deviceSynchronize()
        t_gpu = time.perf_counter() - t0
        match = bool(cp.asnumpy(img_gpu == cp.asarray(img)).all())
        print(f"GPU decode: {t_gpu:.3f}s  ({t_cpu/t_gpu:.1f}x)  matches CPU: {match}")
    else:
        print("GPU decode: no CUDA device -> skipped (runs on the GCP A100 box)")

    if gpu_ingest.has_cuda() and gpu_ingest.has_kvikio():
        path = "/tmp/_kvikio_demo.bin"
        open(path, "wb").write(img.tobytes())
        t0 = time.perf_counter()
        gpu_ingest.cpu_read_bytes(path)
        t_io_cpu = time.perf_counter() - t0
        t0 = time.perf_counter()
        gpu_ingest.kvikio_read_to_gpu(path)
        t_io_gpu = time.perf_counter() - t0
        print(f"I/O: CPU {t_io_cpu*1e3:.1f}ms vs kvikio {t_io_gpu*1e3:.1f}ms "
              f"({t_io_cpu/t_io_gpu:.1f}x)")

    try:
        paths = __import__("fitsio_gpu.synth", fromlist=["write_fits_variants"]) \
            .write_fits_variants(img, "/tmp/_lsst_demo")
        r = cpu_ingest.read_fits(paths["rice"])
        print(f"real FITS anchor: wrote {list(paths)}, RICE_1 read OK, shape {r.shape}")
    except Exception as exc:
        print(f"real FITS anchor: skipped ({exc})")


if __name__ == "__main__":
    main()
