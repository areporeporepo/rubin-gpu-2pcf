"""FITS ingestion benchmark: decompression + I/O, CPU vs GPU.

Two honest, separable measurements (mirroring cuPhoton's two headline numbers):

  decode :  CPU zlib   vs  GPU nvCOMP    (Deflate tiles, same wire format)
  I/O    :  CPU read   vs  kvikio CuFile (GPUDirect Storage)

Reports per-size timings, the GPU speedup, and the compression ratio. The
correctness gate (CPU == GPU decode) runs inside run_demo.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from . import cpu_ingest, gpu_ingest
from .synth import make_image
from .tiles import compress_tiles, compression_ratio


def _time(fn, repeat=3):
    fn()  # warm-up
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run(side_lengths=(1024, 2048, 4096), tile=(256, 256), dtype=np.int32):
    """Benchmark across square images of the given side lengths."""
    have_gpu = gpu_ingest.has_cuda()
    have_kvikio = have_gpu and gpu_ingest.has_kvikio()
    rows = []
    tmpdir = Path(tempfile.mkdtemp(prefix="fitsbench_"))

    for s in side_lengths:
        img = make_image(shape=(s, s), n_sources=s // 2, dtype=dtype)
        blobs, meta = compress_tiles(img, tile=tile)
        ratio = compression_ratio(blobs, meta)
        mpix = (s * s) / 1e6

        t_dec_cpu = _time(lambda: cpu_ingest.decode_tiles(blobs, meta))
        t_dec_gpu = (
            _time(lambda: gpu_ingest.nvcomp_decode_tiles(blobs, meta))
            if have_gpu
            else None
        )

        # I/O: write the raw bytes to disk, then read CPU vs kvikio
        raw_path = tmpdir / f"img_{s}.bin"
        raw_path.write_bytes(img.tobytes())
        t_io_cpu = _time(lambda: gpu_ingest.cpu_read_bytes(str(raw_path)))
        t_io_gpu = (
            _time(lambda: gpu_ingest.kvikio_read_to_gpu(str(raw_path)))
            if have_kvikio
            else None
        )

        rows.append(
            dict(
                mpix=mpix,
                ratio=ratio,
                decode_cpu_s=t_dec_cpu,
                decode_gpu_s=t_dec_gpu,
                decode_speedup=(t_dec_cpu / t_dec_gpu) if t_dec_gpu else None,
                io_cpu_s=t_io_cpu,
                io_gpu_s=t_io_gpu,
                io_speedup=(t_io_cpu / t_io_gpu) if t_io_gpu else None,
            )
        )
        msg = f"{mpix:6.1f} Mpix  ratio={ratio:4.2f}  decode_cpu={t_dec_cpu:7.3f}s"
        if t_dec_gpu:
            msg += f"  decode_gpu={t_dec_gpu:7.3f}s ({t_dec_cpu / t_dec_gpu:5.1f}x)"
        if t_io_gpu:
            msg += f"  io={t_io_cpu / t_io_gpu:5.1f}x"
        print(msg)
    return rows


def plot(rows, path="fits_ingest_scaling.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["mpix"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, [r["decode_cpu_s"] for r in rows], "o-", label="decode: CPU zlib")
    if any(r["decode_gpu_s"] for r in rows):
        ax.plot(x, [r["decode_gpu_s"] for r in rows], "s-", label="decode: GPU nvCOMP")
    if any(r["io_gpu_s"] for r in rows):
        ax.plot(x, [r["io_cpu_s"] for r in rows], "^--", label="I/O: CPU read")
        ax.plot(x, [r["io_gpu_s"] for r in rows], "v--", label="I/O: kvikio (GDS)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("image size [Mpix]")
    ax.set_ylabel("wall time [s]")
    ax.set_title("FITS ingestion mechanism: CPU vs GPU")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    rows = run()
    plot(rows)
