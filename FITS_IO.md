# `fitsio_gpu` — the cuPhoton-style FITS ingestion benchmark

The image-level companion to the catalog clustering demo. It reproduces the
**mechanism** behind NVIDIA cuPhoton's headline numbers on a single accessible
GPU — honestly, with the caveats stated, not buried.

## The two mechanisms cuPhoton exploits

| Mechanism | cuPhoton headline | This module | Tool |
|---|---|---|---|
| Get bytes off disk into the GPU | part of "14,900× ingestion" | `kvikio_read_to_gpu` | **kvikio** (GPUDirect Storage) |
| Decompress tiles in parallel | "8,400× signal processing" | `nvcomp_decode_tiles` | **nvCOMP** |
| CPU baseline being beaten | their reference | `cpu_ingest` | **zlib** / astropy(**CFITSIO**) |

The 14,900× also assumes a **GB200 NVL72 rack** and *their* CPU baseline. We
measure the *mechanism* on one GPU, so expect single-GPU numbers (decode ~10–100×,
I/O depends on whether real GDS is configured), not the rack headline. Saying so
is the whole point — a defensible 30× beats an indefensible 14,900×.

## The honest Rice caveat (read this)

Real LSST FITS uses **RICE_1** tile compression. **nvCOMP has no Rice codec** —
its codecs are LZ4 / Snappy / GDeflate / Deflate / Zstd / etc. So a true GPU
Rice decode requires a **custom CUDA kernel** (almost certainly what cuPhoton
wrote). To keep this benchmark honest *and* runnable, we compress tiles with
**raw DEFLATE** (stdlib `zlib` on CPU, nvCOMP "Deflate" on GPU) — same wire
format on both sides. That measures the *parallel-tile-decode mechanism*
faithfully; it does not measure the Rice codec specifically. Implementing a
GPU Rice decoder is the natural stretch goal (and the part that would genuinely
reproduce cuPhoton rather than approximate it).

We still write/read real `RICE_1` FITS via astropy (whose decompression derives
from the **CFITSIO** algorithms — the library **Jim Bosch forks** at
`github.com/TallJimbo/cfitsio`, branches `stop-writing-rice-one` /
`reset-quantize-level-on-read`) as a real-FITS anchor.

## Run it

```bash
pip install astropy            # real-FITS anchor + CPU path
# on the GCP A100 box:
pip install cupy-cuda12x kvikio-cu12 nvidia-nvcomp-cu12

python examples/run_fits_demo.py     # round-trip check + GPU timings
python -m fitsio_gpu.bench           # scaling -> fits_ingest_scaling.png
```

Runs CPU-only anywhere (stdlib zlib). GPU paths activate when CUDA + kvikio +
nvCOMP are importable.

## Correctness gate

`run_fits_demo` asserts the CPU decode is **lossless** (`array_equal` with the
original) and that the **GPU image matches the CPU image exactly**. No speedup
number is reported unless the decode is bit-exact.

## Version note

nvCOMP's Python API has changed across releases. `gpu_ingest._nvcomp_decompress_batch`
tries the standalone `nvidia.nvcomp` batched API, then `kvikio`'s codec, and
raises a clear message if neither matches your wheel — adjust the algorithm
string there to your installed version.

## Where this fits vs the clustering demo

- **This (`fitsio_gpu`)**: pixels/images. Strong "I can do GPU systems" signal;
  audience is NVIDIA / the Rubin US Data Facility (SLAC) / systems folks.
- **`twopcf`**: catalogs → clustering. The better pitch for **Risa** (her
  bottleneck is inference, not I/O).

Build both; lead with the one that matches your audience.
