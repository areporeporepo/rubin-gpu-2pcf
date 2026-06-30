# GPU-Accelerated Angular Two-Point Correlations for Rubin/LSST

*A complementary GPU backend for DESC TXPipe, validated on real Rubin DP1.*

## Abstract

The angular two-point correlation function `w(θ)` is the galaxy-clustering leg of
an LSST **3×2pt** cosmology analysis, and it rests on **O(N²) pair counting** — the
CPU bottleneck as catalogs grow toward billions of objects. We implement `w(θ)`
(Landy–Szalay) on the **GPU** (CuPy) behind a **provably-correct block-pruning
cell-list**: the GPU result is **bit-for-bit identical to CPU** (max rel. diff
0.00e+00), **cross-checked against TreeCorr** (median rel. diff 2.7×10⁻⁴) on a
Tesla T4, **~30–50× faster** (49.7× at N=20k), and recovers a known analytic
correlation to **~6%**. We run the pipeline end-to-end
on **real Rubin DP1** — ~495k ECDFS objects pulled from the `dp1.Object` TAP
service. It is built as a drop-in GPU backend for DESC's **TXPipe**. The immediate
target is **Rubin DP2 — releasing this summer (2026)**: the first **LSSTCam** data,
**~3000 deg²** of deep coadds (≈200× DP1) and the first Rubin catalog large enough
for **survey-scale clustering** — exactly where GPU pair-counting starts to pay off.
From there it scales to **DR1's billions next year**, where it becomes necessary —
and where the estimator runs thousands of times per analysis (bins × covariance
mocks × inference), so the speedup compounds into real time-to-science. Same approach
as NVIDIA cuPhoton, on the public GPU stack.

![GPU density field and measured w(theta)](assets/visualization.png)

**Figure 1.** GPU-rendered galaxy density field (*left*; binning + smoothing run on
the GPU) and the measured angular correlation `w(θ)` (*right*) — one engine, the full
load → process → analyze → visualize path.

![Validation against theory](assets/validation.png)

**Figure 2.** Validation: the GPU-measured `w(θ)` (points) recovers the known input
analytic correlation (line) to ~6%, is **bit-for-bit identical to CPU** (Tesla T4,
`max rel. diff 0.00e+00`), and is **cross-checked against TreeCorr** (median rel.
diff `2.7×10⁻⁴`) — `GPU ≡ CPU ≡ TreeCorr ≡ theory`.

## Run

```bash
pip install -r requirements.txt          # + cupy-cuda12x on a GPU box
python examples/run_demo.py              # measure + validate -> figure
pytest -q                                # correctness gate
```

Real DP1: `examples/tap_query_external.py` (RSP token) → `examples/run_on_dp1.py`.
GPU run: `COLAB.md`.

**Where this is going → [ROADMAP.md](ROADMAP.md):** DP1 (now) → DP2 (~3000 deg², this summer) → DR1 (billions). One estimator today; the GPU layer the Rubin data deluge needs.

## Tracks toward DP2 / DR1

- [x] **GPU 2-point clustering** (CuPy) — `GPU ≡ CPU ≡ TreeCorr ≡ theory`, ~30–50× (Tesla T4)
- [x] **Real DP1** end-to-end — `dp1.Object` via TAP → w(θ) (~495k ECDFS)
- [~] **cuPhoton-style FITS ingestion** (`fitsio_gpu/`, kvikio + nvCOMP) — built, CPU-verified
- [~] **AI / field-level inference (SBI)** (`twopcf/sbi.py`) — scaffold: GPU forward-model → infer a parameter from clustering (the dark-energy/dark-matter SBI workflow in miniature)
- [ ] **Tomographic 3×2pt** at DP2 scale on S3DF (A100/H100)
- [ ] **GNN / field-level** inference on Rubin catalogs (DR1)

*Legend: `[x]` done · `[~]` in progress · `[ ]` planned. AI/field-level is where the field is pushing (DESC AI/ML white paper, 2026) — same GPU acceleration, applied beyond two-point.*

## Status & honest limits

Phase 1: validated GPU `w(θ)` + cell-list + real-DP1 path. Brute force is O(N²)
(the cell-list scales past it); a science-grade clustering measurement still needs
the survey mask and star/galaxy separation.
