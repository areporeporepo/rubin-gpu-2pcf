# Roadmap — GPU acceleration for the Rubin era

## Why now

Rubin/LSST is the largest optical survey ever attempted: **~20 TB every night,
~20 billion galaxies** mapped over a decade. The cosmology — dark energy, the
nature of dark matter, the galaxy–halo connection — lives in summary statistics
computed across those catalogs. But the workhorse estimators are **O(N²) CPU
pair-counts**, and they hit a wall exactly as the catalogs reach survey scale.
**GPUs clear that wall.** This project is a validated first step, deliberately
timed to ride Rubin's data flow as it arrives.

## Where we are — today, validated

- A GPU `w(θ)` engine that is **bit-for-bit consistent with TreeCorr** and recovers
  known theory to **~6%** (`GPU ≡ CPU ≡ TreeCorr ≡ theory`).
- A **provably-correct block-pruning cell-list** that scales past brute force.
- Run **end-to-end on real Rubin DP1** (~495k ECDFS objects via TAP).
- Shaped as a **complementary GPU backend for DESC's TXPipe**.

Not a pitch deck — a working tool, on real data.

## ▶ Next target: DP2 — this summer

Rubin **Data Preview 2** lands **July–September 2026** — the first data from
**LSSTCam** itself: **~3000 deg²** of deep coadds + catalogs, roughly **200× DP1's
area**, and the **first Rubin dataset large enough for a survey-scale `w(θ)`
measurement**. It's also where catalogs first get big enough that **CPU
pair-counting starts to hurt** — i.e., where this engine earns its keep. We're
built to run on **DP2 the day it drops**. This is the near-term milestone
everything here points at.

## The trajectory — locked to Rubin's releases

| When | Data | Milestone | Why it's exciting |
|---|---|---|---|
| **Now** | DP1 (15 deg²) | validated engine on real data | proof it works |
| **Summer 2026** | **DP2 — ~3000 deg², LSSTCam** | first **survey-scale** `w(θ)` on GPU | ~200× DP1; the moment GPU starts to *matter* |
| **Next year (~2027)** | **DR1 — billions** | GPU / multi-GPU **3×2pt** + inference | CPU can't keep up; GPU becomes *necessary* |

We're built to run on **DP2 the day it drops**, then scale into **DR1 next year**.

## Why the speedup compounds

A clustering analysis runs the 2-point estimator **thousands of times** — across
tomographic bin pairs, **hundreds–thousands of covariance mocks**, systematics
tests, and parameter inference. So a per-call **~30–50×** speedup (measured, 49.7× at N=20k) multiplies:
analysis cycles go **days → hours**, covariance runs **a week → a day**, and
**simulation-based inference** (millions of forward-model evaluations) becomes
feasible. At DP2's hundreds of millions and **DR1's billions**, that's the
difference between an analysis you can iterate and one you can't — i.e.
**time-to-science** for the collaboration.

![Projected analysis time vs catalog size — CPU vs GPU](assets/roadmap_scaling.png)

*Illustrative projection. The GPU/CPU ratio (**~30–50×**, measured on a Tesla T4 —
49.7× at N=20k, and GPU≡CPU bit-for-bit, cross-checked vs TreeCorr at 2.7×10⁻⁴);
absolute times assume O(N) cell-list scaling and order-of-magnitude object counts.
As catalogs grow **DP1 → DP2 → DR1**, GPU keeps a full analysis in hours where CPU
climbs toward a day — multiplied across the thousands of estimator calls per analysis.*

## The science it unlocks

- **3×2pt** (clustering + galaxy–galaxy lensing + shear) → dark energy `w₀, wₐ`
- **Tomographic `w(θ)`** in photo-z bins — the LSST analog of DESI's 3D clustering
- **Satellite-galaxy simulation-based inference** → the nature of dark matter
  (cold? warm? fuzzy?) using public **Symphony** zoom-in sims
- **GPU forward models** → simulation-based inference at a scale CPUs can't reach

## Phases

- **Phase 1 — ✅ this repo.** Validated GPU `w(θ)`, cell-list, real-DP1 path.
- **Phase 2 — satellite DM + SBI.** GPU/JAX forward model over Symphony subhalos →
  a dark-matter limit; the inference loop is the real bottleneck the field feels.
- **Phase 3 — production.** GPU backend merged into TXPipe; tomographic 3×2pt at
  DR1 scale on NERSC/SLAC; the open, validated GPU path for Rubin cosmology.

## The compounding vision

An **open, validated, in-pipeline GPU path for Rubin cosmology** — ready the day
each data release lands, on-mission for the **Center for Decoding the Universe**,
the same direction as NVIDIA cuPhoton but built in the open on the public stack.
Start: one estimator. End: the GPU layer the Rubin data deluge needs.
