# Run on a GPU (Google Colab)

In Colab: **Runtime → Change runtime type → GPU (T4 is fine)**. Then add your
token: **🔑 Secrets** panel (left sidebar) → add `RSP_TOKEN` = your full `gt-….….`
token, toggle "Notebook access" on. Then run the cells below.

```python
# Cell 1 — confirm a GPU is attached
!nvidia-smi -L
```

```python
# Cell 2 — install GPU + query deps (Colab has CUDA 12 -> cupy-cuda12x)
!pip -q install cupy-cuda12x pyvo pandas
```

```python
# Cell 3 — get the code  (needs the repo pushed to GitHub; see note below)
!git clone https://github.com/areporeporepo/rubin-gpu-2pcf.git
%cd rubin-gpu-2pcf
```

```python
# Cell 4 — token from Colab Secrets -> .env  (ephemeral Colab VM; never printed)
from google.colab import userdata
open('.env','w').write('RSP_TOKEN=' + userdata.get('RSP_TOKEN'))
!python examples/check_token.py          # want: looks like a full RSP token: True
```

```python
# Cell 5 — pull the REAL DP1 ECDFS catalog (~495k objects)
!python examples/tap_query_external.py   # -> dp1_ECDFS.csv
```

```python
# Cell 6 — THE SPEEDUP NUMBER: CPU vs GPU scaling on the pair counter
!python -m twopcf.bench                   # -> prints GPU speedup, writes benchmark_scaling.png
```

```python
# Cell 7 — REAL DP1 on GPU, FULL 495k catalog (CPU can't do this tractably)
!python examples/run_on_dp1.py dp1_ECDFS.csv 53.13 -28.10 1.0 0.2 0
# -> w(theta) of real Rubin DP1 galaxies on GPU, writes dp1_wtheta.png
```

```python
# Cell 8 — show the figures
from IPython.display import Image, display
display(Image('benchmark_scaling.png'))
display(Image('dp1_wtheta.png'))
```

## What you get
- **Cell 6:** the measured GPU×CPU speedup (the slide/email number).
- **Cell 7:** `w(θ)` of the **full real DP1 ECDFS field on a GPU** — the headline.
- Both validated by the same engine that matches TreeCorr + theory.

## If `git clone` (Cell 3) fails
The repo isn't on GitHub yet. Either:
- push it (see chat — one OK from you), then Cell 3 works, **or**
- zip `~/rubin-gpu-2pcf`, upload via Colab's Files panel, and `!unzip` instead of Cell 3.

## Honest caveats for the real-data plot
- The catalog has **no star/galaxy cut** (the `i_extendedness` column name needs verifying), so it mixes stars+galaxies — a *first-look* clustering signal, not a clean galaxy w(θ).
- Randoms are a **cone approximation**, not the true DP1 depth mask.
- Both are fine for a demo; state them. A rigorous measurement adds the galaxy cut + real mask.
```
