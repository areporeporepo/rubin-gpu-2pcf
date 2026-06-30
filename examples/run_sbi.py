"""AI / field-level track demo: infer a parameter from clustering, the SBI way.

Generates mock w(theta) summary statistics across a parameter (GPU forward model),
learns the inverse mapping, and reports recovery on a held-out set.

    python examples/run_sbi.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twopcf.paircount import has_gpu  # noqa: E402
from twopcf.sbi import fit_baseline, make_training_set  # noqa: E402


def main():
    backend = "gpu" if has_gpu() else "cpu"
    print(f"backend: {backend}  (forward model = GPU w(theta) when available)")

    # training set: many mocks -> (w(theta), parameter)
    X, y, _ = make_training_set(n_sims=40, backend=backend, seed=0)
    # hold out 10 for testing
    Xtr, ytr, Xte, yte = X[:30], y[:30], X[30:], y[30:]

    predict = fit_baseline(Xtr, ytr)
    yp = predict(Xte)
    rel = np.median(np.abs(yp - yte) / yte)
    print(f"trained on {len(Xtr)} mocks; held-out median recovery error: {100*rel:.1f}%")
    print(" true   predicted")
    for t, p in zip(yte, yp):
        print(f" {t:6.0f}  {p:8.0f}")
    print("\nNote: ridge baseline. Swap in a neural posterior estimator (sbi/lampe/torch)")
    print("for full field-level inference — the GPU forward-model loop is the part that scales.")


if __name__ == "__main__":
    main()
