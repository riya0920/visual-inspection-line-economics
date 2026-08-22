"""Is the pass-2 unseen gap a seed effect or a training-budget effect?

Pass 1 reported unseen-class supervised AUROC 0.958 (one seed, n_train=1400,
14 epochs). Pass 2's three seeds all landed near 0.78 at n_train=1100, 12 epochs.
Three tight seeds cannot be sampling noise, so the difference is configuration --
but "which configuration" is exactly the question, and two variables moved at once.

This holds the seed set fixed and sweeps only the training budget.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import models as M  # noqa: E402
import synth  # noqa: E402

KNOWN = ("pore", "crack", "shrinkage")
UNSEEN = "inclusion"


def auroc(scores, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


def run(seed: int, n_train: int, epochs: int) -> dict:
    rng = np.random.default_rng(seed)
    n_test = 700
    train = synth.make(n_train, 0.35, rng, variants=("A", "B", "C"), defect_types=KNOWN)
    normal_only = synth.make(n_train, 0.0, rng, variants=("A", "B", "C"))
    test_known = synth.make(n_test, 0.35, rng, variants=("A", "B", "C"), defect_types=KNOWN)
    test_unseen = synth.make(n_test // 2, 0.5, rng, variants=("A", "B", "C"),
                             defect_types=(UNSEEN,))
    xtr, ytr, _ = synth.to_arrays(train)
    xn, _, _ = synth.to_arrays(normal_only)
    model, _ = M.train_supervised(xtr, ytr, epochs=epochs, verbose=False)
    anom = M.PatchAnomaly().fit(model, xn)
    out = {"seed": seed, "n_train": n_train, "epochs": epochs}
    for name, samples in (("known", test_known), ("unseen", test_unseen)):
        x, y, _ = synth.to_arrays(samples)
        out[f"{name}_sup"] = auroc(M.supervised_scores(model, x), y)
        out[f"{name}_ano"] = auroc(anom.score_maps(model, x).max(axis=(1, 2)), y)
    out["gap"] = out["unseen_ano"] - out["unseen_sup"]
    return out


rows = []
for n_train, epochs in ((1100, 12), (1400, 14)):
    for seed in (11, 22, 33):
        r = run(seed, n_train, epochs)
        rows.append(r)
        print(f"n={n_train} ep={epochs} seed={seed}: "
              f"unseen sup {r['unseen_sup']:.3f} ano {r['unseen_ano']:.3f} "
              f"gap {r['gap']:+.3f}", flush=True)

out = ROOT / "out" / "budget_probe.json"
out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("wrote", out)
