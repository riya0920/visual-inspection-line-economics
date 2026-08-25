"""Pass 4: the larger MVTec subset, the resumable fetch, and the verdict logic.

The verdict logic gets the most attention here, because it is the part that can
quietly lie. A group mean can be positive because one category in the group won
enormously while another lost, and a report that prints the mean and stops has
grouping-shopped its way to a conclusion.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("rp4", ROOT / "run_pass4.py")
RP = importlib.util.module_from_spec(_spec)
sys.modules["rp4"] = RP
_spec.loader.exec_module(RP)

NPZ = ROOT / "data" / "MVTEC" / "mvtec_subset.npz"
CACHE = ROOT / "data" / "MVTEC" / "cache"
PASS4 = ROOT / "out" / "pass4.json"


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------

def test_auroc_matches_a_known_case():
    assert RP.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert RP.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert RP.auroc([4, 3, 2, 1], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_a_constant_detector_scores_half_not_one():
    """Ties must get average ranks. Without that a detector that outputs the
    same number for everything scores a perfect 1.0."""
    assert RP.auroc([0.5] * 8, [0, 0, 0, 0, 1, 1, 1, 1]) == pytest.approx(0.5)


def test_a_single_class_split_returns_nan_rather_than_a_number():
    assert np.isnan(RP.auroc([0.1, 0.2, 0.3], [1, 1, 1]))


def test_the_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    s = np.r_[rng.normal(0, 1, 40), rng.normal(1.4, 1, 40)]
    lo, hi = RP.bootstrap_auroc_ci(s, y, n_boot=300)
    point = RP.auroc(s, y)
    assert lo < point < hi
    assert hi - lo > 0.02, "an interval this tight on 80 images is not credible"


# ---------------------------------------------------------------------------
# the verdict logic -- the part that can lie
# ---------------------------------------------------------------------------

def _doc(rows, **extra):
    d = {"n_images": 500, "categories": sorted({r["category"] for r in rows}),
         "rows": rows, "backbone": True, "quick": False, "elapsed_s": 100.0}
    graded = [r for r in rows if "delta" in r]
    for kind in ("texture", "object"):
        sub = [r for r in graded if r["kind"] == kind]
        if sub:
            d[f"{kind}_mean_delta"] = float(np.mean([r["delta"] for r in sub]))
            d[f"{kind}_patchcore_wins"] = sum(1 for r in sub if r["delta"] > 0)
            d[f"{kind}_n"] = len(sub)
    ds = [r["delta"] for r in graded]
    d["delta_spread"] = float(max(ds) - min(ds))
    d["group_gap"] = abs(d.get("texture_mean_delta", 0.0)
                         - d.get("object_mean_delta", 0.0))
    d["undertrained"] = [r["category"] for r in graded if r.get("undertrained")]
    d["patchcore_wins"] = sum(1 for r in graded if r["delta"] > 0)
    d["n_graded"] = len(graded)
    d.update(extra)
    return d


def _row(cat, kind, delta, n_train=60, own=0.7):
    return {"category": cat, "kind": kind, "n_train": n_train, "n_test": 60,
            "defect_rate": 0.5, "defect_types": ["scratch"],
            "patchcore_auroc": own + delta, "patchcore_ci": (0.6, 0.9),
            "own_cnn_auroc": own, "own_cnn_ci": (0.6, 0.8), "delta": delta,
            "train_share_of_median": 1.0, "undertrained": False}


def test_a_group_mean_hiding_a_split_is_reported_as_inconsistent():
    """The real case: textures mean +0.173, built from +0.527 and -0.181. A
    report that prints the mean and stops says PatchCore wins on textures."""
    d = _doc([_row("carpet", "texture", +0.527), _row("grid", "texture", -0.181),
              _row("bottle", "object", +0.007), _row("hazelnut", "object", +0.206)])
    assert d["texture_mean_delta"] > 0
    text = RP.report(d)
    assert "Neither attribution survives" in text
    assert "opposite directions" in text
    assert "PatchCore wins consistently" not in text


def test_a_genuinely_consistent_result_is_allowed_to_conclude():
    """Every category the same direction and the groups close together, so the
    spread test has nothing to object to."""
    d = _doc([_row("carpet", "texture", +0.20), _row("grid", "texture", +0.21),
              _row("bottle", "object", +0.22), _row("hazelnut", "object", +0.23)])
    text = RP.report(d)
    assert "PatchCore wins consistently" in text
    assert "Neither attribution survives" not in text


def test_a_clean_texture_object_split_is_reported_as_one():
    d = _doc([_row("carpet", "texture", -0.20), _row("grid", "texture", -0.22),
              _row("bottle", "object", +0.21), _row("hazelnut", "object", +0.23)])
    text = RP.report(d)
    assert "texture versus object" in text


def test_one_category_per_group_refuses_to_conclude():
    d = _doc([_row("grid", "texture", -0.20), _row("bottle", "object", +0.21)])
    text = RP.report(d)
    assert "Not enough categories to conclude" in text


def test_an_undertrained_category_is_flagged_in_the_report():
    r = _row("carpet", "texture", +0.40, n_train=23, own=0.50)
    r["undertrained"] = True
    r["train_share_of_median"] = 0.38
    d = _doc([r, _row("grid", "texture", -0.18),
              _row("bottle", "object", +0.01), _row("hazelnut", "object", +0.21)])
    text = RP.report(d)
    assert "not a fair comparison" in text
    assert "carpet" in text
    assert "23 training images" in text


def test_the_report_always_states_what_it_does_not_settle():
    d = _doc([_row("grid", "texture", -0.2), _row("carpet", "texture", -0.1),
              _row("bottle", "object", +0.2), _row("hazelnut", "object", +0.1)])
    text = RP.report(d)
    assert "is not MVTec" in text
    assert "No segmentation ground truth" in text


# ---------------------------------------------------------------------------
# the resumable fetch
# ---------------------------------------------------------------------------

def test_the_fetcher_writes_its_index_atomically():
    """A resumable fetcher whose index can be truncated by an interrupt is not
    resumable -- and the interrupt is the normal case on this CDN."""
    src = (ROOT / "fetch_mvtec.py").read_text(encoding="utf-8")
    assert "tmp.replace(index_path)" in src
    assert 'index_path.write_text(json.dumps(index)' not in src


def test_building_from_cache_merges_the_earlier_subset():
    """Rebuilding from a cache that post-dates the pass-3 fetch would otherwise
    silently drop the two categories this project has reported on since."""
    src = (ROOT / "fetch_mvtec.py").read_text(encoding="utf-8")
    assert "mvtec_subset_pass3.npz" in src
    assert "def build_from_cache" in src


@pytest.mark.skipif(not CACHE.exists(), reason="no fetch cache")
def test_every_cache_index_entry_points_at_a_real_array():
    """The direction that matters. An index entry with no array would put a
    hole in the rebuilt dataset; an array with no index entry is only a wasted
    file -- and there are some, because the very first fetch ran before the
    index existed.
    """
    idx = json.loads((CACHE / "index.json").read_text(encoding="utf-8"))
    npys = {p.stem for p in CACHE.glob("*.npy")}
    assert npys and idx, "cache is empty"
    dangling = [k for k in idx if k not in npys]
    assert not dangling, f"{len(dangling)} index entries have no array"
    for cat, split, _ in idx.values():
        assert split in ("train", "test")
        assert isinstance(cat, str) and cat


# ---------------------------------------------------------------------------
# against the fetched data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPZ.exists(), reason="MVTec subset not fetched")
def test_the_subset_is_larger_than_pass_3_and_spans_both_kinds():
    z = np.load(NPZ, allow_pickle=True)
    cats = sorted(set(z["category"].tolist()))
    assert len(z["x"]) > 218, "no larger than pass 3"
    assert len(cats) > 2
    kinds = {"texture" if c in RP.TEXTURES else "object" for c in cats}
    assert kinds == {"texture", "object"}


@pytest.mark.skipif(not NPZ.exists(), reason="MVTec subset not fetched")
def test_every_category_has_both_classes_in_its_test_split():
    z = np.load(NPZ, allow_pickle=True)
    cat, split, defect = z["category"], z["split"], z["defect"]
    for c in sorted(set(cat.tolist())):
        te = (cat == c) & (split == "test")
        if te.sum() < 20:
            continue
        y = (defect[te] != "good").astype(int)
        assert 0 < y.sum() < len(y), f"{c} test split is single-class"


@pytest.mark.skipif(not PASS4.exists(), reason="run_pass4 has not been run")
def test_the_measured_result_refutes_both_attributions():
    """The finding: between-category variation dwarfs the group difference, and
    the two textures point in opposite directions."""
    d = json.loads(PASS4.read_text(encoding="utf-8"))
    assert d["n_graded"] >= 4
    assert d["delta_spread"] > 3 * d["group_gap"]
    tex = [r for r in d["rows"] if r.get("kind") == "texture" and "delta" in r]
    assert any(r["delta"] > 0 for r in tex) and any(r["delta"] < 0 for r in tex)
    assert not d["undertrained"], f"still undertrained: {d['undertrained']}"
