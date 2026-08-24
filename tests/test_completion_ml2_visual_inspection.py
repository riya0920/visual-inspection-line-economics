"""Tests for the third-pass modules."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inspection_ops as OPS  # noqa: E402
import patchcore as PC  # noqa: E402
import segmentation as SEG  # noqa: E402


# ---------------------------------------------------------------------------
# coreset selection
# ---------------------------------------------------------------------------

def test_kcenter_covers_better_than_a_random_subset():
    """Coverage, not density, is what a nearest-neighbour detector needs."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (3000, 16)).astype(np.float32)
    _, k = PC.greedy_kcenter(x, 120, seed=0)
    _, r = PC.random_subset(x, 120, seed=0)
    assert k["coverage_radius"] < r["coverage_radius"]


def test_coverage_radius_shrinks_as_the_bank_grows():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (2000, 8)).astype(np.float32)
    small = PC.greedy_kcenter(x, 20, seed=0)[1]["coverage_radius"]
    big = PC.greedy_kcenter(x, 200, seed=0)[1]["coverage_radius"]
    assert big < small


def test_the_pool_is_capped_and_the_capping_is_reported():
    """Regression: k-center is O(pool x selected), so an uncapped pool of 100k
    patches turned a two-minute stage into an hour."""
    x = np.random.default_rng(2).normal(0, 1, (300, 4, 4, 8)).astype(np.float32)
    pc = PC.PatchCore(rate=0.05).fit(x, pool_cap=1000)
    assert pc.info["pool_subsampled"]
    assert pc.info["pool_size"] == 1000
    assert pc.info["patches_available"] == 300 * 16


def test_selecting_more_than_exists_is_clamped():
    x = np.random.default_rng(3).normal(0, 1, (40, 5)).astype(np.float32)
    idx, info = PC.greedy_kcenter(x, 999)
    assert info["n_selected"] == 40 and len(set(idx.tolist())) == 40


# ---------------------------------------------------------------------------
# the bimodal case
# ---------------------------------------------------------------------------

def test_a_gaussian_scores_the_gap_between_two_modes_as_most_normal():
    """The failure PatchCore exists for, isolated.

    A Gaussian fitted to a bimodal normal puts its mean in the empty space
    between the modes, so an anomaly sitting there looks *more* normal than the
    legitimate appearances do.
    """
    tr = PC.make_bimodal_normal(500, dim=4, sep=8.0, seed=0)
    gap = np.zeros((100, 4), dtype=np.float32)
    gap[:, 0] = 4.0                       # exactly between the two modes
    d_gap = PC.mahalanobis_scores(tr, gap).mean()
    d_real = PC.mahalanobis_scores(tr, tr[:100]).mean()
    assert d_gap < d_real, "the Gaussian should call the gap more normal"


def test_patchcore_puts_the_gap_further_away_than_real_normals():
    tr = PC.make_bimodal_normal(500, dim=4, sep=8.0, seed=0)
    pc = PC.PatchCore(rate=0.2).fit(tr.reshape(len(tr), 1, 1, -1))
    gap = np.zeros((100, 4), dtype=np.float32)
    gap[:, 0] = 4.0
    s_gap = pc.score_maps(gap.reshape(len(gap), 1, 1, -1)).mean()
    s_real = pc.score_maps(tr[:100].reshape(100, 1, 1, -1)).mean()
    assert s_gap > s_real


def test_image_score_is_the_max_patch_not_the_mean():
    """A defect is under 1% of the image; any averaging buries it."""
    pc = PC.PatchCore(rate=0.5)
    pc.bank = np.zeros((4, 3), dtype=np.float32)
    patches = np.zeros((1, 4, 4, 3), dtype=np.float32)
    patches[0, 2, 2] = 9.0
    assert pc.image_scores(patches)[0] == pytest.approx(9.0 * np.sqrt(3), rel=1e-3)


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

def test_dice_punishes_the_degenerate_all_background_prediction():
    """Under BCE alone that prediction scores 99%+ pixel accuracy.

    Sized at the project's real prevalence (~0.8% of a 128x128 image). The
    smoothing term is not negligible on a toy 4-pixel mask -- it caps the loss
    at 0.8 there -- so a test at toy scale would understate how hard Dice
    actually punishes an empty prediction.
    """
    import torch
    target = torch.zeros(2, 128, 128)
    target[:, 60:72, 60:72] = 1.0                   # 144 px ~ 0.9% prevalence
    empty = torch.full((2, 128, 128), -20.0)        # sigmoid -> ~0
    assert float(SEG.dice_loss(empty, target)) > 0.98


def test_dice_is_one_when_a_correct_empty_prediction_meets_an_empty_target():
    import torch
    target = torch.zeros(2, 16, 16)
    empty = torch.full((2, 16, 16), -20.0)
    assert float(SEG.dice_loss(empty, target)) == pytest.approx(0.0, abs=0.02)


def test_pixel_accuracy_is_useless_at_this_prevalence():
    """The degenerate baseline is the number every segmentation result must be
    read against."""
    target = np.zeros((10, 64, 64), dtype=np.float32)
    target[:, 30:33, 30:33] = 1.0
    base = SEG.degenerate_baseline(target)
    assert base["pixel_accuracy"] > 0.99
    assert base["iou"] == 0.0


def test_iou_and_dice_reward_actual_overlap():
    target = np.zeros((1, 32, 32), dtype=np.float32)
    target[0, 8:16, 8:16] = 1.0
    perfect = target.copy()
    assert SEG.iou(perfect, target)["iou"] == pytest.approx(1.0)
    half = np.zeros_like(target)
    half[0, 8:12, 8:16] = 1.0
    assert SEG.iou(half, target)["iou"] == pytest.approx(0.5)


def test_shape_normalisation_accepts_both_layouts():
    """synth.to_arrays returns (n,1,H,W) for images and (n,H,W) for masks."""
    assert SEG._chw(np.zeros((3, 8, 8))).shape == (3, 1, 8, 8)
    assert SEG._chw(np.zeros((3, 1, 8, 8))).shape == (3, 1, 8, 8)
    assert SEG._hw(np.zeros((3, 1, 8, 8))).shape == (3, 8, 8)
    with pytest.raises(ValueError):
        SEG._chw(np.zeros((8, 8)))


# ---------------------------------------------------------------------------
# gauge R&R
# ---------------------------------------------------------------------------

def test_no_reacquisition_reports_perfect_repeatability_which_is_the_trap():
    """A deterministic model re-scored on the identical array measures nothing."""
    scores = np.random.default_rng(0).normal(0, 1, (10, 3, 1))
    fixed = np.repeat(scores, 3, axis=2)
    assert OPS.anova_grr(fixed)["pct_repeatability"] == pytest.approx(0.0, abs=1e-9)


def test_repeat_trials_actually_change_the_image():
    img = np.full((2, 8, 8), 0.5, dtype=np.float32)
    a, b = OPS.repeat_trials(img, 2, seed=0)
    assert not np.allclose(a, b), "trials must re-acquire, not re-use"


def test_station_differences_show_up_as_reproducibility():
    rng = np.random.default_rng(0)
    parts = rng.normal(0, 1, (12,))
    scores = np.zeros((12, 3, 3))
    for s in range(3):
        for t in range(3):
            scores[:, s, t] = parts + s * 5.0 + rng.normal(0, 0.01, 12)
    g = OPS.anova_grr(scores)
    assert g["pct_reproducibility"] > g["pct_repeatability"]


def test_grr_verdict_follows_the_aiag_bands():
    parts = np.arange(20, dtype=float) * 10
    clean = np.repeat(parts[:, None, None], 3, 1).repeat(3, 2)
    assert OPS.anova_grr(clean)["verdict"] == "acceptable"


def test_attribute_agreement_is_perfect_when_stations_agree():
    d = np.tile(np.array([[1], [0], [1], [0]]), (1, 3))
    out = OPS.attribute_agreement(d)
    assert out["all_stations_agree"] == 1.0
    assert out["mean_kappa"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the retraining loop
# ---------------------------------------------------------------------------

def test_rarely_flagged_parts_carry_the_most_weight():
    """A part reviewed with probability p stands for 1/p parts like itself."""
    w = OPS.propensity_weights(np.array([1.0, 0.5, 0.1]))
    assert w[0] == pytest.approx(1.0)
    assert w[2] > w[1] > w[0]


def test_propensity_weights_are_clipped():
    """Unclipped, one review at p=0.001 would dominate the whole fit."""
    w = OPS.propensity_weights(np.array([1e-6]), clip=20.0)
    assert w[0] == pytest.approx(20.0)


def test_the_golden_set_is_stratified_and_frozen():
    y = np.array([0] * 80 + [1] * 20)
    x = np.zeros((100, 4))
    a = OPS.golden_set(x, y, n=20, seed=7)
    b = OPS.golden_set(x, y, n=20, seed=7)
    assert np.array_equal(a, b), "the same parts every release, or it is not golden"
    assert y[a].sum() == len(a) // 2


def test_the_requalification_gate_is_asymmetric():
    """A missed defect ships; a false reject costs a re-inspection."""
    old = {"recall": 0.90, "ppv": 0.60}
    recall_drop = OPS.requalification_gate(old, {"recall": 0.86, "ppv": 0.60})
    ppv_drop = OPS.requalification_gate(old, {"recall": 0.90, "ppv": 0.56})
    assert not recall_drop["passed"], "a 4-point recall drop must block"
    assert ppv_drop["passed"], "a 4-point PPV drop is tolerated"


def test_the_gate_passes_an_improvement():
    out = OPS.requalification_gate({"recall": 0.9, "ppv": 0.6},
                                   {"recall": 0.93, "ppv": 0.65})
    assert out["passed"] and not out["reasons"]
