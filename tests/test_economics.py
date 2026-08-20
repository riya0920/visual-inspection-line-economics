"""ML-2 tests: the Bayes arithmetic, the cost model, and the generator's masks.

The PPV arithmetic is the headline claim of the project and it is pure arithmetic,
so it gets checked against hand-computed values rather than against itself.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import economics as E  # noqa: E402
import synth  # noqa: E402


# ---------------------------------------------------------------- Bayes

def test_ppv_on_a_balanced_set_equals_the_textbook_value():
    # p=0.5, Se=0.95, Sp=0.95  ->  0.475 / (0.475 + 0.025) = 0.95
    assert E.ppv(0.5, 0.95, 0.95) == pytest.approx(0.95)


def test_ppv_collapses_at_realistic_prevalence():
    """Same model, same recall, same specificity -- only the base rate changed."""
    balanced = E.ppv(0.5, 0.95, 0.95)
    line = E.ppv(0.005, 0.95, 0.95)
    # 0.00475 / (0.00475 + 0.04975) = 0.0872
    assert line == pytest.approx(0.08716, rel=1e-3)
    assert balanced / line > 10


def test_ppv_is_monotone_in_prevalence_and_specificity():
    assert E.ppv(0.01, 0.9, 0.9) < E.ppv(0.10, 0.9, 0.9)
    assert E.ppv(0.01, 0.9, 0.9) < E.ppv(0.01, 0.9, 0.99)


def test_npv_stays_high_at_low_prevalence():
    """The counterpart nobody quotes: at 0.5% prevalence almost everything really
    is good, so a negative call is nearly always right even for a poor model."""
    assert E.npv(0.005, 0.5, 0.9) > 0.99


def test_perfect_specificity_gives_ppv_one():
    assert E.ppv(0.01, 0.8, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------- cost model

def test_expected_cost_is_linear_in_volume():
    a = E.expected_cost(0.005, 0.95, 0.97, 4.0, 400.0, volume=1_000)
    b = E.expected_cost(0.005, 0.95, 0.97, 4.0, 400.0, volume=10_000)
    assert b == pytest.approx(10 * a)


def test_expected_cost_matches_hand_arithmetic():
    # fr = 0.995*0.03 = 0.02985 ; esc = 0.005*0.05 = 0.00025
    got = E.expected_cost(0.005, 0.95, 0.97, c_false_reject=4.0, c_escape=400.0,
                          volume=200_000)
    want = 200_000 * (0.995 * 0.03 * 4.0 + 0.005 * 0.05 * 400.0)
    assert got == pytest.approx(want)


def test_higher_escape_cost_moves_the_operating_point_toward_recall():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(800, dtype=int), np.ones(200, dtype=int)]
    s = np.r_[rng.normal(0.3, 0.15, 800), rng.normal(0.7, 0.15, 200)]
    cheap = E.optimal_operating_point(s, y, 0.005, 4.0, 40.0)
    dear = E.optimal_operating_point(s, y, 0.005, 4.0, 4000.0)
    assert dear["sensitivity"] >= cheap["sensitivity"]
    assert dear["false_reject_rate"] >= cheap["false_reject_rate"]


# ---------------------------------------------------------------- takt

def test_takt_time_is_sixty_seconds_over_rate():
    t = E.takt_analysis(latency_ms=380.0, parts_per_minute=60.0)
    assert t["takt_ms"] == pytest.approx(1000.0)
    assert t["margin_ms"] == pytest.approx(620.0)
    assert t["fits"]


def test_a_pipeline_slower_than_takt_does_not_fit():
    t = E.takt_analysis(latency_ms=1500.0, parts_per_minute=60.0)
    assert not t["fits"]
    assert t["margin_ms"] < 0
    assert t["max_parts_per_minute_supported"] == pytest.approx(40.0)


# ---------------------------------------------------------------- ROC helpers

def test_sensitivity_and_specificity_are_prevalence_invariant():
    """The reason Se/Sp can be measured on a convenient mix and PPV cannot."""
    rng = np.random.default_rng(1)
    good = rng.normal(0.3, 0.1, 2000)
    bad = rng.normal(0.7, 0.1, 2000)
    balanced_s = np.r_[good, bad]
    balanced_y = np.r_[np.zeros(2000, dtype=int), np.ones(2000, dtype=int)]
    rare_s = np.r_[good, bad[:100]]
    rare_y = np.r_[np.zeros(2000, dtype=int), np.ones(100, dtype=int)]
    a = E.at_false_reject_rate(balanced_s, balanced_y, 0.05)
    b = E.at_false_reject_rate(rare_s, rare_y, 0.05)
    assert a["sensitivity"] == pytest.approx(b["sensitivity"], abs=0.06)


# ---------------------------------------------------------------- the generator

def test_good_samples_have_empty_masks():
    rng = np.random.default_rng(2)
    s = synth.make(60, 0.0, rng)
    assert all(x.label == 0 for x in s)
    assert all(not x.mask.any() for x in s)


def test_every_defect_has_a_nonempty_mask_and_a_type():
    rng = np.random.default_rng(3)
    s = [x for x in synth.make(200, 1.0, rng) if x.label == 1]
    assert s
    for x in s:
        assert x.mask.any(), x.defect_type
        assert x.defect_type in synth.DEFECT_TYPES


def test_defects_are_small_relative_to_the_part():
    """The regression guard for the dilation bug: a 'crack' that covers half the
    image is not a crack, and it would make the hardest class trivial."""
    rng = np.random.default_rng(4)
    s = [x for x in synth.make(300, 1.0, rng) if x.label == 1]
    for kind in synth.DEFECT_TYPES:
        areas = [x.mask.mean() for x in s if x.defect_type == kind]
        if areas:
            assert np.mean(areas) < 0.05, (kind, np.mean(areas))


def test_variants_have_distinguishable_texture():
    rng = np.random.default_rng(5)
    a = np.mean([x.image.std() for x in synth.make(40, 0.0, rng, variants=("A",))])
    d = np.mean([x.image.std() for x in synth.make(40, 0.0, rng, variants=("D",))])
    assert d > a, "variant D is meant to be rougher than A"
