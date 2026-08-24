"""Gauge R&R for a vision system, and a retraining loop that corrects its own
sampling bias.

These are the two remaining items from the README's not-built list, and they are
the two that make the difference between a model and an inspection system.

===========================================================================
PART 1 -- GAUGE R&R, applied to a camera instead of a micrometer
===========================================================================

A quality department will not accept an inspection station on an AUROC. It will
ask for a measurement systems analysis, because AIAG MSA-4 says a gauge must be
qualified before its readings are used to accept product. The vocabulary
transfers exactly:

  REPEATABILITY (equipment variation)
      Same part, same station, measured repeatedly. For a micrometer this is the
      operator's grip. For a vision station it is sensor noise, lighting ripple,
      exposure jitter and any non-determinism in the model. **A deterministic
      model scores perfect repeatability, which is a trap**: it means the metric
      is measuring the CAMERA, not the model, and a station that reports zero
      repeatability variation has almost certainly not re-acquired the image
      between trials.

  REPRODUCIBILITY (appraiser variation)
      Same part, DIFFERENT stations. For a micrometer this is operator to
      operator. For vision it is line 1's camera against line 2's -- different
      lighting, focus, working distance, sensor. This is where vision systems
      actually fail, and it is why a model validated on one station is not
      qualified for another.

  %GRR = 100 * sqrt(repeatability^2 + reproducibility^2) / total variation
      AIAG acceptance: <10% acceptable, 10-30% marginal (acceptable depending on
      application and cost), >30% unacceptable.

THE HONEST DIFFICULTY. A vision system's output is a continuous anomaly score,
so a variance decomposition is well defined. But the DECISION is a binary
accept/reject, and %GRR says nothing about whether the variation matters at the
threshold. A station can have excellent %GRR and still disagree with its
neighbour on every borderline part, because all its variation sits exactly where
the threshold is. So the kappa between stations is reported alongside -- attribute
agreement analysis, which is what MSA-4 actually prescribes for a go/no-go gauge.

===========================================================================
PART 2 -- THE RETRAINING LOOP, and the censoring that poisons it
===========================================================================

docs/EXTENSIONS.md built the operator override log and called it "retraining gold
and a trap", for the same reason:

    gold  the labels are on exactly the parts the model found hard, which is the
          most informative set anyone could buy
    trap  it is CENSORED -- it contains only what stage 1 flagged. Parts the
          screen passed are never reviewed, so they never enter the log, so
          retraining on it alone drifts the model toward the screen's own biases
          and gets worse on everything nobody looked at.

This implements the loop AND the correction. The correction is inverse-propensity
weighting: a reviewed part flagged with probability p represents 1/p parts like
itself, so it is weighted accordingly. Parts from a region the screen almost
always flags get weight ~1; parts from a region it rarely flags get large weight,
because each one stands for many unreviewed siblings.

And a re-qualification gate, because a retrained model that has not been
re-qualified is an unreviewed change to an inspection system. The gate is a
golden-sample set held out of every training loop forever.
"""
from __future__ import annotations

import numpy as np


# ===========================================================================
# gauge R&R
# ===========================================================================

def simulate_stations(x: np.ndarray, n_stations: int = 3, *,
                      gain: float = 0.06, offset: float = 0.05,
                      focus: float = 0.4, seed: int = 0,
                      ) -> list[np.ndarray]:
    """Re-image the same parts on different stations.

    The three perturbations are the three things that actually differ between two
    nominally identical inspection cells, in rough order of how often they are the
    culprit:

      offset  lighting level -- different lamp age or ambient contribution
      gain    exposure / aperture
      focus   working distance, applied as a small blur

    Not modelled: perspective, which is the fourth and would need a real
    homography. Its absence makes this an optimistic estimate of reproducibility.
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_stations):
        g = 1.0 + rng.normal(0, gain)
        o = rng.normal(0, offset)
        img = np.clip(x * g + o, 0.0, 1.0)
        if focus > 0 and s > 0:
            k = float(abs(rng.normal(0, focus)))
            if k > 0.15:
                img = _blur(img, k)
        out.append(img.astype(np.float32))
    return out


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur over the last two axes, shape-agnostic.

    scipy rather than a hand-rolled separable convolution. The hand-rolled one
    reshaped by position and broke the moment the input arrived as (n, 1, H, W)
    instead of (n, H, W) -- and reshape errors of that kind are the ones that
    silently transpose an image when they do not raise.
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(x, dtype=np.float32)
    axes = (a.ndim - 2, a.ndim - 1)
    sig = [0.0] * a.ndim
    sig[axes[0]] = sigma
    sig[axes[1]] = sigma
    return gaussian_filter(a, sigma=sig, mode="nearest")


def repeat_trials(img: np.ndarray, n_trials: int = 3, *, noise: float = 0.012,
                  seed: int = 0) -> list[np.ndarray]:
    """Re-ACQUIRE the same part on the same station.

    Sensor read noise is added on every trial, deliberately. Feeding the identical
    array n times measures nothing -- a deterministic model returns the identical
    score and repeatability comes out as exactly zero, which reads as a perfect
    gauge and is really a broken experiment. Repeatability of a vision station is
    a property of the ACQUISITION, and if the trials do not re-acquire, there is
    no repeatability to measure.
    """
    rng = np.random.default_rng(seed)
    return [np.clip(img + rng.normal(0, noise, img.shape), 0, 1).astype(np.float32)
            for _ in range(n_trials)]


def anova_grr(scores: np.ndarray) -> dict:
    """Crossed ANOVA gauge R&R. `scores` is (parts, stations, trials).

    The standard AIAG decomposition. Variance components can come out negative
    when the true component is near zero and the estimate undershoots; they are
    clamped at zero, which is what MSA-4 prescribes and is worth stating because
    a negative variance in a report means somebody did not.
    """
    s = np.asarray(scores, dtype=float)
    p, k, n = s.shape
    grand = s.mean()
    part_m = s.mean(axis=(1, 2))
    stn_m = s.mean(axis=(0, 2))
    cell_m = s.mean(axis=2)

    ss_part = k * n * ((part_m - grand) ** 2).sum()
    ss_stn = p * n * ((stn_m - grand) ** 2).sum()
    ss_inter = n * ((cell_m - part_m[:, None] - stn_m[None, :] + grand) ** 2).sum()
    ss_rep = ((s - cell_m[:, :, None]) ** 2).sum()

    df_part, df_stn = p - 1, k - 1
    df_inter, df_rep = (p - 1) * (k - 1), p * k * (n - 1)
    ms_part = ss_part / max(df_part, 1)
    ms_stn = ss_stn / max(df_stn, 1)
    ms_inter = ss_inter / max(df_inter, 1)
    ms_rep = ss_rep / max(df_rep, 1) if df_rep else 0.0

    v_rep = max(ms_rep, 0.0)
    v_inter = max((ms_inter - ms_rep) / n, 0.0)
    v_stn = max((ms_stn - ms_inter) / (p * n), 0.0)
    v_part = max((ms_part - ms_inter) / (k * n), 0.0)

    v_reprod = v_stn + v_inter
    v_grr = v_rep + v_reprod
    v_total = v_grr + v_part
    pct = (lambda v: 100.0 * np.sqrt(v / v_total) if v_total > 0 else 0.0)
    grr_pct = pct(v_grr)
    return {
        "var_repeatability": v_rep, "var_reproducibility": v_reprod,
        "var_station": v_stn, "var_interaction": v_inter, "var_part": v_part,
        "var_total": v_total,
        "pct_repeatability": pct(v_rep), "pct_reproducibility": pct(v_reprod),
        "pct_part": pct(v_part), "pct_grr": grr_pct,
        "verdict": ("acceptable" if grr_pct < 10 else
                    "marginal" if grr_pct < 30 else "unacceptable"),
        "ndc": max(int(np.floor(1.41 * np.sqrt(v_part / v_grr))), 0)
        if v_grr > 0 else 99,
        "n_parts": p, "n_stations": k, "n_trials": n,
    }


def attribute_agreement(decisions: np.ndarray) -> dict:
    """Cohen's kappa between stations on the accept/reject CALL.

    %GRR is a variance statistic on a continuous score and can look excellent
    while two stations disagree on every borderline part -- if all the variation
    happens to sit at the threshold. MSA-4 prescribes attribute agreement analysis
    for a go/no-go gauge, and this is it.
    """
    d = np.asarray(decisions, dtype=int)          # (parts, stations)
    p, k = d.shape
    pairs = []
    for a in range(k):
        for b in range(a + 1, k):
            x, y = d[:, a], d[:, b]
            po = float((x == y).mean())
            px, py = x.mean(), y.mean()
            pe = px * py + (1 - px) * (1 - py)
            kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
            pairs.append({"stations": f"{a}~{b}", "agreement": po,
                          "kappa": float(kappa)})
    all_agree = float(np.mean([(row == row[0]).all() for row in d]))
    return {"pairs": pairs, "all_stations_agree": all_agree,
            "mean_kappa": float(np.mean([p_["kappa"] for p_ in pairs]))}


# ===========================================================================
# the retraining loop
# ===========================================================================

def propensity_weights(flag_prob: np.ndarray, clip: float = 20.0) -> np.ndarray:
    """Inverse-propensity weights for a censored review log.

    A part reviewed because the screen flagged it with probability p stands for
    1/p parts like itself. Without this, retraining on the log inherits the
    screen's blind spots -- the model gets better at the parts the screen already
    catches and no better at the ones it misses, which is precisely backwards.

    Clipped, because a part flagged with probability 0.001 would otherwise carry
    weight 1000 and a single review would dominate the fit. Clipping trades a
    little bias for a lot of variance and is the standard practice; the clip value
    is reported so a reader can see how much was traded.
    """
    p = np.clip(np.asarray(flag_prob, dtype=float), 1.0 / clip, 1.0)
    return 1.0 / p


def golden_set(x: np.ndarray, y: np.ndarray, n: int = 60, seed: int = 0):
    """A qualification set held out of every training loop, forever.

    Deliberately stratified and FROZEN. The point of a golden sample set in a
    quality system is that it is the same parts every time, so successive model
    versions are comparable to each other and not merely each to its own test
    split. Re-sampling it per release would destroy exactly that.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    take = min(n // 2, len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, take, replace=False),
                          rng.choice(neg, take, replace=False)])
    idx.sort()
    return idx


def requalification_gate(old_metrics: dict, new_metrics: dict, *,
                         max_recall_drop: float = 0.01,
                         max_ppv_drop: float = 0.05) -> dict:
    """Decide whether a retrained model may reach the line.

    Asymmetric on purpose, and the asymmetry is the whole policy. Recall is
    allowed to drop by 1% and precision by 5%, because a missed defect ships to a
    customer and a false reject costs a re-inspection. A gate with a symmetric
    tolerance has quietly decided those two are equally bad, which no quality
    engineer believes.
    """
    dr = new_metrics["recall"] - old_metrics["recall"]
    dp = new_metrics["ppv"] - old_metrics["ppv"]
    reasons = []
    if dr < -max_recall_drop:
        reasons.append(f"recall dropped {-dr:.3f} (limit {max_recall_drop})")
    if dp < -max_ppv_drop:
        reasons.append(f"PPV dropped {-dp:.3f} (limit {max_ppv_drop})")
    return {"passed": not reasons, "reasons": reasons,
            "delta_recall": float(dr), "delta_ppv": float(dp)}
