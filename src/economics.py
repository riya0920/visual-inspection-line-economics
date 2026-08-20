"""Line economics: PPV at real prevalence, the cost matrix, and takt time.

THE SINGLE MOST-MISSED REALITY IN INSPECTION ML
-----------------------------------------------
A model validated on a balanced test set and reported at "98% accuracy" tells you
nothing about a production line, because production lines do not run at 50%
defect rate. They run at 0.1-2%.

Bayes, with the arithmetic written out because it is the whole argument. At
prevalence p, sensitivity (recall) Se, and specificity Sp:

    PPV = P(defect | flagged) =        p * Se
                                 ---------------------------
                                 p * Se + (1 - p) * (1 - Sp)

At p = 0.5 (a balanced test set) with Se = 0.95 and Sp = 0.95, PPV = 0.95.
At p = 0.005 (a real line) with the SAME model, PPV = 0.087.

Same model. Same recall. Same specificity. Nine out of ten flagged parts are good
parts. Nothing about the model changed -- the base rate did. Any conversation
about an inspection system that has not done this arithmetic is a conversation
about the wrong number.

THE COST MATRIX
---------------
Once PPV is understood, the operating point is not a statistics question, it is an
arithmetic one:

    false reject (a good part scrapped or re-inspected):  C_fr
    escape       (a defective part shipped):              C_esc
    the ratio C_esc / C_fr is the only thing that matters, and it spans three
    orders of magnitude between a cosmetic blemish and a safety-critical casting.

So the operating point is swept across the ratio, not picked once.
"""
from __future__ import annotations

import numpy as np


def ppv(prevalence: float, sensitivity: float, specificity: float) -> float:
    tp = prevalence * sensitivity
    fp = (1 - prevalence) * (1 - specificity)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def npv(prevalence: float, sensitivity: float, specificity: float) -> float:
    tn = (1 - prevalence) * specificity
    fn = prevalence * (1 - sensitivity)
    return float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0


def roc_points(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Sensitivity and specificity at every threshold, from a BALANCED-ish sample.

    Se and Sp are properties of the model and are prevalence-INVARIANT, which is
    exactly why they can be measured on whatever mix is convenient and then
    combined with the real prevalence afterwards. PPV is not, which is why it
    cannot be.
    """
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    P, N = int((y == 1).sum()), int((y == 0).sum())
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    se = tp / max(P, 1)
    sp = 1.0 - fp / max(N, 1)
    return {"thresholds": s, "sensitivity": se, "specificity": sp}


def at_false_reject_rate(scores: np.ndarray, labels: np.ndarray,
                         target_fr: float) -> dict:
    """Recall at a fixed false-reject rate -- the number a plant manager asks for."""
    r = roc_points(scores, labels)
    fr = 1.0 - r["specificity"]
    idx = int(np.searchsorted(fr, target_fr, side="right") - 1)
    idx = max(0, min(idx, len(fr) - 1))
    return {"threshold": float(r["thresholds"][idx]),
            "sensitivity": float(r["sensitivity"][idx]),
            "specificity": float(r["specificity"][idx]),
            "false_reject_rate": float(fr[idx])}


def prevalence_table(scores: np.ndarray, labels: np.ndarray,
                     prevalences=(0.005, 0.02, 0.10, 0.50),
                     target_fr: float = 0.03) -> list[dict]:
    op = at_false_reject_rate(scores, labels, target_fr)
    rows = []
    for p in prevalences:
        rows.append({
            "prevalence": p,
            "sensitivity": op["sensitivity"],
            "specificity": op["specificity"],
            "ppv": ppv(p, op["sensitivity"], op["specificity"]),
            "npv": npv(p, op["sensitivity"], op["specificity"]),
            "good_parts_rejected_per_1000": (1 - p) * (1 - op["specificity"]) * 1000,
            "defects_escaping_per_1000": p * (1 - op["sensitivity"]) * 1000,
        })
    return rows


def expected_cost(prevalence: float, sensitivity: float, specificity: float,
                  c_false_reject: float, c_escape: float,
                  volume: int = 1) -> float:
    """Expected cost per `volume` parts inspected."""
    fr = (1 - prevalence) * (1 - specificity)
    esc = prevalence * (1 - sensitivity)
    return float(volume * (fr * c_false_reject + esc * c_escape))


def optimal_operating_point(scores: np.ndarray, labels: np.ndarray,
                            prevalence: float, c_false_reject: float,
                            c_escape: float, volume: int = 1) -> dict:
    """Argmin of the expected-cost curve over all thresholds."""
    r = roc_points(scores, labels)
    costs = np.array([
        expected_cost(prevalence, se, sp, c_false_reject, c_escape, volume)
        for se, sp in zip(r["sensitivity"], r["specificity"])])
    i = int(np.argmin(costs))
    return {
        "threshold": float(r["thresholds"][i]),
        "sensitivity": float(r["sensitivity"][i]),
        "specificity": float(r["specificity"][i]),
        "false_reject_rate": float(1 - r["specificity"][i]),
        "ppv": ppv(prevalence, float(r["sensitivity"][i]), float(r["specificity"][i])),
        "expected_cost": float(costs[i]),
        "cost_ratio": c_escape / c_false_reject,
    }


def cost_ratio_sweep(scores: np.ndarray, labels: np.ndarray, prevalence: float,
                     c_false_reject: float = 4.0,
                     ratios=(10, 50, 100, 300, 1000),
                     volume: int = 200_000) -> list[dict]:
    out = []
    for ratio in ratios:
        r = optimal_operating_point(scores, labels, prevalence, c_false_reject,
                                    c_false_reject * ratio, volume)
        r["c_escape"] = c_false_reject * ratio
        out.append(r)
    return out


def takt_analysis(latency_ms: float, parts_per_minute: float,
                  stages: dict[str, float] | None = None) -> dict:
    """Does the inspection fit inside the takt time?

    Takt time is the beat of the line: the time available per part if the line is
    to meet demand. At 60 parts/minute the takt is 1,000 ms, and an inspection
    pipeline taking 380 ms has 620 ms of margin. Quoting a model's latency without
    the takt time is quoting half a sentence.

    The margin is not slack to be spent. It absorbs the tail -- p99 rather than
    mean -- plus image acquisition, transfer, and the reject-mechanism actuation
    that has to happen before the part passes the diverter.
    """
    takt_ms = 60_000.0 / parts_per_minute
    return {
        "parts_per_minute": parts_per_minute,
        "takt_ms": takt_ms,
        "inference_ms": latency_ms,
        "margin_ms": takt_ms - latency_ms,
        "utilisation_pct": 100.0 * latency_ms / takt_ms,
        "fits": bool(latency_ms < takt_ms),
        "max_parts_per_minute_supported": 60_000.0 / latency_ms if latency_ms > 0 else float("inf"),
        "stages": stages or {},
    }
