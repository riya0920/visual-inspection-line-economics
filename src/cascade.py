"""The two-stage inspection cascade, actually wired -- plus Grad-CAM and the
operator-override log.

The first build ARGUED for a two-stage architecture and timed both stages
together, which is not the same thing as building one. This is the cascade:

    stage 1  ANOMALY SCREEN on every part. Cheap threshold, tuned for RECALL --
             its job is to let nothing suspicious through, not to be right.
    stage 2  SUPERVISED CLASSIFIER on whatever stage 1 flags. Its job is to say
             WHICH defect, so the part goes to the right disposition.
    output   accept / reject-as-class-X / flag-for-review

WHY A CASCADE AND NOT AN ENSEMBLE. The economics. On a line at 0.5% prevalence,
99%+ of parts are good, and stage 2 only runs on the small flagged fraction --
so the average cost per part is stage-1 cost plus (screen rate x stage-2 cost).
That is what makes a heavier classifier affordable at takt, and it is a
throughput argument rather than an accuracy one.

THE THIRD OUTPUT IS THE ONE THAT MATTERS. "Flag for review" -- stage 1 is
confident something is wrong, stage 2 cannot name it -- is the disposition an
unseen defect should receive. A two-outcome system forces the operator to pick a
wrong class to clear the screen, and then the override log fills with garbage and
the retraining set is poisoned by it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class CascadeConfig:
    screen_threshold: float          # stage-1 anomaly score above which we look closer
    classify_threshold: float        # stage-2 confidence needed to NAME a class
    screen_target_recall: float = 0.99


@dataclass
class CascadeResult:
    verdicts: list[str]
    screened: np.ndarray             # bool: did stage 1 flag it
    named: np.ndarray                # bool: did stage 2 name a class
    screen_rate: float
    stage2_fraction: float
    counts: dict = field(default_factory=dict)


def choose_screen_threshold(anomaly_scores: np.ndarray, labels: np.ndarray,
                            target_recall: float = 0.99) -> float:
    """Lowest threshold that still achieves the target recall on stage 1.

    Stage 1 is tuned for RECALL, deliberately, and its precision is allowed to be
    terrible -- anything it passes is gone forever, while anything it over-flags
    is merely handed to stage 2, which is cheap in aggregate because the flagged
    fraction is small. Tuning stage 1 for accuracy is the classic cascade mistake:
    it optimises the wrong stage's error.
    """
    pos = anomaly_scores[labels == 1]
    if len(pos) == 0:
        return float(np.max(anomaly_scores))
    return float(np.quantile(pos, 1.0 - target_recall))


def run(anomaly_scores: np.ndarray, supervised_scores: np.ndarray,
        cfg: CascadeConfig) -> CascadeResult:
    screened = anomaly_scores >= cfg.screen_threshold
    named = screened & (supervised_scores >= cfg.classify_threshold)
    verdicts = []
    for i in range(len(anomaly_scores)):
        if not screened[i]:
            verdicts.append("ACCEPT")
        elif named[i]:
            verdicts.append("REJECT_CLASSIFIED")
        else:
            verdicts.append("FLAG_FOR_REVIEW")
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    return CascadeResult(
        verdicts=verdicts, screened=screened, named=named,
        screen_rate=float(screened.mean()),
        stage2_fraction=float(screened.mean()),
        counts=counts,
    )


def cascade_cost_per_part(screen_rate: float, stage1_ms: float,
                          stage2_ms: float) -> dict:
    """Average inference time per part under the cascade vs running both always.

    This is the number that justifies the architecture on a line, and it is a
    throughput argument rather than an accuracy one.
    """
    cascade = stage1_ms + screen_rate * stage2_ms
    both = stage1_ms + stage2_ms
    return {
        "stage1_ms": stage1_ms, "stage2_ms": stage2_ms,
        "screen_rate": screen_rate,
        "cascade_avg_ms": cascade,
        "always_both_ms": both,
        "saving_pct": 100.0 * (both - cascade) / both if both else 0.0,
    }


# --------------------------------------------------------------------------
# Grad-CAM: localisation for the SUPERVISED head
# --------------------------------------------------------------------------

def grad_cam(model, x: np.ndarray, layer=None) -> np.ndarray:
    """Grad-CAM heat maps for the supervised classifier.

    The first build's supervised path had ZERO localisation -- it returned a
    scalar per image, which fails the spec's minimum ("Grad-CAM minimum") and,
    more practically, gives a quality engineer nothing to look at. An operator
    handed "this part is defective" with no indication of where cannot verify the
    call, and a verdict that cannot be verified gets rubber-stamped.

    Grad-CAM (Selvaraju et al., 2017): weight each channel of the last
    convolutional feature map by the mean gradient of the score with respect to
    that channel, sum, and ReLU. The result is coarse -- the spatial resolution of
    the last conv layer, upsampled -- which is exactly why the spec calls it a
    minimum rather than a substitute for segmentation.
    """
    model.eval()
    layer = layer or _last_conv(model)
    acts: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []

    h1 = layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        maps = []
        for b in range(0, len(x), 64):
            xb = torch.from_numpy(x[b:b + 64]).float()
            acts.clear()
            grads.clear()
            model.zero_grad()
            out = model(xb)
            score = out.squeeze(-1) if out.ndim > 1 else out
            score.sum().backward()
            a, g = acts[0], grads[0]
            w = g.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * a).sum(dim=1))
            maps.append(cam.detach().numpy())
        cam = np.concatenate(maps, axis=0)
    finally:
        h1.remove()
        h2.remove()
    # Normalise per image so maps are comparable on a review screen.
    flat = cam.reshape(len(cam), -1)
    lo = flat.min(axis=1)[:, None, None]
    hi = flat.max(axis=1)[:, None, None]
    return (cam - lo) / np.maximum(hi - lo, 1e-9)


def _last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("no Conv2d layer found")
    return last


# --------------------------------------------------------------------------
# the operator override log
# --------------------------------------------------------------------------

DISPOSITIONS = ("CONFIRMED_DEFECT", "FALSE_REJECT", "MISSED_DEFECT", "UNCLASSIFIED")


class OverrideLog:
    """Logged operator dispositions at the review station.

    The spec calls this "retraining gold AND a quality-system requirement", and
    both halves are true for the same reason: it is the only place where a human
    judgement about a specific part is captured against the model's judgement of
    the same part.

    What makes it retraining gold is not the volume -- it is the SAMPLING. These
    are not random labels, they are labels on exactly the parts the model found
    hard, which is the most informative set to buy. What makes it a trap is the
    same fact: a model retrained on override data alone is trained on a censored
    sample (only parts stage 1 flagged) and will drift toward the screen's biases.
    Both are stated here because a retraining plan that ignores the second is how
    a v2 model ends up worse than v1 on the parts nobody reviewed.
    """

    def __init__(self):
        self.rows: list[dict] = []

    def record(self, part_id: str, model_verdict: str, model_class: str | None,
               model_score: float, operator: str, disposition: str,
               note: str = "") -> None:
        assert disposition in DISPOSITIONS, disposition
        self.rows.append({
            "part_id": part_id, "model_verdict": model_verdict,
            "model_class": model_class, "model_score": float(model_score),
            "operator": operator, "disposition": disposition, "note": note,
            "ts": time.time(),
        })

    def summary(self) -> dict:
        by: dict[str, int] = {}
        for r in self.rows:
            by[r["disposition"]] = by.get(r["disposition"], 0) + 1
        n = len(self.rows) or 1
        confirmed = by.get("CONFIRMED_DEFECT", 0)
        false_rej = by.get("FALSE_REJECT", 0)
        return {
            "n_reviewed": len(self.rows),
            "by_disposition": by,
            # Operator-confirmed PPV: of the parts the model rejected and a human
            # looked at, how many were genuinely defective. This is the ONLY PPV
            # measurable in production without a destructive audit, and it is
            # measured on the flagged subset rather than on the line -- so it is an
            # upper bound on line PPV, not an estimate of it.
            "operator_confirmed_ppv": confirmed / max(1, confirmed + false_rej),
            "unclassified_rate": by.get("UNCLASSIFIED", 0) / n,
        }
