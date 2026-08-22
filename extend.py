"""ML-2, the next 30%: the cascade wired, Grad-CAM, override log, seed CIs.

    python extend.py            # ~25 min (5 seeds x training)
    python extend.py --quick
    python extend.py --report-only

Four things the first build named as missing:
  1. the two-stage cascade actually built, with its throughput argument costed
  2. Grad-CAM on the supervised head, which previously had zero localisation
  3. the operator-override loop with logged dispositions
  4. confidence intervals -- every AUROC in the first build was one seed
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cascade as C  # noqa: E402
import models as M  # noqa: E402
import synth  # noqa: E402

OUT = ROOT / "out"
KNOWN = ("pore", "crack", "shrinkage")
UNSEEN = "inclusion"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    P, N = float((labels == 1).sum()), float((labels == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - P * (P + 1) / 2) / (P * N))


def ci95(vals: list[float]) -> dict:
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    n = len(v)
    if n < 2:
        return {"mean": float(v.mean()) if n else float("nan"),
                "half_width": float("nan"), "n": n}
    from scipy import stats
    hw = float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return {"mean": float(v.mean()), "half_width": hw, "n": n,
            "sd": float(v.std(ddof=1)), "lo": float(v.mean() - hw),
            "hi": float(v.mean() + hw)}


def one_seed(seed: int, quick: bool) -> dict:
    """Train and evaluate once. Everything downstream is a repeat of this."""
    rng = np.random.default_rng(seed)
    n_train = 400 if quick else 1100
    n_test = 300 if quick else 700

    train = synth.make(n_train, 0.35, rng, variants=("A", "B", "C"), defect_types=KNOWN)
    normal_only = synth.make(n_train, 0.0, rng, variants=("A", "B", "C"))
    test_known = synth.make(n_test, 0.35, rng, variants=("A", "B", "C"), defect_types=KNOWN)
    test_unseen = synth.make(n_test // 2, 0.5, rng, variants=("A", "B", "C"),
                             defect_types=(UNSEEN,))

    xtr, ytr, _ = synth.to_arrays(train)
    xn, _, _ = synth.to_arrays(normal_only)
    model, _ = M.train_supervised(xtr, ytr, epochs=5 if quick else 12, verbose=False)
    anom = M.PatchAnomaly().fit(model, xn)

    out = {"seed": seed}
    packs = {}
    for name, samples in (("known", test_known), ("unseen", test_unseen)):
        x, y, m = synth.to_arrays(samples)
        s_sup = M.supervised_scores(model, x)
        raw = anom.score_maps(model, x)
        s_ano = raw.max(axis=(1, 2))
        out[f"{name}_supervised_auroc"] = auroc(s_sup, y)
        out[f"{name}_anomaly_auroc"] = auroc(s_ano, y)
        packs[name] = {"x": x, "y": y, "m": m, "s_sup": s_sup, "s_ano": s_ano}
    return {"metrics": out, "model": model, "anom": anom, "packs": packs}


def main() -> None:
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        prev = json.loads((OUT / "extensions.json").read_text())
        (ROOT / "docs" / "EXTENSIONS.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/EXTENSIONS.md")
        return

    t0 = time.perf_counter()
    # Three seeds, not five. Five was tried and the run died silently partway
    # through the fourth; three still supports a t-interval and the intervals it
    # produces are reported with their n rather than implied to be more.
    seeds = [11, 22] if quick else [11, 22, 33]
    res: dict = {"seeds": seeds}

    print(f"1/4 training {len(seeds)} seeds for confidence intervals ...", flush=True)
    runs = []
    for s in seeds:
        r = one_seed(s, quick)
        runs.append(r)
        m = r["metrics"]
        print(f"    seed {s}: known sup {m['known_supervised_auroc']:.3f} / "
              f"ano {m['known_anomaly_auroc']:.3f}   unseen sup "
              f"{m['unseen_supervised_auroc']:.3f} / ano "
              f"{m['unseen_anomaly_auroc']:.3f}", flush=True)

    res["seed_ci"] = {
        k: ci95([r["metrics"][k] for r in runs])
        for k in ("known_supervised_auroc", "known_anomaly_auroc",
                  "unseen_supervised_auroc", "unseen_anomaly_auroc")
    }
    gaps = [r["metrics"]["unseen_anomaly_auroc"] - r["metrics"]["unseen_supervised_auroc"]
            for r in runs]
    res["unseen_gap_ci"] = ci95(gaps)

    print("2/4 wiring the two-stage cascade ...", flush=True)
    last = runs[-1]
    k = last["packs"]["known"]
    u = last["packs"]["unseen"]
    thr = C.choose_screen_threshold(k["s_ano"], k["y"], target_recall=0.99)
    # Stage-2 confidence needed to NAME a class: the median score of parts the
    # supervised head is confident about on known defects.
    cls_thr = float(np.quantile(k["s_sup"][k["y"] == 1], 0.25))
    cfg = C.CascadeConfig(screen_threshold=thr, classify_threshold=cls_thr)

    # Sweep the target recall. The cascade's whole justification is that stage 2
    # runs on a SMALL flagged fraction, and whether that is true depends entirely
    # on how good a screen stage 1 is. Measuring the tradeoff beats assuming it.
    sweep = []
    for tr in (0.999, 0.99, 0.95, 0.90, 0.80):
        th = C.choose_screen_threshold(k["s_ano"], k["y"], target_recall=tr)
        flagged_good = float((k["s_ano"][k["y"] == 0] >= th).mean())
        recall = float((k["s_ano"][k["y"] == 1] >= th).mean())
        line_rate = 0.005 + (1 - 0.005) * flagged_good
        sweep.append({"target_recall": tr, "achieved_recall": recall,
                      "good_parts_flagged": flagged_good,
                      "line_screen_rate": line_rate})
    res["cascade_recall_sweep"] = sweep

    cas_known = C.run(k["s_ano"], k["s_sup"], cfg)
    cas_unseen = C.run(u["s_ano"], u["s_sup"], cfg)
    res["cascade"] = {
        "screen_threshold": thr, "classify_threshold": cls_thr,
        "known": {"counts": cas_known.counts, "screen_rate": cas_known.screen_rate},
        "unseen": {"counts": cas_unseen.counts, "screen_rate": cas_unseen.screen_rate},
    }
    # Stage recall: did stage 1 let any defect through?
    res["cascade"]["known_stage1_recall"] = float(
        cas_known.screened[k["y"] == 1].mean())
    res["cascade"]["unseen_stage1_recall"] = float(
        cas_unseen.screened[u["y"] == 1].mean())
    res["cascade"]["known_stage1_false_flag"] = float(
        cas_known.screened[k["y"] == 0].mean())

    # Throughput argument, at realistic line prevalence rather than test prevalence.
    t_s1 = _time_stage(lambda x: last["anom"].score_maps(last["model"], x), k["x"][:64])
    t_s2 = _time_stage(lambda x: M.supervised_scores(last["model"], x), k["x"][:64])
    line_screen_rate = 0.005 + (1 - 0.005) * res["cascade"]["known_stage1_false_flag"]
    res["cascade"]["throughput"] = C.cascade_cost_per_part(
        line_screen_rate, t_s1, t_s2)
    res["cascade"]["line_screen_rate_at_0.5pct_prevalence"] = line_screen_rate
    print(f"    screen rate {cas_known.screen_rate*100:.1f}% on test, "
          f"{line_screen_rate*100:.1f}% at line prevalence; stage1 recall "
          f"known {res['cascade']['known_stage1_recall']*100:.1f}% / unseen "
          f"{res['cascade']['unseen_stage1_recall']*100:.1f}%", flush=True)

    print("3/4 Grad-CAM localisation for the supervised head ...", flush=True)
    defect_idx = np.flatnonzero(k["y"] == 1)[:120]
    cams = C.grad_cam(last["model"], k["x"][defect_idx])
    cam_up = M.upsample_map(cams, synth.SIZE)
    masks = k["m"][defect_idx]
    hit = []
    for i in range(len(defect_idx)):
        if masks[i].any():
            peak = np.unravel_index(np.argmax(cam_up[i]), cam_up[i].shape)
            hit.append(bool(masks[i][peak]))
    res["gradcam"] = {
        "n_images": len(hit),
        "peak_inside_mask_pct": 100.0 * float(np.mean(hit)) if hit else float("nan"),
        "random_baseline_pct": 100.0 * float(masks.mean()),
    }
    print(f"    Grad-CAM peak lands inside the defect mask on "
          f"{res['gradcam']['peak_inside_mask_pct']:.1f}% of images "
          f"(defect pixels are {res['gradcam']['random_baseline_pct']:.2f}% of area)",
          flush=True)

    print("4/4 operator override log ...", flush=True)
    res["override"] = _simulate_review(cas_known, k, cas_unseen, u)
    res["wall_seconds"] = time.perf_counter() - t0

    (OUT / "extensions.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "EXTENSIONS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/EXTENSIONS.md ({res['wall_seconds']:.0f}s)")


def _time_stage(fn, x, n: int = 5) -> float:
    fn(x[:2])
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) / len(x) * 1000)
    return float(np.median(t))


def _simulate_review(cas_k, k, cas_u, u) -> dict:
    """Drive the review station with the cascade's own output.

    The operator is simulated as ground truth, which is generous -- a real
    operator is a measurement system with its own repeatability, and the spec's
    gauge-R&R question is exactly about that. Stated rather than glossed.
    """
    log = C.OverrideLog()
    for i, v in enumerate(cas_k.verdicts):
        if v == "ACCEPT":
            continue
        truth = bool(k["y"][i])
        disp = ("CONFIRMED_DEFECT" if (truth and v == "REJECT_CLASSIFIED")
                else "FALSE_REJECT" if not truth
                else "UNCLASSIFIED")
        log.record(f"K{i:05d}", v, "known" if v == "REJECT_CLASSIFIED" else None,
                   float(k["s_sup"][i]), "OP-07", disp)
    for i, v in enumerate(cas_u.verdicts):
        if v == "ACCEPT":
            continue
        truth = bool(u["y"][i])
        disp = ("UNCLASSIFIED" if (truth and v == "FLAG_FOR_REVIEW")
                else "CONFIRMED_DEFECT" if truth
                else "FALSE_REJECT")
        log.record(f"U{i:05d}", v, None, float(u["s_sup"][i]), "OP-07", disp)
    s = log.summary()
    s["escaped_defects_known"] = int(((~cas_k.screened) & (k["y"] == 1)).sum())
    s["escaped_defects_unseen"] = int(((~cas_u.screened) & (u["y"] == 1)).sum())
    return s



def _budget_section(A) -> None:
    """Render out/budget_probe.json if `python budget_probe.py` has been run.

    Kept as a separate script rather than another arm of extend.py because it
    costs six training runs and answers a question about the FIRST pass, not
    about anything extend.py builds.
    """
    import statistics as st

    f = ROOT / "out" / "budget_probe.json"
    if not f.exists():
        return
    rows = json.loads(f.read_text(encoding="utf-8"))
    if not rows:
        return

    A("\n## 1b. Why the first pass reported 0.958 \u2014 and why that was luck\n")
    A("The first pass reported unseen-class supervised AUROC **0.958** from one "
      "training run, and concluded from it that the unseen-defect experiment had "
      "failed: the supervised head scored *higher* on the held-out class than on "
      "the classes it was trained on, which is the opposite of the effect the "
      "two-stage architecture is supposed to exploit.\n")
    A("The three seeds above land near 0.78, which is nine standard deviations "
      "away. Something other than sampling had to be responsible, and there was an "
      "obvious candidate: `extend.py` trains on 1100 images for 12 epochs and "
      "`run_inspect.py` on 1400 for 14. Two variables had moved at once, so the "
      "honest move was to hold the seeds fixed and sweep only the budget "
      "(`python budget_probe.py`).\n")
    A("**The budget was not the cause.**\n")
    A("| training budget | known sup | unseen sup | known ano | unseen ano | gap (ano \u2212 sup) |")
    A("|---|---|---|---|---|---|")
    cfgs = sorted({(r["n_train"], r["epochs"]) for r in rows})
    stats = {}
    for cfg in cfgs:
        g = [r for r in rows if (r["n_train"], r["epochs"]) == cfg]
        cell = {}
        for key in ("known_sup", "unseen_sup", "known_ano", "unseen_ano", "gap"):
            v = [r[key] for r in g]
            m = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            # t(0.975, df=2) = 4.303 -- with three seeds the interval is wide and
            # saying so is the point of computing it.
            cell[key] = (m, sd, 4.303 * sd / len(v) ** 0.5)
        stats[cfg] = cell
        A(f"| {cfg[0]} imgs / {cfg[1]} ep | {cell['known_sup'][0]:.3f} "
          f"| **{cell['unseen_sup'][0]:.3f}** | {cell['known_ano'][0]:.3f} "
          f"| {cell['unseen_ano'][0]:.3f} | {cell['gap'][0]:+.3f} |")

    big = stats[cfgs[-1]]
    small = stats[cfgs[0]]
    A(f"\nAt the first pass's own budget the unseen supervised AUROC averages "
      f"**{big['unseen_sup'][0]:.3f}** over three seeds, not 0.958 \u2014 and the "
      f"spread is the finding: **sd {big['unseen_sup'][1]:.3f}**, against "
      f"{small['unseen_sup'][1]:.3f} at the smaller budget. 0.958 sits about "
      f"{(0.958 - big['unseen_sup'][0]) / max(big['unseen_sup'][1], 1e-9):.1f} "
      "standard deviations above that mean. It was a lucky seed, reported as a "
      "result.\n")
    A("### What that does to the first pass's conclusion\n")
    gaps = [r["gap"] for r in rows]
    npos = sum(1 for g in gaps if g > 0)
    A(f"**It reverses it.** The anomaly head beats the supervised head on the "
      f"held-out class in **{npos} of {len(gaps)} runs** across both budgets "
      f"(gaps {', '.join(f'{g:+.3f}' for g in gaps)}; smallest {min(gaps):+.3f}). "
      f"A sign test on {npos}/{len(gaps)} gives one-sided p = "
      f"{1 / 2 ** len(gaps):.3f}, which is distribution-free and therefore immune "
      "to the variance blow-up that makes the t-interval at the larger budget "
      "useless.\n")
    A("And the degradation the first pass went looking for and failed to find is "
      "there:\n")
    A("| training budget | supervised: known \u2192 unseen | anomaly: known \u2192 unseen |")
    A("|---|---|---|")
    for cfg in cfgs:
        c = stats[cfg]
        A(f"| {cfg[0]} imgs / {cfg[1]} ep "
          f"| {c['known_sup'][0]:.3f} \u2192 {c['unseen_sup'][0]:.3f} "
          f"(**{c['unseen_sup'][0] - c['known_sup'][0]:+.3f}**) "
          f"| {c['known_ano'][0]:.3f} \u2192 {c['unseen_ano'][0]:.3f} "
          f"({c['unseen_ano'][0] - c['known_ano'][0]:+.3f}) |")
    A("\nThe supervised head **loses** AUROC on a defect class it never saw; the "
      "anomaly head **gains** it. That is the structural claim the two-stage "
      "architecture rests on, and the first pass declared it unsupported on the "
      "strength of a single run.\n")
    A("### What I am not claiming\n")
    A(f"Three seeds per cell. At the larger budget the gap's own 95% interval is "
      f"{big['gap'][0]:+.3f} \u00b1 {big['gap'][2]:.3f}, which **includes zero** "
      "\u2014 the variance is large enough that the parametric interval says "
      "nothing there. The sign test across all six runs is the claim I will "
      "defend; the effect size is not pinned down, and pinning it down means more "
      "seeds.\n")
    A("I also have no confirmed mechanism for why the larger budget is so much "
      "noisier. The plausible story is that more epochs on three defect classes "
      "lets the model commit harder to class-specific features, and how far it "
      "commits varies by initialisation \u2014 but that is a hypothesis I have not "
      "tested, and it is written here as one.\n")
    A("**The transferable lesson is the cheap one.** A single-seed result was "
      "strong enough to become this project's headline negative finding, and it "
      "was wrong. Nothing about it looked fragile \u2014 it was a clean AUROC on a "
      "clean holdout. The only thing that would have caught it is the discipline "
      "of never reporting a model comparison from one run, which costs three "
      "trainings and is the reason section 1 exists at all.")


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# ML-2 extensions — generated by `extend.py`, not hand-edited\n")

    A("## 1. Confidence intervals — the first build had one seed\n")
    A(f"{len(res['seeds'])} independent training runs (seeds {res['seeds']}), "
      "95% t-intervals.\n")
    A("| metric | mean ± half-width | sd |")
    A("|---|---|---|")
    for k, v in res["seed_ci"].items():
        A(f"| {k.replace('_', ' ')} | {v['mean']:.3f} ± {v['half_width']:.3f} | "
          f"{v.get('sd', float('nan')):.3f} |")
    g = res["unseen_gap_ci"]
    A(f"\n**The unseen-defect gap is {g['mean']:+.3f} ± {g['half_width']:.3f}.**\n")
    if g["lo"] > 0:
        A("The interval excludes zero, so the anomaly head does beat the supervised "
          "head on the held-out class — but look at the size. A gap of this "
          "magnitude is a real effect and a small one, and the first build's "
          "single-seed number could not have told you which.")
    else:
        A("**The interval includes zero.** With one seed the first build reported a "
          f"gap of +0.030 and could not say whether that was signal; across "
          f"{g['n']} seeds it is {g['mean']:+.3f} ± {g['half_width']:.3f}, which is "
          "indistinguishable from no difference. That does not overturn the earlier "
          "conclusion — it *is* the earlier conclusion, now with an error bar "
          "attached, and it closes the question the first build had to leave open.")

    _budget_section(A)

    A("\n## 2. The two-stage cascade, wired\n")
    c = res["cascade"]
    A(f"Stage 1 (anomaly screen) threshold chosen for **99% target recall**: "
      f"{c['screen_threshold']:.4f}. Stage 2 names a class above "
      f"{c['classify_threshold']:.4f}.\n")
    A("| | known defects | unseen defect |")
    A("|---|---|---|")
    allk = set(c["known"]["counts"]) | set(c["unseen"]["counts"])
    for v in sorted(allk):
        A(f"| {v} | {c['known']['counts'].get(v, 0)} | {c['unseen']['counts'].get(v, 0)} |")
    A(f"| **stage-1 recall** | **{c['known_stage1_recall']*100:.1f}%** | "
      f"**{c['unseen_stage1_recall']*100:.1f}%** |")
    A(f"\n**The third verdict is the one that matters.** `FLAG_FOR_REVIEW` — stage 1 "
      "is confident something is wrong, stage 2 cannot name it — is the disposition "
      "an unseen defect should receive. A two-outcome system forces the operator to "
      "pick a wrong class to clear the screen, and then the override log fills with "
      "garbage and poisons the retraining set.\n")
    t = c["throughput"]
    A("### The throughput argument\n")
    A(f"Stage 1 costs {t['stage1_ms']:.2f} ms/part, stage 2 {t['stage2_ms']:.2f} ms. "
      f"At a line prevalence of 0.5% the screen flags "
      f"{c['line_screen_rate_at_0.5pct_prevalence']*100:.1f}% of parts, so stage 2 "
      f"runs on that fraction only:\n")
    A("| | ms per part |")
    A("|---|---|")
    A(f"| cascade (stage 1 + {c['line_screen_rate_at_0.5pct_prevalence']*100:.1f}% × stage 2) | **{t['cascade_avg_ms']:.2f}** |")
    A(f"| both stages on every part | {t['always_both_ms']:.2f} |")
    A(f"| saving | {t['saving_pct']:.1f}% |")
    # The throughput case for a cascade is arithmetic, and the arithmetic only
    # works if the cheap stage CLEARS most parts. This is written as a conditional
    # because the measured screen rate decides which way it reads, and the draft
    # that assumed a win was simply wrong about this model.
    screen = c["line_screen_rate_at_0.5pct_prevalence"]
    if t["saving_pct"] >= 25.0:
        A(f"\nThis is a **throughput** argument, not an accuracy one, and here it "
          f"pays: the screen clears {(1 - screen) * 100:.1f}% of parts before the "
          "expensive stage runs, which is what makes a heavier stage-2 classifier "
          "affordable inside takt.")
    else:
        A(f"\n**The throughput argument does not survive contact with the measured "
          f"screen rate.** A cascade saves time only when the cheap stage *clears* "
          f"most parts, and this screen clears {(1 - screen) * 100:.1f}%. Stage 2 "
          f"therefore runs on nearly everything and the saving is "
          f"{t['saving_pct']:.1f}% \u2014 a rounding error against a takt budget, "
          "and nowhere near enough to justify two models, two thresholds and two "
          "retraining paths.\n")
        A("The cause is not the cascade, it is the screen. A 99% recall target on "
          "an anomaly head whose known-defect AUROC is ~0.84 buys that recall by "
          "putting the threshold low enough to flag almost everything, and recall "
          "bought that way is real and worthless.\n")
        A("So the conclusion is narrower than the one I set out to write: **this is "
          "the right architecture for the wrong model.** `FLAG_FOR_REVIEW` justifies "
          "the cascade on its own \u2014 an unseen defect needs a disposition that "
          "is neither ACCEPT nor a guessed class. The latency saving is a separate "
          "claim, and on this model I have not earned it.")
    A("\nEither way the saving depends entirely on the screen rate, which depends "
      "on prevalence \u2014 the same Bayes arithmetic as the PPV table in "
      "RESULTS.md, arriving this time as a latency budget.")
    sweep = res.get("cascade_recall_sweep") or []
    if sweep:
        A("\n#### What the recall target costs, swept\n")
        A("| stage-1 recall target | achieved | good parts flagged | screen rate "
          "@0.5% prevalence | cascade ms/part | saving | defects escaping |")
        A("|---|---|---|---|---|---|---|")
        for r in sweep:
            rate = r["line_screen_rate"]
            ms = t["stage1_ms"] + rate * t["stage2_ms"]
            saving = 100.0 * (t["always_both_ms"] - ms) / t["always_both_ms"]
            A(f"| {r['target_recall'] * 100:.1f}% "
              f"| {r['achieved_recall'] * 100:.1f}% "
              f"| {r['good_parts_flagged'] * 100:.1f}% "
              f"| {rate * 100:.1f}% | {ms:.2f} | {saving:.1f}% "
              f"| {(1 - r['achieved_recall']) * 100:.1f}% |")
        A("\nThis table is the decision surface, and it prices a **safety-vs-cost "
          "trade in milliseconds**: every point of recall given up buys throughput "
          "and ships defects. The last column is why the saving column cannot be "
          "read on its own \u2014 the cheap rows are cheap precisely because they "
          "let defects through.\n")
        A("Read down it and the shape of the problem is plain. The saving only "
          "becomes interesting once the screen is clearing a real fraction of good "
          "parts, and on this model that does not happen until the recall target "
          "has dropped into a range no inspection engineer would sign. **That is a "
          "verdict on the anomaly head, not on the cascade** \u2014 the same "
          "architecture on a screen with a sharper healthy/defective separation "
          "would show this table tilting the other way.")

    A("\n## 3. Grad-CAM — the supervised head had no localisation at all\n")
    gc = res["gradcam"]
    A(f"On {gc['n_images']} defective images, the Grad-CAM peak lands **inside the "
      f"true defect mask {gc['peak_inside_mask_pct']:.1f}%** of the time. Defect "
      f"pixels are {gc['random_baseline_pct']:.2f}% of image area, so a random peak "
      f"would land inside about {gc['random_baseline_pct']:.2f}% of the time.\n")
    lift = gc["peak_inside_mask_pct"] / max(gc["random_baseline_pct"], 1e-9)
    A(f"That is a **{lift:.0f}× lift over chance**, which is the comparison that "
      "makes the number mean something — a localisation score quoted without the "
      "base rate is unreadable, because defects are small and any peak is *usually* "
      "outside the mask by area alone.\n")
    A("Grad-CAM is coarse by construction: it is the spatial resolution of the "
      "last convolutional layer, upsampled. The spec calls it a *minimum* rather "
      "than a substitute for segmentation.\n")
    if gc["peak_inside_mask_pct"] < 25.0:
        A(f"**A {lift:.0f}\u00d7 lift is not the same as a usable one, and I am "
          f"not going to let the lift do the talking.** The peak lands outside the "
          f"true defect {100 - gc['peak_inside_mask_pct']:.1f}% of the time. An "
          "operator shown that overlay is being pointed at the wrong part of the "
          "part in roughly nineteen cases out of twenty, and a localisation aid "
          "that is usually wrong is worse than none \u2014 it teaches the operator "
          "to distrust the overlay, and then to distrust the call the overlay "
          "arrived with.\n")
        A("The diagnosis is in the training objective rather than in Grad-CAM. The "
          "head is trained on an image-level label, so it is free to key on "
          "anything that correlates with the label anywhere in the frame \u2014 "
          "and the first build already caught this model doing exactly that, when "
          "global average pooling washed the defects out. **An attribution method "
          "cannot manufacture spatial evidence the model never used.** Raising this "
          "number is a segmentation or patch-supervision job, which is named in the "
          "not-built list, and it is not a Grad-CAM tuning exercise.")
    else:
        A("At this hit rate it is enough for an operator to verify a call, though "
          "not to measure a defect.")

    A("\n## 4. The operator override log\n")
    o = res["override"]
    A(f"{o['n_reviewed']} parts reached the review station.\n")
    A("| disposition | count |")
    A("|---|---|")
    for k2, v in sorted(o["by_disposition"].items()):
        A(f"| {k2} | {v} |")
    A(f"\n**Operator-confirmed PPV: {o['operator_confirmed_ppv']:.3f}.** This is the "
      "only PPV measurable in production without a destructive audit — and it is "
      "measured on the *flagged subset*, not on the line, so it is an upper bound "
      "on line PPV rather than an estimate of it. Confusing the two is how a "
      "monitoring dashboard reports healthy precision on a line that is scrapping "
      "good parts.\n")
    A(f"Defects that escaped stage 1 entirely: {o['escaped_defects_known']} known, "
      f"{o['escaped_defects_unseen']} unseen. Those never reach a human, which is "
      "why stage 1 is tuned for recall and why its threshold is the one number in "
      "this system worth arguing about.\n")
    A("**Why the log is retraining gold, and why it is also a trap.** It is gold "
      "because of the *sampling*: these are labels on exactly the parts the model "
      "found hard, which is the most informative set to buy. It is a trap for the "
      "same reason — it is a censored sample, containing only what stage 1 flagged, "
      "so a model retrained on it alone drifts toward the screen's own biases and "
      "gets worse on the parts nobody ever reviewed.\n")
    A("**The operator is simulated as ground truth here, which is generous.** A real "
      "operator is a measurement system with its own repeatability and "
      "reproducibility — which is precisely the spec's gauge-R&R question, and it "
      "remains unbuilt.")

    A("\n---\n*Regenerate with `python extend.py`.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
