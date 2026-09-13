"""ML-2 end-to-end: two detectors, real prevalence, cost matrix, takt time.

    python run_inspect.py
    python run_inspect.py --quick
    python run_inspect.py --report-only
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

import economics as E  # noqa: E402
import models as M  # noqa: E402
import synth  # noqa: E402

OUT = ROOT / "out"

# Row label for the held-out-class test. Derived from the config in main() rather
# than hardcoded -- the first version had the string "crack" baked into the report
# and kept printing it after the held-out class changed to `inclusion`, which is a
# small bug with an outsized consequence: a results table that names the wrong
# experiment.
UNSEEN_KEY = "UNSEEN defect (inclusion)"
TAKT_PARTS_PER_MIN = 60.0
SCRAP_COST = 4.0            # $ per good part scrapped or re-inspected


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    P, N = float((labels == 1).sum()), float((labels == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - P * (P + 1) / 2) / (P * N))


def main() -> None:
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        prev = json.loads((OUT / "results.json").read_text())
        (ROOT / "docs").mkdir(exist_ok=True)
        (ROOT / "docs" / "RESULTS.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/RESULTS.md")
        return

    t0 = time.perf_counter()
    rng = np.random.default_rng(20260819)
    n_train = 500 if quick else 1400
    n_test = 400 if quick else 900
    res: dict = {}

    print("1/6 generating images ...", flush=True)
    # THE HELD-OUT CLASS IS THE BRIGHT ONE, and choosing it took two attempts.
    #
    # First attempt: train on (pore, inclusion, shrinkage), hold out `crack`. Both
    # detectors scored AUROC 1.000 on the held-out cracks -- the supervised model
    # was exactly as good on the class it had never seen as the anomaly head was.
    # That is not the supervised model being unexpectedly clever; it is the test
    # being too easy. A crack is a DARK, small, local feature, and two of the
    # three trained classes (pore, shrinkage) are also dark local features. The
    # model had learned "dark local anomaly" and cracks fall inside it.
    #
    # The lesson generalises past this project: "unseen defect" is a property of
    # the FEATURE SPACE, not of the label. Holding out a class the training set
    # already spans tests nothing, and it is the easy mistake to make because the
    # label really was absent from training.
    #
    # So the held-out class is now `inclusion`, the only BRIGHT defect, against a
    # training set of three dark ones. That is novelty along a dimension the
    # training data does not cover -- a detector that learned "defects are darker
    # than their surroundings" has no reason to fire on it.
    #
    # The first attempt's numbers are kept in out/results_crack_holdout.json.
    known = ("pore", "crack", "shrinkage")
    unseen = "inclusion"
    global UNSEEN_KEY
    UNSEEN_KEY = f"UNSEEN defect ({unseen})"

    train = synth.make(n_train, 0.35, rng, variants=("A", "B", "C"), defect_types=known)
    normal_only = [s for s in synth.make(n_train, 0.0, rng, variants=("A", "B", "C"))]
    test_known = synth.make(n_test, 0.35, rng, variants=("A", "B", "C"), defect_types=known)
    test_unseen = synth.make(n_test // 2, 0.5, rng, variants=("A", "B", "C"),
                             defect_types=(unseen,))
    test_variantD = synth.make(n_test // 2, 0.35, rng, variants=("D",), defect_types=known)
    test_lighting = synth.make(n_test // 2, 0.35, rng, variants=("A", "B", "C"),
                               defect_types=known, lighting=0.22, jitter_px=2.0)

    xtr, ytr, _ = synth.to_arrays(train)
    xnorm, _, _ = synth.to_arrays(normal_only)
    res["data"] = {
        "train": len(train), "normal_only_for_anomaly": len(normal_only),
        "test_known": len(test_known), "test_unseen": len(test_unseen),
        "test_variant_D": len(test_variantD), "test_lighting_shift": len(test_lighting),
        "known_classes": list(known), "unseen_class": unseen,
        "train_defect_rate": float(ytr.mean()),
    }
    print(f"    train {len(train)} ({ytr.mean()*100:.0f}% defective), "
          f"normal-only {len(normal_only)}, test {len(test_known)}", flush=True)

    print("2/6 training the supervised CNN ...", flush=True)
    model, secs = M.train_supervised(xtr, ytr, epochs=6 if quick else 14,
                                     verbose=not quick)
    print(f"    {secs:.0f}s", flush=True)

    print("3/6 fitting the anomaly head on NORMAL IMAGES ONLY ...", flush=True)
    anom = M.PatchAnomaly().fit(model, xnorm)

    def evaluate(name: str, samples) -> dict:
        x, y, m = synth.to_arrays(samples)
        s_sup = M.supervised_scores(model, x)
        # ONE score_maps call, reused. The first version computed the maps twice
        # per test set -- once inside image_scores and once for the pixel AUROC --
        # which doubled the most expensive step in the pipeline for nothing.
        raw_maps = anom.score_maps(model, x)
        s_ano = raw_maps.max(axis=(1, 2))
        maps = M.upsample_map(raw_maps, synth.SIZE)
        pix = float("nan")
        if m.any():
            flat_s = maps.reshape(-1)
            flat_y = m.reshape(-1).astype(int)
            # Subsample pixels: 900 images x 16k pixels is 15M points and a rank
            # statistic does not need all of them.
            #
            # WITH replacement, deliberately. `rng.choice(n, size, replace=False)`
            # builds a full permutation of n internally, so on 15M elements it is
            # pathologically slow -- it stalled this stage for twenty minutes
            # before I looked. Sampling 400k indices with replacement from 15M
            # collides on well under 3% of draws and does not move an AUROC.
            n_pix = len(flat_s)
            idx = np.random.default_rng(0).integers(0, n_pix,
                                                    size=min(400_000, n_pix))
            pix = auroc(flat_s[idx], flat_y[idx])
        return {
            "n": len(samples), "defect_rate": float(y.mean()),
            "supervised_auroc": auroc(s_sup, y),
            "anomaly_auroc": auroc(s_ano, y),
            "anomaly_pixel_auroc": pix,
            "_s_sup": s_sup, "_s_ano": s_ano, "_y": y,
        }

    print("4/6 evaluating: known / unseen / new variant / lighting shift ...", flush=True)
    ev = {
        "known defects": evaluate("known", test_known),
        UNSEEN_KEY: evaluate("unseen", test_unseen),
        "new product variant D": evaluate("variantD", test_variantD),
        "lighting + jitter shift": evaluate("lighting", test_lighting),
    }
    res["detection"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in ev.items()}
    for k, v in ev.items():
        print(f"    {k:<24} supervised AUROC {v['supervised_auroc']:.3f}  "
              f"anomaly AUROC {v['anomaly_auroc']:.3f}", flush=True)

    print("5/6 line economics ...", flush=True)
    k = ev["known defects"]
    res["prevalence"] = {
        "supervised": E.prevalence_table(k["_s_sup"], k["_y"]),
        "anomaly": E.prevalence_table(k["_s_ano"], k["_y"]),
    }
    res["cost_sweep"] = E.cost_ratio_sweep(k["_s_sup"], k["_y"], prevalence=0.005,
                                           c_false_reject=SCRAP_COST)
    res["worked_example"] = worked_example()

    print("6/6 takt time ...", flush=True)
    x_one = synth.to_arrays(test_known[:1])[0]
    lat = measure_latency(model, anom, x_one)
    res["takt"] = E.takt_analysis(lat["total_ms"], TAKT_PARTS_PER_MIN,
                                  stages=lat["stages"])
    res["wall_seconds"] = time.perf_counter() - t0

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/RESULTS.md and out/results.json ({res['wall_seconds']:.0f}s)")


def worked_example() -> dict:
    """The interview question, done as arithmetic: 98% recall, 3% false reject,
    200k parts/day, $4 scrap cost. Is this deployable?"""
    volume, fr_rate, cost = 200_000, 0.03, 4.0
    prevalence, recall = 0.005, 0.98
    good = volume * (1 - prevalence)
    scrapped = good * fr_rate
    return {
        "volume_per_day": volume, "false_reject_rate": fr_rate,
        "scrap_cost_each": cost, "prevalence": prevalence, "recall": recall,
        "good_parts_scrapped_per_day": scrapped,
        "daily_cost_of_false_rejects": scrapped * cost,
        "annual_cost": scrapped * cost * 250,
        "defects_caught_per_day": volume * prevalence * recall,
        "defects_escaping_per_day": volume * prevalence * (1 - recall),
        "ppv": E.ppv(prevalence, recall, 1 - fr_rate),
    }


def measure_latency(model, anom, x_one: np.ndarray, n: int = 60) -> dict:
    torch.set_num_threads(1)
    for _ in range(5):
        M.supervised_scores(model, x_one)
        anom.image_scores(model, x_one)
    t = []
    for _ in range(n):
        a = time.perf_counter()
        M.supervised_scores(model, x_one)
        b = time.perf_counter()
        anom.image_scores(model, x_one)
        c = time.perf_counter()
        t.append((b - a, c - b))
    sup = np.array([x[0] for x in t]) * 1000
    ano = np.array([x[1] for x in t]) * 1000
    return {
        "total_ms": float(np.percentile(sup + ano, 99)),
        "stages": {
            "supervised p50": float(np.percentile(sup, 50)),
            "supervised p99": float(np.percentile(sup, 99)),
            "anomaly p50": float(np.percentile(ano, 50)),
            "anomaly p99": float(np.percentile(ano, 99)),
            "combined p50": float(np.percentile(sup + ano, 50)),
            "combined p99": float(np.percentile(sup + ano, 99)),
        },
    }


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    d = res["data"]
    A("# ML-2 results, generated by `run_inspect.py`, not hand-edited\n")
    A("> **Data provenance.** The images are *synthesised* by `src/synth.py`: "
      "filtered-noise casting surfaces with four defect classes and "
      "pixel-accurate ground-truth masks. They are **not MVTec AD** and not the "
      "Kaggle casting set, and no AUROC here is comparable to a published number "
      "on either. What the generator buys is a pixel-accurate mask for every "
      "defect and control over prevalence, defect class, product variant and "
      "lighting, which is what the economics and transfer analyses need.\n")
    A(f"Supervised model trained on **{', '.join(d['known_classes'])}**. "
      f"**`{d['unseen_class']}` is held out entirely** as the unseen defect. The "
      f"anomaly head is fitted on {d['normal_only_for_anomaly']} normal images "
      "only and never sees a defect label.\n")

    A("## 1. The two detectors, and where each one breaks\n")
    A("| test set | n | supervised AUROC | anomaly AUROC | anomaly pixel AUROC |")
    A("|---|---|---|---|---|")
    # Find the held-out row by PREFIX and label it from the config, never from the
    # stored dict key. The key is written by whichever run produced the file, and a
    # results.json from an earlier configuration will carry a label naming the
    # wrong class -- which is exactly what happened here, and a report that
    # confidently mislabels its own experiment is worse than one that crashes.
    unseen_key = next(k for k in res["detection"] if k.startswith("UNSEEN defect"))
    unseen_label = f"UNSEEN defect ({res['data']['unseen_class']})"
    for name, v in res["detection"].items():
        label = unseen_label if name == unseen_key else name
        A(f"| {label} | {v['n']} | {v['supervised_auroc']:.3f} | "
          f"{v['anomaly_auroc']:.3f} | {v['anomaly_pixel_auroc']:.3f} |")
    kn = res["detection"]["known defects"]
    un = res["detection"][unseen_key]
    d = res["data"]
    gap = un["anomaly_auroc"] - un["supervised_auroc"]
    A(f"\n**The unseen-defect row.** Trained on {', '.join(d['known_classes'])}, the "
      f"supervised model scores {kn['supervised_auroc']:.3f} on those same classes. "
      f"On held-out `{d['unseen_class']}` it scores {un['supervised_auroc']:.3f}, "
      f"against {un['anomaly_auroc']:.3f} for the anomaly head: a gap of "
      f"{gap:+.3f}.\n")
    if gap > 0.05:
        A("That gap is structural rather than a tuning problem. The supervised "
          "model learned a boundary between *good* and *these three defects*; a "
          "fourth defect lands wherever the features happen to put it, and there "
          "is no reason for that to be the defective side. The anomaly head has no "
          "concept of a defect class at all, it only knows what normal looks like, "
          "so a novel defect is exactly as detectable to it as a familiar one.\n")
        A("Which is the argument for the **two-stage line architecture**: an "
          "anomaly screen catches the novel thing, and the supervised head tells "
          "the quality engineer which bin to put it in.")
    else:
        A("**This test does not demonstrate what it was built to demonstrate, and "
          "that is the honest headline.** The supervised model handles the "
          f"held-out class about as well as the anomaly head does ({gap:+.3f}), so "
          "there is no measured blindness here to point at. What the anomaly head "
          "does win on is *pixel*-level localisation "
          f"({un['anomaly_pixel_auroc']:.3f}), which is a different claim.\n")
        A("**Two attempts, both of which failed to separate the detectors.** The "
          "first held out `crack` from a training set of (pore, inclusion, "
          "shrinkage): both detectors scored AUROC 1.000. The diagnosis was that a "
          "crack is a *dark* local feature and two trained classes are also dark "
          "local features, so the model had learned \"dark local anomaly\" and "
          "cracks fell inside it. The second, this one, held out `inclusion`, "
          "the only *bright* defect, against three dark ones. That is novelty "
          "along a polarity dimension the training set genuinely does not cover, "
          "and the supervised model still scores "
          f"{un['supervised_auroc']:.3f}.\n")
        A("The most likely reason, and it is a limitation of my **synthetic data** "
          "rather than a discovery about CNNs: `src/synth.py` renders every defect "
          "as a local deviation from a smooth textured background, so a small "
          "convolutional network with max-pooling can learn \"local deviation\" and "
          "generalise across polarity for free. Real defect classes differ in "
          "texture, scale, edge profile and context in ways this generator does "
          "not reproduce, and the published result that supervised models "
          "generalise poorly to unseen defect types is measured on real imagery "
          "(MVTec AD), not on anything like this.\n")
        A("So the structural argument for the two-stage architecture still holds, "
          "a supervised model cannot be *relied* on for a class outside its "
          "training distribution, but **this project does not provide evidence "
          "for it**, and the two-stage recommendation below should be read as "
          "reasoning, not as a measured result. Demonstrating it properly needs "
          "MVTec, which is in the not-built list.\n")
        A("The first attempt's numbers are kept in "
          "`out/results_crack_holdout.json`.")

    A("\n### What the quality engineer sees when a new defect appears\n")
    A("The anomaly head flags the part and produces a **heat map**, because the "
      "score is a per-patch Mahalanobis distance; localisation comes free. The "
      "supervised head returns low confidence on every known class. The correct "
      "disposition is therefore *flag for human review as an unclassified "
      "anomaly*, not *reject as class X*, and the review station has to have that "
      "as an option or the operator will pick a wrong class to clear the screen.")

    A("\n## 2. PPV at realistic prevalence: the arithmetic that changes the decision\n")
    A("Sensitivity and specificity are properties of the model and are "
      "prevalence-invariant. **PPV is not.** So Se and Sp are measured on a "
      "convenient mix and then combined with the real base rate:\n")
    A("```\n            p * Se\nPPV = ---------------------------\n"
      "      p * Se + (1 - p) * (1 - Sp)\n```\n")
    A("Supervised model at a fixed 3% false-reject rate:\n")
    A("| prevalence | sensitivity | specificity | **PPV** | good parts rejected /1000 | defects escaping /1000 |")
    A("|---|---|---|---|---|---|")
    for r in res["prevalence"]["supervised"]:
        A(f"| {r['prevalence']*100:.1f}% | {r['sensitivity']:.3f} | "
          f"{r['specificity']:.3f} | **{r['ppv']:.3f}** | "
          f"{r['good_parts_rejected_per_1000']:.1f} | "
          f"{r['defects_escaping_per_1000']:.2f} |")
    lo = res["prevalence"]["supervised"][0]
    hi = res["prevalence"]["supervised"][-1]
    A(f"\n**Same model, same threshold, same recall.** At the balanced "
      f"{hi['prevalence']*100:.0f}% prevalence of a convenient test set the PPV is "
      f"{hi['ppv']:.3f}. At the {lo['prevalence']*100:.1f}% prevalence of a real "
      f"line it is **{lo['ppv']:.3f}**: roughly "
      f"{(1-lo['ppv'])*100:.0f}% of everything the system rejects is a good part. "
      "Nothing about the model changed; the base rate did. This is the single "
      "most-missed reality in inspection ML and it is arithmetic, not opinion.")

    w = res["worked_example"]
    A("\n### The worked example\n")
    A(f"{w['volume_per_day']:,} parts/day, {w['recall']*100:.0f}% recall, "
      f"{w['false_reject_rate']*100:.0f}% false-reject rate, "
      f"${w['scrap_cost_each']:.0f} scrap cost, "
      f"{w['prevalence']*100:.1f}% defect rate.\n")
    A("| | |")
    A("|---|---|")
    A(f"| good parts scrapped per day | {w['good_parts_scrapped_per_day']:,.0f} |")
    A(f"| **cost of false rejects per day** | **${w['daily_cost_of_false_rejects']:,.0f}** |")
    A(f"| cost per year (250 days) | ${w['annual_cost']:,.0f} |")
    A(f"| defects actually caught per day | {w['defects_caught_per_day']:,.0f} |")
    A(f"| defects escaping per day | {w['defects_escaping_per_day']:,.0f} |")
    A(f"| PPV | {w['ppv']:.3f} |")
    A(f"\n**${w['daily_cost_of_false_rejects']:,.0f} a day of scrapped good parts "
      f"to catch {w['defects_caught_per_day']:,.0f} defects.** Whether that is "
      "deployable depends entirely on what an escape costs, which is the next "
      "table, but a 98%-recall model is not automatically a good model, and the "
      "false-reject line is where high-volume inspection projects die.")

    A("\n## 3. The operating point comes from the cost matrix\n")
    A(f"Prevalence 0.5%, false reject ${SCRAP_COST:.0f}, "
      f"{res['cost_sweep'][0].get('_volume', 200_000):,} parts. The escape cost "
      "spans three orders of magnitude between a cosmetic blemish and a "
      "safety-critical casting, so it is swept rather than assumed.\n")
    A("| escape : false-reject | escape cost | chosen threshold | sensitivity | false-reject rate | PPV | expected cost |")
    A("|---|---|---|---|---|---|---|")
    for r in res["cost_sweep"]:
        A(f"| {r['cost_ratio']:.0f}:1 | ${r['c_escape']:,.0f} | "
          f"{r['threshold']:.4f} | {r['sensitivity']:.3f} | "
          f"{r['false_reject_rate']*100:.2f}% | {r['ppv']:.3f} | "
          f"${r['expected_cost']:,.0f} |")
    first, last = res["cost_sweep"][0], res["cost_sweep"][-1]
    A(f"\nThe operating point moves from a false-reject rate of "
      f"{first['false_reject_rate']*100:.2f}% at {first['cost_ratio']:.0f}:1 to "
      f"{last['false_reject_rate']*100:.2f}% at {last['cost_ratio']:.0f}:1. That is "
      "the same model being asked two different business questions. A single "
      "threshold chosen without this table is a business decision made by whoever "
      "wrote `> 0.5`.")

    t = res["takt"]
    A("\n## 4. Takt time\n")
    A(f"The line runs at **{t['parts_per_minute']:.0f} parts/minute**, so the takt "
      f"time is **{t['takt_ms']:.0f} ms**. The full two-stage pipeline runs in "
      f"**{t['inference_ms']:.1f} ms** (p99) on one CPU thread.\n")
    A("| stage | ms |")
    A("|---|---|")
    for k2, v2 in t["stages"].items():
        A(f"| {k2} | {v2:.2f} |")
    A(f"\nMargin: **{t['margin_ms']:.0f} ms** ({t['utilisation_pct']:.1f}% of takt "
      f"consumed). Maximum line rate this pipeline supports: "
      f"{t['max_parts_per_minute_supported']:,.0f} parts/minute.\n")
    A("The margin is not slack to be spent. It has to absorb image acquisition, "
      "transfer, the p99 tail rather than the mean, and the reject-mechanism "
      "actuation that must complete before the part reaches the diverter. Quoting "
      "a model's latency without the takt time is quoting half a sentence.")

    A("\n## 5. Robustness: the factory floor\n")
    lg = res["detection"]["lighting + jitter shift"]
    vd = res["detection"]["new product variant D"]
    A(f"| scenario | supervised AUROC | anomaly AUROC | vs known-defect baseline |")
    A("|---|---|---|---|")
    A(f"| baseline (known defects) | {kn['supervised_auroc']:.3f} | "
      f"{kn['anomaly_auroc']:.3f} | N/A |")
    A(f"| lighting shift + camera jitter | {lg['supervised_auroc']:.3f} | "
      f"{lg['anomaly_auroc']:.3f} | "
      f"{lg['supervised_auroc']-kn['supervised_auroc']:+.3f} / "
      f"{lg['anomaly_auroc']-kn['anomaly_auroc']:+.3f} |")
    A(f"| new product variant D | {vd['supervised_auroc']:.3f} | "
      f"{vd['anomaly_auroc']:.3f} | "
      f"{vd['supervised_auroc']-kn['supervised_auroc']:+.3f} / "
      f"{vd['anomaly_auroc']-kn['anomaly_auroc']:+.3f} |")
    A("\n**The new-variant row is the re-validation trigger.** Variant D has a "
      "rougher, darker surface than A/B/C, and the anomaly head is fitted on what "
      "normal looks like, for A/B/C. Variant D is *legitimately* unlike its "
      "training normal, so the anomaly head degrades, and the correct response is "
      "not to retune a threshold but to **re-qualify the model against a golden "
      "sample set for that variant**, exactly as a quality team re-qualifies a "
      "gauge after a fixture change.\n")
    A("How the drift would be caught in production without labels: **reject-rate "
      "SPC**. The reject rate is a p-chart statistic, it needs no ground truth, "
      "and a sustained shift in it is a signal whatever the cause: a lighting "
      "change, a new variant, or a genuine process problem. That connects the ML "
      "monitoring to the quality system the plant already runs, and DATA-2 in this "
      "portfolio is the chart engine that would do it. **The two are not wired "
      "together.**")

    A("\n---\n*All images are generated. Every number above is a property of "
      "`src/synth.py`, not of a production line.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
