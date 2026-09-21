"""ML-2 pass 5: the review station writes, and the loop eats what it writes.

    python run_pass5.py            # ~4 min
    python run_pass5.py --quick

Two items from the not-built list, and they turn out to be one item:

  5  the review station renders and does not write
  6  the simulated operator is treated as ground truth

Closing 5 is a store and a POST handler. But the moment the log is real, the
question "what is in it" stops being rhetorical, and item 6 is the answer:
everything in it was written by a human, and the retraining loop in
complete.py never ate a human label in its life. It ate `pool_y` -- the truth
array -- on the parts the screen happened to flag. The censoring was modelled
honestly and the LABEL NOISE was not modelled at all.

So this run does the thing the review station exists to make possible: it
retrains on what operators actually said, with operators who are wrong some of
the time, and asks at what error rate the loop stops being worth running.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import inspection_ops as OPS      # noqa: E402
import models as M                # noqa: E402
import review_service as RS       # noqa: E402
import synth                      # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"
QUICK = "--quick" in sys.argv
EPOCHS = 4 if QUICK else 10
KNOWN = ("pore", "crack", "shrinkage")

# Operator error rates to sweep. 0.0 is complete.py's assumption; the rest are
# the range attribute-agreement studies on visual inspection usually land in.
ERROR_RATES = (0.0, 0.05, 0.10, 0.20, 0.30)

# THREE SEEDS, and the spread is reported. The first version of this run used
# one seed at --quick settings and produced a table where a 5%-error operator
# beat a perfect one by 0.21 AUROC. That is not a finding, it is noise wearing
# a finding's clothes, and the only reason it was visible as noise is that the
# ordering was absurd. A sweep whose steps are smaller than its own seed spread
# cannot answer the question it is asking.
SEEDS = (11, 23, 37)
GOLDEN_N = 120


def auroc(scores, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


def _data(seed: int = 11):
    rng = np.random.default_rng(seed)
    n_tr = 400 if QUICK else 900
    n_te = 250 if QUICK else 600
    d = {}
    for k, (n, p, dt) in (("train", (n_tr, 0.35, KNOWN)),
                          ("normals", (n_tr, 0.0, KNOWN)),
                          ("test", (n_te, 0.35, KNOWN))):
        x, y, _ = synth.to_arrays(
            synth.make(n, p, rng, variants=("A", "B", "C"), defect_types=dt))
        d[k] = {"x": x, "y": np.asarray(y).astype(int)}
    return d


# ---------------------------------------------------------------------------
# the simulated operator, stated plainly
# ---------------------------------------------------------------------------

def simulate_operator(truth: np.ndarray, verdicts, *, error: float,
                      abstain: float, bias: float, rng) -> list[str]:
    """One operator's dispositions on a queue of flagged parts.

    Error is symmetric on purpose except for `bias`, which tilts the mistakes.
    A real inspector is not symmetric -- passing a bad part is cheap in the
    moment and scrapping a good one is not -- and `bias` is how much of the
    error goes the pass-it direction. Guessed, not measured; that is the point
    of sweeping it rather than picking one.

    Abstention is separate from error. "I cannot tell" is not a wrong answer,
    it is a refused one, and a store that folds the two together loses the
    distinction the UNCLASSIFIED button exists to record.
    """
    # bias is the share of the error budget spent MISSING defects rather than
    # scrapping good parts, so the two rates differ and average to `error`.
    p_miss = min(1.0, 2.0 * error * bias)
    p_false = min(1.0, 2.0 * error * (1.0 - bias))
    out = []
    for t, v in zip(truth, verdicts):
        if v == "ACCEPT":
            out.append("UNCLASSIFIED")     # nothing else is coherent
            continue
        if rng.random() < abstain:
            out.append("UNCLASSIFIED")
            continue
        said = bool(t)
        if rng.random() < (p_miss if said else p_false):
            said = not said
        out.append("CONFIRMED_DEFECT" if said else "FALSE_REJECT")
    return out


def _verdicts(scores: np.ndarray, thr: float) -> list[str]:
    """Three-way, as the cascade does it: a band above the screen threshold is
    flagged rather than classified."""
    return ["REJECT_CLASSIFIED" if s > 1.3 * thr else
            "FLAG_FOR_REVIEW" if s > thr else "ACCEPT" for s in scores]


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------

def run_one(seed: int) -> dict:
    D = _data(seed)
    print("  base model ...", flush=True)
    model, _ = M.train_supervised(D["train"]["x"], D["train"]["y"],
                                  epochs=EPOCHS, verbose=False)
    anom = M.PatchAnomaly().fit(model, D["normals"]["x"])

    pool_x, pool_y = D["test"]["x"], D["test"]["y"]
    s_ano = anom.score_maps(model, pool_x).max(axis=(1, 2))
    thr = float(np.quantile(s_ano, 0.55))
    verdicts = _verdicts(s_ano, thr)
    flagged = np.array([v != "ACCEPT" for v in verdicts])

    gold = OPS.golden_set(D["test"]["x"], D["test"]["y"], n=GOLDEN_N, seed=7)
    hold = np.setdiff1d(np.arange(len(pool_x)), gold)
    gx, gy = pool_x[gold], pool_y[gold]

    def metrics(mdl):
        s = M.supervised_scores(mdl, gx)
        t = float(np.quantile(s, 0.5))
        pred = s > t
        tp = int((pred & (gy == 1)).sum()); fp = int((pred & (gy == 0)).sum())
        fn = int((~pred & (gy == 1)).sum())
        return {"recall": tp / max(tp + fn, 1), "ppv": tp / max(tp + fp, 1),
                "auroc": auroc(s, gy)}

    base = metrics(model)
    print(f"  baseline auroc {base['auroc']:.3f}", flush=True)

    # Only parts that are flagged AND not in the golden set may be reviewed:
    # retraining on the set you are about to score on is the oldest way to
    # produce a result that means nothing.
    review_ix = np.array([i for i in hold if flagged[i]])

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ml2rev-"))
    arms, stores = {}, {}
    for err in ERROR_RATES:
        rng = np.random.default_rng(1000 + int(err * 100))
        store = RS.ReviewStore(tmp / f"e{int(err*100)}.db")
        store.queue([{"part_id": f"P{i:05d}", "verdict": verdicts[i],
                      "anomaly_score": float(s_ano[i])} for i in review_ix],
                    "fp-v1", thr)
        truth = pool_y[review_ix]
        vs = [verdicts[i] for i in review_ix]
        for op in ("OP-A", "OP-B"):
            for pid, d in zip([f"P{i:05d}" for i in review_ix],
                              simulate_operator(truth, vs, error=err,
                                                abstain=0.08, bias=0.65,
                                                rng=rng)):
                store.dispose(pid, op, d)
        stores[err] = store

        for arm, rows in (("one operator", None),
                          ("two, consensus only",
                           store.labels_for_retraining(require_consensus=True)),
                          ("two, weighted majority",
                           store.labels_for_retraining(False))):
            if rows is None:
                # One operator: OP-A alone, which is what a station with no
                # second reviewer produces. No consensus is available and no
                # agreement is measurable -- that is the baseline the second
                # operator has to beat.
                rows = []
                for pid in [f"P{i:05d}" for i in review_ix]:
                    ds = [d for d in store.dispositions(pid)
                          if d["operator"] == "OP-A"]
                    lab = RS.LABEL_OF[ds[0]["disposition"]] if ds else None
                    if lab is not None:
                        rows.append({"part_id": pid, "label": lab, "weight": 1.0})
            if not rows:
                continue
            ix = np.array([int(r["part_id"][1:]) for r in rows])
            ry = np.array([r["label"] for r in rows])
            rw = np.array([r.get("weight", 1.0) for r in rows])
            xx = np.concatenate([D["train"]["x"], pool_x[ix]])
            yy = np.concatenate([D["train"]["y"], ry])
            ww = np.concatenate([np.ones(len(D["train"]["x"])), rw])
            mdl, _ = M.train_supervised(xx, yy, epochs=EPOCHS, verbose=False,
                                        sample_weight=ww / ww.mean())
            m = metrics(mdl)
            m["n_labels"] = int(len(rows))
            m["label_error"] = float(np.mean(ry != pool_y[ix]))
            m["gate"] = OPS.requalification_gate(base, m)
            arms.setdefault(arm, {})[err] = m
            print(f"  err={err:.2f} {arm:<24} auroc {m['auroc']:.3f} "
                  f"(labels {len(rows)}, {m['label_error']*100:.0f}% wrong)",
                  flush=True)

    agree = {e: stores[e].agreement() for e in ERROR_RATES}
    return {"baseline": base, "arms": arms, "agreement": agree,
            "n_pool": int(len(pool_x)), "n_flagged": int(flagged.sum()),
            "n_reviewable": int(len(review_ix)), "threshold": thr,
            "error_rates": list(ERROR_RATES), "seed": seed}


def _mean_sd(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def run() -> dict:
    """Every seed, then pooled with the spread kept."""
    runs = []
    for i, sd in enumerate(SEEDS, 1):
        print(f"  seed {sd} ({i}/{len(SEEDS)}) ...", flush=True)
        runs.append(run_one(sd))

    arms = sorted({a for r in runs for a in r["arms"]})
    pooled = {}
    for a in arms:
        pooled[a] = {}
        for e in ERROR_RATES:
            ms = [r["arms"].get(a, {}).get(e) for r in runs]
            ms = [m for m in ms if m]
            if not ms:
                continue
            cell = {"n_seeds": len(ms)}
            for k in ("auroc", "recall", "ppv", "label_error"):
                mu, sd = _mean_sd([m.get(k) for m in ms])
                cell[k], cell[k + "_sd"] = mu, sd
            cell["n_labels"] = int(np.mean([m["n_labels"] for m in ms]))
            cell["n_pass_gate"] = int(sum(m["gate"]["pass"] for m in ms))
            pooled[a][e] = cell

    b_mu, b_sd = _mean_sd([r["baseline"]["auroc"] for r in runs])
    agree = {}
    for e in ERROR_RATES:
        mu, sd = _mean_sd([r["agreement"][e]["defect_or_not_agreement"]
                           for r in runs])
        agree[e] = {"defect_or_not_agreement": mu, "sd": sd}
    return {"seeds": list(SEEDS), "runs": runs,
            "baseline_auroc": b_mu, "baseline_auroc_sd": b_sd,
            "baseline": runs[0]["baseline"], "arms": pooled,
            "agreement": agree, "golden_n": GOLDEN_N,
            "n_pool": runs[0]["n_pool"], "n_flagged": runs[0]["n_flagged"],
            "n_reviewable": runs[0]["n_reviewable"],
            "error_rates": list(ERROR_RATES)}


# ---------------------------------------------------------------------------
# the store, end to end over HTTP
# ---------------------------------------------------------------------------

def service_demo() -> dict:
    import urllib.error
    import urllib.request

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ml2svc-"))
    store = RS.ReviewStore(tmp / "review.db")
    store.queue([{"part_id": "P00001", "verdict": "REJECT_CLASSIFIED",
                  "anomaly_score": 4.1},
                 {"part_id": "P00002", "verdict": "FLAG_FOR_REVIEW",
                  "anomaly_score": 2.0},
                 {"part_id": "P00003", "verdict": "ACCEPT",
                  "anomaly_score": 0.3}], "fp-v1", 1.5)
    h = RS.serve(store, "<h1>review station</h1>")

    def post(body):
        req = urllib.request.Request(
            h["url"] + "/api/dispose", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as f:
                return f.status, json.load(f)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    log = []
    log.append(("a normal disposition",
                post({"part_id": "P00001", "operator": "OP-A",
                      "disposition": "CONFIRMED_DEFECT", "idem_key": "k1"})))
    log.append(("the same request again (double-clicked button)",
                post({"part_id": "P00001", "operator": "OP-A",
                      "disposition": "CONFIRMED_DEFECT", "idem_key": "k1"})))
    log.append(("a second operator disagreeing",
                post({"part_id": "P00001", "operator": "OP-B",
                      "disposition": "FALSE_REJECT"})))
    log.append(("false reject on a part the model ACCEPTED",
                post({"part_id": "P00003", "operator": "OP-A",
                      "disposition": "FALSE_REJECT"})))
    log.append(("a part that is not on the queue",
                post({"part_id": "NOPE", "operator": "OP-A",
                      "disposition": "UNCLASSIFIED"})))

    with urllib.request.urlopen(h["url"] + "/api/agreement") as f:
        ag = json.load(f)
    with urllib.request.urlopen(h["url"] + "/api/labels") as f:
        lab = json.load(f)
    with urllib.request.urlopen(h["url"] + "/api/labels?consensus=1") as f:
        lab_c = json.load(f)
    h["server"].shutdown()
    return {"log": log, "agreement": ag, "labels": lab,
            "labels_consensus_only": lab_c,
            "consensus_P00001": store.consensus("P00001")}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(res: dict, demo: dict) -> str:
    L = []
    A = L.append
    A("# ML-2 pass 5 - the review station writes, and the loop eats it\n")
    A("Two not-built items, and closing the first makes the second "
      "answerable.\n")

    A("## 1. The station writes\n")
    A("`src/review_service.py` is a store and an HTTP API; "
      "`write_review_station(..., post_to=...)` points the existing buttons at "
      "it. The static page still works with no service, because a page that "
      "needs a server running cannot be emailed to a supplier and a page whose "
      "log dies with the tab cannot feed a retraining loop. Neither is a "
      "superset of the other.\n")
    A("What the API does when it is asked for something wrong, end to end:\n")
    A("| request | status | response |")
    A("|---|---|---|")
    for name, (code, body) in demo["log"]:
        cell = (body.get("error") or
                ("replayed" if body.get("replayed") else "recorded"))
        A(f"| {name} | {code} | {cell} |")
    A("")
    A("The two that matter are the middle ones. **A double-clicked button is "
      "one opinion, not two** - without the idempotency key the replay enrols "
      "a second row from the same person, and the agreement statistic below "
      "then reports an operator agreeing with themselves. And **FALSE_REJECT "
      "against ACCEPT is refused**: there was no reject for the operator to "
      "call false, and a store that takes it has quietly recorded a "
      "good-part label nobody intended.\n")
    c = demo["consensus_P00001"]
    in_all = any(r["part_id"] == "P00001" for r in demo["labels"])
    in_cons = any(r["part_id"] == "P00001" for r in demo["labels_consensus_only"])
    A(f"P00001 got two contradicting answers, so it yields no consensus label "
      f"(`{c['why']}`). It appears in `/api/labels` "
      f"{'yes' if in_all else 'no'} and in the consensus-only view "
      f"{'yes' if in_cons else 'no'}.\n")

    A("## 2. The loop had never eaten a human label\n")
    A("`complete.py`'s retraining stage models the CENSORING carefully - only "
      "flagged parts are reviewed, and it corrects for that with inverse "
      "propensity weights. Then it takes the label from `pool_y`. The truth "
      "array. **Label noise was not modelled at all**, which is a strange "
      "thing to notice about a stage whose entire subject is human review.\n")
    A(f"Baseline AUROC on a held-out golden set of {res['golden_n']}: "
      f"**{res['baseline_auroc']:.3f} ± {res['baseline_auroc_sd']:.3f}** "
      f"across {len(res['seeds'])} seeds. "
      f"{res['n_reviewable']} of {res['n_pool']} parts are flagged and "
      f"reviewable; the golden set is excluded from review, because "
      f"retraining on what you are about to score on produces a number that "
      f"means nothing.\n")

    A("### The first version of this table was noise\n")
    A("One seed, `--quick` settings, a golden set of 60. It said a "
      "**5%-error operator beat a perfect one by 0.21 AUROC**, which is not a "
      "result, and the only reason it was recognisable as noise is that the "
      "ordering was absurd rather than merely wrong. Everything below is three "
      "seeds at full settings with the spread printed, because a sweep whose "
      "steps are smaller than its own seed spread cannot answer the question "
      "it is asking.\n")

    A("### AUROC after retraining, by operator error rate\n")
    arms = list(res["arms"])
    A("| operator error | " + " | ".join(arms) + " |")
    A("|---" * (len(arms) + 1) + "|")
    for e in res["error_rates"]:
        cells = []
        for a in arms:
            m = res["arms"][a].get(e)
            cells.append("-" if m is None
                         else f"{m['auroc']:.3f} ± {m['auroc_sd']:.3f}")
        A(f"| {e:.0%} | " + " | ".join(cells) + " |")
    A("")
    A(f"Baseline: **{res['baseline_auroc']:.3f} ± "
      f"{res['baseline_auroc_sd']:.3f}**.\n")

    # Read the table honestly: only differences larger than the pooled spread
    # get called differences.
    spread = max([m["auroc_sd"] for a in arms for m in res["arms"][a].values()]
                 + [res["baseline_auroc_sd"]])
    one = res["arms"].get("one operator", {})
    A(f"The largest seed-to-seed standard deviation anywhere in the table is "
      f"**{spread:.3f}**. Nothing below that is being called a difference.\n")
    if one:
        lo = one.get(min(res["error_rates"]), {}).get("auroc")
        hi = one.get(max(res["error_rates"]), {}).get("auroc")
        if lo is not None and hi is not None:
            drop = lo - hi
            A(f"Across the whole sweep a single operator's arm moves "
              f"{lo:.3f} → {hi:.3f} ({drop:+.3f}) between a perfect operator "
              f"and one wrong 30% of the time. "
              + ("That is larger than the spread, so the degradation is real."
                 if abs(drop) > spread else
                 "**That is inside the spread.** On this data, at this scale, "
                 "the loop is not measurably sensitive to operator error "
                 "across the range swept - a weaker and more useful statement "
                 "than the trend the first table appeared to show.")
              + "\n")

    A("### How often each arm passes the requalification gate\n")
    A(f"Out of {len(res['seeds'])} seeds. The gate is the asymmetric one this "
      "project already uses: 1% recall drop allowed against 5% PPV.\n")
    A("| operator error | " + " | ".join(arms) + " |")
    A("|---" * (len(arms) + 1) + "|")
    for e in res["error_rates"]:
        cells = []
        for a in arms:
            m = res["arms"][a].get(e)
            cells.append("-" if m is None
                         else f"{m['n_pass_gate']}/{len(res['seeds'])}")
        A(f"| {e:.0%} | " + " | ".join(cells) + " |")
    A("")
    A("This is the column a plant would actually read. A gate that passes on "
      "one seed in three is a retraining loop nobody should turn on, whatever "
      "the mean AUROC says.\n")

    A("### What the second operator buys\n")
    A("| operator error | measured agreement | labels, one op | "
      "consensus only | label error, one op | label error, consensus |")
    A("|---|---|---|---|---|---|")
    for e in res["error_rates"]:
        ag = res["agreement"][e]["defect_or_not_agreement"]
        m1 = res["arms"].get("one operator", {}).get(e, {})
        m2 = res["arms"].get("two, consensus only", {}).get(e, {})
        A(f"| {e:.0%} | {'-' if ag is None else f'{ag:.3f}'} | "
          f"{m1.get('n_labels', '-')} | {m2.get('n_labels', '-')} | "
          f"{m1.get('label_error', 0):.1%} | {m2.get('label_error', 0):.1%} |")
    A("")
    A("**This is the table worth having, and it does not depend on the model "
      "at all.** Requiring consensus discards labels and cleans the ones that "
      "survive; both columns move with operator error and neither carries seed "
      "noise, because these are counts rather than a trained model's score.\n")
    A("The agreement column is the number item 6 of the not-built list has "
      "always been about, and it is measurable **only because the store keeps "
      "both answers rather than letting the second write overwrite the "
      "first**. That is not a database detail: last-write-wins is the default "
      "shape of a disposition table, and it destroys the only measurement of "
      "the human that a review station can make for free.\n")

    A("## Honest limits\n")
    for lim in RS.LIMITS:
        A(f"- {lim}")
    A("- The operator is still a SIMULATION. The error rate, the 8% "
      "abstention rate and the 65/35 tilt toward passing bad parts are "
      "guessed. What is no longer assumed is that the rate is ZERO, and the "
      "sweep exists because the right value is not known.")
    A(f"- {len(res['seeds'])} seeds is enough to show that the AUROC "
      "differences are mostly not resolvable and not enough to resolve them. "
      "The honest reading of the AUROC table is *no measurable effect at this "
      "scale*, not *no effect*.")
    A("- The label-count and label-error columns are exact and the AUROC "
      "columns are not. Where they disagree, believe the counts.")
    return "\n".join(L)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    print("1/2 review service, end to end ...", flush=True)
    demo = service_demo()
    print("2/2 retraining on operator labels, 3 seeds ...", flush=True)
    res = run()
    (OUT / "pass5.json").write_text(json.dumps(res, indent=2, default=str),
                                    encoding="utf-8")
    (DOCS / "PASS5.md").write_text(report(res, demo), encoding="utf-8")
    print(f"\nwrote docs/PASS5.md in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
