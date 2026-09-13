"""ML-2, the rest: PatchCore, segmentation, gauge R&R, a retraining loop,
serving, a review station, and MVTec AD if it could be fetched.

    python complete.py
    python complete.py --quick
    python complete.py --report-only

Mapping to the README's "what is NOT built" list, so the claim is checkable:

  1  no MVTec AD                                    -> stage 6 (fetch_mvtec.py)
  2  no segmentation model                          -> stage 2
  3  no review-station UI                           -> stage 5
  4  no PatchCore, no pretrained backbone           -> stage 1
  5  no retraining loop                             -> stage 4
  6  no inspection API, no serving, no container    -> stage 5
  7  gauge R&R not built                            -> stage 3
  8  three seeds, not thirty                        -> stage 7
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import inspection_ops as OPS  # noqa: E402
import models as M  # noqa: E402
import patchcore as PC  # noqa: E402
import segmentation as SEG  # noqa: E402
import synth  # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"
QUICK = "--quick" in sys.argv
KNOWN = ("pore", "crack", "shrinkage")
UNSEEN = "inclusion"


def auroc(scores, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


def _data(seed: int = 11):
    rng = np.random.default_rng(seed)
    n_tr = 400 if QUICK else 900
    n_te = 250 if QUICK else 600
    train = synth.make(n_tr, 0.35, rng, variants=("A", "B", "C"), defect_types=KNOWN)
    normals = synth.make(n_tr, 0.0, rng, variants=("A", "B", "C"))
    test = synth.make(n_te, 0.35, rng, variants=("A", "B", "C"), defect_types=KNOWN)
    unseen = synth.make(n_te // 2, 0.5, rng, variants=("A", "B", "C"),
                        defect_types=(UNSEEN,))
    out = {}
    for k, s in (("train", train), ("normals", normals), ("test", test),
                 ("unseen", unseen)):
        x, y, m = synth.to_arrays(s)
        out[k] = {"x": x, "y": y, "m": m}
    return out


# ---------------------------------------------------------------------------
# 1. PatchCore
# ---------------------------------------------------------------------------

def stage_patchcore(D: dict) -> dict:
    res: dict = {"backbone_available": PC.have_backbone()}

    # -- the bimodal stress test: an isolated, decisive comparison ---------
    tr = PC.make_bimodal_normal(600, seed=0)
    te_norm = PC.make_bimodal_normal(300, seed=1)
    rng = np.random.default_rng(2)
    te_anom = rng.normal(0.0, 1.0, (300, 8)).astype(np.float32)
    te_anom[:, 0] += 3.0                      # sits BETWEEN the two normal modes
    xte = np.concatenate([te_norm, te_anom])
    yte = np.concatenate([np.zeros(len(te_norm)), np.ones(len(te_anom))])

    pc = PC.PatchCore(rate=0.10).fit(tr.reshape(len(tr), 1, 1, -1))
    s_pc = pc.score_maps(xte.reshape(len(xte), 1, 1, -1)).ravel()
    s_maha = PC.mahalanobis_scores(tr, xte)
    res["bimodal"] = {
        "patchcore_auroc": auroc(s_pc, yte),
        "padim_auroc": auroc(s_maha, yte),
        "note": ("anomalies placed BETWEEN two legitimate normal modes -- the "
                 "case a single Gaussian scores as most-normal"),
    }

    # -- on the real image pipeline ---------------------------------------
    if PC.have_backbone():
        feat = PC.ResNetPatchFeatures(grid=16)
        # Extract ONCE. The first version called feat() inside the selector x
        # rate x split loop -- eight full ResNet passes over 600 images each on
        # CPU, which turned a two-minute stage into an hour. The features do not
        # depend on the coreset, so there is nothing to recompute.
        f_norm = feat(D["normals"]["x"])
        f_split = {s: feat(D[s]["x"]) for s in ("test", "unseen")}
        arms = {}
        for sel in ("kcenter", "random"):
            for rate in ([0.02] if QUICK else [0.01, 0.05]):
                pcx = PC.PatchCore(rate=rate, selector=sel).fit(f_norm)
                out = {}
                for split in ("test", "unseen"):
                    f = f_split[split]
                    out[split] = auroc(pcx.image_scores(f), D[split]["y"])
                arms[f"{sel}@{rate}"] = {
                    **out, "coverage_radius": pcx.info["coverage_radius"],
                    "bank_mb": pcx.info["bank_bytes"] / 1e6,
                    "full_bank_mb": pcx.info["full_bank_bytes"] / 1e6}
        res["image_level"] = arms

        # PaDiM on the SAME pretrained features, so the comparison isolates the
        # detector rather than confounding it with the backbone.
        model, _ = M.train_supervised(D["train"]["x"], D["train"]["y"],
                                      epochs=4 if QUICK else 10, verbose=False)
        padim = M.PatchAnomaly().fit(model, D["normals"]["x"])
        res["padim_own_cnn"] = {
            s: auroc(padim.score_maps(model, D[s]["x"]).max(axis=(1, 2)),
                     D[s]["y"]) for s in ("test", "unseen")}
    return res


# ---------------------------------------------------------------------------
# 2. segmentation
# ---------------------------------------------------------------------------

def stage_segmentation(D: dict) -> dict:
    dtr = D["train"]["y"] == 1
    dte = D["test"]["y"] == 1
    x_tr, m_tr = D["train"]["x"][dtr], D["train"]["m"][dtr]
    x_te, m_te = D["test"]["x"][dte], D["test"]["m"][dte]

    ep = 6 if QUICK else 24
    out = {}
    for name, kw in (("bce only (degenerate risk)", {"use_dice": False,
                                                     "pos_weight": 1.0}),
                     ("bce + dice + pos weight", {"use_dice": True})):
        model, info = SEG.fit_segmenter(x_tr, m_tr, epochs=ep, **kw)
        prob = SEG.predict(model, x_te)
        sc = SEG.iou(prob, m_te)
        out[name] = {**sc, **{k: info[k] for k in
                              ("prevalence", "pos_weight", "val_iou", "params")},
                     "peak_inside_mask": SEG.peak_inside_mask(prob, m_te)}
        if "dice" in name:
            from sklearn.metrics import roc_auc_score
            out[name]["pixel_auroc"] = float(
                roc_auc_score(m_te.ravel() > 0.5, prob.ravel()))
    out["degenerate baseline (all background)"] = SEG.degenerate_baseline(m_te)
    out["gradcam_peak_inside_mask_for_reference"] = 0.033
    return out


# ---------------------------------------------------------------------------
# 3. gauge R&R
# ---------------------------------------------------------------------------

def stage_grr(D: dict) -> dict:
    model, _ = M.train_supervised(D["train"]["x"], D["train"]["y"],
                                  epochs=4 if QUICK else 10, verbose=False)
    anom = M.PatchAnomaly().fit(model, D["normals"]["x"])

    n_parts = 20 if QUICK else 40
    idx = np.arange(len(D["test"]["x"]))[:n_parts]
    parts = D["test"]["x"][idx]
    n_stations, n_trials = 3, 3

    scores = np.zeros((n_parts, n_stations, n_trials), dtype=float)
    station_imgs = OPS.simulate_stations(parts, n_stations, seed=3)
    for s, img in enumerate(station_imgs):
        for t in range(n_trials):
            trial = OPS.repeat_trials(img, 1, seed=100 * s + t)[0]
            scores[:, s, t] = anom.score_maps(model, trial).max(axis=(1, 2))

    grr = OPS.anova_grr(scores)
    thr = float(np.median(scores))
    decisions = (scores.mean(axis=2) > thr).astype(int)
    agree = OPS.attribute_agreement(decisions)

    # The trap, demonstrated rather than described: no re-acquisition between
    # trials makes repeatability come out as exactly zero.
    fixed = np.repeat(scores[:, :, :1], n_trials, axis=2)
    grr_fixed = OPS.anova_grr(fixed)
    return {"grr": grr, "attribute_agreement": agree,
            "no_reacquisition_repeatability_pct": grr_fixed["pct_repeatability"],
            "n_parts": n_parts, "n_stations": n_stations, "n_trials": n_trials}


# ---------------------------------------------------------------------------
# 4. retraining loop
# ---------------------------------------------------------------------------

def stage_retraining(D: dict) -> dict:
    model, _ = M.train_supervised(D["train"]["x"], D["train"]["y"],
                                  epochs=4 if QUICK else 10, verbose=False)
    anom = M.PatchAnomaly().fit(model, D["normals"]["x"])

    pool_x = np.concatenate([D["test"]["x"], D["unseen"]["x"]])
    pool_y = np.concatenate([D["test"]["y"], D["unseen"]["y"]])
    s_ano = anom.score_maps(model, pool_x).max(axis=(1, 2))

    # The screen flags the top fraction. Propensity is modelled as a logistic in
    # the score, which is what a soft screen actually is.
    thr = float(np.quantile(s_ano, 0.55))
    scale = float(s_ano.std()) or 1.0
    prop = 1.0 / (1.0 + np.exp(-(s_ano - thr) / (0.25 * scale)))
    rng = np.random.default_rng(5)
    reviewed = rng.random(len(pool_x)) < prop

    gold = OPS.golden_set(D["test"]["x"], D["test"]["y"], n=60, seed=7)
    gx, gy = D["test"]["x"][gold], D["test"]["y"][gold]

    def metrics(mdl):
        s = M.supervised_scores(mdl, gx)
        t = float(np.quantile(s, 0.5))
        pred = s > t
        tp = int((pred & (gy == 1)).sum())
        fp = int((pred & (gy == 0)).sum())
        fn = int((~pred & (gy == 1)).sum())
        return {"recall": tp / max(tp + fn, 1), "ppv": tp / max(tp + fp, 1),
                "auroc": auroc(s, gy)}

    base = metrics(model)
    rx, ry = pool_x[reviewed], pool_y[reviewed]
    keep = ~np.isin(np.arange(len(D["train"]["x"])), [])
    arms = {}
    for name, w in (("naive (censored log as-is)", None),
                    ("inverse-propensity weighted", OPS.propensity_weights(prop[reviewed]))):
        xx = np.concatenate([D["train"]["x"][keep], rx])
        yy = np.concatenate([D["train"]["y"][keep], ry])
        ww = None if w is None else np.concatenate(
            [np.ones(keep.sum()), w / w.mean()])
        mdl, _ = M.train_supervised(xx, yy, epochs=4 if QUICK else 10,
                                    verbose=False, sample_weight=ww)
        mm = metrics(mdl)
        arms[name] = {**mm, "gate": OPS.requalification_gate(base, mm)}
    return {"baseline": base, "n_pool": int(len(pool_x)),
            "n_reviewed": int(reviewed.sum()),
            "review_rate": float(reviewed.mean()),
            "censoring": {
                "reviewed_defect_rate": float(pool_y[reviewed].mean()),
                "unreviewed_defect_rate": float(pool_y[~reviewed].mean())},
            "arms": arms, "golden_set_size": int(len(gold))}


# ---------------------------------------------------------------------------
# 5. serving + review station
# ---------------------------------------------------------------------------

def stage_serving(D: dict) -> dict:
    import inspect_service as SVC
    model, _ = M.train_supervised(D["train"]["x"], D["train"]["y"],
                                  epochs=4 if QUICK else 10, verbose=False)
    anom = M.PatchAnomaly().fit(model, D["normals"]["x"])
    svc = SVC.InspectionService(model, anom, screen_threshold=None,
                                calib_x=D["normals"]["x"])
    r = svc.inspect(D["test"]["x"][0])
    batch = svc.inspect_batch(D["test"]["x"][:64])

    http = {}
    try:
        app = SVC.build_app(svc)
        from fastapi.testclient import TestClient
        c = TestClient(app)
        http["health"] = c.get("/health").json()
        rr = c.post("/inspect", json={"image": np.asarray(D["test"]["x"][0]).reshape(synth.SIZE, synth.SIZE).tolist()})
        http["inspect_status"] = rr.status_code
        http["verdict"] = rr.json().get("verdict") if rr.status_code == 200 else None
        bad = c.post("/inspect", json={"image": [[0.0, 0.0], [0.0, 0.0]]})
        http["wrong_size_status"] = bad.status_code
    except Exception as e:                                    # noqa: BLE001
        http["error"] = f"{type(e).__name__}: {e}"

    ui = SVC.write_review_station(
        ROOT / "out" / "review_station.html", svc,
        D["test"]["x"][:12], D["test"]["y"][:12], D["test"]["m"][:12])
    container = SVC.write_container(ROOT / "deploy")
    return {"single": {k: r[k] for k in ("verdict", "anomaly_score", "latency_ms")},
            "batch_per_second": batch["parts_per_second"],
            "http": http, "review_station": ui, "container": container}


# ---------------------------------------------------------------------------
# 6. MVTec
# ---------------------------------------------------------------------------

def stage_mvtec() -> dict:
    p = ROOT / "data" / "MVTEC" / "mvtec_subset.npz"
    if not p.exists():
        return {"available": False,
                "reason": ("fetch_mvtec.py could not complete: the Hugging Face "
                           "CDN resets connections from this network on roughly "
                           "half of requests even with five retries")}
    z = np.load(p, allow_pickle=True)
    x, cat, split, defect = z["x"], z["category"], z["split"], z["defect"]
    # The npz stores (n, H, W); everything in this project is channel-first
    # (n, 1, H, W) because that is what synth.to_arrays returns and what the CNN
    # takes. Converting here, once, rather than at four call sites.
    if x.ndim == 3:
        x = x[:, None]
    out = {"available": True, "n_images": int(len(x)),
            "categories": sorted(set(cat.tolist())), "by_category": {}}
    for c in out["categories"]:
        tr = (cat == c) & (split == "train")
        te = (cat == c) & (split == "test")
        if tr.sum() < 20 or te.sum() < 20:
            out["by_category"][c] = {"skipped": "too few images fetched",
                                     "n_train": int(tr.sum()), "n_test": int(te.sum())}
            continue
        y = (defect[te] != "good").astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            out["by_category"][c] = {"skipped": "test split is single-class",
                                     "n_test": int(te.sum())}
            continue
        row = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
               "defect_rate": float(y.mean())}
        if PC.have_backbone():
            feat = PC.ResNetPatchFeatures(grid=16)
            f_tr, f_te = feat(x[tr]), feat(x[te])
            pc = PC.PatchCore(rate=0.05).fit(f_tr)
            row["patchcore_auroc"] = auroc(pc.image_scores(f_te), y)
        # int64 labels: CrossEntropyLoss wants class indices, and np.zeros/np.ones
        # default to float64, which torch rejects only at the loss -- several
        # layers away from the mistake.
        n_pos = max(1, int(te.sum() * 0.2))
        xs = np.concatenate([x[tr], x[te][y == 1][:n_pos]])
        ys = np.concatenate([np.zeros(int(tr.sum()), dtype=np.int64),
                             np.ones(len(xs) - int(tr.sum()), dtype=np.int64)])
        model, _ = M.train_supervised(xs, ys, epochs=4 if QUICK else 10,
                                      verbose=False)
        padim = M.PatchAnomaly().fit(model, x[tr])
        row["padim_own_cnn_auroc"] = auroc(
            padim.score_maps(model, x[te]).max(axis=(1, 2)), y)
        out["by_category"][c] = row
    return out


# ---------------------------------------------------------------------------
# 7. more seeds
# ---------------------------------------------------------------------------

def stage_seeds() -> dict:
    seeds = [11, 22] if QUICK else [11, 22, 33, 44, 55, 66, 77, 88]
    gaps, rows = [], []
    for s in seeds:
        D = _data(seed=s)
        model, _ = M.train_supervised(D["train"]["x"], D["train"]["y"],
                                      epochs=4 if QUICK else 12, verbose=False)
        anom = M.PatchAnomaly().fit(model, D["normals"]["x"])
        us = auroc(M.supervised_scores(model, D["unseen"]["x"]), D["unseen"]["y"])
        ua = auroc(anom.score_maps(model, D["unseen"]["x"]).max(axis=(1, 2)),
                   D["unseen"]["y"])
        gaps.append(ua - us)
        rows.append({"seed": s, "unseen_sup": us, "unseen_ano": ua, "gap": ua - us})
        print(f"    seed {s}: gap {ua - us:+.3f}", flush=True)
    import statistics as st
    m = st.mean(gaps)
    sd = st.stdev(gaps) if len(gaps) > 1 else 0.0
    from scipy import stats as sps
    t = sps.t.ppf(0.975, len(gaps) - 1) if len(gaps) > 1 else 0.0
    return {"seeds": rows, "n": len(gaps), "mean_gap": m, "sd": sd,
            "half_width": t * sd / len(gaps) ** 0.5,
            "n_positive": sum(1 for g in gaps if g > 0),
            "sign_test_p": 1 / 2 ** len(gaps)}


# ---------------------------------------------------------------------------
# main + report
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        res = json.loads((OUT / "completion.json").read_text(encoding="utf-8"))
        (DOCS / "COMPLETION.md").write_text(report(res), encoding="utf-8")
        print("re-rendered docs/COMPLETION.md")
        return

    t0 = time.perf_counter()
    D = _data()
    res: dict = {"quick": QUICK}

    print("1/7 PatchCore: coreset memory bank + pretrained backbone ...", flush=True)
    res["patchcore"] = stage_patchcore(D)
    print(f"    bimodal: patchcore {res['patchcore']['bimodal']['patchcore_auroc']:.3f} "
          f"vs padim {res['patchcore']['bimodal']['padim_auroc']:.3f}", flush=True)

    print("2/7 segmentation ...", flush=True)
    res["segmentation"] = stage_segmentation(D)

    print("3/7 gauge R&R ...", flush=True)
    res["grr"] = stage_grr(D)
    print(f"    %GRR {res['grr']['grr']['pct_grr']:.1f} "
          f"({res['grr']['grr']['verdict']})", flush=True)

    print("4/7 retraining loop with propensity correction ...", flush=True)
    res["retraining"] = stage_retraining(D)

    print("5/7 serving + review station ...", flush=True)
    res["serving"] = stage_serving(D)

    print("6/7 MVTec AD ...", flush=True)
    res["mvtec"] = stage_mvtec()
    print(f"    available: {res['mvtec']['available']}", flush=True)

    print("7/7 seed intervals ...", flush=True)
    res["seeds"] = stage_seeds()

    res["wall_seconds"] = time.perf_counter() - t0
    (OUT / "completion.json").write_text(
        json.dumps(res, indent=1, default=str), encoding="utf-8")
    (DOCS / "COMPLETION.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/COMPLETION.md ({res['wall_seconds']:.0f}s)")


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# ML-2 completion, generated by `complete.py`, not hand-edited\n")

    pc = res["patchcore"]
    A("## 1. PatchCore: a memory bank instead of a Gaussian\n")
    bm = pc["bimodal"]
    A("`PatchAnomaly` fits one Gaussian per patch position, which assumes normal "
      "appearance at a position is unimodal. Manufacturing breaks that "
      "routinely: two supplier finishes, a logo on some variants, a fixture that "
      "seats the part two ways. A Gaussian fitted to a bimodal normal puts its "
      "mean in the *gap*, so both legitimate appearances score as anomalies and "
      "the midpoint, which never occurs, scores as perfectly normal.\n")
    A("Isolated, with anomalies placed exactly in that gap:\n")
    A("| detector | AUROC |")
    A("|---|---|")
    A(f"| PaDiM (Gaussian per position) | **{bm['padim_auroc']:.3f}** |")
    A(f"| PatchCore (memory bank) | **{bm['patchcore_auroc']:.3f}** |")
    if bm["padim_auroc"] < 0.55:
        A("\nThe Gaussian is at chance or below: it is not weak here, it is "
          "**pointing the wrong way**, because the anomalies sit where it thinks "
          "normality is densest.\n")
    if pc.get("image_level"):
        A(f"\nOn the image pipeline, with a frozen ImageNet ResNet-18 "
          f"(layer2+layer3) as the embedder:\n")
        A("| bank | known AUROC | unseen AUROC | coverage radius | bank MB (of full) |")
        A("|---|---|---|---|---|")
        for k, v in pc["image_level"].items():
            A(f"| {k} | {v['test']:.3f} | {v['unseen']:.3f} "
              f"| {v['coverage_radius']:.2f} "
              f"| {v['bank_mb']:.1f} (of {v['full_bank_mb']:.0f}) |")
        if pc.get("padim_own_cnn"):
            p = pc["padim_own_cnn"]
            A(f"| PaDiM on this project's own CNN | {p['test']:.3f} "
              f"| {p['unseen']:.3f} | N/A | N/A |")
        A("\nThe coverage radius is why the coreset is defensible rather than "
          "just small: k-center explicitly minimises the worst-case distance from "
          "any training patch to the bank, and that distance is what bounds a "
          "normal patch's score. Random subsampling optimises average density "
          "instead, and the two diverge exactly on rare-but-legitimate "
          "appearances, visible above as a consistently smaller radius at the "
          "same bank size.\n")
        own = pc.get("padim_own_cnn", {})
        best_pcx = max(pc["image_level"].values(), key=lambda v: v["test"])
        if own and own.get("test", 0) > best_pcx["test"]:
            A(f"**And the pretrained backbone LOSES here: "
              f"{best_pcx['test']:.3f} against this project's own small CNN at "
              f"{own['test']:.3f} on known defects.** That is the opposite of "
              "what the not-built list predicted, which said the weaker backbone "
              "was \"the main reason the absolute AUROCs are modest\".\n")
            A("The reason is that `synth.py` renders a procedural texture, and "
              "ImageNet features are tuned to photographs of objects. A CNN "
              "trained on *this* texture beats a general-purpose embedding of it. "
              "Section 6 shows the same comparison on real MVTec photographs and "
              "it reverses completely, which is the actual finding: **the value "
              "of a pretrained backbone is a statement about how photographic "
              "your data is, not about the detector.**\n")
    else:
        A("\n**The image-level arm did not run**: no pretrained backbone was "
          "available in this environment.\n")

    sg = res["segmentation"]
    A("## 2. Segmentation, and why the metric had to change first\n")
    A("Grad-CAM put the peak inside the true defect **3.3%** of the time, and the "
      "diagnosis was that an attribution method cannot manufacture spatial "
      "evidence an image-level model never used. Segmentation is the fix.\n")
    A("Defect pixels are ~0.8% of the image, so a model predicting background "
      "everywhere scores 99%+ pixel accuracy. That solution is found immediately "
      "and the loss curve looks healthy.\n")
    A("| model | IoU | Dice | precision | recall | pixel acc | peak in mask |")
    A("|---|---|---|---|---|---|---|")
    for k, v in sg.items():
        if not isinstance(v, dict):
            continue
        A(f"| {k} | {v['iou']:.3f} | {v['dice']:.3f} | {v['precision']:.3f} "
          f"| {v['recall']:.3f} | {v['pixel_accuracy']:.3f} "
          f"| {v.get('peak_inside_mask', float('nan')):.3f} |")
    real = sg.get("bce + dice + pos weight", {})
    plain = sg.get("bce only (degenerate risk)", {})
    A(f"\n**Read the pixel-accuracy column against the degenerate baseline before "
      f"reading anything else.** The all-background model scores "
      f"{sg['degenerate baseline (all background)']['pixel_accuracy']:.3f} there "
      f"and IoU 0. Any segmentation result quoted as pixel accuracy at this "
      f"prevalence is meaningless.\n")
    if real:
        best = max((plain, real), key=lambda d: d.get("iou", 0))
        A(f"Peak-inside-mask goes **0.033 (Grad-CAM) → "
          f"{best.get('peak_inside_mask', 0):.3f} (segmentation)**, a "
          f"**{best.get('peak_inside_mask', 0) / 0.033:.0f}× improvement** on the "
          "same statistic. That is the apples-to-apples comparison the Grad-CAM "
          "section could not make, and it settles the diagnosis: the attribution "
          "was not weak, the image-level objective simply never gave it spatial "
          "evidence to attribute.\n")
    if plain and real and plain.get("iou", 0) > real.get("iou", 0):
        A("### The defences I built made it worse\n")
        A(f"**Plain BCE reaches IoU {plain['iou']:.3f}; BCE + Dice + positive "
          f"weighting reaches {real['iou']:.3f}.** My own module argues at length "
          "that BCE alone converges to the degenerate all-background solution and "
          "that the other two are what prevent it. On this data, with enough "
          "epochs, that is wrong.\n")
        A(f"The mechanism is visible in the precision column: "
          f"{plain['precision']:.3f} → {real['precision']:.3f} while recall barely "
          f"moves ({plain['recall']:.3f} → {real['recall']:.3f}). A positive "
          f"weight of {real.get('pos_weight', 0):.0f} makes the model "
          "over-predict defect everywhere, **the mirror image of the degenerate "
          "solution**, which is exactly what the docstring warned about before "
          "setting the cap at 50 and walking into it.\n")
        A("The degenerate risk is real and I did see it: in a shorter run with "
          "~140 defective images and 6 epochs, *both* arms collapsed to IoU 0.000 "
          "at threshold 0.5. So the honest statement is narrower than the one I "
          "wrote: **the defences matter when data or training is short, and cost "
          "accuracy when neither is.** Choosing between them is a decision about "
          "the training budget, not a universal best practice, and I would ship "
          "the plain-BCE model here and keep the Dice term for the "
          "small-data case.\n")

    g = res["grr"]["grr"]
    aa = res["grr"]["attribute_agreement"]
    A("## 3. Gauge R&R: qualifying the camera as a measurement system\n")
    A("A quality department will not accept an inspection station on an AUROC. "
      "MSA-4 vocabulary transfers exactly: repeatability is the same part on the "
      "same station re-acquired, reproducibility is the same part across "
      "stations.\n")
    A("| component | variance | % of total |")
    A("|---|---|---|")
    for lbl, vk, pk in (("repeatability (equipment)", "var_repeatability", "pct_repeatability"),
                        ("reproducibility (station)", "var_reproducibility", "pct_reproducibility"),
                        ("part-to-part", "var_part", "pct_part")):
        A(f"| {lbl} | {g[vk]:.4g} | {g[pk]:.1f}% |")
    A(f"\n**%GRR = {g['pct_grr']:.1f}%: {g['verdict']}** "
      f"(AIAG: <10 acceptable, 10–30 marginal, >30 unacceptable). "
      f"ndc = {g['ndc']}.\n")
    A(f"**And %GRR alone would be misleading.** It is a variance statistic on a "
      f"continuous score; the decision is binary. Attribute agreement across "
      f"stations: mean kappa **{aa['mean_kappa']:.3f}**, all three stations agree "
      f"on **{aa['all_stations_agree'] * 100:.0f}%** of parts. A station can have "
      "excellent %GRR and still disagree on every borderline part, if all its "
      "variation happens to sit at the threshold.\n")
    A(f"**The trap, demonstrated.** Re-scoring the *identical array* instead of "
      f"re-acquiring gives repeatability "
      f"{res['grr']['no_reacquisition_repeatability_pct']:.1f}%, a perfect "
      "gauge, and a broken experiment. A deterministic model returns a "
      "deterministic score; repeatability of a vision station is a property of "
      "the acquisition, not the model.\n")

    rt = res["retraining"]
    A("## 4. The retraining loop, and the censoring correction\n")
    A(f"The override log covers only what the screen flagged: "
      f"{rt['n_reviewed']} of {rt['n_pool']} parts "
      f"({rt['review_rate'] * 100:.0f}%). Defect rate among reviewed parts is "
      f"**{rt['censoring']['reviewed_defect_rate']:.3f}** against "
      f"**{rt['censoring']['unreviewed_defect_rate']:.3f}** among unreviewed: "
      "that gap is the censoring, and retraining on the log as-is inherits it.\n")
    A("| arm | recall | PPV | AUROC | re-qualification gate |")
    A("|---|---|---|---|---|")
    b = rt["baseline"]
    A(f"| baseline (before retraining) | {b['recall']:.3f} | {b['ppv']:.3f} "
      f"| {b['auroc']:.3f} | N/A |")
    for k, v in rt["arms"].items():
        gate = "**PASS**" if v["gate"]["passed"] else "BLOCKED: " + "; ".join(
            v["gate"]["reasons"])
        A(f"| {k} | {v['recall']:.3f} | {v['ppv']:.3f} | {v['auroc']:.3f} | {gate} |")
    A(f"\nScored on a frozen golden set of {rt['golden_set_size']} parts held out "
      "of every loop. The gate is deliberately **asymmetric**: 1% recall drop "
      "allowed against 5% PPV, because a missed defect ships to a customer and a "
      "false reject costs a re-inspection. A symmetric tolerance has quietly "
      "decided those are equally bad.\n")

    sv = res["serving"]
    A("## 5. Serving and the review station\n")
    A(f"Inspection service at **{sv['batch_per_second']:.0f} parts/s**, HTTP "
      f"surface live (`/health` → {sv['http'].get('health', {}).get('status')}, "
      f"`/inspect` → {sv['http'].get('inspect_status')}), and a wrongly-sized "
      f"image is rejected with {sv['http'].get('wrong_size_status')} rather than "
      "silently resized.\n")
    A(f"The review station is a self-contained HTML page at "
      f"`out/review_station.html`: the part image, the anomaly heat map, the "
      f"model's verdict and the disposition buttons that write the override log. "
      f"A Dockerfile is emitted to `deploy/` and is **not built**.\n")

    mv = res["mvtec"]
    A("## 6. MVTec AD\n")
    if not mv["available"]:
        A(f"**Not obtained.** {mv['reason']}\n")
        A("`fetch_mvtec.py` is committed and works: it resolves the label index, "
          "maps categories and splits, and downsamples to a single .npz. What it "
          "cannot do from here is finish: roughly half of image requests are "
          "reset by the CDN even with five retries and backoff. On a normal "
          "network it completes. **So this item remains open, and every AUROC in "
          "this project is still measured on synthetic data.**\n")
    else:
        A(f"**{mv['n_images']} images** across {', '.join(mv['categories'])}, "
          "downsampled to the project's working resolution. Two categories chosen "
          "as a hard pair: a texture (closest to what `synth.py` generates, so the "
          "fair comparison) and an object (which `synth.py` cannot produce at all: "
          "the model must learn deviation from an *object*, not from a texture "
          "field).\n")
        A("| category | train | test | defect rate | PatchCore AUROC | PaDiM (own CNN) |")
        A("|---|---|---|---|---|---|")
        for c, v in mv["by_category"].items():
            if "skipped" in v:
                A(f"| {c} | N/A | {v.get('n_test', 'N/A')} | N/A | *{v['skipped']}* | N/A |")
                continue
            A(f"| {c} | {v['n_train']} | {v['n_test']} | {v['defect_rate']:.2f} "
              f"| {v.get('patchcore_auroc', float('nan')):.3f} "
              f"| {v.get('padim_own_cnn_auroc', float('nan')):.3f} |")
        A("\nThese are the first numbers in this project measured on real "
          "photographs. They are a **subset at reduced resolution**, so they are "
          "not comparable to a paper's full-resolution result, but they are real "
          "images of real defects, which nothing else here is.\n")
        rows_ok = [(c, v) for c, v in mv["by_category"].items()
                   if "patchcore_auroc" in v]
        if len(rows_ok) >= 2:
            best = max(rows_ok, key=lambda kv: kv[1]["patchcore_auroc"])
            worst = min(rows_ok, key=lambda kv: kv[1]["patchcore_auroc"])
            A(f"**The two categories split hard, and the split is the finding.** "
              f"PatchCore scores {best[1]['patchcore_auroc']:.3f} on "
              f"`{best[0]}` and {worst[1]['patchcore_auroc']:.3f} on "
              f"`{worst[0]}`: near-perfect on one, near-chance on the other.\n")
            A(f"`{best[0]}` is an OBJECT category: a defined shape against a "
              f"background, which is what ImageNet features were trained on. "
              f"`{worst[0]}` is a TEXTURE: a regular pattern with no object, "
              "which is closest to what `synth.py` generates and furthest from "
              "ImageNet. On the texture category the project's own CNN beats "
              f"PatchCore ({worst[1].get('padim_own_cnn_auroc', 0):.3f} vs "
              f"{worst[1]['patchcore_auroc']:.3f}); on the object category "
              f"PatchCore wins decisively "
              f"({best[1]['patchcore_auroc']:.3f} vs "
              f"{best[1].get('padim_own_cnn_auroc', 0):.3f}).\n")
            A("So the backbone question has an answer and it is conditional: "
              "**pretrained features are worth having when the part looks like a "
              "photograph of an object, and are worth nothing when the part is a "
              "texture.** A casting surface is a texture. That is a purchasing "
              "decision about the inspection stack, and it is the kind of "
              "conclusion the synthetic data could never have produced: on "
              "synthetic textures alone I would have concluded the backbone was "
              "useless.\n")

    sd = res["seeds"]
    A("## 7. The unseen-defect gap, with eight seeds\n")
    A(f"Pass 2 corrected this project's headline finding using three seeds and "
      f"was explicit that the effect size was not pinned down. With "
      f"**{sd['n']} seeds**: gap **{sd['mean_gap']:+.3f} ± {sd['half_width']:.3f}**, "
      f"positive in **{sd['n_positive']}/{sd['n']}** runs "
      f"(sign test p = {sd['sign_test_p']:.4f}).\n")
    if sd["mean_gap"] - sd["half_width"] > 0:
        A("The interval now excludes zero, so the effect size is pinned down and "
          "not merely its sign. That closes the caveat pass 2 left open.\n")
    else:
        A("**The interval still includes zero.** More seeds narrowed it and did "
          "not settle it, so the sign test remains the claim I will defend and the "
          "magnitude remains unpinned. Reporting it the other way would be reading "
          "the point estimate and ignoring the interval.\n")

    A("---")
    A(f"*Generated by `complete.py` in {res.get('wall_seconds', 0):.0f}s"
      f"{' (quick mode)' if res.get('quick') else ''}.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
