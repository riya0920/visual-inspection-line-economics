# ML-2 - Visual Defect Detection with Line Economics

## What it is

A camera checks every part on a production line and decides: **pass** or **reject**.

Most defect-detection projects stop at "the model is 98% accurate". This one asks
the questions a factory actually cares about:

- How many **good parts** does the model throw away, and what does that cost per day?
- Where should the reject threshold sit, given what a missed defect costs?
- Is the model **fast enough** to keep up with the line?
- What happens when a **new kind of defect** shows up that the model never saw?

It compares two detectors:

1. **Supervised** - trained on labelled examples of each defect type.
2. **Anomaly** - trained only on good parts; flags anything that looks unusual.

## What we did

| Step | What |
|---|---|
| 1 | Built a synthetic casting-image generator: 4 product variants, 4 defect types (pore, crack, shrinkage, inclusion), pixel-level defect masks |
| 2 | Trained both detectors and measured them at real factory defect rates (0.5%, 2%), not just on a 50/50 test set |
| 3 | Added a cost layer: dollars lost per day, and a threshold chosen from the cost of a miss vs a false reject |
| 4 | Timed the full pipeline against the line speed (60 parts/min) |
| 5 | Tested robustness: new product variant, lighting change, camera jitter, a defect type held out of training |
| 6 | Built a two-stage cascade with a third outcome, **"flag for review"**, plus Grad-CAM and an operator-override log |
| 7 | Brought in **real images** from MVTec AD (911 images, 6 categories) and a stronger detector (PatchCore, pretrained backbone) |
| 8 | Added a segmentation head, a camera gauge R&R study, and a retraining loop that learns from operator reviews |
| 9 | Built an inspection API and a review station that saves operator decisions to a database |

## Results

### 1. A "good" model can still scrap thousands of good parts

Same model, only the defect rate changes. PPV = of the parts rejected, how many are really bad.

| defect rate | PPV |
|---|---|
| 50% (typical test set) | 0.960 |
| 10% | 0.726 |
| 2% | 0.328 |
| **0.5% (real line)** | **0.107** |

At a real defect rate, **9 of 10 rejected parts are actually good.** The model didn't change - only the world it runs in.

Worked example (200,000 parts/day, 98% recall, 3% false reject, $4 per part):
**5,970 good parts scrapped per day = $23,880/day, about $6M/year**, to catch 980 defects.

### 2. The right threshold depends on the business, not the model

| cost of a miss vs a false reject | parts rejected | expected cost |
|---|---|---|
| 10 : 1 | 0% | $14,169 |
| 100 : 1 | 1% | $138,628 |
| 1000 : 1 | **94%** | $757,847 |

Same model, very different answers. A cosmetic blemish and a safety-critical crack need different thresholds.

### 3. It is fast enough

Full pipeline: **196.5 ms at p99** on one CPU thread, against a 1,000 ms budget per part.
That uses about 20% of the time, and can handle up to ~305 parts/min.

### 4. The anomaly detector wins on defects it has never seen

Train on three defect types, test on the fourth (never seen):

- Supervised model **drops**; anomaly model **improves**.
- Gap: **+0.206 ± 0.083 AUROC**, anomaly ahead in **8 of 8** runs (p = 0.004).

This is why the design uses both: supervised for known defects, anomaly as a safety net for new ones.

### 5. Real images changed the conclusions

On 6 real MVTec categories, PatchCore (pretrained) beat the small CNN on **5 of 6**, but lost badly on one (`grid`).
No simple rule ("real vs synthetic" or "texture vs object") explains which one - the honest answer is **test per product**.

| category | PatchCore | small CNN |
|---|---:|---:|
| bottle | 0.997 | 0.977 |
| carpet | 0.890 | 0.423 |
| grid | 0.534 | **0.752** |
| hazelnut | 0.956 | 0.714 |
| screw | 0.550 | 0.274 |
| transistor | 0.962 | 0.799 |

### 6. Other results

| What | Result |
|---|---|
| Pointing to *where* the defect is | Grad-CAM right 3.3% of the time; segmentation head **76.7%** (23× better) |
| Camera gauge R&R | %GRR 15.5% (marginal), but stations only agree at kappa **0.32** on borderline parts |
| Retraining from operator reviews | Correcting for "operators only see flagged parts": recall **0.767 → 0.867** |
| New product variant | AUROC drops ~0.07-0.09 → needs re-qualification before use |
| Operator mistakes in review labels | Retraining still beats the baseline even at 30% operator error (one seed) |

## How we did it - key decisions

**Use real defect rates, not balanced test sets.** A 50/50 test set hides the false-reject problem. Every headline number is reported at 0.5% and 2%.

**Pick the threshold from costs.** Instead of a default `> 0.5`, the threshold is chosen by minimising expected dollar cost, swept across cost ratios because the cost of a miss is not known.

**Three outcomes, not two.** Pass / reject / **flag for review**. If the system is sure something is wrong but can't name it, an operator looks. Forcing a wrong label would poison the retraining data.

**Run many seeds before believing a result.** A single run first said the anomaly detector had *no* advantage on unseen defects. Three seeds, then eight, showed the opposite. One run was not enough.

**Hold out a defect that is truly different.** Holding out `crack` meant nothing - it looks like the other dark defects the model already learned. `inclusion` (the only bright defect) was the real test.

**Reject wrongly sized images instead of resizing them.** A silent resize turns a broken camera into confident wrong answers. The API returns an error (422) instead.

**Watch the reject rate as a control chart.** Drift can be caught without labels: a sudden change in reject rate is a signal.

**Correct for who gets reviewed.** Operators only see flagged parts, so their labels are biased. Inverse-propensity weighting fixes that during retraining.

**Keep every operator's answer.** The review store never overwrites. Keeping both answers is the only way to measure how much operators agree. A double-clicked button counts once.

**Asymmetric retraining gate.** A new model may lose at most 1% recall but up to 5% PPV, because a missed defect ships to a customer and a false reject only costs a re-check.

### Things that went against our expectations

- Anti-imbalance tricks for segmentation (Dice + positive weight) made it **worse** here (IoU 0.395 vs 0.494 plain), because the weight over-predicted defects.
- The pretrained backbone **lost** on synthetic data but **won** on most real images - synthetic data alone would have led to the wrong choice.
- A cascade was expected to save compute; it saved only **2.5%**, because the first stage has to flag almost everything to keep recall at 99%.

## Limits

- **Casting images are synthetic.** Real images come from MVTec AD only.
- **MVTec is a subset:** 911 of ~5,400 images, 6 of 15 categories, downsized to 128×128. Not comparable to published results.
- **No real pixel masks from MVTec**, so segmentation is trained and scored on synthetic masks.
- **Operators are simulated.** Error rates are guesses; the sweep exists because the true value is unknown.
- **Review service has no login** and uses one SQLite file.
- **Docker image is written but never built.**
- Only two detector designs compared, at settings chosen for this project.

## How to run

```bash
pip install -r requirements.txt
python run_inspect.py        # core results, ~14 min on CPU (--quick for a fast run)
python extend.py             # cascade, Grad-CAM, override log, ~17 min
python budget_probe.py       # multi-seed check, ~50 min
python fetch_mvtec.py        # download real MVTec images (resumable)
python complete.py           # PatchCore, segmentation, gauge R&R, retraining, ~110 min
python run_pass4.py          # 6-category MVTec comparison
python run_pass5.py          # review service + operator-noise sweep, ~4 min
python -m pytest tests       # 90 tests
```

Detailed write-ups: [RESULTS](docs/RESULTS.md) · [EXTENSIONS](docs/EXTENSIONS.md) · [COMPLETION](docs/COMPLETION.md) · [LARGER_MVTEC](docs/LARGER_MVTEC.md) · [PASS5](docs/PASS5.md)

## Layout

```
src/synth.py            synthetic casting images, 4 variants x 4 defect types, pixel masks
src/models.py           supervised CNN + anomaly head trained on good parts only
src/economics.py        defect-rate tables, cost-based threshold, takt budget
src/cascade.py          three-outcome cascade, Grad-CAM, override log
src/patchcore.py        PatchCore with a pretrained ResNet
src/segmentation.py     U-Net segmentation head
src/inspect_service.py  inspection API
src/review_service.py   review-station store and API
fetch_mvtec.py          resumable MVTec download
```

## Brief coverage

| Asked for | Done? |
|---|---|
| Casting defect set + MVTec benchmark | Partly - casting images synthetic; MVTec real (subset) |
| Supervised vs anomaly (PatchCore-style) comparison | Yes |
| Evaluate at 0.5% / 2% defect rates | Yes |
| Cost matrix, cost-based threshold, sensitivity | Yes |
| Throughput vs takt time | Yes |
| Robustness (lighting, jitter, new variant) | Yes |
| Inspection API + review station with logged overrides | Yes |
