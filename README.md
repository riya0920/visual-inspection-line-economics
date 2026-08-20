# ML-2 — Visual Defect Detection with Line Economics

**Status: ~20% slice.** The two-detector comparison, the prevalence arithmetic, the
cost-matrix operating point, and the takt-time budget are built and measured. MVTec
AD, segmentation, the review-station UI, and the operator-override loop are not.

```bash
python run_inspect.py            # ~14 min on CPU
python run_inspect.py --quick
python run_inspect.py --report-only
```

Writes [docs/RESULTS.md](docs/RESULTS.md) and `out/results.json`.

## Data provenance — read this first

**Every image is synthetic.** `src/synth.py` renders a textured casting surface
across four product variants and four defect classes (pore, crack, shrinkage,
inclusion) with pixel-level ground-truth masks. It is **not MVTec AD**, not
Severstal, not a casting dataset. No AUROC here is comparable to a published
number, and the absence of MVTec is the single largest gap in this project — the
spec names it as the field's reference benchmark for good reason.

What the generator buys is pixel masks and controlled shift (a new product variant,
a lighting change), which is what makes the robustness section scoreable.

## The part that works: economics

### PPV collapse at real prevalence

Same model, same operating point, four prevalences:

| prevalence | sensitivity | specificity | **PPV** |
|---|---|---|---|
| 0.5% | 0.699 | 0.971 | **0.107** |
| 2% | 0.699 | 0.971 | **0.328** |
| 10% | 0.699 | 0.971 | 0.726 |
| 50% (balanced) | 0.699 | 0.971 | **0.960** |

**The model did not change. Only the prior did.** At a realistic line prevalence of
0.5%, nine of every ten parts this model rejects are good. Reporting the balanced
number — 0.960 — is not a small optimism, it is a different claim about a different
world, and it is why "98% accuracy on a balanced test set" is the standard way to
make an undeployable inspection model look finished.

### The worked example

200,000 parts/day, 98% recall, 3% false-reject, $4 scrap, 0.5% prevalence:

| | |
|---|---|
| good parts scrapped per day | 5,970 |
| **cost of false rejects per day** | **$23,880** |
| per year (250 days) | $5,970,000 |
| defects caught per day | 980 |
| defects escaping per day | 20 |
| PPV | 0.141 |

**$23,880 a day of scrapped good parts to catch 980 defects.** Whether that is
deployable depends entirely on what an escape costs — a 98%-recall model is not
automatically a good model, and the false-reject line is where high-volume
inspection projects die.

### The operating point comes from the cost matrix

Escape cost spans three orders of magnitude between a cosmetic blemish and a
safety-critical casting, so it is swept rather than assumed:

| escape : false-reject | chosen threshold | sensitivity | false-reject rate | PPV | expected cost |
|---|---|---|---|---|---|
| 10:1 | 0.291 | 0.646 | 0.00% | 1.000 | $14,169 |
| 50:1 | 0.291 | 0.646 | 0.00% | 1.000 | $70,846 |
| 100:1 | 0.227 | 0.674 | 1.03% | 0.247 | $138,628 |
| 300:1 | 0.125 | 0.777 | 10.15% | 0.037 | $347,918 |
| 1000:1 | 0.040 | 0.997 | **93.63%** | 0.005 | $757,847 |

The operating point moves from rejecting **nothing** to rejecting **94% of all
parts** across that range. Same model, two different business questions. A single
threshold chosen without this table is a business decision made by whoever wrote
`> 0.5`.

### Takt time

The line runs at 60 parts/min → **1000 ms takt**. The full two-stage pipeline runs
in **196.5 ms p99** on one CPU thread:

| stage | p50 | p99 |
|---|---|---|
| supervised | 25.1 ms | 63.1 ms |
| anomaly | 48.4 ms | 133.9 ms |
| **combined** | **72.5 ms** | **196.5 ms** |

19.7% of takt consumed; the pipeline supports up to 305 parts/min. The margin is
not slack to be spent — it has to absorb acquisition, transfer, the p99 tail rather
than the mean, and reject-mechanism actuation before the part reaches the diverter.
Quoting a model's latency without the takt time is quoting half a sentence.

## The part that failed: the unseen-defect experiment

This was meant to be the project's other differentiator, and **it did not work —
twice.** The report leads with that rather than burying it.

| test set | supervised AUROC | anomaly AUROC | anomaly pixel AUROC |
|---|---|---|---|
| known defects (pore, crack, shrinkage) | 0.882 | 0.848 | 0.876 |
| **UNSEEN: inclusion** | **0.958** | **0.988** | 0.999 |
| new product variant D | 0.791 | 0.779 | 0.849 |
| lighting shift + camera jitter | 0.800 | 0.819 | 0.877 |

**Attempt 1.** Train on (pore, inclusion, shrinkage), hold out `crack`. Both
detectors scored **AUROC 1.000**. Diagnosis: a crack is a *dark* local feature and
two trained classes are also dark local features, so the model had learned "dark
local anomaly" and cracks fell inside it. **"Unseen defect" is a property of the
feature space, not of the label** — holding out a class the training set already
spans tests nothing, and it's an easy mistake because the label genuinely was
absent from training.

**Attempt 2** (the run above). Hold out `inclusion`, the only *bright* defect,
against three dark ones — novelty along a polarity dimension the training set does
not cover. The supervised model still scores **0.958**, a gap of only +0.030.

The most likely reason is a limitation of the **synthetic data**, not a discovery
about CNNs: `synth.py` renders every defect as a local deviation from a smooth
textured background, so a small conv net with max-pooling learns "local deviation"
and generalises across polarity for free. Real defect classes differ in texture,
scale, edge profile and context in ways this generator does not reproduce — and the
published finding that supervised models generalise poorly to unseen defect types
is measured on real imagery, not on anything like this.

So the structural argument for the two-stage architecture still holds — a
supervised model cannot be *relied* on outside its training distribution — but
**this project provides no evidence for it**, and the recommendation should be read
as reasoning rather than as a result. Demonstrating it properly needs MVTec.

Attempt 1's numbers are kept in `out/results_crack_holdout.json`.

## What the anomaly head does win on

Pixel-level localisation: **0.999 pixel AUROC** on the held-out class, against a
supervised head that produces no localisation at all. That is a different and
smaller claim than "catches unseen defects", and it is the one the numbers support.

## Robustness

| scenario | supervised | anomaly | vs baseline |
|---|---|---|---|
| baseline (known defects) | 0.882 | 0.848 | — |
| new product variant D | 0.791 | 0.779 | −0.091 / −0.069 |
| lighting shift + jitter | 0.800 | 0.819 | −0.082 / −0.029 |

**The new-variant row is the re-validation trigger.** Variant D has a rougher,
darker surface, and the anomaly head is fitted on what normal looks like *for
A/B/C*. Variant D is legitimately unlike its training normal, so the correct
response is not to retune a threshold but to **re-qualify the model against a
golden sample set for that variant** — the same discipline a quality team applies
to a gauge after a fixture change.

How the drift would be caught in production without labels: **reject-rate SPC**.
The reject rate is a p-chart statistic, it needs no ground truth, and a sustained
shift in it signals whatever the cause. That connects ML monitoring to the quality
system the plant already runs — and DATA-2 in this portfolio is the chart engine
that would do it. **The two are not wired together.**

## What is NOT built (the other 80%)

1. **No MVTec AD.** The spec's named benchmark, and its absence means no number
   here is comparable to the literature — and, as above, it is why the
   unseen-defect experiment could not be made to work.
2. **No segmentation model.** The anomaly head produces a heat map from patch
   Mahalanobis distances; there is no supervised segmentation head and no
   Grad-CAM on the classifier, so the supervised path has zero localisation.
3. **No review-station UI and no operator-override loop.** The spec calls the
   logged disposition "retraining gold and a quality-system requirement". Neither
   the UI nor the disposition log exists.
4. **No PatchCore.** `PatchAnomaly` is a PaDiM-style per-patch Gaussian, not a
   coreset memory bank, and there is no pretrained backbone — features come from
   the small CNN trained here, which is weaker than an ImageNet backbone and is
   the main reason the absolute AUROCs are modest.
5. **No two-stage pipeline actually wired.** The architecture is argued for and
   both stages are timed together, but there is no cascade implementation with a
   screening threshold feeding the classifier.
6. **No inspection API, no serving, no container.**
7. **Gauge-R&R equivalent not built.** The spec asks for repeatability on repeated
   images, reproducibility across stations, and a golden-sample set. Only the
   concept is discussed.
8. **One model, one seed, no confidence intervals.** Every AUROC here is a point
   estimate from a single training run.

## Layout

```
src/synth.py       textured casting generator, 4 variants x 4 defect classes, pixel masks
src/models.py      supervised CNN; PaDiM-style patch anomaly head fitted on normals only
src/economics.py   prevalence tables, cost-matrix threshold search, takt budget
run_inspect.py     orchestration; writes docs/RESULTS.md
```
