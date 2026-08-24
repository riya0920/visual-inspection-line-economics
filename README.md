# ML-2 — Visual Defect Detection with Line Economics

**Status: complete.** The two-detector comparison, the prevalence arithmetic, the
cost-matrix operating point, the takt-time budget, the two-stage cascade, Grad-CAM,
the operator-override log, and multi-seed confidence intervals are built and
measured. MVTec AD, segmentation, PatchCore, and the review-station UI are not.

```bash
python run_inspect.py            # ~14 min on CPU
python run_inspect.py --quick
python run_inspect.py --report-only

python extend.py                 # ~17 min: cascade, Grad-CAM, override log, 3-seed CIs
python budget_probe.py           # ~50 min: six training runs behind the correction below
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

## The part that failed, and then un-failed: the unseen-defect experiment

> **Corrected by the second pass.** This section originally concluded that the
> experiment did not work. That conclusion came from a **single training run**, and
> it does not survive three. The methodology critique in *Attempt 1* below still
> stands and is still the most useful thing here. *Attempt 2*'s verdict does not —
> see [the correction](#the-correction-attempt-2-was-under-powered-not-refuted)
> below and §1b of [docs/EXTENSIONS.md](docs/EXTENSIONS.md).

This was meant to be the project's other differentiator, and on the first pass it
appeared not to work **twice**. The report led with that rather than burying it —
which was the right instinct applied to a number that had not earned it.

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

Attempt 1's numbers are kept in `out/results_crack_holdout.json`.

### The correction: attempt 2 was under-powered, not refuted

The paragraph that used to sit here said the structural argument for the two-stage
architecture held as *reasoning* but that **this project provides no evidence for
it**. That was wrong, and the way it was wrong is worth more than the original
finding.

`python budget_probe.py` re-runs attempt 2 at three seeds under both training
budgets. At *this table's own budget* (1400 images, 14 epochs):

| | supervised: known → unseen | anomaly: known → unseen |
|---|---|---|
| 1400 imgs / 14 ep, 3 seeds | 0.888 → **0.740** (−0.147) | 0.863 → **0.951** (+0.088) |
| 1100 imgs / 12 ep, 3 seeds | 0.877 → **0.779** (−0.098) | 0.843 → **0.956** (+0.113) |

**The supervised head loses AUROC on a class it never saw and the anomaly head
gains it** — which is precisely the effect the table above reported as absent. The
0.958 in that table is a single draw from a distribution with **sd 0.096**, about
2.3σ high. The anomaly head wins on the held-out class in **6 of 6 runs** (smallest
gap +0.098; sign test one-sided p = 0.016).

Two things I am deliberately *not* upgrading on the strength of this:

- **The effect size is not pinned down.** With three seeds the gap's 95% interval at
  the larger budget is +0.210 ± 0.241, which includes zero. The sign test is
  distribution-free and is the claim I will defend; the magnitude is not.
- **The synthetic-data caveat below still applies.** The generator renders every
  defect as a local deviation from a smooth background, so this is a weaker test of
  novelty than real imagery. MVTec is still the right way to settle it.

The transferable lesson is cheap and general: **a single-seed model comparison was
strong enough to become this project's headline negative finding, and it was
wrong.** Nothing about it looked fragile — a clean AUROC on a clean holdout is
exactly what a defensible number looks like.

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

## Built in the second pass — see [docs/EXTENSIONS.md](docs/EXTENSIONS.md)

```bash
python extend.py          # ~17 min on CPU
python budget_probe.py    # ~50 min; six training runs
```

Four gaps this README named, and the run **overturned the README's own headline
finding** (above) and **refuted two arguments this README made for the cascade**:

- **Confidence intervals.** Every AUROC here was a point estimate from one run.
  Three seeds, 95% t-intervals, and the finding is the *spread*: unseen-class
  supervised AUROC has sd 0.096 at this project's training budget, which is what
  made the single-seed headline unreliable.
- **The cascade, actually wired.** Three verdicts, not two — and
  `FLAG_FOR_REVIEW` (stage 1 sure something is wrong, stage 2 unable to name it) is
  the disposition an unseen defect should get. A two-outcome system forces the
  operator to pick a wrong class to clear the screen, which fills the override log
  with garbage and poisons the retraining set.
  **But the throughput argument this README made for the cascade collapses.**
  Holding 99% stage-1 recall flags 93.9% of parts at line prevalence, so stage 2
  runs on nearly everything and the saving is **2.5%** — a rounding error against
  takt. The full recall-vs-cost-vs-escapes sweep is in EXTENSIONS.md. The honest
  conclusion is narrower than the one I set out to write: **right architecture,
  wrong model** — and that is the quantitative case for PatchCore and a pretrained
  backbone, rather than a preference for them.
- **Grad-CAM, and what it fails at.** The supervised head had zero localisation.
  It now has some, and *some* is the accurate word: the peak lands inside the true
  defect mask **3.3%** of the time against a 0.80% base rate. That is 4× chance and
  still wrong nineteen times in twenty, which makes it unusable as an operator aid —
  a localisation overlay that is usually wrong teaches operators to distrust the
  overlay and then the call it came with. The cause is the image-level training
  objective, not Grad-CAM: **an attribution method cannot manufacture spatial
  evidence the model never used.**
- **The operator-override log.** 1009 parts reviewed, operator-confirmed PPV 0.309
  — measured on the *flagged subset*, so it is an upper bound on line PPV rather
  than an estimate of it, and conflating the two is how a dashboard reports healthy
  precision on a line that is scrapping good parts. The log is retraining gold
  because of its sampling (labels on exactly the hard parts) and a trap for the
  same reason (censored to what stage 1 flagged).

## Completed in the third pass — see [docs/COMPLETION.md](docs/COMPLETION.md)

```bash
python fetch_mvtec.py     # ~10 min; real MVTec AD images, not redistributed
python complete.py        # ~110 min on CPU
```

- **MVTec AD, fetched.** The benchmark this project has been measured against the
  absence of since pass 1. 218 real images across a texture (`grid`)
  and an object (`hazelnut`) category, downsampled to the working resolution and
  gitignored. **These are the first numbers here measured on real photographs.**
- **PatchCore with a real ImageNet backbone**, and a coreset memory bank replacing
  the per-position Gaussian. On the isolated bimodal case a Gaussian scores
  **0.441 — below chance** — because it puts its mean
  in the gap between two legitimate appearances and calls that gap most normal.
  PatchCore scores 0.609.
- **A segmentation head.** Grad-CAM's peak landed inside the true defect 3.3% of
  the time; segmentation reaches **0.767**, a
  **23× improvement** on the same
  statistic. That settles the pass-2 diagnosis: the attribution method was not
  weak, the image-level objective never gave it spatial evidence to attribute.
- **Gauge R&R for the camera.** %GRR **15.5%
  (marginal)** — and %GRR alone would have been misleading. Mean
  kappa between stations is **0.32**:
  the variance decomposition looks tolerable while the stations disagree on
  borderline parts, which is why MSA-4 prescribes attribute agreement for a
  go/no-go gauge. Re-scoring the identical array instead of re-acquiring reports
  **0.0% repeatability** — a perfect gauge and a broken experiment.
- **A retraining loop with the censoring corrected.** The review log covers only
  what the screen flagged (defect rate 0.53
  reviewed vs 0.19 unreviewed).
  Inverse-propensity weighting takes recall
  0.767 → **0.867**
  against 0.767 for the naive
  arm, scored on a frozen golden set behind an **asymmetric** re-qualification
  gate — 1% recall drop allowed against 5% PPV, because a missed defect ships and
  a false reject costs a re-inspection.
- **An inspection service and a review station.** A wrongly-sized image is
  rejected with 422 rather than resized, because silently resizing means a
  miscalibrated camera produces confident nonsense instead of an alarm.
- **Eight seeds.** The unseen-defect gap is **+0.206 ±
  0.083**, positive in **8/8** runs
  (sign test p = 0.0039). **The interval now excludes zero**,
  which closes the caveat pass 2 left open: the effect size is pinned down, not
  just its sign.

### Three results that went against what I built

**The segmentation defences made it worse.** My module argues that BCE alone
converges to the degenerate all-background solution and that Dice plus positive
weighting prevent it. Plain BCE reaches IoU **0.494**; the defended
version reaches **0.395**. Precision collapses
0.814 → 0.555 while recall barely moves — a
positive weight of 50 over-predicts defect everywhere, **the mirror image of the
degenerate solution**, which the docstring warned about before setting the cap at
50 and walking into it. The degenerate risk is real and I did see it: at ~140
defective images and 6 epochs *both* arms collapsed to IoU 0.000. So the honest
statement is narrower — the defences matter when data or training is short and
cost accuracy when neither is.

**The pretrained backbone loses on synthetic data.** PatchCore with ImageNet
features scores 0.770 on known defects against this project's own
small CNN at **0.858** — the opposite of what the not-built list
predicted when it called the weak backbone "the main reason the absolute AUROCs
are modest".

**And on real photographs it reverses.** On MVTec, PatchCore scores
**0.987** on `hazelnut` (an object) against
0.780 for the own-CNN, and **0.561**
on `grid` (a texture) against 0.699. So the backbone
question has a conditional answer: **pretrained features are worth having when the
part looks like a photograph of an object, and worth nothing when the part is a
texture.** A casting surface is a texture. On synthetic data alone I would have
concluded the backbone was useless — which is exactly what the real data was for.

## What is NOT built

1. **MVTec is a subset at reduced resolution.** Two categories, ~110 images each,
   downsampled to 128×128 — so these numbers are not comparable to a paper's
   full-resolution result on all fifteen categories. The CDN resets roughly half
   of image requests from this network, which is what bounded the fetch.
2. **No pixel-level ground truth from MVTec.** The mirror used carries
   image-level defect labels; the per-pixel masks are not in it, so the
   segmentation head is still trained and scored on synthetic masks only.
3. **No container runtime.** `deploy/Dockerfile` is emitted and never built.
4. **The review station renders and does not write.** The disposition buttons
   build an in-page log; wiring them to a service needs a server.
5. **The simulated operator is treated as ground truth**, which is generous — a
   real operator is a measurement system with its own repeatability, and the
   gauge R&R here measures the camera rather than the human.
6. **Eight seeds, one architecture.** The interval excludes zero now, but every
   number is still one CNN and one PaDiM/PatchCore configuration.

## Layout

```
src/synth.py       textured casting generator, 4 variants x 4 defect classes, pixel masks
src/models.py      supervised CNN; PaDiM-style patch anomaly head fitted on normals only
src/economics.py   prevalence tables, cost-matrix threshold search, takt budget
src/cascade.py     three-verdict cascade, screen-threshold choice, Grad-CAM, override log
run_inspect.py     orchestration; writes docs/RESULTS.md
extend.py          second pass; writes docs/EXTENSIONS.md
budget_probe.py    seed x training-budget sweep; writes out/budget_probe.json
```
