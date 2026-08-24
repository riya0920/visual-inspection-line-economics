"""PatchCore: a coreset memory bank, and why it is a different thing from PaDiM.

WHAT THE FIRST BUILD HAD. `PatchAnomaly` is PaDiM-style: for each patch position
it fits a multivariate Gaussian over the training normals and scores a test patch
by Mahalanobis distance. That is a *parametric* model of normality, and it carries
a specific assumption -- that the normal patches at a given position are unimodal
and roughly Gaussian in feature space.

WHERE THAT ASSUMPTION BREAKS, and it breaks in manufacturing more than most
places: a part with two legitimate appearances at one position. Two supplier
finishes, a stamped logo present on some variants, a fixture that seats the part
two ways. The normal distribution at that patch is bimodal, a Gaussian fitted to
it puts its mean in the empty space between the two modes, and BOTH legitimate
appearances score as anomalies while the midpoint -- which never occurs -- scores
as perfectly normal. That is not a tuning problem; it is the model being wrong
about the shape of normality.

WHAT PATCHCORE DOES INSTEAD. It keeps the normal patch features themselves in a
memory bank and scores a test patch by distance to its nearest neighbour. No
distributional assumption at all: multi-modal normality is represented by simply
having members of both modes in the bank.

THE COST, and the reason the coreset exists. Storing every patch of every
training image is enormous -- 200 images x 16x16 positions x 64 dims is 3.3M
floats, and every test patch must search all of it. PatchCore's contribution is
that a greedy k-center subset of ~1-10% of the bank preserves the coverage that
matters, because what a nearest-neighbour detector needs is not density but
COVERAGE: every region of normal-feature space must have a member near it.

  greedy k-center: repeatedly add the point FURTHEST from everything already
  chosen. It explicitly optimises the worst-case distance from any training
  point to the bank, which is exactly the quantity that decides whether a normal
  test patch gets a low score.

Random subsampling optimises average density instead, and the two differ most
precisely on rare-but-legitimate appearances -- the bimodal case above. Both are
implemented here so the difference is measured rather than asserted.

THE PRETRAINED BACKBONE, and why PatchCore insists on a MID-level layer. Real
PatchCore embeds patches with ImageNet-pretrained features, and the layer choice
is not incidental:

  early layers  (conv1, layer1) are edges and colour blobs. Too generic -- a
                scratch and a legitimate texture edge look alike.
  late layers   (layer4) are ImageNet CLASS semantics: "dog", "car". They have
                discarded exactly the local texture detail a surface defect
                consists of, and they are tuned to a label set with no
                relationship to castings.
  mid layers    (layer2 + layer3) are local texture and part structure. Generic
                enough to transfer to a domain ImageNet never saw, specific
                enough that a defect changes them.

So this uses layer2+layer3 of a ResNet-18, adaptively pooled to a common grid and
concatenated. Frozen -- no fine-tuning, which is the point: the memory bank is
the model, and there is nothing to train.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# pretrained feature extraction
# ---------------------------------------------------------------------------

def have_backbone() -> bool:
    try:
        import torchvision  # noqa: F401
        return True
    except Exception:
        return False


class ResNetPatchFeatures:
    """Frozen ImageNet ResNet-18, layer2+layer3, as a patch embedder.

    Two details that matter and are easy to get wrong:

    1. GREYSCALE TO 3 CHANNELS. The project's images are single-channel; ResNet
       expects 3. Repeating the channel is correct here -- the alternative,
       averaging the RGB weights into one channel, changes the filters the network
       was trained with. Then the ImageNet mean/std normalisation is applied,
       because the frozen weights were fitted under it and skipping it shifts
       every activation.

    2. LOCALLY-AWARE POOLING. PatchCore average-pools each feature vector over its
       3x3 neighbourhood before storing it. Without it the memory bank is a bag of
       single-pixel descriptors and loses the local structure that distinguishes a
       scratch from a speck. It also makes the bank robust to a one-pixel
       misalignment, which matters when parts are not perfectly fixtured.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, grid: int = 16, neighbourhood: int = 3) -> None:
        import torch
        import torchvision

        self.torch = torch
        net = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                        net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.grid = grid
        self.nb = neighbourhood

    def __call__(self, x: np.ndarray, batch: int = 32) -> np.ndarray:
        """(n, H, W) or (n, 1, H, W) in [0,1] -> (n, grid, grid, c).

        Both layouts are accepted because `synth.to_arrays` returns
        channel-first (n, 1, H, W) while the masks and any externally loaded
        images are (n, H, W). Guessing wrong here does not raise -- it silently
        feeds a 5-D tensor into a 4-D op or transposes the image -- so the shape
        is normalised once, explicitly.
        """
        torch = self.torch
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 4 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim != 3:
            raise ValueError(f"expected (n,H,W) or (n,1,H,W), got {a.shape}")
        out = []
        mean = torch.tensor(self.MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.STD).view(1, 3, 1, 1)
        with torch.no_grad():
            for i in range(0, len(a), batch):
                t = torch.from_numpy(a[i:i + batch]).float().unsqueeze(1)
                t = t.repeat(1, 3, 1, 1)
                t = (t - mean) / std
                h = self.stem(t)
                f2 = self.layer2(h)
                f3 = self.layer3(f2)
                g = self.grid
                # NOT `a` -- that is the input array, and reusing the name here
                # clobbered it on the second batch.
                p2 = torch.nn.functional.adaptive_avg_pool2d(f2, g)
                p3 = torch.nn.functional.adaptive_avg_pool2d(f3, g)
                f = torch.cat([p2, p3], dim=1)
                if self.nb > 1:
                    f = torch.nn.functional.avg_pool2d(
                        f, self.nb, stride=1, padding=self.nb // 2)
                out.append(f.permute(0, 2, 3, 1).numpy())
        return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# coreset selection
# ---------------------------------------------------------------------------

def greedy_kcenter(x: np.ndarray, n_select: int, seed: int = 0,
                   ) -> tuple[np.ndarray, dict]:
    """Greedy k-center (farthest-point) subset selection.

    Maintains, for every point, the distance to the nearest already-selected
    point, and repeatedly selects the argmax. O(n * k) distance evaluations with
    an incremental update, rather than the O(n^2 * k) of recomputing.

    Returns the selected indices and the achieved COVERAGE RADIUS -- the maximum
    distance from any training point to the bank. That number is the whole
    justification for the method, so it is returned rather than left implicit: it
    bounds how far a normal patch can be from its nearest neighbour, which bounds
    the false-alarm score.
    """
    n = len(x)
    n_select = int(min(max(n_select, 1), n))
    rng = np.random.default_rng(seed)
    # Squared norms precomputed once, so each iteration is a single matvec
    # rather than an (n, dim) temporary. The naive `norm(x - x[j], axis=1)`
    # allocates 30 MB per iteration at this size and spends most of its time in
    # the allocator; expanding ||a-b||^2 = |a|^2 - 2a.b + |b|^2 makes the inner
    # loop a BLAS call.
    xn = np.einsum("ij,ij->i", x, x)
    first = int(rng.integers(n))
    d = np.maximum(xn - 2.0 * (x @ x[first]) + xn[first], 0.0)
    chosen = [first]
    radii = [float(np.sqrt(d.max()))]
    for _ in range(n_select - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        np.minimum(d, np.maximum(xn - 2.0 * (x @ x[nxt]) + xn[nxt], 0.0), out=d)
        radii.append(float(np.sqrt(d.max())))
    d = np.sqrt(d)
    return np.array(chosen), {
        "coverage_radius": float(d.max()),
        "radius_curve": radii[::max(1, len(radii) // 40)],
        "n_selected": len(chosen), "n_total": n,
    }


def random_subset(x: np.ndarray, n_select: int, seed: int = 0,
                  ) -> tuple[np.ndarray, dict]:
    """Uniform random subset, for comparison. Same size, different objective."""
    n = len(x)
    n_select = int(min(max(n_select, 1), n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_select, replace=False)
    # Same squared-norm expansion as greedy_kcenter. The naive loop here was the
    # real bottleneck in the first run -- k-center got optimised and this one did
    # not, so the "random" comparison arm quietly cost ten minutes while the arm
    # it was being compared against took four seconds.
    xn = np.einsum("ij,ij->i", x, x)
    sel = x[idx]
    d2 = np.full(n, np.inf)
    block = 256
    for i in range(0, len(sel), block):
        c = sel[i:i + block]
        cn = np.einsum("ij,ij->i", c, c)
        dd = xn[:, None] - 2.0 * (x @ c.T) + cn[None, :]
        np.minimum(d2, dd.min(axis=1), out=d2)
    return idx, {"coverage_radius": float(np.sqrt(max(d2.max(), 0.0))),
                 "n_selected": len(idx), "n_total": n}


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------

class PatchCore:
    """Nearest-neighbour patch anomaly detection over a coreset memory bank."""

    def __init__(self, rate: float = 0.02, selector: str = "kcenter",
                 n_neighbours: int = 1, seed: int = 0) -> None:
        self.rate = float(rate)
        self.selector = selector
        self.k = int(n_neighbours)
        self.seed = seed
        self.bank: np.ndarray | None = None
        self.info: dict = {}

    def fit(self, patches: np.ndarray, pool_cap: int = 20000) -> "PatchCore":
        """`patches` is (n_images, h, w, c) of NORMAL images only.

        `pool_cap` bounds the candidate pool before k-center runs, and it is not
        an optimisation detail -- greedy k-center is O(pool x selected) distance
        evaluations, so 100k patches at a 2% rate is 100k x 2k = 200M distance
        computations in 384 dimensions. The published method subsamples the
        patch pool first for exactly this reason.

        The subsample is uniform and therefore unbiased with respect to the
        coverage objective: k-center still picks the farthest-first points, just
        from a random view of the space rather than all of it. What it costs is a
        slightly larger coverage radius, and that radius is measured and
        reported, so the cost is visible rather than assumed away.
        """
        flat = patches.reshape(-1, patches.shape[-1]).astype(np.float32)
        full_n = len(flat)
        pooled = flat
        if full_n > pool_cap:
            rng = np.random.default_rng(self.seed)
            pooled = flat[rng.choice(full_n, size=pool_cap, replace=False)]
        n_sel = max(1, int(round(full_n * self.rate)))
        n_sel = min(n_sel, len(pooled))
        pick = greedy_kcenter if self.selector == "kcenter" else random_subset
        idx, info = pick(pooled, n_sel, seed=self.seed)
        self.bank = pooled[idx]
        self.info = {**info, "selector": self.selector, "rate": self.rate,
                     "bank_bytes": int(self.bank.nbytes),
                     "full_bank_bytes": int(flat.nbytes),
                     "pool_size": int(len(pooled)),
                     "patches_available": int(full_n),
                     "pool_subsampled": bool(full_n > pool_cap)}
        return self

    def score_maps(self, patches: np.ndarray, batch: int = 4096) -> np.ndarray:
        """Per-patch distance to the k-th nearest bank member."""
        if self.bank is None:
            raise RuntimeError("fit() first")
        n, h, w, c = patches.shape
        flat = patches.reshape(-1, c).astype(np.float32)
        out = np.empty(len(flat), dtype=np.float32)
        # ||a-b||^2 = |a|^2 - 2 a.b + |b|^2, so one matmul per block.
        bn = (self.bank ** 2).sum(1)
        for i in range(0, len(flat), batch):
            blk = flat[i:i + batch]
            d2 = (blk ** 2).sum(1)[:, None] - 2 * blk @ self.bank.T + bn[None, :]
            np.maximum(d2, 0, out=d2)
            if self.k == 1:
                out[i:i + batch] = np.sqrt(d2.min(1))
            else:
                part = np.partition(d2, self.k - 1, axis=1)[:, :self.k]
                out[i:i + batch] = np.sqrt(part.mean(1))
        return out.reshape(n, h, w)

    def image_scores(self, patches: np.ndarray) -> np.ndarray:
        """Max over patches -- an image is as anomalous as its worst patch.

        Max rather than mean, and this is the same lesson the first pass learned
        when global average pooling washed the defects out: a defect occupies
        under 1% of the image, so any averaging reduction buries it under the 99%
        that is fine.
        """
        return self.score_maps(patches).max(axis=(1, 2))


# ---------------------------------------------------------------------------
# the bimodal stress test
# ---------------------------------------------------------------------------

def make_bimodal_normal(n: int, dim: int = 8, sep: float = 6.0,
                        minority: float = 0.25, seed: int = 0) -> np.ndarray:
    """Two legitimate appearances at one patch position.

    The failure mode PatchCore exists for, isolated from everything else. A
    Gaussian fitted to this puts its mean in the gap between the modes, so both
    real appearances look anomalous and the midpoint -- which never occurs --
    looks perfectly normal.
    """
    rng = np.random.default_rng(seed)
    n_min = int(n * minority)
    a = rng.normal(0.0, 1.0, (n - n_min, dim))
    b = rng.normal(0.0, 1.0, (n_min, dim))
    b[:, 0] += sep
    return np.concatenate([a, b]).astype(np.float32)


def mahalanobis_scores(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """PaDiM's scoring, for the head-to-head."""
    mu = train.mean(0)
    cov = np.cov(train, rowvar=False) + 1e-3 * np.eye(train.shape[1])
    inv = np.linalg.inv(cov)
    d = test - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))
