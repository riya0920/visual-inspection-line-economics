"""Two complementary detectors. The comparison IS the project.

SUPERVISED  a small CNN trained on labelled defect classes. Strong on what it has
            seen, structurally blind to what it has not: it learned a boundary
            between "good" and "these four defects", and a fifth defect lands
            wherever the features happen to put it -- which is often on the good
            side, confidently.

ANOMALY     trained on NORMAL IMAGES ONLY (PaDiM-style: per-patch Gaussians over
            embeddings, scored by Mahalanobis distance). It has no concept of a
            defect class, so an unseen defect is exactly as detectable as a seen
            one. It pays for that with a worse operating point on the classes the
            supervised model knows.

Neither is the answer. The two-stage line architecture -- anomaly screen, then
supervised classify -- is what real deployments use, and the reason is in the
numbers: the anomaly head catches the novel thing and the supervised head tells
the quality engineer which bin to put it in.

SUBSTITUTION NOTE: real PaDiM/PatchCore use a large ImageNet-pretrained backbone,
because a pretrained embedding is what makes patch Mahalanobis work well. No
pretrained weights are available offline here, so the embedding comes from the
supervised CNN's early layers, trained on this data. That is a WEAKER embedding
and it also means the "anomaly" method is not fully independent of the labels --
it saw them during backbone training. Both facts are stated in the README; the
numbers below should be read as a demonstration of the architecture, not as a
benchmark of PaDiM.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

SEED = 20260819


class InspectCNN(nn.Module):
    """Small conv net. `features()` exposes the early activation map that the
    anomaly head consumes, so both detectors share one forward pass on the line --
    which is what makes the two-stage architecture affordable at takt time."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 64x64
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 32x32
        )
        self.body = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 16x16
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def features(self, x):
        return self.stem(x)          # (N, 48, 32, 32)

    def forward(self, x):
        f = self.body(self.stem(x))
        # GLOBAL MAX POOLING, concatenated with average pooling.
        #
        # The first version used AdaptiveAvgPool2d alone and the model sat at
        # AUROC 0.529 -- chance. The reason is the same one that governs the
        # anomaly score: a defect is LOCAL. A pore covers ~0.35% of the image and
        # a crack ~1%, so averaging the final 16x16 feature map dilutes the
        # evidence by two orders of magnitude and the classifier is left
        # discriminating on global brightness, which is exactly what the lighting
        # augmentation randomises.
        #
        # Max pooling asks "is there anywhere on this part that looks defective",
        # which is the actual question. The average branch is kept because some
        # defects (shrinkage) are diffuse and a max over a noisy map is unstable
        # on its own.
        z = torch.cat([torch.amax(f, dim=(2, 3)), f.mean(dim=(2, 3))], dim=1)
        return self.classifier(z)


def train_supervised(x: np.ndarray, y: np.ndarray, epochs: int = 12,
                     batch: int = 64, lr: float = 2e-3, verbose: bool = False,
                     sample_weight: np.ndarray | None = None):
    """`sample_weight` carries the inverse-propensity correction.

    A review log is CENSORED -- it holds only the parts the screen flagged -- so a
    reviewed part flagged with probability p stands for 1/p parts like itself.
    Passing those weights here is what stops a retrained model inheriting the
    screen's blind spots. It composes with the class weighting below rather than
    replacing it: the two correct different biases (which parts got LOOKED at, and
    how many of each CLASS there are), and dropping either reintroduces its own.
    """
    torch.manual_seed(SEED)
    model = InspectCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Class weighting rather than resampling: the training set is built at a
    # convenient defect rate, and the REAL prevalence is applied later at
    # evaluation time (see economics.py). Mixing the two is how a model ends up
    # calibrated to a prevalence that does not exist on any line.
    w = torch.tensor([1.0, float((y == 0).sum() / max(1, (y == 1).sum()))],
                     dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w, reduction="none")
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    sw = (torch.ones(len(xt)) if sample_weight is None
          else torch.tensor(np.asarray(sample_weight, dtype=np.float32)))
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for b in range(0, len(xt), batch):
            j = perm[b:b + batch]
            opt.zero_grad()
            per = lossf(model(xt[j]), yt[j])
            loss = (per * sw[j]).sum() / sw[j].sum().clamp_min(1e-9)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(j)
        if verbose:
            print(f"    epoch {ep+1:>2}/{epochs} loss {tot/len(xt):.4f}", flush=True)
    return model, time.perf_counter() - t0


@torch.no_grad()
def supervised_scores(model, x: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, len(x), batch):
        p = torch.softmax(model(torch.from_numpy(x[b:b + batch])), dim=1)[:, 1]
        out.append(p.numpy())
    return np.concatenate(out)


class PatchAnomaly:
    """PaDiM-style: a Gaussian per spatial patch position over embeddings.

    Fitted on NORMAL IMAGES ONLY. The score for a patch is its Mahalanobis
    distance from that position's normal distribution, which gives a spatial
    anomaly map for free -- and the map is the localisation the quality engineer
    actually needs. An image-level score of 0.9 tells an operator nothing; a
    heat-map over the part tells them where to look.
    """

    def __init__(self, n_components: int = 40, shrinkage: float = 0.05):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.mu = None
        self.inv = None
        self.proj = None

    @torch.no_grad()
    def _embed(self, model, x: np.ndarray, batch: int = 128) -> np.ndarray:
        model.eval()
        out = []
        for b in range(0, len(x), batch):
            f = model.features(torch.from_numpy(x[b:b + batch]))
            out.append(f.numpy())
        return np.concatenate(out)            # (N, C, H, W)

    def fit(self, model, x_normal: np.ndarray) -> "PatchAnomaly":
        f = self._embed(model, x_normal)
        n, c, h, w = f.shape
        # Random channel projection, as in PaDiM: the full channel covariance per
        # position is both expensive and badly conditioned with few samples.
        rng = np.random.default_rng(SEED)
        k = min(self.n_components, c)
        self.proj = rng.choice(c, size=k, replace=False)
        f = f[:, self.proj, :, :].reshape(n, k, h * w)

        self.mu = f.mean(axis=0)                                   # (k, HW)
        self.inv = np.empty((h * w, k, k), dtype=np.float64)
        for p in range(h * w):
            d = f[:, :, p] - self.mu[:, p]
            cov = np.cov(d, rowvar=False)
            cov = cov + np.eye(k) * (np.trace(cov) / k) * self.shrinkage
            self.inv[p] = np.linalg.pinv(cov)
        self.shape = (h, w)
        return self

    def score_maps(self, model, x: np.ndarray) -> np.ndarray:
        f = self._embed(model, x)
        n, c, h, w = f.shape
        k = len(self.proj)
        f = f[:, self.proj, :, :].reshape(n, k, h * w)
        maps = np.empty((n, h * w), dtype=np.float32)
        for p in range(h * w):
            d = f[:, :, p] - self.mu[:, p]
            maps[:, p] = np.einsum("ij,jk,ik->i", d, self.inv[p], d)
        return np.sqrt(np.maximum(maps, 0)).reshape(n, h, w)

    def image_scores(self, model, x: np.ndarray) -> np.ndarray:
        """Image score = MAX patch distance, not the mean.

        A defect is local. Averaging a 32x32 map dilutes a 4-patch anomaly by a
        factor of 250 and makes a crack indistinguishable from noise -- which is
        the same mistake as scoring a bearing by RMS instead of by kurtosis."""
        return self.score_maps(model, x).max(axis=(1, 2))


def upsample_map(maps: np.ndarray, size: int) -> np.ndarray:
    """Bilinear upsample anomaly maps to image resolution for pixel scoring."""
    t = torch.from_numpy(maps)[:, None, :, :]
    up = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                         align_corners=False)
    return up[:, 0].numpy()


@torch.no_grad()
def gradcam(model, x: np.ndarray) -> np.ndarray:
    """Cheap localisation for the SUPERVISED head: the channel-mean activation of
    the last conv block, upsampled. Not true Grad-CAM (no gradients here) and
    labelled as such -- it is an activation map, which is weaker and honest."""
    model.eval()
    f = model.body(model.stem(torch.from_numpy(x)))
    a = f.mean(dim=1).numpy()
    return a
