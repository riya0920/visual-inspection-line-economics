"""A supervised segmentation head, and the class-imbalance problem that defines it.

WHY THIS EXISTS. The Grad-CAM result in docs/EXTENSIONS.md was that the
supervised head's attribution lands inside the true defect 3.3% of the time
against a 0.80% base rate: 4x chance, and wrong nineteen times in twenty. The
diagnosis there was that an attribution method cannot manufacture spatial
evidence the model never used, because the model is trained on an IMAGE-level
label. The fix named was segmentation, and this is it.

THE PROBLEM THAT MAKES THIS NOT A ROUTINE UNET. Defect pixels are **0.8% of the
image**. Under pixel-wise cross-entropy, a model that predicts "background"
everywhere scores 99.2% pixel accuracy and has a lower loss than most models that
actually try. Gradient descent finds that solution immediately and sits there.
This is the single most common way a first segmentation attempt fails, and it
fails *quietly* -- the loss curve looks healthy.

Three defences, and they address different halves of the problem:

  DICE LOSS         optimises overlap between prediction and mask, normalised by
                    their sizes. An empty prediction scores Dice 0 regardless of
                    how much background it got right, so the degenerate solution
                    is no longer a good one. This is the important one.

  POSITIVE WEIGHTING in BCE, at roughly the inverse class frequency, so a missed
                    defect pixel costs about as much in aggregate as the
                    background does.

  COMBINED LOSS     BCE + Dice rather than either alone. Dice has unstable
                    gradients when the prediction is near-empty (its denominator
                    goes small), and BCE is what gets it off the ground; BCE
                    alone converges to the degenerate solution. They fail in
                    opposite regimes, which is why the sum works.

AND THE METRIC HAS TO CHANGE TOO. Pixel accuracy is meaningless at 0.8%
prevalence -- the degenerate model scores 99.2%. IoU and Dice on the positive
class are reported instead, plus the pixel-level AUROC that the anomaly head is
already scored on, so the two paths are comparable.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


class TinyUNet(nn.Module):
    """A small encoder-decoder with skip connections.

    Skip connections are not decoration here: the encoder downsamples to capture
    context, and a defect is a handful of pixels. Without the skips the decoder is
    upsampling from a coarse grid and cannot recover the boundary -- which is
    exactly the failure Grad-CAM already demonstrated, reproduced inside the
    architecture.
    """

    def __init__(self, ch: int = 16) -> None:
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

        self.e1, self.e2, self.e3 = block(1, ch), block(ch, ch * 2), block(ch * 2, ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.d2 = block(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.d1 = block(ch * 2, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1).squeeze(1)          # logits, (n, H, W)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0):
    """Soft Dice on the positive class.

    `eps` in both numerator and denominator: an image with NO defect has an
    all-zero target, and without smoothing its Dice is 0/0. Smoothing makes a
    correct empty prediction on an empty target score 1 rather than NaN, which is
    the behaviour you want -- the model should not be punished for correctly
    finding nothing.
    """
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2)) + eps
    den = p.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps
    return 1 - (num / den).mean()


def _chw(x: np.ndarray) -> np.ndarray:
    """Normalise to (n, 1, H, W).

    `synth.to_arrays` returns images channel-first as (n, 1, H, W) while masks
    come back as (n, H, W). Unsqueezing blindly turns the first into a 5-D
    tensor, which conv2d rejects -- loudly, which is the good case. The bad case
    is the opposite mistake on the mask, where a stray axis broadcasts instead of
    raising.
    """
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 4 and a.shape[1] == 1:
        return a
    if a.ndim == 3:
        return a[:, None]
    raise ValueError(f"expected (n,H,W) or (n,1,H,W), got {a.shape}")


def _hw(m: np.ndarray) -> np.ndarray:
    """Normalise a mask to (n, H, W)."""
    a = np.asarray(m, dtype=np.float32)
    if a.ndim == 4 and a.shape[1] == 1:
        return a[:, 0]
    if a.ndim == 3:
        return a
    raise ValueError(f"expected a mask of (n,H,W), got {a.shape}")


def fit_segmenter(x: np.ndarray, m: np.ndarray, *, epochs: int = 20,
                  batch: int = 32, lr: float = 2e-3, ch: int = 16,
                  use_dice: bool = True, pos_weight: float | None = None,
                  val_frac: float = 0.15, seed: int = 0, verbose: bool = False,
                  ) -> tuple[TinyUNet, dict]:
    """Train on (images, binary masks). Only defective images are useful here."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, m = _chw(x), _hw(m)
    n = len(x)
    perm = rng.permutation(n)
    nv = max(1, int(n * val_frac))
    vi, ti = perm[:nv], perm[nv:]

    xt = torch.from_numpy(x[ti]).float()
    mt = torch.from_numpy(m[ti]).float()
    xv = torch.from_numpy(x[vi]).float()
    mv = torch.from_numpy(m[vi]).float()

    prevalence = float(mt.mean())
    if pos_weight is None:
        # Inverse class frequency, capped. Uncapped it reaches ~125 at 0.8%
        # prevalence and the model over-predicts defect everywhere -- trading one
        # degenerate solution for its mirror image.
        pos_weight = float(min((1 - prevalence) / max(prevalence, 1e-6), 50.0))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    model = TinyUNet(ch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for i in range(0, len(ti), batch):
            b = slice(i, i + batch)
            opt.zero_grad()
            lg = model(xt[b])
            loss = bce(lg, mt[b]) + (dice_loss(lg, mt[b]) if use_dice else 0.0)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sc = iou(torch.sigmoid(model(xv)).numpy(), mv.numpy())["iou"]
        if sc > best:
            best = sc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    seg ep{ep:02d} val IoU {sc:.3f}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"prevalence": prevalence, "pos_weight": pos_weight,
                   "use_dice": use_dice, "val_iou": best,
                   "seconds": time.perf_counter() - t0,
                   "params": sum(p.numel() for p in model.parameters())}


def predict(model: TinyUNet, x: np.ndarray, batch: int = 64) -> np.ndarray:
    model.eval()
    x = _chw(x)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            t = torch.from_numpy(x[i:i + batch]).float()
            out.append(torch.sigmoid(model(t)).numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# metrics that survive 0.8% prevalence
# ---------------------------------------------------------------------------

def iou(prob: np.ndarray, target: np.ndarray, thr: float = 0.5) -> dict:
    p = prob >= thr
    t = target > 0.5
    inter = float((p & t).sum())
    union = float((p | t).sum())
    return {
        "iou": inter / union if union else 1.0,
        "dice": 2 * inter / (p.sum() + t.sum()) if (p.sum() + t.sum()) else 1.0,
        "precision": inter / p.sum() if p.sum() else 0.0,
        "recall": inter / t.sum() if t.sum() else 1.0,
        # Reported ONLY to show why it must not be used: the degenerate
        # all-background model scores ~99.2% here.
        "pixel_accuracy": float((p == t).mean()),
    }


def degenerate_baseline(target: np.ndarray) -> dict:
    """Score the "predict background everywhere" model.

    Printed next to the real model in the report, because a segmentation number
    without this baseline is unreadable at this prevalence.
    """
    return iou(np.zeros_like(target, dtype=float), target)


def peak_inside_mask(prob: np.ndarray, target: np.ndarray) -> float:
    """Same statistic Grad-CAM was scored on, so the two are directly comparable."""
    hits = 0
    for p, t in zip(prob, target):
        if t.sum() == 0:
            continue
        idx = np.unravel_index(int(np.argmax(p)), p.shape)
        hits += int(t[idx] > 0.5)
    n = int(sum(1 for t in target if t.sum() > 0))
    return hits / max(n, 1)
