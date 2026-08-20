"""Synthetic casting-surface images with pixel-accurate defect masks.

DATA HONESTY, first: these are generated, not photographed. They are not MVTec AD
and not the Kaggle casting set, and no AUROC here is comparable to a published
number on either. What the generator buys is (a) a pixel-accurate ground-truth
mask for every defect, which is what makes pixel-level scoring possible at all,
and (b) control over prevalence, defect type, product variant and lighting, which
is what the economics and transfer analyses need.

Four defect classes, chosen because they fail differently:

  pore        a dark round void. Easy: high contrast, compact, common.
  inclusion   a bright hard particle. Easy but opposite polarity, so a detector
              that learned "defects are dark" fails on it.
  crack       a thin bright/dark line, 1-2 px wide. HARD: almost no area, so it
              barely moves an image-level statistic, and it is the one that
              matters most structurally.
  shrinkage   a diffuse irregular cluster. Hard for a different reason: low
              contrast and no sharp edge, so it looks like texture.

Four product variants (A-D) with different surface texture statistics, so the
"new product variant" transfer test is a real distribution shift rather than a
relabelling: train on A/B/C, test on D.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

SIZE = 128
DEFECT_TYPES = ("pore", "inclusion", "crack", "shrinkage")
VARIANTS = ("A", "B", "C", "D")

# Per-variant surface statistics: (roughness scale, contrast, base grey).
VARIANT_TEXTURE = {
    "A": (3.0, 0.10, 0.52),
    "B": (5.0, 0.13, 0.47),
    "C": (2.0, 0.08, 0.57),
    "D": (7.0, 0.17, 0.43),   # the held-out variant: rougher and darker
}


@dataclass
class Sample:
    image: np.ndarray          # (SIZE, SIZE) float32 in [0,1]
    mask: np.ndarray           # (SIZE, SIZE) bool, True where defective
    label: int                 # 0 = good, 1 = defective
    defect_type: str | None
    variant: str


def _texture(rng: np.random.Generator, variant: str) -> np.ndarray:
    scale, contrast, base = VARIANT_TEXTURE[variant]
    n = rng.standard_normal((SIZE, SIZE))
    t = ndimage.gaussian_filter(n, sigma=scale)
    t = t / (t.std() + 1e-9)
    # A second, finer octave so the surface is not a single smooth blob.
    f = ndimage.gaussian_filter(rng.standard_normal((SIZE, SIZE)), sigma=1.0)
    t = 0.8 * t + 0.2 * (f / (f.std() + 1e-9))
    return np.clip(base + contrast * t, 0, 1).astype(np.float32)


def _lighting(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """A smooth illumination gradient plus a global gain.

    This is the factory-floor variable: somebody replaces a light fixture and the
    gradient changes. It is modelled separately from the texture so the robustness
    test can turn it up without touching the defects.
    """
    if strength <= 0:
        return img
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    ang = rng.uniform(0, 2 * np.pi)
    grad = np.cos(ang) * xx + np.sin(ang) * yy
    gain = 1.0 + strength * rng.uniform(-0.5, 0.5)
    return np.clip((img + strength * 0.5 * (grad - 0.5)) * gain, 0, 1).astype(np.float32)


def _jitter(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, px: float):
    """Camera position jitter: a sub-pixel shift and a small rotation."""
    if px <= 0:
        return img, mask
    dy, dx = rng.uniform(-px, px, 2)
    ang = rng.uniform(-2.0, 2.0)
    img = ndimage.shift(img, (dy, dx), order=1, mode="reflect")
    img = ndimage.rotate(img, ang, reshape=False, order=1, mode="reflect")
    m = ndimage.shift(mask.astype(np.float32), (dy, dx), order=0, mode="constant")
    m = ndimage.rotate(m, ang, reshape=False, order=0, mode="constant")
    return img.astype(np.float32), m > 0.5


def _add_defect(img: np.ndarray, kind: str, rng: np.random.Generator,
                severity: float) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    out = img.copy()
    cy, cx = rng.integers(20, SIZE - 20, 2)

    if kind == "pore":
        r = rng.uniform(3.0, 7.0)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        blob = np.clip(1.0 - d / r, 0, 1) ** 0.7
        mask = blob > 0.25
        out = out - severity * 0.45 * blob

    elif kind == "inclusion":
        r = rng.uniform(2.0, 4.5)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        blob = np.clip(1.0 - d / r, 0, 1)
        mask = blob > 0.3
        out = out + severity * 0.5 * blob

    elif kind == "crack":
        # A short random walk, dilated to 1-2 px. Tiny area on purpose.
        n = int(rng.integers(30, 70))
        ang = rng.uniform(0, 2 * np.pi)
        y, x = float(cy), float(cx)
        pts = []
        for _ in range(n):
            ang += rng.normal(0, 0.28)
            y += np.sin(ang)
            x += np.cos(ang)
            if not (1 <= y < SIZE - 1 and 1 <= x < SIZE - 1):
                break
            pts.append((int(y), int(x)))
        for (py, px_) in pts:
            mask[py, px_] = True
        # iterations must be >= 1: scipy treats iterations=0 as "dilate until
        # convergence", which fills the entire image. The first version drew
        # iterations from {0,1} and produced "cracks" with a 7,700-pixel mask --
        # i.e. the whole part -- which would have made the hardest defect class
        # trivially detectable and the pixel-level AUROC meaningless.
        mask = ndimage.binary_dilation(mask, iterations=int(rng.integers(1, 3)))
        soft = ndimage.gaussian_filter(mask.astype(np.float32), 0.6)
        out = out - severity * 0.55 * soft

    elif kind == "shrinkage":
        blob = ndimage.gaussian_filter(rng.standard_normal((SIZE, SIZE)), 4.0)
        blob = blob / (blob.std() + 1e-9)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        r = rng.uniform(9.0, 16.0)
        window = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        field = blob * window
        mask = field > (field.max() * 0.45)
        out = out - severity * 0.22 * np.clip(field, 0, None) / (field.max() + 1e-9)

    return np.clip(out, 0, 1).astype(np.float32), mask


def make(n: int, defect_rate: float, rng: np.random.Generator,
         variants=("A", "B", "C"), defect_types=DEFECT_TYPES,
         lighting: float = 0.06, jitter_px: float = 0.0,
         severity_range: tuple[float, float] = (0.55, 1.0)) -> list[Sample]:
    out: list[Sample] = []
    for _ in range(n):
        v = str(rng.choice(list(variants)))
        img = _texture(rng, v)
        if rng.random() < defect_rate:
            kind = str(rng.choice(list(defect_types)))
            sev = rng.uniform(*severity_range)
            img, mask = _add_defect(img, kind, rng, sev)
            label = 1
        else:
            kind, mask, label = None, np.zeros((SIZE, SIZE), dtype=bool), 0
        img = _lighting(img, rng, lighting)
        img, mask = _jitter(img, mask, rng, jitter_px)
        out.append(Sample(img, mask, label, kind, v))
    return out


def to_arrays(samples: list[Sample]):
    x = np.stack([s.image for s in samples])[:, None, :, :].astype(np.float32)
    y = np.array([s.label for s in samples], dtype=np.int64)
    m = np.stack([s.mask for s in samples])
    return x, y, m
