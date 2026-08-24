"""Fetch a bounded subset of MVTec AD, the benchmark this project has been
measured against the absence of since pass 1.

WHY IT MATTERS HERE. Every AUROC in RESULTS.md comes from `src/synth.py`, a
procedural casting-surface generator I wrote. The README has said from the first
pass that this is the project's largest caveat, and it named the specific
consequence: the unseen-defect experiment could not be made to work, because
`synth.py` renders every defect as a local deviation from a smooth textured
background, so a small conv net learns "local deviation" and generalises across
defect classes for free. That is a property of my generator, not of CNNs.

MVTec AD is the standard benchmark for exactly this question. Real photographs of
real manufactured parts, real defects, per-pixel ground-truth masks.

WHAT IS FETCHED, and why it is a subset rather than the dataset. Full MVTec AD is
~5 GB across 15 categories at 1024x1024. This takes TWO categories and caps the
image count, then immediately downsamples to the project's working resolution and
stores a single .npz. The PNGs are not kept.

The two categories are chosen to be a hard pair rather than a convenient one:

  grid      a TEXTURE category. Closest to what synth.py generates, so it is the
            fair comparison -- if the pipeline fails here it was never going to
            work anywhere.
  hazelnut  an OBJECT category with a defined shape and a background. This is
            what synth.py cannot produce at all: the model must learn that a
            defect is a deviation from an object, not from a texture field, and
            "anomalous" now includes the object being rotated or misplaced.

PROVENANCE AND LICENCE. MVTec AD is published by MVTec Software GmbH under
CC BY-NC-SA 4.0 -- free for non-commercial and research use with attribution:

    Bergmann, Fauser, Sattlegger, Steger. "MVTec AD -- A Comprehensive Real-World
    Dataset for Unsupervised Anomaly Detection." CVPR 2019.

Fetched here from the Voxel51/mvtec-ad mirror on Hugging Face. Nothing is
redistributed in this repository: data/MVTEC/ is gitignored.

    python fetch_mvtec.py
    python fetch_mvtec.py --check
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DEST = ROOT / "data" / "MVTEC"
REPO = "Voxel51/mvtec-ad"
API = f"https://huggingface.co/api/datasets/{REPO}"
RAW = f"https://huggingface.co/datasets/{REPO}/raw/main"
RES = f"https://huggingface.co/datasets/{REPO}/resolve/main"

CATEGORIES = ("grid", "hazelnut")
MAX_PER_SPLIT = 70           # per category, per split; the CDN resets ~half of requests
SIZE = 128                   # matches src/synth.SIZE


def _get(url: str, tries: int = 5, timeout: int = 60) -> bytes:
    """GET with retries. The HF CDN resets connections here fairly often."""
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return f.read()
        except Exception as e:                                # noqa: BLE001
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"{url}: {type(last).__name__}: {last}")


def _samples() -> list[dict]:
    """The label index. Cached, because it is a few MB of JSON."""
    cache = DEST / "samples.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["samples"]
    blob = _get(f"{RAW}/samples.json", timeout=180)
    DEST.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(blob)
    return json.loads(blob)["samples"]


def _load_png(blob: bytes) -> np.ndarray:
    """Decode, greyscale, resize to SIZE, scale to [0, 1].

    Greyscale on purpose: `src/models.py` takes a single channel, and matching the
    existing input contract is what makes this a test of the PIPELINE rather than
    a new project that happens to share a directory.
    """
    from PIL import Image

    im = Image.open(io.BytesIO(blob)).convert("L").resize(
        (SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    out_path = DEST / "mvtec_subset.npz"
    if "--check" in sys.argv:
        if out_path.exists():
            z = np.load(out_path, allow_pickle=True)
            print(f"present: {out_path.name}, {len(z['x'])} images, "
                  f"categories {sorted(set(z['category'].tolist()))}")
        else:
            print("not fetched")
        return

    print("fetching the label index ...", flush=True)
    samples = _samples()

    # The Voxel51 mirror flattens the MVTec tree, so category / split / defect
    # come from each sample's tags and fields rather than from the path.
    def field(s, *names):
        for n in names:
            v = s.get(n)
            if isinstance(v, dict):
                v = v.get("label") or v.get("classifications")
            if isinstance(v, str):
                return v
        return None

    wanted: dict[tuple, list] = {}
    for s in samples:
        cat = field(s, "category", "object", "class")
        split = field(s, "split", "partition")
        defect = field(s, "defect", "defect_type", "label", "ground_truth")
        fp = s.get("filepath") or s.get("relative_path") or ""
        if cat is None:
            # fall back to parsing whatever path-like string is present
            parts = [p for p in str(fp).replace("\\\\", "/").split("/") if p]
            if len(parts) >= 3:
                cat, split, defect = parts[-4] if len(parts) >= 4 else parts[0], \
                    parts[-3], parts[-2]
        if cat not in CATEGORIES or split not in ("train", "test"):
            continue
        key = (cat, split)
        if len(wanted.setdefault(key, [])) < MAX_PER_SPLIT:
            wanted[key].append({"path": fp, "defect": defect or "good"})

    if not wanted:
        print("could not map samples.json onto categories/splits.\n"
              "Dumping one sample so the field names can be read:")
        print(json.dumps(samples[0], indent=1)[:1500])
        raise SystemExit(1)

    xs, cats, splits, defects = [], [], [], []
    for (cat, split), items in sorted(wanted.items()):
        print(f"  {cat}/{split}: {len(items)} images", flush=True)
        for i, it in enumerate(items):
            rel = str(it["path"]).replace("\\\\", "/").lstrip("./")
            try:
                blob = _get(f"{RES}/{rel}")
                xs.append(_load_png(blob))
            except Exception as e:                            # noqa: BLE001
                print(f"    skip {rel}: {type(e).__name__}", flush=True)
                continue
            cats.append(cat)
            splits.append(split)
            defects.append(it["defect"])
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(items)}", flush=True)

    x = np.stack(xs).astype(np.float32)
    np.savez_compressed(
        out_path, x=x, category=np.array(cats), split=np.array(splits),
        defect=np.array(defects), size=SIZE)
    print(f"\nwrote {out_path} -- {len(x)} images at {SIZE}x{SIZE}, "
          f"{out_path.stat().st_size / 1e6:.1f} MB")
    print("source: MVTec AD (CC BY-NC-SA 4.0), Bergmann et al. CVPR 2019, "
          "via the Voxel51/mvtec-ad mirror. Not redistributed.")


if __name__ == "__main__":
    main()
