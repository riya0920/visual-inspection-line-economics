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

import hashlib
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

# Six categories rather than two, chosen to span the axis that pass 3's result
# turned on. PatchCore's pretrained backbone LOST to a hand-built detector on
# synthetic textures and WON on real photographs, and two categories cannot say
# whether that is about "real vs synthetic" or about "texture vs object".
#
#   textures  grid, carpet          repeating structure, defects break the pattern
#   objects   bottle, hazelnut      a shape on a background, defects break the shape
#   hard      screw, transistor     screw is the category everybody's numbers are
#                                   worst on; transistor's defects are structural
#                                   (misplaced, bent lead) rather than surface marks
CATEGORIES = ("grid", "carpet", "bottle", "hazelnut", "screw", "transistor")
TEXTURES = ("grid", "carpet")
MAX_PER_SPLIT = 100          # per category, per split; the CDN resets ~half of requests
SIZE = 128                   # matches src/synth.SIZE


def _get(url: str, tries: int = 3, timeout: int = 15) -> bytes:
    """GET with retries. The HF CDN resets connections here fairly often."""
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return f.read()
        except Exception as e:                                # noqa: BLE001
            last = e
            # Short waits and a short timeout, deliberately. This CDN either
            # answers quickly or hangs, so a 60-second timeout spends a minute
            # per dead request and a thousand-image fetch never finishes. Fail
            # fast, cache what arrives, and run the whole thing again -- the
            # cache is what makes repeated passes cheap.
            time.sleep(0.4 * (a + 1))
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
    if "--from-cache" in sys.argv:
        build_from_cache(out_path)
        return
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

    # RESUMABLE, because it has to be. The mirror's CDN resets a large fraction
    # of requests from this network, so a thousand-image fetch does not complete
    # in one attempt and never will -- the first run of the six-category version
    # died part-way through the second category. Each decoded image is cached to
    # a small .npy under data/MVTEC/cache/, keyed by the remote path, so a rerun
    # costs nothing for what already arrived and the fetch converges over
    # however many attempts it takes.
    cache = DEST / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # A sidecar index, so a PARTIAL cache is usable. The cache key is a hash of
    # the remote path and carries no category or split, so without this a run
    # that is interrupted -- which is the normal case here -- leaves a directory
    # of anonymous arrays and no way to build a dataset from them. The index is
    # written as each image lands, so it is always at least as complete as the
    # cache.
    index_path = cache / "index.json"
    index = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {}

    def cached(rel: str, cat: str, split: str, defect: str):
        key = hashlib.sha1(rel.encode()).hexdigest()[:20]
        f = cache / f"{key}.npy"
        if f.exists():
            try:
                arr = np.load(f)
                index.setdefault(key, [cat, split, defect])
                return arr, True
            except (ValueError, OSError):
                f.unlink(missing_ok=True)
        blob = _get(f"{RES}/{rel}")
        arr = _load_png(blob)
        np.save(f, arr)
        index[key] = [cat, split, defect]
        # Written via a temp file and replaced atomically. A resumable fetcher
        # whose index can be truncated by an interrupt is not resumable, and the
        # interrupt is the normal case here -- a plain write_text leaves a
        # half-written JSON file if the process dies mid-write, which discards
        # every image fetched so far.
        tmp = index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index), encoding="utf-8")
        tmp.replace(index_path)
        return arr, False

    xs, cats, splits, defects = [], [], [], []
    hits = misses = failed = 0
    for (cat, split), items in sorted(wanted.items()):
        print(f"  {cat}/{split}: {len(items)} images", flush=True)
        for i, it in enumerate(items):
            rel = str(it["path"]).replace("\\\\", "/").lstrip("./")
            try:
                arr, was_cached = cached(rel, cat, split, it["defect"])
                xs.append(arr)
                hits += was_cached
                misses += not was_cached
            except Exception as e:                            # noqa: BLE001
                failed += 1
                print(f"    skip {rel}: {type(e).__name__}", flush=True)
                continue
            cats.append(cat)
            splits.append(split)
            defects.append(it["defect"])
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(items)}  "
                      f"(cache {hits}, fetched {misses}, failed {failed})",
                      flush=True)

    if not xs:
        raise SystemExit("nothing fetched")
    print(f"\n{hits} from cache, {misses} fetched, {failed} still failing")
    x = np.stack(xs).astype(np.float32)
    np.savez_compressed(
        out_path, x=x, category=np.array(cats), split=np.array(splits),
        defect=np.array(defects), size=SIZE)
    print(f"\nwrote {out_path} -- {len(x)} images at {SIZE}x{SIZE}, "
          f"{out_path.stat().st_size / 1e6:.1f} MB")
    print("source: MVTec AD (CC BY-NC-SA 4.0), Bergmann et al. CVPR 2019, "
          "via the Voxel51/mvtec-ad mirror. Not redistributed.")



def build_from_cache(out_path=None) -> dict:
    """Assemble the .npz from whatever is already cached.

    Exists because the fetch is interrupted more often than it completes, and a
    partially fetched subset is still a usable one -- four categories at sixty
    images each answers a question that two categories at a hundred cannot.
    Writing the dataset only at the end of a full pass throws that away.
    """
    cache = DEST / "cache"
    index_path = cache / "index.json"
    out_path = pathlib.Path(out_path or (DEST / "mvtec_subset.npz"))
    if not index_path.exists():
        print("no cache index; run the fetcher first")
        return {"images": 0}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    xs, cats, splits, defects = [], [], [], []
    missing = 0
    for key, (cat, split, defect) in sorted(index.items()):
        f = cache / f"{key}.npy"
        if not f.exists():
            missing += 1
            continue
        try:
            xs.append(np.load(f))
        except (ValueError, OSError):
            missing += 1
            continue
        cats.append(cat)
        splits.append(split)
        defects.append(defect)
    if not xs:
        print("cache index has no readable arrays")
        return {"images": 0}

    # Merge whatever an earlier pass already wrote. The pass-3 subset (grid and
    # hazelnut) was fetched before the cache existed, so it lives only in an
    # .npz -- and rebuilding from the cache alone would silently DROP the two
    # categories the project has been reporting on since pass 3. Duplicates are
    # dropped on the image bytes rather than on a filename, because the two
    # sources do not share one.
    prior = DEST / "mvtec_subset_pass3.npz"
    n_prior = 0
    if prior.exists():
        z = np.load(prior, allow_pickle=True)
        seen = {a.tobytes() for a in xs}
        for i in range(len(z["x"])):
            a = z["x"][i].astype(np.float32)
            if a.tobytes() in seen:
                continue
            seen.add(a.tobytes())
            xs.append(a)
            cats.append(str(z["category"][i]))
            splits.append(str(z["split"][i]))
            defects.append(str(z["defect"][i]))
            n_prior += 1
        print(f"  merged {n_prior} images from {prior.name}")

    x = np.stack(xs).astype(np.float32)
    np.savez_compressed(out_path, x=x, category=np.array(cats),
                        split=np.array(splits), defect=np.array(defects),
                        size=SIZE)
    per = {}
    for c, sp in zip(cats, splits):
        per[f"{c}/{sp}"] = per.get(f"{c}/{sp}", 0) + 1
    print(f"wrote {out_path} -- {len(x)} images "
          f"({len(x) - n_prior} from cache, {n_prior} merged, "
          f"{missing} index entries unreadable)")
    for k in sorted(per):
        print(f"  {k}: {per[k]}")
    return {"images": len(x), "by_split": per, "missing": missing,
            "merged_from_prior": n_prior}

if __name__ == "__main__":
    main()
