"""Inspection service and the operator review station.

WHAT AN INSPECTION ENDPOINT OWES THE LINE, beyond returning a score:

  A VERDICT, NOT A NUMBER. The line does not act on 0.83. It acts on ACCEPT,
  REJECT or FLAG_FOR_REVIEW, and the third one is the whole reason this project
  argued for a cascade -- an unseen defect needs a disposition that is neither
  "pass it" nor a guessed class.

  A LATENCY BUDGET. An inspection station runs inside takt. A service that
  usually answers in 8 ms and occasionally in 400 ms has failed, because the line
  is paced by the worst case. p99 is reported, not the mean.

  REFUSAL ON MALFORMED INPUT. A wrongly-sized image is rejected rather than
  resized. Silently resizing is the tempting one line, and it means a
  miscalibrated camera produces confident nonsense instead of an alarm.

  A DECISION RECORD. Every inspection is logged with the model fingerprint that
  made it. Without that, a recall six months later cannot answer "which model
  passed this part", which in a regulated plant is the first question.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import pathlib
import time

import numpy as np


class InspectionService:
    """Anomaly screen + supervised classifier behind one verdict."""

    def __init__(self, model, anom, *, screen_threshold: float | None = None,
                 classify_threshold: float = 0.5, calib_x: np.ndarray | None = None,
                 target_recall: float = 0.99) -> None:
        self.model = model
        self.anom = anom
        self.classify_threshold = float(classify_threshold)
        self.log: list[dict] = []
        self._lat: list[float] = []

        if screen_threshold is None:
            if calib_x is None:
                raise ValueError("give screen_threshold or calib_x to derive one")
            # Calibrated on NORMAL parts only: the threshold is a quantile of the
            # healthy score distribution, so it carries a false-alarm rate by
            # construction rather than a recall promise it cannot keep without
            # defect labels at install time.
            s = self.anom.score_maps(self.model, calib_x).max(axis=(1, 2))
            screen_threshold = float(np.quantile(s, 0.98))
            self.calibration = {"n_normals": int(len(calib_x)),
                                "quantile": 0.98,
                                "implied_false_flag_rate": 0.02}
        else:
            self.calibration = {"given": True}
        self.screen_threshold = float(screen_threshold)
        self.fingerprint = self._fingerprint()

    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        for p in self.model.parameters():
            h.update(p.detach().numpy().tobytes())
        h.update(f"{self.screen_threshold:.6f}|{self.classify_threshold:.6f}".encode())
        return h.hexdigest()[:16]

    # -- input contract ---------------------------------------------------
    @staticmethod
    def _as_2d(img) -> np.ndarray:
        """Accept (H, W) or (1, H, W).

        `synth.to_arrays` hands back channel-first images, so a service that only
        accepts (H, W) rejects its own project's data. Normalising here rather
        than at each call site is what stops the next caller rediscovering it.
        """
        a = np.asarray(img, dtype=float)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return a

    def check(self, img: np.ndarray) -> list[str]:
        import synth
        problems = []
        a = self._as_2d(img)
        if a.ndim != 2:
            problems.append(f"expected a 2-D image, got shape {a.shape}")
        elif a.shape != (synth.SIZE, synth.SIZE):
            problems.append(
                f"expected {synth.SIZE}x{synth.SIZE}, got {a.shape[0]}x{a.shape[1]}; "
                "resizing here would hide a miscalibrated camera")
        if a.size and (np.nanmin(a) < -1e-6 or np.nanmax(a) > 1 + 1e-6):
            problems.append("pixel values outside [0, 1]")
        if a.size and not np.isfinite(a).all():
            problems.append("non-finite pixel values")
        return problems

    # -- inspection --------------------------------------------------------
    def inspect(self, img: np.ndarray, part_id: str | None = None) -> dict:
        t0 = time.perf_counter()
        problems = self.check(img)
        if problems:
            return {"ok": False, "problems": problems}

        import models as M
        x = self._as_2d(img).astype(np.float32)[None, None]
        amap = self.anom.score_maps(self.model, x)[0]
        a_score = float(amap.max())
        if a_score < self.screen_threshold:
            verdict, cls = "ACCEPT", None
        else:
            cls = float(M.supervised_scores(self.model, x)[0])
            verdict = ("REJECT_CLASSIFIED" if cls > self.classify_threshold
                       else "FLAG_FOR_REVIEW")
        ms = (time.perf_counter() - t0) * 1e3
        self._lat.append(ms)
        rec = {"ok": True, "part_id": part_id, "verdict": verdict,
               "anomaly_score": a_score, "screen_threshold": self.screen_threshold,
               "classifier_score": cls, "model": self.fingerprint,
               "latency_ms": ms}
        self.log.append({k: rec[k] for k in
                         ("part_id", "verdict", "anomaly_score", "model")})
        return rec

    def inspect_batch(self, imgs: np.ndarray) -> dict:
        t0 = time.perf_counter()
        out = [self.inspect(im, part_id=f"P{i:05d}") for i, im in enumerate(imgs)]
        secs = time.perf_counter() - t0
        counts: dict[str, int] = {}
        for r in out:
            counts[r.get("verdict", "REJECTED_INPUT")] = counts.get(
                r.get("verdict", "REJECTED_INPUT"), 0) + 1
        lat = np.array([r["latency_ms"] for r in out if r.get("ok")])
        return {"n": len(out), "counts": counts, "seconds": secs,
                "parts_per_second": len(out) / secs if secs else float("inf"),
                "p50_ms": float(np.percentile(lat, 50)) if len(lat) else None,
                "p99_ms": float(np.percentile(lat, 99)) if len(lat) else None}

    def heatmap(self, img: np.ndarray) -> np.ndarray:
        import models as M
        import synth
        x = self._as_2d(img).astype(np.float32)[None, None]
        return M.upsample_map(self.anom.score_maps(self.model, x), synth.SIZE)[0]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

Req = None          # bound on the first build_app() call; see below


def _request_model():
    """The Pydantic model, built at module scope.

    This file uses `from __future__ import annotations`, so every annotation is
    a string at runtime and FastAPI resolves a handler's parameter type by name
    against the MODULE globals. A class defined inside build_app is not there,
    and FastAPI does not raise -- it silently decides the unresolvable parameter
    must be a query parameter, and every POST returns 422 "field required".
    Exactly the same trap as ML-1's serving module in this portfolio.
    """
    from pydantic import BaseModel

    class Req(BaseModel):
        image: list[list[float]]
        part_id: str | None = None

    return Req


def build_app(svc: InspectionService):
    from fastapi import FastAPI, HTTPException

    global Req
    if Req is None:
        Req = _request_model()
    globals()["Req"] = Req

    app = FastAPI(title="visual inspection", version="1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": svc.fingerprint,
                "screen_threshold": svc.screen_threshold}

    @app.post("/inspect")
    def inspect(r: Req):        # noqa: F821 -- bound above at module scope
        out = svc.inspect(np.asarray(r.image, dtype=float), r.part_id)
        if not out["ok"]:
            raise HTTPException(422, {"problems": out["problems"]})
        return out

    @app.get("/log")
    def log():
        return {"n": len(svc.log), "entries": svc.log[-200:]}

    return app


# ---------------------------------------------------------------------------
# the review station
# ---------------------------------------------------------------------------

def _png_b64(arr: np.ndarray, cmap: str = "gray") -> str:
    from PIL import Image
    a = np.asarray(arr, dtype=float)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    a = (a - a.min()) / (float(np.ptp(a)) or 1.0)   # np.ptp: ndarray.ptp() went away in NumPy 2
    if cmap == "hot":
        r = np.clip(a * 3, 0, 1)
        g = np.clip(a * 3 - 1, 0, 1)
        b = np.clip(a * 3 - 2, 0, 1)
        rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
        im = Image.fromarray(rgb, "RGB")
    else:
        im = Image.fromarray((a * 255).astype(np.uint8), "L")
    buf = io.BytesIO()
    im.resize((256, 256), Image.NEAREST).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def write_review_station(path: pathlib.Path, svc: InspectionService,
                         imgs: np.ndarray, labels: np.ndarray,
                         masks: np.ndarray | None = None) -> dict:
    """A self-contained HTML review station.

    Self-contained on purpose -- one file, no server, no assets. An operator
    terminal on a shop floor is behind an air gap more often than not, and a
    review UI that needs a CDN is a review UI that does not open.

    What it shows per part, and each element is there because leaving it out
    changes the operator's answer:

      the part image        obviously
      the anomaly heat map  so the operator can see WHERE, not just whether
      the model's verdict   including FLAG_FOR_REVIEW, the honest third answer
      the score and threshold  so a borderline call reads as borderline
      disposition buttons   which write the override log the retraining
                            loop consumes
    """
    cards = []
    for i, im in enumerate(imgs):
        r = svc.inspect(im, part_id=f"P{i:05d}")
        if not r.get("ok"):
            raise ValueError(f"image {i} is not inspectable: {r.get('problems')}")
        hm = svc.heatmap(im)
        colour = {"ACCEPT": "#2f855a", "REJECT_CLASSIFIED": "#c53030",
                  "FLAG_FOR_REVIEW": "#b7791f"}.get(r["verdict"], "#4a5568")
        frac = min(1.0, r["anomaly_score"] / (2 * svc.screen_threshold))
        cards.append(f"""
      <div class="card" data-verdict="{r['verdict']}">
        <div class="imgs">
          <figure><img src="data:image/png;base64,{_png_b64(im)}"><figcaption>part</figcaption></figure>
          <figure><img src="data:image/png;base64,{_png_b64(hm, 'hot')}"><figcaption>anomaly</figcaption></figure>
        </div>
        <div class="verdict" style="background:{colour}">{r['verdict'].replace('_', ' ')}</div>
        <div class="meta">
          <div>part <b>{r['part_id']}</b> &middot; truth <b>{'defect' if labels[i] else 'good'}</b></div>
          <div class="bar"><span style="width:{frac * 100:.0f}%"></span></div>
          <div>score <b>{r['anomaly_score']:.2f}</b> &middot; threshold {svc.screen_threshold:.2f}</div>
        </div>
        <div class="btns">
          <button onclick="dispose('{r['part_id']}','CONFIRMED_DEFECT',this)">confirm defect</button>
          <button onclick="dispose('{r['part_id']}','FALSE_REJECT',this)">false reject</button>
          <button onclick="dispose('{r['part_id']}','UNCLASSIFIED',this)">cannot classify</button>
        </div>
      </div>""")

    html = f"""<!doctype html>
<meta charset="utf-8"><title>Review station</title>
<style>
 :root {{ --bg:#f7fafc; --fg:#1a202c; --card:#fff; --line:#e2e8f0; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --bg:#1a202c; --fg:#e2e8f0; --card:#2d3748; --line:#4a5568; }} }}
 body {{ margin:0; padding:24px; font:14px/1.5 system-ui,sans-serif;
        background:var(--bg); color:var(--fg); }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 .sub {{ opacity:.7; margin-bottom:20px; }}
 .grid {{ display:grid; gap:16px;
          grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:12px; }}
 .imgs {{ display:flex; gap:8px; }}
 figure {{ margin:0; flex:1; }}
 img {{ width:100%; border-radius:6px; display:block; image-rendering:pixelated; }}
 figcaption {{ font-size:11px; opacity:.6; text-align:center; margin-top:2px; }}
 .verdict {{ color:#fff; text-align:center; font-weight:600; letter-spacing:.4px;
             border-radius:6px; padding:5px; margin:10px 0 8px; font-size:12px; }}
 .meta {{ font-size:12px; opacity:.85; }}
 .bar {{ height:5px; background:var(--line); border-radius:3px; margin:6px 0; }}
 .bar span {{ display:block; height:100%; background:#3182ce; border-radius:3px; }}
 .btns {{ display:flex; gap:6px; margin-top:10px; }}
 button {{ flex:1; font-size:11px; padding:6px 4px; border:1px solid var(--line);
           background:transparent; color:inherit; border-radius:5px; cursor:pointer; }}
 button:hover {{ background:var(--line); }}
 button.done {{ background:#3182ce; color:#fff; border-color:#3182ce; }}
 #log {{ margin-top:24px; font-family:ui-monospace,monospace; font-size:12px;
         white-space:pre-wrap; opacity:.8; }}
</style>
<h1>Review station</h1>
<div class="sub">model <code>{svc.fingerprint}</code> &middot; screen threshold
 {svc.screen_threshold:.3f} &middot; {len(imgs)} parts queued</div>
<div class="grid">{''.join(cards)}</div>
<div id="log">override log (empty)</div>
<script>
const entries = [];
function dispose(part, d, btn) {{
  entries.push({{part, disposition: d, at: new Date().toISOString()}});
  [...btn.parentElement.children].forEach(b => b.classList.remove('done'));
  btn.classList.add('done');
  document.getElementById('log').textContent =
    'override log (' + entries.length + ' entries)\\n' +
    entries.map(e => e.at + '  ' + e.part + '  ' + e.disposition).join('\\n');
}}
</script>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size,
            "n_parts": int(len(imgs)), "self_contained": True}


def write_container(root: pathlib.Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "Dockerfile").write_text("""# NOT BUILT OR RUN -- emitted by src/inspect_service.py.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pillow scikit-learn fastapi uvicorn \\
    torch --index-url https://download.pytorch.org/whl/cpu
COPY src/ /app/src/
EXPOSE 8000
HEALTHCHECK --interval=20s CMD python -c \\
  "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
""", encoding="utf-8")
    return {"dockerfile": str(root / "Dockerfile"), "built": False,
            "note": "no container runtime in this environment"}
