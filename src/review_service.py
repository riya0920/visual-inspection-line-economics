"""Pass 5: the review station writes.

The not-built list said the station renders and does not write -- the
disposition buttons build an in-page log that dies with the tab. That is the
whole point of the station gone: the log is what the retraining loop is
supposed to eat.

Writing it down turns out to be the easy half. The hard half is that a
disposition is NOT a label, and three things sit between them:

  * a disposition is stated RELATIVE TO A VERDICT. "False reject" means the
    model rejected and the operator disagrees. On a part the model accepted the
    same word means nothing, and a store that accepts it silently has recorded
    a good-part label nobody intended. Refused here, with the verdict that was
    on screen recorded alongside so the mapping stays auditable later.

  * a disposition is a MEASUREMENT, made by an instrument with its own
    repeatability. This project's gauge R&R measured the camera; the operator
    was assumed to be ground truth, which item 6 of the not-built list has
    always admitted is generous. Two operators on one part is the cheapest
    possible measurement of the human, so the store keeps every operator's
    answer rather than letting the last writer win, and reports agreement.

  * a label with two operators behind it is not worth the same as a label with
    one, and a contested label may be worth less than nothing.
    labels_for_retraining() therefore emits weights, and the caller can demand
    consensus.

No authentication: `operator` is a claim, not an identity. Stated in LIMITS
rather than implied by the field name.

NOT AN IMPORT from SE-2 or ML-3, both of which have write-path services in this
portfolio. The pattern transfers; the code does not, because a cross-project
import is what makes two systems impossible to deploy separately. A test greps.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The three the station's buttons emit. Kept as a literal rather than imported
# from cascade.DISPOSITIONS: the store is the contract with the UI, and it
# should fail loudly if the two ever drift apart rather than track silently.
DISPOSITIONS = ("CONFIRMED_DEFECT", "FALSE_REJECT", "UNCLASSIFIED")

# Which verdicts each disposition is meaningful against. FALSE_REJECT against
# ACCEPT is the incoherent one: there was no reject to be false.
COHERENT = {
    "CONFIRMED_DEFECT": {"REJECT_CLASSIFIED", "FLAG_FOR_REVIEW"},
    "FALSE_REJECT": {"REJECT_CLASSIFIED", "FLAG_FOR_REVIEW"},
    "UNCLASSIFIED": {"REJECT_CLASSIFIED", "FLAG_FOR_REVIEW", "ACCEPT"},
}

# disposition -> the label it asserts about the PART. A verdict is about the
# model and goes stale when the model is retrained; a label is about the part
# and does not. That asymmetry is why the store keeps both.
LABEL_OF = {"CONFIRMED_DEFECT": 1, "FALSE_REJECT": 0, "UNCLASSIFIED": None}

LIMITS = (
    "No authentication: `operator` is a claim typed by the client, not an "
    "identity. Anything downstream that treats it as one is wrong.",
    "One process, one SQLite file. No replication and no backup.",
    "Agreement is measured only on parts more than one operator happened to "
    "open. Nothing here routes a part to a second operator on purpose, so the "
    "agreement figure is on a self-selected subset.",
    "A superseded disposition is kept but the retraining view uses only the "
    "latest per operator. An operator who changes their mind twice leaves a "
    "trail that nothing reads.",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS part (
  part_id      TEXT PRIMARY KEY,
  verdict      TEXT NOT NULL,
  score        REAL NOT NULL,
  threshold    REAL NOT NULL,
  fingerprint  TEXT NOT NULL,
  queued_ts    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS disposition (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  part_id      TEXT NOT NULL,
  operator     TEXT NOT NULL,
  disposition  TEXT NOT NULL,
  note         TEXT NOT NULL DEFAULT '',
  verdict_seen TEXT NOT NULL,
  ts           REAL NOT NULL,
  superseded   INTEGER NOT NULL DEFAULT 0,
  idem_key     TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_idem ON disposition(idem_key)
  WHERE idem_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_part ON disposition(part_id);
"""


class ReviewStore:
    """Thread-local connections.

    Not a style choice. A sqlite3.Connection made on one thread and used from
    another raises, and every server in this portfolio is threaded -- this is
    the same bug SE-2's credential store shipped with once, so it is written
    the safe way here from the start.
    """

    def __init__(self, path: pathlib.Path | str):
        self.path = str(path)
        pathlib.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10.0)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return c

    # -- queue ---------------------------------------------------------------

    def queue(self, rows, fingerprint: str, threshold: float,
              now: float | None = None) -> int:
        """Put inspected parts on the review queue.

        Re-queuing a part_id updates its verdict rather than duplicating it: a
        second inspection of the same part is a newer opinion, not a second
        part.
        """
        now = time.time() if now is None else now
        n = 0
        for r in rows:
            self.conn.execute(
                "INSERT INTO part(part_id,verdict,score,threshold,fingerprint,"
                "queued_ts) VALUES(?,?,?,?,?,?) ON CONFLICT(part_id) DO UPDATE "
                "SET verdict=excluded.verdict, score=excluded.score, "
                "threshold=excluded.threshold, fingerprint=excluded.fingerprint,"
                " queued_ts=excluded.queued_ts",
                (r["part_id"], r["verdict"], float(r["anomaly_score"]),
                 float(threshold), fingerprint, now))
            n += 1
        self.conn.commit()
        return n

    def pending(self, operator: str | None = None, limit: int = 200) -> list[dict]:
        """Parts still wanting a disposition.

        With an operator named, "still wanting" means from THAT operator: a
        part one person has already dispositioned is exactly the part worth
        showing a second person, because it is the only way the human ever gets
        measured. Without one, it means nobody has touched it.
        """
        if operator is None:
            q = ("SELECT p.* FROM part p WHERE NOT EXISTS (SELECT 1 FROM "
                 "disposition d WHERE d.part_id=p.part_id AND d.superseded=0) "
                 "ORDER BY p.score DESC LIMIT ?")
            args = (limit,)
        else:
            q = ("SELECT p.* FROM part p WHERE NOT EXISTS (SELECT 1 FROM "
                 "disposition d WHERE d.part_id=p.part_id AND d.operator=? AND "
                 "d.superseded=0) ORDER BY p.score DESC LIMIT ?")
            args = (operator, limit)
        return [dict(r) for r in self.conn.execute(q, args)]

    # -- writing -------------------------------------------------------------

    def dispose(self, part_id: str, operator: str, disposition: str, *,
                note: str = "", idem_key: str | None = None,
                now: float | None = None) -> dict:
        if disposition not in DISPOSITIONS:
            raise ValueError(f"unknown disposition {disposition!r}")
        if not operator.strip():
            raise ValueError("an operator is required: an unattributed "
                             "disposition cannot be audited or measured")
        row = self.conn.execute("SELECT * FROM part WHERE part_id=?",
                                (part_id,)).fetchone()
        if row is None:
            raise KeyError(part_id)
        verdict = row["verdict"]
        if verdict not in COHERENT[disposition]:
            raise ValueError(
                f"{disposition} is not a coherent answer to {verdict}: "
                f"there was no reject for the operator to call false")

        if idem_key is not None:
            prior = self.conn.execute(
                "SELECT * FROM disposition WHERE idem_key=?",
                (idem_key,)).fetchone()
            if prior is not None:
                # A double-clicked button and a retried POST are the same
                # request. Replaying it must not enrol a second opinion from
                # one person, which would then show up as agreement.
                return {**dict(prior), "replayed": True}

        now = time.time() if now is None else now
        prev = self.conn.execute(
            "SELECT id, disposition FROM disposition WHERE part_id=? AND "
            "operator=? AND superseded=0", (part_id, operator)).fetchone()
        if prev is not None:
            self.conn.execute("UPDATE disposition SET superseded=1 WHERE id=?",
                              (prev["id"],))
        cur = self.conn.execute(
            "INSERT INTO disposition(part_id,operator,disposition,note,"
            "verdict_seen,ts,idem_key) VALUES(?,?,?,?,?,?,?)",
            (part_id, operator, disposition, note, verdict, now, idem_key))
        self.conn.commit()
        return {"id": int(cur.lastrowid), "part_id": part_id,
                "operator": operator, "disposition": disposition,
                "verdict_seen": verdict, "ts": now, "replayed": False,
                "superseded_prior": None if prev is None else prev["disposition"]}

    def dispositions(self, part_id: str, include_superseded: bool = False):
        q = "SELECT * FROM disposition WHERE part_id=?"
        if not include_superseded:
            q += " AND superseded=0"
        return [dict(r) for r in self.conn.execute(q + " ORDER BY ts", (part_id,))]

    # -- reading it back as labels -------------------------------------------

    def consensus(self, part_id: str) -> dict:
        rows = self.dispositions(part_id)
        if not rows:
            return {"part_id": part_id, "n": 0, "label": None,
                    "agreed": None, "votes": [],
                    "why": "nobody has looked at it"}
        votes = [r["disposition"] for r in rows]
        labels = [LABEL_OF[v] for v in votes]
        named = [x for x in labels if x is not None]
        agreed = len(set(votes)) == 1
        if not named:
            return {"part_id": part_id, "n": len(rows), "label": None,
                    "agreed": agreed, "votes": votes, "n_naming": 0,
                    "why": "everyone who looked declined to classify it"}
        if len(set(named)) == 1:
            # UNCLASSIFIED is an abstention, not a dissent -- one operator
            # saying "I cannot tell" alongside one saying "defect" is not a
            # contradiction, and treating it as one throws away the only
            # opinion anybody actually held.
            return {"part_id": part_id, "n": len(rows), "label": named[0],
                    "agreed": agreed, "votes": votes, "n_naming": len(named),
                    "why": "unanimous among those who named a label"}
        ones = sum(named)
        if 2 * ones == len(named):
            return {"part_id": part_id, "n": len(rows), "label": None,
                    "agreed": False, "votes": votes, "n_naming": len(named),
                    "why": "tied: the operators contradict each other"}
        maj = 1 if 2 * ones > len(named) else 0
        return {"part_id": part_id, "n": len(rows), "label": maj,
                "agreed": False, "votes": votes, "n_naming": len(named),
                "why": f"majority {max(ones, len(named) - ones)}/{len(named)}"}

    def labels_for_retraining(self, require_consensus: bool = False) -> list[dict]:
        """What the loop is allowed to eat.

        Weight is 1 for an uncontested label and 0.5 for one carried by a
        majority over a dissent. Not a tuned number -- the point is only that a
        contested label is not evidence of the same strength as an agreed one,
        and a loop that cannot express that has no way to use the second
        operator it paid for.
        """
        ids = [r["part_id"] for r in
               self.conn.execute("SELECT DISTINCT part_id FROM disposition "
                                 "WHERE superseded=0")]
        out = []
        for pid in ids:
            c = self.consensus(pid)
            if c["label"] is None:
                continue
            if require_consensus and not c["agreed"]:
                continue
            p = self.conn.execute("SELECT * FROM part WHERE part_id=?",
                                  (pid,)).fetchone()
            out.append({"part_id": pid, "label": int(c["label"]),
                        "weight": 1.0 if c["agreed"] else 0.5,
                        "n_operators": c["n"], "votes": c["votes"],
                        "model_verdict": p["verdict"],
                        "model_score": p["score"]})
        out.sort(key=lambda r: r["part_id"])
        return out

    # -- measuring the operator ----------------------------------------------

    def agreement(self) -> dict:
        """Attribute agreement between operators, on parts more than one saw.

        This is the number item 6 of the not-built list has always been about.
        It is measurable only because the store keeps both answers instead of
        letting the second write overwrite the first.
        """
        rows = self.conn.execute(
            "SELECT part_id FROM disposition WHERE superseded=0 "
            "GROUP BY part_id HAVING COUNT(DISTINCT operator) > 1")
        multi = [r["part_id"] for r in rows]
        exact = binary = n_bin = 0
        for pid in multi:
            c = self.consensus(pid)
            exact += bool(c["agreed"])
            named = [LABEL_OF[v] for v in c["votes"] if LABEL_OF[v] is not None]
            if len(named) > 1:
                n_bin += 1
                binary += len(set(named)) == 1
        return {"n_parts_multiply_reviewed": len(multi),
                "exact_agreement": exact / len(multi) if multi else None,
                "n_binary_comparable": n_bin,
                "defect_or_not_agreement": binary / n_bin if n_bin else None}

    def by_operator(self) -> dict:
        out: dict[str, dict] = {}
        for r in self.conn.execute(
                "SELECT operator, disposition, COUNT(*) n FROM disposition "
                "WHERE superseded=0 GROUP BY operator, disposition"):
            out.setdefault(r["operator"], {})[r["disposition"]] = int(r["n"])
        for op, d in out.items():
            tot = sum(d.values())
            d["n"] = tot
            d["unclassified_rate"] = d.get("UNCLASSIFIED", 0) / tot
        return out


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_handler(store: ReviewStore, page_html: str = ""):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):        # quiet under pytest
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, page_html.encode(),
                                  "text/html; charset=utf-8")
            if path == "/api/pending":
                op = None
                if "operator=" in self.path:
                    op = self.path.split("operator=")[1].split("&")[0]
                return self._json(200, store.pending(op))
            if path == "/api/labels":
                req = "consensus=1" in self.path
                return self._json(200, store.labels_for_retraining(req))
            if path == "/api/agreement":
                return self._json(200, {"agreement": store.agreement(),
                                        "by_operator": store.by_operator()})
            return self._json(404, {"error": "no such route"})

        def do_POST(self):
            if self.path.split("?")[0] != "/api/dispose":
                return self._json(404, {"error": "no such route"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "body is not JSON"})
            missing = [k for k in ("part_id", "operator", "disposition")
                       if not str(body.get(k, "")).strip()]
            if missing:
                return self._json(400, {"error": f"missing: {', '.join(missing)}"})
            try:
                out = store.dispose(
                    body["part_id"], body["operator"], body["disposition"],
                    note=body.get("note", ""), idem_key=body.get("idem_key"))
            except KeyError:
                return self._json(404, {"error": "no such part on the queue"})
            except ValueError as e:
                # A refused disposition is the client's mistake, not the
                # server falling over. 400, with the reason, because the
                # operator needs to read it.
                return self._json(400, {"error": str(e)})
            return self._json(200, out)
    return H


def serve(store: ReviewStore, page_html: str = "", host: str = "127.0.0.1",
          port: int = 0) -> dict:
    srv = ThreadingHTTPServer((host, port), make_handler(store, page_html))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return {"server": srv, "thread": t,
            "url": f"http://{host}:{srv.server_address[1]}"}
