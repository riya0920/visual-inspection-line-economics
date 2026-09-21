"""Pass 5: the review station writes.

The tests that carry weight are the ones about the gap between a DISPOSITION
and a LABEL: a disposition is stated relative to a verdict, it is made by an
instrument with its own repeatability, and a replayed POST is one opinion
rather than two. Everything else here is plumbing.
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import review_service as RS      # noqa: E402


def _rows(n=3):
    verdicts = ["REJECT_CLASSIFIED", "FLAG_FOR_REVIEW", "ACCEPT"]
    return [{"part_id": f"P{i:05d}", "verdict": verdicts[i % 3],
             "anomaly_score": 4.0 - i} for i in range(n)]


@pytest.fixture
def store(tmp_path):
    s = RS.ReviewStore(tmp_path / "r.db")
    s.queue(_rows(), "fp-v1", 1.5)
    return s


# --- the queue ---------------------------------------------------------------

def test_requeuing_a_part_updates_it_rather_than_duplicating(store):
    store.queue([{"part_id": "P00000", "verdict": "ACCEPT",
                  "anomaly_score": 0.1}], "fp-v2", 1.5)
    rows = list(store.conn.execute("SELECT * FROM part WHERE part_id='P00000'"))
    assert len(rows) == 1
    assert rows[0]["verdict"] == "ACCEPT" and rows[0]["fingerprint"] == "fp-v2"


def test_pending_without_an_operator_means_nobody_has_looked(store):
    assert len(store.pending()) == 3
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    assert [p["part_id"] for p in store.pending()] == ["P00001", "P00002"]


def test_pending_for_an_operator_still_offers_what_someone_else_did(store):
    """A part one person dispositioned is exactly the part worth showing a
    second person -- it is the only way the human ever gets measured."""
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    assert "P00000" in [p["part_id"] for p in store.pending("OP-B")]
    assert "P00000" not in [p["part_id"] for p in store.pending("OP-A")]


# --- a disposition is stated relative to a verdict ---------------------------

def test_false_reject_against_accept_is_refused(store):
    """There was no reject for the operator to call false. A store that takes
    it has quietly recorded a good-part label nobody intended."""
    with pytest.raises(ValueError, match="coherent"):
        store.dispose("P00002", "OP-A", "FALSE_REJECT")


def test_confirmed_defect_against_accept_is_refused_too(store):
    with pytest.raises(ValueError, match="coherent"):
        store.dispose("P00002", "OP-A", "CONFIRMED_DEFECT")


def test_unclassified_is_coherent_against_anything(store):
    assert store.dispose("P00002", "OP-A", "UNCLASSIFIED")["id"] > 0


def test_the_verdict_on_screen_is_recorded_with_the_disposition(store):
    """So the disposition -> label mapping stays auditable after the model is
    retrained and the verdict changes."""
    out = store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    assert out["verdict_seen"] == "REJECT_CLASSIFIED"
    store.queue([{"part_id": "P00000", "verdict": "ACCEPT",
                  "anomaly_score": 0.1}], "fp-v2", 1.5)
    assert store.dispositions("P00000")[0]["verdict_seen"] == "REJECT_CLASSIFIED"


def test_an_unknown_disposition_and_an_unknown_part_are_refused(store):
    with pytest.raises(ValueError, match="unknown disposition"):
        store.dispose("P00000", "OP-A", "LOOKS_FINE_TO_ME")
    with pytest.raises(KeyError):
        store.dispose("NOPE", "OP-A", "UNCLASSIFIED")


def test_an_operator_is_required(store):
    with pytest.raises(ValueError, match="operator is required"):
        store.dispose("P00000", "   ", "CONFIRMED_DEFECT")


# --- one opinion, not two ----------------------------------------------------

def test_a_replayed_request_does_not_enrol_a_second_opinion(store):
    """A double-clicked button is one opinion. Two would show up downstream as
    an operator agreeing with themselves."""
    a = store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT", idem_key="k1")
    b = store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT", idem_key="k1")
    assert a["replayed"] is False and b["replayed"] is True
    assert b["id"] == a["id"]
    assert len(store.dispositions("P00000")) == 1


def test_changing_your_mind_supersedes_rather_than_appends(store):
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    out = store.dispose("P00000", "OP-A", "FALSE_REJECT")
    assert out["superseded_prior"] == "CONFIRMED_DEFECT"
    live = store.dispositions("P00000")
    assert len(live) == 1 and live[0]["disposition"] == "FALSE_REJECT"
    assert len(store.dispositions("P00000", include_superseded=True)) == 2


def test_two_operators_are_two_opinions(store):
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00000", "OP-B", "CONFIRMED_DEFECT")
    assert len(store.dispositions("P00000")) == 2


# --- consensus ---------------------------------------------------------------

def test_unanimous_gives_a_label(store):
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00000", "OP-B", "CONFIRMED_DEFECT")
    c = store.consensus("P00000")
    assert c["label"] == 1 and c["agreed"] is True


def test_a_tie_gives_no_label(store):
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00000", "OP-B", "FALSE_REJECT")
    c = store.consensus("P00000")
    assert c["label"] is None and "contradict" in c["why"]


def test_abstention_is_not_dissent(store):
    """One operator saying 'I cannot tell' alongside one saying 'defect' is not
    a contradiction, and treating it as one throws away the only opinion
    anybody held."""
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00000", "OP-B", "UNCLASSIFIED")
    c = store.consensus("P00000")
    assert c["label"] == 1 and c["n_naming"] == 1


def test_everyone_abstaining_gives_no_label(store):
    store.dispose("P00000", "OP-A", "UNCLASSIFIED")
    store.dispose("P00000", "OP-B", "UNCLASSIFIED")
    assert store.consensus("P00000")["label"] is None


def test_a_majority_carries_but_is_marked_contested(store):
    for op, d in (("OP-A", "CONFIRMED_DEFECT"), ("OP-B", "CONFIRMED_DEFECT"),
                  ("OP-C", "FALSE_REJECT")):
        store.dispose("P00000", op, d)
    c = store.consensus("P00000")
    assert c["label"] == 1 and c["agreed"] is False and "majority" in c["why"]


def test_a_part_nobody_looked_at_has_no_consensus(store):
    assert store.consensus("P00001")["label"] is None
    assert store.consensus("P00001")["n"] == 0


# --- labels for retraining ---------------------------------------------------

def test_a_contested_label_weighs_less_than_an_agreed_one(store):
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00000", "OP-B", "CONFIRMED_DEFECT")
    store.dispose("P00001", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00001", "OP-B", "CONFIRMED_DEFECT")
    store.dispose("P00001", "OP-C", "FALSE_REJECT")
    w = {r["part_id"]: r["weight"] for r in store.labels_for_retraining()}
    assert w["P00000"] == 1.0 and w["P00001"] == 0.5


def test_require_consensus_drops_the_contested_ones(store):
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00000", "OP-B", "FALSE_REJECT")
    store.dispose("P00001", "OP-A", "CONFIRMED_DEFECT")
    assert len(store.labels_for_retraining()) == 1
    assert len(store.labels_for_retraining(require_consensus=True)) == 1


def test_unclassified_yields_no_label_at_all(store):
    store.dispose("P00002", "OP-A", "UNCLASSIFIED")
    assert store.labels_for_retraining() == []


# --- measuring the operator --------------------------------------------------

def test_agreement_is_measured_only_where_two_people_looked(store):
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00001", "OP-A", "CONFIRMED_DEFECT")
    assert store.agreement()["n_parts_multiply_reviewed"] == 0
    store.dispose("P00000", "OP-B", "CONFIRMED_DEFECT")
    a = store.agreement()
    assert a["n_parts_multiply_reviewed"] == 1
    assert a["exact_agreement"] == 1.0


def test_disagreement_shows_up_as_disagreement(store):
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00000", "OP-B", "FALSE_REJECT")
    a = store.agreement()
    assert a["exact_agreement"] == 0.0
    assert a["defect_or_not_agreement"] == 0.0


def test_by_operator_counts_only_live_dispositions(store):
    store.dispose("P00000", "OP-A", "CONFIRMED_DEFECT")
    store.dispose("P00000", "OP-A", "FALSE_REJECT")
    d = store.by_operator()["OP-A"]
    assert d["n"] == 1 and d.get("CONFIRMED_DEFECT") is None


# --- HTTP --------------------------------------------------------------------

def _call(url, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


@pytest.fixture
def live(store):
    h = RS.serve(store, "<h1>review</h1>")
    yield h
    h["server"].shutdown()


def test_the_page_is_served_at_the_root(live):
    with urllib.request.urlopen(live["url"] + "/") as f:
        assert b"review" in f.read()


def test_a_disposition_posts_and_shows_up(live, store):
    c, b = _call(live["url"], "/api/dispose",
                 {"part_id": "P00000", "operator": "OP-A",
                  "disposition": "CONFIRMED_DEFECT"})
    assert c == 200 and b["replayed"] is False
    assert len(store.dispositions("P00000")) == 1


def test_an_incoherent_disposition_is_a_400_with_the_reason(live):
    c, b = _call(live["url"], "/api/dispose",
                 {"part_id": "P00002", "operator": "OP-A",
                  "disposition": "FALSE_REJECT"})
    assert c == 400 and "coherent" in b["error"]


def test_an_unknown_part_is_a_404_not_a_500(live):
    c, _ = _call(live["url"], "/api/dispose",
                 {"part_id": "NOPE", "operator": "OP-A",
                  "disposition": "UNCLASSIFIED"})
    assert c == 404


def test_missing_fields_are_named(live):
    c, b = _call(live["url"], "/api/dispose", {"part_id": "P00000"})
    assert c == 400
    assert "operator" in b["error"] and "disposition" in b["error"]


def test_the_idempotency_key_survives_the_http_round_trip(live, store):
    body = {"part_id": "P00000", "operator": "OP-A",
            "disposition": "CONFIRMED_DEFECT", "idem_key": "page:1"}
    assert _call(live["url"], "/api/dispose", body)[1]["replayed"] is False
    assert _call(live["url"], "/api/dispose", body)[1]["replayed"] is True
    assert len(store.dispositions("P00000")) == 1


def test_labels_and_agreement_are_readable_over_http(live):
    _call(live["url"], "/api/dispose", {"part_id": "P00000", "operator": "OP-A",
                                        "disposition": "CONFIRMED_DEFECT"})
    _call(live["url"], "/api/dispose", {"part_id": "P00000", "operator": "OP-B",
                                        "disposition": "FALSE_REJECT"})
    assert _call(live["url"], "/api/labels")[1] == []
    a = _call(live["url"], "/api/agreement")[1]
    assert a["agreement"]["n_parts_multiply_reviewed"] == 1
    assert set(a["by_operator"]) == {"OP-A", "OP-B"}


def test_a_bad_body_is_a_400(live):
    req = urllib.request.Request(live["url"] + "/api/dispose", data=b"{oh no",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req)
        assert False, "should have been rejected"
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_unknown_routes_are_404(live):
    assert _call(live["url"], "/api/nope")[0] == 404
    assert _call(live["url"], "/api/nope", {})[0] == 404


# --- the station page --------------------------------------------------------

def test_the_static_station_still_has_no_service_in_it():
    """A page that needs a server running cannot be emailed to a supplier."""
    import numpy as np
    import inspect_service as SVC

    class FakeSvc:
        fingerprint = "fp"
        screen_threshold = 1.0

        def inspect(self, img, part_id=None):
            return {"ok": True, "verdict": "REJECT_CLASSIFIED",
                    "anomaly_score": 2.0, "part_id": part_id}

        def heatmap(self, img):
            return np.zeros((8, 8))

    imgs = np.zeros((2, 8, 8), dtype=np.float32)
    out = SVC.write_review_station(pathlib.Path(_tmp() / "s.html"), FakeSvc(),
                                   imgs, np.array([0, 1]))
    # Self-contained because POST_TO is null and dispose() returns before it
    # would fetch -- not because the code is absent. The claim under test is
    # that opening this file makes no network request, and a page with no
    # absolute URL in it cannot.
    assert out["self_contained"] is True
    assert "const POST_TO = null;" in out["html"]
    assert "http://" not in out["html"] and "https://" not in out["html"]


def test_the_wired_station_posts_and_keys_each_click():
    import numpy as np
    import inspect_service as SVC

    class FakeSvc:
        fingerprint = "fp"
        screen_threshold = 1.0

        def inspect(self, img, part_id=None):
            return {"ok": True, "verdict": "REJECT_CLASSIFIED",
                    "anomaly_score": 2.0, "part_id": part_id}

        def heatmap(self, img):
            return np.zeros((8, 8))

    imgs = np.zeros((2, 8, 8), dtype=np.float32)
    out = SVC.write_review_station(pathlib.Path(_tmp() / "s.html"), FakeSvc(),
                                   imgs, np.array([0, 1]),
                                   post_to="http://x/api/dispose",
                                   operator="OP-9")
    html = out["html"]
    assert out["self_contained"] is False
    assert "fetch(" in html and '"http://x/api/dispose"' in html
    assert '"OP-9"' in html
    assert "idem_key" in html
    # A failed POST must not look accepted: the operator would believe the part
    # is dealt with and nothing downstream would ever see it.
    assert "failed" in html and "button.failed" in html


def _tmp():
    import tempfile
    return pathlib.Path(tempfile.mkdtemp(prefix="ml2page-"))


# --- boundaries --------------------------------------------------------------

def test_it_does_not_import_another_project():
    """The pattern transfers; the code does not."""
    src = (ROOT / "src" / "review_service.py").read_text(encoding="utf-8")
    code = [l for l in src.splitlines()
            if l.strip().startswith(("import ", "from "))]
    code = " ".join(code).lower()
    for other in ("se2", "se_2", "ml3", "ml_3", "fleet_service",
                  "inspect_service", "cascade"):
        assert other not in code, code
    assert "sys.path" not in src


def test_the_limits_are_stated():
    j = " ".join(RS.LIMITS)
    assert "No authentication" in j
    assert "self-selected subset" in j
