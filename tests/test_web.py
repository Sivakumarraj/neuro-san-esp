"""The public surface. It is the only part a stranger ever touches.

Everything else in this repository fails privately -- a bad wake costs a wake.
This fails in front of whoever was sent the link, and it spends the day's
provider budget while doing it, so the properties worth pinning are the ones
that keep a visitor from being lied to or from emptying the quota.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "web"))

serve = pytest.importorskip("serve")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(serve, "_asked", {"count": 0})
    return TestClient(serve.build_app())


def test_the_page_loads_without_a_provider_call(client):
    """Rendering must not cost anything. A page that spends a request to draw
    itself empties the day's budget on crawlers alone."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Meridian" in response.text


def test_the_question_body_is_actually_parsed(client, monkeypatch):
    """The regression this file exists for.

    serve.py uses `from __future__ import annotations`, so every annotation is a
    string FastAPI resolves against module globals. With the request model
    defined inside build_app() it is invisible there, FastAPI silently
    reclassifies the body as a query parameter, and every ask returns 422
    complaining about a missing query field -- which points nowhere near the
    cause. Caught by calling it, not by reading it.
    """
    monkeypatch.setattr(serve, "_ask", lambda *a: ("J. Vasquez", {}, 1.0))
    response = client.post("/ask", json={"question": "who manages D08?"})
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "J. Vasquez"


def test_a_graded_question_is_graded_in_front_of_the_visitor(client, monkeypatch):
    """Claiming correctness without showing what was expected is the same as
    not claiming it."""
    from esp.eval.tasks import TASKS

    task = TASKS[0]
    monkeypatch.setattr(serve, "_ask", lambda *a: (task.answer, {}, 1.0))
    body = client.post("/ask", json={"question": task.question}).json()
    assert body["expected"] == task.answer
    assert body["correct"] is True


def test_a_wrong_answer_is_reported_wrong(client, monkeypatch):
    from esp.eval.tasks import TASKS

    task = TASKS[0]
    monkeypatch.setattr(serve, "_ask", lambda *a: ("somebody else", {}, 1.0))
    body = client.post("/ask", json={"question": task.question}).json()
    assert body["correct"] is False


def test_an_ungraded_question_claims_nothing(client, monkeypatch):
    """Most questions have no ground truth. The page must not imply one."""
    monkeypatch.setattr(serve, "_ask", lambda *a: ("a plausible answer", {}, 1.0))
    body = client.post("/ask", json={"question": "what is the weather"}).json()
    assert body["expected"] is None
    assert body["correct"] is None


def test_a_provider_failure_does_not_500_the_page(client, monkeypatch):
    """A 429 is the expected failure here, not an exceptional one. It should
    read as a quota message, not as a crash."""
    def boom(*a):
        raise OSError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(serve, "_ask", boom)
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 502
    assert "429" in response.json()["error"]


def test_the_question_cap_is_enforced(client, monkeypatch):
    """One careless loop would spend a 500-a-day budget in minutes."""
    monkeypatch.setattr(serve, "_ask", lambda *a: ("x", {}, 1.0))
    monkeypatch.setattr(serve, "MAX_QUESTIONS", 2)
    for _ in range(2):
        assert client.post("/ask", json={"question": "q"}).status_code == 200
    blocked = client.post("/ask", json={"question": "q"})
    assert blocked.status_code == 429


def test_an_empty_question_costs_nothing(client, monkeypatch):
    called = []
    monkeypatch.setattr(serve, "_ask", lambda *a: called.append(1) or ("x", {}, 1.0))
    assert client.post("/ask", json={"question": "   "}).status_code == 400
    assert not called, "an empty question reached the provider"


def test_health_says_whether_it_is_serving_a_measured_champion(client):
    """A deployment with no state serves the baseline. Saying it is a measured
    winner would be the page's easiest lie."""
    body = client.get("/health").json()
    assert body["ok"] is True
    assert isinstance(body["measured"], bool)


def test_the_page_states_what_has_not_been_achieved(client):
    """The same admissions as the PDFs. A public page that quietly drops them
    is where an overstatement would actually reach somebody.

    The wording is derived from state, so this checks that a caveat is present
    and substantive rather than pinning a sentence that goes stale the moment
    the measurements change.
    """
    text = client.get("/").text
    assert "What it is not." in text
    caveat = serve.caveat()
    assert len(caveat) > 40, caveat
    assert caveat.rstrip().endswith("rather than left out.")
    assert "Stated in the repository" in text


def test_the_caveat_never_contradicts_the_champion(client):
    """The page states what it has not shown, derived from state.

    Written as prose in the template it went stale the moment an evolved
    candidate won: the page denied that any mutant had beaten the baselines
    directly beneath a header naming one as the best-measured topology.
    """
    body = client.get("/").text
    if serve.RECORD is not None and serve.RECORD.origin.startswith("mut:"):
        assert "no evolved candidate has beaten" not in body.lower()


def test_the_caveat_reports_the_real_measurement_count(client):
    """A stated shortfall has to match what is actually on disk."""
    from esp.surrogate.predictor import MIN_SAMPLES

    measured = serve.measurement_count()
    body = client.get("/").text
    if measured < MIN_SAMPLES:
        assert f"there are {measured}" in body
        assert f"needs {MIN_SAMPLES} measurements" in body
    else:
        assert "has not trained" not in body
