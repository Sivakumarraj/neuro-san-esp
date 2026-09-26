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
    monkeypatch.setattr(serve, "_recent", {})
    monkeypatch.setattr(serve, "SPEND_FILE", None)
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


@pytest.mark.parametrize("reply, hint", [
    ("Error from Coordinator: Agent stopped due to exception Error calling model "
     "'gemini-3.5-flash' (UNAUTHENTICATED): 401 UNAUTHENTICATED.", "rejected the key"),
    ("Error from Coordinator: Agent stopped due to exception 429 RESOURCE_EXHAUSTED",
     "limit is used up"),
    ("Error from Coordinator: Agent stopped due to exception 404 NOT_FOUND", "make probe"),
])
def test_an_agent_failure_is_an_error_not_an_answer(client, monkeypatch, reply, hint):
    """neuro-san returns an agent's failure as the reply text. It reached the
    page under a 200, as the network's answer, and nothing said to fix .env."""
    monkeypatch.setattr(serve, "_ask", lambda *a: (reply, {}, 1.0))
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 502
    error = response.json()["error"]
    assert hint in error
    assert reply[:40] in error


def test_the_question_cap_is_enforced(client, monkeypatch):
    """One careless loop would spend a 500-a-day budget in minutes."""
    monkeypatch.setattr(serve, "_ask", lambda *a: ("x", {}, 1.0))
    monkeypatch.setattr(serve, "MAX_QUESTIONS", 2)
    for _ in range(2):
        assert client.post("/ask", json={"question": "q"}).status_code == 200
    blocked = client.post("/ask", json={"question": "q"})
    assert blocked.status_code == 429


def test_one_visitor_cannot_spend_everybodys_day(client, monkeypatch):
    """The daily cap guards the key's bill; the per-visitor share keeps one
    person's loop from using up the page for everyone else."""
    monkeypatch.setattr(serve, "_ask", lambda *a: ("x", {}, 1.0))
    monkeypatch.setattr(serve, "PER_CLIENT_HOURLY", 2)
    for _ in range(2):
        assert client.post("/ask", json={"question": "q"}).status_code == 200
    blocked = client.post("/ask", json={"question": "q"})
    assert blocked.status_code == 429 and "per visitor" in blocked.json()["error"]
    assert serve._asked["count"] == 2, "a refused question was counted"


def test_the_forwarded_address_is_only_trusted_when_told_to(client, monkeypatch):
    """Behind no proxy, X-Forwarded-For is whatever the visitor typed."""
    monkeypatch.setattr(serve, "_ask", lambda *a: ("x", {}, 1.0))
    monkeypatch.setattr(serve, "PER_CLIENT_HOURLY", 1)
    first = client.post("/ask", json={"question": "q"}, headers={"x-forwarded-for": "1.1.1.1"})
    forged = client.post("/ask", json={"question": "q"}, headers={"x-forwarded-for": "2.2.2.2"})
    assert first.status_code == 200 and forged.status_code == 429
    monkeypatch.setattr(serve, "TRUST_PROXY", True)
    monkeypatch.setattr(serve, "_recent", {})
    for address in ("1.1.1.1", "2.2.2.2"):
        response = client.post("/ask", json={"question": "q"},
                               headers={"x-forwarded-for": f"9.9.9.9, {address}"})
        assert response.status_code == 200


def test_the_days_spend_survives_a_restart(client, monkeypatch, tmp_path):
    """Held only in memory, the count went back to zero on every restart, so a
    crash loop had no cap at all."""
    monkeypatch.setattr(serve, "_ask", lambda *a: ("x", {}, 1.0))
    monkeypatch.setattr(serve, "SPEND_FILE", str(tmp_path / "spend.json"))
    monkeypatch.setattr(serve, "_spend_day", {"day": None})
    monkeypatch.setattr(serve, "MAX_QUESTIONS", 3)
    for _ in range(3):
        assert client.post("/ask", json={"question": "q"}).status_code == 200
    # A restart: memory is gone, the file is not.
    monkeypatch.setattr(serve, "_asked", {"count": 0})
    monkeypatch.setattr(serve, "_spend_day", {"day": None})
    monkeypatch.setattr(serve, "_recent", {})
    assert client.post("/ask", json={"question": "q"}).status_code == 429


def test_a_new_day_starts_a_new_budget(client, monkeypatch):
    monkeypatch.setattr(serve, "_ask", lambda *a: ("x", {}, 1.0))
    monkeypatch.setattr(serve, "MAX_QUESTIONS", 1)
    assert client.post("/ask", json={"question": "q"}).status_code == 200
    assert client.post("/ask", json={"question": "q"}).status_code == 429
    monkeypatch.setattr(serve, "_today", lambda: "2999-01-01")
    assert client.post("/ask", json={"question": "q"}).status_code == 200


@pytest.mark.parametrize("secret", [
    "AIzaSyD" + "x" * 32, "AQ.Ex" + "y" * 40, "sk-ant-" + "z" * 40])
def test_no_error_ever_carries_a_key(client, monkeypatch, secret):
    """Errors reach whoever holds the link. A provider that echoes a request
    detail must never put the key on the page."""
    def boom(*a):
        raise OSError(f"401 bad key {secret}")

    monkeypatch.setattr(serve, "_ask", boom)
    error = client.post("/ask", json={"question": "q"}).json()["error"]
    assert secret not in error and "[redacted key]" in error
    reply = ("Error from Coordinator: Agent stopped due to exception "
             f"401 UNAUTHENTICATED {secret}")
    monkeypatch.setattr(serve, "_ask", lambda *a: (reply, {}, 1.0))
    error = client.post("/ask", json={"question": "q"}).json()["error"]
    assert secret not in error


def test_finished_measurements_do_not_accumulate_for_ever(monkeypatch):
    monkeypatch.setattr(serve, "MAX_KEPT_JOBS", 2)
    jobs = {f"j{i}": {"state": "done"} for i in range(5)}
    jobs["live"] = {"state": "running"}
    monkeypatch.setattr(serve, "_jobs", jobs)
    serve._prune_jobs()
    assert list(serve._jobs) == ["j3", "j4", "live"]


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


def test_the_header_names_the_best_measurement_on_disk():
    """The header and the Measure tab's table are read from the same data, so
    they cannot disagree. They did: a committed state/champion.json that
    predated the twelfth measurement put +0.8453 under "best-measured" while
    the table beneath listed +0.8941."""
    from esp.eval import measurements

    best = measurements.best()
    if best is not None:
        assert serve.RECORD.fitness >= best.fitness


def test_the_caveat_does_not_present_an_old_figure_as_current(client):
    """The -0.333 was the surrogate at the first search, on nine samples. Said
    without that context it reads as the predictor's quality today."""
    caveat = serve.caveat()
    if "-0.333" in caveat or "\u22120.333" in caveat:
        assert "first used" in caveat


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


# ------------------------------------------- full answers, any provider

def test_a_long_explained_answer_reaches_the_visitor_whole(client, monkeypatch):
    """The page used to cut every answer at 2,000 characters -- exactly the
    explanation a person came to read."""
    long_answer = "The total penalty owed is 4500. " + "Evidence line. " * 400
    monkeypatch.setattr(serve, "_ask", lambda *a: (long_answer, {}, 1.0))
    body = client.post("/ask", json={"question": "anything"}).json()
    assert body["answer"] == long_answer


def test_the_page_offers_the_four_showcase_questions(client):
    import html

    from esp.serving import SHOWCASE, display_question

    text = client.get("/").text
    for task in SHOWCASE:
        # Escaped, as the page must: "depot's" arrives as "depot&#x27;s".
        assert html.escape(display_question(task)) in text
    assert "Answer with the number only" not in text


def test_a_showcase_question_as_displayed_is_graded(client, monkeypatch):
    from esp.serving import SHOWCASE, display_question

    task = SHOWCASE[3]
    explained = f"The total penalty owed is {task.answer}. It was worked out as follows."
    monkeypatch.setattr(serve, "_ask", lambda *a: (explained, {}, 1.0))
    body = client.post("/ask", json={"question": display_question(task)}).json()
    assert body["expected"] == task.answer
    assert body["correct"] is True


def test_the_answer_names_the_models_it_ran_on(client, monkeypatch):
    monkeypatch.setattr(serve, "_ask", lambda *a: ("x", {}, 1.0))
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["provider"] == serve.SERVED.provider
    assert body["router_model"] and body["worker_model"]


def test_the_page_says_how_the_served_network_differs_from_the_measured_one(client):
    assert "reply format" in client.get("/").text or "not been measured" in client.get("/").text


def test_the_question_limit_message_is_not_about_one_providers_free_tier(client,
                                                                        monkeypatch):
    monkeypatch.setattr(serve, "MAX_QUESTIONS", 0)
    error = client.post("/ask", json={"question": "q"}).json()["error"]
    assert "free" not in error.lower() and "500 requests" not in error


def test_startup_asks_for_the_serving_providers_key_not_googles(monkeypatch, capsys):
    """A deployment holding only a Claude key refused to start at all."""
    from esp.serving import Served

    claude = serve.SERVED.genome.clone()
    claude.default_model = "claude-haiku-4-5"
    monkeypatch.setattr(serve, "SERVED", Served(claude, "anthropic", True))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "AQ.EXAMPLE-not-a-real-key-0000000000000000000000")
    assert serve.main() == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


# ------------------------------------------------------------ measuring

def _report(name_accuracy_tokens):
    from esp.eval.runner import TaskResult
    from esp.measure import Report

    accuracy, tokens = name_accuracy_tokens
    correct = round(accuracy * 4)
    results = [TaskResult(f"Q{i}", 1, i < correct, 1.0, "answer") for i in range(4)]
    return Report(network="n", suite="s", results=results, tokens=tokens, cost=0.0,
                  seconds=4.0, questions={r.task_id: "q" for r in results},
                  expected={r.task_id: "a" for r in results})


def _wait(client, job_id):
    import time
    for _ in range(100):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["state"] != "running":
            return state
        time.sleep(0.05)
    raise AssertionError("the measurement never finished")


@pytest.fixture
def measuring(monkeypatch):
    monkeypatch.setattr(serve, "_measured", {"count": 0})
    monkeypatch.setattr(serve, "_jobs", {})
    monkeypatch.setattr(serve, "SPEND_FILE", None)
    monkeypatch.setattr(serve, "MAX_MEASURE", 1000)
    return TestClient(serve.build_app())


def test_the_committed_networks_are_offered_with_what_they_scored(measuring):
    listed = measuring.get("/api/networks").json()
    committed = [n for n in listed if n["measured"]]
    assert len(committed) == 12
    assert all({"accuracy", "tokens", "agents", "on"} <= set(n["measured"]) for n in committed)


@pytest.mark.parametrize("body,complaint", [
    ({"networks": []}, "at least one"),
    ({"networks": ["a", "b", "c", "d", "e"]}, "at most"),
    ({"networks": ["no-such-network"]}, "unknown"),
    ({"networks": ["__first__"], "suite": "{nope"}, "not JSON"),
    ({"networks": ["__first__"], "suite": "\n".join(
        f'{{"question": "q{i}", "answer": "a"}}' for i in range(60))}, "at most 50"),
])
def test_a_bad_measurement_request_is_refused_before_anything_runs(measuring, body, complaint):
    if body["networks"] == ["__first__"]:
        body = {**body, "networks": [measuring.get("/api/networks").json()[0]["id"]]}
    response = measuring.post("/api/measure", json=body)
    assert response.status_code == 400 and complaint in response.json()["error"]
    assert serve._measured["count"] == 0, "a refused request spent budget"


def test_the_deployment_budget_is_enforced_up_front(measuring, monkeypatch):
    monkeypatch.setattr(serve, "MAX_MEASURE", 10)
    first = measuring.get("/api/networks").json()[0]["id"]
    response = measuring.post("/api/measure", json={"networks": [first]})   # 17 runs
    assert response.status_code == 429 and "left of its 10" in response.json()["error"]


def test_a_measurement_runs_reports_every_network_and_marks_the_front(measuring, monkeypatch):
    listed = measuring.get("/api/networks").json()
    a, b, c = listed[0], listed[1], listed[2]
    # Keyed by id: two committed networks share the name mut:reassign_model.
    outcomes = {a["id"]: (0.75, 1000), b["id"]: (0.5, 2000)}

    def fake_measure(hocon, tasks, suite, on_result):
        name = next(n["id"] for n in listed if serve.candidates()[n["id"]].hocon == hocon)
        if name not in outcomes:
            raise OSError("the whole evaluation cost zero tokens")
        report = _report(outcomes[name])
        for result in report.results:
            on_result(result)
        return report

    monkeypatch.setattr(serve, "measure_network", fake_measure)
    started = measuring.post("/api/measure", json={
        "networks": [a["id"], b["id"], c["id"]],
        "suite": '{"question": "q", "answer": "a"}'})
    assert started.status_code == 200, started.text
    done = _wait(measuring, started.json()["job"])

    assert done["state"] == "done"
    rows = {r["id"]: r for r in done["reports"]}
    assert rows[a["id"]]["pareto"] and not rows[b["id"]]["pareto"], (
        "the better-and-cheaper network is on the front; the other is dominated")
    assert "zero tokens" in rows[c["id"]]["error"], "one failure must not stop the others"


def test_only_one_measurement_runs_at_a_time(measuring, monkeypatch):
    monkeypatch.setattr(serve, "_jobs", {"x": {"state": "running"}})
    first = measuring.get("/api/networks").json()[0]["id"]
    assert measuring.post("/api/measure", json={"networks": [first]}).status_code == 409


def test_the_page_lists_every_benchmark_question_not_four(client):
    import html

    from esp.eval.tasks import TASKS
    from esp.serving import display_question

    text = client.get("/").text
    assert all(html.escape(display_question(t)) in text for t in TASKS)
    assert "Measure networks" in text


def test_the_pages_script_parses(client):
    """A newline escape once turned into a real line break inside a string and
    left the Measure tab dead in every browser. Parsed here, not assumed."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = client.get("/").text.split("<script>")[1].split("</script>")[0]
    done = subprocess.run([node, "-e", "new Function(require('fs').readFileSync(0,'utf8'))"],
                          input=script, capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr


def test_a_busy_router_falls_back_and_says_so(client, monkeypatch):
    """The champion's router runs on a model the free tier allows about twenty
    requests a day, and on the first live run it answered 503 "high demand" to
    three of four questions. The page retries once with every agent on the
    workers' model, and labels the answer as coming from an unmeasured variant."""
    if serve.FALLBACK_HOCON is None:
        pytest.skip("the served champion promotes no agent")
    busy = ("Error from Coordinator: Agent stopped due to exception 503 UNAVAILABLE. "
            "This model is currently experiencing high demand.")
    calls = []

    def ask(hocon, question):
        calls.append(hocon)
        return (busy, {}, 1.0) if hocon == serve.HOCON else ("J. Vasquez", {}, 2.0)

    monkeypatch.setattr(serve, "_ask", ask)
    body = client.post("/ask", json={"question": "who manages D08?"}).json()
    assert calls == [serve.HOCON, serve.FALLBACK_HOCON]
    assert body["answer"] == "J. Vasquez"
    assert "has not been measured" in body["fallback"]
    assert body["router_model"] == body["worker_model"]


def test_a_refused_key_is_not_retried(client, monkeypatch):
    """Only an unavailable model is worth a second paid attempt. A bad key
    fails the same way on every model."""
    calls = []
    refused = ("Error from Coordinator: Agent stopped due to exception 401 "
               "UNAUTHENTICATED API key not valid")
    monkeypatch.setattr(serve, "_ask", lambda *a: calls.append(1) or (refused, {}, 1.0))
    assert client.post("/ask", json={"question": "q"}).status_code == 502
    assert len(calls) == 1


def test_an_answer_from_the_measured_network_claims_no_fallback(client, monkeypatch):
    monkeypatch.setattr(serve, "_ask", lambda *a: ("J. Vasquez", {}, 1.0))
    assert client.post("/ask", json={"question": "q"}).json()["fallback"] is None


def test_idle_visitors_are_forgotten(client, monkeypatch):
    """The per-visitor table must not grow by one entry per address for as
    long as a public page runs."""
    from collections import deque
    monkeypatch.setattr(serve, "MAX_TRACKED_CLIENTS", 2)
    old = serve.time.monotonic() - 7200
    monkeypatch.setattr(serve, "_recent", {f"10.0.0.{i}": deque([old]) for i in range(5)})
    serve._over_client_limit("10.9.9.9")
    assert list(serve._recent) == ["10.9.9.9"]
