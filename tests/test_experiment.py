"""The same-budget experiment, run end to end against a fake provider.

It will spend tens of dollars when it runs for real, so everything about its
logic is checked here first: both arms get exactly the same budget, the control
arm never consults the Predictor, judging uses only the held-out set, a stopped
run resumes from the cache without paying twice, and an exam too easy to rank
anything stops the run before the search is paid for.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from esp.eval import runner
from esp.eval.runner import QuotaExhausted, SuiteRun, TaskResult
from esp.eval.suites import JUDGE, SELECT
from esp.evolve import experiment

ROOT = Path(__file__).resolve().parent.parent


class FakeProvider:
    """Deterministic answers: a network gets a question right or wrong by a hash
    of the two, so the same network always scores the same."""

    def __init__(self, rate: int = 70):
        self.rate = rate
        self.calls = 0
        self.fail_after: int | None = None

    def __call__(self, hocon_path, tasks, **_):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise QuotaExhausted("fake daily cap")
        text = Path(hocon_path).read_text(encoding="utf-8")
        name = Path(hocon_path).stem
        results = []
        for task in tasks:
            roll = int(hashlib.sha256(f"{name}{task.task_id}".encode()).hexdigest(), 16)
            results.append(TaskResult(task.task_id, task.hops, roll % 100 < self.rate,
                                      1.0, "answer"))
        return SuiteRun(results=results, tokens=len(text) * 10, cost=0.0, seconds=1.0)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    provider = FakeProvider()
    monkeypatch.setattr(runner, "run_suite", provider)
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "NETWORK_DIR", tmp_path / "networks")
    # The permutation null is what the gate reads, and its size is not what
    # these tests are about; two shuffles keep the logic and drop minutes.
    from esp.evolve import loop
    monkeypatch.setattr(loop, "NULL_TRIALS", 2)
    return provider


def small_plan() -> experiment.Plan:
    return experiment.Plan(select=SELECT[:6], judge=JUDGE[:8], budget=4,
                           per_generation=2, pool=60)


def test_both_arms_get_the_same_budget_and_are_judged(fake, tmp_path):
    summary = experiment.run(small_plan(), tmp_path / "out")
    assert summary.stopped is None
    assert {arm: info["paid_candidates"] for arm, info in summary.arms.items()} == {
        "predictor": 4, "random": 4}
    assert set(summary.judged) >= {"designer", "predictor"}
    for record in summary.judged.values():
        assert set(record["right"]) == {t.task_id for t in JUDGE[:8]}
    assert "designer vs predictor" in summary.comparisons
    written = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert written["arms"]["random"]["paid_candidates"] == 4


def test_the_control_arm_never_consults_the_predictor(fake, tmp_path):
    experiment.run(small_plan(), tmp_path / "out")
    random_arm = json.loads((tmp_path / "out" / "random" / "history.json").read_text())
    searched = [r for r in random_arm["records"] if r["generation"] > 0]
    assert searched and all(r["predicted"] is None for r in searched)
    predictor_arm = json.loads((tmp_path / "out" / "predictor" / "history.json").read_text())
    assert any(r["predicted"] is not None for r in predictor_arm["records"]
               if r["generation"] > 0)


def test_a_second_run_pays_for_nothing(fake, tmp_path):
    experiment.run(small_plan(), tmp_path / "out")
    first = fake.calls
    again = experiment.run(small_plan(), tmp_path / "out")
    assert again.stopped is None and fake.calls == first


def test_a_quota_stop_resumes_where_it_stopped(fake, tmp_path):
    """A daily cap mid-run is the expected end of a free-tier day, not a crash:
    what was measured is kept, and the next run carries on from it."""
    fake.fail_after = 12
    stopped = experiment.run(small_plan(), tmp_path / "out")
    assert stopped.stopped and "exhausted" in stopped.stopped
    fake.fail_after = None
    assert experiment.run(small_plan(), tmp_path / "out").stopped is None
    paid = fake.calls
    assert experiment.run(small_plan(), tmp_path / "out").stopped is None
    assert fake.calls == paid, "a finished run was paid for again"


def test_an_exam_everyone_passes_stops_before_the_search(fake, tmp_path):
    fake.rate = 100
    summary = experiment.run(small_plan(), tmp_path / "out")
    assert "too easy" in summary.stopped
    assert fake.calls == 1, "only the calibration was paid for"
    assert not summary.arms


def test_the_plan_is_priced_and_spends_nothing(fake, monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import experiment as script
    monkeypatch.setattr(sys, "argv", ["experiment.py", "--budget", "40"])
    assert script.main() == 0
    out = capsys.readouterr().out
    assert "5,640 question-runs" in out and "Nothing was spent" in out
    assert fake.calls == 0


def test_an_unknown_arm_is_refused(fake, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import experiment as script
    monkeypatch.setattr(sys, "argv", ["experiment.py", "--arms", "predictor,oracle"])
    assert script.main() == 2
