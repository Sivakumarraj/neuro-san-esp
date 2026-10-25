"""The penalty autopsy: its parts recompute every answer key, its kinds are
the ones they say, and every figure quoted is what the committed logs give."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from esp import autopsy
from esp.eval.world import build_world

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "autopsy" / "summary.json"
FINDINGS = (ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
WORLD = build_world()


@pytest.fixture(scope="module")
def study() -> dict:
    return autopsy.analyse()


def _tasks():
    from esp.eval.fresh import FRESH
    from esp.eval.suites import JUDGE

    return [t for t in [*JUDGE, *FRESH] if autopsy.is_multi_penalty(t.question)]


def test_the_parts_recompute_every_penalty_answer_key():
    """If the parts were read wrong, every kind below would be too."""
    tasks = _tasks()
    assert len(tasks) == 16
    for task in tasks:
        owed = [h * r for h, r in autopsy.parts(task.question, WORLD)]
        truth = abs(owed[0] - owed[1]) if "exceed" in task.question else sum(owed)
        assert str(truth) == task.answer, task.task_id


def test_the_two_confident_misses_are_the_difference_of_the_rates():
    """J004 and F11, which the cheap network answered the same way twice."""
    by_id = {t.task_id: t for t in _tasks()}
    assert autopsy.parts(by_id["J004"].question, WORLD) == [(12, 4375), (37, 3875)]
    assert autopsy.explain(by_id["J004"].question, "500", False, WORLD) == "rates only"
    assert autopsy.explain(by_id["F11"].question, "3125", False, WORLD) == "rates only"


def test_each_kind_is_what_it_says():
    question = next(t.question for t in _tasks() if t.task_id == "J042")
    owed = [h * r for h, r in autopsy.parts(question, WORLD)]  # three incidents
    assert autopsy.explain(question, str(owed[0] + owed[1]), False, WORLD) == "missed incidents"
    assert autopsy.explain(question, str(sum(owed) * 40), False, WORLD) == "wrong by 10x or more"
    assert autopsy.explain(question, "0", False, WORLD) == "wrong by 10x or more"
    assert autopsy.explain(question, str(sum(owed) + 2000), False, WORLD) == "arithmetic"
    assert autopsy.explain(question, "unknown", False, WORLD) == "not a number"
    assert autopsy.explain(question, "anything", True, WORLD) == "right"


def test_every_run_is_counted_once(study):
    for roles in study.values():
        for r in roles.values():
            assert sum(r["kinds"].values()) == r["share"]["runs"] == 16
            assert r["kinds"]["right"] == r["share"]["runs_right"]


def test_the_committed_summary_is_what_the_logs_say(study):
    assert json.loads(SUMMARY.read_text(encoding="utf-8")) == study


def test_the_docs_quote_the_autopsy_as_computed(study):
    wrong = {role: sum(study[s][role]["share"]["wrong_runs"] for s in study)
             for role in ("cheap", "strong")}
    every = {role: sum(study[s][role]["share"]["all_wrong_runs"] for s in study)
             for role in ("cheap", "strong")}
    missed = sum(study[s][r]["kinds"]["missed incidents"] for s in study for r in study[s])
    for text in (README, FINDINGS):
        assert f"{wrong['cheap']} of its {every['cheap']}" in text
        assert f"{wrong['strong']} of its {every['strong']}" in text
        assert f"{missed} of 60" in text


def test_the_report_script_runs(capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import autopsy_report

    assert autopsy_report.main([]) == 0
    assert "What each wrong number is" in capsys.readouterr().out
