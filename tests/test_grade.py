"""What a measurement counts as a right answer, held against trivial replies.

`score` is containment. As a measurement that let two kinds of reply score
without answering anything: a list of numbers, which contains the answer to
most count questions, and a copy of the question, which contains "not stated"
on every judge-plus question. `grade` is what the runner scores with; these
tests pin that neither trick scores, and that no committed verdict changed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from esp.eval.bank import BANK
from esp.eval.judge_plus import JUDGE_200, JUDGE_PLUS, NOT_STATED
from esp.eval.suites import SELECT
from esp.eval.tasks import TASKS, Task, grade, score

ROOT = Path(__file__).resolve().parent.parent
SETS = {"meridian": TASKS, "bank": BANK, "select": SELECT, "judge-200": JUDGE_200}
SHOTGUN = " ".join(str(n) for n in range(0, 101))


@pytest.mark.parametrize("name", list(SETS))
def test_echoing_the_question_scores_nothing(name):
    assert sum(grade(t, t.question) for t in SETS[name]) == 0


def test_echoing_scored_on_every_unanswerable_question_before():
    """The hole this closes, shown on the old rule so it cannot be argued away."""
    unanswerable = [t for t in JUDGE_PLUS if t.answer == NOT_STATED]
    assert all(score(t.accepted, t.question) for t in unanswerable)
    assert not any(grade(t, t.question) for t in unanswerable)


@pytest.mark.parametrize("name", list(SETS))
def test_a_list_of_numbers_scores_nothing(name):
    tasks = SETS[name]
    assert sum(grade(t, SHOTGUN) for t in tasks) == 0
    assert sum(grade(t, SHOTGUN + " not stated") for t in tasks) == 0


def test_a_list_of_numbers_scored_on_nearly_half_the_judge_set_before():
    assert sum(score(t.accepted, " ".join(str(n) for n in range(21))) for t in JUDGE_200) >= 80


def test_the_best_constant_reply_is_a_quarter_of_judge_plus_at_most():
    """A constant reply is the chance level of an open-answer set. It is not
    zero here, and the documents say so: 'not stated' is right on the 25
    unanswerable questions and '2' on 24 others."""
    counts = Counter(t.answer for t in JUDGE_PLUS)
    best = max(counts.values())
    assert best <= len(JUDGE_PLUS) // 4
    for answer, count in counts.most_common(2):
        task = next(t for t in JUDGE_PLUS if t.answer == answer)
        reply = "not stated" if answer == NOT_STATED else answer
        assert grade(task, reply)
        assert sum(grade(t, reply) for t in JUDGE_PLUS) == count


def test_an_answer_with_its_explanation_still_scores():
    task = Task("X1", "Incident INC-4401 ran late. What is the total penalty owed? "
                "Answer with the number only.", "4500", 2, ("4500",))
    assert grade(task, "4500")
    assert grade(task, "The total penalty owed is 4,500. INC-4401 affected it.")
    assert not grade(task, "INC-4401 was 12 hours late at 375 per hour: 4500")
    assert not grade(task, "4501")


def test_numbers_the_question_states_are_not_a_second_answer():
    task = next(t for t in JUDGE_PLUS if "happened in 20" in t.question)
    year = next(y for y in ("2024", "2025", "2026") if y in task.question)
    assert grade(task, f"{task.answer} incidents happened in {year}.")


def test_an_unanswerable_question_still_accepts_an_abstention():
    task = next(t for t in JUDGE_PLUS if t.answer == NOT_STATED)
    assert grade(task, "Not stated.")
    assert grade(task, "There is no such record in the documents.")
    assert not grade(task, "It was 12 hours late.")


def _committed():
    """Every committed reply with the question it answered and its verdict."""
    by_id = {t.task_id: t for t in TASKS}
    for path in sorted((ROOT / "tests" / "fixtures" / "cache").glob("*.json")):
        for r in json.loads(path.read_text(encoding="utf-8"))["results"]:
            yield by_id[r["task_id"]], r
    for path in sorted((ROOT / "results").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for r in payload.get("results") or []:
            if isinstance(r, dict) and "expected" in r:
                yield Task(r["task_id"], r["question"], r["expected"], r["hops"],
                           (r["expected"],)), r


def test_every_committed_verdict_is_unchanged():
    """The stricter rule must not re-score history: all 272 committed replies
    are graded as they were recorded."""
    seen = 0
    for task, record in _committed():
        seen += 1
        assert grade(task, record["answer"]) is bool(record["correct"]), (task.task_id,
                                                                          record["answer"])
    assert seen == 272
