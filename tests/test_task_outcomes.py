"""A task that never ran is not a task the network got wrong.

neuro-san reports a timeout and a blown recursion cap as ordinary answer
strings. They reach `score()`, compare false against the expected answer, and
land in the cache as incorrect -- indistinguishable from a network that went
and looked and got it wrong.

That is not hypothetical. Every seed measurement committed to this repository
contains it, and it produced the project's headline. All three seed topologies
score 0.8235, which was read as "topology changes cost, not accuracy". Separate
the two kinds of failure and the picture inverts: `solo` answered everything it
finished, and crashed out of three questions; `designer_shaped` timed out on
two and got one wrong. The three are equal only in the count of tasks that
failed, for three different reasons.

The runner already refuses to cache an evaluation poisoned by a provider quota,
with a comment warning about exactly this -- "a plausible-looking partial score
that gets cached forever". The guard just never covered the failures that
actually happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esp.eval.runner import (
    Evaluation,
    TaskResult,
    classify,
    evaluation_from_cache,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "cache"


def result(answer: str, correct: bool = False, error: str = "") -> TaskResult:
    return TaskResult("T01", 2, correct, 1.0, answer, error)


# ------------------------------------------------------ what counts as a crash

@pytest.mark.parametrize("answer", [
    "Agent timed out: max_execution_seconds=600s exceeded.",
    "Error from Answerer: Agent stopped due to exception Recursion limit of 40 reached",
])
def test_a_harness_failure_is_not_a_wrong_answer(answer):
    assert classify([result(answer)])[0].infrastructure is True


def test_an_escaped_exception_is_a_harness_failure():
    """It is recorded with an empty answer and a populated error. Scoring that
    as wrong credits the task set with a measurement nobody made."""
    assert classify([result("", error="RuntimeError: boom")])[0].infrastructure


def test_a_genuinely_wrong_answer_is_not_reclassified():
    """The guard must not launder bad topologies into blameless ones. This is
    the answer `designer_shaped` actually gave to T07, and it is simply wrong."""
    assert classify([result("Anvil Brewing")])[0].infrastructure is False


def test_a_correct_answer_is_never_a_harness_failure():
    assert classify([result("J. Vasquez", correct=True)])[0].infrastructure is False


# ------------------------------------------------- the two accuracies disagree

def evaluation(*results: TaskResult) -> Evaluation:
    marked = classify(list(results))
    return Evaluation("h", round(sum(r.correct for r in marked) / len(marked), 4),
                      0.0, 0, 0.0, 1, 1, marked)


def test_accuracy_and_answered_accuracy_are_different_measurements():
    ev = evaluation(
        result("right", correct=True),
        result("also right", correct=True),
        result("Agent timed out: max_execution_seconds=600s exceeded."),
    )
    assert ev.accuracy == pytest.approx(0.6667, abs=1e-4)   # crash counts as wrong
    assert ev.answered_accuracy() == 1.0                    # nothing it answered was wrong
    assert ev.incomplete == 1


def test_answered_accuracy_is_none_when_nothing_ran():
    ev = evaluation(result("Agent timed out: max_execution_seconds=600s exceeded."))
    assert ev.answered_accuracy() is None
    assert ev.incomplete == 1


# ------------------------------------ the committed measurements are affected

def load(name: str) -> Evaluation:
    return evaluation_from_cache(
        json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))


SEEDS = {"459ac1a66d925b0c": "designer_shaped",
         "c3ee2e0c4a6156c5": "flat_pair",
         "cbe128a617466fe9": "solo"}


def test_every_committed_seed_measurement_contains_a_harness_failure():
    """Recorded so it cannot quietly stop being true. If a re-measurement ever
    clears these, this test failing is the signal to update the README's
    findings -- not to delete the test."""
    for digest, name in SEEDS.items():
        assert load(digest).incomplete > 0, f"{name} no longer has any"


def test_the_identical_accuracy_headline_does_not_survive_the_split():
    """The three seeds agree on `accuracy` and disagree on what they answered.
    Reporting only the first is what made topology look irrelevant to quality.

    The specific ORDER varies run to run (which questions each seed fails on
    depends on task-level timeouts, which are timing-sensitive). The stable
    claim is that ordering EXISTS -- answered_accuracy splits three ways
    where reported accuracy tied them at one number."""
    reported = {name: load(d).accuracy for d, name in SEEDS.items()}
    answered = {name: load(d).answered_accuracy() for d, name in SEEDS.items()}

    assert len(set(reported.values())) == 1, reported
    assert len(set(answered.values())) >= 2, answered


def test_solo_never_attempted_three_of_the_seventeen_tasks():
    """The single-agent baseline hit neuro-san's recursion cap on all three
    3-hop questions. Its accuracy on them is unknown, not zero."""
    solo = load("cbe128a617466fe9")
    assert solo.incomplete == 3
    crashed = [r.task_id for r in solo.results if r.infrastructure]
    assert crashed == ["T06", "T07", "T08"]
