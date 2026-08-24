"""Every answer must be correct by construction -- including the ones in a tie.

This file's subject is the claim the whole fitness function rests on, stated in
`esp/eval/tasks.py` and in the README: the corpus and the ground truth come from
one seeded data model, so an answer cannot be wrong the way a hand-written one
can. Generating the ground truth removes the transcription error. It does not
remove the tie.

"Which contract reference appears in the most incident reports?" has twenty
correct answers in this world -- twenty contracts on two incidents each --
and `Counter.most_common(1)` returned whichever one insertion order put first.
Nineteen correct answers scored wrong: a one-in-twenty lottery wired directly
into the objective the search optimises, which is exactly the poisoned fitness
function the module warns about, reached by the mechanism meant to prevent it.
"""

from __future__ import annotations

from collections import Counter

from esp.eval.tasks import TASKS, build_tasks, score
from esp.eval.world import build_world


def test_every_task_accepts_at_least_its_own_answer():
    for task in TASKS:
        assert task.accepted, f"{task.task_id} accepts nothing"
        assert task.answer in task.accepted, task.task_id


def test_the_most_incidents_question_accepts_every_tied_contract():
    world = build_world()
    counts = Counter(i.contract_ref for i in world.incidents)
    most = max(counts.values())
    tied = sorted(ref for ref, count in counts.items() if count == most)

    task = next(t for t in TASKS if "most incident reports" in t.question)
    assert len(tied) > 1, "the tie is gone; this guard is now pinning nothing"
    assert set(task.accepted) == set(tied), (
        f"{len(tied)} contracts tie on {most} incidents but only "
        f"{len(task.accepted)} are accepted")


def test_a_tied_but_correct_answer_is_marked_correct():
    """The regression in one line: before this, naming any of the other
    nineteen was scored as a wrong answer."""
    task = next(t for t in TASKS if "most incident reports" in t.question)
    for reference in task.accepted:
        assert score(task.accepted, f"The answer is {reference}."), reference


def test_a_genuinely_wrong_reference_is_still_wrong():
    """The fix must not turn the question into a free mark."""
    task = next(t for t in TASKS if "most incident reports" in t.question)
    assert not score(task.accepted, "The answer is C-9999.")


def test_the_uniquely_answered_aggregates_stayed_unique():
    """T06 and T07 have exactly one right answer each, and must not have been
    widened by the fix to the one that did not."""
    world = build_world()

    dearest = max(c.penalty_per_hour for c in world.contracts)
    assert sum(1 for c in world.contracts
               if c.penalty_per_hour == dearest) == 1

    biggest = max(c.annual_value for c in world.contracts)
    assert sum(1 for c in world.contracts if c.annual_value == biggest) == 1

    for task in TASKS:
        if "highest late-delivery penalty" in task.question:
            assert len(task.accepted) == 1, task.accepted
        if "highest annual value" in task.question:
            assert len(task.accepted) == 1, task.accepted


def test_the_task_set_is_deterministic():
    """Two builds of the same world must produce the same questions, answers
    and accepted sets, or a fitness compared across generations means nothing."""
    first, second = build_tasks(), build_tasks()
    assert [(t.task_id, t.question, t.answer, t.accepted, t.hops) for t in first] \
        == [(t.task_id, t.question, t.answer, t.accepted, t.hops) for t in second]
