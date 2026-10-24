"""The fresh replication set: forty questions no network has been asked.

It is only a replication if every answer is right, nothing in it was asked
before, and nothing in it names what the select set names. The answers are
re-derived from the corpus text with the solvers tests/test_suites.py uses,
which share no code with the generator.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_bank import corpus, solve
from test_suites import AGGREGATE, REF, solve_aggregate

from esp.eval.fresh import FRESH, FRESH_SIZE, build_fresh
from esp.eval.judge_plus import JUDGE_PLUS
from esp.eval.suites import JUDGE, SELECT
from esp.eval.tasks import TASKS
from esp.measure import load_suite


def test_every_answer_is_what_the_documents_say():
    depots, contracts, incidents = corpus()
    wrong = []
    for task in FRESH:
        if AGGREGATE.search(task.question):
            derived = solve_aggregate(task.question)
        else:
            derived = solve(task.question, depots, contracts, incidents)
        if derived != task.answer:
            wrong.append((task.task_id, task.answer, derived))
    assert not wrong, f"{len(wrong)} answers disagree with the corpus: {wrong[:5]}"


def test_it_keeps_the_judge_sets_mix():
    assert len(FRESH) == FRESH_SIZE
    assert sum(1 for t in FRESH if AGGREGATE.search(t.question)) == 16


def test_no_question_was_ever_asked_before():
    before = {t.question for t in (*TASKS, *SELECT, *JUDGE, *JUDGE_PLUS)}
    assert not before & {t.question for t in FRESH}
    assert len({t.question for t in FRESH}) == len(FRESH)


def test_it_shares_no_entity_or_cause_and_year_with_the_select_set():
    def named(suite):
        return {ref for t in suite for ref in REF.findall(t.question)}

    def pairs(suite):
        return {m for t in suite for m in re.findall(r"in (\d{4}) (?:were )?caused by ([a-z -]+)",
                                                       t.question)}
    assert not named(SELECT) & named(FRESH)
    assert not pairs(SELECT) & pairs(FRESH)


def test_it_is_deterministic_and_loads_by_name():
    assert [(t.question, t.answer) for t in build_fresh()] == [(t.question, t.answer)
                                                               for t in FRESH]
    assert load_suite("meridian-fresh")[1] == FRESH
