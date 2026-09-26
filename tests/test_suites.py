"""The select and judge suites. A wrong answer, or a question the two share,
would decide the scale-up experiment before any network was asked anything.

Answers are re-derived from the corpus *text* by solvers that share no code
with the generator: the bank's for joins, and one here for aggregates. The
judge set is only a held-out test if nothing in it helped choose, so the two
sets are checked for any shared entity or aggregate parameter.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_bank import corpus, solve

from esp.eval.suites import JUDGE, JUDGE_SIZE, SELECT, SELECT_SIZE, build_suites
from esp.eval.tasks import TASKS
from esp.eval.world import build_world, documents
from esp.measure import SuiteError, load_suite

REF = re.compile(r"\b(?:D\d{2}|C-\d{4}|INC-\d{4})\b")
AGGREGATE = re.compile(r"serviced by depot D\d{2}\?|by depot D\d{2},|depot D\d{2} at its own"
                       r"|incidents in \d{4}|incident in \d{4}")


def records() -> tuple[list[dict], list[dict]]:
    """Contracts and incidents with every field, read from the documents."""
    contracts, incidents = [], []
    for name, text in documents(build_world()).items():
        fields = dict(line.split(": ", 1) for line in text.splitlines()[1:])
        if name.startswith("contract-"):
            contracts.append({"ref": text.split()[1], "depot": fields["Serviced by depot"],
                              "value": int(fields["Annual value"]),
                              "rate": int(fields["Late-delivery penalty"].split()[0])})
        elif name.startswith("incident-"):
            head = text.splitlines()[0]
            incidents.append({"contract": fields["Affected contract"],
                              "hours": int(fields["Hours late"]), "cause": fields["Cause"],
                              "year": int(re.search(r"\((\d{4})\)", head).group(1))})
    return contracts, incidents


def solve_aggregate(question: str) -> str:
    contracts, incidents = records()
    rate = {c["ref"]: c["rate"] for c in contracts}
    depot = re.search(r"depot (D\d{2})", question)
    if depot:
        mine = [c for c in contracts if c["depot"] == depot.group(1)]
        refs = {c["ref"] for c in mine}
        hits = [i for i in incidents if i["contract"] in refs]
        if "How many Meridian Logistics contracts" in question:
            return str(len(mine))
        if "combined annual value" in question:
            return str(sum(c["value"] for c in mine))
        if "How many incidents" in question:
            return str(len(hits))
        if "hours late" in question:
            return str(sum(i["hours"] for i in hits))
        return str(sum(i["hours"] * rate[i["contract"]] for i in hits))
    year = int(re.search(r"in (\d{4})", question).group(1))
    cause = re.search(r"caused by (.+?)[?,]", question).group(1)
    hits = [i for i in incidents if i["year"] == year and i["cause"] == cause]
    if question.startswith("How many incidents"):
        return str(len(hits))
    return str(sum(i["hours"] for i in hits))


@pytest.mark.parametrize("suite", [SELECT, JUDGE], ids=["select", "judge"])
def test_every_answer_is_what_the_documents_say(suite):
    depots, contracts, incidents = corpus()
    wrong = []
    for task in suite:
        if AGGREGATE.search(task.question):
            derived = solve_aggregate(task.question)
        else:
            derived = solve(task.question, depots, contracts, incidents)
        if derived != task.answer:
            wrong.append((task.task_id, task.answer, derived))
    assert not wrong, f"{len(wrong)} answers disagree with the corpus: {wrong[:5]}"


def test_the_sizes_and_the_aggregate_share():
    assert (len(SELECT), len(JUDGE)) == (SELECT_SIZE, JUDGE_SIZE)
    for suite, expected in ((SELECT, 24), (JUDGE, 40)):
        assert sum(1 for t in suite if AGGREGATE.search(t.question)) == expected


def test_no_question_is_asked_twice_anywhere():
    everything = [t.question for t in (*SELECT, *JUDGE, *TASKS)]
    assert len(set(everything)) == len(everything)


def test_the_judge_set_shares_no_entity_with_the_select_set():
    """The judge is a held-out test only if nothing in it helped choose."""
    named = {name: {ref for t in suite for ref in REF.findall(t.question)}
             for name, suite in (("select", SELECT), ("judge", JUDGE))}
    assert not named["select"] & named["judge"], named["select"] & named["judge"]


def test_no_cause_and_year_is_asked_about_in_both():
    def pairs(suite):
        return {m for t in suite for m in re.findall(r"in (\d{4}) (?:were )?caused by ([a-z -]+)",
                                                       t.question)}
    assert not pairs(SELECT) & pairs(JUDGE)


def test_a_calibration_prefix_covers_every_kind():
    first = SELECT[:20]
    assert any(AGGREGATE.search(t.question) for t in first)
    assert {1, 2, 3, 4, 6, 9} <= {t.hops for t in first if not AGGREGATE.search(t.question)}


def test_they_are_deterministic():
    select, judge = build_suites()
    assert [t.question for t in select] == [t.question for t in SELECT]
    assert [t.answer for t in judge] == [t.answer for t in JUDGE]


def test_they_load_by_name():
    assert len(load_suite("meridian-select")[1]) == SELECT_SIZE
    assert len(load_suite("meridian-judge")[1]) == JUDGE_SIZE
    name, tasks = load_suite("meridian-select:20")
    assert name == "meridian-select:20" and tasks == SELECT[:20]
    with pytest.raises(SuiteError):
        load_suite("meridian-judge:101")
