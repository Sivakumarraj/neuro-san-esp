"""The hundred extra judge questions: every answer re-derived from the corpus
text by a solver that shares no code with the generator, nothing shared with
the select set, and "not stated" scored right only where it is right."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_bank import corpus
from test_suites import REF, records

from esp.eval.judge_plus import (
    ABSTENTIONS,
    JUDGE_200,
    JUDGE_PLUS,
    KINDS,
    NOT_STATED,
    build_plus,
    kind_of,
)
from esp.eval.suites import JUDGE, SELECT
from esp.eval.tasks import TASKS, score
from esp.measure import load_suite


def solve(question: str) -> str:
    depots, _, parsed_incidents = corpus()
    contracts, incidents = records()
    by_ref = {c["ref"]: c for c in contracts}
    q = question.split(" Answer with")[0]

    def of(code):
        refs = {c["ref"] for c in contracts if c["depot"] == code}
        return [i for i in incidents if i["contract"] in refs]

    if m := re.search(r"the (.+) of Meridian Logistics depot (D\d{2})\?$", q):
        return NOT_STATED       # no depot document records any such field
    if m := re.search(r"annual value of contract (C-\d{4})\?$", q):
        return str(by_ref[m[1]]["value"]) if m[1] in by_ref else NOT_STATED
    if m := re.search(r"hours late was incident (INC-\d{4})", q):
        found = parsed_incidents.get(m[1])
        return str(found["hours"]) if found else NOT_STATED
    if m := re.search(r"loading bays does Meridian Logistics depot (D\d{2}) have", q):
        return str(depots[m[1]]["bays"]) if m[1] in depots else NOT_STATED
    if m := re.search(r"depot (D\d{2}) or depot (D\d{2}) happened in (\d{4})", q):
        both = of(m[1]) + of(m[2])
        return str(sum(1 for i in both if i["year"] == int(m[3])))
    if m := re.search(r"depot (D\d{2}) happened in (\d{4})", q):
        return str(sum(1 for i in of(m[1]) if i["year"] == int(m[2])))
    if m := re.search(r"depot (D\d{2}) happened before (\d{4})", q):
        return str(sum(1 for i in of(m[1]) if i["year"] < int(m[2])))
    if m := re.search(r"incidents (in|before) (\d{4}) on contracts serviced by depot (D\d{2})", q):
        keep = (lambda y: y == int(m[2])) if m[1] == "in" else (lambda y: y < int(m[2]))
        return str(sum(i["hours"] for i in of(m[3]) if keep(i["year"])))
    if m := re.search(r"depot (D\d{2}) were more than (\d+) hours late", q):
        return str(sum(1 for i in of(m[1]) if i["hours"] > int(m[2])))
    if m := re.search(r"depot (D\d{2}) were caused by (.+)\?$", q):
        return str(sum(1 for i in of(m[1]) if i["cause"] == m[2]))
    if m := re.search(r"depot (D\d{2}) have a late-delivery penalty above (\d+)", q):
        return str(sum(1 for c in contracts if c["depot"] == m[1] and c["rate"] > int(m[2])))
    if m := re.search(r"depot (D\d{2}) whose late-delivery penalty is above (\d+)", q):
        return str(sum(c["value"] for c in contracts
                       if c["depot"] == m[1] and c["rate"] > int(m[2])))
    if m := re.search(r"annual value of contract (C-\d{4}) exceed that of contract (C-\d{4})", q):
        return str(by_ref[m[1]]["value"] - by_ref[m[2]]["value"])
    if m := re.search(r"loading bays does depot (D\d{2}) have than depot (D\d{2})", q):
        return str(depots[m[1]]["bays"] - depots[m[2]]["bays"])
    if m := re.search(r"How many of depots (.+) opened before (\d{4})", q):
        codes = re.findall(r"D\d{2}", m[1])
        return str(sum(1 for c in codes if depots[c]["opened"] < int(m[2])))
    raise AssertionError(f"no solver for: {q}")


def test_every_answer_is_what_the_documents_say():
    wrong = [(t.task_id, t.answer, solve(t.question)) for t in JUDGE_PLUS
             if solve(t.question) != t.answer]
    assert not wrong, wrong[:5]


def test_the_size_and_the_mix():
    assert len(JUDGE_PLUS) == 100 and len(JUDGE_200) == 200
    assert Counter(kind_of(t) for t in JUDGE_PLUS) == {k: 25 for k in KINDS}
    assert sum(t.answer == NOT_STATED for t in JUDGE_PLUS) == 25
    # A network that answers every count with the same small number must not
    # score well: no single answer covers more than a quarter of the set.
    assert max(Counter(t.answer for t in JUDGE_PLUS).values()) <= 25


def test_it_shares_no_entity_with_the_select_set():
    selected = {ref for t in SELECT for ref in REF.findall(t.question)}
    judged = {ref for t in JUDGE_PLUS for ref in REF.findall(t.question)}
    assert not selected & judged, selected & judged


def test_no_question_is_asked_twice_anywhere():
    everything = [t.question for t in (*SELECT, *JUDGE, *JUDGE_PLUS, *TASKS)]
    assert len(set(everything)) == len(everything)


def test_the_rule_is_on_every_question_so_it_gives_nothing_away():
    assert all("'not stated'" in t.question for t in JUDGE_PLUS)


def test_not_stated_is_right_only_where_it_is_right():
    unanswerable = next(t for t in JUDGE_PLUS if t.answer == NOT_STATED)
    answerable = next(t for t in JUDGE_PLUS if t.answer != NOT_STATED)
    assert score(unanswerable.accepted, "The documents do not say: not stated.")
    assert score(unanswerable.accepted, "There is no such contract in the records.")
    assert not score(unanswerable.accepted, "It was 12 hours late.")
    assert not score(answerable.accepted, "Not stated.")
    assert score(answerable.accepted, f"The answer is {answerable.answer}.")
    assert set(ABSTENTIONS) == set(unanswerable.accepted)


def test_it_is_deterministic():
    assert [t.question for t in build_plus()] == [t.question for t in JUDGE_PLUS]


def test_they_load_by_name():
    assert load_suite("meridian-judge-plus")[1] == JUDGE_PLUS
    assert load_suite("meridian-judge-200")[1] == JUDGE_200
    assert len(load_suite("meridian-judge-200:150")[1]) == 150
