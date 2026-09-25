"""The held-out bank. A wrong answer in it would poison every measurement on it.

The strongest test here re-derives every answer from the corpus *text* -- the
documents a network actually reads -- with a solver that shares no code with the
generator. "Correct by construction" was already wrong once in this repository
(twenty tied answers scored as one), so construction is checked, not trusted.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from esp.eval.bank import BANK, MIX, build_bank, to_jsonl
from esp.eval.tasks import TASKS, score
from esp.eval.world import build_world, documents
from esp.measure import SuiteError, load_suite

ROOT = Path(__file__).resolve().parent.parent


def corpus() -> tuple[dict, dict, dict]:
    """Depots, contracts and incidents as parsed from the documents themselves."""
    depots, contracts, incidents = {}, {}, {}
    for name, text in documents(build_world()).items():
        fields = dict(line.split(": ", 1) for line in text.splitlines()[1:])
        if name.startswith("depot-"):
            code = text.split()[3]
            depots[code] = {"city": fields["Location"], "manager": fields["Depot manager"],
                            "bays": int(fields["Loading bays"]), "opened": int(fields["Opened"])}
        elif name.startswith("contract-"):
            ref = text.split()[1]
            contracts[ref] = {"client": fields["Client"], "depot": fields["Serviced by depot"],
                              "goods": fields["Goods"],
                              "rate": int(fields["Late-delivery penalty"].split()[0])}
        else:
            ref = text.split()[1]
            incidents[ref] = {"contract": fields["Affected contract"],
                              "hours": int(fields["Hours late"]), "cause": fields["Cause"]}
    return depots, contracts, incidents


def solve(question: str, depots: dict, contracts: dict, incidents: dict) -> str:
    """Answer a bank question from the parsed corpus, by reading its wording."""
    incs = re.findall(r"INC-\d{4}", question)
    refs = re.findall(r"C-\d{4}", question)
    codes = re.findall(r"depot (D\d{2})", question)
    clients = [c["client"] for c in contracts.values() if c["client"] in question]

    def depot_of_incident(ref):
        return depots[contracts[incidents[ref]["contract"]]["depot"]]

    def owed(ref):
        return incidents[ref]["hours"] * contracts[incidents[ref]["contract"]]["rate"]

    if len(incs) == 3:
        return str(sum(depot_of_incident(r)["bays"] for r in incs))
    if len(incs) == 2:
        if "exceed" in question:
            return str(abs(owed(incs[0]) - owed(incs[1])))
        if "total penalty" in question:
            return str(owed(incs[0]) + owed(incs[1]))
        a, b = (depot_of_incident(r) for r in incs)
        if "years apart" in question:
            return str(abs(a["opened"] - b["opened"]))
        return str(a["bays"] + b["bays"])
    if len(clients) == 2:
        by_client = {c["client"]: c for c in contracts.values()}
        a, b = (depots[by_client[name]["depot"]] for name in clients)
        return str(abs(a["opened"] - b["opened"]))
    if len(incs) == 1:
        ref = incs[0]
        if "cause" in question:
            return incidents[ref]["cause"]
        if "penalty" in question:
            return str(owed(ref))
        contract = contracts[incidents[ref]["contract"]]
        if "client" in question:
            return contract["client"]
        if "goods" in question:
            return contract["goods"]
        home = depots[contract["depot"]]
        for word, field in (("city", "city"), ("manager", "manager"),
                            ("loading bays", "bays"), ("open", "opened")):
            if word in question:
                return str(home[field])
    if clients:
        contract = next(c for c in contracts.values() if c["client"] == clients[0])
        return str(depots[contract["depot"]]["opened"])
    if refs:
        contract = contracts[refs[0]]
        if "Which client" in question:
            return contract["client"]
        if "goods" in question:
            return contract["goods"]
        home = depots[contract["depot"]]
        return home["city"] if "city" in question else home["manager"]
    if codes:
        home = depots[codes[0]]
        for word, field in (("manager", "manager"), ("city", "city"),
                            ("loading bays", "bays")):
            if word in question:
                return str(home[field])
    raise AssertionError(f"no reading for: {question}")


def test_every_answer_is_what_the_documents_say():
    depots, contracts, incidents = corpus()
    wrong = [(t.task_id, t.answer, solve(t.question, depots, contracts, incidents))
             for t in BANK
             if solve(t.question, depots, contracts, incidents) != t.answer]
    assert not wrong, f"{len(wrong)} answers disagree with the corpus: {wrong[:5]}"


def test_it_is_the_size_and_mix_it_claims():
    assert len(BANK) == 250
    assert Counter(t.hops for t in BANK) == Counter(MIX)


def test_it_is_deterministic():
    again = build_bank()
    assert [(t.question, t.answer) for t in again] == [(t.question, t.answer) for t in BANK]


def test_questions_and_ids_are_unique():
    assert len({t.question for t in BANK}) == len(BANK)
    assert len({t.task_id for t in BANK}) == len(BANK)


def test_no_question_touches_an_entity_the_benchmark_names():
    """Held out means held out: the committed networks were selected on the
    seventeen, so the bank asks about entities those never touched."""
    named = set()
    for task in TASKS:
        named.update(re.findall(r"\b(?:D\d{2}|C-\d{4}|INC-\d{4})\b", task.question))
    for task in BANK:
        mentioned = set(re.findall(r"\b(?:D\d{2}|C-\d{4}|INC-\d{4})\b", task.question))
        assert not mentioned & named, f"{task.task_id} names {mentioned & named}"


def test_a_small_prefix_still_asks_every_depth():
    """A free-tier key affords a few dozen questions a day, and the first forty
    must not be forty lookups."""
    assert set(t.hops for t in BANK[:40]) == set(MIX)


def test_the_right_answer_scores_and_a_near_miss_does_not():
    for task in BANK:
        assert score(task.answer, f"The answer is {task.answer}.")
    numeric = [t for t in BANK if t.answer.isdigit()]
    assert not score(numeric[0].answer, f"D{numeric[0].answer} and INC-44{numeric[0].answer}")


def test_the_committed_question_file_matches_the_generator():
    """`make bank` writes it. A hand edit, or a generator change without it,
    would have two different banks under one name."""
    committed = (ROOT / "tasks" / "meridian_bank.jsonl").read_text(encoding="utf-8")
    assert committed == to_jsonl(BANK)
    rows = [json.loads(line) for line in committed.splitlines()]
    assert rows[0] == {"id": "B001", "question": BANK[0].question,
                       "answers": [BANK[0].answer], "hops": BANK[0].hops}


def test_the_bank_loads_by_name_whole_or_in_part():
    name, tasks = load_suite("meridian-bank")
    assert name == "meridian-bank" and len(tasks) == 250
    name, tasks = load_suite("meridian-bank:40")
    assert name == "meridian-bank:40" and [t.task_id for t in tasks] == [
        t.task_id for t in BANK[:40]]


@pytest.mark.parametrize("bad", ["meridian-bank:0", "meridian-bank:251", "meridian-bank:x"])
def test_a_bad_count_is_refused_before_anything_runs(bad):
    with pytest.raises(SuiteError):
        load_suite(bad)
