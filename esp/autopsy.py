"""Why the networks miss the penalty questions, read from the answers they gave.

Questions that charge several incidents at their contracts' penalty rates are a
small share of judge and fresh and most of every network's wrong answers. The
world is generated, so each such question's parts are known exactly: for every
incident, its hours late and its contract's rate per hour. A wrong number can
then be checked against what a network would have said had it gone wrong in one
particular way:

* **rates only**: it found both contracts and answered from their rates alone,
  never multiplying by the hours ("by how much does the larger penalty exceed
  the smaller" answered with the difference of the two rates);
* **missed incidents**: the sum of some of the incidents' penalties and not
  others, which is what a search that never returned a document produces;
* **wrong by 10x or more**: a number no single slip explains;
* **arithmetic**: anything else, within a factor of ten of the truth.

The kinds are tried in that order and the first that fits is the one counted.
A number fitting "missed incidents" may still have been a slip that landed on
a partial sum by chance, so that count is an upper bound on retrieval failures,
and the rest says that the documents were found or the slip was elsewhere.
Only a trace (`esp/trace.py`) shows which agent produced the number.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter

from esp.eval.world import World, build_world

KINDS = ("right", "rates only", "missed incidents", "wrong by 10x or more", "arithmetic",
         "not a number")


def is_multi_penalty(question: str) -> bool:
    """A penalty question over two or more incidents."""
    return "penalty" in question and not question.startswith("Incident INC")


def parts(question: str, world: World) -> list[tuple[int, int]]:
    """(hours late, rate per hour) for every incident the question charges."""
    refs = re.findall(r"INC-\d+", question)
    contracts = {c.ref: c for c in world.contracts}
    if refs:
        by_ref = {i.ref: i for i in world.incidents}
        incidents = [by_ref[r] for r in refs]
    else:
        depot = re.search(r"depot (D\d\d)", question).group(1)
        incidents = [i for i in world.incidents
                     if contracts[i.contract_ref].depot_code == depot]
    return [(i.hours_late, contracts[i.contract_ref].penalty_per_hour) for i in incidents]


def explain(question: str, reply: str, right: bool, world: World) -> str:
    """Which of `KINDS` a reply to a multi-incident penalty question is."""
    if right:
        return "right"
    try:
        said = float(re.sub(r"[,$\s]", "", reply))
    except ValueError:
        return "not a number"
    terms = parts(question, world)
    owed = [hours * rate for hours, rate in terms]
    rates = [rate for _, rate in terms]
    truth = abs(owed[0] - owed[1]) if "exceed" in question else sum(owed)
    if said in {abs(rates[0] - rates[1]) if len(rates) == 2 else -1.0, float(sum(rates))}:
        return "rates only"
    for k in range(1, len(owed)):
        if any(said == sum(subset) for subset in itertools.combinations(owed, k)):
            return "missed incidents"
    if said <= 0 or not truth / 10 < said < truth * 10:
        return "wrong by 10x or more"
    return "arithmetic"


def tally(questions: dict[str, str], answers: dict[str, list[str]],
          right: dict[str, list[bool]], world: World | None = None) -> dict[str, int]:
    """Every run of every multi-incident penalty question, by kind."""
    world = world or build_world()
    counts: Counter[str] = Counter()
    for task, question in questions.items():
        if not is_multi_penalty(question) or task not in answers:
            continue
        for reply, ok in zip(answers[task], right[task], strict=True):
            counts[explain(question, reply, ok, world)] += 1
    return {kind: counts.get(kind, 0) for kind in KINDS}


def share_of_misses(questions: dict[str, str], right: dict[str, list[bool]]) -> dict:
    """How much of a network's wrong runs the penalty questions are."""
    penalty = [t for t, q in questions.items() if is_multi_penalty(q) and t in right]
    wrong = sum(r.count(False) for r in right.values())
    wrong_here = sum(right[t].count(False) for t in penalty)
    return {"questions": len(penalty), "of": len(right),
            "runs_right": sum(right[t].count(True) for t in penalty),
            "runs": sum(len(right[t]) for t in penalty),
            "wrong_runs": wrong_here, "all_wrong_runs": wrong}


def analyse() -> dict:
    """Both networks on judge and on fresh, from the committed logs."""
    from esp import uncertainty
    from esp.eval.fresh import FRESH
    from esp.eval.suites import JUDGE

    world = build_world()
    texts = {"judge": {t.task_id: t.question for t in JUDGE},
             "fresh": {t.task_id: t.question for t in FRESH}}
    out: dict = {}
    for name, pair in uncertainty.PAIRS.items():
        for role, path in zip(("cheap", "strong"), pair, strict=True):
            runs = uncertainty.load(uncertainty.RESULTS / path)
            out.setdefault(name, {})[role] = {
                "share": share_of_misses(texts[name], runs.right),
                "kinds": tally(texts[name], runs.answers, runs.right, world)}
    return out
