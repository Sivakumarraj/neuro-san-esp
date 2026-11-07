"""Better on all four at once: the test docs/ALL-FOUR.md fixed before it ran.

Step 1 chooses among three candidates on the 60 select questions, each run beside
the designer's shape at the same hour: a candidate is eligible when it is more
accurate there and uses fewer tokens and fewer dollars a question, and the most
accurate eligible one wins (a tie to the cheaper, then to fewer tokens). If none
is eligible the test stops.

Step 2 judges the winner against the designer's shape on judge (the test) and
fresh (the replication), on the five pass marks `esp.same_model.verdict` checks.

It reads the committed logs and nothing else, so it costs nothing and needs no key.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from esp import same_model

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "all-four"
SELECT = RESULTS / "select-2026-11"
SETS = {"judge": RESULTS / "judge-2026-11", "fresh": RESULTS / "fresh-2026-11"}

BASELINE = "designer"
CANDIDATES = ("c1", "c2", "c3")
HASHES = {BASELINE: "0378ca26cd9fe977", "c1": "ae41fbfaa513ed47",
          "c2": "e212a77c53bfa809", "c3": "f4e9e494442f9888"}
DESCRIBED = {"c1": "2cc4 on nano, the Calculator on the Coordinator",
             "c2": "2cc4, the Researcher on gpt-4o-mini",
             "c3": "2cc4, the Researcher on gpt-4o-mini, the Calculator on the Coordinator",
             BASELINE: "the designer's shape, every agent on nano"}
# Questions each set asks, and how often each question is asked.
SIZES = {"select": (60, 1), "judge": (100, 2), "fresh": (40, 2)}


def _read(path: Path) -> same_model.Network:
    # Candidates mix models, so the genome, which names every agent's model, is
    # what is checked; the width is checked as in the same-model test.
    return same_model.read(path, hashes=HASHES, model=None)


def _complete(network: same_model.Network, name: str) -> bool:
    questions, repeats = SIZES[name]
    return (len(network.by_question) == questions
            and all(len(runs) == repeats for runs in network.by_question.values()))


def missing_runs(network: same_model.Network, name: str) -> int:
    """How many of a set's planned runs a log lacks."""
    questions, repeats = SIZES[name]
    present = sum(min(len(runs), repeats) for runs in network.by_question.values())
    return questions * repeats - present


def worst_case(network: same_model.Network, name: str) -> same_model.Network:
    """The baseline completed as favourably to it as any outcome could be: every
    missing run right, at no tokens and no cost. A verdict that holds against
    this holds whatever the missing runs would have said.

    Only a missing repeat can be filled; a question never asked at all has no
    chunk to pair, so that log stays incomplete.
    """
    _, repeats = SIZES[name]
    out = copy.deepcopy(network)
    for runs in out.by_question.values():
        while len(runs) < repeats:
            runs.append(True)
            out.runs += 1
            out.right += 1
    for _tasks, usage in out.by_chunk.values():
        while len(usage) < repeats:
            usage.append((0.0, 0.0))
    return out


def screen(folder: Path = SELECT) -> dict:
    """Step 1: which candidates are eligible on select, and which one wins."""
    paths = {label: folder / f"{label}.jsonl" for label in (BASELINE, *CANDIDATES)}
    missing = [label for label, path in paths.items() if not path.exists()]
    if missing:
        return {"complete": False, "missing": missing}
    networks = {label: _read(path) for label, path in paths.items()}
    incomplete = [label for label, n in networks.items() if not _complete(n, "select")]
    if incomplete:
        return {"complete": False, "incomplete": incomplete}
    base = networks[BASELINE]
    eligible = [label for label in CANDIDATES
                if networks[label].accuracy > base.accuracy
                and networks[label].tokens_per_question < base.tokens_per_question
                and networks[label].dollars_per_question < base.dollars_per_question]
    winner = max(eligible, default=None,
                 key=lambda label: (networks[label].accuracy,
                                    -networks[label].dollars_per_question,
                                    -networks[label].tokens_per_question))
    return {"complete": True,
            "networks": {label: {**n.as_dict(), "described": DESCRIBED[label]}
                         for label, n in networks.items()},
            "eligible": eligible, "winner": winner}


# What the calculator can and cannot fix, read from a wrong reply. A question
# that computes a number (a penalty, a total, a difference) answered with a
# number within ten times of the truth is a near miss: the facts were mostly
# there and the working went wrong. A wrong count is enumeration, which no
# calculator fixes.
_COMPUTES = re.compile(r"penalty|total|combined|exceed|difference|apart|sum", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# The morning's all-nano 2cc4 on judge, the network the calculator was meant to help.
NANO_MORNING = ROOT / "results" / "structure" / "judge-nano-2026-11" / "2cc4-nano.jsonl"


def kind_of_miss(question: str, expected: str, reply: str) -> str:
    """"near miss", "wrong count" or "other" for one wrong reply."""
    # A count asks "How many ...", at the start of a sentence. "..., how many hours
    # late is that in total?" is a sum, and is read as one below.
    if "How many" in question:
        return "wrong count"
    truth = _NUMBER.findall(str(expected))
    said = _NUMBER.findall(reply or "")
    if _COMPUTES.search(question) and truth and said:
        expected_value = float(truth[0].replace(",", ""))
        given = float(said[-1].replace(",", ""))
        if expected_value and 0.1 <= abs(given / expected_value) <= 10:
            return "near miss"
    return "other"


def wrong_answers(path: Path, name: str = "judge") -> dict[str, int]:
    """Every wrong run in a log, by `kind_of_miss`."""
    from esp.eval.fresh import FRESH
    from esp.eval.suites import JUDGE

    tasks = {t.task_id: t for t in {"judge": JUDGE, "fresh": FRESH}[name]}
    kinds: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for result in json.loads(line)["results"]:
            if not result["correct"] or result["infrastructure"]:
                task = tasks[result["task_id"]]
                kinds[kind_of_miss(task.question, task.answer, result.get("answer", ""))] += 1
    return {kind: kinds.get(kind, 0) for kind in ("near miss", "wrong count", "other")}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyse(select: Path = SELECT, sets: dict[str, Path] | None = None) -> dict:
    """Both steps, on whatever complete logs exist."""
    out: dict = {"plan": "docs/ALL-FOUR.md", "alpha": same_model.ALPHA,
                 "select": screen(select), "sets": {}, "incomplete": [], "logs": {}}
    winner = out["select"].get("winner")
    paths = sorted(select.glob("*.jsonl")) if out["select"].get("complete") else []
    if winner:
        for name, folder in (sets or SETS).items():
            pair = {label: folder / f"{label}.jsonl" for label in (BASELINE, winner)}
            if not all(path.exists() for path in pair.values()):
                continue
            networks = {label: _read(path) for label, path in pair.items()}
            base = networks[BASELINE]
            questions, _ = SIZES[name]
            fillable = len(base.by_question) == questions
            if not _complete(networks[winner], name) or not fillable:
                out["incomplete"].append(name)
                continue
            result = same_model.verdict(base, networks[winner])
            short = missing_runs(base, name)
            result["missing_runs"] = {BASELINE: short}
            if short:
                # The verdict as measured, and against the most favourable
                # outcome the baseline's missing runs could have had.
                worst = same_model.verdict(worst_case(base, name), networks[winner])
                result["worst_case"] = worst
                result["holds_whatever_is_missing"] = worst["all_hold"]
            out["sets"][name] = result
            paths += list(pair.values())
    if "judge" in out["sets"]:
        folder = (sets or SETS)["judge"]
        out["wrong_answers_on_judge"] = {
            "2cc4 on nano, no calculator": wrong_answers(NANO_MORNING),
            winner: wrong_answers(folder / f"{winner}.jsonl"),
            BASELINE: wrong_answers(folder / f"{BASELINE}.jsonl")}
        paths.append(NANO_MORNING)
    for path in paths:
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        out["logs"][str(shown)] = sha256(path)
    return out
