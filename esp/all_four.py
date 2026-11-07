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

import hashlib
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
            if not all(_complete(n, name) for n in networks.values()):
                out["incomplete"].append(name)
                continue
            out["sets"][name] = same_model.verdict(networks[BASELINE], networks[winner])
            paths += list(pair.values())
    for path in paths:
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        out["logs"][str(shown)] = sha256(path)
    return out
