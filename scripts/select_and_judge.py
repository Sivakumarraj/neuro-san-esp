"""Choose on the select questions, judge on the judge questions: the paid runs.

    python scripts/select_and_judge.py networks                      # list them; nothing spent
    python scripts/select_and_judge.py mutants                       # the search's children of flat
    python scripts/select_and_judge.py measure 2cc4 --suite judge --out DIR --budget 0.40 --go
    python scripts/select_and_judge.py measure 2cc4 --promote Researcher --suite judge ...
    python scripts/select_and_judge.py summary results/headtohead/paid-2026-10

Stages 7 and 8 of the committed runs set ESP_SEARCH_RESULTS=10 (esp/eval/corpus_tool.py);
everything before them used the default of 3. Never compare across the two.

Every network here runs on the configured provider rung for rung (esp/serving.py).
A network is chosen on `meridian-select` (60 questions, one half of the company)
and only then measured on the first 100 judge questions (the other half), so the
questions it is judged on are never the ones it was chosen on.

`measure` measures one network, appending every chunk to <out>/<label>.jsonl as
it lands, so a run stopped by the cap or a crash resumes where it stopped. Run
two networks at the same time, into the same directory, to compare them on the
same hour of the provider. Without --go it prints the plan and spends nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp.config import bootstrap  # noqa: E402

# The children generated from flat in the committed paid search. The seed and
# the operator list are what produced them; changing either produces others.
SEARCH_SEED = 20261002
SEARCH_CHILDREN = 8
# reassign_model is left out: on this ladder it can only promote an agent to the
# 3.7x dearer rung, and the search was for a cheaper network.
SEARCH_OPERATORS = ["add_agent", "remove_agent", "rewire", "split_agent",
                    "merge_agents", "toggle_search"]

# Networks the committed Gemini search measured, named by what they are. Two of
# them, flat and solo, are hand-written seeds the search started from, not
# networks it found; `origin_of` says which.
COMMITTED = {
    "flat": "c3ee2e0c4a6156c5",
    "mergeA": "39fd57c6e575c298",
    "mergeB": "4c20dbbed359ab7a",
    "pair": "59b12db4896e63d8",
    "solo": "cbe128a617466fe9",
    "searchcoord": "cd06778520823bc3",
}


def _committed(genome_hash: str):
    from esp.eval import measurements
    from esp.genome.prune import without_copies

    rows = {r.genome_hash: r for r in measurements.load()}
    if genome_hash not in rows:
        raise SystemExit(f"no committed measurement {genome_hash}")
    return without_copies(rows[genome_hash].genome)


def mutants() -> dict[str, tuple[str, object]]:
    """The search's children of flat, by hash: (operator, genome)."""
    from esp.genome.mutations import InvalidMutant, mutate
    from esp.serving import measurable

    flat = measurable(_committed(COMMITTED["flat"])).genome
    rng = random.Random(SEARCH_SEED)
    seen, out = {flat.genome_hash()}, {}
    for _ in range(500):
        if len(out) == SEARCH_CHILDREN:
            break
        try:
            child, op = mutate(flat, rng, SEARCH_OPERATORS)
        except InvalidMutant:
            continue
        if child.genome_hash() not in seen:
            seen.add(child.genome_hash())
            out[child.genome_hash()] = (op, child)
    return out


# The committed measurements of the three hand-written seeds, by the hash they
# were measured under (tests/fixtures/cache/). Fixed, because a hash recomputed
# here carries whatever provider is configured and would match nothing.
SEED_HASHES = {"459ac1a66d925b0c": "designer_shaped",
               "c3ee2e0c4a6156c5": "flat_pair",
               "cbe128a617466fe9": "solo"}


def origin_of(genome_hash: str) -> str:
    """Where a committed network came from: a hand-written seed, or the search."""
    if genome_hash in SEED_HASHES:
        return f"{genome_hash} (hand-written seed:{SEED_HASHES[genome_hash]}), copies removed"
    return f"{genome_hash} (found by the Gemini search), copies removed"


def network(name: str):
    """A named network, a committed hash, or a child of flat by hash prefix."""
    from esp.genome.seeds import designer_shaped

    if name == "designer":
        return designer_shaped(), "seed:designer_shaped"
    if name in COMMITTED:
        return _committed(COMMITTED[name]), origin_of(COMMITTED[name])
    for genome_hash, (op, genome) in mutants().items():
        if genome_hash.startswith(name):
            return genome, f"{genome_hash}: {op} on flat (paid search)"
    return _committed(name), origin_of(name)


SUITES = ("select", "judge", "judge-plus")


def suite_tasks(name: str) -> list:
    """select: S01-S60. judge: J001-J100. judge-plus: P001-P100, the temporal,
    filtered, compare and unanswerable kinds no other set asks."""
    from esp.eval.judge_plus import JUDGE_PLUS
    from esp.eval.suites import JUDGE, SELECT

    return {"select": SELECT, "judge": JUDGE, "judge-plus": JUDGE_PLUS}[name]


def contender_for(name: str, label: str | None = None, promote: list[str] | None = None):
    """A named network on the configured provider, with any `promote`d agent
    moved to the top model rung after the move, so it lands on the provider's
    stronger model whatever rung the committed genome had it on."""
    from esp.evolve import headtohead as h2h
    from esp.genome.definition import MODEL_TIERS
    from esp.serving import measurable

    genome, origin = network(name)
    served = measurable(genome).genome
    for agent in promote or []:
        if agent not in served.agents:
            raise SystemExit(f"{name} has no agent {agent!r}: {sorted(served.agents)}")
        served.agents[agent].model = MODEL_TIERS[-1]
        origin += f", {agent} on {MODEL_TIERS[-1]}"
    return h2h.Contender(label or name, served, origin)


def summarise(rows: list[dict]) -> dict:
    """Per label: accuracy, tokens and dollars per question and per right answer."""
    by = defaultdict(lambda: {"asked": 0, "right": 0, "tokens": 0, "cost": 0.0,
                              "unfinished": 0, "repeats": set()})
    for r in rows:
        n = by[r["label"]]
        n["asked"] += len(r["results"])
        n["right"] += sum(x["correct"] for x in r["results"])
        n["unfinished"] += sum(x["infrastructure"] for x in r["results"])
        n["tokens"] += r["tokens"]
        n["cost"] += r["cost"]
        n["repeats"].add(r.get("run", 0))
    out = {}
    for label, n in sorted(by.items()):
        out[label] = {
            "question_runs": n["asked"], "correct": n["right"], "unfinished": n["unfinished"],
            "accuracy": round(n["right"] / n["asked"], 4),
            "tokens_per_question": round(n["tokens"] / n["asked"]),
            "cost_per_question": round(n["cost"] / n["asked"], 6),
            "tokens_per_correct": round(n["tokens"] / max(n["right"], 1)),
            "cost_per_correct": round(n["cost"] / max(n["right"], 1), 6),
            "cost": round(n["cost"], 4),
        }
    return out


def paired(rows: list[dict], a: str, b: str) -> dict:
    """Questions each was right on more often, and a two-sided sign test."""
    seen = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for x in r["results"]:
            if not x["infrastructure"]:
                seen[x["task_id"]][r["label"]].append(x["correct"])
    better = {a: [], b: []}
    for task_id, s in sorted(seen.items()):
        if a in s and b in s:
            ra, rb = sum(s[a]) / len(s[a]), sum(s[b]) / len(s[b])
            if ra != rb:
                better[a if ra > rb else b].append(task_id)
    n, k = len(better[a]) + len(better[b]), min(len(better[a]), len(better[b]))
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"better_on": better, "sign_test_p": round(p, 4)}


def load_rows(directory: Path) -> list[dict]:
    """Every chunk under a directory; a subdirectory per independent run."""
    rows = []
    for path in sorted(directory.rglob("*.jsonl")):
        run = path.parent.name
        rows += [{**json.loads(line), "run": run}
                 for line in path.read_text().splitlines() if line.strip()]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("networks")
    sub.add_parser("mutants")
    m = sub.add_parser("measure")
    m.add_argument("network", help="designer, a committed name, a hash, or a child's hash prefix")
    m.add_argument("--label", default=None)
    m.add_argument("--suite", choices=SUITES, default="select")
    m.add_argument("--promote", action="append", default=[], metavar="AGENT",
                   help="run this agent on the top model rung; repeat for more")
    m.add_argument("--repeats", type=int, default=1)
    m.add_argument("--budget", type=float, default=0.40, help="hard cap in dollars")
    m.add_argument("--out", required=True)
    m.add_argument("--go", action="store_true", help="spend: ask the provider")
    s = sub.add_parser("summary")
    s.add_argument("directory")
    args = parser.parse_args(argv)

    os.environ.setdefault("ESP_PIN_MODELS", "1")
    bootstrap()

    if args.command == "summary":
        root = Path(args.directory)
        report = {}
        for stage in sorted(p for p in root.iterdir() if p.is_dir()):
            rows = load_rows(stage)
            labels = sorted({r["label"] for r in rows})
            report[stage.name] = {"networks": summarise(rows)}
            if len(labels) == 2:
                report[stage.name]["paired"] = paired(rows, *labels)
        print(json.dumps(report, indent=2))
        return 0

    from esp.serving import measurable

    if args.command == "networks":
        for name in ["designer", *COMMITTED]:
            genome, origin = network(name)
            served = measurable(genome).genome
            print(f"{name:12} {served.genome_hash()}  {origin}")
        return 0
    if args.command == "mutants":
        for genome_hash, (op, genome) in mutants().items():
            shape = {n: (a.can_search, a.tools) for n, a in sorted(genome.agents.items())}
            print(f"{genome_hash}  {op:14} {shape}")
        return 0

    from esp.evolve import headtohead as h2h

    contender = contender_for(args.network, args.label, args.promote)
    origin = contender.origin
    plan = h2h.Plan(suite_tasks(args.suite), args.repeats, args.budget, [contender])
    print(f"{contender.label}: {contender.genome_hash}  {origin}")
    print(plan.describe())
    if not args.go:
        print("\nplan only: nothing spent. --go measures.")
        return 0
    log = h2h.Log(Path(args.out) / f"{contender.label}.jsonl")
    why = h2h.run(plan, h2h.live_measure, log)
    print(f"\nstopped: {why}")
    print(h2h.render(h2h.summarise(log, plan)))
    return 0 if why == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
