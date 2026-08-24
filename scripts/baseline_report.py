"""Build a report from cached evaluations alone, with no search.

An ESP run needs roughly 165 provider requests per candidate. The free tier
allows 500 per day per model, so a day's budget buys three candidates -- enough
to measure the seed topologies against each other, and not enough to evolve.
When the budget runs out mid-experiment the measurements already made are still
real, and this turns them into the same report the full loop produces.

It reports zero generations, because zero generations happened. The history it
writes is the measured data and nothing else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.eval.runner import CACHE_DIR
from esp.evolve.loop import WEIGHTS, Record, fitness
from esp.genome.seeds import SEEDS


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out.mkdir(parents=True, exist_ok=True)

    # Name each cached evaluation by the seed it came from, where we can.
    origin_of = {}
    for name, build in SEEDS.items():
        origin_of[build().genome_hash()] = f"seed:{name}"

    records: list[Record] = []
    for path in sorted(CACHE_DIR.glob("*.json")):
        raw = json.loads(path.read_text())

        class _E:                      # duck-types Evaluation for fitness()
            accuracy = raw["accuracy"]
            tokens = raw["tokens"]
            agents = raw["agents"]

        digest = raw["genome_hash"]
        records.append(Record(
            genome_hash=digest, generation=0,
            origin=origin_of.get(digest, "mutant"),
            fitness=round(fitness(_E()), 4), accuracy=raw["accuracy"],
            tokens=raw["tokens"], agents=raw["agents"], depth=raw["depth"],
            seconds=raw["seconds"], predicted=None,
        ))

    if not records:
        print("no cached evaluations to report", file=sys.stderr)
        return 1

    def dominated(a: Record, b: Record) -> bool:
        return (b.accuracy >= a.accuracy and b.tokens <= a.tokens
                and b.agents <= a.agents
                and (b.accuracy > a.accuracy or b.tokens < a.tokens
                     or b.agents < a.agents))

    pareto = [r for r in records if not any(dominated(r, o) for o in records)]

    payload = {
        "seed": 20260821,
        "weights": WEIGHTS,
        "real_evaluations": len(records),
        "surrogate_evaluations": 0,
        # No generation ran, so there is no cross-validated surrogate to report.
        # An empty list is the honest value; a fabricated correlation would not be.
        "surrogate_quality": [],
        "records": [r.__dict__ for r in records],
        "pareto": [r.__dict__ for r in pareto],
    }
    (out / "history.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{len(records)} measured evaluations -> {out}/history.json")
    for r in sorted(records, key=lambda r: -r.fitness):
        print(f"  {r.origin:22} acc={r.accuracy:.2f} tok={r.tokens:>7} "
              f"agents={r.agents} fit={r.fitness:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
