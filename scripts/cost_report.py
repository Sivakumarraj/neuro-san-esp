"""Every committed network in tokens and in dollars, ranked both ways.

    python scripts/cost_report.py

Reads only committed measurements; spends nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr

from esp.eval.measurements import FIXTURE_CACHE, load
from esp.eval.pricing import priced
from esp.evolve.loop import scalarise


def rows():
    names = {m.genome_hash: m.name() for m in load()}
    out = []
    for p in priced(FIXTURE_CACHE):
        out.append({"name": names.get(p.genome_hash, p.genome_hash[:8]),
                    "hash": p.genome_hash, "accuracy": p.accuracy,
                    "tokens": p.tokens, "dollars": p.dollars,
                    "per_question": p.dollars_per_question,
                    "per_mtok": p.dollars_per_million_tokens,
                    "token_fitness": scalarise(p.accuracy, p.tokens, p.agents),
                    "dollar_fitness": p.fitness()})
    return sorted(out, key=lambda r: -r["dollar_fitness"])


def main() -> int:
    table = rows()
    designer = next(r for r in table if r["name"] == "seed:designer_shaped")
    print(f"{'network':30} {'acc':>6} {'tokens':>8} {'dollars':>8} {'$/Mtok':>7} "
          f"{'tokens vs designer':>19} {'dollars vs designer':>20}")
    for r in table:
        dt = r["tokens"] / designer["tokens"] - 1
        dd = r["dollars"] / designer["dollars"] - 1
        print(f"{r['name'][:30]:30} {r['accuracy']:6.4f} {r['tokens']:8,d} "
              f"{r['dollars']:8.4f} {r['per_mtok']:7.3f} {dt:+19.0%} {dd:+20.0%}")
    rho = spearmanr([r["token_fitness"] for r in table],
                    [r["dollar_fitness"] for r in table]).statistic
    cheapest = min((r for r in table if r["accuracy"] >= 0.88),
                   key=lambda r: r["dollars"])
    print(f"\nRank agreement between the token and dollar fitness: {rho:+.3f}")
    print(f"Cheapest network at 0.88 or better: {cheapest['name']} "
          f"(${cheapest['dollars']:.4f} for 17 questions)")
    print("Dollars are the provider's own accounting for each run, on the "
          "prices of the day it was measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
