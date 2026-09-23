"""Every published figure about the Predictor, regenerated from committed data.

The README, docs/FINDINGS.md and the dossier quote the surrogate's
cross-validated rank correlations, its permutation nulls, the margins between
them and how often each objective is excluded. Those figures used to be
produced by ad-hoc sweeps that were never committed, so nobody could check
them -- and one of them, the "three input orderings" sweep, could not be
reproduced exactly even by the person who ran it. This is that sweep, written
down, together with every other one the documents quote.

    python scripts/surrogate_figures.py            # everything, a few minutes
    python scripts/surrogate_figures.py --quick    # 5 seeds, for a smoke check

No key, no network, no provider calls: it reads tests/fixtures/cache only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.eval import measurements
from esp.surrogate.outcomes import PREDICTED_OUTCOMES, Outcome, OutcomeSurrogate
from esp.surrogate.predictor import Surrogate


def span(values) -> str:
    return f"[{min(values):+.3f} … {max(values):+.3f}]"


def per_objective(genomes, outcomes, seeds: range, trials: int) -> dict:
    """The per-objective table: spearman, null, margin, exclusions, derived."""
    rows = {name: {"rho": [], "null": [], "margin": [], "excluded": 0}
            for name in PREDICTED_OUTCOMES}
    derived = []
    for seed in seeds:
        quality = OutcomeSurrogate(seed=seed).report_quality(
            genomes, outcomes, seed=seed, null_trials=trials)
        excluded = set(quality.gated_outcomes())
        derived.append(quality.derived.spearman)
        for name in PREDICTED_OUTCOMES:
            rows[name]["rho"].append(quality.per_outcome[name].spearman)
            rows[name]["null"].append(quality.nulls[name])
            rows[name]["margin"].append(quality.margin(name))
            rows[name]["excluded"] += name in excluded
    return {"rows": rows, "derived": derived}


def scalar_sweep(records, seeds: range) -> list[float]:
    """The single scalarised surrogate over three input orderings.

    The orderings are the committed load order, its reverse, and the order of
    the genome hashes. Order matters at all only because tied fitness values
    would otherwise be broken by position, which is exactly what the sweep
    exists to rule out.
    """
    orderings = [list(records), list(reversed(records)),
                 sorted(records, key=lambda r: r.genome_hash)]
    found = []
    for ordered in orderings:
        genomes = [r.genome for r in ordered]
        fitness = [r.fitness for r in ordered]
        for seed in seeds:
            found.append(Surrogate(seed=seed).report_quality(
                genomes, fitness, seed=seed).spearman)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="5 seeds instead of 20/40")
    args = parser.parse_args()

    records = measurements.load()
    genomes = [r.genome for r in records]
    outcomes = [Outcome(accuracy=r.accuracy, tokens=r.tokens) for r in records]
    seeds20 = range(5 if args.quick else 20)
    seeds40 = range(5 if args.quick else 40)
    print(f"{len(records)} committed measurements\n")

    for trials in (12, 40):
        result = per_objective(genomes, outcomes, seeds20, trials)
        print(f"== per objective, {len(seeds20)} fold seeds, {trials} shuffles per null ==")
        for name in PREDICTED_OUTCOMES:
            row = result["rows"][name]
            print(f"  {name:8}  spearman {np.median(row['rho']):+.3f} {span(row['rho'])}"
                  f"  null {np.median(row['null']):+.3f} {span(row['null'])}"
                  f"  margin {np.median(row['margin']):+.3f} {span(row['margin'])}"
                  f"  margin<0 {sum(m < 0 for m in row['margin'])}/{len(seeds20)}"
                  f"  excluded {row['excluded']}/{len(seeds20)}")
        derived = result["derived"]
        print(f"  derived fitness  spearman {np.median(derived):+.3f} {span(derived)}\n")

    every = scalar_sweep(records, seeds40)
    print(f"== the single scalarised surrogate, 3 orderings x {len(seeds40)} seeds ==")
    print(f"  all twelve:  median {np.median(every):+.3f} {span(every)}  "
          f"above +0.2 in {sum(v > 0.2 for v in every)}/{len(every)}")
    # The twelfth measurement is the one the Predictor chose, so "eleven" is
    # the population as it stood when it made that choice.
    best = max(records, key=lambda r: r.fitness)
    eleven = scalar_sweep([r for r in records if r is not best], seeds40)
    print(f"  first eleven: median {np.median(eleven):+.3f} {span(eleven)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
