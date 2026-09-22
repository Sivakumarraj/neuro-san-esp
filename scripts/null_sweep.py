"""Is the exclusion decision stable, or does it depend on the seed you drew?

The gate in `esp/evolve/loop.py` excludes an outcome objective whose model
loses to its own permutation null. That verdict rests on a difference of two
cross-validated rank correlations, and on twelve samples both halves move with
the fold seed -- so a single (seed, shuffles) measurement is not evidence that
the gate fires for a reason. Two figures reported in this repository, a null of
-0.125 and one of -0.234, were the same quantity measured at different seeds,
and for a while both were in the documentation at once.

This sweeps fold seeds and shuffle counts and reports the median, the per-seed
range, and -- the column that matters -- how many seeds actually gate each
objective. Costs no provider budget: it re-uses the committed measurements.

    python scripts/null_sweep.py
    python scripts/null_sweep.py --seeds 50 --trials 12 40 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.eval import measurements
from esp.surrogate.outcomes import PREDICTED_OUTCOMES, Outcome, OutcomeSurrogate


def sweep(genomes: list, outcomes: list, seeds: range, trials: int) -> dict:
    """Per objective: every seed's spearman, null, margin, and gate verdict."""
    rows: dict[str, dict] = {
        name: {"rho": [], "null": [], "margin": [], "gated": 0}
        for name in PREDICTED_OUTCOMES
    }
    for seed in seeds:
        quality = OutcomeSurrogate(seed=seed).report_quality(
            genomes, outcomes, seed=seed, null_trials=trials)
        gated = set(quality.gated_outcomes())
        for name in PREDICTED_OUTCOMES:
            rows[name]["rho"].append(quality.per_outcome[name].spearman)
            rows[name]["null"].append(quality.nulls[name])
            rows[name]["margin"].append(quality.margin(name))
            rows[name]["gated"] += name in gated
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20,
                        help="how many cross-validation fold seeds to sweep")
    parser.add_argument("--trials", type=int, nargs="+", default=[12, 40],
                        help="shuffle counts per null; cost is linear in this")
    args = parser.parse_args()

    measured = measurements.load()
    genomes = [m.genome for m in measured]
    outcomes = [Outcome(accuracy=m.accuracy, tokens=m.tokens) for m in measured]
    seeds = range(args.seeds)

    print(f"{len(genomes)} measured networks, {args.seeds} fold seeds\n")

    def span(values: list[float]) -> str:
        return f"[{min(values):+.3f}, {max(values):+.3f}]"

    for trials in args.trials:
        rows = sweep(genomes, outcomes, seeds, trials)
        print(f"=== {trials} shuffles per null ===")
        for name in PREDICTED_OUTCOMES:
            row = rows[name]
            negative = sum(margin < 0 for margin in row["margin"])
            print(f"  {name:9} "
                  f"rho {np.median(row['rho']):+.3f} {span(row['rho'])}  "
                  f"null {np.median(row['null']):+.3f} {span(row['null'])}  "
                  f"margin {np.median(row['margin']):+.3f} {span(row['margin'])}  "
                  f"excluded {row['gated']}/{args.seeds}  "
                  f"margin negative {negative}/{args.seeds}")
        print()

    print("The exclusion counts are the reproducible result. The null is the\n"
          "noisiest quantity here, so no point estimate of it belongs in a\n"
          "document -- quote the margin as a range and the verdict as a count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
