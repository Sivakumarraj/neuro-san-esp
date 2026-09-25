"""Does a topology chosen on one set of tasks win on tasks it was not chosen on?

The weakest point in everything this repository reports is that a network is
selected on the same seventeen questions it is measured with. This answers the
question from the measurements already committed and costs no provider budget,
because each evaluation recorded the outcome of every individual task.

    python scripts/holdout_report.py
    python scripts/holdout_report.py --splits 500 --held-out 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.eval import holdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=holdout.SPLITS)
    parser.add_argument("--held-out", type=float,
                        default=holdout.HELD_OUT_FRACTION,
                        help="fraction of tasks withheld from selection")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    cache = Path(args.cache) if args.cache else None
    outcome = holdout.analyse(cache, splits=args.splits, seed=args.seed,
                              held_out_fraction=args.held_out)

    print(f"{outcome.population} measured networks, {outcome.splits} random "
          f"splits of {outcome.selection_tasks + outcome.held_out_tasks} tasks")
    print(f"  {outcome.selection_tasks} tasks select the winner, "
          f"{outcome.held_out_tasks} are held back to judge it\n")

    print("Can the winner be identified from half the tasks?")
    print(f"  it also ranked first on the held-out half : "
          f"{outcome.also_won_rate:>6.1%}")
    print(f"  it ranked in the held-out top three       : "
          f"{outcome.top_ranked_rate(3):>6.1%}")
    print(f"  its mean held-out rank                    : "
          f"{outcome.mean_rank:>6.2f} of {outcome.population}")
    print(f"  ordering carried across the split, accuracy only : "
          f"{outcome.mean_accuracy_rho:>+.3f}")
    print(f"  ordering carried across the split, full fitness   : "
          f"{outcome.mean_fitness_rho:>+.3f}")
    if outcome.also_won_rate < 0.5:
        print("\n  NO. Half of seventeen tasks does not identify which network "
              "is best.\n  Accuracy barely carries across the split, and the "
              "cost terms\n  break the ties that are left. Ranking individual "
              "networks needs a\n  larger task set, and that is a limitation of "
              "the measurement rather\n  than of the networks.")

    print("\nWas searching worth it at all?")
    print(f"  searched winner beat the designer's shape : "
          f"{outcome.beat_designer_rate:>6.1%} of splits")
    print(f"  mean held-out fitness margin over it      : "
          f"{outcome.mean_margin:>+6.4f}")
    print(f"  designer's shape mean held-out rank       : "
          f"{outcome.mean_designer_rank:>6.2f} of {outcome.population}")
    if outcome.beat_designer_rate > 0.8:
        print("\n  YES. The population evolution produced is better than the "
              "shape the\n  designer produces, and that holds on tasks which "
              "took no part in\n  choosing the winner. That is the claim this "
              "task set supports.")

    if outcome.evolved_count and outcome.seed_count:
        print(f"\nEvolved networks against hand-written ones "
              f"({outcome.evolved_count} evolved, {outcome.seed_count} seeds):")
        print(f"  mean held-out rank, evolved             : "
              f"{outcome.mean_evolved_rank:>6.2f} of {outcome.population}")
        print(f"  mean held-out rank, seeds               : "
              f"{outcome.mean_seed_rank:>6.2f} of {outcome.population}")
        print(f"  best evolved beat best seed, held out   : "
              f"{outcome.evolved_beat_seed_rate:>6.1%} of splits")
        print("    (each group's best chosen on the selection half, judged on "
              "the other)")
        print(f"  the same, best picked on the held-out half: "
              f"{outcome.evolved_beat_seed_in_sample_rate:>6.1%} of splits")
        print("    (NOT a held-out test: it looks at the half it judges on, and "
              "sets\n    the best of the evolved group against the best of "
              "the seeds.\n    Printed so the figure is never quoted as one.)")

    print("\nWhich network half the tasks picked:")
    for origin, count in sorted(outcome.winners.items(), key=lambda p: -p[1]):
        print(f"  {origin:24} {count:>5}  ({count / outcome.splits:>5.1%})")

    # Reported rather than assumed: a reader should be able to see that the
    # held-out halves are small enough for this to be the dominant caveat.
    print(f"\nAccuracy on {outcome.held_out_tasks} tasks moves in steps of "
          f"{1 / outcome.held_out_tasks:.3f}, and the whole cost penalty "
          f"spans\nless than one of those steps. That is why the ties decide "
          f"the order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
