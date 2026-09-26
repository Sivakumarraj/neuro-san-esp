"""The same-budget experiment. Prints the plan and its price unless told to go.

    make experiment               # plan only: nothing is spent
    make experiment GO=1          # run it, resuming from the cache if stopped
    python scripts/experiment.py --budget 20 --go

See esp/evolve/experiment.py for what it measures and why.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp.config import bootstrap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget", type=int, default=40,
                        help="paid candidates per arm after the seed population")
    parser.add_argument("--per-generation", type=int, default=4)
    parser.add_argument("--arms", default="predictor,random")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", default="results/experiment")
    parser.add_argument("--go", action="store_true", help="spend: without it, plan only")
    parser.add_argument("--force", action="store_true",
                        help="continue even if calibration finds the select set too easy")
    args = parser.parse_args()
    bootstrap()
    os.environ.setdefault("PYTHONPATH", str(ROOT))
    os.environ.setdefault("AGENT_TOOL_PATH", str(ROOT))

    from esp.eval.suites import JUDGE, SELECT
    from esp.evolve import experiment

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = [a for a in arms if a not in experiment.ARMS]
    if unknown:
        print(f"unknown arm(s): {unknown}; choose from {experiment.ARMS}", file=sys.stderr)
        return 2
    plan = experiment.Plan(select=SELECT, judge=JUDGE, budget=args.budget,
                           per_generation=args.per_generation, arms=arms, seed=args.seed)
    print(plan.describe())
    if not args.go:
        print("\nPlan only. Nothing was spent. Add --go (make experiment GO=1) to run it.")
        return 0

    from esp.service.preflight import failures, report, run_checks
    checks = run_checks(live=True)
    print("\npreflight\n" + report(checks))
    if failures(checks):
        print("preflight failed -- nothing was spent", file=sys.stderr)
        return 1

    summary = experiment.run(plan, ROOT / args.out, force=args.force)
    print("\n" + experiment.report(summary))
    print(f"\nwritten to {args.out}/summary.json")
    return 1 if summary.stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
