"""The pool benchmark. Prints the plan and its price unless told otherwise.

    make pool                      # plan only: nothing is spent
    make pool REHEARSE=1           # the whole run against a simulated provider, $0
    make pool GO=1                 # measure the pool and judge the finalists (paid)
    python scripts/pool_benchmark.py --compare results/pool/pool.json

See esp/evolve/pool.py for what it measures and why.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp.config import bootstrap  # noqa: E402


def show(summary: dict) -> str:
    lines = [f"{summary['replicates']} replicate searches over a pool of "
             f"{summary['pool']} (pool best {summary['pool_best']:+.4f}), against "
             f"{summary['permutations']} permuted pools of {summary['per_permutation']}",
             "  each difference is tested against the same difference with outcomes "
             "shuffled across networks; p is Holm-adjusted across every strategy and "
             "budget; 'closes' is the share of random choice's regret removed"]
    for strategy, rows in summary["strategies"].items():
        lines.append(f"  {strategy}")
        for budget, row in rows.items():
            cell = f"    @{budget:<3} found {row['mean_best']:+.4f}"
            if "vs_random" in row:
                low, high = row["vs_random_95"]
                verdict = ("better than random" if row["better_than_random"] else
                           "worse than random" if row["worse_than_random"] else
                           "no difference established")
                cell += (f"  {row['vs_random']:+.4f} vs random [{low:+.4f}, {high:+.4f}]"
                         f" (null {row['null']:+.4f})"
                         f"  closes {row['regret_closed']:+.0%}  p={row['p_holm']:.3f}"
                         f"  {verdict}")
            lines.append(cell)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=120, help="networks in the pool")
    parser.add_argument("--select", type=int, default=60, help="select questions used")
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=19,
                        help="permuted pools the comparison is tested against")
    parser.add_argument("--out", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--go", action="store_true", help="spend: measure and judge")
    mode.add_argument("--rehearse", action="store_true", help="simulated provider, $0")
    mode.add_argument("--compare", metavar="POOL_JSON", help="replicates over a measured pool")
    args = parser.parse_args(argv)
    bootstrap()
    os.environ.setdefault("PYTHONPATH", str(ROOT))
    os.environ.setdefault("AGENT_TOOL_PATH", str(ROOT))

    from esp.eval.suites import SELECT
    from esp.evolve import pool

    plan = pool.Plan(select=SELECT[:args.select], size=args.size, finalists=args.finalists)

    if args.compare:
        members = pool.load(Path(args.compare))
        summary = pool.compare(members, plan.select, replicates=args.replicates,
                               permutations=args.permutations)
        print(show(summary))
        return 0

    print(plan.describe())
    if not (args.go or args.rehearse):
        print("\nPlan only. Nothing was spent. REHEARSE=1 runs it for $0 against a "
              "simulated provider; GO=1 spends.")
        return 0

    if args.rehearse:
        return rehearse(plan, pool, Path(args.out or ROOT / "results" / "rehearsal"),
                        args.replicates, args.permutations)

    from esp.service.preflight import failures, report, run_checks
    checks = run_checks(live=True)
    print("\npreflight\n" + report(checks))
    if failures(checks):
        print("preflight failed -- nothing was spent", file=sys.stderr)
        return 1
    out = Path(args.out or ROOT / "results" / "pool")
    measured = pool.measure(plan, out)
    if measured["stopped"]:
        print(f"\nstopped: {measured['stopped']}\nrun again to resume; nothing is paid twice")
        return 1
    members = pool.load(out / "pool.json")
    summary = pool.compare(members, plan.select, replicates=args.replicates,
                               permutations=args.permutations)
    (out / "compare.json").write_text(json.dumps(summary, indent=1))
    print("\n" + show(summary))
    judged = pool.judge(members, plan, out)
    print(f"\njudged {len(judged['judged'])} networks on {len(plan.judge)} questions; "
          f"written to {out}")
    return 1 if judged["stopped"] else 0


def rehearse(plan, pool, out: Path, replicates: int, permutations: int = 19) -> int:
    """The whole pipeline, with a simulated provider in place of the model."""
    from esp.eval import rehearsal, runner
    from esp.genome.seeds import SEEDS

    genomes = pool.breed(plan.size, plan.seed)
    provider = rehearsal.SimulatedProvider([*genomes, SEEDS["designer_shaped"]()])
    saved = runner.run_suite, runner.CACHE_DIR, runner.NETWORK_DIR
    with tempfile.TemporaryDirectory() as scratch:
        runner.run_suite = provider
        runner.CACHE_DIR = Path(scratch) / "cache"
        runner.NETWORK_DIR = Path(scratch) / "networks"
        try:
            pool.measure(plan, out, genomes)
            members = pool.load(out / "pool.json")
            summary = pool.compare(members, plan.select, replicates=replicates,
                                   permutations=permutations)
            judged = pool.judge(members, plan, out)
        finally:
            runner.run_suite, runner.CACHE_DIR, runner.NETWORK_DIR = saved
    summary["simulated"] = True
    (out / "compare.json").write_text(json.dumps(summary, indent=1))
    totals = {"simulated": True, "question_runs": provider.question_runs,
              "tokens": provider.tokens, "dollars": round(provider.dollars, 2),
              "networks_judged": len(judged["judged"])}
    (out / "totals.json").write_text(json.dumps(totals, indent=1))
    print("\nREHEARSAL -- a simulated provider, not a result\n" + show(summary))
    print(f"\n{provider.question_runs:,} question-runs, {provider.tokens / 1e6:,.0f}M "
          f"simulated tokens, ${provider.dollars:,.2f} simulated; $0 spent")
    better = [f"{s} @{b}" for s, rows in summary["strategies"].items()
              for b, r in rows.items() if r.get("better_than_random")]
    print("beats random choice (Holm-adjusted p below 0.05): " + (", ".join(better) or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
