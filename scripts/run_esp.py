"""Entry point for an ESP run.

    python scripts/run_esp.py --generations 4 --pool 400 --elite 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from esp.config import bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ESP over neuro-san agent networks.")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--pool", type=int, default=400,
                        help="candidates scored by the surrogate each generation")
    parser.add_argument("--elite", type=int, default=4,
                        help="candidates evaluated for real each generation")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    bootstrap()

    root = str(Path(__file__).resolve().parent.parent)
    os.environ.setdefault("PYTHONPATH", root)
    os.environ.setdefault("AGENT_TOOL_PATH", root)

    from esp.eval.ratelimit import stats
    from esp.evolve.loop import Evolution

    evolution = Evolution(seed=args.seed, surrogate_pool=args.pool,
                          real_per_generation=args.elite, out_dir=args.out)
    history = evolution.run(generations=args.generations)

    best = history.best()
    print("\n" + "=" * 72)
    print(f"real evaluations      {history.real_evaluations}")
    print(f"surrogate evaluations {history.surrogate_evaluations}")
    print(f"llm calls / retries   {stats()['calls']} / {stats()['retries']}")
    if best:
        print(f"best                  {best.genome_hash} from {best.origin} "
              f"(gen {best.generation})")
        print(f"                      accuracy={best.accuracy} tokens={best.tokens} "
              f"agents={best.agents} fitness={best.fitness:+.4f}")
    print(f"written to            {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
