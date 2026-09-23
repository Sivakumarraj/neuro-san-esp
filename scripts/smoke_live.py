"""The whole path, end to end, on the key in this machine's .env.

Everything else in the suite is offline by design, which means nothing in it
proves that a real provider answers a real multi-hop question through a real
neuro-san session. This does, in the order a person would hit it: the preflight,
then the four showcase questions put to the champion exactly as the web page
and the studio serve it -- on the configured provider, with the front man
explaining its answer.

    make smoke
    python scripts/smoke_live.py --questions 1     # one question, cheapest check

It spends real money: each question is ten or more model calls. It exits 0 when
every question came back with an answer and tokens were spent, and prints which
answers matched the benchmark. A wrong answer is reported, not failed on -- the
champion's score on a provider it was not measured on is exactly what this
cannot settle -- but no answers at all, or zero tokens, is a broken pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENT_TOOL_PATH", str(ROOT))
os.environ.setdefault("PYTHONPATH", str(ROOT))

from esp.eval import measurements  # noqa: E402
from esp.eval.runner import _ask, _total_tokens, write_network  # noqa: E402
from esp.eval.tasks import score  # noqa: E402
from esp.service.preflight import failures, report, run_checks  # noqa: E402
from esp.serving import SHOWCASE, display_question, presentable  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=int, default=len(SHOWCASE),
                        help="how many showcase questions to ask (each is paid for)")
    args = parser.parse_args()

    checks = run_checks(live=True)
    print("preflight\n" + report(checks) + "\n")
    if failures(checks):
        print("preflight failed -- nothing was asked", file=sys.stderr)
        return 1

    champion = measurements.best()
    if champion is None:
        print("no committed measurements to serve", file=sys.stderr)
        return 1
    served = presentable(champion.genome)
    hocon = str(write_network(served.genome))
    router = served.genome.agents[served.genome.top].model or served.genome.default_model
    print(f"champion {champion.name()} ({champion.genome_hash}), serving on "
          f"{served.provider}: router {router}, workers {served.genome.default_model}")
    print(f"  {served.note(champion.genome.default_model)}\n")

    results = []
    for task in SHOWCASE[:args.questions]:
        question = display_question(task)
        started = time.monotonic()
        try:
            answer, accounting, seconds = _ask(hocon, question)
            error = ""
        except Exception as exc:                    # reported, then judged below
            answer, accounting, seconds = "", {}, time.monotonic() - started
            error = f"{type(exc).__name__}: {exc}"[:300]
        tokens = _total_tokens(accounting)
        correct = bool(answer) and score(task.accepted, answer)
        results.append({"task": task.task_id, "hops": task.hops, "tokens": tokens,
                        "seconds": round(seconds, 1), "correct": correct,
                        "answered": bool(answer), "error": error})
        verdict = "CORRECT" if correct else ("WRONG" if answer else "NO ANSWER")
        print(f"[{verdict:9}] {task.task_id} ({task.hops} hop{'s' * (task.hops != 1)}) "
              f"{question}\n            expected {task.answer!r}; {tokens:,} tokens, "
              f"{seconds:.0f}s{'; ' + error if error else ''}")
        if answer:
            print("            " + answer.strip().replace("\n", "\n            ")[:1200])
        print()

    answered = sum(r["answered"] for r in results)
    spent = sum(r["tokens"] for r in results)
    print(json.dumps({"asked": len(results), "answered": answered,
                      "correct": sum(r["correct"] for r in results),
                      "tokens": spent, "provider": served.provider}))
    if answered < len(results) or spent == 0:
        print("\nbroken pipeline: a question got no answer, or no model was called",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
