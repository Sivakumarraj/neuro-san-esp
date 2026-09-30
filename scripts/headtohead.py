"""The head-to-head: designer against champion, on unseen questions, capped.

    make headtohead                 # the plan and its price; nothing is spent
    make headtohead REHEARSE=1      # the whole run against a simulated provider, $0
    make headtohead GO=1            # measure for real, stopping at the cap

Models are pinned for the run (ESP_PIN_MODELS=1): an exhausted model stops it
rather than being swapped. See esp/evolve/headtohead.py for why.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=int, default=100,
                        help="judge questions asked, from the first")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--budget", type=float, default=4.50,
                        help="hard cap in dollars across the whole run")
    parser.add_argument("--out", default=None, help="directory for the log and summary")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--go", action="store_true", help="spend: ask the provider")
    mode.add_argument("--rehearse", action="store_true",
                      help="run everything against a simulated provider, $0")
    args = parser.parse_args(argv)

    os.environ.setdefault("ESP_PIN_MODELS", "1")
    bootstrap()

    from esp.config import configured_provider
    from esp.eval.judge_plus import JUDGE_200
    from esp.evolve import headtohead as h2h

    if not 0 < args.questions <= len(JUDGE_200):
        parser.error(f"--questions takes 1 to {len(JUDGE_200)}")
    provider = configured_provider()
    plan = h2h.Plan(JUDGE_200[:args.questions], args.repeats, args.budget,
                    h2h.contenders())
    print(f"provider: {provider}")
    print(plan.describe())
    if not (args.go or args.rehearse):
        print("\nplan only: nothing spent. GO=1 measures, REHEARSE=1 simulates.")
        return 0

    if args.rehearse:
        out = Path(args.out or tempfile.mkdtemp(prefix="headtohead-rehearsal-"))
        measure = h2h.rehearsal_measure
    else:
        out = Path(args.out or ROOT / "results" / "headtohead" / provider)
        measure = h2h.live_measure
    out.mkdir(parents=True, exist_ok=True)
    log = h2h.Log(out / "runs.jsonl")
    print(f"\nlog: {log.path} ({len(log.rows)} chunk(s) already measured)")
    stopped = h2h.run(plan, measure, log)
    summary = h2h.summarise(log, plan)
    summary["provider"] = provider
    summary["stopped"] = stopped
    summary["rehearsal"] = bool(args.rehearse)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"\nstopped: {stopped}")
    print(h2h.render(summary))
    if args.rehearse:
        print("\nrehearsal: simulated answers, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
