"""One wake of the ESP optimiser service.

Called on a schedule -- by neuro-san's own periodic interaction, by cron, or by
hand. It takes the lease, spends what today's provider budget allows, writes the
population down after every candidate, and returns.

It refuses to start on a configuration that would produce wrong numbers, which
is a worse outcome than not starting: a misconfigured evaluator scores every
candidate zero and the cache keeps that forever. `--check` runs those checks and
exits without evaluating anything.

Exit codes are meant for a scheduler:
    0  did something, or correctly did nothing
    1  could not run at all, or is misconfigured
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from esp.config import bootstrap
from esp.service.optimizer import wake
from esp.service.preflight import failures, run_checks
from esp.service.preflight import report as configuration_report
from esp.service.state import ServiceState


def main() -> int:
    root = str(Path(__file__).resolve().parent.parent.parent)
    os.environ.setdefault("PYTHONPATH", root)
    os.environ.setdefault("AGENT_TOOL_PATH", root)
    bootstrap()

    # Checked before anything is paid for. A misconfigured evaluator does not
    # crash -- it scores every candidate zero and caches the result, so the
    # search is taught that good topologies are bad. Refusing to start is the
    # cheaper failure by a wide margin.
    checks = run_checks()
    broken = failures(checks)
    if broken:
        print("refusing to start:", file=sys.stderr)
        print(configuration_report(checks), file=sys.stderr)
        return 1
    if "--check" in sys.argv:
        print(configuration_report(checks))
        return 0

    state = ServiceState.load()
    report = wake(state)

    if not report.acquired:
        # Not an error. A scheduler firing every fifteen minutes will overlap a
        # wake that is still evaluating, and declining is the correct response.
        print(f"skipped: {report.note}")
        return 0

    print(json.dumps({
        "wake": state.wakes,
        "generation": report.generation,
        "evaluated_this_wake": report.evaluated,
        "population": len(state.evaluated),
        "best_fitness": report.best_fitness,
        "improved": report.improved,
        "stopped_because": report.stopped_because,
        "exhausted_today": report.exhausted,
        "note": report.note,
    }, indent=2))

    best = state.best()
    if best:
        print(f"\nbest so far: {best.genome_hash}  {best.origin}  "
              f"acc={best.accuracy:.2f} tokens={best.tokens:,} "
              f"agents={best.agents} fitness={best.fitness:+.4f}")

    # Silence is the common outcome and the correct one. A service that reports
    # every wake trains its operator to stop reading it.
    if report.material():
        print("\nMATERIAL: a better topology was found -- worth telling someone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
