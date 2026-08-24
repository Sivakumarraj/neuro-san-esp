"""Render the service's accumulated population as a report.

Without this the service is invisible. `history.json` -- which every figure and
both PDFs read -- is written only by the batch run in esp/evolve/loop.py. The
service writes its population to `state/state.json` and nothing ever joined the
two, so an optimiser could accumulate results for weeks and the report would
still show the afternoon the batch script last managed to finish. Accumulating
is the entire point of running it as a service; a deliverable that cannot see
that accumulation defeats it.

What this deliberately does not do is invent the fields the service does not
record. A service wake does not track how many candidates the surrogate scored
on its way to choosing one, and it does not store a prediction alongside a
measurement. Those come out empty rather than plausible, because the report
reads them to decide whether to say a search happened -- and filling them in
would make it claim one that nobody can point to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Runnable straight from a clone, without `pip install -e` first. Half the
# scripts here already did this and half did not, so `serve_champion.py` --
# which the README puts in the quick start -- died with
# `ModuleNotFoundError: No module named 'esp'` on a fresh Codespace while
# its neighbours ran fine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.evolve.loop import WEIGHTS, History, Record
from esp.service.state import ServiceState


def to_history(state: ServiceState) -> History:
    history = History()
    for entry in state.evaluated:
        history.records.append(Record(
            genome_hash=entry.genome_hash,
            generation=entry.generation,
            origin=entry.origin,
            fitness=entry.fitness,
            accuracy=entry.accuracy,
            tokens=entry.tokens,
            agents=entry.agents,
            depth=entry.depth,
            # Not recorded per candidate by the service. Zero is honest here in
            # a way a guess would not be: nothing reads it except a column that
            # will show 0.0.
            seconds=0.0,
            predicted=None,
        ))
    history.real_evaluations = len(history.records)
    return history


def write(state_dir: Path, out_dir: Path) -> Path:
    state = ServiceState.load(state_dir)
    if not state.evaluated:
        raise SystemExit(
            f"no measurements in {state_dir}/state.json -- the service has not "
            "paid for a candidate yet, and a report of nothing is worse than no "
            "report")

    history = to_history(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": 0,
        "weights": WEIGHTS,
        "real_evaluations": history.real_evaluations,
        # See the module docstring: empty, not guessed. The report uses these to
        # decide whether to say a search ran.
        "surrogate_evaluations": 0,
        "surrogate_quality": [],
        "records": [record.__dict__ for record in history.records],
        "pareto": [record.__dict__ for record in history.pareto()],
        "source": "service state",
        "wakes": state.wakes,
        "last_wake": state.last_wake,
    }
    path = out_dir / "history.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="state",
                        help="the service's state directory")
    parser.add_argument("--out", default="results",
                        help="where to write history.json")
    args = parser.parse_args()

    path = write(Path(args.state), Path(args.out))
    history = json.loads(path.read_text(encoding="utf-8"))
    print(f"wrote {path} from {history['wakes']} wakes, "
          f"{history['real_evaluations']} real evaluations")
    for record in sorted(history["records"], key=lambda r: -r["fitness"]):
        print(f"  {record['genome_hash']}  {record['origin']:<26} "
              f"acc={record['accuracy']:.2f} tokens={record['tokens']:>7,} "
              f"agents={record['agents']} fitness={record['fitness']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
