"""Rebuild results/history.json from the measurements that are committed.

The summary and the detail were written by different code paths at different
times, and drifted: the README once described three seed topologies while
history.json beside it held eleven real evaluations, and the findings document
quoted token figures matching neither. Every published number in this
repository is read off these two files, so they have to agree by construction
rather than by anybody remembering.

This regenerates the records, the Pareto front and the evaluation count from
`tests/fixtures/cache`, which is the detail: one file per candidate holding its
per-task outcomes and the genome that produced them. What it deliberately does
not touch is the surrogate quality reports -- those are measurements taken
*during* a search, not properties of the population, and a rebuild that dropped
them would quietly delete the record of the Predictor ranking worse than
chance.

    python scripts/sync_history.py --check   # fail if they disagree
    python scripts/sync_history.py           # make them agree
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.eval import measurements
from esp.evolve.loop import History, Record

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "results" / "history.json"


def rebuild(cache_dir: Path | None = None) -> dict:
    """The payload history.json should hold, given what is measured."""
    found = measurements.load(cache_dir)
    if not found:
        raise SystemExit(
            f"no usable measurements in "
            f"{cache_dir or measurements.FIXTURE_CACHE} -- refusing to write "
            "a history of nothing")

    existing: dict = {}
    if HISTORY.exists():
        try:
            existing = json.loads(HISTORY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    # Per-candidate fields the cache does not carry. `seconds` is wall-clock,
    # recorded by the batch loop and not by a service wake; `generation` is a
    # property of the search that proposed the candidate. Both are kept from
    # the existing record where there is one rather than invented, and left at
    # their honest defaults where there is not.
    previous = {r["genome_hash"]: r for r in existing.get("records", [])}

    history = History()
    for record in found:
        prior = previous.get(record.genome_hash, {})
        history.records.append(Record(
            genome_hash=record.genome_hash,
            generation=int(prior.get("generation", 0)),
            origin=record.origin or prior.get("origin", ""),
            fitness=record.fitness,
            accuracy=record.accuracy,
            tokens=record.tokens,
            agents=record.agents,
            depth=record.depth,
            seconds=float(prior.get("seconds", 0.0)),
            predicted=prior.get("predicted"),
        ))
    history.records.sort(key=lambda r: r.genome_hash)
    history.real_evaluations = len(history.records)

    payload = dict(existing)
    payload.update({
        "real_evaluations": history.real_evaluations,
        "records": [asdict(r) for r in history.records],
        "pareto": [asdict(r) for r in history.pareto()],
    })
    # Kept, not recomputed: a quality report describes the Predictor at the
    # generation it was measured, and cannot be rederived from the population.
    payload.setdefault("surrogate_quality", [])
    payload.setdefault("surrogate_evaluations", 0)
    payload.setdefault("seed", 0)
    return payload


def _canonical(payload: dict) -> dict:
    """A form in which two histories of the same measurements compare equal."""
    canonical = {k: v for k, v in payload.items()
                 if k not in ("records", "pareto")}
    for key in ("records", "pareto"):
        canonical[key] = {r["genome_hash"]: r
                          for r in payload.get(key, []) or []}
    return canonical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report disagreement, write nothing")
    parser.add_argument("--cache", default=None,
                        help="measurements to rebuild from")
    args = parser.parse_args()

    payload = rebuild(Path(args.cache) if args.cache else None)
    rendered = json.dumps(payload, indent=2) + "\n"
    current = HISTORY.read_text(encoding="utf-8") if HISTORY.exists() else ""

    # Compared as data, not as bytes, and with the two lists keyed by genome
    # rather than by position. Key order, trailing whitespace and the order of
    # a Pareto front are not drift -- the committed front happened to be in
    # hash order while `History.pareto()` returns it best-first -- and a check
    # that reports those is a check nobody keeps running.
    try:
        unchanged = _canonical(json.loads(current)) == _canonical(payload)
    except json.JSONDecodeError:
        unchanged = False

    if unchanged:
        print(f"results/history.json already agrees with its "
              f"{payload['real_evaluations']} committed measurements")
        return 0

    if args.check:
        held = len(json.loads(current or "{}").get("records", []))
        print(f"results/history.json disagrees with the committed "
              f"measurements: {held} record(s) against "
              f"{payload['real_evaluations']} measured", file=sys.stderr)
        return 1

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(rendered, encoding="utf-8")
    best = max(payload["records"], key=lambda r: r["fitness"])
    print(f"wrote results/history.json from {payload['real_evaluations']} "
          f"measurements")
    print(f"best: {best['origin']} ({best['genome_hash']}) "
          f"acc={best['accuracy']:.4f} tokens={best['tokens']:,} "
          f"agents={best['agents']} fitness={best['fitness']:+.4f}")
    print(f"pareto front: {len(payload['pareto'])} network(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
