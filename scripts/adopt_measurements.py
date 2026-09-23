"""Start a deployment from the measurements this repository already paid for.

A fresh clone has an empty service state, so its first wakes go on measuring
the three seeds -- eleven candidates' worth of provider budget has already been
spent on exactly that, and the results are committed. Re-paying for them costs
about four days of a free tier and teaches the Predictor nothing it could not
have been handed.

This adopts them: the committed evaluations become this deployment's starting
population, and the evaluation cache is primed so the runner does not re-measure
a genome whose answer is already known. The first wake afterwards therefore
skips the seeds, trains the Predictor on eleven real samples, and spends its
budget on candidates nobody has measured yet.

Nothing here fabricates a measurement. Every record adopted carries the genome
it was measured on and is refused if that genome does not rebuild to the hash it
was filed under.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.config import provider_for
from esp.eval import measurements
from esp.eval.runner import CACHE_DIR
from esp.genome.definition import DEFAULT_MODEL
from esp.service.state import Evaluated, ServiceState

ROOT = Path(__file__).resolve().parent.parent


def adopt(state_dir: Path, cache_dir: Path, source: Path | None = None,
          force: bool = False) -> tuple[int, int]:
    """Returns (records adopted, cache files primed)."""
    found = measurements.load(source)
    if not found:
        raise SystemExit(
            f"no usable measurements in {source or measurements.FIXTURE_CACHE}")

    # Measurements do not cross providers. The committed twelve were taken on
    # Gemini; adopting them into a deployment configured for Claude would breed
    # children that inherit a Gemini default the run holds no key for, and would
    # rank Claude evaluations against Gemini ones in the same population.
    configured = provider_for(DEFAULT_MODEL)
    measured_on = {provider_for(record.genome.default_model) for record in found}
    if measured_on != {configured}:
        raise SystemExit(
            f"these measurements were taken on {', '.join(sorted(map(str, measured_on)))} "
            f"and this deployment is configured for {configured} ({DEFAULT_MODEL}). "
            "Measurements do not cross providers -- a fitness measured on one model "
            "does not describe the same network on another. Measure the seeds on "
            "your own provider instead: make baseline")

    state = ServiceState.load(state_dir)
    if state.evaluated and not force:
        raise SystemExit(
            f"{state_dir}/state.json already holds {len(state.evaluated)} "
            "measurements -- pass --force to add these to it anyway")

    known = state.seen()
    adopted = 0
    for record in found:
        if record.genome_hash in known:
            continue
        state.add(Evaluated(
            genome_hash=record.genome_hash, origin=record.origin,
            fitness=record.fitness, accuracy=record.accuracy,
            tokens=record.tokens, agents=record.agents, depth=record.depth,
            generation=0, measured_at="",
            model=record.genome.default_model,
            genome=record.genome.canonical()))
        adopted += 1

    # Generation 1, not 0: the adopted population is a completed round, and a
    # wake that thought it was still on generation 0 would file its own
    # candidates alongside measurements it did not take.
    state.generation = max(state.generation, 1)
    state.save(state_dir)

    # Prime the evaluation cache so the runner does not pay again for an answer
    # that is committed. Keyed by genome hash, which is how the runner looks up.
    primed = 0
    directory = source or measurements.FIXTURE_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    for record in found:
        origin = directory / f"{record.genome_hash}.json"
        target = cache_dir / f"{record.genome_hash}.json"
        if origin.exists() and not target.exists():
            shutil.copy2(origin, target)
            primed += 1

    return adopted, primed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="state")
    parser.add_argument("--cache", default=None,
                        help=f"evaluation cache to prime (default {CACHE_DIR})")
    parser.add_argument("--source", default=None,
                        help="measurements to adopt (default: the committed ones)")
    parser.add_argument("--force", action="store_true",
                        help="add to a state that already holds measurements")
    args = parser.parse_args()

    adopted, primed = adopt(
        Path(args.state), Path(args.cache) if args.cache else CACHE_DIR,
        Path(args.source) if args.source else None, args.force)

    state = ServiceState.load(Path(args.state))
    best = state.best()
    print(f"adopted {adopted} measurement(s) into {args.state}/state.json "
          f"({len(state.evaluated)} in the population), primed {primed} "
          f"cache entr{'y' if primed == 1 else 'ies'}")
    if best:
        print(f"best: {best.origin} ({best.genome_hash}) "
              f"acc={best.accuracy:.4f} tokens={best.tokens:,} "
              f"agents={best.agents} fitness={best.fitness:+.4f}")
    print(f"\nThe next wake will skip the seeds and train on "
          f"{len(state.evaluated)} real samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
