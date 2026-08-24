"""Phase B and C only: train the Predictor, then evolve against it.

Zero provider calls. This is the half of ESP that is supposed to be free, and
making it runnable on its own is not a convenience -- it is the claim under
test. If evolving thousands of candidates against a surrogate really costs
nothing, that has to be demonstrable with no API key, no budget and no network,
and the numbers it prints are the ratio the whole method rests on.

What it cannot do is tell you whether the winners are actually good. Only a real
evaluation can, and that is Phase D. This ranks; it does not measure.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.eval.runner import CACHE_DIR as DEFAULT_CACHE_DIR
from esp.evolve.loop import fitness
from esp.genome.mutations import InvalidMutant, mutate
from esp.genome.seeds import SEEDS
from esp.surrogate.predictor import MIN_SAMPLES, Surrogate

ROOT = Path(__file__).resolve().parent.parent

# Seed evaluations that were paid for once and committed, so that the free half
# of ESP is genuinely free to run. CI already points at these; the default path
# did not, which is why `make offline` worked only on a machine that had
# already spent an API budget.
FIXTURE_CACHE = ROOT / "tests" / "fixtures" / "cache"


def _shown(path: Path) -> str:
    """Repo-relative where possible: an absolute path from somebody else's
    machine is noise in a transcript that goes into the dossier."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _cached_measurements(cache_dir: Path) -> tuple[list, list[float]]:
    """Real evaluations already paid for, matched back to their genomes.

    Only seeds can be recovered: a mutant's genome is not stored beside its
    score, and reconstructing one from a hash is not possible. That bounds how
    much training data this mode can ever have, which is worth being explicit
    about rather than quietly training on three points.
    """
    by_hash = {}
    for build in SEEDS.values():
        genome = build()
        by_hash[genome.genome_hash()] = genome

    genomes, values = [], []
    for path in sorted(cache_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        genome = by_hash.get(raw["genome_hash"])
        if genome is None:
            continue

        class _E:
            accuracy = raw["accuracy"]
            tokens = raw["tokens"]
            agents = raw["agents"]

        genomes.append(genome)
        values.append(fitness(_E()))
    return genomes, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=int, default=2000,
                        help="candidates to generate and score against the surrogate")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--cache", default=None,
                        help="directory of cached evaluations "
                             "(default: the live .esp-cache)")
    args = parser.parse_args()

    cache_dir = Path(args.cache) if args.cache else DEFAULT_CACHE_DIR
    genomes, values = _cached_measurements(cache_dir)

    # `make offline` is the one command the README hands a new reader to see the
    # core claim without a key, and on a fresh clone it exited 1: the live cache
    # is gitignored, so it is empty until `make baseline` -- which needs the key
    # the reader was promised they would not need. CI never caught it because CI
    # passes --cache explicitly and so ran a different command than the
    # documented one. Falling back to the committed measurements fixes that, but
    # it is announced rather than silent: training on somebody else's numbers
    # while the reader believes they are their own is the sort of quiet
    # substitution this project refuses everywhere else.
    if len(genomes) < 2 and args.cache is None and cache_dir != FIXTURE_CACHE:
        fallback, fallback_values = _cached_measurements(FIXTURE_CACHE)
        if len(fallback) >= 2:
            print(f"No local measurements in {_shown(cache_dir)} -- falling "
                  f"back to the committed ones in {_shown(FIXTURE_CACHE)}.")
            print("Run `make baseline` with a key to train on your own instead.\n")
            cache_dir, genomes, values = FIXTURE_CACHE, fallback, fallback_values

    if len(genomes) < 2:
        print(f"need at least 2 cached seed evaluations in {cache_dir}, "
              f"found {len(genomes)}. Run `make baseline` first, or point "
              "--cache at tests/fixtures/cache.", file=sys.stderr)
        return 1

    print(f"Phase B -- training the Predictor on {len(genomes)} real evaluations "
          f"from {_shown(cache_dir)}")
    surrogate = Surrogate(seed=args.seed)
    quality = surrogate.report_quality(genomes, values, seed=args.seed)
    print(f"  {quality}")
    surrogate.fit(genomes, values)

    print(f"\nPhase C -- evolving {args.pool} candidates against it")
    rng = random.Random(args.seed)
    started = time.monotonic()
    candidates, rejected = [], 0
    while len(candidates) < args.pool:
        parent = rng.choice(genomes)
        try:
            child, operator = mutate(parent, rng)
        except InvalidMutant:
            # Discarded, never repaired: a repaired mutant is a different mutant
            # than the operator produced.
            rejected += 1
            continue
        candidates.append((child, operator))

    predictions = surrogate.predict([c for c, _ in candidates])
    elapsed = time.monotonic() - started
    ranked = sorted(zip(candidates, predictions, strict=True),
                    key=lambda pair: -pair[1])

    print(f"  {len(candidates)} viable, {rejected} rejected as invalid")
    print(f"  scored in {elapsed:.2f}s with zero provider calls")
    print(f"  ~{elapsed / max(len(candidates), 1) * 1000:.3f} ms per candidate")

    if not surrogate.ranks():
        print(f"\n  !! The surrogate is UNTRAINED on {len(genomes)} samples "
              f"(needs {MIN_SAMPLES}). Every prediction below is the same "
              f"constant, so this is NOT a ranking -- it is the first "
              f"{args.top} candidates in generation order. Phase C's cost "
              f"claim holds; its selection claim does not, until there are "
              f"more real evaluations.")

    heading = ("Top" if surrogate.ranks() else "First")
    ordering = ("by predicted fitness" if surrogate.ranks()
                else "in generation order -- all tied, see above")
    print(f"\n{heading} {args.top} {ordering}:")
    for (genome, operator), predicted in ranked[:args.top]:
        print(f"  {predicted:+.4f}  {genome.genome_hash()}  "
              f"agents={len(genome.reachable())} depth={genome.depth()}  "
              f"via {operator}")

    # The ratio the method rests on. A real evaluation of one candidate takes
    # minutes of model time; the figure above is the alternative.
    print(f"\nPhase C scored {len(candidates)} candidates in {elapsed:.1f}s. "
          "The same number evaluated for real, at roughly 8 minutes each, "
          f"would take about {len(candidates) * 8 / 60:.0f} hours.")
    if surrogate.ranks():
        print("Phase C ranks. It does not measure -- only Phase D does.")
    else:
        print("Phase C generated and scored them for nothing, which is the cost "
              "claim. It did not rank them, and it never measures -- only "
              "Phase D does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
