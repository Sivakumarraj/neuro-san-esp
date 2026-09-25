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
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.eval import measurements
from esp.eval.runner import CACHE_DIR as DEFAULT_CACHE_DIR
from esp.genome.mutations import InvalidMutant, mutate
from esp.surrogate.outcomes import Outcome, OutcomeSurrogate
from esp.surrogate.predictor import MIN_SAMPLES

ROOT = Path(__file__).resolve().parent.parent

# Seed evaluations that were paid for once and committed, so that the free half
# of ESP is genuinely free to run. CI already points at these; the default path
# did not, which is why `make offline` worked only on a machine that had
# already spent an API budget.
FIXTURE_CACHE = measurements.FIXTURE_CACHE


def _shown(path: Path) -> str:
    """Repo-relative where possible: an absolute path from somebody else's
    machine is noise in a transcript that goes into the dossier."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _cached_measurements(cache_dir: Path) -> tuple[list, list[Outcome]]:
    """Real evaluations already paid for, matched back to their genomes.

    Reads through `esp.eval.measurements`, which is also what `make champion`
    and the web front end resolve the best network with. This held its own copy
    of the logic, and the copies disagreed about what counted as a usable
    measurement.

    Returns the *outcomes*, not the fitness. The Predictor is trained on what
    was measured -- accuracy and token cost -- and fitness is derived from its
    predictions afterwards by the same weighting the search selects on.
    """
    found = measurements.load(cache_dir)
    return ([m.genome for m in found],
            [Outcome(accuracy=m.accuracy, tokens=m.tokens) for m in found])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=int, default=2000,
                        help="candidates to generate and score against the surrogate")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--null-trials", type=int, default=12,
                        help="shuffles used to measure each objective's "
                             "permutation null; 0 skips it")
    parser.add_argument("--cache", default=None,
                        help="directory of cached evaluations "
                             "(default: the live .esp-cache)")
    args = parser.parse_args()

    cache_dir = Path(args.cache) if args.cache else DEFAULT_CACHE_DIR
    genomes, outcomes = _cached_measurements(cache_dir)

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
        fallback, fallback_outcomes = _cached_measurements(FIXTURE_CACHE)
        if len(fallback) >= 2:
            print(f"No local measurements in {_shown(cache_dir)} -- falling "
                  f"back to the committed ones in {_shown(FIXTURE_CACHE)}.")
            print("Run `make baseline` with a key to train on your own instead.\n")
            cache_dir, genomes, outcomes = (
                FIXTURE_CACHE, fallback, fallback_outcomes)

    if len(genomes) < 2:
        print(f"need at least 2 cached seed evaluations in {cache_dir}, "
              f"found {len(genomes)}. Run `make baseline` first, or point "
              "--cache at tests/fixtures/cache.", file=sys.stderr)
        return 1

    print(f"Phase B -- training the Predictor on {len(genomes)} real evaluations "
          f"from {_shown(cache_dir)}")
    print("  one model per outcome objective; fitness is derived from their "
          "predictions, not learned")
    surrogate = OutcomeSurrogate(seed=args.seed)
    quality = surrogate.report_quality(genomes, outcomes, seed=args.seed,
                                       null_trials=args.null_trials)
    print(f"  {quality}")
    if quality.nulls:
        print("  the null in brackets is what this same cross-validation "
              "returns on shuffled targets --")
        print("  on a population this small it is not zero, so the margin "
              "over it is the real figure:")
        for name in sorted(quality.nulls):
            margin = quality.margin(name)
            if margin is not None:
                print(f"    {name:9s} margin over its own null: {margin:+.3f}")
    # Printed on its own line because the combined figure will not show it.
    # Token cost on the committed population comes back reliably negative --
    # the Predictor orders candidates by cost backwards -- and the accuracy
    # term is large enough to carry the total into respectable territory
    # regardless. Scalarised into one model this was invisible, which is the
    # whole reason the surrogate now reports per objective.
    # Gated exactly as a service wake gates it. This script used to print the
    # warning and rank with the backwards model anyway, so `make offline` and
    # the service it demonstrates ordered the same candidates differently.
    gated = quality.gated_outcomes()
    for useless in quality.useless_outcomes():
        margin = quality.margin(useless)
        shown = "" if margin is None else f" (margin {margin:+.3f})"
        if useless in gated:
            print(f"  !! the {useless} model does not beat its own null{shown}. "
                  f"It is excluded from the ranking below, as a service wake "
                  f"excludes it.")
        else:
            print(f"  !! the {useless} model does not beat its own null{shown}, "
                  f"but no null was measured for it, so it is kept and its "
                  f"part of the ranking is suspect.")
    surrogate.fit(genomes, outcomes, gated=gated)

    print(f"\nPhase C -- evolving {args.pool} candidates against it")
    rng = random.Random(args.seed)
    started = time.monotonic()

    # Everything already paid for. Phase C exists to find candidates worth
    # buying, and a candidate that has been measured is not one: proposing it
    # spends an elite slot on a known answer, and it flatters the surrogate,
    # which "discovers" a network it was trained on. The measured champion was
    # filling four of the five places in the printed top five.
    known = {genome.genome_hash() for genome in genomes}
    seen: set[str] = set()

    candidates, rejected, repeats, already = [], 0, 0, 0
    # Bounded. The reachable space is finite -- with a small population and
    # strict validity, asking for 2,000 distinct mutants can simply be more
    # than exists, and an unbounded loop would spin instead of saying so.
    attempts = 0
    while len(candidates) < args.pool and attempts < args.pool * 40:
        attempts += 1
        parent = rng.choice(genomes)
        try:
            child, operator = mutate(parent, rng)
        except InvalidMutant:
            # Discarded, never repaired: a repaired mutant is a different mutant
            # than the operator produced.
            rejected += 1
            continue
        digest = child.genome_hash()
        if digest in known:
            already += 1
            continue
        if digest in seen:
            repeats += 1
            continue
        seen.add(digest)
        candidates.append((child, operator))

    bred = time.monotonic() - started

    scoring = time.monotonic()
    predictions = surrogate.predict([c for c, _ in candidates])
    elapsed = time.monotonic() - scoring
    ranked = sorted(zip(candidates, predictions, strict=True),
                    key=lambda pair: -pair[1])

    print(f"  {len(candidates)} distinct viable, {rejected} rejected as invalid, "
          f"{repeats} duplicates, {already} already measured")
    if len(candidates) < args.pool:
        # Said rather than silently delivering a smaller pool. A ratio quoted
        # over 2,000 candidates is wrong if only 600 existed.
        print(f"  NOTE: asked for {args.pool} and the reachable space yielded "
              f"{len(candidates)}. The numbers below describe what was "
              f"actually generated.")
    # Reported apart, because they are different claims. "Ranking is free" is
    # about the Predictor. Breeding and rejecting mutants is ordinary compute,
    # and once duplicates and already-measured genomes are filtered it is most
    # of the clock -- quoting the total as scoring time would overstate the one
    # number the whole method rests on.
    print(f"  bred in {bred:.2f}s, scored in {elapsed:.2f}s, "
          f"zero provider calls")
    print(f"  ~{elapsed / max(len(candidates), 1) * 1000:.3f} ms per candidate "
          f"to rank")

    if not surrogate.ranks():
        print(f"\n  !! The surrogate is UNTRAINED on {len(genomes)} samples "
              f"(needs {MIN_SAMPLES}). Every prediction below is the same "
              f"constant, so this is NOT a ranking -- it is the first "
              f"{args.top} candidates in generation order. Phase C's cost "
              f"claim holds; its selection claim does not, until there are "
              f"more real evaluations.")

    # Ties said out loud. Five candidates at the same predicted fitness are not
    # a top five; the order among them is the order they were bred in.
    if surrogate.ranks() and ranked:
        best = round(float(ranked[0][1]), 4)
        tied = sum(1 for _, predicted in ranked if round(float(predicted), 4) == best)
        if tied > 1:
            print(f"\n  {tied} candidates tie at the top ({best:+.4f}). The "
                  f"Predictor cannot separate them; among them the order below "
                  f"is generation order.")

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
