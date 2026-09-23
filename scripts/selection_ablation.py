"""Does the Predictor pick better networks than a coin would? Offline.

Every other figure about the Predictor is a rank correlation, and a rank
correlation is not what the search uses it for. The search uses it to *choose*:
of the candidates it could pay to measure, which one. So this asks that
question directly, on networks the Predictor was not trained on:

    for every way of holding out three of the twelve committed measurements,
      train on the other nine exactly as a service wake does (cross-validate,
      gate objectives that lose to their own null, fit), pick the held-out
      network it predicts best, and see how that pick really scored.

Against the same held-out triples it reports a random picker (exact, not
simulated), the Predictor with the gate switched off, and with token cost
always excluded, so the gate's own contribution is visible. Regret is the
fitness the pick left on the table against the best network in its triple.
It also counts how often the ungated models put two held-out networks in the
right order on each objective -- the per-objective question the gate asks,
asked on networks the models never saw.

The 220 triples overlap -- each network appears in 55 of them -- so they are
not 220 independent trials, and nothing here is a claim about networks outside
this population. The p-value answers one narrow question: how often a random
picker, on these same triples, does at least as well.

    python scripts/selection_ablation.py            # every triple; ~45 CPU-minutes
    python scripts/selection_ablation.py --quick    # 30 triples spread across the set

The gate's permutation null is most of the cost, so the triples run in
parallel, one process per core.

No key, no network, no provider calls: it reads tests/fixtures/cache only.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.eval import measurements
from esp.surrogate.outcomes import NULL_TRIALS, Outcome, OutcomeSurrogate

_records: list = []


def pick(train, held, gate, seed: int):
    """The Predictor's choice among `held` (None if it cannot rank), the
    objectives excluded, and the fitted Predictor.

    `gate` is "measured" for the gate as a wake runs it, or a fixed list of
    objectives to exclude.
    """
    genomes = [r.genome for r in train]
    outcomes = [Outcome(accuracy=r.accuracy, tokens=r.tokens) for r in train]
    surrogate = OutcomeSurrogate(seed=seed)
    gated = list(gate) if gate != "measured" else surrogate.report_quality(
        genomes, outcomes, seed=seed, null_trials=NULL_TRIALS).gated_outcomes()
    surrogate.fit(genomes, outcomes, gated=gated)
    if not surrogate.ranks():
        return None, gated, surrogate
    return (int(np.argmax(surrogate.predict([r.genome for r in held]))), gated,
            surrogate)


def ordered(held, predicted, objective: str) -> list[bool]:
    """For each held-out pair that truly differs, whether the prediction
    puts it the right way round. Predicted ties count as wrong."""
    truth = [getattr(r, objective) for r in held]
    return [bool(np.sign(truth[a] - truth[b])
                 == np.sign(predicted[a] - predicted[b]))
            for a, b in itertools.combinations(range(len(held)), 2)
            if truth[a] != truth[b]]


def _load() -> None:
    _records[:] = measurements.load()


def one_split(work) -> dict:
    """Score the three pickers on one held-out set."""
    held_idx, seed = work
    held = [_records[i] for i in held_idx]
    train = [r for i, r in enumerate(_records) if i not in held_idx]
    fitness = np.array([r.fitness for r in held])
    best = fitness.max()
    chance = (float(np.mean(fitness == best)), float(best - fitness.mean()))
    out = {"random": chance}
    for name, gate in (("gated", "measured"), ("ungated", ()),
                       ("no_tokens", ("tokens",))):
        chosen, gated, surrogate = pick(train, held, gate, seed)
        if name == "gated":
            out["excluded"] = gated
        if name == "ungated":
            predicted = surrogate.predict_outcomes([r.genome for r in held])
            out["pairs"] = {k: ordered(held, predicted[k], k)
                            for k in ("accuracy", "tokens")}
        # A Predictor that cannot rank leaves the choice to chance, so it is
        # scored as the random picker would be.
        out[name] = ((*chance, False) if chosen is None else
                     (float(fitness[chosen] == best), float(best - fitness[chosen]), True))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = measurements.load()
    splits = list(itertools.combinations(range(len(records)), args.holdout))
    if args.quick:
        # Evenly spaced, not the first thirty: those all hold out network 0.
        splits = splits[::max(1, len(splits) // 30)][:30]

    work = [(held, args.seed) for held in splits]
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1,
                             initializer=_load) as pool:
        results = list(pool.map(one_split, work, chunksize=4))

    random_hit = [r["random"][0] for r in results]
    random_regret = [r["random"][1] for r in results]
    rows = {name: [r[name] for r in results]
            for name in ("gated", "ungated", "no_tokens")}

    n = len(splits)
    print(f"{len(records)} committed measurements, {n} held-out sets of "
          f"{args.holdout}, trained on the other {len(records) - args.holdout}\n")
    print(f"  {"picker":30} {'picked the best':>16} {'mean regret':>12} {'ranked':>8}")
    print(f"  {'random (exact expectation)':28} {np.mean(random_hit):>16.1%} "
          f"{np.mean(random_regret):>12.4f} {'-':>8}")
    for name, label in (("gated", "Predictor, as a wake runs it"),
                        ("ungated", "Predictor, gate switched off"),
                        ("no_tokens", "Predictor, tokens excluded")):
        hits, regret, ranked = zip(*rows[name], strict=True)
        print(f"  {label:28} {np.mean(hits):>16.1%} {np.mean(regret):>12.4f} "
              f"{sum(ranked):>5}/{n}")

    excluded = [name for r in results for name in r["excluded"]]
    print("\n  the gate excluded: " + ", ".join(
        f"{name} in {excluded.count(name)}/{n}" for name in ("accuracy", "tokens")))
    for objective in ("accuracy", "tokens"):
        pairs = [ok for r in results for ok in r["pairs"][objective]]
        print(f"  held-out pairs the ungated {objective} model orders correctly: "
              f"{np.mean(pairs):.1%} of {len(pairs)}")

    # How often a random picker on these same triples does at least as well.
    rng = np.random.default_rng(args.seed)
    draws = (rng.random((20_000, n)) < np.array(random_hit)).sum(axis=1)
    for name in ("gated", "ungated", "no_tokens"):
        hits = sum(h for h, _, _ in rows[name])
        print(f"  P(a random picker does at least as well as the {name} Predictor) "
              f"= {float(np.mean(draws >= hits)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
