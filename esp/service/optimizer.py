"""One wake of the optimiser: spend today's budget, then stop cleanly.

This is the difference between the batch script and a service. `run_esp.py`
plans a fixed number of generations and fails if the provider stops it early. A
wake plans nothing: it evaluates candidates while budget lasts, writes the
population down after each one, and returns. Whether it managed six candidates
or none, the next wake continues from there.

The service is expected to be interrupted. Being stopped by an exhausted quota
is a normal ending, not an error.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from esp.eval import failover
from esp.eval.runner import QuotaExhausted, evaluate
from esp.evolve.loop import fitness
from esp.genome.definition import DEFAULT_MODEL, Genome
from esp.genome.mutations import InvalidMutant, mutate
from esp.genome.seeds import SEEDS
from esp.service.state import Evaluated, Lease, ServiceState
from esp.surrogate.predictor import Surrogate

# How many real evaluations one wake will attempt before stopping voluntarily.
# Lower than a day's budget on purpose: a wake that tries to spend everything
# leaves nothing for the rest of the day, and a service that goes quiet for
# twenty hours after breakfast is not running unattended.
MAX_PER_WAKE = 3
SURROGATE_POOL = 400


@dataclass
class WakeReport:
    """What one wake did. The finaliser decides whether it is worth saying."""

    acquired: bool
    evaluated: int = 0
    generation: int = 0
    best_fitness: float | None = None
    improved: bool = False
    stopped_because: str = ""
    exhausted: list[str] = field(default_factory=list)
    note: str = ""

    def material(self) -> bool:
        """Whether a human needs to hear about this wake.

        Finding a better topology is material. Spending budget and finding
        nothing is the expected case and stays quiet -- a service that reports
        every wake trains its operator to ignore it.
        """
        return self.improved


def _seed_population(state: ServiceState) -> list[Genome]:
    """The genomes to breed from: the seeds, plus anything already measured."""
    return [build() for build in SEEDS.values()]


def _propose(state: ServiceState, parents: list[Genome], rng: random.Random,
             wanted: int) -> list[tuple[Genome, str]]:
    """Rank a pool of mutants with the surrogate and return the best unseen ones.

    Phase C costs nothing, so it runs on every wake even when only one candidate
    can be afforded afterwards -- ranking a large pool for free is the whole
    reason to prefer this over picking mutants at random.
    """
    measured = {e.genome_hash: e.fitness for e in state.evaluated}
    trainable = [(g, measured[g.genome_hash()])
                 for g in parents if g.genome_hash() in measured]

    surrogate = Surrogate(seed=len(state.evaluated) or 1)
    if len(trainable) >= 2:
        surrogate.fit([g for g, _ in trainable], [v for _, v in trainable])
        trained = True
    else:
        trained = False

    seen = state.seen()
    candidates: list[tuple[Genome, str]] = []
    attempts = 0
    while len(candidates) < SURROGATE_POOL and attempts < SURROGATE_POOL * 8:
        attempts += 1
        try:
            child, operator = mutate(rng.choice(parents), rng)
        except InvalidMutant:
            continue
        if child.genome_hash() in seen:
            continue
        candidates.append((child, operator))

    if not candidates:
        return []
    if not trained:
        # Nothing measured yet, so there is nothing to rank against. Say so by
        # taking them in order rather than pretending the surrogate chose.
        return candidates[:wanted]

    scores = surrogate.predict([c for c, _ in candidates])
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: -p[1])
    return [pair for pair, _ in ranked[:wanted]]


def _models_of(genome: Genome) -> list[str]:
    """Every model this candidate can actually call."""
    return sorted({genome.default_model,
                   *(a.model for a in genome.agents.values() if a.model)})


def _exhausted_models(genome: Genome, message: str) -> list[str]:
    """Which models to stop calling for the rest of today.

    Taken from the provider's own message first, because the 429 names the model
    that ran out. Matching the message against the failover ladder instead would
    miss the network default, which is not a ladder member: a run that exhausted
    the default would record nothing, report an empty `exhausted_today`, and
    spend the first calls of every later wake rediscovering the same dead
    model.

    When the message names nothing recognisable, fall back to the candidate's
    own models. Retiring a model that still had budget costs one wake; retrying
    a dead one costs the whole day.
    """
    named = failover.models_named(message)
    return named or _models_of(genome)


def wake(state: ServiceState | None = None, rng: random.Random | None = None,
         max_evaluations: int = MAX_PER_WAKE) -> WakeReport:
    """Run one cycle. Safe to call on a schedule; safe to be killed part-way."""
    state = state if state is not None else ServiceState.load()
    rng = rng or random.Random()

    lease = Lease()
    owner = f"wake-{int(time.time())}"
    if not lease.acquire(owner):
        held = lease.holder() or {}
        # Say when it frees. A wake killed before its finally block leaves the
        # lease held for its full duration, and an operator reading "another
        # wake holds the lease" has no way to tell a busy service from a wedged
        # one. It self-heals at the timeout either way; the message should say
        # when.
        remaining = max(0.0, lease.seconds - (time.time() - float(held.get("taken_at", 0))))
        return WakeReport(
            acquired=False,
            note=(f"another wake holds the lease ({held.get('owner', 'unknown')}, "
                  f"taken {held.get('taken_at_iso', '?')}); "
                  f"it expires in {remaining / 60:.0f} min"))

    try:
        state.wakes += 1
        state.last_wake = datetime.now(UTC).isoformat()

        # The network default is what candidates are built with; the ladder is
        # where a call goes when that runs out. Both have to be live for a wake
        # to be worth starting, and both have to be primed into the failover
        # module or this fresh process will spend calls rediscovering what the
        # last wake already learned.
        fleet = [DEFAULT_MODEL, *failover.LADDER]
        spent = [model for model in fleet if state.is_exhausted(model)]
        failover.prime(spent)
        usable = [model for model in fleet if model not in spent]
        if not usable:
            state.save()
            return WakeReport(acquired=True, generation=state.generation,
                              stopped_because="every model's daily budget is spent",
                              exhausted=spent,
                              note="nothing to do until the quota resets")

        before = state.best()
        best_before = before.fitness if before else None

        # Seeds first: without a measured baseline nothing can be ranked, and a
        # service with no baseline is optimising against nothing.
        parents = _seed_population(state)
        pending: list[tuple[Genome, str]] = [
            (g, f"seed:{name}") for name, g in zip(SEEDS.keys(),
                                                   parents, strict=True)
            if g.genome_hash() not in state.seen()
        ]
        if not pending:
            pending = _propose(state, parents, rng, max_evaluations)

        done = 0
        stopped = ""
        for genome, origin in pending[:max_evaluations]:
            try:
                evaluation = evaluate(genome)
            except QuotaExhausted as exc:
                # Expected ending. Record which model ran out so the next wake
                # today skips it, and stop without touching the population.
                for model in _exhausted_models(genome, str(exc)):
                    state.mark_exhausted(model)
                stopped = "provider budget exhausted"
                break
            except Exception as exc:
                stopped = f"{type(exc).__name__}: {exc}"[:160]
                break

            state.add(Evaluated(
                genome_hash=genome.genome_hash(), origin=origin,
                fitness=round(fitness(evaluation), 4),
                accuracy=evaluation.accuracy, tokens=evaluation.tokens,
                agents=evaluation.agents, depth=evaluation.depth,
                generation=state.generation,
                measured_at=datetime.now(UTC).isoformat(),
                model=genome.default_model,
                # Kept so the winner can actually be served later. Without it a
                # candidate that beats every seed is a score with no network
                # attached, which is the one outcome the whole search exists to
                # produce.
                genome=genome.canonical(),
            ))
            done += 1
            # Saved per candidate, not per wake: the next interruption must not
            # cost an evaluation that has already been paid for.
            state.save()

        if done:
            state.generation += 1
        state.save()

        after = state.best()
        improved = bool(after and (best_before is None or after.fitness > best_before))
        return WakeReport(
            acquired=True, evaluated=done, generation=state.generation,
            best_fitness=after.fitness if after else None,
            improved=improved, stopped_because=stopped,
            exhausted=state.exhausted_now(),
            note="a better topology was found" if improved
                 else "nothing better than what we already had",
        )
    finally:
        lease.release()
