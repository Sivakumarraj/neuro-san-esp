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
from esp.surrogate.outcomes import Outcome, OutcomeSurrogate
from esp.surrogate.predictor import MIN_SAMPLES

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
    # How this wake's candidates were chosen, in words. Empty on a wake that
    # spent its budget on seeds, because nothing was chosen.
    selection: str = ""

    def material(self) -> bool:
        """Whether a human needs to hear about this wake.

        Finding a better topology is material. Spending budget and finding
        nothing is the expected case and stays quiet -- a service that reports
        every wake trains its operator to ignore it.
        """
        return self.improved


def _seeds() -> list[tuple[str, Genome]]:
    """The topologies a wake measures before it is allowed to search."""
    return [(name, build()) for name, build in SEEDS.items()]


# How many of the best measured candidates are allowed to breed. Matches the
# batch loop's `elite`, for the same reason: breeding from the whole population
# hands equal weight to candidates already measured as bad.
ELITE = 3


def _population(state: ServiceState) -> list[Genome]:
    """Every genome the service can put back into the search.

    The seeds are rebuildable from source. Everything else is rebuilt from the
    canonical genome stored beside its score, which is why that field exists.

    This returned the seeds and nothing else, under a docstring that said "plus
    anything already measured". The consequence was not a missing convenience:
    `_propose` derives its training set from what this returns, so the Predictor
    trained on three examples no matter how many the service had paid for -- and
    three is below `MIN_SAMPLES`, so it never trained at all. A service whose
    whole premise is that a population accumulates over weeks was throwing away
    everything it accumulated, on every wake, for ever.

    A record whose stored genome rebuilds to a different hash is skipped rather
    than used. Training it against the score filed under the recorded hash would
    teach the Predictor one network's fitness for another network's structure,
    which is worse than having one fewer sample.
    """
    population = [genome for _, genome in _seeds()]
    known = {genome.genome_hash() for genome in population}

    for record in state.evaluated:
        if not record.genome or record.genome_hash in known:
            continue
        try:
            rebuilt = Genome.from_canonical(record.genome)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue                # unreadable record, not a reason to stop
        if rebuilt.genome_hash() != record.genome_hash:
            continue
        population.append(rebuilt)
        known.add(record.genome_hash)

    return population


@dataclass
class Proposal:
    """The candidates a wake is about to pay for, and how they were chosen.

    The batch loop prints whether the Predictor actually ranked its pool, and
    refuses to report a "best predicted" score over an untrained one. The
    service decided the same thing and told nobody, so an operator watching a
    wake buy three candidates had no way to tell a surrogate-assisted
    generation from a random one. The flag travels with the candidates for the
    same reason the batch loop prints it: a selection nobody can distinguish
    from a coin toss should not be described as a selection.
    """

    candidates: list[tuple[Genome, str]] = field(default_factory=list)
    ranked: bool = False
    samples: int = 0
    # Outcome objectives whose model ranked no better than chance on this
    # wake's own population. Carried because a service nobody is watching has
    # no other way to say it: the fitness the ranking used still weights a
    # prediction that is worthless, and the combined figure hides which one.
    useless: list[str] = field(default_factory=list)

    def how(self) -> str:
        """One line for the wake report."""
        if not self.candidates:
            return "no unseen candidate could be bred"
        if self.ranked:
            line = (f"{len(self.candidates)} chosen by the Predictor, trained "
                    f"on {self.samples} measurements")
            if self.useless:
                line += (f" -- but its {'/'.join(self.useless)} model ranks no "
                         f"better than chance, so that part of the objective "
                         f"is noise")
            return line
        return (f"{len(self.candidates)} taken in the order they were bred -- "
                f"the Predictor is untrained ({self.samples} of "
                f"{MIN_SAMPLES} measurements), so this is a random search")


def _propose(state: ServiceState, population: list[Genome], rng: random.Random,
             wanted: int) -> Proposal:
    """Rank a pool of mutants with the surrogate and return the best unseen ones.

    Phase C costs nothing, so it runs on every wake even when only one candidate
    can be afforded afterwards -- ranking a large pool for free is the whole
    reason to prefer this over picking mutants at random.

    Training uses the whole measured population; breeding uses its elite. Those
    are deliberately different sets, as they are in the batch loop: a badly
    scoring candidate is a useful training example and a poor parent.

    What the Predictor learns is the outcome objectives -- accuracy and token
    cost -- not the fitness. Fitness is derived from its predictions by the
    same weighting the wake selects on. This trained a single model on the
    already-scalarised fitness, which made the surrogate learn the weighting
    along with the world and left no way to see that one of the two objectives
    was being predicted backwards.
    """
    measured = {e.genome_hash: e for e in state.evaluated}
    trainable = [(g, measured[g.genome_hash()])
                 for g in population if g.genome_hash() in measured]

    genomes = [g for g, _ in trainable]
    outcomes = [Outcome(accuracy=r.accuracy, tokens=r.tokens)
                for _, r in trainable]

    surrogate = OutcomeSurrogate(seed=len(state.evaluated) or 1)
    useless: list[str] = []
    if len(trainable) >= 2:
        surrogate.fit(genomes, outcomes)
        if surrogate.ranks():
            # Cross-validated on the wake's own population, because an
            # objective can stop being predictable as the population grows and
            # a figure copied from the README would not notice.
            useless = surrogate.report_quality(
                genomes, outcomes, seed=len(state.evaluated)).useless_outcomes()

    # The elite breed; everything measured trains. Unmeasured genomes cannot be
    # ranked, so they only breed when nothing has been measured yet.
    parents = [g for g, r in sorted(trainable, key=lambda pair: -pair[1].fitness)
               [:ELITE]]
    if not parents:
        parents = population
    if not parents:
        return Proposal(samples=len(trainable), useless=useless)

    # Two sets, not one. `seen` keeps the wake from paying twice for a genome
    # across days; `proposed` keeps it from paying twice inside one batch,
    # which is a different mistake and the one that actually happened: the
    # same mutant was bred twice into a single proposal, and the wake measured
    # it, hit its own cache, and filed it again two milliseconds later. The
    # money was not lost -- the cache saw to that -- but a slot in a
    # three-candidate daily elite was, and the duplicate then sat in the
    # population weighting one measurement twice in everything the Predictor
    # learned afterwards.
    seen = state.seen()
    proposed: set[str] = set()
    candidates: list[tuple[Genome, str]] = []
    attempts = 0
    while len(candidates) < SURROGATE_POOL and attempts < SURROGATE_POOL * 8:
        attempts += 1
        try:
            child, operator = mutate(rng.choice(parents), rng)
        except InvalidMutant:
            continue
        digest = child.genome_hash()
        if digest in seen or digest in proposed:
            continue
        proposed.add(digest)
        candidates.append((child, operator))

    if not candidates:
        return Proposal(samples=len(trainable), useless=useless)

    # `ranks()` rather than a flag of our own. This asked whether it had two
    # samples and then set `trained = True`, but the Surrogate needs
    # MIN_SAMPLES before it fits anything. Between two and seven samples the
    # flag said yes over a predictor that returns one constant for every
    # candidate, so the sort below was a no-op and the wake called an arbitrary
    # slice a selection. Only the surrogate knows whether it trained.
    if not surrogate.ranks():
        return Proposal(candidates[:wanted], ranked=False,
                        samples=len(trainable), useless=useless)

    scores = surrogate.predict([c for c, _ in candidates])
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: -p[1])
    return Proposal([pair for pair, _ in ranked[:wanted]], ranked=True,
                    samples=len(trainable), useless=useless)


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
        pending: list[tuple[Genome, str]] = [
            (genome, f"seed:{name}") for name, genome in _seeds()
            if genome.genome_hash() not in state.seen()
        ]
        selection = ""
        if not pending:
            proposal = _propose(state, _population(state), rng, max_evaluations)
            pending, selection = proposal.candidates, proposal.how()
            print(f"  Phase C -- {selection}", flush=True)

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
            exhausted=state.exhausted_now(), selection=selection,
            note="a better topology was found" if improved
                 else "nothing better than what we already had",
        )
    finally:
        lease.release()
