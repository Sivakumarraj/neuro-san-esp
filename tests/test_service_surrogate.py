"""What the service's Predictor is allowed to learn from, and to claim.

The batch loop trains the surrogate on every candidate it has ever paid for.
The service did not: it trained on the seeds alone, because the function that
was meant to supply the population returned the seeds and nothing else under a
docstring promising "plus anything already measured". Three seeds is below
`MIN_SAMPLES`, so the Predictor never trained -- on a service whose entire
premise is that a population accumulates over weeks.

It then claimed a ranking anyway. `_propose` kept its own `trained` flag, set
when two examples existed, while the Surrogate needs eight before it fits
anything. Between two and seven the flag said yes, every prediction came back
as the same constant, and the wake spent real provider budget on an arbitrary
slice of mutants believing the Predictor had chosen them.

These pin both halves: what trains, and what may be called a selection.
"""

from __future__ import annotations

import random

import numpy as np

from esp.evolve.loop import scalarise
from esp.genome.mutations import InvalidMutant, mutate
from esp.genome.seeds import SEEDS
from esp.service.optimizer import ELITE, Proposal, _population, _propose, _seeds
from esp.service.state import Evaluated, ServiceState
from esp.surrogate.outcomes import (
    PREDICTED_OUTCOMES,
    Outcome,
    OutcomeSurrogate,
)
from esp.surrogate.predictor import MIN_SAMPLES, Surrogate


def measured(genome, fitness: float, origin: str = "mut:add_agent",
             store_genome: bool = True) -> Evaluated:
    """One paid-for evaluation, recorded the way a wake records it.

    Outcomes first, fitness derived from them -- not the other way round.

    This used to record a constant accuracy and a constant token count beside a
    fitness that varied, which is a record no real evaluation can produce:
    fitness is a fixed function of the three outcomes, so identical outcomes
    give identical fitness. The scalarised surrogate never noticed, because it
    trained on the fitness column and never looked at the outcomes. The
    per-objective Predictor trains on the outcomes, and correctly declines to
    fit a population whose objectives never move -- which is how an
    inconsistent fixture that had been passing for months surfaced.

    Callers still say what fitness they want, because that is what the tests
    below are about. The outcomes are chosen to produce it.
    """
    agents = len(genome.reachable())
    # Spread across roughly the range the real measurements span, and
    # correlated with fitness so that a better candidate is also a cheaper one
    # -- an arbitrary fixture convention, not a claim about real networks.
    tokens = 240_000 + round((1.0 - fitness) * 400_000)
    accuracy = fitness - scalarise(0.0, tokens, agents)
    return Evaluated(
        genome_hash=genome.genome_hash(), origin=origin,
        fitness=round(scalarise(accuracy, tokens, agents), 4),
        accuracy=accuracy, tokens=tokens, agents=agents,
        depth=genome.depth(), generation=1,
        measured_at="2026-08-24T00:00:00+00:00", model=genome.default_model,
        genome=genome.canonical() if store_genome else None)


def test_the_fixture_records_a_consistent_evaluation():
    """The fixture above has to be a record the system could actually have
    written, or every test in this file is pinned against something impossible.
    Fitness must be what scalarising its own outcomes gives."""
    genome = next(iter(SEEDS.values()))()
    record = measured(genome, 0.83)

    assert record.fitness == 0.83
    assert record.fitness == round(
        scalarise(record.accuracy, record.tokens, record.agents), 4)


def evolved(count: int, seed: int = 7):
    """`count` distinct single mutants of the seeds, as a wake breeds them.

    Bounded on attempts: a generator that cannot reach `count` must fail the
    assertion that needed them rather than hang the suite. Mutants are
    deliberately not bred back into the parent pool -- chaining mutations grows
    a genome every time, and a few hundred rounds of that produces networks far
    larger than anything the search would measure.
    """
    rng = random.Random(seed)
    parents = [build() for build in SEEDS.values()]
    produced: dict[str, object] = {}
    for _attempt in range(count * 80):
        if len(produced) >= count:
            break
        try:
            child, _operator = mutate(rng.choice(parents), rng)
        except InvalidMutant:
            continue
        produced.setdefault(child.genome_hash(), child)
    return list(produced.values())


def mutants_of(parents: list, count: int, seed: int = 11) -> list:
    """A pool drawn from `parents`, duplicates allowed.

    What Phase C hands the Predictor: many candidates, each one mutation away
    from a measured network. Distinctness is `_propose`'s business, not this
    pool's.
    """
    rng = random.Random(seed)
    pool = []
    for _attempt in range(count * 80):
        if len(pool) >= count:
            break
        try:
            child, _operator = mutate(rng.choice(parents), rng)
        except InvalidMutant:
            continue
        pool.append(child)
    return pool


def populated(count: int = 6) -> ServiceState:
    """A service that has paid for the seeds and `count` evolved candidates."""
    state = ServiceState()
    for name, genome in _seeds():
        state.add(measured(genome, 0.78, origin=f"seed:{name}"))
    for index, genome in enumerate(evolved(count)):
        state.add(measured(genome, 0.80 + 0.01 * index))
    return state


# ------------------------------------------------- what the Predictor learns


def test_every_measured_candidate_can_train_the_predictor():
    """The bug, stated as the property it broke: a candidate the service paid
    for must be available to the Predictor, not just the seeds."""
    state = populated(6)
    hashes = {genome.genome_hash() for genome in _population(state)}

    for record in state.evaluated:
        assert record.genome_hash in hashes, (
            f"{record.origin} was paid for and cannot train the Predictor")


def test_a_population_over_min_samples_actually_trains():
    """Nine measurements were not enough to train an eight-sample model,
    because only three of them reached it."""
    state = populated(6)
    assert len(state.evaluated) >= MIN_SAMPLES

    population = _population(state)
    scores = {e.genome_hash: e.fitness for e in state.evaluated}
    trainable = [(g, scores[g.genome_hash()]) for g in population
                 if g.genome_hash() in scores]
    assert len(trainable) >= MIN_SAMPLES

    surrogate = Surrogate(seed=1)
    surrogate.fit([g for g, _ in trainable], [v for _, v in trainable])
    assert surrogate.ranks()


def test_the_seeds_alone_are_below_the_training_threshold():
    """Why the old behaviour was not merely incomplete. If three seeds were
    enough to train, this would have been a missed opportunity rather than a
    Predictor that never existed."""
    assert len(_seeds()) < MIN_SAMPLES


def test_a_record_with_no_stored_genome_is_skipped_not_guessed():
    """State written before the genome was stored beside the score still
    loads. Those records cannot be rebuilt, and inventing one would train the
    Predictor on a structure nobody measured."""
    state = ServiceState()
    for name, genome in _seeds():
        state.add(measured(genome, 0.78, origin=f"seed:{name}"))
    orphan = evolved(1)[0]
    state.add(measured(orphan, 0.99, store_genome=False))

    hashes = {genome.genome_hash() for genome in _population(state)}
    assert orphan.genome_hash() not in hashes


def test_a_genome_that_rebuilds_to_a_different_hash_is_refused():
    """Worse than one fewer sample: it would file one network's fitness
    against another network's structure."""
    state = ServiceState()
    candidate = evolved(1)[0]
    record = measured(candidate, 0.95)
    record.genome_hash = "0" * 16          # no longer describes this genome
    state.add(record)

    assert record.genome_hash not in {g.genome_hash() for g in _population(state)}


def test_an_unreadable_stored_genome_does_not_stop_the_wake():
    state = ServiceState()
    for name, genome in _seeds():
        state.add(measured(genome, 0.78, origin=f"seed:{name}"))
    broken = measured(evolved(1)[0], 0.9)
    broken.genome = {"not": "a genome"}
    state.add(broken)

    assert len(_population(state)) == len(_seeds())


# -------------------------------------------- what may be called a selection


def test_an_untrained_surrogate_does_not_pretend_to_have_ranked():
    """Two measurements used to be enough to claim a ranking. The Surrogate
    needs eight, so between two and seven the claim covered a constant.

    Sorting on one repeated constant returns the same prefix a stable sort
    started with, so the candidates were not wrong -- the account of how they
    were picked was. That is why the flag is asserted and not the list.
    """
    state = ServiceState()
    for genome, fitness in zip(evolved(3), (0.70, 0.80, 0.90), strict=True):
        state.add(measured(genome, fitness))
    assert 2 <= len(state.evaluated) < MIN_SAMPLES, (
        "the old flag turned on at two samples; pin it inside that window")

    proposal = _propose(state, _population(state), random.Random(3), 3)
    assert proposal.candidates, (
        "a wake with nothing to rank still has to propose something")
    assert not proposal.ranked
    assert proposal.samples == len(state.evaluated)
    assert "random search" in proposal.how()


def test_a_wake_says_which_way_its_candidates_were_chosen():
    """An unattended service that cannot be watched has to say this, or a
    generation of random search is indistinguishable from a guided one."""
    state = populated(6)
    guided = _propose(state, _population(state), random.Random(3), 3)
    assert guided.ranked
    assert "Predictor" in guided.how()
    assert str(len(state.evaluated)) in guided.how()


def test_the_predictor_is_asked_rather_than_assumed(monkeypatch):
    """The fix, as the property it restores: with too few samples to train,
    nothing may consult `predict` at all. The old code called it and sorted the
    answer."""
    state = ServiceState()
    for genome, fitness in zip(evolved(3), (0.70, 0.80, 0.90), strict=True):
        state.add(measured(genome, fitness))

    def refuse(self, genomes):
        raise AssertionError("predict called on an untrained surrogate")

    monkeypatch.setattr(Surrogate, "predict", refuse)
    proposal = _propose(state, _population(state), random.Random(3), 3)
    assert proposal.candidates and not proposal.ranked


def test_a_trained_surrogate_orders_the_pool_it_is_given():
    """The other half: once it has trained, the ranking has to carry
    information. A sort over one repeated constant is not a selection."""
    state = populated(6)
    population = _population(state)
    scores = {e.genome_hash: e.fitness for e in state.evaluated}
    trainable = [(g, scores[g.genome_hash()]) for g in population
                 if g.genome_hash() in scores]
    surrogate = Surrogate(seed=len(state.evaluated))
    surrogate.fit([g for g, _ in trainable], [v for _, v in trainable])
    assert surrogate.ranks()

    pool = mutants_of([g for g, _ in trainable], 120)
    assert len(pool) == 120
    predictions = surrogate.predict(pool)
    assert len(set(np.round(predictions, 6))) > 1, (
        "every candidate predicted the same value; the ranking is a no-op")


def test_only_the_elite_breed():
    """A candidate measured as bad is a useful training example and a poor
    parent. The batch loop already separates the two; the service now does."""
    state = populated(6)
    population = _population(state)
    scores = {e.genome_hash: e.fitness for e in state.evaluated}
    trainable = sorted(((g, scores[g.genome_hash()]) for g in population
                        if g.genome_hash() in scores), key=lambda p: -p[1])

    assert len(trainable) > ELITE, "nothing is being left out, so nothing is pinned"
    worst = trainable[-1][1]
    elite = [value for _, value in trainable[:ELITE]]
    assert min(elite) > worst


def test_proposals_are_never_something_already_paid_for():
    state = populated(6)
    proposal = _propose(state, _population(state), random.Random(5), 3)
    assert proposal.candidates
    for genome, _origin in proposal.candidates:
        assert genome.genome_hash() not in state.seen()


# ----------------------------------------- never pay twice for the same thing

def test_a_proposal_never_contains_the_same_candidate_twice():
    """Within one batch, not just across days.

    `_propose` deduped against the population and not against the batch it was
    building, so the same mutant could be bred twice into one proposal. A live
    wake did exactly that: it measured the genome, hit its own cache on the
    second ask, and filed the record again two milliseconds later. The cache
    meant no provider budget was lost; a slot in a three-candidate daily elite
    was.
    """
    state = populated(6)
    proposal = _propose(state, _population(state), random.Random(5), 3)
    assert len(proposal.candidates) > 1, "nothing to duplicate, so nothing pinned"

    digests = [genome.genome_hash() for genome, _origin in proposal.candidates]
    assert len(digests) == len(set(digests)), f"duplicate proposals: {digests}"


def test_the_population_refuses_a_genome_it_already_holds():
    """The Predictor trains on this list. A genome present twice has its
    fitness counted twice and pulls the fit toward itself."""
    state = ServiceState()
    candidate = evolved(1)[0]

    assert state.add(measured(candidate, 0.81)) is True
    assert state.add(measured(candidate, 0.99)) is False, (
        "a second record for the same genome was accepted")
    assert len(state.evaluated) == 1
    assert state.evaluated[0].fitness == 0.81, "the first measurement stands"


def test_many_proposals_in_a_row_stay_distinct():
    """The loop breeds up to a large pool; the guard has to hold across all of
    it, not only the first few."""
    state = populated(6)
    proposal = _propose(state, _population(state), random.Random(17), 25)
    digests = [genome.genome_hash() for genome, _origin in proposal.candidates]
    assert len(digests) == len(set(digests))
    for digest in digests:
        assert digest not in state.seen()


# ------------------------------- what the Predictor is trained on, and says

def test_the_service_predictor_learns_outcomes_not_fitness():
    """The shape ESP describes, pinned where the service uses it.

    Asked in review by a co-author of the ESP paper: the surrogate should be
    models predicting the outcome objectives, with fitness derived from them.
    This trained one model on the already-scalarised fitness, so the weighting
    lived inside the surrogate and no objective could be inspected on its own.
    """
    state = populated(6)
    population = _population(state)
    records = {e.genome_hash: e for e in state.evaluated}
    trainable = [(g, records[g.genome_hash()]) for g in population
                 if g.genome_hash() in records]
    assert len(trainable) >= MIN_SAMPLES

    surrogate = OutcomeSurrogate(seed=1)
    surrogate.fit([g for g, _ in trainable],
                  [Outcome(accuracy=r.accuracy, tokens=r.tokens)
                   for _, r in trainable])

    assert surrogate.ranks()
    assert set(surrogate.models) == set(PREDICTED_OUTCOMES)
    # Agent count is exact, so it is counted rather than estimated.
    predicted = surrogate.predict_outcomes([g for g, _ in trainable])
    assert list(predicted["agents"]) == [float(len(g.reachable()))
                                         for g, _ in trainable]


def test_a_wake_names_an_objective_its_predictor_gets_wrong():
    """An unattended service is read through its wake report and nothing else.

    A combined quality figure cannot say "one of the two objectives is being
    ranked backwards" -- on the committed population that is exactly the case
    for token cost, and the accuracy term carries the total regardless. If the
    service is going to spend real budget on a ranking, it has to be able to
    say which part of it is noise.
    """
    state = populated(6)
    proposal = Proposal(candidates=[(next(iter(SEEDS.values()))(), "rewire")],
                        ranked=True, samples=len(state.evaluated),
                        useless=["tokens"])

    assert "tokens" in proposal.how()
    assert "no better than chance" in proposal.how()
    # And says nothing of the sort when every objective is usable.
    clean = Proposal(candidates=proposal.candidates, ranked=True,
                     samples=proposal.samples)
    assert "no better than chance" not in clean.how()


def test_the_proposal_carries_the_verdict_it_measured():
    """The field is populated by `_propose` from the wake's own population, not
    copied from a figure in the README that cannot notice the population
    changing underneath it."""
    state = populated(6)
    proposal = _propose(state, _population(state), random.Random(3), 3)

    assert proposal.ranked
    assert isinstance(proposal.useless, list)
    assert set(proposal.useless) <= set(PREDICTED_OUTCOMES)
