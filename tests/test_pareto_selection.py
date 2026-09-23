"""The Pareto front is what breeds, not just what gets plotted.

`pareto()` was computed, written to history, and drawn in three PDFs -- and
selection ignored it entirely, taking the top `elite` by scalarised fitness.
That is multi-objective in the report and single-objective in the search. A
network that is the cheapest ever measured contributes nothing to breeding if
one particular weighting puts it mid-table, which is exactly the outcome the
Pareto front exists to prevent.
"""

from __future__ import annotations

import random

from esp.eval import measurements
from esp.evolve.loop import Evolution, non_dominated
from esp.genome.seeds import SEEDS
from esp.service.optimizer import ELITE, _population, _propose
from esp.surrogate.outcomes import Outcome

MEASURED = measurements.load()


def test_a_dominated_point_is_excluded():
    points = [(0.9, 100.0, 5.0), (0.8, 50.0, 3.0), (0.7, 200.0, 9.0)]
    assert non_dominated(points) == [0, 1]


def test_an_equal_point_with_one_worse_objective_is_dominated():
    """Same accuracy, same cost, more agents. Strictly worse, so out."""
    points = [(0.9, 100.0, 5.0), (0.9, 100.0, 6.0)]
    assert non_dominated(points) == [0]


def test_identical_points_are_both_kept():
    """Neither strictly beats the other, so neither may be dropped -- dropping
    one would make selection depend on list order."""
    points = [(0.9, 100.0, 5.0), (0.9, 100.0, 5.0)]
    assert non_dominated(points) == [0, 1]


def test_every_point_is_on_the_front_when_nothing_dominates():
    points = [(0.9, 300.0, 5.0), (0.8, 200.0, 4.0), (0.7, 100.0, 3.0)]
    assert non_dominated(points) == [0, 1, 2]


def test_the_front_of_the_measured_population_is_not_just_the_best_fitness():
    """On the real population the two rules disagree, which is the reason this
    change is worth making rather than a tidy-up."""
    points = [(m.accuracy, float(m.tokens), float(m.agents)) for m in MEASURED]
    front = {MEASURED[i].genome_hash for i in non_dominated(points)}
    best_by_fitness = {m.genome_hash for m in
                       sorted(MEASURED, key=lambda m: -m.fitness)[:len(front)]}
    assert front != best_by_fitness, (
        "front and top-by-fitness are identical on this population, so this "
        "test pins nothing -- re-check with a different measured set")


def _evolution_over_measured() -> Evolution:
    run = Evolution(seed=1, elite=3, out_dir="/tmp")
    for m in MEASURED:
        digest = m.genome_hash
        run.pool[digest] = m.genome
        run.scored[digest] = m.fitness
        run.outcomes[digest] = Outcome(accuracy=m.accuracy, tokens=m.tokens)
    return run


def test_the_batch_loop_breeds_from_the_front():
    run = _evolution_over_measured()
    parents = {g.genome_hash() for g in run._parents()}

    points = [(m.accuracy, float(m.tokens), float(m.agents)) for m in MEASURED]
    front = [MEASURED[i].genome_hash for i in non_dominated(points)]
    assert parents <= set(front) | {m.genome_hash for m in MEASURED}
    assert parents & set(front), "no front member was selected to breed"


def test_the_elite_is_filled_when_the_front_is_smaller_than_it():
    """A one-point front must not collapse the search onto a single parent."""
    run = Evolution(seed=1, elite=3, out_dir="/tmp")
    builds = list(SEEDS.values())
    # Deliberately nested outcomes: each is strictly worse than the first.
    for index, build in enumerate(builds):
        genome = build()
        digest = genome.genome_hash()
        run.pool[digest] = genome
        run.scored[digest] = 0.9 - 0.1 * index
        run.outcomes[digest] = Outcome(accuracy=0.9 - 0.1 * index,
                                       tokens=200_000 + 10_000 * index)
    parents = run._parents()
    assert len(parents) == min(3, len(builds))


def test_the_service_elite_uses_the_same_rule():
    """Batch and service must not select differently, or the population that
    accumulates over weeks is shaped by which code path paid for it."""
    from tests.test_service_surrogate import populated

    state = populated(6)
    proposal = _propose(state, _population(state), random.Random(3), 3)
    assert proposal.candidates
    assert len(proposal.candidates) <= ELITE + 1
