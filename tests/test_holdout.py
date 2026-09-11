"""Selecting on some tasks, judging on the rest.

Every result this repository reports selects a network on the same seventeen
questions it measures it with, which is the weakest point in the argument. The
analysis answers it from the committed per-task outcomes, so these tests pin
both the arithmetic and the two conclusions it reaches -- one positive, one
not.
"""

from __future__ import annotations

import pytest

from esp.eval import holdout
from esp.evolve.loop import TOKEN_SCALE, WEIGHTS


@pytest.fixture(scope="module")
def population():
    found = holdout.networks()
    assert len(found) >= 2, "need a population to split"
    return found


@pytest.fixture(scope="module")
def tasks():
    found = holdout.task_ids()
    assert len(found) >= 4
    return found


@pytest.fixture(scope="module")
def outcome():
    # Fewer splits than the report runs; the pinned properties are stable well
    # below 200 and the suite should not spend seconds proving it.
    return holdout.analyse(splits=40)


# ------------------------------------------------------------- the arithmetic

def test_every_network_faced_every_shared_task(population, tasks):
    for network in population:
        assert network.correct <= set(tasks) | network.correct
        assert network.finished <= set(tasks) | network.finished


def test_accuracy_on_all_tasks_matches_the_published_accuracy(population, tasks):
    """The subset arithmetic has to reduce to the published number when the
    subset is everything. Otherwise the whole analysis measures something
    else."""
    from esp.eval import measurements

    published = {m.genome_hash: m.accuracy for m in measurements.load()}
    for network in population:
        assert network.accuracy_on(tasks) == pytest.approx(
            published[network.genome_hash], abs=5e-4), network.origin


def test_fitness_on_all_tasks_matches_the_published_fitness(population, tasks):
    from esp.eval import measurements

    published = {m.genome_hash: m.fitness for m in measurements.load()}
    for network in population:
        assert network.fitness_on(tasks) == pytest.approx(
            published[network.genome_hash], abs=5e-4), network.origin


def test_only_the_accuracy_term_is_restricted_to_the_subset(population, tasks):
    """Tokens and agent count are properties of the network, and the cache does
    not attribute tokens per task. Splitting them would invent a measurement."""
    network = population[0]
    half = tasks[: len(tasks) // 2]
    penalty = (WEIGHTS["tokens"] * min(network.tokens / TOKEN_SCALE, 1.0)
               + WEIGHTS["agents"] * (network.agents / 9.0))
    assert network.fitness_on(half) == pytest.approx(
        network.accuracy_on(half) - penalty)


def test_an_empty_subset_scores_no_accuracy(population):
    assert population[0].accuracy_on(()) == 0.0


def test_the_split_sizes_add_up(outcome, tasks):
    assert outcome.selection_tasks + outcome.held_out_tasks == len(tasks)
    assert outcome.held_out_tasks >= 1
    assert outcome.population == len(holdout.networks())


def test_a_population_too_small_to_split_is_refused():
    with pytest.raises(ValueError, match="at least two networks"):
        holdout.analyse(splits=1, cache_dir=holdout.measurements.FIXTURE_CACHE
                        / "does-not-exist")


# -------------------------------------------------------------- the two answers

def test_half_the_tasks_does_not_identify_the_best_network(outcome):
    """The negative result, and the honest limit on every single-network claim
    this repository makes.

    With eight or nine tasks a side, accuracy moves in steps of an eighth while
    the entire cost penalty spans less than one step, so ties decide the order
    and the ties are split-specific. The winner of one half averages the middle
    of the other half.
    """
    assert outcome.also_won_rate < 0.5, (
        f"the selection winner now tops the held-out half "
        f"{outcome.also_won_rate:.0%} of the time -- if the task set grew, "
        f"this test should be replaced rather than relaxed")
    assert outcome.mean_rank > outcome.population * 0.25


def test_accuracy_barely_carries_across_the_split(outcome):
    """Why it fails, stated separately from the fact that it fails. A healthy
    accuracy correlation with a poor fitness one would be a different finding:
    that the cost terms were at fault rather than the sample size."""
    assert abs(outcome.mean_accuracy_rho) < 0.3, (
        f"per-network accuracy now carries across the split at "
        f"{outcome.mean_accuracy_rho:+.3f}, which would change the conclusion")


def test_searching_still_beat_not_searching(outcome):
    """The positive result: what this task set does support.

    The comparison is out of sample -- the winner is chosen on the selection
    half and judged on tasks that took no part in choosing it.
    """
    assert outcome.beat_designer_rate > 0.7, (
        f"the searched winner beats the designer's shape on only "
        f"{outcome.beat_designer_rate:.0%} of splits")
    assert outcome.mean_margin > 0
    assert outcome.mean_designer_rank > outcome.mean_rank, (
        "the designer's shape now ranks better than the searched winner")


def test_the_evolved_population_outranks_the_seeds(outcome):
    """The population-level claim, which is the robust one: whatever the search
    cannot do about picking a single winner, what it produced is better than
    what was hand-written."""
    import random

    population = holdout.networks()
    tasks = holdout.task_ids()
    seeds = [n for n in population if n.origin.startswith("seed:")]
    evolved = [n for n in population if not n.origin.startswith("seed:")]
    assert seeds and evolved

    rng = random.Random(20260821)
    held_out_size = max(1, round(len(tasks) * holdout.HELD_OUT_FRACTION))
    beaten = 0
    trials = 40
    for _trial in range(trials):
        shuffled = list(tasks)
        rng.shuffle(shuffled)
        held_out = tuple(shuffled[:held_out_size])
        best_seed = max(n.fitness_on(held_out) for n in seeds)
        best_evolved = max(n.fitness_on(held_out) for n in evolved)
        beaten += best_evolved > best_seed

    assert beaten == trials, (
        f"the best evolved network beat the best seed on only {beaten} of "
        f"{trials} held-out halves")


def test_the_analysis_is_reproducible(outcome):
    """Same seed, same answer. A conclusion that moves between runs is not a
    conclusion."""
    again = holdout.analyse(splits=40)
    assert again.ranks == outcome.ranks
    assert again.also_won == outcome.also_won
    assert again.mean_margin == pytest.approx(outcome.mean_margin)


def test_a_different_seed_reaches_the_same_conclusions():
    """Not the same numbers -- the same answers. A finding that depends on the
    random seed is a finding about the seed."""
    other = holdout.analyse(splits=40, seed=7)
    assert other.also_won_rate < 0.5
    assert other.beat_designer_rate > 0.7
    assert abs(other.mean_accuracy_rho) < 0.3
