"""What the Predictor predicts, and what reporting it per objective exposed.

Babak Hodjat, a co-author of the ESP paper, read the first version of this
repository and said he could not tell what the predictor surrogate was:

    "Typically, the surrogate model is one or more ML models that act as
    predictors for various outcome objectives we expect from the target we are
    optimizing... The prescription then generates actions optimized against the
    surrogate."

He was right that it was not that. `Surrogate` trains one model on the
already-scalarised fitness, so the weighting lived inside the surrogate and the
three ideas -- predictor, fitness, prescription -- were one object.

Splitting them was supposed to be a documentation fix. It found a bug instead,
and the bug is what most of this file pins: on the committed population the
token model is *anti*-correlated. It ranks candidates by cost backwards, in
every cross-validation seed tried, and the combined figure never showed it
because the accuracy term carried the total.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from esp.eval import measurements
from esp.evolve.loop import scalarise as scalar_fitness
from esp.surrogate.outcomes import (
    PREDICTED_OUTCOMES,
    Outcome,
    OutcomeSurrogate,
    scalarise,
)
from esp.surrogate.predictor import MIN_SAMPLES, Surrogate

MEASURED = measurements.load()
GENOMES = [m.genome for m in MEASURED]
OUTCOMES = [Outcome(accuracy=m.accuracy, tokens=m.tokens) for m in MEASURED]
FITNESSES = [m.fitness for m in MEASURED]

# Enough seeds that a claim about "every seed" means something, few enough that
# the suite stays quick. Each seed is a different KFold partition.
SEEDS_TRIED = range(20)


@cache
def quality(seed: int):
    """Cross-validation over the committed population, computed once per seed.

    Memoised because four tests below assert different things about the same
    twenty reports, and each report fits ten gradient-boosted models. Without
    this the file spends most of its runtime recomputing identical numbers.
    """
    return OutcomeSurrogate(seed=seed).report_quality(GENOMES, OUTCOMES, seed=seed)


@cache
def scalarised_quality(seed: int):
    """The same, for the single-model surrogate this replaced."""
    return Surrogate(seed=seed).report_quality(GENOMES, FITNESSES, seed=seed)


def test_the_committed_population_is_big_enough_to_cross_validate():
    """Everything below is measured on it, so a shrunk fixture must fail here
    rather than quietly turn the assertions into vacuous ones."""
    assert len(MEASURED) >= MIN_SAMPLES


# --------------------------------------------- what the surrogate is made of


def test_one_model_per_measured_outcome_objective():
    """The shape Babak described: predictors for the outcome objectives, not
    for the fitness."""
    surrogate = OutcomeSurrogate(seed=0)
    surrogate.fit(GENOMES, OUTCOMES)
    assert set(surrogate.models) == set(PREDICTED_OUTCOMES)


def test_agent_count_is_counted_rather_than_predicted():
    """It is an exact property of a genome. Asking a regressor to estimate a
    number already in hand adds error and buys nothing."""
    assert "agents" not in PREDICTED_OUTCOMES

    surrogate = OutcomeSurrogate(seed=0)
    surrogate.fit(GENOMES, OUTCOMES)
    predicted = surrogate.predict_outcomes(GENOMES)

    exact = [float(len(g.reachable())) for g in GENOMES]
    assert list(predicted["agents"]) == exact


def test_the_surrogate_never_sees_the_weights():
    """Re-weighting the objectives must not need a refit.

    The point of the split, stated as the property it buys. With the weighting
    inside the surrogate, deciding that token cost matters twice as much meant
    retraining -- and the whole measured population is twelve evaluations, so
    retraining is not free in the only currency this project spends.
    """
    import esp.evolve.loop as loop

    surrogate = OutcomeSurrogate(seed=0)
    surrogate.fit(GENOMES, OUTCOMES)
    before = surrogate.predict(GENOMES)

    original = dict(loop.WEIGHTS)
    try:
        loop.WEIGHTS["tokens"] = original["tokens"] * 10
        after = surrogate.predict(GENOMES)      # no refit
    finally:
        loop.WEIGHTS.clear()
        loop.WEIGHTS.update(original)

    assert not np.allclose(before, after), (
        "changing the weights did not change the derived fitness, so the "
        "weighting is not being applied where it is claimed to be")


def test_the_two_scalarisations_agree():
    """`outcomes.scalarise` is vectorised and `loop.scalarise` is not, because
    `min()` does not broadcast. Two spellings of one formula, pinned against
    each other on the real population rather than trusted."""
    vectorised = scalarise([m.accuracy for m in MEASURED],
                           [m.tokens for m in MEASURED],
                           [m.agents for m in MEASURED])
    scalar = [scalar_fitness(m.accuracy, m.tokens, m.agents) for m in MEASURED]
    assert np.allclose(vectorised, scalar)


def test_fitness_derived_from_measured_outcomes_is_the_measured_fitness():
    """The derivation is the same function the search selects on. If deriving
    fitness from perfect predictions did not reproduce the recorded score, the
    surrogate would be optimising something else entirely."""
    perfect = scalarise([o.accuracy for o in OUTCOMES],
                        [o.tokens for o in OUTCOMES],
                        [float(len(g.reachable())) for g in GENOMES])
    assert np.allclose(perfect, FITNESSES, atol=1e-4)


# ------------------------------------------------------- the bug it uncovered


def test_token_cost_is_anti_predicted_on_the_committed_population():
    """The finding. Not a bad seed, not a weak signal -- a negative one.

    Recorded as a test rather than a note because it is the kind of result that
    quietly gets fixed by a future feature change and never mentioned again.
    If someone makes the token model work, this fails and the README paragraph
    it guards has to be rewritten. That is the intended outcome; what must not
    happen is the claim drifting away from the code in silence.
    """
    rhos = [quality(s).per_outcome["tokens"].spearman for s in SEEDS_TRIED]

    assert all(rho is not None and rho < 0 for rho in rhos), (
        f"token cost is no longer anti-predicted: {rhos}")
    assert max(rhos) < -0.2, (
        f"the worst case is now only weakly negative ({max(rhos):+.3f}); "
        f"re-measure before trusting the documented figure")


def test_accuracy_is_predicted_the_right_way_round():
    """The other half of the same measurement, so "one objective is broken" is
    distinguishable from "the feature set is useless"."""
    rhos = [quality(s).per_outcome["accuracy"].spearman for s in SEEDS_TRIED]
    assert all(rho is not None and rho > 0 for rho in rhos), rhos


def test_the_scalarised_surrogate_hid_it():
    """Why this needed a shape change and not a closer reading.

    The old single-model surrogate reports one respectable number over the same
    data. Nothing in it is wrong; it simply cannot express "one of the two
    objectives is being ranked backwards", because it was never trained on the
    objectives.
    """
    combined = [scalarised_quality(s).spearman for s in SEEDS_TRIED]
    assert all(rho > 0.2 for rho in combined), (
        f"the scalarised surrogate looked healthy in the published figures; "
        f"if it no longer does, this comparison needs rewriting: {combined}")


def test_the_split_did_not_cost_combined_ranking_quality():
    """Deriving fitness from two predictions instead of predicting it directly
    had to be at least as good, or the honesty would have been bought with
    accuracy. On the same seeds it is not worse in the median."""
    derived = sorted(quality(s).derived.spearman for s in SEEDS_TRIED)
    combined = sorted(scalarised_quality(s).spearman for s in SEEDS_TRIED)

    middle = len(derived) // 2
    assert derived[middle] >= combined[middle] - 0.05, (
        f"per-objective median {derived[middle]:+.3f} is materially worse "
        f"than scalarised {combined[middle]:+.3f}")


def test_a_useless_objective_is_named_rather_than_averaged_away():
    report = quality(0)
    assert "tokens" in report.useless_outcomes()
    assert "accuracy" not in report.useless_outcomes()
    assert "tokens" in str(report), (
        "the printed line has to name the objectives, or the per-objective "
        "reporting exists only in the dataclass")


# --------------------------------------------------- refusing to overclaim


def test_below_the_minimum_it_does_not_train_or_rank():
    surrogate = OutcomeSurrogate(seed=0)
    few = GENOMES[:MIN_SAMPLES - 1]
    surrogate.fit(few, OUTCOMES[:MIN_SAMPLES - 1])

    assert not surrogate.trained
    assert not surrogate.ranks()
    predictions = surrogate.predict(few)
    assert len(predictions) == len(few)


def test_too_few_samples_reports_not_measured_rather_than_zero():
    report = OutcomeSurrogate(seed=0).report_quality(
        GENOMES[:3], OUTCOMES[:3], seed=0)
    assert report.derived is None
    assert not report.measured
    assert "NOT MEASURED" in str(report)


def test_a_constant_objective_gets_no_model():
    """Nothing to learn from a column that never varies, and a GBM fitted to
    one would return that constant while looking trained."""
    flat = [Outcome(accuracy=0.8, tokens=m.tokens) for m in MEASURED]
    surrogate = OutcomeSurrogate(seed=0)
    surrogate.fit(GENOMES, flat)

    assert "accuracy" not in surrogate.models
    assert not surrogate.trained, (
        "a surrogate missing one of its objectives must not claim to rank")


def test_an_untrained_surrogate_returns_the_fallback_mean():
    surrogate = OutcomeSurrogate(seed=0)
    surrogate.fit(GENOMES[:2], OUTCOMES[:2])
    predicted = surrogate.predict_outcomes(GENOMES)
    for name in PREDICTED_OUTCOMES:
        assert len(set(np.round(predicted[name], 9))) == 1


# ------------------------------------------------------ what gets written down


def test_the_history_record_keeps_the_keys_older_readers_use():
    """`surrogate_quality` is read by the PDF builder, the web page and a
    README check. The per-objective detail is added beside the old fields
    rather than replacing them -- and never duplicated under a second name,
    which is how two figures for one measurement start disagreeing."""
    record = quality(0).as_record()

    for key in ("samples", "spearman", "mae", "beats_random"):
        assert key in record
    assert record["beats_random"] == (record["spearman"] > 0.2)
    assert record["per_outcome"]["tokens"]["spearman"] < 0
    assert record["useless_outcomes"] == ["tokens"]


@pytest.mark.parametrize("name", PREDICTED_OUTCOMES)
def test_every_predicted_objective_is_actually_measured(name):
    """A predicted objective with nothing behind it would be a model fitted to
    a default."""
    assert all(getattr(o, name) is not None for o in OUTCOMES)
    assert len({getattr(o, name) for o in OUTCOMES}) > 1
