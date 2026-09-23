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
from esp.surrogate.predictor import MIN_SAMPLES, Surrogate, _spearman

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


# ----------------------------------- measuring against the right baseline

@cache
def null_quality(seed: int):
    """A quality report that also measured each objective's permutation null."""
    return OutcomeSurrogate(seed=seed).report_quality(
        GENOMES, OUTCOMES, seed=seed, null_trials=12)


def test_the_token_baseline_is_measurably_negative():
    """The correction, stated as the fact that forced it.

    The first version reported token cost at -0.61 as though zero were the
    baseline. Cross-validation on twelve samples manufactures negative rank
    correlation on its own: hold out a high value, the training mean drops,
    the model predicts low, and the error correlates the wrong way.

    Asserted on tokens only, and on the median across seeds, because a single
    seed's null is noisy -- see the spread test below.
    """
    nulls = [null_quality(s).nulls["tokens"] for s in range(5)]
    assert np.median(nulls) < -0.05, (
        f"the token null came back at {np.median(nulls):+.3f}; if the "
        f"procedure no longer has a negative bias, the published margin "
        f"is wrong")


def test_the_accuracy_baseline_is_near_zero():
    """The other half of the correction, and it went the other way.

    Measuring the null was expected to shrink both figures. It did not:
    accuracy's null sits around -0.03 over 20 seeds, so that figure was sound
    all along and its margin is close to its raw spearman. Recorded because
    "we checked and nothing was wrong" is a result too.
    """
    nulls = [null_quality(s).nulls["accuracy"] for s in range(5)]
    assert abs(np.median(nulls)) < 0.25, (
        f"accuracy's null moved to {np.median(nulls):+.3f}; the README says "
        f"it is near zero")


def test_the_token_effect_is_smaller_than_it_was_published_as():
    """Part artifact, part real -- and twelve samples cannot separate them.

    This stops the dramatic version of the claim coming back. The raw median
    is about -0.53; the margin over the null is about -0.35. Both negative, so
    the finding survives; only its size changes.
    """
    raws, margins = [], []
    for seed in range(5):
        report = null_quality(seed)
        raws.append(report.per_outcome["tokens"].spearman)
        margins.append(report.margin("tokens"))

    assert np.median(raws) < 0 and np.median(margins) < 0, (
        "the finding itself should still hold")
    assert all(m < 0 for m in margins), (
        f"the margin must stay negative in every seed: {margins}")
    assert np.median(margins) > np.median(raws), (
        f"margin median {np.median(margins):+.3f} must be less extreme than "
        f"raw median {np.median(raws):+.3f}, or the null is doing nothing")


def test_uselessness_is_judged_against_the_null_not_against_zero():
    report = null_quality(0)
    assert "tokens" in report.useless_outcomes()
    assert "accuracy" not in report.useless_outcomes()
    # And the printed line says what it was compared against.
    assert "null" in str(report)


def test_a_shuffled_target_does_not_beat_its_own_null():
    """Destroy the signal and the objective must stop clearing the bar.

    Done on **token cost**, not accuracy, and that choice is the finding. A
    permutation test only destroys a relationship if permuting actually moves
    the values. Accuracy takes four distinct values across twelve samples, so
    a shuffle frequently maps a value onto an identical one and leaves the
    ordering largely intact -- shuffled accuracy came back at +0.88 on one
    draw. Token cost is distinct in all twelve, so shuffling it genuinely
    destroys the relationship and the null means what it claims.
    """
    rng = np.random.default_rng(3)
    order = rng.permutation(len(OUTCOMES))
    shuffled = [Outcome(accuracy=OUTCOMES[i].accuracy,
                        tokens=OUTCOMES[order[i]].tokens)
                for i in range(len(OUTCOMES))]
    report = OutcomeSurrogate(seed=1).report_quality(
        GENOMES, shuffled, seed=1, null_trials=12)
    assert "tokens" in report.useless_outcomes()


def test_the_null_is_only_meaningful_on_an_untied_objective():
    """Why the test above uses tokens, pinned so nobody moves it to accuracy.

    A permutation null on a heavily tied target measures less than it looks
    like it does. This records which of the two objectives is safe.
    """
    accuracies = {o.accuracy for o in OUTCOMES}
    tokens = {o.tokens for o in OUTCOMES}

    assert len(tokens) == len(OUTCOMES), (
        "token cost is distinct per network, which is what makes its "
        "permutation null trustworthy")
    assert len(accuracies) < len(OUTCOMES) / 2, (
        "accuracy is heavily tied -- if that stops being true, the caveat in "
        "docs/FINDINGS.md about its null should be revisited")


def test_the_null_is_absent_rather_than_assumed_when_not_measured():
    """A wake that skips the null must not silently get a zero baseline it
    never measured."""
    report = quality(0)
    assert report.nulls == {}
    assert report.margin("tokens") == report.per_outcome["tokens"].spearman


def test_the_record_carries_the_null_and_the_margin():
    record = null_quality(0).as_record()
    tokens = record["per_outcome"]["tokens"]
    assert tokens["null"] is not None
    assert tokens["margin"] == tokens["spearman"] - tokens["null"]


# --- Gating: an objective that loses to its own null stops steering Phase C ---
#
# Naming the defect was the first half. Until the gate below, Phase C still
# weighted the token prediction at full strength every generation, so the
# search was actively steered by a model that orders candidates backwards.
# These pin that it now stops, that it only stops on evidence, and that it
# starts again on its own if the objective ever begins to predict.


def test_a_gated_objective_is_held_at_the_population_mean():
    surrogate = OutcomeSurrogate(seed=0)
    surrogate.fit(GENOMES, OUTCOMES, gated=["tokens"])
    predicted = surrogate.predict_outcomes(GENOMES)

    assert len(set(np.round(predicted["tokens"], 6))) == 1, (
        "a gated objective must be constant across the batch, or it still "
        "carries ordering")
    assert predicted["tokens"][0] == pytest.approx(
        float(np.mean([o.tokens for o in OUTCOMES])))
    # The objective that does predict is untouched.
    assert len(set(np.round(predicted["accuracy"], 6))) > 1


def test_gating_actually_changes_what_phase_c_would_pick():
    """A gate nobody can observe is a comment, not a fix."""
    ungated = OutcomeSurrogate(seed=0)
    ungated.fit(GENOMES, OUTCOMES)
    gated = OutcomeSurrogate(seed=0)
    gated.fit(GENOMES, OUTCOMES, gated=["tokens"])

    before = np.argsort(-ungated.predict(GENOMES))
    after = np.argsort(-gated.predict(GENOMES))
    assert not np.array_equal(before, after), (
        "excluding an anti-correlated objective left the ordering identical, "
        "which would mean it was never steering anything")


def test_gating_needs_a_measured_null_not_a_suspicion():
    """Without a null, +0.2 is a floor twelve samples cannot reach.

    Gating on it would drop objectives for being small-sample rather than for
    being wrong, so `gated_outcomes` stays empty and the loop only warns.
    """
    quality = OutcomeSurrogate(seed=0).report_quality(GENOMES, OUTCOMES, seed=0)
    assert quality.useless_outcomes(), "expected the token model to look bad"
    assert quality.gated_outcomes() == [], (
        "an objective was gated without its null ever being measured")


def test_the_token_model_is_gated_once_its_null_is_measured():
    quality = OutcomeSurrogate(seed=0).report_quality(
        GENOMES, OUTCOMES, seed=0, null_trials=12)
    assert "tokens" in quality.nulls, "the null was not measured"
    assert "tokens" in quality.gated_outcomes(), (
        "token cost loses to its own null on this population and must be "
        "excluded from the derived fitness")
    assert "accuracy" not in quality.gated_outcomes()


def test_gating_every_objective_leaves_only_the_term_that_is_counted():
    """With both *learned* objectives gated, what remains is parsimony.

    Not a constant, and the first version of this test wrongly asserted one.
    Agent count is read off the genome exactly rather than predicted, so it
    keeps ordering candidates -- correctly. What is gone is every contribution
    the surrogate *learned*, which is why `ranks` is False: sorting by a count
    anyone could take by hand is not the Predictor earning its place, and
    Phase C must not report it as a surrogate score.
    """
    surrogate = OutcomeSurrogate(seed=0)
    surrogate.fit(GENOMES, OUTCOMES, gated=list(PREDICTED_OUTCOMES))

    assert not surrogate.ranks(), (
        "no learned model contributes, so Phase C is selecting on parsimony "
        "alone and must say so rather than print a best predicted score")

    predicted = surrogate.predict(GENOMES)
    agents = np.array([float(len(g.reachable())) for g in GENOMES])
    # Fitness penalises agent count, so the order is exactly its reverse.
    assert _spearman(predicted, -agents) == pytest.approx(1.0), (
        "with both learned terms held constant the only signal left should "
        "be the exact agent-count penalty")


def test_a_gate_is_recomputed_every_generation_not_remembered():
    """Gating is a measurement. An objective that starts predicting returns."""
    surrogate = OutcomeSurrogate(seed=0)
    surrogate.fit(GENOMES, OUTCOMES, gated=["tokens"])
    assert surrogate.gated == {"tokens"}

    surrogate.fit(GENOMES, OUTCOMES)
    assert surrogate.gated == set()
    assert len(set(np.round(surrogate.predict_outcomes(GENOMES)["tokens"], 6))) > 1
