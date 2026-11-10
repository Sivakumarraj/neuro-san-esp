"""The Prescriptor: a network from a context to an action, evolved against the
Predictor's tables and never against a measured outcome."""

from __future__ import annotations

import numpy as np
import pytest

from esp import nsga2, prescriptor


def _problem(contexts: int = 60, seed: int = 0):
    """Two actions. The dear one is right where the first feature is set, and
    the cheap one where it is not."""
    rng = np.random.default_rng(seed)
    features = rng.integers(0, 2, size=(contexts, 3)).astype(float)
    needs_dear = features[:, 0] == 1
    right = np.column_stack([np.where(needs_dear, 0.1, 0.9), np.where(needs_dear, 0.9, 0.2)])
    dollars = np.tile([1.0, 3.0], (contexts, 1))
    return features, right, dollars


def test_a_prescriptor_has_a_weight_for_every_connection_and_bias():
    assert prescriptor.size(5, 4) == (5 + 1) * prescriptor.HIDDEN + (prescriptor.HIDDEN + 1) * 4
    assert prescriptor.size(5, 4) == 84


def test_the_constant_prescriptor_prescribes_its_action_for_every_context():
    features = np.random.default_rng(1).normal(size=(20, 5))
    for action in range(4):
        constant = prescriptor.Prescriptor(prescriptor.always(action, 5, 4), 4,
                                           np.zeros(5), np.ones(5))
        assert (constant.prescribe(features) == action).all()


def test_evolution_finds_the_prescription_the_predictor_rewards():
    features, right, dollars = _problem()
    front = prescriptor.evolve(features, right, dollars)
    best = max(front, key=lambda member: member.accuracy)
    assert best.accuracy == pytest.approx(0.9)
    assert list(best.prescribed) == features[:, 0].astype(int).tolist()


def test_the_front_holds_only_prescriptors_no_other_beats_and_runs_cheapest_first():
    features, right, dollars = _problem()
    front = prescriptor.evolve(features, right, dollars)
    scores = np.array([[-member.accuracy, member.dollars] for member in front])
    assert not nsga2.dominance(scores).any()
    assert [member.dollars for member in front] == sorted(member.dollars for member in front)
    assert len({member.prescribed for member in front}) == len(front)


def test_the_front_is_never_worse_than_the_best_single_action():
    """The first generation holds each action prescribed everywhere, so on the
    Predictor nothing on the front is beaten by one of them."""
    features, right, dollars = _problem()
    front = prescriptor.evolve(features, right, dollars, generations=5)
    for action in range(2):
        accuracy, cost = right[:, action].mean(), dollars[:, action].mean()
        assert any(member.accuracy >= accuracy - 1e-12 and member.dollars <= cost + 1e-12
                   for member in front)


def test_what_the_front_says_of_a_prescriptor_is_what_its_prescriptions_give():
    features, right, dollars = _problem()
    rows = np.arange(len(features))
    for member in prescriptor.evolve(features, right, dollars, generations=20):
        prescribed = member.prescriptor.prescribe(features)
        assert tuple(prescribed.tolist()) == member.prescribed
        assert member.accuracy == pytest.approx(right[rows, prescribed].mean())
        assert member.dollars == pytest.approx(dollars[rows, prescribed].mean())


def test_the_same_seed_evolves_the_same_front():
    features, right, dollars = _problem()
    first = prescriptor.evolve(features, right, dollars, seed=7, generations=30)
    again = prescriptor.evolve(features, right, dollars, seed=7, generations=30)
    assert [m.prescribed for m in first] == [m.prescribed for m in again]
    assert all(np.array_equal(a.prescriptor.weights, b.prescriptor.weights)
               for a, b in zip(first, again, strict=True))


def test_a_new_context_is_scaled_as_the_training_contexts_were():
    features, right, dollars = _problem()
    wide = features * np.array([100.0, 1.0, 1.0])       # the same contexts, other units
    best = max(prescriptor.evolve(wide, right, dollars), key=lambda member: member.accuracy)
    assert best.prescriptor.prescribe(np.array([[100.0, 0.0, 0.0]]))[0] == 1
    assert best.prescriptor.prescribe(np.array([[0.0, 0.0, 0.0]]))[0] == 0


def test_a_feature_that_never_varies_is_left_alone():
    features, right, dollars = _problem()
    features[:, 2] = 1.0
    front = prescriptor.evolve(features, right, dollars, generations=5)
    assert np.isfinite(front[0].prescriptor.prescribe(features)).all()


def test_mismatched_tables_are_refused():
    features, right, dollars = _problem()
    with pytest.raises(ValueError):
        prescriptor.evolve(features[:-1], right, dollars)
    with pytest.raises(ValueError):
        prescriptor.evolve(features, right, dollars[:, :1])
    with pytest.raises(ValueError):
        prescriptor.evolve(features, right, dollars, population=1)
