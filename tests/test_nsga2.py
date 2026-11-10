"""NSGA-II's two orderings: which candidates no other beats, and which of a
front to keep when it does not fit."""

from __future__ import annotations

import numpy as np

from esp import nsga2


def test_a_candidate_dominates_when_no_worse_everywhere_and_better_somewhere():
    table = nsga2.dominance(np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 0.5], [1.0, 1.0]]))
    assert table[0, 1] and not table[1, 0]
    # Better on one objective and worse on the other: neither dominates.
    assert not table[0, 2] and not table[2, 0]
    # Equal on both is not better on either.
    assert not table[0, 3] and not table[3, 0]


def test_fronts_peel_off_the_non_dominated_first():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.5]])
    assert nsga2.fronts(points) == [[0], [2, 3], [1]]


def test_every_candidate_is_in_exactly_one_front():
    rng = np.random.default_rng(3)
    points = rng.integers(0, 4, size=(40, 2)).astype(float)
    found = [i for front in nsga2.fronts(points) for i in front]
    assert sorted(found) == list(range(40))


def test_no_member_of_a_front_dominates_another():
    rng = np.random.default_rng(4)
    points = rng.random((30, 2))
    table = nsga2.dominance(points)
    for front in nsga2.fronts(points):
        assert not table[np.ix_(front, front)].any()


def test_the_ends_of_a_front_are_never_crowded_out():
    points = np.array([[0.0, 4.0], [1.0, 3.0], [1.1, 2.9], [4.0, 0.0]])
    distance = nsga2.crowding(points, [0, 1, 2, 3])
    assert np.isinf(distance[0]) and np.isinf(distance[3])
    assert np.isfinite(distance[1]) and np.isfinite(distance[2])


def test_a_front_of_two_is_all_ends():
    assert np.isinf(nsga2.crowding(np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1])).all()


def test_survivors_take_whole_fronts_then_the_loneliest():
    # A first front of two, then a second of three whose middle is the crowded one.
    points = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    assert nsga2.fronts(points) == [[0, 1], [2, 3, 4]]
    assert sorted(nsga2.survivors(points, 4)) == [0, 1, 2, 4]


def test_identical_candidates_survive_the_same_way_every_time():
    """Nothing separates them, so which survive must not depend on the machine:
    the two the sort calls the ends, then the lowest index."""
    points = np.zeros((5, 2))
    assert nsga2.survivors(points, 3) == [0, 4, 1]
    assert nsga2.survivors(points, 3) == nsga2.survivors(points.copy(), 3)


def test_rank_follows_the_fronts():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.5]])
    rank, _ = nsga2.rank_and_crowding(points)
    assert rank.tolist() == [0, 2, 1, 1]
