"""NSGA-II's two orderings, for objectives that are all minimised.

ESP evolves its Prescriptors on several outcomes at once and keeps a front of
them, not one winner: no accuracy is worth any cost, so the choice between the
two is left to whoever reads the front. NSGA-II (Deb et al., 2002) is how the
ESP papers do it, and its two orderings are all of it that is needed here:

* **rank**: the front a candidate sits in. The first front is every candidate
  no other candidate dominates; the second is that again without the first.
* **crowding**: within a front, how far a candidate's neighbours are on each
  objective. The ends of a front are kept first, then the loneliest points, so
  a front stays spread out instead of bunching where it is easy to be.

One candidate dominates another when it is no worse on every objective and
better on at least one. Give a maximised objective negated.
"""

from __future__ import annotations

import numpy as np


def dominance(objectives: np.ndarray) -> np.ndarray:
    """A square table: entry [i, j] says candidate i dominates candidate j."""
    no_worse = (objectives[:, None, :] <= objectives[None, :, :]).all(axis=2)
    better = (objectives[:, None, :] < objectives[None, :, :]).any(axis=2)
    return no_worse & better


def fronts(objectives: np.ndarray) -> list[list[int]]:
    """Every candidate's index, front by front, the non-dominated first."""
    dominates = dominance(np.asarray(objectives, dtype=float))
    beaten_by = dominates.sum(axis=0)
    left = np.ones(len(dominates), dtype=bool)
    out: list[list[int]] = []
    while left.any():
        front = np.flatnonzero(left & (beaten_by == 0))
        out.append(front.tolist())
        left[front] = False
        beaten_by = beaten_by - dominates[front].sum(axis=0)
    return out


def crowding(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """How alone each member of one front is. An end of the front is infinitely
    alone, so the extremes of every objective are never crowded out."""
    values = np.asarray(objectives, dtype=float)[front]
    distance = np.zeros(len(front))
    if len(front) <= 2:
        return np.full(len(front), np.inf)
    for column in values.T:
        order = np.argsort(column, kind="stable")
        distance[order[0]] = distance[order[-1]] = np.inf
        span = column[order[-1]] - column[order[0]]
        if span > 0:
            distance[order[1:-1]] += (column[order[2:]] - column[order[:-2]]) / span
    return distance


def rank_and_crowding(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each candidate's front number and its crowding distance within that front."""
    rank = np.zeros(len(objectives), dtype=int)
    crowd = np.zeros(len(objectives))
    for number, front in enumerate(fronts(objectives)):
        rank[front] = number
        crowd[front] = crowding(objectives, front)
    return rank, crowd


def survivors(objectives: np.ndarray, size: int) -> list[int]:
    """The `size` candidates the next generation keeps: whole fronts in order,
    and of the front that does not fit, its loneliest members. A tie goes to the
    lower index, so the same candidates survive on every machine."""
    kept: list[int] = []
    for front in fronts(objectives):
        if len(kept) + len(front) <= size:
            kept += front
            continue
        distance = crowding(objectives, front)
        order = sorted(range(len(front)), key=lambda i: (-distance[i], front[i]))
        kept += [front[i] for i in order[:size - len(kept)]]
        break
    return kept
