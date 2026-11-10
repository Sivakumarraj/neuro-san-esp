"""The Prescriptor: a neural network from a context to an action, evolved against
the Predictor. The half of ESP this repository did without until now.

ESP (Francon et al., GECCO 2020) has two learned models:

* the **Predictor** maps a context and an action to the outcomes they lead to,
  and is trained on what was measured;
* the **Prescriptor** maps a context to the action to take, and is not trained on
  data at all. Its weights are evolved, each candidate scored by asking the
  Predictor what its prescriptions would lead to. Nothing is paid for until the
  best of them are tried for real.

Here a context is one question's features, an action is one of a fixed list of
networks, and the outcomes are the chance of a right answer and the dollars it
costs. `evolve` takes the Predictor's answers as two tables, one row a context
and one column an action, so it never sees a measured outcome of the contexts it
is later judged on: it cannot, they are not in the tables.

Two outcomes are evolved at once with NSGA-II (`esp.nsga2`), accuracy up and
dollars down, and the result is the front: the prescriptors no other one beats
on both. Choosing one of them is a separate, stated step (`esp.prescription`).

The network is small on purpose: one hidden layer of `HIDDEN` tanh units, an
output per action, and the action with the highest output is prescribed. Its
inputs are standardised with the training contexts' mean and spread, which are
kept with it so a new context is scaled the same way.

The first generation holds one prescriptor per action that prescribes it
everywhere, so the front can never be worse on the Predictor than the best
single action: a Prescriptor has to earn its place over prescribing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from esp import nsga2

HIDDEN = 8
POPULATION = 100
GENERATIONS = 100
CROSSOVER_RATE = 0.5      # chance a weight comes from the second parent
MUTATION_RATE = 0.1       # chance a weight is nudged
MUTATION_SCALE = 0.3      # standard deviation of the nudge
SEED = 20261110


def size(inputs: int, actions: int) -> int:
    """How many weights one prescriptor has."""
    return (inputs + 1) * HIDDEN + (HIDDEN + 1) * actions


def _outputs(weights: np.ndarray, scaled: np.ndarray, actions: int) -> np.ndarray:
    """Every candidate's output for every context: (candidates, contexts, actions)."""
    inputs = scaled.shape[1]
    split = inputs * HIDDEN
    first = weights[:, :split].reshape(-1, inputs, HIDDEN)
    first_bias = weights[:, split:split + HIDDEN]
    rest = weights[:, split + HIDDEN:]
    second = rest[:, :HIDDEN * actions].reshape(-1, HIDDEN, actions)
    second_bias = rest[:, HIDDEN * actions:]
    hidden = np.tanh(np.einsum("nd,pdh->pnh", scaled, first) + first_bias[:, None, :])
    return np.einsum("pnh,phk->pnk", hidden, second) + second_bias[:, None, :]


@dataclass(frozen=True)
class Prescriptor:
    """One evolved network. `prescribe` is all that a deployment runs."""

    weights: np.ndarray
    actions: int
    centre: np.ndarray
    spread: np.ndarray

    def prescribe(self, features: np.ndarray) -> np.ndarray:
        """The index of the action for each context. A tie goes to the first."""
        scaled = (np.atleast_2d(features) - self.centre) / self.spread
        return _outputs(self.weights[None, :], scaled, self.actions)[0].argmax(axis=1)


@dataclass(frozen=True)
class Evolved:
    """A member of the front, with what the Predictor says of it."""

    prescriptor: Prescriptor
    accuracy: float               # predicted share of right answers on the training contexts
    dollars: float                # predicted dollars a question there
    prescribed: tuple[int, ...]   # its action for each training context


def always(action: int, inputs: int, actions: int) -> np.ndarray:
    """The weights of the prescriptor that prescribes one action for every context."""
    weights = np.zeros(size(inputs, actions))
    weights[len(weights) - actions + action] = 1.0
    return weights


def _scores(prescribed: np.ndarray, right: np.ndarray, dollars: np.ndarray) -> np.ndarray:
    """(minus accuracy, dollars) for each candidate: both to be minimised."""
    rows = np.arange(right.shape[0])
    return np.column_stack([-right[rows, prescribed].mean(axis=1),
                            dollars[rows, prescribed].mean(axis=1)])


def evolve(features: np.ndarray, right: np.ndarray, dollars: np.ndarray, *,
           seed: int = SEED, population: int = POPULATION,
           generations: int = GENERATIONS) -> list[Evolved]:
    """The front of prescriptors for these contexts, cheapest first.

    `right` and `dollars` are the Predictor: for each context (row) and action
    (column), the chance of a right answer and the dollars it would cost.
    Prescriptors that prescribe the same action for every training context are
    one member; the first found is kept.
    """
    features = np.asarray(features, dtype=float)
    right = np.asarray(right, dtype=float)
    dollars = np.asarray(dollars, dtype=float)
    contexts, actions = right.shape
    if features.shape[0] != contexts or dollars.shape != right.shape:
        raise ValueError("one row of features, right and dollars for every context")
    if population < actions:
        raise ValueError("the population cannot hold one constant prescriptor per action")
    inputs = features.shape[1]
    centre = features.mean(axis=0)
    spread = features.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    scaled = (features - centre) / spread

    rng = np.random.default_rng(seed)
    genes = size(inputs, actions)
    parents = rng.normal(0.0, 1.0, size=(population, genes))
    for action in range(actions):
        parents[action] = always(action, inputs, actions)

    def prescriptions(weights: np.ndarray) -> np.ndarray:
        return _outputs(weights, scaled, actions).argmax(axis=2)

    scores = _scores(prescriptions(parents), right, dollars)
    for _ in range(generations):
        rank, crowd = nsga2.rank_and_crowding(scores)
        # Binary tournaments: the better front wins, then the lonelier candidate.
        a, b = rng.integers(0, population, size=(2, 2 * population))
        a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] >= crowd[b]))
        chosen = np.where(a_wins, a, b)
        first, second = parents[chosen[:population]], parents[chosen[population:]]
        children = np.where(rng.random((population, genes)) < CROSSOVER_RATE, second, first)
        nudged = rng.random((population, genes)) < MUTATION_RATE
        children = children + nudged * rng.normal(0.0, MUTATION_SCALE, size=(population, genes))

        everyone = np.vstack([parents, children])
        all_scores = np.vstack([scores, _scores(prescriptions(children), right, dollars)])
        kept = nsga2.survivors(all_scores, population)
        parents, scores = everyone[kept], all_scores[kept]

    seen: set[tuple[int, ...]] = set()
    front: list[Evolved] = []
    prescribed = prescriptions(parents)
    for i in nsga2.fronts(scores)[0]:
        pattern = tuple(int(action) for action in prescribed[i])
        if pattern in seen:
            continue
        seen.add(pattern)
        front.append(Evolved(Prescriptor(parents[i].copy(), actions, centre, spread),
                             float(-scores[i, 0]), float(scores[i, 1]), pattern))
    return sorted(front, key=lambda member: (member.dollars, -member.accuracy))
