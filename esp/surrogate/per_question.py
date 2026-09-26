"""A Predictor that learns from every question a network answered.

The v1 Predictor sees one row per measured network: twelve networks, twelve
rows. But each of those networks answered seventeen questions, and which ones
it got right carries far more information than the average does. Seen question
by question, the same runs are 204 training rows; a pool of 120 networks on 60
questions is 7,200.

So this predicts, for a network and a question, the chance the network answers
it correctly, from the network's structure (`esp.surrogate.predictor.features`)
next to the question's own features: how many documents it needs, and which
kind of question it is. A network's predicted accuracy on any set of questions
is the mean over them, which means it can predict a score on a mix of
questions it was never measured on -- the judge set's newer kinds, or a
workload heavy in aggregates.

Uncertainty comes from an ensemble trained on bootstrap resamples of the
*networks* (not the rows: rows from one network are not independent), and the
spread across members is what an upper-confidence search spends on.

Cost is predicted per network, in dollars per question, because the provider
reports cost per run rather than per question. Fitness is derived from the
two predictions with `esp.eval.pricing.fitness_dollars`, never learned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier

from esp.eval.pricing import fitness_dollars
from esp.eval.tasks import Task
from esp.genome.definition import Genome
from esp.surrogate.predictor import features

QUESTION_FEATURE_NAMES = ["hops", "aggregate", "temporal", "filtered", "compare",
                          "unanswerable", "numeric"]

_AGGREGATE = re.compile(r"serviced by depot D\d{2}\?|by depot D\d{2},|depot D\d{2} at its own"
                        r"|incidents in \d{4}|incident in \d{4}")
_TEMPORAL = re.compile(r"happened (in|before) \d{4}|incidents (in|before) \d{4} on")
_FILTERED = re.compile(r"more than \d+ hours late|were caused by|penalty (is )?above \d+")
_COMPARE = re.compile(r"exceed that of|more loading bays|How many of depots")
_UNANSWERABLE = re.compile(r"'not stated'")

# Below this many networks the ensemble has nothing to resample.
MIN_NETWORKS = 6


def question_features(task: Task) -> np.ndarray:
    """What a question asks, read from the question itself."""
    text = task.question
    unanswerable = (task.answer == "not stated")
    return np.array([
        float(task.hops),
        float(bool(_AGGREGATE.search(text))),
        float(bool(_TEMPORAL.search(text))),
        float(bool(_FILTERED.search(text))),
        float(bool(_COMPARE.search(text))),
        float(unanswerable and bool(_UNANSWERABLE.search(text))),
        float(bool(re.fullmatch(r"\d+", task.answer))),
    ])


@dataclass
class Observed:
    """One measured network, question by question."""

    genome: Genome
    right: dict[str, bool]
    dollars_per_question: float


def _rows(observed: list[Observed], tasks: dict[str, Task]):
    xs, ys = [], []
    for item in observed:
        net = features(item.genome)
        for task_id, correct in item.right.items():
            if task_id in tasks:
                xs.append(np.concatenate([net, question_features(tasks[task_id])]))
                ys.append(int(correct))
    return np.array(xs), np.array(ys)


class QuestionPredictor:
    """Chance of a right answer per (network, question), with an ensemble."""

    def __init__(self, seed: int = 0, members: int = 8, max_iter: int = 150):
        self.seed, self.members, self.max_iter = seed, members, max_iter
        self._accuracy: list = []
        self._cost: list = []
        self._constant: float | None = None
        self._dollars = 0.0

    @property
    def trained(self) -> bool:
        return bool(self._accuracy)

    def fit(self, observed: list[Observed], tasks: list[Task]) -> QuestionPredictor:
        lookup = {t.task_id: t for t in tasks}
        self._accuracy, self._cost = [], []
        answers = [c for o in observed for c in o.right.values()]
        self._constant = float(np.mean(answers)) if answers else 0.5
        self._dollars = (float(np.mean([o.dollars_per_question for o in observed]))
                         if observed else 0.0)
        if len(observed) < MIN_NETWORKS:
            return self
        rng = np.random.default_rng(self.seed)
        for member in range(self.members):
            chosen = [observed[i] for i in rng.integers(0, len(observed), len(observed))]
            xs, ys = _rows(chosen, lookup)
            if len(set(ys)) < 2:
                continue
            model = HistGradientBoostingClassifier(
                max_iter=self.max_iter, max_depth=3, learning_rate=0.08,
                random_state=self.seed + member)
            model.fit(xs, ys)
            self._accuracy.append(model)
            cost = GradientBoostingRegressor(n_estimators=30, max_depth=2,
                                             random_state=self.seed + member)
            cost.fit(np.vstack([features(o.genome) for o in chosen]),
                     [o.dollars_per_question for o in chosen])
            self._cost.append(cost)
        return self

    def accuracy(self, genomes: list[Genome], tasks: list[Task]) -> tuple[np.ndarray, np.ndarray]:
        """Predicted accuracy on `tasks` for each genome: (mean, spread)."""
        if not self.trained:
            flat = np.full(len(genomes), self._constant or 0.5)
            return flat, np.zeros(len(genomes))
        qs = np.vstack([question_features(t) for t in tasks])
        nets = np.vstack([features(g) for g in genomes])
        rows = np.hstack([np.repeat(nets, len(tasks), axis=0),
                          np.tile(qs, (len(genomes), 1))])
        per_member = [model.predict_proba(rows)[:, 1].reshape(len(genomes), len(tasks))
                      .mean(axis=1) for model in self._accuracy]
        stacked = np.array(per_member)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def dollars(self, genomes: list[Genome]) -> np.ndarray:
        if not self._cost:
            return np.full(len(genomes), self._dollars)
        matrix = np.vstack([features(g) for g in genomes])
        return np.mean([m.predict(matrix) for m in self._cost], axis=0)

    def fitness(self, genomes: list[Genome], tasks: list[Task],
                optimism: float = 0.0) -> np.ndarray:
        """Derived fitness. `optimism` above zero adds that many spreads of
        accuracy: an upper-confidence bound, so uncertain candidates get tried."""
        mean, spread = self.accuracy(genomes, tasks)
        dollars = self.dollars(genomes)
        agents = [len(g.reachable()) for g in genomes]
        return np.array([fitness_dollars(a + optimism * s, d, n)
                         for a, s, d, n in zip(mean, spread, dollars, agents, strict=True)])
