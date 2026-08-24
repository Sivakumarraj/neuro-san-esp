"""The Predictor half of ESP: estimate a genome's fitness without running it.

Real evaluation costs a minute and real money. The surrogate costs microseconds,
so evolution can search thousands of candidates and spend real evaluations only
on the elite. That trade is the entire reason this is ESP and not a plain
genetic algorithm.

The honest caveat, stated here because it governs how the results should be
read: a surrogate trained on tens of samples is weak. Its job is not to be
right, it is to rank -- to be better than random at telling a promising
topology from a hopeless one. `report_quality` measures exactly that and is
meant to be published even when the answer is unflattering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from esp.genome.definition import MODEL_TIERS, Genome

FEATURE_NAMES = [
    "agents", "depth", "edges", "mean_branching", "max_branching",
    "searchers", "searcher_fraction", "leaves", "top_degree",
    "mean_model_tier", "max_model_tier", "mean_instruction_chars",
    "total_instruction_chars",
]



def _tier(model: str) -> int:
    """Where a model sits on the cost ladder, as a number the surrogate can use.

    Never raises on an unknown name. Calling MODEL_TIERS.index() directly would
    crash feature extraction for any model absent from the list, and failover
    exists precisely to substitute models that are not in it -- a genome
    measured under a swapped model would take the surrogate down with it.
    Unknown models sort after the known ones, which is the honest ordering: we
    do not know what they cost.
    """
    try:
        return MODEL_TIERS.index(model)
    except ValueError:
        return len(MODEL_TIERS)


def features(genome: Genome) -> np.ndarray:
    """Structure and configuration only -- never anything measured.

    Using a measured quantity as a feature would let the surrogate cheat: it
    would need a real evaluation to predict a real evaluation, which defeats
    the purpose.
    """
    live = sorted(genome.reachable())
    agents = [genome.agents[name] for name in live]
    branching = [len([t for t in a.tools if t in live]) for a in agents]
    tiers = [_tier(a.model or genome.default_model) for a in agents]
    lengths = [len(a.instructions) for a in agents]
    searchers = sum(1 for a in agents if a.can_search)

    return np.array([
        len(live),
        genome.depth(),
        sum(branching),
        float(np.mean(branching)) if branching else 0.0,
        float(max(branching)) if branching else 0.0,
        searchers,
        searchers / len(live) if live else 0.0,
        sum(1 for b in branching if b == 0),
        len([t for t in genome.agents[genome.top].tools if t in live]),
        float(np.mean(tiers)) if tiers else 0.0,
        float(max(tiers)) if tiers else 0.0,
        float(np.mean(lengths)) if lengths else 0.0,
        float(sum(lengths)),
    ], dtype=float)


# Below this a GBM memorises rather than generalises, and KFold cannot make a
# held-out fold worth the name. One number, used by both fit and report_quality,
# because they have to agree about when there is enough data.
MIN_SAMPLES = 8


@dataclass
class Quality:
    """What cross-validation found, or that it could not run.

    `spearman` is None when there were too few samples to measure. It used to
    be 0.0 in that case, printed as `spearman=+0.000`, which is indistinguishable
    from a rank correlation that was computed and came out at zero. The project
    was reporting a placeholder as a measurement -- in a repository whose entire
    argument is that its numbers were measured. None cannot be mistaken for a
    result.
    """
    samples: int
    spearman: float | None
    mae: float | None
    beats_random: bool

    @property
    def measured(self) -> bool:
        return self.spearman is not None

    def __str__(self) -> str:
        if not self.measured:
            return (f"surrogate on {self.samples} samples: rank quality NOT "
                    f"MEASURED -- cross-validation needs {MIN_SAMPLES}. The "
                    f"predictor is untrained and returns a constant.")
        verdict = "ranks better than chance" if self.beats_random else "NO BETTER THAN CHANCE"
        return (f"surrogate on {self.samples} samples: spearman={self.spearman:+.3f} "
                f"mae={self.mae:.4f} -- {verdict}")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency. Ranking is what matters --
    the surrogate only has to order candidates, not price them."""
    if len(a) < 3:
        return 0.0
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    if rank_a.std() == 0 or rank_b.std() == 0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


class Surrogate:
    def __init__(self, seed: int = 0):
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=seed,
        )
        self.trained = False
        self._fallback = 0.0

    def fit(self, genomes: list[Genome], fitnesses: list[float]) -> None:
        matrix = np.vstack([features(g) for g in genomes])
        target = np.asarray(fitnesses, dtype=float)
        self._fallback = float(target.mean()) if len(target) else 0.0
        if len(target) >= MIN_SAMPLES and target.std() > 1e-9:
            self.model.fit(matrix, target)
            self.trained = True

    def predict(self, genomes: list[Genome]) -> np.ndarray:
        if not self.trained:
            return np.full(len(genomes), self._fallback)
        return self.model.predict(np.vstack([features(g) for g in genomes]))

    def ranks(self) -> bool:
        """Whether `predict` carries any ordering information at all.

        An untrained surrogate returns the mean of its targets for every
        genome. Sorting on that is a no-op: the "top 5 by predicted fitness"
        is the first five in generation order, and Phase D pays real provider
        budget for them believing they were selected. Callers must say so
        rather than print a ranking that does not exist.
        """
        return self.trained

    def report_quality(self, genomes: list[Genome], fitnesses: list[float],
                       seed: int = 0) -> Quality:
        """Cross-validated ranking quality. Reported whatever it says."""
        target = np.asarray(fitnesses, dtype=float)
        samples = len(target)
        if samples < MIN_SAMPLES or target.std() < 1e-9:
            return Quality(samples, None, None, False)

        matrix = np.vstack([features(g) for g in genomes])
        predictions = np.zeros(samples)
        folds = KFold(n_splits=min(5, samples), shuffle=True, random_state=seed)
        for train_idx, test_idx in folds.split(matrix):
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.9, random_state=seed)
            model.fit(matrix[train_idx], target[train_idx])
            predictions[test_idx] = model.predict(matrix[test_idx])

        rho = _spearman(predictions, target)
        mae = float(np.mean(np.abs(predictions - target)))
        return Quality(samples, rho, mae, rho > 0.2)
