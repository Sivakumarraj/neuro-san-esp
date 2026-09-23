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
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from esp.config import model_rank
from esp.genome.definition import Genome

FEATURE_NAMES = [
    "agents", "depth", "edges", "mean_branching", "max_branching",
    "searchers", "searcher_fraction", "leaves", "top_degree",
    "mean_model_tier", "max_model_tier", "mean_instruction_chars",
    "total_instruction_chars",
]



def _tier(model: str) -> float:
    """Where a model sits on the cost ladder, as a number the surrogate can use.

    A property of the model itself: its rung (cheap or strong), then its
    release within the rung, so gemini-3.1-flash-lite < gemini-3.5-flash-lite
    < gemini-3.5-flash and claude-haiku < claude-sonnet on any machine.

    It used to be the model's position in the configured ladder, with anything
    off the ladder placed above everything on it. Every committed network's
    workers run gemini-3.1-flash-lite, which is not on the default ladder, so
    the cheapest model in the population was encoded as the most expensive --
    and on a machine configured for another provider every committed model fell
    off the ladder at once and the feature went constant. Correcting it moved
    the token-cost margin from -0.47 to its current value and changed no
    exclusion verdict; docs/FINDINGS.md records both.

    Never raises: an unknown model reads as the cheap rung, release 0.
    """
    rung, release = model_rank(model)
    return rung + release / 100


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
    """Rank correlation. Ranking is what matters -- the surrogate only has to
    order candidates, not price them.

    This was hand-rolled as `argsort(argsort(x))` to avoid "a scipy
    dependency" that the project already had: scikit-learn hard-requires
    scipy, so it has been installed all along. What the hand-rolled version
    bought instead was a bug. Ordinal ranks from a double argsort give tied
    values arbitrary distinct ranks in whatever order they arrived, where
    Spearman requires the tied values to share a midrank. So the answer moved
    with the order of the inputs: two identical vectors, permuted identically,
    scored -0.143 and +0.143 on the same data, and the correct answer was 0.

    A tree ensemble predicts a finite set of leaf averages, so ties in the
    predictions are ordinary rather than exotic -- they appeared in 8 of 20
    cross-validation seeds on the committed population. The published figures
    moved by at most 0.015 and no verdict changed, which is luck rather than
    justification: a repository whose argument is that its numbers were
    measured cannot have a measurement that depends on list order.
    """
    if len(a) < 3:
        return 0.0
    # A tolerance, not `== 0`. The variance of six identical floats comes out
    # at 1.1e-16 rather than zero, so an exact test let a constant vector reach
    # scipy, which warned and returned nan -- the right answer arrived only
    # because the nan was caught below. The old code compared integer ranks,
    # where exact zero was safe; this one compares the values. 1e-9 is the same
    # threshold `fit` and `report_quality` already use for a flat target.
    if np.asarray(a, dtype=float).std() < 1e-9:
        return 0.0        # no ordering to correlate against
    if np.asarray(b, dtype=float).std() < 1e-9:
        return 0.0
    result = spearmanr(a, b).statistic
    return 0.0 if np.isnan(result) else float(result)


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
