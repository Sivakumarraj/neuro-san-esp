"""One model per outcome objective, with fitness derived from their predictions.

Written after review feedback from Babak Hodjat, a co-author of the ESP paper,
who read the first version and said:

    "I have a hard time understanding what the predictor surrogate is in the
    ESP for this general use-case. Typically, the surrogate model is one or
    more ML models that act as predictors for various outcome objectives we
    expect from the target we are optimizing... with the fitness being derived
    from the surrogate(s), but this is not very clear."

He is describing the paper's shape, and the first version was not that shape.
`esp/surrogate/predictor.py` trains a single model on the already-scalarised
fitness, so the Predictor learns the weighting rather than the world. This
predicts the outcomes and derives fitness from them, which is what ESP does.

Three things follow from the change, beyond matching the paper:

* **The weights stop being baked in.** Re-weighting accuracy against tokens no
  longer needs the surrogate retrained, because the surrogate never saw the
  weights.
* **Two of the four objectives need no model at all.** Agent count and depth
  are exact properties of a genome -- they can be counted, not guessed. Asking
  a regressor to estimate a number already in hand adds error for nothing.
* **Each objective becomes separately measurable, and that is what found the
  bug.** Reported on its own, token cost turns out to be *anti*-predicted:
  rank correlation around -0.6 on the committed population, consistently
  negative across every seed tried. Scalarised into one target it was hidden,
  because the accuracy term carried the combined score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from esp.genome.definition import Genome
from esp.surrogate.predictor import MIN_SAMPLES, Quality, _spearman, features

# The objectives a model is actually needed for. Agent count and depth are read
# off the genome exactly, so they are computed rather than predicted -- see the
# module docstring.
PREDICTED_OUTCOMES = ("accuracy", "tokens")


@dataclass
class Outcome:
    """One candidate's measured outcomes, as the surrogate needs them."""

    accuracy: float
    tokens: int


@dataclass
class OutcomeQuality:
    """Cross-validated quality, per objective and for the fitness derived from
    them. Reported per objective on purpose: a single combined number is what
    concealed the token model being worse than useless."""

    samples: int
    per_outcome: dict[str, Quality] = field(default_factory=dict)
    derived: Quality | None = None

    @property
    def measured(self) -> bool:
        return self.derived is not None and self.derived.measured

    def useless_outcomes(self) -> list[str]:
        """Objectives whose model ranks no better than chance, or worse.

        Named rather than averaged away. A predictor that is reliably wrong
        about an objective is a fact about the feature set, and it belongs in
        the report instead of inside a mean.
        """
        return [name for name, quality in sorted(self.per_outcome.items())
                if quality.measured and quality.spearman is not None
                and quality.spearman <= 0.2]

    def as_record(self) -> dict:
        """The history entry, shaped so older readers keep working.

        The top level is the derived-fitness Quality, which is what
        `surrogate_quality` has always held and what the PDF tables, the web
        page and the README check read. `per_outcome` is added beside it. A
        second key holding the same figures under a different name would let
        the two drift; a superset cannot.
        """
        derived = self.derived
        return {
            "samples": self.samples,
            "spearman": derived.spearman if derived else None,
            "mae": derived.mae if derived else None,
            "beats_random": bool(derived and derived.beats_random),
            "per_outcome": {
                name: {"spearman": quality.spearman, "mae": quality.mae,
                       "beats_random": quality.beats_random}
                for name, quality in sorted(self.per_outcome.items())},
            "useless_outcomes": self.useless_outcomes(),
        }

    def __str__(self) -> str:
        if not self.measured:
            return (f"outcome surrogate on {self.samples} samples: NOT "
                    f"MEASURED -- cross-validation needs {MIN_SAMPLES}")
        parts = []
        for name, quality in sorted(self.per_outcome.items()):
            shown = ("not measured" if quality.spearman is None
                     else f"{quality.spearman:+.3f}")
            parts.append(f"{name} {shown}")
        derived = self.derived.spearman if self.derived else None
        head = ("fitness not measured" if derived is None
                else f"fitness {derived:+.3f}")
        return (f"outcome surrogate on {self.samples} samples: {head} "
                f"({', '.join(parts)})")


def _model(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9,
        random_state=seed)


def scalarise(accuracy, tokens, agents) -> np.ndarray:
    """The project's fitness, over predicted outcomes rather than measured ones.

    The vectorised twin of `esp.evolve.loop.scalarise`, whose `min()` does not
    broadcast over the arrays a batch of predictions arrives as. The weights
    and the scale are imported, never restated, so the only thing that can
    drift is the shape of the arithmetic -- and
    `test_outcome_surrogate.py::test_the_two_scalarisations_agree` pins that
    against the scalar version on the measured population. A surrogate that
    ranks by a different objective than the search selects on is the quietest
    way to make a search meaningless.
    """
    from esp.evolve.loop import TOKEN_SCALE, WEIGHTS

    return (WEIGHTS["accuracy"] * np.asarray(accuracy, dtype=float)
            - WEIGHTS["tokens"] * np.minimum(
                np.asarray(tokens, dtype=float) / TOKEN_SCALE, 1.0)
            - WEIGHTS["agents"] * (np.asarray(agents, dtype=float) / 9.0))


class OutcomeSurrogate:
    """Predicts each outcome objective; derives fitness from the predictions."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.models: dict[str, GradientBoostingRegressor] = {}
        self.trained = False
        self._fallback: dict[str, float] = {}

    def fit(self, genomes: list[Genome], outcomes: list[Outcome]) -> None:
        matrix = np.vstack([features(g) for g in genomes])
        targets = {
            "accuracy": np.array([o.accuracy for o in outcomes], dtype=float),
            "tokens": np.array([float(o.tokens) for o in outcomes]),
        }
        self._fallback = {name: float(values.mean()) if len(values) else 0.0
                          for name, values in targets.items()}

        if len(genomes) < MIN_SAMPLES:
            return
        fitted = 0
        for name in PREDICTED_OUTCOMES:
            values = targets[name]
            if values.std() <= 1e-9:
                continue        # nothing to learn from a constant objective
            model = _model(self.seed)
            model.fit(matrix, values)
            self.models[name] = model
            fitted += 1
        self.trained = fitted == len(PREDICTED_OUTCOMES)

    def predict_outcomes(self, genomes: list[Genome]) -> dict[str, np.ndarray]:
        """Each objective, predicted or -- where it is exact -- computed."""
        matrix = np.vstack([features(g) for g in genomes])
        predicted: dict[str, np.ndarray] = {}
        for name in PREDICTED_OUTCOMES:
            model = self.models.get(name)
            predicted[name] = (
                model.predict(matrix) if model is not None
                else np.full(len(genomes), self._fallback.get(name, 0.0)))
        # Exact, not estimated. These are properties of the genome.
        predicted["agents"] = np.array(
            [float(len(g.reachable())) for g in genomes])
        return predicted

    def predict(self, genomes: list[Genome]) -> np.ndarray:
        outcomes = self.predict_outcomes(genomes)
        return scalarise(outcomes["accuracy"], outcomes["tokens"],
                         outcomes["agents"])

    def ranks(self) -> bool:
        """Whether `predict` carries ordering information at all."""
        return self.trained

    def report_quality(self, genomes: list[Genome], outcomes: list[Outcome],
                       seed: int = 0) -> OutcomeQuality:
        """Cross-validated, per objective and for the derived fitness."""
        samples = len(genomes)
        report = OutcomeQuality(samples=samples)
        if samples < MIN_SAMPLES:
            return report

        matrix = np.vstack([features(g) for g in genomes])
        agents = np.array([float(len(g.reachable())) for g in genomes])
        targets = {
            "accuracy": np.array([o.accuracy for o in outcomes], dtype=float),
            "tokens": np.array([float(o.tokens) for o in outcomes]),
        }
        folds = list(KFold(n_splits=min(5, samples), shuffle=True,
                           random_state=seed).split(matrix))

        held: dict[str, np.ndarray] = {}
        for name in PREDICTED_OUTCOMES:
            values = targets[name]
            if values.std() <= 1e-9:
                report.per_outcome[name] = Quality(samples, None, None, False)
                held[name] = np.full(samples, float(values.mean()))
                continue
            predictions = np.zeros(samples)
            for train_index, test_index in folds:
                model = _model(seed)
                model.fit(matrix[train_index], values[train_index])
                predictions[test_index] = model.predict(matrix[test_index])
            held[name] = predictions
            rho = _spearman(predictions, values)
            mae = float(np.mean(np.abs(predictions - values)))
            report.per_outcome[name] = Quality(samples, rho, mae, rho > 0.2)

        truth = scalarise(targets["accuracy"], targets["tokens"], agents)
        derived = scalarise(held["accuracy"], held["tokens"], agents)
        rho = _spearman(derived, truth)
        report.derived = Quality(samples, rho,
                                 float(np.mean(np.abs(derived - truth))),
                                 rho > 0.2)
        return report
