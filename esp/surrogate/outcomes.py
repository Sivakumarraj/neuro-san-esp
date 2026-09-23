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
  bug.** Reported on its own, token cost cross-validates around -0.6 on the
  committed population, negative in every seed tried. Scalarised into one
  target it was hidden, because the accuracy term carried the combined score.

**And measuring it separately is not enough -- it has to be measured against
the right null.** A first version of this module reported that -0.6 as though
zero were the no-signal baseline. It is not. Cross-validation on twelve samples
manufactures negative rank correlation on its own: hold out a high value, the
training mean drops, the model predicts low, and the held-out prediction is
wrong in a direction that correlates. `permutation_null` measures that by
destroying the relationship and re-running the identical procedure. On the
committed population the token null is about -0.2, not 0, so part of the
published effect was the procedure and part looks real -- and twelve samples
cannot separate them. Every rank correlation here is now reported against its
own permutation null.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    # Median rank correlation this same procedure produces on the same data
    # with the relationship destroyed. Empty when the null was not measured;
    # never assumed to be zero, because on small samples it is not.
    nulls: dict[str, float] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return self.derived is not None and self.derived.measured

    def margin(self, name: str) -> float | None:
        """How far an objective beats its own no-signal baseline.

        This is the number that means something, and the reason the raw
        spearman alone was misleading: an objective at -0.53 against a null of
        about -0.17 is not "anti-predicted at -0.53", it is roughly 0.35 below
        a baseline that was already negative. Both halves of that subtraction
        move with the fold seed, so the margin is worth quoting as a range
        (medians -0.35 and -0.37 over 20 seeds at 12 and 40 shuffles; `make
        figures`) and never to three decimals.
        """
        quality = self.per_outcome.get(name)
        if quality is None or quality.spearman is None:
            return None
        return quality.spearman - self.nulls.get(name, 0.0)

    def useless_outcomes(self) -> list[str]:
        """Objectives whose model ranks no better than its own null.

        Named rather than averaged away. A predictor that is reliably wrong
        about an objective is a fact about the feature set, and it belongs in
        the report instead of inside a mean.

        Compared against the permutation null where one was measured, and
        against a flat +0.2 otherwise. The first version compared against
        +0.2 always, which on twelve samples judges an objective against a
        baseline the procedure cannot reach.
        """
        out = []
        for name, quality in sorted(self.per_outcome.items()):
            if not quality.measured or quality.spearman is None:
                continue
            floor = self.nulls.get(name, 0.0) + 0.2
            if quality.spearman <= floor:
                out.append(name)
        return out

    def gated_outcomes(self) -> list[str]:
        """Objectives to drop from the derived fitness, not merely warn about.

        Deliberately stricter than `useless_outcomes`: an objective is gated
        only when its own permutation null was actually measured. Without a
        measured null the comparison falls back to a flat +0.2, and on twelve
        samples that is a baseline the procedure cannot reach -- gating on it
        would drop an objective for being small-sample rather than for being
        wrong. Warn on suspicion; act only on evidence.

        Gating is a measurement, never a hardcoded list. An objective that
        starts predicting comes back on its own at the next generation,
        because this is recomputed from the data every time.
        """
        return [name for name in self.useless_outcomes() if name in self.nulls]

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
                       "beats_random": quality.beats_random,
                       "null": self.nulls.get(name),
                       "margin": self.margin(name)}
                for name, quality in sorted(self.per_outcome.items())},
            "useless_outcomes": self.useless_outcomes(),
        }

    def __str__(self) -> str:
        if not self.measured:
            return (f"outcome surrogate on {self.samples} samples: NOT "
                    f"MEASURED -- cross-validation needs {MIN_SAMPLES}")
        parts = []
        for name, quality in sorted(self.per_outcome.items()):
            if quality.spearman is None:
                parts.append(f"{name} not measured")
                continue
            shown = f"{name} {quality.spearman:+.3f}"
            if name in self.nulls:
                shown += f" (null {self.nulls[name]:+.3f})"
            parts.append(shown)
        derived = self.derived.spearman if self.derived else None
        head = ("fitness not measured" if derived is None
                else f"fitness {derived:+.3f}")
        return (f"outcome surrogate on {self.samples} samples: {head} "
                f"({', '.join(parts)})")


# How many shuffles the permutation null averages over. Each one re-runs the
# whole cross-validation, so this is the expensive part of a quality report --
# which is why it is opt-in rather than on by default.
NULL_TRIALS = 12


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


def _cross_validate(matrix: np.ndarray, values: np.ndarray,
                    folds: list, seed: int) -> np.ndarray:
    """Out-of-fold predictions for one objective.

    Extracted so the permutation null runs the *identical* procedure on
    shuffled targets. A null measured by a different code path would not be a
    null of this measurement.
    """
    predictions = np.zeros(len(values))
    for train_index, test_index in folds:
        model = _model(seed)
        model.fit(matrix[train_index], values[train_index])
        predictions[test_index] = model.predict(matrix[test_index])
    return predictions


def permutation_null(matrix: np.ndarray, values: np.ndarray, folds: list,
                     seed: int = 0, trials: int = NULL_TRIALS) -> float:
    """The rank correlation this procedure yields when there is nothing to find.

    Shuffle the targets against the features, so no relationship can survive,
    then cross-validate exactly as the real measurement does. The median over
    `trials` shuffles is the baseline the real figure has to beat.

    On small samples this comes back *negative*, not zero, and that is the
    whole point of measuring it. Holding out a high value drags the training
    mean down, the model predicts low, and the error correlates with the truth
    in the wrong direction. Reporting a real -0.61 against an assumed null of
    0 turns a modest effect into a dramatic one.

    **One limit, and it decides which objective this can be trusted on.** A
    permutation only destroys a relationship if permuting moves the values.
    Accuracy takes four distinct values across the twelve committed
    measurements, so a shuffle often maps a value onto an identical one and
    leaves the ordering largely intact -- one shuffled draw scored +0.88. Its
    null is therefore weak. Token cost is distinct in all twelve, so its null
    is sound, and token cost is the objective the published finding is about.
    A caller reporting this on a tied objective should say so.
    """
    rng = np.random.default_rng(seed)
    found = []
    for trial in range(trials):
        shuffled = values[rng.permutation(len(values))]
        if shuffled.std() <= 1e-9:
            continue
        found.append(_spearman(_cross_validate(matrix, shuffled, folds, trial),
                               shuffled))
    return float(np.median(found)) if found else 0.0


class OutcomeSurrogate:
    """Predicts each outcome objective; derives fitness from the predictions."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.models: dict[str, GradientBoostingRegressor] = {}
        self.trained = False
        self._fallback: dict[str, float] = {}
        # Objectives whose model is measurably worse than its own null. Their
        # predictions are replaced by the population mean, so the term stays
        # on the fitness scale but carries no ordering.
        self.gated: set[str] = set()

    def fit(self, genomes: list[Genome], outcomes: list[Outcome],
            gated: Iterable[str] | None = None) -> None:
        """Train one model per objective.

        `gated` names objectives to exclude from `predict`, normally
        `OutcomeQuality.gated_outcomes()` from the quality report computed on
        this same population. The models are still fitted and kept, so the
        exclusion is a prediction-time decision that reverses itself the
        moment the objective starts ranking.
        """
        self.gated = set(gated or ())
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
            # A gated objective takes the same path as one that never fitted:
            # the training mean. Constant across the batch, so it drops out of
            # the ranking while keeping the derived fitness comparable in
            # magnitude to a measured one.
            model = None if name in self.gated else self.models.get(name)
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
        """Whether `predict` carries ordering information at all.

        Gating every objective leaves `predict` returning a constant, and a
        search that sorts by a constant is a random search. Phase C prints
        that fact rather than a meaningless "best predicted" score, so it has
        to be visible here.
        """
        usable = [name for name in PREDICTED_OUTCOMES
                  if name in self.models and name not in self.gated]
        return self.trained and bool(usable)

    def report_quality(self, genomes: list[Genome], outcomes: list[Outcome],
                       seed: int = 0,
                       null_trials: int = 0) -> OutcomeQuality:
        """Cross-validated, per objective and for the derived fitness.

        `null_trials` above zero also measures each objective's permutation
        null, which is what makes the rank correlations interpretable. It is
        off by default because it costs `null_trials` extra cross-validations
        per objective, and an hourly service wake should not pay that to
        rediscover a constant.
        """
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
            predictions = _cross_validate(matrix, values, folds, seed)
            held[name] = predictions
            rho = _spearman(predictions, values)
            mae = float(np.mean(np.abs(predictions - values)))
            if null_trials > 0:
                report.nulls[name] = permutation_null(
                    matrix, values, folds, seed=seed, trials=null_trials)
            # Beating the null, not beating zero. The two thresholds differ by
            # the null itself -- about 0.13 for token cost on this population,
            # and enough to change the verdict. (Not 0.4: that is the margin,
            # which an earlier version of this comment confused with the gap
            # between the two baselines.)
            floor = report.nulls.get(name, 0.0) + 0.2
            report.per_outcome[name] = Quality(samples, rho, mae, rho > floor)

        truth = scalarise(targets["accuracy"], targets["tokens"], agents)
        derived = scalarise(held["accuracy"], held["tokens"], agents)
        rho = _spearman(derived, truth)
        report.derived = Quality(samples, rho,
                                 float(np.mean(np.abs(derived - truth))),
                                 rho > 0.2)
        return report
