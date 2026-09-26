"""The per-question Predictor, held to a relationship planted where the truth is
known. Each network gets a real chance of answering each kind of question, and
its measured answers are coin flips at that chance -- the noise a real run has.

The planted world is built so that ability differs by kind: more agents help
whole-company totals and hurt simple joins. A network-level model sees only
the average, so it cannot say which network is best at totals; a per-question
model can, and that is the point of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_training import population

from esp.eval.suites import JUDGE, SELECT
from esp.surrogate.per_question import MIN_NETWORKS, Observed, QuestionPredictor, question_features
from esp.surrogate.predictor import features


def chance(genome, task) -> float:
    agents = len(genome.reachable())
    tier = features(genome)[10]
    q = question_features(task)
    # Totals reward more agents; joins reward fewer.
    logit = (-0.6 + 0.55 * (agents - 3) if q[1]
             else 2.0 - 0.9 * (agents - 3) - 0.08 * q[0])
    return float(1 / (1 + np.exp(-(logit + 0.3 * tier))))


def observe(genomes, tasks, seed=0) -> list[Observed]:
    rng = np.random.default_rng(seed)
    return [Observed(g, {t.task_id: bool(rng.random() < chance(g, t)) for t in tasks},
                     0.004 + 0.001 * len(g.reachable())) for g in genomes]


def true_accuracy(genomes, tasks) -> np.ndarray:
    return np.array([np.mean([chance(g, t) for t in tasks]) for g in genomes])


AGGREGATES = [t for t in JUDGE if question_features(t)[1]]


def test_it_ranks_unseen_networks_on_a_mix_it_was_not_trained_on():
    genomes = population(70, seed=3)
    train, held = genomes[:40], genomes[40:]
    model = QuestionPredictor(seed=0).fit(observe(train, SELECT), SELECT)
    predicted, _ = model.accuracy(held, AGGREGATES)
    rho = spearmanr(predicted, true_accuracy(held, AGGREGATES)).statistic
    assert rho > 0.6, rho


def test_a_network_level_average_cannot_do_that():
    """The baseline: rank by each network's overall accuracy, which is all a
    network-level Predictor learns from. On totals alone it ranks worse."""
    genomes = population(70, seed=3)
    held = genomes[40:]
    model = QuestionPredictor(seed=0).fit(observe(genomes[:40], SELECT), SELECT)
    per_question, _ = model.accuracy(held, AGGREGATES)
    overall = true_accuracy(held, SELECT)       # even a perfect overall estimate
    truth = true_accuracy(held, AGGREGATES)
    assert (spearmanr(per_question, truth).statistic
            > spearmanr(overall, truth).statistic + 0.2)


def test_more_networks_train_a_better_predictor():
    def rho(train, seed):
        genomes = population(train + 30, seed=seed + 5)
        model = QuestionPredictor(seed=seed, members=4).fit(
            observe(genomes[:train], SELECT, seed), SELECT)
        predicted, _ = model.accuracy(genomes[train:], SELECT)
        return spearmanr(predicted, true_accuracy(genomes[train:], SELECT)).statistic
    few = np.mean([rho(8, s) for s in range(3)])
    many = np.mean([rho(40, s) for s in range(3)])
    assert many > few, (few, many)


def test_its_uncertainty_shrinks_with_data():
    genomes = population(70, seed=4)
    held = genomes[50:]
    small = QuestionPredictor(seed=1).fit(observe(genomes[:8], SELECT), SELECT)
    large = QuestionPredictor(seed=1).fit(observe(genomes[:50], SELECT), SELECT)
    assert large.accuracy(held, SELECT)[1].mean() < small.accuracy(held, SELECT)[1].mean()


def test_optimism_only_ever_raises_a_candidate():
    genomes = population(30, seed=2)
    model = QuestionPredictor(seed=0).fit(observe(genomes[:20], SELECT), SELECT)
    plain = model.fitness(genomes[20:], SELECT)
    hopeful = model.fitness(genomes[20:], SELECT, optimism=1.0)
    assert (hopeful >= plain).all() and (hopeful > plain).any()


def test_the_same_data_and_seed_train_the_same_model():
    genomes = population(20, seed=6)
    data = observe(genomes, SELECT)
    first = QuestionPredictor(seed=5).fit(data, SELECT).accuracy(genomes, SELECT)[0]
    second = QuestionPredictor(seed=5).fit(data, SELECT).accuracy(genomes, SELECT)[0]
    assert np.array_equal(first, second)


@pytest.mark.parametrize("size", [0, 1, MIN_NETWORKS - 1])
def test_below_the_minimum_it_says_it_cannot_rank(size):
    genomes = population(max(size, 1), seed=1)[:size]
    model = QuestionPredictor().fit(observe(genomes, SELECT), SELECT)
    assert not model.trained
    mean, spread = model.accuracy(population(3), SELECT)
    assert np.ptp(mean) == 0 and not spread.any()


def test_question_features_read_the_kind():
    from esp.eval.judge_plus import JUDGE_PLUS, kind_of
    names = ["temporal", "filtered", "compare", "unanswerable"]
    for task in JUDGE_PLUS:
        flags = question_features(task)[2:6]
        assert flags[names.index(kind_of(task))] == 1 and flags.sum() == 1, task.question
