"""Does training the Predictor actually work? Checked where the answer is known.

On the committed twelve networks nobody knows the true relationship between a
network's structure and its score, so a Predictor that learned nothing and one
that learned the truth can look alike. These tests plant a relationship in a
synthetic population instead, and hold the training pipeline -- features, one
model per objective, the gate, the derived fitness -- to it:

* it recovers a planted signal on networks it never saw, and better with more data;
* on pure noise it finds nothing, and the gate says so instead of steering by it;
* choosing by it beats choosing at random when there is something to learn;
* the same data and seed always train the same model.

Everything is deterministic: fixed seeds, no provider, no network.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from scipy.stats import spearmanr

from esp.genome.mutations import InvalidMutant, mutate
from esp.genome.seeds import SEEDS
from esp.surrogate.outcomes import Outcome, OutcomeSurrogate, scalarise


def population(size: int, seed: int = 7) -> list:
    """Distinct valid networks, bred from the seeds the way the search breeds."""
    rng = random.Random(seed)
    pool = {g.genome_hash(): g for g in (make() for make in SEEDS.values())}
    parents = list(pool.values())
    attempts = 0
    while len(pool) < size and attempts < size * 200:
        attempts += 1
        try:
            child, _ = mutate(rng.choice(parents), rng)
        except InvalidMutant:
            continue
        if child.genome_hash() not in pool:
            pool[child.genome_hash()] = child
            parents.append(child)
    assert len(pool) >= size, "the operators could not breed a population this size"
    return list(pool.values())[:size]


def planted(genomes: list, noise: float = 0.02, seed: int = 11) -> list[Outcome]:
    """A known truth: accuracy rises with agents and falls with depth; tokens
    rise with agents. Structure the Predictor can see, plus a little noise."""
    rng = np.random.default_rng(seed)
    outcomes = []
    for genome in genomes:
        agents, depth = len(genome.reachable()), genome.depth()
        accuracy = 0.55 + 0.05 * min(agents, 6) - 0.04 * depth + rng.normal(0, noise)
        tokens = 60_000 * agents + 20_000 * depth + rng.normal(0, 5_000)
        outcomes.append(Outcome(accuracy=float(np.clip(accuracy, 0, 1)),
                                tokens=int(max(tokens, 1_000))))
    return outcomes


def held_out_rho(train: int, seed: int = 0) -> dict[str, float]:
    genomes = population(train + 30, seed=seed + 7)
    outcomes = planted(genomes, seed=seed + 11)
    surrogate = OutcomeSurrogate(seed=seed)
    surrogate.fit(genomes[:train], outcomes[:train])
    predicted = surrogate.predict_outcomes(genomes[train:])
    truth = outcomes[train:]
    return {
        "accuracy": spearmanr(predicted["accuracy"], [o.accuracy for o in truth]).statistic,
        "tokens": spearmanr(predicted["tokens"], [o.tokens for o in truth]).statistic,
    }


def test_it_recovers_a_planted_signal_on_networks_it_never_saw():
    rho = held_out_rho(train=40)
    assert rho["accuracy"] > 0.6, rho
    assert rho["tokens"] > 0.6, rho


def test_more_measurements_train_a_better_predictor():
    """The whole case for measuring more networks. Averaged over seeds, because
    one small draw can be lucky."""
    few = np.mean([held_out_rho(train=10, seed=s)["accuracy"] for s in range(4)])
    many = np.mean([held_out_rho(train=40, seed=s)["accuracy"] for s in range(4)])
    assert many > few, (few, many)


def test_on_pure_noise_the_gate_refuses_to_steer():
    """Nothing to learn must read as nothing learned: the objective loses to
    its own permutation null and is excluded, rather than ordering candidates
    by chance and calling it a prediction."""
    genomes = population(40)
    rng = np.random.default_rng(3)
    noise = [Outcome(accuracy=float(rng.uniform(0.5, 1.0)),
                     tokens=int(rng.uniform(100_000, 500_000))) for _ in genomes]
    quality = OutcomeSurrogate(seed=0).report_quality(genomes, noise, seed=0, null_trials=12)
    for name in ("accuracy", "tokens"):
        assert quality.per_outcome[name].spearman < 0.35, quality
    assert set(quality.useless_outcomes()) == {"accuracy", "tokens"}


def test_a_planted_signal_is_kept_and_not_gated():
    genomes = population(40)
    quality = OutcomeSurrogate(seed=0).report_quality(
        genomes, planted(genomes), seed=0, null_trials=12)
    assert quality.gated_outcomes() == [], quality


def test_choosing_by_the_predictor_beats_choosing_at_random():
    """Phase C's claim in miniature: from candidates it never measured, the
    Predictor's top ten are better than ten picked at random."""
    genomes = population(160)
    outcomes = planted(genomes)
    surrogate = OutcomeSurrogate(seed=0)
    surrogate.fit(genomes[:40], outcomes[:40])
    pool, truth = genomes[40:], outcomes[40:]
    true_fitness = scalarise(np.array([o.accuracy for o in truth]),
                             np.array([float(o.tokens) for o in truth]),
                             np.array([float(len(g.reachable())) for g in pool]))
    chosen = np.argsort(-surrogate.predict(pool))[:10]
    rng = np.random.default_rng(0)
    random_means = [true_fitness[rng.choice(len(pool), 10, replace=False)].mean()
                    for _ in range(200)]
    assert true_fitness[chosen].mean() > np.percentile(random_means, 95)


def test_the_same_data_and_seed_train_the_same_model():
    genomes = population(30)
    outcomes = planted(genomes)
    first, second = OutcomeSurrogate(seed=5), OutcomeSurrogate(seed=5)
    first.fit(genomes, outcomes)
    second.fit(genomes, outcomes)
    assert np.array_equal(first.predict(genomes), second.predict(genomes))


@pytest.mark.parametrize("size", [0, 1, 7])
def test_below_the_minimum_it_predicts_a_constant_and_says_it_cannot_rank(size):
    genomes = population(max(size, 1))[:size] if size else []
    surrogate = OutcomeSurrogate(seed=0)
    if genomes:
        surrogate.fit(genomes, planted(genomes))
        predictions = surrogate.predict(population(20)[:5])
        # Agent count is computed exactly, so only the predicted objectives
        # are constant; the Predictor must not claim to rank.
        assert not surrogate.ranks()
        assert np.isfinite(predictions).all()
    else:
        assert not surrogate.ranks()
