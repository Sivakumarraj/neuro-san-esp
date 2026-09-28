"""The context seam: the same Predictor, asked for a different workload, must
choose a different network when the planted world says it should.

The planted world is test_per_question's: more agents help whole-company
totals and hurt simple joins. A totals-heavy workload should therefore be
handed bigger teams than a join-heavy one, from the same candidates and the
same trained model. That is what makes a context an input rather than a label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_per_question import observe
from test_training import population

from esp.eval.suites import SELECT
from esp.evolve import pool
from esp.evolve.context import KINDS, Context, kind, predicted_accuracy, prescribe, strategy
from esp.surrogate.per_question import QuestionPredictor


@pytest.fixture(scope="module")
def trained():
    genomes = population(70, seed=3)
    model = QuestionPredictor(seed=0).fit(observe(genomes[:40], SELECT), SELECT)
    return model, genomes[40:]


def test_every_select_question_has_a_kind():
    kinds = {kind(t) for t in SELECT}
    assert kinds <= set(KINDS) and {"join", "aggregate"} <= kinds


def test_the_workload_changes_the_choice(trained):
    model, candidates = trained
    totals = prescribe(model, candidates, SELECT, Context({"aggregate": 1.0}))
    joins = prescribe(model, candidates, SELECT, Context({"join": 1.0}))

    def team(order):
        return np.mean([len(candidates[i].reachable()) for i in order[:5]])
    assert team(totals) > team(joins), (team(totals), team(joins))


def test_a_mixed_workload_sits_between_its_parts(trained):
    model, candidates = trained
    only_totals = predicted_accuracy(model, candidates, SELECT, Context({"aggregate": 1.0}))
    only_joins = predicted_accuracy(model, candidates, SELECT, Context({"join": 1.0}))
    half = predicted_accuracy(model, candidates, SELECT,
                              Context({"aggregate": 0.5, "join": 0.5}))
    assert np.allclose(half, (only_totals + only_joins) / 2)


def test_a_price_cap_puts_dearer_networks_last(trained):
    model, candidates = trained
    dollars = model.dollars(candidates)
    cap = float(np.median(dollars))
    order = prescribe(model, candidates, SELECT, Context({"join": 1.0}, cap))
    within = int((dollars <= cap).sum())
    assert all(dollars[i] <= cap for i in order[:within])
    assert all(dollars[i] > cap for i in order[within:])


def test_a_context_must_name_real_kinds_with_some_weight():
    with pytest.raises(ValueError):
        Context({"poetry": 1.0})
    with pytest.raises(ValueError):
        Context({"join": 0.0})


def test_it_plugs_into_the_pool_benchmark_as_a_strategy():
    """The seam is usable today: the pool benchmark can already compare a
    context-aware chooser with the others, on any measured pool."""
    rng = np.random.default_rng(0)
    ids = [t.task_id for t in SELECT[:20]]
    members = []
    for genome in pool.breed(30, seed=2):
        right = {k: bool(rng.random() < 0.6) for k in ids}
        members.append(pool.Member(genome.genome_hash(), genome.canonical(), right,
                                   float(np.mean(list(right.values()))), 1000, 0.1, 3))
    chooser = strategy(Context({"aggregate": 0.4, "join": 0.6}))
    found = pool.replicate(members, SELECT[:20], chooser, seed=1, budgets=(5, 10))
    assert set(found) == {5, 10} and all(np.isfinite(list(found.values())))
