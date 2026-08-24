"""Scoring, the fitness cache, and the surrogate.

Nothing here calls a language model. These are the properties that must hold
before any real evaluation is worth paying for.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from esp.eval import runner
from esp.eval.corpus_tool import search
from esp.eval.tasks import TASKS, build_tasks, normalise, score
from esp.eval.world import build_world, documents
from esp.genome.mutations import mutate
from esp.genome.seeds import designer_shaped, flat_pair, solo
from esp.surrogate.predictor import FEATURE_NAMES, Surrogate, features

# ------------------------------------------------------------------ the world

def test_world_is_deterministic():
    """Fitness compares candidates across generations. If the corpus differed
    between runs, those numbers would not be comparable at all."""
    assert documents(build_world()) == documents(build_world())
    assert [t.answer for t in build_tasks()] == [t.answer for t in build_tasks()]


def test_every_task_answer_is_findable_in_the_corpus():
    """A task whose answer appears nowhere is unanswerable, and would put a
    permanent ceiling on fitness that no topology could ever lift."""
    corpus = " ".join(documents(build_world()).values()).lower()
    for task in TASKS:
        assert task.answer.lower() in corpus or task.answer.isdigit(), task.task_id


def test_multi_hop_tasks_really_need_more_than_one_document():
    """If a single document answered a 'multi-hop' question, the task set would
    not be measuring what it claims to measure."""
    docs = documents(build_world())
    for task in TASKS:
        if task.hops < 2:
            continue
        holders = [name for name, body in docs.items()
                   if score(task.answer, body)]
        # The answer may sit in one document, but no single document also
        # contains the identifier the question starts from.
        starting_ids = [w for w in task.question.replace("?", "").split()
                        if w.startswith(("D0", "C-", "INC-"))]
        if not starting_ids or not holders:
            continue
        for holder in holders:
            assert not all(i in docs[holder] for i in starting_ids), \
                f"{task.task_id} answerable from {holder} alone"


# ------------------------------------------------------------------ scoring

@pytest.mark.parametrize("expected,produced,want", [
    ("Brindle", "Brindle", True),
    ("Brindle", "The depot is in Brindle.", True),
    ("Brindle", "brindle", True),
    ("4200", "the total is 4,200", True),
    ("4200", "4200", True),
    # the bug this guards: containment scored "4" as correct against the digits
    # inside an identifier, handing marks to a network that never searched
    ("4", "Incident INC-4429 was reported", False),
    ("4", "the depot has 4 bays", True),
    ("31", "312", False),
    ("Brindle", "Ashford", False),
    ("Brindle", "", False),
])
def test_score(expected, produced, want):
    assert score(expected, produced) is want


def test_normalise_strips_only_noise():
    assert normalise("  The Total is 4,200.  ") == "the total is 4200"


# ------------------------------------------------------------------ retrieval

def test_search_is_deterministic_and_identifier_weighted():
    first = search("depot D03 manager")
    assert first == search("depot D03 manager")
    assert first[0]["document"] == "depot-D03.txt"


def test_search_returns_nothing_for_an_empty_query():
    assert search("") == []


# ------------------------------------------------------------------ the cache

def test_cache_round_trips_and_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "NETWORK_DIR", tmp_path / "networks")

    genome = solo()
    calls = {"n": 0}

    def fake_ask(_path, _question):
        calls["n"] += 1
        return "Brindle", {}, 0.01

    monkeypatch.setattr(runner, "_ask", fake_ask)

    first = runner.evaluate(genome, tasks=TASKS[:3])
    assert calls["n"] == 3
    assert first.from_cache is False

    second = runner.evaluate(genome, tasks=TASKS[:3])
    assert calls["n"] == 3, "a cached genome must cost nothing"
    assert second.from_cache is True
    assert second.accuracy == first.accuracy


def test_environment_failure_is_not_cached(tmp_path, monkeypatch):
    """The bug this guards: an unset AGENT_TOOL_PATH once pinned a permanent
    zero on a topology that had never actually run."""
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "NETWORK_DIR", tmp_path / "networks")

    def always_raise(_path, _question):
        raise RuntimeError("no API key")

    monkeypatch.setattr(runner, "_ask", always_raise)

    with pytest.raises(EnvironmentError, match="refusing to cache"):
        runner.evaluate(solo(), tasks=TASKS[:3])
    assert not list((tmp_path / "cache").glob("*.json"))


def test_a_single_task_failure_is_still_cached(tmp_path, monkeypatch):
    """One flaky task is a property of the candidate; every task failing is a
    property of the machine. Only the second is refused."""
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "NETWORK_DIR", tmp_path / "networks")

    state = {"n": 0}

    def flaky(_path, _question):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("transient")
        return "Brindle", {}, 0.01

    monkeypatch.setattr(runner, "_ask", flaky)
    evaluation = runner.evaluate(solo(), tasks=TASKS[:3])
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    assert evaluation.accuracy < 1.0


def test_token_accounting_sums_nested_totals():
    accounting = {"a": {"total_tokens": 10}, "b": [{"total_tokens": 5}]}
    assert runner._total_tokens(accounting) == 15


# ------------------------------------------------------------------ surrogate

def test_feature_vector_matches_its_names():
    assert len(features(designer_shaped())) == len(FEATURE_NAMES)


def test_features_use_no_measured_quantity():
    """The surrogate must predict fitness from structure alone. If a feature
    needed a real evaluation, the surrogate could not save one."""
    genome = designer_shaped()
    assert np.array_equal(features(genome), features(genome.clone()))


def test_surrogate_declines_to_train_on_too_little_data():
    surrogate = Surrogate()
    surrogate.fit([solo()], [0.5])
    assert surrogate.trained is False
    assert surrogate.predict([solo()])[0] == pytest.approx(0.5)


def test_surrogate_learns_a_signal_it_can_actually_see():
    """Agent count is in the feature vector, so a fitness that depends only on
    agent count must be learnable. This tests the plumbing, not the science.

    Mutations are chained rather than all taken from one parent: a single step
    from a fixed genome yields only a few dozen distinct networks, and asking
    for more than exist spins forever.
    """
    rng = random.Random(4)
    genomes, targets, seen = [], [], set()
    frontier = [designer_shaped(), solo()]

    for _ in range(4000):
        if len(genomes) >= 40:
            break
        try:
            mutant, _ = mutate(rng.choice(frontier), rng)
        except Exception:
            continue
        digest = mutant.genome_hash()
        if digest in seen:
            continue
        seen.add(digest)
        genomes.append(mutant)
        targets.append(float(len(mutant.reachable())))
        if len(frontier) < 12:
            frontier.append(mutant)

    assert len(genomes) >= 30, f"only reached {len(genomes)} distinct genomes"

    surrogate = Surrogate(seed=0)
    quality = surrogate.report_quality(genomes, targets)
    assert quality.spearman > 0.5, quality


# --------------------------------------------------------------- fitness shape


def test_cost_objective_is_not_saturated_at_realistic_token_counts():
    """The token penalty must still respond around what candidates actually spend.

    TOKEN_SCALE was once 60,000 while measured candidates were spending
    278,532 to 396,378 tokens on the task set. Every candidate was past the cap,
    min() clipped them all to the same penalty, and the cost objective carried
    no gradient -- the search looked multi-objective while optimising accuracy
    alone. This pins the property rather than the constant.
    """
    from esp.eval.runner import Evaluation
    from esp.evolve.loop import fitness

    def at(tokens: int) -> float:
        return fitness(Evaluation("h", 0.8824, 0.0, tokens, 0.0, 1, 1))

    # Around the measured band, and half and double it.
    cheap, baseline, dear = at(130_000), at(260_160), at(520_000)
    assert cheap > baseline > dear, (cheap, baseline, dear)


def test_accuracy_outranks_cost():
    """A network that answers more must win, however much it spends -- otherwise
    evolution buys cheapness with correctness."""
    from esp.eval.runner import Evaluation
    from esp.evolve.loop import fitness

    accurate_and_dear = fitness(Evaluation("h", 1.0, 0.0, 900_000, 0.0, 6, 3))
    cheap_and_wrong = fitness(Evaluation("h", 0.82, 0.0, 50_000, 0.0, 1, 1))
    assert accurate_and_dear > cheap_and_wrong


def test_quota_failures_are_never_cached_as_a_score():
    """A daily provider cap does not fail every task at once. It starts failing
    them part-way through a candidate, so the all-errored guard misses it and a
    plausible partial score gets cached forever -- telling the search that a
    good network is mediocre. The whole evaluation must be refused instead.
    """
    from esp.eval.runner import TaskResult, _is_quota_failure

    starved = TaskResult("T01", 2, False, 0.0, "",
                         "Error calling model (RESOURCE_EXHAUSTED): 429 ...")
    genuine = TaskResult("T02", 2, False, 3.0, "Bristol", "")
    assert _is_quota_failure(starved)
    assert not _is_quota_failure(genuine)


# ------------------------------------------- a placeholder is not a measurement

def test_rank_quality_below_the_minimum_is_none_not_zero():
    """`spearman=+0.000` was returned without computing anything, and printed
    in the same shape as a real result. This project's whole argument is that
    its numbers were measured; a placeholder that reads like a measurement is
    the one kind of number it cannot ship."""
    from esp.surrogate.predictor import MIN_SAMPLES

    surrogate = Surrogate(seed=0)
    genomes = [designer_shaped(), solo(), flat_pair()]
    quality = surrogate.report_quality(genomes, [0.7868, 0.7842, 0.7816])

    assert quality.samples < MIN_SAMPLES
    assert quality.spearman is None
    assert quality.mae is None
    assert quality.measured is False
    assert "NOT MEASURED" in str(quality)
    assert "+0.000" not in str(quality)


def test_an_untrained_surrogate_admits_it_cannot_rank():
    """It returns the mean of its targets for every genome, so sorting on it is
    a no-op -- and Phase D then pays real budget for what it believes was an
    elite."""
    surrogate = Surrogate(seed=0)
    surrogate.fit([designer_shaped(), solo(), flat_pair()],
                  [0.7868, 0.7842, 0.7816])

    assert surrogate.ranks() is False
    predictions = surrogate.predict([designer_shaped(), solo(), flat_pair()])
    assert len(set(predictions.tolist())) == 1, (
        "predictions differ, so this test no longer pins the degenerate case")
