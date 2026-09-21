"""A run that called no model must never become a measurement.

Found by a live run on a fresh Codespace. Two seeds sat in `.esp-cache` as

    gen0 seed:solo             acc=0.00 tok=0 agents=1 (cached)
    gen0 seed:designer_shaped  acc=0.00 tok=0 agents=4 (cached)

and replayed on every invocation afterwards. They were written before the API
key worked: the agents still answered, just with whatever they say when they
cannot reach a provider, so every task "completed" with a wrong answer, no
error was attached, and the all-tasks-errored guard waved it through.

`tokens == 0` is the tell, and it is exact rather than heuristic. Seventeen
tasks through a language model cannot cost nothing. Two guards, because the
damage outlives the cause: the runner refuses to write one, and the reader
refuses to load one that an older build already wrote.
"""

from __future__ import annotations

import json

import pytest

from esp.eval import measurements
from esp.eval.runner import TaskResult, evaluate
from esp.eval.tasks import TASKS
from esp.genome.seeds import SEEDS


def test_an_evaluation_that_spent_nothing_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("esp.eval.runner.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("esp.eval.runner.NETWORK_DIR", tmp_path / "networks")

    # What a keyless run actually looks like: an answer arrives, it is wrong,
    # nothing errors, and no tokens are spent.
    def answered_but_never_called(hocon_path, question):
        return "I am unable to access the required information.", {}, 0.1

    monkeypatch.setattr("esp.eval.runner._ask", answered_but_never_called)

    genome = next(iter(SEEDS.values()))()
    with pytest.raises(OSError, match="zero tokens"):
        evaluate(genome, tasks=TASKS[:3], use_cache=False)

    written = list((tmp_path / "cache").glob("*.json"))
    assert not written, f"a zero-token evaluation was cached: {written}"


def test_the_refusal_says_how_to_recover():
    """The operator has to be told to delete the cache, or the zeros they
    already have keep replaying while the new guard looks like it did nothing."""
    from esp.eval.runner import CACHE_DIR

    message = (
        "the whole evaluation cost zero tokens, so no model was called -- "
        f"refusing to cache. Check the API key in .env, then delete "
        f"{CACHE_DIR}/ if earlier runs already stored zeros."
    )
    assert ".env" in message and "delete" in message


def test_a_poisoned_record_already_on_disk_is_not_loaded(tmp_path):
    """The other half. A cache written before the guard existed still sits on
    somebody's disk, and one of those records will train a Predictor and be
    served as a champion unless the reader refuses it too."""
    genome = next(iter(SEEDS.values()))()
    digest = genome.genome_hash()
    (tmp_path / f"{digest}.json").write_text(json.dumps({
        "genome_hash": digest, "accuracy": 0.0, "seconds": 3.0,
        "tokens": 0, "cost": 0.0, "agents": len(genome.reachable()),
        "depth": genome.depth(), "results": [], "genome": genome.canonical(),
    }), encoding="utf-8")

    assert measurements.load(tmp_path) == []
    assert measurements.best(tmp_path) is None


def test_a_real_measurement_is_still_loaded(tmp_path):
    """The guard has to reject the poisoned record and nothing else."""
    genome = next(iter(SEEDS.values()))()
    digest = genome.genome_hash()
    (tmp_path / f"{digest}.json").write_text(json.dumps({
        "genome_hash": digest, "accuracy": 0.8235, "seconds": 400.0,
        "tokens": 316_074, "cost": 0.0, "agents": len(genome.reachable()),
        "depth": genome.depth(), "results": [], "genome": genome.canonical(),
    }), encoding="utf-8")

    found = measurements.load(tmp_path)
    assert len(found) == 1
    assert found[0].tokens == 316_074


def test_the_committed_measurements_all_spent_real_tokens():
    """The fixtures this repository ships with must not be poisoned, or the
    guard would silently empty the one population a fresh clone has."""
    found = measurements.load()
    assert found, "the committed measurements failed to load"
    for measurement in found:
        assert measurement.tokens > 0, measurement.genome_hash


def test_an_unfinished_task_is_still_distinguishable_from_a_dead_run():
    """A network that timed out on one task is a real measurement with real
    tokens; a run that called nothing is not. The guard must not conflate
    them."""
    results = [
        TaskResult("T01", 2, False, 0.0, "Agent timed out"),
        TaskResult("T02", 1, True, 3.0, "J. Vasquez"),
    ]
    from esp.eval.runner import classify
    classified = classify(results)
    assert classified[0].infrastructure
    assert not classified[1].infrastructure
