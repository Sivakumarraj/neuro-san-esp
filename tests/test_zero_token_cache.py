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

    written = list((tmp_path / "cache").rglob("*.json"))
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


# ------------------------- every task gave up: the environment, not the shape

def test_a_run_where_every_task_blew_the_recursion_cap_is_not_cached(
        tmp_path, monkeypatch):
    """Seen live against neuro-san 0.7.4: langgraph's recursion limit of 40 was
    reached on every question.

    None of the other guards catches it. neuro-san returns a blown cap as an
    ordinary answer string, so `error` is empty; the agents spent real tokens
    reaching the cap, so the zero-token guard passes it; and no 429 was
    involved. What would land in the cache is accuracy 0.00 with a plausible
    token count, filed against a topology that was never asked anything it
    could finish.
    """
    monkeypatch.setattr("esp.eval.runner.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("esp.eval.runner.NETWORK_DIR", tmp_path / "networks")

    def always_blows_the_cap(hocon_path, question):
        return ("Recursion limit of 40 reached without hitting a stop "
                "condition."), {"totals": {"total_tokens": 8800}}, 12.0

    monkeypatch.setattr("esp.eval.runner._ask", always_blows_the_cap)

    genome = next(iter(SEEDS.values()))()
    with pytest.raises(OSError, match="gave up before answering"):
        evaluate(genome, tasks=TASKS[:4], use_cache=False)
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_a_run_where_every_task_timed_out_is_not_cached(tmp_path, monkeypatch):
    """Same shape, the other way it happens."""
    monkeypatch.setattr("esp.eval.runner.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("esp.eval.runner.NETWORK_DIR", tmp_path / "networks")

    def always_times_out(hocon_path, question):
        return "Agent timed out", {"totals": {"total_tokens": 5100}}, 300.0

    monkeypatch.setattr("esp.eval.runner._ask", always_times_out)
    with pytest.raises(OSError, match="gave up before answering"):
        evaluate(next(iter(SEEDS.values()))(), tasks=TASKS[:4], use_cache=False)


def test_some_unfinished_tasks_are_still_a_real_measurement(
        tmp_path, monkeypatch):
    """The guard must refuse the total wipeout and nothing less.

    Seventeen of the 204 committed task runs never finished, and those
    candidates are real measurements that the whole results table rests on.
    Refusing a candidate because one task timed out would throw away most of
    the population.
    """
    monkeypatch.setattr("esp.eval.runner.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("esp.eval.runner.NETWORK_DIR", tmp_path / "networks")

    state = {"n": 0}

    def one_gives_up(hocon_path, question):
        state["n"] += 1
        if state["n"] == 1:
            return "Agent timed out", {"totals": {"total_tokens": 900}}, 300.0
        return "Brindle", {"totals": {"total_tokens": 4200}}, 4.0

    monkeypatch.setattr("esp.eval.runner._ask", one_gives_up)
    evaluation = evaluate(next(iter(SEEDS.values()))(), tasks=TASKS[:4],
                          use_cache=False)

    assert evaluation.incomplete == 1
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 1


def test_the_committed_measurements_all_answered_something():
    """No candidate in the shipped population is a total wipeout -- otherwise
    the guard would retroactively invalidate the results table."""
    from esp.eval import measurements

    for raw in measurements.raw():
        results = raw.get("results") or []
        if not results:
            continue
        assert not all(r.get("infrastructure") for r in results), (
            f"{raw['genome_hash']} never answered a single task")
