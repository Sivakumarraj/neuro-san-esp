"""Every stored token and dollar figure written before the accounting fix is
twice what the run used. These tests hold the evidence for that, and hold every
reader to one set of units."""

from __future__ import annotations

import json
from pathlib import Path

from esp.eval import accounting as usage

ROOT = Path(__file__).resolve().parent.parent


def _stored_token_counts() -> list[int]:
    counts = []
    for path in sorted((ROOT / "tests" / "fixtures" / "cache").glob("*.json")):
        counts.append(json.loads(path.read_text(encoding="utf-8"))["tokens"])
    for path in sorted((ROOT / "results" / "headtohead").rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts.append(json.loads(line)["tokens"])
    return counts


def test_every_committed_token_count_is_even():
    """Doubling forces an even count. Hundreds of real runs all landing on an
    even number by chance has probability about 2^-n; halving is exact."""
    counts = _stored_token_counts()
    assert len(counts) > 600
    assert all(c % 2 == 0 for c in counts)


def test_a_record_from_before_the_fix_is_halved_and_marked():
    old = {"tokens": 2200, "cost": 0.00065, "reported_cost": 0.0006, "listed_cost": 0.00065,
           "accuracy": 1.0}
    new = usage.normalised(old)
    assert (new["tokens"], new["cost"], new["accuracy"]) == (1100, 0.000325, 1.0)
    assert new["accounting"] == usage.ACCOUNTING
    assert old["tokens"] == 2200          # the stored record is never changed


def test_a_record_counted_once_is_left_alone():
    once = {"tokens": 1100, "cost": 0.000325, "accounting": usage.ACCOUNTING}
    assert usage.normalised(once) == once
    assert usage.tokens_in(once) == 1100


def test_a_history_without_a_version_is_halved_and_its_fitness_kept():
    history = {"records": [{"tokens": 385_280, "fitness": 0.7761}]}
    record = usage.history_in(history)["records"][0]
    assert record == {"tokens": 192_640, "fitness": 0.7761}


# ------------------------------------------------------------- model calls

def test_model_calls_are_counted_once_and_carried_through_a_run(monkeypatch):
    """neuro-san reports successful_requests at the top of the front man's
    structure and again per model, the same shape that once doubled tokens.
    Each question here makes 3 calls, so 2 questions make 6, not 12."""
    from esp.eval import runner
    from esp.eval.tasks import TASKS

    accounting = {"total_tokens": 900, "successful_requests": 3,
                  "models": {"openai": {"gpt-5.4-nano": {"total_tokens": 600,
                                                         "successful_requests": 2},
                                        "gpt-5.4-mini": {"total_tokens": 300,
                                                         "successful_requests": 1}}}}
    assert runner._requests(accounting) == 3
    monkeypatch.setattr(runner, "_ask", lambda *a: ("x", accounting, 0.01))
    run = runner.run_suite("unused.hocon", TASKS[:2], workers=1)
    assert (run.tokens, run.requests) == (1800, 6)


def test_a_record_from_before_calls_were_counted_reads_as_zero():
    from esp.eval.runner import evaluation_from_cache

    raw = json.loads(next((ROOT / "tests" / "fixtures" / "cache").glob("*.json"))
                     .read_text(encoding="utf-8"))
    assert "requests" not in raw
    assert evaluation_from_cache(raw).requests == 0
