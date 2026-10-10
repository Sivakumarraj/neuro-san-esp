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
