"""Choosing the Researcher's model per question: the procedure docs/MODEL-ROUTER.md
fixed, checked as written, and every figure quoted from it recomputed from the logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esp import model_router
from esp.eval.accounting import ACCOUNTING

ROOT = Path(__file__).resolve().parent.parent
PLAN = (ROOT / "docs" / "MODEL-ROUTER.md").read_text(encoding="utf-8")


def test_the_threshold_is_the_most_accurate_within_budget():
    gains = np.array([0.5, 0.4, 0.1])
    cheap, strong = np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0])
    # One question up costs 1.33 a question, two cost 1.67: only one fits under 1.5.
    threshold, accuracy, cost, share = model_router.choose(gains, cheap, strong, 1.0, 2.0, 1.5)
    assert threshold == 0.5 and accuracy == pytest.approx(2 / 3)
    assert cost == pytest.approx(4 / 3) and share == pytest.approx(1 / 3)


def test_a_budget_the_strong_action_never_fits_sends_nothing_up():
    threshold, accuracy, _, share = model_router.choose(
        np.array([0.5, 0.4]), np.array([0.0, 1.0]), np.array([1.0, 1.0]), 1.0, 2.0, 1.0)
    assert np.isinf(threshold) and share == 0 and accuracy == 0.5


def test_a_tie_goes_to_sending_fewer_up():
    """Sending the second question up gains nothing, so it is not paid for."""
    threshold, _, _, share = model_router.choose(
        np.array([0.5, 0.4]), np.array([0.0, 1.0]), np.array([1.0, 1.0]), 1.0, 2.0, 2.0)
    assert threshold == 0.5 and share == 0.5


def test_a_cheap_action_dearer_than_the_budget_is_an_error():
    with pytest.raises(ValueError):
        model_router.choose(np.array([0.1]), np.array([1.0]), np.array([1.0]), 3.0, 4.0, 2.0)


def _log(folder: Path, action: str, right: dict[str, bool], tokens: float, dollars: float,
         genome_hash: str | None = None) -> Path:
    tasks = sorted(right)
    rows = []
    for repeat in range(2):
        for chunk in range(len(tasks) // 4):
            chosen = tasks[4 * chunk:4 * chunk + 4]
            rows.append({"label": action, "chunk": chunk, "repeat": repeat,
                         "genome_hash": genome_hash or model_router.HASHES[action],
                         "settings": {"search_results": 10}, "accounting": ACCOUNTING,
                         "tokens": 4 * (tokens + chunk), "reported_cost": 4 * dollars,
                         "cost": 4 * dollars,
                         "results": [{"task_id": t, "correct": right[t], "infrastructure": False,
                                      "seconds": 10.0} for t in chosen]})
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{action}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class _Sends:
    """A router stub: sends up exactly the questions named."""

    def __init__(self, up: set[str], text: dict[str, str]):
        self.up = {text[t] for t in up}

    def to_strong(self, questions: list[str]) -> np.ndarray:
        return np.array([q in self.up for q in questions])


def test_routing_the_questions_the_cheap_action_misses_passes_every_mark(tmp_path):
    text = model_router._questions("judge")
    ids = sorted(text)
    missed = set(ids[::5])          # wrong on 20 of 100, spread over the chunks
    baseline_wrong = set(ids[2::6][:16])
    paths = {"cheap": _log(tmp_path, "cheap", {t: t not in missed for t in ids}, 4000, 0.0010),
             "strong": _log(tmp_path, "strong", {t: True for t in ids}, 3000, 0.0020),
             "baseline": _log(tmp_path, "baseline", {t: t not in baseline_wrong for t in ids},
                              5000, 0.0014)}
    result = model_router.judge(_Sends(missed, text), "judge", paths)
    assert result["sent_up"] == 20
    assert result["accuracy"]["routed"] == 1.0 and result["accuracy"]["baseline"] == 0.84
    assert result["holds"] == {"accuracy": True, "tokens": True, "dollars": True,
                               "agents": True, "fitness": True}


def test_routing_nothing_is_the_cheap_action(tmp_path):
    text = model_router._questions("judge")
    ids = sorted(text)
    missed = set(ids[::5])
    paths = {"cheap": _log(tmp_path, "cheap", {t: t not in missed for t in ids}, 4000, 0.0010),
             "strong": _log(tmp_path, "strong", {t: True for t in ids}, 3000, 0.0020),
             "baseline": _log(tmp_path, "baseline", {t: t not in set(ids[:16]) for t in ids},
                              5000, 0.00105)}
    result = model_router.judge(_Sends(set(), text), "judge", paths)
    assert result["sent_up"] == 0 and result["accuracy"]["routed"] == result["accuracy"]["cheap"]
    assert not result["holds"]["accuracy"] and not result["all_hold"]
    # Knowing the answers, $0.00005 a question of headroom buys five of the twenty up.
    bound = result["oracle_within_budget"]
    assert bound["sent_up"] == 5 and bound["dollars"] <= 0.00105


def test_a_log_of_another_network_is_refused(tmp_path):
    path = _log(tmp_path, "strong", {"J001": True, "J002": True, "J003": True, "J004": True},
                3000, 0.002, genome_hash="2cc4231b7d3fb81d")
    with pytest.raises(ValueError):
        model_router.read(path, "strong")


def test_the_networks_the_plan_names_are_the_ones_read():
    for genome_hash in model_router.HASHES.values():
        assert genome_hash in PLAN
    assert "05:22 UTC on 7 November 2026" in PLAN
