"""The all-four test: the choice docs/ALL-FOUR.md fixed, checked as written, and
every figure quoted from it recomputed from the committed logs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esp import all_four
from esp.eval.accounting import ACCOUNTING

ROOT = Path(__file__).resolve().parent.parent
PLAN = (ROOT / "docs" / "ALL-FOUR.md").read_text(encoding="utf-8")
AGENTS = {"designer": {"Coordinator": "gpt-5.4-nano", "ContractSpecialist": "gpt-5.4-nano",
                       "DepotSpecialist": "gpt-5.4-nano", "IncidentSpecialist": "gpt-5.4-nano"},
          "c1": {"Coordinator": "gpt-5.4-nano", "Researcher": "gpt-5.4-nano"},
          "c2": {"Coordinator": "gpt-5.4-nano", "Researcher": "gpt-4o-mini"},
          "c3": {"Coordinator": "gpt-5.4-nano", "Researcher": "gpt-4o-mini"}}
SELECT = [f"S{n:02d}" for n in range(1, 61)]


def _log(folder: Path, label: str, right: int, tokens: float, dollars: float,
         questions: list[str] = SELECT, repeats: int = 1, genome_hash: str | None = None) -> None:
    rows = []
    for repeat in range(repeats):
        for chunk in range(len(questions) // 4):
            chosen = questions[4 * chunk:4 * chunk + 4]
            rows.append({"label": label, "genome_hash": genome_hash or all_four.HASHES[label],
                         "models": AGENTS[label], "chunk": chunk, "repeat": repeat,
                         "settings": {"search_results": 10}, "accounting": ACCOUNTING,
                         "tokens": 4 * tokens, "reported_cost": 4 * dollars, "cost": 4 * dollars,
                         "results": [{"task_id": t, "correct": questions.index(t) < right,
                                      "infrastructure": False} for t in chosen]})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                           encoding="utf-8")


def test_the_most_accurate_eligible_candidate_wins(tmp_path):
    _log(tmp_path, "designer", 48, 5000, 0.00137)
    _log(tmp_path, "c1", 54, 4800, 0.00125)          # eligible
    _log(tmp_path, "c2", 56, 3500, 0.00080)          # eligible, more accurate
    _log(tmp_path, "c3", 58, 5200, 0.00090)          # more accurate, but more tokens
    result = all_four.screen(tmp_path)
    assert result["eligible"] == ["c1", "c2"] and result["winner"] == "c2"


def test_a_tie_on_accuracy_goes_to_the_cheaper(tmp_path):
    _log(tmp_path, "designer", 48, 5000, 0.00137)
    _log(tmp_path, "c1", 54, 4800, 0.00125)
    _log(tmp_path, "c2", 54, 4900, 0.00100)
    _log(tmp_path, "c3", 40, 3000, 0.00050)          # cheaper, not more accurate
    result = all_four.screen(tmp_path)
    assert result["eligible"] == ["c1", "c2"] and result["winner"] == "c2"


def test_no_eligible_candidate_stops_the_test(tmp_path):
    _log(tmp_path, "designer", 48, 5000, 0.00137)
    for label in ("c1", "c2", "c3"):
        _log(tmp_path, label, 48, 4000, 0.0010)      # level on accuracy is not better
    study = all_four.analyse(tmp_path, {"judge": tmp_path / "judge"})
    assert study["select"]["winner"] is None and study["sets"] == {}


def test_an_unfinished_screen_is_not_judged(tmp_path):
    _log(tmp_path, "designer", 48, 5000, 0.00137)
    _log(tmp_path, "c1", 30, 4800, 0.00125, questions=SELECT[:40])
    assert all_four.screen(tmp_path)["complete"] is False


def test_a_log_of_another_network_is_refused(tmp_path):
    _log(tmp_path, "c1", 54, 4800, 0.00125, genome_hash="2cc4231b7d3fb81d")
    with pytest.raises(ValueError):
        all_four._read(tmp_path / "c1.jsonl")


def test_the_networks_the_plan_names_are_the_ones_read():
    for genome_hash in all_four.HASHES.values():
        assert genome_hash in PLAN
