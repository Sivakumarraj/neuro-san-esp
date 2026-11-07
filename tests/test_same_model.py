"""The same-model test: the five criteria docs/SAME-MODEL.md fixed before the run,
checked as written, and every figure quoted from it recomputed from the logs.

The paid run cannot be repeated in CI, so the committed logs are the evidence and
these tests recompute the study from them, the way tests/test_route.py does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esp import same_model
from esp.eval.accounting import ACCOUNTING

ROOT = Path(__file__).resolve().parent.parent
PLAN = (ROOT / "docs" / "SAME-MODEL.md").read_text(encoding="utf-8")


def _rows(label: str, right: dict[str, list[bool]], tokens: list[float], dollars: list[float],
          model: str = same_model.MODEL, search: int = same_model.SEARCH_RESULTS,
          genome_hash: str | None = None) -> list[dict]:
    """A log as `select_and_judge.py measure` writes it: chunks of four questions,
    every chunk asked twice. `tokens` and `dollars` are per question, one per chunk."""
    tasks = sorted(right)
    agents = {"designer": ["Coordinator", "ContractSpecialist", "DepotSpecialist",
                           "IncidentSpecialist"],
              "2cc4-nano": ["Coordinator", "Researcher"]}[label]
    rows = []
    for repeat in range(2):
        for chunk in range(len(tasks) // 4):
            chosen = tasks[4 * chunk:4 * chunk + 4]
            rows.append({
                "label": label, "genome_hash": genome_hash or same_model.HASHES[label],
                "models": {agent: model for agent in agents},
                "settings": {"search_results": search, "max_steps": 40,
                             "max_execution_seconds": 600},
                "chunk": chunk, "repeat": repeat, "accounting": ACCOUNTING,
                "tokens": round(4 * tokens[chunk]), "reported_cost": 4 * dollars[chunk],
                "cost": 4 * dollars[chunk],
                "results": [{"task_id": t, "correct": right[t][repeat], "infrastructure": False}
                            for t in chosen]})
    return rows


def _write(folder: Path, label: str, rows: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{label}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


QUESTIONS = [f"J{n:03d}" for n in range(1, 41)]


def _study(tmp_path: Path, candidate_right: int, candidate_tokens: float,
           candidate_dollars: float) -> dict:
    """The designer right on 30 of 40 questions both times; the candidate right on
    `candidate_right`, at the given usage a question in every chunk."""
    designer = {t: [i < 30, i < 30] for i, t in enumerate(QUESTIONS)}
    candidate = {t: [i < candidate_right, i < candidate_right] for i, t in enumerate(QUESTIONS)}
    chunks = len(QUESTIONS) // 4
    folder = tmp_path / "judge"
    _write(folder, "designer", _rows("designer", designer, [5000 + 10 * c for c in range(chunks)],
                                     [0.00135] * chunks))
    _write(folder, "2cc4-nano", _rows("2cc4-nano", candidate,
                                      [candidate_tokens + 10 * c for c in range(chunks)],
                                      [candidate_dollars] * chunks))
    return same_model.analyse({"judge": folder})["suites"]["judge"]


def test_a_network_better_on_every_axis_passes_all_five(tmp_path):
    result = _study(tmp_path, candidate_right=38, candidate_tokens=4000, candidate_dollars=0.0011)
    assert result["holds"] == {"accuracy": True, "tokens": True, "dollars": True,
                               "agents": True, "fitness": True}
    assert result["all_hold"]
    assert result["accuracy"]["candidate_better_on"] == QUESTIONS[30:38]


def test_a_network_that_costs_more_fails_on_cost_and_is_not_called_cheaper(tmp_path):
    result = _study(tmp_path, candidate_right=38, candidate_tokens=4000, candidate_dollars=0.0026)
    assert result["holds"]["accuracy"] and result["holds"]["tokens"]
    assert not result["holds"]["dollars"]
    assert not result["all_hold"]


def test_a_gain_too_small_to_test_is_not_called_more_accurate(tmp_path):
    """Two questions in forty is a sign test of 2 against 0: p = 0.5."""
    result = _study(tmp_path, candidate_right=32, candidate_tokens=4000, candidate_dollars=0.0011)
    assert result["accuracy"]["p"] == 0.5
    assert not result["holds"]["accuracy"]


def test_a_run_that_did_not_finish_counts_as_wrong(tmp_path):
    rows = _rows("2cc4-nano", {t: [True, True] for t in QUESTIONS}, [4000] * 10, [0.001] * 10)
    rows[0]["results"][0]["infrastructure"] = True
    _write(tmp_path, "2cc4-nano", rows)
    network = same_model.read(tmp_path / "2cc4-nano.jsonl")
    assert network.runs == 80 and network.right == 79 and network.unfinished == 1


@pytest.mark.parametrize("change", [{"model": "gpt-5.4-mini"}, {"search": 3},
                                    {"genome_hash": "47322479eb9cba1e"}])
def test_a_log_of_another_experiment_is_refused(tmp_path, change):
    """A promoted Researcher, another search width or another genome is not the
    test the page fixed, and summarising it under that name would be a claim it
    never tested."""
    _write(tmp_path, "2cc4-nano", _rows("2cc4-nano", {t: [True, True] for t in QUESTIONS},
                                        [4000] * 10, [0.001] * 10, **change))
    with pytest.raises(ValueError):
        same_model.read(tmp_path / "2cc4-nano.jsonl")


def test_the_networks_and_settings_are_the_ones_the_plan_names():
    for genome_hash in same_model.HASHES.values():
        assert genome_hash in PLAN
    assert same_model.MODEL in PLAN
    assert f"ESP_SEARCH_RESULTS={same_model.SEARCH_RESULTS}" in PLAN
    assert "p < 0.05" in PLAN and same_model.ALPHA == 0.05

