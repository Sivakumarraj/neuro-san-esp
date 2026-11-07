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
from esp.eval import accounting
from esp.eval.accounting import ACCOUNTING

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "structure"
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



SUMMARY = ROOT / "results" / "structure" / "summary.json"
# Both pages quote the study, with typographic minus signs and bold for the better figure.
PUBLISHED = ("\n".join((ROOT / name).read_text(encoding="utf-8")
                       for name in ("README.md", "docs/RESULTS.md"))
             .replace("\N{MINUS SIGN}", "-").replace("**", ""))


def test_the_committed_summary_is_what_the_logs_say():
    assert same_model.analyse() == json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_the_test_failed_on_accuracy_and_the_pages_say_so():
    """Cheaper and smaller, not more accurate: the README may not call 2cc4 better
    than the designer's shape on every axis while the logs say this."""
    judge = json.loads(SUMMARY.read_text(encoding="utf-8"))["suites"]["judge"]
    assert judge["holds"] == {"accuracy": False, "tokens": True, "dollars": True,
                              "agents": True, "fitness": False}
    assert not judge["all_hold"]
    assert "so the test fails" in PUBLISHED and "It does not pass." in PUBLISHED


def test_every_figure_quoted_is_the_one_the_logs_give():
    study = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for result in study["suites"].values():
        for network in result["networks"].values():
            runs = f"{network['correct']}/{network['question_runs']}"
            for quoted in (f"{network['accuracy']:.1%} ({runs})",
                           f"{network['tokens_per_question']:,}",
                           f"${network['dollars_per_question']:.5f}",
                           f"${network['dollars_per_correct']:.5f}",
                           f"{network['token_fitness']:+.4f}",
                           f"{network['dollar_fitness']:+.4f}"):
                assert quoted in PUBLISHED, quoted
        accuracy = result["accuracy"]
        low, high = accuracy["ci95_points"]
        for quoted in (f"p = {accuracy['p']}", f"{accuracy['points']:+.1f} points",
                       f"{low:+.1f} to {high:+.1f}",
                       f"{len(accuracy['candidate_better_on'])} questions against "
                       f"{len(accuracy['baseline_better_on'])}"):
            assert quoted in PUBLISHED, quoted
        for key in ("tokens", "dollars"):
            test = result[key]
            assert f"p = {test['p']:.2g}" in PUBLISHED, (key, test["p"])
            assert f"{test['change']:.1%}" in PUBLISHED, (key, test["change"])


def test_the_headline_network_rerun_beside_the_test_is_quoted_as_measured():
    rows = [accounting.normalised(json.loads(line)) for line in
            (RESULTS_DIR / "fresh-nano-2026-11" / "evolved.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    runs = [result for row in rows for result in row["results"]]
    right = sum(result["correct"] for result in runs)
    tokens = round(sum(row["tokens"] for row in rows) / len(runs))
    assert {row["genome_hash"] for row in rows} == {"47322479eb9cba1e"}
    assert f"scored {right}/{len(runs)} at {tokens:,} tokens a question" in PUBLISHED
