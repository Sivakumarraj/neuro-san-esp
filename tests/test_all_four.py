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


SUMMARY = ROOT / "results" / "all-four" / "summary.json"
PUBLISHED = ("\n".join((ROOT / name).read_text(encoding="utf-8")
                       for name in ("README.md", "docs/RESULTS.md"))
             .replace("\N{MINUS SIGN}", "-").replace("**", ""))


def test_the_committed_summary_is_what_the_logs_say():
    assert all_four.analyse() == json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_the_winner_is_the_one_the_rule_picks_and_the_pages_name():
    study = json.loads(SUMMARY.read_text(encoding="utf-8"))
    select = study["select"]
    assert select["complete"] and select["winner"] == "c3"
    for label, network in select["networks"].items():
        assert f"{network['accuracy']:.1%}" in PUBLISHED, label
        assert f"{network['tokens_per_question']:,}" in PUBLISHED, label
        assert f"${network['dollars_per_question']:.6f}" in PUBLISHED, label


def test_every_held_out_figure_quoted_is_the_one_the_logs_give():
    study = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert set(study["sets"]) == {"judge", "fresh"} and not study["incomplete"]
    for result in study["sets"].values():
        for network in result["networks"].values():
            runs = f"{network['correct']}/{network['question_runs']}"
            for quoted in (f"{network['accuracy']:.1%} ({runs})",
                           f"{network['tokens_per_question']:,}",
                           f"${network['dollars_per_question']:.5f}",
                           f"{network['token_fitness']:+.4f}",
                           f"{network['dollar_fitness']:+.4f}"):
                assert quoted in PUBLISHED, quoted
        accuracy = result["accuracy"]
        low, high = accuracy["ci95_points"]
        for quoted in (f"p = {accuracy['p']}", f"{low:+.1f} to {high:+.1f}",
                       f"{len(accuracy['candidate_better_on'])} questions against "
                       f"{len(accuracy['baseline_better_on'])}"):
            assert quoted in PUBLISHED, quoted
        for key in ("tokens", "dollars"):
            assert f"p = {result[key]['p']:.2g}" in PUBLISHED, (key, result[key]["p"])
            assert f"{result[key]['change']:.1%}" in PUBLISHED, (key, result[key]["change"])


def _flat(text: str) -> str:
    """One line per page, so a check does not depend on where a paragraph wraps."""
    return " ".join(text.replace("`", "").split())


def test_the_missing_runs_cannot_change_the_verdict_and_the_pages_say_so():
    judge = json.loads(SUMMARY.read_text(encoding="utf-8"))["sets"]["judge"]
    assert judge["all_hold"] and judge["holds_whatever_is_missing"]
    assert judge["missing_runs"] == {"designer": 12}
    worst = judge["worst_case"]
    flat = _flat(PUBLISHED)
    assert f"reaches {worst['networks']['designer']['accuracy']:.1%}" in flat
    assert f"accuracy p = {worst['accuracy']['p']}" in flat
    assert (f"tokens p = {worst['tokens']['p']:.2g}, "
            f"dollars p = {worst['dollars']['p']:.2g}") in flat


def test_the_wrong_answers_quoted_are_the_ones_the_logs_hold():
    kinds = json.loads(SUMMARY.read_text(encoding="utf-8"))["wrong_answers_on_judge"]
    nano, c3, base = kinds["2cc4 on nano, no calculator"], kinds["c3"], kinds["designer"]
    flat = _flat(PUBLISHED)
    for quoted in (f"{sum(nano.values())} wrong judge runs that morning, {nano['near miss']} "
                   f"were near misses and {nano['wrong count']} wrong counts",
                   f"of c3's {sum(c3.values())}, {c3['near miss']} and {c3['wrong count']}",
                   f"{sum(base.values())} that afternoon, {base['near miss']} and "
                   f"{base['wrong count']}",
                   f"fell from {nano['near miss']} runs on 2cc4 on nano to {c3['near miss']}"):
        assert quoted in flat, quoted


@pytest.mark.parametrize("question, expected, reply, kind", [
    ("Adding up every incident on contracts serviced by depot D06, how many hours late is that "
     "in total? Answer with the number only.", "126", "116", "near miss"),
    ("How many incidents affected contracts serviced by depot D05?", "5", "3", "wrong count"),
    ("What is the combined annual value of every contract serviced by depot D13?", "2924000", "2",
     "other"),
    ("Who is the depot manager of depot D08?", "J. Vasquez", "R. Delacroix", "other"),
])
def test_a_wrong_reply_is_sorted_by_what_could_fix_it(question, expected, reply, kind):
    assert all_four.kind_of_miss(question, expected, reply) == kind
