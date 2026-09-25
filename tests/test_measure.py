"""Measuring an arbitrary neuro-san network: the question file and the verdict.

The provider call is replaced throughout, so nothing here spends anything. What
is pinned is the part that has to be right regardless of provider: what a
question file may contain, how each reply is classified, that whole answers
survive into the report, and that a run which measured the environment is
refused rather than reported.
"""

from __future__ import annotations

import json

import pytest

from esp import measure as m
from esp.eval import runner

SUITE = "\n".join([
    json.dumps({"id": "A", "question": "Which city is depot D08 in?", "answer": "Eastgate"}),
    json.dumps({"id": "B", "question": "Highest penalty?", "answers": ["C-2139", "C2139"],
                "hops": 3}),
    json.dumps({"id": "C", "question": "How many bays?", "answer": 4}),
])


def replying(replies: dict[str, str], tokens: int = 120):
    """A stand-in for the model call: a fixed reply per question."""
    def ask(hocon, question):
        return replies[question], {"total_tokens": tokens}, 0.5
    return ask


# ------------------------------------------------------- the question file

def test_json_lines_with_answer_or_answers_and_optional_fields():
    tasks = m.parse_suite(SUITE)
    assert [t.task_id for t in tasks] == ["A", "B", "C"]
    assert tasks[1].accepted == ("C-2139", "C2139") and tasks[1].hops == 3
    assert tasks[2].accepted == ("4",), "a numeric answer is compared as text"


def test_a_json_list_is_accepted_too_and_ids_are_filled_in():
    tasks = m.parse_suite(json.dumps([{"question": "q1", "answer": "a"},
                                      {"question": "q2", "answer": "b"}]))
    assert [t.task_id for t in tasks] == ["Q01", "Q02"]


@pytest.mark.parametrize("text,complaint", [
    ("", "empty"),
    ("{not json", "not JSON"),
    (json.dumps({"answer": "x"}), "no question"),
    (json.dumps({"question": "q"}), "no answer"),
    (json.dumps({"id": "A", "question": "q", "answer": "a"}) + "\n"
     + json.dumps({"id": "A", "question": "q2", "answer": "b"}), "repeats the id"),
    (json.dumps({"question": "q", "answer": "a", "hops": "two"}), "hops"),
])
def test_a_bad_question_file_is_refused_before_anything_is_asked(text, complaint):
    with pytest.raises(m.SuiteError, match=complaint):
        m.parse_suite(text)


def test_the_builtin_suite_is_the_benchmark_every_result_used():
    name, tasks = m.load_suite(None)
    assert name == m.BUILTIN and len(tasks) == 17


# ------------------------------------------------------------- the verdict

def test_each_reply_is_classified_and_kept_whole(monkeypatch):
    long_reply = "The depot is in Eastgate. " + "Evidence. " * 300
    monkeypatch.setattr(runner, "_ask", replying({
        "Which city is depot D08 in?": long_reply,
        "Highest penalty?": "It is contract C2139.",
        "How many bays?": "Agent timed out after max_execution_seconds",
    }))
    report = m.measure("any.hocon", m.parse_suite(SUITE), suite="mine", workers=2)

    by_id = {r.task_id: r for r in report.results}
    assert by_id["A"].correct and by_id["A"].answer == long_reply, "answers are not trimmed"
    assert by_id["B"].correct, "any accepted answer counts"
    assert by_id["C"].infrastructure and not by_id["C"].correct
    assert report.accuracy == round(2 / 3, 4)
    assert report.answered_accuracy == 1.0, "an unfinished question is not a wrong one"
    assert report.unfinished == 1
    assert report.tokens == 360


def test_results_come_back_in_question_order_and_progress_is_reported(monkeypatch):
    monkeypatch.setattr(runner, "_ask", replying({
        "Which city is depot D08 in?": "Eastgate", "Highest penalty?": "C-2139",
        "How many bays?": "4"}))
    seen = []
    report = m.measure("any.hocon", m.parse_suite(SUITE), on_result=seen.append)
    assert [r.task_id for r in report.results] == ["A", "B", "C"]
    assert sorted(r.task_id for r in seen) == ["A", "B", "C"]


def test_a_run_that_called_no_model_is_refused_not_reported(monkeypatch):
    monkeypatch.setattr(runner, "_ask", replying(
        {"Which city is depot D08 in?": "x", "Highest penalty?": "y",
         "How many bays?": "z"}, tokens=0))
    with pytest.raises(OSError, match="zero tokens"):
        m.measure("any.hocon", m.parse_suite(SUITE))


def test_a_report_serialises_with_the_question_and_expected_answer(monkeypatch):
    monkeypatch.setattr(runner, "_ask", replying({
        "Which city is depot D08 in?": "Eastgate", "Highest penalty?": "no idea",
        "How many bays?": "4"}))
    blob = m.measure("net.hocon", m.parse_suite(SUITE), suite="mine").as_dict()
    first = blob["results"][0]
    assert first["question"] == "Which city is depot D08 in?"
    assert first["expected"] == "Eastgate"
    assert blob["questions_asked"] == 3 and blob["suite"] == "mine"
    json.dumps(blob)                                  # must be serialisable


# ------------------------------------------------------------------ the CLI

def test_the_command_line_measures_and_writes_json(monkeypatch, tmp_path, capsys):
    questions = tmp_path / "q.jsonl"
    questions.write_text(SUITE, encoding="utf-8")
    monkeypatch.setattr(runner, "_ask", replying({
        "Which city is depot D08 in?": "Eastgate", "Highest penalty?": "C-2139",
        "How many bays?": "4"}))
    out = tmp_path / "report.json"
    assert m.main(["net.hocon", "--tasks", str(questions), "--json", str(out)]) == 0
    assert "accuracy 100.0%" in capsys.readouterr().out
    assert json.loads(out.read_text())["accuracy"] == 1.0


def test_the_command_line_rejects_a_bad_file_without_asking(monkeypatch, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{nope", encoding="utf-8")
    called = []
    monkeypatch.setattr(runner, "_ask", lambda *a: called.append(1))
    assert m.main(["net.hocon", "--tasks", str(bad)]) == 2
    assert not called
