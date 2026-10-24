"""Agreement between two runs: what counts as agreeing, what each policy costs,
and that every figure quoted about it is what the committed logs give.

Like the routing study, this reads logs of paid runs that cannot be repeated in
CI, so the tests recompute it from those logs and compare.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from esp import route, uncertainty
from esp.eval.tasks import normalise

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "uncertainty" / "summary.json"
README = (ROOT / "README.md").read_text(encoding="utf-8")
FINDINGS = (ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")
LOGS = [*(p for pair in uncertainty.PAIRS.values() for p in pair), uncertainty.CONTROL]


@pytest.fixture(scope="module")
def study() -> dict:
    return uncertainty.analyse()


def _repeats(answers: dict[str, list[str]], right: dict[str, list[bool]],
             dollars: float) -> uncertainty.Repeats:
    return uncertainty.Repeats("toy", "0" * 16, answers, right, dollars)


def test_two_replies_agree_when_either_states_the_others_answer():
    assert uncertainty.same_meaning("R. Delacroix", "Delacroix")
    assert uncertainty.same_meaning("16", "The total is 16.")
    assert not uncertainty.same_meaning("116", "136")
    # A number inside an identifier is not the number.
    assert not uncertainty.same_meaning("4", "INC-4429")


def test_agreeing_is_comparing_normalised_answers_on_every_committed_pair():
    """The choice of what counts as the same answer cannot move a figure: on
    every pair of runs committed, it is plain string equality after
    normalising."""
    for path in LOGS:
        network = uncertainty.load(uncertainty.RESULTS / path)
        for task, (first, second) in network.answers.items():
            assert uncertainty.same_meaning(first, second) == (
                normalise(first) == normalise(second)), (path, task)


def test_every_question_was_asked_twice_and_no_run_failed_on_infrastructure():
    for path in LOGS:
        rows = [json.loads(line) for line in (uncertainty.RESULTS / path).read_text(
            encoding="utf-8").splitlines() if line.strip()]
        assert not any(r["infrastructure"] for row in rows for r in row["results"])
        network = uncertainty.load(uncertainty.RESULTS / path)
        assert {len(a) for a in network.answers.values()} == {2}


def test_the_gate_uses_the_routing_studys_two_networks_on_both_question_sets():
    cheap = route.load(route.RUNS / route.CHEAP[1])
    strong = route.load(route.RUNS / route.STRONG[1])
    for cheap_path, strong_path in uncertainty.PAIRS.values():
        assert uncertainty.load(uncertainty.RESULTS / cheap_path).genome_hash == cheap.genome_hash
        assert uncertainty.load(uncertainty.RESULTS / strong_path).genome_hash == strong.genome_hash


def test_dollars_are_the_routing_studys_count():
    for path in LOGS:
        assert uncertainty.load(uncertainty.RESULTS / path).dollars == pytest.approx(
            route.load(uncertainty.RESULTS / path).dollars, rel=1e-12)


def test_two_agreeing_runs_are_both_right_or_both_wrong():
    """Why the answer a gate keeps is exactly as right as either run."""
    for path in LOGS:
        network = uncertainty.load(uncertainty.RESULTS / path)
        for task, agrees in network.agrees().items():
            if agrees:
                assert len(set(network.right[task])) == 1, (path, task)


def test_the_gate_pays_for_both_cheap_runs_and_the_strong_one_on_disagreement():
    cheap = _repeats({"a": ["1", "1"], "b": ["2", "3"], "c": ["4", "4"], "d": ["5", "6"]},
                     {"a": [True, True], "b": [True, False], "c": [False, False],
                      "d": [False, False]}, dollars=1.0)
    strong = _repeats({t: ["x", "x"] for t in "abcd"},
                      {"a": [True, True], "b": [True, True], "c": [True, True],
                       "d": [False, True]}, dollars=3.0)
    questions = {t: "Who manages depot D05?" for t in "abcd"}  # the rule keeps all cheap
    table = uncertainty.policies(cheap, strong, questions)
    gate = table["gate"]
    assert gate.share_to_strong == 0.5
    assert gate.dollars == pytest.approx(2 * 1.0 + 0.5 * 3.0)
    # a and c kept from the cheap runs (1 and 0), b and d from the strong (1 and 0.5).
    assert gate.accuracy == pytest.approx((1 + 1 + 0 + 0.5) / 4)
    assert table["gate_if_free"].dollars == pytest.approx(1.0 + 0.5 * 3.0)
    assert table["rule"].dollars == pytest.approx(1.0)
    flag = table["flag_cheap"]
    assert (flag.coverage, flag.accuracy, flag.dollars) == (0.5, 0.5, 2.0)


def test_a_policy_is_dominated_only_by_one_at_least_as_right_for_no_more():
    P = uncertainty.Policy
    table = {"gate": P(0.90, 3.0, 1.0, 0.2), "strong": P(0.95, 2.5, 1.0, 1.0),
             "cheap": P(0.80, 1.0, 1.0, 0.0), "gate_if_free": P(0.90, 1.5, 1.0, 0.2),
             "flag_cheap": P(0.99, 2.0, 0.8, 0.0)}
    assert uncertainty.dominated_by("gate", table) == ["strong"]
    assert uncertainty.dominated_by("cheap", table) == []
    assert uncertainty.dominated_by("strong", table) == []


def test_beliefs_are_beta_one_one_updated_with_the_share_of_runs_right():
    cheap = _repeats({"a": ["1", "1"], "b": ["2", "3"]},
                     {"a": [True, True], "b": [True, False]}, dollars=1.0)
    strong = _repeats({"a": ["1", "1"], "b": ["2", "2"]},
                      {"a": [True, True], "b": [True, True]}, dollars=2.0)
    b = uncertainty.beliefs(cheap, strong)
    assert b["cheap_if_agree"].mean == pytest.approx((1 + 1) / (2 + 1))
    assert b["cheap_if_disagree"].mean == pytest.approx((1 + 0.5) / (2 + 1))
    assert b["strong_if_disagree"].mean == pytest.approx((1 + 1) / (2 + 1))
    # Sending b's kind up gains 1/6 of a right answer for 2 dollars.
    assert uncertainty.break_even_value(b, "disagree", 2.0) == pytest.approx(12.0)
    assert uncertainty.break_even_value(b, "agree", 2.0) == float("inf")


def test_beliefs_are_fitted_on_judge_and_checked_on_questions_it_never_asked(study):
    from esp.eval.fresh import FRESH
    from esp.eval.suites import JUDGE, SELECT

    judge = {t.question for t in JUDGE}
    assert not judge & {t.question for t in FRESH}
    assert not {t.question for t in SELECT} & {t.question for t in FRESH}
    fitted = study["beliefs"]["fitted_on_judge"]
    assert fitted["cheap_if_agree"]["questions"] + fitted["cheap_if_disagree"]["questions"] \
        == study["sets"]["judge"]["questions"] == 100
    assert study["sets"]["fresh"]["questions"] == 40


def test_every_disagreement_is_on_a_question_the_rule_sends_up(study):
    """The finding the gate's verdict rests on: on these questions agreement
    carries nothing the one-line rule does not already know."""
    for s in study["sets"].values():
        assert s["disagreed_and_rule_sends_up"] == s["disagreed"] > 0


def test_the_committed_summary_is_what_the_logs_say(study):
    assert json.loads(SUMMARY.read_text(encoding="utf-8")) == study


def test_the_docs_quote_the_study_as_computed(study):
    judge, fresh = study["sets"]["judge"], study["sets"]["fresh"]
    jp, fp = judge["policies"], fresh["policies"]
    cal = study["calibration"]
    for text in (README, FINDINGS):
        for quoted in (f"{cal['judge']['cheap']['right_if_agree']:.1%}",
                       f"{cal['judge']['cheap']['right_if_disagree']:.1%}",
                       f"{cal['fresh']['cheap']['right_if_agree']:.1%}",
                       f"{jp['gate']['accuracy']:.1%}", f"${jp['gate']['dollars']:.5f}",
                       f"{fp['gate']['accuracy']:.1%}", f"${fp['gate']['dollars']:.5f}",
                       f"{judge['disagreed']} of {judge['disagreed']}",
                       f"{fresh['disagreed']} of {fresh['disagreed']}",
                       f"{judge['break_even_ratio']:.2f}"):
            assert quoted in text, quoted
    for quoted in (f"{cal['judge']['strong']['agree']} of {cal['judge']['strong']['agree']}",
                   f"{cal['fresh']['strong']['right_if_agree']:.1%}",
                   f"{cal['fresh']['control']['right_if_agree']:.1%}",
                   f"{judge['random_same_share']:.1%}", f"{fresh['random_same_share']:.1%}",
                   f"${jp['gate_if_free']['dollars']:.5f}",
                   f"{judge['probe_budget']:.0%}",
                   f"${study['beliefs']['break_even_value']['disagree']:.4f}",
                   f"{study['beliefs']['held_out']['gate_accuracy']['fitted_on_judge']:.1%}",
                   *judge["agreed_and_wrong"], *fresh["agreed_and_wrong"]):
        assert quoted in FINDINGS, quoted


def test_the_report_script_runs_on_the_committed_logs(capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import uncertainty_report

    assert uncertainty_report.main([]) == 0
    out = capsys.readouterr().out
    for section in ("OBSERVED", "DERIVED", "NOT KNOWN"):
        assert section in out


def test_probe_budget_is_what_a_probe_could_cost_and_still_match_the_rule(study):
    for name, (cheap_path, strong_path) in uncertainty.PAIRS.items():
        s = study["sets"][name]
        c = uncertainty.load(uncertainty.RESULTS / cheap_path).dollars
        strong = uncertainty.load(uncertainty.RESULTS / strong_path).dollars
        probe = s["probe_budget"] * c
        d = s["policies"]["gate"]["share_to_strong"]
        assert c + probe + d * strong == pytest.approx(s["policies"]["rule"]["dollars"],
                                                       abs=0.01 * c)
        assert np.isclose(s["break_even_ratio"], (1 - d) / 2, atol=1e-3)
