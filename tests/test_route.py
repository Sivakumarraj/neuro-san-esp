"""The router: what it may read, how its threshold is chosen, and that every
figure quoted about it is what the committed logs give.

The study runs on logs of paid runs that cannot be repeated in CI, so the tests
recompute it from those logs and compare, the way tests/test_select_and_judge.py
does for the head-to-head itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from esp import route

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "routing" / "summary.json"
README = (ROOT / "README.md").read_text(encoding="utf-8")
FINDINGS = (ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def study() -> dict:
    return route.analyse()


def test_the_router_reads_the_question_text_and_nothing_else():
    """A router has the question when it arrives and nothing more. The feature
    function takes a string, so it cannot reach `Task.hops` or the answer."""
    a = route.request_features("How many incidents affected contracts serviced by "
                               "depot D05? Answer with the number only.")
    b = route.request_features("Who is the depot manager of Meridian Logistics depot D05?")
    assert a.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert b.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_named_entities_are_counted_whatever_kind_they_are():
    two_incidents = route.request_features(
        "Incidents INC-4431 and INC-4458 each affected a contract serviced by a depot.")
    two_clients = route.request_features(
        "Clients Anvil Brewing and Quire Paper each hold a contract serviced by a depot.")
    one_client = route.request_features(
        "In what year did the depot that services the contract with client Calder Motors open?")
    assert two_incidents[0] == two_clients[0] == 2
    assert one_client[0] == 1


def test_the_strong_network_is_the_same_genome_on_select_and_judge():
    """The study trains on one label and judges on another, which is only
    honest if they are the same network."""
    cheap_select, cheap_judge = (route.load(route.RUNS / p) for p in route.CHEAP)
    strong_select, strong_judge = (route.load(route.RUNS / p) for p in route.STRONG)
    assert cheap_select.genome_hash == cheap_judge.genome_hash
    assert strong_select.genome_hash == strong_judge.genome_hash
    assert cheap_select.genome_hash != strong_select.genome_hash


def test_the_split_cost_adds_back_up_to_what_was_paid():
    for path in route.CHEAP + route.STRONG:
        outcomes = route.load(route.RUNS / path)
        assert np.mean(list(outcomes.dollars_by_time.values())) == pytest.approx(
            outcomes.dollars, rel=1e-9)


def test_the_threshold_is_the_cheapest_within_one_question_of_the_strong_network():
    gains = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    cheap = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    strong = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # The two the cheap network misses carry the largest gains, so sending
    # exactly those two loses nothing.
    assert route.choose_threshold(gains, cheap, strong, slack=0) == 0.4
    # Allowed one miss, the router keeps the second on the cheap network.
    assert route.choose_threshold(gains, cheap, strong, slack=1) == 0.5


def test_a_threshold_above_every_gain_sends_nothing_and_below_every_gain_sends_all():
    cheap_judge = route.load(route.RUNS / route.CHEAP[1])
    strong_judge = route.load(route.RUNS / route.STRONG[1])
    ids = sorted(cheap_judge.right)
    none = route.evaluate(np.zeros(len(ids), bool), ids, cheap_judge, strong_judge)
    every = route.evaluate(np.ones(len(ids), bool), ids, cheap_judge, strong_judge)
    assert (none.accuracy, none.share) == (pytest.approx(0.84), 0.0)
    assert (every.accuracy, every.share) == (pytest.approx(0.945), 1.0)


def test_random_routing_scores_at_the_mixture_of_the_two_networks():
    to_strong = np.array([True] * 4 + [False] * 6)
    cheap = np.array([1.0, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    strong = np.ones(10)
    mean, p = route.permutation_p(to_strong, cheap, strong, rounds=4000, seed=1)
    assert mean == pytest.approx(0.5 + 0.4 * 0.5, abs=0.01)
    assert 0.0 <= p <= 1.0


def test_the_committed_summary_is_what_the_logs_say(study):
    assert json.loads(SUMMARY.read_text(encoding="utf-8")) == study


def test_the_router_was_judged_on_questions_it_was_not_fitted_or_tuned_on(study):
    from esp.eval.suites import JUDGE, SELECT

    assert study["select"]["questions"] == 60
    assert study["judge"]["questions"] == 100
    assert not {t.question for t in SELECT} & {t.question for t in JUDGE}


def test_the_docs_quote_the_study_as_computed(study):
    j = study["judge"]
    router, rule, strong, cheap = j["router"], j["rule"], j["strong"], j["cheap"]
    for text in (README, FINDINGS):
        assert f"{router['accuracy']:.1%}" in text
        assert f"${router['dollars']:.5f}" in text
        assert f"{j['random_same_share']:.1%}" in text
        assert f"p = {j['p_against_random']}" in text
        assert f"{1 - router['dollars'] / strong['dollars']:.0%}" in text
        assert f"${rule['dollars']:.5f}" in text
    for quoted in (f"${cheap['dollars']:.5f}", f"${strong['dollars']:.5f}",
                   f"{study['select']['leave_one_out_accuracy']:.1%}",
                   f"{j['oracle']:.1%}", f"${router['dollars_by_time']:.5f}"):
        assert quoted in FINDINGS, quoted


def test_the_report_script_runs_on_the_committed_logs(capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import route_report

    assert route_report.main([]) == 0
    assert "Judged once on judge" in capsys.readouterr().out
