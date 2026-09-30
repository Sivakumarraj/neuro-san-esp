"""The head-to-head: copies pruned, models pinned, the cap enforced, resumable."""

from __future__ import annotations

import pytest

from esp.eval import failover, measurements
from esp.eval.judge_plus import JUDGE_200
from esp.evolve import headtohead as h2h
from esp.genome.prune import copies, without_copies
from esp.genome.seeds import designer_shaped


def test_a_copy_removed_gives_back_the_network_it_was_copied_from():
    # Three committed networks are the designer's plus a copy of one
    # specialist. Removing the copy must give the designer's genome exactly.
    designer = designer_shaped().genome_hash()
    back = [m for m in measurements.load()
            if copies(m.genome) and without_copies(m.genome).genome_hash() == designer]
    assert len(back) >= 2


def test_the_champion_keeps_its_promoted_router_and_loses_only_the_copy():
    best = measurements.best()
    pruned = without_copies(best.genome)
    assert copies(best.genome) == ["DepotSpecialist1"]
    assert set(best.genome.agents) - set(pruned.agents) == {"DepotSpecialist1"}
    assert pruned.agents[pruned.top].model == best.genome.agents[best.genome.top].model
    assert "DepotSpecialist1" not in pruned.agents[pruned.top].tools


def test_a_network_without_copies_is_unchanged():
    genome = designer_shaped()
    assert copies(genome) == []
    assert without_copies(genome).genome_hash() == genome.genome_hash()


def test_pinned_models_are_never_substituted(monkeypatch):
    failover.reset()
    monkeypatch.setattr(failover, "PINNED", True)
    try:
        assert failover.retire(failover.LADDER[0]) is None
        assert failover.substitute(failover.LADDER[0]) == failover.LADDER[0]
    finally:
        failover.reset()


def _plan(budget: float, questions: int = 8, repeats: int = 2) -> h2h.Plan:
    return h2h.Plan(JUDGE_200[:questions], repeats, budget, h2h.contenders())


def test_every_chunk_is_asked_of_both_networks_in_alternating_order():
    plan = _plan(10.0)
    steps = plan.order()
    assert len(steps) == 2 * len(plan.chunks()) * plan.repeats
    firsts = [steps[i][2].label for i in range(0, len(steps), 2)]
    assert "designer" in firsts and "champion" in firsts


def test_the_cap_stops_the_run_and_a_rerun_resumes(tmp_path):
    log = h2h.Log(tmp_path / "runs.jsonl")
    assert h2h.run(_plan(0.02), h2h.rehearsal_measure, log, say=lambda *_: None) == "budget"
    assert log.spent() <= 0.02
    first = len(log.rows)

    again = h2h.Log(tmp_path / "runs.jsonl")
    assert len(again.rows) == first
    assert h2h.run(_plan(10.0), h2h.rehearsal_measure, again,
                   say=lambda *_: None) == "done"
    keys = [(r["label"], r["repeat"], r["chunk"]) for r in again.rows]
    assert len(keys) == len(set(keys)) == len(_plan(10.0).order())


def test_a_provider_error_stops_the_run_and_keeps_what_was_measured(tmp_path):
    calls = []

    def flaky(contender, chunk):
        calls.append(contender.label)
        if len(calls) == 3:
            raise OSError("quota")
        return h2h.rehearsal_measure(contender, chunk)

    log = h2h.Log(tmp_path / "runs.jsonl")
    stopped = h2h.run(_plan(10.0), flaky, log, say=lambda *_: None)
    assert stopped.startswith("OSError")
    assert len(log.rows) == 2


def test_the_summary_counts_every_question_run(tmp_path):
    plan = _plan(10.0)
    log = h2h.Log(tmp_path / "runs.jsonl")
    h2h.run(plan, h2h.rehearsal_measure, log, say=lambda *_: None)
    summary = h2h.summarise(log, plan)
    for network in summary["networks"].values():
        assert network["question_runs"] == 8 * 2
        assert 0 <= network["accuracy"] <= 1
    assert "designer" in h2h.render(summary)


@pytest.mark.parametrize("model", ["gpt-5.4-nano", "gpt-5.4-mini"])
def test_the_cheap_openai_rungs_have_a_price(model):
    from esp.eval.pricing import blended
    assert blended(model) > 0
