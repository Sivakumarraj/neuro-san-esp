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


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "openai", "openrouter"])
def test_every_default_ladder_can_be_priced(provider):
    """The Anthropic and OpenRouter ladders name models with no price entry, and
    `make headtohead` stopped with a KeyError before it printed its plan."""
    from esp.config import DEFAULT_LADDERS
    from esp.eval.pricing import blended

    for model in DEFAULT_LADDERS[provider]:
        assert blended(model) >= 0


def test_a_log_of_other_questions_is_not_resumed_as_this_plan(tmp_path):
    """Resume matched chunks on (repeat, index, network) alone, so a log measured
    on one question set was resumed on another and reported complete."""
    log = h2h.Log(tmp_path / "runs.jsonl")
    assert h2h.run(_plan(10.0), h2h.rehearsal_measure, log, say=lambda *_: None) == "done"
    other = h2h.Plan(JUDGE_200[100:108], 2, 10.0, h2h.contenders())
    stopped = h2h.run(other, h2h.rehearsal_measure, h2h.Log(tmp_path / "runs.jsonl"),
                      say=lambda *_: None)
    assert stopped.startswith("ValueError") and "new --out" in stopped


def test_a_log_at_another_search_width_is_not_resumed(tmp_path, monkeypatch):
    from esp.eval import corpus_tool, runner

    def measured(contender, chunk):
        return {**h2h.rehearsal_measure(contender, chunk), "settings": runner.run_settings()}

    plan = _plan(10.0)
    assert h2h.run(plan, measured, h2h.Log(tmp_path / "runs.jsonl"), say=lambda *_: None) == "done"
    monkeypatch.setattr(corpus_tool, "MAX_RESULTS", 10)
    stopped = h2h.run(plan, measured, h2h.Log(tmp_path / "runs.jsonl"), say=lambda *_: None)
    assert stopped.startswith("ValueError") and "search_results" in stopped


def test_the_cap_prices_the_next_chunk_at_the_dearest_so_far(tmp_path):
    """Priced at the mean, cheap chunks after a dear one let the next dear one
    through: 0.048 spent, the mean said 0.016 more against a 0.081 cap, and the
    chunk cost 0.04, ending at 0.088."""
    costs = iter([0.04, 0.004, 0.004, 0.04])

    def priced(contender, chunk):
        return {**h2h.rehearsal_measure(contender, chunk), "cost": next(costs)}

    plan = h2h.Plan(JUDGE_200[:16], 1, 0.081, h2h.contenders()[:1])
    log = h2h.Log(tmp_path / "runs.jsonl")
    assert h2h.run(plan, priced, log, say=lambda *_: None) == "budget"
    assert log.spent() <= plan.budget


def test_a_run_the_cap_stopped_does_not_exit_zero(tmp_path, monkeypatch):
    """`make headtohead` exited 0 when the cap or the provider stopped the run,
    so anything reading the exit code took an unfinished run for a done one."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import headtohead as script

    monkeypatch.setenv("ESP_PIN_MODELS", "1")
    common = ["--rehearse", "--questions", "8", "--repeats", "1"]
    assert script.main([*common, "--budget", "0.00001", "--out", str(tmp_path / "cap")]) == 1
    assert script.main([*common, "--budget", "10", "--out", str(tmp_path / "full")]) == 0

