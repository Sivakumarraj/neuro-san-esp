"""The pool benchmark, end to end against the simulated provider: breeding,
measuring with a stop for quota and a resume that pays for nothing twice, the
replicate comparison (paired starts, a planted advantage found, pure noise not
mistaken for one), judging, and the $0 rehearsal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from esp.eval import runner
from esp.eval.rehearsal import SimulatedProvider
from esp.eval.suites import SELECT
from esp.evolve import pool
from esp.genome.seeds import SEEDS

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = SELECT[:20]


@pytest.fixture
def simulated(monkeypatch, tmp_path):
    genomes = pool.breed(30, seed=5)
    provider = SimulatedProvider([*genomes, SEEDS["designer_shaped"]()])
    monkeypatch.setattr(runner, "run_suite", provider)
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "NETWORK_DIR", tmp_path / "networks")
    return provider, genomes


def small_plan() -> pool.Plan:
    return pool.Plan(select=QUESTIONS, judge=SELECT[20:40], size=30, finalists=3, seed=5)


def test_the_pool_is_distinct_valid_networks_and_includes_the_seeds():
    genomes = pool.breed(60)
    hashes = [g.genome_hash() for g in genomes]
    assert len(set(hashes)) == 60
    assert {b().genome_hash() for b in SEEDS.values()} <= set(hashes)
    assert hashes == [g.genome_hash() for g in pool.breed(60)]
    assert len({len(g.reachable()) for g in genomes}) >= 4, "a pool of near-copies"


def test_a_quota_stop_resumes_and_pays_for_nothing_twice(simulated, tmp_path):
    provider, genomes = simulated
    provider.fail_after = 12
    first = pool.measure(small_plan(), tmp_path / "out", genomes)
    assert first["stopped"] and first["measured"] == 12
    provider.fail_after = None
    second = pool.measure(small_plan(), tmp_path / "out", genomes)
    assert second["stopped"] is None and second["measured"] == 30
    assert provider.suites == 30, "a network measured before the stop was paid again"


def test_every_strategy_starts_from_the_same_networks(simulated, tmp_path):
    _, genomes = simulated
    pool.measure(small_plan(), tmp_path / "out", genomes)
    members = pool.load(tmp_path / "out" / "pool.json")
    for seed in range(3):
        runs = {s: pool.replicate(members, QUESTIONS, s, seed, budgets=(5, 10))
                for s in pool.STRATEGIES}
        assert all(set(r) == {5, 10} for r in runs.values())
        assert all(np.isfinite(list(r.values())).all() for r in runs.values())
    # The random start depends on the replicate seed only.
    visible, scored = pool.split(QUESTIONS)
    assert not visible & scored and len(visible | scored) == len(QUESTIONS)


def test_a_planted_advantage_is_found(simulated, tmp_path):
    """Where each network's answers follow its structure without coin-flip
    noise, a Predictor has something to learn, and choosing by it must beat
    choosing at random. (With the simulated provider's noise and only a few
    networks measured it does not, and that is the reason the real run
    measures a pool rather than a handful.)"""
    _, genomes = simulated
    pool.measure(small_plan(), tmp_path / "out", genomes)
    members = pool.load(tmp_path / "out" / "pool.json")
    for member in members:
        genome = member.observed().genome
        # Graded, and inside the range the start networks span: a tree-based
        # Predictor cannot extrapolate past the largest team it has seen.
        ability = 0.55 + 0.05 * min(len(genome.reachable()), 6) - 0.04 * genome.depth()
        member.right = {k: ability > 0.1 + 0.8 * i / len(member.right)
                        for i, k in enumerate(sorted(member.right))}
        member.accuracy = float(np.mean(list(member.right.values())))
        member.dollars = 0.1
    summary = pool.compare(members, QUESTIONS, replicates=40, budgets=(5, 10),
                           strategies=("random", "question"))
    row = summary["strategies"]["question"][10]
    assert row["vs_random"] > 0 and row["better_than_random"], row


def test_pure_noise_is_not_mistaken_for_an_advantage(simulated, tmp_path):
    _, genomes = simulated
    pool.measure(small_plan(), tmp_path / "out", genomes)
    members = pool.load(tmp_path / "out" / "pool.json")
    rng = np.random.default_rng(0)
    for member in members:
        member.right = {k: bool(rng.random() < 0.6) for k in member.right}
        member.accuracy = float(np.mean(list(member.right.values())))
        member.dollars = 0.1
        member.agents = 3       # size is exact and learnable; hold it constant too
    summary = pool.compare(members, QUESTIONS, replicates=24, budgets=(5,),
                           strategies=("random", "question"))
    row = summary["strategies"]["question"][5]
    assert not row["better_than_random"], row


def test_the_finalists_and_the_designer_are_judged(simulated, tmp_path):
    _, genomes = simulated
    plan = small_plan()
    pool.measure(plan, tmp_path / "out", genomes)
    members = pool.load(tmp_path / "out" / "pool.json")
    result = pool.judge(members, plan, tmp_path / "out")
    designer = SEEDS["designer_shaped"]().genome_hash()
    assert designer in result["judged"] and len(result["judged"]) in (3, 4)
    for comparison in result["vs_designer"].values():
        assert comparison["questions"] == 20 and 0 <= comparison["p"] <= 1
    assert json.loads((tmp_path / "out" / "judge.json").read_text())["designer"] == designer


def test_the_plan_is_priced_and_spends_nothing(capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pool_benchmark
    assert pool_benchmark.main([]) == 0
    out = capsys.readouterr().out
    assert "9,000 question-runs" in out and "Nothing was spent" in out


def test_the_rehearsal_runs_everything_for_nothing(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pool_benchmark
    before = runner.run_suite
    assert pool_benchmark.main(["--rehearse", "--size", "20", "--select", "10",
                                "--finalists", "2", "--replicates", "4",
                                "--out", str(tmp_path)]) == 0
    assert runner.run_suite is before, "the rehearsal left the simulated provider in place"
    totals = json.loads((tmp_path / "totals.json").read_text())
    assert totals["simulated"] and totals["question_runs"] == 20 * 10 + 3 * 200
    assert "$0 spent" in capsys.readouterr().out
