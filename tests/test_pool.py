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


def test_a_planted_advantage_is_found():
    """Where each network's answers follow its structure without coin-flip
    noise, a Predictor has something to learn, and choosing by it must beat
    choosing at random -- against the permutation null, not just against
    random choice on this pool.

    Sixty networks on forty questions, because the null is honest about how
    much one pool can show: on the thirty-network pool this test once used,
    the same planted advantage (+0.04) sat inside the spread of permuted pools
    (p = 0.31). A tree-based Predictor cannot extrapolate past the largest
    team it has seen, so ability is graded inside the range the start spans."""
    questions = SELECT[:40]
    ids = sorted(t.task_id for t in questions)
    members = []
    for genome in pool.breed(60, seed=11):
        ability = 0.55 + 0.05 * min(len(genome.reachable()), 6) - 0.04 * genome.depth()
        right = {k: ability > 0.1 + 0.8 * i / len(ids) for i, k in enumerate(ids)}
        members.append(pool.Member(genome.genome_hash(), genome.canonical(), right,
                                   float(np.mean(list(right.values()))), 1000, 0.1, 3))
    summary = pool.compare(members, questions, replicates=40, budgets=(5, 10),
                           strategies=("random", "question"))
    row = summary["strategies"]["question"][10]
    assert row["vs_random"] > row["null"], row
    assert row["better_than_random"], {k: row[k] for k in ("vs_random", "null", "p", "p_holm")}


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
                                "--finalists", "2", "--replicates", "4", "--permutations", "2",
                                "--out", str(tmp_path)]) == 0
    assert runner.run_suite is before, "the rehearsal left the simulated provider in place"
    totals = json.loads((tmp_path / "totals.json").read_text())
    assert totals["simulated"] and totals["question_runs"] == 20 * 10 + 3 * 200
    assert "$0 spent" in capsys.readouterr().out


def noise_pool(size: int, seed: int) -> list[pool.Member]:
    """Answers that are coin flips, cost and size held equal: nothing to find."""
    rng = np.random.default_rng(seed)
    ids = [t.task_id for t in QUESTIONS]
    members = []
    for genome in pool.breed(size, seed=11):
        right = {k: bool(rng.random() < 0.6) for k in ids}
        members.append(pool.Member(genome.genome_hash(), genome.canonical(), right,
                                   float(np.mean(list(right.values()))), 1000, 0.1, 3))
    return members


def fixed_preference(salt: str):
    """A strategy with a firm preference and no information: it ranks networks
    by a hash of their own hash, the same way in every replicate."""
    import hashlib

    def score(known, unknown, seen, seed):
        return [int(hashlib.sha256((salt + m.genome_hash).encode()).hexdigest()[:8], 16)
                for m in unknown]
    score.__name__ = f"fixed-{salt}"
    return score


def test_a_fixed_preference_with_no_information_is_not_called_better():
    """The first version put its interval over the replicates of one pool. A
    strategy that always prefers the same networks then inherits whatever
    luck those networks had on the scored questions, and more replicates only
    made the interval tighter around that luck: on pure noise such a
    strategy came out "better than random" in a third of comparisons and
    worse in half. It is now tested against a permutation null, Holm-adjusted."""
    flagged, raw = 0, 0
    for seed in range(4):
        summary = pool.compare(noise_pool(40, seed), QUESTIONS, replicates=60,
                               budgets=(5, 10),
                               strategies=("random", fixed_preference("a"),
                                           fixed_preference("b")))
        for rows in summary["strategies"].values():
            for row in rows.values():
                if "p" in row:
                    flagged += row["better_than_random"] or row["worse_than_random"]
                    raw += row["vs_random_95"][0] > 0 or row["vs_random_95"][1] < 0
    assert flagged == 0, flagged
    assert raw <= 1, raw


def test_holm_adjusts_a_family_of_p_values():
    adjusted = pool.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert pool.holm({}) == {}


def test_a_network_that_answers_nothing_is_recorded_and_not_paid_for_again(simulated, tmp_path,
                                                                          monkeypatch):
    """The runner refuses a network whose every question failed, rightly. The
    pool used to let that refusal escape, so the run stopped at that network
    and every resume paid for it again and stopped again."""
    provider, genomes = simulated
    broken = genomes[7].genome_hash()
    real = provider.__call__

    def call(hocon_path, tasks, **kw):
        run = real(hocon_path, tasks, **kw)
        if Path(hocon_path).stem == broken:
            for result in run.results:
                result.correct, result.answer = False, "Agent timed out after 600 seconds"
            runner.classify(run.results)      # as the real run_suite does
        return run
    monkeypatch.setattr(runner, "run_suite", call)

    first = pool.measure(small_plan(), tmp_path / "out", genomes)
    assert first["stopped"] is None and first["measured"] == 29
    assert list(first["unmeasurable"]) == [broken]
    paid = provider.suites
    second = pool.measure(small_plan(), tmp_path / "out", genomes)
    assert second["measured"] == 29 and provider.suites == paid, "paid again"


def test_several_failures_in_a_row_stop_the_run_and_blame_nothing(simulated, tmp_path,
                                                                  monkeypatch):
    _, genomes = simulated

    def refuse(*_args, **_kw):
        raise OSError("no key")
    monkeypatch.setattr(runner, "run_suite", refuse)
    result = pool.measure(small_plan(), tmp_path / "out", genomes)
    assert result["stopped"] and "environment" in result["stopped"]
    assert result["unmeasurable"] == {} and result["measured"] == 0


def test_the_price_comes_from_what_committed_runs_cost():
    """The plan was priced at a flat 12,000 tokens a question that nothing
    measured, below the rehearsal's own simulated scale. The range is now the
    select set's calibration run to the twelve networks on the seventeen."""
    low, high = pool.tokens_per_question()
    calibration = json.loads((ROOT / "results" / "calibration" /
                              "designer_select20.json").read_text())
    assert low == round(calibration["tokens"] / len(calibration["results"]))
    rehearsed = json.loads((ROOT / "results" / "rehearsal" / "totals.json").read_text())
    assert low < rehearsed["tokens"] / rehearsed["question_runs"] < high
