"""The paid head-to-head runs: their committed logs, the summary built from them,
and the figures the README quotes from that summary.

The paid runs cannot be repeated in CI, but everything said about them can be
recomputed from the logs in results/headtohead/paid-2026-10/. These tests do
that, so a number in the README cannot drift from the runs it describes.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "headtohead" / "paid-2026-10"
README = (ROOT / "README.md").read_text(encoding="utf-8")

sys.path.insert(0, str(ROOT / "scripts"))
import select_and_judge as sj  # noqa: E402


def _summary() -> dict:
    report = {}
    for stage in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        rows = sj.load_rows(stage)
        labels = sorted({r["label"] for r in rows})
        report[stage.name] = {"networks": sj.summarise(rows)}
        if len(labels) == 2:
            report[stage.name]["paired"] = sj.paired(rows, *labels)
    return report


def test_the_committed_summary_is_what_the_logs_say():
    committed = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    assert _summary() == committed


def test_the_readme_quotes_the_runs_as_measured():
    stages = _summary()

    def line(stage: str, label: str) -> str:
        n = stages[stage]["networks"][label]
        return f"{n['correct']}/{n['question_runs']}"

    for stage, label in (("5-judge-2cc4", "evolved"), ("5-judge-2cc4", "designer"),
                         ("6-judge-plus-P001-P100", "evolved"),
                         ("6-judge-plus-P001-P100", "designer"),
                         ("8-judge-search10", "evolved"), ("8-judge-search10", "designer")):
        quoted = line(stage, label)
        assert quoted in README, f"README is missing {stage} {label} {quoted}"
    p = stages["8-judge-search10"]["paired"]["sign_test_p"]
    assert f"p = {p}" in README


def test_flat_and_solo_are_called_seeds_not_search_results():
    from esp.eval import measurements
    from esp.genome.seeds import SEEDS

    committed = {r.genome_hash: r.genome for r in measurements.load()}
    for genome_hash, seed_name in sj.SEED_HASHES.items():
        assert genome_hash in committed
        assert set(committed[genome_hash].agents) == set(SEEDS[seed_name]().agents)
        assert "hand-written seed" in sj.origin_of(genome_hash)
    assert "Gemini search" in sj.origin_of(sj.COMMITTED["mergeA"])


def test_the_suites_are_the_ones_named():
    assert [len(sj.suite_tasks(s)) for s in sj.SUITES] == [60, 100, 100]
    assert sj.suite_tasks("judge-plus")[0].task_id == "P001"


def test_search_returns_three_documents_unless_told_otherwise(monkeypatch):
    from esp.eval import corpus_tool

    try:
        monkeypatch.delenv("ESP_SEARCH_RESULTS", raising=False)
        assert importlib.reload(corpus_tool).MAX_RESULTS == 3
        monkeypatch.setenv("ESP_SEARCH_RESULTS", "10")
        reloaded = importlib.reload(corpus_tool)
        assert reloaded.MAX_RESULTS == 10
        # "mis-picked" is in eight documents; at three a count over them comes up short.
        assert len(reloaded.search("mis-picked pallet")) > 3
    finally:
        monkeypatch.delenv("ESP_SEARCH_RESULTS", raising=False)
        importlib.reload(corpus_tool)
