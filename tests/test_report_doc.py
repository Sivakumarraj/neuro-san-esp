"""docs/REPORT.md walks a reader through the project with figures from the logs.
Its charts are drawn from the logs and its numbers must be the logs' numbers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import report_figures  # noqa: E402

from esp import route  # noqa: E402
from esp.evolve import paid  # noqa: E402

REPORT = (ROOT / "docs" / "REPORT.md").read_text(encoding="utf-8")


def test_every_chart_is_drawn_from_the_logs_and_linked(tmp_path):
    drawn = report_figures.ReportFigures(tmp_path).draw()
    assert [p.name for p in drawn] == ["report-headline.png", "report-by-kind.png",
                                       "report-frontier.png"]
    for path in drawn:
        assert path.stat().st_size > 10_000
        assert f"../results/figures/{path.name}" in REPORT
        assert (ROOT / "results" / "figures" / path.name).exists()


def test_the_report_quotes_the_head_to_head_and_the_router_as_measured():
    judged = paid.judged()
    judge = route.analyse()["judge"]
    quoted = [
        f"**{judged.accuracy:.1%}** ({judged.correct}/{judged.question_runs})",
        f"{judged.designer_accuracy:.1%} ({judged.designer_correct}/{judged.question_runs})",
        f"**{judged.tokens_per_question:,}** (\N{MINUS SIGN}{-judged.token_change:.0%})",
        f"${judged.cost_per_question:.5f} (+{judged.cost_change:.0%})",
        f"p = {judged.sign_test_p}",
        f"scores {judge['router']['accuracy']:.1%} at ${judge['router']['dollars']:.5f}",
        f"scores {judge['random_same_share']:.1%} (p = {judge['p_against_random']})",
        f"keeps all {judge['rule']['accuracy']:.1%} at ${judge['rule']['dollars']:.5f}",
    ]
    missing = [figure for figure in quoted if figure not in REPORT]
    assert not missing, missing
