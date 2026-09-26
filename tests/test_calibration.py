"""The select set's calibration, as committed, and as the docs quote it."""

from __future__ import annotations

import json
from pathlib import Path

from esp.eval.suites import SELECT
from esp.evolve.experiment import CEILING

ROOT = Path(__file__).resolve().parent.parent
REPORT = json.loads((ROOT / "results" / "calibration" / "designer_select20.json")
                    .read_text(encoding="utf-8"))


def test_it_measured_the_questions_the_select_set_now_holds():
    """A regenerated select set would make this calibration describe a
    different exam. The questions asked must still be the first twenty."""
    asked = {r["task_id"]: r["question"] for r in REPORT["results"]}
    assert asked == {t.task_id: t.question for t in SELECT[:20]}


def test_the_docs_quote_it_and_it_clears_the_ceiling():
    right = sum(1 for r in REPORT["results"] if r["correct"])
    assert (right, REPORT["tokens"]) == (16, 194_828)
    assert REPORT["accuracy"] < CEILING
    findings = (ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "**16 of 20 (80%)**, using 194,828 tokens" in findings
    assert "scored 16 of 20 (80%)" in readme
