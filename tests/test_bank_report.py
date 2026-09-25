"""The held-out bank figures in the docs are read from the committed reports.

The same rule as every other published number here: a reader can rebuild it
with no key, and a test fails if the prose and the data part company.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import bank_report  # noqa: E402


def right(report: dict) -> int:
    return sum(1 for r in report["results"] if r["correct"])


def test_the_committed_reports_say_what_the_docs_say():
    reports = bank_report.load()
    designer, evolved = reports["designer"], reports["evolved"]
    assert designer["suite"] == evolved["suite"] == "meridian-bank:24"
    assert (right(designer), right(evolved)) == (23, 21)
    assert (designer["tokens"], evolved["tokens"]) == (159_086, 176_740)
    findings = (ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")
    assert "| `seed:designer_shaped` | **23 / 24** | 159,086 |" in findings
    assert "| `mut:reassign_model` 6859dd | 21 / 24 | 176,740 |" in findings
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "answered 21 against the designer's 23" in readme


def test_the_reports_name_networks_that_were_measured():
    from esp.eval import measurements
    known = {record.genome_hash for record in measurements.load()}
    for report in bank_report.load().values():
        assert report["genome_hash"] in known


def test_the_paired_test_is_the_exact_one():
    assert bank_report.mcnemar_exact(2, 0) == 0.5
    assert bank_report.mcnemar_exact(0, 0) == 1.0
    assert abs(bank_report.mcnemar_exact(10, 0) - 2 / 1024) < 1e-12


def test_the_report_runs(capsys):
    assert bank_report.main() == 0
    out = capsys.readouterr().out
    assert "exact McNemar two-sided p = 0.500" in out
