"""`make smoke` is the first thing run on a new key. It has to tell an outage
from a wrong answer.

Its first live run, on a free Gemini key, met a router model answering 503 "high
demand" on three of four questions. neuro-san returns an agent's failure as the
reply text, so the script printed WRONG three times, reported four questions
answered, and exited 0 -- an outage presented as the champion being wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

smoke_live = pytest.importorskip("smoke_live")

OUTAGE = ("Error from Coordinator: Agent stopped due to exception 503 UNAVAILABLE. "
          "This model is currently experiencing high demand.")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["smoke_live.py", "--questions", "2"])
    monkeypatch.setattr(smoke_live, "run_checks", lambda live: [])
    monkeypatch.setattr(smoke_live, "write_network", lambda genome: tmp_path / "n.hocon")


def test_an_outage_is_failed_not_wrong_and_the_run_fails(offline, monkeypatch, capsys):
    monkeypatch.setattr(smoke_live, "_ask", lambda *a: (OUTAGE, {}, 1.0))
    assert smoke_live.main() == 1
    out = capsys.readouterr().out
    assert "[FAILED" in out and "[WRONG" not in out
    assert '"answered": 0' in out


def test_a_real_wrong_answer_is_still_wrong(offline, monkeypatch, capsys):
    monkeypatch.setattr(smoke_live, "_ask",
                        lambda *a: ("Nobody", {"total_tokens": 10}, 1.0))
    monkeypatch.setattr(smoke_live, "_total_tokens", lambda accounting: 10)
    assert smoke_live.main() == 0
    out = capsys.readouterr().out
    assert "[WRONG" in out and "[FAILED" not in out
