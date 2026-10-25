"""`make verify` asks for the key the configured model calls with, not Google's.

It demanded GOOGLE_API_KEY whatever the model, so on a machine holding only an
OpenAI or a Claude key the one check a unit test cannot make refused to start.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_periodic  # noqa: E402

from esp.genome import definition  # noqa: E402

PLAUSIBLE_OPENAI = "sk-" + "x" * 40


def test_a_machine_with_only_an_openai_key_is_not_asked_for_googles(monkeypatch):
    monkeypatch.setattr(definition, "DEFAULT_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_API_KEY", PLAUSIBLE_OPENAI)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert verify_periodic.missing_key() == ""


def test_the_configured_models_key_is_still_required(monkeypatch):
    monkeypatch.setattr(definition, "DEFAULT_MODEL", "claude-sonnet")
    monkeypatch.setenv("OPENAI_API_KEY", PLAUSIBLE_OPENAI)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert verify_periodic.missing_key() == (
        "claude-sonnet needs ANTHROPIC_API_KEY, which is not set")


def test_a_placeholder_is_not_a_key(monkeypatch):
    monkeypatch.setattr(definition, "DEFAULT_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "paste-your-key-here")
    assert "needs OPENAI_API_KEY" in verify_periodic.missing_key()
