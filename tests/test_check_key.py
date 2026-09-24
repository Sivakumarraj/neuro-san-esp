"""`make check-key` judges the key the run will call with, and only that one.

The devcontainer copies .env.example to .env. While the example shipped an
active `ANTHROPIC_API_KEY=paste-your-key-here`, everybody who then pasted a
Gemini key was told their key was broken -- by this check, while the preflight
passed and the run used Gemini without trouble.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAUSIBLE_GEMINI = "AQ." + "x" * 40


def _check_key(**env: str) -> subprocess.CompletedProcess:
    environment = {**os.environ, "ESP_NO_DOTENV": "1", "PYTHONPATH": str(ROOT), **env}
    for name in ("ESP_MODEL_TIERS", "ESP_DEFAULT_MODEL", "ESP_PROVIDER", "GOOGLE_API_KEYS",
                 "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                 "OPENROUTER_API_KEY"):
        if name not in env:
            environment.pop(name, None)
    return subprocess.run([sys.executable, "scripts/check_key.py", "--offline"],
                          capture_output=True, text=True, env=environment, cwd=ROOT,
                          timeout=120)


def test_a_placeholder_on_a_provider_not_in_use_is_only_a_note():
    done = _check_key(GOOGLE_API_KEY=PLAUSIBLE_GEMINI,
                      ANTHROPIC_API_KEY="paste-your-key-here")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "note: ANTHROPIC_API_KEY" in done.stdout
    assert "this run uses GOOGLE_API_KEY" in done.stdout


def test_a_placeholder_with_no_other_key_still_fails():
    done = _check_key(ANTHROPIC_API_KEY="paste-your-key-here")
    assert done.returncode == 1
    assert "placeholder" in done.stdout


def test_a_model_whose_key_is_missing_fails_even_when_another_key_is_set():
    """It used to verify whichever key came first and report that one healthy."""
    done = _check_key(GOOGLE_API_KEY=PLAUSIBLE_GEMINI, ESP_DEFAULT_MODEL="claude-sonnet")
    assert done.returncode == 1
    assert "claude-sonnet needs ANTHROPIC_API_KEY" in done.stdout


def test_the_env_example_sets_no_key():
    """Its copy is what a fresh Codespace runs on, so an active placeholder line
    there is a broken key in every Codespace that uses another provider."""
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    active = re.findall(r"^\s*[A-Z_]+_API_KEYS?\s*=.*$", example, flags=re.MULTILINE)
    assert active == []
