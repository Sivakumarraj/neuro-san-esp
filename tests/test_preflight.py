"""Refusing to start beats producing plausible wrong numbers.

A misconfigured evaluator does not crash -- it scores every candidate zero and
caches the result, so the search is taught that good topologies are bad. Both
of this project's real instances of that were configuration, not logic.
"""

from __future__ import annotations

import pytest

from esp.service import preflight


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for name in ("GOOGLE_API_KEY", "AGENT_TOOL_PATH", "PYTHONPATH",
                 "AGENT_NETWORK_DESIGNER_DEMO_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(preflight, "STATE_DIR", tmp_path / "state")
    return monkeypatch


def named(checks, name):
    return next(c for c in checks if c.name == name)


def test_a_missing_key_is_fatal(clean):
    checks = preflight.run_checks()
    assert not named(checks, "provider key").ok
    assert named(checks, "provider key") in preflight.failures(checks)


def test_a_missing_agent_tool_path_is_fatal(clean):
    """The exact bug that made probe_models.py report every healthy model as
    BROKEN: without it neuro-san refuses to build a session at all."""
    clean.setenv("GOOGLE_API_KEY", "x")
    checks = preflight.run_checks()
    assert not named(checks, "AGENT_TOOL_PATH").ok
    assert named(checks, "AGENT_TOOL_PATH").fatal


def test_demo_mode_is_refused(clean):
    """neuro-san-studio's demo mode tells generated agents to invent a
    realistic-looking answer. Accuracy would be measuring fabrication."""
    clean.setenv("GOOGLE_API_KEY", "x")
    clean.setenv("AGENT_TOOL_PATH", "/somewhere")
    clean.setenv("AGENT_NETWORK_DESIGNER_DEMO_MODE", "true")
    checks = preflight.run_checks()
    assert not named(checks, "designer demo mode").ok


def test_a_missing_pythonpath_only_warns(clean):
    """It is usually already importable. Refusing to start over it would make
    the preflight the thing operators route around."""
    clean.setenv("GOOGLE_API_KEY", "x")
    clean.setenv("AGENT_TOOL_PATH", "/somewhere")
    checks = preflight.run_checks()
    assert not named(checks, "PYTHONPATH").fatal


def test_an_unwritable_state_directory_is_fatal(clean, tmp_path):
    """Better to refuse than to spend eight minutes on a candidate that cannot
    be recorded afterwards."""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    clean.setattr(preflight, "STATE_DIR", blocker / "state")
    checks = preflight.run_checks()
    assert not named(checks, "state directory writable").ok


def test_a_healthy_configuration_passes(clean, tmp_path):
    clean.setenv("GOOGLE_API_KEY", "x")
    clean.setenv("AGENT_TOOL_PATH", str(tmp_path))
    checks = preflight.run_checks()
    assert preflight.failures(checks) == []


def test_the_report_names_every_check(clean):
    text = preflight.report(preflight.run_checks())
    assert "provider key" in text and "AGENT_TOOL_PATH" in text


# ------------------------------------------------------------------ the .env


def test_a_key_can_be_pasted_into_a_file(tmp_path, monkeypatch):
    """Asking somebody to export a variable before every command is how a key
    ends up in shell history, or in a script, or committed."""
    from esp import config

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=abc123\n", encoding="utf-8")
    assert config.load_env(env) == ["GOOGLE_API_KEY"]
    import os

    assert os.environ["GOOGLE_API_KEY"] == "abc123"


def test_the_environment_beats_the_file(tmp_path, monkeypatch):
    """A value exported in the shell, set by a container, or injected by a
    secrets manager is the more deliberate one. A stale .env quietly overriding
    it would be miserable to debug."""
    from esp import config

    monkeypatch.setenv("GOOGLE_API_KEY", "from-the-shell")
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=from-the-file\n", encoding="utf-8")
    config.load_env(env)
    import os

    assert os.environ["GOOGLE_API_KEY"] == "from-the-shell"


def test_quotes_and_export_are_stripped(tmp_path, monkeypatch):
    """What a person actually types. A key carrying a stray quote fails with an
    authentication error that says nothing about quoting."""
    from esp import config

    monkeypatch.delenv("ESP_TEST_VALUE", raising=False)
    env = tmp_path / ".env"
    env.write_text('export ESP_TEST_VALUE="quoted"\n', encoding="utf-8")
    config.load_env(env)
    import os

    assert os.environ["ESP_TEST_VALUE"] == "quoted"


def test_comments_and_blank_lines_are_ignored(tmp_path):
    from esp import config

    env = tmp_path / ".env"
    env.write_text("# a comment\n\nnot-a-pair\n", encoding="utf-8")
    assert config.load_env(env) == []


def test_a_missing_env_file_is_not_an_error(tmp_path):
    """The whole offline half of ESP runs with no key at all. Requiring a file
    would break the one path that needs no credentials."""
    from esp import config

    assert config.load_env(tmp_path / "nope.env") == []


def test_the_ring_size_is_reported_when_multiple_keys_are_set(clean):
    """A second key doubles the daily budget, so the preflight has to say so."""
    from esp.eval import ratelimit
    clean.setenv("GOOGLE_API_KEY", "k1")
    clean.setenv("GOOGLE_API_KEYS", "k1,k2")
    clean.setenv("AGENT_TOOL_PATH", "/tmp")
    ratelimit.reload_keyring()
    try:
        checks = preflight.run_checks()
        assert "2 keys in ring" in named(checks, "provider key").detail
        ladder = named(checks, "model ladder").detail
        assert "2000 requests/day (2 keys)" in ladder
        assert "about 12 candidate" in ladder
    finally:
        clean.delenv("GOOGLE_API_KEYS", raising=False)
        ratelimit.reload_keyring()


def test_the_ring_note_is_omitted_for_a_single_key(clean):
    """The single-key case should look the same it always has -- no noise."""
    from esp.eval import ratelimit
    clean.setenv("GOOGLE_API_KEY", "solo")
    clean.setenv("AGENT_TOOL_PATH", "/tmp")
    clean.delenv("GOOGLE_API_KEYS", raising=False)
    ratelimit.reload_keyring()
    try:
        checks = preflight.run_checks()
        assert "keys in ring" not in named(checks, "provider key").detail
        assert "keys)" not in named(checks, "model ladder").detail
    finally:
        ratelimit.reload_keyring()
