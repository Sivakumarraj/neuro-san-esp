"""The .env loader, on the ways people actually write the file."""

from __future__ import annotations

import pytest

from esp import config


@pytest.mark.parametrize("line, expected", [
    ("K = v1", "v1"),
    ("export K=v1", "v1"),
    ('K="v 1"', "v 1"),
    ("K='v1' # note", "v1"),
    ("K=v1  # note", "v1"),
    ('K="a#b"', "a#b"),
    ("K=a#b", "a#b"),
])
def test_values_come_out_as_typed_without_quotes_or_comments(tmp_path, monkeypatch, line, expected):
    monkeypatch.delenv("K", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"# header\n\n{line}\n", encoding="utf-8")
    assert config.load_env(env) == ["K"]
    import os
    assert os.environ["K"] == expected


def test_a_value_already_in_the_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("K", "from-shell")
    env = tmp_path / ".env"
    env.write_text("K=from-file\n", encoding="utf-8")
    assert config.load_env(env) == []
    import os
    assert os.environ["K"] == "from-shell"
