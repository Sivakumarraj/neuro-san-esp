"""The Codespaces setup, checked without opening a Codespace.

A broken devcontainer.json does not announce itself: Codespaces falls back to a
default image and the failure surfaces much later as an import error or a
missing Python version. Since this file exists specifically so that somebody
opening the repository in a browser gets a working environment, the properties
that make it work are worth pinning.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / ".devcontainer/devcontainer.json"


def config() -> dict:
    """Parse it as JSONC. Comments are legal in devcontainer.json and are used
    here to explain the pins, so a plain json.load would reject the real file."""
    text = PATH.read_text(encoding="utf-8")
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    return json.loads(text)


def test_it_is_parseable():
    assert config()["name"] == "neuro-san-esp"


def test_python_is_pinned_to_312():
    """neuro-san's server imports asyncio.eager_task_factory, which does not
    exist on 3.11. On a default image the server dies at startup with an
    AttributeError that says nothing about Python versions."""
    assert "3.12" in config()["image"]


def test_the_project_installs_itself_on_create():
    """A newcomer should not have to know the install command before the first
    thing they try works."""
    assert "pip install -e" in config()["postCreateCommand"]


def test_the_env_file_is_created_but_never_overwritten():
    """cp -n. Clobbering a key somebody already pasted would be a bad first
    impression, and a confusing one."""
    command = config()["postCreateCommand"]
    assert "cp -n .env.example .env" in command


def test_the_requirements_it_installs_actually_exist():
    assert (ROOT / "pyproject.toml").exists()
    assert (ROOT / ".env.example").exists()
