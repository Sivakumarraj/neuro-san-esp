"""The documentation must not drift away from the repository it describes.

Numbers in a README rot silently. This one has already claimed 62 tests when
there were 80, and 80 when there were 114 -- harmless on its own, but a document
whose easily-checkable facts are wrong earns no trust for the facts that are
harder to check, and this project's whole argument is that its claims are
measured.

Only claims that can be checked mechanically are checked here. Prose is not
tested, and pretending otherwise would be theatre.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_every_make_target_the_readme_mentions_exists():
    """A documented command that does not exist is worse than an undocumented
    one: the reader assumes they got it wrong."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-z][\w-]*):", makefile, re.M))
    mentioned = set(re.findall(r"^make ([a-z][\w-]*)", README, re.M))
    missing = mentioned - targets
    assert not missing, f"README documents make targets that do not exist: {missing}"


def test_every_compose_service_the_readme_runs_exists():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    services = set(re.findall(r"^  ([a-z][\w-]*):$", compose, re.M))
    mentioned = set(re.findall(r"docker compose run --rm ([a-z][\w-]*)", README))
    missing = mentioned - services
    assert not missing, f"README runs compose services that do not exist: {missing}"


def test_every_repository_file_the_readme_points_at_exists():
    """Paths in backticks that look like files, checked. A broken pointer in the
    first document a reader opens is the cheapest possible thing to get right."""
    candidates = set(re.findall(r"`([\w./-]+\.(?:py|md|hocon|yaml|toml|pdf))`", README))
    candidates |= set(re.findall(r"\]\(([\w./-]+\.md)\)", README))
    missing = [name for name in candidates
               if "/" in name and not (ROOT / name).exists()]
    assert not missing, f"README points at files that do not exist: {missing}"


def test_dotenv_is_gitignored():
    """The file this project tells people to paste a key into must never be
    committable. This is the single highest-consequence line in .gitignore."""
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored]


def test_the_example_env_holds_no_real_key():
    """It is committed, so anything that looks like a credential in it is a
    published credential."""
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "AIza" not in example, "that looks like a real Google API key"
    assert "sk-" not in example
    assert "paste-your-key-here" in example


def test_git_does_not_track_a_dotenv():
    """Belt and braces: .gitignore only helps for a file that was never added.
    This fails if one was force-added at some point in the past."""
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    assert ".env" not in tracked
