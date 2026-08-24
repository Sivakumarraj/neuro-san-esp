"""The free half of ESP has to run on a machine that has never paid for one.

`make offline` is the one command the README hands a reader who has no API key,
and the claim attached to it is the project's central one: that thousands of
candidates can be ranked for nothing. On a fresh clone it exited 1. The live
cache it read is gitignored, so it holds nothing until `make baseline` -- which
needs the key the reader was just promised they would not need.

CI could not catch it, because CI runs `--cache tests/fixtures/cache` and the
README runs `make offline`. A green pipeline for a command nobody is told to
type says nothing about the command everybody is told to type. These tests run
the documented path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "offline_search.py"


def run(cache_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    """The script as a person runs it, with the live cache pointed somewhere
    empty -- a subprocess because the failure was in `main`, not in a helper."""
    environment = {"PATH": "/usr/bin:/bin", "ESP_CACHE": str(cache_dir),
                   "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, str(SCRIPT), "--pool", "50", *extra],
                          cwd=ROOT, env=environment, capture_output=True,
                          text=True, timeout=300)


def test_it_runs_with_no_local_measurements_at_all(tmp_path):
    """The regression itself: an empty live cache used to be fatal."""
    finished = run(tmp_path / "never-used")
    assert finished.returncode == 0, finished.stderr
    assert "Phase C" in finished.stdout


def test_the_fallback_is_announced_rather_than_silent(tmp_path):
    """Falling back is fine. Letting somebody believe they are looking at their
    own measurements when they are looking at the committed ones is not."""
    finished = run(tmp_path / "never-used")
    assert "falling back to the committed ones" in finished.stdout
    assert "tests/fixtures/cache" in finished.stdout
    assert "make baseline" in finished.stdout


def test_it_says_which_cache_it_trained_on(tmp_path):
    finished = run(tmp_path / "never-used")
    assert "Phase B -- training the Predictor on" in finished.stdout
    assert "from tests/fixtures/cache" in finished.stdout


def test_an_explicitly_named_empty_cache_still_fails(tmp_path):
    """A reader who names a cache means that cache. Quietly serving different
    numbers than the ones asked for is the substitution this guards against --
    the fallback is for the default path only."""
    empty = tmp_path / "mine"
    empty.mkdir()
    finished = run(tmp_path / "never-used", "--cache", str(empty))
    assert finished.returncode == 1
    assert "need at least 2 cached seed evaluations" in finished.stderr
    assert "falling back" not in finished.stdout


def test_local_measurements_are_preferred_over_the_committed_ones(tmp_path):
    """The fallback must not shadow a real cache: if the reader has paid for
    their own evaluations, those are what Phase B trains on."""
    mine = tmp_path / "mine"
    mine.mkdir()
    for fixture in sorted((ROOT / "tests" / "fixtures" / "cache").glob("*.json")):
        (mine / fixture.name).write_text(fixture.read_text(encoding="utf-8"),
                                         encoding="utf-8")
    finished = run(mine)
    assert finished.returncode == 0, finished.stderr
    assert "falling back" not in finished.stdout
