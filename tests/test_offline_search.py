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

import re
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


# --------------------------------------- what Phase C is allowed to propose

def test_the_pool_holds_no_duplicates(tmp_path):
    """Phase C had no dedupe at all, so the same genome filled four of the five
    places in the printed top five. A ranking that lists one network repeatedly
    is not a ranking, and the elite it feeds would buy the same candidate more
    than once."""
    finished = run(tmp_path, "--pool", "200", "--top", "20")
    assert finished.returncode == 0, finished.stderr

    listed = re.findall(r"^\s+[+-]\d\.\d+\s+([0-9a-f]{16})", finished.stdout,
                        re.M)
    assert listed, f"no candidates listed: {finished.stdout[-400:]}"
    assert len(listed) == len(set(listed)), (
        f"the same genome appears more than once: {listed}")


def test_an_already_measured_genome_is_never_proposed(tmp_path):
    """Phase C exists to find candidates worth paying for, and one that has
    been measured is not one. Proposing it spends an elite slot on a known
    answer and flatters the Predictor, which then "discovers" a network it was
    trained on. The measured champion `3bf9c008d880c3fc` was topping the list.
    """
    from esp.eval import measurements

    finished = run(tmp_path, "--pool", "200", "--top", "20")
    assert finished.returncode == 0, finished.stderr

    measured = {m.genome_hash for m in measurements.load()}
    assert measured, "no committed measurements to exclude"

    listed = set(re.findall(r"^\s+[+-]\d\.\d+\s+([0-9a-f]{16})",
                            finished.stdout, re.M))
    overlap = listed & measured
    assert not overlap, f"already-measured genomes proposed: {sorted(overlap)}"


def test_it_says_when_the_reachable_space_is_smaller_than_asked_for(tmp_path):
    """With 12 genomes and a strict validity gate there are only a few hundred
    distinct one-step mutants. Asking for 2,000 and quietly delivering 388
    would make every ratio quoted over the pool wrong."""
    finished = run(tmp_path, "--pool", "5000", "--top", "3")
    assert finished.returncode == 0, finished.stderr
    assert "reachable space yielded" in finished.stdout, finished.stdout[-600:]
    assert "describe what was" in finished.stdout


def test_breeding_and_ranking_are_timed_separately(tmp_path):
    """"Ranking is free" is the claim the whole method rests on, and it is
    about the Predictor. Breeding mutants is ordinary compute and, once
    duplicates are filtered, most of the clock. Quoting the total as scoring
    time would overstate the one number that matters."""
    finished = run(tmp_path, "--pool", "100")
    assert finished.returncode == 0, finished.stderr
    assert "bred in" in finished.stdout
    assert "scored in" in finished.stdout
    assert "to rank" in finished.stdout


def test_it_terminates_when_the_space_is_exhausted(tmp_path):
    """Unbounded, the loop spins forever once every reachable mutant has been
    seen. The subprocess timeout in `run` is the assertion."""
    finished = run(tmp_path, "--pool", "3000", "--top", "1")
    assert finished.returncode == 0, finished.stderr
