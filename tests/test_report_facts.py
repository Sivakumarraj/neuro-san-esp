"""The reports read the population; they do not restate it.

Three PDF generators each held the population's figures as literals -- how many
networks were measured, what the best one scored, what the designer's shape
cost -- so a new measurement meant editing prose in three files and the prose
fell behind. A run added a twelfth network and the primer went on describing
eleven, in a document whose entire argument is that its numbers were measured.

These pin the derivation, and pin the absence of the literals that made the
drift possible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from esp.eval import measurements
from esp.report.facts import NUMBERS, facts, spelled

ROOT = Path(__file__).resolve().parent.parent
HISTORY = json.loads(
    (ROOT / "results" / "history.json").read_text(encoding="utf-8"))
REPORTS = sorted((ROOT / "esp" / "report").glob("*.py"))


@pytest.fixture
def derived():
    return facts()


def test_the_counts_match_the_committed_measurements(derived):
    assert derived.measured == HISTORY["real_evaluations"]
    assert derived.measured == len(HISTORY["records"])
    assert derived.seeds + derived.evolved == derived.measured


def test_the_best_is_the_best_in_the_published_history(derived):
    published = max(HISTORY["records"], key=lambda r: r["fitness"])
    assert derived.best.genome_hash == published["genome_hash"]
    assert derived.best.fitness == pytest.approx(published["fitness"], abs=5e-4)


def test_task_runs_are_the_population_times_the_task_set(derived):
    assert derived.task_runs == derived.measured * derived.tasks
    assert 0 <= derived.unfinished <= derived.task_runs


def test_the_accuracy_gain_is_computed_against_the_designer(derived):
    assert derived.designer is not None, "no designer baseline to compare against"
    expected = (derived.best.accuracy - derived.designer.accuracy) * 100
    assert derived.accuracy_gain_points() == pytest.approx(expected)


def test_the_cheapest_winner_beats_every_seed(derived):
    """The other end of the trade-off. Quoting only the highest fitness hides
    the Pareto front, which is the honest shape of the result."""
    if derived.cheapest_winner is None:
        pytest.skip("only one network beats the seeds")
    seed_best = max(m.accuracy for m in measurements.load() if not m.evolved)
    assert derived.cheapest_winner.accuracy > seed_best
    assert derived.cheapest_winner.evolved


def test_a_count_reads_as_a_word_until_it_stops_helping():
    assert spelled(12) == "twelve"
    assert spelled(max(NUMBERS) + 1) == str(max(NUMBERS) + 1)


def test_no_report_hardcodes_a_measured_token_count():
    """The literals that caused the drift.

    Narrowed to numbers that *are* current measurements: report prose legitimately
    quotes round illustrative figures ("a network costing 300,000") and the
    historical constant behind the saturated-cost bug. What it must not contain is
    a real candidate's token count, because that is a fact about the population
    and the next wake changes the population.
    """
    measured = {m.tokens for m in measurements.load()}
    grouped = {f"{value:,}" for value in measured} | {str(v) for v in measured}

    offences: list[str] = []
    for path in REPORTS:
        if path.name == "facts.py":
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(literal in line for literal in grouped):
                offences.append(f"{path.name}:{number}: {line.strip()[:70]}")
    assert not offences, (
        "a measured token count is typed into report prose, and the next wake "
        "will make it wrong:\n  " + "\n  ".join(offences))


def test_no_report_states_the_population_size_in_words():
    """`eleven measured networks` in prose is the same bug in a different
    font. The count comes from `spelled(facts().measured)`."""
    # The number has to be quantifying the noun, within a couple of words --
    # not merely sharing a sentence with it. "One command writes the
    # best-measured design" is prose, not a population claim.
    pattern = re.compile(
        r'\b(' + "|".join(NUMBERS.values()) + r')\b(?:\s+\w+){0,2}\s+'
        r'\b(real evaluations|network designs|different team designs|'
        r'real tests|measurements|measured networks)\b', re.IGNORECASE)
    offences: list[str] = []
    for path in REPORTS:
        if path.name == "facts.py":
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                offences.append(f"{path.name}:{number}: {line.strip()[:70]}")
    assert not offences, (
        "population size written into report prose:\n  " + "\n  ".join(offences))
