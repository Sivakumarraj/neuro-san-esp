"""The service's population has to reach the report, or running it is pointless.

`history.json` -- which every figure and both PDFs read -- was written only by
the batch run. The service wrote its population somewhere else and nothing
joined the two, so an optimiser could accumulate for weeks while the report
still showed the last afternoon the batch script managed to finish.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from service_report import to_history, write

from esp.service.state import Evaluated, ServiceState


def record(hash_="a" * 16, fitness=0.5, tokens=300_000, agents=2,
           accuracy=0.82) -> Evaluated:
    return Evaluated(hash_, "seed:solo", fitness, accuracy, tokens, agents, 2, 0,
                     "2026-08-22T00:00:00+00:00", "gemini-3.5-flash-lite")


@pytest.fixture
def populated(tmp_path):
    state = ServiceState()
    state.wakes = 4
    state.add(record("a" * 16, 0.78, 278_532, 4))
    state.add(record("b" * 16, 0.76, 396_378, 1))
    state.save(tmp_path)
    return tmp_path


def test_the_measurements_reach_the_report(populated, tmp_path):
    path = write(populated, tmp_path / "out")
    history = json.loads(path.read_text(encoding="utf-8"))
    assert history["real_evaluations"] == 2
    assert {r["genome_hash"] for r in history["records"]} == {"a" * 16, "b" * 16}


def test_the_wake_count_is_carried_through(populated, tmp_path):
    """So a reader can tell four wakes that found two things from two wakes that
    found two things. The second is a much healthier service."""
    history = json.loads(write(populated, tmp_path / "out").read_text())
    assert history["wakes"] == 4


def test_a_surrogate_search_is_not_claimed(populated, tmp_path):
    """The report reads these fields to decide whether to say a search ran. A
    service wake does not record how many candidates the surrogate scored, so
    filling them in would make the report claim a search nobody can point to."""
    history = json.loads(write(populated, tmp_path / "out").read_text())
    assert history["surrogate_evaluations"] == 0
    assert history["surrogate_quality"] == []
    assert all(r["predicted"] is None for r in history["records"])


def test_an_empty_population_refuses_rather_than_reporting_nothing(tmp_path):
    """A report of zero measurements is worse than no report: it looks like a
    result."""
    ServiceState().save(tmp_path)
    with pytest.raises(SystemExit):
        write(tmp_path, tmp_path / "out")


def test_the_pareto_front_is_computed_not_copied(populated, tmp_path):
    """Both records here are non-dominated -- same accuracy, and each is better
    than the other on one of tokens or agents."""
    history = json.loads(write(populated, tmp_path / "out").read_text())
    assert len(history["pareto"]) == 2


def test_a_dominated_candidate_is_left_off_the_front(tmp_path):
    state = ServiceState()
    state.add(record("a" * 16, 0.78, 200_000, 2, accuracy=0.90))
    state.add(record("b" * 16, 0.40, 900_000, 9, accuracy=0.50))
    state.save(tmp_path)
    history = json.loads(write(tmp_path, tmp_path / "out").read_text())
    assert [r["genome_hash"] for r in history["pareto"]] == ["a" * 16]


def test_the_shape_matches_what_the_report_expects(populated, tmp_path):
    """The batch run's history.json is the contract. A missing key here is a
    KeyError in the PDF builder, which is the last place anyone wants one."""
    history = json.loads(write(populated, tmp_path / "out").read_text())
    for key in ("seed", "weights", "real_evaluations", "surrogate_evaluations",
                "surrogate_quality", "records", "pareto"):
        assert key in history, f"the report reads {key!r} and it is missing"


def test_to_history_preserves_every_measurement(populated):
    state = ServiceState.load(populated)
    history = to_history(state)
    assert len(history.records) == len(state.evaluated)
    assert history.records[0].tokens == state.evaluated[0].tokens
