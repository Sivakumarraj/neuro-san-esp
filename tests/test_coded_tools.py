"""The agent's only way into the optimiser.

What is being pinned here is mostly what the tool does *not* do. An autonomous
process that wakes hourly with no human in the loop is dangerous in proportion
to what its language model is allowed to decide, so the model is allowed to
decide nothing: it cannot choose a candidate, cannot spend budget, and cannot
reach anything that deletes or sends. The tool hands it a finished measurement
and a yes/no about whether that measurement is worth mentioning.
"""

from __future__ import annotations

import json

import pytest

from esp.service import coded_tools
from esp.service.optimizer import WakeReport
from esp.service.state import Evaluated, ServiceState


def record(hash_="a" * 16, fitness=0.5) -> Evaluated:
    return Evaluated(hash_, "seed:solo", fitness, 0.82, 300_000, 2, 2, 0,
                     "2026-08-22T00:00:00+00:00", "gemini-3.5-flash-lite")


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Run the tool without running a wake -- a real one costs provider budget."""
    state = ServiceState()

    def install(report: WakeReport, population: list[Evaluated] | None = None):
        state.evaluated = list(population or [])
        state.wakes = 7
        monkeypatch.setattr(ServiceState, "load",
                            classmethod(lambda cls, directory=None: state))
        monkeypatch.setattr(coded_tools, "wake", lambda s: report)
        return state

    return install


def test_the_report_is_json_the_agent_can_read(stub):
    stub(WakeReport(acquired=True, evaluated=2, generation=3,
                    best_fitness=0.78, improved=True, note="better"),
         [record(fitness=0.78)])

    payload = json.loads(coded_tools.RunWake().invoke({}, {}))
    assert payload["evaluated_this_wake"] == 2
    assert payload["improved"] is True
    assert payload["best"]["fitness"] == 0.78


def test_a_declined_lease_is_reported_not_raised(stub):
    """A scheduler that overlaps a running wake is normal. The tool must return
    something the agent can act on rather than failing the interaction."""
    stub(WakeReport(acquired=False, note="another wake holds the lease"))

    payload = json.loads(coded_tools.RunWake().invoke({}, {}))
    assert payload["acquired"] is False
    assert "lease" in payload["note"]


def test_an_empty_population_still_returns_a_readable_report(stub):
    """The first wake of a new deployment has nothing measured yet, and a
    KeyError here would take the whole scheduled interaction down."""
    stub(WakeReport(acquired=True, note="nothing yet"), [])

    payload = json.loads(coded_tools.RunWake().invoke({}, {}))
    assert "best" not in payload
    assert payload["population"] == 0


def test_the_wake_number_reaches_the_bulletin_board(stub):
    stub(WakeReport(acquired=True), [record()])
    sly_data: dict = {}
    coded_tools.RunWake().invoke({}, sly_data)
    assert sly_data["optimizer_wake"] == 7


def test_exhaustion_is_visible_to_the_agent(stub):
    """Not so the model can act on it -- so an operator reading the transcript
    can see why a wake did nothing."""
    stub(WakeReport(acquired=True, stopped_because="provider budget exhausted",
                    exhausted=["gemini-3.1-flash-lite"]), [record()])

    payload = json.loads(coded_tools.RunWake().invoke({}, {}))
    assert payload["exhausted_today"] == ["gemini-3.1-flash-lite"]
    assert payload["stopped_because"] == "provider budget exhausted"


# ------------------------------------------------------- what it cannot do


def test_the_tool_surface_is_read_only():
    """No delete, no send, no post. The agent has exactly two tools and this is
    the one with effects: it evaluates and it writes state. Anything that
    reaches outside the process would be reachable by an unattended hourly
    process with a language model deciding when to use it."""
    surface = {name for name in dir(coded_tools) if not name.startswith("_")}
    forbidden = {"send", "post", "delete", "email", "notify", "publish"}
    assert not (surface & forbidden)


def test_the_agent_cannot_choose_what_to_evaluate():
    """invoke() takes args and ignores them. If the model could name a
    candidate, it could spend the day's budget on something the surrogate never
    ranked -- and the search would no longer be the thing being measured."""
    import inspect

    source = inspect.getsource(coded_tools.RunWake.invoke)
    body = source.split("\n", 1)[1]
    assert "args[" not in body and "args.get" not in body
