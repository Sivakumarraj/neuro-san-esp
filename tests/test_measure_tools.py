"""The evaluator agent: measurement from inside neuro-san.

The model call is replaced throughout, so nothing here spends anything. Pinned:
the agent can only measure networks this deployment already knows, a paid run
past the budget is refused before it starts, every figure the agent is handed
comes from a measurement, and the registry is one neuro-san itself accepts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from esp.eval import runner
from esp.eval.tasks import TASKS
from esp.service import measure_tools as tools

ROOT = Path(__file__).resolve().parent.parent
REGISTRIES = ROOT / "registries"


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(tools, "_spent", {"runs": 0})
    monkeypatch.setattr(tools, "_catalog", None)


def answering(correct_for: set[str]):
    """Right on every benchmark question for networks whose HOCON path contains
    one of `correct_for`; wrong everywhere else."""
    answers = {t.question: t.answer for t in TASKS}

    def ask(hocon, question):
        right = any(key in str(hocon) for key in correct_for)
        return (answers.get(question, "?") if right else "no idea",
                {"total_tokens": 1000}, 0.1)
    return ask


# ---------------------------------------------------------- what it can see

def test_it_lists_the_committed_networks_with_their_measured_scores():
    listed = json.loads(tools.ListNetworks().invoke({}, {}))
    assert len(listed["networks"]) == 12
    assert all(n["measured"]["accuracy"] > 0 for n in listed["networks"])
    assert listed["runs_left"] == tools.MAX_RUNS


def test_networks_resolve_by_name_id_or_hash_and_ambiguity_is_refused():
    first = tools.networks()[0]
    short = first.id[:6]
    for text in (first.name, first.id, short):
        found, unknown = tools.resolve([text])
        assert found == [first] and not unknown, text

    # Two networks are mut:reassign_model; the bare operator name is ambiguous.
    found, unknown = tools.resolve(["mut:reassign_model"])
    assert not found and "matches 2" in unknown[0]


def test_a_path_or_a_network_definition_is_never_measured(monkeypatch):
    """A network names Python classes to import; taking one from a chat
    message would be taking code."""
    called = []
    monkeypatch.setattr(runner, "_ask", lambda *a: called.append(1))
    for wanted in (["registries/optimizer.hocon"], ["/etc/passwd"],
                   ['{"tools": [{"class": "os.system"}]}']):
        reply = json.loads(tools.MeasureNetworks().invoke({"networks": wanted}, {}))
        assert "unknown network" in reply["refused"]
    assert not called


# ---------------------------------------------------------- what it spends

def test_a_run_past_the_budget_is_refused_before_anything_is_asked(monkeypatch):
    called = []
    monkeypatch.setattr(runner, "_ask", lambda *a: called.append(1))
    monkeypatch.setattr(tools, "MAX_RUNS", 20)
    two = [c.id for c in tools.networks()[:2]]         # 2 x 17 = 34 runs
    reply = json.loads(tools.MeasureNetworks().invoke({"networks": two}, {}))
    assert "34 paid runs" in reply["refused"]
    assert not called


def test_bad_questions_are_refused_before_anything_is_asked(monkeypatch):
    called = []
    monkeypatch.setattr(runner, "_ask", lambda *a: called.append(1))
    one = [tools.networks()[0].id]
    reply = json.loads(tools.MeasureNetworks().invoke(
        {"networks": one, "questions": '{"question": "no answer given"}'}, {}))
    assert "no answer" in reply["refused"]
    assert not called


# ---------------------------------------------------------- what it reports

def test_it_measures_compares_and_marks_the_front(monkeypatch):
    best, other = tools.networks()[0], tools.networks()[5]
    monkeypatch.setattr(runner, "_ask", answering({Path(best.hocon).stem}))
    sly: dict = {}
    reply = json.loads(tools.MeasureNetworks().invoke(
        {"networks": [best.id[:6], other.id[:6]]}, sly))

    assert reply["questions"] == 17 and reply["suite"] == "meridian"
    by_name = {r["network"]: r for r in reply["reports"]}
    assert by_name[best.name]["accuracy"] == 1.0 and by_name[best.name]["pareto"]
    assert by_name[other.name]["accuracy"] == 0.0 and not by_name[other.name]["pareto"]
    assert set(by_name[best.name]["per_question"].values()) == {"correct"}
    assert by_name[best.name]["measured_before"]["accuracy"] == best.measured["accuracy"]
    # The whole answers are kept for anything downstream, not handed to the model.
    assert len(sly["measurement"]) == 2 and "answer" in sly["measurement"][0]["results"][0]
    assert tools._spent["runs"] == 34


def test_it_measures_on_the_users_own_questions(monkeypatch):
    monkeypatch.setattr(runner, "_ask", lambda h, q: ("It is 42.", {"total_tokens": 10}, 0.1))
    mine = json.dumps({"question": "What is six times seven?", "answer": "42"})
    reply = json.loads(tools.MeasureNetworks().invoke(
        {"networks": [tools.networks()[0].id], "questions": mine}, {}))
    assert reply["suite"] == "custom" and reply["questions"] == 1
    assert reply["reports"][0]["accuracy"] == 1.0


def test_a_network_whose_run_measured_nothing_is_reported_as_such(monkeypatch):
    """No model called is not a score of zero; the row says it was refused."""
    monkeypatch.setattr(runner, "_ask", lambda h, q: ("x", {"total_tokens": 0}, 0.1))
    reply = json.loads(tools.MeasureNetworks().invoke(
        {"networks": [tools.networks()[0].id]}, {}))
    assert "zero tokens" in reply["reports"][0]["error"]
    assert "accuracy" not in reply["reports"][0]


# --------------------------------------------------------- the registries

@pytest.mark.parametrize("name", ["evaluator.hocon", "optimizer.hocon"])
def test_neuro_san_itself_accepts_the_registry(name):
    pytest.importorskip("neuro_san")
    finished = subprocess.run(
        [sys.executable, "-m", "neuro_san.client.hocon_validator_cli",
         str(REGISTRIES / name)],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
        env={**__import__("os").environ, "AGENT_TOOL_PATH": str(ROOT),
             "PYTHONPATH": str(ROOT)})
    assert "Validation passed" in finished.stdout + finished.stderr, (
        finished.stdout[-2000:] + finished.stderr[-2000:])


@pytest.mark.parametrize("name", ["evaluator.hocon", "optimizer.hocon"])
def test_the_registry_runs_on_any_of_the_three_providers(name):
    """Pinned to one model, the agent failed on every deployment holding only
    another provider's key."""
    from pyhocon import ConfigFactory

    from esp.config import provider_for

    config = ConfigFactory.parse_file(str(REGISTRIES / name))
    models = [entry["model_name"] for entry in config["llm_config"]["fallbacks"]]
    assert {provider_for(m) for m in models} == {"openai", "anthropic", "gemini"}


def test_every_class_the_evaluator_names_exists():
    import importlib

    from pyhocon import ConfigFactory

    config = ConfigFactory.parse_file(str(REGISTRIES / "evaluator.hocon"))
    for tool in config["tools"]:
        if "class" in tool:
            module, _, cls = tool["class"].rpartition(".")
            assert hasattr(importlib.import_module(module), cls), tool["class"]


def test_the_deployed_manifest_keeps_the_evaluator_private_by_default():
    body = (REGISTRIES / "manifest.hocon").read_text(encoding="utf-8")
    block = body[body.index('"evaluator.hocon"'):]
    assert '"public": false' in block.split("}")[0]
