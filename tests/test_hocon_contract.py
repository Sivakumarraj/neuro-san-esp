"""The registries are the service. These tests pin what they must say.

neuro-san's validator checks that a HOCON file is well formed. It cannot know
that the optimiser has to be event-invoked, that neither agent may be public by
default, or that a coded tool's class path still points at a class after a
rename. Each of those fails silently in production: a mistyped class is an
agent that errors on every call, and a public optimiser is an endpoint anyone
can use to empty the day's provider budget.

Parsed with pyhocon, the way neuro-san parses them, rather than read as text.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pyhocon import ConfigFactory

ROOT = Path(__file__).resolve().parent.parent
REGISTRIES = ROOT / "registries"
NETWORKS = ("optimizer.hocon", "evaluator.hocon")


def load(name: str):
    return ConfigFactory.parse_file(str(REGISTRIES / name))


def manifest_entries(monkeypatch) -> dict:
    for variable in ("ESP_OPTIMIZER_PUBLIC", "ESP_EVALUATOR_PUBLIC", "ESP_OPTIMIZER_CRON"):
        monkeypatch.delenv(variable, raising=False)
    manifest = load("manifest.hocon")
    return {str(key).strip('"'): value for key, value in manifest.items()}


def test_every_served_network_is_in_the_manifest(monkeypatch):
    entries = manifest_entries(monkeypatch)
    for name in NETWORKS:
        assert entries[name]["serve"] is True


def test_nothing_that_spends_budget_is_public_by_default(monkeypatch):
    """Both agents spend real provider calls on request. Opening either is an
    operator's explicit choice through an environment variable."""
    entries = manifest_entries(monkeypatch)
    assert entries["optimizer.hocon"]["public"] is False
    assert entries["evaluator.hocon"]["public"] is False


def test_the_optimizer_wakes_on_neuro_sans_own_schedule(monkeypatch):
    interaction = manifest_entries(monkeypatch)["optimizer.hocon"]["periodic"]["interactions"][0]
    assert interaction["enable"] is True
    assert interaction["cron_schedule"] == "0 * * * *"


def test_the_cron_override_reaches_the_schedule(monkeypatch):
    """`make verify` shortens the cron to one minute through this variable. If
    the override stopped applying, verification would wait an hour."""
    monkeypatch.setenv("ESP_OPTIMIZER_CRON", "* * * * *")
    manifest = load("manifest.hocon")
    entry = {str(k).strip('"'): v for k, v in manifest.items()}["optimizer.hocon"]
    assert entry["periodic"]["interactions"][0]["cron_schedule"] == "* * * * *"


def test_the_optimizer_front_man_is_event_invoked():
    """neuro-san's periodic runner only starts an agent declared as an event.
    Without it the schedule fires into nothing."""
    front = load("optimizer.hocon")["tools"][0]
    assert front["name"] == "Optimizer"
    assert front["function"]["invocation"] == "event"
    assert front["tools"] == ["RunWake", "ReportFinding"]


def test_the_optimizer_has_no_tool_that_deletes_or_sends():
    """Its instructions promise the operator this. The promise has to be true of
    the tool list, not only of the prompt."""
    names = {tool["name"] for tool in load("optimizer.hocon")["tools"]}
    assert names == {"Optimizer", "RunWake", "ReportFinding"}


def test_the_evaluator_front_man_measures_and_lists_only():
    front = load("evaluator.hocon")["tools"][0]
    assert front["name"] == "Evaluator"
    assert front["tools"] == ["ListNetworks", "MeasureNetworks"]


@pytest.mark.parametrize("name", NETWORKS)
def test_every_tool_reference_is_defined(name):
    """A name in a `tools` list that no entry defines is a call that fails
    inside the agent, at run time, on the first question that needs it."""
    tools = load(name)["tools"]
    defined = {tool["name"] for tool in tools}
    for tool in tools:
        for reference in tool.get("tools", []):
            assert reference in defined, f"{tool['name']} calls undefined {reference}"


@pytest.mark.parametrize("name", NETWORKS)
def test_every_coded_tool_class_imports(name):
    """The class path is a string until neuro-san imports it on the first call.
    A rename that misses the HOCON would pass every other test in this suite."""
    for tool in load(name)["tools"]:
        path = tool.get("class", None)
        if not path:
            continue
        module, _, attribute = path.rpartition(".")
        assert hasattr(importlib.import_module(module), attribute), path


@pytest.mark.parametrize("name", NETWORKS)
def test_the_front_man_can_run_on_any_supported_provider(name):
    """One fallback per provider, so a deployment holding any one key works."""
    from esp.config import provider_for
    fallbacks = load(name)["llm_config"]["fallbacks"]
    providers = {provider_for(entry["model_name"]) for entry in fallbacks}
    assert {"openai", "anthropic", "gemini"} <= providers
