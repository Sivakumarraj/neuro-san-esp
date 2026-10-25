"""Failures found by reading the code for a production review, each pinned.

None of these showed up in a run yet. Each is a path the code could take:
a quota error that came back as None, a lease released out from under the next
wake, a cache file that stopped every later run, a page that answered 500 on a
damaged spend file. Each test drives the path the review traced.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from esp.eval import ratelimit
from esp.eval.runner import evaluate
from esp.eval.tasks import TASKS
from esp.genome.definition import Agent, Genome
from esp.genome.seeds import SEEDS
from esp.service import state as state_module
from esp.service.state import Lease, ServiceState

ROOT = Path(__file__).resolve().parent.parent

DAILY_429 = ("429 RESOURCE_EXHAUSTED: Quota exceeded for metric "
             "generate_content_free_tier_requests, limit per day")


# ---------------------------------------------------------------- rate limiter

def test_a_quota_error_is_raised_when_key_swaps_use_up_every_attempt(monkeypatch):
    """Each daily-cap swap spends an attempt. With six keys the last attempt
    ended on a swap and the call returned None, which LangChain turned into
    an AttributeError naming no quota -- so the runner's quota guard missed it
    and graded the question wrong. The provider's 429 has to come out."""
    pytest.importorskip("langchain_google_genai")
    from langchain_google_genai import ChatGoogleGenerativeAI

    keys = [f"AIzaSyTESTKEY{i:02d}xxxxxxxxxxxxxxxxxx" for i in range(8)]
    monkeypatch.setenv("GOOGLE_API_KEYS", ",".join(keys))
    ratelimit.reload_keyring()
    calls = []

    def refused(self, *args, **kwargs):
        calls.append(1)
        raise RuntimeError(DAILY_429)

    monkeypatch.setattr(ChatGoogleGenerativeAI, "_generate", refused)
    monkeypatch.setattr(ChatGoogleGenerativeAI, "_esp_rate_limited", False, raising=False)
    try:
        ratelimit.install(rpm=10_000)
        client = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=keys[0])
        with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
            ChatGoogleGenerativeAI._generate(client, [])
        assert len(calls) == ratelimit.MAX_RETRIES
    finally:
        monkeypatch.delenv("GOOGLE_API_KEYS")
        ratelimit.reload_keyring()


# ------------------------------------------------------------------ the lease

def test_an_expired_holder_cannot_release_the_next_wakes_lease(tmp_path):
    """A wake can outlive its lease. When it finished, it deleted the lease
    file whoever held it, and a third wake started beside the second."""
    first = Lease(directory=tmp_path, seconds=0.05)
    assert first.acquire("wake-1")
    time.sleep(0.1)
    second = Lease(directory=tmp_path, seconds=0.05)
    assert second.acquire("wake-2")

    first.release()

    assert (second.holder() or {}).get("owner") == "wake-2"
    assert Lease(directory=tmp_path, seconds=3600).acquire("wake-3") is False
    second.release()
    assert Lease(directory=tmp_path, seconds=3600).acquire("wake-3") is True


def test_a_lease_that_was_never_taken_releases_nothing(tmp_path):
    holder = Lease(directory=tmp_path)
    assert holder.acquire("wake-1")
    Lease(directory=tmp_path).release()
    assert (holder.holder() or {}).get("owner") == "wake-1"


def test_a_wake_reads_the_population_only_once_it_holds_the_lease(tmp_path, monkeypatch):
    """Read before the lease, a wake that waited out the last holder worked
    from the population as it stood before that holder's final save, and
    wrote the stale copy back over it."""
    from esp.service import optimizer

    order = []
    real_acquire = Lease.acquire

    def acquire(self, owner):
        order.append("acquire")
        return real_acquire(self, owner)

    def load(cls, directory=None):
        order.append("load")
        raise RuntimeError("stop here")

    monkeypatch.setattr(state_module, "STATE_DIR", tmp_path)
    monkeypatch.setattr(Lease, "acquire", acquire)
    monkeypatch.setattr(ServiceState, "load", classmethod(load))
    with pytest.raises(RuntimeError, match="stop here"):
        optimizer.wake()
    assert order == ["acquire", "load"]


def test_an_exhausted_model_stays_retired_across_utc_midnight(monkeypatch):
    """The expiry is an hour; the filing key is the day. Read from today's
    record alone, a model marked at 23:50 was fresh again at 00:10."""
    state = ServiceState()
    monkeypatch.setattr(state_module, "_utcnow",
                        lambda: datetime(2026, 10, 25, 23, 50, tzinfo=UTC))
    state.mark_exhausted("gemini-3.5-flash-lite")
    monkeypatch.setattr(state_module, "_utcnow",
                        lambda: datetime(2026, 10, 26, 0, 10, tzinfo=UTC))
    assert state.is_exhausted("gemini-3.5-flash-lite")
    assert state.exhausted_now() == ["gemini-3.5-flash-lite"]
    monkeypatch.setattr(state_module, "_utcnow",
                        lambda: datetime(2026, 10, 26, 0, 51, tzinfo=UTC))
    assert not state.is_exhausted("gemini-3.5-flash-lite")


# ------------------------------------------------------------------ the cache

def test_a_damaged_cache_entry_is_a_miss_not_a_crash(tmp_path, monkeypatch):
    """A half-written entry used to raise on every later run at that genome."""
    monkeypatch.setattr("esp.eval.runner.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("esp.eval.runner.NETWORK_DIR", tmp_path / "networks")
    monkeypatch.setattr("esp.eval.runner._ask",
                        lambda hocon, question: ("Pickering", {"total_tokens": 100}, 0.1))
    genome = SEEDS["solo"]()
    tasks = TASKS[:2]

    from esp.eval.runner import cache_path_for
    path = cache_path_for(genome, tasks)
    path.parent.mkdir(parents=True)
    path.write_text('{"genome_hash": "', encoding="utf-8")

    evaluation = evaluate(genome, tasks=tasks)

    assert not evaluation.from_cache
    assert json.loads(path.read_text(encoding="utf-8"))["genome_hash"] == genome.genome_hash()
    assert not list(path.parent.glob(".*.tmp")), "a scratch file was left behind"


def test_rendered_networks_never_land_in_the_operators_directory(tmp_path):
    """ESP_NETWORKS is where an operator puts networks to be measured. The
    renderer wrote there too, so the catalog listed each committed network
    twice: once as itself and once as its rendered file."""
    script = ("import sys; sys.path.insert(0, '.');"
              "from esp.measure import catalog;"
              "found = catalog(sys.argv[1]);"
              "print(sum(c.id.startswith('file:') for c in found))")
    environment = {**os.environ, "ESP_NETWORKS": str(tmp_path), "ESP_NO_DOTENV": "1",
                   "ESP_RENDERED_NETWORKS": str(tmp_path / "rendered")}
    finished = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=ROOT,
                              env=environment, capture_output=True, text=True, timeout=300)
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "0"


# ------------------------------------------------------------------ rendering

def test_an_agent_name_cannot_write_its_own_hocon_keys():
    """Names were pasted into the HOCON between bare quotes; a name with a quote
    in it closed the string and the rest became keys of the agent."""
    pyhocon = pytest.importorskip("pyhocon")
    name = 'Helper", "class": "os.system'
    genome = Genome(top="Boss", agents={
        "Boss": Agent("Boss", "route", "front", tools=[name]),
        name: Agent(name, "help", "helper"),
    })
    parsed = pyhocon.ConfigFactory.parse_string(genome.to_hocon())
    tools = parsed["tools"]
    assert [tool["name"] for tool in tools] == ["Boss", name]
    assert all("class" not in tool for tool in tools)
    assert tools[0]["tools"] == [name]
