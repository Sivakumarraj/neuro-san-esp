"""The optimiser as a service: it is expected to be interrupted.

A batch run plans a fixed number of generations and fails when the provider
stops it early. A service plans nothing -- it spends what today allows, writes
down where it got to, and carries on tomorrow. These tests pin that behaviour,
because it is the difference between a script that dies at the daily cap and
one that treats the cap as its rhythm.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from esp.service.state import Evaluated, Lease, ServiceState


def make(hash_: str = "a" * 16, fitness: float = 0.5) -> Evaluated:
    return Evaluated(hash_, "seed:solo", fitness, 0.82, 300_000, 2, 2, 0,
                     "2026-08-22T00:00:00+00:00", "gemini-3.5-flash-lite")


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path


# ------------------------------------------------------------------ survival


def test_a_fresh_service_starts_empty(state_dir):
    state = ServiceState.load(state_dir)
    assert state.generation == 0
    assert state.evaluated == []


def test_population_survives_a_restart(state_dir):
    """The whole premise: an evaluation paid for today must still be there
    tomorrow, or the service can never accumulate anything."""
    state = ServiceState.load(state_dir)
    state.add(make())
    state.save(state_dir)

    assert len(ServiceState.load(state_dir).evaluated) == 1


def test_saving_is_atomic(state_dir):
    """Written to a sibling and moved into place. A service killed mid-write
    must not come back to a truncated population."""
    state = ServiceState.load(state_dir)
    state.add(make())
    state.save(state_dir)
    assert not list(state_dir.glob("*.tmp"))
    assert (state_dir / "state.json").exists()


# -------------------------------------------------------------------- budget


def test_spend_is_tracked_per_model(state_dir):
    """One global counter would let an exhausted model stall a service that
    still has budget on another."""
    state = ServiceState.load(state_dir)
    state.record_spend("gemini-3.5-flash-lite", 165)
    state.record_spend("gemini-3.5-flash", 20)
    assert state.spent_today("gemini-3.5-flash-lite") == 165
    assert state.spent_today("gemini-3.5-flash") == 20


def test_exhaustion_is_recorded_per_day_so_the_service_recovers(state_dir):
    """A model retired permanently would leave the service dead after its first
    bad day, which is the opposite of running unattended."""
    state = ServiceState.load(state_dir)
    state.mark_exhausted("gemini-3.5-flash-lite")
    assert state.is_exhausted("gemini-3.5-flash-lite")

    # Same state, a different day: the model is available again.
    state.exhausted = {"1999-01-01": ["gemini-3.5-flash-lite"]}
    assert not state.is_exhausted("gemini-3.5-flash-lite")


def test_usable_skips_only_what_is_spent_today(state_dir):
    state = ServiceState.load(state_dir)
    state.mark_exhausted("b")
    assert state.usable(["a", "b", "c"]) == ["a", "c"]


def test_today_is_utc(state_dir):
    """The provider meters in UTC and the service may move host. A local-time
    boundary would double-spend or skip a day when it does.

    This test asserted `date.today()` -- local time -- under that docstring and
    that name. It agreed with the code, which also called `date.today()`, so
    both were wrong in the same direction and it passed everywhere: on a UTC
    runner because the two coincide, and on the IST machine this was written on
    because the bug and its test moved together. A test that pins the defect it
    is named after is worse than no test, because it certifies it.
    """
    state = ServiceState.load(state_dir)
    state.record_spend("m", 1)
    assert datetime.now(UTC).date().isoformat() in state.spend


# --------------------------------------------------------------------- lease


def test_a_second_wake_cannot_start_while_one_is_running(state_dir):
    """A scheduler firing hourly will overlap a wake that is still evaluating.
    Two optimisers would spend the same daily budget twice."""
    first, second = Lease(state_dir), Lease(state_dir)
    assert first.acquire("wake-1")
    assert not second.acquire("wake-2")


def test_releasing_frees_the_lease(state_dir):
    lease = Lease(state_dir)
    lease.acquire("wake-1")
    lease.release()
    assert Lease(state_dir).acquire("wake-2")


def test_a_stale_lease_expires(state_dir):
    """A wake killed without releasing must not lock the service out forever."""
    lease = Lease(state_dir, seconds=0.0)
    lease.acquire("dead-wake")
    assert Lease(state_dir, seconds=0.0).acquire("new-wake")


def test_a_corrupt_lease_is_a_free_lease(state_dir):
    """Better to risk one overlapped wake than to wedge the service until
    somebody notices and deletes a file by hand."""
    lease = Lease(state_dir)
    lease.acquire("wake-1")
    lease.path.write_text("{ not json", encoding="utf-8")
    assert Lease(state_dir).acquire("wake-2")


# ---------------------------------------------------------------- population


def test_best_is_the_highest_fitness(state_dir):
    state = ServiceState.load(state_dir)
    state.add(make("a" * 16, 0.10))
    state.add(make("b" * 16, 0.90))
    state.add(make("c" * 16, 0.50))
    assert state.best().fitness == 0.90


def test_seen_prevents_paying_for_the_same_genome_twice(state_dir):
    state = ServiceState.load(state_dir)
    state.add(make("a" * 16))
    assert "a" * 16 in state.seen()


def test_best_of_an_empty_population_is_none(state_dir):
    assert ServiceState.load(state_dir).best() is None


# ------------------------------------------------- which model actually ran out


REAL_429 = (
    "QuotaExhausted: 17 of 17 tasks hit a provider quota -- refusing to cache a "
    "score the environment caused, not the topology. First: ResourceExhausted: "
    "429 You exceeded your current quota. quota_metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "quota_id: GenerateRequestsPerDayPerProjectPerModel-FreeTier, "
    "quota_dimensions { key: \"model\" value: \"gemini-3.1-flash-lite\" } "
    "quota_value: 500"
)


def test_the_exhausted_model_is_read_out_of_the_provider_message():
    """The 429 says which name ran out. Believe it.

    This is the bug this test exists for: the first version matched the message
    against the failover ladder, and the model that actually failed was the
    network default, which is not a ladder member. `exhausted_today` came back
    empty after a wake that had plainly hit the cap, so the next wake would have
    spent its first calls on the same dead model -- every hour, all day.
    """
    from esp.eval import failover

    assert failover.models_named(REAL_429) == ["gemini-3.1-flash-lite"]


def test_a_model_outside_the_ladder_is_still_recorded(state_dir):
    from esp.genome.seeds import SEEDS
    from esp.service.optimizer import _exhausted_models

    genome = next(iter(SEEDS.values()))()
    assert _exhausted_models(genome, REAL_429) == ["gemini-3.1-flash-lite"]


def test_an_unreadable_message_falls_back_to_the_candidate_s_own_models(state_dir):
    """Retiring a model that still had budget costs one wake. Retrying a dead
    one costs the rest of the day, so guess towards stopping."""
    from esp.genome.seeds import SEEDS
    from esp.service.optimizer import _exhausted_models

    genome = next(iter(SEEDS.values()))()
    got = _exhausted_models(genome, "connection reset by peer")
    assert genome.default_model in got


def test_priming_stops_a_fresh_process_rediscovering_this_morning(state_dir):
    """State is only worth writing if the next process reads it back."""
    from esp.eval import failover

    failover.reset()
    try:
        failover.prime(["gemini-3.5-flash-lite"])
        assert failover.substitute("gemini-3.5-flash-lite") != "gemini-3.5-flash-lite"
    finally:
        failover.reset()


# ------------------------------------------------- when does budget come back


def test_exhaustion_expires_by_elapsed_time_not_by_calendar_day(state_dir,
                                                                monkeypatch):
    """The bug this replaces, observed directly.

    Exhaustion used to be keyed to the UTC calendar day. The provider resets its
    free tier on its own boundary -- Pacific midnight for Google -- and those are
    eight hours apart. State said gemini-3.1-flash-lite was spent while a probe
    of that same model, inside the same UTC day, answered fine. The service
    would have idled for the rest of the day on a budget it already had back.
    """
    from esp.service import state as state_module

    state = ServiceState()
    state.mark_exhausted("gemini-3.1-flash-lite")
    assert state.is_exhausted("gemini-3.1-flash-lite")

    monkeypatch.setattr(state_module, "RETRY_EXHAUSTED_AFTER", 0.0)
    assert not state.is_exhausted("gemini-3.1-flash-lite"), (
        "a model must be retried once the window passes, not held until midnight")


def test_a_freshly_exhausted_model_is_not_retried_immediately(state_dir):
    """The other half. Retrying every wake would turn a hard daily cap into a
    steady drip of 429s and make the logs useless."""
    state = ServiceState()
    state.mark_exhausted("gemini-3.5-flash")
    assert state.is_exhausted("gemini-3.5-flash")
    assert state.exhausted_now() == ["gemini-3.5-flash"]


def test_old_state_written_as_a_bare_list_is_read_as_expired(state_dir):
    """State on disk from before this change carries no timestamp, so it can
    only mean 'not before tomorrow' -- which is the wrong answer. Read it as
    already expired: the service retries once, costing at most one 429."""
    import json

    (state_dir / "state.json").write_text(json.dumps({
        "generation": 1, "evaluated": [], "spend": {},
        "exhausted": {date.today().isoformat(): ["gemini-3.1-flash-lite"]},
        "last_wake": "", "last_spoke": "", "wakes": 3,
    }), encoding="utf-8")

    state = ServiceState.load(state_dir)
    assert not state.is_exhausted("gemini-3.1-flash-lite")
    assert state.exhausted_now() == []


def test_an_unreadable_timestamp_is_not_a_reason_to_stay_idle(state_dir):
    state = ServiceState()
    state.exhausted[date.today().isoformat()] = {"m": "not-a-timestamp"}
    assert not state.is_exhausted("m")


def test_exhaustion_survives_a_save_and_reload(state_dir):
    state = ServiceState()
    state.mark_exhausted("gemini-3.5-flash")
    state.save(state_dir)
    assert ServiceState.load(state_dir).is_exhausted("gemini-3.5-flash")


# ------------------------------------------ serving an evolved winner


def test_a_genome_round_trips_through_canonical_form():
    """The reconstruction must hash identically, or serving it would quietly
    hand a visitor a network that is not the one that earned the score."""
    from esp.genome.definition import Genome
    from esp.genome.seeds import SEEDS

    for build in SEEDS.values():
        genome = build()
        assert Genome.from_canonical(genome.canonical()).genome_hash() \
            == genome.genome_hash()


def test_the_service_records_the_candidate_not_just_its_score(state_dir):
    """The limitation this closes. Recording a hash and a score meant an
    evolved winner could be measured and then never served again -- a hash
    cannot be turned back into a network, so only seeds could be resurrected,
    and seeds are precisely the thing the search is trying to beat."""
    from esp.genome.seeds import SEEDS

    genome = SEEDS["designer_shaped"]()
    state = ServiceState()
    state.add(Evaluated(
        genome_hash=genome.genome_hash(), origin="split_agent", fitness=0.9,
        accuracy=0.9, tokens=100, agents=4, depth=2, generation=1,
        measured_at="2026-08-23T00:00:00+00:00", model="gemini-3.1-flash-lite",
        genome=genome.canonical()))
    state.save(state_dir)

    reloaded = ServiceState.load(state_dir).best()
    assert reloaded.genome is not None
    from esp.genome.definition import Genome

    assert Genome.from_canonical(reloaded.genome).genome_hash() \
        == genome.genome_hash()


def test_state_written_before_genomes_were_stored_still_loads(state_dir):
    """Old state has no genome field. It must keep working and keep its old
    behaviour rather than crashing a running service on upgrade."""
    import json

    (state_dir / "state.json").write_text(json.dumps({
        "generation": 1, "evaluated": [{
            "genome_hash": "a" * 16, "origin": "seed:solo", "fitness": 0.5,
            "accuracy": 0.82, "tokens": 300, "agents": 1, "depth": 1,
            "generation": 0, "measured_at": "2026-08-22T00:00:00+00:00",
            "model": "gemini-3.1-flash-lite"}],
        "spend": {}, "exhausted": {}, "last_wake": "", "last_spoke": "",
        "wakes": 1,
    }), encoding="utf-8")

    best = ServiceState.load(state_dir).best()
    assert best.genome is None
    assert best.genome_hash == "a" * 16
