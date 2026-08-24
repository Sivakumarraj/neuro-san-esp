"""Model failover: what happens when one model's daily budget runs out.

The distinction this file pins is between a per-minute rate limit and a per-day
cap. The first clears by waiting; the second never clears within a run. Treating
a daily cap as something to sleep through burns the clock and scores every
candidate after it as zero -- which teaches the search that good topologies are
bad, the one failure mode that does not announce itself.
"""

from __future__ import annotations

import pytest

from esp.eval import failover


@pytest.fixture(autouse=True)
def _clean():
    failover.reset()
    yield
    failover.reset()


class _Err(Exception):
    pass


def test_per_day_cap_is_recognised():
    exc = _Err("429 RESOURCE_EXHAUSTED ... quotaId: "
               "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")
    assert failover.is_daily_quota_error(exc)


def test_per_minute_limit_is_not_a_daily_cap():
    """Must be waited out, not failed over -- swapping models on a per-minute
    limit would burn the whole ladder in the first minute of a run."""
    exc = _Err("429 RESOURCE_EXHAUSTED ... quotaId: "
               "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'")
    assert not failover.is_daily_quota_error(exc)


def test_unrelated_errors_are_not_quota_errors():
    assert not failover.is_daily_quota_error(_Err("connection reset by peer"))
    assert not failover.is_daily_quota_error(_Err("400 invalid argument"))


def test_retirement_advances_down_the_ladder():
    first, second = failover.LADDER[0], failover.LADDER[1]
    assert failover.substitute(first) == first
    assert failover.retire(first) == second
    assert failover.substitute(first) == second


def test_a_retired_model_is_never_returned_again():
    for model in failover.LADDER[:-1]:
        failover.retire(model)
    survivor = failover.LADDER[-1]
    for model in failover.LADDER:
        assert failover.substitute(model) == survivor
    assert set(failover.retired()) == set(failover.LADDER[:-1])


def test_exhausting_the_ladder_reports_no_successor():
    for model in failover.LADDER:
        last = failover.retire(model)
    assert last is None


def test_swaps_are_recorded_for_the_writeup():
    """A model swap changes what is being measured, so a run that failed over is
    only comparable within itself. That has to be visible, not silent."""
    failover.retire(failover.LADDER[0])
    swaps = failover.swaps()
    assert swaps and swaps[0]["from"] == failover.LADDER[0]
    assert swaps[0]["to"] == failover.LADDER[1]


def test_ladder_holds_no_known_broken_models():
    """gemini-2.5-* are reachable and in budget but the agent loop fails on them
    ('Agent stopped due to exception'). Failing over onto one would swap a quota
    problem for a broken-run problem and still score the topology as zero."""
    assert not [m for m in failover.LADDER if m.startswith("gemini-2.5")]


def test_quota_exhaustion_is_its_own_exception_type():
    """The caller has to tell 'stop the run, keep what you measured' apart from
    'this candidate is bad'. Matching on OSError text would not survive a
    reworded message, and the run previously died with a traceback that threw
    away every measurement already paid for."""
    from esp.eval.runner import QuotaExhausted

    assert issubclass(QuotaExhausted, OSError)


def test_history_records_that_a_run_stopped_early():
    """A run that ran out of budget is a different artefact from one that
    finished. A reader who cannot tell them apart reads a short fitness curve as
    a converged search."""
    from esp.evolve.loop import History

    assert History().stopped_early == ""


def test_surrogate_survives_a_model_it_has_never_seen():
    """Feature extraction called MODEL_TIERS.index() directly, which raised on
    any model not in the list -- and failover exists precisely to substitute
    models that are not in it. A genome measured under a swapped model would
    have taken the surrogate down with it."""
    from esp.genome.definition import MODEL_TIERS
    from esp.genome.seeds import solo
    from esp.surrogate.predictor import _tier, features

    assert _tier(MODEL_TIERS[0]) == 0
    # Unknown models sort after the known ones: we do not know what they cost.
    assert _tier("some-model-invented-tomorrow") == len(MODEL_TIERS)

    genome = solo()
    genome.default_model = "a-model-not-in-any-tier"
    vector = features(genome)          # must not raise
    assert vector.shape[0] > 0


# ------------------------------------------- the ladder must be affordable


def test_no_rung_is_too_small_to_buy_a_candidate():
    """The invariant the old ladder broke.

    A candidate costs about 165 provider requests. A model capped at 20 a day
    cannot supply one, so failing over to it spends 20 requests, changes what is
    being measured mid-evaluation, and fails anyway. The previous ladder held
    three such models and a real run walked all four rungs in under three
    minutes with nothing measured.
    """
    for model in failover.LADDER:
        cap = failover.DAILY_CAPS[model]
        assert cap >= failover.REQUESTS_PER_CANDIDATE, (
            f"{model} allows {cap}/day, which cannot fund one candidate "
            f"({failover.REQUESTS_PER_CANDIDATE} requests)")


def test_the_ladder_is_derived_from_the_caps_not_hand_written():
    """A prose rule cannot be checked against a list that breaks it. The rule is
    executable now, so this asserts the derivation still holds."""
    affordable = {m for m, cap in failover.DAILY_CAPS.items()
                  if cap >= failover.REQUESTS_PER_CANDIDATE}
    assert set(failover.LADDER) <= affordable


def test_the_ladder_is_not_empty():
    """An empty ladder is not a safe degradation -- it means the first quota
    failure ends every run for the rest of the day."""
    assert failover.LADDER


def test_a_model_with_no_free_tier_never_enters_the_ladder():
    assert "gemini-3.1-pro" not in failover.LADDER
    assert failover.DAILY_CAPS["gemini-3.1-pro"] == 0
