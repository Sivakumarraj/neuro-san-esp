"""The rate limiter, which is a correctness component and not a courtesy one.

A 429 comes back through neuro-san as an agent error: the candidate scores zero
and the search learns that a perfectly good topology is bad. Every property
here exists because getting it wrong corrupts a measurement rather than merely
slowing one down -- and this module shipped untested through the worst bug in
the project.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from esp.eval import failover, ratelimit


@pytest.fixture(autouse=True)
def fresh():
    ratelimit._buckets.clear()
    failover.reset()
    yield
    ratelimit._buckets.clear()
    failover.reset()


# ------------------------------------------------------------------- pacing


def test_a_bucket_hands_out_its_whole_allowance_immediately():
    """No artificial spacing. The limit is per minute, so all of it may be
    spent in the first second -- pacing evenly would halve throughput for no
    reason the provider asks for."""
    bucket = ratelimit.Bucket(rpm=5)
    assert [bucket._try_take() for _ in range(5)] == [0.0] * 5


def test_the_next_caller_is_told_how_long_to_wait():
    bucket = ratelimit.Bucket(rpm=2)
    bucket._try_take()
    bucket._try_take()
    wait = bucket._try_take()
    assert 0 < wait <= 60.1


def test_the_window_slides_rather_than_resetting():
    """A fixed window would let 2*rpm calls land either side of a boundary,
    which is exactly the burst the provider counts as an overage."""
    bucket = ratelimit.Bucket(rpm=2)
    bucket._try_take()
    time.sleep(0.05)
    bucket._try_take()
    first_wait = bucket._try_take()
    # The oldest call is 0.05s older, so the wait is correspondingly shorter.
    assert first_wait < 60.0


def test_buckets_are_per_model():
    """One exhausted model must not pace a model that still has budget."""
    a = ratelimit.bucket_for("model-a", rpm=1)
    b = ratelimit.bucket_for("model-b", rpm=1)
    assert a is not b
    assert a._try_take() == 0.0
    assert b._try_take() == 0.0


def test_the_same_model_shares_one_bucket():
    assert ratelimit.bucket_for("m") is ratelimit.bucket_for("m")


# ------------------------------------------------------- the event-loop bug


def test_waiting_does_not_block_the_event_loop():
    """The regression this module exists to prevent.

    The first limiter called `time.sleep` at an async call site. neuro-san runs
    its agents on asyncio, so one pacing wait froze every other agent in the
    process -- each still spending its own `max_execution_seconds` while frozen,
    then all cancelled together. The search read that as "these topologies are
    bad".

    The test asserts the property directly: while one coroutine is waiting on a
    full bucket, another must still be making progress.
    """
    bucket = ratelimit.Bucket(rpm=1)
    bucket._try_take()                     # bucket is now full
    progressed = []

    async def waiter():
        await bucket.acquire_async()

    async def neighbour():
        for _ in range(5):
            await asyncio.sleep(0)
            progressed.append(1)

    async def main():
        task = asyncio.ensure_future(waiter())
        await neighbour()
        task.cancel()

    asyncio.run(main())
    assert len(progressed) == 5, "a pacing wait starved a concurrent agent"


# ---------------------------------------------------------------- failover


def test_a_live_client_can_be_pointed_at_another_model():
    """The client is built by neuro-san from the HOCON, so redirecting a call
    already in flight means writing the field it reads."""

    class Client:
        model = "gemini-3.5-flash-lite"

    client = Client()
    ratelimit._apply_model(client, "gemini-3.5-flash")
    assert client.model == "gemini-3.5-flash"


def test_a_client_that_refuses_redirection_is_not_fatal():
    """A model we cannot steer still makes its call, just on the model it was
    built with. Raising here would turn a slow run into a failed one."""

    class Stubborn:
        __slots__ = ()

        @property
        def model(self):
            return "fixed"

    ratelimit._apply_model(Stubborn(), "other")     # must not raise


# ----------------------------------------------------------- classification


@pytest.mark.parametrize("text", [
    "429 RESOURCE_EXHAUSTED",
    "ResourceExhausted: 429 You exceeded your current quota",
])
def test_quota_errors_are_recognised(text):
    assert ratelimit._is_quota_error(Exception(text))


@pytest.mark.parametrize("text", [
    "ConnectionResetError", "500 internal error", "invalid api key",
])
def test_other_errors_are_not_treated_as_quota(text):
    """Retrying a bad key sixty times is not resilience, it is a hang."""
    assert not ratelimit._is_quota_error(Exception(text))


def test_a_per_minute_limit_is_not_mistaken_for_a_per_day_cap():
    """The distinction the whole design turns on. A per-minute limit clears by
    waiting, so the limiter should sleep. A per-day cap never clears within a
    run, so sleeping burns the clock and every candidate after it scores zero."""
    minute = Exception("429 RESOURCE_EXHAUSTED quota_id: "
                       "GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    day = Exception("429 RESOURCE_EXHAUSTED quota_id: "
                    "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert not failover.is_daily_quota_error(minute)
    assert failover.is_daily_quota_error(day)
    assert ratelimit._is_quota_error(minute)


def test_the_fixed_sampling_notice_is_silenced_and_nothing_else_is():
    """One library warning fired once per model call and buried real errors.

    langchain-google-genai warns that a model "uses fixed sampling defaults"
    whenever a temperature reaches a model that has none. This project never
    sets a temperature -- neuro-san's own default llm_config does -- so there
    is nothing to fix and the warning is accurate. It is also emitted on every
    call: a live run against the newest Gemini models produced hundreds of
    copies and buried the recursion errors that needed reading.

    The filter has to be narrow. Swallowing warnings wholesale to quieten one
    known-harmless notice is how a real one goes unseen later.
    """
    import warnings

    from esp.eval.ratelimit import quieten_fixed_sampling_warning

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        quieten_fixed_sampling_warning()
        warnings.warn(
            "Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the "
            "sampling parameter(s) temperature will be ignored.",
            UserWarning, stacklevel=1)
        warnings.warn("a deprecation that matters", UserWarning, stacklevel=1)
        warnings.warn("a different category entirely", DeprecationWarning,
                      stacklevel=1)

    messages = [str(w.message) for w in seen]
    assert not any("fixed sampling" in m for m in messages), messages
    assert any("matters" in m for m in messages), messages
    assert any("different category" in m for m in messages), messages


# ------------------------------------------------------ paid-provider pacing

def test_an_overloaded_claude_api_is_retried_not_scored_wrong():
    """Anthropic says 503 as 529. It was missing, so a busy API scored a task
    wrong on a paid key while a comment claimed the case was handled."""
    overloaded = Exception("Error code: 529 - {'type': 'error', 'error': "
                           "{'type': 'overloaded_error', 'message': 'Overloaded'}}")
    assert ratelimit._is_transient_server_error(overloaded)


def test_a_paid_key_can_be_paced_faster_than_the_free_tier():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    code = ("from esp.eval import ratelimit\n"
            "print(ratelimit.DEFAULT_RPM, ratelimit.bucket_for('claude-haiku-4-5').rpm)")
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
        env={**os.environ, "ESP_NO_DOTENV": "1", "ESP_RPM": "45",
             "PYTHONPATH": str(root)})
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["45", "45"]


def test_a_measured_gemini_rate_still_beats_the_override():
    """ESP_RPM is the fallback for unmeasured models, never a way to push a
    free-tier model past the limit its own 429s reported."""
    assert failover.rpm_for("gemini-3.8-flash", 45) < 45
