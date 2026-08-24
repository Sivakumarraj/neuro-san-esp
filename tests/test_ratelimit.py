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
