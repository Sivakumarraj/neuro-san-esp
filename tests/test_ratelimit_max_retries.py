"""max_retries=0 on the client, so the keyring rotates on the first 429.

Without this, langchain's internal tenacity loop retries a per-day 429 six
times before letting the exception propagate to the wrapper -- ~3 minutes per
call spent on a quota that will not clear inside the run, during which the
next key sits idle. Setting it to zero surfaces the 429 immediately, my
rotation fires, and per-minute pacing is still handled by the bucket.
"""

from __future__ import annotations

import pytest

from esp.eval import ratelimit


def test_apply_model_zeros_max_retries():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Fake(BaseModel):
        model: str = "start"
        max_retries: int = 6

    c = Fake()
    ratelimit._apply_model(c, "start")  # same model, still forces max_retries=0
    assert c.max_retries == 0


def test_apply_model_is_idempotent():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Fake(BaseModel):
        model: str = "gemini-x"
        max_retries: int = 0

    c = Fake()
    before = c.model_dump()
    ratelimit._apply_model(c, "gemini-x")
    assert c.model_dump() == before


# ------------------------------------------- transient faults on their side

def test_a_busy_model_is_retried_rather_than_scored():
    """503 UNAVAILABLE means "currently experiencing high demand -- try again
    later". It was raised on the first attempt, because the retry condition
    asked only whether the error was a quota error, so a fault whose own
    message asks to be retried was the one class excluded from the backoff
    loop sitting directly beneath it.

    The cost is measured accuracy, not just wall-clock. In the committed run
    the winning network lost a whole task to a 500 INTERNAL and was recorded
    at 0.8824 having answered correctly everything it actually finished.
    """
    from esp.eval.ratelimit import _is_transient_server_error

    for message in (
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is "
        "currently experiencing high demand. Spikes in demand are usually "
        "temporary. Please try again later.', 'status': 'UNAVAILABLE'}}",
        "500 INTERNAL. {'error': {'code': 500, 'status': 'INTERNAL'}}",
        "504 DEADLINE_EXCEEDED",
    ):
        assert _is_transient_server_error(Exception(message)), message


def test_a_quota_error_is_not_treated_as_a_transient_fault():
    """They are both retryable and they are not the same thing: a per-day cap
    never clears within a run, so it swaps models instead of waiting. Reading
    one as the other would put a dead model back on the backoff loop."""
    from esp.eval.ratelimit import _is_transient_server_error

    for message in (
        "429 RESOURCE_EXHAUSTED. quotaId GenerateRequestsPerDayPerProjectPerModel",
        "429 Too Many Requests",
    ):
        assert not _is_transient_server_error(Exception(message)), message


def test_an_ordinary_error_is_still_raised_at_once():
    """The retry loop must not swallow a real bug. A malformed request or a bad
    key is not going to be fixed by asking again six times."""
    from esp.eval.ratelimit import _is_transient_server_error

    for message in ("400 INVALID_ARGUMENT: model not found",
                    "403 PERMISSION_DENIED: API key not valid",
                    "TypeError: expected str, got int"):
        assert not _is_transient_server_error(Exception(message)), message


def test_the_wrapper_asks_a_busy_model_again_and_keeps_the_answer(monkeypatch):
    """End to end through `install`: a busy model answers on the second ask,
    and the candidate keeps its task instead of scoring it wrong.

    Driven through the real `ChatGoogleGenerativeAI` that `install` patches, so
    this exercises the wrapper rather than a re-implementation of it.
    """
    pytest.importorskip("langchain_google_genai")
    from langchain_google_genai import ChatGoogleGenerativeAI

    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception(
                "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This "
                "model is currently experiencing high demand. Please try "
                "again later.', 'status': 'UNAVAILABLE'}}")
        return "answered"

    # A fresh patch over a known original, undone by monkeypatch afterwards.
    monkeypatch.setattr(ChatGoogleGenerativeAI, "_generate", flaky,
                        raising=False)
    monkeypatch.setattr(ChatGoogleGenerativeAI, "_esp_rate_limited", False,
                        raising=False)
    monkeypatch.setattr(ratelimit, "MAX_RETRIES", 3)
    # No real waiting: the backoff is the policy under test, not the clock.
    monkeypatch.setattr(ratelimit.time, "sleep", lambda _seconds: None)
    assert ratelimit.install(rpm=600)

    limited = ChatGoogleGenerativeAI._generate
    before = ratelimit.stats().get("transient_retries", 0)

    class Client:
        model = "gemini-3.1-flash-lite"
        max_retries = 0

    assert limited(Client()) == "answered"
    assert calls["n"] == 2, "the busy model was never asked again"
    assert ratelimit.stats().get("transient_retries", 0) > before

    monkeypatch.setattr(ChatGoogleGenerativeAI, "_esp_rate_limited", False,
                        raising=False)
