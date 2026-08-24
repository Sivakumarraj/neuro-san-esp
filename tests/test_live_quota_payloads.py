"""The 429s Google actually sent, kept verbatim.

Every payload in this file was captured from a real exhaustion on
2026-08-22 while running the seed network against a live key. They are here
because the previous markers were written against an older payload format and
silently stopped matching: nothing failed, nothing warned, the classifier just
started answering "no" to the question it exists to answer.

That is the expensive direction. A daily cap misread as a transient rate limit
is retried with backoff until MAX_RETRIES, the model is never retired, and
failover -- the entire mechanism for surviving an exhausted model -- never
runs. One task observed here spent 230 seconds doing that before failing.
"""

from __future__ import annotations

import pytest

from esp.eval.failover import is_daily_quota_error, models_named

# Verbatim, from the run. The shape that broke the old classifier: no "PerDay",
# no "per day", and a nine-second retry hint on a budget that does not return
# for hours.
DAILY_2026_08 = (
    "You exceeded your current quota, please check your plan and billing "
    "details. For more information on this error, head to: "
    "https://ai.google.dev/gemini-api/docs/rate-limits. \n"
    "* Quota exceeded for metric: generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests, limit: 500, "
    "model: gemini-3.5-flash-lite\nPlease retry in 9.038661104s. "
    "RESOURCE_EXHAUSTED 429")

# The older shape, still in the wild, which the original markers were built for.
DAILY_LEGACY = (
    "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel "
    "quota_metric generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests limit 500 model gemini-3.1-flash-lite")

PER_MINUTE = (
    "429 RESOURCE_EXHAUSTED Quota exceeded for metric: "
    "generate_requests_per_minute, limit: 15, model: gemini-3.1-flash-lite. "
    "Please retry in 4s.")


@pytest.mark.parametrize("payload", [DAILY_2026_08, DAILY_LEGACY])
def test_a_real_daily_exhaustion_is_recognised(payload):
    """Both formats, because the provider changed the wording once already and
    a classifier that only knows the current one will fail the same way again."""
    assert is_daily_quota_error(Exception(payload)) is True


def test_a_per_minute_limit_is_not_treated_as_a_daily_cap():
    """This one really does clear by waiting. Retiring the model for the rest
    of the day over a four-second throttle would throw away the budget."""
    assert is_daily_quota_error(Exception(PER_MINUTE)) is False


def test_something_that_is_not_a_quota_error_at_all():
    assert is_daily_quota_error(Exception("500 Internal Server Error")) is False
    assert is_daily_quota_error(ValueError("bad config")) is False


@pytest.mark.parametrize("payload,model", [
    (DAILY_2026_08, "gemini-3.5-flash-lite"),
    (DAILY_LEGACY, "gemini-3.1-flash-lite"),
    (PER_MINUTE, "gemini-3.1-flash-lite"),
])
def test_the_model_that_ran_out_is_named(payload, model):
    """Retiring the wrong model is as bad as retiring none: the service keeps
    calling the dead one and stops calling a live one."""
    assert model in models_named(payload)


def test_the_quota_metric_path_is_not_read_as_a_model():
    """`generativelanguage.googleapis.com/generate_content_free_tier_requests`
    is a metric, not a model, and it appears in every one of these payloads."""
    for payload in (DAILY_2026_08, DAILY_LEGACY):
        assert not [m for m in models_named(payload) if "/" in m]
