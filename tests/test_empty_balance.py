"""An account that runs out of credit is stopped at once and named as the cause.

The payload is the one OpenAI returned on 7 November 2026, when the all-four
test's last chunks met an empty prepaid balance. It came as a 429, so it was
retried as a rate limit six times a call, and the run then reported every
unanswered question as a timeout or a blown recursion cap.
"""

from __future__ import annotations

import pytest

from esp.eval import ratelimit, runner
from esp.eval.runner import QuotaExhausted, TaskResult, refuse_unmeasured

EMPTY = ("Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits "
         "to continue using the API at https://platform.openai.com/settings/organization/"
         "billing.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}")
PER_MINUTE = ("Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-5.4-nano on "
              "requests per min (RPM): Limit 500, Used 500, Requested 1.', 'type': 'requests', "
              "'code': 'rate_limit_exceeded'}}")


class Refused(Exception):
    status_code = 429


def test_an_empty_balance_is_not_retried():
    assert ratelimit.is_account_failure(Refused(EMPTY))
    assert ratelimit._backoff(Refused(EMPTY), attempt=0).action == "raise"


def test_a_per_minute_limit_is_still_waited_out():
    assert not ratelimit.is_account_failure(Refused(PER_MINUTE))
    assert ratelimit._backoff(Refused(PER_MINUTE), attempt=0).action == "wait"


def test_a_run_whose_balance_ran_out_is_named_an_account_failure():
    """Every task failed with neuro-san's agent-failure reply around the 429,
    which the all-gave-up guard used to report as timeouts."""
    reply = f"Error from Coordinator: Agent stopped due to exception {EMPTY}"
    results = runner.classify([TaskResult(f"J0{n}", 2, False, 9.0, reply) for n in range(89, 93)])
    assert all(r.infrastructure for r in results)
    with pytest.raises(QuotaExhausted, match="account failure"):
        refuse_unmeasured(results, tokens=4_000, action="report")


def test_a_run_that_only_timed_out_is_still_called_one():
    results = runner.classify([TaskResult(f"T0{n}", 2, False, 300.0, "Agent timed out")
                               for n in range(1, 5)])
    with pytest.raises(OSError, match="gave up before answering") as raised:
        refuse_unmeasured(results, tokens=4_000)
    assert not isinstance(raised.value, QuotaExhausted)
