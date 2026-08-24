"""Which models a run uses has to be configurable, and safe to get wrong.

Provider model names change faster than this repository does, and adding one
meant editing `esp/eval/failover.py`. That is a bad place to keep a fact that
`make probe` can measure in seconds -- and the number of affordable models is
not a detail, it is the number of candidates a day buys, which is the whole
constraint on this project.

The rule that survives every override is the one that was already paid for: a
model whose daily cap cannot fund a candidate does not belong on the ladder. A
run once walked four rungs in three minutes, spending twenty requests a rung
against a candidate that needs a hundred and sixty-five, and measured nothing.
"""

from __future__ import annotations

import pytest

from esp.eval.failover import (
    MEASURED_CAPS,
    REQUESTS_PER_CANDIDATE,
    affordable,
    parse_models,
)

# ----------------------------------------------------------------- parsing

def test_models_are_read_as_name_and_daily_cap():
    assert parse_models("a:500,b:250") == {"a": 500, "b": 250}


def test_whitespace_and_trailing_commas_are_tolerated():
    assert parse_models(" a:500 , b:250 , ") == {"a": 500, "b": 250}


def test_a_model_name_containing_a_colon_keeps_it():
    """Provider names are not guaranteed colon-free; the cap is the last field."""
    assert parse_models("vendor/model:free:300") == {"vendor/model:free": 300}


@pytest.mark.parametrize("spec", ["gemini-2.5-flash", "gemini-2.5-flash:lots",
                                  ":500"])
def test_a_malformed_entry_raises_rather_than_being_skipped(spec):
    """Skipping it would put the run on a different model than was asked for,
    and a fitness compared across models means nothing."""
    with pytest.raises(ValueError):
        parse_models(spec)


# ----------------------------------------------------------------- the rule

def test_a_model_that_cannot_fund_a_candidate_never_reaches_the_ladder():
    caps = {"rich": REQUESTS_PER_CANDIDATE, "poor": REQUESTS_PER_CANDIDATE - 1}
    assert affordable(caps, ["poor", "rich"]) == ["rich"]


def test_an_override_cannot_reintroduce_the_unaffordable_ladder():
    """The bug this rule exists for, expressed as a configuration attempt."""
    caps = {"a": 20, "b": 20, "c": 20, "d": 20}
    assert affordable(caps, ["a", "b", "c", "d"]) == []


def test_an_unknown_model_is_not_assumed_affordable():
    assert affordable({}, ["never-measured"]) == []


def test_preference_order_is_preserved():
    caps = {"first": 500, "second": 500}
    assert affordable(caps, ["second", "first"]) == ["second", "first"]


# --------------------------------------------------------------- the defaults

def test_whatever_is_configured_still_funds_the_ladder():
    """Reads the active caps, not the built-in ones: an override adds models
    that MEASURED_CAPS has never heard of, and this asserted against the wrong
    dict until ESP_MODELS was set in anger."""
    from esp.eval.failover import DAILY_CAPS, LADDER, daily_budget
    assert LADDER, "no model can fund a candidate; a run is impossible"
    assert daily_budget() == sum(DAILY_CAPS[m] for m in LADDER)
    assert all(DAILY_CAPS[m] >= REQUESTS_PER_CANDIDATE for m in LADDER)


def test_the_defaults_alone_fund_a_ladder():
    """The built-in measured caps must stand on their own, with no override."""
    from esp.eval.failover import usable_on_ladder
    built_in = usable_on_ladder(
        MEASURED_CAPS, list(MEASURED_CAPS))
    assert built_in, "the shipped caps fund nothing"


def test_a_known_broken_model_never_reaches_the_ladder_even_if_configured():
    """Affordability was the only rule an override had to pass, so ESP_MODELS
    could put a model the agent loop dies on straight onto the ladder -- which
    scores every topology zero for reasons that are not the topology. The
    existing suite caught it; this pins it."""
    from esp.eval.failover import KNOWN_BROKEN, usable_on_ladder
    broken = next(iter(KNOWN_BROKEN))
    caps = {broken: 100_000, "fine": 500}
    assert usable_on_ladder(caps, [broken, "fine"]) == ["fine"]


def test_an_excluded_model_is_reported_with_its_reason():
    from esp.eval.failover import EXCLUDED
    assert all(why.strip() for why in EXCLUDED.values())


def test_the_preflight_states_which_models_a_run_will_use():
    """"Which model am I on" must not require reading source."""
    from esp.service.preflight import run_checks
    ladder = next(c for c in run_checks() if c.name == "model ladder")
    assert "requests/day" in ladder.detail
    assert "candidate" in ladder.detail
