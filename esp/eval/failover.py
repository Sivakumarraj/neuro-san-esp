"""Move to the next model when the current one runs out of daily quota.

The free tier caps requests per day *per model*, so a long run does not fail
because the provider is down -- it fails because one name is spent while others
still have budget. A 429 that is not handled this way is worse than a slow run:
it comes back as an agent error, the candidate scores zero, and the search
concludes a perfectly good topology is bad.

This sits underneath neuro-san, at the LangChain client, because that is the
only place that sees every call. When a model reports its daily quota
exhausted, it is retired for the rest of the process and subsequent calls are
rewritten to the next model in the ladder.

A model swap changes what is being measured, so a run that failed over is only
comparable within itself. `swaps()` reports what happened for the record.
"""

from __future__ import annotations

import os
import re
import threading

# What one candidate costs, measured: about 165 provider requests across the 17
# tasks. This is the number that decides which models are worth anything here.
REQUESTS_PER_CANDIDATE = 165

# Daily free-tier caps, read out of the provider's own 429 payloads rather than
# from its documentation. Held as data so the ladder can be derived from it
# instead of hand-maintained alongside it: a model whose cap is below
# REQUESTS_PER_CANDIDATE cannot fund a single evaluation, and failing over to
# one spends its budget, changes what is being measured mid-evaluation, and
# fails anyway.
MEASURED_CAPS: dict[str, int] = {
    # OpenRouter's free router picks an available free model per request and
    # moves off exhausted ones itself, which is the failover this module does
    # by hand for Google. 1000/day is the documented free-tier allowance for an
    # account with credits; it has not been measured here, so it is the one
    # number in this table taken from documentation rather than a 429 payload.
    "openrouter/free": 1000,
    "gemini-3.5-flash-lite": 500,
    "gemini-3.1-flash-lite": 500,
    "gemini-3.5-flash": 20,
    "gemini-3.6-flash": 20,
    "gemini-3.7-flash": 20,
    "gemini-3-flash": 20,
    "gemini-3.1-pro": 0,          # no free-tier access at all
}


def parse_models(spec: str) -> dict[str, int]:
    """Read `name:cap,name:cap` into daily caps.

    Provider model names change faster than this file does, and adding one
    meant editing Python. That is a bad place to keep a fact that a `make probe`
    can measure in seconds.

    Malformed input raises rather than being skipped. A typo that silently
    dropped a model would put the run on a different model than the operator
    asked for, and a fitness compared across models means nothing.
    """
    caps: dict[str, int] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, raw = entry.rpartition(":")
        if not separator or not name.strip():
            raise ValueError(
                f"ESP_MODELS entry {entry!r} is not name:daily_cap")
        try:
            caps[name.strip()] = int(raw)
        except ValueError as bad:
            raise ValueError(
                f"ESP_MODELS entry {entry!r} has a non-numeric daily cap") from bad
    return caps


# Reachable, in budget, and still unusable: the agent loop dies on these
# ("Agent stopped due to exception"), so a candidate measured on one scores zero
# for reasons unrelated to its topology. Held as data beside the caps so the
# ladder filter can apply both rules -- affordability is not the only way a
# model can be the wrong choice.
KNOWN_BROKEN: dict[str, str] = {
    "gemini-2.5-flash": "the agent loop dies on it",
    "gemini-2.5-flash-lite": "the agent loop dies on it",
    "gemini-2.5-pro": "the agent loop dies on it",
}


def usable_on_ladder(caps: dict[str, int], preference: list[str],
                     broken: dict[str, str] | None = None) -> list[str]:
    """The ladder: preference order, minus anything that cannot fund a
    candidate and anything known to break the agent loop.

    Both rules apply to measured data and to environment overrides alike. An
    override exempt from them could seat a model that cannot fund an evaluation,
    or one that scores every topology zero for reasons that are not the
    topology.
    """
    broken = KNOWN_BROKEN if broken is None else broken
    return [model for model in preference
            if caps.get(model, 0) >= REQUESTS_PER_CANDIDATE
            and model not in broken]


# Compatibility alias for the name used elsewhere in the module and in tests.
affordable = usable_on_ladder


DAILY_CAPS: dict[str, int] = dict(MEASURED_CAPS)

# Ordered fastest-first among the models that can fund a candidate. Evaluation
# is the bottleneck of the project and the ladder is walked in order. Both
# entries were probed through neuro-san rather than assumed.
_PREFERENCE = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

# ESP_MODELS="name:cap,name:cap" adds or replaces caps and puts those names at
# the front of the preference order, so a new free model can be adopted from
# .env once `make probe` has measured what it allows -- no source edit, and the
# affordability rule above still decides what reaches the ladder.
_OVERRIDE = os.environ.get("ESP_MODELS", "").strip()
if _OVERRIDE:
    _EXTRA = parse_models(_OVERRIDE)
    DAILY_CAPS.update(_EXTRA)
    _PREFERENCE = list(_EXTRA) + [m for m in _PREFERENCE if m not in _EXTRA]

LADDER: list[str] = usable_on_ladder(DAILY_CAPS, _PREFERENCE)

# Named but not on the ladder, with the reason, so the preflight can say so out
# loud. A model silently dropped is the exact surprise this project has already
# paid for -- and somebody who put a name in ESP_MODELS deserves to be told why
# it is not being used rather than to find out from a run of zeros.
EXCLUDED: dict[str, str] = {
    model: KNOWN_BROKEN.get(
        model,
        f"daily cap {DAILY_CAPS.get(model, 0)} < {REQUESTS_PER_CANDIDATE} "
        "needed for one candidate")
    for model in _PREFERENCE if model not in LADDER
}

UNAFFORDABLE: list[str] = [m for m in EXCLUDED if m not in KNOWN_BROKEN]


def daily_budget() -> int:
    """Requests the ladder can spend in a day, and therefore how many
    candidates a day buys -- the number that decides whether a run is possible
    at all."""
    return sum(DAILY_CAPS.get(model, 0) for model in LADDER)


_lock = threading.Lock()
_retired: set[str] = set()
_swaps: list[dict] = []


# How a daily cap announces itself. Matching `PerDay` alone is not sufficient:
# Google's free-tier exhaustion arrives as
#
#   Quota exceeded for metric:
#   generativelanguage.googleapis.com/generate_content_free_tier_requests,
#   limit: 500, model: gemini-3.5-flash-lite. Please retry in 9.0s.
#
# with no "per day" anywhere in it, and a retry hint of nine seconds against a
# budget that does not return for hours. Classified as a transient rate limit
# it would be retried with backoff, the model never retired, and failover never
# reach the next rung -- which is the entire reason failover exists.
_DAILY_MARKERS = ("perday", "per day", "free_tier_requests", "requests_per_day")

# A per-minute limit really does clear by waiting, so it must not be swept up.
_PER_MINUTE_MARKERS = ("perminute", "per minute", "requests_per_minute")


def is_daily_quota_error(exc: BaseException) -> bool:
    """A per-day cap, as opposed to a per-minute rate limit.

    The distinction matters: a per-minute limit clears by waiting, so the
    limiter should sleep. A per-day cap never clears within a run, so waiting
    just burns the clock and every candidate after it scores zero.
    """
    text = str(exc)
    if "RESOURCE_EXHAUSTED" not in text and "429" not in text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _PER_MINUTE_MARKERS):
        return False
    return any(marker in lowered for marker in _DAILY_MARKERS)


# Model names as they appear in a provider's own error payload. The 429 names
# which model ran out, and reading it is more reliable than matching against
# the ladder: a candidate carries its own per-agent `llm_config`, and the
# network default is not a ladder member at all.
#
# Two shapes, because two providers. Google names a bare model
# ("gemini-3.1-flash-lite"); OpenRouter names "vendor/model", optionally with a
# ":free" or ":nitro" variant suffix.
#
# The vendor half forbids dots, and the leading lookbehind is deliberate rather
# than a \b. Google 429 payloads carry the quota metric
# `generativelanguage.googleapis.com/generate_content_free_tier_requests`; a
# permissive "anything/anything" reads that as a model, and \b will happily
# start a match after a dot and read `com/generate_content_...` as one. Either
# would retire a metric name and leave the exhausted model on the ladder.
# OpenRouter vendors are flat slugs ("meta-llama", "mistralai"); domains are
# not.
_MODEL_PATTERN = re.compile(
    r"(?<![\w./-])(?:[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9._-]*(?::[a-z]+)?"
    r"|gemini-[a-z0-9]+(?:[-.][a-z0-9]+)*)",
    re.IGNORECASE)


def models_named(text: str) -> list[str]:
    """Every model name mentioned in a provider message, in order, deduplicated.

    Used to decide which model to stop calling for the rest of the day. Read
    from the message rather than intersected with the ladder: a run on the
    network default would otherwise record nothing, and a scheduled service
    would retry the same exhausted model on every wake until the quota reset.
    """
    found: list[str] = []
    for match in _MODEL_PATTERN.findall(text):
        name = match.rstrip(".-").lower()
        if name not in found:
            found.append(name)
    return found


def prime(models: list[str]) -> None:
    """Retire models already known spent, before any call goes out.

    A wake runs in a fresh process, so the in-memory retired set starts empty
    and the first candidate would rediscover this morning's exhaustion by
    spending real calls on it. What was learned yesterday afternoon is written
    down; this is how it gets read back.
    """
    with _lock:
        _retired.update(models)


def retire(model: str, reason: str = "daily quota exhausted") -> str | None:
    """Retire a model and return the next usable one, or None if none remain."""
    with _lock:
        _retired.add(model)
        for candidate in LADDER:
            if candidate not in _retired:
                _swaps.append({"from": model, "to": candidate, "reason": reason})
                return candidate
        _swaps.append({"from": model, "to": None, "reason": reason})
        return None


def substitute(model: str) -> str:
    """The model to actually call, given one may have been retired."""
    with _lock:
        if model not in _retired:
            return model
        for candidate in LADDER:
            if candidate not in _retired:
                return candidate
        return model          # nothing left; let the caller surface the 429


def retired() -> list[str]:
    with _lock:
        return sorted(_retired)


def swaps() -> list[dict]:
    with _lock:
        return list(_swaps)


def reset() -> None:
    """Test helper. A process only ever retires models forwards."""
    with _lock:
        _retired.clear()
        _swaps.clear()
