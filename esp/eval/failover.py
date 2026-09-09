"""Move to the next model when one model's daily quota is exhausted.

The free tier is model-specific. ESP therefore keeps the usable evolution ladder
separate from models that are worth probing but cannot fund one full candidate.
The default free ladder deliberately stays on the two Flash-Lite models that
provide the repository's 1,000-request/day planning budget.
"""

from __future__ import annotations

import os
import re
import threading

# One candidate currently needs about this many provider requests across the
# 17-task evaluation. A model with a smaller daily cap cannot fund one complete
# candidate and therefore must not be used as a failover rung.
REQUESTS_PER_CANDIDATE = 165

# Current Google free-tier planning values. Gemini 3.8 Flash is supported and
# probed, but its current free RPD is too small to fund one complete candidate.
# The default evolution ladder therefore uses the two Flash-Lite models for a
# combined 1,000-request/day planning budget.
MEASURED_CAPS: dict[str, int] = {
    "gemini-3.8-flash": 20,
    "gemini-3.5-flash-lite": 500,
    "gemini-3.1-flash-lite": 500,
    "gemini-3.5-flash": 20,
    "gemini-3.6-flash": 20,
    "gemini-3.7-flash": 20,
    "gemini-3-flash": 20,
    "gemini-3.1-pro": 0,
}


def parse_models(spec: str) -> dict[str, int]:
    """Read `name:cap,name:cap` into daily caps."""
    caps: dict[str, int] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, raw = entry.rpartition(":")
        if not separator or not name.strip():
            raise ValueError(f"ESP_MODELS entry {entry!r} is not name:daily_cap")
        try:
            caps[name.strip()] = int(raw)
        except ValueError as bad:
            raise ValueError(
                f"ESP_MODELS entry {entry!r} has a non-numeric daily cap"
            ) from bad
    return caps


KNOWN_BROKEN: dict[str, str] = {
    "gemini-2.5-flash": "the agent loop dies on it",
    "gemini-2.5-flash-lite": "the agent loop dies on it",
    "gemini-2.5-pro": "the agent loop dies on it",
}


def usable_on_ladder(
    caps: dict[str, int],
    preference: list[str],
    broken: dict[str, str] | None = None,
) -> list[str]:
    """Keep only models that can fund a complete candidate and are not broken."""
    broken = KNOWN_BROKEN if broken is None else broken
    return [
        model
        for model in preference
        if caps.get(model, 0) >= REQUESTS_PER_CANDIDATE and model not in broken
    ]


# Compatibility alias used by older callers/tests.
affordable = usable_on_ladder

DAILY_CAPS: dict[str, int] = dict(MEASURED_CAPS)

# Free evolution ladder: 500 + 500 = 1,000 requests/day in the default plan.
# Gemini 3.8 Flash remains a first-class probeable model in DAILY_CAPS but is not
# placed on the free evolution ladder because its current 20 RPD cannot fund one
# candidate evaluation.
_PREFERENCE = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

_OVERRIDE = os.environ.get("ESP_MODELS", "").strip()
if _OVERRIDE:
    _EXTRA = parse_models(_OVERRIDE)
    DAILY_CAPS.update(_EXTRA)
    _PREFERENCE = list(_EXTRA) + [m for m in _PREFERENCE if m not in _EXTRA]

LADDER: list[str] = usable_on_ladder(DAILY_CAPS, _PREFERENCE)

EXCLUDED: dict[str, str] = {
    model: KNOWN_BROKEN.get(
        model,
        f"daily cap {DAILY_CAPS.get(model, 0)} < {REQUESTS_PER_CANDIDATE} "
        "needed for one candidate",
    )
    for model in _PREFERENCE
    if model not in LADDER
}

UNAFFORDABLE: list[str] = [m for m in EXCLUDED if m not in KNOWN_BROKEN]


def daily_budget() -> int:
    """Requests the current evolution ladder can spend in a day."""
    return sum(DAILY_CAPS.get(model, 0) for model in LADDER)


_lock = threading.Lock()
_retired: set[str] = set()
_swaps: list[dict] = []

_DAILY_MARKERS = ("perday", "per day", "free_tier_requests", "requests_per_day")
_PER_MINUTE_MARKERS = ("perminute", "per minute", "requests_per_minute")


def is_daily_quota_error(exc: BaseException) -> bool:
    """Distinguish a daily cap from a per-minute rate limit."""
    text = str(exc)
    if "RESOURCE_EXHAUSTED" not in text and "429" not in text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _PER_MINUTE_MARKERS):
        return False
    return any(marker in lowered for marker in _DAILY_MARKERS)


_MODEL_PATTERN = re.compile(
    r"(?<![\w./-])(?:[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9._-]*(?::[a-z]+)?"
    r"|gemini-[a-z0-9]+(?:[-.][a-z0-9]+)*)",
    re.IGNORECASE,
)


def models_named(text: str) -> list[str]:
    """Return model names mentioned in a provider message, deduplicated."""
    found: list[str] = []
    for match in _MODEL_PATTERN.findall(text):
        name = match.rstrip(".-").lower()
        if name not in found:
            found.append(name)
    return found


def prime(models: list[str]) -> None:
    with _lock:
        _retired.update(models)


def retire(model: str, reason: str = "daily quota exhausted") -> str | None:
    with _lock:
        _retired.add(model)
        for candidate in LADDER:
            if candidate not in _retired:
                _swaps.append({"from": model, "to": candidate, "reason": reason})
                return candidate
        _swaps.append({"from": model, "to": None, "reason": reason})
        return None


def substitute(model: str) -> str:
    with _lock:
        if model not in _retired:
            return model
        for candidate in LADDER:
            if candidate not in _retired:
                return candidate
        return model


def retired() -> list[str]:
    with _lock:
        return sorted(_retired)


def swaps() -> list[dict]:
    with _lock:
        return list(_swaps)


def reset() -> None:
    with _lock:
        _retired.clear()
        _swaps.clear()
