"""Token and dollar accounting, and the one correction every stored figure needs.

neuro-san's front man reports a run's usage twice in one structure: once as the
network's total and once per model,

    {"total_tokens": n, "total_cost": c, ...,
     "models": {provider: {model: {"total_tokens": n, ...}}}}

Until accounting version 2 the runner walked that structure and added up every
total it found, so every token count and dollar figure stored before then is
exactly twice what the run used. Every committed token count is even, which is
what the doubling forces; `tests/test_accounting.py` holds that.

Stored measurements are never rewritten. A record written from now on carries
``"accounting": 2``; a record without it is halved on the way in, here and
nowhere else, so every reader works in the same units.
"""

from __future__ import annotations

ACCOUNTING = 2
"""The accounting version written into every record from the fix onward."""

_LEGACY_FACTOR = 0.5
_COST_KEYS = ("cost", "reported_cost", "listed_cost")


def factor(raw: dict) -> float:
    """1.0 for a record counted once, 0.5 for one written before the fix."""
    return 1.0 if raw.get("accounting") == ACCOUNTING else _LEGACY_FACTOR


def tokens_in(raw: dict) -> int:
    """The tokens a stored record actually used."""
    return round(int(raw["tokens"]) * factor(raw))


def cost_in(raw: dict, key: str = "cost") -> float:
    """The dollars a stored record actually cost, under one of its cost keys."""
    return float(raw[key]) * factor(raw)


def normalised(raw: dict) -> dict:
    """A copy of a stored record in version-2 units, marked as such.

    Only the usage fields change; accuracy, answers and timings are untouched.
    """
    if factor(raw) == 1.0:
        return dict(raw)
    out = dict(raw)
    if "tokens" in out:
        out["tokens"] = tokens_in(raw)
    for key in _COST_KEYS:
        if isinstance(out.get(key), (int, float)):
            out[key] = round(cost_in(raw, key), 6)
    out["accounting"] = ACCOUNTING
    return out


def history_in(history: dict) -> dict:
    """A run history (results/history.json) with its records in version-2 units.

    Fitness values are left alone: they were computed against a token scale
    twice the current one, so they already mean the same thing.
    """
    if history.get("accounting") == ACCOUNTING:
        return history
    out = dict(history)
    for key in ("records", "pareto"):
        if isinstance(out.get(key), list):
            out[key] = [{**r, "tokens": round(r["tokens"] * _LEGACY_FACTOR)} if "tokens" in r else r
                        for r in out[key]]
    out["accounting"] = ACCOUNTING
    return out
