"""The one significance test this project uses, in one place."""

from __future__ import annotations

from math import comb


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p for two networks asked the same questions.

    Only the discordant questions carry information -- those one network got
    right and the other wrong. This is how surprising their split would be if
    neither network were better.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(right_a: dict[str, bool], right_b: dict[str, bool]) -> dict:
    """Question-by-question comparison of two networks on one question set."""
    shared = sorted(set(right_a) & set(right_b))
    only_a = [t for t in shared if right_a[t] and not right_b[t]]
    only_b = [t for t in shared if right_b[t] and not right_a[t]]
    return {"questions": len(shared), "only_a": only_a, "only_b": only_b,
            "p": mcnemar_exact(len(only_a), len(only_b))}
