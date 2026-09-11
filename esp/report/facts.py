"""The numbers the reports quote, derived from the measurements once.

Three PDF generators each held the population's figures as literals -- how many
networks were measured, what the best one scored, what the designer's shape
cost. So every new measurement meant hand-editing prose in three files, and the
prose fell behind exactly as the README had: a run added a twelfth network and
the primer went on describing eleven, in a document whose argument is that its
numbers were measured.

Anything that changes when a candidate is measured belongs here and is read,
not typed. Prose that would go stale is written around these values rather than
around a literal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from esp.eval import measurements
from esp.eval.tasks import TASKS

# Written out, because a sentence says "twelve networks" and not "12 networks".
NUMBERS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty",
}


def spelled(count: int) -> str:
    """`twelve`, or `24` once counting words stop helping."""
    return NUMBERS.get(count, str(count))


@dataclass(frozen=True)
class Facts:
    """What the population is, at the moment a report is generated."""

    measured: int
    tasks: int
    task_runs: int
    unfinished: int
    best: measurements.Measurement
    designer: measurements.Measurement | None
    cheapest_winner: measurements.Measurement | None
    cheapest_tokens: int
    dearest_tokens: int
    seeds: int
    evolved: int

    # ------------------------------------------------------------- phrasing

    @property
    def measured_word(self) -> str:
        return spelled(self.measured)

    @property
    def best_correct(self) -> int:
        return round(self.best.accuracy * self.tasks)

    def accuracy_gain_points(self) -> float:
        """Percentage points of accuracy over the designer's shape."""
        if self.designer is None:
            return 0.0
        return (self.best.accuracy - self.designer.accuracy) * 100

    def token_saving_percent(self) -> float:
        """Positive when the best network is cheaper than the designer's."""
        if self.designer is None or not self.designer.tokens:
            return 0.0
        return (1 - self.best.tokens / self.designer.tokens) * 100

    def cost_spread_percent(self) -> float:
        if not self.cheapest_tokens:
            return 0.0
        return (self.dearest_tokens / self.cheapest_tokens - 1) * 100


@lru_cache(maxsize=1)
def facts() -> Facts:
    """Read once per process. Reports are generated in one pass."""
    found = measurements.load()
    if not found:
        raise RuntimeError(
            "no committed measurements, so there is nothing to report")

    runs = unfinished = 0
    for entry in measurements.raw():
        results = entry.get("results") or []
        runs += len(results)
        unfinished += sum(1 for r in results if r.get("infrastructure"))

    designer = next(
        (m for m in found if m.origin == "seed:designer_shaped"), None)
    best = found[0]
    # The other end of the trade-off: the cheapest network that still beats
    # every seed. Quoting only the highest fitness hides the Pareto front,
    # which is the honest shape of the result.
    seed_best = max((m.accuracy for m in found if not m.evolved), default=0.0)
    cheaper = [m for m in found
               if m.evolved and m.accuracy > seed_best and m is not best]
    cheapest_winner = min(cheaper, key=lambda m: m.tokens) if cheaper else None

    return Facts(
        measured=len(found),
        tasks=len(TASKS),
        task_runs=runs,
        unfinished=unfinished,
        best=best,
        designer=designer,
        cheapest_winner=cheapest_winner,
        cheapest_tokens=min(m.tokens for m in found),
        dearest_tokens=max(m.tokens for m in found),
        seeds=sum(1 for m in found if not m.evolved),
        evolved=sum(1 for m in found if m.evolved),
    )
