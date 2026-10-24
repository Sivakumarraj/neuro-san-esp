"""Forty questions no network has been asked: the replication set.

The headline was judged on J001 to J100, and those questions had also been asked
in earlier paid runs, because the judge half of the company had no unused
aggregate questions left at a hundred. Sixteen are left, so this set takes all
of them and keeps the judge set's mix: 40% aggregates and 10% each of joins over
one, two, three, four, six and nine documents.

Drawn from the judge half only (`esp.eval.suites._halves`), so nothing here names
an entity or a cause-and-year pair the select set does, and no question repeats
one from the built-in seventeen, select, judge or judge-plus. Answers come from
the seeded world, and `tests/test_fresh.py` re-derives each from the corpus text.
"""

from __future__ import annotations

from esp.eval.judge_plus import JUDGE_PLUS
from esp.eval.suites import JUDGE, SELECT, SUITE_SEED, _build, _halves, _pairs
from esp.eval.tasks import TASKS, Task
from esp.eval.world import World, build_world

FRESH_SIZE = 40
# Its own seed, so the draw is unlike the judge set's; the halves keep SUITE_SEED.
FRESH_SEED = SUITE_SEED + 29


def build_fresh(world: World | None = None) -> list[Task]:
    """The forty fresh questions. Deterministic."""
    world = world or build_world()
    _, judge_depots = _halves(world, SUITE_SEED)
    _, judge_pairs = _pairs(world, SUITE_SEED)
    taken = {t.question for t in (*TASKS, *SELECT, *JUDGE, *JUDGE_PLUS)}
    return _build(FRESH_SIZE, "F", judge_depots, judge_pairs, FRESH_SEED, world, taken)


FRESH = build_fresh()
