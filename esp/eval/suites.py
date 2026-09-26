"""Two harder question sets for the scale-up: one to select on, one to judge on.

The 250-question bank answered one question -- does the evolved advantage hold on
new questions -- and raised another: the designer's shape scored 96% on it, and
an exam every network passes cannot rank networks. Its questions were joins a
network can finish by following references. The seventeen that did separate
networks included two whole-corpus aggregates, which nearly every network fails.

So these sets are harder by construction, not by tuning:

* **40% aggregates.** "How many incidents affected contracts serviced by depot
  D05?" has no document that states the answer. The corpus search returns three
  documents a call, so an agent has to keep searching until it has every
  matching document, then count or sum them itself.
* **60% joins, one to nine documents**, the bank's shapes.

And they are built so the one that judges never helps choose:

* **`meridian-select`** (60 questions) is what a search selects on.
* **`meridian-judge`** (100) is used only to judge the winners afterwards.

The depots are split in two by a fixed seed. Every lookup and join in one set
names only entities of its own half, and every depot aggregate asks about a
depot of its own half. Aggregates over causes and years span the whole corpus by
nature, so those split by their parameters instead: no cause-and-year pair is
asked about in both sets. Every answer is computed from the seeded world, and
`tests/test_suites.py` re-derives each one from the corpus text.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from esp.eval.bank import NUMBER_ONLY, _Builder
from esp.eval.tasks import TASKS, Task
from esp.eval.world import World, build_world

SUITE_SEED = 20260926
SELECT_SIZE = 60
JUDGE_SIZE = 100

# Share of each kind of question. Depths are documents combined, as in the bank.
MIX = {1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10, 6: 0.10, 9: 0.10, "aggregate": 0.40}


class _HardBuilder(_Builder):
    """The bank's joins restricted to one half of the depots, plus aggregates."""

    def __init__(self, world: World, rng: random.Random, depots: set[str],
                 pairs: list[tuple[str, int]]):
        others = {d.code for d in world.depots if d.code not in depots}
        super().__init__(world, rng, others)
        self.pairs = pairs

    def _contracts_of(self, code: str):
        return [c for c in self.world.contracts if c.depot_code == code]

    def _incidents_of(self, code: str):
        refs = {c.ref for c in self._contracts_of(code)}
        return [i for i in self.world.incidents if i.contract_ref in refs]

    def aggregate(self) -> tuple[str, object, int] | None:
        kind = self.rng.randrange(7)
        if kind < 5:
            depot = self.rng.choice(self.depots)
            contracts = self._contracts_of(depot.code)
            incidents = self._incidents_of(depot.code)
            if kind == 0:
                if len(contracts) < 2:
                    return None
                return (f"How many Meridian Logistics contracts are serviced by depot "
                        f"{depot.code}?{NUMBER_ONLY}", len(contracts), len(contracts))
            if kind == 1:
                if len(contracts) < 2:
                    return None
                total = sum(c.annual_value for c in contracts)
                return (f"What is the combined annual value of every contract "
                        f"serviced by depot {depot.code}?{NUMBER_ONLY}",
                        total, len(contracts))
            if len(incidents) < 2:
                return None
            reading = len(contracts) + len(incidents)
            if kind == 2:
                return (f"How many incidents affected contracts serviced by depot "
                        f"{depot.code}?{NUMBER_ONLY}", len(incidents), reading)
            if kind == 3:
                hours = [i.hours_late for i in incidents]
                if sum(hours) in hours:
                    return None
                return (f"Adding up every incident on contracts serviced by depot "
                        f"{depot.code}, how many hours late is that in total?"
                        f"{NUMBER_ONLY}", sum(hours), reading)
            owed = [i.hours_late * self.world.contract(i.contract_ref).penalty_per_hour
                    for i in incidents]
            if sum(owed) in owed:
                return None
            return (f"Charging every incident on contracts serviced by depot "
                    f"{depot.code} at its own contract's late-delivery penalty rate, "
                    f"what is the total penalty owed?{NUMBER_ONLY}", sum(owed), reading)

        cause, year = self.rng.choice(self.pairs)
        matching = [i for i in self.world.incidents
                    if i.cause == cause and i.year == year]
        if len(matching) < 2:
            return None
        if kind == 5:
            return (f"How many incidents in {year} were caused by {cause}?"
                    f"{NUMBER_ONLY}", len(matching), len(matching))
        hours = [i.hours_late for i in matching]
        if sum(hours) in hours:
            return None
        return (f"Across every incident in {year} caused by {cause}, how many hours "
                f"late is that in total?{NUMBER_ONLY}", sum(hours), len(matching))


def _halves(world: World, seed: int) -> tuple[set[str], set[str]]:
    codes = sorted(d.code for d in world.depots)
    random.Random(seed).shuffle(codes)
    middle = len(codes) // 2
    return set(codes[:middle]), set(codes[middle:])


def _pairs(world: World, seed: int) -> tuple[list, list]:
    pairs = sorted({(i.cause, i.year) for i in world.incidents})
    random.Random(seed + 1).shuffle(pairs)
    middle = len(pairs) // 2
    return pairs[:middle], pairs[middle:]


def _build(size: int, prefix: str, depots: set[str], pairs: list, seed: int,
           world: World, taken: set[str]) -> list[Task]:
    rng = random.Random(seed)
    builder = _HardBuilder(world, rng, depots, pairs)
    makers: dict[object, Callable[[], tuple | None]] = {
        1: builder.one, 2: builder.two, 3: builder.three, 4: builder.four,
        6: builder.six, 9: builder.nine, "aggregate": builder.aggregate}
    wanted = {kind: round(share * size) for kind, share in MIX.items()}
    wanted["aggregate"] += size - sum(wanted.values())

    groups: dict[object, list[tuple[int, str, str]]] = {kind: [] for kind in MIX}
    for kind, count in wanted.items():
        attempts = 0
        while len(groups[kind]) < count:
            attempts += 1
            if attempts > count * 2000:
                raise RuntimeError(f"{prefix}: could not draw {count} distinct "
                                   f"{kind!r} questions from this half of the world")
            drawn = makers[kind]()
            if drawn is None or drawn[0] in taken:
                continue
            question, answer = drawn[0], drawn[1]
            hops = drawn[2] if kind == "aggregate" else kind
            taken.add(question)
            groups[kind].append((hops, question, str(answer)))

    # Round-robin, so the first twenty of a set -- what a calibration run asks --
    # already covers every kind.
    ordered = []
    for position in range(max(len(g) for g in groups.values())):
        for kind in MIX:
            if position < len(groups[kind]):
                ordered.append(groups[kind][position])
    width = len(str(size))
    return [Task(f"{prefix}{n:0{width}d}", question, answer, hops, (answer,))
            for n, (hops, question, answer) in enumerate(ordered, start=1)]


def build_suites(seed: int = SUITE_SEED,
                 world: World | None = None) -> tuple[list[Task], list[Task]]:
    """(select, judge). Deterministic: the same seed yields the same questions."""
    world = world or build_world()
    select_depots, judge_depots = _halves(world, seed)
    select_pairs, judge_pairs = _pairs(world, seed)
    taken = {task.question for task in TASKS}
    select = _build(SELECT_SIZE, "S", select_depots, select_pairs, seed, world, taken)
    judge = _build(JUDGE_SIZE, "J", judge_depots, judge_pairs, seed + 7, world, taken)
    return select, judge


SELECT, JUDGE = build_suites()
