"""A hundred more judge questions, of four kinds the other sets never ask.

Seventeen questions could not rank networks, and a paired test on a hundred can
only register a gap of about a dozen points. These bring the held-out judge set
to two hundred, and they test things the joins and totals do not:

* **Time.** "How many incidents on contracts serviced by depot D05 happened
  before 2026?" Every incident report states its year; the agent has to find
  them all and filter.
* **Filters.** Totals over only the contracts that pass a condition, so a
  network cannot answer by summing everything it found.
* **Comparisons.** Differences between two records, and counts over a few
  named depots. Every answer is a number that no single document states, which
  is what keeps a reply that lists all the candidates from scoring by accident.
* **Questions the documents cannot answer.** A contract that does not exist, or
  a field no document records, such as a depot's telephone number. The right
  answer is "not stated"; a network that invents one is wrong.

Every question here carries the same instruction to answer "not stated" when
the documents do not say, answerable or not, so the instruction itself gives
nothing away.

Built only from the judge half of the company (`esp.eval.suites._halves`), so
nothing here names an entity the select set does. The corpus is unchanged:
every v1 measurement stays valid. `tests/test_judge_plus.py` re-derives every
answer from the document text with a solver that shares no code with this one.
"""

from __future__ import annotations

import random

from esp.eval.suites import JUDGE, SUITE_SEED, _halves
from esp.eval.tasks import TASKS, Task
from esp.eval.world import World, build_world

PLUS_SIZE = 100
KINDS = ("temporal", "filtered", "compare", "unanswerable")
NOT_STATED = "not stated"
ANSWER_RULE = " Answer with the number only, or 'not stated' if the documents do not say."
# Ways a network says the documents do not hold the answer. All are accepted
# for an unanswerable question; none is accepted for an answerable one.
ABSTENTIONS = (NOT_STATED, "not in the documents", "does not exist", "no such",
               "no record", "not found", "no information", "not mentioned",
               "not specified", "not available", "cannot be determined",
               "could not find", "couldn't find", "unknown")
MISSING_FIELDS = ("telephone number", "postcode", "number of staff",
                  "annual fuel budget", "insurance policy number")


class _PlusBuilder:
    def __init__(self, world: World, rng: random.Random, depots: set[str]):
        self.world, self.rng = world, rng
        self.depots = [d for d in world.depots if d.code in depots]
        self.contracts = [c for c in world.contracts if c.depot_code in depots]

    def _contracts_of(self, code: str):
        return [c for c in self.contracts if c.depot_code == code]

    def _incidents_of(self, code: str):
        refs = {c.ref for c in self._contracts_of(code)}
        return [i for i in self.world.incidents if i.contract_ref in refs]

    def temporal(self):
        depot = self.rng.choice(self.depots)
        incidents = self._incidents_of(depot.code)
        kind = self.rng.randrange(5)
        year = self.rng.choice([2024, 2025, 2026])
        if kind == 3:
            other = self.rng.choice([d for d in self.depots if d.code != depot.code])
            first, second = sorted((depot.code, other.code))
            both = incidents + self._incidents_of(other.code)
            hits = [i for i in both if i.year == year]
            if len(hits) < 2:
                return None
            return (f"How many incidents on contracts serviced by depot {first} or "
                    f"depot {second} happened in {year}?", len(hits))
        if kind == 4:
            year = self.rng.choice([2025, 2026])
            hits = [i.hours_late for i in incidents if i.year < year]
            if len(hits) < 2 or sum(hits) in hits:
                return None
            return (f"Adding up the incidents before {year} on contracts serviced by "
                    f"depot {depot.code}, how many hours late is that in total?", sum(hits))
        if kind == 0:
            hits = [i for i in incidents if i.year == year]
            if len(hits) < 2:
                return None
            return (f"How many incidents on contracts serviced by depot {depot.code} "
                    f"happened in {year}?", len(hits))
        if kind == 1:
            year = self.rng.choice([2025, 2026])
            hits = [i for i in incidents if i.year < year]
            if len(hits) < 2 or len(hits) == len(incidents):
                return None
            return (f"How many incidents on contracts serviced by depot {depot.code} "
                    f"happened before {year}?", len(hits))
        hits = [i.hours_late for i in incidents if i.year == year]
        if len(hits) < 2 or sum(hits) in hits:
            return None
        return (f"Adding up the incidents in {year} on contracts serviced by depot "
                f"{depot.code}, how many hours late is that in total?", sum(hits))

    def filtered(self):
        depot = self.rng.choice(self.depots)
        contracts = self._contracts_of(depot.code)
        incidents = self._incidents_of(depot.code)
        kind = self.rng.randrange(4)
        if kind == 0:
            cut = self.rng.choice([10, 15, 20, 25, 30])
            hits = [i for i in incidents if i.hours_late > cut]
            if len(hits) < 2 or len(hits) == len(incidents):
                return None
            return (f"How many incidents on contracts serviced by depot {depot.code} "
                    f"were more than {cut} hours late?", len(hits))
        if kind == 1:
            cause = self.rng.choice(sorted({i.cause for i in incidents}) or [None])
            hits = [i for i in incidents if i.cause == cause]
            if cause is None or len(hits) < 2 or len(hits) == len(incidents):
                return None
            return (f"How many incidents on contracts serviced by depot {depot.code} "
                    f"were caused by {cause}?", len(hits))
        if len(contracts) < 2:
            return None
        rates = [c.penalty_per_hour for c in contracts]
        cuts = [n for n in range(500, 6000, 250) if min(rates) <= n < max(rates)]
        if not cuts:
            return None
        cut = self.rng.choice(cuts)
        above = [c for c in contracts if c.penalty_per_hour > cut]
        if kind == 2:
            if len(above) < 2:
                return None
            return (f"How many contracts serviced by depot {depot.code} have a "
                    f"late-delivery penalty above {cut} per hour?", len(above))
        values = [c.annual_value for c in above]
        if len(values) < 2 or sum(values) in values:
            return None
        return (f"What is the combined annual value of the contracts serviced by depot "
                f"{depot.code} whose late-delivery penalty is above {cut} per hour?",
                sum(values))

    def compare(self):
        kind = self.rng.randrange(3)
        if kind == 0:
            x, y = self.rng.sample(self.contracts, 2)
            if x.annual_value <= y.annual_value:
                x, y = y, x
            gap = x.annual_value - y.annual_value
            if gap == 0:
                return None
            return (f"By how much does the annual value of contract {x.ref} exceed "
                    f"that of contract {y.ref}?", gap)
        if kind == 1:
            a, b = self.rng.sample(self.depots, 2)
            if a.bays <= b.bays:
                a, b = b, a
            gap = a.bays - b.bays
            if gap == 0 or gap in (a.bays, b.bays):
                return None
            return (f"How many more loading bays does depot {a.code} have than depot "
                    f"{b.code}?", gap)
        chosen = sorted(self.rng.sample(self.depots, 4), key=lambda d: d.code)
        year = self.rng.choice([2005, 2008, 2010, 2012, 2015])
        count = sum(1 for d in chosen if d.opened < year)
        if count in (0, len(chosen)):
            return None
        names = ", ".join(d.code for d in chosen[:-1]) + f" and {chosen[-1].code}"
        return f"How many of depots {names} opened before {year}?", count

    def unanswerable(self):
        kind = self.rng.randrange(4)
        if kind == 0:
            ref = f"C-{self.rng.randrange(2180, 2200)}"
            return f"What is the annual value of contract {ref}?", NOT_STATED
        if kind == 1:
            ref = f"INC-{self.rng.randrange(4480, 4500)}"
            return f"How many hours late was incident {ref}?", NOT_STATED
        if kind == 2:
            code = f"D{self.rng.randrange(31, 40)}"
            return f"How many loading bays does Meridian Logistics depot {code} have?", NOT_STATED
        depot = self.rng.choice(self.depots)
        field = self.rng.choice(MISSING_FIELDS)
        return f"What is the {field} of Meridian Logistics depot {depot.code}?", NOT_STATED


def build_plus(seed: int = SUITE_SEED, world: World | None = None) -> list[Task]:
    """The hundred extra judge questions. Deterministic for a seed."""
    world = world or build_world()
    _, judge_depots = _halves(world, seed)
    rng = random.Random(seed + 11)
    builder = _PlusBuilder(world, rng, judge_depots)
    taken = {t.question for t in (*TASKS, *JUDGE)}
    each = PLUS_SIZE // len(KINDS)
    groups: dict[str, list[tuple[str, str]]] = {kind: [] for kind in KINDS}
    for kind in KINDS:
        attempts = 0
        while len(groups[kind]) < each:
            attempts += 1
            if attempts > each * 5000:
                raise RuntimeError(f"could not draw {each} distinct {kind} questions")
            drawn = getattr(builder, kind)()
            if drawn is None:
                continue
            question = drawn[0] + ANSWER_RULE
            if question in taken:
                continue
            taken.add(question)
            groups[kind].append((question, str(drawn[1])))

    tasks = []
    for position in range(each):
        for kind in KINDS:
            question, answer = groups[kind][position]
            accepted = ABSTENTIONS if answer == NOT_STATED else (answer,)
            hops = {"temporal": 5, "filtered": 4, "compare": 2, "unanswerable": 1}[kind]
            tasks.append(Task(f"P{len(tasks) + 1:03d}", question, answer, hops, accepted))
    return tasks


def kind_of(task: Task) -> str:
    """Which of the four kinds a judge-plus question is, from its id."""
    return KINDS[(int(task.task_id[1:]) - 1) % len(KINDS)]


JUDGE_PLUS = build_plus()
JUDGE_200 = [*JUDGE, *JUDGE_PLUS]
