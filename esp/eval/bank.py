"""A large held-out question bank over the same generated world.

Seventeen questions cannot rank individual networks. Split in half, the winner
of one half averages seventh of twelve on the other, and `make holdout` says so.
The fix FINDINGS names is more questions, and nothing cheaper works: a better
estimator cannot recover a signal the sample size does not contain.

This is that bank. Every answer is computed from the seeded world that also
writes the corpus, the same way `tasks.py` builds the seventeen, so every answer
is correct by construction rather than typed by hand.

Three properties matter more than the count:

* **Held out.** No question here names a depot, contract, client or incident
  that any of the seventeen benchmark questions names. The committed networks
  were selected on those seventeen; judged here, they meet questions about
  entities their selection never touched.
* **Hops are documents.** `hops` is the number of corpus documents that must be
  combined to answer, counted the same way for every template. (The seventeen
  label a few of theirs differently; this bank does not reuse their labels.)
  The deepest questions here combine nine documents.
* **Balanced prefixes.** Questions are interleaved across hop counts, so the
  first N of the bank is as balanced as N allows. A free-tier key can afford a
  few dozen questions a day, and `meridian-bank:40` must still ask every kind.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Callable

from esp.eval.tasks import TASKS, Task
from esp.eval.world import World, build_world

BANK_SIZE = 250
BANK_SEED = 20260925

# How many questions of each depth, out of 250. Weighted toward the multi-hop
# shapes, because one-document lookups are what every topology already does.
MIX = {1: 40, 2: 60, 3: 60, 4: 50, 6: 30, 9: 10}

NUMBER_ONLY = " Answer with the number only."

_REF = re.compile(r"\b(D\d{2}|C-\d{4}|INC-\d{4})\b")


def _benchmark_entities(world: World) -> set[str]:
    """Every entity the seventeen benchmark questions name, directly or through
    a join -- a question about a depot reached from a benchmark contract is not
    held out either."""
    named: set[str] = set()
    for task in TASKS:
        named.update(_REF.findall(task.question))
        named.update(c.client for c in world.contracts if c.client in task.question)
    # Follow the joins the benchmark questions make, so their answers' entities
    # are excluded as well as the ones they print.
    for ref in list(named):
        if ref.startswith("INC-"):
            named.add(next(i.contract_ref for i in world.incidents if i.ref == ref))
    for ref in list(named):
        if ref.startswith("C-"):
            contract = world.contract(ref)
            named.update({contract.client, contract.depot_code})
    for client in [n for n in named if not _REF.fullmatch(n)]:
        for contract in world.contracts:
            if contract.client == client:
                named.update({contract.ref, contract.depot_code})
    return named


class _Builder:
    """Draws questions of one depth until the target count is reached."""

    def __init__(self, world: World, rng: random.Random, excluded: set[str]):
        self.world = world
        self.rng = rng
        self.excluded = excluded
        self.depots = [d for d in world.depots if d.code not in excluded]
        self.contracts = [c for c in world.contracts
                          if c.ref not in excluded and c.client not in excluded
                          and c.depot_code not in excluded]
        refs = {c.ref for c in self.contracts}
        self.incidents = [i for i in world.incidents
                          if i.ref not in excluded and i.contract_ref in refs]

    # ------------------------------------------------------------ one document
    def one(self) -> tuple[str, object]:
        kind = self.rng.randrange(6)
        if kind == 0:
            d = self.rng.choice(self.depots)
            return f"Who is the depot manager of Meridian Logistics depot {d.code}?", d.manager
        if kind == 1:
            d = self.rng.choice(self.depots)
            return f"Which city is Meridian Logistics depot {d.code} located in?", d.city
        if kind == 2:
            d = self.rng.choice(self.depots)
            return (f"How many loading bays does Meridian Logistics depot {d.code} "
                    f"have?{NUMBER_ONLY}", d.bays)
        if kind == 3:
            c = self.rng.choice(self.contracts)
            return f"Which client holds Meridian Logistics contract {c.ref}?", c.client
        if kind == 4:
            c = self.rng.choice(self.contracts)
            return f"What goods are carried under contract {c.ref}?", c.goods
        i = self.rng.choice(self.incidents)
        return f"What was the cause of incident {i.ref}?", i.cause

    # ----------------------------------------------------------- two documents
    def two(self) -> tuple[str, object]:
        w, kind = self.world, self.rng.randrange(6)
        if kind == 0:
            c = self.rng.choice(self.contracts)
            return (f"Which city is the depot that services contract {c.ref} "
                    f"located in?", w.depot(c.depot_code).city)
        if kind == 1:
            c = self.rng.choice(self.contracts)
            return (f"Who manages the depot that services contract {c.ref}?",
                    w.depot(c.depot_code).manager)
        if kind == 2:
            i = self.rng.choice(self.incidents)
            return (f"Which client holds the contract affected by incident "
                    f"{i.ref}?", w.contract(i.contract_ref).client)
        if kind == 3:
            i = self.rng.choice(self.incidents)
            return (f"What goods were carried under the contract affected by "
                    f"incident {i.ref}?", w.contract(i.contract_ref).goods)
        if kind == 4:
            i = self.rng.choice(self.incidents)
            rate = w.contract(i.contract_ref).penalty_per_hour
            return (f"Incident {i.ref} ran late. At the late-delivery penalty "
                    f"rate of its contract, what penalty is owed?{NUMBER_ONLY}",
                    i.hours_late * rate)
        c = self.rng.choice(self.contracts)
        return (f"In what year did the depot that services the contract with "
                f"client {c.client} open?{NUMBER_ONLY}", w.depot(c.depot_code).opened)

    # --------------------------------------------------------- three documents
    def three(self) -> tuple[str, object]:
        w, kind = self.world, self.rng.randrange(4)
        i = self.rng.choice(self.incidents)
        home = w.depot(w.contract(i.contract_ref).depot_code)
        if kind == 0:
            return (f"Incident {i.ref} affected a contract. Which city is the "
                    f"depot for that contract located in?", home.city)
        if kind == 1:
            return (f"Incident {i.ref} affected a contract serviced by a depot. "
                    f"Who is that depot's manager?", home.manager)
        if kind == 2:
            return (f"How many loading bays does the depot have that services "
                    f"the contract affected by incident {i.ref}?{NUMBER_ONLY}", home.bays)
        return (f"Incident {i.ref} affected a contract. In what year did the "
                f"depot servicing that contract open?{NUMBER_ONLY}", home.opened)

    # ---------------------------------------------------------- four documents
    #
    # Comparisons ask for a number, never "which of the two". Scoring checks
    # that the reply contains the answer, so a reply naming both candidates
    # would be marked right whichever it chose. A derived number is also
    # redrawn when it equals one of the values it was built from, or a reply
    # that only reached a component would score as if it had combined them.
    def four(self) -> tuple[str, object] | None:
        w, kind = self.world, self.rng.randrange(3)
        a, b = self.rng.sample(self.incidents, 2)
        owed = [x.hours_late * w.contract(x.contract_ref).penalty_per_hour for x in (a, b)]
        if kind == 0:
            gap = abs(owed[0] - owed[1])
            if gap == 0 or gap in owed:
                return None
            return (f"Incidents {a.ref} and {b.ref} both ran late. Charging each "
                    f"at its own contract's late-delivery penalty rate, by how "
                    f"much does the larger penalty exceed the smaller?{NUMBER_ONLY}", gap)
        if kind == 1:
            return (f"Charging incidents {a.ref} and {b.ref} each at its own "
                    f"contract's late-delivery penalty rate, what is the total "
                    f"penalty owed for the two?{NUMBER_ONLY}", sum(owed))
        x, y = self.rng.sample(self.contracts, 2)
        if x.depot_code == y.depot_code:
            return None
        years = abs(w.depot(x.depot_code).opened - w.depot(y.depot_code).opened)
        if years == 0:
            return None
        return (f"Clients {x.client} and {y.client} each hold a contract serviced "
                f"by a depot. How many years apart did those two depots open?"
                f"{NUMBER_ONLY}", years)

    # ----------------------------------------------------------- six documents
    def six(self) -> tuple[str, object] | None:
        w, kind = self.world, self.rng.randrange(2)
        a, b = self.rng.sample(self.incidents, 2)
        homes = [w.depot(w.contract(x.contract_ref).depot_code) for x in (a, b)]
        if homes[0].code == homes[1].code:
            return None
        if kind == 0:
            total = homes[0].bays + homes[1].bays
            if total in (homes[0].bays, homes[1].bays):
                return None
            return (f"Incidents {a.ref} and {b.ref} each affected a contract "
                    f"serviced by a depot. How many loading bays do those two "
                    f"depots have between them?{NUMBER_ONLY}", total)
        years = abs(homes[0].opened - homes[1].opened)
        if years == 0:
            return None
        return (f"Incidents {a.ref} and {b.ref} each affected a contract serviced "
                f"by a depot. How many years apart did those two depots open?"
                f"{NUMBER_ONLY}", years)

    # ---------------------------------------------------------- nine documents
    def nine(self) -> tuple[str, object] | None:
        w = self.world
        trio = self.rng.sample(self.incidents, 3)
        homes = [w.depot(w.contract(x.contract_ref).depot_code) for x in trio]
        if len({d.code for d in homes}) < 3:
            return None
        bays = [d.bays for d in homes]
        total = sum(bays)
        # A partial sum that equals the answer would let a network that read
        # two of the three depots score as if it had read all three.
        partial = {bays[0] + bays[1], bays[0] + bays[2], bays[1] + bays[2], *bays}
        if total in partial:
            return None
        refs = ", ".join(x.ref for x in trio[:2]) + f" and {trio[2].ref}"
        return (f"Incidents {refs} each affected a contract serviced by a "
                f"different depot. How many loading bays do those three depots "
                f"have in total?{NUMBER_ONLY}", total)


def build_bank(size: int = BANK_SIZE, seed: int = BANK_SEED,
               world: World | None = None) -> list[Task]:
    """Deterministic: the same seed always yields the same questions, in the
    same order."""
    world = world or build_world()
    rng = random.Random(seed)
    builder = _Builder(world, rng, _benchmark_entities(world))
    makers: dict[int, Callable[[], tuple[str, object] | None]] = {
        1: builder.one, 2: builder.two, 3: builder.three,
        4: builder.four, 6: builder.six, 9: builder.nine}

    scale = size / sum(MIX.values())
    wanted = {hops: max(1, round(count * scale)) for hops, count in MIX.items()}
    benchmark = {task.question for task in TASKS}
    seen: set[str] = set(benchmark)
    groups: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for hops, count in wanted.items():
        attempts = 0
        while len(groups[hops]) < count:
            attempts += 1
            if attempts > count * 400:
                raise RuntimeError(f"could not draw {count} distinct {hops}-document "
                                   f"questions from this world")
            drawn = makers[hops]()
            if drawn is None or drawn[0] in seen:
                continue
            seen.add(drawn[0])
            groups[hops].append((drawn[0], str(drawn[1])))

    # Round-robin across depths so every prefix of the bank is balanced.
    ordered: list[tuple[int, str, str]] = []
    depth = max(len(g) for g in groups.values())
    for position in range(depth):
        for hops in sorted(groups):
            if position < len(groups[hops]):
                ordered.append((hops, *groups[hops][position]))
    ordered = ordered[:size]
    return [Task(f"B{n:03d}", question, answer, hops, (answer,))
            for n, (hops, question, answer) in enumerate(ordered, start=1)]


def to_jsonl(tasks: list[Task]) -> str:
    """The bank in the question-file format `make measure` and the page read."""
    import json
    return "".join(json.dumps({"id": t.task_id, "question": t.question,
                               "answers": list(t.accepted or (t.answer,)),
                               "hops": t.hops}) + "\n" for t in tasks)


BANK: list[Task] = build_bank()
