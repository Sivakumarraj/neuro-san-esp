"""Evaluation tasks, derived from the same world that produced the corpus.

Each task records how many documents must be combined to answer it. That matters:
a topology that looks strong on one-hop lookups can collapse at four, and a task
set that only asks easy questions will happily evolve a network that cannot do
the job.

The difficulty here was set empirically, not guessed. A first version of this
file was answered perfectly by a single agent with a single tool, which left
evolution nothing to improve -- so the corpus was widened from 40 documents to
124, retrieval was narrowed from five results to three, and the aggregate and
four-hop shapes below were added.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from esp.eval.world import World, build_world


@dataclass(frozen=True)
class Task:
    """One question, and every answer that is correct for it.

    `answer` is the canonical one, for display. `accepted` is the whole set,
    and it exists because "correct by construction" was not. The aggregate
    question "which contract reference appears in the most incident reports?"
    has twenty equally correct answers -- twenty contracts tied on two
    incidents each -- and `Counter.most_common(1)` handed back whichever
    insertion order put first. Nineteen correct answers were scored wrong.

    That is precisely the poisoned fitness function this file's own docstring
    warns about, arrived at through generated ground truth rather than
    hand-written, which is the failure the generation was supposed to prevent.
    """

    task_id: str
    question: str
    answer: str
    hops: int
    accepted: tuple[str, ...] = ()


def normalise(text: str) -> str:
    """Compare answers the way a person would: case, padding, punctuation and
    thousands separators are noise."""
    text = text.strip().lower()
    text = re.sub(r"[,$]", "", text)
    text = re.sub(r"[.!?]+$", "", text)
    return re.sub(r"\s+", " ", text)


def _matches(expected: str, produced: str) -> bool:
    """Containment rather than equality, because an agent asked for a number
    will often say "the total is 4200"; requiring bare equality would punish
    phrasing instead of measuring correctness.

    Numbers are matched on whole-token boundaries instead. Plain containment
    scored an expected answer of "4" as correct against any text holding a
    stray 4 -- including the digits inside "INC-4429" -- which would have handed
    free marks to networks that never found the document. Names and identifiers
    are distinctive enough that substring matching is safe for them.
    """
    want, got = normalise(expected), normalise(produced)
    if re.fullmatch(r"\d+(?:\.\d+)?", want):
        return re.search(rf"(?<![\w.-]){re.escape(want)}(?![\w.-])", got) is not None
    return want in got


def score(expected: str | Sequence[str], produced: str) -> bool:
    """True if any correct answer is present.

    A sequence, because some questions genuinely have more than one right
    answer and marking nineteen of twenty wrong does not measure a topology,
    it measures which one it happened to name.
    """
    if not produced:
        return False
    options = (expected,) if isinstance(expected, str) else tuple(expected)
    return any(_matches(one, produced) for one in options)


def build_tasks(world: World | None = None) -> list[Task]:
    world = world or build_world()
    tasks: list[Task] = []

    def add(question: str, answer: object, hops: int,
            accepted: Sequence[object] | None = None) -> None:
        """`accepted` is every answer that is correct, not just the one the
        generator happened to pick out of a tie."""
        options = tuple(str(a) for a in (accepted if accepted else [answer]))
        tasks.append(Task(f"T{len(tasks) + 1:02d}", question, str(answer),
                          hops, options))

    by_contract = Counter(i.contract_ref for i in world.incidents)

    # --- one hop
    depot = world.depots[7]
    add(f"Who is the depot manager of Meridian Logistics depot {depot.code}?",
        depot.manager, 1)

    # --- two hops: contract -> depot
    for index in (0, 17, 31):
        contract = world.contracts[index]
        add(f"Which city is the depot that services contract {contract.ref} "
            f"located in?", world.depot(contract.depot_code).city, 2)

    contract = world.contracts[23]
    add(f"Who manages the depot that services the contract with client "
        f"{contract.client}?", world.depot(contract.depot_code).manager, 2)

    # --- aggregates: every contract must be compared, so a single search
    #     returning three documents can never be enough
    dearest = max(world.contracts, key=lambda c: c.penalty_per_hour)
    add("Across all forty Meridian Logistics contracts, which contract "
        "reference has the highest late-delivery penalty per hour?",
        dearest.ref, 3)

    biggest = max(world.contracts, key=lambda c: c.annual_value)
    add("Which client holds the Meridian Logistics contract with the highest "
        "annual value?", biggest.client, 3)

    # Twenty contracts tie on two incidents each in this world, so
    # `most_common(1)` was a one-in-twenty lottery presented as ground truth.
    # Every tied reference is accepted, because every tied reference is right.
    worst_count = max(by_contract.values())
    tied_refs = sorted(ref for ref, count in by_contract.items()
                       if count == worst_count)
    add("Which contract reference appears in the most incident reports?",
        tied_refs[0], 3, accepted=tied_refs)

    # --- three hops: incident -> contract -> depot
    for index in (2, 41):
        incident = world.incidents[index]
        linked = world.contract(incident.contract_ref)
        add(f"Incident {incident.ref} affected a contract. Which city is the "
            f"depot for that contract located in?",
            world.depot(linked.depot_code).city, 3)

    # --- four hops: incident -> contract -> depot -> manager, plus a field
    for index in (13, 55):
        incident = world.incidents[index]
        linked = world.contract(incident.contract_ref)
        home = world.depot(linked.depot_code)
        add(f"Incident {incident.ref} affected a contract serviced by a depot. "
            f"Who is that depot's manager?", home.manager, 4)

    incident = world.incidents[29]
    linked = world.contract(incident.contract_ref)
    home = world.depot(linked.depot_code)
    add(f"How many loading bays does the depot have that serviced the contract "
        f"affected by incident {incident.ref}? Answer with the number only.",
        home.bays, 4)

    # --- join plus arithmetic
    for index in (1, 34):
        incident = world.incidents[index]
        linked = world.contract(incident.contract_ref)
        add(f"Incident {incident.ref} ran late. Using the late-delivery penalty "
            f"rate of its contract, what is the total penalty owed? Answer with "
            f"the number only.",
            incident.hours_late * linked.penalty_per_hour, 2)

    # --- arithmetic across several incidents on one contract
    target_ref = by_contract.most_common(2)[1][0]
    total_hours = sum(i.hours_late for i in world.incidents
                      if i.contract_ref == target_ref)
    add(f"Adding up every incident recorded against contract {target_ref}, how "
        f"many hours late is it in total? Answer with the number only.",
        total_hours, 3)

    incident = world.incidents[47]
    linked = world.contract(incident.contract_ref)
    add(f"What goods were being carried under the contract affected by incident "
        f"{incident.ref}?", linked.goods, 2)

    return tasks


TASKS: list[Task] = build_tasks()
