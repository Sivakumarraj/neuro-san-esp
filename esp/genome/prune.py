"""Remove agents that are exact copies of a sibling.

`split_agent` gives a specialist a twin with the same instructions, description,
tools and model, and nothing in the operator set ever makes the two diverge. A
caller that can reach both has two identical choices, so the copy adds no
capability: only its description to every prompt its callers read, and a second
route to the same answer. Two of the evolved networks on the Pareto front
carried one (`DepotSpecialist1`).

This is not part of the search. A mutant is never repaired (esp/genome/
mutations.py), because a repaired mutant is not the one the operator produced.
Pruning is a separate, named step applied to a finished network before it is
judged, and the pruned network is a different genome with its own hash, so it
is measured as what it is rather than credited with its parent's score.
"""

from __future__ import annotations

from esp.genome.definition import Agent, Genome
from esp.genome.mutations import check


def _signature(agent: Agent) -> tuple:
    return (agent.instructions, agent.description, tuple(sorted(agent.tools)),
            agent.model, agent.can_search)


def copies(genome: Genome) -> list[str]:
    """Agents that duplicate an earlier sibling exactly, in name order.

    A copy is removable only if every agent that calls it also calls the
    original: otherwise removing it would cut a route that exists nowhere else.
    The top agent is never a copy.
    """
    callers: dict[str, set[str]] = {name: set() for name in genome.agents}
    for name, agent in genome.agents.items():
        for child in agent.tools:
            callers.setdefault(child, set()).add(name)

    kept: dict[tuple, str] = {}
    found: list[str] = []
    for name in sorted(genome.agents):
        if name == genome.top:
            continue
        key = _signature(genome.agents[name])
        original = kept.get(key)
        if original is not None and callers[name] <= callers[original]:
            found.append(name)
        else:
            kept.setdefault(key, name)
    return found


def without_copies(genome: Genome) -> Genome:
    """The same network with every exact copy removed, checked for viability."""
    pruned = genome.clone()
    for name in copies(genome):
        del pruned.agents[name]
        for agent in pruned.agents.values():
            agent.tools = [child for child in agent.tools if child != name]
    check(pruned)
    return pruned
