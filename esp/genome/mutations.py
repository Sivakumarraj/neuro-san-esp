"""Mutation operators over agent networks.

Seven operators, each changing exactly one thing so that a fitness difference
can be attributed. Every mutant is checked for viability and discarded if it
fails -- never repaired. A repaired mutant is a different mutant than the one
the operator produced, and silently substituting it makes the search unable to
learn what its own operators do.
"""

from __future__ import annotations

import random
import re

from esp.genome.definition import MODEL_TIERS, Agent, Genome

MAX_AGENTS = 9
MIN_AGENTS = 1


class InvalidMutant(Exception):
    """The operator produced a network that cannot be served."""


# ------------------------------------------------------------------ viability

def check(genome: Genome) -> None:
    """Reject anything neuro-san could not serve, or that would waste an
    evaluation. Mirrors studio's StructureNetworkValidator: exactly one top
    agent, a DAG, no unreachable agents, no dangling references."""
    if genome.top not in genome.agents:
        raise InvalidMutant("top agent does not exist")

    live = genome.reachable()
    if not MIN_AGENTS <= len(live) <= MAX_AGENTS:
        raise InvalidMutant(f"agent count {len(live)} out of bounds")

    if len(live) != len(genome.agents):
        raise InvalidMutant("unreachable agents present")

    for name in live:
        agent = genome.agents[name]
        for child in agent.tools:
            if child not in genome.agents:
                raise InvalidMutant(f"{name} points at missing agent {child}")
            if child == name:
                raise InvalidMutant(f"{name} points at itself")

    # A leaf with no corpus access is still legitimate -- it reasons over what
    # its caller passes down, which is what the Arithmetic agent does. Only the
    # network as a whole is required to be able to read anything.

    _assert_acyclic(genome)

    # A network with no way to read the corpus can only guess.
    if not genome.searchers():
        raise InvalidMutant("no agent can search")

    _assert_no_stale_agent_names(genome)


# Names an operator may legitimately introduce, plus the corpus tool, which is
# wired in at render time rather than being an agent.
_ALWAYS_PRESENT = {"CorpusSearch"}


def _assert_no_stale_agent_names(genome: Genome) -> None:
    """Reject an instruction that names an agent which no longer exists.

    Found the hard way: a seed whose coordinator was told to call three
    specialists that had been removed called nothing at all, answered from the
    model's own knowledge, and scored zero across every task while spending
    almost no tokens. Structural mutation changes which agents exist, so any
    instruction naming one is a lie waiting to happen. Agents are described to
    their callers by their `description` field, so prose never needs to name
    them.
    """
    known = set(genome.agents) | _ALWAYS_PRESENT
    for name, agent in genome.agents.items():
        for word in re.findall(r"\b[A-Z][A-Za-z]{3,}\d*\b", agent.instructions):
            if word in _KNOWN_AGENT_WORDS and word not in known:
                raise InvalidMutant(
                    f"{name}'s instructions name '{word}', which is not in the network")


# Only words that have ever been used as an agent name are treated as
# references. Without this, ordinary capitalised prose would trip the check.
_KNOWN_AGENT_WORDS = {
    "Coordinator", "Answerer", "Researcher", "Arithmetic",
    "DepotSpecialist", "ContractSpecialist", "IncidentSpecialist",
}


def _assert_acyclic(genome: Genome) -> None:
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {name: WHITE for name in genome.agents}

    def visit(name: str) -> None:
        colour[name] = GREY
        for child in genome.agents[name].tools:
            if child not in colour:
                continue
            if colour[child] == GREY:
                raise InvalidMutant(f"cycle through {child}")
            if colour[child] == WHITE:
                visit(child)
        colour[name] = BLACK

    for name in list(colour):
        if colour[name] == WHITE:
            visit(name)


# ------------------------------------------------------------------ operators

def _fresh_name(genome: Genome, stem: str) -> str:
    index = 1
    while f"{stem}{index}" in genome.agents:
        index += 1
    return f"{stem}{index}"


def add_agent(genome: Genome, rng: random.Random) -> Genome:
    """Attach a new searching specialist under an existing agent."""
    mutant = genome.clone()
    parent = rng.choice(list(mutant.agents))
    name = _fresh_name(mutant, "Specialist")
    mutant.agents[name] = Agent(
        name=name,
        description="Look up facts in the document store and report them verbatim.",
        instructions=("Search the corpus for what you are asked about and report "
                      "exactly what the documents say, including any identifiers "
                      "they reference. You have no knowledge of Meridian "
                      "Logistics yourself: never answer from memory, never reply "
                      "that you lack access, and call again if one search is not "
                      "enough."),
        can_search=True,
    )
    mutant.agents[parent].tools.append(name)
    return mutant


def remove_agent(genome: Genome, rng: random.Random) -> Genome:
    """Drop an agent and re-parent its children, so removal changes the agent
    count without also silently severing a subtree."""
    mutant = genome.clone()
    candidates = [n for n in mutant.agents if n != mutant.top]
    if not candidates:
        raise InvalidMutant("nothing to remove")

    victim = rng.choice(candidates)
    orphans = mutant.agents[victim].tools
    del mutant.agents[victim]
    for agent in mutant.agents.values():
        if victim in agent.tools:
            agent.tools.remove(victim)
            agent.tools.extend(o for o in orphans if o not in agent.tools)
    return mutant


def rewire(genome: Genome, rng: random.Random) -> Genome:
    """Move one edge. The cheapest way to explore topology at fixed size."""
    mutant = genome.clone()
    edges = [(p, c) for p, a in mutant.agents.items() for c in a.tools]
    if not edges:
        raise InvalidMutant("no edges to move")

    parent, child = rng.choice(edges)
    mutant.agents[parent].tools.remove(child)
    new_parent = rng.choice([n for n in mutant.agents if n != child])
    if child not in mutant.agents[new_parent].tools:
        mutant.agents[new_parent].tools.append(child)
    return mutant


def split_agent(genome: Genome, rng: random.Random) -> Genome:
    """Turn one searching agent into a pair, dividing its children.

    Granularity is the axis the designer never explores -- it is told to prefer
    the fewest agents -- so this operator reaches shapes the baseline cannot.
    """
    mutant = genome.clone()
    candidates = [n for n in mutant.agents if n != mutant.top]
    if not candidates:
        raise InvalidMutant("nothing to split")

    original = rng.choice(candidates)
    twin_name = _fresh_name(mutant, original.rstrip("0123456789") or "Agent")
    source = mutant.agents[original]

    twin = source.clone()
    twin.name = twin_name
    kept, moved = source.tools[::2], source.tools[1::2]
    source.tools = kept
    twin.tools = moved
    mutant.agents[twin_name] = twin

    for agent in mutant.agents.values():
        if original in agent.tools and twin_name not in agent.tools:
            agent.tools.append(twin_name)
    return mutant


def merge_agents(genome: Genome, rng: random.Random) -> Genome:
    """Fold one agent into another, combining their children and abilities."""
    mutant = genome.clone()
    candidates = [n for n in mutant.agents if n != mutant.top]
    if len(candidates) < 2:
        raise InvalidMutant("need two agents to merge")

    keep, absorb = rng.sample(candidates, 2)
    keeper, absorbed = mutant.agents[keep], mutant.agents[absorb]
    keeper.can_search = keeper.can_search or absorbed.can_search
    keeper.tools.extend(t for t in absorbed.tools
                        if t not in keeper.tools and t != keep)
    keeper.description = f"{keeper.description} Also: {absorbed.description}"

    del mutant.agents[absorb]
    for agent in mutant.agents.values():
        if absorb in agent.tools:
            agent.tools.remove(absorb)
            if keep not in agent.tools and agent.name != keep:
                agent.tools.append(keep)
    return mutant


def toggle_search(genome: Genome, rng: random.Random) -> Genome:
    """Give an agent the corpus tool, or take it away."""
    mutant = genome.clone()
    name = rng.choice(list(mutant.agents))
    mutant.agents[name].can_search = not mutant.agents[name].can_search
    return mutant


def reassign_model(genome: Genome, rng: random.Random) -> Genome:
    """Change one agent's model.

    neuro-san overlays a per-agent llm_config over the network default and
    nobody tunes it. This is the main cost/quality knob in a multi-agent
    network: most agents in a deep network are doing routing, not reasoning.
    """
    mutant = genome.clone()
    name = rng.choice(list(mutant.agents))
    current = mutant.agents[name].model or mutant.default_model
    choices = [m for m in MODEL_TIERS if m != current]
    if not choices:
        raise InvalidMutant("only one model tier available")
    mutant.agents[name].model = rng.choice(choices)
    return mutant


OPERATORS = {
    "add_agent": add_agent,
    "remove_agent": remove_agent,
    "rewire": rewire,
    "split_agent": split_agent,
    "merge_agents": merge_agents,
    "toggle_search": toggle_search,
    "reassign_model": reassign_model,
}


def mutate(genome: Genome, rng: random.Random,
           operators: list[str] | None = None) -> tuple[Genome, str]:
    """Apply one random viable operator. Raises InvalidMutant if the chosen
    operator produced something unservable -- the caller decides whether to
    retry, and the count of rejects is itself a measurement."""
    names = operators or list(OPERATORS)
    name = rng.choice(names)
    mutant = OPERATORS[name](genome, rng)
    check(mutant)
    return mutant, name
