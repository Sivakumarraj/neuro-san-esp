"""Starting genomes: the baselines evolution has to beat.

No instruction here names another agent, and that is a hard requirement rather
than a style preference. `remove_agent`, `merge_agents` and `split_agent` change
which agents exist, so an instruction naming an agent a mutation deleted becomes
a lie the network then acts on: the coordinator calls nothing and answers from
the model's own knowledge instead, scoring zero while spending almost no tokens.
Agents are described to each other through their `description` field, which
neuro-san passes to the caller anyway, so naming them in prose is both redundant
and fragile.

`solo` is the floor -- one agent, one tool. `designer_shaped` reproduces what
neuro-san's own `agent_network_designer` produces: exactly one top agent that
connects only to mid-level agents, a shallow DAG, "prefer simplicity, use the
fewest agents necessary" (registries/agent_network_editor.hocon:120-131). It is
a reconstruction of that shape rather than the designer's literal output,
because the designer wires networks against its own toolbox and cannot see this
corpus. The shape is what is being compared, and the shape is faithful.
"""

from __future__ import annotations

from esp.genome.definition import Agent, Genome

_ANSWER_STYLE = (
    "Reply with the answer alone -- a name, a city, a code or a number. "
    "No sentence, no units, no explanation, no thousands separators."
)

# Applied to every agent in every candidate.
#
# Without it, fitness measures the wrong thing. A structurally valid network --
# Coordinator -> Researcher -> CorpusSearch -- scored zero at 2,274 tokens
# because the coordinator looked at its one specialist, decided it could not
# help, and answered "I do not have access to internal company directories"
# from the model's own knowledge. The topology was fine; the model simply
# declined to delegate. Because every candidate carries the same clause this is
# a uniform prior rather than a thumb on the scale for any one shape, and it
# makes the comparison about topology instead of about whether a model felt
# like using its tools on a particular run.
_MUST_DELEGATE = (
    "You have no knowledge of Meridian Logistics yourself, and nothing about it "
    "is in your training data. Never answer from memory and never reply that you "
    "lack access: every fact must come from your tools. If one call is not "
    "enough, call again."
)


def solo() -> Genome:
    return Genome(
        top="Answerer",
        agents={
            "Answerer": Agent(
                name="Answerer",
                description="Answer a question about Meridian Logistics.",
                instructions=(
                    "You answer questions about Meridian Logistics using CorpusSearch.\n"
                    "Search for every identifier in the question (D03, C-2105, INC-4407).\n"
                    "A question often spans two or three documents: find one, read the "
                    "identifier it points to, then search again for that.\n"
                    f"{_MUST_DELEGATE}\n{_ANSWER_STYLE}"
                ),
                can_search=True,
            )
        },
    )


def designer_shaped() -> Genome:
    return Genome(
        top="Coordinator",
        agents={
            "Coordinator": Agent(
                name="Coordinator",
                description="Coordinate a question about Meridian Logistics.",
                instructions=(
                    "You coordinate specialists to answer questions about Meridian "
                    "Logistics.\n"
                    "Choose from the specialists available to you, using their "
                    "descriptions to decide who can help. Questions spanning two "
                    "areas need two calls: ask one, take the identifier it returns, "
                    "then ask the next with that identifier.\n"
                    "Then report the answer.\n"
                    f"{_MUST_DELEGATE}\n{_ANSWER_STYLE}"
                ),
                tools=["DepotSpecialist", "ContractSpecialist", "IncidentSpecialist"],
            ),
            "DepotSpecialist": Agent(
                name="DepotSpecialist",
                description="Look up depots: city, manager, loading bays, opening year.",
                instructions=(
                    "Search the corpus for the depot asked about and report exactly "
                    f"what the document says. {_MUST_DELEGATE}"
                ),
                can_search=True,
            ),
            "ContractSpecialist": Agent(
                name="ContractSpecialist",
                description="Look up contracts: client, depot, goods, penalty rate, annual value.",
                instructions=(
                    "Search the corpus for the contract asked about and report exactly "
                    "what the document says. To compare across all contracts, search "
                    "for 'contract penalty annual value' and read every result. "
                    f"{_MUST_DELEGATE}"
                ),
                can_search=True,
            ),
            "IncidentSpecialist": Agent(
                name="IncidentSpecialist",
                description="Look up incidents: affected contract, hours late, year, cause.",
                instructions=(
                    "Search the corpus for the incident asked about and report exactly "
                    "what the document says, including the contract reference it "
                    f"affected. {_MUST_DELEGATE}"
                ),
                can_search=True,
            ),
        },
    )


def flat_pair() -> Genome:
    """A researcher/checker split -- a shape the designer never produces, because
    it is told to prefer the fewest agents. Included so the seed population is
    not entirely one idea."""
    genome = designer_shaped()
    genome.agents["Coordinator"].tools = ["Researcher", "Arithmetic"]
    genome.agents.pop("DepotSpecialist")
    genome.agents.pop("ContractSpecialist")
    genome.agents.pop("IncidentSpecialist")
    genome.agents["Researcher"] = Agent(
        name="Researcher",
        description="Find the facts needed to answer, following identifiers across documents.",
        instructions=(
            "Search the corpus. Follow every identifier you find to its own document "
            "until you have all the facts. Report the raw facts, not a conclusion. "
            f"{_MUST_DELEGATE}"
        ),
        can_search=True,
    )
    genome.agents["Arithmetic"] = Agent(
        name="Arithmetic",
        description="Multiply, add or compare numbers taken from documents.",
        instructions=(
            "You are given numbers and an operation. Return only the resulting number, "
            "with no separators. Use only the numbers you were handed -- never supply "
            "a figure from memory."
        ),
    )
    return genome


SEEDS: dict[str, callable] = {
    "solo": solo,
    "designer_shaped": designer_shaped,
    "flat_pair": flat_pair,
}
