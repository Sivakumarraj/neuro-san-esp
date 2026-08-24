"""Mutation operators and the viability gate."""

from __future__ import annotations

import contextlib
import random

import pytest

from esp.genome.definition import Agent, Genome
from esp.genome.mutations import OPERATORS, InvalidMutant, check, mutate, reassign_model
from esp.genome.seeds import designer_shaped, solo


def test_every_survivor_of_a_thousand_mutations_is_viable():
    """The headline guarantee: a mutant that reaches evaluation can be served.
    An unservable mutant burns an evaluation slot and scores a misleading zero."""
    rng = random.Random(11)
    base = designer_shaped()
    survivors = 0
    for _ in range(1000):
        try:
            mutant, _ = mutate(base, rng)
        except InvalidMutant:
            continue
        check(mutant)          # must not raise
        survivors += 1
    assert survivors > 800


def test_the_gate_actually_rejects():
    """A check that never fires is indistinguishable from no check at all."""
    rng = random.Random(3)
    rejected = 0
    for _ in range(400):
        try:
            mutate(solo(), rng)
        except InvalidMutant:
            rejected += 1
    assert rejected > 50


def test_network_without_a_searcher_is_rejected():
    genome = solo()
    genome.agents["Answerer"].can_search = False
    with pytest.raises(InvalidMutant, match="no agent can search"):
        check(genome)


def test_cycle_is_rejected():
    genome = Genome("A", {
        "A": Agent("A", "i", "d", tools=["B"]),
        "B": Agent("B", "i", "d", tools=["A"], can_search=True),
    })
    with pytest.raises(InvalidMutant, match="cycle"):
        check(genome)


def test_dangling_reference_is_rejected():
    genome = solo()
    genome.agents["Answerer"].tools.append("Ghost")
    with pytest.raises(InvalidMutant, match="missing agent"):
        check(genome)


def test_unreachable_agent_is_rejected():
    genome = solo()
    genome.agents["Stranded"] = Agent("Stranded", "i", "d", can_search=True)
    with pytest.raises(InvalidMutant, match="unreachable"):
        check(genome)


def test_remove_agent_reparents_children():
    """Removing a middle agent must not silently sever the subtree beneath it,
    or the operator would be doing two things at once and its effect on fitness
    could not be attributed."""
    genome = Genome("Top", {
        "Top": Agent("Top", "i", "d", tools=["Middle"]),
        "Middle": Agent("Middle", "i", "d", tools=["Leaf"]),
        "Leaf": Agent("Leaf", "i", "d", can_search=True),
    })
    rng = random.Random(0)
    reparented = 0
    for _ in range(80):
        mutant = OPERATORS["remove_agent"](genome, rng)
        # The operator picks its victim at random, so Leaf itself is sometimes
        # the one removed. What must never happen is Middle going and taking
        # Leaf's reachability with it.
        if "Middle" not in mutant.agents and "Leaf" in mutant.agents:
            assert "Leaf" in mutant.reachable()
            reparented += 1
    assert reparented > 0, "never exercised the re-parenting path"


def test_reassign_model_changes_exactly_one_agent():
    genome = designer_shaped()
    mutant = reassign_model(genome, random.Random(5))
    changed = [n for n in genome.agents
               if genome.agents[n].model != mutant.agents[n].model]
    assert len(changed) == 1


def test_mutation_never_mutates_its_parent():
    rng = random.Random(9)
    base = designer_shaped()
    before = base.genome_hash()
    for _ in range(200):
        with contextlib.suppress(InvalidMutant):
            mutate(base, rng)
    assert base.genome_hash() == before


def test_instructions_naming_a_removed_agent_are_rejected():
    """The bug this guards cost a whole seed: a coordinator told to call three
    specialists the seed had removed called nothing, answered from the model's
    own knowledge, and scored zero on every task at almost no token cost --
    which reads as a cheap, terrible topology rather than a broken one."""
    genome = solo()
    genome.agents["Answerer"].instructions += " Delegate to DepotSpecialist."
    with pytest.raises(InvalidMutant, match="not in the network"):
        check(genome)


def test_no_seed_instruction_names_another_agent():
    from esp.genome.seeds import SEEDS

    for name, make in SEEDS.items():
        genome = make()
        for agent in genome.agents.values():
            for other in genome.agents:
                if other != agent.name:
                    assert other not in agent.instructions, \
                        f"{name}/{agent.name} names {other}"


def test_every_agent_in_every_seed_refuses_to_answer_from_memory():
    """A uniform prior, not a tuning knob for one shape.

    A valid Coordinator -> Researcher -> CorpusSearch network once scored zero
    at 2,274 tokens because the coordinator decided its single specialist could
    not help and answered from the model's own knowledge instead. Fitness was
    then measuring whether a model felt like delegating rather than measuring
    the topology. Every candidate carries the clause, so no shape is favoured.
    """
    from esp.genome.seeds import SEEDS

    for name, make in SEEDS.items():
        for agent in make().agents.values():
            assert "memory" in agent.instructions.lower(), f"{name}/{agent.name}"


def test_agents_created_by_mutation_carry_the_prior_too():
    genome = OPERATORS["add_agent"](solo(), random.Random(0))
    new = [a for a in genome.agents.values() if a.name.startswith("Specialist")]
    assert new and all("memory" in a.instructions.lower() for a in new)
