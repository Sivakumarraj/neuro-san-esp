"""Genome identity, structure and rendering."""

from __future__ import annotations

import pytest

from esp.genome.definition import Agent, Genome
from esp.genome.seeds import SEEDS, designer_shaped, solo


def test_hash_is_order_independent():
    """Two genomes differing only in dict insertion order are the same network.
    If they hashed differently the fitness cache would miss and every re-visit
    would be paid for again."""
    first = designer_shaped()
    second = Genome(first.top, dict(reversed(list(first.agents.items()))),
                    first.default_model)
    assert first.genome_hash() == second.genome_hash()


def test_hash_changes_when_a_model_changes():
    genome = designer_shaped()
    before = genome.genome_hash()
    genome.agents["DepotSpecialist"].model = "gemini-3.1-flash-lite"
    assert genome.genome_hash() != before


def test_clone_is_independent():
    original = designer_shaped()
    copy = original.clone()
    copy.agents["Coordinator"].tools.append("Nonsense")
    assert "Nonsense" not in original.agents["Coordinator"].tools


def test_unreachable_agents_are_not_rendered():
    genome = solo()
    genome.agents["Orphan"] = Agent("Orphan", "does nothing", "orphan")
    assert "Orphan" not in genome.reachable()
    assert "Orphan" not in genome.to_hocon()


def test_depth_terminates_on_a_diamond():
    """Two paths converging on one agent must not be walked twice forever."""
    genome = Genome("Top", {
        "Top": Agent("Top", "i", "d", tools=["Left", "Right"]),
        "Left": Agent("Left", "i", "d", tools=["Shared"]),
        "Right": Agent("Right", "i", "d", tools=["Shared"]),
        "Shared": Agent("Shared", "i", "d", can_search=True),
    })
    assert genome.depth() == 3


@pytest.mark.parametrize("name", sorted(SEEDS))
def test_seed_renders_valid_hocon(name):
    from pyhocon import ConfigFactory

    text = SEEDS[name]().to_hocon()
    config = ConfigFactory.parse_string(text)
    tools = config.get("tools")
    assert len(tools) >= 1
    names = {tool.get("name") for tool in tools}
    assert SEEDS[name]().top in names


def test_corpus_tool_block_appears_only_when_needed():
    """Look for the tool *definition*, not the name: seed instructions mention
    CorpusSearch in prose, so a substring check passes even with no tool wired."""
    marker = '"class": "esp.eval.corpus_tool.CorpusSearch"'

    genome = solo()
    assert marker in genome.to_hocon()

    genome.agents["Answerer"].can_search = False
    assert marker not in genome.to_hocon()


def test_top_agent_is_rendered_first():
    """neuro-san takes the first entry in `tools` as the front man, so this is
    not cosmetic -- it decides which agent the request enters through.

    Rendering in plain alphabetical order silently handed the network to
    whichever agent sorted first. `designer_shaped` ran with ContractSpecialist
    as its front man and `flat_pair` with Arithmetic, which answered every
    question with "you did not provide any numbers". Both were recorded as bad
    topologies without ever having been run as written.
    """
    import re

    from esp.genome.seeds import SEEDS

    for name, build in SEEDS.items():
        genome = build()
        first = re.search(r'"name": "([^"]+)"', genome.to_hocon()).group(1)
        assert first == genome.top, f"{name}: rendered {first}, top is {genome.top}"


def test_top_agent_is_first_even_when_it_sorts_last():
    from esp.genome.definition import Agent, Genome

    genome = Genome(
        top="Zeta",
        agents={
            "Zeta": Agent("Zeta", "lead", "the top agent", tools=["Alpha"]),
            "Alpha": Agent("Alpha", "help", "a worker", can_search=True),
        },
    )
    names = [line.split('"')[3] for line in genome.to_hocon().splitlines()
             if '"name":' in line]
    assert names[0] == "Zeta", names
