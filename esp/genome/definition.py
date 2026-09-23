"""The genome: a neuro-san agent network as an evolvable object.

The representation is the designer's own `agent_network_definition` --
`{agent_name: {instructions, description, tools}}` -- with one addition, a
per-agent `model`. neuro-san already overlays a per-agent `llm_config` over the
network default (`calling_activation.py:71`), and nobody optimises that today
even though it is the main cost/quality knob: a twelve-agent network where every
agent runs the top model is mostly waste.

Reusing the designer's representation is what makes mutation and round-tripping
free rather than a subsystem.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from esp.config import (
    DEFAULT_LADDERS,
    configured_provider,
    cost_tier,
    default_model_for,
    provider_for,
)

# A candidate's step and time budget.
#
# The time budget is not a model-speed setting. Evaluation runs tasks concurrently
# while a rate limiter paces every LLM call, and an agent waiting its turn in that
# limiter is spending its own execution budget. At four workers against 14 requests
# per minute, a four-hop task can sit in the queue for minutes and be cancelled --
# and a cancelled agent scores zero, so the search concludes a perfectly good
# topology is bad. The budget therefore has to cover queueing, not just thinking.
# It still bounds a runaway loop, which is what it is really for.
MAX_STEPS = int(os.environ.get("ESP_MAX_STEPS", "40"))
MAX_EXECUTION_SECONDS = int(os.environ.get("ESP_MAX_EXECUTION_SECONDS", "600"))

# The network default, and the models a genome may carry, cheapest first.
#
# Chosen the way neuro-san-studio chooses: by provider, not by model. The
# provider is ESP_PROVIDER if set, otherwise whichever of OpenAI, Anthropic or
# Google holds a usable key, in studio's order (`esp.config.configured_provider`).
# Its default ladder uses neuro-san's version-free aliases wherever they exist,
# so a new release is picked up without an edit. Any model neuro-san resolves
# can be named instead with ESP_DEFAULT_MODEL -- the workers then run exactly
# that, and a promoted router gets the same provider's stronger rung.
#
# With no key and nothing set, both describe the Gemini population the
# committed measurements were taken on, so the offline half of the project --
# search, held-out analysis, the null sweep -- runs on a fresh clone with no key
# and every committed genome hash still matches.
#
# Position in the ladder is what `reassign_model` moves along. Changing either
# setting changes every genome hash, because the model is part of the genome --
# which is deliberate. A fitness measured on one model does not describe a
# network running on another, so the cache must miss.
DEFAULT_MODEL = (os.environ.get("ESP_DEFAULT_MODEL", "").strip()
                 or default_model_for(configured_provider()))


def _default_ladder(model: str) -> str:
    """The configured provider's two rungs, with the chosen worker model first.

    The ladder follows the default model's provider. Setting a Claude default
    used to leave the ladder on Gemini, and a `reassign_model` mutation then
    handed an agent a model the run held no key for: every call inside that
    agent failed, the candidate scored zero, and the search learned that a good
    topology is bad.
    """
    provider = provider_for(model) or "gemini"
    cheap, strong = DEFAULT_LADDERS.get(provider, DEFAULT_LADDERS["gemini"])
    # Only the measured default keeps its ladder untouched, so committed genome
    # hashes still match. Any other Gemini model used to keep it too: a run on
    # gemini-3.8-flash promoted its router to the older gemini-3.5-flash.
    if model not in (cheap, strong) and model != default_model_for(provider):
        # The chosen model takes the rung it belongs on: a cheap one becomes
        # what the workers run, a strong one what a router is promoted to.
        # Putting claude-opus below claude-sonnet would make every promotion
        # a downgrade.
        if cost_tier(model):
            strong = model
        else:
            cheap = model
    return f"{cheap},{strong}"


MODEL_TIERS: list[str] = [
    name.strip() for name in os.environ.get(
        "ESP_MODEL_TIERS", _default_ladder(DEFAULT_MODEL)).split(",")
    if name.strip()
]

CORPUS_TOOL = "CorpusSearch"


@dataclass
class Agent:
    name: str
    instructions: str
    description: str
    tools: list[str] = field(default_factory=list)   # names of downstream agents
    model: str | None = None                          # None = inherit network default
    can_search: bool = False                          # gets the corpus tool

    def clone(self) -> Agent:
        return Agent(self.name, self.instructions, self.description,
                     list(self.tools), self.model, self.can_search)


@dataclass
class Genome:
    """A candidate agent network. `top` is the front man the request enters through."""

    top: str
    agents: dict[str, Agent]
    default_model: str = DEFAULT_MODEL

    # ---------------------------------------------------------------- identity

    def canonical(self) -> dict[str, Any]:
        """A stable, order-independent view. Two genomes that differ only in dict
        ordering are the same network and must hash the same, or the fitness
        cache leaks and we pay twice for one topology."""
        return {
            "top": self.top,
            "default_model": self.default_model,
            "agents": {
                name: {
                    "instructions": agent.instructions.strip(),
                    "description": agent.description.strip(),
                    "tools": sorted(agent.tools),
                    "model": agent.model,
                    "can_search": agent.can_search,
                }
                for name, agent in sorted(self.agents.items())
            },
        }

    @classmethod
    def from_canonical(cls, blob: dict[str, Any]) -> Genome:
        """Rebuild a genome from `canonical()`.

        The inverse existed nowhere, and its absence was a real limitation
        rather than an oversight: the service recorded a candidate's hash and
        its score but not the candidate, so an evolved winner could be measured
        and could never be served again. `serve_champion` could only resurrect
        seeds, because seeds are the one thing it can rebuild from source.

        Round-tripping is asserted by a test rather than assumed -- a
        reconstruction that hashes differently is worse than none, because it
        would silently serve a network that is not the one that earned the
        score.
        """
        return cls(
            top=blob["top"],
            agents={
                name: Agent(
                    name=name,
                    instructions=fields["instructions"],
                    description=fields["description"],
                    tools=list(fields.get("tools") or []),
                    model=fields.get("model"),
                    can_search=bool(fields.get("can_search")),
                )
                for name, fields in blob["agents"].items()
            },
            default_model=blob["default_model"],
        )

    def genome_hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def clone(self) -> Genome:
        return Genome(self.top, {n: a.clone() for n, a in self.agents.items()},
                      self.default_model)

    # ---------------------------------------------------------------- structure

    def children(self, name: str) -> list[str]:
        return self.agents[name].tools

    def reachable(self) -> set[str]:
        seen: set[str] = set()
        stack = [self.top]
        while stack:
            current = stack.pop()
            if current in seen or current not in self.agents:
                continue
            seen.add(current)
            stack.extend(self.agents[current].tools)
        return seen

    def depth(self) -> int:
        """Longest path from the top agent. Cycles are rejected before this runs."""
        memo: dict[str, int] = {}

        def walk(name: str, seen: frozenset[str]) -> int:
            if name in memo:
                return memo[name]
            agent = self.agents.get(name)
            if agent is None or not agent.tools:
                return 1
            best = 1 + max(
                (walk(child, seen | {name}) for child in agent.tools
                 if child in self.agents and child not in seen),
                default=0,
            )
            memo[name] = best
            return best

        return walk(self.top, frozenset())

    def searchers(self) -> list[str]:
        return [n for n, a in self.agents.items() if a.can_search]

    # ---------------------------------------------------------------- rendering

    def to_hocon(self) -> str:
        """Render a network neuro-san can serve.

        Only reachable agents are emitted. An unreachable agent costs nothing at
        runtime, but leaving it in would let dead weight accumulate across
        generations and quietly distort the parsimony objective.
        """
        live = self.reachable()
        blocks: list[str] = []

        # The top agent must be emitted first. neuro-san takes the first entry
        # in `tools` as the front man, so ordering here is not cosmetic -- it
        # decides which agent the request actually enters through. Emitting in
        # plain alphabetical order silently handed the network to whichever
        # agent sorted first: `designer_shaped` ran with ContractSpecialist as
        # its front man and `flat_pair` with Arithmetic, which answered every
        # question with "you did not provide any numbers". Both scored as bad
        # topologies when neither had ever been run as written. The rest stay
        # sorted so rendering is deterministic and the file diffs cleanly.
        ordered = [self.top, *sorted(n for n in live if n != self.top)]
        for name in ordered:
            agent = self.agents[name]
            lines: list[str] = ["        {", f'            "name": "{name}",']

            lines.append('            "function": {')
            lines.append(f'                "description": {json.dumps(agent.description)},')
            if name != self.top:
                lines.append('                "parameters": {')
                lines.append('                    "type": "object",')
                lines.append('                    "properties": {')
                lines.append('                        "inquiry": {"type": "string",')
                lines.append('                            "description":'
                             ' "What you need from this agent."}')
                lines.append("                    },")
                lines.append('                    "required": ["inquiry"]')
                lines.append("                },")
            lines.append("            },")

            lines.append(f'            "instructions": {json.dumps(agent.instructions)},')

            downstream = [t for t in agent.tools if t in live]
            if agent.can_search:
                downstream = [*downstream, CORPUS_TOOL]
            if downstream:
                rendered = ", ".join(f'"{t}"' for t in sorted(set(downstream)))
                lines.append(f'            "tools": [{rendered}],')

            if agent.model:
                lines.append('            "llm_config": '
                             f'{{"model_name": "{agent.model}"}},')

            lines.append("        },")
            blocks.append("\n".join(lines))

        if any(self.agents[n].can_search for n in live):
            blocks.append(_CORPUS_TOOL_BLOCK)

        return (
            "{\n"
            f'    "llm_config": {{"model_name": "{self.default_model}"}},\n'
            '    "metadata": {"description": "ESP candidate network."},\n'
            f'    "max_steps": {MAX_STEPS},\n'
            f'    "max_execution_seconds": {MAX_EXECUTION_SECONDS},\n'
            '    "tools": [\n'
            + "\n".join(blocks)
            + "\n    ]\n}\n"
        )


_CORPUS_TOOL_BLOCK = """        {
            "name": "CorpusSearch",
            "function": {
                "description": "Search the Meridian Logistics document store. Returns whole documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                            "description": "Terms to search for. Include any identifier such as D03, C-2105 or INC-4407."}
                    },
                    "required": ["query"]
                },
            },
            "class": "esp.eval.corpus_tool.CorpusSearch",
        },"""
