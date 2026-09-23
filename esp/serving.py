"""The network a person talks to, as opposed to the network that was measured.

Measurement and conversation want different things from the same topology.
Scoring needs an answer it can match exactly, so every measured network tells
its front man to reply with the bare value -- a name, a city, a number -- and
nothing else. That is right for a benchmark and wrong for a person, who asked a
multi-hop question and deserves to see the hops.

So serving changes exactly two things and nothing structural:

* **The reply format.** The one line that demands a bare answer is replaced by
  one that asks for the answer followed by the evidence chain. Agents, wiring,
  tools and per-agent model choices are untouched, so the topology a person
  talks to is the topology that earned the score.
* **The provider, if it has to.** The committed measurements were taken on
  Gemini. A deployment holding only a Claude or GPT key would otherwise refuse
  to start. Each agent keeps its *rung* -- the router that was promoted to the
  stronger model stays promoted -- and moves to the configured provider's model
  for that rung. What that network scores on the new provider has not been
  measured, and every surface that serves one says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from esp.config import cost_tier, model_rank, provider_for
from esp.eval.tasks import TASKS, Task
from esp.genome.definition import DEFAULT_MODEL, MODEL_TIERS, Genome
from esp.genome.seeds import ANSWER_STYLE

# Replaces ANSWER_STYLE on the front man only. Workers already report what the
# documents say; it is the router's reply a person reads.
FULL_ANSWER_STYLE = (
    "Answer in full. Open with the answer itself in one plain sentence. Then "
    "show how it was established: each identifier followed (depot, contract, "
    "incident), what the document for each one says, and any figures combined, "
    "with the calculation written out. If the documents do not settle the "
    "question, say which part is missing instead of guessing."
)

# Format instructions written into the benchmark questions themselves. Shown to
# a person they contradict the full answer the network is now asked for.
_FORMAT_SUFFIX = re.compile(r"\s*Answer with the number only\.?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Served:
    """A servable network, and what distinguishes it from the measured one."""

    genome: Genome
    provider: str
    retargeted: bool

    def note(self, measured_on: str) -> str:
        """One honest sentence for any page that serves this network."""
        if not self.retargeted:
            return (f"Running on {measured_on}, the model it was measured on. "
                    "Only the reply format differs: it explains its answer.")
        return (f"Measured on {measured_on}; running here on {self.provider} with "
                "the same topology and the same agent promoted to the stronger "
                "model. Its score on this provider has not been measured.")


def serving_ladder() -> tuple[str, str]:
    """The cheap and strong models this deployment serves on."""
    return MODEL_TIERS[0], MODEL_TIERS[-1]


def retarget(genome: Genome, cheap: str, strong: str) -> Genome:
    """The same network with every model moved to a new ladder.

    The default keeps its own rung. An agent with a model of its own is placed
    *relative to that default*: ranked above it, the search promoted it, so it
    gets the strong model; ranked below it, the cheap one; equal, the default's
    rung.
    """
    moved = genome.clone()
    ladder = (cheap, strong)
    base = model_rank(genome.default_model)
    default_rung = cost_tier(genome.default_model)
    moved.default_model = ladder[default_rung]
    for agent in moved.agents.values():
        if agent.model is None:
            continue
        rank = model_rank(agent.model)
        rung = 1 if rank > base else 0 if rank < base else default_rung
        agent.model = ladder[rung]
    return moved


def conversational(genome: Genome) -> Genome:
    """The same network, with its front man asked to explain its answer."""
    talking = genome.clone()
    top = talking.agents[talking.top]
    if ANSWER_STYLE in top.instructions:
        top.instructions = top.instructions.replace(ANSWER_STYLE, FULL_ANSWER_STYLE)
    else:
        # A mutation can rewrite instructions. Appending is weaker than
        # replacing, but a network that reached this point without the terse
        # line has nothing to contradict it.
        top.instructions = f"{top.instructions.rstrip()}\n{FULL_ANSWER_STYLE}"
    return talking


def presentable(genome: Genome) -> Served:
    """What a UI should serve for this measured genome, on this deployment."""
    target = provider_for(DEFAULT_MODEL) or "gemini"
    source = provider_for(genome.default_model) or "gemini"
    network = conversational(genome)
    if source == target:
        return Served(network, target, retargeted=False)
    cheap, strong = serving_ladder()
    return Served(retarget(network, cheap, strong), target, retargeted=True)


def measurable(genome: Genome) -> Served:
    """What to *measure* for this genome on this deployment.

    Unlike `presentable`, the reply format is left as measured -- the scorer
    matches the bare value, so explaining would change the protocol, not just
    the manners. Only the provider moves, rung for rung, when it has to; a
    network measured again that way is a new measurement on the new provider,
    which is exactly the one the committed data lacks.
    """
    target = provider_for(DEFAULT_MODEL) or "gemini"
    source = provider_for(genome.default_model) or "gemini"
    if source == target:
        return Served(genome.clone(), target, retargeted=False)
    cheap, strong = serving_ladder()
    return Served(retarget(genome, cheap, strong), target, retargeted=True)


def display_question(task: Task) -> str:
    """A benchmark question as a person should read it."""
    return _FORMAT_SUFFIX.sub("", task.question).strip()


def _showcase() -> list[Task]:
    """One question per difficulty, easiest first.

    Four, chosen by structure rather than by hand: a direct lookup, a two-hop,
    the deepest chain in the set, and one that hops and then calculates.
    Together they exercise every capability the benchmark measures except
    full-corpus aggregation, which nearly every measured network fails and is a
    poor first impression of what the others can do.
    """
    deepest = max(task.hops for task in TASKS)
    picked: list[Task] = []
    wanted = (
        lambda t: t.hops == 1,
        lambda t: t.hops == 2 and "incident" not in t.question.lower(),
        lambda t: t.hops == deepest and "manager" in t.question.lower(),
        lambda t: "penalty owed" in t.question.lower(),
    )
    for want in wanted:
        match = next((t for t in TASKS if want(t) and t not in picked), None)
        if match is not None:
            picked.append(match)
    return picked


SHOWCASE: list[Task] = _showcase()


def graded(question: str) -> Task | None:
    """The benchmark task a question is, if it is one -- matched as displayed."""
    wanted = question.strip()
    for task in TASKS:
        if wanted in (task.question.strip(), display_question(task)):
            return task
    return None
