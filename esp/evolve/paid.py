"""The networks of the paid head-to-head, and the one that won it on held-out questions.

These lived in `scripts/select_and_judge.py`, which only the paid runs used. The
web page and `make smoke` need the winner too, and a script is not something the
package should import, so the network definitions live here and the script uses
them.

The winner, `2cc4` with its Researcher on the stronger model, is not stored
anywhere as a file. It is rebuilt the way the paid search made it: the committed
`flat_pair` seed, moved onto the ladder it was measured on, then the same seeded
mutation sequence. `winner()` refuses to return anything whose hash is not the
one in the committed logs, so what is served is provably what was measured.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from esp.genome.definition import Genome

ROOT = Path(__file__).resolve().parent.parent.parent
PAID_RUNS = ROOT / "results" / "headtohead" / "paid-2026-10"
# The replication on the forty fresh questions, with its model control.
REPLICATION = ROOT / "results" / "replication" / "fresh-2026-10"

# The children generated from flat in the committed paid search. The seed and
# the operator list are what produced them; changing either produces others.
SEARCH_SEED = 20261002
SEARCH_CHILDREN = 8
# reassign_model is left out: on this ladder it can only promote an agent to the
# 3.7x dearer rung, and the search was for a cheaper network.
SEARCH_OPERATORS = ["add_agent", "remove_agent", "rewire", "split_agent",
                    "merge_agents", "toggle_search"]

# Networks the committed Gemini search measured, named by what they are. Two of
# them, flat and solo, are hand-written seeds the search started from, not
# networks it found; `origin_of` says which.
COMMITTED = {
    "flat": "c3ee2e0c4a6156c5",
    "mergeA": "39fd57c6e575c298",
    "mergeB": "4c20dbbed359ab7a",
    "pair": "59b12db4896e63d8",
    "solo": "cbe128a617466fe9",
    "searchcoord": "cd06778520823bc3",
}

# The committed measurements of the three hand-written seeds, by the hash they
# were measured under (tests/fixtures/cache/). Fixed, because a hash recomputed
# here carries whatever provider is configured and would match nothing.
SEED_HASHES = {"459ac1a66d925b0c": "designer_shaped",
               "c3ee2e0c4a6156c5": "flat_pair",
               "cbe128a617466fe9": "solo"}

# The ladder every paid stage ran on: cheap rung, strong rung.
MEASURED_LADDER = ("gpt-5.4-nano", "gpt-5.4-mini")

# The held-out winner, as the stage 8 logs record it.
WINNER_NAME = "2cc4"
WINNER_CHILD = "2cc4231b7d3fb81d"          # flat with Arithmetic removed, all on the cheap rung
WINNER_PROMOTED = ("Researcher",)          # moved to the strong rung for stages 7 and 8
WINNER_HASH = "47322479eb9cba1e"           # the genome hash in 8-judge-search10/evolved.jsonl
WINNER_STAGE = "8-judge-search10"
# The designer's shape it was judged against, every agent on the cheap rung.
DESIGNER_HASH = "0378ca26cd9fe977"         # the genome hash in 8-judge-search10/designer.jsonl
# Stage 8 ran with ESP_SEARCH_RESULTS=10. Served at 3 it is a different system.
WINNER_SEARCH_RESULTS = 10


def committed(genome_hash: str) -> Genome:
    """A committed measurement's genome, with exact copies of a specialist removed."""
    from esp.eval import measurements
    from esp.genome.prune import without_copies

    rows = {r.genome_hash: r for r in measurements.load()}
    if genome_hash not in rows:
        raise SystemExit(f"no committed measurement {genome_hash}")
    return without_copies(rows[genome_hash].genome)


def children_of(parent: Genome) -> dict[str, tuple[str, Genome]]:
    """The paid search's children of `parent`, by hash: (operator, genome)."""
    from esp.genome.mutations import InvalidMutant, mutate

    rng = random.Random(SEARCH_SEED)
    seen, out = {parent.genome_hash()}, {}
    for _ in range(500):
        if len(out) == SEARCH_CHILDREN:
            break
        try:
            child, op = mutate(parent, rng, SEARCH_OPERATORS)
        except InvalidMutant:
            continue
        if child.genome_hash() not in seen:
            seen.add(child.genome_hash())
            out[child.genome_hash()] = (op, child)
    return out


def mutants() -> dict[str, tuple[str, Genome]]:
    """The search's children of flat on the configured provider, by hash."""
    from esp.serving import measurable

    return children_of(measurable(committed(COMMITTED["flat"])).genome)


def origin_of(genome_hash: str) -> str:
    """Where a committed network came from: a hand-written seed, or the search."""
    if genome_hash in SEED_HASHES:
        return f"{genome_hash} (hand-written seed:{SEED_HASHES[genome_hash]}), copies removed"
    return f"{genome_hash} (found by the Gemini search), copies removed"


def network(name: str) -> tuple[Genome, str]:
    """A named network, a committed hash, or a child of flat by hash prefix."""
    from esp.genome.seeds import designer_shaped

    if name == "designer":
        return designer_shaped(), "seed:designer_shaped"
    if name in COMMITTED:
        return committed(COMMITTED[name]), origin_of(COMMITTED[name])
    for genome_hash, (op, genome) in mutants().items():
        if genome_hash.startswith(name):
            return genome, f"{genome_hash}: {op} on flat (paid search)"
    # The children's hashes carry the provider, and the logs name them as they
    # were on the measured ladder, so `2cc4` is found there on any provider.
    from esp.serving import retarget

    measured = children_of(retarget(committed(COMMITTED["flat"]), *MEASURED_LADDER))
    for genome_hash, (op, genome) in measured.items():
        if genome_hash.startswith(name):
            return genome, f"{genome_hash}: {op} on flat (paid search, {MEASURED_LADDER[0]})"
    return committed(name), origin_of(name)


def winner() -> Genome:
    """The held-out winner, exactly as stage 8 measured it, whatever the configured provider.

    Raises when the rebuilt genome's hash is not the measured one: a change to the
    seeds, the operators or the mutation code would otherwise serve a different
    network under the winner's name and score.
    """
    from esp.serving import retarget

    flat = retarget(committed(COMMITTED["flat"]), *MEASURED_LADDER)
    found = children_of(flat).get(WINNER_CHILD)
    if found is None:
        raise RuntimeError(f"the paid search no longer produces {WINNER_CHILD}; "
                           "the seeds or the mutation operators have changed")
    genome = found[1].clone()
    for agent in WINNER_PROMOTED:
        genome.agents[agent].model = MEASURED_LADDER[-1]
    if genome.genome_hash() != WINNER_HASH:
        raise RuntimeError(f"rebuilt {genome.genome_hash()}, measured {WINNER_HASH}")
    return genome


def judged_designer() -> Genome:
    """The designer's shape exactly as stage 8 judged it against the winner.

    Raises when the rebuilt hash is not the measured one, as `winner()` does.
    """
    from esp.genome.seeds import designer_shaped
    from esp.serving import retarget

    genome = retarget(designer_shaped(), *MEASURED_LADDER)
    if genome.genome_hash() != DESIGNER_HASH:
        raise RuntimeError(f"rebuilt {genome.genome_hash()}, measured {DESIGNER_HASH}")
    return genome


@dataclass(frozen=True)
class Judged:
    """What the winner and the designer's shape did on the judge questions in stage 8."""

    correct: int
    question_runs: int
    tokens_per_question: int
    cost_per_question: float
    designer_correct: int
    designer_tokens_per_question: int
    designer_cost_per_question: float
    sign_test_p: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.question_runs

    @property
    def designer_accuracy(self) -> float:
        return self.designer_correct / self.question_runs

    @property
    def token_change(self) -> float:
        """Fractional change in tokens a question against the designer: negative is fewer."""
        return self.tokens_per_question / self.designer_tokens_per_question - 1

    @property
    def cost_change(self) -> float:
        """Fractional change in dollars a question, by neuro-san's own count."""
        return self.cost_per_question / self.designer_cost_per_question - 1


def judged(summary: Path = PAID_RUNS / "summary.json") -> Judged:
    """Stage 8's figures, read from the committed summary that the logs reproduce."""
    if not summary.exists():
        raise FileNotFoundError(
            f"{summary} is missing, so the held-out winner's figures cannot be quoted. "
            "Copy results/headtohead/ into this deployment, or set ESP_SERVE=fitness.")
    stage = json.loads(summary.read_text(encoding="utf-8"))[WINNER_STAGE]
    evolved, designer = stage["networks"]["evolved"], stage["networks"]["designer"]
    return Judged(
        correct=evolved["correct"], question_runs=evolved["question_runs"],
        tokens_per_question=evolved["tokens_per_question"],
        cost_per_question=evolved["reported_cost_per_question"],
        designer_correct=designer["correct"],
        designer_tokens_per_question=designer["tokens_per_question"],
        designer_cost_per_question=designer["reported_cost_per_question"],
        sign_test_p=stage["paired"]["sign_test_p"])


@dataclass(frozen=True)
class Control:
    """The replication's model control: the designer's shape with every specialist
    promoted, beside the all-cheap designer and the winner, on the fresh questions."""

    questions: int
    designer_correct: int
    control_correct: int
    question_runs: int
    winner_better_on: int
    control_better_on: int
    sign_test_p: float

    @property
    def gain(self) -> float:
        """What promoting every specialist bought the designer's shape, in accuracy."""
        return (self.control_correct - self.designer_correct) / self.question_runs


def control(summary: Path = REPLICATION / "summary.json") -> Control | None:
    """The model control, read from the committed replication summary.

    None when the summary is not in this deployment: the page then says the
    control exists rather than quoting figures it cannot read.
    """
    from esp.eval.fresh import FRESH

    if not summary.exists():
        return None
    stage = json.loads(summary.read_text(encoding="utf-8"))["1-fresh"]
    networks = stage["networks"]
    pair = stage["pairs"]["designer-promoted vs evolved"]
    return Control(
        questions=len(FRESH), designer_correct=networks["designer"]["correct"],
        control_correct=networks["designer-promoted"]["correct"],
        question_runs=networks["designer-promoted"]["question_runs"],
        winner_better_on=len(pair["better_on"]["evolved"]),
        control_better_on=len(pair["better_on"]["designer-promoted"]),
        sign_test_p=pair["sign_test_p"])
