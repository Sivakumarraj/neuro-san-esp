"""Does a topology chosen on one set of tasks win on tasks it was not chosen on?

Every result this repository reports selects a network on the same seventeen
questions it measures it with. That is the weakest point in the whole argument:
a search that optimises against a fixed task set will find whatever happens to
suit that task set, and with seventeen tasks there is plenty of room for
something that suits them and nothing else. Nobody can tell the difference by
looking at the winner.

It can be answered from the measurements already committed, because each
evaluation recorded the outcome of every individual task. Split the tasks in
two, rank the population on one half, and see where the winner of that half
lands on the other. Repeat over many random splits, because with seventeen
tasks a single split is mostly luck.

What this cannot do is test generalisation to a different *domain*. Held-out
questions from the same generated world are a weaker test than a new world, and
the honest name for what is measured here is stability across questions, not
transfer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from esp.eval import measurements
from esp.evolve.loop import TOKEN_SCALE, WEIGHTS
from esp.surrogate.predictor import _spearman

# Half the tasks select, half assess. An even split maximises the smaller side,
# which is the one that limits what can be concluded: nine selection tasks
# against eight held-out is as much signal as seventeen tasks allow on both
# sides at once.
HELD_OUT_FRACTION = 0.5

# Enough splits that the answer is a distribution rather than an anecdote.
SPLITS = 200


@dataclass(frozen=True)
class Network:
    """One measured network, with its per-task outcomes kept separately."""

    genome_hash: str
    origin: str
    tokens: int
    agents: int
    correct: frozenset[str]
    finished: frozenset[str]

    def accuracy_on(self, tasks: tuple[str, ...]) -> float:
        if not tasks:
            return 0.0
        return sum(1 for task in tasks if task in self.correct) / len(tasks)

    def fitness_on(self, tasks: tuple[str, ...]) -> float:
        """The project's own fitness, restricted to a subset of the tasks.

        Only the accuracy term is recomputed. Tokens and agent count are
        properties of the network rather than of any one task, and the cache
        does not attribute tokens per task, so both penalties stay whole. That
        keeps the cost ordering between networks exactly as the real fitness
        function sees it, which is what matters for a comparison.
        """
        return (WEIGHTS["accuracy"] * self.accuracy_on(tasks)
                - WEIGHTS["tokens"] * min(self.tokens / TOKEN_SCALE, 1.0)
                - WEIGHTS["agents"] * (self.agents / 9.0))


@dataclass
class Outcome:
    """What the repeated splits found."""

    splits: int
    population: int
    selection_tasks: int
    held_out_tasks: int
    # How often the network that won the selection half also won the other half.
    also_won: int = 0
    # Where the selection winner placed on the held-out half, 1 being best.
    ranks: list[int] = field(default_factory=list)
    # The same, for the shape neuro-san's designer produces -- the baseline a
    # reader would have used instead of searching at all.
    designer_ranks: list[int] = field(default_factory=list)
    # Held-out fitness of the selection winner minus that of the designer's
    # shape, per split. Positive means searching beat not searching.
    margins: list[float] = field(default_factory=list)
    winners: dict[str, int] = field(default_factory=dict)
    # How well the whole population's ordering carries across the split, by
    # fitness and by accuracy alone. The pair is the diagnostic: a low fitness
    # correlation beside a healthy accuracy one says the cost terms are
    # deciding the order, not that the networks fail to generalise.
    fitness_rho: list[float] = field(default_factory=list)
    accuracy_rho: list[float] = field(default_factory=list)

    @property
    def mean_fitness_rho(self) -> float:
        if not self.fitness_rho:
            return 0.0
        return sum(self.fitness_rho) / len(self.fitness_rho)

    @property
    def mean_accuracy_rho(self) -> float:
        if not self.accuracy_rho:
            return 0.0
        return sum(self.accuracy_rho) / len(self.accuracy_rho)

    @property
    def also_won_rate(self) -> float:
        return self.also_won / self.splits if self.splits else 0.0

    @property
    def mean_rank(self) -> float:
        return sum(self.ranks) / len(self.ranks) if self.ranks else 0.0

    @property
    def mean_designer_rank(self) -> float:
        if not self.designer_ranks:
            return 0.0
        return sum(self.designer_ranks) / len(self.designer_ranks)

    @property
    def beat_designer_rate(self) -> float:
        """How often the searched winner beat the designer's shape held out."""
        if not self.margins:
            return 0.0
        return sum(1 for margin in self.margins if margin > 0) / len(self.margins)

    @property
    def mean_margin(self) -> float:
        return sum(self.margins) / len(self.margins) if self.margins else 0.0

    def top_ranked_rate(self, within: int) -> float:
        """How often the selection winner landed in the held-out top `within`."""
        if not self.ranks:
            return 0.0
        return sum(1 for rank in self.ranks if rank <= within) / len(self.ranks)


def networks(cache_dir=None) -> list[Network]:
    """The measured population, with per-task outcomes rather than a score."""
    scored = {m.genome_hash: m for m in measurements.load(cache_dir)}
    found: list[Network] = []

    for entry in measurements.raw(cache_dir):
        digest = entry.get("genome_hash")
        record = scored.get(digest)
        if record is None:
            continue        # not a measurement this deployment can rebuild
        results = entry.get("results") or []
        found.append(Network(
            genome_hash=digest,
            origin=record.origin,
            tokens=record.tokens,
            agents=record.agents,
            correct=frozenset(r["task_id"] for r in results if r.get("correct")),
            finished=frozenset(r["task_id"] for r in results
                               if not r.get("infrastructure")),
        ))
    return found


def task_ids(cache_dir=None) -> tuple[str, ...]:
    """The tasks every network in the population faced.

    Intersected rather than taken from the first record: comparing networks
    over tasks only some of them were asked would not be a comparison, and a
    population measured across a task-set change is exactly the case where
    this would silently stop being one.
    """
    per_network = [
        {r["task_id"] for r in (entry.get("results") or [])}
        for entry in measurements.raw(cache_dir)
    ]
    if not per_network:
        return ()
    shared = set.intersection(*per_network) if per_network else set()
    return tuple(sorted(shared))


def analyse(cache_dir=None, splits: int = SPLITS, seed: int = 20260821,
            held_out_fraction: float = HELD_OUT_FRACTION) -> Outcome:
    """Rank the population on half the tasks; score the winner on the rest."""
    population = networks(cache_dir)
    tasks = task_ids(cache_dir)
    if len(population) < 2 or len(tasks) < 4:
        raise ValueError(
            f"need at least two networks and four shared tasks; have "
            f"{len(population)} and {len(tasks)}")

    held_out_size = max(1, round(len(tasks) * held_out_fraction))
    outcome = Outcome(splits=splits, population=len(population),
                      selection_tasks=len(tasks) - held_out_size,
                      held_out_tasks=held_out_size)
    designer = next((n for n in population
                     if n.origin == "seed:designer_shaped"), None)
    rng = random.Random(seed)

    for _split in range(splits):
        shuffled = list(tasks)
        rng.shuffle(shuffled)
        held_out = tuple(shuffled[:held_out_size])
        selection = tuple(shuffled[held_out_size:])

        # Ties broken by genome hash rather than by list order, so the answer
        # does not depend on the order the cache happened to be read in.
        chosen = max(population,
                     key=lambda n: (n.fitness_on(selection), n.genome_hash))
        outcome.winners[chosen.origin] = outcome.winners.get(chosen.origin, 0) + 1

        by_held_out = sorted(population,
                             key=lambda n: (-n.fitness_on(held_out),
                                            n.genome_hash))
        rank = next(i for i, n in enumerate(by_held_out, start=1)
                    if n.genome_hash == chosen.genome_hash)
        outcome.ranks.append(rank)
        if rank == 1:
            outcome.also_won += 1

        if designer is not None:
            designer_rank = next(i for i, n in enumerate(by_held_out, start=1)
                                 if n.genome_hash == designer.genome_hash)
            outcome.designer_ranks.append(designer_rank)
            outcome.margins.append(chosen.fitness_on(held_out)
                                   - designer.fitness_on(held_out))

        # Tie-aware, because accuracy over eight tasks takes nine values and
        # ties are the normal case. The hand-rolled double-argsort this once
        # used gave tied values arbitrary distinct ranks and moved with input
        # order, which on data this coarse is most of the answer.
        outcome.fitness_rho.append(_spearman(
            [n.fitness_on(selection) for n in population],
            [n.fitness_on(held_out) for n in population]))
        outcome.accuracy_rho.append(_spearman(
            [n.accuracy_on(selection) for n in population],
            [n.accuracy_on(held_out) for n in population]))

    return outcome
