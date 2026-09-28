"""A context for prescription: what questions a network will face, and what an
answer may cost.

ESP prescribes actions for a context, and the Prescriptor is the model that
maps one to the other. Every search in this repository ran against one fixed
question set, so there was no context and nothing for a Prescriptor to take as
input (docs/FINDINGS.md, "there is no context"). The per-question Predictor
changes that. It predicts a network's chance on each kind of question, so a
workload -- a share of each kind -- and a price per answer are a context, and
"which network for this workload" becomes a question with an answer that
depends on it.

This module is the seam, and only the seam:

* `Context` states a workload and an optional price cap.
* `prescribe` ranks candidate networks for a context by the Predictor's
  derived fitness: accuracy on the context's mix of kinds, less dollars, less
  size. It is an argmax over a list someone else bred. It learns nothing.
* `strategy` wraps that as a pool strategy (`esp.evolve.pool.Scorer`), so the
  pool benchmark can already ask whether knowing the workload chooses better.

A learned Prescriptor, which is what ESP means by one, would replace the argmax
with a model that maps a context to a network, evolved against the Predictor.
That is future work and is not claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from esp.eval.pricing import fitness_dollars
from esp.eval.tasks import Task
from esp.genome.definition import Genome
from esp.surrogate.per_question import QuestionPredictor, question_features

KINDS = ("join", "aggregate", "temporal", "filtered", "compare", "unanswerable")


def kind(task: Task) -> str:
    """The kind of a question, read from the same features the Predictor uses."""
    flags = question_features(task)
    for name, index in (("unanswerable", 5), ("temporal", 2), ("filtered", 3),
                        ("compare", 4), ("aggregate", 1)):
        if flags[index]:
            return name
    return "join"


@dataclass(frozen=True)
class Context:
    """A workload -- the share of each kind of question -- and a price cap."""

    workload: dict[str, float] = field(default_factory=lambda: {"join": 1.0})
    max_dollars_per_question: float | None = None

    def __post_init__(self):
        unknown = set(self.workload) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown question kinds {sorted(unknown)}; kinds are {KINDS}")
        if not any(share > 0 for share in self.workload.values()):
            raise ValueError("a workload needs at least one kind with a positive share")


def predicted_accuracy(model: QuestionPredictor, genomes: list[Genome], tasks: list[Task],
                       context: Context) -> np.ndarray:
    """Accuracy on the context's mix: each kind predicted on its own questions,
    weighted by its share. A kind with no example question is ignored, and
    said to be by raising if that leaves nothing."""
    by_kind: dict[str, list[Task]] = {}
    for task in tasks:
        by_kind.setdefault(kind(task), []).append(task)
    total, weight = np.zeros(len(genomes)), 0.0
    for name, share in context.workload.items():
        if share <= 0 or name not in by_kind:
            continue
        mean, _ = model.accuracy(genomes, by_kind[name])
        total += share * mean
        weight += share
    if weight == 0:
        raise ValueError("none of the context's kinds has an example question")
    return total / weight


def prescribe(model: QuestionPredictor, genomes: list[Genome], tasks: list[Task],
              context: Context) -> list[int]:
    """Indices of `genomes`, best first for this context. Networks predicted
    to cost more than the cap come last, in their own order."""
    accuracy = predicted_accuracy(model, genomes, tasks, context)
    dollars = model.dollars(genomes)
    agents = [len(g.reachable()) for g in genomes]
    fitness = np.array([fitness_dollars(a, d, n)
                        for a, d, n in zip(accuracy, dollars, agents, strict=True)])
    over = (np.zeros(len(genomes), dtype=bool) if context.max_dollars_per_question is None
            else dollars > context.max_dollars_per_question)
    return list(np.lexsort((-fitness, over)))


def strategy(context: Context, members: int = 3, max_iter: int = 60):
    """`prescribe` as a pool strategy: trained on the measured networks'
    visible answers, it scores the unmeasured ones for this context."""
    def score(known, unknown, seen, seed):
        model = QuestionPredictor(seed=seed, members=members, max_iter=max_iter)
        model.fit([m.observed({t.task_id for t in seen}) for m in known], seen)
        genomes = [m.observed().genome for m in unknown]
        order = prescribe(model, genomes, seen, context)
        scores = np.empty(len(genomes))
        scores[order] = np.arange(len(genomes), 0, -1)
        return scores
    score.__name__ = "context:" + ",".join(f"{k}={v:g}" for k, v in context.workload.items())
    return score
