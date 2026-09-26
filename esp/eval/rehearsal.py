"""A simulated provider, so the whole paid run can be rehearsed for $0.

It stands in for `esp.eval.runner.run_suite`: same arguments, same `SuiteRun`
back. Answers are coin flips at a chance planted from the network's structure
and the question's kind, and tokens and dollars are drawn to the scale v1
actually measured (about 6,600 to 22,700 tokens a question, $0.29 to $0.50 a
million). Everything is deterministic for a network and a question.

Nothing it produces is a result. It exists to prove the pipeline -- measuring,
caching, stopping for quota and resuming, the replicate comparison, judging --
before real money goes through it, and to count exactly how many question-runs
the real run will make. Every number it writes is labelled simulated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from esp.eval.runner import QuotaExhausted, SuiteRun, TaskResult
from esp.genome.definition import Genome
from esp.surrogate.per_question import question_features
from esp.surrogate.predictor import features


def _unit(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def planted_chance(genome: Genome, question) -> float:
    """A plausible, known relationship: totals and filters reward more agents,
    simple joins reward fewer, a stronger router helps everything a little."""
    f, q = features(genome), question_features(question)
    agents, depth, max_tier = f[0], f[1], f[10]
    if q[1] or q[3] or q[2]:                    # aggregate, filtered, temporal
        logit = -0.4 + 0.45 * min(agents - 3, 3) - 0.15 * depth
    elif q[5]:                                  # unanswerable
        logit = 0.8 - 0.25 * (agents - 3)
    else:                                       # joins and comparisons
        logit = 1.8 - 0.3 * (agents - 3) - 0.07 * q[0]
    return float(1 / (1 + np.exp(-(logit + 0.35 * max_tier))))


class SimulatedProvider:
    """Callable with `run_suite`'s signature. Counts what it was asked."""

    def __init__(self, genomes: list[Genome], fail_after: int | None = None):
        self.by_hash = {g.genome_hash(): g for g in genomes}
        self.fail_after = fail_after
        self.suites = 0
        self.question_runs = 0
        self.tokens = 0
        self.dollars = 0.0

    def add(self, genome: Genome) -> None:
        self.by_hash[genome.genome_hash()] = genome

    def __call__(self, hocon_path, tasks, **_) -> SuiteRun:
        if self.fail_after is not None and self.suites >= self.fail_after:
            raise QuotaExhausted("simulated daily cap")
        self.suites += 1
        digest = Path(hocon_path).stem
        genome = self.by_hash[digest]
        f = features(genome)
        price = 0.29 + 0.12 * max(f[10] - f[9], 0) + 0.05 * f[9]
        results, tokens = [], 0
        for task in tasks:
            right = _unit(digest, task.task_id, "answer") < planted_chance(genome, task)
            spend = int(4_000 + 1_900 * f[0] + 900 * task.hops
                        + 3_000 * _unit(digest, task.task_id, "tokens"))
            tokens += spend
            results.append(TaskResult(task.task_id, task.hops, right, 1.0,
                                      task.answer if right else "simulated miss"))
        dollars = tokens / 1e6 * price
        self.question_runs += len(tasks)
        self.tokens += tokens
        self.dollars += dollars
        return SuiteRun(results=results, tokens=tokens, cost=dollars,
                        seconds=float(len(tasks)))
