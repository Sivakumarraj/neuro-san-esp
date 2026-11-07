"""Same model, fewer agents: the test fixed in docs/SAME-MODEL.md before it ran.

`2cc4` and the designer's shape, every agent of both on `gpt-5.4-nano`, asked the
same questions twice at 10 documents a search, at the same hour. The page lists
five things that must all hold before the README may say `2cc4` is more accurate,
uses fewer tokens, costs less and scores higher. `verdict` checks exactly those:

1. accuracy: right more often on more questions, two-sided sign test p < 0.05,
   a run that did not finish counted as wrong
2. tokens a question: lower, two-sided Wilcoxon signed-rank p < 0.05 over the
   four-question chunks, each chunk's repeats averaged, because the provider
   reports usage per chunk and not per question
3. dollars a question, neuro-san's own reported cost: lower, the same test
4. agents: fewer
5. fitness: higher on the token fitness (`esp.evolve.loop.scalarise`) and on the
   dollar fitness (`esp.eval.pricing.fitness_dollars`)

The token fitness is scaled to the 17-question exam it was calibrated on, so it
is given tokens a question times 17. Every term of both formulas moves the same
way when 1 to 4 hold, so the order the fitness gives cannot depend on that scale.

It reads the committed logs and nothing else, so it costs nothing and needs no key.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from esp.eval import accounting
from esp.eval.pricing import fitness_dollars
from esp.eval.stats import mcnemar_exact
from esp.evolve.loop import scalarise

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "structure"
# The test, then the replication run beside it on the fresh questions.
SUITES = {"judge": RESULTS / "judge-nano-2026-11", "fresh": RESULTS / "fresh-nano-2026-11"}

BASELINE, CANDIDATE = "designer", "2cc4-nano"
# The genomes docs/SAME-MODEL.md names. A log under either label holding another
# genome is a different experiment, and is refused rather than summarised.
HASHES = {BASELINE: "0378ca26cd9fe977", CANDIDATE: "2cc4231b7d3fb81d"}
MODEL = "gpt-5.4-nano"
SEARCH_RESULTS = 10

ALPHA = 0.05
EXAM = 17                # questions the token fitness is scaled to
COST_KEY = "reported_cost"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261107


@dataclass
class Network:
    """One network's log on one question set."""

    label: str
    genome_hash: str
    agents: int
    runs: int = 0
    right: int = 0
    unfinished: int = 0
    tokens: float = 0.0
    dollars: float = 0.0
    # task id -> whether each run was right; chunk -> (task ids, [(tokens, dollars)])
    by_question: dict[str, list[bool]] = field(default_factory=lambda: defaultdict(list))
    by_chunk: dict[int, tuple[tuple[str, ...], list[tuple[float, float]]]] = field(
        default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.right / self.runs if self.runs else 0.0

    @property
    def tokens_per_question(self) -> float:
        return self.tokens / self.runs if self.runs else 0.0

    @property
    def dollars_per_question(self) -> float:
        return self.dollars / self.runs if self.runs else 0.0

    @property
    def dollars_per_correct(self) -> float:
        return self.dollars / self.right if self.right else float("inf")

    @property
    def token_fitness(self) -> float:
        return scalarise(self.accuracy, round(self.tokens_per_question * EXAM), self.agents)

    @property
    def dollar_fitness(self) -> float:
        return fitness_dollars(self.accuracy, self.dollars_per_question, self.agents)

    def as_dict(self) -> dict:
        return {"genome_hash": self.genome_hash, "agents": self.agents,
                "question_runs": self.runs, "correct": self.right,
                "unfinished": self.unfinished, "accuracy": round(self.accuracy, 4),
                "tokens_per_question": round(self.tokens_per_question),
                "dollars_per_question": round(self.dollars_per_question, 6),
                "dollars_per_correct": round(self.dollars_per_correct, 6),
                "token_fitness": round(self.token_fitness, 4),
                "dollar_fitness": round(self.dollar_fitness, 4)}


def read(path: Path, hashes: dict[str, str] | None = None,
         model: str | None = MODEL) -> Network:
    """A log written by `scripts/select_and_judge.py measure`, checked against the plan.

    `hashes` names the genome each label must have run as (this test's by default);
    `model`, when given, is the one model every agent must have run on. A later test
    whose networks mix models passes None and relies on the genome, which names
    every agent's model, to say what ran.
    """
    from esp.evolve.headtohead import read_rows

    plan = HASHES if hashes is None else hashes
    rows = [accounting.normalised(row) for row in read_rows(path)]
    if not rows:
        raise ValueError(f"{path}: no rows")
    label = rows[0]["label"]
    ran = {row["genome_hash"] for row in rows}
    if ran != {plan.get(label)}:
        raise ValueError(f"{path}: {label} ran as {sorted(ran)}, the plan names "
                         f"{plan.get(label)}")
    for row in rows:
        models = set(row["models"].values())
        width = row.get("settings", {}).get("search_results")
        if (model is not None and models != {model}) or width != SEARCH_RESULTS:
            raise ValueError(f"{path}: chunk {row['chunk']} ran on {sorted(models)} at "
                             f"{width} documents, not {model or 'the plan'} at "
                             f"{SEARCH_RESULTS}")
    network = Network(label, rows[0]["genome_hash"], agents=len(rows[0]["models"]))
    for row in rows:
        tasks = tuple(r["task_id"] for r in row["results"])
        known, usage = network.by_chunk.setdefault(row["chunk"], (tasks, []))
        if known != tasks:
            raise ValueError(f"{path}: chunk {row['chunk']} asked {tasks}, earlier {known}")
        usage.append((float(row["tokens"]), float(row[COST_KEY])))
        network.tokens += row["tokens"]
        network.dollars += row[COST_KEY]
        for result in row["results"]:
            network.runs += 1
            network.unfinished += int(result["infrastructure"])
            # Not finishing is not answering: counted wrong, as docs/SAME-MODEL.md says.
            correct = bool(result["correct"]) and not result["infrastructure"]
            network.right += int(correct)
            network.by_question[result["task_id"]].append(correct)
    return network


def _shares(network: Network) -> dict[str, float]:
    return {task: sum(runs) / len(runs) for task, runs in network.by_question.items()}


def accuracy_test(baseline: Network, candidate: Network) -> dict:
    """Questions each was right on more often, the sign test, and a paired 95% interval
    on the difference in accuracy, resampling questions."""
    a, b = _shares(baseline), _shares(candidate)
    shared = sorted(set(a) & set(b))
    better = [t for t in shared if b[t] > a[t]]
    worse = [t for t in shared if b[t] < a[t]]
    diffs = np.array([b[t] - a[t] for t in shared])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = diffs[rng.integers(0, len(diffs), size=(BOOTSTRAP_DRAWS, len(diffs)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"questions": len(shared), "candidate_better_on": better,
            "baseline_better_on": worse, "p": round(mcnemar_exact(len(better), len(worse)), 4),
            "points": round(100 * float(diffs.mean()), 1),
            "ci95_points": [round(100 * float(low), 1), round(100 * float(high), 1)]}


def chunk_test(baseline: Network, candidate: Network, which: int) -> dict:
    """Usage per question in each chunk, repeats averaged, paired chunk by chunk.

    `which` is 0 for tokens and 1 for dollars."""
    shared = sorted(set(baseline.by_chunk) & set(candidate.by_chunk))
    for chunk in shared:
        if baseline.by_chunk[chunk][0] != candidate.by_chunk[chunk][0]:
            raise ValueError(f"chunk {chunk} asked different questions of the two networks")

    def per_question(network: Network, chunk: int) -> float:
        tasks, usage = network.by_chunk[chunk]
        return float(np.mean([u[which] for u in usage])) / len(tasks)

    a = np.array([per_question(baseline, c) for c in shared])
    b = np.array([per_question(candidate, c) for c in shared])
    p = float(wilcoxon(b, a, alternative="two-sided").pvalue) if len(shared) else 1.0
    return {"chunks": len(shared), "candidate_lower_in": int((b < a).sum()),
            "baseline_lower_in": int((a < b).sum()),
            # Six decimals, or two significant figures below that: a network lower in
            # every chunk has an exact p near 6e-08, which six decimals would print as 0.
            "p": round(p, 6) if p >= 1e-6 else float(f"{p:.2g}"),
            "change": round(float(b.sum() / a.sum() - 1), 4) if a.sum() else 0.0}


def verdict(baseline: Network, candidate: Network) -> dict:
    """The five criteria of docs/SAME-MODEL.md, each with its evidence."""
    accuracy = accuracy_test(baseline, candidate)
    tokens = chunk_test(baseline, candidate, 0)
    dollars = chunk_test(baseline, candidate, 1)
    holds = {
        "accuracy": (candidate.accuracy > baseline.accuracy
                     and len(accuracy["candidate_better_on"]) > len(accuracy["baseline_better_on"])
                     and accuracy["p"] < ALPHA),
        "tokens": (candidate.tokens_per_question < baseline.tokens_per_question
                   and tokens["p"] < ALPHA),
        "dollars": (candidate.dollars_per_question < baseline.dollars_per_question
                    and dollars["p"] < ALPHA),
        "agents": candidate.agents < baseline.agents,
        "fitness": (candidate.token_fitness > baseline.token_fitness
                    and candidate.dollar_fitness > baseline.dollar_fitness),
    }
    return {"networks": {baseline.label: baseline.as_dict(),
                         candidate.label: candidate.as_dict()},
            "accuracy": accuracy, "tokens": tokens, "dollars": dollars,
            "holds": holds, "all_hold": all(holds.values())}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyse(suites: dict[str, Path] | None = None) -> dict:
    """Every suite that has both logs, judged on its own; judge is the test."""
    out: dict = {"plan": "docs/SAME-MODEL.md", "alpha": ALPHA, "suites": {}, "logs": {}}
    for name, folder in (suites or SUITES).items():
        paths = {label: folder / f"{label}.jsonl" for label in (BASELINE, CANDIDATE)}
        if not all(path.exists() for path in paths.values()):
            continue
        networks = {label: read(path) for label, path in paths.items()}
        out["suites"][name] = verdict(networks[BASELINE], networks[CANDIDATE])
        for path in paths.values():
            shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            out["logs"][str(shown)] = sha256(path)
    return out
