"""Measure a pool of networks once; compare search strategies on it many times.

The same-budget experiment (`esp.evolve.experiment`) runs one search per arm,
and one search is one sample of a method however large it is. Whether the
Predictor helps cannot be read off a single pair of winners.

Architecture search met the same problem and solved it with tabular
benchmarks (NAS-Bench-101 and 201): measure a space of architectures once,
then run any search method over the measured table as often as statistics
need, for free. This is that, for neuro-san networks.

1. **Pool.** Breed a spread of valid networks from the seeds (1 to 6 mutation
   steps away) and measure each on the `select` questions, once. This is the
   only step that spends.
2. **Replicates.** Each replicate starts from the three seeds plus a few random
   pool members, then pays for `per_step` candidates at a time, chosen by a
   strategy, until the budget is spent. "Paying" reveals a pool member's
   measured answers. A search sees only half the select questions; the
   network it rates best is scored on the other half, which it never saw.
   Scoring on the answers it chose by would reward luck: on a pool of pure
   noise, any strategy with a consistent preference looked significantly
   better than random until this split was added. Strategies are compared
   across hundreds of replicates with bootstrap intervals; every strategy sees
   the same starts, so the comparison is paired.
3. **Judge.** The best networks by `select` fitness, and the designer's shape,
   answer the 200 judge questions they were never chosen on, compared question
   by question with an exact McNemar test.

Fitness is `esp.eval.pricing.fitness_dollars`: accuracy, less dollars per
question, less size. A pool member's value in the replicates is its measured
fitness, so the replicates test choosing, not measuring.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from esp.eval import runner
from esp.eval.judge_plus import JUDGE_200
from esp.eval.pricing import estimate, fitness_dollars
from esp.eval.stats import paired
from esp.eval.suites import SELECT
from esp.eval.tasks import Task
from esp.genome.definition import Genome
from esp.genome.mutations import InvalidMutant, mutate
from esp.genome.seeds import SEEDS
from esp.surrogate.per_question import Observed, QuestionPredictor
from esp.surrogate.predictor import features

POOL_SEED = 20260927
STRATEGIES = ("random", "network", "question", "question-ucb")
BUDGETS = (10, 20, 40)
# Planning figures; `describe` says they are estimates.
TOKENS_PER_QUESTION = 12_000
CALLS_PER_QUESTION = 10


def breed(size: int, seed: int = POOL_SEED, max_steps: int = 6) -> list[Genome]:
    """The seeds, then distinct valid networks 1 to `max_steps` mutations away,
    spread evenly over distance so the pool is not all near-copies."""
    rng = random.Random(seed)
    seeds = [build() for build in SEEDS.values()]
    pool = {g.genome_hash(): g for g in seeds}
    attempts = 0
    while len(pool) < size and attempts < size * 400:
        attempts += 1
        genome = rng.choice(seeds)
        for _ in range(1 + len(pool) % max_steps):
            try:
                genome, _ = mutate(genome, rng)
            except InvalidMutant:
                break
        pool.setdefault(genome.genome_hash(), genome)
    if len(pool) < size:
        raise RuntimeError(f"bred only {len(pool)} distinct networks of {size}")
    return list(pool.values())[:size]


@dataclass
class Plan:
    select: list[Task] = field(default_factory=lambda: list(SELECT))
    judge: list[Task] = field(default_factory=lambda: list(JUDGE_200))
    size: int = 120
    finalists: int = 8
    seed: int = POOL_SEED

    def question_runs(self) -> dict[str, int]:
        return {"pool (select questions)": self.size * len(self.select),
                "judge (finalists and the designer)": (self.finalists + 1) * len(self.judge)}

    def describe(self, models=("claude-haiku-4-5", "claude-sonnet-5",
                               "gemini-3.1-flash-lite")) -> str:
        runs = self.question_runs()
        total = sum(runs.values())
        tokens = total * TOKENS_PER_QUESTION
        lines = [f"Pool benchmark: {self.size} networks on {len(self.select)} select "
                 f"questions, then {self.finalists} finalists and the designer on "
                 f"{len(self.judge)} judge questions."]
        lines += [f"  {name}: {count:,} question-runs" for name, count in runs.items()]
        lines.append(f"  total: {total:,} question-runs, about "
                     f"{total * CALLS_PER_QUESTION:,} model calls and "
                     f"{tokens / 1e6:,.0f}M tokens")
        for model in models:
            lines.append(f"  if every agent ran {model}: about "
                         f"${estimate(tokens, model):,.0f}")
        lines.append(f"Estimates at {TOKENS_PER_QUESTION:,} tokens a question; the "
                     "replicate searches afterwards cost nothing.")
        return "\n".join(lines)


@dataclass
class Member:
    """One measured pool network."""

    genome_hash: str
    genome: dict
    right: dict[str, bool]
    accuracy: float
    tokens: int
    dollars: float
    agents: int

    def observed(self, only: set[str] | None = None) -> Observed:
        right = self.right if only is None else {k: v for k, v in self.right.items()
                                                 if k in only}
        return Observed(Genome.from_canonical(self.genome), right,
                        self.dollars / max(len(self.right), 1))

    @property
    def fitness(self) -> float:
        return self.fitness_on(None)

    def fitness_on(self, only: set[str] | None) -> float:
        """Fitness from the answers to `only` (every answer when None)."""
        answers = [v for k, v in self.right.items() if only is None or k in only]
        accuracy = float(np.mean(answers)) if answers else 0.0
        return fitness_dollars(accuracy, self.dollars / max(len(self.right), 1),
                               self.agents)


def split(tasks: list[Task]) -> tuple[set[str], set[str]]:
    """Questions the search may see, and the ones its choice is scored on.

    Scoring a search's pick on the same answers it chose by rewards luck: on
    pure noise, a strategy with any consistent preference looks significantly
    better or worse than random, because the pool is fixed. Scoring on answers
    the search never saw removes that; only a real relationship survives."""
    ids = [t.task_id for t in tasks]
    return set(ids[0::2]), set(ids[1::2])


def measure(plan: Plan, out_dir: Path, genomes: list[Genome] | None = None) -> dict:
    """Measure every pool network on the select questions. Cached per network,
    so a stop for quota resumes where it stopped and never pays twice."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    genomes = genomes or breed(plan.size, plan.seed)
    members, stopped = [], None
    for genome in genomes:
        try:
            evaluation = runner.evaluate(genome, plan.select)
        except runner.QuotaExhausted as exc:
            stopped = f"quota exhausted after {len(members)} networks: {exc}"
            break
        members.append(Member(genome.genome_hash(), genome.canonical(),
                              {r.task_id: r.correct for r in evaluation.results},
                              evaluation.accuracy, evaluation.tokens, evaluation.cost,
                              evaluation.agents))
    payload = {"plan": {"size": plan.size, "select": len(plan.select), "seed": plan.seed},
               "measured": len(members), "stopped": stopped,
               "members": [asdict(m) for m in members]}
    (out_dir / "pool.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def load(path: Path) -> list[Member]:
    return [Member(**m) for m in json.loads(Path(path).read_text())["members"]]


def _network_scores(train: list[Member], candidates: list[Member], seed: int,
                    visible: set[str]) -> np.ndarray:
    """The network-level baseline: one row per network, fitness as the target."""
    if len(train) < 6:
        return np.zeros(len(candidates))
    model = GradientBoostingRegressor(n_estimators=80, max_depth=2, random_state=seed)
    model.fit(np.vstack([features(m.observed().genome) for m in train]),
              [m.fitness_on(visible) for m in train])
    return model.predict(np.vstack([features(m.observed().genome) for m in candidates]))


def _choose(strategy: str, known: list[Member], unknown: list[Member], k: int,
            tasks: list[Task], visible: set[str], rng: np.random.Generator,
            seed: int) -> list[int]:
    if strategy == "random":
        return list(rng.choice(len(unknown), size=min(k, len(unknown)), replace=False))
    if strategy == "network":
        scores = _network_scores(known, unknown, seed, visible)
    else:
        seen = [t for t in tasks if t.task_id in visible]
        model = QuestionPredictor(seed=seed, members=3, max_iter=60)
        model.fit([m.observed(visible) for m in known], seen)
        genomes = [m.observed().genome for m in unknown]
        scores = model.fitness(genomes, seen,
                               optimism=1.0 if strategy == "question-ucb" else 0.0)
    # Ties broken at random, so a constant score is a random choice, not the
    # pool's file order.
    order = np.lexsort((rng.random(len(unknown)), -np.asarray(scores)))
    return list(order[:k])


def replicate(members: list[Member], tasks: list[Task], strategy: str, seed: int,
              budgets=BUDGETS, per_step: int = 5, start_random: int = 5) -> dict[int, float]:
    """One search over the measured pool.

    The search sees only the `visible` half of the questions: it trains on
    them and keeps the network that scores best on them. What it found is then
    scored on the held-out half. Returns that held-out fitness once each budget
    of paid candidates was spent."""
    visible, scored = split(tasks)
    rng = np.random.default_rng(seed)
    seed_hashes = {build().genome_hash() for build in SEEDS.values()}
    start = [m for m in members if m.genome_hash in seed_hashes]
    others = [m for m in members if m.genome_hash not in seed_hashes]
    # The same random start for every strategy: it depends on the seed only.
    start_rng = np.random.default_rng(seed + 1_000_003)
    picked = set(start_rng.choice(len(others), size=min(start_random, len(others)),
                                  replace=False).tolist())
    known = start + [others[i] for i in sorted(picked)]
    unknown = [m for i, m in enumerate(others) if i not in picked]

    def pick() -> float:
        chosen = max(known, key=lambda m: (m.fitness_on(visible), m.genome_hash))
        return chosen.fitness_on(scored)

    paid, found = 0, {}
    while paid < max(budgets) and unknown:
        k = min(per_step, max(budgets) - paid, len(unknown))
        chosen = _choose(strategy, known, unknown, k, tasks, visible, rng, seed)
        for index in sorted(chosen, reverse=True):
            known.append(unknown.pop(index))
            paid += 1
            if paid in budgets:
                found[paid] = pick()
    for budget in budgets:
        found.setdefault(budget, pick())
    return found


def _interval(values: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(members: list[Member], tasks: list[Task], replicates: int = 200,
            strategies=STRATEGIES, budgets=BUDGETS, per_step: int = 5) -> dict:
    """Every strategy over the same replicate starts. Reports the mean held-out
    fitness of what each found, regret against the pool's best on the held-out
    half, and each strategy's paired difference from random choice with a 95%
    bootstrap interval over replicates. The interval is conditional on this one
    pool: it says how reliably a strategy chooses well here, not across every
    pool that could have been bred."""
    top = max(m.fitness_on(split(tasks)[1]) for m in members)
    runs = {s: [replicate(members, tasks, s, r, budgets, per_step) for r in range(replicates)]
            for s in strategies}
    summary: dict = {"replicates": replicates, "pool": len(members),
                     "pool_best": top, "strategies": {}}
    for strategy, found in runs.items():
        rows = {}
        for budget in budgets:
            values = np.array([f[budget] for f in found])
            row = {"mean_best": float(values.mean()),
                   "mean_regret": float(top - values.mean())}
            if strategy != "random" and "random" in runs:
                diff = values - np.array([f[budget] for f in runs["random"]])
                low, high = _interval(diff, seed=budget)
                row.update({"vs_random": float(diff.mean()),
                            "vs_random_95": [low, high],
                            "better_than_random": low > 0})
            rows[budget] = row
        summary["strategies"][strategy] = rows
    return summary


def finalists(members: list[Member], count: int) -> list[Member]:
    return sorted(members, key=lambda m: -m.fitness)[:count]


def judge(members: list[Member], plan: Plan, out_dir: Path) -> dict:
    """The best by select fitness, and the designer's shape, on the judge set."""
    designer = SEEDS["designer_shaped"]()
    chosen = [Genome.from_canonical(m.genome) for m in finalists(members, plan.finalists)]
    if designer.genome_hash() not in {g.genome_hash() for g in chosen}:
        chosen.append(designer)
    judged, stopped = {}, None
    for genome in chosen:
        try:
            evaluation = runner.evaluate(genome, plan.judge)
        except runner.QuotaExhausted as exc:
            stopped = str(exc)
            break
        per_question = evaluation.cost / max(len(evaluation.results), 1)
        judged[genome.genome_hash()] = {
            "accuracy": evaluation.accuracy, "dollars": evaluation.cost,
            "fitness": fitness_dollars(evaluation.accuracy, per_question, evaluation.agents),
            "right": {r.task_id: r.correct for r in evaluation.results}}
    comparisons = {}
    key = designer.genome_hash()
    if key in judged:
        for digest, record in judged.items():
            if digest != key:
                comparisons[digest] = paired(record["right"], judged[key]["right"])
    result = {"stopped": stopped, "designer": key, "judged": judged,
              "vs_designer": comparisons}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "judge.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
