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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.stats import norm
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
CALLS_PER_QUESTION = 10
ROOT = Path(__file__).resolve().parent.parent.parent


def tokens_per_question(root: Path = ROOT) -> tuple[int, int]:
    """What a question has cost in tokens on committed runs: (low, high).

    A single planning figure of 12,000 used to price the run. Nothing
    committed measured it, and it sat below the rehearsal's own simulated
    scale (about 16,000). The low end is the select set's own calibration run
    (the designer's shape, 20 questions, only three of them aggregates, where
    the full set is 40% aggregates); the high end is the twelve networks on
    the seventeen, whose two whole-corpus aggregates most networks searched
    until they timed out. The select set sits between the two."""
    def per_question(paths, count) -> list[float]:
        found = []
        for path in paths:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                found.append((int(raw["tokens"]), count(raw)))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return [tokens / questions for tokens, questions in found if questions]

    low = per_question(sorted((root / "results" / "calibration").glob("*.json")),
                       lambda raw: len(raw["results"]))
    high = per_question(sorted((root / "tests" / "fixtures" / "cache").glob("*.json")),
                        lambda raw: len(raw["results"]))
    if not low or not high:
        return 12_000, 20_000
    return round(float(np.mean(low))), round(float(np.mean(high)))


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
        low, high = tokens_per_question()
        lines = [f"Pool benchmark: {self.size} networks on {len(self.select)} select "
                 f"questions, then {self.finalists} finalists and the designer on "
                 f"{len(self.judge)} judge questions."]
        lines += [f"  {name}: {count:,} question-runs" for name, count in runs.items()]
        lines.append(f"  total: {total:,} question-runs, about "
                     f"{total * CALLS_PER_QUESTION:,} model calls and "
                     f"{total * low / 1e6:,.0f}M to {total * high / 1e6:,.0f}M tokens")
        for model in models:
            lines.append(f"  if every agent ran {model}: about "
                         f"${estimate(total * low, model):,.0f} to "
                         f"${estimate(total * high, model):,.0f}")
        lines.append(f"Estimates at {low:,} to {high:,} tokens a question, the range "
                     "committed runs have cost; the replicate searches afterwards "
                     "cost nothing.")
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


# A network whose every question fails is refused by the runner, as it should
# be: the number would describe the environment. One such network is a fact
# about that network, and it is recorded and skipped, so a resume does not pay
# for it again. Several in a row are a fact about the environment (a key, a
# model the key cannot reach), and the run stops with nothing recorded against
# them.
UNMEASURABLE_STREAK = 3


def _unmeasurable(path: Path, plan: dict) -> dict[str, str]:
    """The networks an earlier run of the same plan could not measure."""
    try:
        earlier = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(earlier.get("unmeasurable") or {}) if earlier.get("plan") == plan else {}


def measure(plan: Plan, out_dir: Path, genomes: list[Genome] | None = None) -> dict:
    """Measure every pool network on the select questions. Cached per network,
    so a stop for quota resumes where it stopped and never pays twice."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    genomes = genomes or breed(plan.size, plan.seed)
    described = {"size": plan.size, "select": len(plan.select), "seed": plan.seed}
    unmeasurable = _unmeasurable(out_dir / "pool.json", described)
    members, stopped, streak = [], None, {}
    for genome in genomes:
        digest = genome.genome_hash()
        if digest in unmeasurable:
            continue
        try:
            evaluation = runner.evaluate(genome, plan.select)
        except runner.QuotaExhausted as exc:
            stopped = f"quota exhausted after {len(members)} networks: {exc}"
            break
        except OSError as exc:
            streak[digest] = str(exc)[:300]
            if len(streak) >= UNMEASURABLE_STREAK:
                stopped = (f"{len(streak)} networks in a row could not be measured, which "
                           f"points at the environment, not the networks; nothing was "
                           f"recorded against them. Last: {exc}")[:500]
                break
            continue
        unmeasurable.update(streak)
        streak = {}
        members.append(Member(digest, genome.canonical(),
                              {r.task_id: r.correct for r in evaluation.results},
                              evaluation.accuracy, evaluation.tokens, evaluation.cost,
                              evaluation.agents))
    if stopped is None:
        unmeasurable.update(streak)
    payload = {"plan": described, "measured": len(members), "stopped": stopped,
               "unmeasurable": unmeasurable,
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


# A strategy is one of STRATEGIES by name, or any function that scores the
# unmeasured networks from the measured ones -- higher is paid for first. The
# function form is the seam a learned chooser plugs into (see
# esp.evolve.context), and it is how the tests pit an information-free
# preference against random choice.
Scorer = Callable[[list[Member], list[Member], list[Task], int], Sequence[float]]


def _name(strategy: str | Scorer) -> str:
    return strategy if isinstance(strategy, str) else strategy.__name__


def _choose(strategy: str | Scorer, known: list[Member], unknown: list[Member], k: int,
            tasks: list[Task], visible: set[str], rng: np.random.Generator,
            seed: int) -> list[int]:
    if strategy == "random":
        return list(rng.choice(len(unknown), size=min(k, len(unknown)), replace=False))
    seen = [t for t in tasks if t.task_id in visible]
    if callable(strategy):
        scores = strategy(known, unknown, seen, seed)
    elif strategy == "network":
        scores = _network_scores(known, unknown, seed, visible)
    else:
        model = QuestionPredictor(seed=seed, members=3, max_iter=60)
        model.fit([m.observed(visible) for m in known], seen)
        genomes = [m.observed().genome for m in unknown]
        scores = model.fitness(genomes, seen,
                               optimism=1.0 if strategy == "question-ucb" else 0.0)
    # Ties broken at random, so a constant score is a random choice, not the
    # pool's file order.
    order = np.lexsort((rng.random(len(unknown)), -np.asarray(scores, dtype=float)))
    return list(order[:k])


def replicate(members: list[Member], tasks: list[Task], strategy: str | Scorer, seed: int,
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


def permuted(members: list[Member], draw: int) -> list[Member]:
    """The pool with its measured outcomes shuffled across networks: every
    network keeps its genome and size and takes another's answers, tokens and
    dollars. Whatever a strategy achieves here, it achieves without any link
    between a network's structure and what it measured."""
    order = np.random.default_rng(draw + 9_000_011).permutation(len(members))
    return [replace(m, right=members[j].right, accuracy=members[j].accuracy,
                    tokens=members[j].tokens, dollars=members[j].dollars)
            for m, j in zip(members, order, strict=True)]


def holm(p_values: dict) -> dict:
    """Holm-Bonferroni adjusted p-values, for a family of tests at once."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def _against_null(observed: np.ndarray, null: np.ndarray) -> tuple[float, float, float]:
    """(null mean, standard error of observed-minus-null, two-sided p).

    `observed` is one difference per replicate on the real pool; `null` is a
    (permutations x replicates) table of the same difference on permuted
    pools. The null's spread has two parts, which permutation of outcomes
    landed where and the luck of the replicate starts, split the one-way
    random-effects way. The observed mean carries the first part whole and the
    second divided by its own replicates; the null mean carries both divided
    by what estimated it."""
    cells, per_cell = null.shape
    means = null.mean(axis=1)
    within = float(null.var(axis=1, ddof=1).mean()) if per_cell > 1 else 0.0
    between = (max(0.0, float(means.var(ddof=1)) - within / per_cell)
               if cells > 1 else 0.0)
    own = float(observed.var(ddof=1)) if len(observed) > 1 else within
    variance = (between + own / len(observed)
                + between / cells + within / (cells * per_cell))
    excess = float(observed.mean() - means.mean())
    if variance <= 0:
        return float(means.mean()), 0.0, 1.0 if excess == 0 else 0.0
    error = float(np.sqrt(variance))
    return float(means.mean()), error, float(2 * norm.sf(abs(excess) / error))


def compare(members: list[Member], tasks: list[Task], replicates: int = 200,
            strategies=STRATEGIES, budgets=BUDGETS, per_step: int = 5,
            permutations: int = 19, per_permutation: int | None = None) -> dict:
    """Every strategy over the same replicate starts, judged against a
    permutation null.

    Reports the mean held-out fitness of what each strategy found, regret
    against the pool's best on the held-out half, and its paired difference
    from random choice.

    **Whether that difference means anything is decided against the same
    difference on permuted pools**, where the measured outcomes are shuffled
    across networks (`permuted`). A first version put a bootstrap interval
    over the replicates of the one pool. That interval shrinks as replicates
    are added, but which networks happened to be lucky on the scored
    questions does not change with replicates, and a strategy with a fixed
    preference and no information came out "better than random" on pure
    noise in a third of comparisons, and worse in half. A permutation keeps
    every genome and breaks every link between structure and outcome, so a
    strategy that exploits no such link does as well on the permuted pools as
    on the real one. Network size is not permuted: it is exact, and part of
    fitness.

    Every strategy-and-budget test is one of a family, so p-values are
    Holm-adjusted across all of them. `better_than_random` needs the adjusted
    p below 0.05 with the strategy ahead of random choice and of its own null.
    `vs_random_95` is the observed difference with the error the test uses.
    The null costs at least as many replicates again as the observed run.
    `regret_closed` is the effect size: the share of random choice's regret
    that the strategy removes."""
    top = max(m.fitness_on(split(tasks)[1]) for m in members)
    names = [_name(s) for s in strategies]
    # Ten a permutation at least: with fewer the spread within a permutation
    # is estimated too loosely, and on noise pools the unadjusted interval
    # excluded zero twice as often as it should.
    per_permutation = per_permutation or max(10, round(replicates / permutations))

    def run(pool_members: list[Member], count: int, offset: int) -> dict[str, list[dict]]:
        return {name: [replicate(pool_members, tasks, strategy, offset + r, budgets,
                                 per_step) for r in range(count)]
                for strategy, name in zip(strategies, names, strict=True)}

    found = run(members, replicates, 0)
    nulls = ([run(permuted(members, draw), per_permutation, 10_000 * (draw + 1))
              for draw in range(permutations)] if "random" in names else [])
    summary: dict = {"replicates": replicates, "permutations": len(nulls),
                     "per_permutation": per_permutation, "pool": len(members),
                     "pool_best": top,
                     "uncertainty": "against a permutation null, outcomes shuffled across "
                                    "networks; Holm-adjusted across every strategy and "
                                    "budget",
                     "strategies": {}}
    p_values = {}
    for name in names:
        rows = {}
        for budget in budgets:
            values = np.array([f[budget] for f in found[name]])
            row = {"mean_best": float(values.mean()),
                   "mean_regret": float(top - values.mean())}
            if name != "random" and nulls:
                base = np.array([f[budget] for f in found["random"]])
                diff = values - base
                null = np.array([[a[budget] - b[budget]
                                  for a, b in zip(cell[name], cell["random"], strict=True)]
                                 for cell in nulls])
                null_mean, error, p = _against_null(diff, null)
                estimate = float(diff.mean())
                random_regret = float(top - base.mean())
                row.update({"vs_random": estimate, "null": null_mean,
                            "vs_random_95": [estimate - 1.96 * error,
                                             estimate + 1.96 * error],
                            "p": p,
                            "regret_closed": (estimate / random_regret
                                              if random_regret > 0 else 0.0)})
                p_values[(name, budget)] = p
            rows[budget] = row
        summary["strategies"][name] = rows
    for (name, budget), adjusted in holm(p_values).items():
        row = summary["strategies"][name][budget]
        row["p_holm"] = adjusted
        ahead = row["vs_random"] > 0 and row["vs_random"] > row["null"]
        behind = row["vs_random"] < 0 and row["vs_random"] < row["null"]
        row["better_than_random"] = bool(adjusted < 0.05 and ahead)
        row["worse_than_random"] = bool(adjusted < 0.05 and behind)
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
        except OSError as exc:
            # Not judged, and said so, rather than ending the judging for the
            # networks after it.
            judged[genome.genome_hash()] = {"error": str(exc)[:300]}
            continue
        per_question = evaluation.cost / max(len(evaluation.results), 1)
        judged[genome.genome_hash()] = {
            "accuracy": evaluation.accuracy, "dollars": evaluation.cost,
            "fitness": fitness_dollars(evaluation.accuracy, per_question, evaluation.agents),
            "right": {r.task_id: r.correct for r in evaluation.results}}
    comparisons = {}
    key = designer.genome_hash()
    if "right" in judged.get(key, {}):
        for digest, record in judged.items():
            if digest != key and "right" in record:
                comparisons[digest] = paired(record["right"], judged[key]["right"])
    # Every finalist is tested against the designer's shape: one family.
    for digest, adjusted in holm({d: c["p"] for d, c in comparisons.items()}).items():
        comparisons[digest]["p_holm"] = adjusted
    result = {"stopped": stopped, "designer": key, "judged": judged,
              "vs_designer": comparisons}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "judge.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
