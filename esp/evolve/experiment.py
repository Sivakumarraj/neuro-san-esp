"""The same-budget experiment: does the Predictor help the search at all?

Everything published so far is one search, selected and scored on seventeen
questions. That cannot answer the question an ESP reviewer asks first -- would
the search have done as well without the surrogate? -- and the held-out bank
showed why it matters: the one evolved network measured on new questions did not
beat the designer's shape.

So this runs the search twice from the same start, with the same budget:

* **predictor** -- ESP: bred candidates are ranked by the Predictor, and only
  the top of the ranking is paid for.
* **random** -- the control: bred from the same kind of parents, paid for in a
  shuffled order, the Predictor never consulted.

Both select on `meridian-select` (60 questions). Then each arm's winner and the
designer's shape are judged on `meridian-judge` (100 questions about entities
the select set never names), and compared question by question. The judge set
takes no part in any choice, so its verdict is a held-out one.

Nothing is spent unless asked. `plan` prices the run from committed
measurements; `run` starts with a calibration that stops if the select set is
too easy to rank networks, because an experiment on an exam everyone passes
would spend the whole budget to measure nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from esp.eval.runner import QuotaExhausted, evaluate
from esp.eval.stats import paired
from esp.eval.tasks import Task
from esp.evolve.loop import Evolution, fitness
from esp.genome.definition import Genome
from esp.genome.seeds import SEEDS

ROOT = Path(__file__).resolve().parent.parent.parent
ARMS = ("predictor", "random")

# Phase A measures three seeds and six mutants of them before any search, in
# both arms alike -- and only once, because both arms breed them from the same
# seed and the cache serves the second arm.
PHASE_A = 9

# Above this, the select set cannot separate networks: the held-out bank's
# designer score was 96%, and there was nothing left to rank.
CEILING = 0.90

# Model calls per question. The committed benchmark runs cost about 165 calls
# for seventeen questions; aggregates search more, so this is a floor.
CALLS_PER_QUESTION = 10


def cost_per_question(root: Path = ROOT) -> float:
    """Dollars per question, from what the committed held-out runs cost."""
    reports = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((root / "results" / "heldout_bank").glob("*.json"))]
    asked = sum(r["questions_asked"] for r in reports)
    return sum(r["cost"] for r in reports) / asked if asked else 0.0


@dataclass
class Plan:
    select: list[Task]
    judge: list[Task]
    budget: int = 40
    per_generation: int = 4
    arms: tuple[str, ...] = ARMS
    seed: int = 20260926
    pool: int = 400

    def question_runs(self) -> dict[str, int]:
        return {
            "seed population (shared)": PHASE_A * len(self.select),
            "search, all arms": len(self.arms) * self.budget * len(self.select),
            "judging": (len(self.arms) + 1) * len(self.judge),
        }

    def describe(self) -> str:
        runs = self.question_runs()
        total = sum(runs.values())
        rate = cost_per_question()
        lines = [
            f"arms {', '.join(self.arms)}; {self.budget} paid candidates each after "
            f"{PHASE_A} shared seed-population networks; {self.per_generation} per generation",
            f"select on {len(self.select)} questions, judge on {len(self.judge)}",
            "",
            *(f"  {name:28} {count:>7,} question-runs" for name, count in runs.items()),
            f"  {'total':28} {total:>7,} question-runs",
            f"  model calls, at least        {total * CALLS_PER_QUESTION:>7,}",
        ]
        if rate:
            lines.append(f"  cost, from committed runs    ${total * rate:,.2f} to "
                         f"${total * rate * 2:,.2f} (aggregates search more)")
        lines += ["",
                  f"Networks measured on the select set: about "
                  f"{PHASE_A + len(self.arms) * self.budget}. Each arm's Predictor "
                  f"trains on up to {PHASE_A + self.budget}."]
        return "\n".join(lines)


@dataclass
class Summary:
    plan: dict
    stopped: str | None = None
    calibration: dict = field(default_factory=dict)
    arms: dict = field(default_factory=dict)
    judged: dict = field(default_factory=dict)
    comparisons: dict = field(default_factory=dict)


def _judge_record(name: str, genome: Genome, tasks: list[Task]) -> dict:
    evaluation = evaluate(genome, tasks)
    return {"network": name, "genome_hash": genome.genome_hash(),
            "accuracy": evaluation.accuracy, "tokens": evaluation.tokens,
            "agents": evaluation.agents, "fitness": round(fitness(evaluation), 4),
            "right": {r.task_id: bool(r.correct) for r in evaluation.results}}


def run(plan: Plan, out_dir: Path, force: bool = False) -> Summary:
    """Run it. Every measurement is cached per question set, so a run stopped by
    a quota or a crash resumes where it stopped when run again."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = Summary(plan={"budget": plan.budget, "per_generation": plan.per_generation,
                            "arms": list(plan.arms), "seed": plan.seed, "pool": plan.pool,
                            "select": len(plan.select), "judge": len(plan.judge)})

    def finish(why: str | None = None) -> Summary:
        summary.stopped = why
        (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n",
                                              encoding="utf-8")
        return summary

    designer = SEEDS["designer_shaped"]()
    try:
        calibration = evaluate(designer, plan.select)
    except QuotaExhausted as exc:
        return finish(f"provider budget exhausted during calibration: {exc}"[:300])
    summary.calibration = {"network": "seed:designer_shaped",
                           "accuracy": calibration.accuracy, "tokens": calibration.tokens}
    print(f"calibration: the designer's shape scores {calibration.accuracy:.1%} on "
          f"the select set", flush=True)
    if calibration.accuracy >= CEILING and not force:
        return finish(f"the select set is too easy to rank networks: the designer's "
                      f"shape scored {calibration.accuracy:.1%}, at or above "
                      f"{CEILING:.0%}. Nothing more was spent.")

    winners: dict[str, Genome] = {}
    for arm in plan.arms:
        print(f"\n=== arm: {arm} ===", flush=True)
        evolution = Evolution(seed=plan.seed, surrogate_pool=plan.pool,
                              real_per_generation=plan.per_generation,
                              out_dir=str(out_dir / arm), tasks=plan.select,
                              select=arm, budget=plan.budget)
        history = evolution.run(generations=plan.budget)
        best = history.best()
        summary.arms[arm] = {
            "paid_candidates": evolution.search_spent,
            "new_measurements": history.real_evaluations,
            "stopped_early": history.stopped_early,
            "best": asdict(best) if best else None,
            "predictor_quality": history.surrogate_quality[-1:] or None,
        }
        if history.stopped_early:
            return finish(f"arm {arm} stopped early ({history.stopped_early}); run "
                          f"again to resume from the cache")
        if best:
            winners[arm] = evolution.pool[best.genome_hash]

    entrants = {"designer": designer, **winners}
    print("\n=== judging on the held-out set ===", flush=True)
    try:
        for name, genome in entrants.items():
            summary.judged[name] = _judge_record(name, genome, plan.judge)
            record = summary.judged[name]
            print(f"  {name:10} {record['accuracy']:.1%}  tokens {record['tokens']:,}",
                  flush=True)
    except QuotaExhausted as exc:
        return finish(f"provider budget exhausted while judging: {exc}"[:300])

    names = list(summary.judged)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            summary.comparisons[f"{a} vs {b}"] = paired(summary.judged[a]["right"],
                                                        summary.judged[b]["right"])
    return finish(None)


def report(summary: Summary) -> str:
    """The result, in the words it can bear."""
    lines = []
    if summary.calibration:
        lines.append(f"calibration: designer {summary.calibration['accuracy']:.1%} on select")
    for arm, info in summary.arms.items():
        best = info.get("best") or {}
        lines.append(f"{arm:10} paid {info['paid_candidates']}, best on select "
                     f"{best.get('accuracy', 0):.1%} fitness {best.get('fitness', 0):+.4f}")
    if summary.judged:
        lines.append("\nheld out:")
        for name, record in summary.judged.items():
            right = sum(record["right"].values())
            lines.append(f"  {name:10} {right:>3}/{len(record['right'])}  "
                         f"tokens {record['tokens']:,}")
        lines.append("\npaired (exact McNemar):")
        for pair, result in summary.comparisons.items():
            a, b = pair.split(" vs ")
            lines.append(f"  {pair:24} only {a}: {len(result['only_a'])}, only {b}: "
                         f"{len(result['only_b'])}, p = {result['p']:.3f}")
        lines.append("\nA p above 0.05 means no difference is established, in either "
                     "direction.")
    if summary.stopped:
        lines.append(f"\nSTOPPED: {summary.stopped}")
    return "\n".join(lines)
