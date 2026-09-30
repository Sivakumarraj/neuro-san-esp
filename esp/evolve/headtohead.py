"""The head-to-head: the designer's network against the search's champion, on
questions neither was chosen on, repeated, under a hard spending cap.

Why this exists. The committed result -- an evolved network ahead of the
designer on the seventeen built-in questions -- was selected and scored on the
same seventeen, each network measured once. Three things in that data say a
one-question lead there is not evidence:

* Two of the twelve measured networks are the designer's own network plus an
  exact copy of one specialist (esp/genome/prune.py). Removing the copy gives
  back the designer's genome, yet both scored 15/17 to its 14/17.
  Identical capability, one question apart: that is the noise floor.
* The one change the search made that a copy does not explain is the router's
  model: the two leading networks promoted the Coordinator and left every
  worker on the cheap model.
* A live rerun on twenty fresh questions tied on accuracy, and a second one was
  not a comparison at all: the free tier ran out mid-run and failover moved
  every agent of both networks onto one model.

So the contenders are the designer and the champion with its copy removed --
the designer plus a stronger router, the one decision left standing -- each
moved onto the configured provider rung for rung (esp/serving.py). They are
asked the same judge questions, interleaved chunk by chunk so neither gets the
provider on a better hour, with models pinned (ESP_PIN_MODELS) so nothing is
substituted mid-run, and every repeat is kept, because one run of a network is
one draw.

Everything measured is appended to a log as it lands, so a run stopped by the
cap, a quota or a crash resumes where it stopped and nothing paid for is lost.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from esp.eval.tasks import Task
from esp.genome.definition import Genome

CHUNK = 4
# What one question costs a network in tokens, measured: 7k to 23k on the
# committed and live Gemini runs. The plan prices at the top of that range so
# the cap is met early rather than late.
TOKENS_PER_QUESTION = 20_000
# Share of a network's tokens its router spends. The Coordinator reads every
# specialist's report, so it carries more than an equal share.
ROUTER_SHARE = 0.4


@dataclass(frozen=True)
class Contender:
    label: str
    genome: Genome
    origin: str               # which committed measurement it came from, and how

    @property
    def genome_hash(self) -> str:
        return self.genome.genome_hash()

    def models(self) -> dict[str, str]:
        return {name: agent.model or self.genome.default_model
                for name, agent in sorted(self.genome.agents.items())}


def contenders() -> list[Contender]:
    """The designer's network, and the committed champion with its copies
    removed, both on the configured provider."""
    from esp.eval import measurements
    from esp.genome.prune import copies, without_copies
    from esp.genome.seeds import designer_shaped
    from esp.serving import measurable

    designer = designer_shaped()
    best = measurements.best()
    if best is None:
        raise RuntimeError("no committed measurements to take a champion from")
    removed = copies(best.genome)
    champion = without_copies(best.genome)
    return [
        Contender("designer", measurable(designer).genome,
                  "seed:designer_shaped, the network the designer produces"),
        Contender("champion", measurable(champion).genome,
                  f"{best.genome_hash} (committed best), with "
                  f"{', '.join(removed) or 'nothing'} removed"),
    ]


def estimate_cost(contender: Contender, questions: int,
                  tokens_per_question: int = TOKENS_PER_QUESTION) -> float:
    """Dollars for `questions` question-runs, from list prices."""
    from esp.eval.pricing import blended

    router = contender.genome.top
    models = contender.models()
    workers = [m for name, m in models.items() if name != router]
    worker_price = (sum(blended(m) for m in workers) / len(workers)
                    if workers else blended(models[router]))
    price = ROUTER_SHARE * blended(models[router]) + (1 - ROUTER_SHARE) * worker_price
    return questions * tokens_per_question / 1e6 * price


@dataclass
class Plan:
    tasks: list[Task]
    repeats: int
    budget: float
    field_: list[Contender] = field(default_factory=list)

    def chunks(self) -> list[list[Task]]:
        return [self.tasks[i:i + CHUNK] for i in range(0, len(self.tasks), CHUNK)]

    def order(self) -> list[tuple[int, int, Contender]]:
        """(repeat, chunk index, contender), interleaved, alternating which
        network goes first so neither always meets the provider second."""
        steps = []
        for repeat in range(self.repeats):
            for index in range(len(self.chunks())):
                pair = list(self.field_)
                if (repeat + index) % 2:
                    pair.reverse()
                steps.extend((repeat, index, c) for c in pair)
        return steps

    def estimate(self) -> float:
        runs = len(self.tasks) * self.repeats
        return sum(estimate_cost(c, runs) for c in self.field_)

    def describe(self) -> str:
        lines = [f"{len(self.tasks)} judge questions x {len(self.field_)} networks x "
                 f"{self.repeats} repeats = {len(self.tasks) * len(self.field_) * self.repeats}"
                 " question-runs",
                 f"estimated ${self.estimate():.2f} at {TOKENS_PER_QUESTION:,} tokens a "
                 f"question; hard cap ${self.budget:.2f}"]
        for c in self.field_:
            lines.append(f"  {c.label:9} {c.genome_hash}  {c.origin}")
            for agent, model in c.models().items():
                lines.append(f"      {agent:20} {model}")
        return "\n".join(lines)


Measure = Callable[[Contender, list[Task]], dict]


class Log:
    """Append-only record of every chunk measured."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        if path.exists():
            self.rows = [json.loads(line) for line in path.read_text().splitlines()
                         if line.strip()]

    def done(self, repeat: int, index: int, contender: Contender) -> bool:
        return any(r["repeat"] == repeat and r["chunk"] == index
                   and r["genome_hash"] == contender.genome_hash for r in self.rows)

    def spent(self) -> float:
        return sum(r["cost"] for r in self.rows)

    def add(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")
        self.rows.append(row)


def run(plan: Plan, measure: Measure, log: Log, say=print) -> str:
    """Measure the plan until it is done or the cap would be crossed.

    Returns why it stopped: "done", "budget" or the error that stopped it.
    """
    chunks = plan.chunks()
    for repeat, index, contender in plan.order():
        if log.done(repeat, index, contender):
            continue
        chunk = chunks[index]
        mine = [r for r in log.rows if r["genome_hash"] == contender.genome_hash]
        per_question = (sum(r["cost"] for r in mine) / sum(len(r["results"]) for r in mine)
                        if mine else estimate_cost(contender, 1))
        if log.spent() + per_question * len(chunk) > plan.budget:
            say(f"stopping: the next chunk would pass the ${plan.budget:.2f} cap "
                f"(spent ${log.spent():.4f})")
            return "budget"
        try:
            outcome = measure(contender, chunk)
        except OSError as exc:
            say(f"stopping: {contender.label} chunk {index} repeat {repeat}: {exc}")
            return f"{type(exc).__name__}: {exc}"
        log.add({"label": contender.label, "genome_hash": contender.genome_hash,
                 "repeat": repeat, "chunk": index, **outcome})
        right = sum(r["correct"] for r in outcome["results"])
        say(f"{contender.label:9} repeat {repeat} chunk {index:2}: {right}/{len(chunk)}  "
            f"{outcome['tokens']:,} tokens  ${outcome['cost']:.4f}  "
            f"(spent ${log.spent():.4f})")
    return "done"


def summarise(log: Log, plan: Plan) -> dict:
    """Per network: accuracy over every question-run it finished, tokens and
    dollars per question; and the questions the two answered differently."""
    networks = {}
    for c in plan.field_:
        rows = [r for r in log.rows if r["genome_hash"] == c.genome_hash]
        results = [x for r in rows for x in r["results"]]
        asked = len(results)
        unfinished = sum(x["infrastructure"] for x in results)
        right = sum(x["correct"] for x in results)
        tokens = sum(r["tokens"] for r in rows)
        cost = sum(r["cost"] for r in rows)
        networks[c.label] = {
            "genome_hash": c.genome_hash, "origin": c.origin, "models": c.models(),
            "question_runs": asked, "correct": right, "unfinished": unfinished,
            "accuracy": round(right / asked, 4) if asked else None,
            "answered_accuracy": (round(right / (asked - unfinished), 4)
                                  if asked > unfinished else None),
            "tokens": tokens, "cost": round(cost, 6),
            "tokens_per_question": round(tokens / asked) if asked else None,
            "cost_per_question": round(cost / asked, 6) if asked else None,
        }

    # Paired by question: how often each network was right on it, over repeats.
    rate: dict[str, dict[str, list[bool]]] = {}
    for r in log.rows:
        for x in r["results"]:
            if not x["infrastructure"]:
                rate.setdefault(x["task_id"], {}).setdefault(r["label"], []).append(
                    x["correct"])
    labels = [c.label for c in plan.field_]
    better = {label: [] for label in labels}
    if len(labels) == 2:
        a, b = labels
        for task_id, seen in sorted(rate.items()):
            if a in seen and b in seen:
                ra, rb = sum(seen[a]) / len(seen[a]), sum(seen[b]) / len(seen[b])
                if ra > rb:
                    better[a].append(task_id)
                elif rb > ra:
                    better[b].append(task_id)
    return {"networks": networks, "better_on": better,
            "spent": round(log.spent(), 6), "questions": len(plan.tasks),
            "repeats": plan.repeats}


def render(summary: dict) -> str:
    lines = [f"spent ${summary['spent']:.4f} on {summary['questions']} questions "
             f"x {summary['repeats']} repeats"]
    for label, n in summary["networks"].items():
        if not n["question_runs"]:
            lines.append(f"  {label:9} nothing measured")
            continue
        lines.append(
            f"  {label:9} {n['correct']}/{n['question_runs']} = {n['accuracy']:.1%}"
            f"  (unfinished {n['unfinished']})  {n['tokens_per_question']:,} tokens/q"
            f"  ${n['cost_per_question']:.5f}/q  total ${n['cost']:.4f}")
    for label, ids in summary["better_on"].items():
        lines.append(f"  {label} more often right on {len(ids)}: {' '.join(ids) or '-'}")
    return "\n".join(lines)


def rehearsal_measure(contender: Contender, chunk: list[Task]) -> dict:
    """A stand-in provider: deterministic answers and plausible spend, $0.

    Exercises everything a paid run does except the calls. Nothing it
    produces is a measurement.
    """
    results, tokens = [], 0
    for task in chunk:
        rng = random.Random(f"{contender.genome_hash}:{task.task_id}")
        correct = rng.random() < 0.85
        results.append({"task_id": task.task_id, "hops": task.hops, "correct": correct,
                        "seconds": 1.0, "answer": task.answer if correct else "?",
                        "error": "", "infrastructure": False})
        tokens += rng.randrange(6_000, 20_000)
    return {"results": results, "tokens": tokens,
            "cost": estimate_cost(contender, len(chunk), tokens // len(chunk)),
            "seconds": 1.0, "rehearsal": True}


def live_measure(contender: Contender, chunk: list[Task]) -> dict:
    """Ask the real provider."""
    from dataclasses import asdict

    from esp.eval.runner import write_network
    from esp.measure import measure

    path = write_network(contender.genome)
    report = measure(str(path), chunk, suite="meridian-judge-200", workers=CHUNK)
    # The cap is enforced on the larger of what the provider reported and what
    # list prices say the tokens cost. A model the installed neuro-san has no
    # price for reports $0, and a cap that trusts $0 never stops.
    listed = estimate_cost(contender, len(chunk), report.tokens // max(len(chunk), 1))
    return {"results": [asdict(r) for r in report.results], "tokens": report.tokens,
            "cost": max(report.cost, listed), "reported_cost": report.cost,
            "listed_cost": round(listed, 6), "seconds": report.seconds,
            "models": contender.models()}
