"""Measure any neuro-san agent network on any set of questions.

neuro-san can generate an agent network from a sentence and serve it in
seconds. It has no way to say whether that network is any good, or whether one
network is better than another for the same job. This module is that
measurement, for any network neuro-san can load -- not only the ones this
repository evolves:

    python -m esp.measure registries/my_network.hocon --tasks my_questions.jsonl
    python -m esp.measure my_network --tasks my_questions.jsonl   # a manifest name

A question file is JSON Lines, one question per line:

    {"question": "Which city is depot D08 in?", "answer": "Eastgate"}
    {"id": "Q2", "question": "...", "answers": ["C-2139", "C2139"], "hops": 3}

`answer` or `answers` lists what counts as correct; a reply is correct if it
*contains* one of them (numbers match on whole-number boundaries, so "4" does
not match "INC-4429"). `id` and `hops` are optional. With no `--tasks`, the
built-in seventeen-question Meridian Logistics benchmark is used, which needs a
network whose tools can search the Meridian corpus. `--tasks meridian-bank` is the
250-question held-out bank over the same corpus, and `meridian-bank:40` its first
forty, balanced across depths.

What comes back is what this repository's own results are built from:
accuracy, accuracy over the questions the network actually finished, how many
it never finished (a timeout or a blown recursion cap is not a wrong answer),
tokens, cost where the provider reports it, and wall-clock time -- with every
answer kept whole. A run that measured the environment rather than the network
(no model called, every question erroring, a provider quota) is refused rather
than reported, by the same guards that protect the evaluation cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from esp.eval.runner import MAX_WORKERS, TaskResult, refuse_unmeasured, run_suite
from esp.eval.tasks import TASKS, Task

BUILTIN = "meridian"
# The 250-question held-out bank (esp/eval/bank.py). `meridian-bank:40` asks its
# first forty, which the bank orders to be balanced across depths -- a free-tier
# key cannot afford all 250 in a day.
BANK = "meridian-bank"


def _bank() -> list[Task]:
    from esp.eval.bank import BANK as QUESTIONS
    return QUESTIONS


def _select() -> list[Task]:
    from esp.eval.suites import SELECT
    return SELECT


def _judge() -> list[Task]:
    from esp.eval.suites import JUDGE
    return JUDGE


# Every question set that has a name, each loaded only when asked for. The
# scale-up experiment selects on meridian-select and judges on meridian-judge,
# which share no entity (esp/eval/suites.py).
NAMED = {BANK: _bank, "meridian-select": _select, "meridian-judge": _judge}


class SuiteError(ValueError):
    """A question file that cannot be measured against."""


@dataclass
class Report:
    """One network's measurement on one question set."""

    network: str
    suite: str
    results: list[TaskResult]
    tokens: int
    cost: float
    seconds: float
    questions: dict[str, str] = field(default_factory=dict)
    expected: dict[str, str] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return round(sum(r.correct for r in self.results) / len(self.results), 4)

    @property
    def unfinished(self) -> int:
        """Questions the network never answered: a timeout, a recursion cap."""
        return sum(r.infrastructure for r in self.results)

    @property
    def answered_accuracy(self) -> float | None:
        answered = [r for r in self.results if not r.infrastructure]
        if not answered:
            return None
        return round(sum(r.correct for r in answered) / len(answered), 4)

    def as_dict(self) -> dict:
        return {
            "network": self.network, "suite": self.suite,
            "questions_asked": len(self.results), "accuracy": self.accuracy,
            "answered_accuracy": self.answered_accuracy, "unfinished": self.unfinished,
            "tokens": self.tokens, "cost": self.cost, "seconds": self.seconds,
            "results": [{**asdict(r), "question": self.questions.get(r.task_id, ""),
                         "expected": self.expected.get(r.task_id, "")}
                        for r in self.results],
        }


def builtin_suite() -> list[Task]:
    """The seventeen Meridian Logistics questions every committed result used."""
    return list(TASKS)


def parse_suite(text: str) -> list[Task]:
    """Questions from JSON Lines, or from one JSON list of the same objects.

    Validated whole before anything is asked: a malformed line found halfway
    through a paid run is a run paid for twice.
    """
    text = text.strip()
    if not text:
        raise SuiteError("the question file is empty")
    try:
        rows = json.loads(text) if text.startswith("[") else [
            json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise SuiteError(f"not JSON Lines or a JSON list: {exc}") from exc

    tasks: list[Task] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise SuiteError(f"entry {number} is not an object")
        question = str(row.get("question", "")).strip()
        answers = row.get("answers", row.get("answer"))
        if isinstance(answers, (str, int, float)):
            answers = [answers]
        answers = [str(a).strip() for a in (answers or []) if str(a).strip()]
        if not question:
            raise SuiteError(f"entry {number} has no question")
        if not answers:
            raise SuiteError(f"entry {number} has no answer to score against")
        task_id = str(row.get("id") or f"Q{number:02d}")
        if task_id in seen:
            raise SuiteError(f"entry {number} repeats the id {task_id!r}")
        seen.add(task_id)
        try:
            hops = int(row.get("hops", 0))
        except (TypeError, ValueError) as exc:
            raise SuiteError(f"entry {number} has a non-numeric hops") from exc
        tasks.append(Task(task_id, question, answers[0], hops, tuple(answers)))
    return tasks


def load_suite(path: str | Path | None) -> tuple[str, list[Task]]:
    """A named question set: the built-in one, or a file."""
    if path in (None, "", BUILTIN):
        return BUILTIN, builtin_suite()
    named = path.partition(":")[0] if isinstance(path, str) else None
    if named in NAMED:
        questions = NAMED[named]()
        count = path.partition(":")[2]
        if not count:
            return named, list(questions)
        if not count.isdigit() or not 0 < int(count) <= len(questions):
            raise SuiteError(f"{named}:N takes a count from 1 to {len(questions)}, "
                             f"not {count!r}")
        return path, list(questions[:int(count)])
    path = Path(path)
    return path.name, parse_suite(path.read_text(encoding="utf-8"))


def measure(network: str, tasks: list[Task], suite: str = "custom",
            workers: int = MAX_WORKERS, on_result=None) -> Report:
    """Put every question to one network and report what it did.

    `network` is a path to a HOCON file or a network name in the manifest.
    Raises OSError (or QuotaExhausted) when the run measured the environment
    instead of the network; nothing partial is reported in that case.
    """
    if not tasks:
        raise SuiteError("no questions to ask")
    run = run_suite(str(network), tasks, workers=workers, answer_limit=None,
                    on_result=on_result)
    refuse_unmeasured(run.results, run.tokens, action="report it")
    return Report(network=str(network), suite=suite, results=run.results,
                  tokens=run.tokens, cost=run.cost, seconds=run.seconds,
                  questions={t.task_id: t.question for t in tasks},
                  expected={t.task_id: t.answer for t in tasks})


def render(report: Report) -> str:
    """A report as a person reads it in a terminal."""
    answered = report.answered_accuracy
    lines = [
        f"{report.network}  on  {report.suite}",
        f"  accuracy {report.accuracy:.1%}  ({sum(r.correct for r in report.results)}"
        f"/{len(report.results)})   answered-only "
        + (f"{answered:.1%}" if answered is not None else "n/a")
        + f"   unfinished {report.unfinished}",
        f"  tokens {report.tokens:,}   seconds {report.seconds:.0f}"
        + (f"   cost {report.cost:.4f}" if report.cost else ""),
        "",
    ]
    for result in report.results:
        mark = ("unfinished" if result.infrastructure
                else "correct" if result.correct else "wrong")
        reply = (result.answer or result.error).strip().replace("\n", " ")
        lines.append(f"  [{mark:10}] {result.task_id:5} expected "
                     f"{report.expected.get(result.task_id, '')!r}: {reply[:90]}")
    return "\n".join(lines)


def pareto(reports: list[dict]) -> None:
    """Mark the reports no other report beats on accuracy and tokens both.

    Reports that carry an `error` were not measured and are never on the front.
    """
    done = [r for r in reports if not r.get("error")]
    for r in done:
        r["pareto"] = not any(
            o is not r and o["accuracy"] >= r["accuracy"] and o["tokens"] <= r["tokens"]
            and (o["accuracy"] > r["accuracy"] or o["tokens"] < r["tokens"])
            for o in done)


@dataclass(frozen=True)
class Candidate:
    """A network this deployment can measure, and what is known about it."""

    id: str
    name: str
    hocon: str
    measured: dict | None = None        # the committed measurement, if any
    note: str = ""

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "measured": self.measured,
                "note": self.note}


def catalog(extra_dir: str | Path | None = None) -> list[Candidate]:
    """Every network a deployment can measure: the committed twelve, then any
    HOCON files an operator has placed in `extra_dir`.

    The committed networks are rendered for measurement on the configured
    provider, keeping the benchmark's reply format. Files in `extra_dir` are
    measured exactly as written. Nothing is taken from a visitor: a network
    names Python classes to import, so accepting one over the web would be
    accepting code.
    """
    from esp.eval import measurements
    from esp.eval.runner import write_network
    from esp.serving import measurable

    found: list[Candidate] = []
    for record in measurements.load():
        served = measurable(record.genome)
        found.append(Candidate(
            # Operators repeat -- two networks can both be mut:reassign_model --
            # so the name carries enough of the hash to tell them apart.
            id=record.genome_hash, name=f"{record.name()} · {record.genome_hash[:6]}",
            hocon=str(write_network(served.genome)),
            measured={"accuracy": record.accuracy, "tokens": record.tokens,
                      "agents": record.agents, "fitness": record.fitness,
                      "on": record.genome.default_model},
            note=("re-targeted to " + served.provider) if served.retargeted else ""))
    if extra_dir and Path(extra_dir).is_dir():
        for path in sorted(Path(extra_dir).glob("*.hocon")):
            if "manifest" in path.stem:
                continue            # a list of networks to serve, not a network
            found.append(Candidate(id=f"file:{path.stem}", name=path.stem,
                                   hocon=str(path.resolve())))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m esp.measure",
        description="Measure a neuro-san agent network on a set of questions.")
    parser.add_argument("network", help="a HOCON file, or a network name in the manifest")
    parser.add_argument("--tasks", default=BUILTIN,
                        help="JSON Lines question file, or a named set: meridian (the "
                             "built-in 17, default), meridian-bank, meridian-select, "
                             "meridian-judge; NAME:N asks the first N")
    parser.add_argument("--tools", default=None,
                        help="directory holding the network's coded tools "
                             "(sets AGENT_TOOL_PATH)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="questions asked in parallel")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="also write the full report as JSON to this path")
    args = parser.parse_args(argv)

    # neuro-san resolves coded tools from AGENT_TOOL_PATH when the session is
    # built, so it has to be set before the first question is asked.
    if args.tools:
        tools = str(Path(args.tools).resolve())
        os.environ["AGENT_TOOL_PATH"] = tools
        sys.path.insert(0, tools)
    else:
        os.environ.setdefault("AGENT_TOOL_PATH", str(Path(__file__).resolve().parent.parent))

    try:
        suite, tasks = load_suite(args.tasks)
    except (OSError, SuiteError) as exc:
        print(f"cannot read the questions: {exc}", file=sys.stderr)
        return 2
    print(f"measuring {args.network} on {len(tasks)} question(s) from {suite}...",
          file=sys.stderr)
    try:
        report = measure(args.network, tasks, suite=suite, workers=args.workers)
    except OSError as exc:
        print(f"not a measurement of the network: {exc}", file=sys.stderr)
        return 1

    print(render(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.as_dict(), indent=2),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
