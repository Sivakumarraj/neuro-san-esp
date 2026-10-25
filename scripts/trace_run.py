"""Trace questions through both networks: which agent spent what, and said what.

    python scripts/trace_run.py            # the plan and its price; nothing spent
    python scripts/trace_run.py --go       # ask the provider, append to results/trace/
    python scripts/trace_run.py --report   # read results/trace/ and print the split

It asks the two networks of the uncertainty study (`designer`, and `2cc4` with
its Researcher promoted) the multi-incident penalty questions of judge and
fresh, which are most of both networks' wrong answers, and two questions each
network got right as controls. Every message every agent streams is kept, so
the trace answers two things the committed logs cannot:

* how a run's dollars divide between the agents, and so what resampling only
  the front man's reply could cost (docs/FINDINGS.md: a probe must cost at most
  about a third of a cheap run for an uncertainty gate to match the rule);
* which agent produced a wrong number, and from what it was handed.

The networks must be the ones stage 8 measured: on another provider their
genomes differ, the dollars mean something else, and the script refuses unless
told --any-provider. Each question is appended as it lands, so a stopped run
resumes, and it stops at --budget dollars.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from esp import autopsy, trace, uncertainty  # noqa: E402
from esp.config import bootstrap  # noqa: E402

OUT = ROOT / "results" / "trace"
CONTROLS = ("J009", "J021")  # a one-incident penalty and an hours-late total, both right
NETWORKS = (("designer", None, []), ("2cc4", "evolved", ["Researcher"]))


def questions() -> list:
    from esp.eval.fresh import FRESH
    from esp.eval.suites import JUDGE

    chosen = [t for t in [*JUDGE, *FRESH]
              if autopsy.is_multi_penalty(t.question) or t.task_id in CONTROLS]
    return sorted(chosen, key=lambda t: t.task_id)


def stage8_hashes() -> dict[str, str]:
    """The genome each label ran as in the committed stage 8 and fresh logs."""
    hashes = {}
    for pair in uncertainty.PAIRS.values():
        for path in pair:
            run = uncertainty.load(uncertainty.RESULTS / path)
            hashes[run.label] = run.genome_hash
    return hashes


def dollars_per_question() -> dict[str, float]:
    judge = uncertainty.PAIRS["judge"]
    return {label: uncertainty.load(uncertainty.RESULTS / path).dollars
            for label, path in zip(("designer", "evolved"), judge, strict=True)}


def report(out: Path = OUT) -> dict:
    summary = {}
    for path in sorted(out.glob("*.jsonl")):
        rows = trace.read(path)
        summary[path.stem] = trace.summarise(rows)
        summary[path.stem]["right"] = sum(r["correct"] for r in rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--go", action="store_true", help="spend: ask the provider")
    parser.add_argument("--report", action="store_true", help="summarise results/trace/")
    parser.add_argument("--budget", type=float, default=0.20, help="hard cap in dollars")
    parser.add_argument("--any-provider", action="store_true",
                        help="trace even if the networks are not stage 8's")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    if args.report:
        print(json.dumps(report(args.out), indent=2))
        return 0

    # Stage 8's settings, so the networks are the ones it measured.
    os.environ.setdefault("ESP_PIN_MODELS", "1")
    os.environ.setdefault("ESP_SEARCH_RESULTS", "10")
    os.environ.setdefault("ESP_PROVIDER", "openai")
    os.environ.setdefault("ESP_DEFAULT_MODEL", "gpt-5.4-nano")
    os.environ.setdefault("ESP_MODEL_TIERS", "gpt-5.4-nano,gpt-5.4-mini")
    bootstrap()
    from select_and_judge import contender_for

    tasks = questions()
    expected, price = stage8_hashes(), dollars_per_question()
    contenders = [contender_for(*n) for n in NETWORKS]
    total = 0.0
    for c in contenders:
        same = expected.get(c.label) == c.genome_hash
        estimate = price[c.label] * len(tasks)
        total += estimate
        print(f"{c.label:10} {c.genome_hash}  stage 8: {expected.get(c.label)}"
              f"  {'same' if same else 'DIFFERENT'}  ~${estimate:.3f}")
    print(f"{len(tasks)} questions each: {', '.join(t.task_id for t in tasks)}")
    print(f"about ${total:.3f} in all at stage 8's prices; cap ${args.budget:.2f}")

    mismatch = [c.label for c in contenders if expected.get(c.label) != c.genome_hash]
    if mismatch and not args.any_provider:
        print(f"\nrefused: {', '.join(mismatch)} would not be the network stage 8 measured."
              " Configure the provider stage 8 used, or pass --any-provider.")
        return 1
    if not args.go:
        print("\nplan only: nothing spent. --go traces.")
        return 0

    from esp.eval.runner import ask_messages, read_reply, write_network
    from esp.eval.tasks import grade

    args.out.mkdir(parents=True, exist_ok=True)
    spent = sum(r["split"]["total"]["total_cost"]
                for p in args.out.glob("*.jsonl") for r in trace.read(p))
    for c in contenders:
        log = args.out / f"{c.label}.jsonl"
        done = {r["task_id"] for r in trace.read(log)} if log.exists() else set()
        hocon = str(write_network(c.genome))
        for task in tasks:
            if task.task_id in done:
                continue
            if spent >= args.budget:
                print(f"\nstopped at the cap: ${spent:.4f}")
                return 1
            messages, seconds = ask_messages(hocon, task.question)
            answer, _ = read_reply(messages)
            row = trace.record(task.task_id, messages, answer, grade(task, answer), seconds)
            spent += row["split"]["total"]["total_cost"]
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{c.label:10} {task.task_id}  {'right' if row['correct'] else 'wrong':5}"
                  f"  front man {row['split']['front_man_share']:.0%} of"
                  f" ${row['split']['total']['total_cost']:.5f}")
    print(json.dumps(report(args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
