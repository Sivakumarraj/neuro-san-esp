"""A learned Prescriptor, question by question: the test docs/PRESCRIPTOR.md fixed.

Fits ESP's Predictor on the training questions, evolves the Prescriptor against it,
chooses one from the front, and checks it on judge and fresh from the committed
logs. Costs nothing and needs no key.

    python scripts/prescriptor_report.py            # print the verdict
    python scripts/prescriptor_report.py --write    # and rewrite results/prescriptor/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp import prescription  # noqa: E402

SUMMARY = ROOT / "results" / "prescriptor" / "summary.json"


def _mix(shares: dict[str, float]) -> str:
    return ", ".join(f"{share:.0%} {action}" for action, share in shares.items() if share)


def show(study: dict) -> None:
    print(f"The test fixed in {study['plan']}.")
    for name, turn in study["turns"].items():
        chosen = turn["chosen"]
        print(f"\n{name} turn, trained on {' and '.join(turn['trained_on'])}, "
              f"{turn['questions']} questions; budget ${turn['budget']} a question")
        print(f"  static prescription: {turn['static']} for every question")
        print(f"  front of {turn['front']}; chosen: {_mix(chosen['shares'])}; the Predictor "
              f"expects {chosen['predicted']['accuracy']:.1%} at "
              f"${chosen['predicted']['dollars']}")
        for set_name, result in turn["sets"].items():
            base = result["against_baseline"]
            static = result["against_static"]
            random = result["against_random"]
            accuracy = base["accuracy"]
            print(f"\n  {set_name}, {result['questions']} questions: {_mix(result['shares'])}")
            print(f"    against the designer's shape: {accuracy['prescribed']:.1%} against "
                  f"{accuracy['other']:.1%}, sign test p = {accuracy['p']}; "
                  + ", ".join(f"{k} {'holds' if ok else 'FAILS'}"
                              for k, ok in base["holds"].items()))
            print(f"    against {static['static']} alone: {static['accuracy']['prescribed']:.1%} "
                  f"against {static['accuracy']['other']:.1%}, better on "
                  f"{len(static['accuracy']['prescribed_better_on'])}, worse on "
                  f"{len(static['accuracy']['other_better_on'])}, p = "
                  f"{static['accuracy']['p']}; dollars {static['dollars']['change']:+.1%}: "
                  f"{'holds' if static['holds'] else 'FAILS'}")
            print(f"    against the same shares at random: {random['same_shares_accuracy']:.1%}, "
                  f"p = {random['p']}: {'holds' if random['holds'] else 'FAILS'}")
            print(f"    {'PASSES' if result['passes'] else 'DOES NOT PASS'}")
            bound = result["knowing_every_answer"]
            seeds = result["across_seeds"]
            print(f"    not a test: knowing every answer, {bound['accuracy']:.1%} at "
                  f"${bound['dollars']}; over {seeds['seeds']} seeds the chosen prescriptor "
                  f"scores {seeds['accuracy'][0]:.1%} to {seeds['accuracy'][2]:.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help=f"rewrite {SUMMARY.relative_to(ROOT)}")
    args = parser.parse_args(argv)

    study = prescription.analyse()
    show(study)
    if args.write:
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
