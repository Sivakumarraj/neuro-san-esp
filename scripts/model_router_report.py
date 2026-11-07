"""Choosing the Researcher's model per question: the test docs/MODEL-ROUTER.md fixed.

Fits the router on the select questions, then checks the five pass marks on judge and
fresh from the committed logs. Costs nothing and needs no key.

    python scripts/model_router_report.py            # print the verdict
    python scripts/model_router_report.py --write    # and rewrite results/router/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp import model_router  # noqa: E402

SUMMARY = ROOT / "results" / "router" / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help=f"rewrite {SUMMARY.relative_to(ROOT)}")
    args = parser.parse_args(argv)

    study = model_router.analyse()
    select = study["select"]
    price = {action: select["select"][action]["dollars_per_question"]
             for action in ("cheap", "strong")}
    print(f"The test fixed in {study['plan']}.")
    print(f"\nselect, {select['questions']} questions: cheap ${price['cheap']}, strong "
          f"${price['strong']}, baseline ${select['budget']} a question; the budget "
          f"allows {select['most_sent_up_within_budget']} up")
    if select["routes_nothing"]:
        print("  no threshold within the budget beats sending nothing up: the router "
              "is the cheap action")
    else:
        print(f"  threshold {select['threshold']:+.4f}: leave-one-out "
              f"{select['leave_one_out']['accuracy']:.1%} with "
              f"{select['leave_one_out']['share_to_strong']:.0%} sent up")
    for name, result in study["sets"].items():
        acc = result["accuracy"]
        print(f"\n{name}, {result['questions']} questions, {result['sent_up']} sent up")
        print(f"  accuracy: router {acc['routed']:.1%}, baseline {acc['baseline']:.1%} "
              f"(cheap alone {acc['cheap']:.1%}, strong alone {acc['strong']:.1%}); "
              f"sign test p = {acc['p']}")
        for key in ("tokens", "dollars"):
            test = result[key]
            print(f"  {key}: router {test['routed']:,.6g}, baseline {test['baseline']:,.6g} "
                  f"({test['change']:+.1%}); Wilcoxon p = {test['p']:.2g}")
        print("  " + ", ".join(f"{k} {'holds' if ok else 'FAILS'}"
                               for k, ok in result["holds"].items()))
        bound = result["oracle_within_budget"]
        print(f"  not a test: knowing every answer, within the budget, {bound['sent_up']} up "
              f"would score {bound['accuracy']:.1%} (better on {bound['better_on']}, worse "
              f"on {bound['worse_on']}, p = {bound['p']})")
    if args.write:
        SUMMARY.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
