"""Better on all four at once: the test docs/ALL-FOUR.md fixed before it ran.

Step 1, the choice on select, then step 2, the winner against the designer's shape
on judge and fresh, from the committed logs in results/all-four/. Costs nothing.

    python scripts/all_four_report.py            # print both steps
    python scripts/all_four_report.py --write    # and rewrite results/all-four/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from same_model_report import show  # noqa: E402

from esp import all_four  # noqa: E402

SUMMARY = ROOT / "results" / "all-four" / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help=f"rewrite {SUMMARY.relative_to(ROOT)}")
    args = parser.parse_args(argv)

    study = all_four.analyse()
    select = study["select"]
    print(f"The test fixed in {study['plan']}.")
    if not select["complete"]:
        print(f"\nstep 1 is not complete: {select}")
        return 1
    print("\nstep 1, select, 60 questions once each, the same hour")
    print(f"  {'':10}{'accuracy':>9}{'tokens':>8}{'$/question':>12}  network")
    for label, n in select["networks"].items():
        mark = " eligible" if label in select["eligible"] else ""
        print(f"  {label:10}{n['accuracy']:>9.1%}{n['tokens_per_question']:>8,}"
              f"{n['dollars_per_question']:>12.6f}  {n['described']}{mark}")
    winner = select["winner"]
    if winner is None:
        print("  no candidate is more accurate, cheaper and leaner than the designer's "
              "shape on select: the test stops here")
    else:
        print(f"  winner: {winner}, {select['networks'][winner]['described']}")
        for name, result in study["sets"].items():
            show(name, result, all_four.BASELINE, winner,
                 models=all_four.DESCRIBED[winner], titles=("designer", winner))
            short = result["missing_runs"][all_four.BASELINE]
            if short:
                worst = result["worst_case"]
                acc = worst["accuracy"]
                print(f"  the designer is {short} runs short of the plan; against the most "
                      f"favourable outcome for it (all {short} right, at no tokens and no "
                      f"cost): accuracy {worst['networks'][all_four.BASELINE]['accuracy']:.1%}, "
                      f"p = {acc['p']}; tokens p = {worst['tokens']['p']:.2g}; dollars "
                      f"p = {worst['dollars']['p']:.2g}")
                verdict = "YES" if result["holds_whatever_is_missing"] else "NO"
                print(f"  all five hold whatever the missing runs say: {verdict}")
        for name in study["incomplete"]:
            print(f"\n{name}: not complete yet")
    if args.write:
        SUMMARY.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
