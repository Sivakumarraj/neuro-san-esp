"""Same model, fewer agents: the test docs/SAME-MODEL.md fixed before it ran.

Reads the committed logs in results/structure/ and checks the five criteria the
page lists. Costs nothing and needs no key.

    python scripts/same_model_report.py            # print the verdict
    python scripts/same_model_report.py --write    # and rewrite results/structure/summary.json
    python scripts/same_model_report.py --dir runs/same-model   # your own run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp import same_model  # noqa: E402

SUMMARY = ROOT / "results" / "structure" / "summary.json"

ROWS = (("accuracy", "accuracy", "{:.1%}"),
        ("tokens a question", "tokens_per_question", "{:,}"),
        ("dollars a question", "dollars_per_question", "${:.5f}"),
        ("dollars a right answer", "dollars_per_correct", "${:.5f}"),
        ("agents", "agents", "{}"),
        ("token fitness", "token_fitness", "{:+.4f}"),
        ("dollar fitness", "dollar_fitness", "{:+.4f}"))


def show(name: str, result: dict) -> None:
    base = result["networks"][same_model.BASELINE]
    cand = result["networks"][same_model.CANDIDATE]
    print(f"\n{name}: {base['question_runs'] // 2} questions x 2, every agent on "
          f"{same_model.MODEL}, {same_model.SEARCH_RESULTS} documents a search")
    print(f"  {'':24}{'designer':>12}{'2cc4':>12}")
    for title, key, shape in ROWS:
        print(f"  {title:24}{shape.format(base[key]):>12}{shape.format(cand[key]):>12}")
    acc = result["accuracy"]
    low, high = acc["ci95_points"]
    print(f"  accuracy: 2cc4 right more often on {len(acc['candidate_better_on'])} questions, "
          f"the designer on {len(acc['baseline_better_on'])}; sign test p = {acc['p']}; "
          f"{acc['points']:+.1f} points, 95% interval {low:+.1f} to {high:+.1f}")
    for key in ("tokens", "dollars"):
        test = result[key]
        print(f"  {key}: {test['change']:+.1%} for 2cc4, lower in {test['candidate_lower_in']} "
              f"of {test['chunks']} chunks; Wilcoxon p = {test['p']:.2g}")
    print("  " + ", ".join(f"{criterion} {'holds' if ok else 'FAILS'}"
                           for criterion, ok in result["holds"].items()))
    print(f"  all five hold: {'YES' if result['all_hold'] else 'NO'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help=f"rewrite {SUMMARY.relative_to(ROOT)}")
    parser.add_argument("--dir", type=Path, default=None,
                        help="read your own run: a folder holding judge/ or fresh/")
    args = parser.parse_args(argv)
    if args.write and args.dir:
        parser.error("--write records the committed logs only")

    suites = None if args.dir is None else {name: args.dir / name
                                            for name in same_model.SUITES}
    study = same_model.analyse(suites)
    if not study["suites"]:
        print("no logs yet: run the plan in docs/SAME-MODEL.md (make same-model-run)")
        return 1
    print(f"The test fixed in {study['plan']}; judge is the test, fresh the replication.")
    for name, result in study["suites"].items():
        show(name, result)
    if args.write:
        SUMMARY.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
