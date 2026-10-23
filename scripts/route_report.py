"""Route each question to the cheap or the strong network, from the committed runs.

The router is fitted and its threshold chosen on the 60 select questions, then
applied once to the judge questions it never saw. It reads only the logs in
results/headtohead/paid-2026-10/, so it costs nothing and needs no key.

    python scripts/route_report.py            # print the study
    python scripts/route_report.py --write    # and rewrite results/routing/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp import route  # noqa: E402

SUMMARY = ROOT / "results" / "routing" / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help=f"rewrite {SUMMARY.relative_to(ROOT)}")
    args = parser.parse_args(argv)

    study = route.analyse()
    s, j = study["select"], study["judge"]
    cheap, strong = study["networks"]["cheap"], study["networks"]["strong"]
    print(f"cheap  {cheap[0]} ({cheap[1]})   strong  {strong[0]} ({strong[1]})\n")
    print(f"Fitted and tuned on select, {s['questions']} questions")
    print(f"  cheap {s['cheap']:.1%}, strong {s['strong']:.1%}")
    print(f"  threshold {s['threshold']:+.4f}: leave-one-out {s['leave_one_out_accuracy']:.1%}"
          f" with {s['leave_one_out_share']:.0%} sent to the strong network\n")

    print(f"Judged once on judge, {j['questions']} questions, every run of each")
    print(f"  {'':<26}{'accuracy':>9}{'$/question':>12}{'by seconds':>12}{'to strong':>11}")
    for name in ("cheap", "router", "rule", "strong"):
        p = j[name]
        print(f"  {name:<26}{p['accuracy']:>9.1%}{p['dollars']:>12.6f}"
              f"{p['dollars_by_time']:>12.6f}{p['share_to_strong']:>11.0%}")
    print(f"  {'same share, at random':<26}{j['random_same_share']:>9.1%}"
          f"   p = {j['p_against_random']} over {route.PERMUTATIONS:,} routings")
    print(f"  {'oracle (whichever right)':<26}{j['oracle']:>9.1%}")

    saved = 1 - j["router"]["dollars"] / j["strong"]["dollars"]
    print(f"\nThe router costs {saved:.0%} less than the strong network alone and "
          f"scores {100 * (j['strong']['accuracy'] - j['router']['accuracy']):.1f} "
          f"points lower.")
    print("Its choice of questions beats sending the same share at random; it does "
          "not beat\nthe one-line rule, which keeps every answer and saves less.")

    if args.write:
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
