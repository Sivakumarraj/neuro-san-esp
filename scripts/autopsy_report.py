"""Why the networks miss the multi-incident penalty questions.

Every wrong number is checked against the world's own hours and rates (see
esp/autopsy.py). Reads the committed logs only: no key, no cost.

    python scripts/autopsy_report.py            # print it
    python scripts/autopsy_report.py --write    # and rewrite results/autopsy/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp import autopsy  # noqa: E402

SUMMARY = ROOT / "results" / "autopsy" / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help=f"rewrite {SUMMARY.relative_to(ROOT)}")
    args = parser.parse_args(argv)

    study = autopsy.analyse()
    print("Multi-incident penalty questions, every run of each\n")
    print(f"  {'':<16}{'questions':>10}{'right':>8}{'their wrong runs / all wrong':>31}")
    for name, roles in study.items():
        for role, r in roles.items():
            s = r["share"]
            print(f"  {name + ' ' + role:<16}{s['questions']:>4} of {s['of']:<4}"
                  f"{s['runs_right']:>4}/{s['runs']:<3}"
                  f"{s['wrong_runs']:>18} / {s['all_wrong_runs']}")
    print("\nWhat each wrong number is")
    print(f"  {'':<16}" + "".join(f"{k:>22}" for k in autopsy.KINDS[1:]))
    for name, roles in study.items():
        for role, r in roles.items():
            print(f"  {name + ' ' + role:<16}"
                  + "".join(f"{r['kinds'][k]:>22}" for k in autopsy.KINDS[1:]))
    print("\nA number that fits 'missed incidents' may be a slip that landed on a partial"
          "\nsum, so that column is an upper bound on retrieval failures.")

    if args.write:
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
