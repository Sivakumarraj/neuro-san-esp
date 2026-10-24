"""Is the network sure? Agreement between two runs, as a gate and as a flag.

Every judge and fresh question was asked twice of each network, so whether the
two runs agree is already in the committed logs. This reads them and nothing
else: it costs nothing and needs no key. Beliefs are fitted on judge and
checked once on fresh.

The report keeps three kinds of statement apart: what was counted in the logs,
what was computed from those counts, and what these logs cannot say.

    python scripts/uncertainty_report.py            # print the study
    python scripts/uncertainty_report.py --write    # and rewrite results/uncertainty/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp import route, uncertainty  # noqa: E402

SUMMARY = ROOT / "results" / "uncertainty" / "summary.json"

NAMES = {"cheap": "designer's shape, all nano", "strong": "2cc4, Researcher on mini",
         "control": "designer's shape, specialists on mini"}
ORDER = ("cheap", "rule", "strong", "gate", "rule_then_gate", "gate_if_free",
         "flag_cheap", "flag_strong")


def _p(p: float) -> str:
    return f"p < {1 / route.PERMUTATIONS}" if p == 0 else f"p = {p}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help=f"rewrite {SUMMARY.relative_to(ROOT)}")
    args = parser.parse_args(argv)

    study = uncertainty.analyse()
    cal, sets, beliefs = study["calibration"], study["sets"], study["beliefs"]

    print("OBSERVED: counted in the committed logs\n")
    print("When a network's two runs agree, how often is it right?")
    print(f"  {'':<10}{'network':<40}{'agree':>7}{'right if agree':>16}"
          f"{'if not':>8}")
    for name, rows in cal.items():
        for role, c in rows.items():
            print(f"  {name:<10}{NAMES[role]:<40}{c['agree_share']:>7.0%}"
                  f"{c['right_if_agree']:>16.1%}{c['right_if_disagree']:>8.1%}")

    for name, s in sets.items():
        print(f"\n{name}, {s['questions']} questions, every run of each")
        print(f"  {'':<22}{'accuracy':>9}{'$/question':>12}{'answers':>9}{'to strong':>11}")
        for key in ORDER:
            p = s["policies"][key]
            print(f"  {key:<22}{p['accuracy']:>9.1%}{p['dollars']:>12.6f}"
                  f"{p['coverage']:>9.0%}{p['share_to_strong']:>11.0%}")
        print(f"  disagreements the rule sends up anyway: "
              f"{s['disagreed_and_rule_sends_up']} of {s['disagreed']}")
        print(f"  agreed on a wrong answer: {', '.join(s['agreed_and_wrong']) or 'none'}")

    print("\nDERIVED: computed from those counts\n")
    for name, s in sets.items():
        gate = s["policies"]["gate"]
        print(f"{name}:")
        print(f"  the gate is dominated by {', '.join(s['gate_dominated_by'])}: none of"
              f" them is less often right or dearer")
        print(f"  the same share sent up at random scores {s['random_same_share']:.1%};"
              f" the gate's choice {gate['accuracy']:.1%}, {_p(s['p_against_random'])}")
        print(f"  cheap costs {s['price_ratio']:.3f} of strong; a gate is cheaper than the"
              f" strong network alone only below {s['break_even_ratio']:.3f}")
        print(f"  a cheaper probe of agreement would have to cost at most"
              f" {s['probe_budget']:.0%} of a cheap run to cost no more than the rule")
    fit = beliefs["fitted_on_judge"]
    print("\nBeliefs fitted on judge, Beta(1, 1) priors:")
    for key, b in fit.items():
        held = beliefs["held_out"][key]
        print(f"  {key:<20}{b['mean']:>7.1%}   on fresh {held['observed_on_fresh']:.1%}"
              f" over {held['fresh_questions']} questions")
    g = beliefs["held_out"]["gate_accuracy"]
    print(f"  gate accuracy on fresh, predicted {g['fitted_on_judge']:.1%},"
          f" observed {g['observed_on_fresh']:.1%}")
    v = beliefs["break_even_value"]
    print(f"\nSend a question up on disagreement when a right answer is worth more than"
          f" ${v['disagree']:.4f};")
    print(f"on agreement only when it is worth more than ${v['agree']:.4f}.")

    print("\nNOT KNOWN: what these logs cannot say\n")
    print("  - whether agreement measured more cheaply than a whole second run (one")
    print("    agent's reply resampled) keeps the signal: nothing here resampled less")
    print("  - semantic density: the answers are mostly bare numbers, and distance in")
    print("    an embedding is not distance between two numbers")
    print("  - more than two samples: every question was asked exactly twice")
    print("  - another question world: every question is Meridian's")

    if args.write:
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
