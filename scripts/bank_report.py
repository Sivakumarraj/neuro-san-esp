"""What the held-out bank measurements say, from the committed reports alone.

`make bank-report`. No key and no calls: it reads results/heldout_bank/*.json,
written by `python -m esp.measure NETWORK --tasks meridian-bank:N --json ...`,
and sets each network's held-out result beside what it scored on the seventeen
questions it was selected on. The paired comparison is the part that matters:
two networks asked the same questions differ only where one is right and the
other wrong, and with a handful of such questions no difference is established.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from esp.eval import measurements  # noqa: E402
from esp.eval.stats import mcnemar_exact  # noqa: E402

RESULTS = ROOT / "results" / "heldout_bank"


def load(results: Path = RESULTS) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(results.glob("*.json"))}


def main() -> int:
    reports = load()
    if len(reports) < 2:
        print(f"need two reports in {RESULTS.relative_to(ROOT)}, found {len(reports)}")
        return 1
    by_hash = {record.genome_hash: record for record in measurements.load()}

    names = sorted(reports)
    first = reports[names[0]]
    print(f"{len(names)} networks on {first['suite']}, measured "
          f"{first.get('measured_on', '?')} on {first.get('key_tier', '?')}\n")
    print(f"  {'network':34} {'held out':>12} {'tokens':>9}   {'on the 17':>12} {'tokens':>9}")
    for name in names:
        report = reports[name]
        right = sum(1 for r in report["results"] if r["correct"])
        record = by_hash.get(report.get("genome_hash", ""))
        on_17 = (f"{round(record.accuracy * 17):>2}/17 {record.accuracy:.4f}"
                 if record else "           -")
        tokens_17 = f"{record.tokens:>9,}" if record else f"{'-':>9}"
        print(f"  {report['network']:34} {right:>3}/{report['questions_asked']:<3}"
              f"{report['accuracy']:>6.1%} {report['tokens']:>9,}   {on_17} {tokens_17}")

    print("\nBy depth (documents combined), right / asked:")
    depths = sorted({int(r["hops"]) for r in first["results"]})
    print("  " + " " * 34 + "".join(f"{d:>7}" for d in depths))
    for name in names:
        tally = defaultdict(lambda: [0, 0])
        for r in reports[name]["results"]:
            tally[int(r["hops"])][0] += bool(r["correct"])
            tally[int(r["hops"])][1] += 1
        print(f"  {reports[name]['network']:34}"
              + "".join(f"{tally[d][0]:>4}/{tally[d][1]:<2}" for d in depths))

    a, b = names[0], names[1]
    right_a = {r["task_id"]: bool(r["correct"]) for r in reports[a]["results"]}
    right_b = {r["task_id"]: bool(r["correct"]) for r in reports[b]["results"]}
    shared = sorted(set(right_a) & set(right_b))
    only_a = [t for t in shared if right_a[t] and not right_b[t]]
    only_b = [t for t in shared if right_b[t] and not right_a[t]]
    print(f"\nPaired, on the {len(shared)} questions both were asked:")
    print(f"  right only for {a}: {len(only_a)} {only_a}")
    print(f"  right only for {b}: {len(only_b)} {only_b}")
    print(f"  exact McNemar two-sided p = {mcnemar_exact(len(only_a), len(only_b)):.3f}")
    print("\n  With this few discordant questions no difference between the two is\n"
          "  established in either direction. What the run does settle is narrower:\n"
          "  it gives no support, out of sample, to the evolved network being better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
