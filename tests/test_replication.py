"""The replication on the forty fresh questions: its committed logs, the summary built
from them, and every figure the README, RESULTS and REPORT quote from that summary.

The paid run cannot be repeated in CI, but everything said about it can be recomputed
from the logs in results/replication/fresh-2026-10/, so a number in the documents
cannot drift from the run it describes.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "replication" / "fresh-2026-10"
STAGE = RUNS / "1-fresh"
DOCUMENTS = {name: (ROOT / name).read_text(encoding="utf-8")
             for name in ("README.md", "docs/RESULTS.md", "docs/REPORT.md")}
# Matched as tests/test_suites.py and scripts/report_figures.py match aggregates.
AGGREGATE = re.compile(r"serviced by depot D\d{2}\?|by depot D\d{2},|depot D\d{2} at its own"
                       r"|incidents in \d{4}|incident in \d{4}")
LABELS = ("evolved", "designer-promoted", "designer")

sys.path.insert(0, str(ROOT / "scripts"))
import select_and_judge as sj  # noqa: E402

from esp.eval.fresh import FRESH  # noqa: E402
from esp.evolve import paid  # noqa: E402


class Replication:
    """The replication's summary, recomputed from its logs, and the figures it gives."""

    def __init__(self):
        self.rows = sj.load_rows(STAGE)
        self.networks = sj.summarise(self.rows)
        self.pairs = {f"{a} vs {b}": sj.paired(self.rows, a, b)
                      for a, b in (("designer", "evolved"), ("designer-promoted", "evolved"))}

    def change(self, label: str, against: str, key: str) -> int:
        """Whole-percent change in `key` of `label` against `against`."""
        return round(100 * (self.networks[label][key] / self.networks[against][key] - 1))

    def by_repeat(self, label: str) -> list[int]:
        """Right answers in each repeat, in order."""
        right: dict[int, int] = defaultdict(int)
        for row in self.rows:
            if row["label"] == label:
                right[row["repeat"]] += sum(r["correct"] for r in row["results"])
        return [right[repeat] for repeat in sorted(right)]

    def by_kind(self) -> dict[tuple[str, str], int]:
        """Right answers by network and kind of question."""
        kind = {t.task_id: ("Aggregate" if AGGREGATE.search(t.question) else f"{t.hops}-document")
                for t in FRESH}
        right: dict[tuple[str, str], int] = defaultdict(int)
        for row in self.rows:
            for result in row["results"]:
                right[(row["label"], kind[result["task_id"]])] += result["correct"]
        return right


def test_the_committed_summary_is_what_the_logs_say():
    committed = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    run = Replication()
    assert committed == {STAGE.name: {"networks": run.networks, "pairs": run.pairs}}


def test_the_control_the_web_page_quotes_is_what_the_logs_say():
    """apps/web reads the model control through paid.control(); it must be the
    figures the logs give, not a second copy of them."""
    run = Replication()
    control = paid.control()
    beat = run.pairs["designer-promoted vs evolved"]["better_on"]
    assert control.questions == len(FRESH)
    assert control.winner_better_on == len(beat["evolved"])
    assert control.control_better_on == len(beat["designer-promoted"])
    assert control.sign_test_p == run.pairs["designer-promoted vs evolved"]["sign_test_p"]
    assert control.gain == pytest.approx(run.networks["designer-promoted"]["accuracy"]
                                         - run.networks["designer"]["accuracy"])


def test_the_control_is_none_where_the_replication_is_not_shipped(tmp_path):
    assert paid.control(tmp_path / "summary.json") is None


def test_the_run_is_the_one_described():
    run = Replication()
    assert sorted(run.networks) == sorted(LABELS)
    for label in LABELS:
        assert run.networks[label]["question_runs"] == 2 * len(FRESH)
        assert run.networks[label]["unfinished"] == 0
    hashes = {row["label"]: row["genome_hash"] for row in run.rows}
    assert hashes["evolved"] == paid.WINNER_HASH
    assert {row["settings"]["search_results"] for row in run.rows} == {paid.WINNER_SEARCH_RESULTS}
    assert {task for row in run.rows for task in (r["task_id"] for r in row["results"])} == \
        {t.task_id for t in FRESH}


def _row(name: str, cells: list[str]) -> str:
    return f"| {name} | " + " | ".join(cells) + " |"


def test_every_figure_in_the_documents_is_what_the_logs_say():
    run = Replication()
    e, c, d = (run.networks[label] for label in LABELS)
    p_d = run.pairs["designer vs evolved"]["sign_test_p"]
    p_c = run.pairs["designer-promoted vs evolved"]["sign_test_p"]
    beat_d = run.pairs["designer vs evolved"]["better_on"]
    beat_c = run.pairs["designer-promoted vs evolved"]["better_on"]
    # Whole-percent changes of 2cc4 against the designer (d_) and the control (c_).
    d_tok, d_calls = (run.change("evolved", "designer", key)
                      for key in ("tokens_per_question", "requests_per_question"))
    d_cost, d_correct = (run.change("evolved", "designer", key)
                         for key in ("reported_cost_per_question", "reported_cost_per_correct"))
    d_cost_cap, d_correct_cap = (run.change("evolved", "designer", key)
                                 for key in ("cost_per_question", "cost_per_correct"))
    c_tok, c_calls, c_cost, c_correct = (
        run.change("evolved", "designer-promoted", key)
        for key in ("tokens_per_question", "requests_per_question", "reported_cost_per_question",
                    "reported_cost_per_correct"))
    spent = sum(net["cost"] for net in run.networks.values())

    def accuracy(repeats: bool) -> str:
        cells = []
        for label, net in zip(LABELS, (e, c, d), strict=True):
            figure = f"{net['accuracy']:.1%}"
            figure = f"**{figure}**" if net is e else figure
            detail = f"{net['correct']}/{net['question_runs']}"
            if repeats:
                first, second = run.by_repeat(label)
                detail += f"; {first}, then {second}"
            cells.append(f"{figure} ({detail})")
        return _row("Accuracy", cells)

    def money(name: str, key: str) -> str:
        return _row(name, [f"${e[key]:.5f}", f"${c[key]:.5f}", f"**${d[key]:.5f}**"])

    quoted = {
        "README.md": [
            accuracy(repeats=False),
            _row("Tokens per question", [f"**{e['tokens_per_question']:,}**",
                                         f"{c['tokens_per_question']:,}",
                                         f"{d['tokens_per_question']:,}"]),
            _row("Model calls per question", [f"**{e['requests_per_question']:.2f}**",
                                              f"{c['requests_per_question']:.2f}",
                                              f"{d['requests_per_question']:.2f}"]),
            money("Cost per question", "reported_cost_per_question"),
            money("Cost per correct answer", "reported_cost_per_correct"),
            f"on {len(beat_d['evolved'])}\n  questions against {len(beat_d['designer'])} "
            f"(p = {p_d}), with {-d_tok}% fewer tokens and {-d_calls}% fewer model calls",
            f"bought the designer's\n  shape {100 * (c['accuracy'] - d['accuracy']):.1f} points",
            f"on {len(beat_c['evolved'])} questions\n  against {len(beat_c['designer-promoted'])} "
            f"with {-c_tok}% fewer tokens, {-c_calls}% fewer model calls and {-c_correct}% less "
            "per correct answer",
            f"(p = {p_c})",
            f"costs {d_cost}% more a question and {d_correct}%\n  more a correct answer",
            f"{100 * (e['accuracy'] - d['accuracy']):.0f} points of accuracy for\n  about "
            f"${e['reported_cost_per_question'] - d['reported_cost_per_question']:.4f} a question",
            f"The run cost ${spent:.2f}",
        ],
        "docs/RESULTS.md": [
            accuracy(repeats=True),
            f"better on {len(beat_d['evolved'])} questions, worse on {len(beat_d['designer'])} "
            f"({', '.join(beat_d['designer'])})",
            f"better on {len(beat_c['evolved'])} questions, worse on "
            f"{len(beat_c['designer-promoted'])} ({', '.join(beat_c['designer-promoted'])}), "
            f"p = {p_c}",
            f"Tokens \N{MINUS SIGN}{-d_tok}%, model calls \N{MINUS SIGN}{-d_calls}%, cost "
            f"+{d_cost}% a question and +{d_correct}% a correct answer",
            f"(+{d_cost_cap}% and +{d_correct_cap}% on the cap's more cautious count)",
            f"Tokens \N{MINUS SIGN}{-c_tok}%,\n  model calls \N{MINUS SIGN}{-c_calls}%, cost "
            f"\N{MINUS SIGN}{-c_cost}% a question and \N{MINUS SIGN}{-c_correct}% a correct answer",
        ],
        "docs/REPORT.md": [
            accuracy(repeats=False),
            money("Dollars per correct answer", "reported_cost_per_correct"),
            f"p = {p_d}, with {-d_tok}% fewer tokens and {-d_calls}% fewer model calls",
            f"(p = {p_c}, short of 0.05",
            f"costs {d_cost}% more a question and\n  {d_correct}% more a correct answer",
            f"The run spent ${spent:.2f}",
        ],
    }
    missing = [(name, figure) for name, figures in quoted.items() for figure in figures
               if figure not in DOCUMENTS[name]]
    assert not missing, missing


def test_accuracy_by_kind_is_what_the_logs_say():
    right = Replication().by_kind()
    joins = sum(right[(label, f"{h}-document")] for label in ("evolved",) for h in (1, 2, 3))
    assert joins == sum(right[("designer", f"{h}-document")] for h in (1, 2, 3)) == 24
    table = DOCUMENTS["docs/RESULTS.md"]
    expected = [
        f"| Aggregate (32) | **{right[('evolved', 'Aggregate')]}** | "
        f"{right[('designer-promoted', 'Aggregate')]} | {right[('designer', 'Aggregate')]} |",
        f"| 4-document join (8) | **{right[('evolved', '4-document')]}** | "
        f"{right[('designer-promoted', '4-document')]} | {right[('designer', '4-document')]} |",
        f"| 6-document join (8) | {right[('evolved', '6-document')]} | "
        f"{right[('designer-promoted', '6-document')]} | {right[('designer', '6-document')]} |",
        f"| 9-document join (8) | {right[('evolved', '9-document')]} | "
        f"{right[('designer-promoted', '9-document')]} | {right[('designer', '9-document')]} |",
    ]
    missing = [line for line in expected if line not in table]
    assert not missing, missing
