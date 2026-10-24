"""The figures in docs/REPORT.md, drawn from the committed logs. No key, no calls.

    python scripts/report_figures.py            # writes results/figures/report-*.png

Each figure reads the same logs the tests recompute the README from, so a figure
cannot show a number the documents do not.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import select_and_judge as sj  # noqa: E402

from esp import route  # noqa: E402
from esp.evolve import paid  # noqa: E402

OUT = ROOT / "results" / "figures"
STAGE = paid.PAID_RUNS / paid.WINNER_STAGE
REPLICATION = ROOT / "results" / "replication" / "fresh-2026-10" / "1-fresh"

# Fixed by entity, never by rank: the designer's shape is always orange, the
# evolved network always blue, a routing of the two always aqua, and the
# designer's shape on the stronger model always violet.
COLORS = {"designer": "#eb6834", "evolved": "#2a78d6", "router": "#1baf7a", "rule": "#1baf7a",
          "designer-promoted": "#8455c9"}
NAMES = {"designer": "Designer's shape (4 agents)", "evolved": "2cc4 (2 agents)"}
# The replication has two networks of the designer's shape, told apart by model.
FRESH_NAMES = {"evolved": "2cc4 (2 agents)",
               "designer-promoted": "Designer's shape, specialists on gpt-5.4-mini",
               "designer": "Designer's shape, all gpt-5.4-nano"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# The judge set's aggregate questions, matched as tests/test_suites.py matches them.
AGGREGATE = re.compile(r"serviced by depot D\d{2}\?|by depot D\d{2},|depot D\d{2} at its own"
                       r"|incidents in \d{4}|incident in \d{4}")


class ReportFigures:
    """Draws every figure in docs/REPORT.md from the committed paid-run logs."""

    def __init__(self, out: Path = OUT):
        self.out = out
        self.rows = sj.load_rows(STAGE)
        self.summary = sj.summarise(self.rows)
        self.fresh = sj.summarise(sj.load_rows(REPLICATION))

    @staticmethod
    def _style(ax) -> None:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    def headline(self) -> Path:
        """Accuracy, tokens and dollars a question: three charts, one measure each."""
        panels = (("Accuracy", "accuracy", lambda v: f"{v:.1%}"),
                  ("Tokens per question", "tokens_per_question", lambda v: f"{v:,.0f}"),
                  ("Dollars per question", "reported_cost_per_question", lambda v: f"${v:.5f}"))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, (title, key, fmt) in zip(axes, panels, strict=True):
            labels = ["designer", "evolved"]
            values = [self.summary[label][key] for label in labels]
            bars = ax.bar([NAMES[label] for label in labels], values, width=0.55,
                          color=[COLORS[label] for label in labels])
            ax.bar_label(bars, labels=[fmt(v) for v in values], padding=3, color=INK, fontsize=9)
            ax.set_title(title, color=INK, fontsize=11, loc="left")
            ax.set_yticks([])
            ax.margins(y=0.15)
            self._style(ax)
            ax.tick_params(axis="x", labelsize=8)
        fig.suptitle("Judge questions J001-J100, twice each, 10 documents a search",
                     color=MUTED, fontsize=9, x=0.01, ha="left")
        return self._save(fig, "report-headline.png")

    def by_kind(self) -> Path:
        """Where the gain comes from: accuracy by kind of question."""
        from esp.eval.suites import JUDGE

        kind = {t.task_id: ("Aggregate" if AGGREGATE.search(t.question)
                            else f"{t.hops}-document join") for t in JUDGE}
        right: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for row in self.rows:
            for result in row["results"]:
                cell = right[(row["label"], kind[result["task_id"]])]
                cell[0] += result["correct"]
                cell[1] += 1
        kinds = sorted({k for _, k in right}, key=lambda k: (k == "Aggregate", len(k), k))
        fig, ax = plt.subplots(figsize=(10, 3.8))
        width = 0.38
        for offset, label in ((-width / 2, "designer"), (width / 2, "evolved")):
            values = [right[(label, k)][0] / right[(label, k)][1] for k in kinds]
            bars = ax.bar([i + offset for i in range(len(kinds))], values, width=width - 0.03,
                          color=COLORS[label], label=NAMES[label])
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in values], padding=2, fontsize=8,
                         color=INK)
        counts = [right[("designer", k)][1] for k in kinds]
        ax.set_xticks(range(len(kinds)), [f"{k}\n({n} runs)" for k, n in zip(kinds, counts,
                                                                             strict=True)],
                      fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
        ax.set_title("Accuracy by kind of question", color=INK, fontsize=11, loc="left", pad=24)
        ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
        self._style(ax)
        return self._save(fig, "report-by-kind.png")

    def frontier(self) -> Path:
        """Accuracy against dollars a question, for each way of answering."""
        judge = route.analyse()["judge"]
        points = (("Designer's shape alone", judge["cheap"], COLORS["designer"]),
                  ("Router (40% sent up)", judge["router"], COLORS["router"]),
                  ("One-line rule", judge["rule"], COLORS["rule"]),
                  ("2cc4 alone", judge["strong"], COLORS["evolved"]))
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for name, point, color in points:
            ax.scatter(point["dollars"], point["accuracy"], s=90, color=color,
                       edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(f"{name}\n{point['accuracy']:.1%} at ${point['dollars']:.5f}",
                        (point["dollars"], point["accuracy"]), textcoords="offset points",
                        xytext=(8, -4), fontsize=8, color=INK)
        ax.set_xlabel("Dollars per question (neuro-san's own count)", color=MUTED, fontsize=9)
        ax.set_ylabel("Accuracy on judge", color=MUTED, fontsize=9)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0.0012, 0.0032)
        ax.set_title("What each answer costs", color=INK, fontsize=11, loc="left")
        self._style(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(MUTED)
        return self._save(fig, "report-frontier.png")

    def replication(self) -> Path:
        """The fresh questions: the three networks on four measures, one chart each."""
        panels = (("Accuracy", "accuracy", lambda v: f"{v:.1%}"),
                  ("Tokens per question", "tokens_per_question", lambda v: f"{v:,.0f}"),
                  ("Model calls per question", "requests_per_question", lambda v: f"{v:.2f}"),
                  ("Dollars per correct answer", "reported_cost_per_correct",
                   lambda v: f"${v:.5f}"))
        labels = ["evolved", "designer-promoted", "designer"]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
        for ax, (title, key, fmt) in zip(axes, panels, strict=True):
            values = [self.fresh[label][key] for label in labels]
            bars = ax.bar(range(len(labels)), values, width=0.6,
                          color=[COLORS[label] for label in labels])
            ax.bar_label(bars, labels=[fmt(v) for v in values], padding=3, color=INK, fontsize=8)
            ax.set_title(title, color=INK, fontsize=10, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(y=0.15)
            self._style(ax)
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[label]) for label in labels]
        fig.legend(handles, [FRESH_NAMES[label] for label in labels], frameon=False,
                   fontsize=8, loc="lower center", ncol=3)
        fig.suptitle("Fresh questions F01-F40, twice each, 10 documents a search",
                     color=MUTED, fontsize=9, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return self._save(fig, "report-replication.png", tight=False)

    def _save(self, fig, name: str, tight: bool = True) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / name
        if tight:
            fig.tight_layout()
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        return path

    def draw(self) -> list[Path]:
        return [self.headline(), self.by_kind(), self.frontier(), self.replication()]

    @staticmethod
    def main() -> int:
        for path in ReportFigures().draw():
            print("wrote", path.relative_to(ROOT))
        return 0


if __name__ == "__main__":
    raise SystemExit(ReportFigures.main())
