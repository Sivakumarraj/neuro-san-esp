"""Charts for an ESP run.

Three questions, three figures: did fitness improve, what did the search buy
per real evaluation, and what is the accuracy/cost trade-off it found.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#101418"
ACCENT = "#1a4fa0"
GOOD = "#1d7a46"
BAD = "#b3261e"
MUTED = "#5b6672"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9d2db")
    ax.spines["bottom"].set_color("#c9d2db")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color="#eef1f4", linewidth=0.8)
    ax.set_axisbelow(True)


def fitness_curve(history: dict, out: Path) -> Path:
    records = history["records"]
    generations = sorted({r["generation"] for r in records})

    best_so_far, per_gen_best = [], []
    running = float("-inf")
    for generation in generations:
        upto = [r["fitness"] for r in records if r["generation"] <= generation]
        this = [r["fitness"] for r in records if r["generation"] == generation]
        running = max(upto)
        best_so_far.append(running)
        per_gen_best.append(max(this))

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
    ax.plot(generations, best_so_far, "-o", color=ACCENT, linewidth=2,
            markersize=5, label="best so far")
    ax.plot(generations, per_gen_best, "--o", color=MUTED, linewidth=1,
            markersize=4, alpha=0.8, label="best in generation")

    seeds = [r["fitness"] for r in records if r["origin"].startswith("seed:")]
    if seeds:
        ax.axhline(max(seeds), color=BAD, linewidth=1.2, linestyle=":",
                   label="best seed (baseline)")

    ax.set_xlabel("generation", fontsize=9, color=INK)
    ax.set_ylabel("fitness", fontsize=9, color=INK)
    ax.set_title("Fitness over generations", fontsize=11, color=INK, loc="left")
    ax.set_xticks(generations)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def surrogate_scatter(history: dict, out: Path) -> Path:
    """Predicted against real fitness for candidates the surrogate chose.

    The point of ESP is that the Predictor ranks well enough to be worth
    trusting. If these points show no relationship, the run still works but the
    surrogate is not earning its place, and the chart says so plainly.
    """
    pairs = [(r["predicted"], r["fitness"]) for r in history["records"]
             if r.get("predicted") is not None]

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=160)
    if pairs:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.scatter(xs, ys, s=42, color=ACCENT, alpha=0.8, edgecolor="white",
                   linewidth=0.8)
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=":",
                label="perfect prediction")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no surrogate-selected candidates", ha="center",
                color=MUTED, fontsize=9, transform=ax.transAxes)

    ax.set_xlabel("predicted fitness", fontsize=9, color=INK)
    ax.set_ylabel("real fitness", fontsize=9, color=INK)
    ax.set_title("Surrogate calibration", fontsize=11, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def pareto(history: dict, out: Path) -> Path:
    records = history["records"]
    front = {r["genome_hash"] for r in history.get("pareto", [])}

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for record in records:
        on_front = record["genome_hash"] in front
        is_seed = record["origin"].startswith("seed:")
        ax.scatter(record["tokens"], record["accuracy"],
                   s=90 if on_front else 40,
                   color=GOOD if on_front else ("#9aa6b2" if not is_seed else BAD),
                   marker="D" if is_seed else "o",
                   alpha=0.95 if on_front else 0.65,
                   edgecolor="white", linewidth=0.8, zorder=3 if on_front else 2)

    ordered = sorted(history.get("pareto", []), key=lambda r: r["tokens"])
    if len(ordered) > 1:
        ax.step([r["tokens"] for r in ordered], [r["accuracy"] for r in ordered],
                where="post", color=GOOD, linewidth=1.2, alpha=0.7, zorder=1)

    ax.set_xlabel("tokens used across the task set", fontsize=9, color=INK)
    ax.set_ylabel("accuracy", fontsize=9, color=INK)
    ax.set_title("Accuracy against cost  ·  diamonds are seeds, green is the "
                 "Pareto front", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(results_dir: str | Path = "results") -> list[Path]:
    results = Path(results_dir)
    history = json.loads((results / "history.json").read_text())
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    return [
        fitness_curve(history, figures / "fitness.png"),
        surrogate_scatter(history, figures / "surrogate.png"),
        pareto(history, figures / "pareto.png"),
    ]


if __name__ == "__main__":
    for path in render_all():
        print("wrote", path)
