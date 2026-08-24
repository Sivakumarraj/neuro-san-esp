"""Render an ESP run as a PDF.

Whatever the run found, including a negative result. A report that can only
describe a win is not a report.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.platypus import PageBreak, Spacer

from esp.report.layout import (
    AMBER,
    CW,
    GOOD,
    GOOD_BG,
    LEAD,
    WARN_BG,
    Layout,
    _s,
)


class Report(Layout):
    def __init__(self, results_dir: str | Path = "results"):
        super().__init__(results_dir)
        self.history = json.loads((self.dir / "history.json").read_text())


    def compose(self) -> list:
        """Build the run-report story without rendering it.

        Split out so the repository dossier can carry these pages itself rather
        than shipping a second PDF -- a document called "the whole project" that
        needs a companion file is not one document.
        """
        history = self.history
        records = history["records"]
        seeds = [r for r in records if r["origin"].startswith("seed:")]
        best = max(records, key=lambda r: r["fitness"])
        best_seed = max(seeds, key=lambda r: r["fitness"]) if seeds else None
        # A search happened only if the surrogate actually scored candidates and
        # at least one generation produced records beyond the seeds.
        searched = bool(history["surrogate_evaluations"]) and any(
            r["generation"] > 0 for r in records)
        beat = best_seed and best["genome_hash"] != best_seed["genome_hash"] \
            and best["fitness"] > best_seed["fitness"]

        self.h1("Evolving neuro-san agent networks",
                "Evolutionary Surrogate-assisted Prescription applied to agent "
                "topologies &mdash; measured, not asserted.")

        self.p(
            "neuro-san can turn a sentence into a working multi-agent network. It "
            "generates <b>one</b>, and never measures it: there is no fitness function "
            "anywhere in the framework. This run adds the missing half &mdash; a way to "
            "score a network, and a search for a better one.", LEAD)

        # --- headline
        rows = [[
            "Real evaluations", str(history["real_evaluations"]),
            "Surrogate evaluations", f"{history['surrogate_evaluations']:,}",
        ], [
            "Best seed (baseline)",
            f"{best_seed['accuracy']:.2f} acc / {best_seed['tokens']:,} tok"
            if best_seed else "&mdash;",
            # Without a search there is no evolved candidate, and echoing the
            # seed's own numbers under that label would read as one.
            "Best evolved" if searched else "Evolved candidates",
            f"{best['accuracy']:.2f} acc / {best['tokens']:,} tok"
            if searched else "none &mdash; no search ran",
        ]]
        self.table(["", "", "", ""], rows,
                   widths=[CW * 0.26, CW * 0.24, CW * 0.26, CW * 0.24])

        saved = history["surrogate_evaluations"]
        self.callout(
            "What the surrogate bought",
            f"The search examined <b>{saved:,}</b> candidate networks and paid to run "
            f"only <b>{history['real_evaluations']}</b> of them. Every candidate scored "
            "by the Predictor costs microseconds; every real evaluation costs minutes "
            "of language-model time. That ratio is the entire reason this is ESP and "
            "not a plain genetic algorithm."
            if searched else
            "No search was run, so the Predictor bought nothing here. The numbers below "
            "are measurement only.")

        # --- result, honestly
        self.h2("The result")
        if not searched:
            # A search that ran and lost, and a search that never ran, are
            # different claims. Reporting the second as the first would be the
            # kind of quiet overstatement this whole project exists to avoid.
            self.callout(
                "No search was run &mdash; this is a baseline measurement",
                "The provider's free tier allows 500 requests per day per model, and a "
                "candidate costs roughly 165 across the task set. One model's daily "
                "allowance therefore buys three candidates: enough to measure the seed "
                "topologies against each other, not enough to evolve. The quota was "
                "exhausted before any generation completed. What follows is what was "
                "actually measured; no evolved candidate exists to compare against, and "
                "none is claimed.",
                bg=WARN_BG, bar=AMBER)
        elif beat:
            delta_acc = best["accuracy"] - best_seed["accuracy"]
            delta_tok = best_seed["tokens"] - best["tokens"]
            self.callout(
                "Evolution beat the baseline",
                f"Best evolved network: <b>{best['accuracy']:.2f}</b> accuracy on "
                f"<b>{best['tokens']:,}</b> tokens with {best['agents']} agents, found "
                f"in generation {best['generation']} by <i>{best['origin']}</i>. "
                f"Against the best seed that is "
                f"{delta_acc:+.2f} accuracy and {delta_tok:+,} tokens.",
                bg=GOOD_BG, bar=GOOD)
        else:
            self.callout(
                "Evolution did not beat the baseline",
                "No evolved candidate scored above the best seed on the combined "
                "objective. That is the result, and it is reported as found. A clean "
                "harness with an honest negative outcome is a measurement; a tuned "
                "number would not be.",
                bg=WARN_BG, bar=AMBER)

        self.figure("fitness.png",
                    "Fitness over generations. The dotted line is the best seed &mdash; "
                    "the network a one-shot designer would have handed you.")

        self.story.append(PageBreak())

        # --- how it works
        self.h1("Method", "Four phases, repeated. Phase C is the point.")
        self.terminal(
            "Phase A   seed population evaluated for real      -->  (genome, fitness)\n"
            "Phase B   train a Predictor on those pairs\n"
            "Phase C   evolve thousands of candidates against it  --  zero LLM calls\n"
            "Phase D   real-evaluate only the elite, feed back to B")

        self.p(
            "The genome is neuro-san's own <font face='Courier' size='9'>"
            "agent_network_definition</font> with one addition, a per-agent model. "
            "neuro-san already overlays a per-agent "
            "<font face='Courier' size='9'>llm_config</font> over the network default "
            "and nobody tunes it, even though it is the main cost/quality knob in a "
            "multi-agent network.")

        self.h2("Mutation operators")
        self.table(
            ["Operator", "What it changes"],
            [["add_agent / remove_agent", "agent count; removal re-parents children"],
             ["rewire", "one edge, at fixed size"],
             ["split_agent / merge_agents",
              "granularity &mdash; the axis a designer told to &lsquo;prefer the fewest "
              "agents&rsquo; never explores"],
             ["toggle_search", "which agents can read the corpus"],
             ["reassign_model", "per-agent model tier"]],
            widths=[CW * 0.32, CW * 0.68])

        self.p(
            "Every mutant is checked &mdash; one top agent, a DAG, no unreachable "
            "agents, no dangling references, at least one agent able to search &mdash; "
            "and <b>discarded if it fails, never repaired</b>. A repaired mutant is a "
            "different mutant than the operator produced, and silently substituting one "
            "would make the search unable to learn what its own operators do.")

        self.h2("The evaluation world")
        self.p(
            "A fictional logistics company: 24 depots, 40 contracts, 60 incidents, "
            "<b>124 documents</b>, and 17 questions spanning one to four hops. The world "
            "is synthetic so that no answer can be recalled from pretraining &mdash; "
            "fitness measures the topology, not the model's memory. Corpus and ground "
            "truth come from one seeded data model, so every answer is correct by "
            "construction. Retrieval is deterministic term overlap returning three "
            "documents: fitness must be a property of the network, not of a retrieval "
            "layer that drifts between generations.")

        self.story.append(PageBreak())

        # --- surrogate
        self.h1("The Predictor", "Whether the surrogate earned its place.")
        quality = history.get("surrogate_quality") or []
        if quality:
            self.table(
                ["Generation", "Samples", "Spearman", "MAE", "Verdict"],
                [[q["generation"], q["samples"],
                  "not measured" if q.get("spearman") is None
                  else f"{q['spearman']:+.3f}",
                  "&mdash;" if q.get("mae") is None or q["mae"] != q["mae"]
                  else f"{q['mae']:.4f}",
                  "ranks better than chance" if q["beats_random"]
                  else ("too few samples to measure"
                        if q.get("spearman") is None else "no better than chance")]
                 for q in quality],
                widths=[CW * 0.16, CW * 0.14, CW * 0.16, CW * 0.14, CW * 0.40],
                highlight=[i for i, q in enumerate(quality) if q["beats_random"]])

        self.p(
            "A surrogate trained on tens of samples is weak, and this one is trained on "
            "tens of samples. Its job is not to be right &mdash; it is to <i>rank</i>, "
            "well enough that the elite it selects are worth paying to run. The "
            "cross-validated rank correlation is published above whatever it says.")

        self.figure("surrogate.png",
                    "Predicted against real fitness for the candidates the surrogate "
                    "chose. Points on the dotted line would be perfect prediction.",
                    width=CW * 0.62)

        self.story.append(PageBreak())

        # --- pareto
        self.h1("Accuracy against cost",
                "A single fitness number hides the trade-off. The front does not.")
        self.figure("pareto.png",
                    "Every network evaluated. Diamonds are seeds, green is the "
                    "non-dominated front.")

        front = history.get("pareto", [])
        if front:
            self.table(
                ["Network", "Origin", "Accuracy", "Tokens", "Agents", "Depth"],
                [[r["genome_hash"][:10], r["origin"], f"{r['accuracy']:.2f}",
                  f"{r['tokens']:,}", r["agents"], r["depth"]] for r in front],
                widths=[CW * 0.20, CW * 0.26, CW * 0.14, CW * 0.16, CW * 0.12,
                        CW * 0.12])

        self.story.append(PageBreak())

        # --- everything evaluated
        self.h1("Every network evaluated",
                "The full record, in the order it was measured.")
        self.table(
            ["Gen", "Origin", "Network", "Acc", "Tokens", "Agents", "Fitness", "Pred"],
            [[r["generation"], r["origin"], r["genome_hash"][:8],
              f"{r['accuracy']:.2f}", f"{r['tokens']:,}", r["agents"],
              f"{r['fitness']:+.4f}",
              "&mdash;" if r.get("predicted") is None else f"{r['predicted']:+.3f}"]
             for r in records],
            widths=[CW * 0.07, CW * 0.24, CW * 0.13, CW * 0.09, CW * 0.13,
                    CW * 0.10, CW * 0.13, CW * 0.11])

        self.h2("What measurement changed")
        self.callout(
            "Two findings that came from running it, not reasoning about it",
            "<b>Free-tier quota is per-model and wildly uneven.</b> "
            "<font face='Courier' size='8.5'>gemini-3.6-flash</font> allows 20 requests "
            "per <i>day</i>, which cannot support an evolutionary run at all; "
            "<font face='Courier' size='8.5'>gemini-3.5-flash-lite</font> allows 15 per "
            "<i>minute</i>. Rate limiting became a correctness requirement, not a "
            "politeness one: a 429 returns as an agent error, the candidate scores zero, "
            "and the search learns that a good topology is bad.<br/><br/>"
            "<b>The first task set had no headroom.</b> A single agent with a single "
            "tool scored 1.00 on the original 13 questions. There was nothing for "
            "evolution to improve, so the corpus went from 40 documents to 124, "
            "retrieval narrowed from five results to three, and four-hop and aggregate "
            "shapes were added. An experiment whose baseline is already perfect measures "
            "nothing.",
            bg=WARN_BG, bar=AMBER)

        self.story.append(Spacer(1, 4))
        self.h2("Limitations")
        for text in [
            "One task domain. A topology that wins at multi-hop retrieval need not win "
            "elsewhere.",
            "The surrogate is trained on tens of samples. It ranks; it does not price.",
            "The designer baseline is a faithful reconstruction of the shape "
            "<font face='Courier' size='9'>agent_network_designer</font> produces, not "
            "its literal output &mdash; the designer wires networks against its own "
            "toolbox and cannot see this corpus.",
            "Automated agent architecture search is an active field (ADAS, AFlow, "
            "GPTSwarm, MaAS). The idea is not new. What is absent from all of it is "
            "neuro-san, and what is absent from neuro-san is any fitness function.",
        ]:
            self.p(f"&bull;&nbsp; {text}", _s("li", fontSize=9.4, leading=13,
                                              leftIndent=8, spaceAfter=4))

        return self.story

    def build(self, out: str | Path | None = None) -> Path:
        self.compose()
        out = Path(out) if out else self.dir / "ESP-Report.pdf"
        return self.render(out, "Evolving neuro-san agent networks")


if __name__ == "__main__":
    print("wrote", Report().build())
