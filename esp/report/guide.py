"""The beginner's guide: how this repository runs, step by step, with examples.

The Primer and the Explainer say what the project found. This one says how to
use it: what each word means, what the test world looks like, how one question
travels through a network, how a score is computed, and what every command
prints -- for somebody who has never run it.

Every example is taken from the repository at build time, not typed: a real
document from the corpus, a real benchmark question and its answer, the fitness
formula applied to a real measurement, and command output captured from real
runs (`docs/proofs/`). If the code changes, the guide changes with it.

    python -m esp.report.guide      # or: make guide
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.platypus import PageBreak, Spacer

from esp.eval.bank import BANK
from esp.eval.suites import JUDGE, SELECT
from esp.eval.tasks import TASKS
from esp.eval.world import build_world, documents
from esp.evolve.loop import scalarise
from esp.report.layout import ACCENT, SOFT, WARN_BG, Layout, _s

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "docs" / "neuro-san-esp-Beginner-Guide.pdf"

STEP = _s("gstep", fontSize=9.8, leading=14, leftIndent=12, spaceAfter=5)


def _history() -> dict:
    return json.loads((ROOT / "results" / "history.json").read_text(encoding="utf-8"))


def _proof(name: str, lines: int = 40) -> str:
    path = ROOT / "docs" / "proofs" / f"{name}.txt"
    if not path.exists():
        return "(not captured yet: run make proofs)"
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[:lines])


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Guide(Layout):
    def __init__(self, results_dir: str | Path | None = None):
        super().__init__(results_dir or ROOT / "results")

    def step(self, index: int, title: str, text: str) -> None:
        self.p(f'<font color="#1a4fa0"><b>Step {index}. {title}</b></font>',
               _s("gst", fontSize=10.6, leading=14, spaceBefore=8, spaceAfter=2))
        self.p(text, STEP)

    # ------------------------------------------------------------------ pages
    def _cover(self) -> None:
        self.h1("neuro-san-esp: a beginner's guide",
                "What it does, how it runs, and what every command prints.")
        self.p("neuro-san is a framework from Cognizant AI Lab for building teams of AI "
               "agents that work together. It can design such a team from one sentence. "
               "What it cannot do is tell you whether the team it designed is any good, "
               "or whether a different team would do the same job better and cheaper.")
        self.p("This project adds that missing half. It <b>gives every team the same "
               "exam</b>, scores the result, and <b>searches for a better team</b> by "
               "copying good ones with small changes. A cheap predictor guesses which "
               "changes are worth testing, so only the promising ones are paid for.")
        self.callout(
            "The whole project in one picture",
            "Invent many agent teams &rarr; give each the same exam &rarr; score each on "
            "correct answers, cost and size &rarr; keep the best, change them a little, "
            "repeat. A predictor skips testing the changes that look bad.",
            bg=SOFT, bar=ACCENT)

    def _words(self) -> None:
        self.h2("Words you will meet")
        self.table(["Word", "What it means here"], [
            ["Agent", "One AI model with a job description and, sometimes, a tool."],
            ["Agent network", "A team of agents. One is the router (the boss); the "
                              "others are specialists it can ask."],
            ["Router", "The agent a question arrives at. It decides which specialist "
                       "to ask next."],
            ["Tool", "Something an agent can use. Here: a search over the company's "
                     "documents."],
            ["Token", "A small piece of text. Models are paid for by the token, so "
                      "tokens are the cost."],
            ["Multi-hop", "A question that needs several documents chained together."],
            ["Accuracy", "Share of exam questions answered correctly."],
            ["Fitness", "One score from accuracy, tokens and team size. Higher is better."],
            ["Genome", "The full description of one team, as neuro-san reads it."],
            ["Mutation", "A small change to a team: add an agent, rewire, change a model."],
            ["Predictor", "A small machine-learning model that guesses a team's score "
                          "from its design, without running the exam."],
            ["Held-out", "Questions kept aside and never used to choose a winner, so "
                         "the winner can be judged fairly."],
        ], widths=[88, 380])

    def _world(self) -> None:
        self.story.append(PageBreak())
        self.h1("The exam", "An invented company, so no model can answer from memory.")
        world = build_world()
        self.p(f"The exam is about <b>Meridian Logistics</b>, a company that does not "
               f"exist: {len(world.depots)} depots (warehouses), {len(world.contracts)} "
               f"customer contracts and {len(world.incidents)} late-delivery incidents, "
               f"written as {len(documents(world))} short documents. Because it is "
               f"invented, a model can only answer by searching the documents. Here are "
               f"three of them, exactly as an agent sees them:")
        docs = documents(world)
        task = TASKS[1]
        ref = task.question.split("contract ")[1].split()[0]
        contract = world.contract(ref)
        depot = world.depot(contract.depot_code)
        self.terminal(docs[f"contract-{contract.ref}.txt"].rstrip() + "\n\n"
                      + docs[f"depot-{depot.code}.txt"].rstrip() + "\n\n"
                      + docs[f"incident-{world.incidents[0].ref}.txt"].rstrip())
        self.h2("One question, answered by hand")
        self.p(f"<b>Question {task.task_id}:</b> {task.question}")
        self.p(f"<b>Hop 1.</b> Find contract {contract.ref}. It says it is serviced by "
               f"depot <b>{depot.code}</b>.<br/>"
               f"<b>Hop 2.</b> Find depot {depot.code}. It says its location is "
               f"<b>{depot.city}</b>.<br/>"
               f"<b>Answer:</b> {task.answer}. Two documents, so this is a two-hop "
               f"question.", STEP)
        self.h2("The question sets")
        self.table(["Set", "Questions", "Used for"], [
            ["The benchmark", str(len(TASKS)), "Every committed measurement: 1 to 4 hops."],
            ["The bank", str(len(BANK)), "Held-out checks: 1 to 9 documents, entities "
                                          "the benchmark never names."],
            ["meridian-select", str(len(SELECT)), "What the scale-up search chooses on; "
                                                   "40% whole-company totals."],
            ["meridian-judge", str(len(JUDGE)), "Only for judging winners; shares no "
                                                 "entity with the select set."],
        ], widths=[100, 60, 308])
        self.p("Every answer is computed from the same seed that writes the documents, "
               "and a test re-derives each one from the document text, so the exam "
               "itself cannot be wrong.")

    def _answering(self) -> None:
        self.story.append(PageBreak())
        self.h1("How a team answers", "What happens between the question and the reply.")
        self.step(1, "The question reaches the router",
                  "The router reads its instructions and the list of specialists it may "
                  "ask, each with a one-line description.")
        self.step(2, "The router asks a specialist",
                  "For the question above it asks the contract specialist about the "
                  "contract, gets back the depot code, then asks the depot specialist "
                  "about that depot.")
        self.step(3, "Specialists search the documents",
                  "Each specialist calls the search tool, which returns the three "
                  "documents that best match its query, and reports what they say.")
        self.step(4, "The router replies",
                  "In an exam run it replies with the bare answer so it can be marked; "
                  "on the web page it explains which documents it used.")
        self.p("Every one of those steps is a paid model call. A two-hop question is "
               "typically about ten calls; that is why tokens are the cost.")

    def _scoring(self) -> None:
        self.h2("How a team is scored")
        self.p("After the exam each team gets one number:")
        self.terminal("fitness = accuracy - 0.06 x min(tokens / 600000, 1)"
                      " - 0.02 x (agents / 9)")
        records = {r["origin"]: r for r in _history()["records"]}
        designer = records["seed:designer_shaped"]
        best = max(_history()["records"], key=lambda r: r["fitness"])
        rows = []
        for label, r in (("The designer's team", designer), ("The best evolved team", best)):
            value = float(scalarise(r["accuracy"], r["tokens"], r["agents"]))
            rows.append([label, f"{r['accuracy']:.4f}", f"{r['tokens']:,}", str(r["agents"]),
                         f"{value:.4f}"])
        self.table(["Team", "Accuracy", "Tokens", "Agents", "Fitness"], rows,
                   widths=[150, 70, 90, 60, 70])
        self.p(f"Worked through for the designer's team: {designer['accuracy']:.4f} "
               f"&minus; 0.06 &times; {designer['tokens'] / 600000:.3f} &minus; 0.02 "
               f"&times; {designer['agents']}/9 = <b>{designer['fitness']:.4f}</b>. "
               "Correct answers matter most; among equally correct teams, the cheaper "
               "and smaller one wins.")

    def _search(self) -> None:
        self.story.append(PageBreak())
        self.h1("How the search finds a better team", "Four phases, repeated.")
        self.step(1, "Phase A: test the starting teams",
                  "Three hand-made teams, one shaped the way neuro-san's designer "
                  "builds them, sit the real exam. This costs model calls.")
        self.step(2, "Phase B: train the predictor",
                  "A small model learns, from the teams already tested, how a team's "
                  "design relates to its accuracy and its token cost.")
        self.step(3, "Phase C: breed many children, for free",
                  "The best teams are copied with one small change each (seven kinds of "
                  "change). Invalid teams are thrown away. The predictor guesses every "
                  "child's score in a fraction of a millisecond.")
        self.step(4, "Phase D: test only the promising few",
                  "The children the predictor likes best sit the real exam, and their "
                  "results go back into Phase B.")
        self.h2("Example: the change that won")
        self.p("Both winning teams came from the same change, <b>reassign_model</b>: the "
               "router was moved to a stronger model while every specialist stayed on "
               "the cheap one. A per-agent model is a setting neuro-san already has and "
               "nothing tunes; the search found it.")
        self.h2("What the free half prints")
        self.p("<font face='Courier' size='9'>make offline</font> runs phases B and C "
               "on the committed measurements with no key and no cost. Captured output:")
        self.terminal(_proof("offline_search", 24))

    def _running(self) -> None:
        self.story.append(PageBreak())
        self.h1("Running it yourself", "Every command, in the order you would use it.")
        self.step(1, "Install (no key needed)",
                  "Python 3.12 or newer.")
        self.terminal("python3.12 -m venv .venv && source .venv/bin/activate\n"
                      'pip install -e ".[dev]"')
        self.step(2, "Check everything works",
                  "Lint, the docs, neuro-san's own validator, every test, and a real "
                  "offline search. It should end with the tests passing.")
        self.terminal("make validate")
        self.step(3, "Explore the committed results, still free",
                  "These read the measurements already in the repository.")
        self.terminal("make offline       # breed and rank candidates, zero model calls\n"
                      "make holdout       # choose on half the questions, judge on the other\n"
                      "make ablation      # does the predictor pick better than chance?\n"
                      "make bank-report   # the held-out bank comparison\n"
                      "make experiment    # the scale-up plan and its price; spends nothing")
        self.step(4, "Add a key for real runs",
                  "Copy the example settings file and put one provider's key in it. "
                  "The file is never committed.")
        self.terminal("cp .env.example .env     # then edit .env: GOOGLE_API_KEY=...\n"
                      "make check-key           # asks the provider if the key works\n"
                      "make smoke               # four questions to the best team")
        self.step(5, "Talk to the teams in a browser", "")
        self.terminal("python apps/web/serve.py   # http://localhost:7860\n"
                      "make studio                # neuro-san's own UI, http://localhost:4173")
        self.p("The web page has two tabs: <b>Ask</b> sends a question to the best team "
               "and marks benchmark questions right or wrong in front of you; "
               "<b>Measure</b> runs up to four teams on the same questions and shows "
               "which is best on correctness and cost.")
        self.step(6, "Measure your own team", "Any neuro-san network, any questions.")
        self.terminal("make measure NETWORK=registries/my_network.hocon "
                      "TASKS=my_questions.jsonl\n\n"
                      '# my_questions.jsonl, one question per line:\n'
                      '{"question": "Which city is depot D08 in?", "answer": "Eastgate"}')

    def _results(self) -> None:
        self.story.append(PageBreak())
        self.h1("What it has found so far", "In plain words, with what it does not show.")
        bank = {p.stem: _json(p) for p in (ROOT / "results" / "heldout_bank").glob("*.json")}
        calibration = ROOT / "results" / "calibration" / "designer_select20.json"
        rows = [["12 teams on the 17-question benchmark",
                 "The best evolved team answered 16 of 17, the designer's team 14 of 17."]]
        if {"designer", "evolved"} <= set(bank):
            d = sum(r["correct"] for r in bank["designer"]["results"])
            e = sum(r["correct"] for r in bank["evolved"]["results"])
            rows.append([f"{bank['designer']['questions_asked']} new questions (the bank)",
                         f"Designer {d}, evolved {e}: the advantage did not show on "
                         f"new questions. Too few to prove either way."])
        if calibration.exists():
            c = _json(calibration)
            right = sum(r["correct"] for r in c["results"])
            rows.append([f"{c['questions_asked']} harder select questions",
                         f"Designer {right} of {c['questions_asked']}: hard enough to "
                         f"tell teams apart."])
        rows.append(["The predictor, offline",
                     "Picks the best of three unseen teams 62% of the time, against 33% "
                     "by chance."])
        self.table(["Measurement", "Result"], rows, widths=[150, 318])
        self.callout(
            "What is still open",
            "Whether the predictor makes the search better for the same money. The "
            "experiment that answers it (make experiment GO=1) is built and tested, and "
            "needs a paid key: about 56,000 model calls.",
            bg=WARN_BG, bar=ACCENT)
        self.h2("Where things are")
        self.table(["Folder", "What is inside"], [
            ["esp/eval/", "The invented company, the question sets, the exam runner"],
            ["esp/genome/", "Team descriptions and the seven kinds of change"],
            ["esp/surrogate/", "The predictor"],
            ["esp/evolve/", "The search loop and the scale-up experiment"],
            ["apps/web/", "The browser page"],
            ["tests/", "The checks, and the 12 committed measurements"],
            ["docs/", "This guide, the design notes, the findings, the PDFs"],
        ], widths=[100, 368])
        self.story.append(Spacer(1, 6))

    def build(self, out: str | Path | None = None) -> Path:
        self._cover()
        self._words()
        self._world()
        self._answering()
        self._scoring()
        self._search()
        self._running()
        self._results()
        return self.render(Path(out) if out else OUT, "neuro-san-esp: a beginner's guide")


if __name__ == "__main__":
    print("wrote", Guide().build())
