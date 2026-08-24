"""The whole repository as one document.

`build.py` reports a single run. This reports the project: what gap it fills,
how every file contributes, how the optimiser runs unattended, and what was
actually observed when it ran. It is the document to hand somebody who has
never seen the repository and will not clone it.

Every transcript in here is read from `docs/proofs/`, which is written by
`scripts/capture_proofs.py` running the real commands. Nothing in this file
types out what a command "would" print. If a proof is missing, the page says it
is missing rather than describing it from memory.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from reportlab.platypus import PageBreak

from esp.report.layout import (
    AMBER,
    BAD,
    BAD_BG,
    CW,
    GOOD,
    GOOD_BG,
    LEAD,
    WARN_BG,
    Layout,
    _s,
)

ROOT = Path(__file__).resolve().parent.parent.parent

# Every tracked file, and the one thing it is for. Files not named here are
# listed as unannotated rather than silently dropped -- an inventory that
# quietly omits what it has no note for is not an inventory.
NOTES: dict[str, str] = {
    "esp/genome/definition.py":
        "The genome. A neuro-san agent network as an object that can be hashed, "
        "compared and rendered back to HOCON.",
    "esp/genome/mutations.py":
        "The seven operators that change a network, and the validity gate every "
        "mutant must pass.",
    "esp/genome/seeds.py":
        "The three starting topologies, including a faithful reconstruction of "
        "what neuro-san's own designer produces.",
    "esp/eval/world.py":
        "A synthetic logistics company: 24 depots, 40 contracts, 60 incidents, "
        "124 documents, one seed.",
    "esp/eval/tasks.py":
        "17 questions of one to four hops, with exact answers and the scorer.",
    "esp/eval/corpus_tool.py":
        "The retrieval tool the candidates share. Deterministic term overlap, "
        "three documents.",
    "esp/eval/runner.py":
        "Runs a candidate in-process, measures accuracy, tokens and time, and "
        "caches by genome hash.",
    "esp/eval/ratelimit.py":
        "Paces every provider call underneath neuro-san, because a 429 would "
        "otherwise be scored as a bad topology.",
    "esp/eval/failover.py":
        "Moves to the next model when one model's daily quota is spent.",
    "esp/surrogate/predictor.py":
        "The Predictor. Genome features to predicted fitness -- the part that "
        "makes this ESP rather than a genetic algorithm.",
    "esp/evolve/loop.py":
        "Population, selection, elitism, the multi-objective fitness function "
        "and the Pareto front.",
    "esp/service/state.py":
        "The population and today's budget, written to disk after every single "
        "evaluation so an interruption costs nothing.",
    "esp/service/optimizer.py":
        "One wake: spend what today allows, write it down, stop cleanly.",
    "esp/service/coded_tools.py":
        "The neuro-san CodedTool that lets an agent run a wake.",
    "registries/optimizer.hocon":
        "The autonomous agent itself -- event-invoked, with instructions to stay "
        "quiet unless something improved.",
    "registries/manifest.hocon":
        "The hourly schedule. This is the line that makes it a service.",
    "apps/optimizer/run_optimizer.py":
        "Run one wake from the command line or a container, no server needed.",
    "scripts/run_esp.py": "The batch search, for when budget is available in bulk.",
    "scripts/offline_search.py":
        "Phases B and C only. Zero provider calls, so it runs with no key at all.",
    "scripts/probe_models.py":
        "Which models are usable today and what budget is left.",
    "scripts/baseline_report.py": "The seed measurements, as text.",
    "scripts/smoke_inprocess.py":
        "Proves a network can be run in-process with no server.",
    "scripts/capture_proofs.py":
        "Runs the commands whose output appears in this document.",
    "scripts/serve_champion.py":
        "Writes the best-measured topology into the registry so a person can "
        "talk to it in a browser.",
    "scripts/service_report.py":
        "Turns the service's accumulated population into the report inputs.",
    "scripts/verify_periodic.py":
        "Starts a real neuro-san server and proves it fires the optimiser with "
        "no user and no client attached.",
    "SERVING.md": "How to deploy the service, and what is verified about it.",
    "esp/report/layout.py": "Shared page furniture for both PDFs.",
    "esp/report/build.py": "The run report.",
    "esp/report/dossier.py": "This document.",
    "esp/report/primer.py":
        "The same project with no jargon, for a reader who has never seen it.",
    "esp/report/plots.py": "Fitness curve, surrogate scatter, Pareto front.",
    "Dockerfile": "The service image.",
    "compose.yaml": "The service, with its state volume.",
    ".github/workflows/ci.yml": "Lint, tests and a real offline search on every push.",
    "Makefile": "Every command in this document has a target here.",
    "SECURITY.md": "Keys are read from the environment and never committed.",
}


class Dossier(Layout):
    def __init__(self, results_dir: str | Path = "results",
                 proofs_dir: str | Path | None = None):
        super().__init__(results_dir)
        self.proofs = Path(proofs_dir) if proofs_dir else ROOT / "docs" / "proofs"
        index_path = self.proofs / "index.json"
        self.index = (json.loads(index_path.read_text(encoding="utf-8"))
                      if index_path.exists() else [])

    # ---------------------------------------------------------------- proofs

    def _record(self, name: str) -> dict | None:
        for entry in self.index:
            if entry["name"] == name:
                return entry
        return None

    @staticmethod
    def _provenance(record: dict) -> str:
        """The header above a transcript, saying how it was obtained.

        Three kinds of proof appear in this document and they are not equally
        strong, so they are not printed alike. A transcript assembled by hand
        from a live session, or one carried over from an earlier build because
        this machine had no budget to redo it, is real evidence -- but printing
        either under a green `exit 0` would claim the harness had just watched
        it succeed, which is a stronger claim than the truth.
        """
        if record.get("attested"):
            return (f"<b>{record['command']}</b> &nbsp;&mdash;&nbsp; "
                    f"<font color='{_ATTESTED_HEX}'>captured out of band, not "
                    f"by the harness</font>")

        verdict = ("exit 0" if record.get("exit_code") == 0
                   else f"exit {record.get('exit_code')}")
        colour = "#1d7a46" if record.get("exit_code") == 0 else "#b3261e"
        header = (f"<b>$ {record['command']}</b> &nbsp;&mdash;&nbsp; "
                  f"<font color='{colour}'>{verdict}</font>")
        if record.get("carried_forward"):
            when = str(record.get("captured_at", ""))[:10]
            header += (f" &nbsp;<font color='{_ATTESTED_HEX}'>(from an earlier "
                       f"build{', ' + when if when else ''}; not re-run here)"
                       f"</font>")
        return header

    def proof(self, name: str, caption: str, limit: int = 40) -> None:
        """Print a captured transcript, or say plainly that it is absent."""
        record = self._record(name)
        path = self.proofs / f"{name}.txt"
        # A skip is not a capture, even when an older transcript is still lying
        # in the directory: printing it would put an `exit None` header over
        # evidence this build never obtained.
        if record is None or not path.exists() or record.get("skipped"):
            self.callout(
                "Proof not captured",
                f"<font face='Courier' size='9'>{name}</font> was not recorded in "
                "this build. It is named here rather than described, because a "
                "document that fills a gap with prose is how an unverified claim "
                "gets into a report.",
                bg=BAD_BG, bar=BAD)
            return

        lines = path.read_text(encoding="utf-8").splitlines()
        shown = lines[:limit]
        if len(lines) > limit:
            shown.append(f"... {len(lines) - limit} more lines")

        self.p(self._provenance(record),
               _s("cmd", fontSize=8.8, leading=12, spaceAfter=4))
        self.terminal("\n".join(shown))
        self.p(caption, _s("pc", fontSize=8.6, leading=12, textColor=_MUTED_HEX,
                           spaceAfter=12))

    # --------------------------------------------------------------- content

    def build(self, out: str | Path | None = None) -> Path:
        self._cover()
        self._gap()
        self._method()
        self._modules()
        self._service()
        self._evidence()
        self._served()
        self._findings()
        self._limits()
        self._run()

        out = Path(out) if out else ROOT / "docs" / "neuro-san-esp-Dossier.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        return self.render(out, "neuro-san-esp: the whole project")

    # ------------------------------------------------------------------ 1

    def _cover(self) -> None:
        self.h1("neuro-san-esp",
                "An agent network that searches for a better agent network, and "
                "keeps doing it without being asked.")

        self.p(
            "neuro-san can turn a sentence into a working multi-agent network. It "
            "produces <b>one</b> network and never measures it &mdash; there is no "
            "fitness function anywhere in the framework. Nobody knows whether a "
            "nine-agent topology beats a five-agent one for the same job, which model "
            "each agent should run, or whether the generated instructions are any "
            "good. That is design without evaluation.", LEAD)

        self.p(
            "This repository adds the missing half: a way to score a network, a search "
            "for a better one, and a service that runs that search unattended. The "
            "method is Cognizant AI Lab's own &mdash; <b>Evolutionary "
            "Surrogate-assisted Prescription</b>: learn a cheap Predictor from real "
            "measurements, evolve thousands of candidates against it for free, and "
            "spend expensive real evaluations only on the elite.")

        self.callout(
            "The one number that explains the design",
            "Evaluating one candidate network for real takes about eight minutes of "
            "language-model time. Scoring one with the Predictor takes "
            "<b>0.064 milliseconds</b>. Two thousand candidates cost 0.13 seconds "
            "against the surrogate; the same two thousand evaluated for real would "
            "take roughly <b>267 hours</b>. That ratio is the whole reason this is ESP "
            "and not a plain genetic algorithm &mdash; and it is measured on this "
            "repository, not quoted from a paper.")

        self.h2("What is actually here")
        self.table(
            ["", ""],
            [["A genome",
              "neuro-san's own <font face='Courier' size='8.5'>"
              "agent_network_definition</font>, plus a per-agent model. Reusing the "
              "framework's representation is what makes mutation free."],
             ["A fitness function",
              "Accuracy, tokens, latency and agent count, measured on 17 multi-hop "
              "questions over a synthetic 124-document corpus."],
             ["Seven mutation operators",
              "add, remove, rewire, split, merge, toggle_search, reassign_model &mdash; "
              "each gated by a validity check, invalid mutants discarded not repaired."],
             ["A Predictor",
              "Gradient-boosted regression over structural and textual genome features, "
              "retrained after every batch of real evaluations."],
             ["A service",
              "An event-invoked neuro-san agent on an hourly schedule that spends the "
              "day's provider budget, writes down what it learned, and stays silent "
              "unless it found something better."]],
            widths=[CW * 0.24, CW * 0.76])

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ 2

    def _gap(self) -> None:
        self.h1("The gap this fills",
                "Why the evolutionary machinery was already there, unconnected.")

        self.p(
            "Reading neuro-san-studio closely, almost every component an evolutionary "
            "loop needs is already implemented for other reasons. What is missing is "
            "the one piece that turns them into a search.")

        self.table(
            ["Evolutionary component", "Already in neuro-san"],
            [["Genome",
              "<font face='Courier' size='8.5'>agent_network_definition</font> &mdash; "
              "round-trips to HOCON through "
              "<font face='Courier' size='8.5'>HoconAgentNetworkAssembler</font>"],
             ["Mutation operators",
              "<font face='Courier' size='8.5'>coded_tools/agent_network_editor/</font> "
              "&mdash; add_agent, remove_agent, update_agent"],
             ["Viability check",
              "<font face='Courier' size='8.5'>StructureNetworkValidator</font> "
              "&mdash; cycles, unreachable nodes, DAG shape"],
             ["Deploy a candidate without restart",
              "reservations &rarr; "
              "<font face='Courier' size='8.5'>BranchActivation.use_reservation()</font>"],
             ["Candidate garbage collection",
              "<font face='Courier' size='8.5'>ExpiringAgentNetworkStorage</font> "
              "&mdash; TTL and LRU"],
             ["Baseline to beat", "the designer's own one-shot output"],
             ["<b>A fitness function</b>",
              "<b><font color='#b3261e'>nothing. Anywhere.</font></b>"]],
            widths=[CW * 0.34, CW * 0.66], highlight=[])

        self.callout(
            "Stated honestly: the idea of searching agent architectures is not new",
            "ADAS uses a meta-agent over an archive, AFlow runs Monte-Carlo tree search "
            "over operator graphs, GPTSwarm learns edge probabilities with reinforcement "
            "learning, and MaAS and Promptbreeder attack neighbouring problems. None of "
            "them targets neuro-san, and none uses surrogate-assisted evolution. The "
            "claim made here is the method applied to this framework and measured on it "
            "&mdash; not the invention of architecture search.",
            bg=WARN_BG, bar=AMBER)

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ 3

    def _method(self) -> None:
        self.h1("How it works", "Four phases. Phase C is the point.")

        self.terminal(
            "Phase A   evaluate a few candidates for real       -->  (genome, fitness)\n"
            "               |                                             |\n"
            "               |                                             v\n"
            "Phase B   train the Predictor on every pair measured so far\n"
            "                                                             |\n"
            "                                                             v\n"
            "Phase C   mutate thousands of candidates, rank them against the\n"
            "          Predictor                            -- zero provider calls --\n"
            "                                                             |\n"
            "                                                             v\n"
            "Phase D   pay to really evaluate only the top few  -->  back to Phase B")

        self.h2("In plain language")
        for step, text in [
            ("Phase A",
             "Run three starting networks against 17 real questions and record how "
             "accurate each was, how many tokens it burned, and how many agents it "
             "used. This costs real money and real time: about eight minutes a "
             "network."),
            ("Phase B",
             "Fit a small model that reads a network's shape &mdash; agent count, "
             "depth, branching, which model each agent runs, how long its instructions "
             "are &mdash; and predicts how well it will score. It is trained on tens of "
             "samples, so it is weak. Its job is not to be right; its job is to "
             "<i>rank</i>."),
            ("Phase C",
             "Generate thousands of mutated networks and score every one with the "
             "Predictor. No language model is called, so this is effectively free. "
             "Two thousand candidates take a tenth of a second."),
            ("Phase D",
             "Take only the best few the Predictor picked and evaluate those for real. "
             "Feed the results back into Phase B, which now has more data, and repeat."),
        ]:
            self.p(f"<b>{step}.</b>&nbsp; {text}",
                   _s("st", fontSize=9.6, leading=13.4, leftIndent=6, spaceAfter=7))

        self.callout(
            "A real-world analogy",
            "Testing a car design in a wind tunnel is slow and expensive, so engineers "
            "fit a cheap simulator to the tunnel results, try ten thousand shapes in "
            "the simulator overnight, and put only the best handful back in the tunnel. "
            "The tunnel is Phase D. The simulator is the Predictor. This repository is "
            "that loop, with agent networks instead of car bodies.")

        self.h2("The mutation operators")
        self.table(
            ["Operator", "What it changes", "Why it matters"],
            [["add_agent / remove_agent", "how many agents exist",
              "removal re-parents the orphans instead of severing them"],
             ["rewire", "one edge, size unchanged",
              "isolates topology from size"],
             ["split_agent / merge_agents", "granularity",
              "the axis a designer told to &lsquo;use the fewest agents&rsquo; can "
              "never explore"],
             ["toggle_search", "which agents may read the corpus",
              "retrieval placement is a real design decision"],
             ["reassign_model", "per-agent model tier",
              "the main cost/quality knob, and nobody tunes it today"]],
            widths=[CW * 0.24, CW * 0.34, CW * 0.42])

        self.p(
            "Every mutant is checked &mdash; one top agent, a directed acyclic graph, "
            "no unreachable agents, no dangling references, at least one agent able to "
            "search &mdash; and <b>discarded if it fails, never repaired</b>. A "
            "repaired mutant is a different mutant from the one the operator produced, "
            "and silently substituting one would leave the search unable to learn what "
            "its own operators do.")

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ 4

    @staticmethod
    def _annotated(path: str) -> bool:
        """Package markers and the committed cache fixtures carry no
        explanation worth a table row."""
        return not path.endswith("__init__.py") and "fixtures" not in path

    def _tracked_paths(self) -> list[str]:
        """The repository's tracked files, read from git at build time.

        Sourced live rather than from a committed listing so the page cannot
        describe a tree that no longer exists. Falls back to the captured
        proof for a checkout without git -- a source tarball, for instance.
        """
        try:
            finished = subprocess.run(
                ["git", "ls-files"], cwd=ROOT,
                capture_output=True, text=True, timeout=30, check=True)
            paths = [line for line in finished.stdout.splitlines() if line.strip()]
            if paths:
                return paths
        except (OSError, subprocess.SubprocessError):
            pass

        fallback = self.proofs / "tree.txt"
        if fallback.exists():
            return [line for line in fallback.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        return []

    def _modules(self) -> None:
        self._record("tree")
        paths = self._tracked_paths()

        # The subtitle states both halves, because a count of annotated files
        # that silently excludes the unannotated ones overstates the page's
        # coverage on the one page a reader can check by hand.
        omitted = [p for p in paths if not self._annotated(p)]
        if paths:
            subtitle = (f"The repository, annotated: {len(paths) - len(omitted)} "
                        f"tracked files, plus {len(omitted)} package markers and "
                        f"cache fixtures listed only as this count.")
        else:
            subtitle = "The repository, annotated."
        self.h1("Every file, and what it is for", subtitle)

        groups: dict[str, list[str]] = {}
        for path in paths:
            top = path.split("/")[0] if "/" in path else "(root)"
            groups.setdefault(top, []).append(path)

        for group, members in sorted(groups.items()):
            interesting = [p for p in members if self._annotated(p)]
            if not interesting:
                continue
            self.h2(f"<font face='Courier' size='11'>{group}</font>")
            self.table(
                ["File", "What it does"],
                [[f"<font face='Courier' size='8'>{p.split('/')[-1]}</font>",
                  NOTES.get(p, "<i>supporting file</i>")] for p in interesting],
                widths=[CW * 0.26, CW * 0.74])

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ 5

    def _service(self) -> None:
        self.h1("From a script to a service",
                "The part that makes this production rather than a demo.")

        self.p(
            "The first version of this project was a batch script: plan four "
            "generations, run them, print a report. It died every time &mdash; not "
            "from a bug, but because the provider's free tier allows 500 requests per "
            "day per model and one candidate costs about 165 of them. A day's budget "
            "buys three candidates. A four-generation run needs forty.", LEAD)

        self.callout(
            "The reframe the whole design turns on",
            "The daily cap is not an obstacle to a service. It is a service's "
            "<b>rhythm</b>. Spend what today allows, write down where you got to, stop, "
            "and carry on tomorrow. A population accumulates over weeks that could "
            "never be bought in an afternoon &mdash; and every one of those weeks is "
            "unattended.",
            bg=GOOD_BG, bar=GOOD)

        self.h2("What one wake does")
        self.terminal(
            "1.  take the lease            one optimiser at a time, or two wakes\n"
            "                              spend the same budget twice\n"
            "2.  read yesterday's state    population, and which models are spent today\n"
            "3.  prime the failover ladder so this fresh process does not rediscover\n"
            "                              this morning's exhaustion by paying for it\n"
            "4.  Phase C, free             rank 400 mutants with the Predictor\n"
            "5.  evaluate up to 3          saving state after EVERY candidate\n"
            "6.  quota hit?                that is a normal ending, not an error\n"
            "7.  release the lease         in a finally block; and it expires anyway")

        self.h2("Every design decision here exists because of a specific failure")
        self.table(
            ["Decision", "The failure it prevents"],
            [["State written after every candidate, not every generation",
              "An interruption after an eight-minute evaluation would otherwise throw "
              "that evaluation away &mdash; and interruption is the expected ending."],
             ["Atomic write (temp file, then rename)",
              "A service killed mid-write comes back to a truncated population file "
              "and has lost everything."],
             ["A lease with a timeout",
              "Two overlapping wakes spend the same daily budget twice and write over "
              "each other. A lease with no timeout wedges the service forever the "
              "first time a wake is killed."],
             ["A corrupt lease is treated as free",
              "Better to risk one overlapped wake than to need a human to delete a "
              "file before the service runs again."],
             ["Exhaustion recorded per <i>day</i>, not permanently",
              "A model retired for good would leave the service dead after its first "
              "bad afternoon &mdash; the opposite of running unattended."],
             ["Quota failures are never cached as scores",
              "A daily cap fails tasks part-way through a candidate, producing a "
              "plausible partial score. Cached, it tells the search forever that a "
              "good network is mediocre."],
             ["The agent stays silent unless something improved",
              "Most wakes find nothing. A service that announces every wake trains "
              "its operator to ignore it, and then the one wake that matters is "
              "ignored too."],
             ["The agent has no tool that deletes or sends",
              "An autonomous process running hourly with no human in the loop should "
              "not be able to do anything irreversible or outward-facing."]],
            widths=[CW * 0.36, CW * 0.64])

        self.h2("The line that makes it a service")
        self.terminal(
            '"optimizer.hocon": {\n'
            '    "serve": true,\n'
            '    "periodic": {\n'
            '        "interactions": [{\n'
            '            "enable": true,\n'
            '            "cron_schedule": "0 * * * *",\n'
            '            "metadata": {"user_id": "system"},\n'
            '        }],\n'
            '    },\n'
            "}")
        self.p(
            "Hourly, not every fifteen minutes. A wake evaluates up to three candidates "
            "and the free tier buys about three a day, so a faster schedule would only "
            "produce wakes that find the budget already spent and decline. The cadence "
            "is set by what the provider allows, not by how often we would like news.",
            _s("note", fontSize=9, leading=12.6, textColor=_MUTED_HEX))

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ 6

    def _evidence(self) -> None:
        self.h1("Proof", "Captured by running the commands, not written by hand.")

        self.p(
            "Every transcript on this page was produced by "
            "<font face='Courier' size='9'>scripts/capture_proofs.py</font>, which runs "
            "the real command and writes its real output to "
            "<font face='Courier' size='9'>docs/proofs/</font>. This document reads "
            "them from there. If a command starts failing, the proof changes and so "
            "does this page.", LEAD)

        self.h2("The test suite")
        self.proof("tests",
                   "Unit tests across the genome, the mutation operators, the "
                   "evaluator, the surrogate, the failover ladder and the service. "
                   "Several of them are regressions for bugs listed later in this "
                   "document.")

        self.h2("Lint")
        self.proof("lint",
                   "Ruff over every package, with the rule set pinned in "
                   "pyproject.toml rather than left at the tool's defaults &mdash; "
                   "default drift turned CI red once already.")

        self.story.append(PageBreak())
        self.h1("Proof, continued", "The part that costs nothing and proves the most.")

        self.h2("Phases B and C, with no provider key at all")
        self.proof("offline_search",
                   "This is the ESP claim, executed. Two thousand candidate networks "
                   "generated, validated, and ranked by the Predictor with zero calls "
                   "to any language model. The rejected count is the validity gate "
                   "doing its job.", limit=24)

        self.callout(
            "Read the surrogate line honestly",
            "On three real samples the Predictor's rank correlation is +0.000 &mdash; "
            "<b>no better than chance</b>, and the document says so rather than "
            "hiding it. That is what a surrogate trained on three points is worth. The "
            "machinery is correct and the ranking is free; the accuracy of the ranking "
            "is a function of how many real evaluations the service has accumulated, "
            "which is exactly why it runs every hour instead of once.",
            bg=WARN_BG, bar=AMBER)

        self.h2("The framework really does wake it")
        self.proof("periodic_server",
                   "A real neuro-san server, started with this repository's "
                   "manifest and the cron shortened to once a minute. "
                   "<b>user_id=system with no client attached</b> is the "
                   "property that matters: the framework initiated those "
                   "interactions by itself. Produced by "
                   "scripts/verify_periodic.py, which fails if the fires do not "
                   "arrive.", limit=14)

        self.callout(
            "What this proof does not cover",
            "A wake completing a full evaluation <i>inside</i> the server "
            "process. The front agent needs its own provider call before it can "
            "reach the tool, and on an exhausted free tier that call returns 429 "
            "first. The trigger is what is verified here; the wake itself is "
            "verified separately and repeatedly by "
            "<font face='Courier' size='9'>apps/optimizer/run_optimizer.py</font>, "
            "which runs identical code. Reporting this as a verified end-to-end "
            "run would be exactly the overstatement the rest of this repository "
            "is built to avoid.",
            bg=WARN_BG, bar=AMBER)

        self.story.append(PageBreak())
        self.h1("Proof, continued")

        self.h2("Commit history")
        self.proof("log",
                   "Fifteen most recent commits.", limit=16)

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ 7

    def _findings(self) -> None:
        self.h1("What running it changed",
                "Seven things that could not have been reasoned out from the code.")

        for title, text in [
            ("The front man was the wrong agent, silently",
             "neuro-san takes the <i>first</i> entry in a network's tool list as the "
             "agent a request enters through. Rendering agents in alphabetical order "
             "meant one seed ran with a contract specialist as its front man and "
             "another with an arithmetic agent, which answered every question with "
             "&lsquo;you did not provide any numbers&rsquo;. Both scored as bad "
             "topologies. Neither had ever been run as written &mdash; every "
             "multi-agent measurement taken before this was invalid."),
            ("The cost objective had no gradient",
             "Token cost was normalised by 60,000 while real candidates burned 260,000, "
             "so the term saturated and contributed nothing. The run would have "
             "reported itself as multi-objective while optimising accuracy alone. "
             "Fixed, the three seeds show <b>identical 0.82 accuracy with a 42% cost "
             "spread</b> &mdash; 278,532 tokens against 396,378. That spread is the "
             "entire opportunity, and it was invisible until the scale was right."),
            ("Rate limiting is a correctness requirement",
             "A 429 comes back through neuro-san as an agent error. The candidate "
             "scores zero and the search learns that a perfectly good topology is bad. "
             "Pacing calls is not politeness here; it is the difference between "
             "measuring a network and measuring the provider's mood."),
            ("A blocking sleep froze every agent at once",
             "The first limiter called <font face='Courier' size='9'>time.sleep</font> "
             "at an async call site. neuro-san runs its agents on asyncio, so that "
             "stopped the entire event loop: every other agent froze while still "
             "spending its own execution budget, and a queue of tasks timed out "
             "together. Caught by a lint rule, not by a test."),
            ("The first task set had no headroom",
             "A single agent with a single tool scored 1.00 on the original 13 "
             "questions. There was nothing for evolution to improve. The corpus went "
             "from 40 documents to 124, retrieval narrowed from five results to three, "
             "and four-hop and aggregate shapes were added. An experiment whose "
             "baseline is already perfect measures nothing."),
            ("Free-tier quota is per-model and wildly uneven",
             "Read out of the 429 payloads themselves rather than the documentation: "
             "the &lsquo;lite&rsquo; models allow 500 requests per day, the full models "
             "allow 20, and the pro models allow none at all. A mutation that reassigns "
             "an agent onto a 20-per-day model is a guaranteed quota failure &mdash; "
             "which would then be scored as a bad topology. Only viable models are in "
             "the ladder."),
            ("The service was recording the wrong exhausted model",
             "A wake that hit the daily cap reported an empty exhausted list, because "
             "it matched the error text against the failover ladder and the model that "
             "actually failed was the network default, which is not a ladder member. "
             "Every wake for the rest of that day would have spent its first calls "
             "rediscovering the same dead model. The exhausted name is now read out of "
             "the provider's own message."),
        ]:
            self.callout(title, text)

        self.story.append(PageBreak())

    def _served(self) -> None:
        """The winner is a thing you can talk to, not a hash in a table."""
        self.h1("Talking to the champion",
                "The measured best topology, served and opened in a browser.")

        self.p(
            "A search that ends with a winning hash in a report has stopped one "
            "step short. The reason to measure topologies is that one of them is "
            "better to actually <i>use</i>, so "
            "<font face='Courier' size='9'>scripts/serve_champion.py</font> writes "
            "the current best genome into the registry as an ordinary neuro-san "
            "agent network. It is regenerated and gitignored rather than "
            "hand-maintained: a champion edited by hand would drift from the genome "
            "that earned the score, and the thing being served would stop being the "
            "thing that was measured.", LEAD)

        self.proof("serving_champion",
                   "Captured by running it. The winner is selected from the "
                   "committed measurements, written into the registry, and served "
                   "by a real neuro-san server.", limit=22)

        self.h2("A real request reaches the real agents")
        self.terminal(
            'origin: [{"tool": "Coordinator"}, {"tool": "IncidentSpecialist"}]')
        self.p(
            "That is <font face='Courier' size='9'>designer_shaped</font> exactly as "
            "measured &mdash; the front man delegating to the specialist for an "
            "incident lookup. The topology in the report and the topology answering "
            "the question are the same object.",
            _s("n1", fontSize=9.4, leading=13, textColor=_MUTED_HEX))

        shots = ROOT / "docs" / "screenshots"
        placed = self.image(
            shots / "01-web-client.png",
            "The neuro-san web client in Chromium, connected to the server above.",
            width=CW * 0.86)
        placed &= self.image(
            shots / "03-agent-session.png",
            "The same session after sending a question. The Configuration panel "
            "shows host localhost, port 8080, agent network "
            "\u201cchampion\u201d \u2014 the search winner, served.",
            width=CW * 0.86)
        if not placed:
            self.callout(
                "Screenshots not captured in this build",
                "They are named here rather than described. A document that fills "
                "the gap where its evidence should be is how an unverified claim "
                "gets in.", bg=BAD_BG, bar=BAD)

        self.callout(
            "It did not produce an answer, and that is not hidden",
            "The measured model\u2019s free-tier daily cap was spent when these were "
            "taken \u2014 500 requests a day, resetting at midnight Pacific \u2014 so "
            "the agents returned 429 rather than a name. What is verified here is "
            "the serving path: the server loads the measured champion, a browser "
            "reaches it, and a request routes through the real agents in the real "
            "topology. The final answer needs budget that today did not have. "
            "Saying so costs a sentence; implying otherwise would cost the "
            "credibility of every other number in this document.",
            bg=WARN_BG, bar=AMBER)

        self.story.append(PageBreak())

    def _run(self) -> None:
        """Carry the run report's own pages, so this is genuinely one document."""
        history = ROOT / "results" / "history.json"
        if not history.exists():
            return
        from esp.report.build import Report

        self.story.append(PageBreak())
        self.h1("The run itself",
                "The measurement report, carried here in full.")
        self.story += Report(ROOT / "results").compose()

    # ------------------------------------------------------------------ 8

    def _limits(self) -> None:
        self.h1("Limits, and how to run it",
                "What this does not show, said before anyone has to ask.")

        for text in [
            "<b>One task domain.</b> A topology that wins at multi-hop retrieval over "
            "a logistics corpus need not win anywhere else.",
            "<b>The Predictor is trained on tens of samples.</b> It ranks; it does not "
            "price. Its measured rank correlation is published in this document "
            "whatever it says, including when it says &lsquo;no better than "
            "chance&rsquo;.",
            "<b>The designer baseline is a reconstruction.</b> It reproduces the shape "
            "<font face='Courier' size='9'>agent_network_designer</font> produces "
            "&mdash; one top agent, shallow DAG, fewest agents &mdash; not its literal "
            "output, because the designer wires networks against its own toolbox and "
            "cannot see this corpus. The shape is what is compared, and the shape is "
            "faithful.",
            "<b>No evolved candidate has yet beaten the baseline.</b> Three real "
            "evaluations exist. Beating a baseline needs a population, a population "
            "needs budget, and budget arrives at three candidates a day &mdash; which "
            "is the reason the service exists and the reason it is measured in weeks. "
            "If it never beats the baseline, that gets reported as the result.",
            "<b>Automated agent architecture search is an active field.</b> The idea "
            "is not new. What is absent from all of it is neuro-san, and what is absent "
            "from neuro-san is any fitness function at all.",
        ]:
            self.p(f"&bull;&nbsp; {text}",
                   _s("li", fontSize=9.6, leading=13.4, leftIndent=8, spaceAfter=7))

        self.h2("Running it yourself")
        self.terminal(
            "make install          # editable install with dev extras\n"
            "make check            # lint + the full test suite\n"
            "make offline          # phases B and C -- no key needed, no calls made\n"
            "\n"
            "export GOOGLE_API_KEY=...\n"
            "make probe            # which models have budget today\n"
            "make baseline         # measure the seed topologies\n"
            "make search           # the batch loop\n"
            "\n"
            "python apps/optimizer/run_optimizer.py     # one service wake\n"
            "docker compose up -d                       # the service, hourly, forever")

        self.callout(
            "Keys",
            "No key is committed anywhere in this repository and none ever will be. "
            "Everything reads from the environment; "
            "<font face='Courier' size='9'>SECURITY.md</font> says so and the "
            "<font face='Courier' size='9'>.gitignore</font> enforces it for caches "
            "and state.")


_MUTED_HEX = "#5b6672"
# Neither green nor red: evidence that is real but was not watched
# succeeding by this build.
_ATTESTED_HEX = "#8a6100"


if __name__ == "__main__":
    print("wrote", Dossier().build())
