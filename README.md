# neuro-san-esp

**A fitness function for [neuro-san](https://github.com/cognizant-ai-lab/neuro-san)
agent networks, and an evolutionary search that uses it.**

[![ci](https://github.com/Sivakumarraj/neuro-san-esp/actions/workflows/ci.yml/badge.svg)](https://github.com/Sivakumarraj/neuro-san-esp/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

---

## The gap this fills

neuro-san turns a sentence into a working multi-agent network. `agent_network_designer`
generates one, validates it, and serves it in about five seconds. It generates **one**, and
never measures it.

Nothing in neuro-san scores one agent network against another. `neuro_san/test/evaluators/`
holds `assertIn`-style assertions for integration-testing a single network's answer, which
is regression testing rather than a fitness function. So there is no way to answer whether a
nine-agent topology beats a five-agent one for the same job, which model each agent should
run, or whether the generated instructions are any good. That is design without evaluation.

This project supplies the missing half. It **measures** a network — accuracy, token cost and
size over a fixed task set — and then **searches** for a better one using **ESP**
(Evolutionary Surrogate-assisted Prescription), Cognizant AI Lab's own method for
optimisation where every real evaluation is expensive.

## Results

**Twelve networks measured on real model calls, 17 tasks each, 204 task runs. The search
found a better topology than the one neuro-san's designer produces, twice, and both wins
came from the same knob.**

| | Accuracy | Tokens | Agents | Fitness |
|---|---|---|---|---|
| `seed:designer_shaped` — the shape the designer produces | 0.8235 | 385,280 | 4 | 0.7761 |
| `seed:flat_pair` | 0.8235 | 316,074 | 3 | 0.7852 |
| `seed:solo` — one agent, one tool | 0.8235 | 377,716 | 1 | 0.7835 |
| `mut:reassign_model` — cheapest win | 0.8824 | 260,052 | 5 | 0.8453 |
| **`mut:reassign_model` — best measured** | **0.9412** | 359,600 | 5 | **0.8941** |

Against the designer's own shape, the best network answers **two more questions of
seventeen for 7% fewer tokens**. A second network on the front reaches a smaller accuracy
gain for **32% fewer tokens**. Both are on the Pareto front, because the trade-off is the
honest result and one number hides it.

**Both wins are the same finding: put the better model on the router, keep the cheap one on
the workers.** Each was produced by `reassign_model` and each reassigned the front man — the
agent that decides which specialist to ask — while leaving every specialist on the cheap
model. That is a per-agent setting neuro-san already exposes, that nothing in the framework
tunes, and that no one would find by reading the topology. The best network is also the
first in the population to **finish all seventeen tasks**: its one miss is a wrong answer,
not a timeout.

Every number above is recomputed from the committed evaluation cache by a test, so the
README cannot drift away from the run. Every candidate's genome is stored beside its score
in `tests/fixtures/cache/`, so any of them can be rebuilt, served and talked to — `make
studio` puts all twelve in neuro-san's own UI at once.

**Read the table as a measurement, not as a ranking.** Splitting the 17 tasks in two and
selecting on one half, the winner of that half never tops the other half and averages rank
7.2 of 12: seventeen tasks are too few to order individual networks, and the gap between
+0.8941 and +0.8453 is inside the noise. What does survive the split is the population
claim — the best evolved network beat the best seed on held-out tasks in **100% of 200
splits**, and evolved networks outrank the hand-written ones by two and a half places.
`make holdout` reproduces both halves of that from committed data, no key needed, and
[docs/FINDINGS.md](docs/FINDINGS.md#held-out-tasks-what-this-task-set-can-and-cannot-support)
works through it.

**The Predictor has started to earn its place, and has not finished.** At the generation the
first search used it — nine measurements — cross-validated rank correlation was **−0.333**,
worse than chance, and `results/history.json` records that. Refitted over all twelve it is
positive on every split tried, **+0.28 to +0.76**, median near +0.67, and the twelfth
network was the first one it actually chose: a wake trained on eleven real samples ranked a
pool and paid for the top of it. That is the loop working as designed, once. It is not yet
evidence that the surrogate beats picking at random, which needs a run that does both.

Full numbers, the failure analysis and the prior art are in
[docs/FINDINGS.md](docs/FINDINGS.md).

## Limitations

- **The surrogate has never been compared against random selection.** It chose the
  twelfth network, and at the generation the first search used it it cross-validated at
  −0.333, worse than chance. Both facts are published. Which of the two the loop deserves
  credit for needs a run that searches with the Predictor and without it on the same
  budget, and that has not been bought.
- **Twelve real evaluations, two generations of search.** No repeat run, no second random
  seed, no held-out task set. A candidate costs about 165 provider requests against a free
  tier of 500 per day per model — three candidates a day, so eleven is about four days of
  budget. The 118 candidates the surrogate scored in between cost nothing, which is the part
  of ESP that does work as advertised.
- **Seventeen of the 204 task runs never finished** — a timeout or a blown recursion cap
  rather than a wrong answer. They concentrate on two full-corpus aggregation questions.
  `accuracy` counts them as wrong; `answered_accuracy()` excludes them. Both are reported.
- **No baseline other than the seeds.** Random search and evolution-without-a-surrogate
  would each need their own budget, so the claim is that this beat three hand-written
  topologies, not that it beat the alternative search strategies.
- **Seventeen tasks cannot rank individual networks.** Selecting on half of them and
  judging on the other half, per-network accuracy carries across the split at +0.022 — no
  relationship. The population-level result holds out of sample; the per-network ordering
  does not. More tasks is the only fix, and a better estimator is not one.
- **One task domain, and the held-out test stays inside it.** Held-out questions come from
  the same generated world, so what was measured is stability across questions rather than
  transfer to a new domain. A topology that wins at multi-hop retrieval need not win
  elsewhere.
- **The surrogate idea is not novel.** AgentSquare (ICLR 2025) uses a performance predictor
  for the same purpose. What is absent from that work is neuro-san, and what is absent from
  neuro-san is any fitness function at all.

## How it works

Four phases, repeated. Phase C is the point: it is free, so the search can be wide.

```
Phase A   measure the seed topologies for real        ->  (genome, fitness) pairs
Phase B   train a Predictor on those pairs            ->  cheap fitness estimate
Phase C   breed and rank thousands of candidates      ->  zero LLM calls
Phase D   pay for real evaluation of the elite only   ->  feed back into B
```

### The genome is neuro-san's own format

A candidate **is** a neuro-san `agent_network_definition` — the same HOCON the framework
serves — plus a per-agent `model` override. Nothing is invented: a genome renders straight to
a registry file and runs as an ordinary agent network. Defined in
`esp/genome/definition.py`; the three starting topologies are in `esp/genome/seeds.py`.

The configured model is **part of the genome hash**, deliberately. A fitness measured on one
model must not be mistaken for the same network on another.

### Fitness is measured, not asserted

Each candidate answers all 17 tasks through a real neuro-san session. Three objectives are
recorded, then scalarised for selection while the Pareto front is kept separately:

```
fitness = accuracy − 0.06 · min(tokens / 600000, 1) − 0.02 · (agents / 9)
```

The evaluation world (`esp/eval/world.py`) is generated from a fixed seed: 24 depots, 40
contracts, 60 incidents, 124 documents, and 17 questions of one to four hops whose answers
are correct by construction. Retrieval is a deterministic coded tool, so fitness measures the
topology rather than a retrieval layer that drifts between generations. Timeouts and blown
recursion caps are classified as **unfinished rather than wrong** (`esp/eval/runner.py`),
because neuro-san returns them as ordinary answer strings and scoring them as wrong answers
teaches the search that a good topology is bad.

### Seven mutation operators, behind a validity gate

`add_agent`, `remove_agent`, `rewire`, `split_agent`, `merge_agents`, `toggle_search`,
`reassign_model` (`esp/genome/mutations.py`). An invalid mutant — an unreachable agent, a
cycle, no front man — is **discarded, never repaired**, so every candidate that reaches a
real evaluation is a network neuro-san would actually serve.

### The Predictor

A `GradientBoostingRegressor` over **thirteen structural features** of the genome: agent
count, depth, edges, branching, leaves, searchers, model tiers and instruction lengths
(`esp/surrogate/predictor.py`). Never a measured quantity — a feature derived from a
measurement would mean the Predictor needed a real evaluation in order to predict one.

Below **eight** samples it refuses to fit, `predict` returns the training mean, and
`ranks()` reports `False` so that callers say the generation was a random search instead of
printing a ranking over one repeated constant. Quality is reported as cross-validated
Spearman rank correlation, because the Predictor's job is ordering, not pricing.

There is no learned Prescriptor here: the prescription step is mutation plus elite selection.
That is a real departure from canonical ESP, and
[docs/FINDINGS.md](docs/FINDINGS.md#what-the-predictor-is-exactly) says why.

### It runs as a service, not a batch job

The first version was a script that planned forty evaluations and died at the daily cap every
time. The cap is not an obstacle to a service — it is its rhythm. An hourly
`invocation: "event"` agent (`registries/manifest.hocon`) spends what today allows, writes the
population down **after every candidate**, and stops. Budget-aware failover across models
keeps the measured daily caps as data (`esp/eval/failover.py`), and a preflight refuses to
start on a configuration that would produce wrong numbers.

## Quick start

Python 3.12+. Works the same in Codespaces, a devcontainer, or a laptop.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # adds the test tools and the accelerator UI

make check       # ruff + the full test suite
make offline     # phases B and C: 2,000 candidates ranked, zero LLM calls
```

Nothing above needs an account, a key, or a network. `make offline` trains the Predictor on
the eleven measurements committed in `tests/fixtures/cache/` and evolves against them,
announcing which cache it used.

### With an API key

A free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) gives 500
requests per day per model.

```bash
cp .env.example .env      # paste the key in; .env is gitignored
python apps/optimizer/run_optimizer.py --check   # preflight
make probe                                       # which models answer today
```

The preflight reports where the key came from and what the budget buys:

```
[ok  ] provider key: set, GOOGLE_API_KEY, from .env
[ok  ] model ladder: gemini-3.5-flash-lite, gemini-3.1-flash-lite
       -- 1000 requests/day = about 6 candidate(s)
```

**Run the preflight first.** A misconfigured evaluator does not crash. It scores every
candidate zero, and the cache keeps that answer forever, so the search is taught that good
topologies are bad.

Then either re-measure the seeds yourself, or adopt the measurements already paid for and
spend your budget on new candidates instead:

```bash
make baseline                              # measure the seed topologies (~1.5 days of budget)
python scripts/adopt_measurements.py       # or: start from the committed eleven
python apps/optimizer/run_optimizer.py     # one wake: train, rank, pay for the elite
```

### Talk to the agents in a browser

```bash
python apps/web/serve.py  # then open http://localhost:7860
```

One process, no separate backend, no second repository. The page runs questions through the
measured champion on neuro-san's direct session — the same code path the evaluator measures
with, so what you talk to is exactly what was scored. Questions from the graded task set are
**marked against the known answer in front of you**:

```json
{"answer": "R. Delacroix", "expected": "R. Delacroix", "correct": true, "seconds": 48.4}
```

Expect **30–60 seconds** for a multi-hop question: four documents have to be found and
chained, and anything faster would mean it did not really look.

### Open every measured network in the accelerator UI

```bash
pip install -e ".[studio]"   # nsflow, the accelerator UI — included in .[dev]
make studio                  # then http://localhost:4173
```

`localhost` means the machine running the command. On a laptop that is the
browser you already have; in Codespaces or on a remote box, forward port **4173**
(Codespaces does it automatically and gives you a `*.app.github.dev` URL — the
neuro-san server on 8080 stays internal, so 4173 is the only port to open).

This is the comparison, not a demo. Every measured topology is rendered as its
own servable agent and listed by the fitness it earned, so the same question can
be put to the shape neuro-san's designer produces, to the hand-written
alternatives, and to the network that beat them — and the answers, the routing
and the agent count differ in front of you. Reading that off a table is not the
same as watching two topologies answer.

```
studio_evolved_reassign_model
  "Rank 1 of 11 by measured fitness (+0.8453): evolved by the reassign_model
   operator. Scored 88.24% on 17 multi-hop questions using 260,052 tokens
   across 5 agent(s). Genome 6859dda0dfabcf2d."
```

Each agent is the genome that earned its score, rebuilt from the measurement
rather than described, with its numbers in the description the UI shows. The
optimiser is served and stays **private**: it spends the day's whole evaluation
budget when poked.

### Serve the champion as an ordinary agent

```bash
python scripts/serve_champion.py   # writes registries/champion.hocon
export AGENT_MANIFEST_FILE=$PWD/registries/champion_manifest.hocon
export AGENT_TOOL_PATH=$PWD PYTHONPATH=$PWD
python -m neuro_san.service.main_loop.server_main_loop
```

### Run the optimiser as a service

```bash
export AGENT_MANIFEST_FILE=$PWD/registries/manifest.hocon
export AGENT_TOOL_PATH=$PWD PYTHONPATH=$PWD
python -m neuro_san.service.main_loop.server_main_loop
```

The server logs `Found 1 periodic agent interactions` and from then on fires the optimiser on
the cron in `registries/manifest.hocon` with `user_id: system`, no client attached. See
[SERVING.md](SERVING.md) for state, leases, budget and the security model. In a container:
`docker compose up -d optimizer`.

## Repository layout

| Path | What lives there |
|---|---|
| `esp/genome/` | The genome: neuro-san network definitions, the seven mutation operators, three seed topologies |
| `esp/eval/` | The measured world, the 17 scored tasks, the runner, budget-aware model failover |
| `esp/surrogate/` | The Predictor and its honest quality reporting |
| `esp/evolve/` | The batch ESP loop — phases A through D in one sitting |
| `esp/service/` | The same loop as an interruptible service: persistent population, budget, lease |
| `esp/report/` | The generated PDFs, and figures from the run history |
| `apps/web/` | The single-process browser front end |
| `apps/optimizer/` | One wake, runnable by hand or from any scheduler |
| `registries/` | neuro-san manifests: the optimiser agent, and the generated champion |
| `scripts/` | Offline search, model probe, champion and studio serving, report and proof generation |
| `tests/` | The suite, plus the eleven committed measurements in `tests/fixtures/cache/` |
| `results/` | `results/history.json` and the figures the reports read |

## Testing and verification

```bash
make check      # ruff + the full suite, exactly what CI runs
make verify     # start a real neuro-san server and prove it fires the optimiser
make offline    # the free half of ESP, end to end, no key
make holdout    # select on half the tasks, judge on the other half, no key
```

The suite covers the genome and its validity gate, the scored tasks, outcome
classification, the surrogate's refusal to train below eight samples, budget arithmetic and
quota-payload parsing, the lease and its UTC clock, the container and devcontainer
definitions, and the documented numbers themselves — several tests fail if this README
disagrees with `results/history.json`.

Evidence captured from real runs, rather than described, is in `docs/proofs/` and rendered
into the dossier: the full test run, the offline search, a live neuro-san server firing the
optimiser on a shortened cron, and a browser reaching the champion through the real agents.

## Documentation

| | |
|---|---|
| [docs/FINDINGS.md](docs/FINDINGS.md) | Measurements, failure analysis, what the Predictor is, prior art |
| [SERVING.md](SERVING.md) | Deployment, state, budget, and what the agent may do |
| [SECURITY.md](SECURITY.md) | Reporting a vulnerability |
| [Dossier](docs/neuro-san-esp-Dossier.pdf) | The technical report, with captured evidence |
| [Primer](docs/neuro-san-esp-Primer.pdf) | The same result without the jargon |
| [Explainer](docs/neuro-san-esp-Explainer.pdf) | For a reader who knows nothing about agents, models or tokens |
| [Verification](docs/neuro-san-esp-Verification.pdf) | Every check in the dossier, re-run |

Built on [neuro-san](https://github.com/cognizant-ai-lab/neuro-san) by Cognizant AI Lab.
Licensed under Apache 2.0.
