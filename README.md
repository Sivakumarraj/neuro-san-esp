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
size over a fixed task set — and then **searches** for a better one, borrowing the
surrogate-assisted structure of **ESP** (Evolutionary Surrogate-assisted Prescription),
Cognizant AI Lab's own method for optimisation where every real evaluation is expensive.

**Said plainly up front, because the name invites the question:** in the
[ESP paper](https://arxiv.org/abs/2002.05368) the Prescriptor is itself a neural network,
evolved to maximise the surrogate's predictions. **There is no learned Prescriptor here** —
prescription is seven mutation operators plus elite selection. What is borrowed is the
Predictor and the sample-efficiency argument. In shape this is nearer to **LEAF**
(*Evolutionary Neural AutoML for Deep Learning*, GECCO 2019), which evolves architectures
and size, with agents where LEAF had layers.

## Measure your own network

The measurement is not tied to the networks this repository evolved. **Any network neuro-san
can load can be scored on any questions you can check the answers to**, from three places:

```bash
# 1. A terminal. NETWORK is a HOCON path or a name in your manifest.
make measure NETWORK=registries/my_network.hocon TASKS=my_questions.jsonl

# 2. A browser: the "Measure networks" tab, side by side, with a Pareto front.
python apps/web/serve.py                     # http://localhost:7860

# 3. Inside neuro-san: the evaluator agent, in the studio chat panel.
make studio                                  # then ask it "which networks can you measure?"
```

A question file is JSON Lines, one question per line; `answers` may list several accepted
forms, and `id` and `hops` are optional:

```json
{"question": "Which city is depot D08 in?", "answer": "Pickering"}
{"id": "Q2", "question": "Which contract has the highest penalty?", "answers": ["C-2139", "C2139"]}
```

Each network gets accuracy; accuracy over the questions it finished, because a timeout or a
blown recursion cap is not a wrong answer; the unfinished count; tokens; cost where the
provider reports it; and time. Every answer is kept whole, and `--json` writes the full
report. The file is validated before anything is paid for. A run that measured the
environment rather than the network is refused, not reported: no model called, every
question erroring, or a provider quota. With no question file, the built-in
seventeen-question benchmark below is used.

## Results

**Twelve networks measured on real model calls, 17 tasks each, 204 task runs. The search
found a better topology than the one neuro-san's designer produces, twice, and both wins
came from the same knob.**

| | Accuracy | Tokens | Agents | Fitness |
| --- | --- | --- | --- | --- |
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
- **One of the two predicted objectives does not beat its own null, and is now excluded.**
  Token cost cross-validates at −0.53 against a permutation null of about −0.17, so it sits
  **roughly 0.35 below the no-signal baseline**, and the Predictor orders candidates by cost
  backwards. Phase C used to weight it at full strength anyway. It no longer does: each
  generation measures every objective against its own permutation null, and an objective
  that loses to that null is held at the population mean, so it stays on the fitness scale
  but cannot order anything. Steering a search with a predictor that ranks backwards is
  worse than not predicting that objective at all.

  **The exclusion is stable even though the number under it is noisy**, and the two have to
  be reported separately. Across 20 cross-validation seeds the token margin is negative in
  **20 of 20** and the objective is excluded in **20 of 20**, at both 12 and 40 shuffles per
  null. The null itself is far less settled: its median moves from −0.17 to −0.16 between
  those shuffle counts, and individual seeds range from −0.42 to +0.03. So the margin is a
  range — medians **−0.35 and −0.37**, single seeds from −0.12 to −0.70 — and any
  single-seed figure is one draw from that spread rather than the result.

  The gate is a measurement, not a hardcoded exclusion — the objective returns on its own
  the generation it starts predicting — and it only fires where the null was actually
  measured, because against an assumed baseline of zero a twelve-sample procedure would
  drop objectives for being small-sample rather than for being wrong. Accuracy clears its
  null by +0.73 and is untouched, in 20 of 20 seeds. **What this does not do is fix the
  prediction.** Thirteen structural features still do not predict what a network will spend,
  nothing here says what would, and twelve samples cannot say how much of the −0.35 is real.
- **One feature was mislabelled; it is fixed, and no conclusion moved.** The model-tier
  feature placed a model by its position on the configured ladder, so the workers' cheapest
  model read as the most expensive. It is now a property of the model — rung, then release.
  The token margin moved from −0.47 to −0.35, accuracy's from +0.63 to +0.73, and every
  exclusion verdict stayed the same. Every Predictor figure here is after the fix, and `make
  figures` regenerates them all from committed data; the before-and-after table is in
  [docs/FINDINGS.md](docs/FINDINGS.md#a-defect-in-the-model-tier-feature-measured-and-fixed).
- **There is no context, so this is not ESP's loop.** ESP prescribes *actions for a context*
  and the Prescriptor is the model that maps one to the other. This project has **no context
  variable at all** — every candidate is scored against the same fixed task set. That is the
  reason there is no learned Prescriptor rather than an oversight: with nothing to map from,
  mutation operators are the only thing that can fill that slot. What this implements is
  surrogate-assisted architecture search, which is nearer to LEAF than to ESP.
- **Twelve real evaluations, two generations of search.** No repeat run, no second random
  seed, no held-out task set. A candidate costs about 165 model calls and a quarter to half a
  million tokens, which is why there are twelve and not twelve hundred. The 118 candidates the
  surrogate scored in between cost nothing, which is the part of ESP that does work as
  advertised.
- **Every measurement was taken on one provider.** All twelve ran on Gemini. The code runs
  unchanged on Anthropic and OpenAI, and the champion is served there with its promotion
  intact, but nothing here says how the ranking holds on another provider's models.
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

```text
Phase A   measure the seed topologies for real        ->  (genome, outcomes) pairs
Phase B   train one Predictor per outcome objective   ->  cheap outcome estimates
Phase C   breed candidates, rank by derived fitness   ->  zero LLM calls
Phase D   pay for real evaluation of the elite only   ->  feed back into B
```

Fitness is **derived** from the Predictor's outputs, never learned by it — see
[Predictor, fitness, prescription](#predictor-fitness-prescription--which-is-which) below for
why that distinction matters and what it exposed.

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

```text
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

**The Pareto front breeds.** Parents are chosen by non-dominated sorting on
(accuracy up, tokens down, agents down), topped up with the best scalarised fitness when the
front is smaller than the elite. Selection used to take the top few by fitness alone while
the front was computed, plotted in three documents and never consulted — multi-objective in
the report and single-objective in the search.

### Predictor, fitness, prescription — which is which

Three words get used loosely about a loop like this, and conflating them is how a search
comes to optimise something nobody chose. In this repository they are three separate things:

| | What it is | Where it lives |
| --- | --- | --- |
| **Predictor** | The surrogate. **One `GradientBoostingRegressor` per outcome objective**, learned from real evaluations. It predicts *accuracy* and *token cost*. It never sees a fitness and never sees the weights below. | `esp/surrogate/outcomes.py` |
| **Fitness** | A fixed weighting applied to outcomes — not learned, not fitted, just arithmetic. Over **measured** outcomes it scores Phases A and D. Over **predicted** outcomes it ranks Phase C. Same function both times. | `esp/evolve/loop.py::scalarise` |
| **Prescription** | What proposes the next candidate. Seven mutation operators plus Pareto-front selection, run against the Predictor. **Not** a learned model — see the departure noted at the top. | `esp/genome/mutations.py` |

**There is no context here, and that is the deeper departure.** In ESP a Prescriptor maps a
*context* to the actions to take in it. Every candidate in this project is evaluated against
the same fixed task set, so there is no context to map from — which is why mutation operators
occupy the prescription slot rather than a learned model. Building a real Prescriptor starts
with deciding what the context is; that decision is the research, not the network.

Agent count is a fourth objective and has **no model at all**: it is an exact property of a
genome, so it is counted rather than estimated. A regressor asked to guess a number already
in hand only adds error.

The Predictor's input is **thirteen structural features** of the genome — agent count, depth,
edges, branching, leaves, searchers, model tiers, instruction lengths — and never a measured
quantity. A feature derived from a measurement would mean the Predictor needed a real
evaluation in order to predict one.

Below **eight** samples it refuses to fit, `predict` returns the training mean, and
`ranks()` reports `False` so that callers say the generation was a random search instead of
printing a ranking over one repeated constant. Quality is reported as cross-validated
Spearman rank correlation, because the Predictor's job is ordering, not pricing.

**This split was made after review feedback and it immediately found a defect.** The first
version trained a single model directly on the scalarised fitness, which put the weighting
inside the surrogate. Reported per objective instead, on the twelve measured networks:

| Objective | Spearman | Permutation null | Margin over null | Excluded from Phase C |
| --- | --- | --- | --- | --- |
| accuracy | **+0.63** [+0.27 … +0.72] | −0.07 [−0.36 … +0.30] | **+0.73**, positive in 20/20 seeds | 0/20 seeds |
| token cost | **−0.53** [−0.80 … −0.38] | −0.17 [−0.42 … +0.03] | **−0.35**, negative in 20/20 seeds | **20/20 seeds** |

Medians over 20 cross-validation seeds, with the per-seed range in brackets; each null is
the median of 12 shuffles. Repeating the sweep at 40 shuffles moves the null medians to
−0.13 and −0.16 and leaves both margin signs and both exclusion counts unchanged.

**Read the last column, not the third.** The null is the noisiest quantity here — a single
seed can put token cost's anywhere from −0.42 to +0.03 — so no point estimate of it is worth
quoting. What is stable is the decision it feeds: token cost loses to its own null under
every seed and every shuffle count tried, and accuracy under none. `make figures`
reproduces the whole table from committed data, no key needed.

**The null column is the point, and the first version of this table did not have it.** A rank
correlation only means something against the baseline the same procedure produces when there
is no signal to find, and on twelve samples that baseline is not automatically zero.
Cross-validation can manufacture negative correlation on its own: hold out a high value, the
training mean drops, the model predicts low. `permutation_null()` measures it by shuffling
the targets and re-running the identical cross-validation.

Two things came out of measuring it. **Accuracy's margin survives it comfortably** — at worst
+0.40 across the seeds tried — though its null is not the settled −0.03 an earlier draft
claimed: it is −0.07 at 12 shuffles and −0.13 at 40, and single seeds reach +0.30, for the tie
reason below. **Token cost's null is about −0.17, so the real effect is −0.35, not −0.53.**
The finding holds: the margin is negative in every seed tried, and the Predictor does order
candidates by cost backwards. It is a third smaller than the raw correlation, and twelve
samples cannot cleanly separate the real part from the artifact.

**Two caveats the tests record.** The null is only trustworthy on an untied objective — a
permutation destroys a relationship only if permuting moves the values, and accuracy takes
just four distinct values across twelve networks, so its null is weak. Token cost is distinct
in all twelve, so its null is sound, and token cost is what the finding is about. And the
spread is wide: the token null itself ranges from −0.42 to +0.03 across seeds. The margin is
the right statistic; it is not a precise one. Full account in
[docs/FINDINGS.md](docs/FINDINGS.md#what-the-predictor-is-exactly).

### It runs as a service, not a batch job

The first version was a script that planned forty evaluations and died when the provider
stopped it. A budget is not an obstacle to a service — it is its rhythm. An hourly
`invocation: "event"` agent (`registries/manifest.hocon`) spends what the budget allows,
writes the population down **after every candidate**, and stops, so an interruption costs at
most the candidate in flight. A preflight refuses to start on a configuration that would
produce wrong numbers.

## Quick start

Python 3.12+. Works the same in Codespaces, a devcontainer, or a laptop.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # adds the test tools and the accelerator UI

make check       # ruff + the full test suite
make offline     # phases B and C: breed and rank candidates, zero LLM calls
```

Nothing above needs an account, a key, or a network. `make offline` trains the Predictor on
the twelve measurements committed in `tests/fixtures/cache/` and evolves against them,
announcing which cache it used.

### With an API key

Runs on **Anthropic, OpenAI or Google Gemini**, chosen the way neuro-san-studio chooses: by
provider, not by model. Put a key in `.env` and that provider is used. With several keys set,
studio's order decides (OpenAI, then Anthropic, then Gemini), and `ESP_PROVIDER` forces one.
neuro-san picks the client class from the model name, so nothing else changes.

Each provider has a two-rung ladder, a model for the workers and a stronger one the search can
promote a router to, because that promotion is the one change the measurements found worth
making. Wherever neuro-san has a version-free alias the ladder uses it, as studio's own config
does with `claude-sonnet`, so a new release is picked up without an edit:

| Provider | Key | Workers | Router, when promoted |
| --- | --- | --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-haiku` (newest Haiku) | `claude-sonnet` (newest Sonnet) |
| OpenAI | `OPENAI_API_KEY` | `gpt-5.4-mini` | `gpt-5.5` |
| Google Gemini | `GOOGLE_API_KEY` | `gemini-3.1-flash-lite` | `gemini-3.5-flash` |

**Any model neuro-san resolves works.** `ESP_DEFAULT_MODEL=claude-opus` runs the workers on
Opus, and the model takes the rung it belongs on, so a strong choice becomes the top of the
ladder rather than sitting under Sonnet. `ESP_MODEL_TIERS` sets both rungs outright. OpenAI has
no version-free alias in neuro-san's registry, so its rungs are the newest named there. Gemini's
are the ones the committed measurements were taken on.

```bash
cp .env.example .env      # paste your key in; .env is gitignored
make check-key            # asks the provider whether the key works
python apps/optimizer/run_optimizer.py --check   # the full preflight
```

```text
[ok  ] provider key: set, ANTHROPIC_API_KEY, from .env
[ok  ] model provider: claude-haiku needs ANTHROPIC_API_KEY
[ok  ] model tiers: claude-haiku, claude-sonnet
[ok  ] population provider: all anthropic
[ok  ] pacing: 14 requests/minute per model (ESP_RPM); paid API, no daily cap
```

**Run the preflight first.** A misconfigured evaluator does not crash. It scores every
candidate zero, and the cache keeps that answer forever, so the search is taught that good
topologies are bad. The preflight asks the provider whether the key works, rather than whether
the variable is set, and refuses to start on the mistakes that otherwise surface from inside an
agent:

```text
[FAIL] provider key: ANTHROPIC_API_KEY still the placeholder from .env.example
[FAIL] model tiers: gemini-3.5-flash needs GOOGLE_API_KEY -- set ESP_MODEL_TIERS to models
       of the provider you hold a key for
[FAIL] population provider: state holds measurements taken on gemini, and this run is
       configured for anthropic. Measurements do not cross providers
```

**What a run costs.** One candidate is seventeen questions and about 165 model calls. The twelve
committed measurements used between 242,670 and 473,450 tokens each; those were taken on
Gemini, and another provider's tokenizer and verbosity will land in the same range rather than
on the same number. `ESP_RPM` sets the pace. The default of 14 requests a minute is safe on any
account and slow on a paid one, at about twelve minutes of queueing per candidate. Set it to one
under the per-minute limit your provider's console shows.

**The model is part of the genome hash, so measurements do not cross providers.** The twelve
committed results are all on `gemini-3.1-flash-lite`. On Claude every hash changes, the cache
misses correctly, and the population starts again from the seeds. A fitness measured on one
model does not describe the same network on another. The web page and the studio still serve
the measured champion on your provider: the same topology, with the same agent promoted to the
stronger model. Each surface says that the score on your provider has not been measured.

**The key is read once, at launch.** Editing `.env` under a running server changes nothing
until it restarts, and `.env.example` is the committed template. It is not read for a key, so
put the key in `.env` and leave the example alone.

You need a measured population before anything can be searched. There are two
ways to get one. **They are alternatives, not steps — run one, not both.**

**Option A — measure the seed topologies on your own provider.** Three networks, about 500
model calls.

```bash
make baseline
```

**Option B — adopt the twelve measurements this repository already paid for.** Only on
Gemini, where they were taken: `adopt_measurements.py` refuses to mix them into a run on
another provider. Seconds, and no key needed.

```bash
python scripts/adopt_measurements.py
```

Then, with a population in place, run one wake — train the Predictor, rank a
free pool, and pay only for the elite:

```bash
python apps/optimizer/run_optimizer.py
```

Check the whole path end to end on your key, from preflight to four answered questions:

```bash
make smoke
```

> **If a run ever reports `acc=0.00 tok=0 (cached)`**, an earlier run wrote
> zeros before the key worked. `tok=0` means no model was called at all. Newer
> builds refuse to cache that, but zeros already on disk keep replaying:
> `rm -rf .esp-cache` and run the preflight again.

#### On Google's free tier

Gemini's free tier caps requests per model per day, and some models cannot fund a single
candidate. For that case only, the runner fails over between models as each cap is reached,
with measured caps and per-model pacing kept as data in `esp/eval/failover.py`. `make probe`
re-measures them against your key. None of this machinery engages on Anthropic or OpenAI,
which have per-minute limits and no daily cap.

### Talk to the agents, and measure them, in a browser

```bash
python apps/web/serve.py  # then open http://localhost:7860
```

The page has two tabs. **Ask a network** is described below. **Measure networks** puts the
same questions to up to four networks at once: the twelve committed ones, plus any HOCON you
place in `ESP_NETWORKS`. It uses the built-in benchmark or a JSON Lines file you paste, shows
progress, and marks the Pareto front. It also gives a question-by-question grid and a JSON
download. Measuring is paid for, so a deployment caps the total runs (`ESP_WEB_MAX_MEASURE`)
and runs one measurement at a time. A network is never uploaded: it names Python classes to
import.

One process, no separate backend, no second repository. The page runs questions through the
measured champion on neuro-san's direct session, the same code path the evaluator measures
with. It changes one line, and says so: the benchmark tells the front man to reply with the
bare value so it can be scored, and the page asks it instead to **explain its answer**. It
gives the answer first, then every identifier it followed, what each document said, and any
calculation written out. The topology, tools and models are the ones that earned the score.

Four questions are offered as one click each, one per kind of difficulty. They are a direct
lookup, a two-document hop, the deepest four-document chain in the set, and one that hops and
then calculates. **All seventeen benchmark questions** are listed beneath them, easiest
first, and any other question about Meridian Logistics works too. Benchmark questions are
**marked against the known answer in front of you**, and full answers are never truncated:

```json
{"answer": "The total penalty owed for incident INC-4401 is 4500. ...",
 "expected": "4500", "correct": true, "provider": "anthropic",
 "router_model": "claude-sonnet", "worker_model": "claude-haiku", "seconds": 48.4}
```

Expect **30–90 seconds** for a multi-hop question: up to four documents have to be found and
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

```text
studio_evolved_reassign_model
  "Rank 1 of 12 by measured fitness (+0.8941): evolved by the reassign_model
   operator. Scored 94.12% on 17 multi-hop questions using 359,600 tokens
   across 5 agent(s), measured on gemini-3.1-flash-lite. Genome 3bf9c008d880c3fc."
```

Each agent is the genome that earned its score, rebuilt from the measurement
rather than described, with its numbers in the description the UI shows, and
the same four one-click questions the web page offers. As on the web page, the
front man explains its answer, and on another provider the models move to that
provider's ladder with the promotion kept; each description says which.

The **evaluator** is served beside them, so a measurement is a chat message: "measure the
designer's shape against the best evolved network". It can only measure networks the
deployment already knows. It is told to state no number its tools did not return, and
`ESP_EVAL_MAX_RUNS` caps what it may spend. The optimiser is served and stays **private**:
poking it starts a paid evaluation.

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
| --- | --- |
| `esp/genome/` | The genome: neuro-san network definitions, the seven mutation operators, three seed topologies |
| `esp/measure.py` | Measure any neuro-san network on any question file — the library behind `make measure`, the web page and the evaluator |
| `esp/eval/` | The measured world, the 17 scored tasks, the runner, budget-aware model failover |
| `esp/surrogate/` | The Predictor and its honest quality reporting |
| `esp/evolve/` | The batch ESP loop — phases A through D in one sitting |
| `esp/service/` | The same loop as an interruptible service: persistent population, budget, lease; the evaluator's tools |
| `esp/report/` | The generated PDFs, and figures from the run history |
| `apps/web/` | The single-process browser front end: ask a network, measure networks |
| `apps/optimizer/` | One wake, runnable by hand or from any scheduler |
| `registries/` | neuro-san manifests: the optimiser and evaluator agents, and the generated champion |
| `scripts/` | Offline search, Predictor figures and the selection ablation, model probe, champion and studio serving, report and proof generation |
| `tests/` | The suite, plus the twelve committed measurements in `tests/fixtures/cache/` |
| `results/` | `results/history.json` and the figures the reports read |

## Testing and verification

```bash
make check      # ruff + the full suite, exactly what CI runs
make verify     # start a real neuro-san server and prove it fires the optimiser
make offline    # the free half of ESP, end to end, no key
make holdout    # select on half the tasks, judge on the other half, no key
make figures    # every published Predictor figure, regenerated, no key
make ablation   # does the Predictor pick better than chance? no key
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
| --- | --- |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Measurements, failure analysis, what the Predictor is, prior art |
| [SERVING.md](SERVING.md) | Deployment, state, budget, and what the agent may do |
| [SECURITY.md](SECURITY.md) | Reporting a vulnerability |
| [Dossier](docs/neuro-san-esp-Dossier.pdf) | The technical report, with captured evidence |
| [Primer](docs/neuro-san-esp-Primer.pdf) | The same result without the jargon |
| [Explainer](docs/neuro-san-esp-Explainer.pdf) | For a reader who knows nothing about agents, models or tokens |
| [Verification](docs/neuro-san-esp-Verification.pdf) | Every check in the dossier, re-run |

Built on [neuro-san](https://github.com/cognizant-ai-lab/neuro-san) by Cognizant AI Lab.
Licensed under Apache 2.0.
