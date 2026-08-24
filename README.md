# neuro-san-esp

**A fitness function for [neuro-san](https://github.com/cognizant-ai-lab/neuro-san)
agent networks, and an evolutionary search that uses it.**

[![ci](https://github.com/Sivakumarraj/neuro-san-esp/actions/workflows/ci.yml/badge.svg)](https://github.com/Sivakumarraj/neuro-san-esp/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

neuro-san turns a sentence into a working multi-agent network. `agent_network_designer`
generates one, validates it, and serves it in about five seconds. It generates **one**,
and never measures it.

Nothing in neuro-san scores an agent network against another. `neuro_san/test/evaluators/`
holds `assertIn`-style assertions for integration testing a single network's answer, which
is regression testing rather than a fitness function. So nobody can answer whether a
nine-agent topology beats a five-agent one for the same job, which model each agent should
run, or whether the generated instructions are any good. That is design without evaluation.

This project supplies the missing half. It measures a network — accuracy, token cost and
size over a fixed task set — and then searches for a better one using **ESP**
(Evolutionary Surrogate-assisted Prescription), Cognizant AI Lab's own method for
optimisation where every real evaluation is expensive: learn a cheap Predictor from real
measurements, evolve thousands of candidates against it for free, and pay for real
evaluation only on the elite.

## Features

- **A measured fitness function** for neuro-san topologies: accuracy, tokens and agent
  count, scalarised for selection with the Pareto front recorded separately.
- **The genome is neuro-san's own** `agent_network_definition`, plus a per-agent `model` —
  the main cost/quality knob the framework already supports and nobody tunes.
- **Seven mutation operators** — add, remove, rewire, split, merge, toggle search, reassign
  model. Invalid mutants are discarded, never repaired.
- **A synthetic evaluation world** — 24 depots, 40 contracts, 60 incidents, 124 documents
  and 17 one-to-four-hop questions, generated so every answer is correct by construction.
- **Deterministic retrieval**, so fitness measures the topology rather than a retrieval
  layer that drifts between generations.
- **Runs as a service, not a batch job** — an hourly `invocation: "event"` agent that
  spends what the daily free tier allows, saves state after every candidate, and stops.
- **Provider-agnostic** — Google or OpenRouter, selected by model name in `.env`.
- **Budget-aware failover** across models, with daily caps kept as data and a preflight
  that refuses to start on a configuration that would produce wrong numbers.
- **A full test suite**, and reports that decline to print a number nobody measured.

## Quick start

Python 3.12+. Works the same in Codespaces, a devcontainer, or a laptop.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make check       # ruff + the full test suite
make offline     # Phase B and C: 2,000 candidates scored, zero LLM calls
```

Nothing above needs an account, a key, or a network. `make offline` trains the Predictor
on the seed measurements committed in `tests/fixtures/cache` and evolves against it,
announcing which cache it used.

### With an API key

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env      # paste the key in; .env is gitignored
python apps/optimizer/run_optimizer.py --check   # preflight
make probe                                       # which models answer today
make baseline                                    # measure the seed topologies
```

The preflight reports where the key came from and which models the run will use:

```
[ok  ] provider key: set, GOOGLE_API_KEY, from .env
[ok  ] model ladder: gemini-3.5-flash-lite, gemini-3.1-flash-lite
       -- 1000 requests/day = about 6 candidate(s)
```

**Run the preflight first.** A misconfigured evaluator does not crash. It scores every
candidate zero, and the cache keeps that answer forever, so the search is taught that good
topologies are bad.

### The front end — talk to the agents in a browser

This is the whole thing working end to end: a page in Chrome, real agents behind it, real
model calls.

```bash
cp .env.example .env      # paste your key in
python apps/web/serve.py  # then open http://localhost:7860
```

One process. No separate backend to start, no second repository — the page runs questions
through the champion topology on neuro-san's direct session, the same code path the
evaluator measures with, so what you talk to is exactly what was scored.

Questions from the graded task set are **marked against the known answer in front of you**:

```json
{"answer": "R. Delacroix", "expected": "R. Delacroix", "correct": true, "seconds": 48.4}
```

Expect **30–60 seconds** for a multi-hop question. Four documents have to be found and
chained; anything faster would mean it did not really look.

### Running the service

```bash
export AGENT_MANIFEST_FILE=$PWD/registries/manifest.hocon
export AGENT_TOOL_PATH=$PWD PYTHONPATH=$PWD
python -m neuro_san.service.main_loop.server_main_loop
```

The server logs `Found 1 periodic agent interactions` and from then on fires the optimiser
on the cron in `registries/manifest.hocon` with `user_id: system`, no client attached.
See [SERVING.md](SERVING.md) for state, leases, budget and the security model.

In a container:

```bash
docker compose up -d optimizer
```

### Running tests

```bash
make check                # ruff + the full suite, exactly what CI runs
make verify               # start a real server and prove it fires the optimiser
```

## Results

Three seed topologies over the same 17 tasks on one model. The cost spread is real: the
four-agent network is the cheapest and the single-agent network the most expensive, a 42%
difference for the same score. The matching accuracy is **not** a result — all three fail
the same three tasks for three different reasons, one of them by crashing out of them
entirely.

Full numbers, the failure analysis, and the prior art this sits beside are in
[docs/FINDINGS.md](docs/FINDINGS.md). Three PDFs are generated: a technical [dossier](docs/neuro-san-esp-Dossier.pdf), a jargon-free
[primer](docs/neuro-san-esp-Primer.pdf), and an
[explainer](docs/neuro-san-esp-Explainer.pdf) (`make explainer`) written for a reader who knows
nothing about agents, models or tokens — every idea anchored to something ordinary, a restaurant
kitchen or an electricity bill. `scripts/verification_report.py` produces a
[verification report](docs/neuro-san-esp-Verification.pdf) by running every check in it.

## Limitations

- **The surrogate is not trained on the data this project has.** It needs eight samples and
  three exist, so it returns a constant and Phase C ranks nothing. `report_quality`
  publishes that it was *not measured* rather than a number.
- **No evolved candidate has beaten the baseline.** A candidate costs about 165 provider
  requests against a free tier of 500 per day per model, so no search has completed.
- **Six of the fifty-one seed task runs were timeouts or crashes**, not answers. `accuracy`
  counts them as wrong; `answered_accuracy()` excludes them. Both are reported.
- **One task domain.** A topology that wins at multi-hop retrieval need not win elsewhere.
- **The surrogate idea is not novel.** AgentSquare (ICLR 2025) uses a performance predictor
  for the same purpose. What is absent from that work is neuro-san, and what is absent from
  neuro-san is any fitness function at all.

## Documentation

| | |
|---|---|
| [SERVING.md](SERVING.md) | Deployment, state, budget, and what the agent may do |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Measurements, failure analysis, prior art |
| [SECURITY.md](SECURITY.md) | Reporting a vulnerability |

Built on [neuro-san](https://github.com/cognizant-ai-lab/neuro-san) by Cognizant AI Lab.
Licensed under Apache 2.0.
