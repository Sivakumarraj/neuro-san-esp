# neuro-san-esp

**A fitness function for [neuro-san](https://github.com/cognizant-ai-lab/neuro-san) agent
networks, and an evolutionary search that uses it.**

[![ci](https://github.com/Sivakumarraj/neuro-san-esp/actions/workflows/ci.yml/badge.svg)](https://github.com/Sivakumarraj/neuro-san-esp/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

neuro-san turns a sentence into a working multi-agent network: `agent_network_designer`
generates one, validates it and serves it. It generates **one**, and never measures it.
Nothing in neuro-san scores one network against another, so there is no way to say whether a
nine-agent topology beats a five-agent one for the same job, or which model each agent should
run.

This project supplies that missing half. It **measures** a network (accuracy, token cost and
size over a fixed set of questions) and **searches** for a better one, borrowing the
surrogate-assisted structure of **ESP** (Evolutionary Surrogate-assisted Prescription),
Cognizant AI Lab's method for optimisation where every real evaluation is expensive.

In the [ESP paper](https://arxiv.org/abs/2002.05368) the Prescriptor is itself a learned
model. **There is no learned Prescriptor here**: prescription is seven mutation operators
plus Pareto-front selection, and what is borrowed is the Predictor and the sample-efficiency
argument. In shape this is nearer to **LEAF** (*Evolutionary Neural AutoML for Deep
Learning*, GECCO 2019), which evolves architectures, with agents where LEAF had layers.

## Headline result

On questions it was never chosen on, an **evolved two-agent network beat the network
neuro-san's designer would build**: more accurate, fewer tokens, lower cost, on both repeats.

| 100 held-out judge questions × 2 repeats, `gpt-5.4-nano` | Evolved `2cc4` | Designer's shape |
| --- | --- | --- |
| Accuracy | **76.5%** (153/200) | 69.0% (138/200) |
| Tokens per question | **7,493** (−17%) | 9,018 |
| Cost per question | **$0.00229** (−17%) | $0.00275 |
| Cost per correct answer | **$0.00299** (−25%) | $0.00399 |
| Agents | **2** | 4 |

- **Chosen and judged on different questions.** The network was selected on the 60
  `meridian-select` questions and judged once on 100 judge questions built from the other
  half of the company. The two sets share no depot, contract or incident.
- **Found by the search, not by hand.** `2cc4` is a Coordinator over one searching
  Researcher, produced by the `remove_agent` operator; most of its gain is on 4-hop
  questions (18 against 10 of 28).
- **Stated at its real strength.** Right more often on 16 questions against 6: sign test
  p = 0.052, a strong lead on accuracy; the token and cost saving held on both repeats.

Every measured chunk is committed in `results/headtohead/paid-2026-10/`, and
`scripts/select_and_judge.py` reproduces the networks, the search and the summary
([details](#reproducing-the-head-to-head)).

## What is included

- **A measurement for any neuro-san network** on any question file you can check the
  answers to: `make measure`, a browser page, or an evaluator agent inside neuro-san.
- **The ESP loop**: a seed population measured for real, one Predictor per outcome
  objective, thousands of candidates ranked for free, and only the most promising paid for.
- **A generated test world**, Meridian Logistics: 24 depots, 40 contracts, 60 incidents in
  124 documents, invented so no model can answer from memory.
- **Held-out question sets**: `meridian-select` (60) to choose on and `meridian-judge-200` to
  judge on, from disjoint halves of the company, plus the original 17 and a 250-question bank.
- **Cost in dollars, beside tokens**, a per-question Predictor, a pool benchmark that
  rehearses for $0, and a capped head-to-head that resumes where it stopped.
- **A service, a browser front end and neuro-san's accelerator UI**, Docker images from a
  pinned lockfile, and a `make validate` gate.

About 15,700 lines of application code and 9,600 lines of tests (Python 3.12+).

## Architecture

```mermaid
flowchart LR
    subgraph World["Generated test world"]
        Docs["124 documents"]
        Q["Question sets<br/>17 · bank 250 · select 60 · judge 200"]
    end

    subgraph Search["ESP search"]
        Seeds["Phase A<br/>seed networks, measured"]
        Pred["Phase B<br/>Predictor per objective"]
        Breed["Phase C<br/>mutate + rank, zero calls"]
        Pay["Phase D<br/>measure the elite"]
        Seeds --> Pred --> Breed --> Pay --> Pred
    end

    subgraph Measure["Measurement"]
        Net["neuro-san network"] --> Score["accuracy · tokens · agents<br/>→ fitness"]
    end

    Q --> Net
    Docs --> Net
    Seeds --> Net
    Pay --> Net
    Score --> Pred

    Score --> Out["Web page · accelerator UI<br/>service · reports"]
```

Each candidate **is** a neuro-san `agent_network_definition` plus a per-agent model choice,
so every network the search produces runs as an ordinary neuro-san network. Fitness is
**derived** from measured or predicted outcomes by a fixed formula, never learned:

```text
fitness = accuracy − 0.06 · min(tokens / 600000, 1) − 0.02 · (agents / 9)
```

[docs/DESIGN.md](docs/DESIGN.md) explains every part: the genome, the seven operators, the
validity gate, what the Predictor is and is not, and the service.

## Quick start

Python 3.12+. Nothing in this section needs an account, a key, or a network.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make validate    # lint, docs, HOCON validator, the full suite, an offline search
make offline     # phases B and C: breed and rank candidates, zero LLM calls
make holdout     # select on half the questions, judge on the other half
```

`make offline` trains the Predictor on the 12 committed measurements in
`tests/fixtures/cache/` and evolves against them.

### With a key

Runs on whichever provider's key is present in `.env` (`.env.example` lists them), chosen the
way neuro-san-studio chooses:

```bash
cp .env.example .env      # uncomment your provider's key line; .env is gitignored
make check-key            # asks the provider whether the key works
make smoke                # four questions to the champion, end to end
```

A search needs a measured population first. There are two ways to get one. **They are
alternatives, not steps; run one, not both.**

```bash
make baseline             # A: measure the three seed networks on your provider
```

```bash
python scripts/adopt_measurements.py   # B: adopt the committed Gemini measurements
```

> **If a run ever reports `acc=0.00 tok=0 (cached)`**, an earlier run wrote zeros before the
> key worked. `tok=0` means no model was called. `rm -rf .esp-cache` and run the preflight
> again.

[docs/GUIDE.md](docs/GUIDE.md) covers providers and models, the preflight, what a run costs,
the web page, the accelerator UI, and running the optimiser as a service.

## Results of the search on the 17 built-in questions

The first search ran on Gemini. **Twelve networks measured on real model calls, 17 questions
each**, all on `gemini-3.1-flash-lite`:

| | Accuracy | Tokens | Agents | Fitness |
| --- | --- | --- | --- | --- |
| `seed:designer_shaped` — the shape the designer produces | 0.8235 | 385,280 | 4 | 0.7761 |
| `seed:flat_pair` | 0.8235 | 316,074 | 3 | 0.7852 |
| `seed:solo` — one agent, one tool | 0.8235 | 377,716 | 1 | 0.7835 |
| `mut:reassign_model` — cheapest win | 0.8824 | 260,052 | 5 | 0.8453 |
| **`mut:reassign_model` — best measured** | **0.9412** | 359,600 | 5 | **0.8941** |

- **Seventeen questions cannot rank individual networks**, which is why the headline result
  is judged on held-out questions. Selecting on half and judging on the other half, the
  searched winner beats the designer's shape in 90% of 200 splits, but a random network
  from the population does so 85.5% of the time (`make holdout`). On 24 held-out bank
  questions the evolved network answered 21 against the designer's 23 (`make bank-report`).
- **That champion's gain was a stronger router model.** Re-measured on held-out questions
  it tied the designer at 2.2× the cost, so the search was re-run for structure instead.
- **The select set leaves room to rank.** The designer's shape scored 16 of 20 (80%) on its
  first 20 questions, against 96% on the bank.
- **The Predictor beats chance offline.** At nine measurements its rank correlation was
  **−0.333**; trained on nine of the twelve it picks the best of three unseen networks 62%
  of the time against 33% by chance (`make ablation`), level with a rule that picks the
  network with the most agents.

Every number above is recomputed from committed data by a test.
[docs/FINDINGS.md](docs/FINDINGS.md) has the full measurements and the failure analysis.

## Limitations

- **One task domain.** Held-out questions come from the same generated world, so what is
  measured is stability across questions, not transfer to a new domain.
- **One generation of search on the held-out run.** `2cc4` is the best of eight children of
  one parent, judged on 100 questions twice; a longer search and more repeats would tighten
  the accuracy gap's error.
- **Two models measured.** The search on `gemini-3.1-flash-lite`, the head-to-head on
  `gpt-5.4-nano`. Nothing here says how the ranking holds on stronger models.
- **A tree-based Predictor cannot extrapolate** beyond the shapes it has seen, and twelve
  measured networks are too few to show that it helps the search. The pool benchmark is
  built to answer that.
- **Not novel as an idea.** AgentSquare (ICLR 2025) also uses a performance predictor to
  search agent designs. What is new here is doing it for neuro-san, which has no fitness
  function at all.

## Reproducing the head-to-head

Put your key in `.env` (gitignored):

```bash
OPENAI_API_KEY=sk-...your-key...
ESP_PROVIDER=openai
ESP_DEFAULT_MODEL=gpt-5.4-nano
ESP_MODEL_TIERS=gpt-5.4-nano,gpt-5.4-mini
ESP_PIN_MODELS=1          # a failing model stops the run instead of being swapped
```

```bash
python scripts/select_and_judge.py networks            # every network and its hash; $0
python scripts/select_and_judge.py mutants             # the search's 8 children of flat
python scripts/select_and_judge.py measure 2cc4 --suite judge --out runs/r1 --go &
python scripts/select_and_judge.py measure designer --suite judge --out runs/r1 --go
python scripts/select_and_judge.py summary results/headtohead/paid-2026-10
```

Both networks run at the same time under a dollar cap, and a stopped run resumes from its log.

| Stage | Measured | Outcome |
| --- | --- | --- |
| 1 | `make headtohead GO=1`: the designer against the Gemini champion | 68 / 100 each; the champion cost 2.2× per question |
| 2 | Six searched networks and the designer on select; the best on judge × 2 | `flat`: 77.5% against 67.5% (p = 0.011), equal tokens |
| 3 | One search generation from `flat` (8 children) on select; the best on judge × 2 | `2cc4`: the headline result |

Selection rules were fixed before each stage. The designer scored 67 to 70 of 100 in every
run, so about ±2 questions is run-to-run noise. Total: $5.18 for 1,888 question-runs.
[The paid runs explained](docs/neuro-san-esp-Paid-Runs.pdf) walks through it step by step.

## Pool benchmark

One search per method is one sample of that method. The pool benchmark measures 120 networks
once on the select questions, compares search strategies over that pool in hundreds of free
replicate searches, tests each against a permutation null (Holm-adjusted), and judges the
winners on the judge questions.

```bash
make pool                # prints the plan and its price; spends nothing
make pool REHEARSE=1     # the whole run against a simulated provider, for $0
make pool GO=1           # measures and judges, and resumes from the cache if stopped
```

## Repository layout

| Path | What lives there |
| --- | --- |
| `esp/genome/` | The genome: neuro-san network definitions, the seven mutation operators, three seed networks |
| `esp/eval/` | The world, the question sets, the runner, the scorer, model failover |
| `esp/surrogate/` | The Predictor and its quality reporting |
| `esp/evolve/` | The ESP loop and the same-budget experiment |
| `esp/service/` | The loop as an interruptible service, and the evaluator's tools |
| `esp/measure.py` | Measure any neuro-san network on any question file |
| `esp/report/` | The PDFs and the figures |
| `apps/web/` | The browser front end: ask a network, measure networks |
| `apps/optimizer/` | One optimiser wake, runnable by hand or from any scheduler |
| `registries/` | neuro-san manifests for the optimiser and the evaluator |
| `scripts/` | Command-line entry points behind the make targets |
| `tests/` | The suite, and the committed measurements in `tests/fixtures/cache/` |
| `results/` | Run history, held-out bank reports and figures |

## Testing

```bash
make validate      # lint, docs, HOCON validator, the full suite, an offline search
make verify        # a real neuro-san server firing the optimiser on its schedule
make figures       # every Predictor figure, regenerated from committed data
make bank-report   # the held-out bank comparison, from committed reports
make cost-report   # every network in tokens and in dollars
make pool REHEARSE=1   # the paid run end to end against a simulated provider, $0
```

The suite re-derives every question's answer from the documents, checks the registries
against contract tests, runs the experiment end to end against a fake provider, and fails if
this README disagrees with the committed measurements. Transcripts of real runs are in
`docs/proofs/`.

## Documentation

| | |
| --- | --- |
| [docs/DESIGN.md](docs/DESIGN.md) | How the search works, part by part |
| [docs/GUIDE.md](docs/GUIDE.md) | Running it: providers, web page, accelerator UI, service |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Measurements, failure analysis, prior art |
| [SERVING.md](SERVING.md) | Deployment, state, budget and the security model |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and checks |
| [SECURITY.md](SECURITY.md) | Keys, the public page, dependencies, reporting a vulnerability |
| [Beginner's guide](docs/neuro-san-esp-Beginner-Guide.pdf) | How it runs, step by step, with real examples |
| [Dossier](docs/neuro-san-esp-Dossier.pdf) | The technical report, with captured evidence |
| [Primer](docs/neuro-san-esp-Primer.pdf) | The same result without the jargon |
| [Paid runs](docs/neuro-san-esp-Paid-Runs.pdf) | The paid head-to-head runs, explained for a beginner |

Built on [neuro-san](https://github.com/cognizant-ai-lab/neuro-san) by Cognizant AI Lab.
Licensed under Apache 2.0.
