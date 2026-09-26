# Running it

Everything that needs a provider key or a server. The offline half needs neither; see the
[README](../README.md#quick-start).

## With an API key

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

**Any model of the three providers works** (`claude-opus-5-5`, `gpt-5.5`, `gemini-3.8-flash`),
including one released after the installed neuro-san: a name neuro-san does not list is sent
with its provider's class, and neuro-san passes it to the provider unchanged.
`ESP_DEFAULT_MODEL=claude-opus` runs the workers on Opus, and the model takes the rung it belongs
on, so a strong choice becomes the top of the ladder rather than sitting under Sonnet.
`ESP_MODEL_TIERS` sets both rungs outright. OpenAI has no version-free alias in neuro-san's
registry, so its rungs are the newest named there. Gemini's are the ones the committed
measurements were taken on.

```bash
cp .env.example .env      # uncomment your provider's key line, paste the key; .env is gitignored
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

### The paid run

The pool benchmark is the run a paid key is for. Rehearse it first; it costs nothing and
prints exactly how many question-runs the real one makes:

```bash
make pool                 # the plan and its price
make pool REHEARSE=1      # every stage against a simulated provider, $0
make pool GO=1            # the real run: preflight, pool, replicates, judging
```

Before `GO=1`, measure the designer's shape on 20 select questions with the new key
(`python -m esp.measure registries/r11_seed_designer_shaped.hocon --tasks meridian-select:20`,
well under a dollar) to see what a
question really costs on that provider, and set a spending limit on the key itself in the
provider's console. A stop for quota or a restart resumes from the cache; nothing is paid
for twice.

### On Google's free tier

Gemini's free tier caps requests per model per day, and some models cannot fund a single
candidate. For that case only, the runner fails over between models as each cap is reached,
with measured caps and per-model pacing kept as data in `esp/eval/failover.py`. `make probe`
re-measures them against your key. None of this machinery engages on Anthropic or OpenAI,
which have per-minute limits and no daily cap.

## Talk to the agents, and measure them, in a browser

```bash
python apps/web/serve.py  # then open http://localhost:7860
```

In a Codespace, open port 7860 from the **Ports** tab with the globe icon, in a real browser tab.
VS Code's preview shows the page but does not send the port's login with its requests, so every
question fails with "could not reach its server" and no `POST /ask` line appears in the terminal.
`make studio` serves neuro-san's own UI on port 4173 the same way.

The page has two tabs. **Ask a network** is described below. **Measure networks** puts the
same questions to up to four networks at once: the twelve committed ones, plus any HOCON you
place in `ESP_NETWORKS`. It uses the built-in benchmark or a JSON Lines file you paste, shows
progress, and marks the Pareto front. It also gives a question-by-question grid and a JSON
download. Measuring is paid for, so a deployment caps the runs per UTC day
(`ESP_WEB_MAX_MEASURE`) and runs one measurement at a time. Questions are capped per day
(`ESP_WEB_MAX_QUESTIONS`) and per visitor per hour (`ESP_WEB_PER_CLIENT_HOURLY`), and the
counts survive a restart when `ESP_WEB_SPEND_FILE` is set, as the Docker images do. A
network is never uploaded: it names Python classes to import.

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

## Open every measured network in the accelerator UI

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
r01_evolved_reassign_model_3bf9c008
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

## Serve the champion as an ordinary agent

```bash
python scripts/serve_champion.py   # writes registries/champion.hocon
export AGENT_MANIFEST_FILE=$PWD/registries/champion_manifest.hocon
export AGENT_TOOL_PATH=$PWD PYTHONPATH=$PWD
python -m neuro_san.service.main_loop.server_main_loop
```

## Run the optimiser as a service

```bash
export AGENT_MANIFEST_FILE=$PWD/registries/manifest.hocon
export AGENT_TOOL_PATH=$PWD PYTHONPATH=$PWD
python -m neuro_san.service.main_loop.server_main_loop
```

The server logs `Found 1 periodic agent interactions` and from then on fires the optimiser on
the cron in `registries/manifest.hocon` with `user_id: system`, no client attached. See
[SERVING.md](../SERVING.md) for state, leases, budget and the security model. In a container:
`docker compose up -d optimizer`.
