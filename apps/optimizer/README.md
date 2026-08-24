# The Optimiser

An agent network that searches for a better agent network, on a schedule, with
nobody watching.

It is the reason this repository is a service and not a script. The batch
version — `scripts/run_esp.py` — plans a fixed number of generations and died
every single time it was run, not from a bug but from arithmetic: the free tier
allows 500 requests per day per model, one candidate costs about 165 of them, so
a day buys three candidates and a four-generation run needs forty.

**The daily cap is not an obstacle to a service. It is its rhythm.**

## What one wake looks like

```
1.  take the lease            one optimiser at a time, or two wakes spend the
                              same budget twice and write over each other
2.  read yesterday's state    the population, and which models are spent today
3.  prime the failover ladder so this fresh process does not rediscover this
                              morning's exhaustion by paying for it again
4.  Phase C, free             mutate 400 candidates, rank them with the
                              Predictor, zero provider calls
5.  evaluate up to 3          the best unseen ones -- saving state after EVERY
                              candidate, not at the end
6.  quota hit?                stop. That is a normal ending, not an error, and
                              the population is untouched
7.  release the lease         in a finally block. And it expires anyway, in
                              case this process never reaches the finally block
```

Steps 1–3 and 7 are all about being interrupted, because interruption is the
expected ending rather than a failure.

## The two agents

| Agent | What it does |
|---|---|
| `Optimizer` | The front man. `invocation: "event"`, so neuro-san fires it with no user and no request. Calls `RunWake`, then decides whether the result is worth mentioning. |
| `RunWake` | A `CodedTool`. Runs the wake and returns a finished measurement as JSON. **Ignores its arguments**, so the model cannot direct it. |
| `ReportFinding` | Drafts three sentences for an engineer who has not looked at this in a week. It cannot send them anywhere. |

## The rules it runs by

**The model decides nothing that costs money.** Which candidate to evaluate is
chosen by the surrogate and the mutation operators, in code, from measured
fitness. The model's job is to notice that something improved and write it up.

**Silence is the correct outcome.** Most wakes find nothing — the day buys three
candidates, so progress is measured in weeks. The agent is instructed to say
nothing at all when `improved` is false. A service that announces every wake
trains its operator to ignore it, and then the one wake that matters is ignored
too.

**It cannot do anything irreversible or outward-facing.** There is no tool that
deletes and no tool that sends. This is a process that wakes hourly with a
language model in it and no human in the loop; the surface it can reach is the
whole safety argument, and the surface is two read-and-measure tools.

**Exhaustion is remembered per day, not permanently.** A model retired for good
would leave the service dead after its first bad afternoon. Keyed by UTC day, it
recovers by itself at the quota reset — and UTC because the service may move
host, and a local-time boundary would double-spend or skip a day when it does.

**A quota failure is never recorded as a score.** A daily cap does not fail
every task at once; it starts failing them part-way through a candidate, which
produces a plausible-looking partial score. Cached, that tells the search
forever that a good network is mediocre.

## Run it

Check the configuration before spending anything:

```bash
python apps/optimizer/run_optimizer.py --check
```

A wake refuses to start if a fatal check fails. A misconfigured evaluator does
not crash — it scores every candidate zero and caches it — so refusing to start
is by far the cheaper failure.

One wake, by hand or from any scheduler:

```bash
export PYTHONPATH=$PWD AGENT_TOOL_PATH=$PWD
export ESP_STATE=$PWD/state
export GOOGLE_API_KEY=...

python apps/optimizer/run_optimizer.py
```

```json
{
  "wake": 1,
  "generation": 1,
  "evaluated_this_wake": 3,
  "population": 3,
  "best_fitness": 0.7868,
  "improved": true,
  "stopped_because": "",
  "exhausted_today": [],
  "note": "a better topology was found"
}

best so far: 459ac1a66d925b0c  seed:designer_shaped  acc=0.82 tokens=278,532 agents=4 fitness=+0.7868

MATERIAL: a better topology was found -- worth telling someone.
```

As a service, on neuro-san's own schedule — see [SERVING.md](../../SERVING.md):

```bash
AGENT_MANIFEST_FILE=$PWD/registries/manifest.hocon \
python -m neuro_san.service.main_loop.server_main_loop
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | did something, **or correctly did nothing** |
| `1` | could not run at all (no `GOOGLE_API_KEY`) |

A declined lease exits `0`. A scheduler overlapping a wake that is still
evaluating is normal operation:

```
skipped: another wake holds the lease (wake-1787401390,
taken 2026-08-22T12:23:10.884733+00:00); it expires in 57 min
```

The expiry time is in that message deliberately. Without it, an operator reading
"another wake holds the lease" cannot tell a busy service from a wedged one.

## Settings

| Variable | Default | What it does |
|---|---|---|
| `ESP_STATE` | `state` | population, per-day spend, exhausted models, lease |
| `ESP_LEASE_SECONDS` | `3600` | how long a wake may hold the lease before it counts as dead |
| `ESP_OPTIMIZER_CRON` | `0 * * * *` | overrides the manifest schedule |
| `MAX_PER_WAKE` | `3` (in code) | real evaluations one wake will attempt |
| `SURROGATE_POOL` | `400` (in code) | candidates ranked per wake — free, so it runs even when only one can be afforded |

## Honest limits

**It has not yet beaten the baseline.** Three real evaluations exist. Beating a
baseline needs a population, a population needs budget, and budget arrives at
three candidates a day. If it never beats the baseline, that gets reported as
the result — `esp/report/build.py` renders a negative outcome as readily as a
positive one, and distinguishes "a search ran and lost" from "no search ran",
because those are different claims.

**The Predictor is currently worse than useless.** On three training samples its
cross-validated rank correlation is `+0.000` — no better than chance, printed as
such by `make offline`. The machinery is correct and the ranking costs nothing;
its *quality* is a function of how many real evaluations have accumulated, which
is exactly why this runs every hour instead of once.

**A full wake through the server is unverified.** The scheduling path is
verified — the server fires the interaction with `user_id: system` and no client
attached. What has not been observed is a wake completing a full evaluation
*inside* that server process, because the provider quota was exhausted when it
was tried. The wake itself is verified repeatedly through
`run_optimizer.py`, which runs the identical code.
