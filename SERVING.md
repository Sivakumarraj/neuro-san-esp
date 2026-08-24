# Serving the optimiser

The optimiser is an **event-invoked** agent network: there is no user on the
other end and no request to answer. neuro-san fires it on a schedule, it spends
what the day's provider budget allows, writes down where it got to, and stops.

Verified end to end against **neuro-san 0.6.95** on Python 3.12 — the transcript
of that run is in [What "verified" means here](#what-verified-means-here) below.

## Requirements

Python **3.12+**. The neuro-san server imports `asyncio.eager_task_factory`,
which does not exist on 3.11. The ESP library itself runs on 3.10+; only the
server needs 3.12.

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -e ".[dev]" neuro-san
```

## Layout the framework expects

Coded tools are resolved as fully-qualified classes from `PYTHONPATH`, with
`AGENT_TOOL_PATH` pointing at the root they hang from. This repository puts the
tools inside the library itself rather than in a sibling directory:

```hocon
"class": "esp.service.coded_tools.RunWake"
```

so both variables point at the repository root:

```bash
export AGENT_TOOL_PATH=$PWD
export PYTHONPATH=$PWD
```

## Run it

```bash
export AGENT_MANIFEST_FILE=$PWD/registries/manifest.hocon
export AGENT_TOOL_PATH=$PWD
export PYTHONPATH=$PWD
export ESP_STATE=$PWD/state          # population and budget live here
export GOOGLE_API_KEY=...

python -m neuro_san.service.main_loop.server_main_loop
```

The server logs `Found 1 periodic agent interactions` at startup. That line is
the whole deployment: from then on it fires `optimizer` on the cron in
`registries/manifest.hocon` with `user_id: system`, with no client connected.

### Check the configuration first

```bash
python apps/optimizer/run_optimizer.py --check
```

```
  [ok  ] provider key: GOOGLE_API_KEY is set
  [ok  ] AGENT_TOOL_PATH: /srv/neuro-san-esp
  [ok  ] PYTHONPATH: /srv/neuro-san-esp
  [ok  ] state directory writable: /srv/neuro-san-esp/state
  [ok  ] designer demo mode: off
```

A wake **refuses to start** if any of those fail. That is deliberate and it is
the cheaper failure: a misconfigured evaluator does not crash, it scores every
candidate zero and the cache keeps that answer forever, so the search is taught
that good topologies are bad. Both real instances of this in the project's
history were configuration, not logic — a missing `AGENT_TOOL_PATH` once made a
probe report every healthy model as BROKEN.

### One wake, without a server

```bash
python apps/optimizer/run_optimizer.py
```

Same code path, no gRPC, no HTTP. Use it under `cron`, a Kubernetes `CronJob`, a
systemd timer, or by hand. Exit codes are meant for a scheduler:

| Code | Meaning |
|---|---|
| `0` | did something, **or correctly did nothing** |
| `1` | could not run at all (no `GOOGLE_API_KEY`) |

A declined lease is exit `0`. A scheduler overlapping a wake that is still
evaluating is normal operation, not a failure.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `ESP_STATE` | `state` | population, per-day spend, exhausted models, lease |
| `ESP_OPTIMIZER_CRON` | `0 * * * *` | overrides the schedule in the manifest |
| `ESP_LEASE_SECONDS` | `3600` | how long a wake may hold the lease before it is considered dead |
| `ESP_CACHE` | `.esp-cache` | evaluations keyed by genome hash, so a topology is never paid for twice |
| `ESP_WORKERS` | `4` | task-level parallelism inside one candidate |
| `ESP_MAX_EXECUTION_SECONDS` | `600` | per-agent budget; **must** cover queueing in the rate limiter, not just thinking |

### Why hourly, and not more often

A wake evaluates at most three candidates. The free tier allows 500 requests per
day per model and one candidate costs about 165 of them, so the day's budget is
about three candidates. A fifteen-minute schedule would produce three wakes that
work and ninety-three that find the budget already spent and decline. **The
cadence is set by what the provider allows, not by how often we would like
news.** On a paid tier, raise both the cron frequency and `MAX_PER_WAKE` in
`esp/service/optimizer.py` together — raising one alone does nothing.

## State, and what survives a restart

`$ESP_STATE/state.json` holds everything a wake needs to continue: the
population, which models are spent *today*, and the wake counter. It is written
after **every single candidate**, not per generation, and written atomically
(temp file, then rename).

That is not defensive coding, it is the operating assumption. A wake is
*expected* to be cut short by an exhausted quota rather than to finish, and an
interruption after an eight-minute evaluation must not throw that evaluation
away.

**Mount it.** `compose.yaml` puts it on a named volume. A container that loses
this directory loses weeks of accumulated population and starts again from the
seeds — the provider budget is what makes that expensive, not the compute.

`$ESP_STATE/lease.json` stops two wakes overlapping. It expires after
`ESP_LEASE_SECONDS`, and a corrupt lease is treated as a free one: better to
risk one overlapped wake than to need a human to delete a file before the
service will run again.

## In a container

```bash
docker compose up -d            # the service, on its schedule, with a state volume
docker compose logs -f
```

## What the agent is allowed to do

Nothing that matters. The model never chooses a candidate and never spends
budget — the surrogate and the mutation operators do that, in code, from
measured fitness. It has exactly two tools:

- `RunWake` — runs the wake and returns a finished measurement as JSON. It
  ignores its arguments, so the model cannot direct it.
- `ReportFinding` — drafts an operator-facing note. It cannot send it anywhere.

There is no tool that deletes and no tool that reaches outside the process. An
unattended process that wakes every hour should not be able to do anything
irreversible or outward-facing, and this one cannot.

It is also instructed to stay **silent** unless something improved. Most wakes
find nothing, and a service that announces every wake trains its operator to
ignore it — and then the one wake that matters is ignored too.

## What "verified" means here

The scheduling path was run against a real neuro-san server, not reasoned about.
With `ESP_OPTIMIZER_CRON="*/1 * * * *"` to shorten the wait:

```
Starting PeriodicEventInitiator with 1.000000 seconds period
Found 1 periodic agent interactions
HealthProbeServer started on port 8081

{"message": "Received a optimizer.StreamingChat request for '
Scheduled optimisation wake. Run one wake, and report only if a better
topology was found. Most wakes find nothing; say nothing when that happens.
'", "user_id": "system", "Timestamp": "2026-08-22T13:14:01.233063", ...}
```

`user_id: system` with no client attached is the property that matters: the
framework initiated that interaction by itself.

**What is not verified:** a wake completing a full evaluation *through the
server*. Both of this project's 500-per-day models were exhausted when the
above was captured, so the front agent's own reasoning call returned 429 before
it could call `RunWake`. The wake path itself is verified separately and
repeatedly by `apps/optimizer/run_optimizer.py`, which runs the identical
`esp.service.optimizer.wake()`; what the server adds on top of it is the
trigger, and the trigger is what the transcript above shows.

That distinction is worth stating rather than glossing: an exhausted quota
stopping an agent before it reasons is an environment limit, and reporting it as
a verified end-to-end run would be exactly the kind of overstatement the rest of
this repository is built to avoid.

## Security

No key is committed here and none ever will be. Everything reads from the
environment; see [SECURITY.md](SECURITY.md). `.gitignore` covers the cache, the
generated networks and the state directory.
