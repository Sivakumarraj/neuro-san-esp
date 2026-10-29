"""A single-process web front end for the measured champion topology.

The repository could already serve the champion -- but only as two processes
(a neuro-san server plus a separate web client from another repository), on a
developer's machine, reachable by nobody. That is not "online", and the winner
of a search that nobody can talk to is still just a hash in a report.

This is one process with no external client: FastAPI serves a page, the page
posts a question, and the question runs through the champion via neuro-san's
direct session -- the same code path the evaluator measures with. What a
visitor talks to is the measured topology with one line changed: the front man
explains its answer instead of returning the bare value the scorer needs. If the
deployment holds a key for a different provider from the one the champion was
measured on, the models move rung for rung and the page says so
(`esp/serving.py`).

It binds 7860 because that is what Hugging Face Spaces expects, and Spaces is
the cheapest way to put this behind a URL somebody can click. It works the same
under `docker run -p 7860:7860` or bare `python apps/web/serve.py`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
import logging  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

from esp.config import bootstrap  # noqa: E402

# Load .env BEFORE anything downstream reads env at import time. The rate
# limiter builds its keyring at module load from GOOGLE_API_KEYS; leaving
# bootstrap for main() froze that keyring empty for anyone who only set
# their keys in .env, so the first per-day 429 stopped the page instead
# of rotating.
bootstrap()

import hmac  # noqa: E402
import html  # noqa: E402
import ipaddress  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from collections import deque  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

try:
    # Module level, because `from __future__ import annotations` makes FastAPI
    # resolve every handler annotation against these globals -- see test_web.
    from fastapi import Request
except ImportError:                 # the page cannot run without fastapi anyway
    Request = None

from esp.config import key_name_for, key_problem  # noqa: E402
from esp.eval import corpus_tool, measurements  # noqa: E402
from esp.eval.runner import _ask, never_answered, write_network  # noqa: E402
from esp.eval.tasks import TASKS, score  # noqa: E402
from esp.evolve import paid  # noqa: E402
from esp.genome.definition import Genome  # noqa: E402
from esp.genome.seeds import SEEDS  # noqa: E402
from esp.measure import BUILTIN, Candidate, SuiteError, catalog, pareto, parse_suite  # noqa: E402
from esp.measure import measure as measure_network  # noqa: E402
from esp.service.state import Evaluated, ServiceState  # noqa: E402
from esp.serving import (  # noqa: E402
    SHOWCASE,
    demoted,
    display_question,
    graded,
    presentable,
    unavailable,
)
from esp.surrogate.predictor import MIN_SAMPLES  # noqa: E402

# Which network the page serves. "heldout", the default, is the network that won
# on questions it was never chosen on (esp/evolve/paid.py). "fitness" is the best
# on the built-in seventeen, across this deployment's own state and the committed
# cache: what a deployment running its own search wants. The page served the
# fitness choice alone until October 2026, and that network tied the designer on
# held-out questions at 2.1 times the cost.
SERVE = os.environ.get("ESP_SERVE", "heldout").strip().lower()
if SERVE not in {"heldout", "fitness"}:
    raise ValueError(f"ESP_SERVE={SERVE!r}: use heldout or fitness")

# A visitor is not a benchmark run, and every question is paid for: a multi-hop
# answer is ten or more model calls. A public deployment caps how many it will
# answer before it stops, so one careless loop cannot run up the key's bill.
MAX_QUESTIONS = int(os.environ.get("ESP_WEB_MAX_QUESTIONS", "40"))
_asked = {"count": 0}

# One visitor must not be able to spend everybody's day. The daily cap above is
# the budget guarantee; this is fairness inside it. Keyed on the connecting
# address -- behind a proxy that is the proxy, so ESP_WEB_TRUST_PROXY=true reads
# the last X-Forwarded-For entry instead: the address the proxy itself saw. A
# proxy appends to that header, so earlier entries are whatever the visitor sent.
# Only set it behind exactly one proxy that does so (Hugging Face Spaces).
PER_CLIENT_HOURLY = int(os.environ.get("ESP_WEB_PER_CLIENT_HOURLY", "10"))
TRUST_PROXY = os.environ.get("ESP_WEB_TRUST_PROXY", "").lower() in {"1", "true", "yes"}
_recent: dict[str, deque] = {}
MAX_TRACKED_CLIENTS = 10_000

# Both caps are per UTC day, because provider quotas are. Held in memory, the
# count went back to zero on every restart, so a crash loop had no cap at all.
# ESP_WEB_SPEND_FILE keeps it across restarts; the images set it under state/.
SPEND_FILE = os.environ.get("ESP_WEB_SPEND_FILE") or None
_budget_lock = threading.Lock()
_spend_day = {"day": None}

# Finished measurements kept for their result pages. Without a bound, a
# long-running deployment grows this for ever.
MAX_KEPT_JOBS = 50

# Measuring is the expensive half of the page: one network on the built-in
# benchmark is seventeen multi-hop runs, about 165 model calls. A deployment
# caps how many question-runs it will spend in total, how many networks one
# measurement may compare, and how long a visitor's question file may be; and
# it runs one measurement at a time, so a second visitor queues behind a key's
# rate limit rather than doubling it.
MAX_MEASURE = int(os.environ.get("ESP_WEB_MAX_MEASURE", "68"))
MAX_NETWORKS = 4
MAX_SUITE_QUESTIONS = 50
MAX_SUITE_BYTES = 200_000
# The longest question either form sends to a network. /ask cut at this and a
# question file had no limit at all: one 192,027-character question went to
# every chosen network and was charged as one run.
MAX_QUESTION_CHARS = 1_000
# Every POST body must declare its length and stay under this, checked before a
# byte of it is read. The schema's own limits apply only after a body is read and
# parsed whole: a 150 MB question was read, parsed and echoed back in a 422.
# Twice the largest question file, because JSON escaping can grow it.
MAX_BODY_BYTES = 2 * MAX_SUITE_BYTES
# Who may show this page in a frame. The page has paid buttons, so another site
# must not frame it and click them through. Hugging Face Spaces shows an app in
# an iframe on huggingface.co, which is why that origin is allowed by default.
FRAME_ANCESTORS = os.environ.get("ESP_WEB_FRAME_ANCESTORS", "'self' https://huggingface.co")

# When set, every paid request needs "Authorization: Bearer <token>". Unset, the
# page stays open to anyone with the link and the daily caps are the only guard.
ACCESS_TOKEN = os.environ.get("ESP_WEB_TOKEN", "")

logger = logging.getLogger("esp.web")

# A caller's own request id is kept only if it is short and plain, so it cannot
# put anything odd in the log line it ends up on.
_REQUEST_ID = re.compile(r"[A-Za-z0-9-]{1,64}")


def authorized(header: str) -> bool:
    """Whether a request carries the deployment's token. Constant time, so the
    reply's timing says nothing about how much of a guess was right."""
    if not ACCESS_TOKEN:
        return True
    scheme, _, token = header.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token.encode(), ACCESS_TOKEN.encode())
# Networks an operator adds by placing HOCON files here. Never uploaded: a
# network names Python classes to import, so taking one from a visitor would be
# taking code.
NETWORKS_DIR = os.environ.get("ESP_NETWORKS") or None
_measured = {"count": 0}
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()
_candidates: dict[str, Candidate] | None = None


# Anything shaped like a provider key. Errors are shown to whoever holds the
# link, and a provider that echoes a request detail must never put a key on it.
_KEY_SHAPES = re.compile(
    r"(AIza[0-9A-Za-z_-]{16,}|AQ\.[0-9A-Za-z_-]{16,}|sk-[0-9A-Za-z_-]{16,})")


def redact(text: str) -> str:
    return _KEY_SHAPES.sub("[redacted key]", text)


# A file path in an exception says where this server keeps things. The visitor
# needs the provider's words, not the layout of the machine.
_PATHS = re.compile(r"(?:/[\w.\-]+){2,}")


def public_error(text: str) -> str:
    """An error as a visitor may see it: keys masked, server paths hidden."""
    return _PATHS.sub("[path]", redact(text))


def log_failure(where: str) -> None:
    """The whole traceback, to the server's own log. The visitor sees one
    redacted line; the operator needs the stack to fix anything, and an
    unattended page that only ever shows the short form cannot be debugged."""
    print(f"[{where}] " + redact(traceback.format_exc()), file=sys.stderr, flush=True)


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _roll_day() -> None:
    """Start a new day's counts, loading today's from disk once if kept there.
    Call with _budget_lock held."""
    today = _today()
    if _spend_day["day"] == today:
        return
    _spend_day["day"] = today
    _asked["count"], _measured["count"] = 0, 0
    if SPEND_FILE:
        try:
            saved = json.loads(Path(SPEND_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(saved, dict) and saved.get("day") == today:
            # A field that is not a count is ignored, never fatal. int() on a
            # null or a string raised under the budget lock, and every request
            # that day answered 500.
            _asked["count"] = _count(saved.get("asked"))
            _measured["count"] = _count(saved.get("measured"))


def _count(value) -> int:
    """A saved count, or 0 when the file holds something else there."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _save_spend() -> None:
    """Write the day's counts. Call with _budget_lock held."""
    if not SPEND_FILE:
        return
    path = Path(SPEND_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_suffix(".tmp")
        scratch.write_text(json.dumps({"day": _spend_day["day"], "asked": _asked["count"],
                                       "measured": _measured["count"]}), encoding="utf-8")
        scratch.replace(path)
    except OSError as exc:        # a full disk must not take the page down
        print(f"could not keep the spend count: {exc}", file=sys.stderr)


def client_key(request) -> str:
    if request is None:
        return "unknown"
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return _bucket(forwarded.split(",")[-1].strip())
    return _bucket(request.client.host) if request.client else "unknown"


def _bucket(address: str) -> str:
    """The share an address draws on: itself, or its /64 for IPv6.

    One IPv6 subscriber is routed a whole /64 -- 2^64 addresses. Keyed on the
    full address, a visitor took a fresh hourly share per question by changing
    the low bits, and spent the day's cap alone.
    """
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return address
    if parsed.version == 4:
        return str(parsed)
    if parsed.ipv4_mapped is not None:
        return str(parsed.ipv4_mapped)
    return str(ipaddress.ip_network(f"{parsed}/64", strict=False))


def _over_client_limit(key: str) -> bool:
    """Record one question for this client unless it is over its hourly share.
    Call with _budget_lock held."""
    now = time.monotonic()
    # Forget visitors idle for an hour. Without this the table grows by one
    # entry per address that ever asked, for as long as the page runs.
    if len(_recent) > MAX_TRACKED_CLIENTS:
        for idle in [k for k, w in _recent.items() if not w or now - w[-1] > 3600]:
            del _recent[idle]
    window = _recent.setdefault(key, deque())
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= PER_CLIENT_HOURLY:
        return True
    window.append(now)
    return False


def _prune_jobs() -> None:
    """Drop the oldest finished measurements beyond MAX_KEPT_JOBS. Call with
    _job_lock held; dicts keep insertion order, so the first are the oldest."""
    finished = [job_id for job_id, job in _jobs.items() if job["state"] != "running"]
    for job_id in finished[:max(0, len(finished) - MAX_KEPT_JOBS)]:
        del _jobs[job_id]


def candidates() -> dict[str, Candidate]:
    """The measurable networks, rendered once per process."""
    global _candidates
    if _candidates is None:
        _candidates = {c.id: c for c in catalog(NETWORKS_DIR)}
    return _candidates


def run_job(job: dict, chosen: list[Candidate], tasks: list, suite: str) -> None:
    """Measure each chosen network in turn. One failing network does not stop
    the others: its row says why, and the rest are still measured."""
    def tick(_result) -> None:
        job["done"] += 1

    try:
        for candidate in chosen:
            job["current"] = candidate.name
            before = job["done"]
            try:
                report = measure_network(candidate.hocon, tasks, suite=suite, on_result=tick)
                job["reports"].append({"id": candidate.id, "name": candidate.name,
                                       **report.as_dict()})
            except (OSError, SuiteError) as exc:
                job["reports"].append({"id": candidate.id, "name": candidate.name,
                                       "error": public_error(str(exc))[:300]})
            # A refused network still used up its share of the run, so progress
            # reaches the total whatever happened to it.
            job["done"] = before + len(tasks)
        pareto(job["reports"])
        job["state"] = "done"
    except Exception as exc:                       # reported to the page, never raised
        job["state"], job["error"] = "error", public_error(f"{type(exc).__name__}: {exc}")[:300]
        log_failure("measurement job")
    finally:
        job["current"] = None


def champion():
    """(name, genome, record, judged): the network this page serves.

    With ESP_SERVE=heldout, the held-out winner and its stage 8 figures, and no
    record: it was never measured on the built-in seventeen, so it has no
    fitness to compare with theirs. Otherwise `fitness_champion()`.
    """
    if SERVE == "heldout":
        return paid.WINNER_NAME, paid.winner(), None, paid.judged()
    return (*fitness_champion(), None)


def fitness_champion():
    """The measured best, or the designer-shaped seed when nothing is measured.

    Sources: this deployment's own `state/champion.json`, its service
    population, and the evaluation cache committed with the repository; the
    designer-shaped seed when there is none. Falling back rather than failing is deliberate: a
    fresh deployment with no state should still answer, and the page says which
    case it is in.

    The committed cache was missing from that list, and the omission was not
    cosmetic. A fresh clone has no state, so every deployment anybody has ever
    started from one served `designer_shaped` at +0.7761 -- the *worst* of the
    eleven measured networks -- under a page headed "measured champion", while
    the network that actually won at +0.8453 sat in the repository with its
    genome beside its score.

    Every source is scored on the same benchmark, so the highest fitness among
    them wins rather than the first that exists. Taking the first served a
    committed manifest that predated the twelfth measurement: +0.8453 under
    "best-measured", beside a table listing +0.8941.
    """
    state_dir = Path(os.environ.get("ESP_STATE", ROOT / "state"))
    found = []
    manifest = state_dir / "champion.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        genome = Genome.from_canonical(payload["genome"])
        measured = payload["measured"]
        found.append((payload.get("origin", "measured"), genome, Evaluated(
            genome_hash=payload["hash"], origin=payload.get("origin", "measured"),
            fitness=measured["fitness"], accuracy=measured["accuracy"],
            tokens=measured["tokens"], agents=measured["agents"],
            depth=len(genome.reachable()), generation=0, measured_at="",
            model=genome.default_model, genome=payload["genome"])))

    best = ServiceState.load(state_dir).best()
    if best is not None:
        for name, build in SEEDS.items():
            genome = build()
            if genome.genome_hash() == best.genome_hash:
                found.append((name, genome, best))

    committed = measurements.best()
    if committed is not None:
        found.append((committed.name(), committed.genome, Evaluated(
            genome_hash=committed.genome_hash, origin=committed.origin,
            fitness=committed.fitness, accuracy=committed.accuracy,
            tokens=committed.tokens, agents=committed.agents,
            depth=committed.depth, generation=0, measured_at="",
            model=committed.genome.default_model,
            genome=committed.genome.canonical())))

    if found:
        return max(found, key=lambda entry: entry[2].fitness)
    return "designer_shaped", SEEDS["designer_shaped"](), None


NAME, GENOME, RECORD, JUDGED = champion()
CONTROL = paid.control() if JUDGED is not None else None
SERVED = presentable(GENOME)
HOCON = str(write_network(SERVED.genome))
# Every agent on the default model, for when a promoted model is busy or out of
# quota. Rendered once, at startup, like the network it stands in for.
_FALLBACK = demoted(SERVED.genome)
FALLBACK_HOCON = str(write_network(_FALLBACK)) if _FALLBACK else None

# Declared at module scope, and that is not a style choice. This file uses
# `from __future__ import annotations`, so every annotation is a string that
# FastAPI resolves against module globals -- a request model defined inside
# build_app() is invisible there, and FastAPI silently reclassifies the
# parameter as a query field. The symptom is a 422 saying the body field is a
# missing query parameter, which points nowhere near the cause.
# Bounded at the schema, so an oversized body is refused (422) before anything
# reads it, instead of being parsed whole and then cut down.
class Question(BaseModel):
    # Refused above the length /ask would send, rather than accepted and cut:
    # questions of 1,001 to 4,000 characters were silently shortened.
    question: str = Field(max_length=MAX_QUESTION_CHARS)


class MeasureRequest(BaseModel):
    networks: list[str] = Field(max_length=16)
    suite: str | None = Field(default=None, max_length=MAX_SUITE_BYTES)   # JSON Lines


# The page lives beside this file as plain HTML, so the markup can be read and
# linted as markup. Placeholders in __CAPS__ are filled by page(); every value a
# visitor supplies reaches the DOM through textContent, never innerHTML.
PAGE = (Path(__file__).with_name("page.html")).read_text(encoding="utf-8")


def measurement_count() -> int:
    """How many real evaluations this deployment can see.

    Counted through the same reader the champion comes from, so the page cannot
    report a population it did not actually resolve -- a cache file whose
    genome will not rebuild is not a measurement this deployment can see.
    """
    return len(measurements.load())


def surrogate_quality() -> dict | None:
    """The most recent cross-validated quality report, if a run produced one.

    Read from the run history rather than recomputed, so the page reports the
    predictor that was actually used rather than one fitted here for display.
    """
    history = ROOT / "results" / "history.json"
    if not history.exists():
        return None
    try:
        entries = json.loads(history.read_text(encoding="utf-8")).get(
            "surrogate_quality") or []
    except (json.JSONDecodeError, OSError):
        return None
    return entries[-1] if entries else None


def caveat() -> str:
    """What this deployment has *not* shown, derived rather than written down.

    The previous version was prose fixed in the template, and it went stale the
    moment an evolved candidate won: the page claimed no mutant had beaten the
    baselines directly beneath a header naming one as the best-measured
    topology. A caveat that contradicts the page above it is worse than none.
    """
    if JUDGED is not None:
        # The control was a sentence saying it had not been run, and it went
        # stale the day the replication ran it. Read from its summary instead.
        model = ("Its Researcher runs a dearer model than any agent in the designer's "
                 "shape. ")
        if CONTROL is None:
            model += ("The control that promotes the designer's specialists is in "
                      "results/replication/, which this deployment does not carry. ")
        else:
            model += (f"On {CONTROL.questions} fresh questions, promoting every "
                      f"specialist bought the designer's shape {100 * CONTROL.gain:.1f} "
                      f"points of accuracy, and "
                      f"{NAME} still beat that control on {CONTROL.winner_better_on} "
                      f"questions against {CONTROL.control_better_on}, which is not "
                      f"significant on so few (p = {CONTROL.sign_test_p}). ")
        return (model + f"It costs {JUDGED.cost_change:.0%} more per question than "
                "the designer's shape, and it is one measured mutation of a "
                "hand-written seed, judged in one task domain, not a network the "
                "Predictor chose. Stated in the repository rather than left out.")
    parts: list[str] = []
    evolved = RECORD is not None and RECORD.origin.startswith("mut:")
    if not evolved:
        parts.append("no evolved candidate has beaten the hand-designed "
                     "baselines yet")

    measured = measurement_count()
    if measured < MIN_SAMPLES:
        parts.append(f"the surrogate has not trained &mdash; it needs "
                     f"{MIN_SAMPLES} measurements and there are {measured}, so "
                     f"the search is still exploring rather than ranking")
    else:
        # Having enough samples to train is not the same as training usefully.
        # Reporting only the count would let the page imply a working predictor
        # on the strength of a threshold it merely cleared.
        quality = surrogate_quality()
        if quality is not None and not quality.get("beats_random", False):
            parts.append(
                f"when the search first used the surrogate it had "
                f"{quality.get('samples', measured)} samples and ranked worse "
                f"than chance (Spearman {quality.get('spearman', float('nan')):+.3f}); "
                f"trained on nine of the twelve committed networks it picks the "
                f"best of three unseen ones 62% of the time against 33% for chance "
                f"(make ablation), but twelve networks is too few to call it a "
                f"reliable selector")

    if not parts:
        parts.append("the search has run few generations, so the Pareto front "
                     "is thin")
    body = "; ".join(parts)
    return (body[0].upper() + body[1:] +
            ". Stated in the repository rather than left out.")


def page() -> str:
    if JUDGED is not None:
        sub = (f"<b>{NAME}</b> &mdash; the network that won on questions it was never "
               f"chosen on: {JUDGED.accuracy:.1%} correct against "
               f"{JUDGED.designer_accuracy:.1%} for the shape neuro-san's designer "
               f"builds, over {JUDGED.question_runs} question-runs each, with "
               f"{-JUDGED.token_change:.0%} fewer tokens a question "
               f"(sign test p = {JUDGED.sign_test_p}).")
    elif RECORD is not None:
        sub = (f"<b>{NAME}</b> &mdash; the best-measured topology: "
               f"{RECORD.accuracy:.0%} correct on {len(TASKS)} multi-hop questions, "
               f"{RECORD.tokens:,} tokens, {RECORD.agents} agents. "
               "Chosen by measurement, not by guessing.")
    else:
        sub = (f"<b>{NAME}</b> &mdash; the shape neuro-san's own designer produces. "
               "No measurements are loaded in this deployment, so this is the "
               "baseline rather than a measured winner.")
    labels = ("Direct lookup", "Two documents", "Deepest chain", "Hop, then calculate")
    examples = "".join(
        f'<button data-q="{html.escape(display_question(task), quote=True)}" '
        f'onclick="pick(this)"><b>{label} &middot; {task.hops} hop'
        f'{"s" if task.hops != 1 else ""}</b>{html.escape(display_question(task))}</button>'
        for label, task in zip(labels, SHOWCASE, strict=False))
    everything = "".join(
        f'<button data-q="{html.escape(display_question(task), quote=True)}" '
        f'onclick="pick(this)"><b>{task.task_id} &middot; {task.hops} hop'
        f'{"s" if task.hops != 1 else ""}</b>{html.escape(display_question(task))}</button>'
        for task in sorted(TASKS, key=lambda t: (t.hops, t.task_id)))
    served = html.escape(SERVED.note(GENOME.default_model))
    return (PAGE.replace("__SUB__", sub)
                .replace("__SERVED__", served)
                .replace("__EXAMPLES__", examples)
                .replace("__ALLQUESTIONS__", everything)
                .replace("__COUNT__", str(len(TASKS)))
                .replace("__MAXNETS__", str(MAX_NETWORKS))
                .replace("__CAVEAT__", caveat()))


# What a person can do about each way a provider refuses, keyed on the words the
# three providers use for it. Checked in order: an expired key can also be
# reported alongside a 404 for the model it was trying to reach.
_FAILURE_HINTS = (
    (("UNAUTHENTICATED", "API key not valid", "PERMISSION_DENIED",
      "authentication_error", "Incorrect API key", "invalid x-api-key"),
     "The provider rejected the key. Fix it in .env, run `make check-key`, "
     "then restart this server -- the key is read once, at startup."),
    # neuro-san's own wording for a rejected key or an account out of credit.
    (("environment variable must be correctly set", "credit balance", "Insufficient credits"),
     "The provider refused the key, or the account is out of credit. Check both "
     "in the provider's console, run `make check-key`, then restart this server."),
    (("RESOURCE_EXHAUSTED", "429", "rate_limit", "quota"),
     "The key's rate or daily limit is used up. Wait and ask again; on a free "
     "Gemini key, GOOGLE_API_KEYS spreads the day over several projects."),
    (("NOT_FOUND", "not_found_error", "model_not_found", "does not exist"),
     "The key cannot use a model this network is set to. `make probe` lists "
     "the models it can use; ESP_DEFAULT_MODEL picks one."),
    (("Agent timed out", "max_execution_seconds"),
     "An agent ran out of time, usually while queueing in the rate limiter. "
     "ESP_MAX_EXECUTION_SECONDS raises the budget."),
)


def failure_hint(reply: str) -> str:
    """What to do about an agent failure, ahead of the provider's own words."""
    for markers, hint in _FAILURE_HINTS:
        if any(marker in reply for marker in markers):
            return hint
    return "An agent failed before the network could answer."


def _charge_retry() -> bool:
    """Count the fallback retry against the day's cap, if the cap allows one.

    The retry is a second paid run of the whole network. Uncounted, a page
    whose router was busy all day spent up to twice ESP_WEB_MAX_QUESTIONS.
    """
    with _budget_lock:
        _roll_day()
        if _asked["count"] >= MAX_QUESTIONS:
            return False
        _asked["count"] += 1
        _save_spend()
        return True


def build_app():
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import HTMLResponse, JSONResponse

    # No /docs, /redoc or /openapi.json: they described every paid endpoint to
    # anyone, and /openapi.json answered 500.
    app = FastAPI(title="neuro-san-esp champion", docs_url=None, redoc_url=None,
                  openapi_url=None)

    @app.middleware("http")
    async def guard(request, call_next):
        """Refuse an unbounded or oversized body before it is read, and keep the
        page out of other sites' frames. Each request gets an id, returned in
        X-Request-ID and on its log line, so a visitor's report can be found."""
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex[:12]
        started = time.monotonic()
        response = await _guarded(request, call_next)
        response.headers["X-Request-ID"] = request_id
        logger.info("%s %s %s %.0fms id=%s", request.method, request.url.path,
                    response.status_code, (time.monotonic() - started) * 1000, request_id)
        return response

    async def _guarded(request, call_next):
        """The checks themselves, apart from the logging around them."""
        if request.method == "POST" and not authorized(request.headers.get("authorization", "")):
            return JSONResponse({"error": "this deployment needs an access token"},
                                status_code=401)
        if request.method == "POST":
            length = request.headers.get("content-length", "")
            if not length.isdigit():
                return JSONResponse({"error": "a request body must declare its length"},
                                    status_code=411)
            if int(length) > MAX_BODY_BYTES:
                return JSONResponse({"error": "the request is too large"}, status_code=413)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = f"frame-ancestors {FRAME_ANCESTORS}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid(_request, exc):
        # FastAPI's own 422 echoes every input back; name the fields instead.
        fields = sorted({".".join(str(part) for part in error.get("loc", ())[1:]) or "body"
                         for error in exc.errors()})
        return JSONResponse({"error": f"invalid request: {', '.join(fields)}"}, status_code=422)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "champion": NAME,
                "measured": RECORD is not None or JUDGED is not None,
                "judged_held_out": JUDGED is not None,
                "asked": _asked["count"], "cap": MAX_QUESTIONS}

    @app.post("/ask")
    def ask(payload: Question, request: Request = None) -> JSONResponse:
        question = payload.question.strip()[:MAX_QUESTION_CHARS]
        if not question:
            return JSONResponse({"error": "empty question"}, status_code=400)

        # Checked and counted in one step. Handlers run on a thread pool, and
        # a read-then-increment without the lock let two questions through on
        # the last unit of budget.
        with _budget_lock:
            _roll_day()
            if _asked["count"] >= MAX_QUESTIONS:
                return JSONResponse(
                    {"error": f"this deployment has answered its limit of "
                              f"{MAX_QUESTIONS} questions for today -- every answer "
                              f"is paid for, so a public page stops rather than "
                              f"running up the key's bill. ESP_WEB_MAX_QUESTIONS "
                              f"raises it."},
                    status_code=429)
            if _over_client_limit(client_key(request)):
                return JSONResponse(
                    {"error": f"you have asked {PER_CLIENT_HOURLY} questions in the "
                              f"last hour, this page's share per visitor. Please "
                              f"try again later."},
                    status_code=429)
            _asked["count"] += 1
            _save_spend()

        started = time.monotonic()
        try:
            answer, _, seconds = _ask(HOCON, question)
        except Exception as exc:      # a provider 429 must not 500 the page
            log_failure("/ask")
            return JSONResponse(
                {"error": public_error(f"{type(exc).__name__}: {exc}")[:300]}, status_code=502)
        # A promoted router on a model that is busy or out of its daily quota
        # (gemini-3.5-flash allows about twenty requests a day on a free key)
        # left the page erroring on most questions. One retry with every agent
        # on the workers' model, labelled, rather than an error.
        fallback = None
        router = (SERVED.genome.agents[SERVED.genome.top].model
                  or SERVED.genome.default_model)
        if (never_answered(answer) and unavailable(answer) and FALLBACK_HOCON
                and _charge_retry()):
            try:
                retry, _, retry_seconds = _ask(FALLBACK_HOCON, question)
            except Exception:           # the original failure is the one to report
                retry = ""
            if retry and not never_answered(retry):
                answer, seconds = retry, (seconds or 0) + (retry_seconds or 0)
                router = SERVED.genome.default_model
                fallback = (f"The measured stronger model was unavailable, so this "
                            f"answer ran with every agent on "
                            f"{SERVED.genome.default_model}. That variant has not "
                            f"been measured.")

        # neuro-san hands an agent's failure back as the reply text. Shown as an
        # answer, a key the provider refused read as the network talking
        # nonsense, under a 200 -- nothing on the page said to go and fix .env.
        if never_answered(answer):
            return JSONResponse(
                {"error": f"{failure_hint(answer)} The provider said: "
                          f"{public_error(answer)[:500]}"},
                status_code=502)

        # If it is one of the benchmark questions, grade it in front of the
        # visitor. Claiming correctness without showing the expected answer
        # would be the same as not claiming it. Matched as displayed, because
        # the page shows benchmark questions without their "number only"
        # suffix -- the network is now asked to explain, not to be terse.
        task = graded(question)
        expected = task.answer if task else None
        return JSONResponse({
            # Never truncated. The old 2,000-character cut existed to protect a
            # one-word benchmark answer from nothing, and it clipped exactly the
            # explanation a person came to read.
            "answer": answer,
            "agents": len(SERVED.genome.reachable()),
            "seconds": round(seconds or (time.monotonic() - started), 1),
            "provider": SERVED.provider,
            "router_model": router,
            "worker_model": SERVED.genome.default_model,
            # Every agent's model, because the promoted agent is not always the
            # router: the held-out winner promotes its Researcher.
            "models": {name: SERVED.genome.default_model if fallback else
                       (agent.model or SERVED.genome.default_model)
                       for name, agent in sorted(SERVED.genome.agents.items())
                       if name in SERVED.genome.reachable()},
            "fallback": fallback,
            "expected": expected,
            "correct": score(expected, answer) if expected else None,
        })

    @app.get("/api/networks")
    def networks() -> list[dict]:
        return [c.public() for c in candidates().values()]

    @app.post("/api/measure")
    def start_measure(payload: MeasureRequest) -> JSONResponse:
        pool = candidates()
        wanted = list(dict.fromkeys(payload.networks))
        if not wanted:
            return JSONResponse({"error": "pick at least one network"}, status_code=400)
        if len(wanted) > MAX_NETWORKS:
            return JSONResponse({"error": f"at most {MAX_NETWORKS} networks at once"},
                                status_code=400)
        unknown = [n for n in wanted if n not in pool]
        if unknown:
            return JSONResponse({"error": f"unknown network(s): {', '.join(unknown)}"},
                                status_code=400)
        if payload.suite is None:
            suite, tasks = BUILTIN, list(TASKS)
        else:
            if len(payload.suite.encode("utf-8")) > MAX_SUITE_BYTES:
                return JSONResponse({"error": "the question file is too large"},
                                    status_code=400)
            try:
                suite, tasks = "your questions", parse_suite(payload.suite)
            except SuiteError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            if len(tasks) > MAX_SUITE_QUESTIONS:
                return JSONResponse(
                    {"error": f"at most {MAX_SUITE_QUESTIONS} questions per measurement"},
                    status_code=400)
            if any(len(task.question) > MAX_QUESTION_CHARS for task in tasks):
                return JSONResponse(
                    {"error": f"a question is longer than {MAX_QUESTION_CHARS:,} characters"},
                    status_code=400)

        runs = len(wanted) * len(tasks)
        with _job_lock, _budget_lock:
            _roll_day()
            _prune_jobs()
            if any(j["state"] == "running" for j in _jobs.values()):
                return JSONResponse({"error": "a measurement is already running on this "
                                              "deployment -- try again when it finishes"},
                                    status_code=409)
            if _measured["count"] + runs > MAX_MEASURE:
                left = max(0, MAX_MEASURE - _measured["count"])
                return JSONResponse(
                    {"error": f"that is {runs} question-runs and this deployment has {left} "
                              f"left of its {MAX_MEASURE}. Every run is paid for; "
                              "ESP_WEB_MAX_MEASURE raises the limit."}, status_code=429)
            _measured["count"] += runs
            _save_spend()
            job_id = uuid.uuid4().hex[:12]
            job = {"id": job_id, "state": "running", "done": 0, "total": runs,
                   "suite": suite, "current": None, "reports": [], "error": None}
            _jobs[job_id] = job
        threading.Thread(target=run_job, args=(job, [pool[n] for n in wanted], tasks, suite),
                         daemon=True).start()
        return JSONResponse({"job": job_id, "runs": runs})

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        job = _jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "no such measurement"}, status_code=404)
        # Reports carry every network's replies and every provider error as they
        # came back, so a key a provider echoed would reach whoever holds the
        # link. Masked here, on the one way out.
        return JSONResponse(json.loads(redact(json.dumps(job))))

    return app


def serve_at_judged_width() -> int:
    """Set the search width the served network was judged at, and return it.

    The winner was judged with ten documents a search, and at three it is a
    different system. An explicit ESP_SEARCH_RESULTS still wins. Set on the
    module, which the search reads each time it runs (esp/eval/corpus_tool.py),
    and only when the server starts: importing this file must not change the
    width for anything else in the process, the test suite included.
    """
    if JUDGED is not None and "ESP_SEARCH_RESULTS" not in os.environ:
        corpus_tool.MAX_RESULTS = paid.WINNER_SEARCH_RESULTS
    return corpus_tool.MAX_RESULTS


def main() -> int:
    # The key this deployment needs is the one for the provider it serves on,
    # which is the configured one -- not the one the champion was measured on.
    # This used to demand GOOGLE_API_KEY unconditionally, so a deployment
    # holding only a Claude key refused to start at all.
    wanted = key_name_for(SERVED.genome.default_model)
    problem = key_problem(os.environ.get(wanted or "", ""), wanted or "")
    if wanted is None or problem:
        print(f"{wanted or 'no provider key'} is {problem or 'unknown'} -- the "
              "page would load and every question would fail. Put the key in "
              ".env; see .env.example.", file=sys.stderr)
        return 1
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    serve_at_judged_width()
    port = int(os.environ.get("PORT", "7860"))
    print(f"champion={NAME} measured={RECORD is not None or JUDGED is not None} "
          f"search_results={corpus_tool.MAX_RESULTS} "
          f"provider={SERVED.provider} retargeted={SERVED.retargeted} port={port}")
    # Never let uvicorn rewrite the client address from X-Forwarded-For: it
    # trusts the header from 127.0.0.1 by default, so behind a local proxy every
    # visitor chose their own address and their own hourly share. client_key
    # reads the header itself, and only when ESP_WEB_TRUST_PROXY says to.
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info",
                proxy_headers=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
