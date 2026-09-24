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

from pydantic import BaseModel  # noqa: E402

from esp.config import bootstrap  # noqa: E402

# Load .env BEFORE anything downstream reads env at import time. The rate
# limiter builds its keyring at module load from GOOGLE_API_KEYS; leaving
# bootstrap for main() froze that keyring empty for anyone who only set
# their keys in .env, so the first per-day 429 stopped the page instead
# of rotating.
bootstrap()

import html  # noqa: E402
import threading  # noqa: E402
import uuid  # noqa: E402

from esp.config import key_name_for, key_problem  # noqa: E402
from esp.eval import measurements  # noqa: E402
from esp.eval.runner import _ask, never_answered, write_network  # noqa: E402
from esp.eval.tasks import TASKS, score  # noqa: E402
from esp.genome.definition import Genome  # noqa: E402
from esp.genome.seeds import SEEDS  # noqa: E402
from esp.measure import BUILTIN, Candidate, SuiteError, catalog, pareto, parse_suite  # noqa: E402
from esp.measure import measure as measure_network  # noqa: E402
from esp.service.state import Evaluated, ServiceState  # noqa: E402
from esp.serving import SHOWCASE, display_question, graded, presentable  # noqa: E402
from esp.surrogate.predictor import MIN_SAMPLES  # noqa: E402

# A visitor is not a benchmark run, and every question is paid for: a multi-hop
# answer is ten or more model calls. A public deployment caps how many it will
# answer before it stops, so one careless loop cannot run up the key's bill.
MAX_QUESTIONS = int(os.environ.get("ESP_WEB_MAX_QUESTIONS", "40"))
_asked = {"count": 0}

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
# Networks an operator adds by placing HOCON files here. Never uploaded: a
# network names Python classes to import, so taking one from a visitor would be
# taking code.
NETWORKS_DIR = os.environ.get("ESP_NETWORKS") or None
_measured = {"count": 0}
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()
_candidates: dict[str, Candidate] | None = None


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
                                       "error": str(exc)[:300]})
            # A refused network still used up its share of the run, so progress
            # reaches the total whatever happened to it.
            job["done"] = before + len(tasks)
        pareto(job["reports"])
        job["state"] = "done"
    except Exception as exc:                       # reported to the page, never raised
        job["state"], job["error"] = "error", f"{type(exc).__name__}: {exc}"[:300]
    finally:
        job["current"] = None


def champion():
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


NAME, GENOME, RECORD = champion()
SERVED = presentable(GENOME)
HOCON = str(write_network(SERVED.genome))

# Declared at module scope, and that is not a style choice. This file uses
# `from __future__ import annotations`, so every annotation is a string that
# FastAPI resolves against module globals -- a request model defined inside
# build_app() is invisible there, and FastAPI silently reclassifies the
# parameter as a query field. The symptom is a 422 saying the body field is a
# missing query parameter, which points nowhere near the cause.
class Question(BaseModel):
    question: str


class MeasureRequest(BaseModel):
    networks: list[str]
    suite: str | None = None          # JSON Lines; None is the built-in benchmark


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
    if RECORD is not None:
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


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="neuro-san-esp champion")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "champion": NAME,
                "measured": RECORD is not None,
                "asked": _asked["count"], "cap": MAX_QUESTIONS}

    @app.post("/ask")
    def ask(payload: Question) -> JSONResponse:
        if _asked["count"] >= MAX_QUESTIONS:
            return JSONResponse(
                {"error": f"this deployment has answered its limit of "
                          f"{MAX_QUESTIONS} questions -- every answer is paid "
                          f"for, so a public page stops rather than running up "
                          f"the key's bill. ESP_WEB_MAX_QUESTIONS raises it."},
                status_code=429)
        question = payload.question.strip()[:1000]
        if not question:
            return JSONResponse({"error": "empty question"}, status_code=400)

        _asked["count"] += 1
        started = time.monotonic()
        try:
            answer, _, seconds = _ask(HOCON, question)
        except Exception as exc:      # a provider 429 must not 500 the page
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"[:300]}, status_code=502)
        # neuro-san hands an agent's failure back as the reply text. Shown as an
        # answer, a key the provider refused read as the network talking
        # nonsense, under a 200 -- nothing on the page said to go and fix .env.
        if never_answered(answer):
            return JSONResponse(
                {"error": f"{failure_hint(answer)} The provider said: {answer[:500]}"},
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
            "router_model": SERVED.genome.agents[SERVED.genome.top].model
                            or SERVED.genome.default_model,
            "worker_model": SERVED.genome.default_model,
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

        runs = len(wanted) * len(tasks)
        with _job_lock:
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
        return JSONResponse(job)

    return app


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
    port = int(os.environ.get("PORT", "7860"))
    print(f"champion={NAME} measured={RECORD is not None} "
          f"provider={SERVED.provider} retargeted={SERVED.retargeted} port={port}")
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
