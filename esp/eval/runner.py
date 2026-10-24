"""Run a candidate genome against the task set and measure it.

Evaluation is the bottleneck of the whole project, so three things matter here
more than elegance: no HTTP round trip, no repeated work, and no silent failure.
A candidate that crashes must score zero and keep the run alive -- an exception
that escapes kills a generation and costs an hour.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from esp.eval import accounting as usage
from esp.eval.ratelimit import install as install_rate_limit
from esp.eval.ratelimit import install_others as install_other_rate_limits
from esp.eval.tasks import TASKS, Task, grade
from esp.genome.definition import Genome

CACHE_DIR = Path(os.environ.get("ESP_CACHE", ".esp-cache"))
NETWORK_DIR = Path(os.environ.get("ESP_NETWORKS", ".esp-networks"))
MAX_WORKERS = int(os.environ.get("ESP_WORKERS", "4"))


@dataclass
class TaskResult:
    task_id: str
    hops: int
    correct: bool
    seconds: float
    answer: str
    error: str = ""
    # True when the network never got to answer: it timed out, hit neuro-san's
    # recursion cap, or raised. `correct=False` alone cannot say which, and the
    # difference is the difference between a wrong topology and a broken run.
    infrastructure: bool = False


@dataclass
class Evaluation:
    genome_hash: str
    accuracy: float
    seconds: float
    tokens: int
    cost: float
    agents: int
    depth: int
    results: list[TaskResult] = field(default_factory=list)
    from_cache: bool = False
    # Model calls (API requests) the run made, as neuro-san counts them. 0 in
    # every record written before it was kept, which is all of the committed ones.
    requests: int = 0

    def accuracy_by_hops(self) -> dict[int, float]:
        buckets: dict[int, list[bool]] = {}
        for result in self.results:
            buckets.setdefault(result.hops, []).append(result.correct)
        return {hops: sum(v) / len(v) for hops, v in sorted(buckets.items())}

    @property
    def incomplete(self) -> int:
        """Tasks the network never actually answered."""
        return sum(1 for r in self.results if r.infrastructure)

    def answered_accuracy(self) -> float | None:
        """Accuracy over the tasks that ran to an answer.

        `accuracy` counts a timeout as a wrong answer, which is one defensible
        convention -- a topology too slow to reply is no use. But it is not the
        same measurement, and reporting only the first is how three topologies
        came to look equally accurate when one of them had crashed out of three
        questions rather than getting them wrong. Both numbers, always.
        """
        attempted = [r for r in self.results if not r.infrastructure]
        if not attempted:
            return None
        return round(sum(r.correct for r in attempted) / len(attempted), 4)


_FACTORY = None
_FACTORY_LOCK = threading.Lock()


def _session_factory():
    """One factory per process. Building it reads neuro-san's manifest, which
    takes about eight seconds.

    The lock is the point. Tasks are evaluated on a thread pool, and an
    unsynchronised check-then-build lets every worker miss simultaneously and
    build its own -- four manifest reads, thirty wasted seconds, on every
    candidate. That was the observed behaviour before this lock existed.
    """
    global _FACTORY
    if _FACTORY is not None:
        return _FACTORY
    with _FACTORY_LOCK:
        # Re-checked inside the lock: another worker may have built it while
        # this one waited.
        if _FACTORY is None:
            from neuro_san.client.direct_agent_session_factory import (
                DirectAgentSessionFactory,
            )
            install_rate_limit()
            # Anthropic and OpenAI too, when their drivers are present. An
            # unpaced provider is not a smaller version of a paced one: a rate
            # limit comes back through neuro-san as an agent error, the
            # candidate scores zero, and the cache keeps that answer.
            install_other_rate_limits()
            _FACTORY = DirectAgentSessionFactory()
    return _FACTORY


def _ask(hocon_path: str, question: str) -> tuple[str, dict, float]:
    factory = _session_factory()
    session = factory.create_session(hocon_path)
    request = {
        "user_message": {"type": "HUMAN", "text": question},
        "sly_data": {},
        "chat_filter": {"chat_filter_type": "MAXIMAL"},
    }

    started = time.monotonic()
    answer = ""
    accounting: dict = {}
    for result in session.streaming_chat(request):
        message = result.get("response")
        if message is None:
            continue
        kind = str(message.get("type", ""))
        text = message.get("text") or ""
        # 101/AGENT_FRAMEWORK carries the final answer; 4/AI is the front man's
        # own reply and stands in when no framework message arrives.
        if kind in ("4", "AI", "101", "AGENT_FRAMEWORK") and text:
            answer = text
        # Token accounting rides on a front-man AGENT message with a structure.
        if kind in ("100", "AGENT"):
            structure = message.get("structure")
            origin = message.get("origin") or []
            if structure and len(origin) <= 1:
                accounting = structure
    return answer, accounting, time.monotonic() - started


def _sum_once(accounting: dict, keys: tuple[str, ...]) -> float:
    """Add up a usage figure, counting every model call once.

    neuro-san's front man reports a run's total at the top of the structure and
    the same usage again per model under "models". A node that carries its own
    total is the answer for everything beneath it, so nothing under it is
    added; only where no total sits are the parts summed. Summing both used to
    double every token and dollar figure; see esp/eval/accounting.py.
    """
    total = 0.0

    def walk(node) -> None:
        nonlocal total
        if isinstance(node, dict):
            here = [node[k] for k in keys if isinstance(node.get(k), (int, float))]
            if here:
                total += float(here[0])
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(accounting)
    return total


def _total_tokens(accounting: dict) -> int:
    """Tokens the run used, counted once."""
    return int(_sum_once(accounting, ("total_tokens",)))


def _requests(accounting: dict) -> int:
    """Model calls the run made: neuro-san's successful_requests, counted once."""
    return int(_sum_once(accounting, ("successful_requests",)))


def _cost(accounting: dict) -> float:
    """Dollars the provider accounting reports for the run, counted once."""
    return round(_sum_once(accounting, ("total_cost", "cost")), 6)



class QuotaExhausted(OSError):
    """The provider refused, so the score belongs to the environment.

    A distinct type rather than a bare OSError because the caller has to tell
    this apart from a genuine failure: this one means "stop the run and keep
    what you measured", not "this candidate is bad".
    """


# A rejected key or an account out of credit. neuro-san replaces either with one
# fixed message (its api_key_error_check.py) and hands it back as the reply, so a
# balance that ran out part-way through a run read as a wrong answer to every
# question after it, and the run was cached. The raw provider phrases cover the
# error path. Like a daily cap, it says nothing about the network.
_ACCOUNT_MARKERS = ("environment variable must be correctly set", "credit balance is too low",
                    "Insufficient credits", "Incorrect API key provided")
_QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", "429", "quota", "exceeded your current quota",
                  *_ACCOUNT_MARKERS)
# Markers no answer about the Meridian corpus can contain by accident.
_UNAMBIGUOUS_QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", "exceeded your current quota",
                              *_ACCOUNT_MARKERS)


def _is_quota_failure(result: TaskResult) -> bool:
    """Whether a task failed because the provider refused, not because the
    network was wrong. Timeouts count: when a daily cap is hit, calls queue
    behind a retry that never succeeds and the agent is cancelled.

    The loose markers are read only where the harness or neuro-san speaks: the
    error, and a reply that is an agent failure. Read in any reply they matched
    a network's own words -- incident INC-4429 is in the corpus, and three
    questions ask about it -- and refused a real measurement as a quota
    failure, which a resume then paid for again, every time."""
    if result.correct:
        return False
    if any(marker in result.error for marker in _QUOTA_MARKERS):
        return True
    markers = _QUOTA_MARKERS if never_answered(result.answer) else _UNAMBIGUOUS_QUOTA_MARKERS
    return any(marker in result.answer for marker in markers)


# Failures that are the harness giving up, not the network answering wrongly.
# neuro-san reports these as an ordinary answer string, so they reach `score()`
# and compare false against the expected answer -- silently indistinguishable
# from a network that looked and got it wrong.
_INFRASTRUCTURE_MARKERS = (
    "Agent timed out",
    "max_execution_seconds",
    "Recursion limit of",
    "Agent stopped due to exception",
    "environment variable must be correctly set",
)


def never_answered(reply: str) -> bool:
    """Whether a reply is neuro-san reporting an agent failure, not an answer."""
    return any(marker in reply for marker in _INFRASTRUCTURE_MARKERS)


def _is_infrastructure_failure(result: TaskResult) -> bool:
    """Whether this task never produced an answer.

    An exception that escaped `_ask` counts: it is recorded with an empty
    answer and a populated error, and scoring that as "wrong" would credit the
    task set with a measurement nobody made.
    """
    if result.correct:
        return False
    if result.error:
        return True
    return never_answered(result.answer)


def classify(results: list[TaskResult]) -> list[TaskResult]:
    """Mark the results that are not measurements of the topology.

    Applied on the way out of a fresh run and on the way in from the cache, so
    that records written before this existed are classified too rather than
    staying quietly clean.
    """
    for result in results:
        result.infrastructure = _is_infrastructure_failure(result)
    return results


@dataclass
class SuiteRun:
    """What one network did on one list of tasks, before any verdict."""

    results: list[TaskResult]
    tokens: int
    cost: float
    seconds: float
    requests: int = 0


def run_suite(hocon_path: str, tasks: list[Task], workers: int = MAX_WORKERS,
              answer_limit: int | None = 400,
              on_result: Callable[[TaskResult], None] | None = None) -> SuiteRun:
    """Put every task to one network, in parallel, and classify what came back.

    `hocon_path` is any neuro-san network: a file path, or a name in the
    manifest. `answer_limit` trims stored answers -- the cache keeps 400
    characters, a report keeps them whole. `on_result` is called as each task
    finishes, in completion order, so a caller can show progress; the returned
    results are always in task order.
    """
    def run_one(task: Task) -> tuple[TaskResult, dict]:
        try:
            answer, accounting, seconds = _ask(hocon_path, task.question)
            kept = answer if answer_limit is None else answer[:answer_limit]
            # neuro-san reports a blown cap as the reply, and "Recursion limit
            # of 40" contains the answer to every question whose answer is 40.
            # A reply that is an agent failure is never graded as an answer.
            correct = grade(task, answer) and not never_answered(answer)
            return TaskResult(task.task_id, task.hops, correct,
                              round(seconds, 2), kept), accounting
        except Exception as exc:
            return TaskResult(task.task_id, task.hops, False, 0.0, "",
                              f"{type(exc).__name__}: {exc}"[:300]), {}

    started = time.monotonic()
    pairs: list[tuple[TaskResult, dict] | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending = {pool.submit(run_one, task): index for index, task in enumerate(tasks)}
        for future in as_completed(pending):
            pair = future.result()
            pairs[pending[future]] = pair
            if on_result is not None:
                on_result(classify([pair[0]])[0])
    elapsed = time.monotonic() - started

    done = [pair for pair in pairs if pair is not None]
    return SuiteRun(
        results=classify([pair[0] for pair in done]),
        tokens=sum(_total_tokens(pair[1]) for pair in done),
        cost=round(sum(_cost(pair[1]) for pair in done), 6),
        seconds=round(elapsed, 2),
        requests=sum(_requests(pair[1]) for pair in done),
    )


def refuse_unmeasured(results: list[TaskResult], tokens: int,
                      action: str = "cache") -> None:
    """Raise when a run measured the environment instead of the network.

    Shared by the cache-writing evaluator and by `esp.measure`, because a
    number the environment produced is as wrong in a report as in the cache --
    it is only more permanent in the cache. `action` names what is being
    refused in the message: the evaluator caches, a measurement reports.
    """
    # A genome that scores zero because every task raised is not a bad genome,
    # it is a broken environment -- a missing key, an unset AGENT_TOOL_PATH.
    # Caching that would pin a permanent zero on a topology that was never
    # actually run, and the cache would keep returning it long after the
    # environment was fixed.
    if all(r.error for r in results):
        raise OSError(
            f"every task failed identically -- refusing to {action}. First error: "
            + results[0].error
        )

    # Nothing was measured. Every real evaluation burns tokens -- seventeen
    # tasks through a language model cannot cost zero -- so a zero total means
    # no model was ever called: a missing key, an unset AGENT_TOOL_PATH, a
    # provider the driver could not reach. The tasks still "completed", just
    # with whatever the agent says when it cannot work, so `accuracy` comes
    # back 0.00 with no error attached and the all-errored check above waves it
    # through.
    #
    # This is the single most expensive thing that can be written to the cache.
    # It is indistinguishable from a real zero once stored, it never expires,
    # and it poisons everything downstream: the seeds replay as 0.00 for ever,
    # the Predictor trains on them, and the search concludes that networks
    # which were never run are bad. A live run hit exactly this -- two seeds
    # cached at acc=0.00 tok=0, replayed on every invocation afterwards.
    if tokens == 0:
        raise OSError(
            "the whole evaluation cost zero tokens, so no model was called -- "
            f"refusing to {action}. Check the API key in .env"
            + (f", then delete {CACHE_DIR}/ if earlier runs already stored zeros."
               if action == "cache" else ".")
        )

    # Every task gave up before answering. Not one wrong answer among them --
    # seventeen timeouts, or seventeen blown recursion caps, which is what a
    # live run against neuro-san 0.7.4 produces when langgraph's limit of 40 is
    # reached on every question.
    #
    # None of the guards above sees this. neuro-san hands a blown cap back as
    # an ordinary answer string, so `error` is empty and the all-errored check
    # passes it; the agents burned real tokens on the way to giving up, so the
    # zero-token check passes it too; and no 429 was involved. What lands in
    # the cache is accuracy 0.00 with a plausible token count, filed against a
    # topology that was never actually asked anything it could finish.
    #
    # `classify()` has always marked these `infrastructure`, and
    # `answered_accuracy()` has always excluded them. Neither helps once the
    # record is written: the cached `accuracy` is what trains the Predictor and
    # what the champion resolver reads. A candidate with some unfinished tasks
    # is still a measurement and is still cached -- only the total wipeout is
    # refused, because that one is a fact about the environment.
    if results and all(r.infrastructure for r in results):
        raise OSError(
            f"all {len(results)} tasks gave up before answering "
            f"(timeout or recursion cap) -- refusing to {action}. This measures "
            f"the environment, not the topology. First: "
            f"{(results[0].answer or results[0].error)[:200]}"
        )

    # Quota exhaustion is not a property of the topology. A daily cap does not
    # fail every task at once -- it starts failing them part-way through a
    # candidate, so the all-errored check above does not catch it. What lands
    # instead is a plausible-looking partial score that gets cached forever and
    # tells the search a good network is mediocre. Refuse the whole evaluation
    # rather than record a number the environment produced.
    starved = [r for r in results if _is_quota_failure(r)]
    if starved:
        raise QuotaExhausted(
            f"{len(starved)} of {len(results)} tasks hit a provider quota or an "
            f"account failure (key or credit) -- "
            f"refusing to {action} a score the environment caused, not the topology. "
            f"First: {starved[0].error[:200]}"
        )



def write_network(genome: Genome) -> Path:
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    path = NETWORK_DIR / f"{genome.genome_hash()}.hocon"
    path.write_text(genome.to_hocon(), encoding="utf-8")
    return path


def evaluation_from_cache(raw: dict) -> Evaluation:
    """Rebuild an Evaluation from a cached payload.

    Shared rather than inlined because a cache entry carries more than the
    measurement: the genome is stored beside the score so the candidate can be
    replayed, and it is not a field of Evaluation. Every reader has to drop it
    the same way, and the ones that constructed Evaluation(**raw) by hand broke
    the moment it was added.
    """
    raw = usage.normalised(raw)
    raw.pop("accounting", None)
    raw.pop("settings", None)
    raw.pop("genome", None)
    raw.pop("from_cache", None)
    raw["results"] = classify([TaskResult(**r) for r in raw["results"]])
    return Evaluation(**raw)


def suite_key(tasks: list[Task] | None) -> str | None:
    """A short fingerprint of a question set, or None for the built-in 17.

    The cache used to be keyed by genome alone, so a network measured on any
    other questions was handed its old seventeen-question result from the
    cache -- a measurement of the wrong exam, returned as if it were this one.
    The built-in set keeps the flat layout every committed measurement and
    reader already use; any other set gets a directory of its own.
    """
    if tasks is None or list(tasks) == list(TASKS):
        return None
    import hashlib
    body = json.dumps([[t.task_id, t.question, list(t.accepted or (t.answer,))]
                       for t in tasks], sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def run_settings() -> dict[str, int]:
    """The settings a run used that the genome hash cannot see.

    How many documents a search returns and how many steps an agent may take
    are environment settings, not part of a network, yet they change what it
    can answer: a network measured at 3 documents was served from the cache to
    a run at 10. Read at call time, so a reloaded module is seen.
    """
    from esp.eval import corpus_tool
    from esp.genome import definition
    return {"search_results": corpus_tool.MAX_RESULTS, "max_steps": definition.MAX_STEPS,
            "max_execution_seconds": definition.MAX_EXECUTION_SECONDS}


# What every committed measurement ran with. These keep the flat layout; a run
# under any other value gets a directory of its own.
_KEYED_DEFAULTS = {"search_results": 3, "max_steps": 40}


def cache_path_for(genome: Genome, tasks: list[Task] | None = None) -> Path:
    key = suite_key(tasks)
    base = CACHE_DIR if key is None else CACHE_DIR / "suites" / key
    settings = run_settings()
    if any(settings[name] != value for name, value in _KEYED_DEFAULTS.items()):
        base = base / f"search{settings['search_results']}-steps{settings['max_steps']}"
    return base / f"{genome.genome_hash()}.json"


def evaluate(genome: Genome, tasks: list[Task] | None = None,
             use_cache: bool = True) -> Evaluation:
    tasks = tasks or TASKS
    digest = genome.genome_hash()
    cache_path = cache_path_for(genome, tasks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists():
        evaluation = evaluation_from_cache(json.loads(cache_path.read_text()))
        evaluation.from_cache = True
        return evaluation

    hocon_path = str(write_network(genome))
    run = run_suite(hocon_path, tasks)
    results = run.results

    evaluation = Evaluation(
        genome_hash=digest,
        accuracy=round(sum(r.correct for r in results) / len(results), 4),
        seconds=run.seconds,
        tokens=run.tokens,
        cost=run.cost,
        agents=len(genome.reachable()),
        depth=genome.depth(),
        results=results,
        requests=run.requests,
    )
    refuse_unmeasured(results, run.tokens)

    payload = asdict(evaluation)
    payload.pop("from_cache", None)
    # The genome, not just its hash. A score with no network attached can be
    # read back but never replayed, which bounded the offline search to the
    # seeds it could rebuild from source -- three points out of every eleven
    # measured. A hash cannot be inverted; the canonical form can.
    payload["genome"] = genome.canonical()
    payload["accounting"] = usage.ACCOUNTING
    payload["settings"] = run_settings()
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return evaluation
