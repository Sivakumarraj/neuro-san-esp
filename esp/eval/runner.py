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

from esp.eval.ratelimit import install as install_rate_limit
from esp.eval.ratelimit import install_others as install_other_rate_limits
from esp.eval.tasks import TASKS, Task, score
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


def _total_tokens(accounting: dict) -> int:
    """Token accounting nests per-agent totals; sum whatever integer token
    counts are present rather than guessing at one schema."""
    total = 0

    def walk(node) -> None:
        nonlocal total
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("total_tokens", "completion_tokens", "prompt_tokens") \
                        and isinstance(value, (int, float)):
                    if key == "total_tokens":
                        total += int(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(accounting)
    return total


def _cost(accounting: dict) -> float:
    total = 0.0

    def walk(node) -> None:
        nonlocal total
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("total_cost", "cost") and isinstance(value, (int, float)):
                    total += float(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(accounting)
    return round(total, 6)



class QuotaExhausted(OSError):
    """The provider refused, so the score belongs to the environment.

    A distinct type rather than a bare OSError because the caller has to tell
    this apart from a genuine failure: this one means "stop the run and keep
    what you measured", not "this candidate is bad".
    """


_QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", "429", "quota", "exceeded your current quota")


def _is_quota_failure(result: TaskResult) -> bool:
    """Whether a task failed because the provider refused, not because the
    network was wrong. Timeouts count: when a daily cap is hit, calls queue
    behind a retry that never succeeds and the agent is cancelled."""
    blob = f"{result.error} {result.answer}"
    return any(marker in blob for marker in _QUOTA_MARKERS)


# Failures that are the harness giving up, not the network answering wrongly.
# neuro-san reports these as an ordinary answer string, so they reach `score()`
# and compare false against the expected answer -- silently indistinguishable
# from a network that looked and got it wrong.
_INFRASTRUCTURE_MARKERS = (
    "Agent timed out",
    "max_execution_seconds",
    "Recursion limit of",
    "Agent stopped due to exception",
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
            return TaskResult(task.task_id, task.hops,
                              score(task.accepted or (task.answer,), answer),
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
            f"{len(starved)} of {len(results)} tasks hit a provider quota -- "
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
    raw = dict(raw)
    raw.pop("genome", None)
    raw.pop("from_cache", None)
    raw["results"] = classify([TaskResult(**r) for r in raw["results"]])
    return Evaluation(**raw)


def evaluate(genome: Genome, tasks: list[Task] | None = None,
             use_cache: bool = True) -> Evaluation:
    tasks = tasks or TASKS
    digest = genome.genome_hash()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{digest}.json"

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
    )
    refuse_unmeasured(results, run.tokens)

    payload = asdict(evaluation)
    payload.pop("from_cache", None)
    # The genome, not just its hash. A score with no network attached can be
    # read back but never replayed, which bounded the offline search to the
    # seeds it could rebuild from source -- three points out of every eleven
    # measured. A hash cannot be inverted; the canonical form can.
    payload["genome"] = genome.canonical()
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return evaluation
