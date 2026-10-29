"""A global rate limiter for the LLM calls neuro-san makes on our behalf.

The free tier allows 15 requests per minute per model. Evaluation runs many
candidates in parallel, and neuro-san owns the call sites, so pacing has to
happen underneath it: this patches the LangChain Google client so every
generate call takes a token from a per-model bucket first.

Without this the run does not merely go slower -- it fails wrongly. A 429 comes
back as an agent error, the candidate scores zero, and the search learns that a
perfectly good topology is bad. Rate limiting is therefore a correctness
requirement here, not a politeness one.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import random
import re
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass

from esp.eval import failover

# Requests per minute for any model without a measured limit of its own.
#
# 14 sits one under the 15 the Gemini free tier allows, and it is a safe floor
# on every provider. It is not a sensible ceiling on a paid key: at 14 a minute
# the ~165 calls one candidate makes take twelve minutes of pure queueing.
# `ESP_RPM` raises it -- set it to one under the requests-per-minute limit your
# provider console shows for your account.
DEFAULT_RPM = int(os.environ.get("ESP_RPM", "14"))
if DEFAULT_RPM < 1:
    # 0 or less made every paced call raise IndexError from an empty window.
    raise ValueError(f"ESP_RPM={DEFAULT_RPM}: the pace is at least one request a minute")
MAX_RETRIES = 6

_buckets: dict[str, Bucket] = {}
_buckets_lock = threading.Lock()
_stats = {"waits": 0, "wait_seconds": 0.0, "retries": 0, "calls": 0,
          "model_swaps": 0, "transient_retries": 0}


def _apply_model(client, model: str) -> None:
    """Point a live client at a different model.

    The client is constructed by neuro-san from the HOCON, so the only way to
    redirect a call already in flight is to set the field it reads. Pydantic
    models refuse unknown attributes, hence the guarded assignment.

    Also forces max_retries=0 on the way through. langchain-google-genai runs
    an internal tenacity retry loop on 429; on a per-day 429 that loop burns
    minutes on a quota that will not clear, and my keyring rotation only fires
    once it exhausts. Zeroing the internal retry surfaces the first 429 to my
    wrapper immediately -- rotation is instant, and per-minute pacing is still
    handled by the sliding-window bucket above.
    """
    if getattr(client, "max_retries", None) != 0:
        with contextlib.suppress(Exception):
            object.__setattr__(client, "max_retries", 0)
    if getattr(client, "model", None) == model:
        return
    with contextlib.suppress(Exception):
        object.__setattr__(client, "model", model)


def _parse_keyring() -> list[str]:
    """Read the multi-key env, falling back to the single-key one.

    Google's free tier is per-project-per-model. A second key from a second
    project doubles the daily budget for the same code path. `GOOGLE_API_KEYS=
    "k1,k2"` opts in; `GOOGLE_API_KEY=k1` alone is the old behaviour.
    """
    import os
    multi = os.environ.get("GOOGLE_API_KEYS", "").strip()
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("GOOGLE_API_KEY", "").strip()
    return [single] if single else []


_KEYRING: list[str] = _parse_keyring()
_key_exhausted: dict[tuple[str, str], bool] = {}


def keyring() -> list[str]:
    """The keys in use for this process. Public for tests and preflight."""
    return list(_KEYRING)


def reload_keyring() -> None:
    """Re-read the keyring env vars. Tests use this; production sets once."""
    global _KEYRING
    _KEYRING = _parse_keyring()
    _key_exhausted.clear()


def next_key_for(model: str, current: str) -> str | None:
    """Find the next unexhausted key for a model, starting after current.

    Returns None only when every key in the ring has been marked exhausted
    for this model -- the signal that the model itself should be retired.
    """
    if not _KEYRING:
        return None
    start = (_KEYRING.index(current) + 1) % len(_KEYRING) if current in _KEYRING else 0
    for offset in range(len(_KEYRING)):
        candidate = _KEYRING[(start + offset) % len(_KEYRING)]
        if not _key_exhausted.get((candidate, model), False):
            return candidate
    return None


def mark_key_exhausted(key: str, model: str) -> None:
    """Record that a (key, model) pair is spent for the day."""
    if key:
        _key_exhausted[(key, model)] = True


def _apply_key(client, key: str) -> None:
    """Point a live client at a different API key.

    Mirrors _apply_model. The Google client stores the key as SecretStr;
    object.__setattr__ steers the field the request uses without tripping
    Pydantic's frozen-model check.
    """
    if not key:
        return
    try:
        from pydantic.types import SecretStr
    except ImportError:  # pragma: no cover
        return
    current = getattr(client, "google_api_key", None)
    # Only a value that is not a SecretStr can fail here; anything else wrong
    # with the client should surface, not vanish.
    with contextlib.suppress(AttributeError, TypeError):
        if current is not None and current.get_secret_value() == key:
            return
    with contextlib.suppress(Exception):
        object.__setattr__(client, "google_api_key", SecretStr(key))




class Bucket:
    """Sliding-window limiter. Fair across threads: whoever waits longest goes
    next, because the alternative starves a worker for the length of a run."""

    def __init__(self, rpm: int = DEFAULT_RPM):
        self.rpm = rpm
        self.window: deque[float] = deque()
        self.lock = threading.Lock()

    def _try_take(self) -> float:
        """Take a token, or report how many seconds until one frees up.

        Returns 0.0 when a token was taken. The lock is deliberately not held
        across the wait, so a sleeping caller cannot stop the others re-checking.
        """
        with self.lock:
            now = time.monotonic()
            while self.window and now - self.window[0] >= 60.0:
                self.window.popleft()
            if len(self.window) < self.rpm:
                self.window.append(now)
                return 0.0
            return 60.0 - (now - self.window[0]) + 0.05

    def acquire(self) -> None:
        """Blocking acquire, for calls made on an ordinary thread."""
        while True:
            sleep_for = self._try_take()
            if sleep_for == 0.0:
                return
            _count("waits")
            _count("wait_seconds", sleep_for)
            time.sleep(sleep_for)

    async def acquire_async(self) -> None:
        """Acquire from inside a coroutine.

        The blocking version cannot be used here. neuro-san runs its agents on
        asyncio, so a `time.sleep` at an async call site stops the whole event
        loop rather than just its caller: every other agent in the process
        freezes for the length of the wait. They are all spending their own
        `max_execution_seconds` while frozen, so a queue of concurrent tasks
        times out together and scores zero -- which the search then reads as
        "this topology is bad", when the topology was never allowed to run.
        """
        while True:
            sleep_for = self._try_take()
            if sleep_for == 0.0:
                return
            _count("waits")
            _count("wait_seconds", sleep_for)
            await asyncio.sleep(sleep_for)


def quieten_fixed_sampling_warning() -> None:
    """Silence one library warning that fires once per model call.

    langchain-google-genai warns that a model "uses fixed sampling defaults;
    the sampling parameter(s) temperature will be ignored" whenever a
    temperature reaches a model that has none. This project never sets a
    temperature -- neuro-san's own default llm_config does -- so there is
    nothing here to fix, and the warning is accurate and harmless.

    It is also emitted on every single call. A live run against the newest
    Gemini models produced hundreds of copies, and they buried the recursion
    errors that actually needed reading. An operator who cannot see a real
    failure in their own terminal is the cost being paid for a notice about a
    parameter nobody chose.

    Narrow on purpose: this message, this category. Anything else the driver
    has to say still gets through.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*fixed sampling defaults.*",
        category=UserWarning,
    )


def bucket_for(model: str, rpm: int = DEFAULT_RPM) -> Bucket:
    """One bucket per model, paced at that model's own measured limit.

    `rpm` is the fallback for a model nobody has measured. It used to be the
    rate for *every* model, which is right for the lite tier and far too fast
    for the newest flash models -- 14 a minute against a measured 5. Selecting
    one of those then produced 429s the runner scored as candidate failures.
    """
    with _buckets_lock:
        if model not in _buckets:
            from esp.eval.failover import rpm_for
            _buckets[model] = Bucket(rpm_for(model, rpm))
        return _buckets[model]


def stats() -> dict:
    return dict(_stats)


_stats_lock = threading.Lock()


def _count(name: str, by: float = 1) -> None:
    """Add to one of the run's counters.

    Under a lock because the wrappers run on every worker thread at once, and
    `+=` on a shared dict entry is a read and a write that two threads can
    interleave, losing a count.
    """
    with _stats_lock:
        _stats[name] = _stats.get(name, 0) + by

# An HTTP status where a provider puts one in its message: at the start
# ("429 RESOURCE_EXHAUSTED", "ClientError: 503 UNAVAILABLE"), after "Error
# code:" (Anthropic and OpenAI), or as a JSON "code" field (Google). Anchored
# on purpose. Matched anywhere, "500" was found inside "65000" and a request
# that could never succeed was asked five more times.
_STATUS_IN_TEXT = re.compile(
    r"(?:^\s*(?:[\w.]+:\s*)?|error code:?\s*|status(?: code)?[:=]?\s*|['\"]code['\"]:\s*)"
    r"(\d{3})\b", re.IGNORECASE)

QUOTA_STATUS = 429
# Busy or broken on the provider's side; 529 is Anthropic's way of saying 503.
TRANSIENT_STATUSES = frozenset({500, 502, 503, 504, 529})

# Words that mean the same as a status when no status is given.
_QUOTA_WORDS = ("RESOURCE_EXHAUSTED", "ResourceExhausted", "Too Many Requests",
                "exceeded your current quota")
_TRANSIENT_WORDS = ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "overloaded")


def status_in(text: str) -> int | None:
    """The HTTP status a provider error message states, or None."""
    match = _STATUS_IN_TEXT.search(text)
    return int(match.group(1)) if match else None


def status_of(exc: BaseException) -> int | None:
    """The HTTP status a provider error carries, or None when it carries none.

    The SDKs put it on the exception -- `status_code` on OpenAI and Anthropic
    errors, `code` on google-genai's -- so that is read first, and the message
    only when the exception has no such field.
    """
    for attribute in ("status_code", "code", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    return status_in(str(exc))


def _is_quota_error(exc: BaseException) -> bool:
    if status_of(exc) == QUOTA_STATUS:
        return True
    text = str(exc)
    return any(word in text for word in _QUOTA_WORDS)


def _is_transient_server_error(exc: BaseException) -> bool:
    """Whether waiting and asking again is the right response.

    These were raised on the first attempt with no retry, because the retry
    condition asked only whether the error was a quota error. In the committed
    run the best network lost task T03 to a 500 INTERNAL and was recorded at
    0.8824 when it had answered correctly everything it actually finished.

    A stated status decides: a 400 is never transient, whatever words follow
    it. The words are read only when no status is given.
    """
    if _is_quota_error(exc):
        return False        # quota is handled separately, and differently
    status = status_of(exc)
    if status is not None:
        return status in TRANSIENT_STATUSES
    text = str(exc)
    return any(word in text for word in _TRANSIENT_WORDS)


def _out_of_attempts(last: Exception | None) -> Exception:
    """What a paced call raises when every attempt went on a key or model swap.

    A daily-cap 429 swaps the key or the model and goes round again, and that
    swap spends an attempt. With six or more keys in GOOGLE_API_KEYS, or keys
    and ladder rungs that add up to six, the last attempt ended on a swap, the
    loop ran out, and the call returned None. LangChain then failed on the
    None with an AttributeError that names no quota, so the runner's quota
    guard did not see it: the question was graded wrong and the run could be
    cached. The provider's own 429 is raised instead, so the guard sees it.
    """
    return last if last is not None else RuntimeError("no attempt was made")


@dataclass(frozen=True)
class Next:
    """What a paced call does after one failed attempt."""

    action: str                 # "raise", "again" (after a swap) or "wait"
    seconds: float = 0.0


def _backoff(exc: BaseException, attempt: int) -> Next:
    """Wait and ask again, or give up: the policy every provider shares."""
    transient = _is_transient_server_error(exc)
    if (not _is_quota_error(exc) and not transient) or attempt == MAX_RETRIES - 1:
        return Next("raise")
    _count("retries")
    if transient:
        _count("transient_retries")
    # Jittered: synchronised retries from parallel workers would re-collide on
    # the same second.
    return Next("wait", min(60.0, 2 ** attempt) + random.random())


def _secret(client) -> str:
    """The key a Google client is using, as a plain string."""
    current = getattr(client, "google_api_key", None)
    if current is None:
        return ""
    with contextlib.suppress(AttributeError, TypeError):
        return current.get_secret_value()
    return ""


def _google_next(client, exc: BaseException, attempt: int, model: str) -> Next:
    """The Google policy: a per-day cap swaps the key, then the model.

    A per-day cap never clears within a run. Sleeping on it burns the clock and
    every candidate afterwards scores zero, so the call moves to the next key
    in GOOGLE_API_KEYS, then down the model ladder, instead of waiting.
    Everything else is the shared backoff.
    """
    if not failover.is_daily_quota_error(exc):
        return _backoff(exc, attempt)
    key = _secret(client)
    mark_key_exhausted(key, model)
    nxt_key = next_key_for(model, key)
    if nxt_key is not None and nxt_key != key:
        _apply_key(client, nxt_key)
        _count("key_swaps")
        return Next("again")
    nxt = failover.retire(model)
    _count("model_swaps")
    if nxt is None:
        return Next("raise")
    _apply_model(client, nxt)
    return Next("again")


def paced(original, bucket_name, decide, is_async: bool, rpm: int, before=None):
    """Wrap one client method in pacing and the retry policy.

    One body for the sync and the async path, so a fix lands once. The four
    hand-written copies this replaced are how a missing `raise` survived in one
    of them. `bucket_name(client)` picks the rate bucket, paced at `rpm` when
    the model has no measured limit; `decide` is the retry policy; and
    `before(client)` runs at the start of every attempt (Google uses it to apply
    the current model from the failover ladder).
    """
    if is_async:
        async def limited(self, *args, **kwargs):
            last: Exception | None = None
            for attempt in range(MAX_RETRIES):
                if before is not None:
                    before(self)
                # Async all the way down: neither the pacing wait nor the
                # backoff may block the event loop the agents run on.
                await bucket_for(bucket_name(self), rpm).acquire_async()
                try:
                    _count("calls")
                    return await original(self, *args, **kwargs)
                except Exception as exc:
                    last = exc
                    step = decide(self, exc, attempt)
                    if step.action == "raise":
                        raise
                    if step.action == "wait":
                        await asyncio.sleep(step.seconds)
            raise _out_of_attempts(last)
    else:
        def limited(self, *args, **kwargs):
            last: Exception | None = None
            for attempt in range(MAX_RETRIES):
                if before is not None:
                    before(self)
                bucket_for(bucket_name(self), rpm).acquire()
                try:
                    _count("calls")
                    return original(self, *args, **kwargs)
                except Exception as exc:
                    last = exc
                    step = decide(self, exc, attempt)
                    if step.action == "raise":
                        raise
                    if step.action == "wait":
                        time.sleep(step.seconds)
            raise _out_of_attempts(last)
    return limited


def install(rpm: int = DEFAULT_RPM) -> bool:
    """Patch the Google chat model. Idempotent; returns False if unavailable."""
    quieten_fixed_sampling_warning()
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return False

    if getattr(ChatGoogleGenerativeAI, "_esp_rate_limited", False):
        return True

    def current_model(client) -> str:
        # Resolved every attempt: another worker may have retired this model
        # since the last one.
        return failover.substitute(getattr(client, "model", "unknown"))

    def before(client) -> None:
        _apply_model(client, current_model(client))

    def decide(client, exc: BaseException, attempt: int) -> Next:
        return _google_next(client, exc, attempt, current_model(client))

    for method_name, is_async in (("_generate", False), ("_agenerate", True)):
        original = getattr(ChatGoogleGenerativeAI, method_name, None)
        if original is None:
            continue
        setattr(ChatGoogleGenerativeAI, method_name,
                paced(original, current_model, decide, is_async, rpm, before))
    ChatGoogleGenerativeAI._esp_rate_limited = True
    return True


# The other providers this project can run on. Google keeps its own policy
# above because the daily-cap failover, the model ladder and the key ring are
# all free-tier Google concepts measured out of Google's own 429 payloads.
# Anthropic and OpenAI are paid APIs with per-minute limits and no daily cap to
# retire a model against, so they get the half that applies: pacing, backoff on
# a rate limit, and retry on a transient server fault.
#
# Without this they were unpaced and unretried. A 529 from Anthropic or a 503
# from OpenAI would come back through neuro-san as an agent error, the
# candidate would score zero, and the search would learn that a perfectly good
# topology is bad -- the exact failure already fixed once for Google.
_OTHER_PROVIDERS = (
    ("langchain_anthropic", "ChatAnthropic"),
    ("langchain_openai", "ChatOpenAI"),
)


def _model_of(client) -> str:
    """The model a paid-provider client calls.

    Clients disagree about the attribute; both are read rather than assuming,
    because pacing the wrong bucket name silently shares one rate limit
    between every model.
    """
    for attribute in ("model", "model_name"):
        value = getattr(client, attribute, None)
        if value:
            return str(value)
    return "unknown"


def install_others(rpm: int = DEFAULT_RPM) -> list[str]:
    """Pace Anthropic and OpenAI. Returns the client classes actually patched.

    Absent drivers are skipped rather than raising: a run on Gemini should not
    need the Anthropic package installed to start.
    """
    patched: list[str] = []

    def decide(_client, exc: BaseException, attempt: int) -> Next:
        return _backoff(exc, attempt)

    for module_name, class_name in _OTHER_PROVIDERS:
        try:
            module = importlib.import_module(module_name)
            client = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        if getattr(client, "_esp_rate_limited", False):
            patched.append(class_name)
            continue
        for method_name, is_async in (("_generate", False), ("_agenerate", True)):
            original = getattr(client, method_name, None)
            if original is None:
                continue
            setattr(client, method_name, paced(original, _model_of, decide, is_async, rpm))
        client._esp_rate_limited = True
        patched.append(class_name)

    return patched
