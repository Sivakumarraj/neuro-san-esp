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
import random
import threading
import time
from collections import deque

from esp.eval import failover

DEFAULT_RPM = 14          # one under the documented 15, to leave headroom
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
    try:
        if current is not None and current.get_secret_value() == key:
            return
    except Exception:
        pass
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
            _stats["waits"] += 1
            _stats["wait_seconds"] += sleep_for
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
            _stats["waits"] += 1
            _stats["wait_seconds"] += sleep_for
            await asyncio.sleep(sleep_for)


def bucket_for(model: str, rpm: int = DEFAULT_RPM) -> Bucket:
    with _buckets_lock:
        if model not in _buckets:
            _buckets[model] = Bucket(rpm)
        return _buckets[model]


def stats() -> dict:
    return dict(_stats)


def _is_quota_error(exc: BaseException) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


# Transient faults on the provider's side: the model is busy or something broke
# in their stack, and the response says so. Distinct from a quota error in
# cause and identical in the only respect that matters here -- waiting fixes
# it.
_TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE",          # "currently experiencing high demand"
    "500", "INTERNAL",             # their side fell over
    "504", "DEADLINE_EXCEEDED",    # their gateway gave up, not our timeout
)


def _is_transient_server_error(exc: BaseException) -> bool:
    """Whether waiting and asking again is the right response.

    These were raised on the first attempt with no retry, because the retry
    condition below asked only whether the error was a quota error. The
    backoff loop was already sitting there; the one error class whose own
    message says "please try again later" was the class excluded from it.

    It costs measured accuracy, not just time. In the committed run the best
    network lost task T03 to a 500 INTERNAL and was recorded at 0.8824 when it
    had answered correctly everything it actually finished -- a whole task of
    seventeen, five accuracy points, thrown away on provider weather that a
    ten-second wait would have cleared.

    Matched on the message because that is all the provider hands back through
    langchain. Broad on purpose: a 503 misread as transient costs one retry,
    and a transient fault misread as an answer costs a task.
    """
    text = str(exc)
    if _is_quota_error(exc):
        return False        # quota is handled above this, and differently
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def install(rpm: int = DEFAULT_RPM) -> bool:
    """Patch the Google chat model. Idempotent; returns False if unavailable."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return False

    if getattr(ChatGoogleGenerativeAI, "_esp_rate_limited", False):
        return True

    def wrap(method_name: str, is_async: bool):
        original = getattr(ChatGoogleGenerativeAI, method_name, None)
        if original is None:
            return

        if is_async:
            async def limited(self, *args, **kwargs):
                for attempt in range(MAX_RETRIES):
                    # Resolved every attempt: another worker may have retired
                    # this model since the last one.
                    model = failover.substitute(getattr(self, "model", "unknown"))
                    _apply_model(self, model)
                    # Async all the way down: neither the pacing wait nor the
                    # backoff may block the event loop the agents run on.
                    await bucket_for(model, rpm).acquire_async()
                    try:
                        _stats["calls"] += 1
                        return await original(self, *args, **kwargs)
                    except Exception as exc:
                        # A per-day cap never clears within a run. Sleeping on it
                        # burns the clock and every candidate afterwards scores
                        # zero, so swap models instead of waiting.
                        if failover.is_daily_quota_error(exc):
                            current_key = getattr(self, "google_api_key", None)
                            current_key_str = ""
                            if current_key is not None:
                                with contextlib.suppress(Exception):
                                    current_key_str = current_key.get_secret_value()
                            mark_key_exhausted(current_key_str, model)
                            nxt_key = next_key_for(model, current_key_str)
                            if nxt_key is not None and nxt_key != current_key_str:
                                _apply_key(self, nxt_key)
                                _stats["key_swaps"] = _stats.get("key_swaps", 0) + 1
                                continue
                            nxt = failover.retire(model)
                            _stats["model_swaps"] += 1
                            if nxt is None:
                                raise
                            _apply_model(self, nxt)
                            continue
                        transient = _is_transient_server_error(exc)
                        if ((not _is_quota_error(exc) and not transient)
                                or attempt == MAX_RETRIES - 1):
                            raise
                        _stats["retries"] += 1
                        if transient:
                            _stats["transient_retries"] = _stats.get(
                                "transient_retries", 0) + 1
                        # Jittered backoff: synchronised retries from parallel
                        # workers would re-collide on the same second.
                        await asyncio.sleep(min(60.0, 2 ** attempt) + random.random())
                return None
        else:
            def limited(self, *args, **kwargs):
                for attempt in range(MAX_RETRIES):
                    model = failover.substitute(getattr(self, "model", "unknown"))
                    _apply_model(self, model)
                    bucket_for(model, rpm).acquire()
                    try:
                        _stats["calls"] += 1
                        return original(self, *args, **kwargs)
                    except Exception as exc:
                        if failover.is_daily_quota_error(exc):
                            current_key = getattr(self, "google_api_key", None)
                            current_key_str = ""
                            if current_key is not None:
                                with contextlib.suppress(Exception):
                                    current_key_str = current_key.get_secret_value()
                            mark_key_exhausted(current_key_str, model)
                            nxt_key = next_key_for(model, current_key_str)
                            if nxt_key is not None and nxt_key != current_key_str:
                                _apply_key(self, nxt_key)
                                _stats["key_swaps"] = _stats.get("key_swaps", 0) + 1
                                continue
                            nxt = failover.retire(model)
                            _stats["model_swaps"] += 1
                            if nxt is None:
                                raise
                            _apply_model(self, nxt)
                            continue
                        transient = _is_transient_server_error(exc)
                        if ((not _is_quota_error(exc) and not transient)
                                or attempt == MAX_RETRIES - 1):
                            raise
                        _stats["retries"] += 1
                        if transient:
                            _stats["transient_retries"] = _stats.get(
                                "transient_retries", 0) + 1
                        time.sleep(min(60.0, 2 ** attempt) + random.random())
                return None

        setattr(ChatGoogleGenerativeAI, method_name, limited)

    wrap("_generate", is_async=False)
    wrap("_agenerate", is_async=True)
    ChatGoogleGenerativeAI._esp_rate_limited = True
    return True


# The other providers this project can run on. Google keeps its own wrapper
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


def install_others(rpm: int = DEFAULT_RPM) -> list[str]:
    """Pace Anthropic and OpenAI. Returns the client classes actually patched.

    Absent drivers are skipped rather than raising: a run on Gemini should not
    need the Anthropic package installed to start.
    """
    patched: list[str] = []

    for module_name, class_name in _OTHER_PROVIDERS:
        try:
            module = importlib.import_module(module_name)
            client = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        if getattr(client, "_esp_rate_limited", False):
            patched.append(class_name)
            continue

        def wrap(target, method_name: str, is_async: bool):
            original = getattr(target, method_name, None)
            if original is None:
                return

            def model_of(self) -> str:
                # Clients disagree about the attribute; both are read rather
                # than assuming, because pacing the wrong bucket name silently
                # shares one rate limit between every model.
                for attribute in ("model", "model_name"):
                    value = getattr(self, attribute, None)
                    if value:
                        return str(value)
                return "unknown"

            if is_async:
                async def limited(self, *args, **kwargs):
                    for attempt in range(MAX_RETRIES):
                        await bucket_for(model_of(self), rpm).acquire_async()
                        try:
                            _stats["calls"] += 1
                            return await original(self, *args, **kwargs)
                        except Exception as exc:
                            transient = _is_transient_server_error(exc)
                            if ((not _is_quota_error(exc) and not transient)
                                    or attempt == MAX_RETRIES - 1):
                                raise
                            _stats["retries"] += 1
                            if transient:
                                _stats["transient_retries"] = _stats.get(
                                    "transient_retries", 0) + 1
                            await asyncio.sleep(
                                min(60.0, 2 ** attempt) + random.random())
                    return None
            else:
                def limited(self, *args, **kwargs):
                    for attempt in range(MAX_RETRIES):
                        bucket_for(model_of(self), rpm).acquire()
                        try:
                            _stats["calls"] += 1
                            return original(self, *args, **kwargs)
                        except Exception as exc:
                            transient = _is_transient_server_error(exc)
                            if ((not _is_quota_error(exc) and not transient)
                                    or attempt == MAX_RETRIES - 1):
                                raise
                            _stats["retries"] += 1
                            if transient:
                                _stats["transient_retries"] = _stats.get(
                                    "transient_retries", 0) + 1
                            time.sleep(min(60.0, 2 ** attempt) + random.random())
                    return None

            setattr(target, method_name, limited)

        wrap(client, "_generate", is_async=False)
        wrap(client, "_agenerate", is_async=True)
        client._esp_rate_limited = True
        patched.append(class_name)

    return patched
