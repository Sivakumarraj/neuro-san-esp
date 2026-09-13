"""Load a local .env, so a key can be pasted into a file instead of a shell.

Written here rather than pulled in as a dependency. `python-dotenv` is present
in this environment, but only as a transitive dependency of something else --
and relying on a package nobody declared is the exact bug that broke CI when
`reportlab` turned out to be undeclared. Fifteen lines are cheaper than either
a new dependency or a repeat of that.

The environment always wins over the file. A value exported in the shell, set by
a container, or injected by a secrets manager is the more deliberate one, and a
stale .env silently overriding it would be very hard to debug.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Either provider satisfies the preflight. neuro-san ships policies for both
# and picks the class from the model name, so the only thing that has to
# agree is that the key for the model being used is present.
KEY_NAMES = ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
             "OPENROUTER_API_KEY")

# Which provider a model name belongs to, and which key that provider needs.
#
# neuro-san picks the client class from the model name and ships classes for
# all of these, so switching provider is a model name and a key -- no code
# path here changes. What does change is every genome hash, because the model
# is part of the genome: a fitness measured on Gemini does not describe the
# same network on Claude, the cache must miss, and the measurements start
# again. That is deliberate, and it is why this map exists rather than a
# single global "provider" setting.
#
# Owned here rather than read from neuro-san's own registry at runtime: its
# HOCON keys are quoted and dotted in ways pyhocon resolves inconsistently
# across versions, and a provider misread at startup is a run that fails on
# every call. `tests/test_providers.py` checks this map against that registry,
# so drift is caught where it is cheap.
PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("openrouter/", "openrouter"),
    ("gemini-", "gemini"),
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
)

KEY_FOR_PROVIDER = {
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def provider_for(model: str) -> str | None:
    """Which provider serves this model name, or None if nothing claims it."""
    name = (model or "").strip().lower()
    for prefix, provider in PROVIDER_PREFIXES:
        if name.startswith(prefix):
            return provider
    return None


def key_name_for(model: str) -> str | None:
    """The environment variable a run on this model needs."""
    provider = provider_for(model)
    return KEY_FOR_PROVIDER.get(provider) if provider else None


def load_env(path: Path | None = None) -> list[str]:
    """Read KEY=value lines into the environment. Returns the names it set."""
    path = path or ROOT / ".env"
    if not path.exists():
        return []

    applied: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        # Quotes are what a person types when a value has spaces in it; they are
        # not part of the value. An API key carrying a stray quote fails with an
        # authentication error that says nothing about quoting.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value
            applied.append(name)
    return applied


# What .env.example ships on the key line. Someone who copies the example and
# pastes badly leaves this behind, and it is a perfectly good non-empty string:
# `bool(value)` is true, the preflight passes, and the provider answers
# "API key not valid" on the first real call -- eight minutes into a candidate,
# or in front of somebody being shown the UI.
PLACEHOLDER = "paste-your-key-here"


# What each provider's keys look like. Checked only as a hint, because a
# provider can change its format and a shape check that refuses a live key is
# worse than one that lets a dead key through -- `verify_key` catches the dead
# one, and nothing catches a false refusal except a confused person.
_KEY_PREFIXES = {
    "ANTHROPIC_API_KEY": "sk-ant-",
    "OPENROUTER_API_KEY": "sk-or-",
}


def key_problem(value: str, name: str = "") -> str:
    """Why this string cannot be a usable API key, or "" if it looks like one.

    A shape check, not an authentication check -- only the provider can say
    whether a well-formed key is live, and `verify_key` asks it. What this
    catches is the class of mistake that makes a key obviously wrong before any
    call is made, because the preflight's whole job is to refuse to start on a
    configuration that would produce wrong numbers.

    The report is about the string, never the string itself: a message that
    echoed the value would put a live credential into terminal scrollback,
    screen shares and pasted logs.
    """
    if not value:
        return "unset"
    if PLACEHOLDER in value:
        if value.strip() == PLACEHOLDER:
            return ("still the placeholder from .env.example -- replace "
                    f"{PLACEHOLDER!r} with the key itself")
        return (f"the placeholder {PLACEHOLDER!r} is still in the value, with "
                "the key pasted next to it rather than over it")
    if any(character.isspace() for character in value):
        return ("contains a space, tab or newline -- a key pasted across a "
                "line break, or with the shell prompt caught on the end")
    if len(value) < 20:
        return f"only {len(value)} characters, which is too short to be a key"

    # Named-key hints, last so that a wrong-looking-but-live key still fails on
    # the specific complaint rather than a generic one. A mismatch here is
    # nearly always a key pasted into the wrong variable, which otherwise
    # surfaces as an authentication error naming the wrong provider.
    expected = _KEY_PREFIXES.get(name)
    if expected and not value.startswith(expected):
        return (f"does not start with {expected!r}, so it does not look like "
                f"an {name.removesuffix('_API_KEY').lower()} key -- check it "
                "is not pasted into the wrong variable")
    return ""


def provider_keys() -> list[str]:
    """Which provider keys are set to something that could be a key.

    A placeholder is not one. This returned every non-empty value, so
    `GOOGLE_API_KEY=paste-your-key-here` reported as a key that was set and the
    preflight passed on it.
    """
    return [name for name in KEY_NAMES
            if os.environ.get(name)
            and not key_problem(os.environ[name], name)]


def unusable_keys() -> dict[str, str]:
    """Keys that are set to something unusable, and why. For the preflight."""
    return {name: problem for name in KEY_NAMES
            if (value := os.environ.get(name))
            and (problem := key_problem(value, name))}


# Where each provider will answer "is this key live?" for free. Listing models
# costs nothing everywhere, which matters: on a free tier a verification that
# spent a generation request would take a bite out of a budget that buys three
# candidates a day.
_VERIFY_ENDPOINTS = {
    "GOOGLE_API_KEY": (
        "https://generativelanguage.googleapis.com/v1beta/models",
        lambda key: {"x-goog-api-key": key}, "models", "Google"),
    "ANTHROPIC_API_KEY": (
        "https://api.anthropic.com/v1/models",
        lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"},
        "data", "Anthropic"),
    "OPENAI_API_KEY": (
        "https://api.openai.com/v1/models",
        lambda key: {"Authorization": f"Bearer {key}"}, "data", "OpenAI"),
}


def verify_key(name: str = "GOOGLE_API_KEY", timeout: float = 20.0
               ) -> tuple[bool, str]:
    """Ask the provider whether the key is live. Returns (ok, what it said).

    Lists models rather than generating anything: the model list is free
    everywhere, so a key can be checked without spending a request from a daily
    budget that buys three candidates. A shape check cannot tell a revoked key
    from a live one, and "API key not valid" arriving in front of an audience is
    the failure this exists to move earlier.

    stdlib urllib, because every HTTP client in this environment is somebody
    else's transitive dependency and this module exists to avoid relying on one.
    """
    import json
    import urllib.error
    import urllib.request

    value = os.environ.get(name, "")
    problem = key_problem(value, name)
    if problem:
        return False, problem

    endpoint = _VERIFY_ENDPOINTS.get(name)
    if endpoint is None:
        # OpenRouter has no free unauthenticated model list worth relying on,
        # and a provider nobody has added here is not a provider this can
        # speak for. Silence beats a guess.
        return True, f"not checked -- no verification endpoint for {name}"

    url, headers, collection, provider = endpoint
    request = urllib.request.Request(url, headers=headers(value))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        count = len(payload.get(collection) or [])
        return True, f"accepted by {provider}, {count} models visible"
    except urllib.error.HTTPError as failure:
        body = failure.read().decode("utf-8", "replace")
        if failure.code in (400, 401, 403):
            reason = next((marker for marker in
                           ("API_KEY_INVALID", "invalid_api_key",
                            "authentication_error", "invalid_request_error")
                           if marker in body), "")
            return False, (
                f"rejected by {provider} ({failure.code}"
                f"{', ' + reason if reason else ''}) -- the key is wrong, "
                f"revoked, or from an account without API access")
        return False, f"could not be checked: HTTP {failure.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as failure:
        # Not a verdict on the key. A machine behind a proxy that blocks the
        # provider is a different problem, and reporting it as a bad key would
        # send somebody to rotate a credential that was fine.
        return True, f"not checked -- could not reach {provider} ({failure})"


def key_source(name: str | None = None) -> str:
    """Where a provider key came from, for the preflight to report.

    Worth saying out loud: "the key is set" is not useful when someone is
    debugging why the key they just pasted is not the one being used.
    """
    present = provider_keys()
    name = name or (present[0] if present else None)
    if name is None or not os.environ.get(name):
        return "not set"
    if (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.strip().removeprefix("export ").startswith(name):
                return f"{name}, from .env"
    return f"{name}, from the environment"


def bootstrap() -> None:
    """What every entry point calls first: load .env, then the legacy dev file.

    /tmp/.gk is a development convenience from before .env existed. It stays
    because a running deployment may rely on it, but it is checked last: a key
    a person pasted into the repository should beat one left in /tmp months ago.
    """
    load_env()
    legacy = Path("/tmp/.gk")
    if legacy.exists() and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = legacy.read_text(encoding="utf-8").strip()
