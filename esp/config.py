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
KEY_NAMES = ("GOOGLE_API_KEY", "OPENROUTER_API_KEY")


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


def key_problem(value: str) -> str:
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
    return ""


def provider_keys() -> list[str]:
    """Which provider keys are set to something that could be a key.

    A placeholder is not one. This returned every non-empty value, so
    `GOOGLE_API_KEY=paste-your-key-here` reported as a key that was set and the
    preflight passed on it.
    """
    return [name for name in KEY_NAMES
            if os.environ.get(name) and not key_problem(os.environ[name])]


def unusable_keys() -> dict[str, str]:
    """Keys that are set to something unusable, and why. For the preflight."""
    return {name: problem for name in KEY_NAMES
            if (value := os.environ.get(name))
            and (problem := key_problem(value))}


def verify_key(name: str = "GOOGLE_API_KEY", timeout: float = 20.0
               ) -> tuple[bool, str]:
    """Ask Google whether the key is live. Returns (ok, what it said).

    Lists models rather than generating anything: the model list is free, so a
    key can be checked without spending a request from a daily budget that buys
    three candidates. A shape check cannot tell a revoked key from a live one,
    and "API key not valid" arriving in front of an audience is the failure this
    exists to move earlier.

    stdlib urllib, because every HTTP client in this environment is somebody
    else's transitive dependency and this module exists to avoid relying on one.
    """
    import json
    import urllib.error
    import urllib.request

    value = os.environ.get(name, "")
    problem = key_problem(value)
    if problem:
        return False, problem
    if name != "GOOGLE_API_KEY":
        return True, "not checked -- only Google keys can be verified here"

    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": value})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        count = len(payload.get("models") or [])
        return True, f"accepted by Google, {count} models visible"
    except urllib.error.HTTPError as failure:
        body = failure.read().decode("utf-8", "replace")
        if failure.code in (400, 401, 403):
            reason = "API_KEY_INVALID" if "API_KEY_INVALID" in body else ""
            return False, (
                f"rejected by Google ({failure.code}"
                f"{', ' + reason if reason else ''}) -- the key is wrong, "
                "revoked, or from a project without the Generative Language "
                "API enabled")
        return False, f"could not be checked: HTTP {failure.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as failure:
        # Not a verdict on the key. A machine behind a proxy that blocks Google
        # is a different problem, and reporting it as a bad key would send
        # somebody to rotate a credential that was fine.
        return True, f"not checked -- could not reach Google ({failure})"


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
