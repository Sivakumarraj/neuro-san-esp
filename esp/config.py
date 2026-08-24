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


def provider_keys() -> list[str]:
    """Which provider keys are actually set."""
    return [name for name in KEY_NAMES if os.environ.get(name)]


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
