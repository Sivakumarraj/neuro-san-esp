"""Refuse to start on a configuration that would produce wrong numbers.

A misconfigured evaluator does not crash. It produces plausible numbers that
are wrong, and the cache then keeps them.

Two failures make the point. Without `AGENT_TOOL_PATH` neuro-san refuses to
build a session, so every model fails identically and a probe reports healthy
models as broken. A search started on an exhausted model does not stop either --
every candidate scores zero, and the search concludes that good topologies are
bad.

So the checks run before anything is paid for, and a failure is a refusal to
start rather than a warning nobody reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from esp.config import key_source, provider_keys
from esp.eval.failover import (
    EXCLUDED,
    LADDER,
    REQUESTS_PER_CANDIDATE,
    daily_budget,
)
from esp.eval.ratelimit import keyring
from esp.service.state import STATE_DIR


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True


def run_checks(root: Path | None = None) -> list[Check]:
    root = root or Path(__file__).resolve().parent.parent.parent
    checks: list[Check] = []

    # Either provider satisfies this. neuro-san picks the client class from the
    # model name, so what matters is that the key for the model in use is set.
    present = provider_keys()
    # Where it came from, not just that it exists. "The key is set" is useless
    # to somebody debugging why the key they just pasted is not the one in use.
    keys = keyring()
    ring_note = f"; {len(keys)} keys in ring" if len(keys) > 1 else ""
    checks.append(Check(
        "provider key", bool(present),
        f"set, {key_source()}{ring_note}" if present
        else "unset -- every task would fail identically. Copy .env.example to "
             ".env and paste GOOGLE_API_KEY or OPENROUTER_API_KEY in"))

    tool_path = os.environ.get("AGENT_TOOL_PATH", "")
    checks.append(Check(
        "AGENT_TOOL_PATH", bool(tool_path),
        f"{tool_path}" if tool_path
        else "unset -- neuro-san cannot resolve CorpusSearch and refuses to "
             "build a session, which reads as every topology being broken"))

    python_path = os.environ.get("PYTHONPATH", "")
    checks.append(Check(
        "PYTHONPATH", str(root) in python_path.split(os.pathsep),
        python_path or "unset",
        fatal=False))

    # State has to be writable before a wake spends eight minutes on a candidate
    # it will then be unable to record.
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        probe = STATE_DIR / ".writable"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        writable = True
        detail = str(STATE_DIR.resolve())
    except OSError as exc:
        writable = False
        detail = f"{STATE_DIR}: {exc}"
    checks.append(Check("state directory writable", writable, detail))

    # The demo mode in neuro-san-studio instructs generated agents to invent a
    # realistic-looking answer. Fitness would be measuring fabrication quality.
    demo = os.environ.get("AGENT_NETWORK_DESIGNER_DEMO_MODE", "").lower()
    checks.append(Check(
        "designer demo mode", demo not in ("true", "1", "yes"),
        "off" if demo not in ("true", "1", "yes")
        else "ON -- agents are told to make up realistic answers, so accuracy "
             "would measure fabrication rather than retrieval"))

    # Which models this run will actually use, and what a day of them buys.
    # The ladder is derived from measured caps and can be overridden from the
    # environment, so "which model am I on" must not require reading source.
    ladder = ", ".join(LADDER) if LADDER else "none"
    key_multiplier = max(1, len(keyring()))
    daily = daily_budget() * key_multiplier
    candidates = daily // REQUESTS_PER_CANDIDATE
    key_note = f" ({key_multiplier} keys)" if key_multiplier > 1 else ""
    detail = (f"{ladder} -- {daily} requests/day{key_note} "
              f"= about {candidates} candidate(s)")
    if EXCLUDED:
        # Silently dropping these is how a four-rung ladder came to spend 20
        # requests a rung and measure nothing. Named with the reason, because
        # somebody who configured one deserves better than a run of zeros.
        detail += "; excluded: " + ", ".join(
            f"{model} ({why})" for model, why in EXCLUDED.items())
    checks.append(Check("model ladder", bool(LADDER), detail))

    return checks


def report(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        mark = "ok  " if check.ok else ("FAIL" if check.fatal else "warn")
        lines.append(f"  [{mark}] {check.name}: {check.detail}")
    return "\n".join(lines)


def failures(checks: list[Check]) -> list[Check]:
    return [c for c in checks if not c.ok and c.fatal]
