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

from esp.config import (
    key_name_for,
    key_source,
    provider_for,
    provider_keys,
    unusable_keys,
    verify_key,
)
from esp.eval.failover import (
    EXCLUDED,
    LADDER,
    REQUESTS_PER_CANDIDATE,
    daily_budget,
)
from esp.eval.ratelimit import DEFAULT_RPM, keyring
from esp.genome.definition import DEFAULT_MODEL, MODEL_TIERS
from esp.service.state import STATE_DIR


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True


def run_checks(root: Path | None = None, live: bool = False) -> list[Check]:
    """Every check that can be made before a candidate is paid for.

    `live` adds one network call that asks the provider whether the key is
    actually accepted. Off by default so the suite and a keyless clone stay
    offline; the documented preflight turns it on, because a shape check cannot
    tell a revoked key from a live one and "API key not valid" is not a thing
    to discover from inside an agent.
    """
    root = root or Path(__file__).resolve().parent.parent.parent
    checks: list[Check] = []

    # Either provider satisfies this. neuro-san picks the client class from the
    # model name, so what matters is that the key for the model in use is set.
    present = provider_keys()
    # Where it came from, not just that it exists. "The key is set" is useless
    # to somebody debugging why the key they just pasted is not the one in use.
    keys = keyring()
    ring_note = f"; {len(keys)} keys in ring" if len(keys) > 1 else ""

    # A key that is set to something unusable is worse than no key, because it
    # passes a check for presence. `GOOGLE_API_KEY=paste-your-key-here` used to
    # report as set, and the first real call came back "API key not valid" --
    # from inside an agent, eight minutes into a candidate.
    broken = unusable_keys()
    if present:
        detail = f"set, {key_source()}{ring_note}"
    elif broken:
        detail = "; ".join(f"{name} {problem}"
                           for name, problem in sorted(broken.items()))
    else:
        detail = ("unset -- every task would fail identically. Copy "
                  ".env.example to .env and paste GOOGLE_API_KEY or "
                  "OPENROUTER_API_KEY in")
    checks.append(Check("provider key", bool(present), detail))

    # Asked of the provider, not inferred. Free: it lists models rather than
    # generating anything, so it spends nothing from a daily budget that buys
    # three candidates. Non-fatal, because an unreachable provider is a
    # different problem from a bad key and must not be reported as one.
    # The check that saves a whole run. neuro-san picks the client class from
    # the model name, so a network configured for claude-sonnet-5 with only
    # GOOGLE_API_KEY set does not fail at startup -- it fails on every call,
    # inside every agent, and scores every candidate zero. The cache then keeps
    # those zeros, and the search is taught that good topologies are bad.
    wanted = key_name_for(DEFAULT_MODEL)
    if wanted is None:
        checks.append(Check(
            "model provider", False,
            f"nothing here claims {DEFAULT_MODEL!r} -- add its prefix to "
            "PROVIDER_PREFIXES so the right key can be required",
            fatal=False))
    else:
        matched = wanted in present
        checks.append(Check(
            "model provider", matched,
            f"{DEFAULT_MODEL} needs {wanted}"
            + ("" if matched else
               f", which is not set. Present: {', '.join(present) or 'none'}"
               ". Every call would fail and every candidate would score zero")))

    # The ladder is checked as well as the default, because `reassign_model`
    # hands agents every model on it. A Claude default with a Gemini ladder
    # passed the check above and then failed inside whichever agents the search
    # promoted -- scoring those candidates zero, which reads as the promotion
    # having been a bad idea.
    # Only rungs needing a *different* key from the default's: a missing or
    # placeholder default key is already reported above, and repeating it here
    # would send the reader to change a ladder that is fine.
    stranded = sorted({
        f"{model} needs {key}" for model in MODEL_TIERS
        if (key := key_name_for(model)) is not None
        and key != wanted and key not in present})
    checks.append(Check(
        "model tiers", not stranded,
        ", ".join(MODEL_TIERS) if not stranded else
        "; ".join(stranded) + " -- set ESP_MODEL_TIERS to models of the "
        "provider you hold a key for"))

    # Verify the key the configured model will actually use, not whichever key
    # happens to be first. With three providers set, checking the wrong one
    # reports health for a key this run never touches.
    if live:
        target = wanted if wanted in present else (present[0] if present else None)
        if target:
            accepted, verdict = verify_key(target)
            checks.append(Check(f"{target} accepted", accepted, verdict,
                                fatal=accepted is False))

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

    # A population measured on one provider cannot be continued on another:
    # its children inherit a default model this run holds no key for, and its
    # fitness scores do not describe the same networks on the new provider.
    # Reachable by adopting the committed Gemini measurements and then
    # switching .env to Claude.
    from esp.service.state import ServiceState

    configured = provider_for(DEFAULT_MODEL)
    try:
        population = ServiceState.load(STATE_DIR).evaluated
    except (OSError, ValueError, TypeError) as exc:
        # Reported, not raised: a preflight that dies on the thing it exists to
        # check tells the reader less than the check would have.
        checks.append(Check("population provider", False,
                            f"{STATE_DIR}/state.json is unreadable: {exc}"[:200]))
    else:
        foreign = sorted({provider_for(record.model) or record.model
                          for record in population
                          if record.model and provider_for(record.model) != configured})
        checks.append(Check(
            "population provider", not foreign,
            f"all {configured}" if not foreign else
            f"{STATE_DIR} holds measurements taken on {', '.join(foreign)}, and this "
            f"run is configured for {configured}. Measurements do not cross "
            "providers -- point ESP_STATE at a fresh directory and run make baseline"))

    # The demo mode in neuro-san-studio instructs generated agents to invent a
    # realistic-looking answer. Fitness would be measuring fabrication quality.
    demo = os.environ.get("AGENT_NETWORK_DESIGNER_DEMO_MODE", "").lower()
    checks.append(Check(
        "designer demo mode", demo not in ("true", "1", "yes"),
        "off" if demo not in ("true", "1", "yes")
        else "ON -- agents are told to make up realistic answers, so accuracy "
             "would measure fabrication rather than retrieval"))

    # What a day of evaluation buys, for the one provider where that is a
    # meaningful question. Google's free tier caps requests per model per day,
    # and the failover ladder exists to spread a run across those caps. A paid
    # key has no daily cap -- only a per-minute limit and a bill -- so on
    # Anthropic or OpenAI the useful figure is the pace, not a daily budget.
    if provider_for(DEFAULT_MODEL) != "gemini":
        checks.append(Check(
            "pacing", True,
            f"{DEFAULT_RPM} requests/minute per model (ESP_RPM); paid API, no "
            f"daily cap -- one candidate is about {REQUESTS_PER_CANDIDATE} calls"))
        return checks

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
