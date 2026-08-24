"""Verify that neuro-san really fires the optimiser on its own.

This is the one claim in the repository that cannot be checked by a unit test:
that an `invocation: "event"` agent on a cron schedule is started by the
framework with no user and no client attached. So it is checked by starting a
real server and watching for the interaction.

The cron is overridden to once a minute for the duration -- the shipped schedule
is hourly, and a verification that takes an hour is a verification nobody runs.

It does not assert that a wake completes. The front agent needs its own provider
call to reason, and on an exhausted free tier that call fails before it reaches
the tool. What is being verified here is the trigger; the wake itself is
verified by apps/optimizer/run_optimizer.py, which runs identical code.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# Runnable straight from a clone, without `pip install -e` first. Half the
# scripts here already did this and half did not, so `serve_champion.py` --
# which the README puts in the quick start -- died with
# `ModuleNotFoundError: No module named 'esp'` on a fresh Codespace while
# its neighbours ran fine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.config import bootstrap

ROOT = Path(__file__).resolve().parent.parent
WANTED_FIRES = int(os.environ.get("ESP_VERIFY_FIRES", "3"))
TIMEOUT = float(os.environ.get("ESP_VERIFY_TIMEOUT", "300"))

STARTUP = ("Starting PeriodicEventInitiator", "periodic agent interactions",
           "HealthProbeServer started")
FIRE = "Received a optimizer.StreamingChat request"


def _stamp(line: str) -> str:
    found = re.search(r'"Timestamp": "([^"]+)"', line)
    return found.group(1)[:19] if found else ""


def _message(line: str) -> str:
    found = re.search(r'"message": "([^"]*)"', line)
    return found.group(1) if found else ""


def main() -> int:
    bootstrap()
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set", file=sys.stderr)
        return 1

    log = ROOT / ".esp-verify.log"
    environment = {
        **os.environ,
        "AGENT_MANIFEST_FILE": str(ROOT / "registries/manifest.hocon"),
        "AGENT_TOOL_PATH": str(ROOT),
        "PYTHONPATH": str(ROOT),
        # A dedicated state directory: this must not disturb a real population,
        # and it must not fight a real wake for the lease.
        "ESP_STATE": str(ROOT / ".esp-verify-state"),
        "ESP_OPTIMIZER_CRON": "*/1 * * * *",
    }

    with log.open("w") as sink:
        server = subprocess.Popen(
            [sys.executable, "-m", "neuro_san.service.main_loop.server_main_loop"],
            cwd=ROOT, env=environment, stdout=sink, stderr=subprocess.STDOUT,
            start_new_session=True)

    try:
        deadline = time.time() + TIMEOUT
        fires: list[str] = []
        startup: list[str] = []
        while time.time() < deadline and len(fires) < WANTED_FIRES:
            time.sleep(2)
            text = log.read_text(errors="replace")
            startup = [f"{_stamp(ln)}  user_id=None     {_message(ln)}"
                       for ln in text.splitlines()
                       if any(k in ln for k in STARTUP) and _message(ln)]
            # These records embed the prompt with its newlines intact, so they
            # are not valid JSON and cannot be parsed -- only matched.
            fires = [f"{stamp[:19]}  user_id={user:<8} {FIRE}"
                     for user, stamp in re.findall(
                         rf'"message": "{FIRE} for .*?"user_id": "([^"]*)", '
                         r'"Timestamp": "([^"]+)"', text, re.S)]
    finally:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        server.wait(timeout=30)

    report = "\n".join([*startup, *sorted(fires)])
    report += (f"\n\n--- {len(fires)} scheduled fires, one per minute, "
               "no client connected ---")
    print(report)

    if len(fires) < WANTED_FIRES:
        print(f"\nEXPECTED {WANTED_FIRES} fires, saw {len(fires)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
