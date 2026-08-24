"""Capture the evidence the dossier prints, by running it.

The point of this file is that the transcripts in the PDF are not typed by
hand. Each entry below is a real command; its real output is written to
`docs/proofs/` and the dossier reads it from there. A claim in the document
therefore cannot drift away from what the repository actually does -- if a
command starts failing, the proof it produces changes, and so does the page.

Commands that need provider budget are skipped when there is no key, and the
skip is recorded rather than hidden: an absent proof should be visible as
absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Runnable straight from a clone, without `pip install -e` first. Half the
# scripts here already did this and half did not, so `serve_champion.py` --
# which the README puts in the quick start -- died with
# `ModuleNotFoundError: No module named 'esp'` on a fresh Codespace while
# its neighbours ran fine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.config import bootstrap

ROOT = Path(__file__).resolve().parent.parent
PROOF_DIR = ROOT / "docs" / "proofs"

# name -> (command, needs_provider_budget)
#
# Order matters: the git snapshots are taken first so the suite and lint
# transcripts below describe the tree exactly as it was listed.
COMMANDS: dict[str, tuple[list[str], bool]] = {
    "tree": (["git", "ls-files"], False),
    "log": (["git", "log", "--oneline", "-15"], False),
    "tests": ([sys.executable, "-m", "pytest", "tests", "-q"], False),
    "lint": ([sys.executable, "-m", "ruff", "check", "esp", "tests",
              "scripts", "apps"], False),
    "offline_search": ([sys.executable, "scripts/offline_search.py",
                        "--pool", "2000"], False),
    # Starts a real neuro-san server and watches it fire the optimiser on a
    # schedule. Needs budget only because the server refuses to start without a
    # key, not because the verification spends one.
    "periodic_server": ([sys.executable, "scripts/verify_periodic.py"], True),
}

# Transcripts that are real but that no single subprocess here can reproduce:
# a server, a browser, and curl in one sitting, assembled by hand. They are
# carried into the index so the dossier can print the evidence that exists --
# it was rendering "Proof not captured" over a transcript sitting on disk --
# and flagged so the document can never present them as harness-captured.
# The flag is the point: an out-of-band proof printed like a machine-captured
# one is a stronger claim than the truth.
ATTESTED: dict[str, str] = {
    "serving_champion":
        "assembled by hand from one live session: serve_champion.py, a real "
        "neuro-san server, curl, and a browser",
}


def run(name: str, command: list[str]) -> dict:
    started = datetime.now(UTC)
    environment = {**os.environ, "PYTHONPATH": str(ROOT),
                   "AGENT_TOOL_PATH": str(ROOT)}
    finished = subprocess.run(command, cwd=ROOT, env=environment,
                              capture_output=True, text=True, timeout=1800)
    output = (finished.stdout + finished.stderr).strip()
    (PROOF_DIR / f"{name}.txt").write_text(output, encoding="utf-8")
    return {
        "name": name,
        "command": " ".join(command),
        "exit_code": finished.returncode,
        "captured_at": started.isoformat(),
        "lines": len(output.splitlines()),
    }


def load_index() -> dict[str, dict]:
    """Whatever the last capture recorded, by name."""
    path = PROOF_DIR / "index.json"
    if not path.exists():
        return {}
    try:
        return {entry["name"]: entry
                for entry in json.loads(path.read_text(encoding="utf-8"))}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def carry_forward(name: str, command: list[str],
                  previous: dict[str, dict]) -> dict:
    """A skip must not delete evidence.

    The index is rewritten wholesale on every run, so without this a machine
    lacking a provider key would replace a real `periodic_server` capture with
    a bare skip: the transcript would stay on disk unreachable, and the dossier
    would print it under an `exit None` header. A proof that cannot be
    reproduced here is still a proof; it is kept, dated, and marked as not
    re-verified in this build.
    """
    earlier = previous.get(name)
    if earlier and (PROOF_DIR / f"{name}.txt").exists() and "exit_code" in earlier:
        return {**earlier, "carried_forward": True}
    return {"name": name, "command": " ".join(command),
            "skipped": "no provider key in this environment"}


def main() -> int:
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    bootstrap()
    has_key = bool(os.environ.get("GOOGLE_API_KEY"))

    previous = load_index()

    index = []
    for name, (command, needs_budget) in COMMANDS.items():
        if needs_budget and not has_key:
            record = carry_forward(name, command, previous)
            index.append(record)
            # Say which of the two actually happened. Printing "skipped" over a
            # record that was carried forward understates the evidence, and
            # printing "carried forward" when nothing was kept would overstate
            # it -- the whole point of this directory is that the two are not
            # the same thing.
            if record.get("carried_forward"):
                when = str(record.get("captured_at", ""))[:10]
                print(f"{name:16} carried forward from {when} "
                      f"(exit {record.get('exit_code')}), no key to re-run it")
            else:
                print(f"{name:16} skipped, no provider key and nothing to keep")
            continue
        record = run(name, command)
        index.append(record)
        print(f"{name:16} exit={record['exit_code']} "
              f"{record['lines']:4d} lines")

    for name, provenance in ATTESTED.items():
        path = PROOF_DIR / f"{name}.txt"
        if not path.exists():
            continue
        index.append({
            "name": name,
            "command": f"({provenance})",
            "attested": provenance,
            "lines": len(path.read_text(encoding="utf-8").splitlines()),
        })
        print(f"{name:16} attested")

    (PROOF_DIR / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8")
    # A failing proof is still a proof, so this does not fail the build. The
    # exit codes are recorded and the dossier prints them.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
