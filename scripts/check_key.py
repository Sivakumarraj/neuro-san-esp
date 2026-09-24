"""Is the provider key usable? Asked before something long or public starts.

The failure this exists to move earlier looks like success for a while. A key
set to the placeholder from `.env.example` is a perfectly good non-empty
string, so a check for presence passes, the UI opens, twelve topologies render,
and the first question comes back

    Error from Coordinator: ... 400 INVALID_ARGUMENT.
    API key not valid. Please pass a valid API key.

from inside an agent -- which is a bad place to find that out and a worse one
to find it out in front of somebody being shown the project.

Exits 0 when the key looks usable or when the verdict cannot be reached (a
machine that cannot see Google is a different problem, and reporting it as a
bad key sends somebody to rotate a working credential). Exits 1 when the key is
definitely unusable, so a caller can decide whether that is fatal: `make
studio` warns and opens the UI anyway, because twelve topologies are worth
looking at without a key, while the optimiser preflight refuses to start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.config import bootstrap, key_name_for, provider_keys, unusable_keys, verify_key


def problems(live: bool = True) -> tuple[list[str], list[str]]:
    """What stops a run, then what is only left over, in words a person can act on.

    Only the key the configured model calls with can stop a run. The devcontainer
    copies .env.example to .env, so a placeholder on another provider's line is
    what everybody who pastes a different provider's key is left with. Failing
    on it sent people to fix a line their run never reads, while the preflight,
    which checks the key in use, passed.
    """
    bootstrap()
    from esp.genome.definition import DEFAULT_MODEL  # reads the env .env just set

    broken = unusable_keys()
    usable = provider_keys()
    wanted = key_name_for(DEFAULT_MODEL)
    in_use = wanted if wanted in usable else (None if wanted else next(iter(usable), None))

    if in_use is None:
        if broken:
            return [f"{name} {problem}" for name, problem in sorted(broken.items())], []
        if usable:
            return [f"{DEFAULT_MODEL} needs {wanted}, which is not set -- set it, or "
                    "ESP_DEFAULT_MODEL to a model of the provider you hold a key for"], []
        return ["no provider key set -- in .env, uncomment the line for your "
                "provider and paste its key in"], []

    notes = [f"{name} {problem} -- ignored, this run uses {in_use}; comment the "
             "line out to silence this" for name, problem in sorted(broken.items())]
    if live:
        accepted, verdict = verify_key(in_use)
        if not accepted:
            return [f"{in_use} {verdict}"], notes
    return [], notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="check the shape only, ask the provider nothing")
    parser.add_argument("--warn", action="store_true",
                        help="report and exit 0, for a caller that continues anyway")
    parser.add_argument("--quiet-when-fine", action="store_true",
                        help="say nothing when the key is usable")
    args = parser.parse_args()

    found, notes = problems(live=not args.offline)
    if not found:
        if not args.quiet_when_fine:
            print("provider key: usable")
            for note in notes:
                print(f"  note: {note}")
        return 0

    print()
    for problem in found:
        print(f"  WARNING: {problem}" if args.warn else f"  {problem}")
    if args.warn:
        print("  The UI will open and the agents will not answer.")
        print("  Fix .env, then stop this and start it again: the key is read "
              "once at launch,")
        print("  so editing .env under a running server changes nothing until "
              "it restarts.")
    print()
    return 0 if args.warn else 1


if __name__ == "__main__":
    raise SystemExit(main())
