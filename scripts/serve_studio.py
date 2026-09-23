"""Serve every measured network at once, in neuro-san's own studio UI.

`serve_champion.py` writes one network and answers the question "can I talk to
the winner". This answers a better one: **why** is it the winner. It renders
every measured topology as its own servable agent, so the accelerator UI lists
them side by side and the same question can be put to each -- the shape the
designer produces, the hand-written alternatives, and the evolved network that
beat them. The difference between a demo and an instrument is whether the
comparison is visible, and the comparison is the entire result of this project.

Nothing here is a mock-up. Each agent is the genome that earned its score,
rendered to the HOCON neuro-san serves, and the scores quoted in each agent's
description are the measured ones. Two things differ from the measured form, and
both are stated in the description the UI shows (`esp/serving.py`): the front
man explains its answer instead of returning the bare value the scorer needs,
and on a deployment configured for another provider the models move rung for
rung. Each network also carries the four showcase questions, which the UI
offers as one-click prompts.

    python scripts/serve_studio.py          # write the registries
    make studio                             # write them, then launch the UI

The evaluator is included and public, so a measurement can be asked for from
the same chat panel ("measure the designer's shape against the best evolved
network"). The optimiser is included but stays private, as it is in the
deployed manifest: it spends the day's whole evaluation budget when poked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.config import bootstrap
from esp.eval import measurements
from esp.eval.tasks import TASKS
from esp.serving import SHOWCASE, display_question, presentable

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registries"

# Generated files, one per measured network plus the manifest that lists them.
# Prefixed so `make clean` and .gitignore can name them as a set, and so a
# hand-edited copy cannot be mistaken for something maintained.
PREFIX = "studio_"


def _base_name(record) -> str:
    """A name a person can pick out of a list in the UI.

    The genome hash identifies a network and is not what anybody wants to
    read. `mut:reassign_model` becomes `evolved_reassign_model`, and the hash
    goes in the description where it can still be matched back to
    results/history.json.
    """
    origin = record.origin or record.genome_hash
    if origin.startswith("seed:"):
        return f"seed_{origin.removeprefix('seed:')}"
    stem = re.sub(r"[^a-z0-9_]+", "_", origin.removeprefix("mut:").lower())
    return f"evolved_{stem}" if stem else f"evolved_{record.genome_hash[:8]}"


def agent_names(records) -> list[str]:
    """One distinct filename per network, readable where it can be.

    An operator can win twice. `split_agent` produced two of the eleven
    measured networks, and naming both after the operator alone had the second
    overwrite the first on disk while the manifest listed the same file twice
    -- so eleven measured networks became nine served ones, silently, and the
    two that vanished were ranked second and third. Only the colliding names
    carry a hash, so the common case stays legible.
    """
    bases = [_base_name(record) for record in records]
    repeated = {name for name in bases if bases.count(name) > 1}
    return [f"{name}_{record.genome_hash[:8]}" if name in repeated else name
            for name, record in zip(bases, records, strict=True)]


def describe(record, rank: int, total: int) -> str:
    """What this network is, in the one line the UI shows next to its name."""
    kind = ("the shape neuro-san's agent_network_designer produces"
            if record.origin == "seed:designer_shaped" else
            "a hand-written starting topology" if not record.evolved else
            f"evolved by the {record.origin.removeprefix('mut:')} operator")
    return (f"Rank {rank} of {total} by measured fitness ({record.fitness:+.4f}): "
            f"{kind}. Scored {record.accuracy:.2%} on {len(TASKS)} multi-hop "
            f"questions using {record.tokens:,} tokens across {record.agents} "
            f"agent(s), measured on {record.genome.default_model}. "
            f"Genome {record.genome_hash}. Chosen by measurement, not guessing.")


def write(records, include_optimizer: bool = True) -> tuple[list[Path], Path]:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    entries: list[str] = []

    names = agent_names(records)
    for rank, (record, name) in enumerate(zip(records, names, strict=True),
                                         start=1):
        path = REGISTRY / f"{PREFIX}{name}.hocon"
        served = presentable(record.genome)
        metadata = {
            "description": (describe(record, rank, len(records)) + " "
                            + served.note(record.genome.default_model)),
            # What nsflow offers as one-click prompts. Without it the UI shows
            # only its own generic "What all can you help us with?", which this
            # network -- built to answer questions about one invented company
            # -- cannot usefully answer.
            "sample_queries": [display_question(task) for task in SHOWCASE],
        }
        body = served.genome.to_hocon().replace(
            '"metadata": {"description": "ESP candidate network."},',
            '"metadata": ' + json.dumps(metadata) + ",")
        path.write_text(body, encoding="utf-8")
        written.append(path)
        entries.append(f'    "{path.name}": {{"serve": true, "public": true}},')

    manifest = REGISTRY / f"{PREFIX}manifest.hocon"
    lines = [
        "{",
        "    # Generated by scripts/serve_studio.py -- do not edit.",
        "    #",
        "    # Every network here was measured, and is listed worst-last by the",
        "    # fitness it earned. Ask them all the same question: that comparison",
        "    # is the result this project exists to produce, and reading it off a",
        "    # table is not the same as watching two topologies answer.",
        *entries,
    ]
    lines += [
        "",
        "    # Measures any of the networks above on request, from the chat panel.",
        "    # Public here because this UI is the operator's own; every request is",
        "    # paid, and ESP_EVAL_MAX_RUNS caps them for the life of the server.",
        '    "evaluator.hocon": {"serve": true, "public": true},',
    ]
    if include_optimizer:
        lines += [
            "",
            "    # Private, as in the deployed manifest. Poking it starts a",
            "    # real evaluation, which is about 165 paid model calls.",
            '    "optimizer.hocon": {"serve": true, "public": false},',
        ]
    lines += ["}", ""]
    manifest.write_text("\n".join(lines), encoding="utf-8")

    # Every measured network has to reach the UI. A name collision used to lose
    # two of them to an overwrite, which is exactly the kind of quiet shortfall
    # this project keeps finding, so it is checked rather than assumed.
    if len({path.name for path in written}) != len(records):
        raise SystemExit(
            f"{len(records)} networks produced "
            f"{len({p.name for p in written})} distinct registry files -- "
            "refusing to serve a manifest that has lost one")
    return written, manifest


def main() -> int:
    bootstrap()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=0,
                        help="serve only the N best networks (default: all)")
    parser.add_argument("--no-optimizer", action="store_true",
                        help="leave the optimiser out of the manifest")
    args = parser.parse_args()

    records = measurements.load()
    if not records:
        raise SystemExit(
            "nothing measured, so there is nothing to serve -- the committed "
            "measurements in tests/fixtures/cache are missing")
    if args.top:
        records = records[:args.top]

    written, manifest = write(records, include_optimizer=not args.no_optimizer)

    print(f"{len(written)} measured network(s) rendered into "
          f"{REGISTRY.relative_to(ROOT)}/:\n")
    for rank, (record, path) in enumerate(zip(records, written, strict=True),
                                          start=1):
        print(f"  {rank:>2}. {path.stem.removeprefix(PREFIX):26} "
              f"fitness {record.fitness:+.4f}  acc {record.accuracy:.4f}  "
              f"{record.tokens:>8,} tokens  {record.agents} agent(s)")

    print(f"\n  manifest: {manifest.relative_to(ROOT)}")
    print("\nOpen them in the accelerator UI:\n")
    print(f"  AGENT_MANIFEST_FILE={manifest} \\")
    print(f"  AGENT_TOOL_PATH={ROOT} PYTHONPATH={ROOT} \\")
    print("  nsflow run")
    print("\n  ... then http://localhost:4173 -- or just `make studio`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
