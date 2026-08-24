"""Serve the best topology the service has found, so a person can talk to it.

Until this existed the winner of the search was a hash in a report, which is one
step short of the point: the reason to measure topologies is that one of them is
better to actually use, and nothing here let anyone use it.

This writes the current best genome into the registry as a normal, servable
agent network. Start the server afterwards and it answers questions in the
browser like any other neuro-san agent -- except this one was chosen by
measurement rather than by a person guessing.

It is regenerated rather than hand-maintained on purpose, and the generated
files are gitignored. A champion edited by hand would drift from the genome that
earned the score, and then the thing being served would not be the thing that
was measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Runnable straight from a clone, without `pip install -e` first. Half the
# scripts here already did this and half did not, so `serve_champion.py` --
# which the README puts in the quick start -- died with
# `ModuleNotFoundError: No module named 'esp'` on a fresh Codespace while
# its neighbours ran fine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.config import bootstrap
from esp.genome.definition import DEFAULT_MODEL, Genome
from esp.genome.seeds import SEEDS
from esp.service.state import Evaluated, ServiceState

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registries"


def committed_best() -> Evaluated | None:
    """The best topology in `results/history.json`, which is committed.

    `make champion` read only the service's own state, so on a fresh clone it
    exited with "nothing measured yet -- run a wake first" -- and a wake needs a
    key. The measurements this repository reports everywhere else were sitting
    in `results/` the whole time, unreachable. Same shape as the bug that had
    `make offline` demanding an API budget to run the half of ESP that is meant
    to be free.
    """
    history = ROOT / "results" / "history.json"
    if not history.exists():
        return None
    try:
        records = json.loads(history.read_text(encoding="utf-8"))["records"]
    except (json.JSONDecodeError, KeyError):
        return None
    seeds = [r for r in records if str(r.get("origin", "")).startswith("seed:")]
    if not seeds:
        return None
    best = max(seeds, key=lambda r: r["fitness"])
    return Evaluated(
        genome_hash=best["genome_hash"], origin=best["origin"],
        fitness=best["fitness"], accuracy=best["accuracy"],
        tokens=best["tokens"], agents=best["agents"], depth=best["depth"],
        generation=best.get("generation", 0), measured_at="",
        model=best.get("model", ""))


def champion_genome(state: ServiceState):
    """The measured best, matched back to a genome we can render.

    Only seeds can be resurrected today: the service records a genome's hash and
    its score, not the genome itself. That is a real limitation and it is stated
    rather than papered over -- an evolved winner cannot yet be served, which is
    the next thing worth building.
    """
    best = state.best()
    if best is None:
        # Announced, never silent: serving somebody a topology from the
        # committed run while they believe it is their own service's winner
        # would misreport what they are talking to.
        best = committed_best()
        if best is not None:
            print("No service population yet -- serving the best topology from "
                  "the committed measurements in results/history.json.")
            print("Run a wake with a key to serve your own.\n")
    if best is None:
        raise SystemExit(
            "nothing measured yet, and results/history.json holds no seed "
            "measurement either -- run a wake first")

    for name, build in SEEDS.items():
        genome = build()
        if genome.genome_hash() == best.genome_hash:
            return name, genome, best

    # An evolved winner, rebuilt from the genome the service now stores beside
    # the score. This is the outcome the whole search exists to produce, and
    # until the state carried the candidate itself it was the one outcome that
    # could not be served -- a hash cannot be turned back into a network.
    #
    # The reconstruction is checked rather than trusted: a genome that rebuilds
    # to a different hash is worse than none, because it would quietly serve a
    # network that is not the one that earned the score.
    if best.genome:
        rebuilt = Genome.from_canonical(best.genome)
        if rebuilt.genome_hash() != best.genome_hash:
            raise SystemExit(
                f"stored genome for {best.genome_hash} rebuilds to "
                f"{rebuilt.genome_hash()} -- refusing to serve a network that "
                "is not the one that was measured")
        return best.origin or "evolved", rebuilt, best

    # The hash did not match, which has two very different causes and this
    # reported only the rarer one. The model is part of the genome, so measuring
    # on gemini-3.1-flash-lite and then setting ESP_DEFAULT_MODEL to something
    # else re-hashes every seed: `designer_shaped` goes from 459ac1a6 to
    # 4bf49cf0 without a single agent changing. The record's `origin` still
    # names the seed, so ask it rather than concluding the genome is
    # unreconstructable and sending somebody to look at ServiceState.
    seed_name = best.origin.removeprefix("seed:") if best.origin else ""
    if seed_name in SEEDS:
        genome = SEEDS[seed_name]()
        print(f"NOTE: {seed_name} hashes to {genome.genome_hash()} on "
              f"{DEFAULT_MODEL}, but the measurement recorded "
              f"{best.genome_hash}.")
        print("The topology is the same; the model is part of the genome, so a "
              "different model gives a different hash.")
        print("Serving the topology. The numbers below describe the run on the "
              "model it was measured on, NOT what you are about to talk to.\n")
        return seed_name, genome, best

    raise SystemExit(
        f"best genome {best.genome_hash} came from origin {best.origin!r}, "
        "which is not one of the seeds, so it is an evolved candidate -- and "
        "the service stores scores rather than genomes, so it cannot be "
        "rebuilt. Serve a seed champion, or extend ServiceState to keep the "
        "genome.")


def write_registry(name: str, genome, record, model: str | None = None
                   ) -> tuple[Path, Path]:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    hocon = REGISTRY / "champion.hocon"

    if model:
        # Serving on a different model than the one that earned the score is a
        # real change: the model is part of the genome hash precisely because it
        # changes both the answers and the cost. It is allowed because a spent
        # daily quota should not make the winner unreachable, but what is served
        # is then the topology, not the measurement.
        genome = genome.clone()
        genome.default_model = model

    served_on = model or record.model
    body = genome.to_hocon().replace(
        '"metadata": {"description": "ESP candidate network."},',
        '"metadata": {"description": '
        + json.dumps(
            f"The best-measured topology so far: {name}, "
            f"{record.accuracy:.0%} correct on 17 multi-hop questions using "
            f"{record.tokens:,} tokens across {record.agents} agent(s), "
            f"measured on {record.model}. Chosen by measurement, not guessing."
            + (f" Being served on {served_on}, which is not the model it was "
               "measured on." if served_on != record.model else ""))
        + "},")
    hocon.write_text(body, encoding="utf-8")

    manifest = REGISTRY / "champion_manifest.hocon"
    manifest.write_text(
        "{\n"
        "    # Public on purpose: this one answers questions and spends only\n"
        "    # what the person asking spends. The optimiser stays private --\n"
        "    # it spends the day's whole evaluation budget when poked.\n"
        '    "champion.hocon": {"serve": true, "public": true},\n'
        "}\n", encoding="utf-8")
    return hocon, manifest


def main() -> int:
    bootstrap()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="state")
    parser.add_argument(
        "--model", default=None,
        help="serve on this model instead of the one it was measured on, for "
             "when the measured model's daily quota is spent. The topology is "
             "still the topology, but the numbers no longer describe what you "
             "are talking to.")
    args = parser.parse_args()

    state = ServiceState.load(Path(args.state))
    name, genome, record = champion_genome(state)
    hocon, manifest = write_registry(name, genome, record, args.model)

    print(f"champion: {name} ({record.genome_hash})")
    print(f"  accuracy {record.accuracy:.2f}   tokens {record.tokens:,}   "
          f"agents {record.agents}   fitness {record.fitness:+.4f}")
    print(f"  wrote {hocon.relative_to(ROOT)}")
    print(f"  wrote {manifest.relative_to(ROOT)}")
    if args.model and args.model != record.model:
        print(f"  NOTE: serving on {args.model}, measured on {record.model} -- "
              "same topology, but the numbers above are not what you will get")
    print("\nServe it:")
    print(f"  AGENT_MANIFEST_FILE={manifest} \\")
    print(f"  AGENT_TOOL_PATH={ROOT} PYTHONPATH={ROOT} \\")
    print("  python -m neuro_san.service.main_loop.server_main_loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
