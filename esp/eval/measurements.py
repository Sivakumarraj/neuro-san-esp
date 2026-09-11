"""The evaluations this repository has already paid for.

Three places needed the same thing -- the best network anybody has actually
measured, as a genome they can serve or breed from -- and each reached for a
different, worse approximation of it. `offline_search.py` read the evaluation
cache properly. `serve_champion.py` read `results/history.json`, which records
scores and no genomes, so it could only resurrect seeds and served `flat_pair`
at +0.7852 on a fresh clone. The web front end read the service's own state,
found nothing on a fresh clone, and fell back to `designer_shaped` at +0.7761 --
the *worst* of the eleven measured networks, presented as the champion.

Meanwhile the real champion, at +0.8453, was committed the whole time in
`tests/fixtures/cache/` with its genome beside its score. Same shape as the bug
that had `make offline` demanding an API budget to run the half of ESP that is
meant to be free: the measurement was there, and nothing could reach it. So
there is now one reader, and the three callers share it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from esp.genome.definition import Genome
from esp.genome.seeds import SEEDS

ROOT = Path(__file__).resolve().parent.parent.parent

# Paid for once, committed, so the free half of ESP needs no account. This is
# also the only population a fresh clone has, which is why every default here
# points at it.
FIXTURE_CACHE = ROOT / "tests" / "fixtures" / "cache"
HISTORY = ROOT / "results" / "history.json"
# The live service population, which knows the origin of a candidate it
# measured before that candidate reaches the published history.
STATE = Path(os.environ.get("ESP_STATE", ROOT / "state")) / "state.json"

# Weights live with the loop that selects on them; imported lazily inside
# `_fitness` so that reading measurements does not drag the evolution module in.


@dataclass(frozen=True)
class Measurement:
    """One real evaluation, with the network that produced it."""

    genome_hash: str
    genome: Genome
    fitness: float
    accuracy: float
    tokens: int
    agents: int
    depth: int
    origin: str = ""

    @property
    def evolved(self) -> bool:
        return not self.origin.startswith("seed:")

    def name(self) -> str:
        """Something to call it in a UI or a log line."""
        return self.origin or self.genome_hash


def _fitness(accuracy: float, tokens: int, agents: int) -> float:
    from esp.evolve.loop import TOKEN_SCALE, WEIGHTS
    return (WEIGHTS["accuracy"] * accuracy
            - WEIGHTS["tokens"] * min(tokens / TOKEN_SCALE, 1.0)
            - WEIGHTS["agents"] * (agents / 9.0))


def _origins() -> dict[str, str]:
    """Which operator or seed produced each measured hash.

    The cache does not record it -- it is keyed by genome and knows nothing
    about the search that proposed the genome -- so provenance comes from the
    two files that do: the published run history, and the live service state.
    A champion labelled `mut:reassign_model` rather than by its hash is the
    difference between a result and a checksum.

    The service state is read second and does not overwrite the history,
    because the history is what was published. It is read at all because a
    candidate a wake measured this morning is in the state and in the cache
    and not yet in the history -- and without this it would surface as a
    network nobody could say the origin of, which is how a freshly measured
    winner ends up published with a blank provenance.
    """
    origins: dict[str, str] = {}

    for path, key in ((STATE, "evaluated"), (HISTORY, "records")):
        if not path.exists():
            continue
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))[key] or []
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            continue
        for entry in entries:
            digest = entry.get("genome_hash")
            origin = entry.get("origin") or ""
            if digest and origin:
                origins[digest] = origin

    return origins


def _normalise_origin(origin: str) -> str:
    """`reassign_model` and `mut:reassign_model` are the same provenance.

    The batch loop prefixes an operator with `mut:` and the service records the
    bare operator name, so the same candidate reads differently depending on
    which one measured it. Normalising here keeps the published history
    consistent with itself rather than recording the accident of which code
    path paid for the candidate.
    """
    if not origin or origin.startswith(("seed:", "mut:")):
        return origin
    return f"mut:{origin}"


def load(cache_dir: Path | None = None) -> list[Measurement]:
    """Every cached evaluation whose network can be rebuilt, best first.

    A record is skipped rather than guessed at when its genome is missing and
    it is not a seed, and when the stored genome rebuilds to a different hash.
    The second case matters more than it looks: serving or training on a
    network that is not the one that earned the score is worse than having one
    fewer measurement, because nothing downstream can tell.
    """
    directory = cache_dir or FIXTURE_CACHE
    if not directory.is_dir():
        return []

    seeds = {}
    for name, build in SEEDS.items():
        genome = build()
        seeds[genome.genome_hash()] = (f"seed:{name}", genome)
    origins = _origins()

    found: list[Measurement] = []
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            digest = raw["genome_hash"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue

        stored = raw.get("genome")
        if stored:
            try:
                genome = Genome.from_canonical(stored)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if genome.genome_hash() != digest:
                continue
        elif digest in seeds:
            genome = seeds[digest][1]       # older cache, seeds only
        else:
            continue

        try:
            accuracy = float(raw["accuracy"])
            tokens = int(raw["tokens"])
            agents = int(raw["agents"])
        except (KeyError, TypeError, ValueError):
            continue

        found.append(Measurement(
            genome_hash=digest, genome=genome,
            fitness=round(_fitness(accuracy, tokens, agents), 4),
            accuracy=accuracy, tokens=tokens, agents=agents,
            depth=int(raw.get("depth", genome.depth())),
            origin=_normalise_origin(
                origins.get(digest)
                or (seeds[digest][0] if digest in seeds else ""))))

    return sorted(found, key=lambda m: -m.fitness)


def best(cache_dir: Path | None = None) -> Measurement | None:
    """The highest-scoring network anybody here has actually paid to measure."""
    found = load(cache_dir)
    return found[0] if found else None


def raw(cache_dir: Path | None = None) -> list[dict]:
    """The cached evaluations as written, for readers that need the per-task
    detail rather than the genome.

    `load` returns what can be searched and served; the outcome of each
    individual task is not part of that and is what the reports count
    unfinished runs from.
    """
    directory = cache_dir or FIXTURE_CACHE
    if not directory.is_dir():
        return []
    entries = []
    for path in sorted(directory.glob("*.json")):
        try:
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return entries
