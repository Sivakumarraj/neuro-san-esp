"""The committed measurements have to agree with each other.

`results/history.json` is a summary. `tests/fixtures/cache/` holds the real
thing: one file per candidate with its per-task outcomes and the genome that
produced them. Everything this repository claims about its results is read off
those two files, and they were written by separate code paths at separate
times -- which is exactly how a repository ends up reporting one run in its
README, a second in its findings document, and a third in its artifacts.

So: every score in the summary must be recomputable from the detail, every
genome must rebuild to the hash it was filed under, and the whole population
must have been measured on one model. Those are the three assumptions every
published number here rests on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esp.eval import measurements
from esp.evolve.loop import TOKEN_SCALE, WEIGHTS
from esp.genome.definition import Genome
from esp.surrogate.predictor import Surrogate

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tests" / "fixtures" / "cache"
HISTORY = json.loads((ROOT / "results" / "history.json").read_text(encoding="utf-8"))
RECORDS = {record["genome_hash"]: record for record in HISTORY["records"]}


def cached() -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(CACHE.glob("*.json"))}


def scalarised(entry: dict) -> float:
    """The project's own fitness, recomputed from a cached measurement."""
    return (WEIGHTS["accuracy"] * entry["accuracy"]
            - WEIGHTS["tokens"] * min(entry["tokens"] / TOKEN_SCALE, 1.0)
            - WEIGHTS["agents"] * (entry["agents"] / 9.0))


def test_the_summary_describes_the_same_population_as_the_cache():
    assert set(cached()) == set(RECORDS), (
        "results/history.json and tests/fixtures/cache describe different runs")
    assert HISTORY["real_evaluations"] == len(RECORDS)


@pytest.mark.parametrize("digest", sorted(cached()))
def test_every_published_fitness_is_recomputable(digest):
    """A score in the summary that cannot be rederived from the measurement is
    a number with no measurement behind it."""
    entry = cached()[digest]
    assert scalarised(entry) == pytest.approx(RECORDS[digest]["fitness"], abs=5e-4)
    for field in ("accuracy", "tokens", "agents", "depth"):
        assert entry[field] == RECORDS[digest][field], field


@pytest.mark.parametrize("digest", sorted(cached()))
def test_every_measured_candidate_can_be_rebuilt(digest):
    """The winner of a search that cannot be reconstructed is a hash. Serving
    it, re-measuring it or showing somebody its topology all need the genome,
    and it is only trustworthy if it rebuilds to the hash it was filed under."""
    entry = cached()[digest]
    assert "genome" in entry, "measured and then lost: no genome stored"
    assert Genome.from_canonical(entry["genome"]).genome_hash() == digest


def test_the_whole_population_was_measured_on_one_model():
    """The model is part of the genome hash precisely so that fitnesses are not
    compared across models. A mixed population would make the results table a
    comparison of providers wearing the clothes of a comparison of topologies."""
    models = {Genome.from_canonical(entry["genome"]).default_model
              for entry in cached().values()}
    assert len(models) == 1, f"measured on more than one model: {sorted(models)}"


def test_every_candidate_faced_the_same_task_set():
    counts = {digest: len(entry["results"]) for digest, entry in cached().items()}
    assert len(set(counts.values())) == 1, (
        f"candidates were scored over different task sets: {counts}")


def test_an_unfinished_run_is_marked_rather_than_scored_as_wrong():
    """A timeout is not an answer. The distinction has to survive in the cache,
    because `answered_accuracy` is computed from it and the README quotes it."""
    for digest, entry in cached().items():
        for result in entry["results"]:
            assert "infrastructure" in result, (
                f"{digest} {result['task_id']}: no outcome classification")
            if result["infrastructure"]:
                assert not result["correct"]


def test_the_recorded_surrogate_quality_is_not_a_placeholder():
    """`spearman: 0.0` used to mean "never computed". Anything present here now
    is a measurement, and its verdict has to match its own number."""
    for report in HISTORY["surrogate_quality"]:
        assert report["spearman"] is not None
        assert report["beats_random"] == (report["spearman"] > 0.2)


# ---------------------------------------------- what the Predictor is worth

def test_the_predictor_ranks_better_than_chance_on_the_whole_population():
    """The claim the docs make about the refitted surrogate, and the reason it
    is stated as a range.

    At eleven samples a single cross-validated spearman is an artefact of how
    `KFold` split: the same measurement yields +0.24 to +0.65 depending on the
    fold seed and on the order the samples arrive in. What survives all of that
    is the sign and the verdict, so that is what is pinned. A number quoted to
    three decimals from one seed would be pinned to the seed rather than to the
    data.
    """
    found = measurements.load()
    orders = {
        "as loaded": found,
        "reversed": list(reversed(found)),
        "by hash": sorted(found, key=lambda m: m.genome_hash),
    }
    surrogate = Surrogate()
    seen = []
    for label, sequence in orders.items():
        for seed in range(4):
            quality = surrogate.report_quality(
                [m.genome for m in sequence], [m.fitness for m in sequence],
                seed=seed)
            assert quality.measured, f"{label}/{seed}: not measured"
            assert quality.beats_random, (
                f"{label}/{seed}: spearman={quality.spearman:+.3f} no longer "
                f"beats chance")
            seen.append(quality.spearman)

    assert min(seen) > 0, f"a negative correlation appeared: {min(seen):+.3f}"


def test_the_findings_quote_the_range_they_measured():
    findings = (ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")
    assert "+0.236 to +0.645" in findings, (
        "the measured spread is the finding; a single figure is a seed")


# ------------------------------------------------- reaching the measurements

def test_the_best_committed_measurement_is_the_best_one():
    found = measurements.best()
    assert found is not None
    assert found.fitness == max(m.fitness for m in measurements.load())
    assert found.genome.genome_hash() == found.genome_hash


def test_the_champion_is_an_evolved_candidate_and_can_be_served():
    """The whole point of the search, and the thing three separate readers used
    to be unable to reach. If this ever becomes a seed again, either the search
    regressed or something stopped resolving the evolved winner."""
    found = measurements.best()
    assert found.evolved, f"the best measured network is a seed: {found.origin}"
    assert found.origin, "no origin, so nothing can name what won"
    assert found.genome.to_hocon(), "the winner does not render as a network"


def test_a_cache_entry_whose_genome_does_not_rebuild_is_skipped(tmp_path):
    entry = json.loads(
        next(CACHE.glob("*.json")).read_text(encoding="utf-8"))
    entry["genome_hash"] = "f" * 16         # no longer describes the genome
    (tmp_path / ("f" * 16 + ".json")).write_text(json.dumps(entry),
                                                 encoding="utf-8")
    assert measurements.load(tmp_path) == []


def test_an_empty_or_missing_cache_is_not_an_error(tmp_path):
    assert measurements.load(tmp_path) == []
    assert measurements.load(tmp_path / "nope") == []
    assert measurements.best(tmp_path) is None


def test_unreadable_cache_entries_are_stepped_over(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
    good = next(CACHE.glob("*.json"))
    (tmp_path / good.name).write_text(good.read_text(encoding="utf-8"),
                                      encoding="utf-8")
    assert len(measurements.load(tmp_path)) == 1
