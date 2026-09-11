"""Every measured network reaches neuro-san's accelerator UI.

`serve_champion` answers "can I talk to the winner". The studio answers the
better question -- why is it the winner -- by serving every measured topology
at once so the same question can be put to the designer's shape and to the
network that beat it. That only works if all of them arrive: a name collision
silently dropped two of eleven, and the two lost were ranked second and third.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import serve_studio  # noqa: E402

from esp.eval import measurements  # noqa: E402


@pytest.fixture
def records():
    found = measurements.load()
    assert found, "no committed measurements to serve"
    return found


def test_every_measured_network_gets_its_own_name(records):
    names = serve_studio.agent_names(records)
    assert len(names) == len(records)
    assert len(set(names)) == len(records), (
        f"two networks share a name and one would overwrite the other: {names}")


def test_an_operator_that_wins_twice_is_disambiguated_by_hash(records):
    """`split_agent` produced two of the eleven. Both have to be servable."""
    names = serve_studio.agent_names(records)
    repeated = [record for record in records
                if sum(1 for r in records if r.origin == record.origin) > 1]
    assert repeated, "no operator appears twice, so nothing is pinned here"

    for record in repeated:
        name = names[records.index(record)]
        assert record.genome_hash[:8] in name, (
            f"{record.origin} appears more than once and {name} does not say "
            f"which one it is")


def test_a_uniquely_named_network_keeps_the_readable_name(records):
    """Only the collisions carry a hash. A list of hashes is not a list anybody
    can read."""
    names = serve_studio.agent_names(records)
    unique = [r for r in records
              if sum(1 for other in records if other.origin == r.origin) == 1]
    assert unique
    for record in unique:
        assert record.genome_hash[:8] not in names[records.index(record)]


def test_the_manifest_lists_one_entry_per_network(tmp_path, monkeypatch, records):
    monkeypatch.setattr(serve_studio, "REGISTRY", tmp_path)
    written, manifest = serve_studio.write(records)

    assert len(written) == len(records)
    assert len({path.name for path in written}) == len(records)

    body = manifest.read_text(encoding="utf-8")
    for path in written:
        assert f'"{path.name}"' in body, f"{path.name} is served but unlisted"


def test_the_optimiser_is_served_but_never_public(tmp_path, monkeypatch, records):
    """It spends the day's whole evaluation budget when poked, and the day buys
    about three evaluations. An endpoint anyone can reach is an endpoint that
    can empty it before the scheduler gets a turn."""
    monkeypatch.setattr(serve_studio, "REGISTRY", tmp_path)
    _written, manifest = serve_studio.write(records)
    body = manifest.read_text(encoding="utf-8")

    assert '"optimizer.hocon"' in body
    optimiser_line = next(line for line in body.splitlines()
                          if "optimizer.hocon" in line)
    assert '"public": false' in optimiser_line, optimiser_line


def test_every_served_network_renders_as_a_real_neuro_san_network(
        tmp_path, monkeypatch, records):
    """Not a mock-up: each file is the genome that earned the score, in the
    HOCON neuro-san serves, with its measured numbers in the description."""
    pytest.importorskip("pyhocon")
    from pyhocon import ConfigFactory

    monkeypatch.setattr(serve_studio, "REGISTRY", tmp_path)
    written, _manifest = serve_studio.write(records)

    by_name = {serve_studio.agent_names(records)[i]: records[i]
               for i in range(len(records))}
    for path in written:
        config = ConfigFactory.parse_file(str(path))
        tools = config["tools"]
        assert tools, f"{path.name} serves no agents"

        record = by_name[path.stem.removeprefix(serve_studio.PREFIX)]
        # The network's own metadata, which is what the UI shows beside its
        # name -- not the front agent's function description, which describes
        # what that one agent does.
        description = config["metadata"]["description"]
        assert record.genome_hash in description, (
            f"{path.name} cannot be matched back to its measurement")
        assert f"{record.tokens:,}" in description
        assert f"{record.fitness:+.4f}" in description


def test_the_write_refuses_a_manifest_that_lost_a_network(
        tmp_path, monkeypatch, records):
    """The guard itself. If naming ever collapses again it must fail loudly
    rather than serve nine networks and call them eleven."""
    monkeypatch.setattr(serve_studio, "REGISTRY", tmp_path)
    monkeypatch.setattr(serve_studio, "agent_names",
                        lambda _records: ["same"] * len(records))

    with pytest.raises(SystemExit, match="refusing to serve"):
        serve_studio.write(records)
