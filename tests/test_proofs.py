"""The evidence layer has to describe this repository, not an earlier one.

`docs/proofs/` holds the transcripts the dossier prints, and the argument for
printing them is that they were produced by running the thing. That argument
only holds while they still match the tree they describe. Nothing checked, and
they rotted: the tests transcript claimed 114 passing when 158 passed, and the
page annotating every file in the repository was built from a listing missing
seventeen tracked files, `esp/config.py` among them.

A separate failure had the opposite shape. `serving_champion.txt` was captured,
committed, and never added to the index, so the dossier printed a red "Proof not
captured" box over a transcript sitting inches away on disk -- understating the
evidence rather than overstating it, which is the better direction to fail in
and still wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROOFS = ROOT / "docs" / "proofs"

sys.path.insert(0, str(ROOT / "scripts"))

from capture_proofs import ATTESTED, carry_forward, load_index  # noqa: E402


def index() -> list[dict]:
    path = PROOFS / "index.json"
    if not path.exists():
        pytest.skip("no proofs captured in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------- the committed set

def test_the_dossier_reads_its_file_listing_from_git():
    """The annotated-files page is built from `git ls-files` at render time.

    Sourcing it live is what keeps the page honest without a committed
    snapshot that has to be regenerated in lockstep with every added file.
    """
    from esp.report.dossier import Dossier

    paths = Dossier()._tracked_paths()
    assert paths, "the dossier could not obtain a file listing"
    assert "pyproject.toml" in paths
    assert any(p.startswith("esp/") for p in paths)


def test_every_committed_transcript_is_reachable_through_the_index():
    """The dossier finds a proof by name in the index. A transcript that is not
    in it cannot be printed, however real it is."""
    named = {entry["name"] for entry in index()}
    on_disk = {path.stem for path in PROOFS.glob("*.txt")}
    assert on_disk <= named, (
        f"captured but unreachable: {sorted(on_disk - named)}. Add the name to "
        "COMMANDS or ATTESTED in scripts/capture_proofs.py.")


def test_every_index_entry_that_claims_a_capture_has_one():
    """The other direction: an index entry with no transcript behind it would
    have the document print a header over nothing."""
    missing = [entry["name"] for entry in index()
               if not entry.get("skipped")
               and not (PROOFS / f"{entry['name']}.txt").exists()]
    assert not missing, f"index claims transcripts that do not exist: {missing}"


def test_an_attested_proof_declares_how_it_was_obtained():
    """Attestation is a weaker claim than capture, and the document says so out
    loud. An empty provenance string would let it print as though it were not."""
    for entry in index():
        if "attested" in entry:
            assert entry["attested"].strip(), f"{entry['name']} attests nothing"


def test_the_champion_transcript_is_declared_somewhere():
    """The specific regression: it existed, and nothing pointed at it."""
    if not (PROOFS / "serving_champion.txt").exists():
        pytest.skip("champion has not been served in this checkout")
    assert "serving_champion" in ATTESTED


# ------------------------------------------------------- a skip loses nothing

def test_a_skip_carries_the_previous_capture_forward(tmp_path, monkeypatch):
    """Capturing rewrites the index wholesale, so a machine with no key used to
    replace a real capture with a bare skip -- the transcript stayed on disk,
    unreachable, and the page printed it under `exit None`."""
    import capture_proofs
    monkeypatch.setattr(capture_proofs, "PROOF_DIR", tmp_path)
    (tmp_path / "periodic_server.txt").write_text("fired", encoding="utf-8")

    earlier = {"periodic_server": {
        "name": "periodic_server", "command": "python scripts/verify_periodic.py",
        "exit_code": 0, "captured_at": "2026-08-22T13:21:39.231271+00:00",
        "lines": 8}}
    kept = carry_forward("periodic_server", ["ignored"], earlier)

    assert kept["exit_code"] == 0
    assert kept["captured_at"] == "2026-08-22T13:21:39.231271+00:00"
    assert kept["carried_forward"] is True
    assert "skipped" not in kept


def test_a_skip_with_nothing_to_carry_is_recorded_as_a_skip(tmp_path, monkeypatch):
    """An absent proof stays visibly absent. Inventing one would be worse than
    the gap."""
    import capture_proofs
    monkeypatch.setattr(capture_proofs, "PROOF_DIR", tmp_path)
    skipped = carry_forward("periodic_server", ["python", "verify.py"], {})
    assert skipped["skipped"]
    assert "exit_code" not in skipped


def test_a_deleted_transcript_is_not_carried_forward(tmp_path, monkeypatch):
    """The record is only worth keeping while the transcript it points at is."""
    import capture_proofs
    monkeypatch.setattr(capture_proofs, "PROOF_DIR", tmp_path)
    earlier = {"periodic_server": {"name": "periodic_server", "command": "x",
                                   "exit_code": 0, "lines": 8}}
    result = carry_forward("periodic_server", ["python", "verify.py"], earlier)
    assert result["skipped"]


def test_an_unreadable_index_is_not_fatal(tmp_path, monkeypatch):
    """Capturing proofs must not be the thing that fails on a corrupt index."""
    import capture_proofs
    monkeypatch.setattr(capture_proofs, "PROOF_DIR", tmp_path)
    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
    assert load_index() == {}
