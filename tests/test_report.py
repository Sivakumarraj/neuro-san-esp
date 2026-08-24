"""The documents, and the one property that makes them worth reading.

A report is only evidence if it cannot quietly describe something that did not
happen. Two failures of that kind have already occurred in this project: the
run report once rendered "evolution did not beat the baseline" when no search
had run at all, and echoed the seed's own numbers under "best evolved". Both
are stronger claims than the truth. The tests here pin the honesty properties,
not the layout.
"""

from __future__ import annotations

import json

import pytest

from esp.report.build import Report
from esp.report.dossier import Dossier

HISTORY = {
    "real_evaluations": 3,
    "surrogate_evaluations": 0,
    "records": [
        {"generation": 0, "origin": "seed:designer_shaped",
         "genome_hash": "459ac1a66d925b0c", "accuracy": 0.82, "tokens": 278532,
         "agents": 4, "depth": 2, "fitness": 0.7868, "predicted": None},
        {"generation": 0, "origin": "seed:solo",
         "genome_hash": "bb3340f7f3c91f05", "accuracy": 0.82, "tokens": 396378,
         "agents": 1, "depth": 1, "fitness": 0.7602, "predicted": None},
    ],
    "surrogate_quality": [],
    "pareto": [],
}


@pytest.fixture
def results(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps(HISTORY), encoding="utf-8")
    (tmp_path / "figures").mkdir()
    return tmp_path


def text_of(pdf_path) -> str:
    """Crude, deliberately. Reading the rendered text back is what proves the
    claim reached the page rather than only the data structure."""
    from pypdf import PdfReader
    return "\n".join(page.extract_text() for page in PdfReader(str(pdf_path)).pages)


# ------------------------------------------------------------ the run report


def test_a_run_with_no_search_does_not_claim_a_search_lost(results, tmp_path):
    """"Evolution did not beat the baseline" and "no search was run" are
    different claims. The report made the first one for a while, which is the
    kind of quiet overstatement this project exists to avoid."""
    out = Report(results).build(tmp_path / "r.pdf")
    body = text_of(out)
    assert "No search was run" in body
    assert "did not beat the baseline" not in body


def test_the_seed_numbers_are_not_reprinted_as_an_evolved_result(results, tmp_path):
    out = Report(results).build(tmp_path / "r.pdf")
    assert "no search ran" in text_of(out)


def test_a_real_search_that_lost_says_so(results, tmp_path):
    """The other side of the same property: when a search genuinely ran and
    found nothing better, that must be reported as a negative result and not
    softened into "no search"."""
    history = json.loads((results / "history.json").read_text())
    history["surrogate_evaluations"] = 2000
    history["records"].append(
        {"generation": 1, "origin": "split_agent", "genome_hash": "c" * 16,
         "accuracy": 0.70, "tokens": 300000, "agents": 3, "depth": 2,
         "fitness": 0.60, "predicted": 0.79})
    (results / "history.json").write_text(json.dumps(history), encoding="utf-8")

    body = text_of(Report(results).build(tmp_path / "r.pdf"))
    assert "did not beat the baseline" in body
    assert "No search was run" not in body


# --------------------------------------------------------------- the dossier


def test_a_missing_proof_is_declared_missing(results, tmp_path):
    """The property that makes the dossier evidence rather than prose. If a
    transcript was not captured, the page must say so -- filling the gap with a
    description is how an unverified claim gets into a report."""
    empty = tmp_path / "no-proofs"
    empty.mkdir()
    dossier = Dossier(results, proofs_dir=empty)
    body = text_of(dossier.build(tmp_path / "d.pdf"))
    assert "Proof not captured" in body


def test_a_captured_proof_is_printed_verbatim(results, tmp_path):
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    (proofs / "tests.txt").write_text("80 passed in 2.10s", encoding="utf-8")
    (proofs / "index.json").write_text(json.dumps(
        [{"name": "tests", "command": "pytest tests -q", "exit_code": 0,
          "lines": 1}]), encoding="utf-8")

    body = text_of(Dossier(results, proofs_dir=proofs).build(tmp_path / "d.pdf"))
    assert "80 passed" in body


def test_a_failing_proof_is_still_shown(results, tmp_path):
    """A proof that fails is a proof. Hiding it would leave the document
    describing a repository that works better than this one does."""
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    (proofs / "lint.txt").write_text("E501 line too long", encoding="utf-8")
    (proofs / "index.json").write_text(json.dumps(
        [{"name": "lint", "command": "ruff check esp", "exit_code": 1,
          "lines": 1}]), encoding="utf-8")

    body = text_of(Dossier(results, proofs_dir=proofs).build(tmp_path / "d.pdf"))
    assert "exit 1" in body
    assert "line too long" in body


def test_an_attested_proof_reaches_the_page(results, tmp_path):
    """Some evidence cannot be produced by one subprocess: the champion
    transcript is a server, a browser and curl in one sitting. It sat on disk
    while the page rendered "Proof not captured" over it, because nothing had
    put it in the index."""
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    (proofs / "serving_champion.txt").write_text(
        "champion: designer_shaped (459ac1a66d925b0c)", encoding="utf-8")
    (proofs / "index.json").write_text(json.dumps(
        [{"name": "serving_champion", "command": "(by hand from a live session)",
          "attested": "by hand from a live session", "lines": 1}]),
        encoding="utf-8")

    body = text_of(Dossier(results, proofs_dir=proofs).build(tmp_path / "d.pdf"))
    assert "designer_shaped" in body
    assert "captured out of band" in body


def test_a_carried_forward_proof_reaches_the_page(results, tmp_path):
    """A capture on a machine with no budget must not delete the transcript it
    cannot reproduce. It is kept and printed."""
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    (proofs / "periodic_server.txt").write_text(
        "scheduled fire observed  user_id=system", encoding="utf-8")
    (proofs / "index.json").write_text(json.dumps(
        [{"name": "periodic_server", "command": "python scripts/verify_periodic.py",
          "exit_code": 0, "captured_at": "2026-08-22T13:21:39.231271+00:00",
          "lines": 1, "carried_forward": True}]), encoding="utf-8")

    body = text_of(Dossier(results, proofs_dir=proofs).build(tmp_path / "d.pdf"))
    assert "user_id=system" in body
    assert "from an earlier build" in body


# The header above a transcript is where a proof could overclaim, so it is
# checked directly rather than through the crude text of a 22-page PDF.

def test_a_harness_captured_proof_says_exit_zero():
    header = Dossier._provenance(
        {"name": "tests", "command": "pytest tests -q", "exit_code": 0})
    assert "exit 0" in header
    assert "out of band" not in header
    assert "earlier build" not in header


def test_an_attested_proof_never_claims_an_exit_code():
    """`exit 0` means this build watched the command succeed. Nothing watched
    a browser session succeed, so the header must not say it did."""
    header = Dossier._provenance(
        {"name": "serving_champion", "command": "(by hand from a live session)",
         "attested": "by hand from a live session"})
    assert "captured out of band" in header
    assert "exit" not in header


def test_a_carried_forward_proof_is_dated_and_flagged():
    """It really did exit 0 -- on another day, on another machine. Both halves
    of that belong in the header."""
    header = Dossier._provenance(
        {"name": "periodic_server", "command": "python scripts/verify_periodic.py",
         "exit_code": 0, "captured_at": "2026-08-22T13:21:39.231271+00:00",
         "carried_forward": True})
    assert "exit 0" in header
    assert "from an earlier build" in header
    assert "2026-08-22" in header
    assert "not re-run here" in header


def test_the_file_page_counts_what_it_leaves_out(results, tmp_path):
    """The annotated-files page states both halves of its own coverage.

    A count of annotated files that silently drops the unannotated ones
    overstates the page on the one page a reader can check against
    `git ls-files` by hand.
    """
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    (proofs / "index.json").write_text(json.dumps(
        [{"name": "tree", "command": "git ls-files", "exit_code": 0,
          "lines": 3}]), encoding="utf-8")

    dossier = Dossier(results, proofs_dir=proofs)
    tracked = dossier._tracked_paths()
    annotated = [p for p in tracked if dossier._annotated(p)]
    omitted = len(tracked) - len(annotated)

    body = text_of(dossier.build(tmp_path / "d.pdf"))
    assert "Nothing omitted" not in body
    assert f"{len(annotated)} tracked files, plus {omitted} package markers" in body


def test_a_skipped_proof_is_declared_missing_even_with_a_transcript_on_disk(
        results, tmp_path):
    """The transcript can outlive the capture that made it -- a clone has the
    committed file but an index that never ran here. A skip is not a capture,
    and rendering the leftover would head it `exit None`."""
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    (proofs / "periodic_server.txt").write_text("fired once", encoding="utf-8")
    (proofs / "index.json").write_text(json.dumps(
        [{"name": "periodic_server", "command": "python scripts/verify_periodic.py",
          "skipped": "no provider key in this environment"}]), encoding="utf-8")

    body = text_of(Dossier(results, proofs_dir=proofs).build(tmp_path / "d.pdf"))
    assert "Proof not captured" in body
    assert "exit None" not in body
