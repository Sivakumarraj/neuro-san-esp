"""The documentation must not drift away from the repository it describes.

Numbers in a README rot silently. This one has already claimed 62 tests when
there were 80, and 80 when there were 114 -- harmless on its own, but a document
whose easily-checkable facts are wrong earns no trust for the facts that are
harder to check, and this project's whole argument is that its claims are
measured.

Only claims that can be checked mechanically are checked here. Prose is not
tested, and pretending otherwise would be theatre.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

# Counts in the README are written out in words, because that is how a sentence
# reads. The checks below need them as numbers.
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "thirteen": 13}
WORDS_TO_TEXT = {value: word for word, value in WORDS.items()}


def test_every_make_target_the_readme_mentions_exists():
    """A documented command that does not exist is worse than an undocumented
    one: the reader assumes they got it wrong."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-z][\w-]*):", makefile, re.M))
    mentioned = set(re.findall(r"^make ([a-z][\w-]*)", README, re.M))
    missing = mentioned - targets
    assert not missing, f"README documents make targets that do not exist: {missing}"


def test_every_compose_service_the_readme_runs_exists():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    services = set(re.findall(r"^  ([a-z][\w-]*):$", compose, re.M))
    mentioned = set(re.findall(r"docker compose run --rm ([a-z][\w-]*)", README))
    missing = mentioned - services
    assert not missing, f"README runs compose services that do not exist: {missing}"


def test_every_repository_file_the_readme_points_at_exists():
    """Paths in backticks that look like files, checked. A broken pointer in the
    first document a reader opens is the cheapest possible thing to get right."""
    candidates = set(re.findall(r"`([\w./-]+\.(?:py|md|hocon|yaml|toml|pdf))`", README))
    candidates |= set(re.findall(r"\]\(([\w./-]+\.md)\)", README))
    missing = [name for name in candidates
               if "/" in name and not (ROOT / name).exists()]
    assert not missing, f"README points at files that do not exist: {missing}"


def test_dotenv_is_gitignored():
    """The file this project tells people to paste a key into must never be
    committable. This is the single highest-consequence line in .gitignore."""
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored]


def test_the_example_env_holds_no_real_key():
    """It is committed, so anything that looks like a credential in it is a
    published credential."""
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "AIza" not in example, "that looks like a real Google API key"
    assert "sk-" not in example
    assert "paste-your-key-here" in example


def test_git_does_not_track_a_dotenv():
    """Belt and braces: .gitignore only helps for a file that was never added.
    This fails if one was force-added at some point in the past."""
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    assert ".env" not in tracked


def test_the_readme_results_table_matches_the_committed_measurements():
    """The numbers in the README's headline table, checked against the run that
    produced them.

    This is the drift that actually happened. The README described three seed
    topologies, an untrained Predictor and no completed search, while
    `results/history.json` beside it held eleven real evaluations and an
    evolved candidate that beat every seed. Both were committed. A reader had
    no way to know which one the repository meant.
    """
    history = json.loads(
        (ROOT / "results" / "history.json").read_text(encoding="utf-8"))
    records = {record["genome_hash"]: record for record in history["records"]}

    results = README[README.index("## Results"):README.index("## Limitations")]
    rows = [line for line in results.splitlines()
            if line.startswith("|") and "---" not in line][1:]
    assert rows, "the README's results table has gone"

    for row in rows:
        cells = [cell.strip(" *`") for cell in row.strip("|").split("|")]
        origin = cells[0].split(" ")[0].split("—")[0].strip(" *`")
        accuracy, tokens, agents, fitness = cells[1:5]
        matching = [r for r in records.values() if r["origin"] == origin]
        assert matching, f"the README reports {origin!r}, which was never measured"
        assert any(
            f"{r['accuracy']:.4f}" == accuracy
            and f"{r['tokens']:,}" == tokens
            and str(r["agents"]) == agents
            and f"{r['fitness']:.4f}" == fitness
            for r in matching), (
            f"the README's row for {origin!r} matches no measured candidate: "
            f"{accuracy} {tokens} {agents} {fitness}")


def test_the_readme_does_not_overstate_how_much_was_measured():
    history = json.loads(
        (ROOT / "results" / "history.json").read_text(encoding="utf-8"))
    measured = history["real_evaluations"]
    claimed = {int(n) for n in re.findall(r"(\w+) networks measured", README)
               if n.isdigit()}
    claimed |= {WORDS[n.lower()] for n in re.findall(r"(\w+) networks measured", README)
                if n.lower() in WORDS}
    assert claimed == {measured}, (
        f"the README claims {claimed or 'nothing'} networks measured; "
        f"results/history.json records {measured}")


def test_the_readme_reports_the_surrogate_quality_the_search_actually_used():
    """The flattering number is the one computed afterwards over the whole
    population. The README has to carry the one the search ran on."""
    history = json.loads(
        (ROOT / "results" / "history.json").read_text(encoding="utf-8"))
    reports = history["surrogate_quality"]
    assert reports, "nothing recorded, so nothing to check"
    for report in reports:
        shown = f"{abs(report['spearman']):.3f}"
        # The README writes a negative correlation with a typographic minus,
        # which is the right character for prose and not the one Python emits.
        assert shown in README.replace("\N{MINUS SIGN}", "-"), (
            f"the search measured spearman={report['spearman']:+.3f} at "
            f"{report['samples']} samples and the README does not say so")


def test_the_documented_predictor_features_are_the_real_ones():
    """Asked in review, answered in docs/FINDINGS.md, and therefore able to
    drift away from the code that computes them."""
    from esp.surrogate.predictor import FEATURE_NAMES

    findings = (ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")
    # Anchored on the feature subsection, not on "What the Predictor is". That
    # section now also carries a table separating Predictor from fitness from
    # prescription, and a table of per-objective rank correlations -- both of
    # which name things in backticks that are not features. Scraping every
    # table under the heading made this check fail on the documentation it
    # exists to protect.
    section = findings[findings.index("### The feature set"):]
    section = section[:section.index("\n## ", 1)]

    # Only the feature table counts. Backticked names elsewhere in the prose
    # are functions and thresholds, not features, and treating them as claims
    # made this check fail on its own explanation.
    table = [line for line in section.splitlines()
             if line.startswith("|") and "---" not in line]
    documented = set()
    for row in table:
        documented |= set(re.findall(r"`(\w+)`", row))

    assert documented == set(FEATURE_NAMES), (
        f"the documented feature table and the Predictor disagree -- "
        f"omitted: {set(FEATURE_NAMES) - documented or 'none'}, "
        f"invented: {documented - set(FEATURE_NAMES) or 'none'}")


def test_the_documented_predictor_thresholds_are_the_real_ones():
    from esp.surrogate.predictor import MIN_SAMPLES

    findings = (ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")
    section = findings[findings.index("## What the Predictor is"):]
    assert WORDS_TO_TEXT[MIN_SAMPLES] in section, (
        f"the Predictor needs {MIN_SAMPLES} samples and the section does not "
        f"say so")

    from esp.surrogate.predictor import FEATURE_NAMES
    assert WORDS_TO_TEXT[len(FEATURE_NAMES)] in section, (
        f"the Predictor reads {len(FEATURE_NAMES)} features and the section "
        f"does not say so")


def test_no_committed_file_holds_a_credential_shaped_string():
    """The whole tree, not just .env.example.

    A key-shaped literal was committed as a test fixture while these checks
    watched the one file they were told to watch. The fixtures were written by
    copying a real key, which is the obvious thing to reach for and puts a live
    credential into git history -- where removing it means rewriting history and
    rotating the key anyway.

    Anything genuinely key-shaped in a committed file has to say it is not one.
    """
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()

    # Shapes, not values: two Google formats and OpenRouter's.
    shapes = (
        re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
        re.compile(r"AQ\.[0-9A-Za-z_\-]{30,}"),
        re.compile(r"sk-or-v1-[0-9a-f]{40,}"),
        re.compile(r"sk-[A-Za-z0-9]{40,}"),
    )
    # A fixture has to announce itself. Anything matching a shape and saying
    # none of these is treated as real.
    disclaimers = ("EXAMPLE", "example", "not-a-real", "paste-your",
                   "fake", "FAKE", "xxxx", "XXXX")

    offences: list[str] = []
    for name in tracked:
        path = ROOT / name
        if not path.is_file() or path.suffix in (".png", ".pdf", ".db"):
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for shape in shapes:
            for hit in shape.findall(body):
                if any(word in hit for word in disclaimers):
                    continue
                offences.append(f"{name}: {hit[:12]}... ({len(hit)} chars)")

    assert not offences, (
        "credential-shaped strings in committed files -- if any of these is "
        "real it is now published and must be rotated:\n  "
        + "\n  ".join(offences))
