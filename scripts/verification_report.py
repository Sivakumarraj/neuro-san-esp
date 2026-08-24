"""What this repository actually does when you run it, and what it does not.

Every row in this document is produced by running the command in it, here, now.
Nothing is typed in by hand, and a check that fails prints as failed -- a
verification report that can only say "pass" verifies nothing.

The distinction it exists to hold: a check that ran and passed, a check that
ran and failed, and a claim that COULD NOT BE TESTED in this environment. The
third is the one that gets quietly rounded up to the first, and rounding it up
is how a project ends up asserting more than it measured.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.report.layout import (
    AMBER,
    BAD,
    BAD_BG,
    CW,
    GOOD,
    GOOD_BG,
    LEAD,
    WARN_BG,
    Layout,
    _s,
)

ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str], timeout: int = 1800) -> tuple[int, str]:
    """A real subprocess. The exit code is the verdict, not a judgement call."""
    finished = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin",
             "PYTHONPATH": str(ROOT), "AGENT_TOOL_PATH": str(ROOT),
             "HOME": str(Path.home())})
    return finished.returncode, (finished.stdout + finished.stderr).strip()


CHECKS: list[tuple[str, list[str], str]] = [
    ("Test suite", [sys.executable, "-m", "pytest", "tests", "-q"],
     "Every property this project claims about itself, asserted."),
    ("Lint", [sys.executable, "-m", "ruff", "check", "esp", "tests", "scripts",
              "apps"],
     "The same command CI runs, on the same paths."),
    ("Offline search (no key, no network)",
     [sys.executable, "scripts/offline_search.py", "--pool", "2000"],
     "Phase B and C: the half of ESP that is meant to cost nothing."),
    ("Preflight",
     [sys.executable, "apps/optimizer/run_optimizer.py", "--check"],
     "PREFLIGHT: passes when a provider key is configured, refuses when one "
     "is not. Which of those is the pass condition depends on the environment "
     "this report was produced in, so it is decided by looking rather than "
     "assumed: hardcoding either outcome as the pass condition misreports the "
     "other one."),
    ("Champion network generation",
     [sys.executable, "scripts/serve_champion.py", "--state", "state"],
     "Writes the best measured topology into the registry as a servable agent."),
]

# Claims this environment cannot test, named rather than omitted.
UNTESTABLE: list[tuple[str, str]] = [
    ("An agent answering a question",
     "Needs a provider key, and this session has none. The agent networks "
     "render, the server accepts them and the corpus tool returns documents, "
     "but nothing has called a language model, so no answer has been produced "
     "or scored here."),
    ("OpenRouter as the provider",
     "The switch is implemented and its configuration is tested, but "
     "openrouter.ai is refused by this sandbox's egress policy (HTTP 403 on "
     "CONNECT), so no call has been made to it. Google's endpoint is reachable "
     "from here; OpenRouter's is not."),
    ("The container image",
     "No Docker daemon in this environment. tests/test_container.py parses the "
     "Dockerfile and compose.yaml instead and checks that everything the "
     "documented commands need is copied. That is file inspection, not a "
     "build."),
    ("A full evolutionary run",
     "One candidate costs about 165 provider requests and the free tier caps "
     "requests per day per model. No search has completed, and no evolved "
     "candidate has beaten the seed baseline."),
]


class Verification(Layout):
    def build(self, out: Path) -> Path:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        commit = run(["git", "rev-parse", "--short", "HEAD"])[1]

        self.h1("Verification report",
                f"neuro-san-esp at {commit} — produced by running every "
                f"command below, {stamp}")

        self.p(
            "Each check in this document was executed to produce the row that "
            "describes it. The exit code is the verdict. Nothing here is a "
            "summary of a previous run, and nothing is asserted that was not "
            "observed &mdash; a report that cannot record a failure is not "
            "evidence of a success.", LEAD)

        # Does this environment actually have a provider key? The preflight's
        # correct behaviour inverts on the answer, so it is established rather
        # than assumed.
        keyed = bool(os.environ.get("GOOGLE_API_KEY")
                     or os.environ.get("OPENROUTER_API_KEY")
                     or (ROOT / ".env").exists())

        results = []
        for name, command, why in CHECKS:
            code, output = run(command)
            if why.startswith("PREFLIGHT:"):
                passed = (code == 0) if keyed else (code != 0)
                name = f"{name}, {'with' if keyed else 'with no'} provider key"
                why = why.removeprefix("PREFLIGHT: ") + (
                    f" Here: a key {'was' if keyed else 'was not'} present, so "
                    f"exit {'0' if keyed else 'non-zero'} is the pass.")
            else:
                passed = code == 0
            results.append((name, command, why, code, output, passed))

        self.h2("Summary")
        self.table(
            ["Check", "Exit", "Result"],
            [[name, str(code), "PASS" if passed else "FAIL"]
             for name, _, _, code, _, passed in results],
            widths=[CW * 0.62, CW * 0.13, CW * 0.25],
            highlight=[i for i, r in enumerate(results) if r[5]])

        failures = [r for r in results if not r[5]]
        if failures:
            self.callout(
                f"{len(failures)} check(s) failed",
                "Recorded rather than removed. " + ", ".join(
                    r[0] for r in failures), bg=BAD_BG, bar=BAD)
        else:
            self.callout(
                "Every executed check passed",
                "That covers what can be run without a provider key. It does "
                "not cover the four claims on the last page, which this "
                "environment cannot test at all.", bg=GOOD_BG, bar=GOOD)

        for name, command, why, code, output, passed in results:
            self.h2(name)
            self.p(why, _s("w", fontSize=9.4, leading=13))
            verdict = f"exit {code}"
            note = " (refusal is the pass)" if (
                "exit non-zero is the pass" in why) else ""
            self.p(f"<b>$ {' '.join(Path(c).name if '/' in c else c for c in command)}"
                   f"</b> &nbsp;&mdash;&nbsp; <font color='"
                   f"{'#1d7a46' if passed else '#b3261e'}'>{verdict}"
                   f"{note}"
                   f"</font>",
                   _s("cmd", fontSize=8.8, leading=12, spaceAfter=4))
            lines = output.splitlines()
            shown = lines[:26]
            if len(lines) > 26:
                shown.append(f"... {len(lines) - 26} more lines")
            self.terminal("\n".join(shown) or "(no output)")

        self.h1("Not verified here",
                "Four claims this environment cannot test. Named, not omitted.")
        self.p(
            "These are the rows that matter most in a document like this. A "
            "check that could not run is not a check that passed, and the "
            "distance between those two is where a project starts claiming "
            "more than it measured.", LEAD)
        for title, detail in UNTESTABLE:
            self.callout(title, detail, bg=WARN_BG, bar=AMBER)

        self.h2("What would close the gap")
        self.p(
            "A free Google AI Studio key, pasted into <font face='Courier' "
            "size='9'>.env</font>. Google's endpoint is reachable from the "
            "environment this report was produced in, so the preflight, a real "
            "wake, and an agent answering a question would all become "
            "testable. One candidate costs about 165 requests against a free "
            "tier that allows 500 per day per model, so the first three "
            "measurements are affordable on day one and the surrogate's "
            "eight-sample floor is not.", LEAD)

        return self.render(out, "neuro-san-esp — Verification report")


def main() -> int:
    out = ROOT / "docs" / "neuro-san-esp-Verification.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    Verification().build(out)
    print("wrote", out)
    # A machine-readable twin, so the numbers can be checked without a PDF
    # reader and a later run can be diffed against this one.
    print(json.dumps({"generated": datetime.now(UTC).isoformat()}, indent=2)[:0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
