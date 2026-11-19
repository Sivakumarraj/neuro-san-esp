"""The `esp` command: every documented step, the same on Windows, macOS and Linux.

The Makefile sets PYTHONPATH and AGENT_TOOL_PATH with POSIX shell syntax and
needs `make`, so none of its recipes run in PowerShell or cmd. This command sets
the same variables with `os.pathsep`, loads `.env` once, and runs each step
with the interpreter it was started from, so `esp validate` on Windows is the
same check CI runs on Linux. neuro-san-studio ships its `ns` command the same
way (`neuro_san_studio.commands.cli`).

    esp validate          lint, docs, HOCON, the test suite, an offline search
    esp check-key         does the provider accept the key in .env
    esp smoke             live questions to the served network (paid)
    esp loop              the ESP loop (see esp/loop/); offline unless --live
    esp studio            every network in neuro-san's accelerator UI
    esp web               the single-page web front end on port 7860
    esp report NAME       recompute a committed result for $0
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Results recomputed from committed logs, no key and no network.
REPORTS: dict[str, str] = {
    "all-four": "scripts/all_four_report.py",
    "prescriptor": "scripts/prescriptor_report.py",
    "same-model": "scripts/same_model_report.py",
    "model-router": "scripts/model_router_report.py",
    "route": "scripts/route_report.py",
    "holdout": "scripts/holdout_report.py",
    "uncertainty": "scripts/uncertainty_report.py",
    "autopsy": "scripts/autopsy_report.py",
    "ablation": "scripts/selection_ablation.py",
    "figures": "scripts/surrogate_figures.py",
    "cost": "scripts/cost_report.py",
    "bank": "scripts/bank_report.py",
}

SERVED_REGISTRIES = ("registries/optimizer.hocon", "registries/evaluator.hocon")


class Cli:
    """Parses one `esp` command line and runs the steps it names."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        if not (root / "scripts").is_dir():
            raise SystemExit(
                f"esp runs from a clone of neuro-san-esp; {root} has no scripts/ directory. "
                "Clone the repository and run `pip install -e .[dev]` inside it.")

    def environment(self) -> dict[str, str]:
        """This process's environment with .env loaded and the repository on the paths."""
        # Imported here so `esp --help` works before .env is read.
        from esp.config import bootstrap  # pylint: disable=import-outside-toplevel
        bootstrap()
        env = dict(os.environ)
        root = str(self.root)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root if not existing else root + os.pathsep + existing
        env.setdefault("AGENT_TOOL_PATH", root)
        env.setdefault("PYTHONUTF8", "1")       # Windows consoles default to cp1252
        return env

    def run(self, *args: str, env: dict[str, str] | None = None) -> int:
        """Run `python <args>` from the repository root and return its exit code."""
        command = [sys.executable, *args]
        print("$ python " + " ".join(args), flush=True)
        return subprocess.run(command, cwd=self.root, env=env or self.environment(),
                              check=False).returncode

    def steps(self, *steps: tuple[str, ...]) -> int:
        """Run steps in order and stop at the first that fails."""
        env = self.environment()
        for step in steps:
            code = self.run(*step, env=env)
            if code != 0:
                return code
        return 0

    # ------------------------------------------------------------ commands

    def lint(self) -> int:
        """ruff over the code, pymarkdown over the docs."""
        docs = sorted(str(path.relative_to(self.root))
                      for folder in (self.root, self.root / "docs")
                      for path in folder.glob("*.md"))
        return self.steps(("-m", "ruff", "check", "esp", "tests", "scripts", "apps"),
                          ("-m", "pymarkdown", "--config", ".pymarkdownlint.yaml", "scan", *docs))

    def hocon(self) -> int:
        """neuro-san's own validator over every served registry."""
        return self.steps(*(("-m", "neuro_san.client.hocon_validator_cli", path)
                            for path in SERVED_REGISTRIES))

    def test(self, extra: list[str]) -> int:
        """The offline test suite."""
        return self.run("-m", "pytest", "tests", "-q", *extra)

    def offline_check(self) -> int:
        """Phases B and C over the committed measurements, zero model calls."""
        return self.run("scripts/offline_search.py", "--cache", "tests/fixtures/cache",
                        "--pool", "200")

    def validate(self) -> int:
        """Everything CI checks, in the order CI checks it. No key, no network."""
        for step in (self.lint, self.hocon, lambda: self.test([]), self.offline_check):
            code = step()
            if code != 0:
                return code
        return 0

    def studio(self) -> int:
        """Write the measured networks, then open neuro-san's accelerator UI on them."""
        if importlib.util.find_spec("nsflow") is None:
            print('esp studio needs nsflow, neuro-san\'s UI: pip install -e ".[studio]"',
                  file=sys.stderr)
            return 1
        env = self.environment()
        code = self.run("scripts/serve_studio.py", env=env)
        if code != 0:
            return code
        self.run("scripts/check_key.py", "--warn", "--quiet-when-fine", env=env)
        env["AGENT_MANIFEST_FILE"] = str(self.root / "registries" / "studio_manifest.hocon")
        return self.run("-m", "nsflow.run", env=env)

    # ------------------------------------------------------------ parsing

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """The argument parser, kept separate so a test can read its commands."""
        parser = argparse.ArgumentParser(prog="esp", description=__doc__.splitlines()[0])
        sub = parser.add_subparsers(dest="command", required=True)
        for name, text in (("validate", "lint, docs, HOCON, tests and an offline search"),
                           ("lint", "ruff and pymarkdown"),
                           ("hocon", "neuro-san's validator over the served registries"),
                           ("offline-check", "phases B and C over the committed measurements"),
                           ("offline", "phases B and C over a 2,000-candidate pool"),
                           ("check-key", "ask the provider whether the key works"),
                           ("studio", "every network in neuro-san's accelerator UI"),
                           ("web", "the web front end on http://localhost:7860"),
                           ("champion", "print the network the web page serves")):
            sub.add_parser(name, help=text)
        test = sub.add_parser("test", help="the offline test suite")
        test.add_argument("pytest_args", nargs=argparse.REMAINDER)
        smoke = sub.add_parser("smoke", help="live questions to the served network (paid)")
        smoke.add_argument("--questions", type=int, default=None)
        measure = sub.add_parser("measure", help="measure any neuro-san network (paid)")
        measure.add_argument("network")
        measure.add_argument("--tasks", default="meridian")
        loop = sub.add_parser("loop", help="the ESP loop: predictor, prescriptor, repeat")
        loop.add_argument("loop_args", nargs=argparse.REMAINDER)
        report = sub.add_parser("report", help="recompute a committed result for $0")
        report.add_argument("name", choices=sorted(REPORTS))
        return parser

    def main(self, argv: list[str] | None = None) -> int:
        """Dispatch one command line."""
        argv = sys.argv[1:] if argv is None else argv
        if argv[:1] == ["test"]:
            return self.test(argv[1:])
        if argv[:1] == ["loop"]:
            return self.run("-m", "esp.loop", *argv[1:])
        args = self.parser().parse_args(argv)
        command: str = args.command
        if command == "validate":
            return self.validate()
        if command == "lint":
            return self.lint()
        if command == "hocon":
            return self.hocon()
        if command == "offline-check":
            return self.offline_check()
        if command == "offline":
            return self.run("scripts/offline_search.py", "--pool", "2000")
        if command == "check-key":
            return self.run("scripts/check_key.py")
        if command == "smoke":
            extra = [] if args.questions is None else ["--questions", str(args.questions)]
            return self.run("scripts/smoke_live.py", *extra)
        if command == "measure":
            return self.run("-m", "esp.measure", args.network, "--tasks", args.tasks)
        if command == "studio":
            return self.studio()
        if command == "web":
            return self.run("apps/web/serve.py")
        if command == "champion":
            state = os.environ.get("ESP_STATE", "state")
            return self.run("scripts/serve_champion.py", "--state", state)
        if command == "report":
            return self.run(REPORTS[args.name])
        raise SystemExit(f"unknown command {command}")    # argparse already refuses these


def main() -> int:
    """Console-script entry point declared in pyproject.toml."""
    return Cli().main()


if __name__ == "__main__":
    sys.exit(main())
