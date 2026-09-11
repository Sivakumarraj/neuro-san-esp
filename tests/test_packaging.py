"""Every third-party module the library imports must be declared as a dependency.

This bug is invisible on a developer machine by construction. A working venv has
whatever was installed for anything else, so an undeclared import keeps working
locally forever; CI is the only place that installs from the dependency list
alone, and it is the only place the mistake shows up. reportlab was undeclared
for several commits and nothing noticed until a test imported the report module.

Reading the imports out of the source is the fix, rather than remembering to
update two places whenever an import is added.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Import name -> distribution name, which differ often enough to matter.
DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "neuro_san": "neuro-san",
    # Pulled in by neuro-san itself; the limiter patches it rather than owning
    # it, so depending on it directly would pin a version neuro-san chooses.
    "langchain_google_genai": None,
}


def imported_modules(package: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in (ROOT / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in sys.stdlib_module_names or name == package:
                    continue
                found.setdefault(name, set()).add(
                    str(path.relative_to(ROOT)))
    return found


def declared() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = list(data["project"]["dependencies"])
    for extra in data["project"].get("optional-dependencies", {}).values():
        requirements.extend(extra)
    names = set()
    for requirement in requirements:
        for separator in (">=", "==", "<", ">", "~=", "["):
            requirement = requirement.split(separator)[0]
        names.add(requirement.strip().lower())
    return names


def test_every_import_in_esp_is_declared():
    available = declared()
    missing = []
    for module, users in sorted(imported_modules("esp").items()):
        distribution = DISTRIBUTION.get(module, module)
        if distribution is None:
            continue
        if distribution.lower() not in available:
            missing.append(f"{module} (as {distribution}), imported by "
                           f"{sorted(users)[0]}")
    assert not missing, (
        "undeclared dependencies -- these work locally and break a clean "
        "install:\n  " + "\n  ".join(missing))


def test_the_report_dependencies_are_runtime_not_dev():
    """`make dossier` is a documented command, not a test helper. Putting
    reportlab in the dev extra would make the shipped container unable to run
    it."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = " ".join(data["project"]["dependencies"]).lower()
    assert "reportlab" in runtime
    assert "matplotlib" in runtime


def test_every_import_in_scripts_and_apps_is_declared():
    """The library is not the only thing that gets run.

    `scripts/` and `apps/` hold the documented commands -- the preflight, the
    offline search, the champion server, the web front end. An undeclared
    import there breaks a quick-start step rather than a test, which is worse:
    the first person to try it is someone following the README on a clean
    machine.
    """
    available = declared()
    missing = []
    for package in ("scripts", "apps"):
        if not (ROOT / package).is_dir():
            continue
        for module, users in sorted(imported_modules(package).items()):
            if module in ("esp", "serve_studio", "serve_champion",
                          "sync_history", "holdout_report"):
                continue        # this project's own modules
            distribution = DISTRIBUTION.get(module, module)
            if distribution is None:
                continue
            if distribution.lower() not in available:
                missing.append(f"{module} (as {distribution}), imported by "
                               f"{sorted(users)[0]}")
    assert not missing, (
        "undeclared dependencies in the documented commands:\n  "
        + "\n  ".join(missing))


def test_every_module_a_make_target_runs_is_declared():
    """The bug this was written for, which no import scan could have caught.

    `make studio` runs `python -m nsflow.run`. nsflow was never declared, and
    it was installed in the environment where the target was written, so it
    worked there and failed on the first clean install with a bare
    ModuleNotFoundError. The dependency was invoked by a Makefile rather than
    imported by a module, so scanning Python for imports could not see it --
    the third time this repository has shipped a package it did not declare,
    and the first time the invocation was not an import.
    """
    import re

    available = declared()
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    missing = []
    for module in re.findall(r"python -m ([A-Za-z_][\w.]*)", makefile):
        top = module.split(".")[0]
        if top in sys.stdlib_module_names or top == "esp":
            continue
        distribution = DISTRIBUTION.get(top, top)
        if distribution is None:
            continue
        if distribution.lower() not in available:
            missing.append(f"{top} (as {distribution}), run as "
                           f"`python -m {module}` by the Makefile")

    assert not missing, (
        "a make target runs a module this project does not declare -- it works "
        "wherever it happens to be installed and fails everywhere else:\n  "
        + "\n  ".join(missing))


def test_the_accelerator_ui_is_an_extra_not_a_runtime_dependency():
    """nsflow bundles a built frontend, speech recognition and graphviz.
    Everything except `make studio` runs without it, so a plain install should
    not pay for it -- but the quick start's own `pip install -e ".[dev]"` has
    to be enough for the documented sequence to work."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = " ".join(data["project"]["dependencies"])
    extras = data["project"]["optional-dependencies"]

    assert "nsflow" not in runtime, "the UI does not belong in a plain install"
    assert any("nsflow" in requirement for requirement in extras.get("studio", []))
    assert any("nsflow" in requirement for requirement in extras.get("dev", [])), (
        "the quick start installs .[dev] and then runs make studio")


def test_the_declared_nsflow_can_run_against_the_declared_neuro_san():
    """A version floor that pip will silently resolve the wrong way.

    nsflow 0.6.x pins `neuro-san<0.7`. This project's floor resolves neuro-san
    to 0.7.x, so an unbounded `nsflow` requirement installs a 0.6.x release
    into an environment it cannot run in -- and the failure surfaces later, as
    an import error inside the UI rather than as a resolution conflict.
    """
    import re

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for extra, requirements in data["project"]["optional-dependencies"].items():
        for requirement in requirements:
            if not requirement.startswith("nsflow"):
                continue
            floor = re.search(r">=\s*(\d+)\.(\d+)", requirement)
            assert floor, f"{extra}: nsflow needs a version floor: {requirement}"
            major, minor = int(floor.group(1)), int(floor.group(2))
            assert (major, minor) >= (0, 7), (
                f"{extra}: nsflow {major}.{minor} pins neuro-san<0.7, which "
                f"this project no longer installs")
