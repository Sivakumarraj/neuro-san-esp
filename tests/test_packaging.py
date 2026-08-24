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
