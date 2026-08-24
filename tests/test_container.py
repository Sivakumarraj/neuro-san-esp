"""The image must contain what the documentation tells people to run.

This is a bug that no amount of local testing finds, because everything works
in the checkout. The Dockerfile copied esp/, scripts/ and tests/ but not apps/
or registries/ -- so the image could run the offline search and could not run
the optimiser service, which is the thing SERVING.md and compose.yaml both
describe it as running. `docker compose up -d` was a documented command that
could not work.

Parsing the Dockerfile is crude, and deliberately cheaper than the alternative:
building the image in CI would take minutes on every push to catch a mistake
that is a missing line in a COPY list.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
COMPOSE = (ROOT / "compose.yaml").read_text(encoding="utf-8")


def copied() -> set[str]:
    """Top-level paths the image receives."""
    paths: set[str] = set()
    for line in DOCKERFILE.splitlines():
        if not line.startswith("COPY "):
            continue
        # COPY <sources...> <destination>
        parts = line.split()[1:-1]
        for part in parts:
            paths.add(part.strip("./").split("/")[0])
    return paths


def test_the_service_entry_point_is_in_the_image():
    """apps/ holds run_optimizer.py, which compose runs and the healthcheck
    calls. Without it the container starts and can do nothing."""
    assert "apps" in copied()


def test_the_registries_are_in_the_image():
    """registries/manifest.hocon is the cron schedule -- it *is* the service.
    A server started without it has no periodic interaction to fire."""
    assert "registries" in copied()


def test_every_path_the_compose_commands_need_is_copied():
    """Read the commands out of compose.yaml rather than listing them here, so
    adding a service cannot silently outrun the image."""
    needed = set()
    for command in re.findall(r"command: (.+)", COMPOSE):
        for token in command.split():
            if "/" in token and not token.startswith("-"):
                needed.add(token.strip('"').split("/")[0])
    missing = {path for path in needed if path not in copied()}
    assert not missing, f"compose runs from {missing}, which the image lacks"


def test_the_state_directory_exists_in_the_image():
    """A wake writes state after every candidate. Failing on the first write
    would throw away an evaluation that has already been paid for."""
    assert "mkdir -p /app/state" in DOCKERFILE


def test_the_state_volume_is_named_not_anonymous():
    """The population is weeks of provider budget. An anonymous volume is
    discarded by `docker compose down -v` and by most orchestrators."""
    assert "esp-state:/app/state" in COMPOSE
    assert re.search(r"^volumes:\n  esp-state:", COMPOSE, re.M)


def test_no_key_is_baked_into_the_image():
    """Keys come from the environment at run time. An image is copied around
    and pushed to registries; a key inside one is a key published."""
    for marker in ("GOOGLE_API_KEY=", "AIza", "sk-"):
        assert marker not in DOCKERFILE


def test_the_container_does_not_run_as_root():
    """An evaluation harness executes model-chosen tool calls. A small blast
    radius is cheap to have."""
    assert "USER esp" in DOCKERFILE


def test_the_lint_command_in_the_image_matches_ci():
    """A check service that lints less than CI does gives false confidence."""
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    ci_lint = re.search(r"ruff check ([\w /]+)", workflow).group(1).split()
    image_lint = re.search(r"ruff check ([\w /]+)", COMPOSE).group(1).split()
    assert set(ci_lint) == set(image_lint)
