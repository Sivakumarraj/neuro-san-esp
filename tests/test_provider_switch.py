"""Running on a different provider must be configuration, not a rewrite.

neuro-san ships policies for both Google and OpenRouter and resolves the client
class from the model name, so the framework was never the obstacle -- this
repository was. The provider key name, the model tiers and the network default
were all baked into Python, which meant the project could only ever be run
against one account.

`openrouter/free` matters specifically: it is OpenRouter's own free router,
which picks an available free model per request and moves off exhausted ones
itself. That is the same job `esp/eval/failover.py` does by hand for Google.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from esp.eval.failover import models_named

ROOT = Path(__file__).resolve().parent.parent

# The quota metric in a real Google 429 is a URL path, and it is the reason the
# model parser cannot simply look for "something/something".
REAL_GOOGLE_429 = (
    "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel "
    "quota_metric generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests model gemini-3.1-flash-lite limit 500")


# ------------------------------------------------- reading a 429 from either

def test_a_google_model_is_still_read_out_of_a_real_payload():
    assert models_named(REAL_GOOGLE_429) == ["gemini-3.1-flash-lite"]


def test_a_quota_metric_url_is_not_mistaken_for_a_model():
    """Widening the pattern for `vendor/model` matched the metric path first,
    so the service would have retired `com/generate_content_free_tier_requests`
    and left the model that actually ran out on the ladder."""
    assert not [m for m in models_named(REAL_GOOGLE_429) if "/" in m]


@pytest.mark.parametrize("text,expected", [
    ("429 for openrouter/free", "openrouter/free"),
    ("rate limited: meta-llama/llama-3.3-70b-instruct:free",
     "meta-llama/llama-3.3-70b-instruct:free"),
    ("upstream 429 from mistralai/mistral-7b-instruct",
     "mistralai/mistral-7b-instruct"),
])
def test_an_openrouter_model_is_read_out_of_a_429(text, expected):
    """Without this the service records nothing, and the next wake spends real
    calls rediscovering the exhaustion it already found."""
    assert expected in models_named(text)


# --------------------------------------------------------- provider keys

def test_either_provider_key_satisfies_the_preflight(monkeypatch):
    from esp.config import provider_keys

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert provider_keys() == ["OPENROUTER_API_KEY"]

    monkeypatch.setenv("GOOGLE_API_KEY", "y")
    assert set(provider_keys()) == {"GOOGLE_API_KEY", "OPENROUTER_API_KEY"}


def test_no_key_at_all_is_still_a_refusal(monkeypatch):
    from esp.config import provider_keys

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert provider_keys() == []


def test_the_preflight_names_which_key_it_found(monkeypatch):
    """"A key is set" is useless to somebody debugging why the key they pasted
    is not the one being used."""
    from esp.config import key_source

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert "OPENROUTER_API_KEY" in key_source()


# ------------------------------------------------- the model reaches the wire

def render(env: dict[str, str]) -> str:
    """A subprocess, because the model names are read at import."""
    code = ("from esp.genome.seeds import designer_shaped;"
            "g=designer_shaped();print(g.genome_hash());print(g.to_hocon())")
    finished = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT), **env})
    assert finished.returncode == 0, finished.stderr
    return finished.stdout


def test_the_configured_model_is_what_the_network_asks_for():
    out = render({"ESP_DEFAULT_MODEL": "openrouter/free",
                  "ESP_MODEL_TIERS": "openrouter/free"})
    assert '"model_name": "openrouter/free"' in out
    assert "gemini" not in out


def test_changing_the_model_changes_the_genome_hash():
    """The model is part of the genome, so the cache must miss. A fitness
    measured on Gemini does not describe a network running on Llama, and a
    cache that quietly returned the old number would say it did."""
    google = render({}).splitlines()[0]
    openrouter = render({"ESP_DEFAULT_MODEL": "openrouter/free",
                         "ESP_MODEL_TIERS": "openrouter/free"}).splitlines()[0]
    assert google != openrouter


def test_the_openrouter_driver_is_declared_not_lazily_installed():
    """neuro-san lazily pip-installs langchain-openrouter on first use. An
    undeclared dependency is exactly what broke CI when reportlab turned out to
    be missing, so it is an extra rather than a surprise at runtime."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "langchain-openrouter" in pyproject
    assert "openrouter = [" in pyproject


def test_the_env_example_documents_the_switch():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for needle in ("OPENROUTER_API_KEY", "ESP_DEFAULT_MODEL", "openrouter/free"):
        assert needle in example, needle
    assert "sk-" not in example, "no key-shaped string belongs in a committed file"


# ------------------------------- the champion survives a model change

def test_a_model_change_does_not_look_like_an_evolved_candidate():
    """The model is part of the genome, so configuring a different one
    re-hashes every seed. serve_champion matched the committed measurement by
    hash alone, so it concluded the winner was an unreconstructable evolved
    candidate and pointed the reader at ServiceState -- when the actual cause
    was ESP_DEFAULT_MODEL, and the record's `origin` named the seed all along.
    """
    finished = subprocess.run(
        [sys.executable, "scripts/serve_champion.py", "--state", "no-such-state"],
        cwd=ROOT, text=True, capture_output=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT),
             "ESP_DEFAULT_MODEL": "gemini-3.5-flash-lite",
             "ESP_MODEL_TIERS": "gemini-3.5-flash-lite"})

    assert finished.returncode == 0, finished.stderr
    assert "evolved candidate" not in finished.stdout + finished.stderr

    # A seed name, not a specific one. This used to assert "designer_shaped",
    # which coupled the test to whichever seed happened to be winning in
    # results/history.json -- and that file is rewritten by every real search.
    # The first search to change the ranking turned a passing test red without
    # anything being broken. What the test is actually about is that the seed
    # is *named* rather than reported as unreconstructable.
    from esp.genome.seeds import SEEDS

    assert any(name in finished.stdout for name in SEEDS), (
        f"no seed name in output: {finished.stdout[:300]}")
    # and it must not present the old numbers as describing the new model
    assert "NOT what you are about to talk to" in finished.stdout


def test_a_genuinely_evolved_winner_still_reports_honestly():
    """The fix must not swallow the real case: an origin that is not a seed
    cannot be rebuilt, and saying so is the point."""
    from esp.service.state import Evaluated

    sys.path.insert(0, str(ROOT / "scripts"))
    import serve_champion

    evolved = Evaluated("deadbeefdeadbeef", "mut:add_agent", 0.5, 0.8, 1000,
                        3, 2, 1, "", "")

    class _State:
        def best(self): return evolved

    with pytest.raises(SystemExit) as raised:
        serve_champion.champion_genome(_State())
    assert "evolved candidate" in str(raised.value)
