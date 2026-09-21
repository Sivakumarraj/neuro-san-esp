"""Running on Gemini, Claude or GPT — whichever key is present.

neuro-san picks the client class from the model name and ships classes for all
three, so switching provider is a model name and a key. What that leaves this
project responsible for is the part neuro-san does not do: requiring the key
the configured model actually needs, pacing every provider rather than one, and
refusing to start on a combination that would fail on every call.

The failure this guards against is specific and quiet. A network configured for
`claude-sonnet-5` with only `GOOGLE_API_KEY` set does not fail at startup. It
fails inside every agent, scores every candidate zero, and the cache keeps those
zeros — so the search is taught that good topologies are bad.
"""

from __future__ import annotations

import os

import pytest

from esp.config import (
    KEY_FOR_PROVIDER,
    KEY_NAMES,
    key_name_for,
    key_problem,
    provider_for,
    verify_key,
)

GEMINI = "AQ.EXAMPLE-not-a-real-key-0000000000000000000000000000"
ANTHROPIC = "sk-ant-EXAMPLE-not-a-real-key-000000000000000000000"
OPENAI = "sk-proj-EXAMPLE-not-a-real-key-00000000000000000000"


# ------------------------------------------------- which provider owns a model

@pytest.mark.parametrize(("model", "provider"), [
    ("gemini-3.1-flash-lite", "gemini"),
    ("gemini-3.5-flash", "gemini"),
    ("claude-sonnet-5", "anthropic"),
    ("claude-haiku-4-5", "anthropic"),
    ("claude-opus-5", "anthropic"),
    ("gpt-5-mini", "openai"),
    ("gpt-4o-mini", "openai"),
    ("o3-mini", "openai"),
    ("openrouter/free", "openrouter"),
])
def test_a_model_name_resolves_to_its_provider(model, provider):
    assert provider_for(model) == provider


def test_an_unknown_model_claims_no_provider():
    """Silence beats a guess. A model matched to the wrong provider would ask
    for the wrong key and fail on every call."""
    assert provider_for("llama3") is None
    assert provider_for("") is None
    assert key_name_for("mistral-nemo") is None


def test_every_provider_has_a_key_and_every_key_is_declared():
    for provider, key in KEY_FOR_PROVIDER.items():
        assert key in KEY_NAMES, f"{provider} wants {key}, which is not declared"


@pytest.mark.parametrize(("model", "key"), [
    ("gemini-3.1-flash-lite", "GOOGLE_API_KEY"),
    ("claude-sonnet-5", "ANTHROPIC_API_KEY"),
    ("gpt-5-mini", "OPENAI_API_KEY"),
    ("openrouter/free", "OPENROUTER_API_KEY"),
])
def test_the_right_key_is_demanded_for_the_model(model, key):
    assert key_name_for(model) == key


def test_the_provider_map_agrees_with_neuro_sans_own_registry():
    """Drift caught where it is cheap.

    The map is owned here rather than read from neuro-san at runtime, because
    its HOCON keys are quoted and dotted in ways pyhocon resolves
    inconsistently between versions — and a provider misread at startup is a
    run that fails on every call. The registry is still the source of truth, so
    it is checked.
    """
    pytest.importorskip("pyhocon")
    import neuro_san
    from pyhocon import ConfigFactory

    path = (os.path.dirname(neuro_san.__file__)
            + "/internals/run_context/langchain/llms/default_llm_info.hocon")
    if not os.path.exists(path):
        pytest.skip("neuro-san moved its LLM registry")

    registry = ConfigFactory.parse_file(path)
    mismatched = []
    for raw, entry in dict(registry).items():
        name = str(raw).strip('"')
        if name in ("classes", "default_config"):
            continue
        try:
            theirs = entry.get("class", None)
        except Exception:
            continue        # not a model entry, or a shape pyhocon dislikes
        if not theirs:
            continue
        ours = provider_for(name)
        # Only names this project claims are checked. neuro-san serves ollama,
        # bedrock, azure and nvidia too, and this project does not.
        if ours is not None and ours != theirs:
            mismatched.append(f"{name}: neuro-san says {theirs}, we say {ours}")

    assert not mismatched, (
        "the provider map disagrees with neuro-san's registry:\n  "
        + "\n  ".join(mismatched))


# ---------------------------------------------------------- key shape per key

def test_each_provider_accepts_its_own_key_shape():
    assert key_problem(GEMINI, "GOOGLE_API_KEY") == ""
    assert key_problem(ANTHROPIC, "ANTHROPIC_API_KEY") == ""
    assert key_problem(OPENAI, "OPENAI_API_KEY") == ""


def test_a_key_in_the_wrong_variable_is_caught():
    """The commonest three-key mistake: the right key, the wrong slot. It
    otherwise surfaces as an authentication error naming a provider the person
    was not trying to use."""
    problem = key_problem(OPENAI, "ANTHROPIC_API_KEY")
    assert "sk-ant-" in problem
    assert "wrong variable" in problem


def test_the_placeholder_is_rejected_for_every_provider():
    from esp.config import PLACEHOLDER

    for name in KEY_NAMES:
        assert key_problem(PLACEHOLDER, name), name


def test_no_key_shape_message_contains_the_key():
    """A message that echoed the value would put a live credential into
    terminal scrollback and screen shares."""
    for name, value in (("ANTHROPIC_API_KEY", OPENAI),
                        ("OPENAI_API_KEY", "sk-Zx9Qw"),
                        ("GOOGLE_API_KEY", GEMINI + " ")):
        problem = key_problem(value, name)
        if problem:
            assert value.strip() not in problem, problem


# ------------------------------------------------------- live verification

@pytest.mark.parametrize("name", ["GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
                                  "OPENAI_API_KEY"])
def test_every_provider_has_a_verification_endpoint(name):
    """Listing models is free at all three, which is the point: a key can be
    checked without spending a request from a budget that buys three candidates
    a day."""
    from esp.config import _VERIFY_ENDPOINTS

    assert name in _VERIFY_ENDPOINTS


def test_verification_refuses_a_bad_shape_before_reaching_the_network(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong-prefix-but-long-enough-xx")
    accepted, said = verify_key("ANTHROPIC_API_KEY")
    assert not accepted
    assert "sk-ant-" in said


def test_a_provider_with_no_endpoint_is_reported_as_unchecked(monkeypatch):
    """OpenRouter has no free model list worth relying on. Saying so beats
    guessing, and beats reporting an unchecked key as good."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "0" * 56)
    accepted, said = verify_key("OPENROUTER_API_KEY")
    assert accepted
    assert "not checked" in said


# ------------------------------------------------------------------- pacing

def test_anthropic_and_openai_are_paced_when_their_drivers_are_present():
    """Google had a wrapper and the others did not. An unpaced provider is not
    a smaller version of a paced one: a rate limit comes back through neuro-san
    as an agent error, the candidate scores zero, and the cache keeps it."""
    from esp.eval import ratelimit

    patched = ratelimit.install_others()
    for module_name, class_name in ratelimit._OTHER_PROVIDERS:
        try:
            __import__(module_name)
        except ImportError:
            continue
        assert class_name in patched, f"{class_name} installed but not paced"


def test_installing_pacing_twice_is_harmless():
    from esp.eval import ratelimit

    first = ratelimit.install_others()
    assert ratelimit.install_others() == first


def test_a_missing_driver_is_skipped_rather_than_raising():
    """A run on Gemini must not need the Anthropic package to start."""
    from esp.eval import ratelimit

    original = ratelimit._OTHER_PROVIDERS
    try:
        ratelimit._OTHER_PROVIDERS = (("no_such_module_at_all", "Nope"),)
        assert ratelimit.install_others() == []
    finally:
        ratelimit._OTHER_PROVIDERS = original


# ------------------------------------------- per-model rate limits, measured

def test_the_newest_flash_models_are_paced_slower_than_the_lite_tier():
    """The pacing was one global number for every model.

    14 requests a minute is right for the lite tier and nearly three times too
    fast for the newest flash models, which measured 5. Asking for one
    therefore produced a stream of 429s that the runner scored as candidate
    failures -- so a per-model limit is not a refinement, it is what makes
    those models usable at all.
    """
    from esp.eval.failover import rpm_for

    assert rpm_for("gemini-3.8-flash") < rpm_for("gemini-3.1-flash-lite")
    assert rpm_for("gemini-3.8-flash") <= 5
    # Anything unmeasured keeps the caller's default rather than guessing.
    assert rpm_for("some-model-nobody-measured", default=11) == 11


def test_a_measured_rpm_leaves_headroom():
    """One under the measured figure, so a burst landing on the same second as
    the window rolls does not spend the last slot."""
    from esp.eval.failover import MEASURED_RPM, rpm_for

    for model, measured in MEASURED_RPM.items():
        assert rpm_for(model) < measured, model
        assert rpm_for(model) >= 1, model


def test_the_newest_flash_model_cannot_fund_a_candidate_and_says_so():
    """The reason the default is still a lite model, pinned as arithmetic.

    gemini-3.8-flash is the newest model and it is genuinely available on the
    free tier. It is also capped at roughly 20 requests a day against the 165
    one candidate needs, so it cannot complete a single evaluation. That is a
    fact about budget, not about quality, and the docs say which.
    """
    from esp.eval.failover import DAILY_CAPS, REQUESTS_PER_CANDIDATE

    assert DAILY_CAPS["gemini-3.8-flash"] < REQUESTS_PER_CANDIDATE
    assert DAILY_CAPS["gemini-3.1-flash-lite"] >= REQUESTS_PER_CANDIDATE


def test_the_buckets_use_the_measured_rate_not_the_default():
    from esp.eval import ratelimit
    from esp.eval.failover import rpm_for

    ratelimit._buckets.clear()
    assert ratelimit.bucket_for("gemini-3.8-flash").rpm == rpm_for("gemini-3.8-flash")
    assert ratelimit.bucket_for("gemini-3.1-flash-lite").rpm == rpm_for(
        "gemini-3.1-flash-lite")
    ratelimit._buckets.clear()
