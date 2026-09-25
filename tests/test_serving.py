"""Serving a measured network to a person, on whichever provider holds the key.

What these pin: a served network keeps the topology that earned its score, a
person gets a full answer instead of the bare value the scorer needs, and a
deployment holding only a Claude or GPT key can serve a champion measured on
Gemini without pretending the champion was measured there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from esp.config import DEFAULT_LADDERS, cost_tier, provider_for
from esp.eval import measurements
from esp.eval.tasks import TASKS, score
from esp.genome.seeds import ANSWER_STYLE
from esp.serving import (
    FULL_ANSWER_STYLE,
    SHOWCASE,
    conversational,
    display_question,
    graded,
    presentable,
    retarget,
)

ROOT = Path(__file__).resolve().parent.parent
ANTHROPIC_KEY = "sk-ant-EXAMPLE-not-a-real-key-000000000000000000000"
OPENAI_KEY = "sk-proj-EXAMPLE-not-a-real-key-00000000000000000000"
CHAMPION = measurements.best()


# ------------------------------------------------------------- cost tiers

@pytest.mark.parametrize("model,tier", [
    ("gemini-3.1-flash-lite", 0),
    ("gemini-3.5-flash-lite", 0),
    ("gemini-3.5-flash", 1),
    ("claude-haiku-4-5", 0),
    ("claude-sonnet-5", 1),
    ("claude-opus-5", 1),
    ("gpt-5-mini", 0),
    ("gpt-5", 1),
    ("some-model-nobody-knows", 0),
])
def test_a_model_is_placed_on_its_own_familys_rung(model, tier):
    assert cost_tier(model) == tier


def test_every_ladder_is_cheap_then_strong_within_one_provider():
    for provider, (cheap, strong) in DEFAULT_LADDERS.items():
        assert provider_for(cheap) == provider_for(strong) == provider, provider
        if provider != "openrouter":        # one router, both rungs
            assert (cost_tier(cheap), cost_tier(strong)) == (0, 1), provider


def test_every_ladder_model_is_one_neuro_san_can_resolve():
    """A name neuro-san does not know fails inside every agent, not at start.

    That is a whole paid candidate scored zero, so the ladders are checked
    against the registry that will actually resolve them.
    """
    pytest.importorskip("pyhocon")
    import neuro_san
    from pyhocon import ConfigFactory

    path = (os.path.dirname(neuro_san.__file__)
            + "/internals/run_context/langchain/llms/default_llm_info.hocon")
    if not os.path.exists(path):
        pytest.skip("neuro-san moved its LLM registry")
    known = {str(name).strip('"') for name in ConfigFactory.parse_file(path)}
    missing = [model for provider, ladder in DEFAULT_LADDERS.items()
               if provider != "openrouter" for model in ladder if model not in known]
    assert not missing, f"neuro-san's registry does not resolve: {missing}"


# ------------------------------------------------------ retarget, converse

def test_retargeting_keeps_the_topology_and_the_promotion():
    """The finding is *which* agent got the stronger model. Keep it."""
    moved = retarget(CHAMPION.genome, "claude-haiku-4-5", "claude-sonnet-5")

    assert moved.top == CHAMPION.genome.top
    assert sorted(moved.agents) == sorted(CHAMPION.genome.agents)
    for name, agent in CHAMPION.genome.agents.items():
        assert moved.agents[name].tools == agent.tools
        assert moved.agents[name].can_search == agent.can_search
        assert moved.agents[name].instructions == agent.instructions
    assert moved.agents[CHAMPION.genome.top].model == "claude-sonnet-5"
    assert moved.default_model == "claude-haiku-4-5"


@pytest.mark.parametrize("record", measurements.load(), ids=lambda r: r.genome_hash[:8])
def test_every_measured_promotion_survives_retargeting(record):
    """Checked on the whole population, because one winner promoted its router
    from gemini-3.1-flash-lite to gemini-3.5-flash-lite -- cheap to cheap -- and
    a rule that looked only at the rung served it with no promotion at all."""
    moved = retarget(record.genome, "claude-haiku-4-5", "claude-sonnet-5")
    default = record.genome.default_model
    for name, agent in record.genome.agents.items():
        if agent.model is None:
            assert moved.agents[name].model is None
        elif agent.model != default:
            assert moved.agents[name].model == "claude-sonnet-5", (
                f"{name} was moved off {default} to {agent.model} by the search "
                "and lost that promotion when served on Claude")


def test_a_demotion_below_the_default_stays_a_demotion():
    genome = CHAMPION.genome.clone()
    genome.default_model = "gemini-3.5-flash"
    worker = next(n for n in genome.agents if n != genome.top)
    genome.agents[worker].model = "gemini-3.1-flash-lite"
    moved = retarget(genome, "claude-haiku-4-5", "claude-sonnet-5")
    assert moved.default_model == "claude-sonnet-5"
    assert moved.agents[worker].model == "claude-haiku-4-5"


def test_retargeting_leaves_the_measured_genome_untouched():
    before = CHAMPION.genome.canonical()
    retarget(CHAMPION.genome, "claude-haiku-4-5", "claude-sonnet-5")
    conversational(CHAMPION.genome)
    assert CHAMPION.genome.canonical() == before


def test_only_the_front_mans_reply_format_changes():
    talking = conversational(CHAMPION.genome)
    top = talking.agents[talking.top]

    assert ANSWER_STYLE not in top.instructions
    assert FULL_ANSWER_STYLE in top.instructions
    for name, agent in CHAMPION.genome.agents.items():
        if name != talking.top:
            assert talking.agents[name].instructions == agent.instructions


def test_a_network_without_the_terse_line_still_gets_the_full_answer_line():
    genome = CHAMPION.genome.clone()
    top = genome.agents[genome.top]
    top.instructions = top.instructions.replace(ANSWER_STYLE, "")
    assert FULL_ANSWER_STYLE in conversational(genome).agents[genome.top].instructions


def test_same_provider_is_served_as_measured_apart_from_the_reply_format():
    served = presentable(CHAMPION.genome)       # suite runs on the Gemini default
    assert not served.retargeted
    assert served.genome.default_model == CHAMPION.genome.default_model
    assert "not been measured" not in served.note(CHAMPION.genome.default_model)


def _in_subprocess(code: str, **env: str) -> str:
    environment = {**os.environ, "ESP_NO_DOTENV": "1", "PYTHONPATH": str(ROOT), **env}
    for name in ("ESP_MODEL_TIERS", "ESP_DEFAULT_MODEL", "ESP_PROVIDER",
                 "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        if name not in env:
            environment.pop(name, None)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=environment, cwd=ROOT, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_a_claude_deployment_serves_the_gemini_champion_on_claude_and_says_so():
    out = json.loads(_in_subprocess(
        "import json\n"
        "from esp.eval import measurements\n"
        "from esp.serving import presentable\n"
        "g = measurements.best().genome\n"
        "s = presentable(g)\n"
        "print(json.dumps({'retargeted': s.retargeted, 'provider': s.provider,\n"
        "  'models': sorted({a.model for a in s.genome.agents.values() if a.model}\n"
        "                   | {s.genome.default_model}),\n"
        "  'note': s.note(g.default_model)}))",
        ANTHROPIC_API_KEY=ANTHROPIC_KEY))

    assert out["retargeted"] is True
    assert out["provider"] == "anthropic"
    assert out["models"] == ["claude-haiku", "claude-sonnet"], (
        "a Claude key alone should serve on neuro-san's version-free aliases")
    assert "has not been measured" in out["note"], (
        "a retargeted network must never be presented as the measured one")


# --------------------------------------------------------------- showcase

def test_the_showcase_is_four_distinct_questions_of_rising_difficulty():
    assert len(SHOWCASE) == 4
    assert len({task.task_id for task in SHOWCASE}) == 4
    hops = [task.hops for task in SHOWCASE[:3]]
    assert hops == sorted(hops) and hops[0] == 1
    assert hops[2] == max(task.hops for task in TASKS), "the deepest chain is shown"
    assert "penalty" in SHOWCASE[3].question.lower(), "one question calculates"


def test_the_showcase_leaves_out_what_nearly_every_network_fails():
    """Full-corpus aggregation is the benchmark's hardest case, not a demo."""
    assert not any(task.question.lower().startswith(("across all", "which client holds",
                                                     "which contract reference appears"))
                   for task in SHOWCASE)


def test_a_question_is_shown_without_the_benchmarks_format_instruction():
    numeric = next(task for task in TASKS if "number only" in task.question)
    assert "number only" not in display_question(numeric)
    assert display_question(numeric).endswith("?")


def test_a_displayed_question_is_still_graded_against_the_benchmark():
    for task in SHOWCASE:
        assert graded(display_question(task)) is task
        assert graded(task.question) is task
    assert graded("What is the capital of France?") is None


def test_a_full_answer_containing_the_value_is_still_scored_correct():
    """The scorer matches containment, so explaining does not cost the grade."""
    task = SHOWCASE[3]
    explained = (f"The total penalty owed is {task.answer}. INC-4401 affected a "
                 "contract whose late-delivery rate was applied to the hours late.")
    assert score(task.answer, explained)


# ------------------------------------------------------------------ studio

def test_the_studio_offers_the_showcase_as_one_click_prompts(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import scripts.serve_studio as studio

    monkeypatch.setattr(studio, "REGISTRY", tmp_path)
    written, _ = studio.write(measurements.load()[:2], include_optimizer=False)

    from pyhocon import ConfigFactory
    for path in written:
        config = ConfigFactory.parse_file(str(path))
        assert list(config["metadata"]["sample_queries"]) == [
            display_question(task) for task in SHOWCASE]
        assert "Answer in full." in path.read_text()
        assert "Reply with the answer alone" not in path.read_text()


# ------------------------------------------------------------- .env order

def test_a_provider_chosen_only_in_dotenv_reaches_every_module(tmp_path):
    """Five entry points imported the model settings before loading .env.

    A machine configured for Claude in .env alone rendered Gemini networks.
    Checked the way a person hits it: the package beside a .env, a fresh
    interpreter, nothing exported, and the first import being the module that
    fixes the model -- which is exactly the order that used to lose.
    """
    import shutil

    shutil.copytree(ROOT / "esp", tmp_path / "esp",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-ant-EXAMPLE-not-a-real-key-000000000000000000000\n"
        "ESP_PROVIDER=anthropic\n", encoding="utf-8")
    environment = {k: v for k, v in os.environ.items()
                   if k not in ("ESP_NO_DOTENV", "ESP_DEFAULT_MODEL", "ESP_MODEL_TIERS",
                                "ANTHROPIC_API_KEY")}
    environment["PYTHONPATH"] = str(tmp_path)
    done = subprocess.run(
        [sys.executable, "-c",
         "from esp.genome.definition import DEFAULT_MODEL, MODEL_TIERS\n"
         "print(DEFAULT_MODEL, ','.join(MODEL_TIERS))"],
        capture_output=True, text=True, env=environment, cwd=tmp_path, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["claude-haiku", "claude-haiku,claude-sonnet"]


def test_esp_no_dotenv_really_turns_loading_off(tmp_path):
    import shutil

    shutil.copytree(ROOT / "esp", tmp_path / "esp",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / ".env").write_text("ESP_DEFAULT_MODEL=claude-haiku-4-5\n", encoding="utf-8")
    environment = {k: v for k, v in os.environ.items()
                   if k not in ("ESP_DEFAULT_MODEL", "ESP_MODEL_TIERS")}
    environment.update(PYTHONPATH=str(tmp_path), ESP_NO_DOTENV="1")
    done = subprocess.run(
        [sys.executable, "-c",
         "from esp.genome.definition import DEFAULT_MODEL\nprint(DEFAULT_MODEL)"],
        capture_output=True, text=True, env=environment, cwd=tmp_path, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "gemini-3.1-flash-lite"


def test_the_package_loads_dotenv_before_any_setting_is_read():
    """The fix itself: `import esp` runs bootstrap before anything else."""
    source = (ROOT / "esp" / "__init__.py").read_text(encoding="utf-8")
    assert "_bootstrap()" in source


def test_the_ladder_follows_the_default_models_provider():
    out = _in_subprocess(
        "from esp.genome.definition import MODEL_TIERS\nprint(','.join(MODEL_TIERS))",
        ESP_DEFAULT_MODEL="gpt-5-mini")
    assert out == "gpt-5-mini,gpt-5.5", (
        "an OpenAI default with a Gemini ladder hands agents a model the run "
        "holds no key for")


def test_an_explicit_ladder_still_wins():
    out = _in_subprocess(
        "from esp.genome.definition import MODEL_TIERS\nprint(','.join(MODEL_TIERS))",
        ESP_DEFAULT_MODEL="claude-haiku-4-5",
        ESP_MODEL_TIERS="claude-haiku-4-5,claude-opus-5")
    assert out == "claude-haiku-4-5,claude-opus-5"


def test_a_newer_gemini_default_is_never_promoted_to_an_older_model():
    out = _in_subprocess(
        "from esp.genome.definition import MODEL_TIERS\nprint(','.join(MODEL_TIERS))",
        ESP_DEFAULT_MODEL="gemini-3.8-flash")
    assert out == "gemini-3.5-flash-lite,gemini-3.8-flash", (
        "the Gemini ladder ignored the chosen model, so a router promoted from "
        "gemini-3.8-flash got gemini-3.5-flash")


# Stand-ins for a release newer than the installed neuro-san. Real names go
# stale: claude-opus-5-5 was one until neuro-san 0.7.5 listed it, and the tests
# then failed on the correct behaviour.
_UNRELEASED = [("claude-opus-99", "anthropic"), ("gpt-99", "openai"),
               ("gemini-99-flash", "gemini")]


@pytest.mark.parametrize("model, provider", _UNRELEASED)
def test_a_model_newer_than_neuro_san_is_passed_to_its_provider(model, provider, monkeypatch):
    """A bare name neuro-san does not list fails inside every agent."""
    from neuro_san.internals.run_context.langchain.llms.default_llm_factory import (
        DefaultLlmFactory,
    )

    from esp.config import KEY_FOR_PROVIDER, llm_config
    monkeypatch.setenv(KEY_FOR_PROVIDER[provider], "not-a-real-key")
    config = llm_config(model)
    assert config == {"class": provider, "model_name": model}
    factory = DefaultLlmFactory()
    factory.load()
    built = factory.create_llm(config).get_model()
    assert model in (getattr(built, "model", None), getattr(built, "model_name", None))


def test_a_listed_model_is_left_for_neuro_san_to_resolve():
    """Naming the class beside `claude-sonnet` sends Anthropic the alias as-is."""
    from esp.config import DEFAULT_LADDERS, llm_config
    for model in {m for ladder in DEFAULT_LADDERS.values() for m in ladder}:
        assert llm_config(model) == {"model_name": model}


def test_a_network_on_a_model_newer_than_neuro_san_names_its_class():
    out = _in_subprocess(
        "from esp.eval import measurements\n"
        "from esp.serving import presentable\n"
        "print(presentable(measurements.best().genome).genome.to_hocon())",
        ESP_DEFAULT_MODEL="claude-opus-99", ANTHROPIC_API_KEY="not-a-real-key")
    assert '{"class": "anthropic", "model_name": "claude-opus-99"}' in out


def test_the_gemini_default_is_unchanged_so_committed_hashes_still_match():
    from esp.genome.definition import DEFAULT_MODEL, MODEL_TIERS
    assert DEFAULT_MODEL == "gemini-3.1-flash-lite"
    assert MODEL_TIERS == ["gemini-3.5-flash-lite", "gemini-3.5-flash"]


# ------------------------------------------------ measurements never cross

def test_the_gemini_measurements_are_not_adopted_into_a_claude_deployment(tmp_path):
    """Adopting them would breed children with a Gemini default the run holds no
    key for, and rank Claude evaluations against Gemini ones."""
    environment = {**os.environ, "ESP_NO_DOTENV": "1", "PYTHONPATH": str(ROOT),
                   "ESP_DEFAULT_MODEL": "claude-haiku-4-5"}
    environment.pop("ESP_MODEL_TIERS", None)
    done = subprocess.run(
        [sys.executable, "scripts/adopt_measurements.py",
         "--state", str(tmp_path / "state"), "--cache", str(tmp_path / "cache")],
        capture_output=True, text=True, env=environment, cwd=ROOT, timeout=120)
    assert done.returncode != 0
    assert "do not cross providers" in done.stderr
    assert "make baseline" in done.stderr
    assert not (tmp_path / "state" / "state.json").exists()


def test_a_population_from_another_provider_fails_the_preflight(tmp_path, monkeypatch):
    from esp.service import preflight
    from esp.service.state import Evaluated, ServiceState

    state = ServiceState()
    state.add(Evaluated(genome_hash="a" * 16, origin="seed:solo", fitness=0.7,
                        accuracy=0.8, tokens=1, agents=1, depth=1, generation=0,
                        measured_at="", model="gemini-3.1-flash-lite", genome=None))
    state.save(tmp_path)
    monkeypatch.setattr(preflight, "STATE_DIR", tmp_path)
    monkeypatch.setattr(preflight, "DEFAULT_MODEL", "claude-haiku-4-5")
    check = next(c for c in preflight.run_checks() if c.name == "population provider")
    assert not check.ok and check.fatal
    assert "gemini" in check.detail and "anthropic" in check.detail


# ------------------------------------------- chosen by provider, not model

_MODELS = ("from esp.genome.definition import DEFAULT_MODEL, MODEL_TIERS\n"
           "print(DEFAULT_MODEL, ','.join(MODEL_TIERS))")


def test_a_claude_key_alone_selects_claude_on_version_free_aliases():
    """neuro-san keeps `claude-haiku` and `claude-sonnet` pointed at the newest
    release, which is how neuro-san-studio names Claude in its own config."""
    assert _in_subprocess(_MODELS, ANTHROPIC_API_KEY=ANTHROPIC_KEY).split() == [
        "claude-haiku", "claude-haiku,claude-sonnet"]


def test_several_keys_follow_neuro_san_studios_order():
    out = _in_subprocess(_MODELS, ANTHROPIC_API_KEY=ANTHROPIC_KEY, OPENAI_API_KEY=OPENAI_KEY)
    assert out.split()[0].startswith("gpt-"), "studio falls back OpenAI, Anthropic, Gemini"


def test_esp_provider_chooses_among_several_keys():
    out = _in_subprocess(_MODELS, ANTHROPIC_API_KEY=ANTHROPIC_KEY, OPENAI_API_KEY=OPENAI_KEY,
                         ESP_PROVIDER="claude")
    assert out.split()[0] == "claude-haiku"


def test_a_placeholder_is_not_a_key_and_selects_nothing():
    out = _in_subprocess(_MODELS, ANTHROPIC_API_KEY="paste-your-key-here")
    assert out.split()[0] == "gemini-3.1-flash-lite"


def test_an_unknown_provider_name_is_refused_not_guessed(monkeypatch):
    from esp.config import configured_provider

    monkeypatch.setenv("ESP_PROVIDER", "mistral")
    with pytest.raises(ValueError, match="ESP_PROVIDER"):
        configured_provider()


def test_any_model_of_the_provider_can_be_chosen_and_takes_its_own_rung():
    """A strong choice tops the ladder; putting claude-opus under claude-sonnet
    would make every promotion a downgrade."""
    assert _in_subprocess(_MODELS, ANTHROPIC_API_KEY=ANTHROPIC_KEY,
                          ESP_DEFAULT_MODEL="claude-opus").split() == [
        "claude-opus", "claude-haiku,claude-opus"]
    assert _in_subprocess(_MODELS, ANTHROPIC_API_KEY=ANTHROPIC_KEY,
                          ESP_DEFAULT_MODEL="claude-haiku-4-5").split() == [
        "claude-haiku-4-5", "claude-haiku-4-5,claude-sonnet"]


def test_the_committed_analysis_does_not_depend_on_the_configured_provider():
    """The surrogate's model-tier feature reads each model's own provider ladder,
    so `make offline` prints the published figures on a Claude machine too."""
    code = ("import numpy as np\n"
            "from esp.eval import measurements\n"
            "from esp.surrogate.predictor import features\n"
            "print(repr(np.vstack([features(m.genome) for m in measurements.load()]).tolist()))")
    assert _in_subprocess(code) == _in_subprocess(code, ANTHROPIC_API_KEY=ANTHROPIC_KEY)


def test_the_demoted_network_is_the_same_network_on_one_model():
    from esp.serving import demoted
    champion = measurements.best().genome
    down = demoted(champion)
    assert down is not None, "the champion promotes its router"
    assert all(agent.model is None for agent in down.agents.values())
    assert set(down.agents) == set(champion.agents) and down.top == champion.top
    assert champion.agents[champion.top].model, "demoting must not touch the original"


def test_nothing_is_demoted_when_nothing_was_promoted():
    from esp.genome.seeds import SEEDS
    from esp.serving import demoted
    assert all(demoted(make()) is None for make in SEEDS.values())
