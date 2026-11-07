"""The calculator CodedTool and the genome flag that gives an agent one.

The tool is handed text a model wrote, so the tests that matter most are the ones
that try to make it do something other than arithmetic.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from esp.eval.calculator_tool import CalculationError, Calculator, evaluate, shown
from esp.genome.definition import CALCULATOR_INSTRUCTIONS, CALCULATOR_TOOL, Genome
from esp.genome.seeds import designer_shaped, flat_pair, solo

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("expression, expected", [
    ("375 * 12", "4500"),
    ("375 * 12 + 250 * 8", "6500"),
    ("(23 + 19) * 250", "10500"),
    ("2472000 + 650000", "3122000"),
    ("1,250 * 4", "5000"),                       # thousands separators
    ("90 \N{MULTIPLICATION SIGN} 4", "360"),     # the signs models write
    ("100 \N{MINUS SIGN} 7.5", "92.5"),
    ("-4 + 10", "6"),
    ("10 / 4", "2.5"),
    ("0.1 + 0.2", "0.3"),                        # no float noise in the answer
    ("  42  ", "42"),
])
def test_it_works_out_arithmetic_exactly(expression, expected):
    assert shown(evaluate(expression)) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('true')",
    "open('/etc/passwd').read()",
    "x + 1",
    "(1).__class__",
    "[1, 2][0]",
    "2 ** 10",                                   # powers are refused, not computed
    "9 ** 9 ** 9",
    "lambda: 1",
    "1 if 1 else 2",
    "1; 2",
    "True + 1",
    "",
    "1 +",
    "1 / 0",
    "1e400",
    "9" * 401,
])
def test_anything_but_arithmetic_is_refused(expression):
    with pytest.raises(CalculationError):
        evaluate(expression)


def test_the_tool_answers_errors_in_words_and_counts_its_calls():
    board: dict = {}
    tool = Calculator()
    assert tool.invoke({"expression": "6 * 7"}, board) == "42"
    assert tool.invoke({"expression": "os.system('x')"}, board).startswith("Error: ")
    assert tool.invoke({}, board) == "Error: no expression given."
    assert board == {"calculator_calls": 3}


def test_either_call_gives_the_same_answer():
    sync_board, async_board = {}, {}
    args = {"expression": "(12 + 9) * 375"}
    assert (asyncio.run(Calculator().async_invoke(args, async_board))
            == Calculator().invoke(args, sync_board) == "7875")
    assert async_board == sync_board


@pytest.mark.parametrize("make", [designer_shaped, flat_pair, solo])
def test_a_network_without_the_calculator_hashes_as_it_always_did(make):
    """Every committed measurement is keyed on a hash computed before the flag
    existed; the flag left unset must not appear in the canonical form."""
    genome = make()
    assert all("can_calculate" not in a for a in genome.canonical()["agents"].values())
    assert CALCULATOR_TOOL not in genome.to_hocon()


def test_the_committed_networks_keep_their_hashes():
    sys.path.insert(0, str(ROOT / "scripts"))
    from esp.evolve import paid

    assert paid.winner().genome_hash() == paid.WINNER_HASH


def test_giving_an_agent_the_calculator_wires_it_and_tells_it():
    genome = flat_pair()
    with_tool = genome.with_calculator("Coordinator")
    agent = with_tool.agents["Coordinator"]
    assert agent.can_calculate and agent.instructions.endswith(CALCULATOR_INSTRUCTIONS)
    assert not genome.agents["Coordinator"].can_calculate       # the original is untouched
    assert with_tool.genome_hash() != genome.genome_hash()
    hocon = with_tool.to_hocon()
    assert hocon.count('"class": "esp.eval.calculator_tool.Calculator"') == 1
    coordinator = hocon[hocon.index('"name": "Coordinator"'):hocon.index('"name": "Researcher"')]
    assert '"Calculator"' in coordinator
    # Asking twice does not repeat the instruction.
    assert with_tool.with_calculator("Coordinator").genome_hash() == with_tool.genome_hash()


def test_the_calculator_survives_a_round_trip_through_the_canonical_form():
    genome = designer_shaped().with_calculator("Coordinator")
    rebuilt = Genome.from_canonical(json.loads(json.dumps(genome.canonical())))
    assert rebuilt.genome_hash() == genome.genome_hash()
    assert rebuilt.agents["Coordinator"].can_calculate


def test_an_unknown_agent_is_an_error():
    with pytest.raises(KeyError):
        solo().with_calculator("Nobody")


def test_the_calculator_network_passes_neuro_sans_validator(tmp_path, monkeypatch):
    """The validator `make hocon` runs over the registries, run over a rendered
    network with the tool, so a malformed block fails here and not mid-run."""
    import subprocess

    path = tmp_path / "calc.hocon"
    path.write_text(flat_pair().with_calculator("Coordinator").to_hocon(), encoding="utf-8")
    env = {"AGENT_TOOL_PATH": str(ROOT), "PYTHONPATH": str(ROOT),
           "PATH": __import__("os").environ.get("PATH", "")}
    done = subprocess.run([sys.executable, "-m", "neuro_san.client.hocon_validator_cli",
                           str(path)], capture_output=True, text=True, env=env, timeout=120,
                          check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Validation passed" in done.stdout


def test_select_and_judge_builds_the_variants_it_is_asked_for(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import select_and_judge as sj

    contender = sj.contender_for("2cc4", "x", models=["Researcher=gpt-4o-mini"],
                                 calculator=["Coordinator"])
    models = contender.models()
    assert models["Researcher"] == "gpt-4o-mini"
    assert contender.genome.agents["Coordinator"].can_calculate
    assert "Researcher on gpt-4o-mini, Calculator on Coordinator" in contender.origin
    with pytest.raises(SystemExit):
        sj.contender_for("2cc4", "x", models=["Researcher"])
    with pytest.raises(SystemExit):
        sj.contender_for("2cc4", "x", calculator=["Nobody"])
