"""The held-out winner the web page and `make smoke` serve, rebuilt from source.

The page used to serve the best network on the seventeen built-in questions,
which tied the designer on held-out questions at 2.1 times the cost. It now
serves the network that won on questions it was never chosen on. That claim is
only honest if the network served is the one the logs measured, so these tests
rebuild it and compare hashes against the committed stage 8 log.
"""

from __future__ import annotations

import json

from esp.evolve import paid


def _logged_hash(label: str) -> str:
    path = paid.PAID_RUNS / paid.WINNER_STAGE / f"{label}.jsonl"
    hashes = {json.loads(line)["genome_hash"]
              for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert len(hashes) == 1, hashes
    return hashes.pop()


def test_the_winner_is_the_genome_stage_8_measured():
    assert paid.winner().genome_hash() == paid.WINNER_HASH == _logged_hash("evolved")


def test_the_winner_is_two_agents_with_only_the_researcher_promoted():
    genome = paid.winner()
    models = {name: genome.agents[name].model or genome.default_model
              for name in sorted(genome.reachable())}
    assert models == {"Coordinator": "gpt-5.4-nano", "Researcher": "gpt-5.4-mini"}


def test_the_winner_does_not_depend_on_the_configured_provider():
    """The suite runs on the default Gemini configuration, where `mutants()`
    produces other hashes. The winner is rebuilt on the ladder it was measured
    on, so it comes out the same."""
    from esp.genome.definition import DEFAULT_MODEL

    assert not DEFAULT_MODEL.startswith("gpt-")
    assert paid.winner().genome_hash() == paid.WINNER_HASH


def test_the_judged_figures_are_the_committed_summary():
    judged = paid.judged()
    assert (judged.correct, judged.designer_correct, judged.question_runs) == (189, 168, 200)
    assert round(judged.accuracy, 3) == 0.945 and round(judged.designer_accuracy, 2) == 0.84
    assert round(judged.token_change, 2) == -0.37
    assert round(judged.cost_change, 2) == 0.94
    assert judged.sign_test_p == 0.0007


def test_a_deployment_without_the_logs_is_told_what_to_do(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="ESP_SERVE=fitness"):
        paid.judged(tmp_path / "summary.json")
