"""Taking a run apart by agent, from messages shaped as neuro-san streams them.

The shapes follow neuro-san's token counter: an agent below the front man sends
one AGENT message with the usage of its subtree when it finishes, and the front
man sends the main network's accounting and then the request total, the total
with a "models" breakdown. Coded tools stream text and no usage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from esp import trace
from esp.eval import runner

ROOT = Path(__file__).resolve().parent.parent


def _origin(*path: str | tuple[str, int]) -> list[dict]:
    out = []
    for step in path:
        tool, index = (step, 1) if isinstance(step, str) else step
        out.append({"tool": tool, "instantiation_index": index})
    return out


def _usage(tokens: int, dollars: float, requests: int = 1, **extra) -> dict:
    return {"total_tokens": tokens, "prompt_tokens": tokens * 9 // 10,
            "completion_tokens": tokens // 10,
            "successful_requests": requests, "total_cost": dollars, **extra}


def _agent(origin: list[dict], structure: dict) -> dict:
    return {"type": "AGENT", "origin": origin, "structure": structure}


def _run() -> list[dict]:
    """Coordinator -> DepotSpecialist; Coordinator -> ContractSpecialist ->
    IncidentSpecialist; ContractSpecialist also calls the CorpusSearch tool."""
    fm = ("Coordinator",)
    return [
        {"type": "AI", "origin": _origin(*fm, "ContractSpecialist", "CorpusSearch"),
         "text": "C-2118: Late-delivery penalty: 4375 per hour"},
        _agent(_origin(*fm, "ContractSpecialist", "IncidentSpecialist"), _usage(200, 0.2, 2)),
        {"type": "AI", "origin": _origin(*fm, "ContractSpecialist"), "text": "rate 4375"},
        _agent(_origin(*fm, "ContractSpecialist"), _usage(500, 0.5, 4)),
        _agent(_origin(*fm, "DepotSpecialist"), _usage(300, 0.3, 2)),
        _agent(_origin(*fm), _usage(1000, 1.0, 9)),
        _agent(_origin(*fm), _usage(1000, 1.0, 9, models={"openai": {}})),
        {"type": "AGENT_FRAMEWORK", "origin": _origin(*fm), "text": "90875"},
    ]


def test_each_invocation_owns_its_subtree_less_the_subtrees_below_it():
    found = trace.invocations(_run())
    own = {trace.name(o): inv.own["total_tokens"] for o, inv in found.items()}
    assert own == {"Coordinator.ContractSpecialist.IncidentSpecialist": 200,
                   "Coordinator.ContractSpecialist": 300,
                   "Coordinator.DepotSpecialist": 300,
                   "Coordinator": 200}


def test_the_parts_add_back_up_to_the_request_total():
    split = trace.split(_run())
    for key in trace.USAGE:
        assert sum(u[key] for u in split.by_agent.values()) == pytest.approx(split.total[key])
    assert split.consistent
    assert split.front_man == "Coordinator"
    assert split.front_man_share == pytest.approx(0.2)


def test_the_front_man_total_is_the_structure_with_models_whatever_follows():
    messages = [*_run(), _agent(_origin("Coordinator"), _usage(5, 0.005))]
    assert trace.split(messages).total["total_tokens"] == 1000


def test_an_agent_called_twice_is_two_invocations_of_one_agent():
    fm = ("Coordinator",)
    messages = [_agent(_origin(*fm, ("DepotSpecialist", 1)), _usage(100, 0.1)),
                _agent(_origin(*fm, ("DepotSpecialist", 2)), _usage(150, 0.15)),
                _agent(_origin(*fm), _usage(400, 0.4, models={}))]
    split = trace.split(messages)
    assert split.calls["DepotSpecialist"] == 2
    assert split.by_agent["DepotSpecialist"]["total_tokens"] == 250
    assert split.by_agent["Coordinator"]["total_tokens"] == 150


def test_a_child_reporting_more_than_its_parent_is_flagged_not_hidden():
    messages = [_agent(_origin("Coordinator", "DepotSpecialist"), _usage(900, 0.9)),
                _agent(_origin("Coordinator"), _usage(400, 0.4, models={}))]
    assert not trace.split(messages).consistent


def test_a_run_without_exactly_one_front_man_is_refused():
    with pytest.raises(ValueError):
        trace.split([_agent(_origin("Coordinator", "DepotSpecialist"), _usage(1, 0.0))])


def test_the_measured_path_reads_the_same_stream_as_before():
    """read_reply is what `_ask` always did: the last answer and the front
    man's accounting, every agent below it read past."""
    answer, accounting = runner.read_reply(_run())
    assert answer == "90875"
    assert accounting["total_tokens"] == 1000 and "models" in accounting


def test_the_transcript_keeps_who_said_what_and_cuts_long_text():
    messages = [*_run(), {"type": "AI", "origin": _origin("Coordinator"),
                          "text": "x" * (trace.TEXT_LIMIT + 5)}]
    said = trace.transcript(messages)
    assert said[0]["agent"] == "Coordinator.ContractSpecialist.CorpusSearch"
    assert said[-1]["cut"] and len(said[-1]["text"]) == trace.TEXT_LIMIT


def test_a_record_and_its_summary(tmp_path):
    rows = [trace.record("J004", _run(), "90875", True, 12.3)] * 2
    path = tmp_path / "designer.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = trace.summarise(trace.read(path))
    assert summary["questions"] == 2 and summary["consistent"]
    assert summary["agent_share"]["Coordinator"] == pytest.approx(0.2)
    assert summary["front_man_share_median"] == pytest.approx(0.2)


@pytest.fixture()
def trace_run():
    sys.path.insert(0, str(ROOT / "scripts"))
    import trace_run

    return trace_run


def test_the_traced_questions_are_every_multi_incident_penalty_and_two_controls(trace_run):
    ids = [t.task_id for t in trace_run.questions()]
    assert len(ids) == 18
    assert {"J004", "F11", *trace_run.CONTROLS} <= set(ids)


def _plan(**env: str):
    """The script in a fresh interpreter: model tiers are fixed at import, and
    the script sets stage 8's settings in its own environment, not this one."""
    import os
    import subprocess

    clean = {k: v for k, v in os.environ.items()
             if not k.startswith("ESP_") and not k.endswith("_API_KEY")}
    clean.update(ESP_NO_DOTENV="1", **env)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "trace_run.py")],
                          capture_output=True, text=True, env=clean, cwd=ROOT, timeout=120)


def test_on_stage_8s_settings_the_plan_is_stage_8s_networks():
    done = _plan()
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.count("same") == 2 and "plan only" in done.stdout


def test_the_plan_refuses_networks_stage_8_did_not_measure():
    done = _plan(ESP_PROVIDER="gemini", ESP_DEFAULT_MODEL="gemini-3.1-flash-lite",
                 ESP_MODEL_TIERS="gemini-3.5-flash-lite,gemini-3.5-flash")
    assert done.returncode == 1
    assert "refused" in done.stdout
