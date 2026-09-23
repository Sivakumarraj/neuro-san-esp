"""Measurement as a neuro-san agent: the evaluator's tools.

`esp.measure` answers "is this network any good" from a terminal and the web
page answers it from a browser. These answer it from inside neuro-san, so the
question can be put in the same chat UI the networks themselves are served in:
"measure the designer's shape against the evolved one" is a message, not a
command line.

The opposite trust arrangement from the optimiser's tools. There the language
model decides nothing; here choosing what to measure is the whole job, so the
model does choose -- but only among networks this deployment already knows
(`esp.measure.catalog`), never a path or a network definition it was handed,
because a network names Python classes to import. And every run is paid for,
so one process caps the question-runs it will spend in total.

Every number the agent reports comes back from these tools. The instructions in
registries/evaluator.hocon forbid it to state any figure they did not return.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from neuro_san.interfaces.coded_tool import CodedTool

from esp.measure import Candidate, SuiteError, catalog, load_suite, measure, pareto, parse_suite

# Four networks on the built-in benchmark: about 680 model calls, a few
# dollars on a paid key. ESP_EVAL_MAX_RUNS raises it for a deliberate study.
MAX_RUNS = int(os.environ.get("ESP_EVAL_MAX_RUNS", "68"))
MAX_NETWORKS = 4

_spent = {"runs": 0}
_lock = threading.Lock()
_catalog: list[Candidate] | None = None


def networks() -> list[Candidate]:
    """What this process can measure, rendered once."""
    global _catalog
    if _catalog is None:
        _catalog = catalog(os.environ.get("ESP_NETWORKS") or None)
    return _catalog


def resolve(wanted: list[str]) -> tuple[list[Candidate], list[str]]:
    """Networks by id, by full name, or by an unambiguous prefix of either.

    A model repeats what it was shown, and what it was shown is a name like
    "mut:reassign_model · 3bf9c0" -- so the six-character hash alone, or the
    name without it when only one network has that name, has to work too.
    """
    found, unknown = [], []
    for text in (str(w).strip() for w in wanted):
        exact = [c for c in networks() if text in (c.id, c.name)]
        loose = exact or [c for c in networks()
                          if text and (c.id.startswith(text) or c.name.startswith(text))]
        if len(loose) == 1 and loose[0] not in found:
            found.append(loose[0])
        elif len(loose) != 1:
            unknown.append(text if not loose else
                           f"{text} (matches {len(loose)}: be more specific)")
    return found, unknown


class ListNetworks(CodedTool):
    """The networks that can be measured, with what each scored when measured."""

    def invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> str:
        return json.dumps({
            "networks": [c.public() for c in networks()],
            "runs_left": MAX_RUNS - _spent["runs"],
            "max_networks_per_measurement": MAX_NETWORKS,
        })


class MeasureNetworks(CodedTool):
    """Put the same questions to up to four networks and compare them."""

    def invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> str:
        wanted = args.get("networks") or []
        if isinstance(wanted, str):
            wanted = [part for part in wanted.split(",") if part.strip()]
        chosen, unknown = resolve(wanted)
        if unknown:
            return _refuse(f"unknown network(s): {', '.join(unknown)}. "
                           "Call ListNetworks for the names.")
        if not chosen:
            return _refuse("name at least one network to measure")
        if len(chosen) > MAX_NETWORKS:
            return _refuse(f"at most {MAX_NETWORKS} networks at once")

        try:
            questions = args.get("questions")
            suite, tasks = (("custom", parse_suite(questions)) if questions
                            else load_suite(None))
        except SuiteError as exc:
            return _refuse(f"the questions cannot be used: {exc}")

        runs = len(chosen) * len(tasks)
        with _lock:
            if _spent["runs"] + runs > MAX_RUNS:
                return _refuse(
                    f"that is {runs} paid runs and {MAX_RUNS - _spent['runs']} are "
                    "left in this server's budget (ESP_EVAL_MAX_RUNS raises it)")
            _spent["runs"] += runs

        reports: list[dict] = []
        full: list[dict] = []
        for candidate in chosen:
            try:
                report = measure(candidate.hocon, tasks, suite=suite)
            except (OSError, SuiteError) as exc:
                reports.append({"network": candidate.name, "error": str(exc)[:300]})
                continue
            full.append(report.as_dict())
            reports.append({
                "network": candidate.name,
                "accuracy": report.accuracy,
                "answered_accuracy": report.answered_accuracy,
                "unfinished": report.unfinished,
                "tokens": report.tokens,
                "seconds": round(report.seconds),
                "measured_before": candidate.measured,
                "note": candidate.note,
                "per_question": {
                    r.task_id: ("unfinished" if r.infrastructure
                                else "correct" if r.correct else "wrong")
                    for r in report.results},
            })
        pareto(reports)
        # The full answers are long; the agent gets verdicts, and the whole
        # reports go on the bulletin board for anything downstream.
        sly_data["measurement"] = full
        return json.dumps({"suite": suite, "questions": len(tasks), "reports": reports})


def _refuse(reason: str) -> str:
    return json.dumps({"refused": reason})
