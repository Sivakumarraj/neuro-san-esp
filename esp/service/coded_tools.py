"""The optimiser's tools, as neuro-san CodedTools.

The decision this file deliberately does not delegate is *what to evaluate*.
That is chosen by the surrogate and the mutation operators, in code, from
measured fitness. The language model's job is to notice that something improved
and write it up -- it never picks a candidate, never spends budget, and cannot
reach anything that deletes or sends.
"""

from __future__ import annotations

import json
from typing import Any

from neuro_san.interfaces.coded_tool import CodedTool

from esp.service.optimizer import wake
from esp.service.state import ServiceState


class RunWake(CodedTool):
    """Run one wake and return what happened, as JSON the front agent can read."""

    def invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> str:
        state = ServiceState.load()
        report = wake(state)

        best = state.best()
        payload = {
            "acquired": report.acquired,
            "evaluated_this_wake": report.evaluated,
            "population": len(state.evaluated),
            "generation": report.generation,
            "improved": report.improved,
            "best_fitness": report.best_fitness,
            "stopped_because": report.stopped_because,
            "exhausted_today": report.exhausted,
            "note": report.note,
        }
        if best:
            payload["best"] = {
                "genome": best.genome_hash, "origin": best.origin,
                "accuracy": best.accuracy, "tokens": best.tokens,
                "agents": best.agents, "fitness": best.fitness,
            }
        # The run id and population size go on the bulletin board so a later
        # tool can correlate without the front agent having to relay them.
        sly_data["optimizer_wake"] = state.wakes
        return json.dumps(payload)
