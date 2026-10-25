"""Inside one run: which agent spent what, and what each one passed up.

Every measurement in this repository reads a network from outside: the answer,
and one token total from the front man. That cannot answer the question the
uncertainty study left open (docs/FINDINGS.md, "Agreement between two runs"):
could agreement be measured by resampling one agent's reply instead of the whole
network? A gate in front of `2cc4` costs no more than the one-line rule only if
that probe costs at most about a third of a cheap run. Whether it can depends on
how a run's cost splits between the agents, which nothing recorded.

neuro-san already reports it. Its token counter
(`neuro_san/internals/run_context/langchain/token_counting/langchain_token_counter.py`)
sends an AGENT message with a usage structure every time an agent finishes:

* an agent below the front man reports the usage of its **subtree**, its own
  calls and every agent it called on the way;
* the front man reports two structures last, the main network's and the
  request's total, the total carrying a "models" breakdown.

The runner kept only the front man's total and read past the rest. This keeps
every message and takes the subtrees apart: an invocation's own usage is its
subtree less the subtrees of the nearest invocations below it that reported.
Coded tools call no model and report nothing, so they cost nothing here.

An invocation is one origin path: (agent, instantiation index) from the front
man down. An agent called three times is three invocations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

USAGE = ("total_tokens", "prompt_tokens", "completion_tokens", "successful_requests",
         "total_cost")
TEXT_LIMIT = 4_000  # characters of one message kept in a trace; documents can be long

Origin = tuple[tuple[str, int], ...]


def origin_of(message: dict) -> Origin:
    return tuple((str(o.get("tool")), int(o.get("instantiation_index", 1)))
                 for o in message.get("origin") or [])


def name(origin: Origin) -> str:
    return ".".join(f"{tool}-{index}" if index > 1 else tool for tool, index in origin)


def _usage(structure: dict) -> dict[str, float]:
    return {k: float(structure.get(k) or 0) for k in USAGE}


@dataclass
class Invocation:
    origin: Origin
    subtree: dict[str, float]
    own: dict[str, float] = field(default_factory=dict)

    @property
    def agent(self) -> str:
        return self.origin[-1][0]


def invocations(messages: list[dict]) -> dict[Origin, Invocation]:
    """Every invocation that reported usage, with its subtree and its own share.

    For the front man the request total is its subtree: the structure with a
    "models" breakdown if one came, else the last it sent."""
    reported: dict[Origin, dict] = {}
    for message in messages:
        structure = message.get("structure")
        if str(message.get("type", "")) not in ("100", "AGENT") or not isinstance(structure, dict):
            continue
        if "total_tokens" not in structure:
            continue
        origin = origin_of(message)
        if not origin:
            continue
        if len(origin) == 1 and "models" in reported.get(origin, {}) \
                and "models" not in structure:
            continue
        reported[origin] = structure

    found = {o: Invocation(o, _usage(s)) for o, s in reported.items()}
    for origin, inv in found.items():
        below = [d for d in found if len(d) > len(origin) and d[:len(origin)] == origin]
        nearest = [d for d in below
                   if not any(len(origin) < len(m) < len(d) and d[:len(m)] == m for m in below)]
        inv.own = {k: inv.subtree[k] - sum(found[d].subtree[k] for d in nearest)
                   for k in USAGE}
    return found


@dataclass(frozen=True)
class Split:
    """How one run's usage divides between its agents."""

    total: dict[str, float]
    by_agent: dict[str, dict[str, float]]  # own usage, summed over invocations
    calls: dict[str, int]                  # invocations of each agent
    front_man: str
    consistent: bool                       # no invocation's own usage came out negative

    @property
    def front_man_share(self) -> float:
        """The front man's own dollars over the request's. Its last call writes
        the answer, so this bounds what resampling only that reply could cost."""
        spent = self.total["total_cost"]
        return self.by_agent[self.front_man]["total_cost"] / spent if spent else 0.0

    def to_dict(self) -> dict:
        return {"front_man": self.front_man, "consistent": self.consistent,
                "front_man_share": round(self.front_man_share, 4),
                "total": {k: round(v, 6) for k, v in self.total.items()},
                "calls": self.calls,
                "by_agent": {a: {k: round(v, 6) for k, v in u.items()}
                             for a, u in self.by_agent.items()}}


def split(messages: list[dict]) -> Split:
    found = invocations(messages)
    roots = [o for o in found if len(o) == 1]
    if len(roots) != 1:
        raise ValueError(f"expected one front man reporting usage, found {len(roots)}")
    front = roots[0]
    by_agent: dict[str, dict[str, float]] = {}
    calls: dict[str, int] = {}
    for inv in found.values():
        totals = by_agent.setdefault(inv.agent, dict.fromkeys(USAGE, 0.0))
        for k in USAGE:
            totals[k] += inv.own[k]
        calls[inv.agent] = calls.get(inv.agent, 0) + 1
    consistent = all(v >= -1e-9 for inv in found.values() for v in inv.own.values())
    return Split(found[front].subtree, by_agent, calls, front[0][0], consistent)


def transcript(messages: list[dict]) -> list[dict]:
    """What each agent said, in order: the record to read when an answer is wrong."""
    out = []
    for message in messages:
        text = message.get("text") or ""
        if not text:
            continue
        out.append({"agent": name(origin_of(message)), "type": str(message.get("type", "")),
                    "text": text[:TEXT_LIMIT], "cut": len(text) > TEXT_LIMIT})
    return out


def record(task_id: str, messages: list[dict], answer: str, correct: bool,
           seconds: float) -> dict:
    """One traced question, as a line of a trace log."""
    return {"task_id": task_id, "answer": answer, "correct": correct,
            "seconds": round(seconds, 2), "split": split(messages).to_dict(),
            "transcript": transcript(messages)}


def summarise(rows: list[dict]) -> dict:
    """Over every traced question: each agent's share of the dollars, and the
    front man's own share question by question."""
    spent = sum(r["split"]["total"]["total_cost"] for r in rows)
    agents: dict[str, float] = {}
    for r in rows:
        for agent, usage in r["split"]["by_agent"].items():
            agents[agent] = agents.get(agent, 0.0) + usage["total_cost"]
    shares = sorted(r["split"]["front_man_share"] for r in rows)
    return {"questions": len(rows),
            "consistent": all(r["split"]["consistent"] for r in rows),
            "dollars": round(spent, 6),
            "agent_share": {a: round(v / spent, 4) if spent else 0.0
                            for a, v in sorted(agents.items())},
            "front_man_share_median": round(shares[len(shares) // 2], 4) if shares else None,
            "front_man_share_max": round(shares[-1], 4) if shares else None}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
