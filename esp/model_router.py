"""Choose the Researcher's model question by question: the test fixed in
docs/MODEL-ROUTER.md before it ran.

`2cc4` is one topology, Coordinator and Researcher, run two ways: every agent on
`gpt-5.4-nano` (the cheap action) or with the Researcher on `gpt-5.4-mini` (the
strong action). The same-model test (esp/same_model.py) found the cheap action
14.6% cheaper than the designer's shape at about its accuracy, and the strong
action far more accurate at 94% more a question. Sending only the questions that
need it to the strong action could be more accurate than the designer's shape and
still cheaper. This checks whether it is.

* **Predictor**: `esp.route.Router`, unchanged: one logistic regression per action
  from the question's text (`esp.route.request_features`) to its chance of a
  right answer, fitted on the 60 select questions only.
* **Prescription**: send a question to the strong action when its predicted gain
  is at least a threshold. The threshold is chosen on select, from leave-one-out
  gains, as the most accurate whose select cost a question is at or below the
  designer's shape's; a tie goes to the threshold sending fewer questions up.
* **Judged** on the judge and fresh questions, every action measured on
  7 November 2026, against the five pass marks of the same-model test.

The provider reports usage per four-question chunk, so a routed question's tokens
and dollars are its action's average for that chunk; `dollars_by_time`, the
chunk's cost split by the seconds each question took, is a second estimate.

It reads the committed logs and nothing else, so it costs nothing and needs no key.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from esp import route
from esp.eval import accounting
from esp.eval.pricing import fitness_dollars
from esp.eval.stats import mcnemar_exact
from esp.evolve.loop import scalarise

ROOT = Path(__file__).resolve().parent.parent
PAID = ROOT / "results" / "headtohead" / "paid-2026-10"
STRUCTURE = ROOT / "results" / "structure"
OUT = ROOT / "results" / "router"
ACTIONS = ("cheap", "strong", "baseline")
HASHES = {"cheap": "2cc4231b7d3fb81d", "strong": "47322479eb9cba1e",
          "baseline": "0378ca26cd9fe977"}
AGENTS = {"cheap": 2, "strong": 2, "baseline": 4}
ROUTED_AGENTS = 2        # both actions are the same two agents

SELECT = {"cheap": PAID / "7-select-search10" / "2cc4.jsonl",
          "strong": PAID / "7-select-search10" / "2cc4-mini.jsonl",
          "baseline": PAID / "7-select-search10" / "designer.jsonl"}
SETS = {
    "judge": {"cheap": STRUCTURE / "judge-nano-2026-11" / "2cc4-nano.jsonl",
              "strong": OUT / "judge-2026-11" / "2cc4-mini.jsonl",
              "baseline": STRUCTURE / "judge-nano-2026-11" / "designer.jsonl"},
    "fresh": {"cheap": STRUCTURE / "fresh-nano-2026-11" / "2cc4-nano.jsonl",
              "strong": STRUCTURE / "fresh-nano-2026-11" / "evolved.jsonl",
              "baseline": STRUCTURE / "fresh-nano-2026-11" / "designer.jsonl"},
    # The strong action as October measured it, a fortnight before the others.
    "judge-october-strong": {"cheap": STRUCTURE / "judge-nano-2026-11" / "2cc4-nano.jsonl",
                             "strong": PAID / "8-judge-search10" / "evolved.jsonl",
                             "baseline": STRUCTURE / "judge-nano-2026-11" / "designer.jsonl"},
}

ALPHA = 0.05
EXAM = 17
COST_KEY = "reported_cost"
SEARCH_RESULTS = 10
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261107


@dataclass
class Log:
    """One action on one question set: right answers, and usage by chunk."""

    genome_hash: str
    right: dict[str, float] = field(default_factory=dict)       # task -> share of runs right
    chunk_of: dict[str, int] = field(default_factory=dict)      # task -> chunk
    tokens: dict[int, float] = field(default_factory=dict)      # chunk -> tokens a question
    dollars: dict[int, float] = field(default_factory=dict)     # chunk -> dollars a question
    by_time: dict[str, float] = field(default_factory=dict)     # task -> dollars, split by seconds
    runs: int = 0

    @property
    def dollars_per_question(self) -> float:
        return float(np.mean([self.dollars[self.chunk_of[t]] for t in self.right]))


def read(path: Path, action: str) -> Log:
    """A log written by scripts/select_and_judge.py measure, checked against the plan."""
    from esp.evolve.headtohead import read_rows

    rows = [accounting.normalised(row) for row in read_rows(path)]
    hashes = {row["genome_hash"] for row in rows}
    if hashes != {HASHES[action]}:
        raise ValueError(f"{path}: the {action} action ran as {sorted(hashes)}, "
                         f"the plan names {HASHES[action]}")
    for row in rows:
        width = row.get("settings", {}).get("search_results", SEARCH_RESULTS)
        if width != SEARCH_RESULTS:
            raise ValueError(f"{path}: chunk {row['chunk']} searched {width} documents")
    log = Log(HASHES[action])
    seen: dict[str, list[bool]] = defaultdict(list)
    usage: dict[int, list[tuple[float, float]]] = defaultdict(list)
    split: dict[str, list[float]] = defaultdict(list)
    tasks_in: dict[int, tuple[str, ...]] = {}
    for row in rows:
        tasks = tuple(r["task_id"] for r in row["results"])
        if tasks_in.setdefault(row["chunk"], tasks) != tasks:
            raise ValueError(f"{path}: chunk {row['chunk']} asked different questions")
        usage[row["chunk"]].append((row["tokens"] / len(tasks), row[COST_KEY] / len(tasks)))
        seconds = sum(r["seconds"] for r in row["results"]) or 1.0
        for r in row["results"]:
            log.chunk_of[r["task_id"]] = row["chunk"]
            # Not finishing is not answering: wrong, as in the same-model test.
            seen[r["task_id"]].append(bool(r["correct"]) and not r["infrastructure"])
            split[r["task_id"]].append(row[COST_KEY] * r["seconds"] / seconds)
            log.runs += 1
    log.right = {t: sum(v) / len(v) for t, v in seen.items()}
    log.tokens = {c: float(np.mean([u[0] for u in v])) for c, v in usage.items()}
    log.dollars = {c: float(np.mean([u[1] for u in v])) for c, v in usage.items()}
    log.by_time = {t: float(np.mean(v)) for t, v in split.items()}
    return log


def _questions(name: str) -> dict[str, str]:
    from esp.eval.fresh import FRESH
    from esp.eval.suites import JUDGE, SELECT

    tasks = {"select": SELECT, "judge": JUDGE, "fresh": FRESH}[name.split("-")[0]]
    return {t.task_id: t.question for t in tasks}


def choose(gains: np.ndarray, cheap: np.ndarray, strong: np.ndarray, price_cheap: float,
           price_strong: float, budget: float) -> tuple[float, float, float, float]:
    """(threshold, accuracy, cost a question, share sent up): the most accurate
    threshold whose cost stays within `budget`, a tie going to the one sending
    fewer up. Infinity, sending nothing up, is always a candidate."""
    best: tuple[float, float, float, float] | None = None
    for threshold in [float("inf"), *sorted(set(gains.tolist()), reverse=True)]:
        up = gains >= threshold
        share = float(up.mean())
        cost = share * price_strong + (1 - share) * price_cheap
        accuracy = float(np.where(up, strong, cheap).mean())
        if cost <= budget + 1e-12 and (best is None or accuracy > best[1] + 1e-12):
            best = (threshold, accuracy, cost, share)
    if best is None:
        raise ValueError("the cheap action alone already costs more than the budget")
    return best


def fit(paths: dict[str, Path] | None = None) -> dict:
    """The Predictor and the threshold, from the select questions alone."""
    logs = {a: read(p, a) for a, p in (paths or SELECT).items()}
    text = _questions("select")
    ids = [t for t in text if all(t in logs[a].right for a in ACTIONS)]
    questions = [text[t] for t in ids]
    cheap = np.array([logs["cheap"].right[t] for t in ids])
    strong = np.array([logs["strong"].right[t] for t in ids])
    gains = route.leave_one_out_gain(questions, cheap, strong)

    price = {a: logs[a].dollars_per_question for a in ACTIONS}
    budget = price["baseline"]
    threshold, accuracy, cost, share = choose(gains, cheap, strong, price["cheap"],
                                              price["strong"], budget)
    router = route.Router(threshold).fit(questions, cheap, strong)
    return {"router": router, "questions": len(ids), "threshold": threshold,
            "leave_one_out": {"accuracy": round(accuracy, 4), "dollars": round(cost, 6),
                              "share_to_strong": round(share, 4)},
            "select": {a: {"accuracy": round(float(np.mean([logs[a].right[t] for t in ids])), 4),
                           "dollars_per_question": round(price[a], 6)} for a in ACTIONS},
            "budget": round(budget, 6),
            "most_sent_up_within_budget": int(np.floor(
                (budget - price["cheap"]) / (price["strong"] - price["cheap"]) * len(ids)))}


def _bootstrap(diffs: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = diffs[rng.integers(0, len(diffs), size=(BOOTSTRAP_DRAWS, len(diffs)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return [round(100 * float(low), 1), round(100 * float(high), 1)]


def judge(router: route.Router, name: str, paths: dict[str, Path]) -> dict:
    """The routing on one question set, against the five pass marks."""
    logs = {a: read(p, a) for a, p in paths.items()}
    text = _questions(name)
    ids = [t for t in text if all(t in logs[a].right for a in ACTIONS)]
    for t in ids:
        chunks = {logs[a].chunk_of[t] for a in ACTIONS}
        if len(chunks) != 1:
            raise ValueError(f"{name}: {t} sits in different chunks: {chunks}")
    up = dict(zip(ids, router.to_strong([text[t] for t in ids]).tolist(), strict=True))

    def chosen(t: str) -> str:
        return "strong" if up[t] else "cheap"

    routed_right = np.array([logs[chosen(t)].right[t] for t in ids])
    base_right = np.array([logs["baseline"].right[t] for t in ids])
    better = [t for t, r, b in zip(ids, routed_right, base_right, strict=True) if r > b]
    worse = [t for t, r, b in zip(ids, routed_right, base_right, strict=True) if r < b]

    def per_question(field_name: str, action_of) -> dict[int, float]:
        values: dict[int, list[float]] = defaultdict(list)
        for t in ids:
            log = logs[action_of(t)]
            values[log.chunk_of[t]].append(getattr(log, field_name)[log.chunk_of[t]])
        return {c: float(np.mean(v)) for c, v in values.items()}

    usage = {}
    for field_name in ("tokens", "dollars"):
        routed = per_question(field_name, chosen)
        base = per_question(field_name, lambda _t: "baseline")
        chunks = sorted(routed)
        a = np.array([routed[c] for c in chunks])
        b = np.array([base[c] for c in chunks])
        usage[field_name] = {
            "routed": round(float(np.mean([getattr(logs[chosen(t)], field_name)[
                logs[chosen(t)].chunk_of[t]] for t in ids])), 6),
            "baseline": round(float(np.mean([getattr(logs["baseline"], field_name)[
                logs["baseline"].chunk_of[t]] for t in ids])), 6),
            "chunks": len(chunks), "routed_lower_in": int((a < b).sum()),
            "p": round(float(wilcoxon(a, b, alternative="two-sided").pvalue), 6),
            "change": round(float(a.sum() / b.sum() - 1), 4)}

    accuracy = float(routed_right.mean())
    base_accuracy = float(base_right.mean())
    tokens, dollars = usage["tokens"]["routed"], usage["dollars"]["routed"]
    base_tokens, base_dollars = usage["tokens"]["baseline"], usage["dollars"]["baseline"]
    fitness = {
        "routed": {"token": round(scalarise(accuracy, round(tokens * EXAM), ROUTED_AGENTS), 4),
                   "dollar": round(fitness_dollars(accuracy, dollars, ROUTED_AGENTS), 4)},
        "baseline": {"token": round(scalarise(base_accuracy, round(base_tokens * EXAM),
                                              AGENTS["baseline"]), 4),
                     "dollar": round(fitness_dollars(base_accuracy, base_dollars,
                                                     AGENTS["baseline"]), 4)}}
    p_accuracy = round(mcnemar_exact(len(better), len(worse)), 4)
    holds = {
        "accuracy": accuracy > base_accuracy and len(better) > len(worse) and p_accuracy < ALPHA,
        "tokens": tokens < base_tokens and usage["tokens"]["p"] < ALPHA,
        "dollars": dollars < base_dollars and usage["dollars"]["p"] < ALPHA,
        "agents": AGENTS["baseline"] > ROUTED_AGENTS,
        "fitness": all(fitness["routed"][k] > fitness["baseline"][k] for k in ("token", "dollar")),
    }
    up_mask = np.array([up[t] for t in ids])
    cheap_right = np.array([logs["cheap"].right[t] for t in ids])
    strong_right = np.array([logs["strong"].right[t] for t in ids])
    random_accuracy, p_random = route.permutation_p(up_mask, cheap_right, strong_right)

    # Not a router and not a test: the routing that knows every answer, sending up
    # the questions the strong action wins most, cheapest first, while the cost a
    # question stays at or below the baseline's. Whether these two actions could
    # ever pass, whatever the Predictor.
    def price(action: str, t: str) -> float:
        return logs[action].dollars[logs[action].chunk_of[t]]

    spend = sum(price("cheap", t) for t in ids)
    allowed = base_dollars * len(ids)
    oracle_up: set[str] = set()
    for t in sorted((t for t in ids if logs["strong"].right[t] > logs["cheap"].right[t]),
                    key=lambda t: (logs["cheap"].right[t] - logs["strong"].right[t],
                                   price("strong", t) - price("cheap", t))):
        extra = price("strong", t) - price("cheap", t)
        if spend + extra <= allowed + 1e-12:
            oracle_up.add(t)
            spend += extra
    oracle_right = np.array([logs["strong" if t in oracle_up else "cheap"].right[t] for t in ids])
    oracle_better = int((oracle_right > base_right).sum())
    oracle_worse = int((oracle_right < base_right).sum())
    return {
        "questions": len(ids), "sent_up": int(up_mask.sum()),
        "share_to_strong": round(float(up_mask.mean()), 4),
        "accuracy": {"routed": round(accuracy, 4), "baseline": round(base_accuracy, 4),
                     "cheap": round(float(cheap_right.mean()), 4),
                     "strong": round(float(strong_right.mean()), 4),
                     "routed_better_on": better, "baseline_better_on": worse,
                     "p": p_accuracy,
                     "points": round(100 * (accuracy - base_accuracy), 1),
                     "ci95_points": _bootstrap(routed_right - base_right),
                     "random_same_share": round(random_accuracy, 4),
                     "p_against_random": round(p_random, 4)},
        "tokens": usage["tokens"], "dollars": usage["dollars"],
        "dollars_by_time": {
            "routed": round(float(np.mean([logs[chosen(t)].by_time[t] for t in ids])), 6),
            "baseline": round(float(np.mean([logs["baseline"].by_time[t] for t in ids])), 6)},
        "fitness": fitness, "holds": holds, "all_hold": all(holds.values()),
        "oracle_within_budget": {
            "accuracy": round(float(oracle_right.mean()), 4), "sent_up": len(oracle_up),
            "dollars": round(spend / len(ids), 6), "better_on": oracle_better,
            "worse_on": oracle_worse, "p": round(mcnemar_exact(oracle_better, oracle_worse), 4)}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyse(select: dict[str, Path] | None = None,
            sets: dict[str, dict[str, Path]] | None = None) -> dict:
    """Fit on select, then judge every set whose three logs exist."""
    fitted = fit(select)
    router = fitted.pop("router")
    threshold = fitted["threshold"]
    # No threshold within the budget beat routing nothing, which JSON cannot write
    # as an infinite threshold, so it is said in words instead.
    out: dict = {"plan": "docs/MODEL-ROUTER.md", "alpha": ALPHA,
                 "select": {**fitted,
                            "threshold": None if np.isinf(threshold) else round(threshold, 6),
                            "routes_nothing": bool(np.isinf(threshold))},
                 "sets": {}, "logs": {}}
    paths = list((select or SELECT).values())
    out["incomplete"] = []
    for name, logs in (sets or SETS).items():
        if not all(path.exists() for path in logs.values()):
            continue
        # A run still being measured covers only some questions, and a set judged
        # on part of itself is not the set the plan names.
        questions = set(_questions(name))
        if any(set(read(path, action).right) != questions for action, path in logs.items()):
            out["incomplete"].append(name)
            continue
        out["sets"][name] = judge(router, name, logs)
        paths += list(logs.values())
    for path in paths:
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        out["logs"][str(shown)] = sha256(path)
    return out
