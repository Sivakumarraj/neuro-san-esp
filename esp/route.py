"""Route each question to the network that should answer it: ESP with a context.

Everything else in this repository prescribes one network for every question.
ESP proper prescribes per context: a Predictor learns the outcomes of
(context, action), and the prescription picks, for each context, the action the
Predictor says is best. Here:

* **Context** is the question as it arrives, read by `request_features` from its
  text alone. Nothing a router would not have at that moment: not the number of
  documents the question needs (`Task.hops`, which the generator knows and a
  router never does) and never the answer.
* **Actions** are two networks: a cheap one and a strong one.
* **Outcomes** are whether the answer was right and what it cost.
* **Predictor**: per network, a logistic regression from the question's features
  to its chance of a right answer, trained on the select questions only.
* **Prescription**: send a question to the strong network when the predicted gain
  in that chance is at least `threshold`. The threshold is chosen on select, from
  leave-one-out predictions, as the cheapest that stays within one question of
  the strong network's select score. Only then is it applied to judge.

This is not a learned Prescriptor. With two actions, a threshold on the predicted
gain is the whole decision, so there is nothing for one to learn.

Cost is each network's average dollars per question, because the provider
reports cost per chunk of four questions, not per question. A hard question
costs more than an easy one on the same network, and the router sends the hard
ones to the strong network, so `analyse` also prices every routing with the
chunk's cost split by the seconds each question took: a second estimate, not a
measurement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "headtohead" / "paid-2026-10"

# The one pair of networks measured on both select and judge with the search at
# 10 documents (stages 7 and 8). The strong network is 2cc4 with its Researcher on
# gpt-5.4-mini: stage 8 calls it "evolved", stage 7 "2cc4-mini", and both logs
# carry genome 47322479eb9cba1e. Results at 3 documents are never mixed in.
CHEAP = ("7-select-search10/designer.jsonl", "8-judge-search10/designer.jsonl")
STRONG = ("7-select-search10/2cc4-mini.jsonl", "8-judge-search10/evolved.jsonl")

FEATURE_NAMES = ["named", "aggregate", "arithmetic", "chain", "numeric"]

_IDENTIFIER = re.compile(r"INC-\d+|\bC-\d+|\bD\d\d\b")
_AGGREGATE = re.compile(r"\bevery\b|How many incidents|contracts are serviced by depot")
_ARITHMETIC = re.compile(r"total|combined|between them|apart|exceed|penalty")
_CHAIN = re.compile(r"affected a contract|that services|affected by incident|serviced by a")

# Within one question of the strong network on select, so a router is never
# chosen for a saving that already costs answers on the set that chose it.
SLACK_QUESTIONS = 1
PERMUTATIONS = 10_000
PERMUTATION_SEED = 20261010


def request_features(question: str) -> np.ndarray:
    """What a question asks, from its text and nothing else.

    named       entities it names: incident, contract and depot identifiers,
                and clients ("Clients A and B" names two)
    aggregate   asks across every record of a kind, which no single document states
    arithmetic  needs a sum, a difference or a penalty worked out
    chain       has to follow one record to another
    numeric     asks for a number only
    """
    clients = 2 if question.startswith("Clients ") else int(" client " in question)
    return np.array([
        float(len(_IDENTIFIER.findall(question)) + clients),
        float(bool(_AGGREGATE.search(question))),
        float(bool(_ARITHMETIC.search(question))),
        float(bool(_CHAIN.search(question))),
        float("number only" in question),
    ])


def rule_prefers_strong(question: str) -> bool:
    """The one-line rule a learned router has to beat: send a question to the
    strong network when it names two or more entities or asks across every
    record of a kind. Written from the select questions the cheap network got
    wrong, not fitted."""
    named, aggregate = request_features(question)[:2]
    return named >= 2 or bool(aggregate)


@dataclass
class Outcomes:
    """One network on one question set, question by question."""

    label: str
    genome_hash: str
    right: dict[str, float]      # share of a question's runs it got right
    dollars: float               # average dollars per question
    dollars_by_time: dict[str, float]  # each chunk's cost split by seconds


def load(path: Path) -> Outcomes:
    """A committed log. Runs that failed on infrastructure, not on the
    question, are left out, as `scripts/select_and_judge.py` leaves them out."""
    runs: dict[str, list[bool]] = {}
    shares: dict[str, list[float]] = {}
    cost, asked, label, genome_hash = 0.0, 0, "", ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        label, genome_hash = chunk["label"], chunk["genome_hash"]
        seconds = sum(r["seconds"] for r in chunk["results"]) or 1.0
        for r in chunk["results"]:
            shares.setdefault(r["task_id"], []).append(chunk["cost"] * r["seconds"] / seconds)
            if not r["infrastructure"]:
                runs.setdefault(r["task_id"], []).append(r["correct"])
        asked += len(chunk["results"])
        cost += chunk["cost"]
    return Outcomes(label, genome_hash,
                    {t: sum(v) / len(v) for t, v in runs.items()},
                    cost / asked if asked else 0.0,
                    {t: float(np.mean(v)) for t, v in shares.items()})


def _model(features: np.ndarray, right: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=1000).fit(features, (right >= 0.5).astype(int))


class Router:
    """The Predictor, one logistic regression per network, and the threshold."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.cheap: LogisticRegression | None = None
        self.strong: LogisticRegression | None = None

    def fit(self, questions: list[str], right_cheap: np.ndarray,
            right_strong: np.ndarray) -> Router:
        features = np.vstack([request_features(q) for q in questions])
        self.cheap = _model(features, right_cheap)
        self.strong = _model(features, right_strong)
        return self

    def gain(self, questions: list[str]) -> np.ndarray:
        """Predicted rise in the chance of a right answer from the strong network."""
        features = np.vstack([request_features(q) for q in questions])
        return (self.strong.predict_proba(features)[:, 1]
                - self.cheap.predict_proba(features)[:, 1])

    def to_strong(self, questions: list[str]) -> np.ndarray:
        return self.gain(questions) >= self.threshold


def leave_one_out_gain(questions: list[str], right_cheap: np.ndarray,
                       right_strong: np.ndarray) -> np.ndarray:
    """Each question's predicted gain from a router that never saw it."""
    gains = np.zeros(len(questions))
    for i in range(len(questions)):
        keep = np.arange(len(questions)) != i
        router = Router().fit([q for q, k in zip(questions, keep, strict=True) if k],
                              right_cheap[keep], right_strong[keep])
        gains[i] = router.gain([questions[i]])[0]
    return gains


def choose_threshold(gains: np.ndarray, right_cheap: np.ndarray,
                     right_strong: np.ndarray,
                     slack: float = SLACK_QUESTIONS) -> float:
    """The highest threshold, so the fewest questions sent to the strong network,
    whose routing scores within `slack` questions of the strong network alone."""
    floor = right_strong.mean() - slack / len(gains)
    for threshold in sorted(set(gains.tolist()), reverse=True):
        routed = np.where(gains >= threshold, right_strong, right_cheap)
        if routed.mean() >= floor - 1e-9:
            return float(threshold)
    return float(gains.min())


@dataclass(frozen=True)
class Point:
    """One routing on one question set."""

    accuracy: float
    dollars: float
    dollars_by_time: float
    share: float  # of questions sent to the strong network


def evaluate(to_strong: np.ndarray, task_ids: list[str], cheap: Outcomes,
             strong: Outcomes) -> Point:
    right = [strong.right[t] if s else cheap.right[t]
             for t, s in zip(task_ids, to_strong, strict=True)]
    by_time = [strong.dollars_by_time[t] if s else cheap.dollars_by_time[t]
               for t, s in zip(task_ids, to_strong, strict=True)]
    share = float(np.mean(to_strong))
    return Point(float(np.mean(right)),
                 share * strong.dollars + (1 - share) * cheap.dollars,
                 float(np.mean(by_time)), share)


def permutation_p(to_strong: np.ndarray, right_cheap: np.ndarray,
                  right_strong: np.ndarray, rounds: int = PERMUTATIONS,
                  seed: int = PERMUTATION_SEED) -> tuple[float, float]:
    """(mean accuracy, p) of sending the same number of questions to the strong
    network at random. p is the share of random routings that score at least
    as well as this one: what the router's choice of questions is worth."""
    observed = np.where(to_strong, right_strong, right_cheap).mean()
    rng = np.random.default_rng(seed)
    scores = np.array([np.where(rng.permutation(to_strong), right_strong, right_cheap).mean()
                       for _ in range(rounds)])
    return float(scores.mean()), float(np.mean(scores >= observed - 1e-12))


def _paired(cheap: Outcomes, strong: Outcomes, tasks) -> tuple[list[str], list[str],
                                                               np.ndarray, np.ndarray]:
    ids = [t.task_id for t in tasks if t.task_id in cheap.right and t.task_id in strong.right]
    text = {t.task_id: t.question for t in tasks}
    return (ids, [text[t] for t in ids], np.array([cheap.right[t] for t in ids]),
            np.array([strong.right[t] for t in ids]))


def _point(point: Point) -> dict:
    return {"accuracy": round(point.accuracy, 4), "dollars": round(point.dollars, 6),
            "dollars_by_time": round(point.dollars_by_time, 6),
            "share_to_strong": round(point.share, 2)}


def analyse(runs: Path = RUNS) -> dict:
    """The routing study on the committed logs. Fitted and tuned on select,
    then applied once to judge."""
    from esp.eval.suites import JUDGE, SELECT

    cheap_select, cheap_judge = (load(runs / p) for p in CHEAP)
    strong_select, strong_judge = (load(runs / p) for p in STRONG)
    _, questions, right_cheap, right_strong = _paired(cheap_select, strong_select, SELECT)

    gains = leave_one_out_gain(questions, right_cheap, right_strong)
    threshold = choose_threshold(gains, right_cheap, right_strong)
    chosen = np.where(gains >= threshold, right_strong, right_cheap)
    router = Router(threshold).fit(questions, right_cheap, right_strong)

    judge_ids, judge_questions, judge_cheap, judge_strong = _paired(
        cheap_judge, strong_judge, JUDGE)
    routed = router.to_strong(judge_questions)
    rule = np.array([rule_prefers_strong(q) for q in judge_questions])
    random_accuracy, p = permutation_p(routed, judge_cheap, judge_strong)

    def alone(to_strong: bool) -> dict:
        return _point(evaluate(np.full(len(judge_ids), to_strong), judge_ids,
                               cheap_judge, strong_judge))

    return {
        "networks": {"cheap": [cheap_select.label, cheap_judge.genome_hash],
                     "strong": [strong_select.label, strong_judge.genome_hash]},
        "select": {"questions": len(questions),
                   "cheap": round(float(right_cheap.mean()), 4),
                   "strong": round(float(right_strong.mean()), 4),
                   "threshold": round(threshold, 4),
                   "leave_one_out_accuracy": round(float(chosen.mean()), 4),
                   "leave_one_out_share": round(float(np.mean(gains >= threshold)), 2)},
        "judge": {"questions": len(judge_ids),
                  "cheap": alone(False), "strong": alone(True),
                  "router": _point(evaluate(routed, judge_ids, cheap_judge, strong_judge)),
                  "rule": _point(evaluate(rule, judge_ids, cheap_judge, strong_judge)),
                  "random_same_share": round(random_accuracy, 4),
                  "p_against_random": round(p, 4),
                  "oracle": round(float(np.maximum(judge_cheap, judge_strong).mean()), 4)},
        "coefficients": {
            "cheap": dict(zip(FEATURE_NAMES, np.round(router.cheap.coef_[0], 2).tolist(),
                              strict=True)),
            "strong": dict(zip(FEATURE_NAMES, np.round(router.strong.coef_[0], 2).tolist(),
                               strict=True))},
    }
