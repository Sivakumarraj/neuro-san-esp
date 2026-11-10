"""A learned Prescriptor, question by question: the test fixed in
docs/PRESCRIPTOR.md before it ran.

Every earlier test here prescribes with a rule a person wrote: one network for
every question, or a threshold on a predicted gain. ESP proper has a second
learned model, the Prescriptor, evolved against the Predictor
(`esp.prescriptor`). This runs ESP as its paper describes it, on the four
networks measured on all three question sets at 10 documents a search:

* **Context**: a question's text features (`esp.route.request_features`).
* **Actions**: which of the four networks answers it.
* **Outcomes**: whether the answer is right, and what it cost.
* **Predictor**: per action, a logistic regression from the features to the
  chance of a right answer (`esp.route`'s, unchanged), and the action's average
  cost a question. The Prescriptor is evolved against each training question's
  leave-one-out chance, so no question's own outcome tells it where to go.
* **Prescriptor**: a neural network from the features to an action, evolved with
  NSGA-II on predicted accuracy and predicted dollars together. The result is a
  front of prescriptors.
* **Choice**: from the front, the most accurate by the Predictor whose predicted
  cost a question is at or below the designer's shape's on the training
  questions; a tie goes to the cheaper.

ESP is a loop: what its prescriptions lead to becomes training data. So there
are two turns. The first is trained on select and judged on judge and fresh.
The second adds the judge outcomes to the training data and is judged on fresh.

A Prescriptor has to beat prescribing nothing: the single network the same
budget rule picks from the training questions, without reading the question.
`judge` checks the five marks every test here uses against the designer's
shape, then that, then whether the questions it chose matter at all.

It reads the committed logs and nothing else, so it costs nothing and needs no key.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from esp import model_router, prescriptor, route
from esp.eval.pricing import fitness_dollars
from esp.eval.stats import mcnemar_exact
from esp.evolve.loop import scalarise

ROOT = Path(__file__).resolve().parent.parent
PAID = ROOT / "results" / "headtohead" / "paid-2026-10"
STRUCTURE = ROOT / "results" / "structure"
ROUTER = ROOT / "results" / "router"
ALL_FOUR = ROOT / "results" / "all-four"

# Every network measured on select, judge and fresh at 10 documents a search.
ACTIONS = ("designer", "2cc4-nano", "2cc4-mini", "c3")
BASELINE = "designer"
HASHES = {"designer": "0378ca26cd9fe977", "2cc4-nano": "2cc4231b7d3fb81d",
          "2cc4-mini": "47322479eb9cba1e", "c3": "f4e9e494442f9888"}
AGENTS = {"designer": 4, "2cc4-nano": 2, "2cc4-mini": 2, "c3": 2}
DESCRIBED = {
    "designer": "the designer's shape, every agent on gpt-5.4-nano",
    "2cc4-nano": "2cc4, every agent on gpt-5.4-nano",
    "2cc4-mini": "2cc4 with its Researcher on gpt-5.4-mini",
    "c3": "2cc4 with its Researcher on gpt-4o-mini and a calculator on its Coordinator",
}
LOGS = {
    "select": {"designer": PAID / "7-select-search10" / "designer.jsonl",
               "2cc4-nano": PAID / "7-select-search10" / "2cc4.jsonl",
               "2cc4-mini": PAID / "7-select-search10" / "2cc4-mini.jsonl",
               "c3": ALL_FOUR / "select-2026-11" / "c3.jsonl"},
    "judge": {"designer": STRUCTURE / "judge-nano-2026-11" / "designer.jsonl",
              "2cc4-nano": STRUCTURE / "judge-nano-2026-11" / "2cc4-nano.jsonl",
              "2cc4-mini": ROUTER / "judge-2026-11" / "2cc4-mini.jsonl",
              "c3": ALL_FOUR / "judge-2026-11" / "c3.jsonl"},
    "fresh": {"designer": STRUCTURE / "fresh-nano-2026-11" / "designer.jsonl",
              "2cc4-nano": STRUCTURE / "fresh-nano-2026-11" / "2cc4-nano.jsonl",
              "2cc4-mini": STRUCTURE / "fresh-nano-2026-11" / "evolved.jsonl",
              "c3": ALL_FOUR / "fresh-2026-11" / "c3.jsonl"},
}
# (name, the sets it is trained on, the sets it is judged on)
TURNS = (("first", ("select",), ("judge", "fresh")),
         ("second", ("select", "judge"), ("fresh",)))

ALPHA = model_router.ALPHA
EXAM = model_router.EXAM
# The verdict is the first seed's. The rest say how much of it is the seed.
SEEDS = tuple(range(prescriptor.SEED, prescriptor.SEED + 20))
PERMUTATIONS = 10_000
PERMUTATION_SEED = 20261110
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261110


@dataclass
class Table:
    """Every action's outcome on every question of one or more sets."""

    ids: list[str]
    features: np.ndarray       # question by feature
    right: np.ndarray          # question by action: share of its runs right
    dollars: np.ndarray        # question by action: its chunk's dollars a question
    tokens: np.ndarray         # question by action: its chunk's tokens a question
    chunks: list[tuple[str, int]]   # each question's (set, chunk)


def table(name: str, logs: dict[str, Path]) -> Table:
    """One question set. Refused unless every action's log covers every question
    of it, in the same four-question chunks."""
    read = {a: model_router.read_log(logs[a], HASHES[a], a) for a in ACTIONS}
    text = model_router._questions(name)
    for action, log in read.items():
        if set(log.right) != set(text):
            raise ValueError(f"{name}: the {action} log covers {len(log.right)} of "
                             f"{len(text)} questions")
    ids = list(text)
    for t in ids:
        if len({read[a].chunk_of[t] for a in ACTIONS}) != 1:
            raise ValueError(f"{name}: {t} sits in different chunks")
    return Table(
        ids, np.vstack([route.request_features(text[t]) for t in ids]),
        np.array([[read[a].right[t] for a in ACTIONS] for t in ids]),
        np.array([[read[a].dollars[read[a].chunk_of[t]] for a in ACTIONS] for t in ids]),
        np.array([[read[a].tokens[read[a].chunk_of[t]] for a in ACTIONS] for t in ids]),
        [(name, read[BASELINE].chunk_of[t]) for t in ids])


def stack(tables: list[Table]) -> Table:
    return Table([t for part in tables for t in part.ids],
                 np.vstack([part.features for part in tables]),
                 np.vstack([part.right for part in tables]),
                 np.vstack([part.dollars for part in tables]),
                 np.vstack([part.tokens for part in tables]),
                 [c for part in tables for c in part.chunks])


def chance(train_features: np.ndarray, train_right: np.ndarray,
           features: np.ndarray) -> np.ndarray:
    """One action's Predictor: the chance of a right answer for each context.
    An action never wrong, or never right, in training has nothing to fit."""
    labels = train_right >= 0.5
    if labels.all() or not labels.any():
        return np.full(len(features), float(labels[0]))
    return route._model(train_features, train_right).predict_proba(features)[:, 1]


def leave_one_out(features: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Question by action: each question's chance from a Predictor that never saw
    it. Rounded, so the evolution that reads it is the same on every machine."""
    out = np.zeros_like(right, dtype=float)
    for i in range(len(features)):
        keep = np.arange(len(features)) != i
        for action in range(right.shape[1]):
            out[i, action] = chance(features[keep], right[keep, action], features[[i]])[0]
    return np.round(out, 6)


def static_choice(right: np.ndarray, dollars: np.ndarray, budget: float) -> int:
    """Prescribing without reading the question: the single action, measured most
    accurate on the training questions, whose cost a question is within the
    budget. A tie goes to the cheaper."""
    accuracy, price = right.mean(axis=0), dollars.mean(axis=0)
    within = [a for a in range(len(price)) if price[a] <= budget + 1e-12]
    if not within:
        raise ValueError("no action costs as little as the budget")
    return min(within, key=lambda a: (-round(float(accuracy[a]), 9), price[a]))


def choose(front: list[prescriptor.Evolved], budget: float) -> prescriptor.Evolved:
    """The prescriptor that is judged: the most accurate by the Predictor whose
    predicted cost a question is within the budget. A tie goes to the cheaper."""
    within = [member for member in front if member.dollars <= budget + 1e-12]
    if not within:
        raise ValueError("no prescriptor on the front costs as little as the budget")
    return min(within, key=lambda member: (-round(member.accuracy, 9), member.dollars))


@dataclass
class Fitted:
    """One turn before it is judged: everything decided on its training questions."""

    train: Table
    predicted: np.ndarray      # leave-one-out chance, question by action
    prices: np.ndarray         # each action's average dollars a question
    budget: float
    static: int


def prepare(train: Table) -> Fitted:
    base = ACTIONS.index(BASELINE)
    budget = float(train.dollars[:, base].mean())
    return Fitted(train, leave_one_out(train.features, train.right),
                  np.round(train.dollars.mean(axis=0), 9), budget,
                  static_choice(train.right, train.dollars, budget))


def evolved(fitted: Fitted, seed: int) -> list[prescriptor.Evolved]:
    prices = np.tile(fitted.prices, (len(fitted.train.ids), 1))
    return prescriptor.evolve(fitted.train.features, fitted.predicted, prices, seed=seed)


def _shares(prescribed: np.ndarray) -> dict[str, float]:
    return {a: round(float(np.mean(prescribed == i)), 4) for i, a in enumerate(ACTIONS)}


def _by_chunk(values: np.ndarray, chunks: list[tuple[str, int]]) -> np.ndarray:
    order = sorted(set(chunks))
    return np.array([values[[c == chunk for c in chunks]].mean() for chunk in order])


def _lower(prescribed: np.ndarray, other: np.ndarray, chunks: list[tuple[str, int]]) -> dict:
    """Usage a question against another prescription, tested over the chunks."""
    a, b = _by_chunk(prescribed, chunks), _by_chunk(other, chunks)
    # Identical in every chunk: the same prescription, and nothing to test.
    p = 1.0 if np.allclose(a, b) else float(wilcoxon(a, b, alternative="two-sided").pvalue)
    return {"prescribed": round(float(prescribed.mean()), 6),
            "other": round(float(other.mean()), 6), "chunks": len(a),
            "prescribed_lower_in": int((a < b).sum()), "p": round(p, 6),
            "change": round(float(a.sum() / b.sum() - 1), 4)}


def _bootstrap(diffs: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = diffs[rng.integers(0, len(diffs), size=(BOOTSTRAP_DRAWS, len(diffs)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return [round(100 * float(low), 1), round(100 * float(high), 1)]


def _against(prescribed_right: np.ndarray, other_right: np.ndarray, ids: list[str]) -> dict:
    """Accuracy, question by question, against another prescription."""
    better = [t for t, p, o in zip(ids, prescribed_right, other_right, strict=True) if p > o]
    worse = [t for t, p, o in zip(ids, prescribed_right, other_right, strict=True) if p < o]
    return {"prescribed": round(float(prescribed_right.mean()), 4),
            "other": round(float(other_right.mean()), 4),
            "prescribed_better_on": better, "other_better_on": worse,
            "p": round(mcnemar_exact(len(better), len(worse)), 4),
            "points": round(100 * float(prescribed_right.mean() - other_right.mean()), 1),
            "ci95_points": _bootstrap(prescribed_right - other_right)}


def at_random(prescribed: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """(mean accuracy, p) of handing out the same number of questions to each
    action at random. p is the share of such draws at least as accurate as this
    prescription: what its choice of questions is worth."""
    rows = np.arange(len(prescribed))
    observed = right[rows, prescribed].mean()
    rng = np.random.default_rng(PERMUTATION_SEED)
    scores = np.array([right[rows, rng.permutation(prescribed)].mean()
                       for _ in range(PERMUTATIONS)])
    return float(scores.mean()), float(np.mean(scores >= observed - 1e-12))


def judge(prescribed: np.ndarray, test: Table, static: int) -> dict:
    """One prescription on one question set, against every mark the plan names."""
    rows = np.arange(len(test.ids))
    base = ACTIONS.index(BASELINE)
    right = test.right[rows, prescribed]
    tokens, dollars = test.tokens[rows, prescribed], test.dollars[rows, prescribed]
    agents = max(AGENTS[ACTIONS[a]] for a in set(prescribed.tolist()))

    accuracy = _against(right, test.right[:, base], test.ids)
    usage = {"tokens": _lower(tokens, test.tokens[:, base], test.chunks),
             "dollars": _lower(dollars, test.dollars[:, base], test.chunks)}
    fitness = {
        "prescribed": {
            "token": round(scalarise(float(right.mean()), round(float(tokens.mean()) * EXAM),
                                     agents), 4),
            "dollar": round(fitness_dollars(float(right.mean()), float(dollars.mean()),
                                            agents), 4)},
        "baseline": {
            "token": round(scalarise(float(test.right[:, base].mean()),
                                     round(float(test.tokens[:, base].mean()) * EXAM),
                                     AGENTS[BASELINE]), 4),
            "dollar": round(fitness_dollars(float(test.right[:, base].mean()),
                                            float(test.dollars[:, base].mean()),
                                            AGENTS[BASELINE]), 4)}}
    holds = {
        "accuracy": (accuracy["points"] > 0 and len(accuracy["prescribed_better_on"])
                     > len(accuracy["other_better_on"]) and accuracy["p"] < ALPHA),
        "tokens": (usage["tokens"]["prescribed"] < usage["tokens"]["other"]
                   and usage["tokens"]["p"] < ALPHA),
        "dollars": (usage["dollars"]["prescribed"] < usage["dollars"]["other"]
                    and usage["dollars"]["p"] < ALPHA),
        "agents": agents < AGENTS[BASELINE],
        "fitness": all(fitness["prescribed"][k] > fitness["baseline"][k]
                       for k in ("token", "dollar")),
    }

    # The Prescriptor against prescribing nothing: one action for every question.
    against_static = _against(right, test.right[:, static], test.ids)
    static_dollars = _lower(dollars, test.dollars[:, static], test.chunks)
    random_accuracy, p_random = at_random(prescribed, test.right)
    beats_static = (against_static["points"] > 0
                    and len(against_static["prescribed_better_on"])
                    > len(against_static["other_better_on"])
                    and against_static["p"] < ALPHA)
    reads_the_question = p_random < ALPHA

    # Not a prescriptor and not a test: for each question, the cheapest action
    # among those that got it right most often. A bound on any Predictor.
    best = np.array([min(range(len(ACTIONS)),
                         key=lambda a, i=i: (-test.right[i, a], test.dollars[i, a]))
                     for i in rows])
    return {
        "questions": len(test.ids), "shares": _shares(prescribed), "agents": agents,
        "one_action_only": len(set(prescribed.tolist())) == 1,
        "against_baseline": {"accuracy": accuracy, "tokens": usage["tokens"],
                             "dollars": usage["dollars"], "fitness": fitness,
                             "holds": holds, "all_hold": all(holds.values())},
        "against_static": {"static": ACTIONS[static], "accuracy": against_static,
                           "dollars": static_dollars, "holds": beats_static},
        "against_random": {"same_shares_accuracy": round(random_accuracy, 4),
                           "p": round(p_random, 4), "holds": reads_the_question},
        "passes": all(holds.values()) and beats_static and reads_the_question,
        "single_actions": {a: {"accuracy": round(float(test.right[:, i].mean()), 4),
                               "dollars": round(float(test.dollars[:, i].mean()), 6)}
                           for i, a in enumerate(ACTIONS)},
        "knowing_every_answer": {"accuracy": round(float(test.right[rows, best].mean()), 4),
                                 "dollars": round(float(test.dollars[rows, best].mean()), 6),
                                 "shares": _shares(best)},
    }


def _front_on(front: list[prescriptor.Evolved], test: Table) -> list[dict]:
    """Each member of the front as it really did on a question set, and whether
    one action alone was at least as accurate for no more money."""
    rows = np.arange(len(test.ids))
    single = [(float(test.right[:, a].mean()), float(test.dollars[:, a].mean()))
              for a in range(len(ACTIONS))]
    out = []
    for member in front:
        prescribed = member.prescriptor.prescribe(test.features)
        accuracy = float(test.right[rows, prescribed].mean())
        dollars = float(test.dollars[rows, prescribed].mean())
        beaten = any(a >= accuracy - 1e-12 and d <= dollars + 1e-12
                     and (a > accuracy + 1e-12 or d < dollars - 1e-12) for a, d in single)
        out.append({"predicted": {"accuracy": round(member.accuracy, 4),
                                  "dollars": round(member.dollars, 6)},
                    "accuracy": round(accuracy, 4), "dollars": round(dollars, 6),
                    "shares": _shares(prescribed), "beaten_by_one_action": beaten})
    return out


def _across_seeds(fitted: Fitted, test: Table, first: np.ndarray) -> dict:
    """The chosen prescriptor of every seed on one question set: how much of the
    first seed's result is the seed."""
    rows = np.arange(len(test.ids))
    accuracy, dollars, same = [], [], 0
    for seed in SEEDS:
        prescribed = choose(evolved(fitted, seed), fitted.budget).prescriptor.prescribe(
            test.features)
        accuracy.append(float(test.right[rows, prescribed].mean()))
        dollars.append(float(test.dollars[rows, prescribed].mean()))
        same += int(np.array_equal(prescribed, first))
    return {"seeds": len(SEEDS), "same_prescriptions_as_the_first": same,
            "accuracy": [round(min(accuracy), 4), round(float(np.median(accuracy)), 4),
                         round(max(accuracy), 4)],
            "dollars": [round(min(dollars), 6), round(float(np.median(dollars)), 6),
                        round(max(dollars), 6)]}


def turn(train_on: tuple[str, ...], judged_on: tuple[str, ...],
         logs: dict[str, dict[str, Path]] | None = None) -> dict:
    """One turn of the loop: fit on the training sets, then judge."""
    logs = logs or LOGS
    fitted = prepare(stack([table(name, logs[name]) for name in train_on]))
    front = evolved(fitted, SEEDS[0])
    chosen = choose(front, fitted.budget)
    train = fitted.train
    out: dict = {
        "trained_on": list(train_on), "questions": len(train.ids),
        "budget": round(fitted.budget, 6), "static": ACTIONS[fitted.static],
        "training": {a: {"accuracy": round(float(train.right[:, i].mean()), 4),
                         "predicted": round(float(fitted.predicted[:, i].mean()), 4),
                         "dollars": round(float(fitted.prices[i]), 6)}
                     for i, a in enumerate(ACTIONS)},
        "front": len(front),
        "chosen": {"predicted": {"accuracy": round(chosen.accuracy, 4),
                                 "dollars": round(chosen.dollars, 6)},
                   "shares": _shares(np.array(chosen.prescribed)),
                   "one_action_only": len(set(chosen.prescribed)) == 1},
        "sets": {}}
    for name in judged_on:
        test = table(name, logs[name])
        prescribed = chosen.prescriptor.prescribe(test.features)
        out["sets"][name] = {**judge(prescribed, test, fitted.static),
                             "front": _front_on(front, test),
                             "across_seeds": _across_seeds(fitted, test, prescribed)}
    return out


def analyse(logs: dict[str, dict[str, Path]] | None = None) -> dict:
    """Both turns, from the committed logs."""
    logs = logs or LOGS
    out: dict = {
        "plan": "docs/PRESCRIPTOR.md", "alpha": ALPHA,
        "actions": {a: {"genome": HASHES[a], "agents": AGENTS[a]} for a in ACTIONS},
        "baseline": BASELINE,
        "evolution": {"hidden": prescriptor.HIDDEN, "population": prescriptor.POPULATION,
                      "generations": prescriptor.GENERATIONS, "seed": SEEDS[0],
                      "seeds": len(SEEDS)},
        "turns": {name: turn(train_on, judged_on, logs)
                  for name, train_on, judged_on in TURNS},
        "logs": {}}
    for paths in logs.values():
        for path in paths.values():
            shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            out["logs"][str(shown)] = model_router.sha256(path)
    return out
