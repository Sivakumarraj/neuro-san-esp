"""Is the network sure? Agreement between two runs of it, as a gate and as a flag.

HASSUM (Knowlton, Guha and Miikkulainen, arXiv:2608.14707) samples a worker's
reply several times, scores the samples' semantic entropy, and has the
orchestrator reprompt or escalate when they disagree. The ICML 2026 position
paper "Agentic AI orchestration should be Bayes-consistent" (Papamarkou et al.,
arXiv:2605.00742) asks the control layer to keep beliefs about what each action
will deliver and to choose by expected utility. This module asks both questions
of neuro-san networks, on runs that were already paid for.

Every judge question (stage 8) and every fresh question (the replication) was
asked twice of each network, so each network already has two samples of every
answer:

* **Semantic entropy over two samples** is 0 when the two answers mean the same
  and ln 2 when they do not: one bit, agree or disagree. Two answers mean the
  same when either states the other's, by the scorer's own
  `esp.eval.tasks.score`. Semantic density needs an embedding of each answer,
  and for answers that are mostly bare numbers ("116" against "136") distance
  in an embedding is not distance in meaning, so it is not computed.
* **As a gate**: run the cheap network twice, keep its answer when the two
  agree, and ask the strong network when they do not. The two networks are the
  routing study's pair (`esp/route.py`), so the gate has the router's bar to
  beat: the one-line rule, which matches the strong network for 14% less.
* **As a flag**: run a network twice and call the answer unsure when the two
  disagree. A flag does not change the answer; it says which ones to check.

**Beliefs** are the Bayes-consistent part. Each is a Beta(1, 1) prior updated
with what judge showed: how often the cheap network is right when its two runs
agree, and when they do not, and how often the strong network is right on the
same two kinds of question. With them, the escalation rule is not tuned: send a
question up when the expected gain in right answers is worth the strong
network's price. They are fitted on judge and checked once on fresh, which no
part of this study was chosen on.

Cost is neuro-san's own count, as in `esp/route.py`: each network's average
dollars per question run, halved for logs that predate accounting version 2.
A gate pays for both cheap runs on every question, whatever it then does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from esp import route
from esp.eval import accounting
from esp.eval.tasks import score

ROOT = route.ROOT
RESULTS = ROOT / "results"

# The pair every gate here uses: the routing study's cheap and strong networks,
# on the two question sets where each was asked every question twice. Judge is
# where beliefs are fitted; fresh is where they are checked.
PAIRS = {
    "judge": ("headtohead/paid-2026-10/8-judge-search10/designer.jsonl",
              "headtohead/paid-2026-10/8-judge-search10/evolved.jsonl"),
    "fresh": ("replication/fresh-2026-10/1-fresh/designer.jsonl",
              "replication/fresh-2026-10/1-fresh/evolved.jsonl"),
}
# The replication's control, the designer's shape with every specialist on
# gpt-5.4-mini. It is a third network to calibrate the flag on, never a gate.
CONTROL = "replication/fresh-2026-10/1-fresh/designer-promoted.jsonl"


def same_meaning(a: str, b: str) -> bool:
    """Whether two replies give the same answer: either states the other's.

    The scorer's containment, read both ways, so "R. Delacroix" and "Delacroix"
    agree and "116" and "136" do not. On every committed pair of runs this is
    the same as comparing the normalised strings, and a test holds it there."""
    return score(a, b) or score(b, a)


@dataclass
class Repeats:
    """One network on one question set, every run of every question kept apart."""

    label: str
    genome_hash: str
    answers: dict[str, list[str]]
    right: dict[str, list[bool]]
    dollars: float  # average per question run

    def agrees(self) -> dict[str, bool]:
        """Per question, whether its first two runs gave the same answer."""
        return {t: same_meaning(a[0], a[1]) for t, a in self.answers.items() if len(a) >= 2}

    def share_right(self) -> dict[str, float]:
        return {t: float(np.mean(r)) for t, r in self.right.items()}


def load(path: Path) -> Repeats:
    """A committed log, in repeat order. A question with any run that failed on
    infrastructure is left out whole: whether two runs agree cannot be read
    from one that never answered."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows.sort(key=lambda r: (r["repeat"], r["chunk"]))
    answers: dict[str, list[str]] = {}
    right: dict[str, list[bool]] = {}
    failed: set[str] = set()
    paid, asked = 0.0, 0
    for row in rows:
        paid += accounting.cost_in(row, route.COST_KEY)
        asked += len(row["results"])
        for r in row["results"]:
            if r["infrastructure"]:
                failed.add(r["task_id"])
            answers.setdefault(r["task_id"], []).append(r["answer"])
            right.setdefault(r["task_id"], []).append(bool(r["correct"]))
    keep = sorted(set(answers) - failed)
    return Repeats(rows[0]["label"], rows[0]["genome_hash"],
                   {t: answers[t] for t in keep}, {t: right[t] for t in keep},
                   paid / asked if asked else 0.0)


@dataclass(frozen=True)
class Calibration:
    """How often a network is right when its two runs agree, and when not."""

    questions: int
    agree: int
    right_if_agree: float     # share of runs right, over the questions that agree
    right_if_disagree: float  # the same over the questions that do not

    def to_dict(self) -> dict:
        return {"questions": self.questions, "agree": self.agree,
                "agree_share": round(self.agree / self.questions, 4),
                "right_if_agree": round(self.right_if_agree, 4),
                "right_if_disagree": round(self.right_if_disagree, 4)}


def calibrate(network: Repeats) -> Calibration:
    agree, right = network.agrees(), network.share_right()
    on = [right[t] for t, a in agree.items() if a]
    off = [right[t] for t, a in agree.items() if not a]
    return Calibration(len(agree), len(on), float(np.mean(on)) if on else float("nan"),
                       float(np.mean(off)) if off else float("nan"))


@dataclass(frozen=True)
class Policy:
    """One way of answering a question set: how often right, at what price."""

    accuracy: float         # over the questions it answers
    dollars: float          # per question asked, every run it makes
    coverage: float         # share of questions it answers; a flag abstains on the rest
    share_to_strong: float  # share of questions the strong network is asked

    def to_dict(self) -> dict:
        return {"accuracy": round(self.accuracy, 4), "dollars": round(self.dollars, 6),
                "coverage": round(self.coverage, 2),
                "share_to_strong": round(self.share_to_strong, 2)}


def policies(cheap: Repeats, strong: Repeats, questions: dict[str, str]) -> dict[str, Policy]:
    """Every way of answering compared here, on the questions both networks ran.

    Accuracy is the share of runs right, as `esp/route.py` counts it. Where two
    cheap runs agree, both are right or both are wrong, so the answer the gate
    keeps is exactly as right as either run."""
    ids = sorted(set(cheap.answers) & set(strong.answers))
    agree = cheap.agrees()
    up = np.array([not agree[t] for t in ids])
    hard = np.array([route.rule_prefers_strong(questions[t]) for t in ids])
    rc = np.array([cheap.share_right()[t] for t in ids])
    rs = np.array([strong.share_right()[t] for t in ids])
    agree_s = strong.agrees()
    sure_s = np.array([agree_s[t] for t in ids])
    c, s = cheap.dollars, strong.dollars

    def mix(to_strong: np.ndarray, dollars: float) -> Policy:
        return Policy(float(np.where(to_strong, rs, rc).mean()), dollars, 1.0,
                      float(to_strong.mean()))

    easy_up = ~hard & up
    return {
        "cheap": mix(np.zeros(len(ids), bool), c),
        "strong": mix(np.ones(len(ids), bool), s),
        "rule": mix(hard, hard.mean() * s + (1 - hard.mean()) * c),
        # Both cheap runs on every question, the strong network on disagreement.
        "gate": mix(up, 2 * c + up.mean() * s),
        # The same choice of questions if knowing the agreement cost nothing: a
        # bound on any cheaper probe, not something that can be bought.
        "gate_if_free": mix(up, c + up.mean() * s),
        # The rule's hard questions straight up, its easy ones through the gate.
        "rule_then_gate": mix(hard | easy_up,
                              hard.mean() * s + (~hard).mean() * 2 * c + easy_up.mean() * s),
        "flag_cheap": Policy(float(rc[~up].mean()), 2 * c, float((~up).mean()), 0.0),
        "flag_strong": Policy(float(rs[sure_s].mean()), 2 * s, float(sure_s.mean()), 1.0),
    }


def dominated_by(name: str, table: dict[str, Policy]) -> list[str]:
    """The policies that answer every question at least as often right for no
    more money. A dominated policy is never the expected-utility choice, at any
    price put on a right answer. Bounds and flags are left out: a flag answers
    fewer questions, and a bound cannot be bought."""
    me = table[name]
    return sorted(other for other, p in table.items()
                  if other != name and p.coverage == 1.0 and other != "gate_if_free"
                  and p.accuracy >= me.accuracy - 1e-12 and p.dollars <= me.dollars + 1e-12
                  and (p.accuracy > me.accuracy + 1e-12 or p.dollars < me.dollars - 1e-12))


@dataclass(frozen=True)
class Belief:
    """A Beta(1, 1) prior updated with fractional successes: each question adds
    the share of its runs that were right."""

    right: float
    questions: int

    @property
    def mean(self) -> float:
        return (1 + self.right) / (2 + self.questions)

    def to_dict(self) -> dict:
        return {"right": round(self.right, 2), "questions": self.questions,
                "mean": round(self.mean, 4)}


def beliefs(cheap: Repeats, strong: Repeats) -> dict[str, Belief]:
    """What each network delivers, given what the cheap network's two runs did."""
    agree, rc, rs = cheap.agrees(), cheap.share_right(), strong.share_right()
    out = {}
    for kind, want in (("agree", True), ("disagree", False)):
        ids = [t for t, a in agree.items() if a == want and t in rs]
        out[f"cheap_if_{kind}"] = Belief(sum(rc[t] for t in ids), len(ids))
        out[f"strong_if_{kind}"] = Belief(sum(rs[t] for t in ids), len(ids))
    return out


def break_even_value(belief: dict[str, Belief], kind: str, strong_dollars: float) -> float:
    """The least a right answer must be worth, in dollars, for sending a
    question up on this signal to raise expected utility: the strong network's
    price over the expected gain in right answers. Infinite when no gain is
    expected."""
    gain = belief[f"strong_if_{kind}"].mean - belief[f"cheap_if_{kind}"].mean
    return strong_dollars / gain if gain > 0 else float("inf")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyse(results: Path = RESULTS) -> dict:
    """The whole study on the committed logs: calibration of every network that
    ran twice, every policy on judge and on fresh, and beliefs fitted on judge
    and checked on fresh."""
    from esp.eval.fresh import FRESH
    from esp.eval.suites import JUDGE

    texts = {"judge": {t.task_id: t.question for t in JUDGE},
             "fresh": {t.task_id: t.question for t in FRESH}}
    loaded = {name: tuple(load(results / p) for p in pair) for name, pair in PAIRS.items()}
    control = load(results / CONTROL)

    sets = {}
    for name, (cheap, strong) in loaded.items():
        table = policies(cheap, strong, texts[name])
        ids = sorted(set(cheap.answers) & set(strong.answers))
        agree = cheap.agrees()
        up = np.array([not agree[t] for t in ids])
        random_accuracy, p = route.permutation_p(
            up, np.array([cheap.share_right()[t] for t in ids]),
            np.array([strong.share_right()[t] for t in ids]))
        d = float(up.mean())
        sets[name] = {
            "questions": len(ids),
            "price_ratio": round(cheap.dollars / strong.dollars, 3),
            # A gate is cheaper than the strong network alone only below this.
            "break_even_ratio": round((1 - d) / 2, 3),
            # How much a probe of the agreement could cost, as a share of one
            # cheap run, for the gate to cost no more than the rule.
            "probe_budget": round((table["rule"].dollars - cheap.dollars - d * strong.dollars)
                                  / cheap.dollars, 2),
            "policies": {k: v.to_dict() for k, v in table.items()},
            "gate_dominated_by": dominated_by("gate", table),
            "random_same_share": round(random_accuracy, 4),
            "p_against_random": round(p, 4),
            "agreed_and_wrong": sorted(t for t in ids
                                       if agree[t] and not any(cheap.right[t])),
            # Of the questions the cheap network's runs disagreed on, how many
            # the one-line rule sends up anyway.
            "disagreed": int(up.sum()),
            "disagreed_and_rule_sends_up": int(sum(
                not agree[t] and route.rule_prefers_strong(texts[name][t]) for t in ids)),
        }

    calibration = {
        "judge": {"cheap": calibrate(loaded["judge"][0]).to_dict(),
                  "strong": calibrate(loaded["judge"][1]).to_dict()},
        "fresh": {"cheap": calibrate(loaded["fresh"][0]).to_dict(),
                  "strong": calibrate(loaded["fresh"][1]).to_dict(),
                  "control": calibrate(control).to_dict()},
    }

    fitted = beliefs(*loaded["judge"])
    observed = beliefs(*loaded["fresh"])
    strong_judge = loaded["judge"][1].dollars
    fresh_agree = calibration["fresh"]["cheap"]["agree_share"]
    held_out = {
        key: {"fitted_on_judge": round(fitted[key].mean, 4),
              "observed_on_fresh": (round(observed[key].right / observed[key].questions, 4)
                                    if observed[key].questions else None),
              "fresh_questions": observed[key].questions}
        for key in fitted}
    held_out["gate_accuracy"] = {
        "fitted_on_judge": round(fresh_agree * fitted["cheap_if_agree"].mean
                                 + (1 - fresh_agree) * fitted["strong_if_disagree"].mean, 4),
        "observed_on_fresh": sets["fresh"]["policies"]["gate"]["accuracy"],
        "fresh_questions": sets["fresh"]["questions"]}

    networks = {name: {"cheap": [c.label, c.genome_hash], "strong": [s.label, s.genome_hash]}
                for name, (c, s) in loaded.items()}
    networks["fresh"]["control"] = [control.label, control.genome_hash]
    sources = [*(p for pair in PAIRS.values() for p in pair), CONTROL]
    return {
        "networks": networks,
        "calibration": calibration,
        "sets": sets,
        "beliefs": {"fitted_on_judge": {k: v.to_dict() for k, v in fitted.items()},
                    "break_even_value": {
                        kind: round(break_even_value(fitted, kind, strong_judge), 4)
                        for kind in ("agree", "disagree")},
                    "held_out": held_out},
        # Which bytes this was computed from, so a figure quoted elsewhere can be
        # traced to the exact logs that produced it.
        "sources": {p: digest(results / p) for p in sources},
    }
