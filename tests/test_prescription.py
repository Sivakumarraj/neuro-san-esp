"""A learned Prescriptor, question by question: the procedure docs/PRESCRIPTOR.md
fixed, checked as written, and every figure quoted from it recomputed from the logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esp import model_router, prescription, prescriptor, route
from esp.eval.accounting import ACCOUNTING

ROOT = Path(__file__).resolve().parent.parent
# One line, so a figure is found wherever the page happens to wrap.
PLAN = " ".join((ROOT / "docs" / "PRESCRIPTOR.md").read_text(encoding="utf-8")
                .replace("\N{MINUS SIGN}", "-").split())
ACTIONS = prescription.ACTIONS


def _member(accuracy: float, dollars: float) -> prescriptor.Evolved:
    return prescriptor.Evolved(None, accuracy, dollars, ())


def test_the_chosen_prescriptor_is_the_most_accurate_within_the_budget():
    front = [_member(0.90, 1.0), _member(0.95, 1.4), _member(0.99, 2.0)]
    assert prescription.choose(front, 1.5).accuracy == 0.95


def test_a_tie_on_accuracy_goes_to_the_cheaper_prescriptor():
    front = [_member(0.95, 1.4), _member(0.95, 1.2)]
    assert prescription.choose(front, 1.5).dollars == 1.2


def test_a_front_wholly_over_the_budget_is_an_error():
    with pytest.raises(ValueError):
        prescription.choose([_member(0.9, 2.0)], 1.5)


def test_the_static_prescription_is_the_most_accurate_action_within_the_budget():
    right = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    dollars = np.tile([1.0, 3.0, 1.2], (3, 1))
    # The second action is the most accurate and over the budget.
    assert prescription.static_choice(right, dollars, 1.5) == 2
    assert prescription.static_choice(right, dollars, 3.0) == 1


def test_a_tie_between_actions_goes_to_the_cheaper():
    right = np.ones((4, 2))
    assert prescription.static_choice(right, np.tile([2.0, 1.0], (4, 1)), 5.0) == 1


def test_no_action_within_the_budget_is_an_error():
    with pytest.raises(ValueError):
        prescription.static_choice(np.ones((2, 2)), np.full((2, 2), 3.0), 1.0)


def test_an_action_never_wrong_in_training_predicts_its_own_score():
    features = np.arange(8, dtype=float).reshape(4, 2)
    assert prescription.chance(features, np.ones(4), features).tolist() == [1.0] * 4
    assert prescription.chance(features, np.zeros(4), features).tolist() == [0.0] * 4


def test_the_leave_one_out_chance_never_uses_a_question_s_own_outcome():
    """Flip one question's outcome and its own chance must not move, since the
    Predictor that scores it never saw it."""
    rng = np.random.default_rng(5)
    features = rng.integers(0, 2, size=(24, 3)).astype(float)
    right = np.column_stack([features[:, 0], 1 - features[:, 1]])
    before = prescription.leave_one_out(features, right)
    flipped = right.copy()
    flipped[0] = 1 - flipped[0]
    after = prescription.leave_one_out(features, flipped)
    assert np.array_equal(before[0], after[0])
    assert not np.array_equal(before[1:], after[1:])
    assert np.array_equal(before, np.round(before, 6))


def _table(right: np.ndarray, dollars: list[float], tokens: list[float]) -> prescription.Table:
    questions = len(right)
    return prescription.Table(
        [f"Q{i:03d}" for i in range(questions)], np.zeros((questions, 5)), right,
        np.tile(dollars, (questions, 1)), np.tile(tokens, (questions, 1)),
        [("judge", i // 4) for i in range(questions)])


def test_prescribing_one_action_everywhere_cannot_beat_itself():
    right = np.ones((40, 4))
    right[::4, 0] = 0.0               # the designer's shape misses a quarter
    test = _table(right, [0.0014, 0.0012, 0.0028, 0.0008], [5000, 4500, 3200, 3000])
    c3 = ACTIONS.index("c3")
    result = prescription.judge(np.full(40, c3), test, c3)
    assert result["one_action_only"] and result["shares"]["c3"] == 1.0
    assert result["against_baseline"]["all_hold"]
    static = result["against_static"]
    assert static["static"] == "c3" and not static["holds"]
    assert static["accuracy"]["p"] == 1.0 and static["dollars"]["p"] == 1.0
    assert not result["against_random"]["holds"] and not result["passes"]


def test_sending_the_right_questions_to_the_strong_action_passes_every_mark():
    right = np.ones((40, 4))
    hard = np.arange(40) % 4 == 1            # one question in every chunk
    right[hard, 0] = right[hard, 1] = right[hard, 3] = 0.0
    test = _table(right, [0.0014, 0.0012, 0.0016, 0.0008], [5000, 4500, 3200, 3000])
    mini, c3 = ACTIONS.index("2cc4-mini"), ACTIONS.index("c3")
    result = prescription.judge(np.where(hard, mini, c3), test, c3)
    assert result["shares"] == {"designer": 0.0, "2cc4-nano": 0.0, "2cc4-mini": 0.25,
                                "c3": 0.75}
    assert result["against_baseline"]["holds"] == {
        "accuracy": True, "tokens": True, "dollars": True, "agents": True, "fitness": True}
    assert result["against_static"]["holds"] and result["against_random"]["holds"]
    assert result["against_static"]["accuracy"]["prescribed"] == 1.0
    assert result["passes"]


def test_sending_the_wrong_questions_up_is_no_better_than_chance():
    right = np.ones((40, 4))
    hard = np.arange(40) % 4 == 1
    right[hard, 0] = right[hard, 1] = right[hard, 3] = 0.0
    test = _table(right, [0.0014, 0.0012, 0.0016, 0.0008], [5000, 4500, 3200, 3000])
    mini, c3 = ACTIONS.index("2cc4-mini"), ACTIONS.index("c3")
    result = prescription.judge(np.where(np.arange(40) % 4 == 2, mini, c3), test, c3)
    assert not result["against_static"]["holds"] and not result["against_random"]["holds"]
    assert not result["passes"]


def test_a_prescription_that_uses_the_designer_s_shape_has_no_fewer_agents():
    test = _table(np.ones((8, 4)), [0.0014, 0.0012, 0.0016, 0.0008],
                  [5000, 4500, 3200, 3000])
    prescribed = np.array([0, 3, 3, 3, 3, 3, 3, 3])
    result = prescription.judge(prescribed, test, 3)
    assert result["agents"] == 4 and not result["against_baseline"]["holds"]["agents"]


def test_knowing_every_answer_takes_the_cheapest_action_that_is_right():
    right = np.array([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 1.0, 0.5]])
    test = _table(right, [0.0014, 0.0012, 0.0016, 0.0008], [5000, 4500, 3200, 3000])
    bound = prescription.judge(np.full(4, 3), test, 3)["knowing_every_answer"]
    assert bound["accuracy"] == 0.75
    # The cheapest right action each time: 2cc4-nano, c3, c3 (none right), 2cc4-mini.
    assert bound["shares"] == {"designer": 0.0, "2cc4-nano": 0.25, "2cc4-mini": 0.25,
                               "c3": 0.5}


def test_random_draws_keep_the_number_of_questions_each_action_gets():
    right = np.zeros((20, 4))
    right[:5, 2] = 1.0
    prescribed = np.array([2] * 5 + [3] * 15)
    accuracy, p = prescription.at_random(prescribed, right)
    # Five questions on the strong action at random are right a quarter of the time.
    assert accuracy == pytest.approx(5 * 5 / 20 / 20, abs=0.01)
    assert p < 0.01


def _logs(folder: Path, name: str, repeats: int) -> dict[str, Path]:
    """Four logs of one set in which only 2cc4-mini answers an aggregate
    question, the other three answer everything else, and 2cc4-mini misses the
    questions that follow one record to another."""
    text = model_router._questions(name)
    ids = list(text)
    features = {t: route.request_features(text[t]) for t in ids}
    aggregate = {t for t in ids if features[t][1] == 1}
    chained = {t for t in ids if features[t][3] == 1} - aggregate
    usage = {"designer": (5000, 0.0014), "2cc4-nano": (4500, 0.0012),
             "2cc4-mini": (3200, 0.0015), "c3": (3000, 0.0008)}
    paths = {}
    for action, (tokens, dollars) in usage.items():
        rows = []
        for repeat in range(repeats):
            for chunk in range(len(ids) // 4):
                rows.append({
                    "label": action, "chunk": chunk, "repeat": repeat,
                    "genome_hash": prescription.HASHES[action],
                    "settings": {"search_results": 10}, "accounting": ACCOUNTING,
                    "tokens": 4 * tokens, "reported_cost": 4 * dollars, "cost": 4 * dollars,
                    "results": [{"task_id": t, "infrastructure": False, "seconds": 10.0,
                                 "correct": (t not in chained if action == "2cc4-mini"
                                             else t not in aggregate)}
                                for t in ids[4 * chunk:4 * chunk + 4]]})
        path = folder / name / f"{action}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paths[action] = path
    return paths


def test_a_prescriptor_learns_to_send_the_questions_one_action_alone_answers(tmp_path):
    logs = {"select": _logs(tmp_path, "select", 1), "judge": _logs(tmp_path, "judge", 2)}
    result = prescription.turn(("select",), ("judge",), logs)
    assert result["static"] == "c3" and not result["chosen"]["one_action_only"]
    judge = result["sets"]["judge"]
    text = model_router._questions("judge")
    aggregates = np.mean([route.request_features(q)[1] for q in text.values()])
    # Every aggregate goes to the one action that answers it, and few others do.
    assert aggregates <= judge["shares"]["2cc4-mini"] < aggregates + 0.1
    assert judge["shares"]["designer"] == judge["shares"]["2cc4-nano"] == 0.0
    assert judge["against_static"]["accuracy"]["prescribed"] == 1.0
    assert judge["against_static"]["accuracy"]["other"] == pytest.approx(1 - aggregates)
    assert judge["passes"]
    assert judge["across_seeds"]["seeds"] == len(prescription.SEEDS)
    # Every member of the front is reported with what it really scored.
    assert len(judge["front"]) == result["front"]


def test_a_log_of_another_network_is_refused(tmp_path):
    logs = _logs(tmp_path, "select", 1)
    logs["c3"] = logs["designer"]
    with pytest.raises(ValueError):
        prescription.table("select", logs)


def test_a_log_that_misses_a_question_is_refused(tmp_path):
    logs = _logs(tmp_path, "select", 1)
    kept = logs["c3"].read_text(encoding="utf-8").splitlines()[:-1]
    logs["c3"].write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        prescription.table("select", logs)


def test_the_networks_and_settings_the_plan_names_are_the_ones_used():
    for action in ACTIONS:
        assert prescription.HASHES[action] in PLAN
        assert f"`{action}`" in PLAN
    assert prescription.TURNS == (("first", ("select",), ("judge", "fresh")),
                                  ("second", ("select", "judge"), ("fresh",)))
    assert f"seed {prescriptor.SEED}" in PLAN and prescription.SEEDS[0] == prescriptor.SEED
    assert f"{prescription.SEEDS[0]} to {prescription.SEEDS[-1]}" in PLAN
    assert (f"5 features to {prescriptor.HIDDEN} tanh units to {len(ACTIONS)} outputs, "
            f"{prescriptor.size(5, len(ACTIONS))} weights") in PLAN
    assert (f"{prescriptor.POPULATION} prescriptors for {prescriptor.GENERATIONS} generations"
            in PLAN)
    for folders in prescription.LOGS.values():
        for path in folders.values():
            assert f"`{path.parent.relative_to(ROOT).as_posix()}/`" in PLAN, path


def test_what_the_plan_says_of_select_is_what_the_select_logs_say():
    """Everything the plan records before the judge questions were touched,
    recomputed from the select logs alone."""
    fitted = prescription.prepare(prescription.table("select", prescription.LOGS["select"]))
    train = fitted.train
    assert f"designer's shape's ${fitted.budget:.6f} a question" in PLAN
    assert ACTIONS[fitted.static] == "c3" and "**The static prescription is `c3`**" in PLAN
    for i, action in enumerate(ACTIONS):
        row = (f"| `{action}` | {train.right[:, i].mean():.1%} | "
               f"{fitted.predicted[:, i].mean():.1%} | ${round(float(fitted.prices[i]), 6)} |")
        assert row in PLAN, row

    mini, static = ACTIONS.index("2cc4-mini"), fitted.static
    room = int(np.floor((fitted.budget - fitted.prices[static])
                        / (fitted.prices[mini] - fitted.prices[static]) * len(train.ids)))
    assert f"the budget allows {room} of the 60 questions on `2cc4-mini`" in PLAN
    gain = 100 * (fitted.predicted[:, mini] - fitted.predicted[:, static])
    assert (f"positive on {int((gain > 0).sum())} of the 60 questions, between "
            f"{gain.min():.1f} and {gain.max():+.1f} points, and "
            f"{np.median(gain):+.1f} at the median") in PLAN

    front = prescription.evolved(fitted, prescription.SEEDS[0])
    chosen = prescription.choose(front, fitted.budget)
    counts = {a: chosen.prescribed.count(i) for i, a in enumerate(ACTIONS)}
    assert f"**The front** has {len(front)} prescriptors" in PLAN
    assert f"`c3` everywhere at ${front[0].dollars:.6f} a question" in PLAN
    last = np.array(front[-1].prescribed)
    assert (f"{np.mean(last == mini):.0%} of the questions to `2cc4-mini` at "
            f"${front[-1].dollars:.6f}") in PLAN
    assert (f"sends {counts['2cc4-mini']} of the 60 questions to `2cc4-mini` and "
            f"{counts['c3']} to `c3`") in PLAN
    assert counts["designer"] == counts["2cc4-nano"] == 0
    assert (f"expects {chosen.accuracy:.1%} of it at ${chosen.dollars:.6f} a question, "
            f"against {fitted.predicted[:, static].mean():.1%} of `c3` alone") in PLAN

    rows = np.arange(len(train.ids))
    right = round(float(train.right[rows, np.array(chosen.prescribed)].sum()))
    assert (f"right on {right} of 60 against `c3`'s "
            f"{round(float(train.right[:, static].sum()))}") in PLAN


def test_every_seed_chooses_the_same_select_questions():
    fitted = prescription.prepare(prescription.table("select", prescription.LOGS["select"]))
    chosen = {prescription.choose(prescription.evolved(fitted, seed), fitted.budget).prescribed
              for seed in prescription.SEEDS}
    assert len(chosen) == 1
    assert f"All {len(prescription.SEEDS)} seeds" in PLAN
