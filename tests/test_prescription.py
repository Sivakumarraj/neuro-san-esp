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


SUMMARY = ROOT / "results" / "prescriptor" / "summary.json"
PUBLISHED = " ".join(
    "\n".join((ROOT / name).read_text(encoding="utf-8") for name in ("README.md",
                                                                      "docs/RESULTS.md"))
    .replace("\N{MINUS SIGN}", "-").replace("**", "").split())
ROWS = (("first", "judge", "First turn, judge"), ("first", "fresh", "First turn, fresh"),
        ("second", "fresh", "Second turn, fresh"))


def _study() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_the_committed_summary_is_what_the_logs_say():
    assert prescription.analyse() == _study()


def test_the_prescriptor_did_not_pass_and_the_pages_say_so():
    study = _study()
    for turn, name, _ in ROWS:
        result = study["turns"][turn]["sets"][name]
        assert not result["passes"]
        assert not result["against_static"]["holds"] and not result["against_random"]["holds"]
        assert result["against_baseline"]["holds"] == {
            "accuracy": True, "tokens": True, "dollars": False, "agents": True, "fitness": True}
    assert "The Prescriptor does not pass." in PLAN
    assert "The Prescriptor does not pass." in PUBLISHED
    assert "It does not pass." in (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")


def test_every_row_of_the_result_table_is_what_the_logs_give():
    study = _study()
    for turn, name, title in ROWS:
        result = study["turns"][turn]["sets"][name]
        static, random = result["against_static"], result["against_random"]
        accuracy = static["accuracy"]
        row = (f"| {title} | {result['shares']['2cc4-mini']:.0%} | "
               f"{accuracy['prescribed']:.1%} | {accuracy['other']:.1%} | "
               f"{len(accuracy['prescribed_better_on'])}, {len(accuracy['other_better_on'])}, "
               f"p = {accuracy['p']} | {random['same_shares_accuracy']:.1%}, "
               f"p = {random['p']} | {static['dollars']['change']:+.1%} |")
        assert row in PLAN, row
        assert row in PUBLISHED, row


def test_the_first_turn_on_judge_is_quoted_as_the_logs_give_it():
    study = _study()
    judge = study["turns"]["first"]["sets"]["judge"]
    base, static, random = (judge["against_baseline"], judge["against_static"],
                            judge["against_random"])
    sent = round(judge["shares"]["2cc4-mini"] * judge["questions"])
    kept = round(judge["shares"]["c3"] * judge["questions"])
    assert f"sends {sent} of the 100 judge questions to `2cc4-mini` and {kept} to `c3`" in PLAN
    assert f"it sends {sent} questions to `2cc4`" in PUBLISHED
    assert f"and {kept} to `c3`" in PUBLISHED
    singles = judge["single_actions"]
    assert (f"| Accuracy | {static['accuracy']['prescribed']:.1%} | "
            f"{singles['c3']['accuracy']:.1%} | {singles['designer']['accuracy']:.1%} |") in PLAN
    assert (f"| Cost a question | ${base['dollars']['prescribed']} | ${singles['c3']['dollars']} "
            f"| ${singles['designer']['dollars']} |") in PLAN

    accuracy = base["accuracy"]
    assert accuracy["p"] < 0.0001 and "(p < 0.0001)" in PLAN
    assert (f"more often on {len(accuracy['prescribed_better_on'])} questions and less often "
            f"on {len(accuracy['other_better_on'])}") in PLAN
    tokens, dollars = base["tokens"], base["dollars"]
    assert (f"uses {tokens['prescribed']:,.0f} tokens a question against "
            f"{tokens['other']:,.0f} ({tokens['change']:+.1%}, lower in "
            f"{tokens['prescribed_lower_in']} of {tokens['chunks']} chunks)") in PLAN
    assert f"Its cost is {-dollars['change']:.1%} lower" in PLAN
    assert (f"only {dollars['prescribed_lower_in']} of the {dollars['chunks']} chunks: "
            f"Wilcoxon p = {dollars['p']:.4f}") in PLAN
    for page in (PLAN, PUBLISHED):
        assert f"{tokens['change']:+.1%}" in page
        assert f"{-dollars['change']:.1%} lower" in page
        assert f"p = {dollars['p']:.4f}" in page

    better = static["accuracy"]["prescribed_better_on"]
    worse = static["accuracy"]["other_better_on"]
    assert (f"often on {len(better)} questions ({', '.join(better)}) and less often on "
            f"{len(worse)} ({', '.join(worse)}): p = {static['accuracy']['p']}") in PLAN
    low, high = static["accuracy"]["ci95_points"]
    assert f"runs from {low:.1f} to {high:+.1f} points" in PLAN
    assert f"It costs {static['dollars']['change']:.1%} more." in PLAN
    assert (f"scores {random['same_shares_accuracy']:.1%} on average, and "
            f"{random['p']:.2%} of 10,000 such draws") in PLAN
    assert (f"at ${base['dollars']['prescribed']} a question against "
            f"${singles['c3']['dollars']}") in PUBLISHED
    assert f"at {static['dollars']['change']:.1%} more a question" in PUBLISHED
    assert (f"score {random['same_shares_accuracy']:.1%} (p = {random['p']})") in PUBLISHED


def test_the_second_turn_is_quoted_as_the_logs_give_it():
    study = _study()
    turn = study["turns"]["second"]
    chosen = turn["chosen"]
    sent = round(chosen["shares"]["2cc4-mini"] * turn["questions"])
    assert turn["static"] == "c3" and turn["questions"] == 160
    assert (f"160 questions, a budget of ${turn['budget']} a question, `c3` again the static "
            f"prescription, and a front of {turn['front']}") in PLAN
    assert (f"sends {sent} of the 160 training questions to `2cc4-mini`, and the Predictor "
            f"expects {chosen['predicted']['accuracy']:.1%} of it at "
            f"${chosen['predicted']['dollars']}") in PLAN
    fresh = turn["sets"]["fresh"]["against_static"]["accuracy"]
    assert (f"scores {fresh['prescribed']:.1%} on fresh against {fresh['other']:.1%} (better on "
            f"{len(fresh['prescribed_better_on'])} questions, worse on "
            f"{len(fresh['other_better_on'])}, p = {fresh['p']})") in PUBLISHED
    first = study["turns"]["first"]["sets"]["fresh"]["against_baseline"]["dollars"]
    second = turn["sets"]["fresh"]["against_baseline"]["dollars"]
    assert (f"{-first['change']:.1%} lower in the first (p = {first['p']:.4f}), and "
            f"{second['change']:.1%} higher in the second") in PLAN


def test_what_is_reported_beside_the_result_is_what_the_logs_give():
    study = _study()
    first = study["turns"]["first"]["sets"]
    for name, said in (("judge", "answer every judge question at"),
                       ("fresh", "of the fresh questions at")):
        bound = first[name]["knowing_every_answer"]
        for page in (PLAN, PUBLISHED):
            assert said in page and f"${bound['dollars']}" in page
    assert first["judge"]["knowing_every_answer"]["accuracy"] == 1.0
    assert f"{first['fresh']['knowing_every_answer']['accuracy']:.1%} of the fresh" in PLAN
    for name in ("judge", "fresh"):
        assert (first[name]["knowing_every_answer"]["dollars"]
                < first[name]["single_actions"]["designer"]["dollars"])

    def unbeaten(name: str) -> tuple[int, float]:
        front = first[name]["front"]
        return (sum(not member["beaten_by_one_action"] for member in front),
                max(member["accuracy"] for member in front))

    count, best = unbeaten("judge")
    assert (f"{count} of its {len(first['judge']['front'])} prescriptors were not beaten by any "
            f"one network, and the dearer ones scored up to {best:.1%}") in PLAN
    count, best = unbeaten("fresh")
    assert (f"On fresh {count} of the {len(first['fresh']['front'])} were not beaten, and none "
            f"scored above {best:.1%}") in PLAN

    seeds = first["judge"]["across_seeds"]
    assert seeds["accuracy"][0] == seeds["accuracy"][2]
    assert (f"All {seeds['seeds']} seeds' chosen prescriptors score {seeds['accuracy'][0]:.1%} "
            f"on judge, at ${seeds['dollars'][0]} to ${seeds['dollars'][2]} a question") in PLAN
    assert (f"{seeds['same_prescriptions_as_the_first']} of the {seeds['seeds']} prescribe "
            f"exactly as the first does on judge") in PLAN
    for turn, name in (("first", "fresh"), ("second", "fresh")):
        across = study["turns"][turn]["sets"][name]["across_seeds"]
        assert across["same_prescriptions_as_the_first"] == across["seeds"]


def test_the_single_network_the_pages_call_cheapest_and_nearly_best_is():
    study = _study()
    for name in ("judge", "fresh"):
        singles = study["turns"]["first"]["sets"][name]["single_actions"]
        assert min(singles, key=lambda a: singles[a]["dollars"]) == "c3"
        best = max(single["accuracy"] for single in singles.values())
        assert best - singles["c3"]["accuracy"] <= 0.02 + 1e-9
    changes = [study["turns"][turn]["sets"][name]["against_static"]["dollars"]["change"]
               for turn, name, _ in ROWS]
    assert all(0.60 <= change <= 0.80 for change in changes)
    assert "costs 60% to 80% more" in PLAN


def test_the_plan_was_pushed_before_the_result_and_says_which_commit():
    assert "Pushed as `ec1d00a` at 12:36 UTC on 10 November 2026" in PLAN

