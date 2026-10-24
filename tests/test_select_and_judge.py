"""The paid head-to-head runs: their committed logs, the summary built from them,
and the figures the README quotes from that summary.

The paid runs cannot be repeated in CI, but everything said about them can be
recomputed from the logs in results/headtohead/paid-2026-10/. These tests do
that, so a number in the README cannot drift from the runs it describes.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "headtohead" / "paid-2026-10"
# The README carries the headline and docs/RESULTS.md every stage behind it; a figure
# quoted in either has to be the one the logs give.
README = "\n".join((ROOT / name).read_text(encoding="utf-8")
                   for name in ("README.md", "docs/RESULTS.md"))
FINDINGS = (ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")

sys.path.insert(0, str(ROOT / "scripts"))
import select_and_judge as sj  # noqa: E402


def _summary() -> dict:
    report = {}
    for stage in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        rows = sj.load_rows(stage)
        labels = sorted({r["label"] for r in rows})
        report[stage.name] = {"networks": sj.summarise(rows)}
        if len(labels) == 2:
            report[stage.name]["paired"] = sj.paired(rows, *labels)
    return report


def test_the_committed_summary_is_what_the_logs_say():
    committed = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    assert _summary() == committed


def test_the_readme_quotes_the_runs_as_measured():
    stages = _summary()

    def line(stage: str, label: str) -> str:
        n = stages[stage]["networks"][label]
        return f"{n['correct']}/{n['question_runs']}"

    for stage, label in (("5-judge-2cc4", "evolved"), ("5-judge-2cc4", "designer"),
                         ("6-judge-plus-P001-P100", "evolved"),
                         ("6-judge-plus-P001-P100", "designer"),
                         ("8-judge-search10", "evolved"), ("8-judge-search10", "designer")):
        quoted = line(stage, label)
        assert quoted in README, f"README is missing {stage} {label} {quoted}"
    p = stages["8-judge-search10"]["paired"]["sign_test_p"]
    assert f"p = {p}" in README


def _plain(text: str) -> str:
    """Prose as a reader sees it: no bold markers, one space, an ASCII minus."""
    return " ".join(text.replace("**", "").replace("\N{MINUS SIGN}", "-").split())


def _pct(accuracy: float) -> str:
    return f"{100 * accuracy:.1f}%"


def _change(a: float, b: float) -> int:
    return round(100 * (a / b - 1))


def _repeats(stage: str, label: str) -> list[tuple[int, int]]:
    """(right, asked) for each repeat of one network, in order."""
    counts: dict = defaultdict(lambda: [0, 0])
    for r in sj.load_rows(RUNS / stage):
        if r["label"] == label:
            key = (r["run"], r.get("repeat", 0))
            counts[key][0] += sum(x["correct"] for x in r["results"])
            counts[key][1] += len(r["results"])
    return [tuple(counts[key]) for key in sorted(counts)]


def _totals(first: int, last: int) -> dict:
    rows = [r for stage in sorted(p for p in RUNS.iterdir() if p.is_dir())
            if first <= int(stage.name.split("-")[0]) <= last for r in sj.load_rows(stage)]
    return {"runs": sum(len(r["results"]) for r in rows),
            "reported": sum(r["reported_cost"] for r in rows),
            "cap": sum(r["cost"] for r in rows)}


def _p(report: dict) -> str:
    """A stage's two-sided sign test as the documents print it: three decimals,
    one significant figure below 0.001."""
    import math

    a, b = (len(found) for found in report["paired"]["better_on"].values())
    n, k = a + b, min(a, b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1g}"


def _quoted() -> list[tuple[str, str]]:
    """Every paid-run figure the README and FINDINGS state, as each should read,
    computed from the logs. A document that drifts from its runs fails here."""
    from esp.eval.pricing import blended

    s = _summary()
    net = {stage: report["networks"] for stage, report in s.items()}
    p = {stage: _p(report) for stage, report in s.items() if "paired" in report}
    times = "\N{MULTIPLICATION SIGN}"
    e8, d8 = net["8-judge-search10"]["evolved"], net["8-judge-search10"]["designer"]
    e5, d5 = net["5-judge-2cc4"]["evolved"], net["5-judge-2cc4"]["designer"]
    e6, d6 = net["6-judge-plus-P001-P100"]["evolved"], net["6-judge-plus-P001-P100"]["designer"]
    f3, d3 = net["3-judge-flat"]["flat"], net["3-judge-flat"]["designer"]
    c1, d1 = net["1-gemini-champion"]["champion"], net["1-gemini-champion"]["designer"]
    s2, s7 = net["2-select-gemini-networks"], net["7-select-search10"]
    better = s["8-judge-search10"]["paired"]["better_on"]
    (e8a, _), (e8b, _) = _repeats("8-judge-search10", "evolved")
    (d8a, _), (d8b, _) = _repeats("8-judge-search10", "designer")
    early, late = _totals(0, 5), _totals(6, 8)

    def change(a: dict, b: dict, key: str) -> int:
        # Nested only to keep each figure on one readable line below.
        return _change(a[key], b[key])

    tok8, tok5, tok6 = (change(e8, d8, "tokens_per_question"),
                        change(e5, d5, "tokens_per_question"),
                        change(e6, d6, "tokens_per_question"))
    assert abs(change(f3, d3, "tokens_per_question")) < 2           # "equal tokens"

    designer_runs = [right for stage in ("1-gemini-champion", "3-judge-flat", "5-judge-2cc4",
                                         "6-judge-plus-P001-P100")
                     for right, asked in _repeats(stage, "designer") if asked == 100]
    champion, designer = _repeats("1-gemini-champion", "champion"), _repeats(
        "1-gemini-champion", "designer")
    assert champion[0] == designer[0] == (68, 100)
    second = champion[1][1] + designer[1][1]
    ratio = c1["reported_cost_per_question"] / d1["reported_cost_per_question"]

    children = net["4-select-children-of-flat"]
    short = sorted((n["question_runs"] for n in children.values() if n["question_runs"] < 60),
                   reverse=True)
    best = max(n["correct"] for n in children.values())
    assert all(n["correct"] + 60 - n["question_runs"] < best
               for n in children.values() if n["question_runs"] < 60)
    words = {4: "four"}
    found = children["m_2cc4231b7d3fb81d"]

    misses = [x for r in sj.load_rows(RUNS / "7-select-search10") for x in r["results"]
              if not x["correct"]]
    zeros = [x for x in misses if x["answer"].strip() == "0"
             or "cannot be determined" in x["answer"].lower()]
    on_sums = [x for x in zeros if x["task_id"] in {"S43", "S46", "S51"}]
    # The two misses quoted as examples, both 2cc4's.
    answer = {x["task_id"]: x["answer"] for r in sj.load_rows(RUNS / "7-select-search10")
              if r["label"] == "2cc4" for x in r["results"] if not x["correct"]}
    expected = {t.task_id: t.answer for t in sj.suite_tasks("select")}
    words.update({5: "five", 6: "Six"})

    # Each logged row names every agent of the network it measured.
    two, four = (len({agent for r in sj.load_rows(RUNS / "8-judge-search10")
                      if r["label"] == label for agent in r["models"]})
                 for label in ("evolved", "designer"))
    words.update({len(s2) - 1: "Six"} if len(s2) == 7 else {})

    cost_q = ("reported_cost_per_question", "cost_per_question")
    cost_c = ("reported_cost_per_correct", "cost_per_correct")
    return [
        ("README", f"scored {_pct(e8['accuracy'])} against {_pct(d8['accuracy'])}"),
        ("README", f"({e8['correct']}/{e8['question_runs']}; {e8a}, then {e8b})"),
        ("README", f"({d8['correct']}/{d8['question_runs']}; {d8a}, then {d8b})"),
        ("README", f"{e8['tokens_per_question']:,} ({tok8:+d}%) | {d8['tokens_per_question']:,}"),
        ("README", f"${e8[cost_q[0]]:.5f} ({change(e8, d8, cost_q[0]):+d}%) | "
                   f"${d8[cost_q[0]]:.5f}"),
        ("README", f"${e8[cost_c[0]]:.5f} ({change(e8, d8, cost_c[0]):+d}%) | "
                   f"${d8[cost_c[0]]:.5f}"),
        ("README", f"{change(e8, d8, cost_q[1]):+d}% a question and "
                   f"{change(e8, d8, cost_c[1]):+d}% a correct answer"),
        ("README", f"| Agents | {two} | {four} |"),
        ("README", f"| 2, 3 | {words[len(s2) - 1]} committed networks and the designer on select"),
        ("README", f"{len(better['evolved'])} questions against {len(better['designer'])}: "
                   f"sign test p = {p['8-judge-search10']}"),
        ("README", f"scored {s7['2cc4-mini']['correct']}/60 there"),
        ("README", f"{blended('gpt-5.4-mini') / blended('gpt-5.4-nano'):.1f}{times} dearer"),
        ("README", f"rose from {100 * d5['accuracy']:.0f}% to {100 * d8['accuracy']:.0f}%"),
        ("README", f"scored {e5['correct']}/200 ({_pct(e5['accuracy'])}) against the designer's "
                   f"{d5['correct']}/200 ({_pct(d5['accuracy'])}) with {-tok5}% fewer tokens "
                   f"(p = {p['5-judge-2cc4']})"),
        ("README", f"scored {e6['correct']}/100 against {d6['correct']}/100 "
                   f"(p = {p['6-judge-plus-P001-P100']}), but used {tok6}% more tokens"),
        ("README", f"68 / 100 each (the second repeat stopped after {second} question-runs); "
                   f"the champion cost {ratio:.1f}{times} per question"),
        ("README", f"{_pct(f3['accuracy'])} against {_pct(d3['accuracy'])} "
                   f"(p = {p['3-judge-flat']}), equal tokens"),
        ("README", f"{_pct(e5['accuracy'])} against {_pct(d5['accuracy'])} "
                   f"(p = {p['5-judge-2cc4']}), {-tok5}% fewer tokens"),
        ("README", f"{e6['correct']} against {d6['correct']} "
                   f"(p = {p['6-judge-plus-P001-P100']}), {tok6}% more tokens"),
        ("README", f"{s7['2cc4-mini']['correct']}, {s7['flat']['correct']}, "
                   f"{s7['2cc4']['correct']} and {s7['designer']['correct']} of 60"),
        ("README", f"{_pct(e8['accuracy'])} against {_pct(d8['accuracy'])} "
                   f"(p = {p['8-judge-search10']})"),
        ("README", f"8 children, {words[len(short)]} stopped short of 60"),
        ("README", f"scored {min(designer_runs)} to {max(designer_runs)} of 100 in every run "
                   "at 3 documents"),
        ("README", f"stages 0 to 5 cost ${early['reported']:.2f} for {early['runs']:,} "
                   f"question-runs and stages 6 to 8 ${late['reported']:.2f} more "
                   f"(${early['cap']:.2f} and ${late['cap']:.2f} on the cap's basis)"),
        ("FINDINGS", f"68 / 100 each in the first repeat (the second stopped after {second} "
                     f"question-runs); champion {ratio:.1f}{times} the cost per question"),
        ("FINDINGS", f"`flat` {s2['flat']['correct']} / 60, designer "
                     f"{s2['designer']['correct']} / 60"),
        ("FINDINGS", f"{f3['correct']} / 200 against {d3['correct']} / 200, "
                     f"p = {p['3-judge-flat']}, equal tokens"),
        ("FINDINGS", f"`2cc4` {found['correct']} / 60 at {found['tokens_per_question']:,} tokens "
                     f"a question; {words[len(short)]} stopped short of 60 (at "
                     f"{', '.join(map(str, short[:-1]))} and {short[-1]} questions) and none "
                     f"could have reached {best}"),
        ("FINDINGS", f"{e5['correct']} / 200 against {d5['correct']} / 200, "
                     f"p = {p['5-judge-2cc4']}, {-tok5}% fewer tokens"),
        ("FINDINGS", f"{e6['correct']} / 100 against {d6['correct']} / 100, "
                     f"p = {p['6-judge-plus-P001-P100']}, {tok6}% more tokens"),
        ("FINDINGS", f"`gpt-5.4-mini` {s7['2cc4-mini']['correct']}, `flat` "
                     f"{s7['flat']['correct']}, `2cc4` {s7['2cc4']['correct']}, designer "
                     f"{s7['designer']['correct']}, of 60"),
        ("FINDINGS", f"{e8['correct']} / 200 against {d8['correct']} / 200, "
                     f"p = {p['8-judge-search10']}, {-tok8}% fewer tokens, "
                     f"{change(e8, d8, cost_c[0])}% more per correct answer"),
        ("FINDINGS", f"{change(e8, d8, cost_c[0])}% more per correct answer by neuro-san's own "
                     f"count ({change(e8, d8, cost_c[1])}% on the cap's more cautious basis)"),
        ("FINDINGS", f"scored {found['correct']} and {children['m_f8134cc80a0a8689']['correct']} "
                     "of 60: their instructions differ"),
        ("FINDINGS", f"raised the designer's shape from {100 * d5['accuracy']:.0f}% to "
                     f"{100 * d8['accuracy']:.0f}%"),
        ("FINDINGS", f"took `2cc4` from {s7['2cc4']['correct']} to {s7['2cc4-mini']['correct']} "
                     f"of 60 on select and to {_pct(e8['accuracy'])} on judge"),
        ("FINDINGS", f"{answer['S05']} for {expected['S05']}, {answer['S18']} for "
                     f"{int(expected['S18']):,}"),
        ("FINDINGS", f"{words[len(zeros)]} of the {len(misses)} were refusals or zeros: "
                     f"{words[len(on_sums)]} on the depot-wide penalty sums"),
    ]


def test_every_paid_figure_in_the_docs_is_what_the_logs_say():
    """The README and FINDINGS say this file recomputes every figure they quote
    about the paid runs. It used to check six fractions and one p-value, and a
    README with nineteen of its figures changed still passed."""
    documents = {"README": _plain(README), "FINDINGS": _plain(FINDINGS)}
    missing = [(where, figure) for where, figure in _quoted()
               if _plain(figure) not in documents[where]]
    assert not missing, f"documents disagree with the logs: {missing}"


def test_flat_and_solo_are_called_seeds_not_search_results():
    from esp.eval import measurements
    from esp.genome.seeds import SEEDS

    committed = {r.genome_hash: r.genome for r in measurements.load()}
    for genome_hash, seed_name in sj.SEED_HASHES.items():
        assert genome_hash in committed
        assert set(committed[genome_hash].agents) == set(SEEDS[seed_name]().agents)
        assert "hand-written seed" in sj.origin_of(genome_hash)
    assert "Gemini search" in sj.origin_of(sj.COMMITTED["mergeA"])


def test_the_suites_are_the_ones_named():
    assert [len(sj.suite_tasks(s)) for s in sj.SUITES] == [60, 100, 100, 40]
    assert sj.suite_tasks("judge-plus")[0].task_id == "P001"


def test_search_returns_three_documents_unless_told_otherwise(monkeypatch):
    from esp.eval import corpus_tool

    try:
        monkeypatch.delenv("ESP_SEARCH_RESULTS", raising=False)
        assert importlib.reload(corpus_tool).MAX_RESULTS == 3
        monkeypatch.setenv("ESP_SEARCH_RESULTS", "10")
        reloaded = importlib.reload(corpus_tool)
        assert reloaded.MAX_RESULTS == 10
        # "mis-picked" is in eight documents; at three a count over them comes up short.
        assert len(reloaded.search("mis-picked pallet")) > 3
    finally:
        monkeypatch.delenv("ESP_SEARCH_RESULTS", raising=False)
        importlib.reload(corpus_tool)


def test_the_search_width_is_read_when_the_search_runs(monkeypatch):
    """The cache key and the search must agree on the width. The search used to
    take its default when it was defined, so raising MAX_RESULTS in process
    changed the key to 10 while every search still returned 3."""
    from esp.eval import corpus_tool
    from esp.eval.runner import run_settings

    monkeypatch.setattr(corpus_tool, "MAX_RESULTS", 10)
    assert len(corpus_tool.search("mis-picked pallet")) > 3
    assert run_settings()["search_results"] == 10


def test_model_calls_are_reported_only_when_every_chunk_counted_them():
    """The committed logs predate the count, so their summary must not claim
    zero calls; a run that counted them reports calls per question."""
    row = {"label": "a", "genome_hash": "h", "tokens": 10, "cost": 0.1,
           "results": [{"task_id": "J001", "correct": True, "infrastructure": False}] * 2}
    assert "requests_per_question" not in sj.summarise([row])["a"]
    counted = sj.summarise([{**row, "requests": 9}, {**row, "requests": 7}])["a"]
    assert counted["requests_per_question"] == 4.0


def test_one_label_never_pools_two_networks():
    """Measuring 2cc4, then 2cc4 with a promoted Researcher, under one label into
    one --out reported 120 question-runs for a 60-question set."""
    import pytest

    rows = [{"label": "evolved", "genome_hash": "2cc4231b7d3fb81d", "results": [],
             "tokens": 0, "cost": 0.0},
            {"label": "evolved", "genome_hash": "47322479eb9cba1e", "results": [],
             "tokens": 0, "cost": 0.0}]
    with pytest.raises(ValueError, match="several networks"):
        sj.summarise(rows)
    with pytest.raises(ValueError, match="several networks"):
        sj.paired(rows, "evolved", "designer")

