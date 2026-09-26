"""Dollar cost, next to tokens. The case for it is in the committed data: the
best-measured network used fewer tokens than the designer's shape and cost
much more, because its router runs a pricier model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from esp.eval.measurements import FIXTURE_CACHE
from esp.eval.pricing import DOLLAR_SCALE, blended, estimate, fitness_dollars, priced

ROOT = Path(__file__).resolve().parent.parent
BEST, DESIGNER, CHAMPION, SPLIT = "3bf9c008", "459ac1a6", "6859dda0", "deca12e8"


def by_prefix():
    return {p.genome_hash[:8]: p for p in priced(FIXTURE_CACHE)}


def test_every_committed_measurement_recorded_its_cost():
    assert len(priced(FIXTURE_CACHE)) == 12


def test_the_best_network_used_fewer_tokens_and_cost_more_dollars():
    table = by_prefix()
    best, designer = table[BEST], table[DESIGNER]
    assert round(best.tokens / designer.tokens - 1, 2) == -0.07
    assert round(best.dollars / designer.dollars - 1, 2) == 0.63


def test_the_champion_is_not_the_cheapest_network_at_its_accuracy():
    table = by_prefix()
    assert table[CHAMPION].accuracy == table[SPLIT].accuracy
    assert table[SPLIT].dollars < table[CHAMPION].dollars
    assert table[CHAMPION].tokens < table[SPLIT].tokens


def test_planning_prices():
    assert blended("claude-haiku-4-5") == pytest.approx(1.40)
    assert blended("claude-sonnet-5") == pytest.approx(2.80)
    assert blended("gemini-3.1-flash-lite") == pytest.approx(0.45)
    assert estimate(1e6, "claude-opus-5-5") == pytest.approx(5.60)
    with pytest.raises(KeyError, match="no price"):
        blended("some-model-nobody-priced")


def test_the_dollar_fitness_charges_money_and_caps_it():
    cheap = fitness_dollars(0.9, 0.001, 4)
    dear = fitness_dollars(0.9, 0.05, 4)
    assert cheap > dear
    assert fitness_dollars(0.9, DOLLAR_SCALE, 4) == fitness_dollars(0.9, 10.0, 4)
    assert fitness_dollars(0.95, 0.05, 4) > fitness_dollars(0.9, 0.001, 4)


def test_the_report_runs_and_spends_nothing(capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cost_report
    assert cost_report.main() == 0
    out = capsys.readouterr().out
    assert "+63%" in out and "mut:split_agent" in out
