"""The ablation's missing baseline: a rule that learns nothing.

`make ablation` reports the Predictor picking the best of three unseen
networks 62% of the time against 33% for chance. Chance is not the only
baseline that matters. On these twelve networks the larger teams are also the
better ones, so "pick the network with the most agents" does as well, with no
training at all. The documents say so; this pins the figure they quote.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import selection_ablation

from esp.eval import measurements


def test_picking_the_most_agents_matches_the_gated_predictor():
    records = measurements.load()
    assert len(records) == 12
    picks = [selection_ablation.most_agents([records[i] for i in held])
             for held in itertools.combinations(range(len(records)), 3)]
    assert len(picks) == 220
    assert round(float(np.mean([hit for hit, _ in picks])), 3) == 0.626
    assert round(float(np.mean([regret for _, regret in picks])), 4) == 0.0106
