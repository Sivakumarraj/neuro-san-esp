# The same-model test, fixed before it was run

This page was written and pushed before any judge run of the pair below, so the claim it
tests cannot have been fitted to the result. The commit that adds it is the timestamp.

## The claim

On questions it was never chosen on, with **every agent of both networks on the same
model**, `2cc4` (two agents) beats the designer's shape (four agents) on every axis at once:
more accurate, fewer tokens a question, less money a question, fewer agents, and a higher
fitness.

## Why this test

The headline network promotes `2cc4`'s Researcher to `gpt-5.4-mini`. That buys accuracy and
costs 94% more a question, so it is not a structure result. Holding the model equal is the
only way to credit the gain to the network itself.

`2cc4` on `gpt-5.4-nano` has been judged only at 3 documents a search (stage 5: 76.5%
against 69.0%, 17% fewer tokens, 17% less a question, sign test p = 0.0525) and, at 10
documents, only on the select questions it was chosen on (stage 7: 52/60 against 48/60).

## The run

| | |
| --- | --- |
| Networks | designer's shape `0378ca26cd9fe977`; `2cc4` `2cc4231b7d3fb81d`; every agent `gpt-5.4-nano` |
| Questions | J001 to J100 (`meridian-judge`), never used to choose `2cc4`, each asked twice |
| Settings | `ESP_SEARCH_RESULTS=10`, `ESP_PIN_MODELS=1`, 40 steps, 600 seconds |
| Order | both networks at the same time, so they meet the same hour of the provider |
| Caps | $0.40 a network |
| Logs | `results/structure/judge-nano-2026-11/` |

```bash
export ESP_PROVIDER=openai ESP_DEFAULT_MODEL=gpt-5.4-nano ESP_MODEL_TIERS=gpt-5.4-nano,gpt-5.4-mini
export ESP_SEARCH_RESULTS=10 ESP_PIN_MODELS=1
OUT=results/structure/judge-nano-2026-11
python scripts/select_and_judge.py measure designer --suite judge --repeats 2 --out $OUT --budget 0.40 --go &
python scripts/select_and_judge.py measure 2cc4 --label 2cc4-nano --suite judge --repeats 2 --out $OUT \
    --budget 0.40 --go
```

## What has to hold

All five, or the README does not claim dominance on the axis that failed and reports what
was measured instead.

1. **Accuracy.** `2cc4` is right more often than the designer's shape on more questions than
   the reverse, two-sided sign test p < 0.05. A run that did not finish counts as wrong.
2. **Tokens a question.** Lower for `2cc4`, paired over the 25 four-question chunks (each
   chunk's two repeats averaged), two-sided Wilcoxon signed-rank p < 0.05.
3. **Money a question.** neuro-san's own reported cost, lower for `2cc4`, the same test.
4. **Agents.** 2 against 4.
5. **Fitness.** Higher on both formulas the repository uses: the token fitness
   (`esp/evolve/loop.py`) and the dollar fitness (`esp/eval/pricing.py`).

## Also run, and not the test

The 40 fresh questions F01 to F40, at the same settings, were measured on 7 November 2026
for both networks and for the headline network. That run was already under way, and partly
seen, when this page was written, so it is a replication rather than the test.
