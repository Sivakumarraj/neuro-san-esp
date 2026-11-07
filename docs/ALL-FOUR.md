# Better on all four at once: the third test

This page was written and pushed before any of the networks below was run. It fixes which
networks are tried, how one is chosen on the select questions, and the five things the
chosen one must then show on questions it was not chosen on. Whatever happens, the README
reports it.

## Why these networks

Neither earlier test found a network better than the designer's shape on accuracy, tokens
and cost at once ([SAME-MODEL.md](SAME-MODEL.md), [MODEL-ROUTER.md](MODEL-ROUTER.md)).
`2cc4` on `gpt-5.4-nano` is 14.6% cheaper and two points less accurate; with its Researcher
on `gpt-5.4-mini` it is eleven points more accurate and 75% dearer.

Two changes could close that gap:

- **A calculator.** On judge, 13 of the all-nano `2cc4`'s 36 wrong runs are totals that no
  subset of the true records adds up to: the facts were found and the sum slipped. Another
  4 are penalty sums. `esp/eval/calculator_tool.py` is a CodedTool, not an agent.
- **A cheaper strong Researcher.** `gpt-4o-mini` costs $0.15 and $0.60 a million input and
  output tokens against `gpt-5.4-nano`'s $0.20 and $1.25, and is the only cheaper OpenAI
  model neuro-san 0.7.5 prices correctly. Its table has `gpt-5-mini`'s output at $20 a
  million and no price for `gpt-5-nano`, so neither can be costed by neuro-san's own count.

**What has been seen.** The calculator idea comes from the wrong answers on judge and fresh
(the autopsy in [FINDINGS.md](FINDINGS.md) and the count above). Those question sets are
therefore blind to the networks, none of which has ever been run, but not to the idea. The
choice among the networks is made on select alone.

## The candidates

| | Network | Genome |
| --- | --- | --- |
| C1 | `2cc4`, every agent `gpt-5.4-nano`, the Calculator on the Coordinator | `ae41fbfaa513ed47` |
| C2 | `2cc4`, the Researcher on `gpt-4o-mini` | `e212a77c53bfa809` |
| C3 | `2cc4`, the Researcher on `gpt-4o-mini`, the Calculator on the Coordinator | `f4e9e494442f9888` |
| Baseline | the designer's shape, every agent `gpt-5.4-nano` | `0378ca26cd9fe977` |

Every one runs at 10 documents a search, 40 steps and 600 seconds, with models pinned.

## Step 1: choosing on select

The baseline and C1 to C3 each answer the 60 select questions once, at the same hour.

- A candidate is **eligible** when, on select, it is more accurate than the baseline and uses
  fewer tokens and fewer dollars a question.
- The **winner** is the most accurate eligible candidate; a tie goes to the cheaper, then to
  the one using fewer tokens.
- **If none is eligible, the test stops** and nothing more is spent.

```bash
export ESP_PROVIDER=openai ESP_DEFAULT_MODEL=gpt-5.4-nano ESP_MODEL_TIERS=gpt-5.4-nano,gpt-5.4-mini
export ESP_SEARCH_RESULTS=10 ESP_PIN_MODELS=1
OUT=results/all-four/select-2026-11
python scripts/select_and_judge.py measure designer --suite select --out $OUT --go &
python scripts/select_and_judge.py measure 2cc4 --label c1 --calculator Coordinator \
    --suite select --out $OUT --new-network --go &
python scripts/select_and_judge.py measure 2cc4 --label c2 --model Researcher=gpt-4o-mini \
    --suite select --out $OUT --new-network --go &
python scripts/select_and_judge.py measure 2cc4 --label c3 --model Researcher=gpt-4o-mini \
    --calculator Coordinator --suite select --out $OUT --new-network --go
```

## Step 2: judging the winner

The winner and the baseline, at the same hour, on J001 to J100 twice (the test) and F01 to
F40 twice (the replication), logged in `results/all-four/judge-2026-11/` and
`results/all-four/fresh-2026-11/`. On judge, all five must hold:

1. **Accuracy.** Right more often on more questions than the baseline, two-sided sign test
   p < 0.05; a run that did not finish counts as wrong.
2. **Tokens a question.** Lower, two-sided Wilcoxon signed-rank p < 0.05 over the 25
   four-question chunks, each chunk's two repeats averaged.
3. **Money a question.** neuro-san's own reported cost, lower, the same test.
4. **Agents.** Fewer. The Calculator is a CodedTool, as CorpusSearch is, not an agent.
5. **Fitness.** Higher on the token fitness and on the dollar fitness.

If any of 1 to 3 fails, the README says which and does not claim the network is better on
that axis.

Each network runs under a $0.40 cap a question set. The whole test costs about $1.10.
