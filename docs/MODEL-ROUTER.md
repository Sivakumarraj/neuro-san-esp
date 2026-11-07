# Choosing the Researcher's model per question

The second test, after [the same-model test](SAME-MODEL.md). This page was written and
pushed before any routing was computed on the judge or fresh questions. It records one
result that was computed first, on the select questions alone, because the procedure
needs it: the threshold.

## The claim

`2cc4` is one topology, a Coordinator and a Researcher. Run with every agent on
`gpt-5.4-nano` it is 14.6% cheaper than the designer's shape at about its accuracy; run
with the Researcher on `gpt-5.4-mini` it is far more accurate at 94% more a question. A
router that sends only the questions that need it to the stronger Researcher could be
more accurate than the designer's shape and still cheaper. The claim is that it is, on
the same five pass marks as the same-model test.

## The procedure

| | |
| --- | --- |
| Cheap action | `2cc4` `2cc4231b7d3fb81d`, every agent `gpt-5.4-nano` |
| Strong action | `2cc4` `47322479eb9cba1e`, the Researcher on `gpt-5.4-mini` |
| Baseline | the designer's shape `0378ca26cd9fe977`, every agent `gpt-5.4-nano` |
| Predictor | `esp.route.Router`, unchanged: a logistic regression per action from the question's text |
| Fitted on | the 60 select questions, stage 7 (`results/headtohead/paid-2026-10/7-select-search10/`) |
| Threshold | from leave-one-out gains on select: the most accurate whose select cost a question is at or below the baseline's; a tie goes to sending fewer up |
| Judged on | J001 to J100 and F01 to F40, every action measured on 7 November 2026 |

The strong action had been judged only in October, so it was measured again on judge, at
the same settings as the other two, starting at 05:22 UTC on 7 November 2026
(`results/router/judge-2026-11/`). October's run is kept as a second check.

Cost is neuro-san's own count. The provider reports usage per four-question chunk, so a
routed question's tokens and dollars are its action's average for that chunk.

## What the select questions already decide

On select the cheap action costs $0.001263 a question, the strong action $0.00279 and the
baseline $0.001367. The budget therefore allows at most 4 of the 60 questions up. The
strong action beat the cheap one on 6 select questions and lost on 1, and the Predictor
does not find them: its single highest-ranked question is one the strong action got wrong,
and its next group is 14 questions. No threshold within the budget scores better on select
than sending nothing up, so the procedure sends nothing up.

The router is therefore the cheap action on every question, and it can pass only where the
cheap action passes. On judge the same-model test has already shown that it does not: it is
cheaper, and not more accurate. This test is recorded rather than abandoned because it was
announced, and because the strong action's same-day judge run answers a second question:
whether a router that knew the answers could pass within the budget, which says whether a
better Predictor would be worth building.

## What has to hold

The same five as [SAME-MODEL.md](SAME-MODEL.md), for the router against the baseline:
accuracy higher (sign test p < 0.05), tokens and dollars a question lower (Wilcoxon over
the chunks, p < 0.05), fewer agents, and a higher token and dollar fitness.

## Not a test

The routing that knows every answer, sending up the questions the strong action wins most
while the cost a question stays at or below the baseline's, is reported beside the result.
It is a bound on any Predictor for these two actions, computed on the questions it is
reported on, and is never quoted as a result.
