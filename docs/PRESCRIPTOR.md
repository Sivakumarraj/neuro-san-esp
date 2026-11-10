# A learned Prescriptor, question by question: the fourth test

This page and the code it describes were committed and pushed before any prescriptor was
evaluated on the judge or fresh questions. It records what was computed first, on the select
questions alone, because the procedure needs it. Whatever happens, the README reports it.

## Why this test

ESP, as its paper describes it, has two learned models. The Predictor maps a context and an
action to their outcomes and is trained on measurements. The Prescriptor maps a context to an
action, and its weights are evolved against the Predictor, so nothing is paid for until the
best prescriptors are tried for real.

Until now this repository had only the first. Its search prescribes with mutation operators
and selection, and its routers ([MODEL-ROUTER.md](MODEL-ROUTER.md), `make route`) prescribe with
a threshold a person chose a rule for. With two actions that is all there is to decide. With
more than two it is not, and since the [third test](ALL-FOUR.md) four networks have been
measured on all three question sets. This test adds the Prescriptor and asks whether it earns
its place.

## The claim

A Prescriptor evolved against the Predictor, choosing which network answers each question from
the question's text, is better than prescribing one network for every question.

## The procedure

| | |
| --- | --- |
| Context | a question's text features, `esp.route.request_features`, unchanged |
| Actions | which of the four networks below answers the question |
| Outcomes | whether the answer is right, and neuro-san's own count of what it cost |
| Predictor | per action, `esp.route`'s logistic regression from the features to the chance of a right answer, and the action's average cost a question |
| Prescriptor | a neural network, 5 features to 8 tanh units to 4 outputs, 84 weights; the action with the highest output is prescribed (`esp/prescriptor.py`) |
| Evolution | NSGA-II (`esp/nsga2.py`) on predicted accuracy, up, and predicted dollars a question, down; 100 prescriptors for 100 generations, seed 20261110 |
| Scored against | each training question's leave-one-out chance, so no question's own outcome tells the Prescriptor where to send it |
| Choice | from the front, the most accurate by the Predictor whose predicted cost a question is at or below the designer's shape's on the training questions; a tie goes to the cheaper |
| Against | the **static prescription**: the single network, measured most accurate on the training questions, within the same budget |

The first generation holds, beside random networks, the four prescriptors that prescribe one
action everywhere. The front can therefore never be worse on the Predictor than the best
single network, and the Prescriptor has to earn its place over prescribing nothing.

### The actions

Every network measured on select, judge and fresh at 10 documents a search:

| Action | Network | Genome | Agents |
| --- | --- | --- | --- |
| `designer` | the designer's shape, every agent on `gpt-5.4-nano`; also the baseline | `0378ca26cd9fe977` | 4 |
| `2cc4-nano` | `2cc4`, every agent on `gpt-5.4-nano` | `2cc4231b7d3fb81d` | 2 |
| `2cc4-mini` | `2cc4` with its Researcher on `gpt-5.4-mini` | `47322479eb9cba1e` | 2 |
| `c3` | `2cc4` with its Researcher on `gpt-4o-mini` and a calculator on its Coordinator | `f4e9e494442f9888` | 2 |

### The logs

| Set | `designer`, `2cc4-nano` | `2cc4-mini` | `c3` |
| --- | --- | --- | --- |
| Select, 60 questions once | `results/headtohead/paid-2026-10/7-select-search10/` | the same folder | `results/all-four/select-2026-11/` |
| Judge, 100 questions twice | `results/structure/judge-nano-2026-11/` | `results/router/judge-2026-11/` | `results/all-four/judge-2026-11/` |
| Fresh, 40 questions twice | `results/structure/fresh-nano-2026-11/` | the same folder | `results/all-four/fresh-2026-11/` |

Nothing new is measured. The provider reports usage per four-question chunk, so a prescribed
question's tokens and dollars are its action's average for that chunk. A run that did not
finish counts as wrong.

### Two turns

ESP is a loop: what its prescriptions lead to becomes the Predictor's training data.

- **First turn.** Trained on the 60 select questions. Judged on judge, and reported on fresh.
- **Second turn.** The judge outcomes join the training data, 160 questions, and the Predictor,
  the evolution and the choice are run again with nothing else changed. Judged on fresh.

## What has been seen

Each action's accuracy and cost on judge and on fresh, as one network for every question, is
already published in this repository. So it is known before this test that on judge `c3` is
the most accurate network within the budget, and that `2cc4-mini` is two points above it at
more than three times its cost. The room for a prescriptor is therefore known to be small:
sending a quarter of the judge questions to `2cc4-mini` at random would be expected to gain
half a point.

What has not been computed is any prescription that mixes the actions on judge or fresh: which
question goes where, and what that scores. The Predictor, the Prescriptor and the choice are
fixed on select below, and for the second turn on select and judge.

The four judge logs were all measured on 7 November 2026, not at the same hour. The designer's
shape scored 84.0% in one judge run that day and 80.3% in another, so a difference of a few
points between two logs is within what one network does from one run to the next.

## What the select questions already decide

| Action | Right on select | Leave-one-out chance, averaged | Cost a question |
| --- | --- | --- | --- |
| `designer` | 80.0% | 80.1% | $0.001367 |
| `2cc4-nano` | 86.7% | 86.6% | $0.001263 |
| `2cc4-mini` | 95.0% | 95.0% | $0.00279 |
| `c3` | 93.3% | 93.3% | $0.000854 |

- **The budget** is the designer's shape's $0.001367 a question.
- **The static prescription is `c3`**: the most accurate of the three networks within the
  budget. Beside it, the budget allows 15 of the 60 questions on `2cc4-mini`.
- **The Predictor expects little.** Its leave-one-out gain from `2cc4-mini` over `c3` is positive
  on 40 of the 60 questions, between −4.6 and +12.5 points, and +0.7 at the median.
- **The front** has 36 prescriptors, from `c3` everywhere at $0.000854 a question to one
  sending 68% of the questions to `2cc4-mini` at $0.002177.
- **The chosen prescriptor** sends 15 of the 60 questions to `2cc4-mini` and 45 to `c3`, and
  nothing to the other two. The Predictor expects 94.8% of it at $0.001338 a question, against
  93.3% of `c3` alone.
- **The seed does not matter to the choice.** All 20 seeds, 20261110 to 20261129, choose the
  same 15 questions.

On select itself the chosen prescriptor is right on 58 of 60 against `c3`'s 56. Select is what
the Predictor was trained on, so that proves nothing, and it is not quoted as a result.

## What has to hold

On judge, for the first turn's chosen prescriptor, all of:

1. **The five marks every test here uses, against the designer's shape**: accuracy higher
   (right more often on more questions, two-sided sign test p < 0.05); tokens and dollars a
   question lower (two-sided Wilcoxon over the chunks, p < 0.05); fewer agents, counting the
   largest network it sends any question to; and a higher token and dollar fitness.
2. **Better than the static prescription**: more accurate than `c3` alone, right more often on
   more questions, two-sided sign test p < 0.05.
3. **Its choice of questions matters**: more accurate than handing the same number of
   questions to each action at random, in fewer than 5% of 10,000 random draws
   (seed 20261110).

The second turn is held to the same three on fresh, and the first turn's prescriptor is
reported on fresh beside it. Fresh has 40 questions: a difference that would be significant on
100 can fail to be on 40, and a failure there is reported as that.

If any mark fails, the test fails and the README says the Prescriptor did not earn its place.

## Not a test

Reported beside the result, and never quoted as one:

- **The whole front on judge and fresh**: what each of its prescriptors really scored and
  cost, and whether one network alone was at least as accurate for no more money.
- **Knowing every answer**: for each question, the cheapest of the actions that got it right
  most often. A bound on any Predictor for these four actions, computed on the questions it is
  reported on.
- **The other 19 seeds**: the lowest, median and highest accuracy and cost of their chosen
  prescriptors, to show how much of the result is the seed.

## What a pass would not mean

The four networks were still found by mutation and selection, and by a person adding a
calculator. A pass would show that prescribing among measured networks per question, with
ESP's two models, beats prescribing one. It would not make the search that found the networks
ESP, and the README will keep saying so.

```bash
make prescriptor     # both turns, from the committed logs, for $0
```
