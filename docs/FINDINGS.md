# Findings

The long-form record: what the measurements said, what running the
system changed about them, and the prior art this sits next to. The README
keeps the summary; this keeps the argument.

## What was measured

**Twelve networks, 17 tasks each, 204 real task runs on real model calls.** Three seeds
and nine mutants, every one of them cached with its per-task outcomes and its genome in
`tests/fixtures/cache/`, summarised in `results/history.json`.

| Origin | Accuracy | Of what it answered | Never finished | Tokens | Agents | Fitness |
| --- | --- | --- | --- | --- | --- | --- |
| **`mut:reassign_model`** | **0.9412** | **0.94 (16/17)** | **0** | 359,600 | 5 | **0.8941** |
| `mut:reassign_model` | 0.8824 | 1.00 (15/15) | 2 | **260,052** | 5 | 0.8453 |
| `mut:split_agent` | 0.8824 | 0.94 (15/16) | 1 | 272,068 | 5 | 0.8441 |
| `mut:merge_agents` | 0.8824 | 0.94 (15/16) | 1 | 381,778 | 3 | 0.8376 |
| `mut:toggle_search` | 0.8824 | 0.94 (15/16) | 1 | 345,226 | 5 | 0.8368 |
| `mut:split_agent` | 0.8824 | 0.94 (15/16) | 1 | 394,388 | 5 | 0.8319 |
| `mut:add_agent` | 0.8235 | 0.93 (14/15) | 2 | 249,662 | 2 | 0.7941 |
| `mut:merge_agents` | 0.8235 | 0.93 (14/15) | 2 | **242,670** | 3 | 0.7926 |
| `seed:flat_pair` | 0.8235 | 0.88 (14/16) | 1 | 316,074 | 3 | 0.7852 |
| `seed:solo` (one agent, one tool) | 0.8235 | 1.00 (14/14) | 3 | 377,716 | 1 | 0.7835 |
| `seed:designer_shaped` (the designer's shape) | 0.8235 | 0.88 (14/16) | 1 | 385,280 | 4 | 0.7761 |
| `mut:rewire` | 0.7647 | 0.87 (13/15) | 2 | 473,450 | 3 | 0.7107 |

**The search beat all three seeds twice, and both wins came from the same knob.** The best
network answers 16 of 17 against `designer_shaped`'s 14, on 7% fewer tokens. A second one
reaches 15 of 17 for 32% fewer. Six networks now sit on the Pareto front; five of the six
are evolved, and the one seed on it is there only for being the smallest.

**Both winners reassigned the front man's model and left every specialist alone.** The best
puts `gemini-3.5-flash` on the Coordinator and keeps all four specialists on
`gemini-3.1-flash-lite`; the cheaper winner does the same with `gemini-3.5-flash-lite`. The
agent that decides *who to ask* is the one worth paying for, and the agents that do the
looking are not. That is a per-agent setting neuro-san already supports, that nothing in
the framework tunes, and that is invisible in the topology — two networks with an identical
shape and a different model assignment are a 5.9-point accuracy difference apart here.

**The best network is the first to finish all seventeen tasks.** Every other measured
topology lost at least one run to a timeout or a blown recursion cap. Its single miss is a
wrong answer to T07 — it said `Bright Circuit` — which is a different and more respectable
failure than not finishing. A better router does not merely choose better; it stops the
network wandering until the clock runs out.

**The cost spread across the population is 95%**, from 242,670 tokens to 473,450, for
accuracies inside 18 points of each other. Topology and per-agent model assignment changed
what answering cost by a factor of two, and that is precisely the measurement neuro-san
cannot make today. It is only visible because of the `TOKEN_SCALE` fix below —
under the saturated scale everything clipped to the same penalty and scored as
indistinguishable.

**The Predictor did not contribute to this result.** `results/history.json` records one
cross-validated quality report, at generation 1 on nine samples: **spearman −0.333, mae
0.049, `beats_random: false`**. That is the surrogate the search actually ranked with, and
it ranked worse than chance.

Refitting `report_quality` over all twelve committed measurements says something better,
with a caveat attached. Across three input orderings and forty cross-validation seeds — 120
runs — spearman came out **positive every time, +0.261 to +0.725, median +0.618**, beating
the 0.2 threshold in all 120. At eleven measurements the same sweep gave +0.219 to +0.636,
median +0.509: the estimate is improving as samples accumulate, which is the expected shape
and not yet a result. `make figures` reruns both sweeps from committed data; the figures are
after the model-tier fix [below](#a-defect-in-the-model-tier-feature-measured-and-fixed).

The spread is still part of the finding. At this sample size the number moves by almost 0.5
depending on how `KFold` happens to split, so **any single figure quoted from it is an
artefact of a seed** — which is why a range is given here and nowhere is one value.

**The twelfth network is the first the Predictor actually chose.** A service wake trained on
eleven real samples, ranked a pool of mutants, paid for the top of it, and the candidate it
picked beat everything measured before. That is the ESP loop working end to end, once. On
its own it is not evidence that the surrogate beats picking at random. The offline selection
test [below](#does-the-predictor-pick-better-than-chance) is: trained on nine networks, it
picks the best of three unseen ones 62% of the time against 33% for chance. A search with the
Predictor and one without it, on the same budget, has still not been run. So the
improvement is credited to the evolutionary search, with the Predictor's contribution
measured offline only.

`esp/surrogate/predictor.py` reports `spearman` as `None` rather than `0.000` when there
are too few samples to cross-validate, because a placeholder printed in a measurement's
format is worse than an absence.

### The accuracy tie among the seeds is not a finding

All three seeds score 0.8235, and this table used to read *"identical accuracy — topology
did not change what these networks could answer"*, over a row of three 0.82s and the words
*"zero errors"*. They each failed three of the seventeen tasks, for different reasons:

- `solo` hit neuro-san's **recursion cap** on all three (T06, T07, T08) and never produced
  an answer to any of them. Its accuracy on those questions is unknown, not zero.
- `designer_shaped` **timed out** on one at `max_execution_seconds=600` and got two
  genuinely wrong.
- `flat_pair` timed out on one and got two genuinely wrong.

neuro-san returns a timeout and a blown recursion cap as ordinary answer strings, so they
reached the scorer, compared false against the expected answer, and were cached as wrong
answers. The runner already refused to cache an evaluation poisoned by a provider quota —
with a comment warning about *"a plausible-looking partial score that gets cached
forever"* — but the guard did not cover the two failures that actually happened.

Split the two apart and the ordering among the seeds inverts: on the questions each
actually finished, `solo` got everything right and `designer_shaped` did not. What topology
changed there was **how often the network finished at all**.

`tests/test_task_outcomes.py` pins this, including on the committed fixtures, so it cannot
quietly stop being true.

### Where the unfinished runs are

Seventeen of the 204 task runs never finished, all of them among the first eleven networks. They are not spread evenly:

| Task | Shape | Networks that never finished it |
| --- | --- | --- |
| T08 | 3-hop full-corpus aggregation | 9 of 12 |
| T06 | 3-hop full-corpus aggregation | 6 of 12 |
| T03 | 2-hop | 1 of 12 |
| T07 | 3-hop | 1 of 12 |

T06 and T08 are full-corpus aggregations — *"across all forty contracts, which has the
highest…"* — and `CorpusSearch` returns three documents a query, so answering one means
about fourteen successive searches. Three of the twelve managed T08. They are not free to
fail: whatever a network spends looping on a question it cannot answer is charged to it as
cost, so the spread above measures efficiency *and* failure mode together. Separating them
needs a re-measurement this project has not been able to buy.

The winner's two unfinished runs are worth naming precisely, because one of them is not
its fault: T08 was the 600-second timeout every network hits, and T03 came back as a
provider `500 INTERNAL`. It answered correctly everything it finished.

All twelve were measured on the same base model, **`gemini-3.1-flash-lite`**. That is not
incidental — the model is part of the genome hash, so a fitness compared across models
would not mean anything, and the table above is only a comparison because one model
produced all of it. `reassign_model` changes a *per-agent* model override inside that
network, which is why the winner is still comparable with the rest.

**The search that ran is one generation deep, and the free tier is why.** A candidate
costs about 165 provider requests across the task set, and the tier caps requests *per
day, per model* — 500 a day here, so one day's allowance buys three candidates. Eleven
candidates is roughly four days of budget spent in the right order: three seeds, then six
mutants bred from them, then a second generation of two. `results/history.json` records
one generation-1 quality report and no generation 2, because the budget ran out rather
than the search converging. A deeper search is a budget problem, not a code problem.

The free half did work as advertised: **118 candidates were scored by the surrogate** on
the way to those eleven real evaluations, at no provider cost. That ratio is the entire
argument for ESP over plain evolution, and it holds here even though the Predictor's
ranking did not.

`esp/eval/failover.py` moves to another model when a daily budget runs out, retiring the
spent one and rewriting in-flight calls. The measured caps live in `DAILY_CAPS` there, not
here, so there is one place to correct when a provider changes its mind.

**The ladder itself was wrong, and a run proved it.** It held three 20/day models, and a
candidate needs about 165 requests — so failing over to one of them spends 20 requests,
changes what is being measured mid-evaluation, and fails anyway. On 22 August a run
walked all four rungs in under three minutes with nothing measured. The rule was stated
correctly in a comment (*"the full ones allow 20/day, which buys no candidates at all"*)
directly above a list that broke it, which is why the caps are now **data** and the
ladder is derived from them:

```python
LADDER = [m for m in _PREFERENCE if DAILY_CAPS[m] >= REQUESTS_PER_CANDIDATE]
```

A prose rule cannot be checked against a list that contradicts it. An executable one can,
and a test now does.

The report states the budget ceiling on its first page rather than presenting one
generation as a converged search. Those are different claims, and the weaker one is the
true one.

## Held-out tasks: what this task set can and cannot support

Every result above selects a network on the same seventeen questions it
measures it with. That is the weakest point in the whole argument, and it can be
tested without spending anything, because each evaluation recorded the outcome
of every individual task. Split the seventeen in two, rank the population on one
half, and see where that half's winner lands on the other. Two hundred random
splits, nine tasks selecting and eight judging (`make holdout`).

It gives two answers, and they point in opposite directions.

**No, half of seventeen tasks cannot identify which network is best.**

| | |
| --- | --- |
| Selection winner also ranked first on the held-out half | 0% of splits |
| Selection winner ranked in the held-out top three | 0% of splits |
| Its mean held-out rank | 7.2 of 12 |
| Population ordering carried across the split, accuracy alone | +0.022 |
| Population ordering carried across the split, full fitness | +0.108 |

A rank correlation of +0.022 is no relationship at all. The reason is arithmetic
rather than mysterious: accuracy over eight tasks takes nine possible values and
moves in steps of 0.125, while the entire cost penalty across this population
spans about 0.03 — less than a quarter of one step. So held-out accuracy sorts
the networks into three or four large ties, and the cost terms decide the order
inside them. Which networks land in which tie is split-specific.

**This is a limitation of the measurement, not of the networks, and it
invalidates one kind of claim made above.** "This network is the best of the
twelve" does not survive a held-out split. Anything resting on the ordering of
individual networks — which operator is best, whether +0.8941 genuinely beats
+0.8453 — is inside the noise of a seventeen-task set. The fix is more tasks,
and nothing cheaper works: a better estimator cannot recover a signal the
sample size does not contain.

**Yes, searching beat not searching, and that does hold out of sample.**

| | |
| --- | --- |
| Searched winner beat the designer's shape on held-out tasks | 90% of splits |
| Mean held-out fitness margin over it | +0.0224 |
| The designer's shape, mean held-out rank | 10.4 of 12 |
| Mean held-out rank, the nine evolved networks | 5.8 of 12 |
| Mean held-out rank, the three seeds | 8.5 of 12 |
| Best evolved beat best seed, each chosen on the selection half | 66% of splits |

The comparison is genuinely out of sample: the winner is chosen on the selection
half and judged on tasks that took no part in choosing it. Evolved networks
outrank hand-written ones by two and a half places on average.

**A correction.** This table used to end with "best evolved network beat the
best seed on held-out tasks: 100% of splits". No command printed it, and it was
not a held-out result: it picked each group's best by its score on the held-out
half itself, and it set the best of nine evolved networks against the best of
three seeds, which favours the larger group before any network is compared.
Choosing each group's best on the selection half and judging it on the other,
as every other row here does, gives 66%. `make holdout` now prints both, the
in-sample one labelled as such, so the old figure cannot be quoted as a
held-out result again.

So the claim this evidence supports is the population-level one — **evolutionary
search over neuro-san topologies produces better networks than the shape the
designer produces, robustly** — and not the per-network one. The headline table
is still the honest report of what was measured; what it cannot bear is being
read as a ranking.

Two caveats on the test itself. Held-out questions come from the same generated
world, so this measures stability across questions rather than transfer to a new
domain, which would be a stronger test and needs a second world. And the
selection half is itself only nine tasks, so the winner it picks is noisier than
a real search's would be after a fuller run.

## The held-out bank: the first out-of-sample measurement

Every figure above selects and judges networks on the same seventeen questions, or on
halves of them. `esp/eval/bank.py` generates 250 more over the same corpus, about
entities the seventeen never name, with every answer re-derived from the corpus text by a
test. The first live run on it, on 25 September 2026 with a free Gemini key, measured two
networks on the first 24 questions, four at each depth. `make bank-report` reproduces this
table from `results/heldout_bank/`, with no key.

| | Held out (24) | Tokens | On the 17 | Tokens |
| --- | --- | --- | --- | --- |
| `seed:designer_shaped` | **23 / 24** | 159,086 | 14 / 17 | 385,280 |
| `mut:reassign_model` 6859dd | 21 / 24 | 176,740 | 15 / 17 | 260,052 |

**The evolved network did not beat the designer's shape on questions it was not selected
on.** It answered two fewer and spent 11% more tokens, reversing the 32% saving it showed
on the seventeen. The whole difference sits in one shape, the four-document penalty
arithmetic, where it got 1 of 4 and the designer 3 of 4; at every other depth, including
the nine-document sums, both answered 4 of 4. Paired question by question, two questions
separate them and both favour the designer, an exact McNemar p of 0.5: **no difference is
established in either direction**. What the run settles is narrower, and it matters for
the headline: out of sample, it gives no support to the evolved network being better.

**The bank is easier than the seventeen for these networks.** The designer scored 96% here
against 82% on the seventeen, whose two whole-corpus aggregates (T07 and T08) nearly every
network fails, and the bank asks nothing like them. A bank that separates networks needs
harder shapes, such as aggregates over all forty contracts that a three-document retrieval
cannot answer in one call. That is the next change, and it is not made here.

**Why only two networks and 24 questions.** The free tier allows 500 requests a day on each
flash-lite model and a question costs about ten. The champion, `3bf9c0`, could not be
measured at all: its router runs on gemini-3.5-flash, which the free tier allows about 20
requests a day, and which answered 503 "high demand" to three of four smoke-test questions
the same day. Network `6859dd` was chosen because it is evolved, ranked second, and runs
only on the 500-a-day models.

### The harder select set, calibrated

The bank could not rank networks because the designer's shape scored 96% on it. The select
set was built harder on purpose, and its first live measurement, on 26 September 2026 with a
free Gemini key, confirms it: on its first 20 questions, which cover every kind it asks, the
designer's shape answered **16 of 20 (80%)**, using 194,828 tokens. That is under the 90%
ceiling the experiment checks before it spends anything, so the set leaves room to tell
networks apart. All four misses are the same kind of question: late-delivery penalties
combined across incidents, hours times rate, summed or differenced. That is the shape the
evolved network also missed on the bank; arithmetic over several documents, not retrieval,
is where these networks fail. The report is in `results/calibration/`.

### The same-budget experiment, built and not yet run

Twelve networks and seventeen questions cannot say whether the Predictor helps, and the
bank above cannot rank networks. `make experiment` is built to answer both, and has not been
run: it needs a paid key. It searches twice from the same start with the same budget, once
choosing candidates by the Predictor and once at random; selects on 60 harder questions, 40%
whole-corpus aggregates; and judges each winner and the designer's shape on 100 questions
about entities the select set never names. At the default budget it measures about 89
networks, so each arm's Predictor trains on up to 49 rather than 12. Its logic is tested end
to end against a fake provider (`tests/test_experiment.py`); its answer is not known yet, and
this section will report it whichever way it falls.

### The token price hid the router

The fitness counts tokens. Every committed measurement also records what the provider charged
for it, and in dollars the ranking tells a different story (`make cost-report`):

| Network | Accuracy | Tokens vs designer | Dollars vs designer |
| --- | --- | --- | --- |
| `mut:reassign_model`, best measured | 0.9412 | −7% | +63% |
| `mut:reassign_model`, the champion | 0.8824 | −33% | −9% |
| `mut:split_agent` | 0.8824 | −29% | −22% |

The best network is cheaper in tokens and much dearer in money, because its router runs
gemini-3.5-flash while the designer's shape runs everything on gemini-3.1-flash-lite: 0.504
dollars a million tokens against 0.288. The champion, chosen as the cheapest win by tokens,
is not the cheapest network at its accuracy in dollars; `mut:split_agent` is. On the 24 bank
questions the champion also cost 31% more than the designer's shape. So the v1 fitness is
left as it was, because every committed result was selected under it, and the pool benchmark
selects on `esp.eval.pricing.fitness_dollars` instead.

### Two hundred judge questions, four new kinds

A paired test on seventeen questions needs a gap of more than 30 accuracy points to register;
on a hundred, about a dozen. `meridian-judge-200` adds a hundred questions to the judge set,
from the same half of the company, of four kinds none of the other sets asks: counts and sums
over a time window, totals over only the records that pass a condition, differences between
two records, and questions the documents cannot answer (a contract that does not exist, a
field no document records). Every one of them carries the same instruction to answer "not
stated" when the documents do not say, so the instruction gives nothing away, and a network
that invents an answer is wrong. No answer covers more than a quarter of the set, so a
network cannot score by always saying "2". The corpus is unchanged, so every v1 measurement
stands. `tests/test_judge_plus.py` re-derives every answer from the document text.

### The pool benchmark, and two things its rehearsal found

One search per method is one sample of that method. The pool benchmark (`esp/evolve/pool.py`,
`make pool`) measures a pool of 120 networks once and compares strategies over it in hundreds
of replicate searches for free. It has not been run on a real provider. Building it against a
simulated provider found two things worth recording before anyone pays for it.

- **Scoring a pick on the answers it was chosen by rewards luck.** On a pool of pure noise, a
  per-question Predictor looked significantly better than random choice: its preference was
  consistent, the pool is fixed, and it happened to prefer the network that was lucky on
  those questions. A search now sees half the select questions and what it picks is scored on
  the other half. On pure noise the advantage disappears, as it should, and a planted
  relationship is still found (`tests/test_pool.py`).
- **A tree-based Predictor cannot extrapolate.** Where larger teams do better and the
  measured start contained no team as large as the best in the pool, the Predictor could not
  guess it and did no better than chance. It ranks inside the range it has seen, which is
  why the paid run measures a broad pool rather than a handful.

The full-size rehearsal (`make pool REHEARSE=1`) made exactly the 9,000 question-runs the plan
prices, 7,200 for the pool and 1,800 for judging, in about seven minutes and for nothing. Its
comparisons are of a simulated world with a planted relationship, so they say nothing about
real networks; `results/rehearsal/` keeps them, labelled as simulated.

### A Predictor that learns per question

`esp/surrogate/per_question.py` predicts, for a network and a question, the chance of a right
answer, from the network's structure beside the question's kind, so twelve networks on
seventeen questions are 204 training rows rather than 12. On a planted population where more
agents help totals and hurt joins, it ranks unseen networks on totals alone at a rank
correlation above 0.6, where even a perfect estimate of each network's overall average does
worse, because the average cannot say which network is good at which kind. It is tested for
learning more from more networks, narrower uncertainty with more data, and determinism
(`tests/test_per_question.py`). On real measurements it has not been tested yet; the pool
benchmark is where it will be.

## What the Predictor is, exactly

Asked directly in review — twice, and the second time by a co-author of the ESP paper, who
said he had *"a hard time understanding what the predictor surrogate is in the ESP for this
general use-case"* and described what it ought to be:

> Typically, the surrogate model is one or more ML models that act as predictors for various
> outcome objectives we expect from the target we are optimizing — in this case, the
> Neuro-san agent network. The prescription then generates actions optimized against the
> surrogate.

The first version was not that, and the confusion was the code's fault rather than the
reader's. It trained **one** model directly on the already-scalarised fitness. That single
choice fused the three things a reader has to be able to separate — what is predicted, how
it is scored, what proposes the next candidate — into one object, and no amount of prose
around it would have made them distinct.

### The three things, separated

| | What it is | Learned? | Where |
| --- | --- | --- | --- |
| **Predictor** | One `GradientBoostingRegressor` per outcome objective — 200 trees, depth 3, learning rate 0.05, subsample 0.9 — fitted on `(genome → outcome)`. Predicts **accuracy** and **token cost**. | Yes, from real evaluations | `esp/surrogate/outcomes.py` |
| **Fitness** | `accuracy − 0.06·min(tokens/600000, 1) − 0.02·(agents/9)`. A fixed weighting. Applied to *measured* outcomes it scores Phases A and D; applied to *predicted* outcomes it ranks Phase C. | No — arithmetic | `esp/evolve/loop.py::scalarise` |
| **Prescription** | Seven mutation operators plus elite selection, run against the Predictor. | No | `esp/genome/mutations.py` |

**Agent count is an objective with no model.** It is an exact property of a genome, so
`predict_outcomes` counts it. Asking a regressor to estimate a number already in hand adds
error and buys nothing. Depth is excluded from fitness entirely and is reported only.

Two things follow from making the split, beyond matching the paper. The weights stop being
baked into a fitted model, so re-weighting no longer needs a retrain on a population that
cost four days of provider budget to collect. And each objective becomes separately
measurable — which is how the next section exists at all.

### Splitting it found a defect: token cost does not beat its own null

Reported per objective over the twelve measured networks:

| Objective | spearman | permutation null | margin over null | margin sign | excluded |
| --- | --- | --- | --- | --- | --- |
| accuracy | +0.626 [+0.268 … +0.723] | **−0.072** [−0.361 … +0.298] | **+0.733** [+0.397 … +1.008] | positive 20 / 20 | 0 / 20 |
| **token cost** | **−0.531** [−0.795 … −0.382] | **−0.168** [−0.423 … +0.025] | **−0.350** [−0.552 … −0.116] | **negative 20 / 20** | **20 / 20** |
| derived fitness | +0.601 [+0.431 … +0.683] | — | — | — | — |
| *(the old single model, same data)* | *+0.618* [+0.261 … +0.725] | *—* | *—* | *—* | *—* |

Medians over 20 cross-validation seeds, per-seed range in brackets; each null is itself the
median of 12 shuffles. The single-model row is the 120-run sweep above. All of it is after
the model-tier fix and regenerated by `make figures`.

Re-run at 40 shuffles per null, the medians move — accuracy's null to −0.127 and token
cost's to −0.157, margins to +0.741 and −0.374 — and **nothing that matters moves**: both
margin signs hold at 20 / 20 and both exclusion counts are identical. The spearman column is
unchanged to three decimals, as it must be, since the shuffle count cannot touch it.

**The last column is the load-bearing one.** A margin is a difference of two noisy
quantities, and the null is the noisier: one seed can place token cost's anywhere between
−0.42 and +0.03. No point estimate of it should be quoted. What replicates is the decision —
token cost loses to its own null under every seed and shuffle count tried, accuracy under
none. That is what the gate acts on, and it is why the gate is recomputed per generation
rather than written down as a constant.

**The null column was not in the first version of this table, and leaving it out overstated
the finding.** That version reported −0.608 (−0.531 after the tier fix) as though zero were the no-signal baseline.
Cross-validation on twelve samples can manufacture negative rank correlation on its own:
hold out a high value and the training mean drops, so the model predicts low and the error
correlates with the truth in the wrong direction. The baseline has to be measured, not
assumed.

`permutation_null()` measures it the only way that means anything — shuffle the targets so no
relationship can survive, run the **identical** cross-validation, take the median over twelve
shuffles.

Measuring it corrected the record in both directions:

- **Accuracy's margin clears the null under every seed** — +0.73 at 12 shuffles, +0.74 at
  40, never below +0.39. The accuracy finding is not at risk. Its *null*, however, is not the
  settled −0.03 this bullet first claimed: it is −0.07 at 12 shuffles and −0.13 at 40, and
  individual seeds reach +0.30, for the tie reason below. The margin survives because it is
  large, not because the baseline is known precisely.
- **Token cost's null is about −0.17, so the real effect is −0.35, not −0.53.** A third
  smaller than the raw correlation. The finding survives — the margin is negative in every one of 20
  seeds at both shuffle counts, and the Predictor genuinely orders candidates by cost
  backwards — but the dramatic version of the number does not.

**A limit on the null itself, which decides which objective it can be trusted on.** A
permutation destroys a relationship only if permuting moves the values. Accuracy takes
**four distinct values across twelve networks**, so a shuffle frequently maps a value onto an
identical one and leaves the ordering largely intact. Its null is therefore weak, and its
+0.73 margin should be read as indicative rather than measured. The sweep shows it:
accuracy's null drifts from −0.07 to −0.13 with the shuffle count, and single seeds reach
+0.30. The margin survives because it is large, never below +0.39, not because the baseline
is known. Token cost is distinct in all twelve, so its null
is sound — and token cost is the objective the finding is about. `test_the_null_is_only_meaningful_on_an_untied_objective`
pins this so nobody moves the check to the tied objective.

**And the spread is wide even where the null is sound.** The median is stable; a single draw
is not. Individual shuffled token correlations spread widely at this sample size, and the
*null itself* ranges from −0.42 to +0.03 across cross-validation seeds. The
margin over the null is the right statistic. It is not a precise one, and no figure in this
table should be quoted to three decimal places as though it were.

**Note the last two rows.** The combined figure barely differs — +0.60 derived from the
per-objective models, +0.62 from the old single model. The
scalarised surrogate looked healthy, was healthy by its own measure, and was concealing
this, because the accuracy term is large enough to carry the total on its own. That is the
part worth generalising: a single scalarised quality number cannot express *which*
objective is broken, and a multi-objective search reporting one number will not notice when
half of it is noise.

It is now said out loud in three places rather than left in a table: `make offline` prints a
warning under the quality line, the service puts it in the wake report an unattended
operator reads, and `tests/test_outcome_surrogate.py::test_token_cost_is_anti_predicted_on_the_committed_population`
fails if it ever stops being true — so fixing it forces this section to be rewritten instead
of allowing it to go quietly stale.

**What has not been done:** nothing here diagnoses *why*, and no feature has been added to
try to fix it. Both need more than twelve evaluations to be worth doing, and inventing a
fix that cannot be validated would be worse than reporting the defect.

### Does the Predictor pick better than chance?

Every figure above describes the Predictor. The search uses it to **choose**: of the
candidates it could pay to measure, which one. `make ablation` asks that directly, with no
key and no calls. There are 220 ways to hold out three of the twelve measurements. For each
one it trains on the other nine exactly as a service wake does: cross-validate, gate, fit.
Then it picks the held-out network it predicts best and looks up how that network actually
scored.

| Picker | Picked the best of three | Mean regret (fitness) | Could rank |
| --- | --- | --- | --- |
| random (exact expectation) | 33.3% | 0.0387 | — |
| **Predictor, as a wake runs it** | **62.1%** | **0.0162** | 212 / 220 |
| Predictor, gate switched off | 71.8% | 0.0066 | 220 / 220 |
| Predictor, token cost always excluded | 62.3% | 0.0144 | 220 / 220 |

Regret is the fitness the pick left behind against the best network in its three. Where
the Predictor could not rank, because the gate excluded both objectives, its pick is scored
as chance.

**The Predictor picks better than chance.** It picks the best network nearly twice as often
as a random picker does, with well under half the regret. Of 20,000 simulated random
pickers on the same held-out sets, none did as well. That is a description, not a p-value:
the simulation treats the 220 sets as independent, and they are not, since each network is
in 55 of them, so the effective sample is nearer twelve than 220. It says nothing about
networks outside this population. It is still the first direct evidence in this repository
that the surrogate helps the search choose.

**The gate makes the choice worse, and why is not understood.** The gate excluded token cost
in 205 of the 220 training sets, as the per-objective figures predict, and accuracy in 11.
Excluding token cost costs about ten points of "picked the best" and more than doubles
regret. The tokens-excluded row shows that exclusion alone accounts for nearly all of it.
Yet on these same held-out networks the token model is backwards. It puts a held-out pair
in the right order only 28.8% of the time (660 pairs), where chance is 50%, while the
accuracy model manages 84.8% (460 pairs). A term that ranks its own objective backwards is
improving the choice on fitness. Token spend and accuracy are uncorrelated across the twelve networks (Spearman
−0.08), so the obvious explanation, that spending tracks accuracy, does not hold. Twelve
networks cannot separate a mechanism from a coincidence.

**The gate stays on, and this is recorded against it.** The gate asks whether a model
predicts its own objective. On held-out networks the answer for token cost is no, as it is
in cross-validation. The ablation asks the question the search actually depends on, and on
that question the gate loses. Keeping a term because it helps for no known reason is how a
search comes to rest on an accident of twelve samples. Removing a safeguard on the same
evidence would be the same mistake the other way round. The gate is one argument to `fit`,
and `make ablation` will show when the evidence moves. The change this points to is gating
an objective on held-out *selection* rather than on its own rank correlation. Testing that
needs more measurements than exist.

**What this still does not show** is that a search using the Predictor finds better networks
than one that does not, for the same budget. Picking the best of three measured networks is
the step the search depends on, not the search itself. The online comparison needs paid runs
on both arms, and they have not been bought.

### The feature set

**The Predictor's input is thirteen numbers describing the network's structure and
configuration, and nothing that was measured:**

| | |
| --- | --- |
| Shape | `agents`, `depth`, `edges`, `mean_branching`, `max_branching`, `leaves`, `top_degree` |
| Tools | `searchers`, `searcher_fraction` |
| Models | `mean_model_tier`, `max_model_tier` |
| Prompts | `mean_instruction_chars`, `total_instruction_chars` |

The exclusion is deliberate and it is the reason the surrogate is worth anything: a feature
derived from a measurement would mean the Predictor needed a real evaluation in order to
predict a real evaluation. `_tier` maps a model name to its position on the cost ladder and
never raises on an unfamiliar one, because failover substitutes models that are not on the
list and a genome measured under a swapped model must not take feature extraction down
with it.

A gradient-boosted tree ensemble rather than anything larger because the training set is
**tens of samples**, and there is no neural network anywhere in this repository. Below eight
samples it refuses to fit at all: `predict` then returns the mean of whatever it has seen,
`ranks()` reports `False`, and both the batch loop and the service say out loud that the
generation is a random search rather than printing a ranking over one repeated constant. An
objective whose measured values are constant gets no model either, and a surrogate missing
one of its objectives does not claim to rank.

Quality is reported as **cross-validated Spearman rank correlation**, not error, because
the Predictor's job is ordering. It never has to price a topology correctly — it has to put
the promising ones above the hopeless ones so that real budget goes to the top of the list.
`report_quality` runs `KFold` over the whole scored population, publishes a figure per
objective **and** for the derived fitness, and names any objective that ranks no better than
chance rather than letting it disappear into a mean. It publishes whatever it says,
including the −0.333 above and the −0.608 in the section before this one.

**The deeper departure: there is no context.** ESP prescribes *actions for a context*, and
the Prescriptor is precisely the model that maps one to the other. This project has **no
context variable anywhere** — every candidate is evaluated against the same fixed seventeen
tasks. That is not an omission that could be patched by bolting on a network: with nothing to
map from, a Prescriptor has no input, which is why seven mutation operators occupy that slot
instead. Building a real Prescriptor here starts with deciding what the context *is* — a
task distribution, a budget, a domain — and that decision is the research, not the network.
Stated before a reader has to ask, because "ESP" without it invites exactly the confusion the
review raised.

**Where this is not canonical ESP.** In ESP as Cognizant AI Lab published it, the
Prescriptor is *also* a learned model — a network mapping context to actions, evolved
against the Predictor. Here there is no learned Prescriptor. The prescription step is the
seven mutation operators plus elite selection over measured fitness, so what this
implements is the surrogate-assisted half of ESP with an ordinary evolutionary search in
place of an evolved prescriptor. That is a deliberate simplification for a genome that is a
HOCON agent network rather than a fixed-length action vector, and it is a real difference
rather than a detail: anyone comparing this against the ESP papers should expect to find
one model here, not two.

### A defect in the model-tier feature, measured and fixed

Two of the thirteen features, `mean_model_tier` and `max_model_tier`, used to place each
agent's model by its position on the configured ladder, `gemini-3.5-flash-lite,
gemini-3.5-flash`, with anything off the ladder placed above everything on it. The workers in
every committed network run `gemini-3.1-flash-lite`, which is not on that ladder, so the
cheapest model was encoded as the *highest* tier. On a machine configured for Claude or
OpenAI every committed model fell off the ladder at once and the feature went constant.

The tier is now a property of the model: its rung (cheap or strong, from the model family),
then its release within the rung, so `gemini-3.1-flash-lite < gemini-3.5-flash-lite <
gemini-3.5-flash` and `claude-haiku < claude-sonnet` on any machine
(`esp/config.py::model_rank`). The 20-seed sweep at 12 shuffles, before and after:

| | token margin | token excluded | accuracy margin | accuracy excluded | derived fitness |
| --- | --- | --- | --- | --- | --- |
| before the fix | −0.472 [−0.661 … −0.238] | 20 / 20 | +0.628 [+0.367 … +0.989] | 0 / 20 | +0.643 |
| after the fix | −0.350 [−0.552 … −0.116] | 20 / 20 | +0.733 [+0.397 … +1.008] | 0 / 20 | +0.601 |

**No conclusion changes.** Token cost still loses to its own null under every seed and is
still excluded, and accuracy still clears its null under every seed. The token margin is
about a quarter smaller, the accuracy margin a little larger, and accuracy still carries the
combined figure.

Every Predictor figure in this document and the README is the after-fix one, and
`scripts/surrogate_figures.py` (`make figures`) regenerates all of them from committed data.
The one exception is the −0.333 in `results/history.json`: that is the Predictor the first
search actually ranked with, and it stays as recorded.

### The Pareto front now breeds

`pareto()` was computed on every run, written to `history.json`, and drawn in three PDFs.
Selection ignored it completely and took the top `elite` candidates by scalarised fitness.
That is multi-objective in the report and single-objective in the search, and the gap has a
cost: the cheapest network ever measured contributes nothing to breeding if one particular
weighting puts it mid-table — which is the exact outcome a Pareto front exists to prevent.

Parents are now chosen by non-dominated sorting on (accuracy up, tokens down, agents down),
topped up with the best scalarised fitness when the front is smaller than the elite, so a
one-point front cannot collapse the search onto a single parent. The batch loop and the
service wake share one `non_dominated()` so the front the reports draw and the front the
search breeds from cannot drift apart.

## What measurement changed

Every item here was found by running the system, not by reasoning about it. Most of
them were bugs that made a *good* topology score zero — the one failure mode that
does not announce itself, because a search that is being lied to still produces a
smooth curve.

**The top agent was not the front man.** `to_hocon()` emitted agents alphabetically,
but neuro-san takes the *first* entry in `tools` as the front man. Ordering was therefore
not cosmetic — it decided which agent the request entered through. `designer_shaped`, the
baseline the whole experiment compares against, ran with `ContractSpecialist` in front
and still scored 0.82; `flat_pair` ran with `Arithmetic` in front, which answered *"you
did not provide any numbers or an operation"* to all seventeen questions and scored
**0.00**. Neither had ever been run as written. With the fix, `flat_pair` scores 0.82 —
it was never a bad topology.

**The cost objective had no gradient.** `TOKEN_SCALE` was 60,000 while the cheapest
possible topology — one agent, one tool — spends over 250,000 tokens on the task set.
Every candidate sat past the cap, so `min()` clipped them all to the same penalty: a
network costing 300k and one costing 900k scored *identically* on cost. The run would
have printed a Pareto front and called itself multi-objective while optimising accuracy
alone. Two tests now pin the property rather than the constant.

**Rate limiting starved the agents it paced.** `Bucket.acquire()` blocks with
`time.sleep`, and it was being called from an `async` call site. neuro-san runs agents on
asyncio, so every pacing wait froze the whole event loop — every concurrent agent at
once, each still spending its own `max_execution_seconds`, then all cancelled together.
Ruff's `ASYNC251` is what caught it.

**Free-tier quota is per-model, and there are two of them.** A per-minute rate *and* a
per-day cap, both applied per model, and they need opposite responses: a per-minute limit
clears by waiting, so the limiter sleeps on it; a per-day cap never clears within a run,
so sleeping on it burns the clock and every candidate after it scores zero. They are told
apart by the `quotaId` in the 429 payload. `esp/eval/ratelimit.py` paces every call
through a per-model sliding window, because a 429 returns as an agent error and teaches
the search that a good topology is bad.

**The service recorded the wrong exhausted model.** A wake that hit the daily cap
reported an empty exhausted list, because it matched the error text against the failover
ladder — and the model that actually failed was the *network default*, which is not a
ladder member. Every wake for the rest of that day would have spent its first calls
rediscovering the same dead model. The name is now read out of the provider's own 429.

**The published package was missing a piece.** `reportlab` — needed to build the PDFs
this repository documents as deliverables — was in no dependency list at all. It kept
working here because a developer venv had it from something else, which is precisely why
this class of bug survives: a machine that already has a package can never detect a
missing declaration, and CI is the only place that installs from the list alone. Anyone
cloning this cleanly would have got a project that could not build its own report.

**The image could not run the service it was documented as running.** The `Dockerfile`
copied `esp/`, `scripts/` and `tests/` — not `apps/` and not `registries/`. The manifest
*is* the service, and it was not in the image, so `docker compose up -d` was a documented
command that could not work. `tests/test_container.py` now parses both files and checks
every path the documented commands need; five of its eight tests fail against the
previous versions.

**The first task set had no headroom.** A single agent with a single tool scored **1.00**
on the original 13 questions — perfect, including three-hop joins and arithmetic. There
was nothing for evolution to improve. The corpus was widened from 40 documents to 124,
retrieval narrowed from five results to three, and four-hop and aggregate shapes added.
An experiment whose baseline is already perfect measures nothing.

## Prior art

### The two papers this sits directly on top of

**ESP.** *Effective Reinforcement Learning through Evolutionary Surrogate-Assisted
Prescription* — Francon, Gonzalez, Hodjat, Meyerson, Miikkulainen, Qiu, Shahrzad, GECCO
2020 ([arXiv:2002.05368](https://arxiv.org/abs/2002.05368)). This repository is named
after it, so the departure has to be stated first rather than found:

> **In the paper the Prescriptor is a neural network, evolved to maximise the surrogate's
> predictions. Here there is no learned Prescriptor at all.** Prescription is seven
> mutation operators plus elite selection over measured fitness. One learned model, not
> two.

That is a real difference, not a detail. What is faithful is the Predictor's role and the
sample-efficiency argument: evaluate cheaply on a surrogate, spend real evaluations only
on the elite. The paper's surrogate is "a random forest or a neural network trained with
gradient descent", so a `GradientBoostingRegressor` is squarely in family.

**LEAF.** *Evolutionary Neural AutoML for Deep Learning* — Liang, Meyerson, Hodjat, Fink,
Mutch, Miikkulainen, GECCO 2019. LEAF evolves network **architectures and size**, not just
hyperparameters, and reports that architecture optimisation beats hyperparameter
optimisation while shrinking the network.

**LEAF is the closer relative, and the more honest lineage.** This evolves the topology of
an agent network and the model assigned to each agent, scored against a multi-objective
fitness that prices accuracy, tokens and size — which is LEAF's shape, one level up, with
agents where LEAF had layers. Calling the whole thing "ESP" overstates the
Prescriptor half and understates what it actually resembles.

### Automated agent architecture search

A crowded, fast-moving field: ADAS (meta-agent
plus archive), AFlow (MCTS over operator graphs), GPTSwarm (RL over edge probabilities),
MaAS and AutoMaAS (agentic supernets), AgentSquare (modular design space), EvoMAS
(evolutionary generation of multi-agent systems), Promptbreeder. **The idea of searching
agent architectures is not new here, and neither is the surrogate.**

This section previously claimed the surrogate-assisted variant as a contribution, on the
grounds that *"the systems above do not use"* it. That was wrong.
**AgentSquare (ICLR 2025) introduces a performance predictor implementing an in-context
surrogate model, precisely so it can skip unpromising candidates without paying for a real
evaluation** — the same idea, published first, and it was not in the list above. Predictor-
based screening for agentic workflows is an established technique, not an insight of this
repository.

What is actually left, once that is subtracted:

- **No fitness function for neuro-san exists.** Verified against neuro-san 0.6.96: zero
  occurrences of fitness, genetic, evolve, pareto or surrogate anywhere in the package,
  and no comparative scoring between topologies in neuro-san-studio either.
  `agent_network_designer` generates a network; nothing measures one.
- **neuro-san does contain evaluators, and they are not this.**
  `neuro_san/test/evaluators/` holds `assertIn` / `assertLess` style assertions that check
  one network's answer in an integration test. That is pass/fail regression testing of a
  single network, not a scalar objective over a population, and the distinction is the
  whole project. Claiming neuro-san "has no evaluation" is false; claiming it has no
  *fitness function* is true.
- **neuro-san already ships ESP — at the other level.**
  neuro-san's own `esp_decision_assistant` registry is an agent network that uses ESP's
  Context/Actions/Outcomes framing to help a *person* make a decision ("Should I buy a new
  car or lease one?"), with prescriptor and predictor agents inside it. Cognizant AI Lab's
  own *NeuroSAN+NeuroAI* (Miikkulainen, Fink, Francon et al., 2025) works at that level
  too. There, ESP is what the agent network **does**. Here, ESP is what **designs** the
  agent network. Same vocabulary, opposite direction, and anyone from that lab will ask
  which one this is within a minute.

So the honest contribution is narrow and it is engineering, not research: a working,
measured fitness function for neuro-san topologies, a budget-aware service that can
actually run one on a free tier, and the bugs that only running it could find.
