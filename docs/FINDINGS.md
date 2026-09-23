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
runs — spearman came out **positive every time, +0.280 to +0.755, median +0.671**, beating
the 0.2 threshold in all 120. At eleven measurements the same sweep gave +0.236 to +0.645,
median +0.518: the estimate is improving and its spread is narrowing as samples accumulate,
which is the expected shape and not yet a result.

The spread is still part of the finding. At this sample size the number moves by almost 0.5
depending on how `KFold` happens to split, so **any single figure quoted from it is an
artefact of a seed** — which is why a range is given here and nowhere is one value.

**The twelfth network is the first the Predictor actually chose.** A service wake trained on
eleven real samples, ranked a pool of mutants, paid for the top of it, and the candidate it
picked beat everything measured before. That is the ESP loop working end to end, once. It is
not evidence that the surrogate beats picking at random, because nothing in this repository
has yet run both on the same budget — and until that exists, the honest attribution for the
improvement is the evolutionary search with the Predictor unproven alongside it.

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
| Best evolved network beat the best seed on held-out tasks | **100% of splits** |

The comparison is genuinely out of sample: the winner is chosen on the selection
half and judged on tasks that took no part in choosing it. Evolved networks
outrank hand-written ones by two and a half places on average, and in two
hundred splits there was not one where the best seed beat the best evolved
network.

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
| accuracy | +0.610 [+0.306 … +0.724] | **−0.032** [−0.313 … +0.251] | **+0.628** [+0.367 … +0.989] | positive 20 / 20 | 0 / 20 |
| **token cost** | **−0.608** [−0.725 … −0.476] | **−0.125** [−0.388 … −0.011] | **−0.472** [−0.661 … −0.238] | **negative 20 / 20** | **20 / 20** |
| derived fitness | +0.648 | — | — | — | — |
| *(the old single model, same data)* | *+0.648* | *—* | *—* | *—* | *—* |

Medians over 20 cross-validation seeds, per-seed range in brackets; each null is itself the
median of 12 shuffles.

Re-run at 40 shuffles per null, the medians move — accuracy's null to −0.092 and token
cost's to −0.143, margins to +0.696 and −0.483 — and **nothing that matters moves**: both
margin signs hold at 20 / 20 and both exclusion counts are identical. The spearman column is
unchanged to three decimals, as it must be, since the shuffle count cannot touch it.

**The last column is the load-bearing one.** A margin is a difference of two noisy
quantities, and the null is the noisier: one seed can place token cost's anywhere between
−0.39 and −0.01. No point estimate of it should be quoted. What replicates is the decision —
token cost loses to its own null under every seed and shuffle count tried, accuracy under
none. That is what the gate acts on, and it is why the gate is recomputed per generation
rather than written down as a constant.

**The null column was not in the first version of this table, and leaving it out overstated
the finding.** That version reported −0.608 as though zero were the no-signal baseline.
Cross-validation on twelve samples can manufacture negative rank correlation on its own:
hold out a high value and the training mean drops, so the model predicts low and the error
correlates with the truth in the wrong direction. The baseline has to be measured, not
assumed.

`permutation_null()` measures it the only way that means anything — shuffle the targets so no
relationship can survive, run the **identical** cross-validation, take the median over twelve
shuffles.

Measuring it corrected the record in both directions:

- **Accuracy's margin clears the null under every seed** — +0.63 at 12 shuffles, +0.70 at
  40, never below +0.37. The accuracy finding is not at risk. Its *null*, however, is not the
  settled −0.03 this bullet first claimed: it drifts to −0.09 when measured over more
  shuffles and individual seeds reach +0.25, for the tie reason below. The margin survives
  because it is large, not because the baseline is known precisely.
- **Token cost's null is about −0.13, so the real effect is −0.47, not −0.61.** About 20%
  smaller than published. The finding survives — the margin is negative in every one of 20
  seeds at both shuffle counts, and the Predictor genuinely orders candidates by cost
  backwards — but the dramatic version of the number does not.

**A limit on the null itself, which decides which objective it can be trusted on.** A
permutation destroys a relationship only if permuting moves the values. Accuracy takes
**four distinct values across twelve networks**, so a shuffle frequently maps a value onto an
identical one and leaves the ordering largely intact — one shuffled draw came back at +0.88.
Its null is therefore weak, and its +0.63 margin should be read as indicative rather than
measured. The sweep shows it: accuracy's null drifts from −0.03 to −0.09 with the shuffle
count, and single seeds reach +0.25. The margin survives because it is large, never below
+0.37, not because the baseline is known. Token cost is distinct in all twelve, so its null
is sound — and token cost is the objective the finding is about. `test_the_null_is_only_meaningful_on_an_untied_objective`
pins this so nobody moves the check to the tied objective.

**And the spread is wide even where the null is sound.** The median is stable; a single draw
is not. Individual shuffled token correlations run from roughly −0.55 to +0.59 at this sample
size, and the *null itself* ranges from −0.39 to −0.01 across cross-validation seeds. The
margin over the null is the right statistic. It is not a precise one, and no figure in this
table should be quoted to three decimal places as though it were.

**Note the last row.** The combined figure is unchanged — median +0.648 either way. The
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

### A known defect in the model-tier feature, measured and left in place

Two of the thirteen features, `mean_model_tier` and `max_model_tier`, place each agent's model
on the ladder `gemini-3.5-flash-lite, gemini-3.5-flash`. The workers in every committed network
run `gemini-3.1-flash-lite`, which is not on that ladder, so it is encoded as the *highest*
tier. The cheapest model is labelled as costlier than the strongest one.

Correcting the ordering and re-running the 20-seed sweep at 12 shuffles:

| | token margin | token excluded | accuracy margin | accuracy excluded | derived fitness |
| --- | --- | --- | --- | --- | --- |
| as published | −0.472 [−0.661 … −0.238] | 20 / 20 | +0.628 [+0.367 … +0.989] | 0 / 20 | +0.643 |
| ordering corrected | −0.350 [−0.552 … −0.116] | 20 / 20 | +0.733 [+0.397 … +1.008] | 0 / 20 | +0.601 |

**No conclusion changes.** Token cost still loses to its own null under every seed and is still
excluded, and accuracy still clears its null under every seed. The token margin is about a
quarter smaller. The accuracy margin is a little larger. Accuracy still carries the combined
figure.

It is recorded rather than silently fixed, because every figure in this document describes the
Predictor as it actually ran. That includes the one whose choices `results/history.json`
records. Correcting the feature changes the model those figures describe, so the fix belongs
with a re-derivation of all of them, not ahead of it.

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
