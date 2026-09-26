# Design

How neuro-san-esp searches for a better agent network, and what each part is. The
[README](../README.md) has the summary; [FINDINGS.md](FINDINGS.md) has the measurements.

## The loop

Four phases, repeated. Phase C is the point: it is free, so the search can be wide.

```text
Phase A   measure the seed topologies for real        ->  (genome, outcomes) pairs
Phase B   train one Predictor per outcome objective   ->  cheap outcome estimates
Phase C   breed candidates, rank by derived fitness   ->  zero LLM calls
Phase D   pay for real evaluation of the elite only   ->  feed back into B
```

Fitness is **derived** from the Predictor's outputs, never learned by it — see
[Predictor, fitness, prescription](#predictor-fitness-prescription--which-is-which) below for
why that distinction matters and what it exposed.

## The genome is neuro-san's own format

A candidate **is** a neuro-san `agent_network_definition` — the same HOCON the framework
serves — plus a per-agent `model` override. Nothing is invented: a genome renders straight to
a registry file and runs as an ordinary agent network. Defined in
`esp/genome/definition.py`; the three starting topologies are in `esp/genome/seeds.py`.

The configured model is **part of the genome hash**, deliberately. A fitness measured on one
model must not be mistaken for the same network on another.

## Fitness is measured, not asserted

Each candidate answers all 17 tasks through a real neuro-san session. Three objectives are
recorded, then scalarised for selection while the Pareto front is kept separately:

```text
fitness = accuracy − 0.06 · min(tokens / 600000, 1) − 0.02 · (agents / 9)
```

The evaluation world (`esp/eval/world.py`) is generated from a fixed seed: 24 depots, 40
contracts, 60 incidents, 124 documents, and 17 questions of one to four hops whose answers
are correct by construction. Retrieval is a deterministic coded tool, so fitness measures the
topology rather than a retrieval layer that drifts between generations. Timeouts and blown
recursion caps are classified as **unfinished rather than wrong** (`esp/eval/runner.py`),
because neuro-san returns them as ordinary answer strings and scoring them as wrong answers
teaches the search that a good topology is bad.

## Seven mutation operators, behind a validity gate

`add_agent`, `remove_agent`, `rewire`, `split_agent`, `merge_agents`, `toggle_search`,
`reassign_model` (`esp/genome/mutations.py`). An invalid mutant — an unreachable agent, a
cycle, no front man — is **discarded, never repaired**, so every candidate that reaches a
real evaluation is a network neuro-san would actually serve.

**The Pareto front breeds.** Parents are chosen by non-dominated sorting on
(accuracy up, tokens down, agents down), topped up with the best scalarised fitness when the
front is smaller than the elite. Selection used to take the top few by fitness alone while
the front was computed, plotted in three documents and never consulted — multi-objective in
the report and single-objective in the search.

## Predictor, fitness, prescription — which is which

Three words get used loosely about a loop like this, and conflating them is how a search
comes to optimise something nobody chose. In this repository they are three separate things:

| | What it is | Where it lives |
| --- | --- | --- |
| **Predictor** | The surrogate. **One `GradientBoostingRegressor` per outcome objective**, learned from real evaluations. It predicts *accuracy* and *token cost*. It never sees a fitness and never sees the weights below. | `esp/surrogate/outcomes.py` |
| **Fitness** | A fixed weighting applied to outcomes — not learned, not fitted, just arithmetic. Over **measured** outcomes it scores Phases A and D. Over **predicted** outcomes it ranks Phase C. Same function both times. | `esp/evolve/loop.py::scalarise` |
| **Prescription** | What proposes the next candidate. Seven mutation operators plus Pareto-front selection, run against the Predictor. **Not** a learned model — see the departure noted at the top. | `esp/genome/mutations.py` |

**There is no context here, and that is the deeper departure.** In ESP a Prescriptor maps a
*context* to the actions to take in it. Every candidate in this project is evaluated against
the same fixed task set, so there is no context to map from — which is why mutation operators
occupy the prescription slot rather than a learned model. Building a real Prescriptor starts
with deciding what the context is; that decision is the research, not the network.

Agent count is a fourth objective and has **no model at all**: it is an exact property of a
genome, so it is counted rather than estimated. A regressor asked to guess a number already
in hand only adds error.

The Predictor's input is **thirteen structural features** of the genome — agent count, depth,
edges, branching, leaves, searchers, model tiers, instruction lengths — and never a measured
quantity. A feature derived from a measurement would mean the Predictor needed a real
evaluation in order to predict one.

Below **eight** samples it refuses to fit, `predict` returns the training mean, and
`ranks()` reports `False` so that callers say the generation was a random search instead of
printing a ranking over one repeated constant. Quality is reported as cross-validated
Spearman rank correlation, because the Predictor's job is ordering, not pricing.

**This split was made after review feedback and it immediately found a defect.** The first
version trained a single model directly on the scalarised fitness, which put the weighting
inside the surrogate. Reported per objective instead, on the twelve measured networks:

| Objective | Spearman | Permutation null | Margin over null | Excluded from Phase C |
| --- | --- | --- | --- | --- |
| accuracy | **+0.63** [+0.27 … +0.72] | −0.07 [−0.36 … +0.30] | **+0.73**, positive in 20/20 seeds | 0/20 seeds |
| token cost | **−0.53** [−0.80 … −0.38] | −0.17 [−0.42 … +0.03] | **−0.35**, negative in 20/20 seeds | **20/20 seeds** |

Medians over 20 cross-validation seeds, with the per-seed range in brackets; each null is
the median of 12 shuffles. Repeating the sweep at 40 shuffles moves the null medians to
−0.13 and −0.16 and leaves both margin signs and both exclusion counts unchanged.

**Read the last column, not the third.** The null is the noisiest quantity here — a single
seed can put token cost's anywhere from −0.42 to +0.03 — so no point estimate of it is worth
quoting. What is stable is the decision it feeds: token cost loses to its own null under
every seed and every shuffle count tried, and accuracy under none. `make figures`
reproduces the whole table from committed data, no key needed.

**The null column is the point, and the first version of this table did not have it.** A rank
correlation only means something against the baseline the same procedure produces when there
is no signal to find, and on twelve samples that baseline is not automatically zero.
Cross-validation can manufacture negative correlation on its own: hold out a high value, the
training mean drops, the model predicts low. `permutation_null()` measures it by shuffling
the targets and re-running the identical cross-validation.

Two things came out of measuring it. **Accuracy's margin survives it comfortably** — at worst
+0.40 across the seeds tried — though its null is not the settled −0.03 an earlier draft
claimed: it is −0.07 at 12 shuffles and −0.13 at 40, and single seeds reach +0.30, for the tie
reason below. **Token cost's null is about −0.17, so the real effect is −0.35, not −0.53.**
The finding holds: the margin is negative in every seed tried, and the Predictor does order
candidates by cost backwards. It is a third smaller than the raw correlation, and twelve
samples cannot cleanly separate the real part from the artifact.

**Two caveats the tests record.** The null is only trustworthy on an untied objective — a
permutation destroys a relationship only if permuting moves the values, and accuracy takes
just four distinct values across twelve networks, so its null is weak. Token cost is distinct
in all twelve, so its null is sound, and token cost is what the finding is about. And the
spread is wide: the token null itself ranges from −0.42 to +0.03 across seeds. The margin is
the right statistic; it is not a precise one. Full account in
[FINDINGS.md](FINDINGS.md#what-the-predictor-is-exactly).

## It runs as a service, not a batch job

The first version was a script that planned forty evaluations and died when the provider
stopped it. A budget is not an obstacle to a service — it is its rhythm. An hourly
`invocation: "event"` agent (`registries/manifest.hocon`) spends what the budget allows,
writes the population down **after every candidate**, and stops, so an interruption costs at
most the candidate in flight. A preflight refuses to start on a configuration that would
produce wrong numbers.

## The token-cost Predictor, in full

**One of the two predicted objectives does not beat its own null, and is now excluded.**
Token cost cross-validates at −0.53 against a permutation null of about −0.17, so it sits
**roughly 0.35 below the no-signal baseline**, and the Predictor orders candidates by cost
backwards. Phase C used to weight it at full strength anyway. It no longer does: each
generation measures every objective against its own permutation null, and an objective
that loses to that null is held at the population mean, so it stays on the fitness scale
but cannot order anything. Steering a search with a predictor that ranks backwards
looked worse than not predicting that objective at all. The selection ablation says
otherwise on these twelve networks, for no reason yet understood (see the bullet above),
and the gate stays on until that is explained.

**The exclusion is stable even though the number under it is noisy**, and the two have to
be reported separately. Across 20 cross-validation seeds the token margin is negative in
**20 of 20** and the objective is excluded in **20 of 20**, at both 12 and 40 shuffles per
null. The null itself is far less settled: its median moves from −0.17 to −0.16 between
those shuffle counts, and individual seeds range from −0.42 to +0.03. So the margin is a
range — medians **−0.35 and −0.37**, single seeds from −0.12 to −0.70 — and any
single-seed figure is one draw from that spread rather than the result.

The gate is a measurement, not a hardcoded exclusion — the objective returns on its own
the generation it starts predicting — and it only fires where the null was actually
measured, because against an assumed baseline of zero a twelve-sample procedure would
drop objectives for being small-sample rather than for being wrong. Accuracy clears its
null by +0.73 and is untouched, in 20 of 20 seeds. **What this does not do is fix the
prediction.** Thirteen structural features still do not predict what a network will spend,
nothing here says what would, and twelve samples cannot say how much of the −0.35 is real.

## A mislabelled feature, fixed

**One feature was mislabelled; it is fixed, and no conclusion moved.** The model-tier
feature placed a model by its position on the configured ladder, so the workers' cheapest
model read as the most expensive. It is now a property of the model — rung, then release.
The token margin moved from −0.47 to −0.35, accuracy's from +0.63 to +0.73, and every
exclusion verdict stayed the same. Every Predictor figure here is after the fix, and `make
figures` regenerates them all from committed data; the before-and-after table is in
[FINDINGS.md](FINDINGS.md#a-defect-in-the-model-tier-feature-measured-and-fixed).
