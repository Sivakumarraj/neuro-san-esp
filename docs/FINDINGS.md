# Findings

The long-form record: what the measurements said, what running the
system changed about them, and the prior art this sits next to. The README
keeps the summary; this keeps the argument.

## What was measured

Three seed topologies, same 17 tasks, same model:

| Topology | Accuracy | Of what it answered | Never finished | Tokens | Agents |
|---|---|---|---|---|---|
| `designer_shaped` (the shape the designer produces) | 0.82 | 0.93 (14/15) | 2 | **278,532** | 4 |
| `flat_pair` | 0.82 | 0.88 (14/16) | 1 | 326,364 | 3 |
| `solo` (one agent, one tool) | 0.82 | **1.00 (14/14)** | 3 | **396,378** | 1 |

**The 42% spread in cost is real, and part of it is the cost of failing.** Topology
changed what answering cost, and that is precisely the measurement neuro-san cannot make
today. It is only visible because of the `TOKEN_SCALE` fix below — under the saturated
scale all three clipped to the same penalty and scored as indistinguishable.

The qualifier matters. The three tasks below are full-corpus aggregations — *"across all
forty contracts, which has the highest…"* — and `CorpusSearch` returns three documents a
query, so answering one means about fourteen successive searches. No topology managed it.
They are not free to fail: they consumed **44% of `designer_shaped`'s wall-clock, 34% of
`flat_pair`'s and 60% of `solo`'s**, on 18% of the tasks. Whatever a topology spends
looping on a question it cannot answer is charged to it as cost, so the spread measures
efficiency *and* failure mode together. Separating them needs a re-measurement this
project has not been able to buy.

**The identical accuracy is not real, and this table used to say so.** It read
*"identical accuracy — topology did not change what these networks could answer"*, over a
row of three 0.82s and the words *"zero errors"*. All three do score 0.8235, because all
three failed exactly three of the seventeen tasks. They failed them for three different
reasons:

- `solo` hit neuro-san's **recursion cap** on all three, and never produced an answer to
  any of them. Its accuracy on those questions is unknown, not zero.
- `designer_shaped` **timed out** on two at `max_execution_seconds=600` and got one
  genuinely wrong.
- `flat_pair` timed out on one and got two genuinely wrong.

neuro-san returns a timeout and a blown recursion cap as ordinary answer strings, so they
reached the scorer, compared false against the expected answer, and were cached as wrong
answers. The runner already refuses to cache an evaluation poisoned by a provider quota —
with a comment warning about *"a plausible-looking partial score that gets cached
forever"* — but the guard did not cover the two failures that actually happened.

Split the two apart and the ordering inverts: on the questions each topology actually
finished, `solo` got everything right and `designer_shaped` did not. What topology changed
here was **how often the network finished at all**, which is a different and more
interesting finding than the one this table used to report. It is also a weaker one: three
measurements, and six of the fifty-one task runs are not measurements of anything.

`tests/test_task_outcomes.py` pins this, including on the committed fixtures, so it cannot
quietly stop being true.

All three were measured on the same model, **`gemini-3.1-flash-lite`**. That is not
incidental — the model is part of the genome hash, so a fitness compared across models
would not mean anything, and the comparison above is only a comparison because one model
produced all of it.

**No search was run.** A candidate costs about 165 provider requests across the task set,
and the free tier caps requests *per day, per model* — 500 a day here, so one day's
allowance buys three candidates. That is enough to measure the seed topologies against
each other and not enough to evolve. The budget was spent before a generation completed.

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

The report states this on its first page rather than reporting a search that lost. Those
are different claims, and the weaker one is the true one.

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

Automated agent architecture search is a crowded, fast-moving field: ADAS (meta-agent
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
