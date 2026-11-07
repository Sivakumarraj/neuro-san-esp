# Results

Every measured result behind the [README](../README.md), with the detail the README leaves out.
Every figure on this page is recomputed from the committed logs by a test
(`tests/test_select_and_judge.py`, `tests/test_route.py`, `tests/test_docs.py`), so this page
cannot drift from the runs it describes. [FINDINGS.md](FINDINGS.md) has the failure analysis,
and [DESIGN.md](DESIGN.md) how each part works.

## The held-out head-to-head

On questions it was never chosen on, a **two-agent network, with the stronger model on
the one agent that reads the documents, scored 94.5% against 84.0%** for a hand-written
four-agent network in the shape neuro-san's designer builds, using 37% fewer tokens but costing
more.

| 100 judge questions × 2 repeats, search returns 10 documents | `2cc4`, Researcher on `gpt-5.4-mini` | Designer's shape, all `gpt-5.4-nano` |
| --- | --- | --- |
| Accuracy | **94.5%** (189/200; 91, then 98) | 84.0% (168/200; 82, then 86) |
| Tokens per question | **3,250** (−37%) | 5,153 |
| Cost per question | $0.00263 (+94%) | **$0.00135** |
| Cost per correct answer | $0.00278 (+72%) | **$0.00161** |
| Agents | **2** | 4 |

- **Significant.** Right more often on 17 questions against 2: sign test p = 0.0007.
- **Cost is neuro-san's own count**: every call's input and output tokens at the
  provider's list prices, with no cache discount. The dollar cap ran on a more cautious
  figure per chunk (the larger of that count and a blended estimate), on which the gap is
  +71% a question and +52% a correct answer. Earlier versions of this page quoted those.
- **Token and dollar figures were corrected on 10 October 2026.** neuro-san reports a run's
  usage as a total and again per model, and the runner added both, so every token and
  dollar figure published before then was twice the real one. Accuracy, every ratio and
  every fitness value are unchanged. The committed logs are not rewritten:
  `esp/eval/accounting.py` halves an old record as it is read.
- **Chosen on select, judged on judge.** Every change was decided on the 60
  `meridian-select` questions (`2cc4` with the stronger Researcher scored 57/60 there).
  The judge questions come from the other half of the company and were never used to
  choose. They are not fresh: J001 to J100 were also asked in the earlier runs below,
  because the judge half has no unused aggregate questions left to draw.
- **Not a pure structure result.** The winner promotes one agent to a model 3.7× dearer
  per token; the designer's shape was not measured here with its specialists promoted. The
  [replication](#replication-on-40-fresh-questions) measured that control.
- **The benchmark changed.** The search tool returned 3 documents in every earlier run.
  At 3, a question such as "how many incidents were caused by a mis-picked pallet" cannot
  be answered: the word is in 8 documents. With 10 (`ESP_SEARCH_RESULTS=10`), the
  designer's shape rose from 69% to 84% on the same questions. Numbers at 3 and at 10
  are never compared with each other.
- **Where `2cc4` came from.** It is the hand-written `flat_pair` seed with its Arithmetic
  agent removed by the `remove_agent` operator: one mutation, chosen by measurement on
  select. The Predictor was not used in any paid run.

Before the search fix, on all-`gpt-5.4-nano` networks and the same 100 judge questions,
`2cc4` scored 153/200 (76.5%) against the designer's 138/200 (69.0%) with 17% fewer
tokens (p = 0.052). On 100 questions of four kinds no network had been asked before
(P001 to P100: temporal, filtered, compare, unanswerable) it scored 77/100 against
68/100 (p = 0.108), but used 7% more tokens there.

Every measured chunk is committed in `results/headtohead/paid-2026-10/`, and
`scripts/select_and_judge.py` reproduces the networks, the search and the summary
([details](#reproducing-the-head-to-head)). `tests/test_select_and_judge.py` recomputes
every figure in this section from those logs.

## Replication on 40 fresh questions

`make replicate` asked 40 questions no network had been asked (F01 to F40, drawn from the
judge half by `esp/eval/fresh.py` with the judge set's mix), twice each at 10 documents, of
the designer's shape, `2cc4` with its Researcher promoted, and the designer's shape with all
three specialists promoted to `gpt-5.4-mini`. The three ran at the same time, each under its
own dollar cap, and every one of the 240 question-runs finished.

| 40 fresh questions × 2 | `2cc4` | Control: designer's shape, specialists on mini | Designer's shape, all nano |
| --- | --- | --- | --- |
| Accuracy | **90.0%** (72/80; 37, then 35) | 77.5% (62/80; 30, then 32) | 75.0% (60/80; 31, then 29) |
| Tokens per question | **3,342** | 4,493 | 5,773 |
| Model calls per question | **5.62** | 7.39 | 7.47 |
| Cost per question | $0.00278 | $0.00340 | **$0.00153** |
| Cost per correct answer | $0.00309 | $0.00438 | **$0.00204** |

| Kind of question (runs) | `2cc4` | Control | Designer's shape |
| --- | --- | --- | --- |
| Aggregate (32) | **29** | 20 | 23 |
| 1-, 2- and 3-document join (24) | 24 | 24 | 24 |
| 4-document join (8) | **4** | 2 | 0 |
| 6-document join (8) | 7 | 8 | 8 |
| 9-document join (8) | 8 | 8 | 5 |

- **Against the designer's shape**: better on 9 questions, worse on 1 (F12), sign test
  p = 0.0215. Tokens −42%, model calls −25%, cost +81% a question and +51% a correct answer
  by neuro-san's own count (+60% and +33% on the cap's more cautious count).
- **Against the control**: better on 7 questions, worse on 1 (F12), p = 0.0703. Tokens −26%,
  model calls −24%, cost −18% a question and −29% a correct answer. Promoting every
  specialist raised the designer's shape by only 2.5 points, and its aggregates fell from 23
  to 20, so the stronger model alone does not explain the headline.
- **Harder than judge.** Both networks scored lower than on J001 to J100, mostly on the
  four-document joins, which compare two computed penalties: the all-nano designer got none
  of its 8 right, against 60% on judge.
- **One question went the other way.** F12, a six-document join, was answered right in both
  repeats by both designer networks and in one of two by `2cc4`.

Every chunk is committed in `results/replication/fresh-2026-10/1-fresh/`, with
`summary.json` beside it, and `tests/test_replication.py` recomputes this section from them.

## Same model, half the agents

The test [SAME-MODEL.md](SAME-MODEL.md) fixed, and pushed, before the run: `2cc4` (genome
`2cc4231b7d3fb81d`) against the designer's shape (`0378ca26cd9fe977`), every agent of both on
`gpt-5.4-nano`, 10 documents a search, each question twice, both networks at the same hour on
7 November 2026. It passes only if `2cc4` is more accurate, uses fewer tokens, costs less, has
fewer agents and scores a higher fitness. It does not pass.

| Judge, 100 questions × 2 (the test) | `2cc4` | Designer's shape |
| --- | --- | --- |
| Accuracy | 82.0% (164/200) | **84.0%** (168/200) |
| Tokens per question | **4,596** | 5,385 |
| Cost per question | **$0.00119** | $0.00139 |
| Cost per correct answer | **$0.00145** | $0.00166 |
| Agents | **2** | 4 |
| Token fitness | +0.7999 | **+0.8128** |
| Dollar fitness | +0.8134 | **+0.8285** |

| Fresh, 40 questions × 2 (the replication) | `2cc4` | Designer's shape |
| --- | --- | --- |
| Accuracy | **80.0%** (64/80) | 70.0% (56/80) |
| Tokens per question | **5,560** | 5,700 |
| Cost per question | **$0.00142** | $0.00148 |
| Cost per correct answer | **$0.00177** | $0.00212 |
| Agents | **2** | 4 |
| Token fitness | **+0.7766** | +0.6717 |
| Dollar fitness | **+0.7929** | +0.6884 |

- **Judge: cheaper, not more accurate.** Tokens −14.6%, lower in 17 of 25 chunks (Wilcoxon
  p = 0.027); dollars −14.6%, lower in 18 (p = 0.0087). Accuracy −2.0 points with a 95%
  interval of −7.5 to +3.5: right more often on 8 questions against 12, sign test p = 0.5034.
  Accuracy and fitness fail; tokens, cost and agents pass.
- **Fresh: more accurate, not cheaper, neither significantly.** Accuracy +10.0 points (95%
  interval −3.8 to +23.8; 11 questions against 5, p = 0.2101); tokens −2.5% (p = 0.92) and
  dollars −4.3% (p = 0.85). Fresh was the replication, run beside the test and partly seen
  while the plan was written, so it cannot rescue the test.
- **So the shape is a saving, not a gain.** With the model held equal, two agents answer about
  as well as four for about 15% less on judge. The headline's accuracy comes with the stronger
  model on the Researcher.
- **The same networks, a fortnight apart.** The designer's shape scored 168/200 on judge on
  7 November and 168/200 in October; on fresh it scored 56/80 against 60/80. `2cc4` with its
  Researcher promoted, run on fresh the same day (`results/structure/fresh-nano-2026-11/evolved.jsonl`),
  scored 70/80 at 3,148 tokens a question against 72/80 at 3,342 in October.
- **Cost** is neuro-san's own count at list prices, as everywhere on this page. Tokens and
  dollars are tested over the four-question chunks, each chunk's two repeats averaged, because
  the provider reports usage per chunk and not per question.

`make same-model` recomputes both tables and every test from the logs in `results/structure/`,
`tests/test_same_model.py` holds this page to them, and `make same-model-run GO=1` repeats the
run (about $0.60 for the pair on judge).

## Choosing the Researcher's model per question

The test [MODEL-ROUTER.md](MODEL-ROUTER.md) fixed: `2cc4` with its Researcher on `gpt-5.4-nano`
(`2cc4231b7d3fb81d`) or on `gpt-5.4-mini` (`47322479eb9cba1e`), chosen per question by
`esp.route.Router` fitted on select, at no more cost a question than the designer's shape
there. The same five pass marks as the same-model test. It does not pass.

| Select, 60 questions, stage 7 | `2cc4`, nano | `2cc4`, Researcher on mini | Designer's shape |
| --- | --- | --- | --- |
| Accuracy | 86.7% | 95.0% | 80.0% |
| Cost per question | $0.001263 | $0.00279 | $0.001367 |

- **The budget is 4 questions of 60.** Sending more than 4 up costs more a question than the
  designer's shape. Mini beat nano on 6 select questions and lost on 1, and the Predictor's
  single top-ranked question is one mini got wrong; its next group is 14 questions. No
  threshold within budget scores better than sending nothing up, so the router sends nothing
  up, and on judge it is `2cc4` on nano: 82.0% against 84.0%, tokens and dollars 14.6% lower.
  Accuracy and fitness fail; tokens, cost and agents pass.
- **The strong action, measured again on judge the same day** (`results/router/judge-2026-11/`),
  scored 95.5% (191/200) at 3,020 tokens and $0.002431 a question, against 94.5% (189/200) at
  3,250 tokens in October.

Not a test: the routing that knows every answer, sending up the questions mini wins most while
the cost a question stays at or below the designer's shape's. It is computed on the questions
it is reported on, and bounds what any Predictor could do with these two networks.

| Knowing every answer, within the designer's cost | Sent up | Accuracy | Cost per question | Against the designer's shape |
| --- | --- | --- | --- | --- |
| Judge, mini measured 7 November | 17 | 94.5% | $0.001384 | better on 18, worse on 1, p = 0.0001 |
| Judge, mini measured in October | 15 | 93.0% | $0.001387 | better on 16, worse on 2, p = 0.0013 |
| Fresh | 3 | 87.5% | $0.001481 | better on 12, worse on 2, p = 0.0129 |

So the two networks leave room to beat the designer's shape on accuracy at no more cost a
question, and the question-text Predictor does not find it. `make model-router` recomputes every
figure here from the logs, and `tests/test_model_router.py` holds this page to them.

## Reproducing the head-to-head

Put your key in `.env` (gitignored):

```bash
OPENAI_API_KEY=sk-...your-key...
ESP_PROVIDER=openai
ESP_DEFAULT_MODEL=gpt-5.4-nano
ESP_MODEL_TIERS=gpt-5.4-nano,gpt-5.4-mini
ESP_PIN_MODELS=1          # a failing model stops the run instead of being swapped
```

```bash
python scripts/select_and_judge.py networks            # every network and its hash; $0
python scripts/select_and_judge.py mutants             # the search's 8 children of flat
python scripts/select_and_judge.py measure 2cc4 --suite judge --out runs/r1 --go &
python scripts/select_and_judge.py measure designer --suite judge --out runs/r1 --go
python scripts/select_and_judge.py summary results/headtohead/paid-2026-10
```

For the headline run add `ESP_SEARCH_RESULTS=10` to `.env`, and run the evolved network as
`measure 2cc4 --promote Researcher --suite judge --repeats 2`. `--suite judge-plus` asks
P001 to P100. Both networks run at the same time, each under its own dollar cap, so the
pair can spend up to twice the cap. A stopped run resumes from its log, and refuses a log
measured on other questions or at another search width.

| Stage | Measured | Outcome |
| --- | --- | --- |
| 1 | `make headtohead GO=1`: the designer against the Gemini champion | 68 / 100 each (the second repeat stopped after 12 question-runs); the champion cost 2.1× per question |
| 2, 3 | Six committed networks and the designer on select (four found by the Gemini search, two hand-written seeds); the best on judge × 2 | `flat` (the `flat_pair` seed): 77.5% against 67.5% (p = 0.011), equal tokens |
| 4, 5 | One search generation from `flat` (8 children, four stopped short of 60) on select; the best on judge × 2 | `2cc4`: 76.5% against 69.0% (p = 0.052), 17% fewer tokens |
| 6 | `2cc4` and the designer on P001 to P100, once | 77 against 68 (p = 0.108), 7% more tokens |
| 7 | Search fixed to 10 documents; three networks and `2cc4` with its Researcher promoted, on select | 57, 53, 52 and 48 of 60 |
| 8 | The best of stage 7 and the designer on judge × 2 | the headline: 94.5% against 84.0% (p = 0.0007) |

The designer scored 67 to 70 of 100 in every run at 3 documents, so about ±2 questions is
run-to-run noise. Stages are numbered as their logs are, and stage 0 was a four-question
check. By neuro-san's own count, stages 0 to 5 cost $2.31 for 1,888 question-runs and stages
6 to 8 $1.42 more ($2.59 and $1.55 on the cap's basis).
[The paid runs explained](neuro-san-esp-Paid-Runs.pdf) walks through stages 0 to 5; it
was written before stages 6 to 8, before the logs were committed and before the token
correction, so its token and dollar figures are twice the real ones.

## Pool benchmark

One search per method is one sample of that method. The pool benchmark measures 120 networks
once on the select questions, compares search strategies over that pool in hundreds of free
replicate searches, tests each against a permutation null (Holm-adjusted), and judges the
winners on the judge questions.

The full run is 9,000 question-runs: about $61 to $125 if every agent ran
`claude-haiku-4-5`, $123 to $251 on `claude-sonnet-5`, and $20 to $40 on
`gemini-3.1-flash-lite`, at 4,871 to 9,946 tokens a question, the range committed runs
have cost. The rehearsal takes about 30 minutes on four cores.

```bash
make pool                # prints the plan and its price; spends nothing
make pool REHEARSE=1     # the whole run against a simulated provider, for $0
make pool GO=1           # measures and judges, and resumes from the cache if stopped
```

## Results of the search on the 17 built-in questions

The first search ran on Gemini. **Twelve networks measured on real model calls, 17 questions
each**, all on `gemini-3.1-flash-lite`:

| | Accuracy | Tokens | Agents | Fitness |
| --- | --- | --- | --- | --- |
| `seed:designer_shaped` — the shape the designer produces | 0.8235 | 192,640 | 4 | 0.7761 |
| `seed:flat_pair` | 0.8235 | 158,037 | 3 | 0.7852 |
| `seed:solo` — one agent, one tool | 0.8235 | 188,858 | 1 | 0.7835 |
| `mut:reassign_model` — cheapest win | 0.8824 | 130,026 | 5 | 0.8453 |
| **`mut:reassign_model` — best measured** | **0.9412** | 179,800 | 5 | **0.8941** |

- **Seventeen questions cannot rank individual networks**, which is why the headline result
  is judged on held-out questions. Selecting on half and judging on the other half, the
  searched winner beats the designer's shape in 90% of 200 splits, but a random network
  from the population does so 85.5% of the time (`make holdout`). On 24 held-out bank
  questions the evolved network answered 21 against the designer's 23 (`make bank-report`).
- **That champion's gain was a stronger router model.** Re-measured on held-out questions
  it tied the designer at 2.1× the cost, so the search was re-run for structure instead.
- **The select set leaves room to rank.** The designer's shape scored 16 of 20 (80%) on its
  first 20 questions, against 96% on the bank.
- **The Predictor beats chance offline.** At nine measurements its rank correlation was
  **−0.333**; trained on nine of the twelve it picks the best of three unseen networks 62%
  of the time against 33% by chance (`make ablation`), level with a rule that picks the
  network with the most agents.

Every number above is recomputed from committed data by a test.
[FINDINGS.md](FINDINGS.md) has the full measurements and the failure analysis.
