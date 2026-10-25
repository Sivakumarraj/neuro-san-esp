---
title: neuro-san-esp — measured agent network
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Talking to a measured agent network

neuro-san can design a multi-agent network from one sentence. It designs **one** and never
measures it — there is no fitness function anywhere in the framework.

This Space serves `2cc4` from [neuro-san-esp](https://github.com/Sivakumarraj/neuro-san-esp),
which adds that missing half: the network that beat the shape neuro-san's designer builds on 100
questions it was never chosen on.

Ask it about **Meridian Logistics** — a company that does not exist. 24 depots, 40 contracts,
124 documents, all invented, so nothing about it is in any model's training data. The only way to
answer is to go and read, which is what makes the score a measurement of the *topology* rather
than of recall.

Questions that appear in the graded task set are **graded live against ground truth** in front of
you.

## Deploying your own

1. Create a Space → **Docker** → blank.
2. Copy this directory's `Dockerfile` and this `README.md` to the Space root, and the repository
   alongside it (or point the Space at the GitHub repo).
3. **Settings → Variables and secrets → New secret**: one provider's key. `OPENAI_API_KEY`
   serves `2cc4` on the models it was judged on. With `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY`
   its models move rung for rung, and the page says that score was not measured there.
   Never put the key in the Dockerfile or the repo — it is baked into every layer if you do.
4. Optional: `ESP_WEB_MAX_QUESTIONS` (default 40) caps how many questions the Space answers per
   UTC day, and `ESP_WEB_PER_CLIENT_HOURLY` (default 10) caps each visitor's share per hour.
   Every answer is several paid model calls, and one careless loop spends the day.

## Limits

- **What is measured.** `2cc4`, with its Researcher on `gpt-5.4-mini` and its Coordinator on
  `gpt-5.4-nano`, answered 94.5% of 100 judge questions asked twice, against 84.0% for the
  designer's shape on `gpt-5.4-nano` (sign test p = 0.0007), with every search returning 10
  documents. On 40 fresh questions it answered 90.0% against 75.0% (p = 0.0215).
  The [README](https://github.com/Sivakumarraj/neuro-san-esp#headline-result) has every figure
  and the logs they come from.
- **Not a pure structure result.** Its one document-reading agent runs a model 3.7× dearer per
  token. Promoting every specialist in the designer's shape bought it 2.5 points on the fresh
  questions, and `2cc4` still beat that control on 7 questions against 1, which on 40 questions
  is not significant (p = 0.0703).
- **It costs more.** 37% fewer tokens, but 94% more dollars a question than the designer's
  shape, which is still the cheapest network measured.
- **One task domain, one mutation.** Every question comes from the same invented company, and
  `2cc4` is one measured mutation of a hand-written seed, not a network the Predictor chose.
- **The Predictor is weak.** Trained on the twelve measurements it picks the best of three
  unseen networks 62% of the time against 33% by chance (`make ablation`), level with a rule
  that picks the network with the most agents.
