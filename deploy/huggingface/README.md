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

This Space serves the best-measured topology from
[neuro-san-esp](https://github.com/Sivakumarraj/neuro-san-esp), which adds that missing half.

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
3. **Settings → Variables and secrets → New secret**: `GOOGLE_API_KEY`.
   Never put the key in the Dockerfile or the repo — it is baked into every layer if you do.
4. Optional: `ESP_WEB_MAX_QUESTIONS` (default 40) caps how many questions one deployment answers.
   The free provider tier is 500 requests a day and one careless loop spends all of it.

## Honest limits

- **No evolved candidate has beaten the baseline.** The search has not completed one.
- **The surrogate has not trained.** It needs 8 real measurements.
- Answers take 30–60 seconds. Multi-hop retrieval across four documents is genuinely that slow,
  and a spinner that lies about it would be worse.
