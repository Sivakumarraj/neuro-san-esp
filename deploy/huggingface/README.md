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
4. Optional: `ESP_WEB_MAX_QUESTIONS` (default 40) caps how many questions the Space answers per
   UTC day, and `ESP_WEB_PER_CLIENT_HOURLY` (default 10) caps each visitor's share per hour.
   The free provider tier is 500 requests a day and one careless loop spends all of it.

## Limits

- **What is measured.** Twelve networks, each on 17 multi-hop questions, all on
  gemini-3.1-flash-lite. The served network scored 16 of 17 against 14 of 17 for the shape
  neuro-san's designer produces. Seventeen questions cannot rank individual networks; what
  holds on held-out questions is that the searched winner beats the designer's shape in 90%
  of splits (`make holdout`). On 24 genuinely new questions, the one evolved network measured
  so far answered 21 against the designer's 23 (`make bank-report`), so the advantage has not
  reproduced out of sample. [docs/FINDINGS.md](../../docs/FINDINGS.md) has the full account.
- **The Predictor is weak.** Trained on the twelve measurements it picks the best of three
  unseen networks 62% of the time against 33% by chance (`make ablation`). Its token-cost
  model ranks backwards and is excluded from the fitness it ranks with.
- **Free-tier quota.** The served network's router runs on a stronger Gemini model than its
  workers, and Google's free tier allows that model about 20 requests a day. A free key
  answers a handful of questions a day; a paid key has no such cap.
- Answers take 30–60 seconds. Multi-hop retrieval across four documents is genuinely that slow,
  and a spinner that lies about it would be worse.
