# Security

This is an evaluation harness. It runs language-model agents that choose their
own tool calls, over a corpus, on a schedule someone else sets. The risks are
mostly about blast radius and about what gets recorded.

## What a candidate network can reach

**One tool: `CorpusSearch`.** It reads a fixed, synthetic, in-memory corpus
built at import from `esp/eval/world.py`. It takes a query string and returns
documents. There is no file path in its signature, no URL, no shell, and no
write path — a mutated network cannot acquire capabilities its operators do not
grant, because `toggle_search` only flips whether an agent may call the one tool
that exists.

**No network egress from a candidate.** Retrieval is deterministic term overlap
over an in-process dict. No embeddings service, no HTTP. The only outbound
traffic in the whole system is the model provider call.

**Deliberately synthetic data.** Nothing in Meridian Logistics is real: no real
company, person, address or contract appears in the corpus or the task set. A
prompt injected into a document could not exfiltrate anything of value even if
an agent obeyed it, because there is nothing of value there.

## Credentials

`GOOGLE_API_KEY` is read from the environment and never written to disk, never
logged, and never placed in a genome, a HOCON file, a cache entry or a report.
The repository contains no key material; `.gitignore` excludes the cache and
generated networks.

Rotate any key that has been pasted into a chat window, an issue or a log,
whether or not it was used.

## What is recorded

Cache entries under `.esp-cache/` hold a candidate's accuracy, token count,
timings, and the **first 400 characters of each answer**. Answers are model
output about a fictional corpus, but they are model output: treat the cache as
untrusted text, not as something to render unescaped.

Evaluation results are keyed by genome hash. The hash covers the network's
structure, instructions and models — not its answers — so a cache entry cannot
be used to recover anything about the environment it ran in.

## Failure modes that matter more than they look

**A quota failure is not a bad topology.** If the provider refuses, the
candidate scores zero, and an unguarded harness will conclude the network was
bad and select against it. `esp/eval/runner.py` refuses to cache any evaluation
where a task hit a provider quota, and `esp/eval/failover.py` moves to another
model rather than recording the zero. This is the single most important
correctness property in the repository.

**Timeouts are bounded, not open.** `MAX_EXECUTION_SECONDS` and `MAX_STEPS` cap
what one agent can spend. They exist to stop a runaway loop, and the time budget
is deliberately larger than the model needs because it must also cover queueing
behind the rate limiter.

## Reporting

Open an issue. There is no production deployment and no user data here, so
please just describe the problem in the open.
