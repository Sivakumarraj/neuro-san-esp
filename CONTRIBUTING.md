# Contributing

Thanks for looking. This repository measures agent networks and publishes what the
measurements say, including where they fall short. Contributions are held to the same rule
the code is: **a claim is only as good as the check that would catch it being wrong.**

## Set up

Python 3.12 or newer.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make validate
```

`make validate` is the gate: run it before every commit. It runs what CI runs (lint over
the code and the docs, the whole suite, a real offline search over the committed
measurements) and puts every registry through neuro-san's own HOCON validator as well. The
suite is offline by design. No test calls a model provider, and none needs a key, so it runs the
same on every machine and never spends money.

The docs are linted with the same markdown linter neuro-san uses:

```bash
pymarkdown --config .pymarkdownlint.yaml scan *.md docs/*.md
```

## Before you open a pull request

- **Run the offline half end to end.** `make offline`, `make holdout` and `make null-sweep`
  all run on the twelve committed measurements with no key.
- **Run the live path if you touched it.** `make smoke` puts four real questions to the
  champion on the key in your `.env`. It costs a few dozen model calls.
- **Fill in the pull request template.** It asks for the evidence, not a summary of it.

## The rules this repository keeps

**Every number in the documentation is recomputed by a test.** The README's results table,
the measurement counts, the surrogate figures and the held-out split are all checked against
the committed data in `tests/fixtures/cache/` and `results/history.json`. If you change a
figure, change or add the test that recomputes it. If you cannot write that test, the
figure does not belong in the docs yet.

**Report ranges, not lucky draws.** Anything cross-validated on twelve samples moves with
the fold seed. Quote a range over seeds (`scripts/null_sweep.py` shows how), and quote a
decision, such as "excluded in 20 of 20 seeds", in preference to the noisy number behind it.

**Measurements do not cross providers.** The model is part of every genome hash, so a fitness
measured on Gemini does not describe the same network on Claude. Do not mix them in one
population, and do not compare figures across providers without saying so. The code refuses
to do either. Keep it that way.

**Never cache a measurement a provider failure produced.** A missing key, a spent quota or a
busy API has to fail loudly or be retried. It must never be scored as a wrong answer, because
the cache keeps it forever and the search learns that a good topology is bad. Several guards
in `esp/eval/runner.py` and `esp/eval/ratelimit.py` exist because this happened.

**Say what did not work.** A limitation found and published is worth more than one fixed
quietly. Each has its place in the README's *Limitations*, in `docs/FINDINGS.md` or in a
test that fails if it stops being true.

## Commit messages

One change per commit. The subject line is a conventional prefix (`fix:`, `feat:`, `docs:`,
`test:`, `build:`) followed by what is now true, in the present tense. The body explains why:
what was wrong, how it was found, and what evidence shows it is fixed.

## Secrets

Keys go in `.env`, which is gitignored. A test scans every tracked file for
credential-shaped strings and fails the build if it finds one. If a key is ever exposed in
a chat, an issue or a log, rotate it; do not only delete the message.

## Conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
