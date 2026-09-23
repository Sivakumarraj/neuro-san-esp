<!-- Remove any section that does not apply. -->

## Description

<!-- What this changes, and why. Link the issue if there is one. -->

## Evidence

<!-- What you ran and what it showed. For a change to a number anywhere in the
     documentation, the test that recomputes it. For a change to measurement,
     say which provider and model the new figures came from. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Measurement or analysis (changes a published figure)
- [ ] Refactoring
- [ ] Documentation

## Checklist

- [ ] `make check` passes (lint and the full offline suite)
- [ ] `pymarkdown --config .pymarkdownlint.yaml scan *.md docs/*.md` is clean
- [ ] Tests added or updated for the behaviour changed
- [ ] Every figure in the docs that this touches is recomputed by a test, not typed
- [ ] No key, `.env` or provider output with a key in it is committed

---

By submitting this pull request, I confirm that my contribution is made under the terms of the
project's [Apache 2.0 License](../LICENSE).
