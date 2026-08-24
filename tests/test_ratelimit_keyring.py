"""Multi-key rotation: on daily quota, try the next key before retiring model.

A search that stops the moment ONE key exhausts its per-model daily budget
still leaves budget on every other key in the ring. The rule the run learnt
the hard way: a per-day 429 exhausts (key, model), not the model.
"""

from __future__ import annotations

import contextlib
import os

import pytest

from esp.eval import ratelimit


@contextlib.contextmanager
def _env(**pairs):
    old = {k: os.environ.get(k) for k in pairs}
    for k, v in pairs.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_keyring_parses_multi_key_env():
    with _env(GOOGLE_API_KEYS="a,b,c", GOOGLE_API_KEY=None):
        ratelimit.reload_keyring()
        assert ratelimit.keyring() == ["a", "b", "c"]


def test_keyring_falls_back_to_single_key():
    with _env(GOOGLE_API_KEYS=None, GOOGLE_API_KEY="solo"):
        ratelimit.reload_keyring()
        assert ratelimit.keyring() == ["solo"]


def test_keyring_empty_when_no_env():
    with _env(GOOGLE_API_KEYS=None, GOOGLE_API_KEY=None):
        ratelimit.reload_keyring()
        assert ratelimit.keyring() == []


def test_next_key_rotates_from_current():
    with _env(GOOGLE_API_KEYS="a,b,c", GOOGLE_API_KEY=None):
        ratelimit.reload_keyring()
        assert ratelimit.next_key_for("m1", "a") == "b"
        assert ratelimit.next_key_for("m1", "b") == "c"
        assert ratelimit.next_key_for("m1", "c") == "a"


def test_next_key_skips_exhausted():
    with _env(GOOGLE_API_KEYS="a,b,c", GOOGLE_API_KEY=None):
        ratelimit.reload_keyring()
        ratelimit.mark_key_exhausted("b", "m1")
        assert ratelimit.next_key_for("m1", "a") == "c"


def test_next_key_returns_none_when_all_exhausted():
    with _env(GOOGLE_API_KEYS="a,b", GOOGLE_API_KEY=None):
        ratelimit.reload_keyring()
        ratelimit.mark_key_exhausted("a", "m1")
        ratelimit.mark_key_exhausted("b", "m1")
        assert ratelimit.next_key_for("m1", "a") is None


def test_exhaustion_is_per_model():
    with _env(GOOGLE_API_KEYS="a,b", GOOGLE_API_KEY=None):
        ratelimit.reload_keyring()
        ratelimit.mark_key_exhausted("a", "m1")
        # a is still fresh on a different model
        assert ratelimit.next_key_for("m2", "b") == "a"


def test_apply_key_swaps_pydantic_secretstr():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel
    from pydantic.types import SecretStr

    class Fake(BaseModel):
        google_api_key: SecretStr | None = None

    c = Fake(google_api_key=SecretStr("k1"))
    ratelimit._apply_key(c, "k2")
    assert c.google_api_key.get_secret_value() == "k2"


def test_apply_key_no_op_when_same():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel
    from pydantic.types import SecretStr

    class Fake(BaseModel):
        google_api_key: SecretStr | None = None

    c = Fake(google_api_key=SecretStr("k1"))
    ratelimit._apply_key(c, "k1")
    assert c.google_api_key.get_secret_value() == "k1"
