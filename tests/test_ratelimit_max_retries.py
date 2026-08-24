"""max_retries=0 on the client, so the keyring rotates on the first 429.

Without this, langchain's internal tenacity loop retries a per-day 429 six
times before letting the exception propagate to the wrapper -- ~3 minutes per
call spent on a quota that will not clear inside the run, during which the
next key sits idle. Setting it to zero surfaces the 429 immediately, my
rotation fires, and per-minute pacing is still handled by the bucket.
"""

from __future__ import annotations

import pytest

from esp.eval import ratelimit


def test_apply_model_zeros_max_retries():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Fake(BaseModel):
        model: str = "start"
        max_retries: int = 6

    c = Fake()
    ratelimit._apply_model(c, "start")  # same model, still forces max_retries=0
    assert c.max_retries == 0


def test_apply_model_is_idempotent():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Fake(BaseModel):
        model: str = "gemini-x"
        max_retries: int = 0

    c = Fake()
    before = c.model_dump()
    ratelimit._apply_model(c, "gemini-x")
    assert c.model_dump() == before
