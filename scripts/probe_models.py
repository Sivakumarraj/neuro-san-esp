"""Which models are usable, and which have daily budget left.

Worth running before any search. The free tier caps requests per day *per
model*, so a search that starts on an exhausted model does not fail loudly --
every candidate scores zero and the search concludes that good topologies are
bad. One cheap call per model answers the question for a few seconds of time.

Three outcomes are distinguished, because they need different responses:
  OK          usable now
  QUOTA GONE  usable tomorrow; the cap is per day, not per hour
  BROKEN      never usable -- reachable and in budget, but the agent loop dies
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esp.config import bootstrap
from esp.eval.failover import DAILY_CAPS, LADDER

# Everything worth asking about: the ladder, every model with a recorded cap
# (so a re-probe can tell whether a rejection still holds), and a few known
# names that are in neither. Deduplicated in order -- the list was written by
# hand with `gemini-3.1-flash-lite` in it *and* in LADDER, so every probe run
# spent two calls asking about the same model and printed it twice.
_ALSO_WORTH_ASKING = ["gemini-2.5-flash-lite", "gemini-2.0-flash"]

CANDIDATES: list[str] = list(dict.fromkeys(
    [*LADDER, *DAILY_CAPS, *_ALSO_WORTH_ASKING]))

_NETWORK = """{
    "llm_config": {"model_name": "%s"},
    "max_steps": 6,
    "max_execution_seconds": 45,
    "tools": [{"name": "A",
               "function": {"description": "Answer a question."},
               "instructions": "Reply with the answer alone."}]
}"""


def probe(factory, model: str) -> tuple[str, float]:
    started = time.monotonic()
    answer = ""
    sink = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.hocon"
        path.write_text(_NETWORK % model, encoding="utf-8")
        try:
            session = factory.create_session(str(path))
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                for chunk in session.streaming_chat(
                        {"user_message": {"type": "HUMAN", "text": "What is 6 times 7?"}}):
                    message = (chunk or {}).get("response") or {}
                    kind = getattr(message.get("type"), "name", str(message.get("type")))
                    if (kind in ("AI", "AGENT_FRAMEWORK", "4", "101")
                            and (message.get("text") or "").strip()):
                        answer = message["text"].strip()
        except Exception as exc:
            answer = f"EXC {type(exc).__name__}"

    blob = answer + sink.getvalue()
    elapsed = time.monotonic() - started
    if "RESOURCE_EXHAUSTED" in blob or "429" in blob:
        cap = re.search(r"limit: (\d+), model", blob)
        return f"QUOTA GONE (cap {cap.group(1) if cap else '?'}/day)", elapsed
    if "42" in answer:
        return "OK", elapsed
    return f"BROKEN: {answer[:50]}", elapsed


def main() -> int:
    # neuro-san resolves coded tools against AGENT_TOOL_PATH and refuses to build
    # a session without a usable one. Setting it here rather than relying on the
    # caller matters: without it every model fails identically with a ValueError,
    # and the probe reports "BROKEN" for models that are perfectly fine -- which
    # is precisely the wrong answer from a tool whose job is to tell you what
    # works.
    root = str(Path(__file__).resolve().parent.parent)
    os.environ.setdefault("AGENT_TOOL_PATH", root)
    os.environ.setdefault("PYTHONPATH", root)

    bootstrap()
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set", file=sys.stderr)
        return 2

    warnings.filterwarnings("ignore")
    for noisy in ("neuro_san", "httpx", "httpcore", "urllib3",
                  "langchain", "langchain_google_genai"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    from neuro_san.client.direct_agent_session_factory import DirectAgentSessionFactory

    factory = DirectAgentSessionFactory()          # reads the manifest once
    usable = []
    for model in CANDIDATES:
        verdict, elapsed = probe(factory, model)
        marker = "*" if model in LADDER else " "
        print(f"{marker} {model:34} {elapsed:5.1f}s  {verdict}", flush=True)
        if verdict == "OK":
            usable.append(model)

    print(f"\n{len(usable)} usable now: {', '.join(usable) or 'none'}")
    print("* = in the failover ladder")
    # Non-zero when nothing can run, so `make probe` is usable as a gate.
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
