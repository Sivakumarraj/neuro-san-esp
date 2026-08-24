"""Phase 0 smoke test: run one agent network in-process, no server.

Evaluation is the bottleneck of the whole project, so the candidate runner must
not pay an HTTP round trip. This proves DirectAgentSessionFactory can load a
network straight from a .hocon path and answer a question.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Runnable straight from a clone, without `pip install -e` first. Half the
# scripts here already did this and half did not, so `serve_champion.py` --
# which the README puts in the quick start -- died with
# `ModuleNotFoundError: No module named 'esp'` on a fresh Codespace while
# its neighbours ran fine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuro_san.client.direct_agent_session_factory import DirectAgentSessionFactory
from neuro_san.session.session_invocation_context import SessionInvocationContext  # noqa: F401

from esp.config import bootstrap


def run(hocon_path: str, question: str) -> dict:
    factory = DirectAgentSessionFactory()
    session = factory.create_session(hocon_path)

    request = {
        "user_message": {"type": "HUMAN", "text": question},
        "sly_data": {},
        # MAXIMAL so we can see every agent message, not just the final answer.
        "chat_filter": {"chat_filter_type": "MAXIMAL"},
    }

    started = time.monotonic()
    answer: str = ""
    seen: list[tuple[str, str]] = []
    for result in session.streaming_chat(request):
        message = result.get("response")
        if message is None:
            continue
        # The direct session leaves ChatMessageType as its int value; the HTTP
        # path converts it to the enum name. Accept either.
        kind = str(message.get("type", "?"))
        text = message.get("text") or ""
        seen.append((kind, text[:80]))
        if kind in ("4", "AI", "101", "AGENT_FRAMEWORK") and text:
            answer = text
    elapsed = time.monotonic() - started

    return {"answer": answer, "seconds": round(elapsed, 2), "messages": seen}


if __name__ == "__main__":
    bootstrap()

    # Defaults to the network this repo ships for exactly this purpose, so the
    # smoke test runs with no arguments rather than dying on an IndexError.
    default_hocon = Path(__file__).resolve().parent.parent / "esp/eval/networks/smoke.hocon"
    hocon = sys.argv[1] if len(sys.argv) > 1 else str(default_hocon)
    question = sys.argv[2] if len(sys.argv) > 2 else "What is 2 plus 2?"
    print(json.dumps(run(hocon, question), indent=2))
