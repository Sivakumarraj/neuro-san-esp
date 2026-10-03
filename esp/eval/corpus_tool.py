"""The one tool every candidate network gets: search over the fixed corpus.

Deterministic on purpose. Fitness must be a property of the topology, so the
retrieval underneath it cannot be allowed to vary between generations. No
embeddings, no network calls, no model — scoring is a transparent term overlap
that returns the same documents for the same query forever.
"""

from __future__ import annotations

import os
import re
from typing import Any

from neuro_san.interfaces.coded_tool import CodedTool

from esp.eval.world import build_world, documents

_DOCS: dict[str, str] = documents(build_world())
_STOP = {"the", "a", "an", "of", "for", "is", "are", "was", "were", "which",
         "what", "who", "and", "or", "to", "in", "on", "at", "by", "with",
         "that", "this", "it", "its", "answer", "number", "only"}

# Every committed measurement used 3. At 3 a "find every incident that..."
# question cannot be answered from one search: "mis-picked" appears in 8
# documents and depot D05 in 4, so the rest are cut off and the network counts
# short. ESP_SEARCH_RESULTS raises it for a run; results measured at one value
# are never compared with results measured at another.
MAX_RESULTS = int(os.environ.get("ESP_SEARCH_RESULTS", "3"))


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9\-]+", text.lower()) if t not in _STOP]


def search(query: str, max_results: int = MAX_RESULTS) -> list[dict[str, Any]]:
    """Rank documents by how many query terms they contain.

    Identifiers (D03, C-2105, INC-4407) are weighted heavily because they are
    the join keys of a multi-hop question — matching one is far stronger
    evidence than matching a common word.
    """
    query_terms = _terms(query)
    if not query_terms:
        return []

    scored: list[tuple[float, str]] = []
    for name, body in _DOCS.items():
        body_terms = set(_terms(body))
        hits = 0.0
        for term in query_terms:
            if term in body_terms:
                hits += 4.0 if re.fullmatch(r"(d\d{2}|c-\d{4}|inc-\d{4})", term) else 1.0
        if hits:
            scored.append((hits, name))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [{"document": name, "score": hits, "content": _DOCS[name]}
            for hits, name in scored[:max_results]]


class CorpusSearch(CodedTool):
    """Search the Meridian Logistics document store."""

    def invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> Any:
        query = args.get("query") or args.get("search_terms") or ""
        if not query:
            return "Error: no 'query' provided."

        results = search(str(query))
        if not results:
            return "No documents matched that query."

        # Count retrievals so the evaluator can see how hard a topology worked,
        # not just whether it was right.
        sly_data["corpus_calls"] = sly_data.get("corpus_calls", 0) + 1

        return "\n\n".join(
            f"--- {item['document']} ---\n{item['content'].strip()}"
            for item in results
        )
