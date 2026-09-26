"""What a network costs in dollars, and a fitness that sees it.

The v1 fitness charges for tokens. Tokens are a fair price only while every
agent runs the same model, and the search's best move breaks exactly that: it
puts a stronger model on the router. The best-measured network used 7% fewer
tokens than the designer's shape and cost 63% more, because a router token on
gemini-3.5-flash is dearer than a worker token on gemini-3.1-flash-lite. A
token count cannot see that; a dollar count can.

Dollars are what neuro-san itself reports for each run (`Evaluation.cost`,
from the provider's own accounting), so a measured network needs no price
table. The table here is for *planning*: pricing a run before it is paid for.
It is dated, because prices move, and every figure derived from it says so.

The v1 fitness in `esp.evolve.loop` is left exactly as it was. Every committed
result was selected under it, and changing it would silently re-rank history.
`fitness_dollars` is what the pool benchmark selects on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# List prices in dollars per million tokens, (input, output). Claude figures
# are Anthropic's first-party API list prices. The Gemini row is not a list
# price: it is what this project's own committed runs cost per million tokens,
# blended, because that is the number a plan should be priced with.
PRICES_AS_OF = "2026-09-26"
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
# Blended $/M tokens measured on committed Gemini runs (0.29 to 0.50 across
# networks, depending on which model the router runs).
GEMINI_MEASURED_BLENDED = 0.45
# Agent traffic is mostly input: documents and instructions go in, short
# answers and tool calls come out.
OUTPUT_SHARE = 0.10

# Dollar fitness. One question on the designer's shape cost about $0.0065 in
# v1; the scale puts a network ten times dearer than that at the cap, which is
# where the token scale of the v1 fitness sat relative to the designer too.
DOLLAR_WEIGHT = 0.06
DOLLAR_SCALE = 0.065
AGENT_WEIGHT = 0.02


def blended(model: str, output_share: float = OUTPUT_SHARE) -> float:
    """Dollars per million tokens for one model, input and output mixed."""
    if model.startswith("gemini"):
        return GEMINI_MEASURED_BLENDED
    try:
        price_in, price_out = PRICES[model]
    except KeyError:
        raise KeyError(f"no price for {model!r}; add it to PRICES "
                       f"(prices as of {PRICES_AS_OF})") from None
    return (1 - output_share) * price_in + output_share * price_out


def estimate(tokens: float, model: str) -> float:
    """Planning estimate of what `tokens` cost on `model`."""
    return tokens / 1e6 * blended(model)


def fitness_dollars(accuracy: float, dollars_per_question: float,
                    agents: int) -> float:
    """Accuracy, less what an answer costs in money, less size.

    The same shape as the v1 fitness with tokens replaced by dollars per
    question, so a stronger router is charged what it actually costs.
    """
    return (accuracy
            - DOLLAR_WEIGHT * min(dollars_per_question / DOLLAR_SCALE, 1.0)
            - AGENT_WEIGHT * (agents / 9.0))


@dataclass(frozen=True)
class Priced:
    """One measured network, in tokens and in dollars."""

    genome_hash: str
    accuracy: float
    tokens: int
    dollars: float
    questions: int
    agents: int

    @property
    def dollars_per_question(self) -> float:
        return self.dollars / self.questions if self.questions else 0.0

    @property
    def dollars_per_million_tokens(self) -> float:
        return self.dollars / self.tokens * 1e6 if self.tokens else 0.0

    def fitness(self) -> float:
        return fitness_dollars(self.accuracy, self.dollars_per_question,
                               self.agents)


def priced(cache_dir: Path) -> list[Priced]:
    """Every cached measurement that recorded what it cost."""
    found = []
    for path in sorted(Path(cache_dir).glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            found.append(Priced(raw["genome_hash"], float(raw["accuracy"]),
                                int(raw["tokens"]), float(raw["cost"]),
                                len(raw["results"]), int(raw["agents"])))
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            continue
    return [p for p in found if p.tokens > 0 and p.dollars > 0]
