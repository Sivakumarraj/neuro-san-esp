"""Exact arithmetic as a neuro-san CodedTool.

Most of the numbers the networks get wrong are sums and products of figures they
had already found: on the judge questions, 13 of the all-nano `2cc4`'s 36 wrong
runs are totals no subset of the true records adds up to, which a dropped record
cannot explain (docs/ALL-FOUR.md). A model adding seven figures in its head
slips; a calculator does not.

The expression is parsed, never evaluated: `ast` reads it, and only numbers,
`+ - * /`, unary signs and parentheses are walked. Names, calls, attributes,
subscripts and powers are refused before anything is computed, so a model cannot
reach Python through this tool however it phrases the request. Size and the
magnitude of every intermediate result are bounded.

Deterministic and microseconds long, so like `CorpusSearch` it runs on neuro-san's
event loop rather than in a thread.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from neuro_san.interfaces.coded_tool import CodedTool

# Longer than any expression a question here needs, short enough that a model
# pasting a document into the tool is refused rather than parsed.
MAX_CHARS = 400
# Every committed answer is under ten billion; a result past this is a mistake.
MAX_MAGNITUDE = 1e15

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
# Signs a model writes that Python does not read. Commas are thousands
# separators here: no expression this tool accepts has another use for one.
_SPELLINGS = {",": "", "\N{MULTIPLICATION SIGN}": "*", "\N{DIVISION SIGN}": "/",
              "\N{MINUS SIGN}": "-", "\N{EN DASH}": "-"}


class CalculationError(ValueError):
    """An expression this tool will not, or cannot, work out."""


def _walk(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        result = float(node.value)
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _walk(node.left), _walk(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalculationError("division by zero")
        result = _BINARY[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        result = _UNARY[type(node.op)](_walk(node.operand))
    else:
        raise CalculationError("only numbers, + - * / and parentheses are allowed")
    if abs(result) > MAX_MAGNITUDE:
        raise CalculationError("the result is too large to be an answer here")
    return result


def evaluate(expression: str) -> float:
    """The value of an arithmetic expression, or CalculationError."""
    text = str(expression or "")
    for written, read in _SPELLINGS.items():
        text = text.replace(written, read)
    text = text.strip()
    if not text:
        raise CalculationError("no expression given")
    if len(text) > MAX_CHARS:
        raise CalculationError(f"an expression is at most {MAX_CHARS} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise CalculationError(f"not an arithmetic expression: {text[:60]}") from exc
    return _walk(tree.body)


def shown(value: float) -> str:
    """A result the way the questions want it: no separators, no trailing zeros."""
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


class Calculator(CodedTool):
    """Work out an arithmetic expression exactly."""

    def invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> str:
        sly_data["calculator_calls"] = sly_data.get("calculator_calls", 0) + 1
        try:
            return shown(evaluate(args.get("expression", "")))
        except CalculationError as exc:
            return f"Error: {exc}."

    async def async_invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> str:
        """What neuro-san calls first; the work is microseconds, so it runs here."""
        return self.invoke(args, sly_data)
