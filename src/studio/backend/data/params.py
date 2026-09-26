"""
Parameters: flattening of raw experiment JSON, the search grammar, and schedule curves.

Search grammar: `term (AND term)*`. A term is either `path_suffix op value` with
`op` in `=, !=, >, <, >=, <=, ~`, or free text. Whitespace-separated terms are also combined with AND.
"""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .extract import CLASS_KEY, as_int

ParamKind = Literal["object", "schedule", "array", "number", "string", "boolean", "null"]
Op = Literal["=", "!=", ">", "<", ">=", "<=", "~"]
CURVE_POINTS = 64


@dataclass
class ParamRow:
    path: str
    key: str
    depth: int
    kind: ParamKind
    value: Any
    cls: str | None = None
    curve: dict[str, list[float]] | None = None

    def to_json(self) -> dict[str, Any]:
        """The API's `ParamRow` (non-finite floats become null). @ai-generated"""
        value = self.value
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        return {
            "path": self.path,
            "key": self.key,
            "depth": self.depth,
            "kind": self.kind,
            "value": value,
            "cls": self.cls,
            "curve": self.curve,
        }


# ---------------------------------------------------------------- Schedules


def _num(node: dict, key: str) -> float | None:
    """@ai-generated"""
    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def schedule_value(node: Any, t: float) -> float | None:
    """
    Value at time step `t` of a serialized schedule, mirroring `marl.utils.schedule`
    (Linear, Exp, Constant and Rounded schedules). None for unknown or malformed schedules.

    @ai-generated
    """
    if not isinstance(node, dict):
        return None
    cls = node.get(CLASS_KEY)
    if cls == "RoundedSchedule":
        inner = schedule_value(node.get("schedule"), t)
        digits = as_int(node.get("n_digits", 0))
        return None if inner is None or digits is None else round(inner, digits)
    start = _num(node, "start_value")
    if cls == "ConstantSchedule":
        return start
    end, n = _num(node, "end_value"), _num(node, "n_steps")
    if start is None or end is None or n is None:
        return None
    if t >= n:
        return end
    if cls == "LinearSchedule":
        return (end - start) / n * t + start
    if cls == "ExpSchedule":
        if n <= 1 or start == 0:
            return start
        try:
            return start * (end / start) ** (t / (n - 1))
        except (ZeroDivisionError, OverflowError, ValueError):
            return None
    return None


def schedule_curve(node: Any, horizon: int | None = None, points: int = CURVE_POINTS) -> dict[str, list[float]] | None:
    """
    Sampled curve `{x, y}` of a schedule over `[0, max(n_steps, horizon)]`. None if unknown.

    @ai-generated
    """
    if schedule_value(node, 0) is None:
        return None
    n = _schedule_steps(node)
    end = max(n or 0, horizon or 0) or 1
    xs = sorted({round(end * i / (points - 1)) for i in range(points)} | ({n} if n and n <= end else set()))
    ys = [schedule_value(node, x) for x in xs]
    if any(y is None or not math.isfinite(y) for y in ys):
        return None
    return {"x": [float(x) for x in xs], "y": [float(y) for y in ys]}  # type: ignore[arg-type]


def _schedule_steps(node: dict) -> int | None:
    """@ai-generated"""
    if node.get(CLASS_KEY) == "RoundedSchedule":
        inner = node.get("schedule")
        return _schedule_steps(inner) if isinstance(inner, dict) else None
    return as_int(node.get("n_steps"))


# ---------------------------------------------------------------- Flatten


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _leaf_kind(value: Any) -> ParamKind:
    """@ai-generated"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def flatten(raw: Any, horizon: int | None = None) -> list[ParamRow]:
    """
    Depth-first rows of a raw JSON object. `class-name` becomes `cls` of object nodes, `name` keys
    duplicating the class name are hidden, and lists of scalars are leaves. Schedules (class name
    ending in `Schedule`) get a curve over `horizon` (default: the root's `n_steps`).

    @ai-generated
    """
    if not isinstance(raw, dict):
        return []
    if horizon is None:
        horizon = as_int(raw.get("n_steps"))
    rows = list[ParamRow]()

    def visit_children(node: Any, prefix: str, depth: int):
        if isinstance(node, dict):
            cls = node.get(CLASS_KEY)
            items: Iterable[tuple[str, Any]] = ((k, v) for k, v in node.items() if k != CLASS_KEY and not (k == "name" and v == cls))
        else:
            items = ((str(i), v) for i, v in enumerate(node))
        for key, value in items:
            visit(key, value, f"{prefix}.{key}" if prefix else key, depth)

    def visit(key: str, value: Any, path: str, depth: int):
        if isinstance(value, dict):
            cls = value.get(CLASS_KEY)
            cls = cls if isinstance(cls, str) else None
            if cls is not None and cls.endswith("Schedule"):
                rows.append(ParamRow(path, key, depth, "schedule", None, cls, schedule_curve(value, horizon)))
            else:
                rows.append(ParamRow(path, key, depth, "object", None, cls))
            visit_children(value, path, depth + 1)
        elif isinstance(value, list):
            if all(_is_scalar(v) for v in value):
                rows.append(ParamRow(path, key, depth, "array", value))
            else:
                rows.append(ParamRow(path, key, depth, "array", None))
                visit_children(value, path, depth + 1)
        else:
            rows.append(ParamRow(path, key, depth, _leaf_kind(value), value))

    visit_children(raw, "", 0)
    return rows


# ---------------------------------------------------------------- Search


_TOKEN_RE = re.compile(r'(?P<path>[\w.\-\[\]]+)\s*(?P<op>>=|<=|!=|=|>|<|~)\s*(?P<value>"[^"]*"|\S+)|(?P<text>"[^"]*"|\S+)')


@dataclass(frozen=True)
class Term:
    path: str | None
    """None for free text."""
    op: Op | None
    value: str


@dataclass
class SearchContext:
    """What a query is matched against: identity fields and flattened parameters."""

    id: str
    name: str
    algo: str | None = None
    env: str | None = None
    test_env: str | None = None
    status: str | None = None
    rows: list[ParamRow] = field(default_factory=list)

    def all_rows(self) -> list[ParamRow]:
        """Parameter rows plus virtual rows for the identity fields. @ai-generated"""
        virtual = [
            ParamRow(key, key, 0, "string", value)
            for key, value in (("id", self.id), ("name", self.name), ("algo", self.algo), ("env", self.env), ("status", self.status))
            if value is not None
        ]
        return [*virtual, *self.rows]


def parse_query(q: str) -> list[Term]:
    """@ai-generated"""
    terms = list[Term]()
    for match in _TOKEN_RE.finditer(q or ""):
        if match.group("text") is not None:
            text = match.group("text").strip('"')
            if text and text.upper() != "AND":
                terms.append(Term(None, None, text))
        else:
            terms.append(Term(match.group("path"), match.group("op"), match.group("value").strip('"')))  # type: ignore[arg-type]
    return terms


def _to_number(value: str) -> float | None:
    """@ai-generated"""
    try:
        return float(value)
    except ValueError:
        return None


def _compare_row(row: ParamRow, op: Op, value: str) -> bool:
    """Whether `row <op> value` holds; object and schedule nodes compare their class name. @ai-generated"""
    target: Any = row.cls if row.kind in ("object", "schedule") else row.value
    if row.kind == "array":
        target = row.value
        if op == "~" and isinstance(target, list):
            return any(value.casefold() in str(v).casefold() for v in target)
        return False
    if op == "~":
        return target is not None and value.casefold() in str(target).casefold()
    number = _to_number(value)
    if isinstance(target, (int, float)) and not isinstance(target, bool) and number is not None:
        return {
            "=": target == number,
            "!=": target != number,
            ">": target > number,
            "<": target < number,
            ">=": target >= number,
            "<=": target <= number,
        }[op]
    if op not in ("=", "!="):
        return False
    if target is None:
        equal = value.casefold() in ("null", "none")
    elif isinstance(target, bool):
        equal = str(target).casefold() == value.casefold()
    else:
        equal = str(target).casefold() == value.casefold()
    return equal if op == "=" else not equal


def _path_matches(path: str, suffix: str) -> bool:
    return path == suffix or path.endswith("." + suffix)


def match_term(term: Term, ctx: SearchContext, rows: list[ParamRow] | None = None) -> bool:
    """@ai-generated"""
    rows = ctx.all_rows() if rows is None else rows
    if term.path is None or term.op is None:
        needle = term.value.casefold()
        for row in rows:
            for candidate in (row.cls, row.value if row.kind != "array" else None):
                if candidate is not None and not isinstance(candidate, bool) and needle in str(candidate).casefold():
                    return True
        return False
    candidates = [row for row in rows if _path_matches(row.path, term.path)]
    if not candidates:
        return False
    if term.op == "!=":
        return all(_compare_row(row, "!=", term.value) for row in candidates)
    return any(_compare_row(row, term.op, term.value) for row in candidates)


def matches(query: str | list[Term], ctx: SearchContext) -> bool:
    """All terms of the query match (an empty query matches everything). @ai-generated"""
    terms = parse_query(query) if isinstance(query, str) else query
    rows = ctx.all_rows()
    return all(match_term(term, ctx, rows) for term in terms)
