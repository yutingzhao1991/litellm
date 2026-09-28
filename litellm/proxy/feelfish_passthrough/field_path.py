"""
Field path evaluation for FeelFish response-field pricing.

Supports the small path dialect used by ``response_field_pricing`` rules:

- dotted keys: ``usage.text_words``
- array index: ``items[0].text``
- array wildcard: ``items[*].text`` (returns a list of values)
- header names (case-insensitive) when the value source is a header mapping
- NDJSON / SSE text buffers when the value source is ``response_text``

Everything is pure and framework free, so the billing engine can be unit
tested without the proxy or httpx.
"""

import json
import re
from typing import Any, List, Mapping, Optional, Sequence, Union

WILDCARD = "*"

_INDEX_PATTERN = re.compile(r"\[(\*|\d+)\]")
_SEGMENT_PATTERN = re.compile(r"^([^\[\]]*)((?:\[\*\]|\[\d+\])*)$")

PathToken = Union[str, int]


def parse_path(path: str) -> List[PathToken]:
    """
    Parse ``a.b[*].c`` into tokens (``str`` keys, ``int`` indexes, ``*`` wildcards).

    Segments are split on ``.`` first so keys may contain any character except
    ``[`` / ``]`` / ``.``.
    """
    tokens: List[PathToken] = []
    for segment in (path or "").split("."):
        if not segment:
            continue
        match = _SEGMENT_PATTERN.match(segment)
        if match is None:
            # Malformed segment (stray brackets): treat it as a literal key so the
            # rule simply resolves to "missing" instead of raising.
            tokens.append(segment)
            continue
        key, indexes = match.group(1), match.group(2)
        if key:
            tokens.append(key)
        for index in _INDEX_PATTERN.findall(indexes):
            tokens.append(WILDCARD if index == "*" else int(index))
    return tokens


def evaluate_path(node: Any, tokens: Sequence[PathToken]) -> Any:
    """
    Resolve ``tokens`` against ``node``.

    Returns ``None`` when any segment is missing. A wildcard returns the list
    of resolved values (``None`` values removed); if nothing resolves it
    returns ``None`` rather than an empty list so callers can treat it as
    "missing".
    """
    if not tokens:
        return node
    if node is None:
        return None

    token, rest = tokens[0], tokens[1:]

    if token == WILDCARD:
        if not isinstance(node, list):
            return None
        if not node:
            # An empty list is a resolved value (count = 0), not a missing field;
            # otherwise zero usage would wrongly trigger the fallback chain.
            return []
        resolved = [evaluate_path(item, rest) for item in node]
        values = [value for value in resolved if value is not None]
        return values or None

    if isinstance(token, int):
        if not isinstance(node, list) or token >= len(node):
            return None
        return evaluate_path(node[token], rest)

    if isinstance(node, dict):
        return evaluate_path(node.get(token), rest)

    if isinstance(node, list):
        # Implicit wildcard: ``items.text`` over a list of objects.
        if not node:
            return []
        resolved = [evaluate_path(item, tokens) for item in node]
        values = [value for value in resolved if value is not None]
        return values or None

    return None


def lookup(data: Any, path: str) -> Any:
    """Resolve ``path`` against ``data`` (see module docstring for the dialect)."""
    return evaluate_path(data, parse_path(path))


def lookup_header(headers: Optional[Mapping[str, str]], name: str) -> Optional[str]:
    """Case-insensitive header lookup."""
    if not headers or not name:
        return None
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def parse_json_lines(text: Optional[str]) -> List[Any]:
    """
    Parse an NDJSON / SSE buffer into a list of JSON values.

    Handles ``data: {...}`` SSE prefixes and silently skips lines that are not
    JSON (keep-alive comments, ``[DONE]``, partial trailing lines).
    """
    if not text:
        return []
    parsed: List[Any] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if not line or line == "[DONE]":
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return parsed
