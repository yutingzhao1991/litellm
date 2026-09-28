"""
Response-field billing engine for the FeelFish pass-through provider.

Pure functions: given a validated :class:`BillingConfig` and a
:class:`BillingContext` (request body / response body / response text /
response headers) it returns the USD cost plus an auditable breakdown.

Design notes
------------

- ``cost = max(min_cost, sum(value * unit_price * scale) * multiplier)``.
- A rule that cannot resolve a value contributes ``0`` and is marked
  ``matched: false``. If a ``required`` rule is missing, the **whole primary
  result is discarded** and the fallback chain is used (fallback rules >
  ``cost_per_request`` > ``0`` + warning). That keeps a partially-resolved rule
  set from silently under-charging.
- Lists are flattened before aggregation, so ``items[*].cost`` and ``path:
  items`` + ``aggregate: count`` behave predictably.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .config import (
    Aggregate,
    BillingConfig,
    BillingRule,
    FieldSource,
    MissingPolicy,
    Transform,
)
from .field_path import lookup, lookup_header, parse_json_lines

# Costs are tracked in USD; keep enough precision for per-character pricing.
COST_PRECISION = 10


@dataclass
class BillingContext:
    """Everything a billing rule may read."""

    request_body: Optional[Any] = None
    response_body: Optional[Any] = None
    response_text: Optional[str] = None
    response_headers: Optional[Mapping[str, str]] = None


@dataclass
class RuleResult:
    """Evaluation of a single rule (also used for the breakdown payload)."""

    rule: BillingRule
    value: Optional[float]
    matched: bool
    cost: float

    def to_breakdown(self) -> Dict[str, Any]:
        return {
            "from": self.rule.from_.value,
            "path": self.rule.path,
            "aggregate": self.rule.aggregate.value,
            "transform": self.rule.transform.value,
            "value": self.value,
            "unit": self.rule.unit,
            "unit_price": self.rule.unit_price,
            "scale": self.rule.scale,
            "required": self.rule.required,
            "matched": self.matched,
            "cost": self.cost,
        }


@dataclass
class BillingResult:
    """Final cost plus the audit trail for one request."""

    cost: float
    rules: List[RuleResult] = field(default_factory=list)
    fallback_used: bool = False
    cost_per_request_used: bool = False
    missing_required: List[str] = field(default_factory=list)

    @property
    def has_missing_required(self) -> bool:
        return len(self.missing_required) > 0

    def to_breakdown(self) -> Dict[str, Any]:
        return {
            "total": self.cost,
            "rules": [result.to_breakdown() for result in self.rules],
            "fallback_used": self.fallback_used,
            "cost_per_request_used": self.cost_per_request_used,
            "missing_required": list(self.missing_required),
        }


def evaluate_billing(
    config: BillingConfig,
    context: BillingContext,
    incomplete_response: bool = False,
) -> BillingResult:
    """
    Compute the USD cost for a single request.

    ``incomplete_response`` marks a streamed response whose middle was dropped by
    the billing buffer. Aggregations that need the whole stream (``sum`` /
    ``count`` / ``min`` / ``max``) would undercharge from a partial window, so
    such rules are treated as unresolved and the fallback chain applies instead.
    """
    if incomplete_response and _needs_complete_stream(config.rules):
        unresolved = _RulesOutcome(
            cost=0.0,
            results=[
                RuleResult(rule=rule, value=None, matched=False, cost=0.0)
                for rule in config.rules
            ],
            missing_required=[rule.path for rule in config.rules],
            matched_any=False,
        )
        return _apply_fallback_chain(
            config, context, unresolved, incomplete_response=incomplete_response
        )

    primary = _evaluate_rules(config.rules, context)

    if primary.missing_required and config.on_missing == MissingPolicy.FALLBACK:
        return _apply_fallback_chain(
            config, context, primary, incomplete_response=incomplete_response
        )

    cost = primary.cost
    used_cost_per_request = False
    if not primary.matched_any and config.cost_per_request > 0:
        cost = config.cost_per_request
        used_cost_per_request = True

    return BillingResult(
        cost=_finalize_cost(cost, config),
        rules=primary.results,
        fallback_used=False,
        cost_per_request_used=used_cost_per_request,
        missing_required=primary.missing_required,
    )


COMPLETE_STREAM_AGGREGATES = frozenset(
    {Aggregate.SUM, Aggregate.COUNT, Aggregate.MIN, Aggregate.MAX}
)

# Only streamed response bodies can be truncated; request bodies and response
# headers are always complete.
TRUNCATABLE_SOURCES = frozenset({FieldSource.RESPONSE_BODY, FieldSource.RESPONSE_TEXT})


def _needs_complete_stream(rules: Sequence[BillingRule]) -> bool:
    """
    Whether any rule aggregates over a stream that may have been truncated.

    ``sum`` / ``count`` / ``min`` / ``max`` over the response need every event;
    the same aggregations over the request body or headers are unaffected by a
    truncated response and keep working.
    """
    return any(
        rule.from_ in TRUNCATABLE_SOURCES and rule.aggregate in COMPLETE_STREAM_AGGREGATES
        for rule in rules
    )


def _apply_fallback_chain(
    config: BillingConfig,
    context: BillingContext,
    primary: "_RulesOutcome",
    incomplete_response: bool = False,
) -> BillingResult:
    fallback = _evaluate_rules(config.fallback, context) if config.fallback else None

    # A fallback set is only usable when all of *its* required fields resolved and
    # it does not itself aggregate over a truncated stream: accepting either would
    # silently undercharge and bypass ``cost_per_request``.
    fallback_incomplete = bool(
        incomplete_response
        and config.fallback
        and _needs_complete_stream(config.fallback)
    )
    if (
        fallback is not None
        and fallback.matched_any
        and not fallback.missing_required
        and not fallback_incomplete
    ):
        return BillingResult(
            cost=_finalize_cost(fallback.cost, config),
            rules=primary.results + fallback.results,
            fallback_used=True,
            cost_per_request_used=False,
            missing_required=primary.missing_required,
        )

    missing_required = primary.missing_required + (
        fallback.missing_required if fallback is not None else []
    )

    if config.cost_per_request > 0:
        return BillingResult(
            cost=_finalize_cost(config.cost_per_request, config),
            rules=primary.results + (fallback.results if fallback is not None else []),
            fallback_used=bool(config.fallback),
            cost_per_request_used=True,
            missing_required=missing_required,
        )

    # Nothing to bill with: record the miss so operators can alert on it.
    return BillingResult(
        cost=_finalize_cost(0.0, config),
        rules=primary.results + (fallback.results if fallback is not None else []),
        fallback_used=bool(config.fallback),
        cost_per_request_used=False,
        missing_required=missing_required,
    )


def _finalize_cost(cost: float, config: BillingConfig) -> float:
    return round(max(config.min_cost, cost * config.multiplier), COST_PRECISION)


@dataclass
class _RulesOutcome:
    cost: float
    results: List[RuleResult]
    missing_required: List[str]
    matched_any: bool


def _evaluate_rules(
    rules: Sequence[BillingRule], context: BillingContext
) -> _RulesOutcome:
    results: List[RuleResult] = []
    missing_required: List[str] = []
    cost = 0.0
    matched_any = False

    for rule in rules:
        value = _resolve_rule_value(rule, context)
        if value is None:
            results.append(RuleResult(rule=rule, value=None, matched=False, cost=0.0))
            if rule.required:
                missing_required.append(rule.path)
            continue
        rule_cost = value * rule.unit_price * rule.scale
        cost += rule_cost
        matched_any = True
        results.append(
            RuleResult(rule=rule, value=value, matched=True, cost=rule_cost)
        )

    return _RulesOutcome(
        cost=cost,
        results=results,
        missing_required=missing_required,
        matched_any=matched_any,
    )


def _resolve_rule_value(
    rule: BillingRule, context: BillingContext
) -> Optional[float]:
    raw_values = _collect_raw_values(rule, context)
    if raw_values is None:
        return None
    if not raw_values:
        # The field resolved, but to an empty collection: only `count` has a
        # meaningful value (0). Treating it as "missing" would wrongly fire the
        # fallback/cost_per_request chain for a legitimate zero-usage response.
        return 0.0 if rule.aggregate == Aggregate.COUNT else None
    aggregated = _aggregate(raw_values, rule.aggregate)
    if aggregated is None:
        return None
    return _apply_transform(aggregated, rule.transform)


def _collect_raw_values(
    rule: BillingRule, context: BillingContext
) -> Optional[List[Any]]:
    """
    Collect the raw values for a rule.

    Returns ``None`` when the field/source is missing, or a (possibly empty) list
    when it resolved.
    """
    source = rule.from_

    if source == FieldSource.REQUEST_BODY:
        return _flatten_resolved(lookup(context.request_body, rule.path))

    if source == FieldSource.RESPONSE_BODY:
        return _flatten_resolved(lookup(context.response_body, rule.path))

    if source == FieldSource.RESPONSE_HEADERS:
        return _flatten_resolved(lookup_header(context.response_headers, rule.path))

    if source == FieldSource.RESPONSE_TEXT:
        events = parse_json_lines(context.response_text)
        if not events:
            return None
        values: List[Any] = []
        resolved_any = False
        for event in events:
            raw = lookup(event, rule.path)
            if raw is None:
                continue
            resolved_any = True
            values.extend(_flatten([raw]))
        # No event carried the field at all -> missing (fallback applies), as
        # opposed to a field that resolved to an empty collection.
        return values if resolved_any else None

    return None


def _flatten_resolved(value: Any) -> Optional[List[Any]]:
    """Flatten a resolved value; ``None`` means the path did not resolve."""
    if value is None:
        return None
    return _flatten([value])


def _flatten(values: Sequence[Any]) -> List[Any]:
    """
    Recursively flatten list results.

    A wildcard path evaluated against a streamed event list yields nested lists
    (one per event), so a single level of flattening would drop the inner values
    and silently bill 0 where the non-streaming path bills correctly.
    """
    flattened: List[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            flattened.extend(_flatten(value))
        else:
            flattened.append(value)
    return flattened


def _aggregate(values: Sequence[Any], aggregate: Aggregate) -> Any:
    if not values:
        return None
    if aggregate == Aggregate.FIRST:
        return values[0]
    if aggregate == Aggregate.LAST:
        return values[-1]
    if aggregate == Aggregate.COUNT:
        return float(len(values))

    numbers = [
        number
        for number in (_to_number(value) for value in values)
        if number is not None
    ]
    if not numbers:
        return None
    if aggregate == Aggregate.SUM:
        return float(sum(numbers))
    if aggregate == Aggregate.MIN:
        return float(min(numbers))
    if aggregate == Aggregate.MAX:
        return float(max(numbers))
    return None


def _apply_transform(value: Any, transform: Transform) -> Optional[float]:
    if transform == Transform.NONE:
        return _to_number(value)
    if transform == Transform.CHARACTERS:
        return float(len(str(value)))
    if transform == Transform.CHARACTERS_NO_SPACE:
        return float(len("".join(str(value).split())))
    if transform == Transform.BYTES:
        return float(len(str(value).encode("utf-8")))
    if transform == Transform.MS_TO_SECONDS:
        number = _to_number(value)
        return None if number is None else number / 1000.0
    if transform == Transform.COUNT_ITEMS:
        if isinstance(value, (list, dict, str)):
            return float(len(value))
        return None
    if transform == Transform.TRUTHY:
        return 1.0 if _is_truthy(value) else 0.0
    return None


def _to_number(value: Any) -> Optional[float]:
    """Coerce a JSON scalar to ``float``; non-numeric values are "missing"."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError):
            return None
        # "NaN" / "Infinity" parse successfully but are not billable values.
        return number if math.isfinite(number) else None
    return None


def _is_truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "none", "null")
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True
