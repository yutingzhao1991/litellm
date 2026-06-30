from datetime import datetime, time, timezone
from typing import Any, List, Optional, TypedDict
from zoneinfo import ZoneInfo

from litellm._logging import verbose_logger
from litellm.types.utils import ModelInfo


class TimeBasedPricingResult(TypedDict, total=False):
    multiplier: float
    rule_name: Optional[str]
    timezone: Optional[str]


_DEFAULT_TIME_BASED_PRICING_RESULT: TimeBasedPricingResult = {"multiplier": 1.0}

_DAY_NAME_TO_WEEKDAY = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def _default_time_based_pricing_result() -> TimeBasedPricingResult:
    return dict(_DEFAULT_TIME_BASED_PRICING_RESULT)


def _parse_time_string(time_string: Any) -> Optional[time]:
    if not isinstance(time_string, str):
        return None
    try:
        return datetime.strptime(time_string, "%H:%M").time()
    except ValueError:
        return None


def _parse_days(days: Any) -> Optional[List[int]]:
    if days is None:
        return None
    if not isinstance(days, list):
        return []

    parsed_days: List[int] = []
    for day in days:
        if not isinstance(day, str):
            return []
        weekday = _DAY_NAME_TO_WEEKDAY.get(day.lower())
        if weekday is None:
            return []
        parsed_days.append(weekday)

    return parsed_days


def _is_time_in_window(local_time: time, start_time: time, end_time: time) -> bool:
    if start_time <= end_time:
        return start_time <= local_time < end_time

    return local_time >= start_time or local_time < end_time


def _get_pricing_datetime(
    pricing_datetime: Optional[datetime],
    pricing_timezone: ZoneInfo,
) -> datetime:
    selected_datetime = pricing_datetime or datetime.now(timezone.utc)
    if selected_datetime.tzinfo is None:
        selected_datetime = selected_datetime.astimezone()

    return selected_datetime.astimezone(pricing_timezone)


def get_time_based_pricing_result(
    model_info: Optional[ModelInfo],
    pricing_datetime: Optional[datetime] = None,
) -> TimeBasedPricingResult:
    """
    Return the configured time-based pricing multiplier for a model.

    Missing or invalid config falls back to multiplier=1.0 so cost calculation
    remains backwards compatible and resilient to bad model price entries.
    """
    if model_info is None:
        return _default_time_based_pricing_result()

    pricing_config = model_info.get("time_based_pricing")
    if not isinstance(pricing_config, dict):
        return _default_time_based_pricing_result()

    timezone_name = pricing_config.get("timezone", "UTC")
    if not isinstance(timezone_name, str):
        verbose_logger.debug(
            "Invalid time_based_pricing timezone=%s. Falling back to multiplier=1.0",
            timezone_name,
        )
        return _default_time_based_pricing_result()

    try:
        pricing_timezone = ZoneInfo(timezone_name)
    except Exception:
        verbose_logger.debug(
            "Invalid time_based_pricing timezone=%s. Falling back to multiplier=1.0",
            timezone_name,
        )
        return _default_time_based_pricing_result()

    rules = pricing_config.get("rules")
    if not isinstance(rules, list):
        return _default_time_based_pricing_result()

    local_datetime = _get_pricing_datetime(
        pricing_datetime=pricing_datetime,
        pricing_timezone=pricing_timezone,
    )
    local_time = local_datetime.time()
    local_weekday = local_datetime.weekday()

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        start_time = _parse_time_string(rule.get("start_time"))
        end_time = _parse_time_string(rule.get("end_time"))
        multiplier = rule.get("multiplier")
        days = _parse_days(rule.get("days"))

        if (
            start_time is None
            or end_time is None
            or not isinstance(multiplier, (int, float))
            or multiplier <= 0
            or days == []
        ):
            verbose_logger.debug(
                "Invalid time_based_pricing rule=%s. Skipping rule.",
                rule,
            )
            continue

        if days is not None and local_weekday not in days:
            continue

        if _is_time_in_window(local_time, start_time, end_time):
            rule_name = rule.get("name")
            return {
                "multiplier": float(multiplier),
                "rule_name": rule_name if isinstance(rule_name, str) else None,
                "timezone": timezone_name,
            }

    return _default_time_based_pricing_result()


def apply_time_based_pricing(
    prompt_cost: float,
    completion_cost: float,
    model_info: Optional[ModelInfo],
    pricing_datetime: Optional[datetime] = None,
) -> tuple[float, float, TimeBasedPricingResult]:
    pricing_result = get_time_based_pricing_result(
        model_info=model_info,
        pricing_datetime=pricing_datetime,
    )
    multiplier = pricing_result["multiplier"]

    return prompt_cost * multiplier, completion_cost * multiplier, pricing_result
