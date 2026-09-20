import json
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, TypedDict
from zoneinfo import ZoneInfo

from litellm._logging import verbose_logger
from litellm.types.utils import ModelInfo


class TimeBasedPricingResult(TypedDict, total=False):
    multiplier: float
    rule_name: Optional[str]
    timezone: Optional[str]
    date_classes: Optional[List[str]]


class CompiledCalendar(TypedDict):
    holiday_dates: FrozenSet[date]
    workday_dates: FrozenSet[date]
    is_valid: bool
    invalid_entries: Tuple[str, ...]


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

# Date class names that carry built-in meaning. Every other string in a rule's
# `dates` list is treated as a literal `YYYY-MM-DD` date, so one-off dates do not
# need calendar entries.
_RESERVED_DATE_CLASSES = frozenset({"workday", "weekend", "holiday", "all"})

_CALENDAR_KEY_BY_CLASS = {
    "holiday": "holidays",
    "workday": "workdays",
}

_CALENDAR_COMPILE_CACHE_SIZE = 256


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
    """Match a local wall-clock time against a rule window.

    Start is inclusive and end is exclusive. `start == end` means the whole day,
    which is how a rule such as "holidays are off-peak all day" is expressed.
    `start > end` wraps past midnight.
    """
    if start_time == end_time:
        return True
    if start_time < end_time:
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


def _empty_calendar(
    invalid_entries: Tuple[str, ...] = (),
) -> CompiledCalendar:
    return {
        "holiday_dates": frozenset(),
        "workday_dates": frozenset(),
        "is_valid": True,
        "invalid_entries": invalid_entries,
    }


def _parse_iso_date(raw_entry: Any) -> Optional[date]:
    if not isinstance(raw_entry, str):
        return None
    try:
        return date.fromisoformat(raw_entry.strip())
    except ValueError:
        return None


def _parse_calendar_entry(raw_entry: Any) -> Optional[Tuple[date, date]]:
    """Parse one calendar entry into an inclusive date range.

    Accepts `"YYYY-MM-DD"`, `"YYYY-MM-DD..YYYY-MM-DD"`, and the object form
    `{"start": ..., "end": ...}`. Returns None for anything unusable so a single
    typo cannot invalidate the rest of the calendar.
    """
    if isinstance(raw_entry, str):
        parts = raw_entry.split("..")
        if len(parts) == 1:
            single = _parse_iso_date(parts[0])
            return None if single is None else (single, single)
        if len(parts) == 2:
            start = _parse_iso_date(parts[0])
            end = _parse_iso_date(parts[1])
            if start is None or end is None:
                return None
            return (end, start) if end < start else (start, end)
        return None

    if isinstance(raw_entry, Mapping):
        start = _parse_iso_date(raw_entry.get("start"))
        end = _parse_iso_date(raw_entry.get("end"))
        if start is None:
            return None
        if end is None:
            end = start
        return (end, start) if end < start else (start, end)

    return None


def _collect_calendar_dates(
    calendar_config: Mapping[str, Any],
    key: str,
) -> Tuple[List[date], List[str]]:
    """Expand one calendar list into concrete dates plus invalid entries."""
    entries = calendar_config.get(key)
    if entries is None:
        return [], []
    if not isinstance(entries, list):
        return [], [repr(entries)]

    collected: List[date] = []
    invalid: List[str] = []
    for raw_entry in entries:
        parsed = _parse_calendar_entry(raw_entry)
        if parsed is None:
            invalid.append(repr(raw_entry))
            continue
        start, end = parsed
        current = start
        while current <= end:
            collected.append(current)
            # Stop at the range end: adding a day to `date.max` overflows, and one
            # such entry must not take down the whole calendar.
            if current == end:
                break
            current += timedelta(days=1)

    return collected, invalid


def compile_time_based_pricing_calendar(
    pricing_config: Optional[Mapping[str, Any]],
) -> CompiledCalendar:
    """Compile `time_based_pricing.calendar` into concrete date sets.

    A missing or unusable `calendar` yields an empty (but valid) calendar, which
    keeps date matching limited to the built-in `workday` / `weekend` classes.
    Individual entries that fail to parse are dropped and reported instead of
    disabling the whole calendar.
    """
    if not isinstance(pricing_config, Mapping):
        return _empty_calendar()

    calendar_config = pricing_config.get("calendar")
    if calendar_config is None:
        return _empty_calendar()
    if not isinstance(calendar_config, Mapping):
        verbose_logger.debug(
            "Invalid time_based_pricing calendar=%s. Ignoring calendar.",
            calendar_config,
        )
        return _empty_calendar()

    holidays, invalid_holidays = _collect_calendar_dates(calendar_config, "holidays")
    workdays, invalid_workdays = _collect_calendar_dates(calendar_config, "workdays")
    invalid_entries = tuple(invalid_holidays + invalid_workdays)
    if invalid_entries:
        verbose_logger.debug(
            "Invalid time_based_pricing calendar entries=%s. Skipping them.",
            invalid_entries,
        )

    return {
        "holiday_dates": frozenset(holidays),
        "workday_dates": frozenset(workdays),
        "is_valid": True,
        "invalid_entries": invalid_entries,
    }


def _canonical_cache_key(value: Any) -> Optional[str]:
    """Return a stable string key for a config value, or None if not canonicalizable."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return None


@lru_cache(maxsize=_CALENDAR_COMPILE_CACHE_SIZE)
def _compile_calendar_cached(canonical_calendar_config: str) -> CompiledCalendar:
    """Compile a canonicalized calendar config. Keyed by a string so it is hashable."""
    return compile_time_based_pricing_calendar(
        {"calendar": json.loads(canonical_calendar_config)}
    )


def _get_compiled_calendar(pricing_config: Mapping[str, Any]) -> CompiledCalendar:
    """Return the compiled calendar for a config, compiling at most once per config."""
    calendar_config = pricing_config.get("calendar")
    if calendar_config is None:
        return _empty_calendar()

    cache_key = _canonical_cache_key(calendar_config)
    if cache_key is None:
        return compile_time_based_pricing_calendar(pricing_config)

    return _compile_calendar_cached(cache_key)


def _parse_rule_dates(
    raw_dates: Any,
) -> Optional[List[str]]:
    """Normalize a rule's `dates` list.

    Returns None when the rule does not restrict by date (missing `dates`), an
    empty list when the config is unusable, otherwise the normalized entries.

    One unrecognized entry makes the whole list unusable, so a rule with a typo
    such as `["workday", "hoilday"]` is skipped instead of silently applying to
    workdays. Partial-validity here would make a bad config look like it works.
    """
    if raw_dates is None:
        return None
    if not isinstance(raw_dates, list):
        return []

    normalized: List[str] = []
    for entry in raw_dates:
        if not isinstance(entry, str) or not entry.strip():
            return []
        candidate = entry.strip().lower()
        if candidate in _RESERVED_DATE_CLASSES:
            normalized.append(candidate)
            continue
        if _parse_calendar_entry(candidate) is None:
            return []
        normalized.append(candidate)

    return normalized


def _get_date_classes(
    local_date: date,
    calendar: CompiledCalendar,
) -> FrozenSet[str]:
    """Return the calendar classes satisfied by a local date.

    A date in `calendar.workdays` is a workday even when it falls on a weekend.
    That override is for providers that charge peak on an adjusted weekend;
    providers that bill every weekend as off-peak leave `workdays` unset, and
    those dates keep the natural `weekend` class.
    """
    holiday_dates = calendar["holiday_dates"]
    workday_dates = calendar["workday_dates"]

    is_weekend = local_date.weekday() >= 5
    if is_weekend:
        classes = {"weekend"}
    else:
        classes = {"workday"}

    if local_date in workday_dates:
        classes = {"workday"}
    elif local_date in holiday_dates:
        classes = {"holiday"}

    classes.add("all")
    return frozenset(classes)


def _matches_rule_dates(
    rule_dates: Optional[List[str]],
    local_date: date,
    calendar: CompiledCalendar,
) -> bool:
    if rule_dates is None:
        return True

    date_classes = _get_date_classes(local_date, calendar)
    for entry in rule_dates:
        if entry in _RESERVED_DATE_CLASSES:
            if entry in date_classes:
                return True
            continue

        # Literal dates and inclusive ranges, so a one-off promotion or a holiday
        # block can be expressed directly in a rule without calendar entries.
        window = _parse_calendar_entry(entry)
        if window is None:
            continue
        start, end = window
        if start <= local_date <= end:
            return True

    return False


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
    local_date = local_datetime.date()

    calendar = _get_compiled_calendar(pricing_config)
    date_classes = _get_date_classes(local_date, calendar)

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        start_time = _parse_time_string(rule.get("start_time"))
        end_time = _parse_time_string(rule.get("end_time"))
        multiplier = rule.get("multiplier")
        days = _parse_days(rule.get("days"))
        rule_dates = _parse_rule_dates(rule.get("dates"))

        if (
            start_time is None
            or end_time is None
            or not isinstance(multiplier, (int, float))
            or multiplier <= 0
            or days == []
            or rule_dates == []
        ):
            verbose_logger.debug(
                "Invalid time_based_pricing rule=%s. Skipping rule.",
                rule,
            )
            continue

        if days is not None and local_weekday not in days:
            continue

        if not _matches_rule_dates(rule_dates, local_date, calendar):
            continue

        if _is_time_in_window(local_time, start_time, end_time):
            rule_name = rule.get("name")
            return {
                "multiplier": float(multiplier),
                "rule_name": rule_name if isinstance(rule_name, str) else None,
                "timezone": timezone_name,
                "date_classes": sorted(date_classes),
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
